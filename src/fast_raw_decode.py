"""
Fast full-resolution RAW decode with exact LibRaw color parity.

Replaces ``rawpy.postprocess()`` for the sensor-resolution zoom decode only
(export and the Adjust-panel edit base keep their existing rawpy paths).
LibRaw still does the unpack (file decode); this module reimplements the
*pixel math* with cv2/numpy primitives that are SIMD + multithreaded.
Honest end-to-end numbers (fresh imread each run, min of 3, Apple M1):
1.4-1.7x faster than rawpy's LINEAR demosaic path (what the sensor decode
used before) and 2.3-3.1x faster than AHD — with *better* demosaic quality
than LINEAR (cv2 edge-aware vs bilinear). LibRaw unpack (100-400ms,
CPU-bound, not offloadable) is the Amdahl floor on further gains. GPU offload (OpenCL/Metal) was measured and
rejected: cv2's CPU demosaic is already memory-bound on unified memory
(19ms GPU vs 17ms CPU for a 33MP frame), so a GPU backend adds dependency
weight for no gain. See docs/local/UPCOMING_FEATURES.md section E.

Color parity (the E2 gate that blocked the earlier GPU PoC) is exact by
construction — every step below replicates dcraw/LibRaw semantics for the
app's decode params (use_camera_wb, no_auto_bright, output_bps=8,
gamma=(2.222, 4.5), user_flip=0, highlight_mode=clip):

1. Per-channel black subtraction and white-balance scaling exactly as
   dcraw ``scale_colors``: camera multipliers divided by their *minimum*
   (all >= 1) and the result clipped to 65535 in camera space *before* the
   color matrix — this is what forces clipped highlights to saturate to
   neutral white. (The earlier PoC normalized by max and clipped after the
   matrix, which is exactly what produced its shifted highlight colors.)
2. ``rgb_cam`` derived exactly as dcraw ``cam_xyz_coeff``: forward matrix
   ``cam_rgb = cam_xyz @ XYZ_RGB`` row-normalized (rows sum to 1, mapping
   camera white to sRGB white) then pseudo-inverted. ``raw.color_matrix``
   cannot be used directly — LibRaw leaves it zeroed for most cameras
   (verified on Sony ARW and Canon CR3).
3. LibRaw ``adjust_maximum`` (default adjust_maximum_thr=0.75) replicated
   for the effective saturation level.
4. Output gamma is dcraw's BT.709 curve for gamma=(2.222, 4.5) — linear toe
   below the solved breakpoint, ``(1+g4)*x^(1/2.222) - g4`` above — with the
   8-bit output taken as ``curve >> 8`` like dcraw's write_ppm.

Verified: the demosaic-free half-size reference path below matches
``rawpy.postprocess(half_size=True, ...)`` within +/-1 8-bit LSB on 100% of
pixels across Sony ARW + 12 Canon CR3 golden files (run
``scripts/fast_raw_decode_parity_gate.py``). Highlights included.

Scope guards: Bayer 2x2 / 3-color / uint16 sensors only — X-Trans, Foveon,
monochrome, linear DNG, float DNG and 4-color CFAs all return None and fall
back to the existing rawpy path. Disable entirely with
``RAWVIEWER_FAST_RAW_DECODE=0``.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DecodeCancelled(Exception):
    """Raised mid-decode when the owning load task was cancelled.

    The chunked stages check ``cancel_check`` between chunks so a cancelled
    background/prefetch decode releases its RAW worker slot within ~10ms
    instead of running its remaining ~1s to completion -- this is what lets
    a navigation's CURRENT-priority decode start immediately.
    """

# dcraw xyz_rgb: sRGB primaries -> XYZ (D65). Must match dcraw exactly —
# these are not the higher-precision IEC values.
_XYZ_RGB = np.array(
    [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ],
    dtype=np.float64,
)

# OpenCV Bayer code naming is anti-intuitive (it names the pattern by the
# 2x2 *below-right* of the origin): an RGGB sensor needs COLOR_BayerBG2RGB.
# Anchor RGGB->BG was verified empirically against rawpy output (meandiff
# 0.63 vs 4-7 for the other three codes); the rest follow the same shift.
_BAYER_TO_CV2 = {"RGGB": "BG", "BGGR": "RG", "GRBG": "GB", "GBRG": "GR"}

_GAMMA_LUT8: Optional[np.ndarray] = None

# Keys the sensor-decode params may contain without changing color semantics.
# Any unexpected key disables the fast path (conservative: new rawpy params
# added upstream must be reviewed for parity before the fast path honors them).
_ALLOWED_PARAM_KEYS = {
    "use_camera_wb",
    "use_auto_wb",
    "output_bps",
    "no_auto_bright",
    "gamma",
    "bright",
    "user_flip",
    "half_size",
    "demosaic_algorithm",  # fast path substitutes its own (EA >= LINEAR quality)
    "dcb_enhance",  # demosaic-only knob, no color impact
}


def _import_cv2():
    """Import cv2 defensively against cross-thread lazy-import races.

    With lazy `import cv2` spread across worker threads (this app's style),
    Python's importlib deadlock-breaker can hand one thread a partially
    initialized cv2 module (observed in the wild as `partially initialized
    module 'cv2' has no attribute 'COLOR_BayerBG2RGB_EA'`). Retry briefly;
    give up with None so the caller falls back to rawpy for that decode.
    """
    import time as _time

    for _ in range(3):
        import cv2

        if hasattr(cv2, "COLOR_BayerBG2RGB_EA"):
            return cv2
        _time.sleep(0.1)
    return None


def _chunked_copy(a: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Contiguous copy in row chunks so no single memcpy holds the GIL long."""
    out = np.empty(a.shape, dtype=a.dtype)
    for y0 in range(0, a.shape[0], chunk):
        out[y0 : y0 + chunk] = a[y0 : y0 + chunk]
    return out


def _env_disabled() -> bool:
    return str(os.environ.get("RAWVIEWER_FAST_RAW_DECODE", "")).strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }


def prefer_gpu_decode_enabled() -> bool:
    """GPU (MPS/CUDA) full-resolution demosaic, default OFF.

    Measured no speed benefit on real files (45-46MP CR3/NEF: ~358-365ms
    either way, GPU occasionally slower after warmup) -- consistent with
    this module's own top-of-file docstring, which already reached the same
    conclusion for the earlier OpenCL/Metal PoC. GPU decode also feeds the
    MPS unified-memory pressure that needed a dedicated
    release_cached_gpu_memory() mitigation (image_cache.py); skipping it by
    default when it buys nothing removes that cost too. Set
    RAWVIEWER_PREFER_GPU_DECODE=1 to re-enable (e.g. to re-verify on
    different hardware, or for the parity gate's GPU-vs-CPU comparison).
    """
    return str(os.environ.get("RAWVIEWER_PREFER_GPU_DECODE", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def gpu_decode_max_megapixels() -> float:
    """Skip GPU demosaic above this mosaic size (CPU usually wins + less VRAM).

    100MP RAF was measured ~10s on CUDA with heavy RSS growth while browsing;
    fall back to CPU past the cap. Default 70 covers a7CR-class bodies while
    still keeping extreme medium-format / GFX off GPU. Override with
    RAWVIEWER_GPU_DECODE_MAX_MP (0 disables the cap).
    """
    raw = str(os.environ.get("RAWVIEWER_GPU_DECODE_MAX_MP", "70")).strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 70.0


def wb_sanity_enabled() -> bool:
    """As-shot WB sanity check vs embedded JPEG (RAWVIEWER_WB_SANITY=0 disables)."""
    return str(os.environ.get("RAWVIEWER_WB_SANITY", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


# Midtone-masked median of per-pixel log(R/G), log(B/G) differences between a
# tiny LibRaw-pipeline render and the camera's embedded JPEG. A genuine WB
# misparse shifts EVERY pixel the same direction (median catches it); tone
# curve / Picture Style saturation differences are roughly symmetric around
# zero (median rejects them). Measured on 23 golden files: correctly parsed
# bodies score <= 0.21, the misparsed EOS R6 Mark III files score >= 1.12 --
# threshold 0.5 sits in the middle of a >5x gap.
_WB_SANITY_THRESHOLD = 0.5
# key: normcase(abspath). Value: corrected cam_mul list, or None = checked, OK.
_WB_CORRECTION_CACHE: Dict[str, Optional[list]] = {}
# Per-MODEL verdict: the WB misparse is a parser-vs-model property, so once
# one file of a model measures clean, later files of that model skip the
# embedded-JPEG check entirely -- it costs ~38ms of thumb extract + decode +
# metric per file, a real tax on every fresh navigation when paid per-file.
# Misparsed models still measure per file (the as-shot WB is per shot).
# True = model suspect, False = model verified clean.
#
# The key must NOT be the color matrix alone: LibRaw aliases bodies it does
# not know to an older model's matrix (verified: the misparsed EOS R6 Mark
# III returns byte-identical rgb_cam to a correctly-parsed older body in the
# golden set), which let the older body's clean verdict suppress the R6 III
# correction. Matrix + visible-sensor dims + black levels + white level
# separates them (4639x6959/black [0,35,102,69] vs 4024x6022/black 2048x4).
_WB_MODEL_VERDICT: Dict[bytes, bool] = {}


def _wb_model_key(
    rgb_cam: Optional[np.ndarray],
    mosaic: np.ndarray,
    black: np.ndarray,
    white: float,
) -> bytes:
    if rgb_cam is None:
        return b""
    return (
        rgb_cam.tobytes()
        + np.asarray(mosaic.shape, dtype=np.int64).tobytes()
        + np.asarray(black, dtype=np.float64).tobytes()
        + np.float64(white).tobytes()
    )


def get_corrected_camera_wb(file_path: str) -> Optional[list]:
    """Session-cached corrected camera WB multipliers for ``file_path``.

    Non-None only when this file's as-shot WB failed the embedded-JPEG sanity
    check during :func:`unpack_raw` (e.g. bodies newer than the bundled
    LibRaw's makernote parser, like the EOS R6 Mark III). Callers that decode
    via rawpy directly (edit base, EDR) should pass it as ``user_wb`` with
    ``use_camera_wb=False`` so every pipeline shows the same corrected color.
    """
    key = os.path.normcase(os.path.abspath(file_path))
    return _WB_CORRECTION_CACHE.get(key)


def _srgb_to_linear01(arr8: np.ndarray) -> np.ndarray:
    x = arr8.astype(np.float64) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _wb_correction_from_jpeg(
    file_path: str,
    thumb_bytes: Optional[bytes],
    mosaic: np.ndarray,
    pattern: np.ndarray,
    black: np.ndarray,
    scale_mul: np.ndarray,
    rgb_cam: np.ndarray,
    cam_mul: np.ndarray,
) -> Optional[list]:
    """Corrected cam_mul when the as-shot WB disagrees with the embedded JPEG.

    Returns None when the WB is fine, the thumb is unusable, or the solve is
    implausible. Never raises.
    """
    try:
        if not thumb_bytes:
            return None
        cv2 = _import_cv2()
        if cv2 is None:
            return None
        arr = cv2.imdecode(
            np.frombuffer(thumb_bytes, np.uint8), cv2.IMREAD_REDUCED_COLOR_8
        )
        if arr is None or arr.ndim != 3:
            return None

        # Black-subtracted tiny CFA planes (WB-independent, computed once).
        w2 = mosaic.shape[1] // 2
        step = max(1, w2 // 192)
        raw_planes: list = []  # (role, channel_index, plane)
        for (dy, dx), ci in np.ndenumerate(pattern):
            plane = mosaic[dy::2, dx::2][::step, ::step].astype(np.float64) - black[ci]
            role = "R" if ci == 0 else ("B" if ci == 2 else "G")
            raw_planes.append((role, ci, plane))
        roles = {r for r, _, _ in raw_planes}
        if "R" not in roles or "B" not in roles or "G" not in roles:
            return None
        hmin = min(p.shape[0] for _, _, p in raw_planes)
        wmin = min(p.shape[1] for _, _, p in raw_planes)
        raw_planes = [(r, ci, p[:hmin, :wmin]) for r, ci, p in raw_planes]
        m3 = rgb_cam.astype(np.float64)
        m3_inv = np.linalg.pinv(m3)
        # scale_mul = (cam_mul / cam_mul.min()) * 65535 / denom
        denom = (cam_mul[0] / cam_mul.min()) * 65535.0 / scale_mul[0]

        def _build_jpg(src_bgr: np.ndarray) -> np.ndarray:
            return _srgb_to_linear01(
                cv2.cvtColor(
                    cv2.resize(src_bgr, (wmin, hmin), interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2RGB,
                )
            )

        def _render_and_metric(mul: np.ndarray, jpg: np.ndarray, lj: np.ndarray):
            sc = (mul / mul.min()) * 65535.0 / denom
            acc: Dict[str, np.ndarray] = {}
            n_g = 0
            for role, ci, p in raw_planes:
                v = np.clip(p * sc[ci], 0, 65535)
                if role == "G":
                    acc["G"] = acc.get("G", 0) + v
                    n_g += 1
                else:
                    acc[role] = v
            cam3 = np.stack([acc["R"], acc["G"] / max(n_g, 1), acc["B"]], -1)
            render = np.clip(cam3 @ m3.T, 0, 65535) / 65535.0
            lr = 0.2126 * render[..., 0] + 0.7152 * render[..., 1] + 0.0722 * render[..., 2]
            mask = (
                (lr > 0.01) & (lr < 0.6) & (lj > 0.01) & (lj < 0.6)
                & (render > 1e-4).all(-1) & (jpg > 1e-4).all(-1)
            )
            if int(mask.sum()) < 200:
                return None
            dr = float(np.median(
                np.log(render[..., 0][mask] / render[..., 1][mask])
                - np.log(jpg[..., 0][mask] / jpg[..., 1][mask])
            ))
            db = float(np.median(
                np.log(render[..., 2][mask] / render[..., 1][mask])
                - np.log(jpg[..., 2][mask] / jpg[..., 1][mask])
            ))
            cbar = cam3.reshape(-1, 3)[mask.reshape(-1)].mean(axis=0)
            return dr, db, max(abs(dr), abs(db)), cbar

        # Align JPEG aspect to CFA half-planes. Some bodies embed a portrait
        # JPEG while the mosaic buffer stays sensor-native landscape (or the
        # reverse). Stretching across that mismatch poisons the median metric.
        plane_portrait = hmin > wmin
        jpeg_portrait = arr.shape[0] > arr.shape[1]
        if plane_portrait == jpeg_portrait:
            cand_imgs = (arr, cv2.rotate(arr, cv2.ROTATE_180))
        else:
            cand_imgs = (
                cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE),
                cv2.rotate(arr, cv2.ROTATE_90_COUNTERCLOCKWISE),
            )
        best = None  # (first_dist, jpg, lj)
        for src in cand_imgs:
            jpg_t = _build_jpg(src)
            lj_t = (
                0.2126 * jpg_t[..., 0]
                + 0.7152 * jpg_t[..., 1]
                + 0.0722 * jpg_t[..., 2]
            )
            m0 = _render_and_metric(cam_mul, jpg_t, lj_t)
            if m0 is None:
                continue
            dist0 = m0[2]
            if best is None or dist0 < best[0]:
                best = (dist0, jpg_t, lj_t)
        if best is None:
            return None
        first_dist, jpg, lj = best
        if first_dist <= _WB_SANITY_THRESHOLD:
            return None

        # Iterate the mean-vector Newton step: one step lands well below the
        # trigger threshold but can leave a visible residual tint (the solve
        # is exact only for the masked MEAN, while the metric is a median and
        # the mask itself shifts as color converges). 2-3 iterations settle
        # under ~0.06 (imperceptible against JPEG tone-curve noise).
        cur_mul = cam_mul.astype(np.float64).copy()
        for it in range(5):
            m = _render_and_metric(cur_mul, jpg, lj)
            if m is None:
                if it == 0:
                    return None
                break
            dr, db, dist, cbar = m
            if it > 0 and dist <= 0.06:
                break
            # w = (M^-1 (g_srgb * (M cbar))) / cbar -- exact for the mean.
            g_srgb = np.array([np.exp(-dr), 1.0, np.exp(-db)], dtype=np.float64)
            if np.any(cbar <= 0):
                break
            w = m3_inv @ (g_srgb * (m3 @ cbar)) / cbar
            if np.any(~np.isfinite(w)) or np.any(w <= 0):
                break
            w = np.clip(w, 0.3, 3.5)
            cur_mul[0] *= w[0]
            cur_mul[1] *= w[1]
            cur_mul[2] *= w[2]
            cur_mul[3] *= w[1]

        # NOTE (evaluated, rejected): fitting a 3x3 crosstalk matrix against
        # the JPEG on top of this diagonal was tried for the residual
        # per-region tint (quadrant medians +/-0.24 after convergence). The
        # scene's colors are near-collinear so plain LSQ is degenerate, and
        # ridge-regularized fits reduced nothing: the region disagreement
        # comes from Canon's spatially-LOCAL JPEG processing (ALO etc.),
        # which no global transform can match. Diagonal-only is the robust
        # stopping point until LibRaw ships a real profile for the body.
        new_mul = [float(x) for x in cur_mul]
        logger.warning(
            "[FAST_RAW] as-shot WB failed embedded-JPEG sanity for %s "
            "(dist=%.2f, threshold=%.2f) -- corrected multipliers %s -> %s "
            "(LibRaw likely predates this camera's makernote layout)",
            os.path.basename(file_path),
            first_dist,
            _WB_SANITY_THRESHOLD,
            [round(float(x), 1) for x in cam_mul],
            [round(x, 1) for x in new_mul],
        )
        return new_mul
    except Exception as e:
        logger.debug("[FAST_RAW] WB sanity check failed for %s: %s", file_path, e)
        return None


@dataclass
class UnpackedRaw:
    """LibRaw unpack output + derived color metadata, decoupled from rawpy.

    Produced by :func:`unpack_raw` (the 100-400ms I/O+unpack stage, the
    Amdahl floor of every decode). Both display tiers render from one of
    these — :func:`decode_half_from_unpacked` for the fit-view paint and
    :func:`finish_full_decode` for the sensor-resolution tier — so a
    navigation that later zooms (or pauses into the idle full decode) pays
    the unpack exactly once. ~2 bytes/sensor-pixel resident (uint16 mosaic).
    """

    file_path: str
    mosaic: np.ndarray  # uint16 visible-area copy, CFA phase at origin
    pattern: np.ndarray  # 2x2 LibRaw channel indices at visible origin
    pat_str: str  # RGGB-style, guaranteed key of _BAYER_TO_CV2
    black: np.ndarray  # per-channel black levels, shape (4,)
    scale_mul: np.ndarray  # dcraw scale_colors multipliers, shape (4,)
    rgb_cam: np.ndarray  # float32 3x3 camera->sRGB matrix


def params_supported(params: Dict[str, Any], return_linear: bool = False) -> bool:
    """True when the rawpy params match the semantics this module replicates."""
    # Allowed keys for both modes
    allowed = _ALLOWED_PARAM_KEYS.copy()
    if return_linear:
        allowed.add("highlight_mode")

    if set(params) - allowed:
        return False
    if not params.get("use_camera_wb", False):
        return False
    if params.get("use_auto_wb", False):
        return False
    if not params.get("no_auto_bright", False):
        return False
    
    if return_linear:
        if int(params.get("output_bps", 16)) != 16:
            return False
        # Linear adjustments use highlight_mode > 0 (usually 2 for blend)
        hm = params.get("highlight_mode", 0)
        if getattr(hm, "value", hm) not in (0, 1, 2):
            return False
    else:
        if int(params.get("output_bps", 8)) != 8:
            return False
        gamma = tuple(params.get("gamma", (2.222, 4.5)))
        if len(gamma) != 2 or abs(gamma[0] - 2.222) > 1e-6 or abs(gamma[1] - 4.5) > 1e-6:
            return False

    if float(params.get("bright", 1.0)) != 1.0:
        return False
    if int(params.get("user_flip", 0)) != 0:
        return False
    if params.get("half_size", False):
        return False
    return True


def params_supported_half(params: Dict[str, Any], return_linear: bool = False) -> bool:
    """Like :func:`params_supported` but for the half_size display decode."""
    if not params.get("half_size", False):
        return False
    relaxed = {k: v for k, v in params.items() if k != "half_size"}
    return params_supported(relaxed, return_linear=return_linear)


_GAMMA_CURVE16: Optional[np.ndarray] = None


def gamma_curve16() -> np.ndarray:
    """dcraw BT.709 curve, 16-bit in -> 16-bit out (see _gamma_lut8).

    Public: the edit pipeline's display/export encode uses this same curve
    so an edited image at default settings renders identically to browse.
    """
    _gamma_lut8()  # builds both LUTs
    return _GAMMA_CURVE16


def _gamma_lut8() -> np.ndarray:
    """dcraw gamma_curve for gamma=(2.222, 4.5): BT.709 OETF, 16-bit -> 8-bit.

    The toe breakpoint is solved by bisection exactly like dcraw does, and the
    8-bit value is ``curve >> 8`` (truncation) matching dcraw's write_ppm.
    """
    global _GAMMA_LUT8, _GAMMA_CURVE16
    if _GAMMA_LUT8 is not None:
        return _GAMMA_LUT8
    pwr, ts = 1.0 / 2.222, 4.5
    bnd = [0.0, 1.0]
    g2 = 0.0
    for _ in range(48):
        g2 = (bnd[0] + bnd[1]) / 2
        if ((g2 / ts) ** -pwr - 1) / pwr - 1 / g2 > -1:
            bnd[1] = g2
        else:
            bnd[0] = g2
    g3 = g2 / ts
    g4 = g2 * (1 / pwr - 1)
    x = np.arange(65536, dtype=np.float64) / 65535.0
    y = np.where(x < g3, x * ts, (1 + g4) * np.power(np.maximum(x, 1e-12), pwr) - g4)
    curve16 = np.clip(y * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    _GAMMA_CURVE16 = curve16
    _GAMMA_LUT8 = (curve16 >> 8).astype(np.uint8)
    return _GAMMA_LUT8


def _rgb_cam_from_cam_xyz(cam_xyz: np.ndarray) -> Optional[np.ndarray]:
    """dcraw cam_xyz_coeff: row-normalize the forward matrix, then pinv.

    Rows of the result sum to 1 (camera white -> sRGB white); getting this
    wrong (e.g. normalizing the inverse, or transposing) produces global
    channel gains — the failure mode is easy to spot because
    ``raw.daylight_whitebalance`` equals 1/rowsum of the forward matrix and
    can be used to cross-check.
    """
    cam = np.asarray(cam_xyz, dtype=np.float64)[:3, :3]
    if cam.shape != (3, 3) or np.allclose(cam, 0.0):
        return None
    cam_rgb = cam @ _XYZ_RGB
    rowsum = cam_rgb.sum(axis=1)
    if np.any(rowsum <= 0):
        return None
    cam_rgb = cam_rgb / rowsum[:, None]
    try:
        return np.linalg.pinv(cam_rgb).astype(np.float32)
    except np.linalg.LinAlgError:
        return None


def _visible_pattern(raw: Any) -> Optional[np.ndarray]:
    """2x2 CFA channel indices at the *visible-area* origin.

    Uses ``raw_colors_visible`` rather than ``raw_pattern`` because odd
    top/left sensor margins shift the pattern phase between the full frame
    and the visible crop.
    """
    try:
        pat = np.asarray(raw.raw_pattern)
        if pat.shape != (2, 2):
            return None  # X-Trans (6x6) and friends
        vis = np.asarray(raw.raw_colors_visible[:2, :2])
        if vis.shape != (2, 2):
            return None
        return vis.astype(np.int32)
    except Exception:
        return None


def _pattern_string(pattern: np.ndarray) -> Optional[str]:
    """RGGB-style string from a 2x2 of LibRaw channel indices (3=second green)."""
    names = {0: "R", 1: "G", 2: "B", 3: "G"}
    try:
        s = "".join(names[int(pattern[y, x])] for y in (0, 1) for x in (0, 1))
    except KeyError:
        return None
    return s if s in _BAYER_TO_CV2 else None


def _resolve_black(raw: Any, cblack: np.ndarray) -> np.ndarray:
    """True per-channel black level, repairing LibRaw's incomplete report.

    LibRaw splits black into a base (``color.black``) plus per-channel deltas
    (``color.cblack[0..3]``), and rawpy's ``black_level_per_channel`` exposes
    only the deltas. On bodies LibRaw parses fully the deltas carry the whole
    level (EOS R5: [512]*4) and this returns them untouched. On a body whose
    makernote layout the bundled LibRaw predates, the base holds the level and
    the deltas come back as small ISO trims over a ZERO floor -- the EOS R6
    Mark III reports [0, 33, 100, 67] against a true black of 2049. Subtracting
    only those leaves a ~2048/16383 = 12%-of-range pedestal in every channel,
    which the WB multipliers then amplify unequally into a heavy color cast:
    the "new camera color shift". (rawpy's own postprocess renders these files
    with the same pedestal, so it is no use as a reference here -- the camera's
    embedded JPEG is, and the repaired black is what matches it.)

    ``cblack.min() <= 0`` is the signature of the missing base: a real sensor
    floor is never zero. Only then do we measure the sensor's masked
    optical-black columns, which are the physical zero reference, and use that
    flat level for all four channels (measured per CFA phase, the R6 III's
    phases agree to 1 LSB -- LibRaw's lopsided deltas are NOT a per-channel
    floor and must not be added on top).

    The measurement is deliberately gated rather than universal: on a Sony ARW
    in the sample set the left margin reads 884 against a true black of 512, so
    it is not clean masked black on every body. Bodies LibRaw already gets right
    keep LibRaw's answer and cannot regress.
    """
    try:
        if float(cblack.min()) > 0.0:
            return cblack  # LibRaw's level is complete -- trust it.
        s = raw.sizes
        lm = int(getattr(s, "left_margin", 0) or 0)
        full = raw.raw_image
        if full is None or full.ndim != 2 or lm < 24:
            return cblack
        # Skip the outermost columns (often tapered/dead rather than a clean
        # masked reference) and subsample rows: this is a median over a uniform
        # strip, not a measurement that needs every pixel.
        strip = full[::4, 8 : lm - 8]
        if strip.size < 1024:
            return cblack
        level = float(np.median(strip))
        white = float(raw.white_level)
        if not (0.0 < level < white * 0.5):
            return cblack  # implausible -- keep LibRaw's answer.
        logger.info(
            "[FAST_RAW] LibRaw reported an incomplete black level %s; using the "
            "measured optical-black floor %.0f (body newer than this LibRaw)",
            [round(float(x)) for x in cblack],
            level,
        )
        return np.full(4, level, dtype=np.float64)
    except Exception:
        return cblack


def unpack_raw(
    file_path: str,
    rawpy_lock: Optional[Any] = None,
) -> Optional[UnpackedRaw]:
    """LibRaw open+unpack and color-metadata derivation, no pixel math.

    Returns None for anything the fast pixel path can't handle (X-Trans,
    4-color CFA, linear/float DNG, missing camera WB, ...) — never raises.
    """
    if _env_disabled():
        return None
    from perf_metrics import perf_mark

    _t_unpack = time.perf_counter()
    try:
        import rawpy

        if rawpy_lock is not None:
            with rawpy_lock:
                raw_ctx = rawpy.imread(file_path)
        else:
            raw_ctx = rawpy.imread(file_path)
        with raw_ctx as raw:
            if getattr(raw, "num_colors", 0) != 3:
                return None
            pattern = _visible_pattern(raw)
            if pattern is None:
                return None
            pat_str = _pattern_string(pattern)
            if pat_str is None:
                return None
            mosaic = raw.raw_image_visible
            if mosaic is None or mosaic.ndim != 2 or mosaic.dtype != np.uint16:
                return None  # linear/float DNG, monochrome, sRAW
            # Copy while the rawpy handle is open (visible view borrows its
            # buffer). Chunked like the gathers below to bound GIL holds.
            mosaic = _chunked_copy(mosaic)
            black = np.asarray(raw.black_level_per_channel, dtype=np.float64)
            black = _resolve_black(raw, black)
            white = float(raw.white_level)
            cam_mul = np.asarray(raw.camera_whitebalance, dtype=np.float64).copy()
            rgb_cam = _rgb_cam_from_cam_xyz(raw.rgb_xyz_matrix)
            # WB sanity needs the embedded JPEG; extract while the handle is
            # open, but only on this file's first unpack this session, and
            # only when the camera model isn't already verified clean (the
            # per-model verdict skips the ~38ms thumb+metric cost for every
            # subsequent file from a well-parsed body).
            thumb_bytes = None
            wb_key = os.path.normcase(os.path.abspath(file_path))
            model_key = _wb_model_key(rgb_cam, mosaic, black, white)
            if (
                wb_sanity_enabled()
                and wb_key not in _WB_CORRECTION_CACHE
                and _WB_MODEL_VERDICT.get(model_key) is not False
            ):
                try:
                    t = raw.extract_thumb()
                    if getattr(t, "format", None) == rawpy.ThumbFormat.JPEG:
                        thumb_bytes = t.data
                except Exception:
                    thumb_bytes = None
        if rgb_cam is None or black.shape[0] < 4 or cam_mul.shape[0] < 4:
            return None
        if cam_mul[0] <= 0 or cam_mul[1] <= 0 or cam_mul[2] <= 0:
            return None  # no camera WB recorded; rawpy path handles its fallback
        if cam_mul[3] <= 0:
            cam_mul[3] = cam_mul[1]

        # dcraw scale_colors (highlight_mode=0): divide by the minimum so all
        # multipliers >= 1, scale to 65535 over the effective range, clip in
        # camera space. LibRaw adjust_maximum (thr=0.75) replicated for the
        # saturation point.
        pre_mul = cam_mul / cam_mul.min()
        # Plain reduction, not chunked: unlike the LUT-gather chunking below
        # (a genuinely measured 60-120ms GIL stall from fancy indexing --
        # see finish_full_decode's comment), a simple ndarray.max() releases
        # the GIL for its C-level reduction loop almost continuously.
        # Verified: on a 100MP array, plain .max() showed a 0.03ms max
        # scheduling gap for a concurrent Python thread vs 0.94ms for the
        # chunked-generator version, at the same ~4ms wall time -- the
        # chunking here bought nothing, just copied the nearby gather's
        # pattern onto a different operation with different GIL behavior.
        data_max = float(mosaic.max())
        maximum = white
        if 0 < data_max < maximum and data_max > maximum * 0.75:
            maximum = data_max
        denom = maximum - black.min()
        if denom <= 0:
            return None
        scale_mul = pre_mul * 65535.0 / denom

        # As-shot WB sanity vs the camera's own JPEG (see get_corrected_camera_wb).
        if wb_sanity_enabled():
            if wb_key not in _WB_CORRECTION_CACHE:
                if _WB_MODEL_VERDICT.get(model_key) is False:
                    # Model already verified clean this session -- skip.
                    _WB_CORRECTION_CACHE[wb_key] = None
                else:
                    result = _wb_correction_from_jpeg(
                        file_path, thumb_bytes, mosaic, pattern, black,
                        scale_mul, rgb_cam, cam_mul,
                    )
                    _WB_CORRECTION_CACHE[wb_key] = result
                    # Record the model verdict only from a real measurement:
                    # a missing/undecodable thumb must not mark a potentially
                    # misparsed model as clean for the whole session.
                    if result is not None:
                        _WB_MODEL_VERDICT[model_key] = True
                    elif thumb_bytes:
                        _WB_MODEL_VERDICT.setdefault(model_key, False)
            corrected = _WB_CORRECTION_CACHE[wb_key]
            if corrected is not None:
                cam_mul = np.asarray(corrected, dtype=np.float64)
                pre_mul = cam_mul / cam_mul.min()
                scale_mul = pre_mul * 65535.0 / denom
        perf_mark(
            "unpack",
            (time.perf_counter() - _t_unpack) * 1000.0,
            file_path,
            mp=mosaic.shape[0] * mosaic.shape[1] / 1e6,
        )
        return UnpackedRaw(
            file_path=file_path,
            mosaic=mosaic,
            pattern=pattern,
            pat_str=pat_str,
            black=black,
            scale_mul=scale_mul,
            rgb_cam=rgb_cam,
        )
    except Exception as e:
        logger.warning("[FAST_RAW] unpack failed for %s: %s", file_path, e)
        return None


def finish_full_decode(
    unpacked: UnpackedRaw,
    cancel_check: Optional[Callable[[], bool]] = None,
    return_linear: bool = False,
    *,
    prefer_gpu: bool = False,
) -> Optional[np.ndarray]:
    """Full-resolution pixel math (scale/demosaic/matrix/gamma) on an unpack.

    Returns uint8 sRGB in sensor orientation, or a ``DeviceRgb`` CUDA wrapper
    when ``RAWVIEWER_GPU_CUDA_GL=1`` and the GPU path retains the device buffer.
    Returns None when cv2 is unavailable. Raises :class:`DecodeCancelled`
    between chunks when ``cancel_check`` fires.

    ``prefer_gpu`` defaults to False: measured no speed benefit from the MPS
    path on real files (45-46MP CR3/NEF, ~358-365ms either way, GPU
    occasionally slower after warmup) -- consistent with this module's own
    top-of-file docstring, which already concluded the same for the earlier
    OpenCL/Metal PoC ("19ms GPU vs 17ms CPU... adds dependency weight for no
    gain"). GPU decode also feeds the MPS unified-memory pressure that
    needed a dedicated release_cached_gpu_memory() mitigation elsewhere
    (image_cache.py) -- avoiding it by default when it buys nothing removes
    that cost too. Set True explicitly for a caller that specifically wants
    to exercise the GPU path (e.g. stashed full-tier zoom when
    ``prefer_gpu_decode_enabled()``, or a parity/benchmark script).
    """
    if prefer_gpu:
        try:
            mp_cap = gpu_decode_max_megapixels()
            mosaic_mp = (
                float(unpacked.mosaic.shape[0] * unpacked.mosaic.shape[1]) / 1e6
                if getattr(unpacked, "mosaic", None) is not None
                else 0.0
            )
            if mp_cap > 0 and mosaic_mp > mp_cap:
                logger.info(
                    "[FAST_RAW] Skipping GPU finish_full for %s "
                    "(%.1f MP > %.1f MP cap); using CPU",
                    os.path.basename(unpacked.file_path or ""),
                    mosaic_mp,
                    mp_cap,
                )
            else:
                import torch_bootstrap

                if torch_bootstrap.wait_for_gpu_backend_ready(timeout=2.0):
                    from gpu_raw_processor import try_gpu_decode_from_unpacked

                    gpu_out = try_gpu_decode_from_unpacked(
                        unpacked, cancel_check=cancel_check, return_linear=return_linear
                    )
                    if gpu_out is not None:
                        return gpu_out
        except DecodeCancelled:
            raise
        except Exception as e:
            logger.debug("[FAST_RAW] GPU finish_full_decode skipped: %s", e)

    cv2 = _import_cv2()
    if cv2 is None:
        return None
    from perf_metrics import perf_mark

    _t0 = time.perf_counter()

    def _abort_if_cancelled() -> None:
        if cancel_check is not None and cancel_check():
            raise DecodeCancelled(unpacked.file_path)

    # The numpy LUT gathers below hold the GIL (unlike cv2 calls and
    # rawpy's postprocess, which release it) -- a single full-frame
    # gather is a 60-120ms GIL stall that visibly freezes the Qt event
    # loop when this runs on an in-process worker thread (the default on
    # macOS, where the LibRaw process pool is off). Chunking rows bounds
    # each GIL hold to a few ms; the interpreter can schedule the GUI
    # thread between chunks. Output is byte-identical to the unchunked
    # version. Chunk height must stay even to preserve the CFA phase.
    chunk = 256
    mosaic, pattern = unpacked.mosaic, unpacked.pattern
    luts = _scale_luts(unpacked)
    h = mosaic.shape[0]
    scaled = np.empty_like(mosaic)
    for y0 in range(0, h, chunk):
        _abort_if_cancelled()
        src = mosaic[y0 : y0 + chunk]
        dst = scaled[y0 : y0 + chunk]
        for (dy, dx), ci in np.ndenumerate(pattern):
            dst[dy::2, dx::2] = luts[ci][src[dy::2, dx::2]]

    _abort_if_cancelled()
    code = getattr(cv2, f"COLOR_Bayer{_BAYER_TO_CV2[unpacked.pat_str]}2RGB_EA")
    rgb16 = cv2.demosaicing(scaled, code)
    _abort_if_cancelled()
    # uint16 in/out: cv2.transform saturates to [0, 65535] for free
    # (within 1 LSB of the float path on ~0.001% of pixels).
    srgb16 = cv2.transform(rgb16, unpacked.rgb_cam)
    if return_linear:
        perf_mark(
            "decode_full_cpu_linear",
            (time.perf_counter() - _t0) * 1000.0,
            unpacked.file_path,
            mp=srgb16.shape[0] * srgb16.shape[1] / 1e6,
        )
        return srgb16

    glut = _gamma_lut8()
    out = np.empty(srgb16.shape, dtype=np.uint8)
    for y0 in range(0, srgb16.shape[0], chunk):
        _abort_if_cancelled()
        out[y0 : y0 + chunk] = glut[srgb16[y0 : y0 + chunk]]
    perf_mark(
        "decode_full_cpu",
        (time.perf_counter() - _t0) * 1000.0,
        unpacked.file_path,
        mp=out.shape[0] * out.shape[1] / 1e6,
    )
    return out


def _scale_luts(unpacked: UnpackedRaw) -> list:
    """Per-channel 16-bit LUTs implementing dcraw black-subtract + WB scale."""
    x = np.arange(65536, dtype=np.float64)
    return [
        np.clip((x - unpacked.black[c]) * unpacked.scale_mul[c], 0, 65535).astype(
            np.uint16
        )
        for c in range(4)
    ]


def decode_half_from_unpacked(
    unpacked: UnpackedRaw,
    cancel_check: Optional[Callable[[], bool]] = None,
    return_linear: bool = False,
) -> Optional[np.ndarray]:
    """Demosaic-free half-size tier from an existing unpack (fit-view paint)."""
    from perf_metrics import perf_timer

    with perf_timer("decode_half", unpacked.file_path):
        return _decode_half_from_unpacked_impl(unpacked, cancel_check, return_linear)


def _decode_half_from_unpacked_impl(
    unpacked: UnpackedRaw,
    cancel_check: Optional[Callable[[], bool]] = None,
    return_linear: bool = False,
) -> Optional[np.ndarray]:
    cv2 = _import_cv2()
    if cv2 is None:
        return None

    def _abort_if_cancelled() -> None:
        if cancel_check is not None and cancel_check():
            raise DecodeCancelled(unpacked.file_path)

    mosaic = unpacked.mosaic
    h, w = mosaic.shape
    if h % 2 or w % 2:
        # rawpy half_size ceils odd dims and zero-fills the missing samples
        # (raw 0 -> clips to black after black-subtract, greens average the
        # zero in) -- parity-verified against rawpy on a 5463x8191 CR3.
        mosaic = np.pad(mosaic, ((0, h % 2), (0, w % 2)), mode="constant")
    acc = {"R": None, "B": None}
    greens = []
    for (dy, dx), ci in np.ndenumerate(unpacked.pattern):
        _abort_if_cancelled()
        # dcraw scale_colors as one saturating SIMD op: uint16 saturation
        # is exactly the camera-space [0, 65535] clip (the step that keeps
        # clipped highlights neutral). ~7x faster than the numpy float
        # equivalent; quantization is <=0.5 LSB16 pre-matrix, parity-gated.
        plane = np.ascontiguousarray(mosaic[dy::2, dx::2])
        sc = float(unpacked.scale_mul[ci])
        scaled = cv2.addWeighted(plane, sc, plane, 0.0, -float(unpacked.black[ci]) * sc)
        if ci == 0:
            acc["R"] = scaled
        elif ci == 2:
            acc["B"] = scaled
        else:
            greens.append(scaled)
    if acc["R"] is None or acc["B"] is None or not greens:
        return None
    green = (
        greens[0]
        if len(greens) == 1
        else cv2.addWeighted(greens[0], 0.5, greens[1], 0.5, 0.0)
    )
    _abort_if_cancelled()
    cam3 = cv2.merge([acc["R"], green, acc["B"]])
    # uint16 in/out: saturates to [0, 65535], same as the full-res path.
    srgb16 = cv2.transform(cam3, unpacked.rgb_cam)
    _abort_if_cancelled()
    if return_linear:
        return srgb16
    return _gamma_lut8()[srgb16]


def try_fast_raw_decode(
    file_path: str,
    params: Dict[str, Any],
    rawpy_lock: Optional[Any] = None,
    cancel_check: Optional[Any] = None,
    return_linear: bool = False,
) -> Optional[np.ndarray]:
    """Full-resolution decode matching rawpy semantics for ``params``.

    Returns an (H, W, 3) uint8 sRGB array in sensor orientation (user_flip=0
    semantics — the caller applies EXIF orientation, same as the rawpy path),
    or None to signal "use the rawpy fallback" (unsupported sensor/params, or
    any error — this function never raises except DecodeCancelled).
    """
    if _env_disabled():
        return None
    try:
        is_half = params.get("half_size", False)
        supported = params_supported_half(params, return_linear=return_linear) if is_half else params_supported(params, return_linear=return_linear)
        if not supported:
            logger.info(
                "[FAST_RAW] params not supported, using rawpy: %s",
                {k: v for k, v in params.items() if k != "demosaic_algorithm"},
            )
            return None
        t0 = time.perf_counter()
        unpacked = unpack_raw(file_path, rawpy_lock=rawpy_lock)
        if unpacked is None:
            return None
            
        # If half_size is requested, try to run the fast half-size decode
        if params.get("half_size", False):
            # GPU half-size isn't implemented yet, so use CPU
            out = decode_half_from_unpacked(unpacked, cancel_check=cancel_check, return_linear=return_linear)
            if out is not None:
                logger.info(
                    "[FAST_RAW] %s decoded half-size in %.0fms (pattern=%s)",
                    os.path.basename(file_path),
                    (time.perf_counter() - t0) * 1000.0,
                    unpacked.pat_str,
                )
                return out

        if prefer_gpu_decode_enabled():
            try:
                mp_cap = gpu_decode_max_megapixels()
                mosaic_mp = (
                    float(unpacked.mosaic.shape[0] * unpacked.mosaic.shape[1]) / 1e6
                    if getattr(unpacked, "mosaic", None) is not None
                    else 0.0
                )
                if mp_cap > 0 and mosaic_mp > mp_cap:
                    logger.info(
                        "[FAST_RAW] Skipping GPU demosaic for %s (%.1f MP > %.1f MP cap); using CPU",
                        os.path.basename(file_path),
                        mosaic_mp,
                        mp_cap,
                    )
                else:
                    # torch/kornia must never be imported for the first time on this
                    # (background) thread -- macOS aborts the process if PyTorch's
                    # OpenMP runtime initializes off the main thread. The main thread
                    # imports gpu_raw_processor right after showing the window
                    # (torch_bootstrap.py); wait for that here instead of importing
                    # it ourselves. A timeout falls back to CPU decode below via the
                    # existing `except Exception` path, same as any other GPU-decode
                    # failure.
                    import torch_bootstrap
                    if not torch_bootstrap.wait_for_gpu_backend_ready(timeout=10.0):
                        raise RuntimeError("GPU backend not ready within timeout")
                    from gpu_raw_processor import try_gpu_decode_from_unpacked
                    gpu_out = try_gpu_decode_from_unpacked(unpacked, cancel_check=cancel_check, return_linear=return_linear)
                    if gpu_out is not None:
                        logger.info(
                            "[FAST_RAW] %s decoded (GPU) %dx%d in %.0fms (pattern=%s)",
                            os.path.basename(file_path),
                            gpu_out.shape[1],
                            gpu_out.shape[0],
                            (time.perf_counter() - t0) * 1000.0,
                            unpacked.pat_str,
                        )
                        return gpu_out
            except DecodeCancelled:
                raise
            except Exception as e:
                logger.warning("[FAST_RAW] GPU decode attempt failed, falling back to CPU: %s", e)

        out = finish_full_decode(
            unpacked, cancel_check=cancel_check, return_linear=return_linear, prefer_gpu=False
        )
        if out is None:
            return None
        logger.info(
            "[FAST_RAW] %s decoded %dx%d in %.0fms (pattern=%s)",
            os.path.basename(file_path),
            out.shape[1],
            out.shape[0],
            (time.perf_counter() - t0) * 1000.0,
            unpacked.pat_str,
        )
        return out
    except DecodeCancelled:
        raise
    except Exception as e:
        logger.warning("[FAST_RAW] failed for %s, falling back to rawpy: %s", file_path, e)
        return None


def halfsize_reference_decode(file_path: str) -> Optional[np.ndarray]:
    """Demosaic-free half-size decode for the parity gate (tests only).

    2x2-bins the mosaic exactly like rawpy's half_size mode (greens averaged),
    then runs the same color math as the fast path. Must match
    ``rawpy.postprocess(half_size=True, use_camera_wb=True,
    no_auto_bright=True, output_bps=8, user_flip=0)`` within +/-1 LSB —
    demosaic is bypassed on both sides, so any difference is a color bug.
    """
    try:
        import rawpy

        with rawpy.imread(file_path) as raw:
            if getattr(raw, "num_colors", 0) != 3:
                return None
            if np.asarray(raw.raw_pattern).shape != (2, 2):
                return None
            mosaic = raw.raw_image_visible
            if mosaic is None or mosaic.ndim != 2:
                return None
            mosaic = mosaic.astype(np.float32)
            colors = np.asarray(raw.raw_colors_visible)
            black = np.asarray(raw.black_level_per_channel, dtype=np.float64)
            black = _resolve_black(raw, black)
            white = float(raw.white_level)
            cam_mul = np.asarray(raw.camera_whitebalance, dtype=np.float64).copy()
            rgb_cam = _rgb_cam_from_cam_xyz(raw.rgb_xyz_matrix)
        if rgb_cam is None:
            return None
        if cam_mul[0] <= 0 or cam_mul[1] <= 0 or cam_mul[2] <= 0:
            return None
        if cam_mul[3] <= 0:
            cam_mul[3] = cam_mul[1]
        pre_mul = cam_mul / cam_mul.min()
        data_max = float(mosaic.max())
        maximum = white
        if 0 < data_max < maximum and data_max > maximum * 0.75:
            maximum = data_max
        scale_mul = pre_mul * 65535.0 / (maximum - black.min())

        h, w = mosaic.shape
        h2, w2 = (h + 1) // 2, (w + 1) // 2
        mp = np.zeros((h2 * 2, w2 * 2), np.float32)
        mp[:h, :w] = mosaic
        cp = np.full((h2 * 2, w2 * 2), -1, np.int8)
        cp[:h, :w] = colors
        m = mp.reshape(h2, 2, w2, 2)
        c = cp.reshape(h2, 2, w2, 2)

        # Accumulate by CFA *position role* (R / B / average-of-greens) so
        # sensors indexing both greens as channel 1 work the same as RGBG.
        acc = {"R": None, "B": None}
        greens = []
        for ci in range(4):
            mask = c == ci
            cnt = mask.sum(axis=(1, 3))
            if not cnt.any():
                continue
            vals = np.where(mask, m, 0).sum(axis=(1, 3)) / np.maximum(cnt, 1)
            vals = np.clip((vals - black[ci]) * scale_mul[ci], 0, 65535)
            if ci == 0:
                acc["R"] = vals
            elif ci == 2:
                acc["B"] = vals
            else:
                # weight greens by how many positions in the 2x2 carry them
                greens.append((vals, float(cnt.max())))
        if acc["R"] is None or acc["B"] is None or not greens:
            return None
        gsum = sum(v * n for v, n in greens)
        gn = sum(n for _, n in greens)
        cam3 = np.stack([acc["R"], gsum / gn, acc["B"]], axis=-1)
        lin = np.clip(cam3 @ rgb_cam[:, :3].T, 0, 65535)
        return _gamma_lut8()[(lin + 0.5).astype(np.uint16)]
    except Exception as e:
        logger.warning("[FAST_RAW] halfsize reference failed for %s: %s", file_path, e)
        return None
