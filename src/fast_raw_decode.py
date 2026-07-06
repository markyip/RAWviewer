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


def params_supported(params: Dict[str, Any]) -> bool:
    """True when the rawpy params match the semantics this module replicates."""
    if set(params) - _ALLOWED_PARAM_KEYS:
        return False
    if not params.get("use_camera_wb", False):
        return False
    if params.get("use_auto_wb", False):
        return False
    if not params.get("no_auto_bright", False):
        return False
    if int(params.get("output_bps", 8)) != 8:
        return False
    if float(params.get("bright", 1.0)) != 1.0:
        return False
    if int(params.get("user_flip", 0)) != 0:
        return False
    if params.get("half_size", False):
        return False
    gamma = tuple(params.get("gamma", (2.222, 4.5)))
    if len(gamma) != 2 or abs(gamma[0] - 2.222) > 1e-6 or abs(gamma[1] - 4.5) > 1e-6:
        return False
    return True


def params_supported_half(params: Dict[str, Any]) -> bool:
    """Like :func:`params_supported` but for the half_size display decode."""
    if not params.get("half_size", False):
        return False
    relaxed = {k: v for k, v in params.items() if k != "half_size"}
    return params_supported(relaxed)


def _gamma_lut8() -> np.ndarray:
    """dcraw gamma_curve for gamma=(2.222, 4.5): BT.709 OETF, 16-bit -> 8-bit.

    The toe breakpoint is solved by bisection exactly like dcraw does, and the
    8-bit value is ``curve >> 8`` (truncation) matching dcraw's write_ppm.
    """
    global _GAMMA_LUT8
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
            white = float(raw.white_level)
            cam_mul = np.asarray(raw.camera_whitebalance, dtype=np.float64).copy()
            rgb_cam = _rgb_cam_from_cam_xyz(raw.rgb_xyz_matrix)
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
        data_max = float(max(mosaic[y : y + 256].max() for y in range(0, mosaic.shape[0], 256)))
        maximum = white
        if 0 < data_max < maximum and data_max > maximum * 0.75:
            maximum = data_max
        denom = maximum - black.min()
        if denom <= 0:
            return None
        scale_mul = pre_mul * 65535.0 / denom
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
) -> Optional[np.ndarray]:
    """Full-resolution pixel math (scale/demosaic/matrix/gamma) on an unpack.

    Returns uint8 sRGB in sensor orientation, or None when cv2 is
    unavailable. Raises :class:`DecodeCancelled` between chunks when
    ``cancel_check`` fires.
    """
    cv2 = _import_cv2()
    if cv2 is None:
        return None

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
    glut = _gamma_lut8()
    out = np.empty(srgb16.shape, dtype=np.uint8)
    for y0 in range(0, srgb16.shape[0], chunk):
        _abort_if_cancelled()
        out[y0 : y0 + chunk] = glut[srgb16[y0 : y0 + chunk]]
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
) -> Optional[np.ndarray]:
    """Demosaic-free half-size tier from an existing unpack (fit-view paint).

    Same 2x2 binning + color math as rawpy's ``half_size=True`` (greens
    averaged; parity gate holds it to +/-1 8-bit LSB), but reusing the
    already-unpacked mosaic — its marginal cost is quarter-resolution pixel
    math only, so a fit-view paint plus a later full tier pays LibRaw's
    unpack once instead of twice. Returns None when cv2 is unavailable.
    """
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
    return _gamma_lut8()[srgb16]


def try_fast_raw_decode(
    file_path: str,
    params: Dict[str, Any],
    rawpy_lock: Optional[Any] = None,
    cancel_check: Optional[Any] = None,
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
        if not params_supported(params):
            logger.info(
                "[FAST_RAW] params not supported, using rawpy: %s",
                {k: v for k, v in params.items() if k != "demosaic_algorithm"},
            )
            return None
        t0 = time.perf_counter()
        unpacked = unpack_raw(file_path, rawpy_lock=rawpy_lock)
        if unpacked is None:
            return None
            
        try:
            from gpu_raw_processor import try_gpu_decode_from_unpacked
            gpu_out = try_gpu_decode_from_unpacked(unpacked, cancel_check=cancel_check)
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
            
        out = finish_full_decode(unpacked, cancel_check=cancel_check)
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
