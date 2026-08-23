"""HDR and Panorama stitching engine for RAWviewer.

Domain contract: every engine here expects **display-referred** frames (uint8,
or uint16 carrying the dcraw BT.709 encode). The Adjust-panel edit base is
scene-linear, so the caller converts once via :func:`to_display_referred`
before handing frames over. This is not cosmetic -- exposure fusion's
well-exposedness Gaussians are defined on display-referred values, MTB's
``exclude_range`` band is measured in 8-bit code values, and the panorama
Stitcher quantizes to 8 bits internally. Fed scene-linear data, all three
misbehave in ways that never raise.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Dict

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Every frame stays resident with working copies on top, and merges now run at
# full resolution. Cap the set (override for machines with the RAM to spare).
try:
    _MAX_MERGE_FRAMES = max(2, int(os.environ.get("RAWVIEWER_MERGE_MAX_FRAMES", "48")))
except ValueError:
    _MAX_MERGE_FRAMES = 48


@dataclass
class StitchResult:
    """Result of an HDR, Panorama, or HDR Panorama stitching operation."""
    success: bool
    image: Optional[np.ndarray] = None
    output_path: str = ""
    error_message: str = ""
    rejected_paths: List[str] = None

    def __post_init__(self):
        if self.rejected_paths is None:
            self.rejected_paths = []


def to_display_referred(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Scene-linear edit base -> display-referred uint16 (browse/Adjust parity).

    Applies the same dcraw BT.709 encode the viewer and the Adjust panel use
    (``fast_raw_decode.gamma_curve16``, the curve behind
    ``raw_edit_pipeline.linear_to_display_uint8``), so a merged frame matches
    what the user saw while browsing the sources.

    uint8 input is passed through untouched: 8-bit bases in this app are
    embedded JPEGs and ordinary rasters, which are display-referred already --
    encoding them again would wash them out.
    """
    if img is None:
        return None
    if img.dtype == np.uint8:
        return img

    from fast_raw_decode import gamma_curve16

    if img.dtype == np.uint16:
        lin = img.astype(np.float32) / 65535.0
    else:
        # Scene-linear float where 1.0 = camera clip; values above it are the
        # Blend-reconstructed highlight headroom, which display clips to white
        # exactly as browse does.
        lin = np.clip(img.astype(np.float32), 0.0, 1.0)
    idx = np.clip(lin * 65535.0 + 0.5, 0.0, 65535.0).astype(np.uint16)
    return gamma_curve16()[idx]


def _luma_u8(img: np.ndarray) -> np.ndarray:
    """8-bit luma of a display-referred frame, whatever its depth."""
    from focus_stacking import _to_gray_u8

    return _to_gray_u8(img)


def _translate(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Exact integer translation (nearest sampling, zero fill outside)."""
    if dx == 0 and dy == 0:
        return img
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        img, m, (img.shape[1], img.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def align_hdr_images(images: List[np.ndarray]) -> Tuple[List[np.ndarray], bool]:
    """Align handheld bracketed exposures using OpenCV Median Threshold Bitmap.

    Frames must be display-referred (see the module docstring). Returns
    (aligned_images, success); on failure the inputs come back untouched so the
    merge can still run on unaligned frames.

    Two things this deliberately does NOT do, both of which used to make it a
    silent no-op:

    * ``AlignMTB.process(src, dst)`` does not fill a Python list passed as
      ``dst`` -- it only works when the SAME list is passed for both, and
      returns None either way. Every merge was therefore fusing an empty list.
      Shifts are taken from ``calculateShift`` instead, which has a plain
      return value there is no way to misread.
    * MTB's bitmaps are 8-bit only: handed uint16 or float frames, ``process``
      returns without aligning and without raising. The shift is measured on
      8-bit luma and applied to the frames at their own depth.
    """
    if not images or len(images) < 2:
        return list(images), True

    try:
        # cut=False: the crop is done here, from the shifts we computed, so the
        # reference is cropped identically to the frames that moved.
        align_mtb = cv2.createAlignMTB(max_bits=4, exclude_range=4, cut=False)
        ref_gray = _luma_u8(images[0])
        shifts: List[Tuple[int, int]] = [(0, 0)]
        for idx in range(1, len(images)):
            dx, dy = align_mtb.calculateShift(ref_gray, _luma_u8(images[idx]))
            shifts.append((int(dx), int(dy)))

        h, w = images[0].shape[:2]
        # Common valid rectangle: a frame shifted by +dx has no data for the
        # first dx columns, so keep only what every frame covers.
        x0 = max(0, max(s[0] for s in shifts))
        x1 = w + min(0, min(s[0] for s in shifts))
        y0 = max(0, max(s[1] for s in shifts))
        y1 = h + min(0, min(s[1] for s in shifts))
        if (x1 - x0) < w * 0.5 or (y1 - y0) < h * 0.5:
            logger.warning(
                "MTB shifts %s would crop away more than half the frame; "
                "merging unaligned", shifts,
            )
            return list(images), False

        aligned = [_translate(im, dx, dy)[y0:y1, x0:x1] for im, (dx, dy) in zip(images, shifts)]
        logger.info("MTB aligned %d frames, shifts=%s, crop=%dx%d",
                    len(aligned), shifts[1:], x1 - x0, y1 - y0)
        return aligned, True
    except Exception as exc:
        logger.warning("MTB HDR alignment failed: %s", exc)
        return list(images), False


def crop_panorama_borders(img: np.ndarray) -> np.ndarray:
    """Crop the warped black border off a panorama, leaving a clean rectangle.

    Both steps key off the stitcher's *coverage* rather than off luminance,
    because a photograph legitimately contains black pixels:

    * The largest contour is used, not ``contours[0]``. RETR_EXTERNAL returns
      every bright blob, and on a dark-background panorama (night sky, black
      backdrop) the first one OpenCV happened to find was a speck -- a
      600x1200 stitch cropped to 1x2 px.
    * The inner rectangle is measured on the FILLED contour mask, not on the
      grey values. A single black scene pixel used to disqualify its entire
      row, so a normal frame lost ~38% of its width.
    """
    if img is None or img.size == 0:
        return img
    try:
        if img.ndim == 3:
            gray = cv2.cvtColor(
                img, cv2.COLOR_RGB2GRAY if img.shape[2] == 3 else cv2.COLOR_RGBA2GRAY
            )
        else:
            gray = img
        if gray.dtype != np.uint8:
            gray = _luma_u8(img)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        # The warp border is the black that REACHES THE EDGE of the canvas.
        # Flood-filling it from the border separates "no data here" from a
        # black object in the photograph, which no threshold can tell apart:
        # dark scene pixels enclosed by picture stay covered, and only the
        # border-connected black is carved away.
        border_black = np.zeros((thresh.shape[0] + 2, thresh.shape[1] + 2), np.uint8)
        filled = thresh.copy()
        for seed in ((0, 0), (thresh.shape[1] - 1, 0),
                     (0, thresh.shape[0] - 1), (thresh.shape[1] - 1, thresh.shape[0] - 1)):
            if filled[seed[1], seed[0]] == 0:
                cv2.floodFill(filled, border_black, seed, 255)
        # `filled` is now coverage plus the border region; the border region
        # alone is what floodFill added.
        thresh = np.where(filled > 0, 255, 0).astype(np.uint8)
        thresh[border_black[1:-1, 1:-1] > 0] = 0
        # Bridge the 1px gaps warping leaves inside the covered area so the
        # outer contour traces the panorama, not its speckle.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img
        biggest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest)
        if w < 2 or h < 2:
            return img
        cropped = img[y:y + h, x:x + w]

        coverage = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(coverage, [biggest], -1, 255, thickness=cv2.FILLED)
        sub = coverage[y:y + h, x:x + w]
        valid_rows = np.where(sub.min(axis=1) > 0)[0]
        valid_cols = np.where(sub.min(axis=0) > 0)[0]
        if len(valid_rows) > 0 and len(valid_cols) > 0:
            r_start, r_end = valid_rows[0], valid_rows[-1] + 1
            c_start, c_end = valid_cols[0], valid_cols[-1] + 1
            if (r_end - r_start) > h * 0.5 and (c_end - c_start) > w * 0.5:
                cropped = cropped[r_start:r_end, c_start:c_end]
        return cropped
    except Exception as exc:
        logger.warning("Auto-crop panorama borders failed: %s", exc)
        return img


def estimate_ev_offsets(images: List[np.ndarray]) -> List[float]:
    """Estimate relative EV offsets across a set of images based on mean luminance."""
    if not images:
        return []
    lum_list = []
    for img in images:
        if img.ndim == 3:
            lum = np.mean(0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2])
        else:
            lum = np.mean(img)
        lum_list.append(float(lum))
    
    mid_idx = len(lum_list) // 2
    ref_lum = max(1e-4, lum_list[mid_idx])
    
    ev_offsets = []
    for l in lum_list:
        ratio = max(1e-4, l) / ref_lum
        ev = math.log2(ratio)
        ev_offsets.append(round(ev, 1))
    return ev_offsets


def _exif_ratio_to_float(tag: Any) -> Optional[float]:
    """Best-effort float from an exifread Ratio/IfdTag value ('−2/1', Fraction, int)."""
    if tag is None:
        return None
    try:
        values = getattr(tag, "values", None)
        v = values[0] if values else tag
        num = getattr(v, "num", None)
        den = getattr(v, "den", None)
        if num is not None and den:
            return float(num) / float(den)
        return float(str(v).strip())
    except Exception:
        s = str(tag).strip()
        if "/" in s:
            try:
                num_s, den_s = s.split("/", 1)
                den_f = float(den_s)
                if den_f:
                    return float(num_s) / den_f
            except Exception:
                return None
        return None


def exposure_ev_offsets_for_paths(
    paths: List[str], images: Optional[List[np.ndarray]] = None
) -> List[float]:
    """Per-frame relative EV offsets for a bracketed set, best source first.

    1. EXIF ExposureBiasValue when present on every frame and not uniform
       (the camera's own bracketing record -- exact).
    2. EXIF ExposureTime: log2(t / median t) -- correct when the bracket
       was shot varying shutter (the normal case; aperture/ISO brackets
       fall through to 3).
    3. Pixel-luminance estimate via estimate_ev_offsets(images) when
       decoded frames are available, else all zeros.

    Used by the HDR dialog for its exposure-ladder sort, EV badges and the
    default shadow/highlight recovery prior -- so this returns a usable
    list for ANY input rather than raising.
    """
    n = len(paths)
    if n == 0:
        return []

    biases: List[Optional[float]] = []
    times: List[Optional[float]] = []
    for p in paths:
        tags: Dict[str, Any] = {}
        try:
            import metadata_backend

            tags = metadata_backend.process_file_from_path(p, details=False) or {}
        except Exception:
            tags = {}
        biases.append(_exif_ratio_to_float(tags.get("EXIF ExposureBiasValue")))
        times.append(_exif_ratio_to_float(tags.get("EXIF ExposureTime")))

    if all(b is not None for b in biases) and len({round(b, 2) for b in biases}) > 1:
        return [round(float(b), 1) for b in biases]

    valid_times = [t for t in times if t is not None and t > 0]
    if len(valid_times) == n and len({round(t, 6) for t in valid_times}) > 1:
        ref = sorted(valid_times)[n // 2]
        return [round(math.log2(t / ref), 1) for t in valid_times]

    if images:
        return estimate_ev_offsets(images)
    return [0.0] * n


def capture_times_for_paths(paths: List[str]) -> List[Optional[float]]:
    """Unix timestamps from DateTimeOriginal (plus SubSec when present).

    Missing EXIF yields None -- file mtime is *not* used, because a copy onto
    a working volume would then sort by copy order rather than shoot order.
    """
    from common_image_loader import parse_capture_time_to_timestamp

    out: List[Optional[float]] = []
    for p in paths:
        tags: Dict[str, Any] = {}
        try:
            import metadata_backend

            tags = metadata_backend.process_file_from_path(p, details=False) or {}
        except Exception:
            tags = {}
        ts = 0.0
        for tag_name in (
            "EXIF DateTimeOriginal",
            "Image DateTime",
            "EXIF DateTime",
        ):
            raw = tags.get(tag_name)
            if not raw:
                continue
            ts = parse_capture_time_to_timestamp(str(raw))
            if ts > 0:
                break
        if ts <= 0:
            out.append(None)
            continue
        sub = tags.get("EXIF SubSecTimeOriginal") or tags.get("EXIF SubSecTime")
        if sub is not None:
            try:
                digits = "".join(c for c in str(sub) if c.isdigit())
                if digits:
                    ts += int(digits[:6].ljust(6, "0")) / 1e6
            except Exception:
                pass
        out.append(ts)
    return out


def _natural_name_key(path: str) -> tuple:
    import re

    name = os.path.basename(path).lower()
    parts = re.split(r"(\d+)", name)
    return tuple(int(p) if p.isdigit() else p for p in parts)


def infer_bracket_period(evs: List[float]) -> Optional[int]:
    """Smallest repeating EV period, or None when the set is not a bracketed pano."""
    n = len(evs)
    if n < 4 or len({round(e, 1) for e in evs}) <= 1:
        return None
    for g in range(2, n // 2 + 1):
        if n % g:
            continue
        if len({round(evs[i], 1) for i in range(g)}) < g:
            continue
        if all(abs(evs[i] - evs[i % g]) <= 0.34 for i in range(n)):
            return g
    return None


def reorder_merge_paths(
    paths: List[str],
    mode: str = "hdr",
    *,
    evs: Optional[List[float]] = None,
    capture_times: Optional[List[Optional[float]]] = None,
    group_size: Optional[int] = None,
) -> Tuple[List[str], str]:
    """Put a merge selection into shoot / bracket order.

    Gallery selection order is whatever the user clicked, which is rarely the
    order the camera wrote the bracket. Returns ``(paths, note)``; ``note`` is
    empty when the order did not change.

    * HDR: darkest → brightest (EXIF EV), capture time as a tie-break.
    * Panorama / focus stack: capture time (filename if EXIF time is missing).
    * Panorama HDR: capture time for panel order, then darkest → brightest
      inside each bracket of ``group_size``.
    """
    n = len(paths)
    if n < 2:
        return list(paths), ""

    mode = (mode or "hdr").lower()
    if evs is None:
        try:
            evs = exposure_ev_offsets_for_paths(list(paths))
        except Exception:
            evs = [0.0] * n
    if len(evs) != n:
        evs = list(evs) + [0.0] * n
        evs = evs[:n]
    if capture_times is None:
        try:
            capture_times = capture_times_for_paths(list(paths))
        except Exception:
            capture_times = [None] * n
    if len(capture_times) != n:
        capture_times = list(capture_times) + [None] * n
        capture_times = capture_times[:n]

    def seq_key(i: int) -> tuple:
        t = capture_times[i]
        return (t is None, t if t is not None else 0.0, _natural_name_key(paths[i]), i)

    if mode == "hdr":
        order = sorted(range(n), key=lambda i: (float(evs[i]),) + seq_key(i))
        note = "Sorted darkest → brightest from EXIF."
    elif mode == "hdr_panorama":
        order = sorted(range(n), key=seq_key)
        ordered_evs = [evs[i] for i in order]
        g = int(group_size) if group_size else infer_bracket_period(ordered_evs)
        if g and n % g == 0:
            grouped: List[int] = []
            for start in range(0, n, g):
                chunk = order[start:start + g]
                chunk.sort(key=lambda i: (float(evs[i]), i))
                grouped.extend(chunk)
            order = grouped
            note = (
                "Sorted by capture time into panels, "
                "darkest → brightest within each bracket."
            )
        else:
            note = "Sorted by capture time."
    else:
        # panorama, focus_stack, anything else: shooting order
        order = sorted(range(n), key=seq_key)
        note = "Sorted by capture time."

    ordered = [paths[i] for i in order]
    if ordered == list(paths):
        return ordered, ""
    return ordered, note


def default_recovery_weights(ev_offsets: List[float]) -> Tuple[List[float], List[float]]:
    """(shadow_weights, highlight_weights) prior from a set's EV offsets.

    The darkest frame is the natural highlight donor (its bright areas are
    the only unclipped record of the scene's highlights) and the brightest
    frame the shadow donor -- so they open biased that way (1.5x/0.5x)
    while middle frames stay neutral. All-equal offsets (no usable EV
    info) return all-neutral so the dialog never fakes a confident prior.
    """
    n = len(ev_offsets)
    if n == 0:
        return [], []
    shadow = [1.0] * n
    highlight = [1.0] * n
    if n >= 2 and len({round(e, 2) for e in ev_offsets}) > 1:
        darkest = int(np.argmin(ev_offsets))
        brightest = int(np.argmax(ev_offsets))
        highlight[darkest] = 1.5
        shadow[darkest] = 0.5
        shadow[brightest] = 1.5
        highlight[brightest] = 0.5
    return shadow, highlight


def merge_hdr_exposure_fusion(
    images: List[np.ndarray],
    highlight_weight: float = 1.0,
    shadow_weight: float = 1.0,
    midtone_weight: float = 1.0,
    image_weights: Optional[List[float]] = None,
    contrast_weight: float = 1.0,
    saturation_weight: float = 1.0,
    highlight_weights: Optional[List[float]] = None,
    shadow_weights: Optional[List[float]] = None,
    return_weight_maps: bool = False,
):
    """Mertens Exposure Fusion with custom Highlight, Shadow, Midtone & Per-Image weights.

    Frames must be display-referred (see the module docstring).

    Every control reaches the output, because the fusion is always a
    Laplacian-pyramid blend of the weight maps built below
    (focus_stacking._fuse_pyramid, seam-free like Mertens itself).
    cv2.createMergeMertens used to run whenever no per-image list was given,
    and its process() honours only the three constructor scalars -- so the
    maps were computed and discarded, and the dialog's Highlight and Shadow
    sliders moved the output by nothing more than cv2's own 1-ULP
    parallel-reduction jitter. The maps carry the same three Mertens metrics
    (well-exposedness, contrast, saturation), so this is the same fusion with
    the extra controls actually connected.

    ``highlight_weights`` / ``shadow_weights``: optional per-image exponent
    lists (same 0-2.0 scale as the global scalars, which they override
    per-image when given) -- "recover highlights mostly from frame 1,
    shadows from frame 3".

    ``return_weight_maps``: also return the normalized per-image weight
    maps (list of float32 HxW summing to 1 per pixel) for contribution
    visualization -- returns ``(result, weight_maps)`` instead of just
    ``result``.
    """
    if not images:
        raise ValueError("No images provided for HDR merge")
    if len(images) == 1:
        res_single = images[0].copy()
        if return_weight_maps:
            return res_single, [np.ones(images[0].shape[:2], dtype=np.float32)]
        return res_single

    is_uint8 = (images[0].dtype == np.uint8)
    norm_imgs = []
    for img in images:
        if img.dtype == np.uint8:
            norm_imgs.append(img.astype(np.float32) / 255.0)
        elif img.dtype == np.uint16:
            norm_imgs.append(img.astype(np.float32) / 65535.0)
        else:
            norm_imgs.append(np.clip(img.astype(np.float32), 0.0, 1.0))

    n = len(norm_imgs)
    if n > _MAX_MERGE_FRAMES:
        raise ValueError(
            f"Too many frames for an HDR merge ({n}; limit {_MAX_MERGE_FRAMES}). "
            "Raise RAWVIEWER_MERGE_MAX_FRAMES if this machine has the RAM."
        )
    if image_weights is None or len(image_weights) != n:
        image_weights = [1.0] * n
    per_image_hl = highlight_weights is not None and len(highlight_weights) == n
    per_image_sh = shadow_weights is not None and len(shadow_weights) == n

    weights_maps = []
    for idx, img in enumerate(norm_imgs):
        user_w = max(0.0, float(image_weights[idx]))
        if user_w < 1e-4:
            weights_maps.append(np.zeros((img.shape[0], img.shape[1]), dtype=np.float32))
            continue

        hl_exp = float(highlight_weights[idx]) if per_image_hl else float(highlight_weight)
        sh_exp = float(shadow_weights[idx]) if per_image_sh else float(shadow_weight)

        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

        # Well-exposedness, per channel and multiplied, exactly as Mertens
        # defines it: a pixel blown in one channel only is still penalised,
        # which a luma-only Gaussian misses.
        well_exp = np.ones(gray.shape, dtype=np.float32)
        for c in range(3):
            well_exp *= np.exp(-0.5 * ((img[:, :, c] - 0.5) ** 2) / (0.2 ** 2))
        if midtone_weight != 1.0:
            well_exp = well_exp ** midtone_weight

        # Highlight/shadow recovery as a MODULATION that is identity at 1.0,
        # not an added term. Summing three Gaussians (the old shape) meant the
        # dark and bright lobes diluted well-exposedness even at neutral
        # settings, flattening every merge: measured 0.158 output contrast
        # against cv2 Mertens' 0.184 on the same bracket. Now neutral reduces
        # to plain Mertens weighting, and the sliders lean the fusion toward
        # the dark frame's highlight record (hl > 1) or the bright frame's
        # shadow record (sh > 1) from there. Linear in the exponent, so it
        # cannot blow up the way a negative power of a near-zero Gaussian can.
        exp_w = well_exp
        if hl_exp != 1.0:
            hl_map = np.exp(-0.5 * ((gray - 0.15) ** 2) / (0.25 ** 2))
            exp_w = exp_w * np.maximum(1.0 + (hl_exp - 1.0) * hl_map, 0.0)
        if sh_exp != 1.0:
            sh_map = np.exp(-0.5 * ((gray - 0.85) ** 2) / (0.25 ** 2))
            exp_w = exp_w * np.maximum(1.0 + (sh_exp - 1.0) * sh_map, 0.0)

        if contrast_weight > 0.0:
            # On the float gray, not a uint8 round-trip, and NOT divided by
            # this frame's own maximum: per-frame normalisation made one
            # specular speck scale down the whole frame's contrast term, so
            # frames were compared on incommensurable scales.
            lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
            contrast_w = (lap + 1e-4) ** contrast_weight
        else:
            contrast_w = 1.0

        if saturation_weight > 0.0:
            # Floored for the same reason Mertens floors it: in a flat grey
            # region every frame's saturation is ~0, and without the floor the
            # normalised weights there are decided by rounding noise.
            sat_w = (np.std(img, axis=-1) + 1e-4) ** saturation_weight
        else:
            sat_w = 1.0

        w_combined = (exp_w * contrast_w * sat_w * user_w) + 1e-12
        weights_maps.append(w_combined.astype(np.float32))

    sum_w = np.zeros_like(weights_maps[0])
    for w in weights_maps:
        sum_w += w
    sum_w = np.maximum(sum_w, 1e-12)

    for i in range(n):
        weights_maps[i] /= sum_w

    # Seam-free multi-band blend of the maps above (same primitive focus
    # stacking uses). Deterministic, and unlike cv2 Mertens it cannot ignore
    # the weights it is handed.
    from focus_stacking import _fuse_pyramid

    res = np.clip(_fuse_pyramid(norm_imgs, weights_maps), 0.0, 1.0)
    del norm_imgs

    if is_uint8:
        out = (res * 255.0).round().clip(0, 255).astype(np.uint8)
    else:
        out = (res * 65535.0).round().clip(0, 65535).astype(np.uint16)
    if return_weight_maps:
        return out, weights_maps
    return out


def stitch_panorama(
    images: List[np.ndarray],
    paths: Optional[List[str]] = None,
    auto_crop: bool = True,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> StitchResult:
    """Stitch images into a linear panorama with automatic rectangular border cropping.

    Frames must be display-referred (see the module docstring). cv2.Stitcher is
    8-bit only, so the result is uint8 whatever the input depth -- which is
    lossy but sane on display-referred data. On scene-linear input the same
    quantization reached only 169 of 256 display levels, with the darkest codes
    stepping 0 -> 4 -> 8 -> 14: visible shadow banding in every panorama.
    """
    if not images or len(images) < 2:
        return StitchResult(
            success=False,
            error_message="Panorama stitching requires at least 2 images."
        )

    n = len(images)
    paths = paths or [f"Image {i+1}" for i in range(n)]
    if n > _MAX_MERGE_FRAMES:
        return StitchResult(
            success=False,
            error_message=(
                f"Too many frames for a panorama ({n}; limit {_MAX_MERGE_FRAMES}). "
                "Every frame is held in memory at once. Select fewer, or raise "
                "RAWVIEWER_MERGE_MAX_FRAMES."
            ),
            rejected_paths=paths,
        )

    if progress_callback:
        progress_callback(10, "Detecting features & matching keypoints…")

    prep_imgs = []
    for img in images:
        if img.dtype == np.uint8:
            img_u8 = img
        elif img.dtype == np.uint16:
            img_u8 = (img / 257.0).astype(np.uint8)
        elif np.issubdtype(img.dtype, np.floating):
            # Any float depth, not just float32: .astype(uint8) on a float64
            # [0, 1] frame used to fall through below and produce black.
            img_u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        prep_imgs.append(img_u8)

    try:
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

        if progress_callback:
            progress_callback(40, "Estimating camera poses & homography…")

        status, stitched = stitcher.stitch(prep_imgs)

        if status == cv2.Stitcher_OK and stitched is not None and stitched.size > 0:
            if auto_crop:
                if progress_callback:
                    progress_callback(85, "Auto-cropping warped black borders…")
                stitched = crop_panorama_borders(stitched)
            if progress_callback:
                progress_callback(95, "Finalizing panorama blend…")
            return StitchResult(
                success=True,
                image=stitched,
            )

        error_reasons = {
            cv2.Stitcher_ERR_NEED_MORE_IMGS: "Insufficient feature overlap between images (minimum 20% overlap required).",
            cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Camera homography estimation failed due to severe misalignment or parallax.",
            cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera pose parameter adjustment failed.",
        }
        err_detail = error_reasons.get(status, f"Stitching failed with status code {status}.")
        logger.warning("Stitcher failed for %d images: %s", n, err_detail)

        return StitchResult(
            success=False,
            error_message=f"Stitching failed: {err_detail}",
            rejected_paths=paths,
        )

    except Exception as exc:
        logger.exception("Panorama stitching exception: %s", exc)
        return StitchResult(
            success=False,
            error_message=f"Panorama stitching failed: {str(exc)}",
            rejected_paths=paths,
        )


def infer_bracket_group_size(
    paths: List[str], images: Optional[List[np.ndarray]] = None
) -> int:
    """Exposures per panel for a bracketed panorama, read off the EV pattern.

    A bracketed pano is shot panel by panel, so its EV offsets repeat with the
    bracket's period: [-2, 0, +2, -2, 0, +2, ...] is 3 exposures per panel. The
    period is the smallest divisor of n whose first cycle holds distinct EVs and
    which every later frame matches.

    Replaces ``2 if n % 2 == 0 else 3``, which was right only by luck: 12 frames
    shot as 4 panels x 3 stops -- an entirely ordinary bracketed pano -- grouped
    in twos and fused exposures belonging to *different panels*. Falls back to
    that old guess when no usable EXIF exposure record exists, so the caller
    always gets a workable default to show the user.
    """
    n = len(paths)
    if n < 4:
        return max(1, n)

    try:
        evs = exposure_ev_offsets_for_paths(list(paths), images=images)
    except Exception:
        evs = []

    period = infer_bracket_period(evs) if len(evs) == n else None
    if period:
        return period

    return 2 if n % 2 == 0 else 3


def merge_hdr_panorama(
    images: List[np.ndarray],
    paths: List[str],
    weights: Optional[Dict[str, float]] = None,
    auto_crop: bool = True,
    group_size: Optional[int] = None,
    align: bool = True,
    image_weights: Optional[List[float]] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> StitchResult:
    """Combined Panorama HDR processing: fuses bracketed exposures then stitches panels.

    ``group_size``: exposures per panel. Defaults to
    :func:`infer_bracket_group_size` when not given; the dialog passes the
    user's choice.

    ``image_weights``: one per source photo, in the order given. Sliced per
    bracket, so a weight applies to the exposure it was set on.
    """
    if not images or len(images) < 4:
        return StitchResult(
            success=False,
            error_message="Panorama HDR requires at least 4 images (bracketed exposure pairs)."
        )

    weights = weights or {}
    hl_w = float(weights.get("highlight", 1.0))
    sh_w = float(weights.get("shadow", 1.0))
    mid_w = float(weights.get("midtone", 1.0))

    n = len(images)
    if group_size is None:
        group_size = infer_bracket_group_size(paths, images=images)
    group_size = max(2, int(group_size))
    if n % group_size:
        return StitchResult(
            success=False,
            error_message=(
                f"{n} photos do not divide into brackets of {group_size}. "
                "Set the exposures-per-panel to match how the set was shot, or "
                "select a complete set of panels."
            ),
            rejected_paths=list(paths),
        )
    n_panels = n // group_size
    logger.info("Panorama HDR: %d frames as %d panels x %d exposures",
                n, n_panels, group_size)

    panels = []
    panel_paths = []
    for panel_idx, i in enumerate(range(0, n, group_size)):
        if progress_callback:
            progress_callback(
                10 + int(35 * panel_idx / max(1, n_panels)),
                f"Fusing panel {panel_idx + 1}/{n_panels}…",
            )
        chunk = images[i:i + group_size]
        chunk_paths = paths[i:i + group_size]
        chunk_weights = (
            list(image_weights[i:i + group_size])
            if image_weights and len(image_weights) == n
            else None
        )
        if len(chunk) < 2:
            panels.append(chunk[0])
            panel_paths.append(chunk_paths[0])
            continue
        if align:
            chunk, _ok = align_hdr_images(chunk)
        hdr_panel = merge_hdr_exposure_fusion(
            chunk,
            highlight_weight=hl_w,
            shadow_weight=sh_w,
            midtone_weight=mid_w,
            image_weights=chunk_weights,
        )
        panels.append(hdr_panel)
        panel_paths.append(chunk_paths[0])

    # Aligning each panel crops it to its own bracket's common rectangle, so
    # panels can end up a few pixels apart. The Stitcher does not care about
    # differing sizes, but crop_panorama_borders does -- give it panels of one
    # size by trimming to the smallest.
    if len({p.shape[:2] for p in panels}) > 1:
        min_h = min(p.shape[0] for p in panels)
        min_w = min(p.shape[1] for p in panels)
        panels = [p[:min_h, :min_w] for p in panels]

    if progress_callback:
        progress_callback(50, "Stitching HDR panels into panorama…")

    return stitch_panorama(
        panels,
        paths=panel_paths,
        auto_crop=auto_crop,
        progress_callback=progress_callback,
    )
