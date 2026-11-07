"""Apply adjustments: linear + gamma-space pipelines, banded processing, tone helpers."""

from __future__ import annotations

import math
import os

import numpy as np

from .checks import is_default_adjustments, uses_recovery_tone_map, wb_reference_temperature
from .defaults import DEFAULT_ADJUSTMENTS, _kelvin_to_rgb
from .saved_edits import _decode_srgb_uint8_to_linear


def _format_xmp_value(key: str, value: float) -> str:
    if key == "Exposure2012":
        return f"{value:.2f}".rstrip("0").rstrip(".") if abs(value) < 10 else f"{value:.1f}"
    if key == "Temperature":
        return str(int(round(value)))
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def apply_adjustments_to_rgb(rgb_image: np.ndarray, adj: dict[str, float]) -> np.ndarray:
    """Apply adjustments via the scene-linear edit pipeline.

    uint16/float32 buffers are already scene-linear. uint8 display buffers
    (embedded JPEG / LibRaw postprocess) are decoded with the sRGB EOTF,
    processed with the same pipeline as the Adjust panel, then encoded back
    to display uint8 ΓÇö the legacy gamma-space path is no longer used here
    (it diverged on PV2012, HSL, vignette, dehaze, LUT, NR, and AsShot WB).
    """
    if rgb_image is None:
        return rgb_image
    if adj is None:
        return rgb_image
    from raw_adjustments.pipeline import linear_to_display_uint8, process_linear_edit_buffer

    merged = dict(DEFAULT_ADJUSTMENTS)
    merged.update(adj or {})
    if rgb_image.dtype in (np.uint16, np.float32):
        use_recovery = uses_recovery_tone_map(merged)
        if is_default_adjustments(merged) and not use_recovery:
            processed = process_linear_edit_buffer(rgb_image, {}, preview=True)
            return linear_to_display_uint8(processed, {})
        processed = process_linear_edit_buffer(rgb_image, merged, preview=True)
        return linear_to_display_uint8(processed, merged)
    if rgb_image.dtype != np.uint8 or getattr(rgb_image, "ndim", 0) != 3:
        return rgb_image
    if is_default_adjustments(merged) and not uses_recovery_tone_map(merged):
        return rgb_image
    lin = _decode_srgb_uint8_to_linear(np.ascontiguousarray(rgb_image))
    processed = process_linear_edit_buffer(lin, merged, preview=True)
    return linear_to_display_uint8(processed, merged)


def apply_adjustments_to_linear(rgb_image: np.ndarray, adj: dict[str, float]) -> np.ndarray:
    """Scene-linear edit path: PV2012 + NLM chroma, tone-map, encode sRGB uint8."""
    from raw_adjustments.pipeline import linear_to_display_uint8, process_linear_edit_buffer

    if rgb_image is None:
        return rgb_image
    merged = dict(DEFAULT_ADJUSTMENTS)
    merged.update(adj or {})
    use_recovery = uses_recovery_tone_map(merged)
    if is_default_adjustments(merged) and not use_recovery:
        processed = process_linear_edit_buffer(rgb_image, {}, preview=True)
        return linear_to_display_uint8(processed, {})
    processed = process_linear_edit_buffer(rgb_image, merged, preview=True)
    return linear_to_display_uint8(processed, merged)


def apply_adjustments_to_linear_uint16(rgb_image: np.ndarray, adj: dict[str, float]) -> np.ndarray:
    """Full-quality export: PV2012 pipeline ΓåÆ 16-bit display-referred sRGB."""
    from raw_adjustments.pipeline import linear_to_export_uint16_srgb, process_linear_edit_buffer

    if rgb_image is None:
        return rgb_image
    merged = dict(DEFAULT_ADJUSTMENTS)
    merged.update(adj or {})
    processed = process_linear_edit_buffer(rgb_image, merged, preview=False, chroma_denoise=True)
    return linear_to_export_uint16_srgb(processed, merged)


def _apply_adjustments_to_srgb(rgb_image: np.ndarray, adj: dict[str, float]) -> np.ndarray:
    """Legacy gamma-space path for 8-bit sRGB buffers (cached previews / JPEG)."""
    if rgb_image is None or not adj:
        return rgb_image
    merged = dict(DEFAULT_ADJUSTMENTS)
    merged.update(adj)
    if is_default_adjustments(merged):
        return rgb_image

    # Geometry first (straighten/perspective/crop, raw_transform.py) so the
    # gamma path -- gallery tiles, fit previews, cached uint8 redisplay --
    # shows the same framing as the linear edit pipeline. Must run before the
    # row-band split below: it changes the buffer's shape.
    from raw_adjustments.transform import apply_geometry

    rgb_image = apply_geometry(rgb_image, merged)

    # Row-band parallelism: every stage below is per-pixel except the small-
    # radius Gaussian blurs inside detail-enhance (Sharpness/Clarity/
    # Defringe), so splitting rows across threads with enough padding to
    # cover those blur radii gives byte-identical output, just computed
    # concurrently -- numpy/cv2 release the GIL for these array sizes, so a
    # plain ThreadPoolExecutor gets real parallelism without the IPC cost of
    # multiprocessing. Dodge & Burn is excluded (its mask needs its own
    # per-band slicing, not implemented here) and falls back to the
    # single-threaded path below.
    from raw_adjustments.dodge_burn import MASK_KEY as _db_mask_key

    mask_serial = str(merged.get(_db_mask_key, "") or "")
    n_workers = _detail_pipeline_worker_count(rgb_image.shape[0])
    if not mask_serial and n_workers > 1:
        return _apply_adjustments_to_srgb_banded(rgb_image, merged, n_workers)

    img = rgb_image.astype(np.float32) / 255.0
    img = _apply_adjustments_to_srgb_core(img, merged)
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def banded_worker_count(height: int) -> int:
    """Row-band thread count for the banded pixel pipelines (shared by
    raw_adjustments and raw_edit_pipeline -- one tuning point, not several).
    """
    # Thread-pool overhead isn't worth it below a few hundred rows.
    if height < 512:
        return 1
    return min(8, os.cpu_count() or 1)


# Back-compat alias (older call sites/tests reference the private name).
_detail_pipeline_worker_count = banded_worker_count


def band_ranges(h: int, n_workers: int, pad_px: int = 0) -> list:
    """Split ``h`` rows into up to ``n_workers`` contiguous bands.

    Returns [(y0, y1, pad_top, pad_bot), ...] where pads are clamped to the
    image edges. Single source of truth for the band-boundary math used by
    every banded pipeline (a boundary off-by-one fixed here fixes all of
    them; previously three near-identical copies existed).
    """

    band_h = math.ceil(h / max(1, n_workers))
    bands = []
    for i in range(n_workers):
        y0 = i * band_h
        y1 = min(h, y0 + band_h)
        if y0 >= y1:
            break
        bands.append((y0, y1, min(pad_px, y0), min(pad_px, h - y1)))
    return bands


_BANDED_EXECUTOR = None
_BANDED_EXECUTOR_LOCK = None


def banded_executor():
    """Shared lazily-created pool for banded pixel stages.

    The banded functions used to build (spawn + join up to 8 OS threads) a
    fresh ThreadPoolExecutor per call -- on the Adjust-panel live-preview
    path that's pool churn on every slider tick. One idle pool costs nothing
    between calls; concurrent callers share it safely (Executor.map is
    thread-safe).
    """
    global _BANDED_EXECUTOR, _BANDED_EXECUTOR_LOCK
    import threading

    if _BANDED_EXECUTOR_LOCK is None:
        _BANDED_EXECUTOR_LOCK = threading.Lock()
    with _BANDED_EXECUTOR_LOCK:
        if _BANDED_EXECUTOR is None:
            from concurrent.futures import ThreadPoolExecutor

            _BANDED_EXECUTOR = ThreadPoolExecutor(
                max_workers=min(8, os.cpu_count() or 1),
                thread_name_prefix="banded-pixels",
            )
        return _BANDED_EXECUTOR


# Comfortably larger than 3x any Gaussian sigma used in apply_detail_enhancements
# (defringe 1.5, sharpness 0.9, clarity's downsampled effective ~10) -- a band's
# padded region always has enough real neighboring pixels for every blur stage
# to produce the same result as processing the full image in one pass.
_BAND_PAD_PX = 48


def _apply_adjustments_to_srgb_banded(
    rgb_image: np.ndarray, merged: dict[str, float], n_workers: int
) -> np.ndarray:
    """Row-band parallel version of _apply_adjustments_to_srgb's core pipeline."""
    h = rgb_image.shape[0]
    bands = band_ranges(h, n_workers, pad_px=_BAND_PAD_PX)

    def _process_band(band):
        y0, y1, pad_top, pad_bot = band
        src = rgb_image[y0 - pad_top : y1 + pad_bot]
        img = src.astype(np.float32) / 255.0
        out = _apply_adjustments_to_srgb_core(img, merged)
        out_u8 = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
        return out_u8[pad_top : pad_top + (y1 - y0)]

    import cv2

    prev_threads = cv2.getNumThreads()
    # Avoid oversubscription: cv2's own internal (IPP/TBB) threading would
    # otherwise compete with our outer thread pool for the same cores.
    cv2.setNumThreads(1)
    try:
        results = list(banded_executor().map(_process_band, bands))
    finally:
        cv2.setNumThreads(prev_threads)

    return np.concatenate(results, axis=0)


def _apply_adjustments_to_srgb_core(img: np.ndarray, merged: dict[str, float]) -> np.ndarray:
    """All per-pixel/local-only adjustment stages on a float32 [0,1] (H,W,3)
    buffer -- may be a padded row-band slice, see _apply_adjustments_to_srgb_banded.
    """
    # 1. Temperature & Tint (neutral gray preserved ΓÇö scale R/B vs G)
    # Use AsShot / authored reference ΓÇö never hard-code 5500K (that made
    # legacy uint8 WB disagree with the linear Adjust pipeline).
    temp_val = float(merged.get("Temperature", DEFAULT_ADJUSTMENTS["Temperature"]))
    tint_val = float(merged.get("Tint", 0.0))
    ref_temp = wb_reference_temperature(merged)
    if temp_val > 1000.0 and abs(temp_val - ref_temp) > 0.5:
        r_tgt, g_tgt, b_tgt = _kelvin_to_rgb(temp_val)
        r_ref, g_ref, b_ref = _kelvin_to_rgb(ref_temp)
        if g_ref > 1e-5 and g_tgt > 1e-5:
            img[:, :, 0] *= (r_tgt / r_ref) / (g_tgt / g_ref)
            img[:, :, 2] *= (b_tgt / b_ref) / (g_tgt / g_ref)
    elif abs(temp_val) > 1e-4 and temp_val <= 1000.0:
        # Lightroom-style relative offset when not stored as Kelvin
        scale = temp_val / 100.0
        img[:, :, 0] *= 1.0 + scale * 0.1
        img[:, :, 2] *= 1.0 - scale * 0.1

    if abs(tint_val) > 1e-4:
        img[:, :, 1] *= 1.0 - (tint_val / 150.0) * 0.1

    # 2. Exposure
    exp_val = float(merged.get("Exposure2012", 0.0))
    if abs(exp_val) > 1e-4:
        img *= 2.0 ** exp_val

    # 2b. Dodge & burn (see raw_dodge_burn.py). This legacy gamma-space path
    # is used for cached uint8 full-image redisplay (browse revisit/zoom of
    # an edited photo, not the live Adjust-panel preview) -- applying the
    # mask here too keeps that path visually consistent with the scene-
    # linear pipeline's result (process_linear_edit_buffer).
    from raw_adjustments.dodge_burn import MASK_KEY as _db_mask_key

    mask_serial = str(merged.get(_db_mask_key, "") or "")
    if mask_serial:
        from raw_adjustments.dodge_burn import DEFAULT_STRENGTH as _db_default_strength
        from raw_adjustments.dodge_burn import STRENGTH_KEY as _db_strength_key
        from raw_adjustments.dodge_burn import _deserialize_mask_cached, apply_dodge_burn

        mask = _deserialize_mask_cached(mask_serial)
        if mask is not None:
            stops = float(merged.get(_db_strength_key, _db_default_strength))
            img = apply_dodge_burn(img, mask, stops)

    # 2c. Spot heal (cv2.inpaint) ΓÇö same rationale as dodge/burn above.
    from raw_adjustments.spot_heal import MASK_KEY as _heal_mask_key

    heal_serial = str(merged.get(_heal_mask_key, "") or "")
    if heal_serial:
        from raw_adjustments.spot_heal import _deserialize_mask_cached as _heal_deserialize
        from raw_adjustments.spot_heal import apply_spot_heal

        heal_mask = _heal_deserialize(heal_serial)
        if heal_mask is not None:
            img = apply_spot_heal(img, heal_mask)

    # 3. Contrast
    contrast_val = float(merged.get("Contrast2012", 0.0))
    if abs(contrast_val) > 1e-4:
        c = contrast_val / 100.0
        factor = (259.0 * (c * 100.0 + 255.0)) / (255.0 * (259.0 - c * 100.0))
        img = factor * (img - 0.5) + 0.5

    # 4. Highlights & Shadows ΓÇö luminance-weighted (hue-preserving, PaintFE-style)
    hi_val = float(merged.get("Highlights2012", 0.0))
    sh_val = float(merged.get("Shadows2012", 0.0))
    if abs(hi_val) > 1e-4 or abs(sh_val) > 1e-4:
        img = _apply_highlights_shadows(img, hi_val, sh_val)

    # 5. Whites & Blacks
    white_val = float(merged.get("Whites2012", 0.0))
    black_val = float(merged.get("Blacks2012", 0.0))
    if abs(white_val) > 1e-4 or abs(black_val) > 1e-4:
        lum = _channel_luminance(img)
        if abs(black_val) > 1e-4:
            img = _apply_masked_luminance_adjust(
                img, lum, _black_region_weight(lum), black_val / 100.0, lift=True,
                lift_up_strength=0.03,
            )
            lum = _channel_luminance(img)
        if abs(white_val) > 1e-4:
            img = _apply_masked_luminance_adjust(
                img, lum, _highlight_region_weight(lum), white_val / 100.0, lift=True
            )

    # 6. Saturation & Vibrance
    sat_val = float(merged.get("Saturation", 0.0))
    vib_val = float(merged.get("Vibrance", 0.0))
    if abs(sat_val) > 1e-4 or abs(vib_val) > 1e-4:
        luma = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
        luma = np.expand_dims(luma, axis=-1)

        sat_scale = 1.0
        if abs(sat_val) > 1e-4:
            sat_scale += sat_val / 100.0

        if abs(vib_val) > 1e-4:
            # np.max/min(img, axis=-1) is ~8x slower than a direct elementwise
            # chain for a size-3 last axis (measured 356ms vs 45ms at 32MP).
            r_c, g_c, b_c = img[:, :, 0:1], img[:, :, 1:2], img[:, :, 2:3]
            max_val = np.maximum(np.maximum(r_c, g_c), b_c)
            min_val = np.minimum(np.minimum(r_c, g_c), b_c)
            s = (max_val - min_val) / (max_val + 1e-5)
            vib_factor = (vib_val / 100.0) * (1.0 - s)
            sat_scale += vib_factor

        img = luma + (img - luma) * sat_scale

    from raw_adjustments.detail_enhance import apply_detail_enhancements

    img = np.clip(img, 0.0, 1.0)
    img = apply_detail_enhancements(img, merged)
    return img


def _linear_float_from_buffer(rgb_image: np.ndarray) -> np.ndarray:
    """Normalize uint16 linear or uint8 inputs to float32 scene-linear."""
    if rgb_image.dtype == np.uint16:
        return rgb_image.astype(np.float32) / 65535.0
    if rgb_image.dtype == np.float32:
        return np.clip(rgb_image, 0.0, None)
    return rgb_image.astype(np.float32) / 255.0


def _linear_buffer_to_display_uint8(rgb_image: np.ndarray) -> np.ndarray:
    if rgb_image is None:
        return rgb_image
    from raw_adjustments.pipeline import linear_to_display_uint8, process_linear_edit_buffer

    processed = process_linear_edit_buffer(rgb_image, {}, preview=True)
    return linear_to_display_uint8(processed)


def _tone_map_linear_to_srgb8(
    img: np.ndarray,
    encode_srgb8,
    luminance_fn,
) -> np.ndarray:
    """Luminance-preserving Reinhard shoulder, then sRGB encode."""
    lum = luminance_fn(img)
    mapped = lum / (1.0 + np.maximum(lum, 0.0))
    scale = mapped / np.maximum(lum, 1e-6)
    display = img * scale[..., np.newaxis]
    return encode_srgb8(display)


_SHADOW_LUM_MAX = 0.22
_HIGHLIGHT_LUM_MIN = 0.55
_BLACK_LUM_MAX = 0.07


def _channel_luminance(img: np.ndarray) -> np.ndarray:
    return 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]


def _shadow_region_weight(lum: np.ndarray) -> np.ndarray:
    return np.clip((_SHADOW_LUM_MAX - lum) / _SHADOW_LUM_MAX, 0.0, 1.0) ** 2.5


def _highlight_region_weight(lum: np.ndarray) -> np.ndarray:
    span = max(1.0 - _HIGHLIGHT_LUM_MIN, 1e-6)
    return np.clip((lum - _HIGHLIGHT_LUM_MIN) / span, 0.0, 1.0) ** 2.0


def _black_region_weight(lum: np.ndarray) -> np.ndarray:
    return np.clip((_BLACK_LUM_MAX - lum) / _BLACK_LUM_MAX, 0.0, 1.0) ** 2.0


def _apply_masked_luminance_adjust(
    img: np.ndarray,
    lum: np.ndarray,
    mask: np.ndarray,
    amount: float,
    *,
    lift: bool,
    lift_up_strength: float = 0.55,
) -> np.ndarray:
    """
    Hue-preserving tone tweak: scale RGB by a luminance ratio inside ``mask`` only.
    Positive amount lifts; negative amount darkens the masked region.

    ``lift_up_strength`` (the ``lift and amount > 0`` branch) is the only branch
    whose safe coefficient depends on how steep ``mask`` is: the shared 0.55
    default is safe for ``_highlight_region_weight`` (worst slope -1.0) but not
    for ``_shadow_region_weight`` (-12.36) or ``_black_region_weight`` (-29.57,
    a narrower 0.07-wide region) -- both produced verified tone-curve reversals
    (banding) well within slider range. Callers using those masks must pass a
    safe override (see call sites below); the other three branches are safe
    for every mask used in this module at their existing fixed coefficients.
    """
    if abs(amount) < 1e-6:
        return img
    if lift and amount > 0:
        lum_new = lum + mask * amount * (1.0 - lum) * lift_up_strength
    elif lift and amount < 0:
        lum_new = lum + mask * amount * lum * 0.45
    elif not lift and amount < 0:
        knee = np.maximum(lum - _HIGHLIGHT_LUM_MIN, 0.0)
        lum_new = lum + mask * amount * knee * 0.65
    else:
        lum_new = lum + mask * amount * (1.0 - lum) * 0.35
    ratio = lum_new / np.maximum(lum, 1e-6)
    # A per-channel loop with a sliced out= beats both a plain (H,W,3)*(H,W)
    # broadcast and pre-tiling ratio to (H,W,3) via np.repeat (measured at
    # 32MP: broadcast 142ms, repeat-tile 119ms, per-channel-out 91ms) --
    # numpy's broadcasting iterator doesn't take the fast contiguous SIMD
    # path for a trailing size-1 axis, and repeat pays its own full-size
    # allocation+copy. Bit-identical to the broadcast result.
    out = np.empty_like(img)
    for c in range(img.shape[-1]):
        np.multiply(img[:, :, c], ratio, out=out[:, :, c])
    return out


def _apply_highlights_shadows(img: np.ndarray, hi_val: float, sh_val: float) -> np.ndarray:
    """Gamma-space HS using the same masked luminance ratios."""
    lum = _channel_luminance(img)
    if abs(sh_val) > 1e-4:
        img = _apply_masked_luminance_adjust(
            img, lum, _shadow_region_weight(lum), sh_val / 100.0, lift=True,
            lift_up_strength=0.07,
        )
        lum = _channel_luminance(img)
    if abs(hi_val) > 1e-4:
        amt = hi_val / 100.0
        if amt < 0:
            img = _apply_masked_luminance_adjust(
                img, lum, _highlight_region_weight(lum), amt, lift=False
            )
        else:
            img = _apply_masked_luminance_adjust(
                img, lum, _highlight_region_weight(lum), amt, lift=True
            )
    return img
