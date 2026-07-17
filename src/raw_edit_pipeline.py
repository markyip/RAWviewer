"""Shared edit decode → denoise → PV2012 adjust → display / 16-bit export."""

from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np

from raw_adjustments import (
    DEFAULT_ADJUSTMENTS,
    RECOVERY_BASELINE_KEY,
    is_default_adjustments,
    uses_recovery_tone_map,
)
from raw_chroma_denoise import apply_chroma_denoise, apply_luma_denoise, chroma_denoise_enabled
from raw_dodge_burn import DEFAULT_STRENGTH as _DB_DEFAULT_STRENGTH
from raw_dodge_burn import MASK_KEY as _DB_MASK_KEY
from raw_dodge_burn import STRENGTH_KEY as _DB_STRENGTH_KEY
from raw_dodge_burn import resolve_mask_from_adj as _resolve_db_mask
from raw_pv2012 import apply_pv2012_tone_rgb
from raw_tone_curve import TONE_CURVE_SERIAL_KEY


def _linear_float_from_buffer(rgb_image: np.ndarray) -> np.ndarray:
    if rgb_image.dtype == np.uint16:
        return rgb_image.astype(np.float32) / 65535.0
    if rgb_image.dtype == np.float32:
        return np.clip(rgb_image.astype(np.float32), 0.0, None)
    return rgb_image.astype(np.float32) / 255.0


def _kelvin_to_rgb(k: float) -> tuple[float, float, float]:
    from raw_adjustments import _kelvin_to_rgb as kelvin

    return kelvin(k)


def _apply_wb_tint(img: np.ndarray, merged: dict[str, float]) -> np.ndarray:
    from raw_adjustments import wb_reference_temperature

    ref_temp = wb_reference_temperature(merged)
    temp_val = float(merged.get("Temperature", ref_temp))
    tint_val = float(merged.get("Tint", 0.0))
    if temp_val > 1000.0 and abs(temp_val - ref_temp) > 1.0:
        r_tgt, g_tgt, b_tgt = _kelvin_to_rgb(temp_val)
        r_ref, g_ref, b_ref = _kelvin_to_rgb(ref_temp)
        if g_ref > 1e-5 and g_tgt > 1e-5:
            img[:, :, 0] *= (r_tgt / r_ref) / (g_tgt / g_ref)
            img[:, :, 2] *= (b_tgt / b_ref) / (g_tgt / g_ref)
    elif abs(temp_val) > 1e-4 and temp_val <= 1000.0:
        scale = temp_val / 100.0
        img[:, :, 0] *= 1.0 + scale * 0.1
        img[:, :, 2] *= 1.0 - scale * 0.1
    if abs(tint_val) > 1e-4:
        img[:, :, 1] *= 1.0 - (tint_val / 150.0) * 0.1
    return img


def _apply_saturation_vibrance(img: np.ndarray, merged: dict[str, float]) -> np.ndarray:
    """
    Saturation / vibrance via HSV S-channel scale in gamma-encoded (perceptual) space.

    Scaling chroma (img - luma) directly on scene-linear RGB made the result wildly
    hue-dependent: hues with large linear-light channel spread (green, yellow) blew
    past the gamut boundary and clipped to a flat, fully-saturated block, while hues
    with small linear spread (skin tones, blue sky) barely moved — the "colors look
    off / no pop" symptom. Round-tripping through the sRGB OETF (nearest-index LUT,
    not np.power — see docs/EDIT_PIPELINE.md) and scaling HSV S there matches
    how a "Saturation" slider is perceived, and S is bounded to [0, 1] by construction
    so it can't overshoot the gamut the way additive chroma scaling did.
    """
    import cv2

    from raw_tone_recovery import _decode_srgb_float01, _encode_srgb_float01

    sat_val = float(merged.get("Saturation", 0.0))
    vib_val = float(merged.get("Vibrance", 0.0))
    if abs(sat_val) < 1e-4 and abs(vib_val) < 1e-4:
        return img

    rgb_lin = np.clip(img.astype(np.float32), 0.0, 1.0)
    rgb_enc = _encode_srgb_float01(rgb_lin)
    bgr = np.ascontiguousarray(rgb_enc[..., ::-1])
    h, s, v = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))

    scale = np.ones_like(s)
    if abs(sat_val) > 1e-4:
        scale = scale + sat_val / 100.0
    if abs(vib_val) > 1e-4:
        scale = scale + (vib_val / 100.0) * (1.0 - s)
    s = np.clip(s * scale, 0.0, 1.0)

    bgr_out = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    rgb_enc_out = np.clip(bgr_out[..., ::-1], 0.0, 1.0)
    return _decode_srgb_float01(rgb_enc_out)


def _tone_map_reinhard_display(img: np.ndarray) -> np.ndarray:
    """Scene-linear float → display-linear float in [0, 1] (Reinhard shoulder).

    RETIRED as the default editor transform (kept only for the legacy
    recovery-look path): Reinhard maps linear 1.0 to display 0.5, so an
    edited image at DEFAULT settings rendered whites at ~151/255 vs the
    browse render's 225 -- measured as the reported "grey overcast" -- and
    compressed chroma spread by 24%. The default path is now
    _tone_map_clip_display + the dcraw BT.709 encode, matching browse
    exactly in tone reproduction.
    """
    from raw_tone_recovery import _luminance

    lum = _luminance(img)
    mapped = lum / (1.0 + np.maximum(lum, 0.0))
    scale = mapped / np.maximum(lum, 1e-6)
    return np.clip(img * scale[..., np.newaxis], 0.0, None)


def _tone_map_clip_display(img: np.ndarray) -> np.ndarray:
    """Scene-linear float → display-linear [0, 1] by hard clip (browse parity).

    The browse pipeline saturates highlights at the camera clip point and
    applies the dcraw BT.709 curve; clipping here reproduces that identically
    for the editor's default rendering.
    """
    return np.clip(img.astype(np.float32, copy=False), 0.0, 1.0)


def _tone_map_recovery_display(img: np.ndarray) -> np.ndarray:
    """Recovery preview tone path (matches P-key recovery, including edge cap)."""
    from raw_tone_recovery import (
        RECOVERY_MAX_EDGE,
        _cap_max_edge_linear,
        apply_local_shadow_highlight_recovery_display,
        linear_tone_map_to_display,
    )

    h, w = img.shape[:2]
    rgb_f = img.astype(np.float32)
    if max(h, w) <= RECOVERY_MAX_EDGE:
        display = linear_tone_map_to_display(rgb_f)
        return apply_local_shadow_highlight_recovery_display(display)

    small = _cap_max_edge_linear(rgb_f, RECOVERY_MAX_EDGE)
    polished = apply_local_shadow_highlight_recovery_display(
        linear_tone_map_to_display(small)
    )
    import cv2

    up = cv2.resize(polished, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(up, 0.0, 1.0).astype(np.float32)


def _tone_map_for_display(img: np.ndarray, merged: dict[str, float]) -> np.ndarray:
    if uses_recovery_tone_map(merged):
        # Legacy sessions only: the recovery-look button was removed from
        # the Adjust panel (2026-07); nothing sets the baseline flag anymore.
        return _tone_map_recovery_display(img)
    return _tone_map_clip_display(img)


def _apply_display_stage(img: np.ndarray, merged: dict[str, float]) -> np.ndarray:
    """Tone map → sat/vib → HSL → detail (display-linear float [0, 1])."""
    from raw_detail_enhance import apply_detail_enhancements
    from raw_hsl import apply_hsl_adjustments

    display = _tone_map_for_display(img, merged)
    if merged:
        display = _apply_display_color_adjustments(display, merged, preview=False)
        display = apply_hsl_adjustments(display, merged)
    display = apply_detail_enhancements(display, merged)
    return display


def _linear_pipeline_worker_count(height: int) -> int:
    # Shared tuning point with raw_adjustments' banded sRGB pipeline.
    from raw_adjustments import banded_worker_count

    return banded_worker_count(height)


def process_linear_edit_buffer(
    rgb_image: np.ndarray,
    adj: dict[str, float],
    *,
    preview: bool = False,
    chroma_denoise: Optional[bool] = None,
) -> np.ndarray:
    """
    Scene-linear uint16/float → adjusted scene-linear float32.

    Order: WB → exposure → chroma/luma denoise (bilateral) → PV2012 tone → saturation.
    """
    if rgb_image is None:
        return rgb_image
    merged = dict(DEFAULT_ADJUSTMENTS)
    merged.update(adj or {})
    img = _linear_float_from_buffer(rgb_image)

    if is_default_adjustments(merged) and not uses_recovery_tone_map(merged):
        return img

    # Geometry first: every later stage (dodge/burn coordinates included)
    # works in the transformed frame. No-op unless transform keys are set.
    from raw_transform import apply_geometry

    img = apply_geometry(img, merged)

    mask = _resolve_db_mask(merged)
    do_denoise = chroma_denoise if chroma_denoise is not None else chroma_denoise_enabled()
    nr_amount = float(merged.get("ColorNoiseReduction", 0.0))
    luma_nr_amount = float(merged.get("LuminanceNoiseReduction", 0.0))
    denoise_active = nr_amount > 1e-4 or (do_denoise and not preview) or luma_nr_amount > 1e-4

    # Row-band parallelism: WB, exposure and PV2012 tone are all elementwise
    # (no cross-row dependency), so splitting rows across threads gives
    # identical output computed concurrently -- numpy releases the GIL for
    # arrays this size (same rationale as _apply_adjustments_to_srgb_banded
    # in raw_adjustments.py). Denoise (real spatial radius) and dodge/burn
    # (its mask needs its own per-band slicing, not implemented here) fall
    # back to the single-threaded path below, same as that precedent.
    n_workers = _linear_pipeline_worker_count(img.shape[0])
    if mask is None and not denoise_active and n_workers > 1:
        return _process_linear_edit_buffer_banded(img, merged, n_workers)

    return _process_linear_edit_tail(img, merged, preview=preview, chroma_denoise=chroma_denoise)


def _process_linear_edit_tail(
    img: np.ndarray,
    merged: dict[str, float],
    *,
    preview: bool,
    chroma_denoise: Optional[bool],
) -> np.ndarray:
    """WB -> exposure -> dodge/burn -> denoise -> PV2012 tone on a (possibly banded) buffer."""
    img = _apply_wb_tint(img, merged)

    exp_val = float(merged.get("Exposure2012", 0.0))
    if abs(exp_val) > 1e-4:
        img = img * (2.0 ** exp_val)

    # Dodge & burn: local per-pixel exposure, applied right after the
    # global exposure gain and before denoise/tone -- local brightness
    # changes should see the same noise/tone response as global ones
    # (matches how a real exposure difference at capture time would look).
    mask = _resolve_db_mask(merged)
    if mask is not None:
        from raw_dodge_burn import apply_dodge_burn

        stops = float(merged.get(_DB_STRENGTH_KEY, _DB_DEFAULT_STRENGTH))
        img = apply_dodge_burn(img, mask, stops)

    do_denoise = chroma_denoise if chroma_denoise is not None else chroma_denoise_enabled()
    nr_amount = float(merged.get("ColorNoiseReduction", 0.0))
    method = int(float(merged.get("DenoiseMethod", 0.0)))
    if nr_amount > 1e-4:
        img = apply_chroma_denoise(
            img, strength=min(1.5, nr_amount / 100.0 * 1.25), method=method, preview=preview
        )
    elif do_denoise and not preview:
        img = apply_chroma_denoise(img, method=method, preview=False)

    luma_nr_amount = float(merged.get("LuminanceNoiseReduction", 0.0))
    if luma_nr_amount > 1e-4:
        img = apply_luma_denoise(img, strength=luma_nr_amount / 100.0, method=method, preview=preview)

    if not uses_recovery_tone_map(merged):
        img = apply_pv2012_tone_rgb(img, merged)
    return np.clip(img, 0.0, None)


def _process_linear_edit_buffer_banded(
    img: np.ndarray, merged: dict[str, float], n_workers: int
) -> np.ndarray:
    """Row-band parallel version of process_linear_edit_buffer's WB/exposure/tone tail.

    Only called when denoise is inactive and no dodge/burn mask is set (see
    caller), so every stage in ``_process_linear_edit_tail`` is purely
    elementwise -- bands need no padding, unlike the detail-enhance blur
    stages in raw_adjustments._apply_adjustments_to_srgb_banded.
    """
    from raw_adjustments import band_ranges, banded_executor

    bands = band_ranges(img.shape[0], n_workers)

    def _process_band(band):
        y0, y1, _pt, _pb = band
        return _process_linear_edit_tail(img[y0:y1], merged, preview=True, chroma_denoise=False)

    results = list(banded_executor().map(_process_band, bands))
    return np.concatenate(results, axis=0)


class PreviewStageCache:
    """
    Per-open-file memoization for live Adjust-panel preview recompute.

    Every slider tick used to rerun the entire WB -> exposure -> denoise ->
    PV2012 tone -> tonemap -> sat/vibrance -> HSL -> detail chain from
    scratch, even when only one late-stage control (e.g. Sharpness) changed
    since the previous tick -- see docs/EDIT_PIPELINE.md Performance
    review #2, the root cause of "sluggish/laggy while dragging". This cache
    chains a cheap key comparison per stage: if a stage's own inputs (plus
    whatever stage feeds it) haven't changed since the last call, its cached
    output is reused and every stage after it in the chain is skipped too.

    Only the interactive preview path (_AdjustPreviewWorker) uses this. Export
    always calls process_linear_edit_buffer / _apply_display_stage directly
    with no cache, so a bug here cannot corrupt a baked export.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.base_ref = None
        self.stage_keys: dict[str, tuple] = {}
        self.stage_out: dict[str, np.ndarray] = {}


_PRE_TONE_KEYS = (
    "Temperature",
    "Tint",
    "Exposure2012",
    "ColorNoiseReduction",
    "LuminanceNoiseReduction",
    "DenoiseMethod",
    _DB_MASK_KEY,
    _DB_STRENGTH_KEY,
    # Geometry runs at the head of the pipeline (before WB), so any transform
    # change invalidates pre_tone and everything chained to it.
    "CropAngle",
    "PerspectiveVertical",
    "PerspectiveHorizontal",
    "CropLeft",
    "CropRight",
    "CropTop",
    "CropBottom",
)

# Every key uses_recovery_tone_map() and apply_pv2012_tone_rgb() read, so the
# "tone" stage key alone is enough to detect any change that would flip the
# recovery-vs-PV2012 branch or alter the PV2012 tone result.
_TONE_KEYS = (
    "Contrast2012",
    "Highlights2012",
    "Shadows2012",
    "Whites2012",
    "Blacks2012",
    "ParametricShadows",
    "ParametricDarks",
    "ParametricLights",
    "ParametricHighlights",
    RECOVERY_BASELINE_KEY,
    TONE_CURVE_SERIAL_KEY,
)

def _hsl_color_keys() -> tuple[str, ...]:
    from raw_hsl import HSL_COLOR_NAMES

    return tuple(
        f"{prefix}Adjustment{color}"
        for color in HSL_COLOR_NAMES
        for prefix in ("Hue", "Saturation", "Luminance")
    )


_COLOR_KEYS = (
    "Saturation",
    "Vibrance",
    "PostCropVignetteAmount",
    "PostCropVignetteMidpoint",
    "Dehaze",
) + _hsl_color_keys()

_DETAIL_KEYS = ("Sharpness", "Clarity2012", "Defringe")


def _stage_key(merged: dict, keys: tuple) -> tuple:
    parts = []
    for k in keys:
        v = merged.get(k, 0.0)
        if v is None:
            # Missing/blank XMP leftovers: string-ish keys stay "", numeric stay 0.
            parts.append(
                ""
                if k in ("CreativeLUTName", "CreativeLUTWorkingSpace")
                or str(k).endswith("Mask")
                or "Curve" in str(k)
                else 0.0
            )
            continue
        if isinstance(v, str):
            # Dodge/burn mask serials are multi-MB base64 PNGs — hash so
            # stage-key compares stay O(1) and don't pin giant strings in
            # the cache dict (still invalidates whenever the mask changes).
            if len(v) > 256 or str(k).endswith("Mask") or k.endswith("Curve") or "Curve" in str(k):
                import hashlib

                parts.append(hashlib.sha1(v.encode("utf-8", errors="replace")).hexdigest())
            else:
                parts.append(v)
            continue
        # Harden against blank leftovers — a TypeError here used to be swallowed
        # by _AdjustPreviewWorker and leave the live preview stuck on the last frame.
        try:
            parts.append(round(float(v), 6))
        except (TypeError, ValueError):
            parts.append(0.0)
    return tuple(parts)


def process_linear_edit_buffer_staged(
    rgb_image: np.ndarray,
    adj: dict[str, float],
    cache: "PreviewStageCache",
    *,
    preview: bool = True,
    chroma_denoise: Optional[bool] = None,
    preview_lite: bool = False,
    settle_fast: bool = False,
) -> np.ndarray:
    """Cached equivalent of process_linear_edit_buffer for the preview path.

    ``preview_lite`` selects the cheaper PV2012 path (live-drag only). Live
    and settle use separate PreviewStageCache instances in main.py, so lite
    tone outputs cannot be reused after a full-quality settle.
    ``settle_fast`` keeps y0 guided filter but skips chroma-damp on settle.
    """
    if rgb_image is None:
        return rgb_image
    merged = dict(DEFAULT_ADJUSTMENTS)
    merged.update(adj or {})

    with cache.lock:
        if cache.base_ref is not rgb_image:
            cache.reset()
            cache.base_ref = rgb_image

        if is_default_adjustments(merged) and not uses_recovery_tone_map(merged):
            # Fast path bypasses pre_tone/tone entirely, so it must still
            # stamp a distinct "tone" key -- otherwise a later call landing
            # back on this fast path (or the transition into it) would leave
            # the stale key from the last *real* computation in place, and
            # _apply_display_stage_staged's tonemap stage (keyed only off
            # this metadata, not off buffer identity) would wrongly reuse a
            # tonemap output computed from different pixel data.
            processed = _linear_float_from_buffer(rgb_image)
            cache.stage_keys["tone"] = ("__default_fastpath__",)
            cache.stage_out["tone"] = processed
            return processed

        pre_key = _stage_key(merged, _PRE_TONE_KEYS)
        if cache.stage_keys.get("pre_tone") != pre_key or "pre_tone" not in cache.stage_out:
            img = _linear_float_from_buffer(rgb_image)
            from raw_transform import apply_geometry

            img = apply_geometry(img, merged)
            img = _apply_wb_tint(img, merged)

            exp_val = float(merged.get("Exposure2012", 0.0))
            if abs(exp_val) > 1e-4:
                img *= 2.0 ** exp_val

            mask = _resolve_db_mask(merged)
            if mask is not None:
                from raw_dodge_burn import apply_dodge_burn

                stops = float(merged.get(_DB_STRENGTH_KEY, _DB_DEFAULT_STRENGTH))
                img = apply_dodge_burn(img, mask, stops)

            do_denoise = chroma_denoise if chroma_denoise is not None else chroma_denoise_enabled()
            nr_amount = float(merged.get("ColorNoiseReduction", 0.0))
            method = int(float(merged.get("DenoiseMethod", 0.0)))
            if nr_amount > 1e-4:
                img = apply_chroma_denoise(
                    img, strength=min(1.5, nr_amount / 100.0 * 1.25), method=method, preview=preview
                )
            elif do_denoise and not preview:
                img = apply_chroma_denoise(img, method=method, preview=False)

            luma_nr_amount = float(merged.get("LuminanceNoiseReduction", 0.0))
            if luma_nr_amount > 1e-4:
                img = apply_luma_denoise(img, strength=luma_nr_amount / 100.0, method=method, preview=preview)

            cache.stage_out["pre_tone"] = img
            cache.stage_keys["pre_tone"] = pre_key
        pre_tone_out = cache.stage_out["pre_tone"]

        # Chained to pre_key: the tone stage's *input* is pre_tone_out, so a
        # change to a pre-tone-only key (e.g. Exposure2012) must invalidate
        # the tone stage too, even though no _TONE_KEYS value moved.
        # preview_lite is part of the key so flipping lite↔full never reuses
        # the other path's tone buffer (belt-and-suspenders vs dual caches).
        tone_key = (
            pre_key,
            _stage_key(merged, _TONE_KEYS),
            bool(preview_lite),
            bool(settle_fast),
        )
        if cache.stage_keys.get("tone") != tone_key or "tone" not in cache.stage_out:
            if uses_recovery_tone_map(merged):
                toned = pre_tone_out
            else:
                toned = apply_pv2012_tone_rgb(
                    pre_tone_out,
                    merged,
                    preview_lite=bool(preview_lite),
                    settle_fast=bool(settle_fast),
                )
            cache.stage_out["tone"] = np.clip(toned, 0.0, None)
            cache.stage_keys["tone"] = tone_key
        return cache.stage_out["tone"]


def _apply_display_stage_staged(
    img: np.ndarray, merged: dict[str, float], cache: "PreviewStageCache"
) -> np.ndarray:
    """Cached equivalent of _apply_display_stage for the preview path."""
    from raw_detail_enhance import apply_detail_enhancements
    from raw_hsl import apply_hsl_adjustments

    with cache.lock:
        tone_key = cache.stage_keys.get("tone")

        tonemap_key = (tone_key,)
        if cache.stage_keys.get("tonemap") != tonemap_key or "tonemap" not in cache.stage_out:
            cache.stage_out["tonemap"] = _tone_map_for_display(img, merged)
            cache.stage_keys["tonemap"] = tonemap_key
        display = cache.stage_out["tonemap"]

        color_key = (cache.stage_keys.get("tonemap"), _stage_key(merged, _COLOR_KEYS))
        if cache.stage_keys.get("color") != color_key or "color" not in cache.stage_out:
            out = _apply_display_color_adjustments(display, merged, preview=True)
            out = apply_hsl_adjustments(out, merged)
            cache.stage_out["color"] = out
            cache.stage_keys["color"] = color_key
        colored = cache.stage_out["color"]

        detail_key = (cache.stage_keys.get("color"), _stage_key(merged, _DETAIL_KEYS))
        if cache.stage_keys.get("detail") != detail_key or "detail" not in cache.stage_out:
            cache.stage_out["detail"] = apply_detail_enhancements(colored, merged)
            cache.stage_keys["detail"] = detail_key
        return cache.stage_out["detail"]


def render_adjust_preview_uint8(
    rgb_image: np.ndarray,
    adj: dict[str, float],
    cache: "PreviewStageCache",
    *,
    preview_lite: bool = False,
    settle_fast: bool = False,
) -> np.ndarray:
    """
    Staged/cached equivalent of process_linear_edit_buffer(preview=True) +
    linear_to_display_uint8, reusing intermediate buffers across calls on the
    same base image when only later-stage adjustment keys changed since the
    previous call. Used exclusively by the interactive Adjust-panel preview
    path (_AdjustPreviewWorker in main.py).

    ``preview_lite=True`` for the downsampled live-drag base only.
    ``settle_fast=True`` for Adjust settle (keeps y0 guide, skips chroma damp);
    export keeps the full PV2012 path.

    Encode must match ``linear_to_display_uint8`` (dcraw BT.709 / ``_gamma_lut8``),
    not IEC sRGB OETF — otherwise staged preview diverges from the uncached
    path and smoke tests (and on-screen parity with settle/export) break.
    """
    from fast_raw_decode import _gamma_lut8
    from raw_lut import apply_pipeline_lut_encoded, apply_pipeline_lut_linear
    from raw_tone_curve import apply_channel_curves_encoded

    merged = dict(DEFAULT_ADJUSTMENTS)
    merged.update(adj or {})
    processed = process_linear_edit_buffer_staged(
        rgb_image,
        merged,
        cache,
        preview=True,
        preview_lite=preview_lite,
        settle_fast=settle_fast,
    )
    display = _apply_display_stage_staged(processed, merged, cache)
    display = apply_pipeline_lut_linear(display, merged)
    # Copy before encode: display may be a live cache buffer; returning a view
    # into stage_out would let the next worker tick mutate pixels already
    # queued for the UI thread's QPixmap conversion.
    idx = np.clip(display * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    encoded = np.ascontiguousarray(_gamma_lut8()[idx])
    encoded = apply_pipeline_lut_encoded(encoded, merged, 255.0)
    return apply_channel_curves_encoded(encoded, merged, 255.0)


def _apply_display_color_adjustments(
    display_linear: np.ndarray,
    merged: dict[str, float],
    *,
    preview: bool = False,
) -> np.ndarray:
    """Saturation / vibrance / vignette / dehaze in display-linear space.

    Creative LUT is applied later via ``apply_pipeline_lut_*`` (Rec.709 after
    gamma encode by default; Linear keeps the pre-encode path).
    """
    from raw_effects import (
        VIGNETTE_MIDPOINT_DEFAULT,
        VIGNETTE_MIDPOINT_KEY,
        apply_dehaze,
        apply_vignette,
    )

    out = _apply_saturation_vibrance(display_linear, merged)
    dehaze = float(merged.get("Dehaze", 0.0) or 0.0)
    if abs(dehaze) > 1e-3:
        out = apply_dehaze(out, dehaze, preview=preview)
    vignette = float(merged.get("PostCropVignetteAmount", 0.0) or 0.0)
    if abs(vignette) > 1e-3:
        midpoint = float(
            merged.get(VIGNETTE_MIDPOINT_KEY, VIGNETTE_MIDPOINT_DEFAULT)
            or VIGNETTE_MIDPOINT_DEFAULT
        )
        out = apply_vignette(out, vignette, midpoint=midpoint)
    return out


def linear_to_display_uint8(img: np.ndarray, adj: dict[str, float] | None = None) -> np.ndarray:
    # dcraw BT.709 encode, NOT the IEC sRGB OETF: browse renders encode with
    # this exact curve, and the two differ most in the toe (12.92x vs 4.5x
    # linear slope) -- the sRGB encode lifted the deepest shadows ~3x over
    # the browse render, surfacing sensor noise ("grainy" report). One curve
    # everywhere = editor default is pixel-comparable with browse.
    from fast_raw_decode import _gamma_lut8
    from raw_lut import apply_pipeline_lut_encoded, apply_pipeline_lut_linear
    from raw_tone_curve import apply_channel_curves_encoded

    merged = dict(adj or {})
    n_workers = _linear_pipeline_worker_count(img.shape[0])
    if not uses_recovery_tone_map(merged) and n_workers > 1:
        display = _apply_display_stage_banded(img, merged, n_workers)
    else:
        display = _apply_display_stage(img, merged)
    display = apply_pipeline_lut_linear(display, merged)
    idx = np.clip(display * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    encoded = apply_pipeline_lut_encoded(_gamma_lut8()[idx], merged, 255.0)
    return apply_channel_curves_encoded(encoded, merged, 255.0)


def _apply_display_stage_banded(
    img: np.ndarray, merged: dict[str, float], n_workers: int
) -> np.ndarray:
    """Row-band parallel version of _apply_display_stage (tone map, sat/vib,
    HSL, detail-enhance). Only called for the clip tone-map path (the
    recovery path downsamples the whole image via cv2.resize and isn't
    row-independent). Reuses raw_adjustments._BAND_PAD_PX -- same blur radii,
    same padding math as _apply_adjustments_to_srgb_banded.

    Vignette and dehaze are whole-frame effects (radial center / atmospheric
    light). Applying them inside a row band re-centers the vignette on each
    strip and yields large preview/export mismatches — fall back to the
    single-pass path when either is active.
    """
    from raw_adjustments import _BAND_PAD_PX, band_ranges, banded_executor
    from raw_effects import DEHAZE_KEY, VIGNETTE_KEY

    if (
        abs(float(merged.get(VIGNETTE_KEY, 0.0) or 0.0)) > 1e-3
        or abs(float(merged.get(DEHAZE_KEY, 0.0) or 0.0)) > 1e-3
    ):
        return _apply_display_stage(img, merged)

    bands = band_ranges(img.shape[0], n_workers, pad_px=_BAND_PAD_PX)

    def _process_band(band):
        y0, y1, pad_top, pad_bot = band
        src = img[y0 - pad_top : y1 + pad_bot]
        out = _apply_display_stage(src, merged)
        return out[pad_top : pad_top + (y1 - y0)]

    import cv2

    prev_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        results = list(banded_executor().map(_process_band, bands))
    finally:
        cv2.setNumThreads(prev_threads)
    return np.concatenate(results, axis=0)


def linear_to_export_uint16_srgb(img: np.ndarray, adj: dict[str, float] | None = None) -> np.ndarray:
    """Display-referred 16-bit export, same dcraw BT.709 curve as display.

    (Name kept for call-site stability; the output is the app's standard
    BT.709-encoded rendering, tagged/treated as sRGB exactly like every
    LibRaw/dcraw-derived pipeline does.)
    """
    from fast_raw_decode import gamma_curve16
    from raw_lut import apply_pipeline_lut_encoded, apply_pipeline_lut_linear
    from raw_tone_curve import apply_channel_curves_encoded

    merged = dict(adj or {})
    display = _apply_display_stage(img, merged)
    display = apply_pipeline_lut_linear(display, merged)
    idx = np.clip(display * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    encoded = apply_pipeline_lut_encoded(gamma_curve16()[idx], merged, 65535.0)
    return apply_channel_curves_encoded(encoded, merged, 65535.0)


def _ensure_parent_dir(output_path: str) -> None:
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _chroma_denoise_for_export(adj: dict[str, float]) -> bool:
    return float((adj or {}).get("ColorNoiseReduction", 0.0)) > 0 or chroma_denoise_enabled()


def _process_for_export(rgb_linear: np.ndarray, adj: dict[str, float]) -> np.ndarray:
    return process_linear_edit_buffer(
        rgb_linear,
        adj,
        preview=False,
        chroma_denoise=_chroma_denoise_for_export(adj),
    )


def _xmp_extratags(embed_xmp_path: Optional[str]) -> list:
    if not embed_xmp_path or not os.path.isfile(embed_xmp_path):
        return []
    with open(embed_xmp_path, "rb") as f:
        xmp_bytes = f.read()
    return [(700, "B", len(xmp_bytes), xmp_bytes, False)]


def _write_16bit_rgb_tiff(
    output_path: str,
    rgb_uint16: np.ndarray,
    *,
    embed_xmp_path: Optional[str] = None,
) -> None:
    """
    Write a genuine 16-bit-per-channel, 3-sample RGB TIFF via ``tifffile``.

    Pillow's ``"RGB"`` mode is defined as 8-bit-per-channel -- there is no
    Pillow mode for 16-bit 3-channel data, so ``Image.fromarray(arr,
    mode="RGB")`` with a uint16 array has no valid type-lookup entry. On
    Pillow >=~10 (this project pins ">=10.0.0" with no upper bound) that
    raises outright; workarounds via ``Image.frombuffer`` with a "RGB;16*"
    raw mode "succeed" but silently discard the low byte, since the
    underlying storage is 8-bit regardless. Both are unacceptable for a
    "16-bit TIFF" export, so this path bypasses Pillow's Image model
    entirely for 16-bit RGB. ``compression="deflate"`` (not the previous
    LZW) needs no extra codec package; ``lzw`` requires ``imagecodecs``,
    which isn't a project dependency.
    """
    import tifffile

    extratags = _xmp_extratags(embed_xmp_path)
    _ensure_parent_dir(output_path)
    tifffile.imwrite(
        output_path,
        rgb_uint16,
        photometric="rgb",
        compression="deflate",
        extratags=extratags or None,
    )


def export_adjusted_tiff16(
    rgb_linear: np.ndarray,
    adj: dict[str, float],
    output_path: str,
    *,
    embed_xmp_path: Optional[str] = None,
    nn_denoise: bool = False,
) -> None:
    """Bake adjustments to 16-bit sRGB TIFF; optionally embed XMP packet."""
    processed = _process_for_export(rgb_linear, adj)
    out = linear_to_export_uint16_srgb(processed, adj)
    if nn_denoise:
        from raw_nn_denoise import denoise_display_uint16

        out = denoise_display_uint16(out)
    _write_16bit_rgb_tiff(output_path, out, embed_xmp_path=embed_xmp_path)


def export_adjusted_jpeg(
    rgb_linear: np.ndarray,
    adj: dict[str, float],
    output_path: str,
    *,
    quality: int = 92,
    nn_denoise: bool = False,
) -> None:
    """Bake adjustments to 8-bit JPEG."""
    processed = _process_for_export(rgb_linear, adj)
    out = linear_to_display_uint8(processed, adj)
    if nn_denoise:
        from raw_nn_denoise import denoise_display_uint8

        out = denoise_display_uint8(out)
    from PIL import Image

    im = Image.fromarray(out, mode="RGB")
    _ensure_parent_dir(output_path)
    im.save(
        output_path,
        format="JPEG",
        quality=max(1, min(100, int(quality))),
        subsampling=0,
        # Deliberately no optimize=True: that flag combined with
        # subsampling=0 triggers a libjpeg "Suspension not allowed here"
        # encoder crash on some Pillow/libjpeg builds -- reproduced with a
        # plain synthetic array, unrelated to any RAWViewer-specific data.
        # subsampling=0 (no chroma subsampling) matters for color fidelity;
        # optimize is a pure file-size optimization with no visible quality
        # difference, not worth the crash risk.
    )


def export_adjusted_webp(
    rgb_linear: np.ndarray,
    adj: dict[str, float],
    output_path: str,
    *,
    quality: int = 88,
    nn_denoise: bool = False,
) -> None:
    """Bake adjustments to 8-bit WebP."""
    processed = _process_for_export(rgb_linear, adj)
    out = linear_to_display_uint8(processed, adj)
    if nn_denoise:
        from raw_nn_denoise import denoise_display_uint8

        out = denoise_display_uint8(out)
    from PIL import Image

    im = Image.fromarray(out, mode="RGB")
    _ensure_parent_dir(output_path)
    im.save(
        output_path,
        format="WEBP",
        quality=max(1, min(100, int(quality))),
        method=4,
    )


EXPORT_FORMAT_TIFF16 = "tiff16"
EXPORT_FORMAT_JPEG = "jpeg"
EXPORT_FORMAT_WEBP = "webp"


class ExportCancelled(Exception):
    """Raised when the user cancels an in-progress export."""


def export_adjusted_image(
    export_format: str,
    *,
    output_path: str,
    rgb_linear: Optional[np.ndarray],
    adj: dict[str, float],
    embed_xmp_path: Optional[str] = None,
    cancel_check=None,
    progress_cb=None,
) -> None:
    """Dispatch baked export (TIFF16 / JPEG / WebP; "_nn" suffix = AI denoise).

    The "<fmt>_nn" variants run the realPLKSR export denoise
    (raw_nn_denoise.py) on the display-referred buffer just before encoding.
    Export-only: inference costs ~830 ms/MP even at fp16/CUDA, so it never
    belongs in the live preview path.
    """
    fmt = (export_format or EXPORT_FORMAT_TIFF16).strip().lower()
    nn = fmt.endswith("_nn")
    if nn:
        fmt = fmt[: -len("_nn")]
    if rgb_linear is None:
        raise RuntimeError("Full-resolution RAW decode failed")
    if cancel_check is not None and cancel_check():
        raise ExportCancelled()
    if nn:
        from raw_nn_denoise import set_cancel_check, set_progress_cb

        set_cancel_check(cancel_check)
        set_progress_cb(progress_cb)
    try:
        if fmt == EXPORT_FORMAT_JPEG:
            export_adjusted_jpeg(rgb_linear, adj, output_path, nn_denoise=nn)
        elif fmt == EXPORT_FORMAT_WEBP:
            export_adjusted_webp(rgb_linear, adj, output_path, nn_denoise=nn)
        else:
            export_adjusted_tiff16(
                rgb_linear, adj, output_path, embed_xmp_path=embed_xmp_path,
                nn_denoise=nn,
            )
    except Exception as e:
        if type(e).__name__ == "NNDenoiseCancelled":
            raise ExportCancelled() from e
        raise
    finally:
        if nn:
            from raw_nn_denoise import set_cancel_check, set_progress_cb

            set_cancel_check(None)
            set_progress_cb(None)
