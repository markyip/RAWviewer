"""Tone curve, WB, as-shot temperature, and default-adjustment checks."""

from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np

from processing.raw_decode.gil import rawpy_imread_warm

from .defaults import (
    _EXPORT_ONLY_DENOISE_KEYS,
    _PV2012_RECOVERY_EXCLUSIVE_KEYS,
    _PV2012_TONE_SLIDER_KEYS,
    AS_SHOT_TEMP_KEY,
    DEFAULT_ADJUSTMENTS,
    RECOVERY_BASELINE_HIGHLIGHTS2012,
    RECOVERY_BASELINE_KEY,
    RECOVERY_BASELINE_SHADOWS2012,
    _kelvin_to_rgb,
)


def tone_curve_serial_is_active(serial: object) -> bool:
    """True when a tone-curve serial encodes a non-identity edit."""
    text = str(serial or "").strip()
    if not text:
        return False
    from raw_adjustments.tone_curve import (
        deserialize_tone_curve_points,
        is_identity_tone_curve_points,
    )

    points = deserialize_tone_curve_points(text)
    if len(points) < 2:
        return False
    return not is_identity_tone_curve_points(points)


def uses_recovery_tone_map(adj: dict[str, float] | None) -> bool:
    """True when adjust should use P-key recovery tone instead of Reinhard + PV2012."""
    if not adj:
        return False
    if float(adj.get(RECOVERY_BASELINE_KEY, 0.0)) <= 0.5:
        return False
    from raw_adjustments.tone_curve import CHANNEL_CURVE_KEYS, TONE_CURVE_SERIAL_KEY

    if tone_curve_serial_is_active(adj.get(TONE_CURVE_SERIAL_KEY)):
        return False
    for key in CHANNEL_CURVE_KEYS:
        if tone_curve_serial_is_active(adj.get(key)):
            return False
    for key in _PV2012_RECOVERY_EXCLUSIVE_KEYS:
        default = float(DEFAULT_ADJUSTMENTS.get(key, 0.0))
        if abs(float(adj.get(key, default)) - default) > 1e-4:
            return False
    return True


def recovery_baseline_slider_hints() -> dict[str, float]:
    """PV2012 Highlights/Shadows readouts shown when recovery look is enabled."""
    return {
        "Shadows2012": RECOVERY_BASELINE_SHADOWS2012,
        "Highlights2012": RECOVERY_BASELINE_HIGHLIGHTS2012,
    }


def is_pv2012_tone_slider(key: str) -> bool:
    return key in _PV2012_TONE_SLIDER_KEYS

_AS_SHOT_TEMP_CACHE: OrderedDict[str, float] = OrderedDict()
_AS_SHOT_TEMP_CACHE_MAX = 2048


def _as_shot_temp_cache_put(cache_key: str, value: float) -> None:
    _AS_SHOT_TEMP_CACHE[cache_key] = float(value)
    _AS_SHOT_TEMP_CACHE.move_to_end(cache_key)
    while len(_AS_SHOT_TEMP_CACHE) > _AS_SHOT_TEMP_CACHE_MAX:
        _AS_SHOT_TEMP_CACHE.popitem(last=False)


def read_as_shot_temperature(image_path: str) -> float:
    """Best-effort as-shot CCT from EXIF or RAW metadata (not a UI preset)."""
    if not image_path:
        return DEFAULT_ADJUSTMENTS["Temperature"]
    cache_key = os.path.normcase(os.path.abspath(image_path))
    cached = _AS_SHOT_TEMP_CACHE.get(cache_key)
    if cached is not None:
        _AS_SHOT_TEMP_CACHE.move_to_end(cache_key)
        return cached
    result = DEFAULT_ADJUSTMENTS["Temperature"]
    resolved = False
    try:
        from metadata import backend as metadata_backend

        tags = metadata_backend.process_file_from_path(image_path, details=False)
        for tag in (
            "EXIF ColorTemperature",
            "MakerNote ColorTemperature",
            "EXIF WB_RGGBLevelsAsShot",
        ):
            raw = tags.get(tag)
            if raw is None:
                continue
            if tag.endswith("ColorTemperature"):
                val = int(float(str(raw).split()[0]))
                if 2000 <= val <= 12000:
                    result = float(val)
                    resolved = True
                    break
    except Exception:
        pass
    if not resolved:
        try:

            from enhanced_raw_processor import _rawpy_global_lock

            with _rawpy_global_lock:
                raw_ctx = rawpy_imread_warm(image_path)
            with raw_ctx as raw:
                cam = np.array(raw.camera_whitebalance[:3], dtype=np.float64)
                day = np.array(raw.daylight_whitebalance[:3], dtype=np.float64)
                if np.all(cam > 0) and np.all(day > 0):
                    rb_cam = cam[0] / cam[2]
                    rb_day = day[0] / day[2]
                    ratio = rb_cam / max(rb_day, 1e-6)
                    est = 5500.0 * (ratio ** -0.35)
                    result = float(np.clip(est, 2000.0, 12000.0))
        except Exception:
            pass
    _as_shot_temp_cache_put(cache_key, result)
    return result

def wb_reference_temperature(adj: dict[str, float] | None) -> float:
    if not adj:
        return DEFAULT_ADJUSTMENTS["Temperature"]
    return float(adj.get(AS_SHOT_TEMP_KEY, DEFAULT_ADJUSTMENTS["Temperature"]))


def is_neutral_wb(adj: dict[str, float] | None) -> bool:
    if not adj:
        return True
    ref = wb_reference_temperature(adj)
    temp = float(adj.get("Temperature", ref))
    tint = float(adj.get("Tint", 0.0))
    return abs(temp - ref) <= 0.5 and abs(tint) <= 1e-4


def solve_white_balance_from_sample(
    r: float, g: float, b: float, ref_temp: float
) -> tuple[float, float]:
    """
    White-balance dropper: solve Temperature/Tint so a sampled scene-linear RGB
    pixel (as-shot camera WB, pre Temperature/Tint) becomes neutral gray.

    Exact inverse of ``_apply_wb_tint`` rather than a new color model: Temperature
    scales R/B oppositely relative to G, so the R/B ratio depends only on
    Temperature (bisection, monotonic over the 2000-12000K slider range); Tint then
    scales G alone, solved algebraically once Temperature is fixed. Samples outside
    what the slider range / Tint's +/-10% G scaling can correct are clamped to the
    nearest achievable value rather than raising an error.
    """
    r = max(float(r), 1e-6)
    g = max(float(g), 1e-6)
    b = max(float(b), 1e-6)
    ref_temp = float(ref_temp) if ref_temp and ref_temp > 0 else DEFAULT_ADJUSTMENTS["Temperature"]
    r_ref, g_ref, b_ref = _kelvin_to_rgb(ref_temp)

    def rb_scale_ratio(t: float) -> float:
        r_t, g_t, b_t = _kelvin_to_rgb(t)
        scale_r = (r_t / r_ref) / (g_t / g_ref)
        scale_b = (b_t / b_ref) / (g_t / g_ref)
        return scale_r / scale_b

    target = b / r
    lo, hi = 2000.0, 12000.0
    ratio_lo, ratio_hi = rb_scale_ratio(lo), rb_scale_ratio(hi)  # monotonically decreasing
    if target >= ratio_lo:
        temperature = lo
    elif target <= ratio_hi:
        temperature = hi
    else:
        for _ in range(40):
            mid = (lo + hi) / 2.0
            if rb_scale_ratio(mid) > target:
                lo = mid
            else:
                hi = mid
        temperature = (lo + hi) / 2.0

    r_t, g_t, b_t = _kelvin_to_rgb(temperature)
    scale_r = (r_t / r_ref) / (g_t / g_ref)
    final_r = r * scale_r
    tint = ((1.0 - final_r / g) / 0.1) * 150.0 if g > 1e-6 else 0.0
    tint = max(-150.0, min(150.0, tint))
    return temperature, tint

def is_default_adjustments(adj: dict[str, float] | None) -> bool:
    if not adj:
        return True
    from raw_adjustments.dodge_burn import MASK_KEY
    from raw_adjustments.lut import LUT_AMOUNT_KEY, LUT_NAME_KEY
    from raw_adjustments.spot_heal import MASK_KEY as HEAL_MASK_KEY
    from raw_adjustments.tone_curve import CHANNEL_CURVE_KEYS, TONE_CURVE_SERIAL_KEY

    # Identity serials ("0,0;255,255") are visually a no-op; treat them as
    # default so Reset / clear can delete the sidecar and skip baking a stale
    # edited preview back into gallery thumbs.
    if tone_curve_serial_is_active(adj.get(TONE_CURVE_SERIAL_KEY)):
        return False
    for key in CHANNEL_CURVE_KEYS:
        if tone_curve_serial_is_active(adj.get(key)):
            return False
    if str(adj.get(MASK_KEY, "") or "").strip():
        return False
    if str(adj.get(HEAL_MASK_KEY, "") or "").strip():
        return False
    # Live paint injects mask objects without a PNG serial yet ΓÇö still non-default.
    try:
        from raw_adjustments.dodge_burn import MASK_OBJ_KEY
        from raw_adjustments.spot_heal import MASK_OBJ_KEY as HEAL_MASK_OBJ_KEY

        if adj.get(MASK_OBJ_KEY) is not None or adj.get(HEAL_MASK_OBJ_KEY) is not None:
            return False
    except Exception:
        pass
    lut_name = str(adj.get(LUT_NAME_KEY, "") or "").strip()
    try:
        lut_amt = float(adj.get(LUT_AMOUNT_KEY, 0.0) or 0.0)
    except (TypeError, ValueError):
        lut_amt = 0.0
    if lut_name and abs(lut_amt) > 1e-4:
        return False
    ref_temp = wb_reference_temperature(adj)
    for key, default in DEFAULT_ADJUSTMENTS.items():
        # Strength only matters with a painted mask; alone it must not keep
        # an otherwise-empty sidecar after Reset.
        if key == "DodgeBurnStrength":
            continue
        # Export-only AI-denoise knobs: changes nothing in the always-shown
        # preview, so they must not mark the image as edited on their own.
        if key in _EXPORT_ONLY_DENOISE_KEYS:
            continue
        try:
            av = float(adj.get(key, default) if adj.get(key, default) is not None else default)
        except (TypeError, ValueError):
            return False
        bv = float(default)
        if key == "Temperature":
            if abs(av - ref_temp) > 0.5:
                return False
        elif abs(av - bv) > 1e-4:
            return False
    return True


def is_default_adjustments_for_sidecar(adj: dict[str, float] | None) -> bool:
    """Like is_default_adjustments, but does NOT ignore export-only denoise knobs.

    is_default_adjustments() deliberately skips DenoiseDetail / AIDenoiseAmount
    (they are export-only knobs invisible to the always-shown preview, so they
    shouldn't mark a file "edited" for the gallery badge or skip the
    pixel-pipeline shortcut). But write_xmp_adjustments reused that SAME
    check to decide whether to write or clear the sidecar -- so a
    denoise-setting-only change was treated as "no edit" and silently
    dropped. Callers that persist the sidecar must use this variant.
    """
    if not is_default_adjustments(adj):
        return False
    if not adj:
        return True
    for key in _EXPORT_ONLY_DENOISE_KEYS:
        default = DEFAULT_ADJUSTMENTS[key]
        try:
            value = float(adj.get(key, default))
        except (TypeError, ValueError):
            return False
        if abs(value - default) > 1e-4:
            return False
    return True


def adjustments_equal(a: dict[str, float], b: dict[str, float]) -> bool:
    from raw_adjustments.lut import LUT_NAME_KEY, LUT_SPACE_KEY, lut_working_space

    if str(a.get(LUT_NAME_KEY, "") or "").strip() != str(b.get(LUT_NAME_KEY, "") or "").strip():
        return False
    if lut_working_space(a) != lut_working_space(b):
        # Only matters when a LUT is active; still compare so UI dirty-checks work.
        if str(a.get(LUT_NAME_KEY, "") or "").strip() or str(b.get(LUT_NAME_KEY, "") or "").strip():
            return False
        if str(a.get(LUT_SPACE_KEY, "") or "").strip() != str(b.get(LUT_SPACE_KEY, "") or "").strip():
            return False
    ref_a = wb_reference_temperature(a)
    ref_b = wb_reference_temperature(b)
    for key, default in DEFAULT_ADJUSTMENTS.items():
        av = float(a.get(key, default))
        bv = float(b.get(key, default))
        if key == "Temperature":
            da = av - ref_a
            db = bv - ref_b
            if abs(da - db) > 0.5:
                return False
        elif abs(av - bv) > 1e-4:
            return False
    return True
