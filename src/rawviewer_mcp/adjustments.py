"""Clamp and public-view helpers for MCP adjustment dicts."""

from __future__ import annotations

from typing import Any, Dict

from raw_adjustments import DEFAULT_ADJUSTMENTS, SLIDER_SPECS

# Keys that are masks / paint data — never returned or accepted in v1.
_MASK_KEY_NAMES = (
    "DodgeBurnMask",
    "SpotHealMask",
    "MaskLayers",
)

_NOTABLE_STRING_KEYS = (
    "CreativeLUTName",
    "CreativeLUTWorkingSpace",
    "_tone_curve_pv2012",
    "_tone_curve_pv2012_red",
    "_tone_curve_pv2012_green",
    "_tone_curve_pv2012_blue",
)


def _mask_keys() -> frozenset[str]:
    keys = set(_MASK_KEY_NAMES)
    try:
        from raw_dodge_burn import MASK_KEY, MASK_OBJ_KEY

        keys.add(MASK_KEY)
        keys.add(MASK_OBJ_KEY)
    except Exception:
        pass
    try:
        from raw_spot_heal import MASK_KEY as HEAL
        from raw_spot_heal import MASK_OBJ_KEY as HEAL_OBJ

        keys.add(HEAL)
        keys.add(HEAL_OBJ)
    except Exception:
        pass
    try:
        from raw_mask_layers import MASK_LAYERS_KEY, MASK_LAYERS_OBJ_KEY

        keys.add(MASK_LAYERS_KEY)
        keys.add(MASK_LAYERS_OBJ_KEY)
    except Exception:
        pass
    return frozenset(keys)


def _spec_range(key: str) -> tuple[float, float] | None:
    for spec in SLIDER_SPECS:
        if spec.key == key:
            lo = float(spec.slider_to_value(spec.minimum))
            hi = float(spec.slider_to_value(spec.maximum))
            if lo > hi:
                lo, hi = hi, lo
            return lo, hi
    return None


def _fallback_range(key: str) -> tuple[float, float] | None:
    if key in ("CropLeft", "CropRight", "CropTop", "CropBottom"):
        return 0.0, 1.0
    if key == "AnamorphicRatio":
        return 1.0, 2.0
    if key in ("LensCorrectionEnabled", "DenoiseMethod"):
        return 0.0, 1.0
    if key == "DodgeBurnStrength":
        return 0.0, 4.0
    if key.startswith("HueAdjustment") or key.startswith("SaturationAdjustment") or key.startswith(
        "LuminanceAdjustment"
    ):
        return -100.0, 100.0
    if key in ("ColorNoiseReduction", "CreativeLUTAmount"):
        return 0.0, 100.0
    if key in DEFAULT_ADJUSTMENTS:
        default = float(DEFAULT_ADJUSTMENTS[key])
        # Conservative band around defaults for keys without a slider spec.
        return default - 100.0, default + 100.0
    return None


def value_range_for_key(key: str) -> tuple[float, float] | None:
    spec = _spec_range(key)
    if spec is not None:
        return spec
    return _fallback_range(key)


def clamp_adjustment_value(key: str, value: Any) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Adjustment {key!r} must be numeric") from exc
    bounds = value_range_for_key(key)
    if bounds is None:
        raise ValueError(f"Unknown adjustment key: {key}")
    lo, hi = bounds
    if num < lo:
        return lo
    if num > hi:
        return hi
    return num


def clamp_patch(patch: Dict[str, Any]) -> Dict[str, float]:
    """Clamp slider keys; reject mask / unknown keys."""
    if not isinstance(patch, dict):
        raise ValueError("adjustments must be an object")
    masks = _mask_keys()
    out: Dict[str, float] = {}
    unknown: list[str] = []
    for key, raw in patch.items():
        if key in masks or key in _MASK_KEY_NAMES:
            raise ValueError("Brush / heal / mask keys are not supported in v1")
        if key in _NOTABLE_STRING_KEYS:
            # String develop keys are not patched via set_adjustments in v1
            # (presets may still apply them through apply_preset).
            unknown.append(str(key))
            continue
        if value_range_for_key(str(key)) is None:
            unknown.append(str(key))
            continue
        out[str(key)] = clamp_adjustment_value(str(key), raw)
    if unknown:
        raise ValueError("Unknown or unsupported adjustment keys: " + ", ".join(unknown))
    return out


def public_adjustments(adj: Dict[str, Any] | None) -> Dict[str, Any]:
    """Numeric sliders plus notable string keys; no mask blobs."""
    src = dict(adj or {})
    masks = _mask_keys()
    out: Dict[str, Any] = {}
    for key, default in DEFAULT_ADJUSTMENTS.items():
        if key in masks:
            continue
        val = src.get(key, default)
        try:
            out[key] = float(val)
        except (TypeError, ValueError):
            continue
    from raw_adjustments import AS_SHOT_TEMP_KEY

    if AS_SHOT_TEMP_KEY in src:
        try:
            out[AS_SHOT_TEMP_KEY] = float(src[AS_SHOT_TEMP_KEY])
        except (TypeError, ValueError):
            pass
    try:
        from raw_tone_curve import CHANNEL_CURVE_KEYS, TONE_CURVE_SERIAL_KEY

        string_keys = (TONE_CURVE_SERIAL_KEY,) + CHANNEL_CURVE_KEYS
    except Exception:
        string_keys = _NOTABLE_STRING_KEYS
    for key in (*string_keys, "CreativeLUTName", "CreativeLUTWorkingSpace"):
        if key in masks:
            continue
        val = src.get(key)
        if val is None:
            continue
        text = str(val).strip()
        if text:
            out[key] = text
    return out
