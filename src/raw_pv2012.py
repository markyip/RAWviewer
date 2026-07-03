"""
Adobe Lightroom / Camera Raw Process Version 2012 tone engine.

PV2012 (XMP ProcessVersion 11.0) applies a built-in medium-contrast base curve,
then parametric Exposure / Contrast / Highlights / Shadows / Whites / Blacks in
perceptual space. Tone work is luminance-only to preserve hue.
"""

from __future__ import annotations

import numpy as np

from raw_tone_curve import apply_tone_curve_perceptual

PROCESS_VERSION = "11.0"
TONE_CURVE_NAME_2012 = "Linear"

# Medium-contrast base baked into PV2012 “Linear” (0–255 coordinates).
_BASE_CURVE_XY = np.array(
    [
        [0.0, 0.0],
        [32.0 / 255.0, 22.0 / 255.0],
        [64.0 / 255.0, 56.0 / 255.0],
        [128.0 / 255.0, 128.0 / 255.0],
        [192.0 / 255.0, 196.0 / 255.0],
        [1.0, 1.0],
    ],
    dtype=np.float32,
)

# Default parametric region splits (Shadows | Darks | Lights | Highlights).
_SPLIT_SHADOWS = 0.25
_SPLIT_DARKS = 0.50
_SPLIT_LIGHTS = 0.75

_LUT_SIZE = 65536
_BASE_LUT = np.interp(
    np.linspace(0.0, 1.0, _LUT_SIZE, dtype=np.float32),
    _BASE_CURVE_XY[:, 0],
    _BASE_CURVE_XY[:, 1],
).astype(np.float32)

# Cap multiplicative lift in deep shadows (prevents demosaic green blow-up).
_PERCEPTUAL_LUM_FLOOR = 0.03
_MAX_TONE_RATIO = 3.0
_MIN_TONE_RATIO = 0.5


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    )


def _region_weight_highlights(y: np.ndarray) -> np.ndarray:
    span = max(1.0 - _SPLIT_LIGHTS, 1e-6)
    t = np.clip((y - _SPLIT_LIGHTS) / span, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _region_weight_shadows(y: np.ndarray) -> np.ndarray:
    t = np.clip((_SPLIT_SHADOWS - y) / max(_SPLIT_SHADOWS, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _apply_base_curve(y: np.ndarray) -> np.ndarray:
    yi = np.clip(y, 0.0, 1.0) * (_LUT_SIZE - 1)
    lo = np.floor(yi).astype(np.int32)
    hi = np.minimum(lo + 1, _LUT_SIZE - 1)
    frac = yi - lo
    return _BASE_LUT[lo] * (1.0 - frac) + _BASE_LUT[hi] * frac


def _scene_to_perceptual(lum: np.ndarray) -> np.ndarray:
    """Scene-linear luminance → PV2012 perceptual working space."""
    t = np.clip(lum.astype(np.float32), 0.0, None)
    t = t / (t + 0.25)
    return _apply_base_curve(t)


def _apply_pv2012_contrast(y: np.ndarray, contrast_val: float) -> np.ndarray:
    c = contrast_val / 100.0
    factor = (259.0 * (c * 100.0 + 255.0)) / (255.0 * (259.0 - c * 100.0))
    return (y - 0.5) * factor + 0.5


def _apply_whites_blacks(y: np.ndarray, whites: float, blacks: float) -> np.ndarray:
    w = whites / 100.0
    b = blacks / 100.0
    white_pt = 1.0 + w * 0.12
    black_pt = b * 0.12
    span = max(white_pt - black_pt, 1e-6)
    return np.clip((y - black_pt) / span, 0.0, 1.0)


def _apply_highlights_shadows(y: np.ndarray, highlights: float, shadows: float) -> np.ndarray:
    hi = highlights / 100.0
    sh = shadows / 100.0
    if abs(hi) < 1e-6 and abs(sh) < 1e-6:
        return y
    out = y.copy()
    if abs(sh) > 1e-6:
        sw = _region_weight_shadows(out)
        if sh > 0:
            out = out + sw * sh * (1.0 - out) * 0.42
        else:
            out = out + sw * sh * out * 0.38
    if abs(hi) > 1e-6:
        hw = _region_weight_highlights(out)
        if hi < 0:
            knee = np.maximum(out - _SPLIT_LIGHTS, 0.0)
            out = out + hw * hi * knee * 0.55
        else:
            out = out + hw * hi * (1.0 - out) * 0.35
    return np.clip(out, 0.0, 1.0)


def _apply_pv2012_perceptual(
    y: np.ndarray,
    *,
    contrast: float = 0.0,
    highlights: float = 0.0,
    shadows: float = 0.0,
    whites: float = 0.0,
    blacks: float = 0.0,
) -> np.ndarray:
    out = y.copy()
    if abs(contrast) > 1e-4:
        out = _apply_pv2012_contrast(out, contrast)
    if abs(highlights) > 1e-4 or abs(shadows) > 1e-4:
        out = _apply_highlights_shadows(out, highlights, shadows)
    if abs(whites) > 1e-4 or abs(blacks) > 1e-4:
        out = _apply_whites_blacks(out, whites, blacks)
    return np.clip(out, 0.0, 1.0)


def apply_pv2012_tone_rgb(img: np.ndarray, adj: dict[str, float]) -> np.ndarray:
    """Hue-preserving PV2012 tone; ratio capped in perceptual space."""
    lum = _luminance(img)
    y0 = _scene_to_perceptual(lum)
    y0 = apply_tone_curve_perceptual(y0, adj)
    y1 = _apply_pv2012_perceptual(
        y0,
        contrast=float(adj.get("Contrast2012", 0.0)),
        highlights=float(adj.get("Highlights2012", 0.0)),
        shadows=float(adj.get("Shadows2012", 0.0)),
        whites=float(adj.get("Whites2012", 0.0)),
        blacks=float(adj.get("Blacks2012", 0.0)),
    )

    ratio = y1 / np.maximum(y0, _PERCEPTUAL_LUM_FLOOR)
    ratio = np.clip(ratio, _MIN_TONE_RATIO, _MAX_TONE_RATIO)
    out = img * ratio[..., np.newaxis]

    # Shadow lift amplifies demosaic chroma noise — pull chroma toward luma.
    sh = float(adj.get("Shadows2012", 0.0))
    if sh > 1e-4:
        sw = _region_weight_shadows(y0)
        damp = 1.0 - sw[..., np.newaxis] * min(sh / 100.0, 1.0) * 0.55
        luma = lum[..., np.newaxis]
        chroma = out - luma
        out = luma + chroma * damp

    return np.clip(out, 0.0, None)
