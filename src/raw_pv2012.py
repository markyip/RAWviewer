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
    """
    Whites raises the input needed to hit output 1.0 (more headroom = less
    clipping = darker at the top when whites > 0 is wrong -- whites > 0 must
    brighten/clip highlights, matching Lightroom and this module's own legacy
    gamma-space path). Signs are therefore inverted from the naive "raise the
    point" reading: whites > 0 lowers white_pt (less input needed to clip),
    blacks > 0 lowers black_pt (extends below 0, lifting shadows off the floor).
    """
    w = whites / 100.0
    b = blacks / 100.0
    # 0.25, not the original 0.12: at +-12% max shift, Whites/Blacks barely
    # moved the black/white point at all -- nowhere near strong enough to
    # visibly clip highlights/crush shadows the way real Whites/Blacks
    # sliders do at their extremes. 0.25 gives genuine clipping at +-100
    # (worst-case combined span, both maxed toward each other, is 0.5 --
    # still comfortably away from a degenerate/near-zero span).
    white_pt = 1.0 - w * 0.25
    black_pt = -b * 0.25
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
            # Coefficient capped at 0.15 (not the more intuitive-looking 0.42): the
            # shadow-lift weight's slope reaches ~5.85 at its steepest (y~0.11), so
            # any coefficient above ~0.17 makes d(out)/dy go negative there --
            # a real, verified tone-curve inversion (banding) that showed up at
            # Shadows>=~45 with the old 0.42. 0.15 keeps a full-range monotonic
            # curve at Shadows=100 with margin to spare.
            out = out + sw * sh * (1.0 - out) * 0.15
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
    # y0 must stay the true pre-curve, pre-PV2012 baseline: it's the ratio's
    # denominator below. Reassigning it to the curve-adjusted value here (as
    # this used to do) makes the point tone curve's effect appear in *both*
    # the numerator (y1, computed from the curved value) and the denominator,
    # so it exactly cancels out of the ratio -- the point curve UI was fully
    # wired end-to-end (widget -> XMP -> pipeline) but had zero visible
    # effect on the image, reported as "the tone curve is unresponsive".
    y_curve = apply_tone_curve_perceptual(y0, adj)
    y1 = _apply_pv2012_perceptual(
        y_curve,
        contrast=float(adj.get("Contrast2012", 0.0)),
        highlights=float(adj.get("Highlights2012", 0.0)),
        shadows=float(adj.get("Shadows2012", 0.0)),
        whites=float(adj.get("Whites2012", 0.0)),
        blacks=float(adj.get("Blacks2012", 0.0)),
    )

    # Floor applies to *both* numerator and denominator: it exists only to
    # avoid a near-zero-denominator blowup, not to bias the ratio itself. A
    # floor on the denominator alone made "no sliders touched" (y1 == y0)
    # into a non-identity operation for any pixel below the floor -- e.g.
    # y0=0.005 produced ratio=0.5 (max darkening) with every slider at
    # default, silently crushing deep shadows/blacks before any user
    # adjustment, and fighting against genuine shadow/black recovery
    # attempts in exactly the tonal range users reach for it.
    ratio = np.maximum(y1, _PERCEPTUAL_LUM_FLOOR) / np.maximum(y0, _PERCEPTUAL_LUM_FLOOR)
    ratio = np.clip(ratio, _MIN_TONE_RATIO, _MAX_TONE_RATIO)
    out = img * ratio[..., np.newaxis]

    # Shadow lift amplifies demosaic chroma noise — pull chroma toward luma.
    sh = float(adj.get("Shadows2012", 0.0))
    if sh > 1e-4:
        sw = _region_weight_shadows(y_curve)
        damp = 1.0 - sw[..., np.newaxis] * min(sh / 100.0, 1.0) * 0.55
        luma = lum[..., np.newaxis]
        chroma = out - luma
        out = luma + chroma * damp

    return np.clip(out, 0.0, None)
