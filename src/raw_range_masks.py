"""Luminance / color range mask alphas for Mask Editing.

Both helpers take a **display-referred** RGB buffer (see ``to_display_rgb``)
and return a float32 (H, W) alpha in [0, 1]. The result is a *static*
coverage layer the user can then brush / erase / invert like any other mask
-- range params are not re-evaluated live (Lightroom-style parametric re-open
is a later step).

The space is not a detail. Every threshold below is a fraction of a
perceptual [0, 1] range, and the app's edit base is SCENE-LINEAR, where most
of a normal frame sits under 0.2. Handing these functions a linear buffer
therefore made one click select nearly the whole photo: measured on real CR3
frames, a centre click covered **99.2%** of the frame in linear against 81%
display-referred at the same settings. Callers must convert first --
``to_display_rgb`` -- and must sample the SAME buffer they measure, or the
centre and the band are in different spaces and the mask is arbitrary.

This is the same class of mistake raw_auto_adjust.py's header warns about at
length: thresholds are meaningless without the space they were sized for.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def to_display_rgb(rgb: np.ndarray) -> np.ndarray:
    """Scene-linear RGB -> display-referred sRGB, the space the user sees.

    A no-op for buffers already display-referred (uint8). Range selection is
    a judgement about the rendered photo -- the user clicks what is on screen
    -- so the measurement has to happen in the space the screen is in.
    """
    arr = np.asarray(rgb)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) * (1.0 / 255.0)
    from raw_tone_recovery import _encode_srgb_float01

    return _encode_srgb_float01(np.clip(arr.astype(np.float32, copy=False), 0.0, 1.0))


def _to_float_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) * (1.0 / 255.0)
    return arr.astype(np.float32, copy=False)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    # Rec. 709 luma on whatever space the edit base is in -- good enough
    # for "this tonal band" selection; absolute colorimetry is not the goal.
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def luminance_range_alpha(
    rgb: np.ndarray,
    *,
    center: float,
    # Measured on 6 real CR3 frames, centre click, display-referred: the old
    # 0.18/0.08 selected 74% of the frame on average (up to 91%), which is not
    # a selection. 0.06/0.05 lands at ~41% -- a tonal band is legitimately
    # broad, and the user widens it by brushing rather than starting from
    # "almost everything".
    width: float = 0.06,
    feather: float = 0.05,
) -> np.ndarray:
    """Soft band of coverage around ``center`` luminance in [0, 1].

    ``width`` is the half-width of the full-strength plateau; ``feather``
    is how far outside that plateau the alpha falls to zero.
    """
    img = _to_float_rgb(rgb)
    y = _luminance(img)
    c = float(np.clip(center, 0.0, 1.0))
    half = max(1e-4, float(width))
    soft = max(1e-4, float(feather))
    dist = np.abs(y - c)
    # Plateau then linear falloff.
    alpha = 1.0 - (dist - half) / soft
    np.clip(alpha, 0.0, 1.0, out=alpha)
    return alpha.astype(np.float32, copy=False)


def color_range_alpha(
    rgb: np.ndarray,
    sample_rgb: Sequence[float],
    *,
    # Same measurement as luminance_range_alpha: 0.18/0.10 averaged 53% of the
    # frame, 0.12/0.08 lands at ~37%.
    tolerance: float = 0.12,
    feather: float = 0.08,
) -> np.ndarray:
    """Soft coverage of pixels near ``sample_rgb`` in RGB space.

    Distance is Euclidean in linear RGB (cheap, stable). Chroma-only
    distance would miss brightness shifts the user usually wants included
    when sampling a sky or skin tone.
    """
    img = _to_float_rgb(rgb)
    target = np.asarray(sample_rgb, dtype=np.float32).reshape(3)
    target = np.clip(target, 0.0, 1.0)
    diff = img - target.reshape(1, 1, 3)
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    tol = max(1e-4, float(tolerance))
    soft = max(1e-4, float(feather))
    alpha = 1.0 - (dist - tol) / soft
    np.clip(alpha, 0.0, 1.0, out=alpha)
    return alpha.astype(np.float32, copy=False)


def sample_rgb_at(
    rgb: np.ndarray, x: float, y: float, *, radius: int = 2
) -> Optional[tuple[float, float, float]]:
    """Average a small neighborhood around (x, y) for stable color picks."""
    img = _to_float_rgb(rgb)
    h, w = img.shape[:2]
    if h < 1 or w < 1:
        return None
    cx = int(round(float(x)))
    cy = int(round(float(y)))
    if not (0 <= cx < w and 0 <= cy < h):
        return None
    r = max(0, int(radius))
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    patch = img[y0:y1, x0:x1]
    mean = patch.reshape(-1, 3).mean(axis=0)
    return float(mean[0]), float(mean[1]), float(mean[2])


def sample_luminance_at(rgb: np.ndarray, x: float, y: float, *, radius: int = 2) -> Optional[float]:
    sample = sample_rgb_at(rgb, x, y, radius=radius)
    if sample is None:
        return None
    r, g, b = sample
    return float(0.2126 * r + 0.7152 * g + 0.0722 * b)
