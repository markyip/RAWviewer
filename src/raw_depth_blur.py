"""Continuous depth-of-field / background blur for mask layers.

Two modes share one gather-from-blur-bands path:

* **Depth (continuous CoC):** ``sigma ∝ |nearness − focus|``. Nearness is the
  Depth mask's alpha (1 = nearest, 0 = farthest) — the same convention as
  ``raw_ai_masks.estimate_depth``. ``focus`` picks the sharp plane.
* **Masked (soft region blur):** ``sigma ∝ weight`` where weight is the
  layer's effective coverage. Smart Object + Invert + Blur is the usual
  "blur the background" portrait path without a depth map.

Neither path invents bokeh discs; they are optical-ish Gaussian DOF. A
hexagonal / spectral bokeh kernel would be a later swap at the band-blur
step only.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# At Blur=100, farthest-from-focus pixels get this Gaussian sigma (pixels at
# the working buffer). Soft enough for a clear portrait separation without
# turning into a flat fog on typical half-res edit bases.
_MAX_SIGMA = 20.0
_PREVIEW_MAX_SIGMA = 12.0
_FULL_BANDS = 6
_PREVIEW_BANDS = 4


def _gaussian_blur_rgb(img: np.ndarray, sigma: float) -> np.ndarray:
    """Blur RGB; large sigmas downsample first (same idea as Clarity)."""
    import cv2

    sigma = float(sigma)
    if sigma < 0.15:
        return img
    work = np.ascontiguousarray(img, dtype=np.float32)
    if sigma >= 4.0:
        h, w = work.shape[:2]
        scale = 4 if sigma >= 8.0 else 2
        small = cv2.resize(
            work,
            (max(1, w // scale), max(1, h // scale)),
            interpolation=cv2.INTER_AREA,
        )
        blurred = cv2.GaussianBlur(small, (0, 0), sigma / float(scale))
        return cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(work, (0, 0), sigma)


def blur_pad_px(amount: float, *, preview: bool = False) -> int:
    """How many pixels of context a Blur of ``amount`` (0–100) needs."""
    amt = max(0.0, float(amount))
    if amt < 1e-4:
        return 0
    max_sigma = _PREVIEW_MAX_SIGMA if preview else _MAX_SIGMA
    return int(min(96, max(8, round(max_sigma * (amt / 100.0) * 2.5))))


def apply_variable_blur(
    img: np.ndarray,
    sigma_map: np.ndarray,
    *,
    preview: bool = False,
) -> np.ndarray:
    """Gather a continuously-varying Gaussian blur from discrete bands.

    ``sigma_map`` is per-pixel sigma in pixels (same resolution as ``img``).
    Pixels at sigma≈0 keep the sharp source.
    """
    if img is None or sigma_map is None:
        return img
    h, w = img.shape[:2]
    if sigma_map.shape[:2] != (h, w):
        import cv2

        sigma_map = cv2.resize(
            sigma_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
        )
    sigma_map = np.asarray(sigma_map, dtype=np.float32)
    peak = float(np.max(sigma_map)) if sigma_map.size else 0.0
    if peak < 0.15:
        return img

    n_bands = _PREVIEW_BANDS if preview else _FULL_BANDS
    # Band 0 = sharp; remaining bands evenly space up to peak.
    sigmas = [0.0]
    for i in range(1, n_bands):
        sigmas.append(peak * (i / float(n_bands - 1)))

    levels = [np.ascontiguousarray(img, dtype=np.float32)]
    for s in sigmas[1:]:
        levels.append(_gaussian_blur_rgb(levels[0], s))

    # Map each pixel onto the band continuum.
    t = np.clip(sigma_map / max(peak, 1e-6) * (n_bands - 1), 0.0, float(n_bands - 1))
    i0 = np.floor(t).astype(np.int32)
    i1 = np.minimum(i0 + 1, n_bands - 1)
    f = (t - i0.astype(np.float32))[..., np.newaxis]

    out = np.zeros_like(levels[0])
    # Vectorized scatter-add. A Python per-channel loop over 12 MP frames
    # was the gather cost; n_bands is still only 4–6.
    fade = f[..., 0]
    for b in range(n_bands):
        w0 = ((i0 == b).astype(np.float32) * (1.0 - fade))[..., np.newaxis]
        out += levels[b] * w0
        w1 = (((i1 == b) & (i0 != i1)).astype(np.float32) * fade)[..., np.newaxis]
        out += levels[b] * w1
    return out


def sigma_map_from_depth(
    nearness: np.ndarray,
    amount: float,
    focus: float,
    *,
    preview: bool = False,
) -> np.ndarray:
    """Circle-of-confusion map: sharp at ``focus``, soft farther away.

    ``nearness`` / ``focus`` in [0, 1] (1 = nearest).
    """
    amt = max(0.0, min(100.0, float(amount))) / 100.0
    if amt < 1e-6:
        return np.zeros(nearness.shape[:2], dtype=np.float32)
    foc = float(np.clip(focus, 0.0, 1.0))
    near = np.clip(np.asarray(nearness, dtype=np.float32), 0.0, 1.0)
    coc = np.abs(near - foc)
    max_sigma = _PREVIEW_MAX_SIGMA if preview else _MAX_SIGMA
    return (coc * amt * max_sigma).astype(np.float32)


def sigma_map_from_weight(
    weight: np.ndarray,
    amount: float,
    *,
    preview: bool = False,
) -> np.ndarray:
    """Soft region blur: more coverage → more blur (masked background path)."""
    amt = max(0.0, min(100.0, float(amount))) / 100.0
    if amt < 1e-6:
        return np.zeros(weight.shape[:2], dtype=np.float32)
    w = np.clip(np.asarray(weight, dtype=np.float32), 0.0, 1.0)
    max_sigma = _PREVIEW_MAX_SIGMA if preview else _MAX_SIGMA
    return (w * amt * max_sigma).astype(np.float32)


def apply_mask_blur(
    img: np.ndarray,
    weight: np.ndarray,
    amount: float,
    *,
    continuous_depth: bool = False,
    focus: float = 1.0,
    preview: bool = False,
) -> np.ndarray:
    """Blur ``img`` using ``weight`` as depth nearness or coverage.

    When ``continuous_depth`` is True, ``weight`` is nearness and ``focus``
    selects the sharp plane. Otherwise ``weight`` is blur coverage (0–1).
    """
    if img is None or float(amount) < 1e-4:
        return img
    if continuous_depth:
        sigma = sigma_map_from_depth(weight, amount, focus, preview=preview)
    else:
        sigma = sigma_map_from_weight(weight, amount, preview=preview)
    return apply_variable_blur(img, sigma, preview=preview)
