"""Display-space sharpness, clarity, and chromatic defringe for the edit pipeline."""

from __future__ import annotations

import numpy as np


def _luma(img: np.ndarray) -> np.ndarray:
    from raw_tone_recovery import _luminance

    return _luminance(img)


def _gaussian_blur_luma(y: np.ndarray, sigma: float) -> np.ndarray:
    y = np.ascontiguousarray(y.astype(np.float32))
    try:
        import cv2

        sigma = float(sigma)
        # A wide blur (e.g. Clarity's sigma=10) is inherently low-frequency,
        # so it can't represent detail finer than its own radius -- doing it
        # at full resolution is wasted work. Downsample 4x, blur at a
        # correspondingly smaller sigma, upsample back: ~15x faster (measured
        # 358ms -> 24ms at 32MP) with negligible error (mean abs diff
        # ~0.0007, max ~0.014 on a 0-1 scale). Left alone for small sigmas
        # (sharpness ~0.9, defringe ~1.5) where full-res detail matters and
        # the blur is already cheap.
        if sigma >= 4.0:
            h, w = y.shape[:2]
            small = cv2.resize(
                y, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA
            )
            blurred_small = cv2.GaussianBlur(small, (0, 0), sigma / 4.0)
            return cv2.resize(blurred_small, (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.GaussianBlur(y, (0, 0), sigma)
    except Exception:
        return y


def _unsharp_luma(img: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    if abs(amount) < 1e-5:
        return img
    work = np.ascontiguousarray(np.clip(img.astype(np.float32), 0.0, 1.0))
    y = _luma(work)
    blurred = _gaussian_blur_luma(y, sigma)
    y2 = np.clip(y + (y - blurred) * float(amount), 0.0, 1.0)
    scale = y2 / np.maximum(y, 1e-4)
    return np.clip(work * scale[..., np.newaxis], 0.0, 1.0)


def apply_sharpness(display_linear: np.ndarray, amount: float) -> np.ndarray:
    """Lightroom-style sharpening on display-linear RGB (0–150)."""
    val = float(amount)
    if val <= 0.0:
        return display_linear
    strength = (val / 150.0) * 1.35
    return _unsharp_luma(display_linear, sigma=0.9, amount=strength)


def apply_clarity2012(display_linear: np.ndarray, amount: float) -> np.ndarray:
    """Local midtone contrast (Clarity2012, −100…+100)."""
    val = float(amount)
    if abs(val) < 1e-4:
        return display_linear
    strength = (val / 100.0) * 0.75
    return _unsharp_luma(display_linear, sigma=10.0, amount=strength)


def apply_defringe(display_linear: np.ndarray, amount: float) -> np.ndarray:
    """
    Suppress purple/green fringing at high-contrast edges (display-linear RGB).

    Two fixes over the previous version:
    - ``green`` used a 1x gain (``g - 0.5*(r+b)``) vs. ``purple``'s 2x gain
      (``r+b - 2*g``) for the same absolute color imbalance, so green fringing
      was flagged as half as severe as equally-strong purple fringing. Both are
      now ``2*g`` vs. ``r+b`` scale, i.e. exact negations of each other
      (``fringe == |r + b - 2*g|``).
    - There was no edge/locality gate at all: any pixel whose hue leaned
      purple/green relative to its own chroma got desaturated, including
      uniform, unfringed regions (e.g. a purple flower lost ~90% saturation at
      Defringe=100 with no edge anywhere nearby) -- because ``fringe / span``
      is scale-invariant along the purple/green hue axis, not proportional to
      how strong the cast is. ``edge_weight`` gates the effect to pixels near
      an actual luminance transition (same Gaussian-blur local-contrast trick
      as clarity/sharpness, at a tight sigma matching typical fringe width),
      so flat colored regions are left alone.
    """
    val = float(amount)
    if val <= 0.0:
        return display_linear
    s = val / 100.0
    img = np.clip(display_linear.astype(np.float32), 0.0, 1.0)
    lum2d = _luma(img)
    lum = lum2d[..., np.newaxis]

    r, g, b = img[:, :, 0:1], img[:, :, 1:2], img[:, :, 2:3]
    purple = np.clip(r + b - 2.0 * g, 0.0, None)
    green = np.clip(2.0 * g - r - b, 0.0, None)
    fringe = np.maximum(purple, green)
    # np.max/min(img, axis=-1) is ~8x slower than a direct 3-way elementwise
    # chain for a size-3 last axis (measured 356ms vs 45ms at 32MP) -- numpy's
    # generic axis-reduce machinery doesn't specialize for this shape.
    span = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    color_mask = np.clip(fringe / (span + 1e-4), 0.0, 1.0)

    edge = np.abs(lum2d - _gaussian_blur_luma(lum2d, 1.5))[..., np.newaxis]
    edge_weight = edge / (edge + 0.04)  # soft knee: ~0 in flat regions, -> 1 at real edges

    mask = color_mask * edge_weight * s
    return np.clip(lum + (img - lum) * (1.0 - mask * 0.9), 0.0, None)


def apply_detail_enhancements(display_linear: np.ndarray, merged: dict[str, float]) -> np.ndarray:
    """Sharpness → clarity → defringe on display-linear float RGB."""
    if display_linear is None:
        return display_linear
    out = display_linear
    sharp = float(merged.get("Sharpness", 0.0))
    clarity = float(merged.get("Clarity2012", 0.0))
    defringe = float(merged.get("Defringe", 0.0))
    if sharp > 0.0:
        out = apply_sharpness(out, sharp)
    if abs(clarity) > 1e-4:
        out = apply_clarity2012(out, clarity)
    if defringe > 0.0:
        out = apply_defringe(out, defringe)
    return out
