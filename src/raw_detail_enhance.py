"""Display-space sharpness, clarity, and chromatic defringe for the edit pipeline."""

from __future__ import annotations

import numpy as np


def _luma(img: np.ndarray) -> np.ndarray:
    from raw_tone_recovery import _luminance

    return _luminance(img)


def _unsharp_luma(img: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    if abs(amount) < 1e-5:
        return img
    import cv2

    y = _luma(img).astype(np.float32)
    blurred = cv2.GaussianBlur(y, (0, 0), float(sigma))
    y2 = np.clip(y + (y - blurred) * float(amount), 0.0, None)
    scale = y2 / np.maximum(y, 1e-6)
    return np.clip(img * scale[..., np.newaxis], 0.0, None)


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
    """Suppress purple/green fringing on high-chroma edge regions."""
    val = float(amount)
    if val <= 0.0:
        return display_linear
    s = val / 100.0
    img = display_linear.astype(np.float32)
    lum = _luma(img)[..., np.newaxis]
    purple = np.clip(img[:, :, 0:1] + img[:, :, 2:3] - 2.0 * img[:, :, 1:2], 0.0, None)
    green = np.clip(img[:, :, 1:2] - 0.5 * (img[:, :, 0:1] + img[:, :, 2:3]), 0.0, None)
    fringe = np.maximum(purple, green)
    span = np.max(img, axis=-1, keepdims=True) - np.min(img, axis=-1, keepdims=True)
    mask = np.clip(fringe / (span + 1e-4), 0.0, 1.0) * s
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
