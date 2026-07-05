"""Lightroom-style HSL per-color adjustments in display-linear RGB."""

from __future__ import annotations

import numpy as np

HSL_COLOR_NAMES: tuple[str, ...] = (
    "Red",
    "Orange",
    "Yellow",
    "Green",
    "Aqua",
    "Blue",
    "Purple",
    "Magenta",
)

# Hue centers (degrees) and falloff width for smooth bands.
_HSL_BANDS: tuple[tuple[str, float, float], ...] = (
    ("Red", 0.0, 42.0),
    ("Orange", 30.0, 38.0),
    ("Yellow", 55.0, 40.0),
    ("Green", 120.0, 50.0),
    ("Aqua", 175.0, 40.0),
    ("Blue", 220.0, 45.0),
    ("Purple", 275.0, 40.0),
    ("Magenta", 315.0, 45.0),
)


def _rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import cv2

    bgr = np.clip(rgb[..., ::-1], 0.0, 1.0).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].copy()
    s = hsv[:, :, 1].copy()
    v = hsv[:, :, 2].copy()
    return h, s, v


def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    import cv2

    hsv = np.stack([np.clip(h, 0.0, 360.0), np.clip(s, 0.0, 1.0), np.clip(v, 0.0, 1.0)], axis=-1)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return np.clip(bgr[..., ::-1], 0.0, 1.0)


def _hue_distance(h: np.ndarray, center: float) -> np.ndarray:
    d = np.abs(h - center)
    return np.minimum(d, 360.0 - d)


def _band_weight(h: np.ndarray, center: float, width: float) -> np.ndarray:
    d = _hue_distance(h, center)
    t = np.clip(1.0 - d / max(width, 1e-3), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def apply_hsl_adjustments(display_linear: np.ndarray, adj: dict[str, float]) -> np.ndarray:
    """Per-color hue / saturation / luminance (Lightroom HSL panel)."""
    if display_linear is None:
        return display_linear
    active = False
    for color in HSL_COLOR_NAMES:
        if any(
            abs(float(adj.get(f"{kind}Adjustment{color}", 0.0))) > 1e-4
            for kind in ("Hue", "Saturation", "Luminance")
        ):
            active = True
            break
    if not active:
        return display_linear

    rgb = np.clip(display_linear.astype(np.float32), 0.0, 1.0)
    h, s, v = _rgb_to_hsv(rgb)
    dh = np.zeros_like(h)
    ds = np.zeros_like(s)
    dv = np.zeros_like(v)
    w_sum = np.zeros_like(h)

    for color, center, width in _HSL_BANDS:
        hue_k = f"HueAdjustment{color}"
        sat_k = f"SaturationAdjustment{color}"
        lum_k = f"LuminanceAdjustment{color}"
        hv = float(adj.get(hue_k, 0.0))
        sv = float(adj.get(sat_k, 0.0))
        lv = float(adj.get(lum_k, 0.0))
        if abs(hv) < 1e-4 and abs(sv) < 1e-4 and abs(lv) < 1e-4:
            continue
        w = _band_weight(h, center, width)
        w_sum += w
        dh += w * (hv / 100.0) * 35.0
        ds += w * (sv / 100.0)
        dv += w * (lv / 100.0) * 0.35

    w_sum = np.maximum(w_sum, 1e-6)
    dh = dh / w_sum
    ds = ds / w_sum
    dv = dv / w_sum

    h = (h + dh) % 360.0
    s = np.clip(s * (1.0 + ds), 0.0, 1.0)
    v = np.clip(v + dv, 0.0, 1.0)
    return _hsv_to_rgb(h, s, v)
