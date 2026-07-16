"""Lightroom-style HSL per-color adjustments in display-linear RGB.

OpenCV float32 ``COLOR_BGR2HSV`` uses H in [0, 360] and S/V in [0, 1]
(unlike uint8 HSV where H is [0, 179] and S/V are [0, 255]). Callers must
never rescale float HSV as if it were uint8 — that was Performance review #6.
"""

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
    """RGB float [0,1] → H [0,360], S/V [0,1] (OpenCV float32 HSV convention)."""
    import cv2

    bgr = np.ascontiguousarray(np.clip(rgb[..., ::-1], 0.0, 1.0), dtype=np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Defensive: some builds historically surprised callers with uint8-like
    # packing even on float input. Detect and convert once.
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    if float(np.nanmax(s)) > 1.5 or float(np.nanmax(v)) > 1.5:
        # Treat as uint8-scaled channels packed into float.
        h = h * (360.0 / 179.0) if float(np.nanmax(h)) <= 180.5 else h
        s = s / 255.0
        v = v / 255.0
    return h.astype(np.float32, copy=False), s.astype(np.float32, copy=False), v.astype(
        np.float32, copy=False
    )


def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """H [0,360], S/V [0,1] → RGB float [0,1]."""
    import cv2

    hsv = np.stack(
        [
            np.clip(h, 0.0, 360.0).astype(np.float32, copy=False),
            np.clip(s, 0.0, 1.0).astype(np.float32, copy=False),
            np.clip(v, 0.0, 1.0).astype(np.float32, copy=False),
        ],
        axis=-1,
    )
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
