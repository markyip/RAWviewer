"""Post-crop vignette and dehaze (display-linear RGB float [0, 1]).

Both are global Look/Effects-style controls — cheap enough for live preview
when amount is non-zero, true no-ops when at default. Applied after tone-map
and before HSL/detail so they sit with other perceptual display adjustments.
"""

from __future__ import annotations

import numpy as np

VIGNETTE_KEY = "PostCropVignetteAmount"
VIGNETTE_MIDPOINT_KEY = "PostCropVignetteMidpoint"
DEHAZE_KEY = "Dehaze"
EFFECT_KEYS = (VIGNETTE_KEY, VIGNETTE_MIDPOINT_KEY, DEHAZE_KEY)
VIGNETTE_MIDPOINT_DEFAULT = 50.0


def apply_vignette(
    img: np.ndarray,
    amount: float,
    midpoint: float = VIGNETTE_MIDPOINT_DEFAULT,
    *,
    y_range: tuple[int, int] | None = None,
    total_h: int | None = None,
) -> np.ndarray:
    """Post-crop vignette matching Lightroom Amount / Midpoint polarity.

    ``amount`` in [-100, 100] (``crs:PostCropVignetteAmount``):
      negative → darken corners, positive → lighten corners.

    ``midpoint`` in [0, 100] (``crs:PostCropVignetteMidpoint``, default 50):
      lower → vignette reaches further toward the center (larger);
      higher → effect stays near the corners (smaller).

    Paint Overlay–style blend (lerp toward black / white by a radial mask).
    """
    a = float(amount)
    if abs(a) < 1e-3 or img is None or img.ndim != 3:
        return img
    h, w = img.shape[:2]
    if h < 1 or w < 2:
        return img

    mid = float(np.clip(midpoint, 0.0, 100.0))
    full_h = total_h if (total_h is not None and total_h >= h) else h
    y_start = y_range[0] if (y_range is not None and len(y_range) == 2) else 0
    y_end = y_start + h

    # Radius normalized so image corners are 1.0 (edge midpoints ≈ 0.71).
    yy, xx = np.ogrid[y_start:y_end, 0:w]
    cy = (full_h - 1) * 0.5
    cx = (w - 1) * 0.5
    corner_r = max(np.hypot(cx, cy), 1.0)
    r = (np.hypot(xx - cx, yy - cy) / corner_r).astype(np.float32)

    # Midpoint → onset radius. Pack high values tightly toward the corners so
    # Midpoint=100 is a small corner-only vignette (LR-like), not a wide rim.
    #   0 → 0.00,  50 → ~0.55,  100 → 0.93
    u = mid / 100.0
    inner = (0.35 * u + 0.65 * (u * u)) * 0.93
    span = max(0.05, 1.0 - inner)
    t = np.clip((r - inner) / span, 0.0, 1.0)
    mask = (t * t * (3.0 - 2.0 * t)).astype(np.float32)  # smoothstep
    strength = abs(a) / 100.0
    wgt = (strength * mask)[..., np.newaxis]
    src = np.clip(img, 0.0, None).astype(np.float32, copy=False)
    if a < 0:
        out = src * (1.0 - wgt)
    else:
        out = src + (1.0 - np.clip(src, 0.0, 1.0)) * wgt
    return np.clip(out, 0.0, None).astype(np.float32)


def apply_dehaze(img: np.ndarray, amount: float, *, preview: bool = False) -> np.ndarray:
    """Lite dark-channel dehaze / haze. ``amount`` in [-100, 100].

    Positive = remove haze (Dark Channel Prior–inspired, downsampled guide).
    Negative = add a soft atmospheric wash. Keeps Mac/Win parity with cv2+numpy
    only — no ML weights.
    """
    a = float(amount)
    if abs(a) < 1e-3 or img is None or img.ndim != 3:
        return img
    try:
        import cv2
    except Exception:
        return img

    src = np.clip(img, 0.0, None).astype(np.float32)
    h, w = src.shape[:2]
    # Work on a guide at most ~640 long edge for speed; upsample transmission.
    max_edge = 480 if preview else 720
    scale = min(1.0, float(max_edge) / float(max(h, w)))
    if scale < 0.999:
        small = cv2.resize(
            src,
            (max(8, int(w * scale)), max(8, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = src

    # Dark channel (min over RGB, then min-filter)
    dark = small.min(axis=2)
    k = 5 if preview else 7
    dark = cv2.erode(dark, np.ones((k, k), np.uint8))

    # Atmospheric light: mean of brightest dark-channel pixels
    flat = dark.reshape(-1)
    n_pick = max(1, int(flat.size * 0.001))
    idx = np.argpartition(flat, -n_pick)[-n_pick:]
    ys, xs = np.unravel_index(idx, dark.shape)
    A = small[ys, xs].mean(axis=0)
    A = np.clip(A, 0.08, 1.0).astype(np.float32)

    omega = 0.95
    t = 1.0 - omega * (dark / np.maximum(A.max(), 1e-3))
    t = np.clip(t, 0.12, 1.0).astype(np.float32)
    # Soften transmission
    t = cv2.GaussianBlur(t, (0, 0), sigmaX=2.0 if preview else 3.0)

    if scale < 0.999:
        t = cv2.resize(t, (w, h), interpolation=cv2.INTER_LINEAR)

    strength = abs(a) / 100.0
    if a > 0:
        # Dehaze: J = (I - A)/t + A, blended by strength
        t3 = np.maximum(t, 0.12)[..., np.newaxis]
        cleared = (src - A) / t3 + A
        out = src * (1.0 - strength) + cleared * strength
    else:
        # Add haze: blend toward atmospheric light with inverted transmission
        haze_mix = strength * (1.0 - t)[..., np.newaxis] * 0.85
        out = src * (1.0 - haze_mix) + A * haze_mix

    # Mild midtone contrast restore on positive dehaze (avoids flat look)
    if a > 0:
        luma = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]
        mid = np.clip((luma - 0.15) / 0.70, 0.0, 1.0)
        boost = 1.0 + 0.12 * strength * mid
        out = out * boost[..., np.newaxis]

    return np.clip(out, 0.0, None).astype(np.float32)
