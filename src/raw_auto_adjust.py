"""One-shot automatic adjustments: Auto white balance and Auto straighten.

Both are ESTIMATORS feeding the existing manual controls -- Auto WB produces a
neutral-gray sample for raw_adjustments.solve_white_balance_from_sample (the
same solver the WB dropper uses), Auto straighten produces a CropAngle for the
Transform slider. The user sees exactly which sliders moved and can refine or
undo with the normal controls; nothing here writes pixels or sidecars itself.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def estimate_neutral_rgb(linear_rgb: np.ndarray) -> Optional[tuple]:
    """Robust gray-world neutral estimate from a scene-linear RGB buffer.

    Plain gray-world (mean of everything) is famously wrecked by dominant
    subject colors. Two standard robustifiers:
      - midtone gate: only pixels between the 25th and 90th luminance
        percentile vote (shadows carry mostly noise, highlights clip);
      - saturation gate: only the least-saturated half of those vote --
        near-neutral surfaces are the pixels that SHOULD be gray, strongly
        colored ones say nothing about the illuminant.
    Returns (r, g, b) means of the voting pixels, or None if too few.
    """
    try:
        a = linear_rgb
        if a is None or a.ndim != 3 or a.shape[2] < 3:
            return None
        # Subsample for speed: the estimate is a global mean, ~100k pixels
        # is plenty.
        h, w = a.shape[:2]
        step = max(1, int((h * w / 150_000) ** 0.5))
        s = a[::step, ::step, :3].astype(np.float32)
        lum = 0.2126 * s[..., 0] + 0.7152 * s[..., 1] + 0.0722 * s[..., 2]
        lo, hi = np.percentile(lum, (25.0, 90.0))
        mid = (lum > max(lo, 1e-5)) & (lum < hi)
        if int(mid.sum()) < 500:
            return None
        px = s[mid]
        mx = px.max(axis=1)
        mn = px.min(axis=1)
        sat = (mx - mn) / np.maximum(mx, 1e-6)
        keep = sat <= np.percentile(sat, 50.0)
        px = px[keep]
        if px.shape[0] < 300:
            return None
        r, g, b = (float(px[:, i].mean()) for i in range(3))
        if min(r, g, b) <= 1e-6:
            return None
        return (r, g, b)
    except Exception as e:
        logger.debug("[AUTO_WB] estimate failed: %s", e)
        return None


def _to_uint8_gray(rgb: np.ndarray) -> Optional[np.ndarray]:
    """Accept display uint8, float[0,1], or scene-linear float → uint8 gray."""
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] < 3:
        return None
    a = rgb[..., :3]
    if a.dtype == np.uint8:
        g = a.astype(np.float32)
    else:
        a = a.astype(np.float32)
        # Scene-linear buffers are often >>1 in highlights; tone-map lightly.
        if float(np.percentile(a, 99)) > 1.5:
            a = a / (1.0 + a)
        g = np.clip(a, 0.0, 1.0) * 255.0
    gray = (
        0.2126 * g[..., 0] + 0.7152 * g[..., 1] + 0.0722 * g[..., 2]
    ).astype(np.uint8)
    return gray


def estimate_straighten_angle(display_rgb: np.ndarray) -> Optional[float]:
    """Dominant small tilt (degrees, for CropAngle) from near-axis lines.

    Pipeline:
      1. Canny + probabilistic Hough for long near-H/V segments
      2. Fallback: gradient-orientation histogram peak near the axes

    Each Hough segment within ±12° of horizontal or vertical votes
    (length-weighted) for the correction that would bring it true.
    Returns None when there is nothing lawful to align to.

    ``CropAngle`` is +CCW image rotation (see raw_transform._warped_quad).
    A line whose image-space angle is +dev (CCW from +X) needs CropAngle
    = -dev so the line becomes axis-aligned after the warp.
    """
    try:
        import cv2

        gray = _to_uint8_gray(display_rgb)
        if gray is None:
            return None
        h, w = gray.shape[:2]
        # Work at a bounded size for stable thresholds across megapixel RAWs.
        max_edge = 1400.0
        scale = min(1.0, max_edge / float(max(h, w)))
        if scale < 0.999:
            gray = cv2.resize(
                gray,
                (max(32, int(w * scale)), max(32, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        # Mild blur keeps Canny from latching onto sensor noise / fine foliage.
        gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2)

        edges = cv2.Canny(gray, 40, 140, apertureSize=3)
        min_side = min(gray.shape[:2])
        # Was 0.15 — too strict on many interiors / horizons; 0.06 catches
        # typical horizon + architectural lines without drowning in noise.
        min_len = max(24, int(0.06 * min_side))
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 360.0,
            threshold=max(30, min_len // 3),
            minLineLength=min_len,
            maxLineGap=max(6, min_len // 8),
        )

        votes: list[float] = []
        weights: list[float] = []
        tol = 12.0
        if lines is not None:
            # OpenCV may return (N,1,4) or (N,4) depending on build/version.
            for seg in lines:
                x1, y1, x2, y2 = (float(v) for v in np.asarray(seg).reshape(-1)[:4])
                length = float(np.hypot(x2 - x1, y2 - y1))
                if length < 1.0:
                    continue
                # Image coords: +Y down. arctan2(dy, dx) in degrees.
                ang = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                # Deviation from nearest axis, signed, in (-45, 45].
                folded = ((ang + 45.0) % 90.0) - 45.0
                if abs(folded) <= tol:
                    votes.append(folded)
                    weights.append(length)

        med: Optional[float] = None
        if len(votes) >= 2:
            v = np.asarray(votes, dtype=np.float64)
            wts = np.asarray(weights, dtype=np.float64)
            order = np.argsort(v)
            cw = np.cumsum(wts[order])
            med = float(v[order][np.searchsorted(cw, cw[-1] / 2.0)])
            spread = float(np.sqrt(np.average((v - med) ** 2, weights=wts)))
            # Was 3.0 — slightly more tolerant of mixed architecture + horizon.
            if spread > 4.5:
                med = None

        if med is None:
            med = _gradient_orientation_peak(gray, tol=tol)
        if med is None:
            return None
        if abs(med) < 0.08:
            return None  # already straight enough
        # CropAngle +CCW; line at +med needs opposite rotation.
        return float(np.clip(-med, -45.0, 45.0))
    except Exception as e:
        logger.debug("[AUTO_STRAIGHTEN] estimate failed: %s", e)
        return None


def _gradient_orientation_peak(gray_u8: np.ndarray, *, tol: float = 12.0) -> Optional[float]:
    """Fallback: Sobel orientation histogram peak near H/V axes."""
    try:
        import cv2

        gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        # Edge direction is perpendicular to gradient; line angle = atan2(gy,gx)+90
        # but we want the line's deviation the same way as Hough: use gradient
        # angle folded to nearest axis of the *edge* (gradient ±90°).
        ang = np.degrees(np.arctan2(gy, gx))  # gradient angle
        # Edge orientation = gradient + 90°, then fold to (-45, 45]
        edge = ang + 90.0
        folded = ((edge + 45.0) % 90.0) - 45.0
        strong = mag > max(12.0, float(np.percentile(mag, 90)))
        if int(strong.sum()) < 500:
            return None
        sel = folded[strong]
        # Keep near-axis only
        near = sel[np.abs(sel) <= tol]
        if near.size < 300:
            return None
        # Weighted by magnitude
        w = mag[strong][np.abs(sel) <= tol]
        order = np.argsort(near)
        cw = np.cumsum(w[order])
        med = float(near[order][np.searchsorted(cw, cw[-1] / 2.0)])
        if abs(med) < 0.08:
            return None
        return med
    except Exception:
        return None
