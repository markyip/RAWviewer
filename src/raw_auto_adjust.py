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


def estimate_straighten_angle(display_rgb: np.ndarray) -> Optional[float]:
    """Dominant small tilt (degrees, for CropAngle) from near-axis lines.

    Canny + probabilistic Hough; every detected segment within +-8 degrees of
    horizontal or vertical votes (length-weighted) for the correction that
    would bring it true. Median vote wins -- robust to a minority of
    genuinely diagonal content. Returns None when there is nothing lawful to
    align to (no segments, or the votes disagree wildly, e.g. organic
    scenes), so the caller can say "no dominant line found" instead of
    applying a garbage rotation.
    """
    try:
        import cv2

        a = display_rgb
        if a is None or a.ndim != 3:
            return None
        h, w = a.shape[:2]
        scale = 1200.0 / max(h, w)
        if scale < 1.0:
            a = cv2.resize(a, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 60, 160, apertureSize=3)
        min_len = int(0.15 * min(a.shape[:2]))
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 360.0, threshold=60,
            minLineLength=min_len, maxLineGap=8,
        )
        if lines is None or len(lines) == 0:
            return None
        votes = []
        weights = []
        tol = 8.0
        for x1, y1, x2, y2 in lines[:, 0]:
            dx, dy = float(x2 - x1), float(y2 - y1)
            length = float(np.hypot(dx, dy))
            if length < 1.0:
                continue
            ang = float(np.degrees(np.arctan2(dy, dx)))  # -180..180
            # Fold to deviation from the nearest axis (horizontal/vertical).
            dev_h = ((ang + 90.0) % 180.0) - 90.0        # deviation from horizontal
            dev_v = ((ang % 180.0) - 90.0)               # deviation from vertical
            dev = dev_h if abs(dev_h) <= abs(dev_v) else dev_v
            if abs(dev) <= tol:
                votes.append(dev)
                weights.append(length)
        if len(votes) < 3:
            return None
        votes = np.asarray(votes)
        weights = np.asarray(weights)
        order = np.argsort(votes)
        cw = np.cumsum(weights[order])
        med = float(votes[order][np.searchsorted(cw, cw[-1] / 2.0)])
        # Agreement check: if the length-weighted spread around the median is
        # wide, there is no single dominant tilt -- refuse.
        spread = float(
            np.sqrt(np.average((votes - med) ** 2, weights=weights))
        )
        if spread > 3.0 or abs(med) < 0.05:
            return None
        # CropAngle rotates the IMAGE by +deg CCW; a line tilted +dev needs
        # the opposite rotation to come true.
        return float(np.clip(-med, -45.0, 45.0))
    except Exception as e:
        logger.debug("[AUTO_STRAIGHTEN] estimate failed: %s", e)
        return None
