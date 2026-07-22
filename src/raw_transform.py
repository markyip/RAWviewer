"""Slider-driven geometry stage: straighten, crop insets, keystone perspective.

Applied ONCE at the head of both edit pipelines (scene-linear and the legacy
gamma path), before any tone/color work, so dodge/burn strokes and every
downstream stage operate in the transformed frame.

Hard constraint (by design, per user requirement): the output never contains
pixels that were not in the source frame. Rotation/keystone warp within the
original canvas, then the result is cropped to the largest centered
original-aspect rectangle that lies fully inside the transformed image quad
(binary-search inscribed rect), so no border fill, no canvas expansion, no
inpainting -- output dimensions are always <= the input's.

Slider model (matches Lightroom's Transform panel, which is also
slider-based -- no corner-drag UI needed):
  CropAngle              degrees, +CCW, straighten
  PerspectiveVertical    +100 tilts the top away (converging verticals fix)
  PerspectiveHorizontal  +100 tilts the right side away
  CropLeft/Right/Top/Bottom  additional user crop, inset fraction per edge
"""

from __future__ import annotations

import numpy as np

TRANSFORM_KEYS = (
    "CropAngle",
    "PerspectiveVertical",
    "PerspectiveHorizontal",
    "CropLeft",
    "CropRight",
    "CropTop",
    "CropBottom",
    "AnamorphicRatio",
)

# Keystone strength at slider 100: the far edge shrinks by this fraction.
_KEYSTONE_MAX = 0.35


def has_geometry(adj: dict | None) -> bool:
    if not adj:
        return False
    try:
        if any(abs(float(adj.get(k, 0.0) or 0.0)) > 1e-4 for k in (
            "CropAngle",
            "PerspectiveVertical",
            "PerspectiveHorizontal",
            "CropLeft",
            "CropRight",
            "CropTop",
            "CropBottom",
        )):
            return True
        ratio = float(adj.get("AnamorphicRatio", 1.0) or 1.0)
        return abs(ratio - 1.0) > 1e-4
    except Exception:
        return False


def _warped_quad(w: float, h: float, angle_deg: float, pv: float, ph: float) -> np.ndarray:
    """Destination positions of the 4 source corners (TL, TR, BR, BL)."""
    corners = np.array(
        [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=np.float64
    )
    cx, cy = w / 2.0, h / 2.0
    # Rotation about the center.
    a = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = (corners - (cx, cy)) @ rot.T + (cx, cy)
    # Keystone: shrink the far edge toward the center line. pv > 0 pulls the
    # TOP corners inward horizontally (verticals converge less after warp);
    # ph > 0 pulls the RIGHT corners inward vertically.
    kv = pv * _KEYSTONE_MAX
    kh = ph * _KEYSTONE_MAX
    for i in range(4):
        x, y = pts[i]
        fy = 1.0 - (y / h)  # 1 at top, 0 at bottom
        fx = x / w          # 0 at left, 1 at right
        if kv >= 0:
            pts[i, 0] = cx + (x - cx) * (1.0 - kv * fy)
        else:
            pts[i, 0] = cx + (x - cx) * (1.0 + kv * (1.0 - fy))
        if kh >= 0:
            pts[i, 1] = cy + (y - cy) * (1.0 - kh * fx)
        else:
            pts[i, 1] = cy + (y - cy) * (1.0 + kh * (1.0 - fx))
    return pts.astype(np.float32)


def _point_in_quad(px: np.ndarray, quad: np.ndarray) -> bool:
    """All points inside the convex quad (winding cross-product test)."""
    for i in range(4):
        a = quad[i]
        b = quad[(i + 1) % 4]
        cross = (b[0] - a[0]) * (px[:, 1] - a[1]) - (b[1] - a[1]) * (px[:, 0] - a[0])
        # TL->TR->BR->BL is clockwise in y-DOWN image coordinates, which makes
        # the interior the POSITIVE-cross side of every edge.
        if np.any(cross < -1e-6):
            return False
    return True


def _inscribed_rect(
    w: int, h: int, quad: np.ndarray, *, iterations: int = 24
) -> tuple[int, int, int, int]:
    """Largest centered rect of the ORIGINAL aspect fully inside ``quad``.

    Binary search on scale -- ~24 iterations bound the error under a pixel at
    any realistic size, and the centered-original-aspect constraint is what
    guarantees "no pixels invented": every point of the returned rect maps
    from real source content. Live preview may pass fewer iterations.
    """
    cx, cy = w / 2.0, h / 2.0
    lo, hi = 0.0, 1.0
    n = max(8, min(24, int(iterations)))
    for _ in range(n):
        mid = (lo + hi) / 2.0
        hw, hh = cx * mid, cy * mid
        pts = np.array(
            [
                [cx - hw, cy - hh],
                [cx + hw, cy - hh],
                [cx + hw, cy + hh],
                [cx - hw, cy + hh],
            ]
        )
        if _point_in_quad(pts, quad):
            lo = mid
        else:
            hi = mid
    hw, hh = cx * lo, cy * lo
    x0, y0 = int(np.ceil(cx - hw)), int(np.ceil(cy - hh))
    x1, y1 = int(np.floor(cx + hw)), int(np.floor(cy + hh))
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def apply_geometry(
    img: np.ndarray,
    adj: dict | None,
    *,
    preview: bool = False,
) -> np.ndarray:
    """Straighten + keystone + crop a (H, W, 3) buffer; dtype preserved.

    No-op (identity, same object) when no transform keys are set. Works on
    uint8 display buffers and float linear edit bases alike.

    ``preview=True`` uses a cheaper warp (nearest) and a shorter inscribed-rect
    search — for live Transform-slider ticks only. Export / settle keep the
    default high-quality path.
    """
    if not has_geometry(adj):
        return img
    try:
        import cv2

        h, w = img.shape[:2]
        angle = float(adj.get("CropAngle", 0.0) or 0.0)
        pv = float(adj.get("PerspectiveVertical", 0.0) or 0.0) / 100.0
        ph = float(adj.get("PerspectiveHorizontal", 0.0) or 0.0) / 100.0
        pv = max(-1.0, min(1.0, pv))
        ph = max(-1.0, min(1.0, ph))

        out = img
        if abs(angle) > 1e-4 or abs(pv) > 1e-4 or abs(ph) > 1e-4:
            src = np.array(
                [[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32
            )
            dst = _warped_quad(w, h, angle, pv, ph)
            m = cv2.getPerspectiveTransform(src, dst)
            # INTER_NEAREST is ~2–4× faster than LINEAR on large float buffers
            # and is fine for 640px live-drag / instant overlay feedback.
            flags = cv2.INTER_NEAREST if preview else cv2.INTER_LINEAR
            out = cv2.warpPerspective(
                img, m, (w, h), flags=flags,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            iters = 12 if preview else 24
            x0, y0, x1, y1 = _inscribed_rect(w, h, dst, iterations=iters)
            out = out[y0:y1, x0:x1]

        # User crop insets, applied within whatever survived the warp crop.
        ch, cw = out.shape[:2]
        li = max(0.0, min(0.45, float(adj.get("CropLeft", 0.0) or 0.0)))
        ri = max(0.0, min(0.45, float(adj.get("CropRight", 0.0) or 0.0)))
        ti = max(0.0, min(0.45, float(adj.get("CropTop", 0.0) or 0.0)))
        bi = max(0.0, min(0.45, float(adj.get("CropBottom", 0.0) or 0.0)))
        if li + ri + ti + bi > 1e-4:
            x0 = int(cw * li)
            x1 = cw - int(cw * ri)
            y0 = int(ch * ti)
            y1 = ch - int(ch * bi)
            if x1 - x0 >= 8 and y1 - y0 >= 8:
                out = out[y0:y1, x0:x1]

        # Anamorphic lens desqueeze: stretch width by ratio (1.33x, 1.5x, 1.6x, 2.0x).
        ratio = float(adj.get("AnamorphicRatio", 1.0) or 1.0)
        if abs(ratio - 1.0) > 1e-4 and ratio > 0.1:
            ch, cw = out.shape[:2]
            new_w = max(1, int(round(cw * ratio)))
            interp = cv2.INTER_NEAREST if preview else (cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_CUBIC)
            out = cv2.resize(out, (new_w, ch), interpolation=interp)

        return np.ascontiguousarray(out)
    except Exception:
        return img


def map_display_point_to_buffer(
    x: float,
    y: float,
    *,
    display_w: int,
    display_h: int,
    buffer_w: int,
    buffer_h: int,
) -> tuple[float, float]:
    """Scale a display-pixmap point into a same-framing buffer (D&B / WB pick).

    Adjust live-drag may show a 640px lite tier while the sample/mask buffer
    is half-res or post-geometry; both share the same crop/framing, so a
    uniform scale is the correct mapping (same approach as dodge/burn strokes).
    """
    sx = float(buffer_w) / float(max(1, int(display_w)))
    sy = float(buffer_h) / float(max(1, int(display_h)))
    return float(x) * sx, float(y) * sy
