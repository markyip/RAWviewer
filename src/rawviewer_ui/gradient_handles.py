"""On-canvas handles for parametric gradient masks.

Two halves that must agree exactly:

* ``GradientHandleItem`` draws the geometry in scene coordinates (== image
  pixels), with a cosmetic pen so line weight stays constant at any zoom.
* ``hit_test`` / ``apply_drag`` are plain functions over normalised params, so
  the interaction is testable without a scene, a viewport or a mouse.

Handle layout follows Lightroom and darktable, because a gradient's handles are
one of the few UI conventions photographers already have muscle memory for:

  linear  a dot at each end of the axis, plus the axis line. Dragging an end
          moves that end; dragging the line moves the whole gradient without
          changing its direction or spread.
  radial  a dot at the centre and one at each of the four ellipse extremes.
          Centre moves it; an extreme resizes that axis only.

Hit radius is in SCREEN pixels, converted by the caller, so a handle stays
grabbable when zoomed out -- a radius in image pixels becomes untouchable on a
40MP frame fitted to a window.
"""

from __future__ import annotations

import math
from typing import Optional

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
from PyQt6.QtWidgets import QGraphicsItem

# Handle names are returned by hit_test and consumed by apply_drag.
H_LINEAR_START = "linear_start"
H_LINEAR_END = "linear_end"
H_LINEAR_LINE = "linear_line"
H_RADIAL_CENTRE = "radial_centre"
H_RADIAL_LEFT = "radial_left"
H_RADIAL_RIGHT = "radial_right"
H_RADIAL_TOP = "radial_top"
H_RADIAL_BOTTOM = "radial_bottom"

_DOT_R = 6.0  # screen px
_MIN_RADIUS = 1e-3


def handle_points(kind: str, params: dict, width: int, height: int) -> dict:
    """Handle name -> (x, y) in image pixels."""
    p = params or {}
    w = max(1, int(width) - 1)
    h = max(1, int(height) - 1)
    if kind == "linear":
        return {
            H_LINEAR_START: (float(p.get("x0", 0.5)) * w, float(p.get("y0", 0.0)) * h),
            H_LINEAR_END: (float(p.get("x1", 0.5)) * w, float(p.get("y1", 0.45)) * h),
        }
    if kind == "radial":
        cx = float(p.get("cx", 0.5)) * w
        cy = float(p.get("cy", 0.5)) * h
        rx = max(_MIN_RADIUS, float(p.get("rx", 0.3))) * w
        ry = max(_MIN_RADIUS, float(p.get("ry", 0.3))) * h
        rot = math.radians(float(p.get("rotation", 0.0)))
        cos_r, sin_r = math.cos(rot), math.sin(rot)

        def rotated(dx, dy):
            return (cx + dx * cos_r - dy * sin_r, cy + dx * sin_r + dy * cos_r)

        return {
            H_RADIAL_CENTRE: (cx, cy),
            H_RADIAL_LEFT: rotated(-rx, 0.0),
            H_RADIAL_RIGHT: rotated(rx, 0.0),
            H_RADIAL_TOP: rotated(0.0, -ry),
            H_RADIAL_BOTTOM: rotated(0.0, ry),
        }
    return {}


def _distance_to_segment(px, py, ax, ay, bx, by) -> float:
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def hit_test(
    kind: str,
    params: dict,
    x: float,
    y: float,
    width: int,
    height: int,
    *,
    tolerance: float,
) -> Optional[str]:
    """Which handle is under image-pixel (x, y), or None.

    Dots are tested before the line so grabbing an endpoint that sits on the
    axis moves that end rather than sliding the whole gradient.
    """
    points = handle_points(kind, params, width, height)
    best = None
    best_dist = tolerance
    for name, (hx, hy) in points.items():
        d = math.hypot(x - hx, y - hy)
        if d <= best_dist:
            best, best_dist = name, d
    if best is not None:
        return best

    if kind == "linear":
        (ax, ay) = points[H_LINEAR_START]
        (bx, by) = points[H_LINEAR_END]
        if _distance_to_segment(x, y, ax, ay, bx, by) <= tolerance:
            return H_LINEAR_LINE
    return None


def apply_drag(
    kind: str,
    params: dict,
    handle: str,
    x: float,
    y: float,
    width: int,
    height: int,
    *,
    grab: Optional[tuple] = None,
) -> dict:
    """Params updated for a drag of ``handle`` to image-pixel (x, y).

    ``grab`` is (params_at_press, press_x, press_y) and is only needed by the
    whole-gradient move, which must translate from where the gesture started --
    without it the gradient jumps so its centre snaps under the cursor on the
    first move.
    """
    p = dict(params or {})
    w = max(1, int(width) - 1)
    h = max(1, int(height) - 1)
    nx = max(0.0, min(1.0, float(x) / w))
    ny = max(0.0, min(1.0, float(y) / h))

    if handle == H_LINEAR_START:
        p["x0"], p["y0"] = nx, ny
    elif handle == H_LINEAR_END:
        p["x1"], p["y1"] = nx, ny
    elif handle == H_LINEAR_LINE and grab is not None:
        start, gx, gy = grab
        dx = (float(x) - float(gx)) / w
        dy = (float(y) - float(gy)) / h
        for key, delta in (("x0", dx), ("x1", dx), ("y0", dy), ("y1", dy)):
            p[key] = max(0.0, min(1.0, float(start.get(key, 0.0)) + delta))
    elif handle == H_RADIAL_CENTRE:
        p["cx"], p["cy"] = nx, ny
    elif handle in (H_RADIAL_LEFT, H_RADIAL_RIGHT):
        cx = float(p.get("cx", 0.5))
        p["rx"] = max(_MIN_RADIUS, abs(nx - cx))
    elif handle in (H_RADIAL_TOP, H_RADIAL_BOTTOM):
        cy = float(p.get("cy", 0.5))
        p["ry"] = max(_MIN_RADIUS, abs(ny - cy))
    return p


class GradientHandleItem(QGraphicsItem):
    """Draws a gradient's axis/ellipse and its handles in image coordinates."""

    def __init__(self) -> None:
        super().__init__()
        self._kind = ""
        self._params: dict = {}
        self._w = 0
        self._h = 0
        self.setZValue(26)  # above the brush cursor (25) so handles stay visible
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, False)
        self.hide()

    def set_geometry(self, kind: str, params: dict, width: int, height: int) -> None:
        self._kind = kind or ""
        self._params = dict(params or {})
        self._w = int(width or 0)
        self._h = int(height or 0)
        self.prepareGeometryChange()
        self.update()
        self.setVisible(bool(self._kind) and self._w > 0 and self._h > 0)

    def clear(self) -> None:
        self._kind = ""
        self._params = {}
        self.hide()

    def boundingRect(self) -> QRectF:
        if self._w <= 0 or self._h <= 0:
            return QRectF()
        # The whole frame: a linear axis can be dragged anywhere in it, and an
        # exact rect would need recomputing on every param change for no gain.
        return QRectF(0.0, 0.0, float(self._w), float(self._h))

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: ARG002
        if not self._kind or self._w <= 0 or self._h <= 0:
            return
        points = handle_points(self._kind, self._params, self._w, self._h)
        if not points:
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Cosmetic pens: constant on-screen weight at any zoom. Drawn twice,
        # dark under light, so the geometry stays readable over both a blown
        # sky and a black shadow.
        under = QPen(QColor(0, 0, 0, 160), 3.0)
        under.setCosmetic(True)
        over = QPen(QColor(255, 255, 255, 230), 1.4)
        over.setCosmetic(True)

        if self._kind == "linear":
            a = QPointF(*points[H_LINEAR_START])
            b = QPointF(*points[H_LINEAR_END])
            for pen in (under, over):
                painter.setPen(pen)
                painter.drawLine(a, b)
            # Ticks along the axis ends mark the extent of the fade, the way
            # Lightroom's three graduated-filter lines do.
            dx, dy = b.x() - a.x(), b.y() - a.y()
            length = math.hypot(dx, dy)
            if length > 1e-6:
                nx, ny = -dy / length, dx / length
                tick = max(self._w, self._h) * 0.05
                for point in (a, b):
                    p1 = QPointF(point.x() - nx * tick, point.y() - ny * tick)
                    p2 = QPointF(point.x() + nx * tick, point.y() + ny * tick)
                    for pen in (under, over):
                        painter.setPen(pen)
                        painter.drawLine(p1, p2)
        elif self._kind == "radial":
            cx, cy = points[H_RADIAL_CENTRE]
            rx = abs(points[H_RADIAL_RIGHT][0] - cx) or 1.0
            ry = abs(points[H_RADIAL_BOTTOM][1] - cy) or 1.0
            rect = QRectF(cx - rx, cy - ry, rx * 2.0, ry * 2.0)
            for pen in (under, over):
                painter.setPen(pen)
                painter.drawEllipse(rect)

        # Handle dots, sized in screen pixels via the inverse view scale.
        scale = 1.0
        if painter.worldTransform().m11():
            scale = abs(painter.worldTransform().m11())
        r = _DOT_R / max(1e-6, scale)
        painter.setPen(QPen(QColor(0, 0, 0, 190), 1.2))
        painter.setBrush(QBrush(QColor(255, 255, 255, 235)))
        for name, (hx, hy) in points.items():
            if name == H_LINEAR_LINE:
                continue
            painter.drawEllipse(QRectF(hx - r, hy - r, r * 2.0, r * 2.0))
