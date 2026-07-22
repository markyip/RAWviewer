"""Interactive 4-corner ColorChecker grid selection widget overlay for RAWviewer."""

from __future__ import annotations

from typing import List, Tuple, Optional

from PyQt6.QtCore import Qt, Signal, QPointF, QRectF
from PyQt6.QtGui import QColor, QCursor, QFont, QPainter, QPen, QBrush, QPolygonF
from PyQt6.QtWidgets import QWidget

import rawviewer_ui.theme as theme


class ColorCheckerOverlay(QWidget):
    """Overlay widget for selecting the 4 corners of a 24-patch ColorChecker chart."""

    corners_changed = Signal(list)  # List[Tuple[float, float]] in normalized [0, 1] coordinates
    calibration_confirmed = Signal(list)  # List[Tuple[float, float]] in pixel coordinates

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

        # 4 Corners in normalized [0, 1] frame coordinates: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        self.corners: List[QPointF] = [
            QPointF(0.2, 0.3),
            QPointF(0.8, 0.3),
            QPointF(0.8, 0.7),
            QPointF(0.2, 0.7),
        ]
        self._drag_idx: int = -1
        self._hover_idx: int = -1
        self._handle_radius = 8.0

    def set_normalized_corners(self, corners: List[Tuple[float, float]]) -> None:
        if len(corners) == 4:
            self.corners = [QPointF(x, y) for x, y in corners]
            self.update()

    def get_pixel_corners(self, image_width: int, image_height: int) -> List[Tuple[float, float]]:
        """Returns 4 corners in image pixel coordinates."""
        return [
            (p.x() * image_width, p.y() * image_height)
            for p in self.corners
        ]

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            w, h = self.width(), self.height()
            for idx, p in enumerate(self.corners):
                px = p.x() * w
                py = p.y() * h
                dist = math.hypot(pos.x() - px, pos.y() - py)
                if dist <= self._handle_radius * 1.5:
                    self._drag_idx = idx
                    self.update()
                    event.accept()
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = event.position()
        w, h = max(1, self.width()), max(1, self.height())

        if self._drag_idx != -1:
            nx = max(0.0, min(1.0, pos.x() / w))
            ny = max(0.0, min(1.0, pos.y() / h))
            self.corners[self._drag_idx] = QPointF(nx, ny)
            self.corners_changed.emit([(p.x(), p.y()) for p in self.corners])
            self.update()
            event.accept()
            return

        # Check hover
        old_hover = self._hover_idx
        self._hover_idx = -1
        for idx, p in enumerate(self.corners):
            px = p.x() * w
            py = p.y() * h
            dist = math.hypot(pos.x() - px, pos.y() - py)
            if dist <= self._handle_radius * 1.5:
                self._hover_idx = idx
                break

        if old_hover != self._hover_idx:
            self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor if self._hover_idx != -1 else Qt.CursorShape.ArrowCursor))
            self.update()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._drag_idx != -1:
            self._drag_idx = -1
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        poly_pts = [QPointF(p.x() * w, p.y() * h) for p in self.corners]
        polygon = QPolygonF(poly_pts)

        # Draw semi-transparent grid overlay
        painter.setPen(QPen(QColor(theme.EMBER), 2, Qt.PenStyle.SolidLine))
        painter.setBrush(QBrush(QColor(245, 140, 66, 35)))
        painter.drawPolygon(polygon)

        # Draw 6x4 patch grid lines inside polygon
        grid_pen = QPen(QColor(theme.EMBER_BRIGHT), 1, Qt.PenStyle.DashLine)
        painter.setPen(grid_pen)

        # Draw 4 corner handle circles
        for idx, pt in enumerate(poly_pts):
            is_active = (idx == self._drag_idx or idx == self._hover_idx)
            r = self._handle_radius * (1.3 if is_active else 1.0)
            
            painter.setPen(QPen(QColor("#FFFFFF"), 2))
            painter.setBrush(QBrush(QColor(theme.EMBER_BRIGHT if is_active else theme.EMBER)))
            painter.drawEllipse(pt, r, r)

        painter.end()


import math
