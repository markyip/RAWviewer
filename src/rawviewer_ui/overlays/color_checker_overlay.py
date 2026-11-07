"""Interactive pin overlay for a 24-patch ColorChecker chart."""

from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QCursor, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QWidget

try:
    import rawviewer_ui.theme as theme
except ModuleNotFoundError:
    from core import theme as theme


class ColorCheckerOverlay(QWidget):
    """User-draggable quad that maps chart corners onto a preview photo."""

    corners_changed = pyqtSignal(list)
    calibration_confirmed = pyqtSignal(list)

    _HANDLE = 8.0
    _SEED = ((0.22, 0.28), (0.78, 0.28), (0.78, 0.72), (0.22, 0.72))

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.corners = [QPointF(x, y) for x, y in self._SEED]
        self._drag_index = -1
        self._hover_index = -1

    def reset_to_seed(self) -> None:
        self.corners = [QPointF(x, y) for x, y in self._SEED]
        self.update()

    def set_normalized_corners(self, corners: list[tuple[float, float]]) -> None:
        if len(corners) != 4:
            return
        self.corners = [QPointF(x, y) for x, y in corners]
        self.update()

    def get_pixel_corners(self, width: int, height: int) -> list[tuple[float, float]]:
        return [(p.x() * width, p.y() * height) for p in self.corners]

    def _nearest_handle(self, pos, w: int, h: int) -> int:
        limit = self._HANDLE * 1.6
        for i, p in enumerate(self.corners):
            if math.hypot(pos.x() - p.x() * w, pos.y() - p.y() * h) <= limit:
                return i
        return -1

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            idx = self._nearest_handle(event.position(), self.width(), self.height())
            if idx >= 0:
                self._drag_index = idx
                self.update()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        w, h = max(1, self.width()), max(1, self.height())
        pos = event.position()
        if self._drag_index >= 0:
            nx = max(0.0, min(1.0, pos.x() / w))
            ny = max(0.0, min(1.0, pos.y() / h))
            self.corners[self._drag_index] = QPointF(nx, ny)
            self.corners_changed.emit([(p.x(), p.y()) for p in self.corners])
            self.update()
            event.accept()
            return
        nxt = self._nearest_handle(pos, w, h)
        if nxt != self._hover_index:
            self._hover_index = nxt
            shape = (
                Qt.CursorShape.PointingHandCursor
                if nxt >= 0
                else Qt.CursorShape.ArrowCursor
            )
            self.setCursor(QCursor(shape))
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._drag_index >= 0:
            self._drag_index = -1
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        poly = QPolygonF([QPointF(p.x() * w, p.y() * h) for p in self.corners])
        qp.setPen(QPen(QColor(theme.EMBER), 2))
        qp.setBrush(QBrush(QColor(245, 140, 66, 32)))
        qp.drawPolygon(poly)
        for i, pt in enumerate(poly):
            lit = i in (self._drag_index, self._hover_index)
            rad = self._HANDLE * (1.25 if lit else 1.0)
            qp.setPen(QPen(QColor("#FFFFFF"), 2))
            fill = theme.EMBER if lit else theme.EMBER_DIM
            qp.setBrush(QBrush(QColor(fill)))
            qp.drawEllipse(pt, rad, rad)
        qp.end()
