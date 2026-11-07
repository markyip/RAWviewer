"""ResizeGripIndicator: bottom-right resize hint for frameless windows."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QWidget

from core import theme


class ResizeGripIndicator(QWidget):
    """Bottom-right resize hint for frameless windows (overlay, not app chrome)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(16, 16)
        self._hint_visible = False
        self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.hide()

    def set_hint_visible(self, visible: bool) -> None:
        visible = bool(visible)
        if self._hint_visible == visible:
            return
        self._hint_visible = visible
        self.update()

    def _viewer(self):
        return self.parent()

    def mousePressEvent(self, event):
        viewer = self._viewer()
        if (
            viewer is not None
            and hasattr(viewer, "_begin_resize_at_edge")
            and viewer._begin_resize_at_edge(event, "bottom_right")
        ):
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        viewer = self._viewer()
        if (
            viewer is not None
            and hasattr(viewer, "_continue_resize_drag")
            and viewer._continue_resize_drag(event)
        ):
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        viewer = self._viewer()
        if viewer is not None and hasattr(viewer, "_finish_resize_drag") and viewer._finish_resize_drag():
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def enterEvent(self, event):
        self.set_hint_visible(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.set_hint_visible(False)
        super().leaveEvent(event)

    def paintEvent(self, event):
        if not self._hint_visible:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(*theme.INK_MUTED_RGB, 100))
        spacing = 3
        dot = 2
        origin_x = self.width() - dot
        origin_y = self.height() - dot
        for row in range(3):
            count = 3 - row
            for col in range(count):
                x = origin_x - (count - 1 - col) * spacing
                y = origin_y - row * spacing
                painter.drawEllipse(x, y, dot, dot)
        painter.end()
