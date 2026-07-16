"""Interactive crop rectangle overlay for the GPU single-image view.

Scene coordinates == image pixels. Insets are fractions of width/height
(CropLeft/Right/Top/Bottom), matching raw_transform.apply_geometry.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QPainter, QPen
from PyQt6.QtWidgets import QGraphicsObject

import theme

# Handle hit size in *view* pixels (cosmetic).
_HANDLE_VIEW_PX = 10.0
_MIN_FRAC = 0.02  # keep at least ~96% of each dimension usable
_MAX_INSET = 0.45


class CropOverlayItem(QGraphicsObject):
    """Dim outside the crop rect; drag edges/corners/body to adjust insets."""

    insetsChanged = pyqtSignal(float, float, float, float)  # L, R, T, B
    editingFinished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setZValue(20)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setAcceptHoverEvents(True)
        self._img_w = 0
        self._img_h = 0
        self._left = 0.0
        self._right = 0.0
        self._top = 0.0
        self._bottom = 0.0
        self._aspect: Optional[float] = None  # width/height, None = free
        self._drag_mode: Optional[str] = None
        self._drag_origin = QPointF()
        self._drag_start_insets = (0.0, 0.0, 0.0, 0.0)
        self.hide()

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, max(1, self._img_w), max(1, self._img_h))

    def set_image_size(self, width: int, height: int) -> None:
        self.prepareGeometryChange()
        self._img_w = max(0, int(width))
        self._img_h = max(0, int(height))
        self.update()

    def set_insets(self, left: float, right: float, top: float, bottom: float) -> None:
        self._left = float(np_clip(left, 0.0, _MAX_INSET))
        self._right = float(np_clip(right, 0.0, _MAX_INSET))
        self._top = float(np_clip(top, 0.0, _MAX_INSET))
        self._bottom = float(np_clip(bottom, 0.0, _MAX_INSET))
        self._normalize_min_size()
        self.update()

    def insets(self) -> tuple[float, float, float, float]:
        return (self._left, self._right, self._top, self._bottom)

    def set_aspect_ratio(self, aspect: Optional[float]) -> None:
        """``aspect`` = width/height, or None for free crop."""
        self._aspect = float(aspect) if aspect and aspect > 0 else None
        if self._aspect is not None:
            self._apply_aspect_from_center()
        self.update()

    def crop_rect(self) -> QRectF:
        w, h = float(self._img_w), float(self._img_h)
        return QRectF(
            w * self._left,
            h * self._top,
            w * (1.0 - self._left - self._right),
            h * (1.0 - self._top - self._bottom),
        )

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: ARG002
        if self._img_w <= 0 or self._img_h <= 0:
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        full = QRectF(0, 0, self._img_w, self._img_h)
        crop = self.crop_rect()
        # Dim outside the crop (four rects — avoid even-odd fill quirks).
        dim = QColor(0, 0, 0, 140)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(dim)
        painter.drawRect(QRectF(full.left(), full.top(), full.width(), crop.top() - full.top()))
        painter.drawRect(
            QRectF(full.left(), crop.bottom(), full.width(), full.bottom() - crop.bottom())
        )
        painter.drawRect(
            QRectF(full.left(), crop.top(), crop.left() - full.left(), crop.height())
        )
        painter.drawRect(
            QRectF(crop.right(), crop.top(), full.right() - crop.right(), crop.height())
        )

        pen = QPen(QColor(theme.INK))
        pen.setWidthF(1.25)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(crop)

        # Rule-of-thirds guides inside the crop.
        guide = QPen(QColor(*theme.INK_RGB, 90))
        guide.setWidthF(1.0)
        guide.setCosmetic(True)
        painter.setPen(guide)
        for i in (1, 2):
            x = crop.left() + crop.width() * i / 3.0
            y = crop.top() + crop.height() * i / 3.0
            painter.drawLine(QPointF(x, crop.top()), QPointF(x, crop.bottom()))
            painter.drawLine(QPointF(crop.left(), y), QPointF(crop.right(), y))

        # Corner handles.
        handle = QPen(QColor(theme.EMBER))
        handle.setWidthF(2.0)
        handle.setCosmetic(True)
        painter.setPen(handle)
        hs = self._handle_scene_size()
        for hx, hy in (
            (crop.left(), crop.top()),
            (crop.right(), crop.top()),
            (crop.left(), crop.bottom()),
            (crop.right(), crop.bottom()),
        ):
            painter.drawRect(QRectF(hx - hs / 2, hy - hs / 2, hs, hs))

    def _handle_scene_size(self) -> float:
        # Approximate view→scene: keep handles usable when zoomed out.
        return max(8.0, _HANDLE_VIEW_PX)

    def _hit_test(self, pos: QPointF) -> str:
        crop = self.crop_rect()
        hs = self._handle_scene_size() * 1.25
        corners = {
            "tl": QPointF(crop.left(), crop.top()),
            "tr": QPointF(crop.right(), crop.top()),
            "bl": QPointF(crop.left(), crop.bottom()),
            "br": QPointF(crop.right(), crop.bottom()),
        }
        for name, pt in corners.items():
            if abs(pos.x() - pt.x()) <= hs and abs(pos.y() - pt.y()) <= hs:
                return name
        edge = hs * 0.85
        if abs(pos.y() - crop.top()) <= edge and crop.left() <= pos.x() <= crop.right():
            return "t"
        if abs(pos.y() - crop.bottom()) <= edge and crop.left() <= pos.x() <= crop.right():
            return "b"
        if abs(pos.x() - crop.left()) <= edge and crop.top() <= pos.y() <= crop.bottom():
            return "l"
        if abs(pos.x() - crop.right()) <= edge and crop.top() <= pos.y() <= crop.bottom():
            return "r"
        if crop.contains(pos):
            return "move"
        return ""

    def hoverMoveEvent(self, event) -> None:
        mode = self._hit_test(event.pos())
        cursors = {
            "tl": Qt.CursorShape.SizeFDiagCursor,
            "br": Qt.CursorShape.SizeFDiagCursor,
            "tr": Qt.CursorShape.SizeBDiagCursor,
            "bl": Qt.CursorShape.SizeBDiagCursor,
            "t": Qt.CursorShape.SizeVerCursor,
            "b": Qt.CursorShape.SizeVerCursor,
            "l": Qt.CursorShape.SizeHorCursor,
            "r": Qt.CursorShape.SizeHorCursor,
            "move": Qt.CursorShape.SizeAllCursor,
        }
        if mode in cursors:
            self.setCursor(QCursor(cursors[mode]))
        else:
            self.unsetCursor()
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event) -> None:
        mode = self._hit_test(event.pos())
        if not mode:
            event.ignore()
            return
        self._drag_mode = mode
        self._drag_origin = QPointF(event.pos())
        self._drag_start_insets = self.insets()
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if not self._drag_mode or self._img_w <= 0 or self._img_h <= 0:
            event.ignore()
            return
        dx = (event.pos().x() - self._drag_origin.x()) / self._img_w
        dy = (event.pos().y() - self._drag_origin.y()) / self._img_h
        l, r, t, b = self._drag_start_insets
        mode = self._drag_mode
        if mode == "move":
            # Preserve size; clamp so we don't push past edges.
            width_frac = 1.0 - l - r
            height_frac = 1.0 - t - b
            l = np_clip(l + dx, 0.0, 1.0 - width_frac)
            t = np_clip(t + dy, 0.0, 1.0 - height_frac)
            r = 1.0 - width_frac - l
            b = 1.0 - height_frac - t
        else:
            if "l" in mode:
                l = np_clip(l + dx, 0.0, _MAX_INSET)
            if "r" in mode:
                r = np_clip(r - dx, 0.0, _MAX_INSET)
            if "t" in mode:
                t = np_clip(t + dy, 0.0, _MAX_INSET)
            if "b" in mode:
                b = np_clip(b - dy, 0.0, _MAX_INSET)
            if self._aspect is not None:
                l, r, t, b = self._constrain_aspect(l, r, t, b, mode)
        self._left, self._right, self._top, self._bottom = l, r, t, b
        self._normalize_min_size()
        self.update()
        self.insetsChanged.emit(self._left, self._right, self._top, self._bottom)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_mode:
            self._drag_mode = None
            self.editingFinished.emit()
            event.accept()
            return
        event.ignore()

    def _normalize_min_size(self) -> None:
        if self._left + self._right > 1.0 - _MIN_FRAC:
            overflow = self._left + self._right - (1.0 - _MIN_FRAC)
            self._left = max(0.0, self._left - overflow / 2)
            self._right = max(0.0, self._right - overflow / 2)
        if self._top + self._bottom > 1.0 - _MIN_FRAC:
            overflow = self._top + self._bottom - (1.0 - _MIN_FRAC)
            self._top = max(0.0, self._top - overflow / 2)
            self._bottom = max(0.0, self._bottom - overflow / 2)

    def _apply_aspect_from_center(self) -> None:
        if self._aspect is None or self._img_w <= 0 or self._img_h <= 0:
            return
        crop = self.crop_rect()
        cx, cy = crop.center().x(), crop.center().y()
        # Fit largest aspect-correct rect around center that stays in frame.
        max_w = min(cx, self._img_w - cx) * 2
        max_h = min(cy, self._img_h - cy) * 2
        if max_w / self._aspect <= max_h:
            w = max_w
            h = w / self._aspect
        else:
            h = max_h
            w = h * self._aspect
        l = (cx - w / 2) / self._img_w
        r = 1.0 - (cx + w / 2) / self._img_w
        t = (cy - h / 2) / self._img_h
        b = 1.0 - (cy + h / 2) / self._img_h
        self._left, self._right, self._top, self._bottom = (
            max(0.0, l),
            max(0.0, r),
            max(0.0, t),
            max(0.0, b),
        )
        self._normalize_min_size()
        self.insetsChanged.emit(self._left, self._right, self._top, self._bottom)

    def _constrain_aspect(
        self, l: float, r: float, t: float, b: float, mode: str
    ) -> tuple[float, float, float, float]:
        assert self._aspect is not None
        img_aspect = self._img_w / max(1.0, float(self._img_h))
        # crop_w/crop_h in pixels = aspect; in fraction space:
        # (w_frac * img_w) / (h_frac * img_h) = aspect
        # => w_frac / h_frac = aspect / img_aspect
        target = self._aspect / img_aspect
        w_frac = max(_MIN_FRAC, 1.0 - l - r)
        h_frac = max(_MIN_FRAC, 1.0 - t - b)
        if mode in ("t", "b"):
            new_w = h_frac * target
            cx = l + w_frac / 2.0
            l = np_clip(cx - new_w / 2.0, 0.0, _MAX_INSET)
            r = np_clip(1.0 - new_w - l, 0.0, _MAX_INSET)
        elif mode in ("l", "r"):
            new_h = w_frac / target
            cy = t + h_frac / 2.0
            t = np_clip(cy - new_h / 2.0, 0.0, _MAX_INSET)
            b = np_clip(1.0 - new_h - t, 0.0, _MAX_INSET)
        else:
            # Corner: width is authoritative; pin the opposite corner.
            new_h = w_frac / target
            if "t" in mode:
                t = np_clip(1.0 - b - new_h, 0.0, _MAX_INSET)
            else:
                b = np_clip(1.0 - t - new_h, 0.0, _MAX_INSET)
        return l, r, t, b


def np_clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))
