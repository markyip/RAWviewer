"""RGB and luminance histogram for single-image view."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from exposure_clipping import clipping_counts
from PyQt6.QtCore import Qt, QRect, QRectF, QPointF
from PyQt6.QtGui import QColor, QCursor, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import QSizePolicy, QWidget


def _qimage_to_rgb_array(image: QImage) -> Optional[np.ndarray]:
    img = image.convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    if w < 1 or h < 1:
        return None
    bpl = img.bytesPerLine()
    nbytes = h * bpl
    bits = img.constBits()
    if bits is None:
        return None
    try:
        if hasattr(bits, "asstring"):
            raw = bits.asstring(nbytes)
        else:
            raw = bytes(memoryview(bits)[:nbytes])
    except (BufferError, TypeError, AttributeError):
        out = np.empty((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                c = img.pixel(x, y)
                out[y, x, 0] = (c >> 16) & 0xFF
                out[y, x, 1] = (c >> 8) & 0xFF
                out[y, x, 2] = c & 0xFF
        return out
    arr = np.frombuffer(bytearray(raw), dtype=np.uint8).reshape(h, bpl)
    return np.ascontiguousarray(arr[:, : w * 3].reshape(h, w, 3))


def _histograms_from_rgb(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = rgb[:, :, 0].ravel()
    g = rgb[:, :, 1].ravel()
    b = rgb[:, :, 2].ravel()
    hr = np.bincount(r, minlength=256).astype(np.float64)
    hg = np.bincount(g, minlength=256).astype(np.float64)
    hb = np.bincount(b, minlength=256).astype(np.float64)
    lum = (
        0.2126 * r.astype(np.float64)
        + 0.7152 * g.astype(np.float64)
        + 0.0722 * b.astype(np.float64)
    ).astype(np.int32)
    lum = np.clip(lum, 0, 255).astype(np.uint8)
    hl = np.bincount(lum, minlength=256).astype(np.float64)
    return hr, hg, hb, hl


class ImageHistogramWidget(QWidget):
    """Small 16:9 card with RGB + luminance histogram; call set_pixmap / clear from the main viewer."""

    _MAX_SAMPLE_SIDE = 384
    # 16:9 card (width × height), exact ratio 352:198 = 16:9
    _CARD_W = 352
    _CARD_H = 198
    _BG = QColor(30, 30, 30, 200)
    _BORDER = QColor(255, 255, 255, 45)
    _RADIUS = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hr: Optional[np.ndarray] = None
        self._hg: Optional[np.ndarray] = None
        self._hb: Optional[np.ndarray] = None
        self._hl: Optional[np.ndarray] = None
        self._hi_clip = 0
        self._lo_clip = 0
        self._clip_total = 0
        self._polys: Optional[list] = None  # cached QPolygonF per channel
        self._peaks: Optional[list] = None  # cached peak value per channel
        self.setFixedSize(self._CARD_W, self._CARD_H)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        self._drag_press_global = None
        self._drag_pos_at_press = None

    def enterEvent(self, event):
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_press_global = event.globalPosition().toPoint()
            self._drag_pos_at_press = self.pos()
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (
            self._drag_press_global is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            parent = self.parentWidget()
            if parent:
                d = event.globalPosition().toPoint() - self._drag_press_global
                nx = self._drag_pos_at_press.x() + d.x()
                ny = self._drag_pos_at_press.y() + d.y()
                nx = max(0, min(nx, parent.width() - self.width()))
                ny = max(0, min(ny, parent.height() - self.height()))
                self.move(nx, ny)
                if hasattr(parent, "mark_histogram_user_moved"):
                    parent.mark_histogram_user_moved()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_press_global = None
            self._drag_pos_at_press = None
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def clear(self) -> None:
        self._hr = self._hg = self._hb = self._hl = None
        self._hi_clip = self._lo_clip = self._clip_total = 0
        self._polys = self._peaks = None
        self.update()

    def set_pixmap(self, pixmap: QPixmap) -> None:
        if pixmap is None or pixmap.isNull():
            self.clear()
            return
        img = pixmap.toImage()
        if img.isNull():
            self.clear()
            return
        w, h = img.width(), img.height()
        m = max(w, h)
        if m > self._MAX_SAMPLE_SIDE:
            scale = self._MAX_SAMPLE_SIDE / m
            img = img.scaled(
                max(1, int(w * scale)),
                max(1, int(h * scale)),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        rgb = _qimage_to_rgb_array(img)
        if rgb is None:
            self.clear()
            return
        self._hr, self._hg, self._hb, self._hl = _histograms_from_rgb(rgb)
        self._hi_clip, self._lo_clip, self._clip_total = clipping_counts(rgb)
        self._polys = self._peaks = None  # invalidate cached geometry
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect().adjusted(1, 1, -1, -1)), self._RADIUS, self._RADIUS)
        painter.fillPath(path, self._BG)
        painter.strokePath(path, QPen(self._BORDER, 1))

        inner = self.rect().adjusted(7, 6, -7, -7)
        chart = inner
        if self._hr is None or chart.width() < 4 or chart.height() < 4:
            painter.setPen(QColor(160, 160, 160, 200))
            painter.drawText(chart, Qt.AlignmentFlag.AlignCenter, "—")
            return

        if self._polys is None:
            x0, y0 = chart.left(), chart.top()
            rw, rh = chart.width(), chart.height()
            polys = []
            peaks = []
            for hist in (self._hr, self._hg, self._hb, self._hl):
                peak = float(np.max(hist))
                if peak < 1.0:
                    peak = 1.0
                peaks.append(peak)
                poly = QPolygonF()
                for i in range(256):
                    vx = x0 + (i / 255.0) * rw
                    vy = y0 + rh - (hist[i] / peak) * rh
                    poly.append(QPointF(vx, vy))
                polys.append(poly)
            self._polys = polys
            self._peaks = peaks

        for poly, color, width_f in (
            (self._polys[0], QColor(220, 90, 90, 200), 1.0),
            (self._polys[1], QColor(90, 200, 110, 200), 1.0),
            (self._polys[2], QColor(100, 160, 255, 200), 1.0),
            (self._polys[3], QColor(230, 230, 230, 255), 1.35),
        ):
            painter.setPen(QPen(color, width_f))
            painter.drawPolyline(poly)

        if self._clip_total > 0:
            if self._lo_clip > 0:
                painter.fillRect(
                    chart.left(),
                    chart.bottom() - 5,
                    5,
                    5,
                    QColor(90, 150, 255, 240),
                )
            if self._hi_clip > 0:
                painter.fillRect(
                    chart.right() - 4,
                    chart.bottom() - 5,
                    5,
                    5,
                    QColor(255, 70, 70, 240),
                )
