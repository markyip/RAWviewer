"""Draggable PV2012 point tone curve editor for the adjust panel."""

from __future__ import annotations

from typing import List, Optional, Tuple

from PyQt6.QtCore import QPoint, QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QMouseEvent, QPainter, QPen
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from raw_tone_curve import (
    default_tone_curve_points,
    deserialize_tone_curve_points,
    insert_tone_curve_point,
    move_tone_curve_point,
    normalize_tone_curve_points,
    remove_tone_curve_point,
    sample_point_curve_for_display,
    serialize_tone_curve_points,
)

Point = Tuple[float, float]


class ToneCurveWidget(QWidget):
    """Lightroom-style point curve on 0–255 input/output axes."""

    points_changed = pyqtSignal()
    editing_finished = pyqtSignal()

    _HIT_RADIUS = 9.0
    _MARGIN = 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self._points: List[Point] = default_tone_curve_points()
        self._drag_index: Optional[int] = None
        self._hover_index: Optional[int] = None
        self.setFixedHeight(108)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))

    def points(self) -> List[Point]:
        return list(self._points)

    def serialized_points(self) -> str:
        return serialize_tone_curve_points(self._points)

    def load_serial(self, serial: str) -> None:
        pts = deserialize_tone_curve_points(serial)
        self._points = (
            normalize_tone_curve_points(pts) if pts else default_tone_curve_points()
        )
        self.update()

    def reset_linear(self) -> None:
        self._points = default_tone_curve_points()
        self.update()

    def _plot_rect(self):
        m = self._MARGIN
        return m, m, self.width() - 2 * m, self.height() - 2 * m

    def _to_widget(self, x: float, y: float) -> QPointF:
        px, py, pw, ph = self._plot_rect()
        wx = px + (x / 255.0) * pw
        wy = py + (1.0 - y / 255.0) * ph
        return QPointF(wx, wy)

    def _from_widget(self, pos: QPointF) -> Point:
        px, py, pw, ph = self._plot_rect()
        if pw <= 0 or ph <= 0:
            return 0.0, 0.0
        x = (pos.x() - px) / pw * 255.0
        y = (1.0 - (pos.y() - py) / ph) * 255.0
        return float(max(0.0, min(255.0, x))), float(max(0.0, min(255.0, y)))

    def _hit_index(self, pos: QPointF) -> Optional[int]:
        best: Optional[int] = None
        best_d = self._HIT_RADIUS * self._HIT_RADIUS
        for i, pt in enumerate(self._points):
            wp = self._to_widget(*pt)
            dx = wp.x() - pos.x()
            dy = wp.y() - pos.y()
            d = dx * dx + dy * dy
            if d <= best_d:
                best_d = d
                best = i
        return best

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        px, py, pw, ph = self._plot_rect()

        painter.fillRect(self.rect(), QColor(22, 22, 22, 0))
        painter.setPen(QPen(QColor(55, 55, 55), 1))
        painter.drawRect(int(px), int(py), int(pw), int(ph))

        # Grid
        painter.setPen(QPen(QColor(48, 48, 48), 1))
        for i in range(1, 4):
            gx = px + pw * i / 4.0
            gy = py + ph * i / 4.0
            painter.drawLine(int(gx), int(py), int(gx), int(py + ph))
            painter.drawLine(int(px), int(gy), int(px + pw), int(gy))

        # Identity reference
        painter.setPen(QPen(QColor(70, 70, 70), 1, Qt.PenStyle.DashLine))
        p0 = self._to_widget(0.0, 0.0)
        p1 = self._to_widget(255.0, 255.0)
        painter.drawLine(p0, p1)

        # Smooth monotonic-cubic curve -- sampled from the same PCHIP fit
        # actually applied to the image (sample_point_curve_for_display),
        # not a straight connect-the-dots polyline between knots.
        painter.setPen(QPen(QColor(144, 202, 249), 2))
        samples = sample_point_curve_for_display(self._points, n_samples=96)
        if samples:
            prev: Optional[QPointF] = None
            for x, y in samples:
                wp = self._to_widget(x, y)
                if prev is not None:
                    painter.drawLine(prev, wp)
                prev = wp

        # Control points
        for i, pt in enumerate(self._points):
            wp = self._to_widget(*pt)
            is_endpoint = i == 0 or i == len(self._points) - 1
            active = i == self._drag_index or i == self._hover_index
            radius = 5.5 if active else 4.5
            fill = QColor(210, 230, 255) if active else QColor(170, 200, 235)
            if is_endpoint:
                fill = QColor(190, 190, 190) if not active else QColor(220, 220, 220)
            painter.setBrush(fill)
            painter.setPen(QPen(QColor(30, 30, 30), 1))
            painter.drawEllipse(wp, radius, radius)

        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = QPointF(event.position())
        hit = self._hit_index(pos)
        if hit is not None:
            self._drag_index = hit
            event.accept()
            return
        if event.modifiers() & Qt.KeyboardModifier.AltModifier:
            event.accept()
            return
        x, y = self._from_widget(pos)
        self._points = insert_tone_curve_point(self._points, x, y)
        self._drag_index = self._hit_index(pos)
        self.update()
        self.points_changed.emit()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = QPointF(event.position())
        if self._drag_index is not None:
            x, y = self._from_widget(pos)
            self._points = move_tone_curve_point(self._points, self._drag_index, x, y)
            self.update()
            self.points_changed.emit()
            event.accept()
            return
        hover = self._hit_index(pos)
        if hover != self._hover_index:
            self._hover_index = hover
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._drag_index is not None:
            self._drag_index = None
            self.editing_finished.emit()
            event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = QPointF(event.position())
        hit = self._hit_index(pos)
        if hit is not None and 0 < hit < len(self._points) - 1:
            self._points = remove_tone_curve_point(self._points, hit)
            self.update()
            self.points_changed.emit()
            self.editing_finished.emit()
            event.accept()

    def leaveEvent(self, _event) -> None:
        self._hover_index = None
        self.update()


class ToneCurveEditorRow(QWidget):
    """Curve widget with a compact reset control."""

    points_changed = pyqtSignal()
    editing_finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        hint = QLabel("Drag points · click to add · double-click to remove")
        hint.setStyleSheet("color: #707070; font-size: 9px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.curve = ToneCurveWidget()
        self.curve.points_changed.connect(self.points_changed.emit)
        self.curve.editing_finished.connect(self.editing_finished.emit)
        row.addWidget(self.curve, 1)

        reset_btn = QPushButton("Linear")
        reset_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        reset_btn.setFixedWidth(52)
        reset_btn.setToolTip("Reset point curve to linear")
        reset_btn.clicked.connect(self._on_reset)
        row.addWidget(reset_btn, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(row)

    def _on_reset(self) -> None:
        self.curve.reset_linear()
        self.points_changed.emit()
        self.editing_finished.emit()

    def serialized_points(self) -> str:
        return self.curve.serialized_points()

    def load_serial(self, serial: str) -> None:
        self.curve.load_serial(serial)

    def reset_linear(self) -> None:
        self.curve.reset_linear()
