"""Draggable GPS cluster map overlay for single-image view."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QCursor, QMouseEvent, QPainter, QWheelEvent
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from gps_neighbors import GpsCluster, GpsMapView, build_map_view, corpus_gps_points, current_point_for_path
from location_map_engine import (
    DEFAULT_TILE_STYLE,
    LocationMapModel,
    WIDGET_H,
    WIDGET_W,
    default_map_cache_dir,
    paint_map_content,
    probe_map_tiles_online,
)

logger = logging.getLogger(__name__)


class _MapBuildWorker(QThread):
    finished = pyqtSignal(object, object)  # LocationMapModel | None, str error

    def __init__(self, map_view: GpsMapView, cache_dir: Path, tile_style: str = DEFAULT_TILE_STYLE):
        super().__init__()
        self._map_view = map_view
        self._cache_dir = cache_dir
        self._tile_style = tile_style

    def run(self) -> None:
        try:
            model = LocationMapModel(self._map_view, self._cache_dir, tile_style=self._tile_style)
            self.finished.emit(model, "")
        except Exception as exc:
            logger.warning("Map build failed: %s", exc, exc_info=True)
            self.finished.emit(None, str(exc))


class _MapCanvas(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._model: Optional[LocationMapModel] = None
        self._on_cluster_clicked: Optional[Callable[[GpsCluster], None]] = None
        self.setFixedSize(WIDGET_W, WIDGET_H)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)

    def set_model(
        self,
        model: Optional[LocationMapModel],
        *,
        on_cluster_clicked: Optional[Callable[[GpsCluster], None]] = None,
    ) -> None:
        self._model = model
        self._on_cluster_clicked = on_cluster_clicked
        self.update()

    def paintEvent(self, event) -> None:
        del event
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._model is None:
            p.fillRect(self.rect(), Qt.GlobalColor.black)
            p.end()
            return
        paint_map_content(
            p,
            self._model.cropped,
            self._model.pin_pixels,
            self._model.view_w,
            self._model.view_h,
            WIDGET_W,
            WIDGET_H,
            self._model.attribution,
        )
        p.end()

    def wheelEvent(self, event: QWheelEvent) -> None:
        model = self._model
        if model is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        changed = model.zoom_in() if delta > 0 else model.zoom_out()
        if changed:
            self.update()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        model = self._model
        if model is None or event.button() != Qt.MouseButton.LeftButton:
            return
        picked = model.hit_test_pin(event.position().x(), event.position().y())
        if picked is None:
            return
        if self._on_cluster_clicked:
            self._on_cluster_clicked(picked.cluster)
        event.accept()


class ImageLocationMapWidget(QWidget):
    """352×264 draggable map card (cluster pins, OSM tiles)."""

    _CARD_W = WIDGET_W
    _CARD_H = WIDGET_H + 22
    _BG = Qt.GlobalColor.transparent

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._model: Optional[LocationMapModel] = None
        self._worker: Optional[_MapBuildWorker] = None
        self._on_cluster_clicked: Optional[Callable[[GpsCluster], None]] = None
        self._drag_press_global = None
        self._drag_pos_at_press = None
        self._online: Optional[bool] = None

        self.setFixedSize(self._CARD_W, self._CARD_H)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._status = QLabel("Map")
        self._status.setStyleSheet("color: #bbb; font-size: 10px; padding-left: 4px;")
        self._canvas = _MapCanvas(self)
        layout.addWidget(self._status)
        layout.addWidget(self._canvas)

    def set_cluster_click_handler(self, handler: Optional[Callable[[GpsCluster], None]]) -> None:
        self._on_cluster_clicked = handler
        self._canvas.set_model(self._model, on_cluster_clicked=handler)

    def clear(self) -> None:
        self._cancel_worker()
        self._model = None
        self._canvas.set_model(None)
        self._status.setText("Map")
        self.hide()

    def load_for_files(
        self,
        file_paths: list[str],
        current_path: Optional[str],
        *,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self._cancel_worker()
        self._model = None
        self._canvas.set_model(None, on_cluster_clicked=self._on_cluster_clicked)

        if not current_path:
            self.clear()
            return

        corpus = corpus_gps_points(file_paths)
        current_pt = current_point_for_path(corpus, current_path)
        if current_pt is None:
            self.clear()
            return

        if self._online is False:
            self.clear()
            return

        map_view = build_map_view(current_pt, corpus)
        if map_view is None:
            self.clear()
            return

        self._status.setText("Loading map…")
        self.show()
        worker = _MapBuildWorker(map_view, cache_dir or default_map_cache_dir())
        self._worker = worker

        def _done(model: object, err: str) -> None:
            if self._worker is not worker:
                return
            self._worker = None
            if model is None:
                self._status.setText("Map unavailable")
                logger.warning("Location map failed: %s", err or "unknown")
                self.hide()
                return
            self._model = model
            z = model.zoom
            self._status.setText(f"GPS · z{z} ({model.base_zoom}–{model.max_zoom_limit})")
            self._canvas.set_model(model, on_cluster_clicked=self._on_cluster_clicked)
            self._canvas.update()

        worker.finished.connect(_done)
        worker.start()

    def ensure_online(self, callback: Callable[[bool], None]) -> None:
        if self._online is not None:
            callback(self._online)
            return

        class _Probe(QThread):
            result = pyqtSignal(bool)

            def run(self_inner) -> None:
                self_inner.result.emit(probe_map_tiles_online())

        probe = _Probe(self)

        def _on_result(ok: bool) -> None:
            self._online = ok
            callback(ok)

        probe.result.connect(_on_result)
        probe.finished.connect(probe.deleteLater)
        probe.start()

    def _cancel_worker(self) -> None:
        worker = self._worker
        if worker is None:
            return
        try:
            worker.finished.disconnect()
        except Exception:
            pass
        if worker.isRunning():
            worker.requestInterruption()
            worker.wait(200)
        self._worker = None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.childAt(
            event.position().toPoint()
        ) is self._status:
            self._drag_press_global = event.globalPosition().toPoint()
            self._drag_pos_at_press = self.pos()
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
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
                if hasattr(parent, "mark_map_user_moved"):
                    parent.mark_map_user_moved()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_press_global = None
            self._drag_pos_at_press = None
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
            event.accept()
            return
        super().mouseReleaseEvent(event)
