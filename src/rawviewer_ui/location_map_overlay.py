"""Draggable GPS cluster map overlay for single-image view."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QCursor, QMouseEvent, QPainter, QWheelEvent
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from gps_neighbors import GpsCluster, GpsMapView, build_map_view, cluster_points, corpus_gps_points, current_point_for_path, DEFAULT_CLUSTER_RADIUS_M
from location_map_engine import (
    DEFAULT_TILE_STYLE,
    LocationMapModel,
    WIDGET_H,
    WIDGET_W,
    default_map_cache_dir,
    is_map_frame_edge,
    paint_map_content,
    probe_map_tiles_online,
)

logger = logging.getLogger(__name__)


class _MapLoadWorker(QThread):
    """Build GPS corpus + tile model off the UI thread (POC-style folder scan)."""

    finished = pyqtSignal(object, str)  # LocationMapModel | None, error

    def __init__(
        self,
        file_paths: list[str],
        current_path: str,
        cache_dir: Path,
        tile_style: str = DEFAULT_TILE_STYLE,
    ):
        super().__init__()
        self._file_paths = list(file_paths)
        self._current_path = current_path
        self._cache_dir = cache_dir
        self._tile_style = tile_style

    def run(self) -> None:
        try:
            if self.isInterruptionRequested():
                return
            corpus = corpus_gps_points(self._file_paths)
            if self.isInterruptionRequested():
                return
            current_pt = current_point_for_path(corpus, self._current_path)
            if current_pt is None:
                self.finished.emit(None, "no_gps")
                return
            map_view = build_map_view(current_pt, corpus)
            if map_view is None:
                self.finished.emit(None, "no_cluster")
                return
            if self.isInterruptionRequested():
                return
            all_clusters = cluster_points(corpus, DEFAULT_CLUSTER_RADIUS_M)
            model = LocationMapModel(
                map_view,
                self._cache_dir,
                tile_style=self._tile_style,
                all_clusters=all_clusters,
            )
            if self.isInterruptionRequested():
                return
            self.finished.emit(model, "")
        except Exception as exc:
            if self.isInterruptionRequested():
                return
            logger.warning("Map load failed: %s", exc, exc_info=True)
            self.finished.emit(None, str(exc))


class _MapCanvas(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._model: Optional[LocationMapModel] = None
        self._on_cluster_clicked: Optional[Callable[[GpsCluster], None]] = None
        self._press_pos = None
        self._panning = False
        self._pending_cluster: Optional[GpsCluster] = None
        self.setFixedSize(WIDGET_W, WIDGET_H)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))

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

    def zoom_in(self) -> bool:
        model = self._model
        if model is None or not model.zoom_in():
            return False
        self.update()
        return True

    def zoom_out(self) -> bool:
        model = self._model
        if model is None or not model.zoom_out():
            return False
        self.update()
        return True

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
            parent = self.parentWidget()
            if parent is not None and hasattr(parent, "_sync_zoom_buttons"):
                parent._sync_zoom_buttons()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        model = self._model
        if model is None or event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position()
        if is_map_frame_edge(pos.x(), pos.y()):
            if model.reset_view():
                self.update()
            event.accept()
            return
        self._press_pos = pos
        self._panning = False
        picked = model.hit_test_pin(event.position().x(), event.position().y())
        self._pending_cluster = picked.cluster if picked is not None else None
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        model = self._model
        if (
            model is None
            or self._press_pos is None
            or not (event.buttons() & Qt.MouseButton.LeftButton)
        ):
            return
        delta = event.position() - self._press_pos
        if not self._panning and delta.manhattanLength() > 8:
            self._panning = True
            self._pending_cluster = None
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
        if self._panning:
            if model.pan_by_widget_delta(delta.x(), delta.y()):
                self._press_pos = event.position()
                self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._panning:
            self._panning = False
        elif self._pending_cluster is not None and self._on_cluster_clicked:
            self._on_cluster_clicked(self._pending_cluster)
        self._press_pos = None
        self._pending_cluster = None
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        event.accept()


class ImageLocationMapWidget(QWidget):
    """352×264 draggable map card (cluster pins, OSM tiles)."""

    _CARD_W = WIDGET_W
    _CARD_H = WIDGET_H + 22
    _BG = Qt.GlobalColor.transparent

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._model: Optional[LocationMapModel] = None
        self._worker: Optional[_MapLoadWorker] = None
        self._orphan_workers: list[_MapLoadWorker] = []
        self._load_generation = 0
        self._probe: Optional[QThread] = None
        self._orphan_probes: list[QThread] = []
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

        header = QWidget(self)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(4, 0, 4, 0)
        header_layout.setSpacing(4)

        self._status = QLabel("Map")
        self._status.setStyleSheet("color: #bbb; font-size: 10px;")
        header_layout.addWidget(self._status, 1)

        zoom_btn_style = (
            "QPushButton { color: #ddd; background: rgba(40,40,40,200);"
            " border: 1px solid #555; border-radius: 3px; min-width: 22px; max-width: 22px;"
            " min-height: 18px; max-height: 18px; padding: 0; font-size: 13px; font-weight: bold; }"
            "QPushButton:hover { background: rgba(70,70,70,220); }"
            "QPushButton:disabled { color: #666; border-color: #444; }"
        )
        self._zoom_out_btn = QPushButton("−")
        self._zoom_out_btn.setFlat(True)
        self._zoom_out_btn.setStyleSheet(zoom_btn_style)
        self._zoom_out_btn.setToolTip("Zoom out")
        self._zoom_out_btn.clicked.connect(self._on_zoom_out_clicked)
        self._zoom_in_btn = QPushButton("+")
        self._zoom_in_btn.setFlat(True)
        self._zoom_in_btn.setStyleSheet(zoom_btn_style)
        self._zoom_in_btn.setToolTip("Zoom in")
        self._zoom_in_btn.clicked.connect(self._on_zoom_in_clicked)
        header_layout.addWidget(self._zoom_out_btn)
        header_layout.addWidget(self._zoom_in_btn)

        self._canvas = _MapCanvas(self)
        layout.addWidget(header)
        layout.addWidget(self._canvas)

    def set_cluster_click_handler(self, handler: Optional[Callable[[GpsCluster], None]]) -> None:
        self._on_cluster_clicked = handler
        self._canvas.set_model(self._model, on_cluster_clicked=handler)

    def _sync_zoom_buttons(self) -> None:
        model = self._model
        can_in = bool(model and model.can_zoom_in())
        can_out = bool(model and model.can_zoom_out())
        self._zoom_in_btn.setEnabled(can_in)
        self._zoom_out_btn.setEnabled(can_out)

    def _on_zoom_in_clicked(self) -> None:
        if self._canvas.zoom_in():
            self._sync_zoom_buttons()

    def _on_zoom_out_clicked(self) -> None:
        if self._canvas.zoom_out():
            self._sync_zoom_buttons()

    def clear(self) -> None:
        self._cancel_worker()
        self._model = None
        self._canvas.set_model(None)
        self._status.setText("Map")
        self._zoom_in_btn.setEnabled(False)
        self._zoom_out_btn.setEnabled(False)
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

        if self._online is False:
            self.clear()
            return

        self._status.setText("Loading map…")
        self._zoom_in_btn.setEnabled(False)
        self._zoom_out_btn.setEnabled(False)
        self.show()
        paths = list(file_paths)
        self._load_generation += 1
        load_generation = self._load_generation
        worker = _MapLoadWorker(
            paths,
            current_path,
            cache_dir or default_map_cache_dir(),
        )
        self._worker = worker

        def _done(model: object, err: str) -> None:
            if load_generation != self._load_generation or self._worker is not worker:
                return
            self._worker = None
            self._release_worker(worker)
            if model is None:
                if err == "no_gps":
                    logger.info(
                        "Location map hidden: no GPS for %s",
                        os.path.basename(current_path),
                    )
                elif err == "no_cluster":
                    logger.warning(
                        "Location map hidden: no cluster for %s",
                        os.path.basename(current_path),
                    )
                else:
                    logger.warning("Location map failed: %s", err or "unknown")
                self._status.setText("Map")
                self.hide()
                return
            self._model = model
            self._status.setText("Map")
            self._canvas.set_model(model, on_cluster_clicked=self._on_cluster_clicked)
            self._canvas.update()
            self._sync_zoom_buttons()
            logger.info(
                "Location map ready for %s (%d folder paths scanned)",
                os.path.basename(current_path),
                len(paths),
            )

        worker.finished.connect(_done)
        worker.start()

    def ensure_online(self, callback: Callable[[bool], None]) -> None:
        if self._online is not None:
            callback(self._online)
            return

        self._cancel_probe()

        class _Probe(QThread):
            result = pyqtSignal(bool)

            def run(self_inner) -> None:
                if self_inner.isInterruptionRequested():
                    return
                self_inner.result.emit(probe_map_tiles_online())

        probe = _Probe(self)
        self._probe = probe

        def _on_result(ok: bool) -> None:
            if self._probe is not probe:
                return
            self._probe = None
            self._online = ok
            callback(ok)

        def _probe_finished() -> None:
            if self._probe is probe:
                self._probe = None
            self._release_probe(probe)

        probe.result.connect(_on_result)
        probe.finished.connect(_probe_finished)
        probe.start()

    def _release_worker(self, worker: _MapLoadWorker) -> None:
        try:
            worker.finished.disconnect()
        except (TypeError, RuntimeError):
            pass
        if worker in self._orphan_workers:
            self._orphan_workers.remove(worker)
        worker.deleteLater()

    def _release_probe(self, probe: QThread) -> None:
        try:
            probe.result.disconnect()
        except (TypeError, RuntimeError):
            pass
        try:
            probe.finished.disconnect()
        except (TypeError, RuntimeError):
            pass
        if probe in self._orphan_probes:
            self._orphan_probes.remove(probe)
        probe.deleteLater()

    def _cancel_probe(self) -> None:
        probe = self._probe
        if probe is not None:
            self._probe = None
            try:
                probe.result.disconnect()
            except (TypeError, RuntimeError):
                pass
            if probe.isRunning():
                probe.requestInterruption()
                probe.finished.connect(lambda p=probe: self._release_probe(p))
                self._orphan_probes.append(probe)
            else:
                self._release_probe(probe)
        self._orphan_probes = [p for p in self._orphan_probes if p.isRunning()]

    def _cancel_worker(self) -> None:
        worker = self._worker
        if worker is None:
            return
        self._worker = None
        self._load_generation += 1
        try:
            worker.finished.disconnect()
        except (TypeError, RuntimeError):
            pass
        if worker.isRunning():
            worker.requestInterruption()
            worker.finished.connect(lambda w=worker: self._release_worker(w))
            self._orphan_workers.append(worker)
        else:
            self._release_worker(worker)
        self._orphan_workers = [w for w in self._orphan_workers if w.isRunning()]

    def closeEvent(self, event) -> None:
        self._cancel_probe()
        self._cancel_worker()
        for worker in list(self._orphan_workers):
            if worker.isRunning():
                worker.requestInterruption()
                if not worker.wait(5000):
                    worker.terminate()
                    worker.wait(1000)
            self._release_worker(worker)
        for probe in list(self._orphan_probes):
            if probe.isRunning():
                probe.requestInterruption()
                if not probe.wait(5000):
                    probe.terminate()
                    probe.wait(1000)
            self._release_probe(probe)
        super().closeEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.childAt(
            event.position().toPoint()
        ) in (self._status, self._zoom_in_btn, self._zoom_out_btn):
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
