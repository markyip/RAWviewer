"""Draggable GPS map overlay for single-image view."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Callable

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal, QPoint, QRectF
from PyQt6.QtGui import QCursor, QMouseEvent, QPainter, QPainterPath, QColor, QPen
from PyQt6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

import metadata_backend
from location_map_engine import (
    DEFAULT_ZOOM,
    LocationMapModel,
    WIDGET_H,
    WIDGET_W,
    default_map_cache_dir,
    probe_map_tiles_online,
)

logger = logging.getLogger(__name__)

_active_workers = set()
_active_probes = set()
_about_to_quit_connected = False

def _cleanup_threads():
    for w in list(_active_workers):
        if w.isRunning():
            w.requestInterruption()
            w.wait(1000)
    for p in list(_active_probes):
        if p.isRunning():
            p.requestInterruption()
            p.wait(1000)
    _active_workers.clear()
    _active_probes.clear()


def _tag_text(tags: dict, *keys: str) -> str:
    for key in keys:
        tag = tags.get(key)
        if tag is None:
            continue
        val = getattr(tag, "values", tag)
        if isinstance(val, (list, tuple)) and val:
            val = val[0]
        s = str(val).strip()
        if s:
            return s
    return ""


def _ratio_to_float(v: any) -> Optional[float]:
    try:
        if hasattr(v, "num") and hasattr(v, "den"):
            den = float(v.den) if float(v.den) != 0 else 1.0
            return float(v.num) / den
        if isinstance(v, str) and "/" in v:
            num, den = v.split("/", 1)
            d = float(den) if float(den) != 0 else 1.0
            return float(num) / d
        return float(v)
    except Exception:
        return None


def gps_to_decimal(gps_vals: any, ref: str) -> Optional[float]:
    try:
        vals = getattr(gps_vals, "values", gps_vals)
        if isinstance(vals, str):
            import re
            parts = re.split(r"[\s,]+", vals.strip())
            if len(parts) >= 3:
                parsed = []
                for p in parts:
                    if "/" in p:
                        num, den = p.split("/", 1)
                        parsed.append(float(num) / (float(den) if float(den) != 0 else 1.0))
                    else:
                        parsed.append(float(p))
                vals = parsed
        if not vals or not isinstance(vals, (list, tuple)) or len(vals) < 3:
            return None
        d = _ratio_to_float(vals[0])
        m = _ratio_to_float(vals[1])
        s = _ratio_to_float(vals[2])
        if d is None or m is None or s is None:
            return None
        dec = float(d) + float(m) / 60.0 + float(s) / 3600.0
        ref_str = str(ref or "").strip().upper()
        if ref_str in ("S", "W"):
            dec = -dec
        return dec
    except Exception:
        return None


def extract_gps_coordinates(file_path: str) -> Optional[tuple[float, float]]:
    try:
        tags = metadata_backend.process_file_from_path(
            file_path,
            details=False,
            stop_tag="GPS GPSLongitudeRef",
        )
    except Exception:
        return None
    lat = tags.get("GPS GPSLatitude") or tags.get("EXIF GPSLatitude") or tags.get("GPSLatitude")
    lon = tags.get("GPS GPSLongitude") or tags.get("EXIF GPSLongitude") or tags.get("GPSLongitude")
    if not lat or not lon:
        return None
    lat_ref = _tag_text(tags, "GPS GPSLatitudeRef", "EXIF GPSLatitudeRef", "GPSLatitudeRef")
    lon_ref = _tag_text(tags, "GPS GPSLongitudeRef", "EXIF GPSLongitudeRef", "GPSLongitudeRef")
    lat_dec = gps_to_decimal(lat, lat_ref)
    lon_dec = gps_to_decimal(lon, lon_ref)
    if lat_dec is None or lon_dec is None:
        return None
    if abs(lat_dec) <= 0.001 and abs(lon_dec) <= 0.001:
        return None
    return lat_dec, lon_dec


class _MapLoadWorker(QThread):
    """Background thread loader for extracting EXIF coordinates and loading map tiles."""

    finished = pyqtSignal(object, str)  # LocationMapModel | None, error

    def __init__(
        self,
        current_path: str,
        cache_dir: Path,
    ):
        super().__init__()
        self._current_path = current_path
        self._cache_dir = cache_dir

    def run(self) -> None:
        try:
            if self.isInterruptionRequested():
                return
            
            # 1. Extract GPS for current file
            coords = extract_gps_coordinates(self._current_path)
            if coords is None:
                self.finished.emit(None, "no_gps")
                return
            
            if self.isInterruptionRequested():
                return
            
            lat, lon = coords
            
            # 2. Build LocationMapModel
            model = LocationMapModel(
                lat,
                lon,
                self._cache_dir,
            )
            
            if self.isInterruptionRequested():
                return
                
            self.finished.emit(model, "")
        except Exception as exc:
            if self.isInterruptionRequested():
                return
            logger.warning("Map build worker failed: %s", exc, exc_info=True)
            self.finished.emit(None, str(exc))


class _MapCanvas(QWidget):
    """Inner QWidget that draws the stitched map tiles and location pins."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._model: Optional[LocationMapModel] = None
        self.setFixedSize(WIDGET_W, WIDGET_H)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # Enable dragging the map card by clicking on the canvas as well
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def set_model(self, model: Optional[LocationMapModel]) -> None:
        self._model = model
        self.update()

    def paintEvent(self, event) -> None:
        del event
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        
        # Round the canvas edges to match the card container
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), 4.0, 4.0)
        p.setClipPath(path)

        if self._model is None:
            p.fillRect(self.rect(), QColor(20, 22, 26))
            p.end()
            return
            
        # Draw cropped background tiles
        p.drawPixmap(0, 0, self._model.cropped)
        # Draw pins
        self._model.paint_pins(p)
        # Draw attribution at bottom left
        p.setPen(QPen(QColor(255, 255, 255, 180)))
        font = p.font()
        font.setPointSize(8)
        p.setFont(font)
        p.drawText(6, WIDGET_H - 6, "© OpenStreetMap contributors")
        p.end()


class ImageLocationMapWidget(QWidget):
    """352×264 draggable map card overlay showing stitched tiles without zoom/pan controls."""

    _CARD_W = WIDGET_W + 8
    _CARD_H = WIDGET_H + 8

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._model: Optional[LocationMapModel] = None
        self._worker: Optional[_MapLoadWorker] = None
        self._load_generation = 0
        self._probe: Optional[QThread] = None
        self._drag_press_global: Optional[QPoint] = None
        self._drag_pos_at_press: Optional[QPoint] = None
        self._online: Optional[bool] = None

        self.setFixedSize(self._CARD_W, self._CARD_H)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)

        self._canvas = _MapCanvas(self)
        layout.addWidget(self._canvas)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # Premium dark glassmorphic card container
        painter.setBrush(QColor(25, 27, 31, 220))
        painter.setPen(QPen(QColor(255, 255, 255, 45), 1.2))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.6, 0.6, -0.6, -0.6), 6.0, 6.0)
        painter.end()

    def clear(self) -> None:
        self._cancel_worker()
        self._model = None
        self._canvas.set_model(None)
        self.hide()

    def load_for_file(
        self,
        current_path: Optional[str],
        *,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self._cancel_worker()
        self._model = None
        self._canvas.set_model(None)

        if not current_path or not os.path.isfile(current_path):
            self.clear()
            return

        if self._online is False:
            self.clear()
            return
        
        self._load_generation += 1
        load_generation = self._load_generation
        
        worker = _MapLoadWorker(
            current_path,
            cache_dir or default_map_cache_dir(),
        )
        self._worker = worker
        
        _active_workers.add(worker)
        worker.finished.connect(lambda: _active_workers.discard(worker))
        worker.finished.connect(worker.deleteLater)

        global _about_to_quit_connected
        if not _about_to_quit_connected:
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                try:
                    app.aboutToQuit.connect(_cleanup_threads)
                    _about_to_quit_connected = True
                except Exception:
                    pass

        def _done(model: object, err: str) -> None:
            if load_generation != self._load_generation or self._worker is not worker:
                return
            self._worker = None
            
            if model is None:
                if err == "no_gps":
                    logger.debug("Location map hidden: no GPS metadata for %s", os.path.basename(current_path))
                else:
                    logger.warning("Location map load failed: %s", err or "unknown")
                self.clear()
                return
                
            self._model = model
            self._canvas.set_model(model)
            self._canvas.update()
            self.show()

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

        probe = _Probe()
        self._probe = probe
        
        _active_probes.add(probe)
        probe.finished.connect(lambda: _active_probes.discard(probe))
        probe.finished.connect(probe.deleteLater)

        global _about_to_quit_connected
        if not _about_to_quit_connected:
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                try:
                    app.aboutToQuit.connect(_cleanup_threads)
                    _about_to_quit_connected = True
                except Exception:
                    pass

        def _on_result(ok: bool) -> None:
            if self._probe is not probe:
                return
            self._probe = None
            self._online = ok
            callback(ok)

        def _probe_finished() -> None:
            if self._probe is probe:
                self._probe = None

        probe.result.connect(_on_result)
        probe.finished.connect(_probe_finished)
        probe.start()

    def _cancel_worker(self) -> None:
        worker = self._worker
        if worker is not None:
            self._worker = None
            self._load_generation += 1
            if worker.isRunning():
                worker.requestInterruption()

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

    def closeEvent(self, event) -> None:
        self._cancel_probe()
        self._cancel_worker()
        super().closeEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
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
