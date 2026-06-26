"""Draggable GPS map overlay for single-image view."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Callable

from PyQt6.QtCore import QObject, Qt, pyqtSignal, QPoint, QRectF, QRunnable, QThreadPool
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


class MapLoadSignals(QObject):
    finished = pyqtSignal(object, str)  # LocationMapModel | None, error
    gps_found = pyqtSignal()  # emitted once GPS coords are confirmed


class _MapLoadWorker(QRunnable):
    """Background loader for extracting EXIF coordinates and loading map tiles."""

    def __init__(
        self,
        current_path: str,
        cache_dir: Path,
        signals: MapLoadSignals,
    ):
        super().__init__()
        self._current_path = current_path
        self._cache_dir = cache_dir
        self.signals = signals
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            if self._cancelled:
                return
            
            # 1. Extract GPS for current file
            coords = extract_gps_coordinates(self._current_path)
            if coords is None:
                if not self._cancelled:
                    self.signals.finished.emit(None, "no_gps")
                return
            
            if self._cancelled:
                return

            # GPS confirmed — tell the UI to show the loading placeholder now
            self.signals.gps_found.emit()

            if self._cancelled:
                return
            
            lat, lon = coords
            
            # 2. Build LocationMapModel
            model = LocationMapModel(
                lat,
                lon,
                self._cache_dir,
            )
            
            if self._cancelled:
                return
                
            self.signals.finished.emit(model, "")
        except Exception as exc:
            if self._cancelled:
                return
            logger.warning("Map build worker failed: %s", exc, exc_info=True)
            self.signals.finished.emit(None, str(exc))


class ProbeSignals(QObject):
    result = pyqtSignal(bool)


class _Probe(QRunnable):
    """Background loader/probe to check if OpenStreetMap tiles are online."""

    def __init__(self, signals: ProbeSignals):
        super().__init__()
        self.signals = signals
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        if self._cancelled:
            return
        res = probe_map_tiles_online()
        if not self._cancelled:
            self.signals.result.emit(res)


class _MapCanvas(QWidget):
    """Inner QWidget that draws the stitched map tiles and location pins."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._model: Optional[LocationMapModel] = None
        self._loading: bool = False
        self.setFixedSize(WIDGET_W, WIDGET_H)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # Enable dragging the map card by clicking on the canvas as well
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def set_model(self, model: Optional[LocationMapModel]) -> None:
        self._model = model
        self.update()

    def set_loading(self, loading: bool) -> None:
        self._loading = loading
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
            if self._loading:
                # Show a subtle loading indicator while tiles are being fetched
                p.setPen(QPen(QColor(180, 180, 180, 200)))
                font = p.font()
                font.setPointSize(11)
                p.setFont(font)
                p.drawText(
                    self.rect(),
                    Qt.AlignmentFlag.AlignCenter,
                    "Loading map\u2026",
                )
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
        p.drawText(6, WIDGET_H - 6, "\u00a9 OpenStreetMap contributors")
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
        self._probe: Optional[_Probe] = None
        self._drag_press_global: Optional[QPoint] = None
        self._drag_pos_at_press: Optional[QPoint] = None
        self._online: Optional[bool] = None
        self._active_signals = set()

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
        self._canvas.set_loading(False)
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
        # Always start hidden; only show once GPS is confirmed by the worker.
        # This prevents a spurious "Loading map…" popup on images without GPS data.
        self._canvas.set_loading(False)
        self.hide()

        if not current_path or not os.path.isfile(current_path):
            return

        if self._online is False:
            return

        self._load_generation += 1
        load_generation = self._load_generation

        signals = MapLoadSignals()
        self._active_signals.add(signals)

        worker = _MapLoadWorker(
            current_path,
            cache_dir or default_map_cache_dir(),
            signals,
        )
        self._worker = worker

        def _on_gps_found() -> None:
            """GPS coordinates confirmed — now it is safe to show the loading card."""
            if load_generation != self._load_generation or self._worker is not worker:
                return
            self._canvas.set_loading(True)
            self.show()
            # _layout_map() skips hidden widgets, so re-layout now that we are
            # visible so the container can position and raise the card correctly.
            self._relayout_parent()

        def _done(model: object, err: str) -> None:
            self._active_signals.discard(signals)
            if load_generation != self._load_generation or self._worker is not worker:
                return
            self._worker = None

            # Always clear the loading state first
            self._canvas.set_loading(False)

            if model is None:
                if err == "no_gps":
                    logger.debug("Location map hidden: no GPS metadata for %s", os.path.basename(current_path))
                else:
                    logger.warning("Location map load failed: %s", err or "unknown")
                self.hide()
                return

            self._model = model
            self._canvas.set_model(model)
            self._canvas.update()
            # Ensure the widget is visible (gps_found will have shown it already
            # for the loading phase, but show() is idempotent).
            self.show()
            self._relayout_parent()

        signals.gps_found.connect(_on_gps_found)
        signals.finished.connect(_done)
        QThreadPool.globalInstance().start(worker)

    def _relayout_parent(self) -> None:
        """Ask the parent single-view container to re-position and raise this widget."""
        parent = self.parentWidget()
        if parent is not None and hasattr(parent, "relayout_map"):
            parent.relayout_map()

    def ensure_online(self, callback: Callable[[bool], None]) -> None:
        if self._online is not None:
            callback(self._online)
            return

        self._cancel_probe()

        signals = ProbeSignals()
        self._active_signals.add(signals)

        probe = _Probe(signals)
        self._probe = probe

        def _on_result(ok: bool) -> None:
            self._active_signals.discard(signals)
            if self._probe is not probe:
                return
            self._probe = None
            self._online = ok
            callback(ok)

        signals.result.connect(_on_result)
        QThreadPool.globalInstance().start(probe)

    def _cancel_worker(self) -> None:
        worker = self._worker
        if worker is not None:
            self._worker = None
            self._load_generation += 1
            try:
                worker.cancel()
                self._active_signals.discard(worker.signals)
            except RuntimeError:
                pass

    def _cancel_probe(self) -> None:
        probe = self._probe
        if probe is not None:
            self._probe = None
            try:
                probe.cancel()
                self._active_signals.discard(probe.signals)
            except RuntimeError:
                pass

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
