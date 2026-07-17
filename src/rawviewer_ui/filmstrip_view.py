"""Bottom film-strip overlay for single-image view (virtualized justified thumbnails)."""

from __future__ import annotations

import bisect
import os
from typing import Dict, List, Optional, Set

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QTimer, QUrl, QMimeData, QPoint
from PyQt6.QtGui import QPixmap, QImage, QWheelEvent, QKeyEvent, QTransform, QDrag
from PyQt6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QWidget,
    QApplication,
)

from rawviewer_ui.widgets import stamp_rawviewer_export_drag

# Single justified row
ROW_H = 78
MIN_GAP = 2
SIDE_MARGIN = 4
MIN_CELL_W = 14
MAX_CELL_W = 140
DEFAULT_ASPECT = 1.5
def _filmstrip_env_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default)).strip()))
    except (TypeError, ValueError):
        return default


BUFFER_PX = _filmstrip_env_int("RAWVIEWER_FILMSTRIP_BUFFER_PX", 180, minimum=80)
BUFFER_CELLS = _filmstrip_env_int("RAWVIEWER_FILMSTRIP_BUFFER_CELLS", 4, minimum=1)
_LAZY_WARM_THRESHOLD = _filmstrip_env_int(
    "RAWVIEWER_FILMSTRIP_LAZY_WARM_THRESHOLD", 400, minimum=50
)
_WARM_BAND = _filmstrip_env_int("RAWVIEWER_FILMSTRIP_WARM_BAND", 56, minimum=8)
# Uniform inset inside each cell so portrait/landscape share the same padding.
CELL_BORDER = 2
THUMB_PAD_V = 2
THUMB_PAD_H = 2
# Rebuild layout when stored aspect differs from measured thumb aspect.
ASPECT_REBUILD_EPS = 0.02


def _path_key(path: str) -> str:
    try:
        if not path:
            return ""
        return os.path.normcase(os.path.normpath(path))
    except Exception:
        return path or ""


def _thumbnail_to_pixmap(thumbnail) -> Optional[QPixmap]:
    if thumbnail is None:
        return None
    if isinstance(thumbnail, QPixmap):
        return thumbnail if not thumbnail.isNull() else None
    if isinstance(thumbnail, QImage):
        if thumbnail.isNull():
            return None
        rgb = thumbnail.convertToFormat(QImage.Format.Format_RGB888)
        return QPixmap.fromImage(rgb) if not rgb.isNull() else None
    try:
        arr = np.ascontiguousarray(thumbnail)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        h, w, c = arr.shape[:3]
        if h <= 0 or w <= 0:
            return None
        if c == 3:
            bpl = arr.strides[0]
            qimg = QImage(arr.data, w, h, bpl, QImage.Format.Format_RGB888).copy()
        elif c == 4:
            bpl = arr.strides[0]
            qimg = QImage(arr.data, w, h, bpl, QImage.Format.Format_RGBA8888).copy()
        else:
            return None
        return QPixmap.fromImage(qimg)
    except Exception:
        return None


class FilmstripCell(QLabel):
    clicked = pyqtSignal(int, object)  # index, QMouseEvent

    def __init__(self, parent=None):
        super().__init__(parent)
        self.index = -1
        self.file_path = ""
        self._drag_start_pos = None
        self._drag_started = False

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.file_path:
            self._drag_start_pos = event.position().toPoint()
            self._drag_started = False
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.MouseButton.LeftButton) or not self.file_path:
            super().mouseMoveEvent(event)
            return

        if self._drag_start_pos is not None and not self._drag_started:
            dist = (event.position().toPoint() - self._drag_start_pos).manhattanLength()
            if dist >= QApplication.startDragDistance():
                self._drag_started = True
                self._start_drag()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            drag_started = self._drag_started
            self._drag_start_pos = None
            self._drag_started = False
            if not drag_started and self.index >= 0:
                self.clicked.emit(self.index, event)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def _start_drag(self):
        if not self.file_path or not os.path.exists(self.file_path):
            return
        
        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setUrls([QUrl.fromLocalFile(self.file_path)])
        stamp_rawviewer_export_drag(mime_data)
        drag.setMimeData(mime_data)
        
        px = self.pixmap()
        if px and not px.isNull():
            drag_px = px.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            drag.setPixmap(drag_px)
            drag.setHotSpot(QPoint(drag_px.width() // 2, drag_px.height() // 2))
            
        drag.exec(Qt.DropAction.CopyAction)


class FilmStripBar(QFrame):
    """Horizontal virtualized strip with gallery-style justified spacing."""

    selection_changed = pyqtSignal(int)
    committed = pyqtSignal(int)
    thumbnails_needed = pyqtSignal(list)

    def __init__(self, parent=None, viewer=None):
        super().__init__(parent)
        self._viewer = viewer
        self.setObjectName("filmstrip_bar")
        self.setStyleSheet(
            """
            QFrame#filmstrip_bar {
                background-color: rgba(20, 18, 15, 230);
                border-top: 1px solid #3A332A;
            }
            QLabel#filmstrip_cell {
                background-color: transparent;
                border: 2px solid transparent;
                border-radius: 3px;
                color: #888;
                font-size: 10px;
                padding: 0px;
            }
            QLabel#filmstrip_cell_selected {
                background-color: rgba(51, 51, 51, 120);
                border: 2px solid #d9691e;
                border-radius: 3px;
                padding: 0px;
            }
            QLabel#filmstrip_batch_badge {
                color: #D9A441;
                font-size: 11px;
                font-weight: 700;
                background: transparent;
                border: none;
                padding: 0px;
            }
            """
        )
        self._files: List[str] = []
        self._metadata_cache: Dict[str, dict] = {}
        self._aspects: List[float] = []
        self._starts: List[int] = []
        self._widths: List[int] = []
        self._inner_width = 0
        self._row_offset = 0
        self._current_index = -1
        self._pending_index = -1
        self._cells: Dict[int, QLabel] = {}
        self._cell_pool: List[QLabel] = []
        self._active_paths: Set[str] = set()
        self._thumb_gen: Dict[str, int] = {}
        self._path_to_index: Dict[str, int] = {}
        self._generation = 0
        self._bookmarked_norm: Set[str] = set()
        # Measured slot widths from decoded/scaled thumbs (survives layout rebuilds).
        self._measured_widths: Dict[str, int] = {}

        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._scroll = QScrollArea(self)
        self._scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._scroll.setWidgetResizable(False)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setStyleSheet("background: transparent;")
        self._scroll.viewport().setStyleSheet("background: transparent;")

        self._content = QWidget(self._scroll)
        self._content.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._content.setStyleSheet("background: transparent;")
        self._scroll.setWidget(self._content)

        layout_outer = __import__("PyQt6.QtWidgets", fromlist=["QVBoxLayout"]).QVBoxLayout(self)
        layout_outer.setContentsMargins(0, 2, 0, 2)
        layout_outer.addWidget(self._scroll)

        self._scroll.horizontalScrollBar().valueChanged.connect(self._on_scroll)
        self._reload_timer = QTimer(self)
        self._reload_timer.setSingleShot(True)
        self._reload_timer.timeout.connect(self._reload_visible)

        self._scroll_refresh_timer = QTimer(self)
        self._scroll_refresh_timer.setSingleShot(True)
        self._scroll_refresh_timer.timeout.connect(self._on_scroll_settled)

        self._periodic_refresh_timer = QTimer(self)
        self._periodic_refresh_timer.setInterval(2000)
        self._periodic_refresh_timer.timeout.connect(self._periodic_refresh)

        self.setMouseTracking(True)
        self._scroll.setMouseTracking(True)
        self._content.setMouseTracking(True)

    def sizeHint(self):
        return QSize(-1, ROW_H + 4)

    def showEvent(self, event):
        super().showEvent(event)
        self._periodic_refresh_timer.start()
        QTimer.singleShot(0, lambda: self.refresh_visible_thumbnails(refresh_cache=True))

    def hideEvent(self, event):
        self._periodic_refresh_timer.stop()
        super().hideEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scroll.setGeometry(0, 0, self.width(), self.height())
        self._rebuild_layout_slots()
        QTimer.singleShot(0, self._reload_visible)

    def current_files(self) -> tuple:
        return tuple(self._files)

    def set_files(
        self,
        files: List[str],
        bulk_metadata: Optional[dict] = None,
        *,
        select_index: Optional[int] = None,
    ) -> None:
        self._generation += 1
        # Recycle every live cell — do not just clear the dict. Orphaned QLabel
        # widgets would stay visible with stale thumbnails/selection styling.
        for idx in list(self._cells.keys()):
            self._recycle_cell(idx)
        self._files = list(files) if files else []
        self._active_paths.clear()
        self._thumb_gen.clear()
        self._measured_widths.clear()
        self._rebuild_path_index()
        if bulk_metadata:
            self._metadata_cache.update(bulk_metadata)
        self._fetch_missing_metadata()
        if self._files:
            if select_index is not None and select_index >= 0:
                idx = max(0, min(select_index, len(self._files) - 1))
            else:
                idx = 0
        else:
            idx = -1
        if len(self._files) > _LAZY_WARM_THRESHOLD and idx >= 0:
            self._warm_slot_widths_from_cache(center_index=idx)
        else:
            self._warm_slot_widths_from_cache()
        self._rebuild_layout_slots()
        if self._files:
            self._current_index = idx
            self._pending_index = idx
        else:
            self._current_index = -1
            self._pending_index = -1
        QTimer.singleShot(0, self._scroll_to_index_centered)
        QTimer.singleShot(0, self._reload_visible)

    def set_current_index(self, index: int, center: bool = True) -> None:
        if not self._files:
            return
        index = max(0, min(index, len(self._files) - 1))
        if index == self._current_index and index == self._pending_index:
            self._update_selection_style()
            if center:
                self._scroll_to_index_centered()
            return
        self._current_index = index
        self._pending_index = index
        self._update_selection_style()
        if center:
            self._scroll_to_index_centered()
        self.refresh_visible_thumbnails(refresh_cache=True)

    def center_on_current(self) -> None:
        """Scrolls the filmstrip to center the currently selected image."""
        self._scroll_to_index_centered()

    def refresh_visible_thumbnails(
        self, force: bool = False, refresh_cache: bool = True
    ) -> None:
        """Rebind visible cells and optionally reload pixmaps from cache."""
        if force:
            for cell in list(self._cells.values()):
                self._clear_cell_thumbnail(cell)
            for cell in self._cell_pool:
                self._clear_cell_thumbnail(cell)
        elif refresh_cache:
            before = len(self._measured_widths)
            center = self._current_index if self._current_index >= 0 else self._pending_index
            if len(self._files) > _LAZY_WARM_THRESHOLD and center >= 0:
                self._warm_slot_widths_from_cache(center_index=center)
            else:
                self._warm_slot_widths_from_cache()
            if len(self._measured_widths) != before:
                self._rebuild_layout_slots()
            for idx, cell in list(self._cells.items()):
                if 0 <= idx < len(self._files):
                    path = self._files[idx]
                    if getattr(cell, "_filmstrip_path", None) == path:
                        self._try_apply_cache_for_cell(idx, path, cell)
        self._reload_visible()

    def invalidate_thumbnails_for_path(self, file_path: str) -> None:
        """Drop the cached cell pixmap for a path (e.g. after a saved/reset edit).

        refresh_visible_thumbnails() keeps a bound cell's existing pixmap when
        the global cache misses, so after image_cache.invalidate_file() the
        strip kept showing the stale edited render until the cell was recycled.
        Clearing the cell forces _reload_visible's cache-miss path, which
        requests a fresh async decode.
        """
        if not file_path:
            return
        self._measured_widths.pop(_path_key(file_path), None)
        idx = self._index_for_path(file_path)
        if idx >= 0:
            cell = self._cells.get(idx)
            if cell is not None:
                self._clear_cell_thumbnail(cell)
                cell.setText("…")
        self._reload_visible()

    def current_index(self) -> int:
        return self._pending_index if self._pending_index >= 0 else self._current_index

    def wants_thumbnail(self, file_path: str) -> bool:
        return file_path in self._active_paths

    def paths_with_displayed_thumbnails(self) -> List[str]:
        out: List[str] = []
        for idx, cell in self._cells.items():
            if idx < 0 or idx >= len(self._files):
                continue
            px = cell.pixmap()
            if px is not None and not px.isNull():
                out.append(self._files[idx])
        return out

    def note_cached_thumbnail(self, file_path: str, thumbnail) -> None:
        """Record slot width when a thumbnail lands in cache before the cell is visible."""
        if not file_path or file_path not in self._files:
            return
        key = _path_key(file_path)
        if key in self._measured_widths:
            return
        pixmap = _thumbnail_to_pixmap(thumbnail)
        if pixmap is None or pixmap.isNull():
            return
        rot = self._get_rotation_degrees_for_path(file_path)
        if rot:
            pixmap = self._rotate_pixmap(pixmap, rot)
        scaled = self._scale_pixmap_to_content(pixmap)
        if scaled is None or scaled.isNull():
            return
        width = self._cell_width_for_scaled(scaled)
        self._measured_widths[key] = width
        idx = self._index_for_path(file_path)
        if idx >= 0 and idx < len(self._widths) and self._widths[idx] != width:
            self._widths[idx] = width
            self._recompute_slot_starts()

    def apply_thumbnail(self, file_path: str, thumbnail) -> None:
        if file_path not in self._active_paths:
            return
        gen = self._thumb_gen.get(file_path, -1)
        if gen != self._generation:
            return
        idx = self._index_for_path(file_path)
        if idx < 0:
            return
        cell = self._cells.get(idx)
        if cell is None:
            pixmap = _thumbnail_to_pixmap(thumbnail)
            if pixmap is None or pixmap.isNull():
                return
            rot = self._get_rotation_degrees_for_path(file_path)
            if rot:
                pixmap = self._rotate_pixmap(pixmap, rot)
            scaled = self._scale_pixmap_to_content(pixmap)
            if scaled is None or scaled.isNull():
                return
            width = self._cell_width_for_scaled(scaled)
            self._measured_widths[_path_key(file_path)] = width
            if idx < len(self._widths) and self._widths[idx] != width:
                self._widths[idx] = width
                self._recompute_slot_starts()
            return
        if _path_key(getattr(cell, "_filmstrip_path", "")) != _path_key(file_path):
            return
        self._apply_pixmap_to_cell(idx, file_path, thumbnail)

    def refresh_cell_for_path(self, file_path: str) -> None:
        """Redraw one cell after non-destructive visual rotation changes."""
        idx = self._index_for_path(file_path)
        if idx < 0 or idx >= len(self._files):
            return
        path = self._files[idx]
        self._measured_widths.pop(_path_key(path), None)
        cell = self._cells.get(idx)
        if cell is not None:
            self._clear_cell_thumbnail(cell)
            cell._filmstrip_path = path
            if self._try_apply_cache_for_cell(idx, path, cell):
                self._recompute_slot_starts()
                return
            cell.setText("…")
        else:
            scaled = self._scaled_thumb_for_path(path)
            if scaled is not None and not scaled.isNull():
                self._record_cell_width(path, idx, self._cell_width_for_scaled(scaled))
                self._recompute_slot_starts()
        self._reload_visible()

    def on_visual_rotation_changed(self, file_path: str) -> None:
        """Update slot geometry and thumbnail after single-view rotation."""
        if not file_path:
            return
        self.refresh_cell_for_path(file_path)

    def _index_for_path(self, file_path: str) -> int:
        return self._path_to_index.get(_path_key(file_path), -1)

    def _rebuild_path_index(self) -> None:
        self._path_to_index = {_path_key(p): i for i, p in enumerate(self._files)}

    def _clear_cell_thumbnail(self, cell: QLabel) -> None:
        cell.setPixmap(QPixmap())
        cell.setText("")
        cell._filmstrip_path = None

    def _content_height(self) -> int:
        """Drawable thumb height inside a row cell (border + vertical padding)."""
        return max(12, ROW_H - 2 * CELL_BORDER - 2 * THUMB_PAD_V)

    def _max_content_width(self) -> int:
        return max(12, MAX_CELL_W - 2 * CELL_BORDER - 2 * THUMB_PAD_H)

    def _scale_pixmap_to_content(self, pixmap: QPixmap) -> QPixmap:
        """Scale to row height first so width follows the image aspect ratio."""
        if pixmap is None or pixmap.isNull():
            return pixmap
        content_h = self._content_height()
        max_cw = self._max_content_width()
        scaled = pixmap.scaledToHeight(
            content_h, Qt.TransformationMode.SmoothTransformation
        )
        if scaled.width() > max_cw:
            scaled = pixmap.scaled(
                max_cw,
                content_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        return scaled

    def _cell_width_for_scaled(self, scaled: QPixmap) -> int:
        if scaled is None or scaled.isNull():
            return MIN_CELL_W
        # Horizontal room is only for the selection border; no extra fixed box padding.
        return scaled.width() + 2 * CELL_BORDER

    def _scaled_thumb_for_path(self, path: str) -> Optional[QPixmap]:
        """Build the display-sized pixmap for a path when it is not on-screen yet."""
        viewer = self._viewer
        if viewer is None or not hasattr(viewer, "image_cache"):
            return None
        try:
            # FAST IN-MEMORY CHECK ONLY to avoid blocking UI thread on thousands of files!
            # Grid tier (~512px) is now the film strip's authoritative source, same as
            # gallery tiles; 256px thumbnail tier is only a fallback.
            thumb = viewer.image_cache.grid_cache.get(path)
            if thumb is None:
                thumb = viewer.image_cache.thumbnail_cache.get(path)
            if thumb is None:
                return None
        except Exception:
            return None
        pixmap = _thumbnail_to_pixmap(thumb)
        if pixmap is None or pixmap.isNull():
            return None
        rot = self._get_rotation_degrees_for_path(path)
        if rot:
            pixmap = self._rotate_pixmap(pixmap, rot)
        return self._scale_pixmap_to_content(pixmap)

    def _record_cell_width(self, path: str, idx: int, width: int) -> None:
        if not path or idx < 0 or idx >= len(self._widths):
            return
        width = max(MIN_CELL_W, min(MAX_CELL_W, int(width)))
        self._measured_widths[_path_key(path)] = width
        if self._widths[idx] == width:
            return
        self._widths[idx] = width
        self._recompute_slot_starts()

    def _slot_width_for_path(self, path: str, idx: int) -> int:
        """Resolve layout slot width; never reserve a wide box from wrong EXIF."""
        key = _path_key(path)
        cached = self._measured_widths.get(key)
        if cached is not None:
            return cached

        cell = self._cells.get(idx)
        if cell is not None:
            px = cell.pixmap()
            if px is not None and not px.isNull():
                return self._cell_width_for_scaled(px)

        scaled = self._scaled_thumb_for_path(path)
        if scaled is not None and not scaled.isNull():
            width = self._cell_width_for_scaled(scaled)
            self._measured_widths[key] = width
            return width

        # Unknown until decode: keep a narrow placeholder instead of a landscape EXIF guess.
        return MIN_CELL_W

    def _get_rotation_degrees_for_path(self, file_path: str) -> int:
        viewer = self._viewer
        if viewer is None:
            return 0
        getter = getattr(viewer, "_get_visual_rotation_degrees", None)
        if getter is None:
            return 0
        try:
            return int(getter(file_path)) % 360
        except Exception:
            return 0

    def _display_aspect(self, file_path: str, base_aspect: float) -> float:
        if base_aspect <= 0:
            return base_aspect
        if self._get_rotation_degrees_for_path(file_path) in (90, 270):
            return 1.0 / base_aspect
        return base_aspect

    def _rotate_pixmap(self, pixmap: QPixmap, degrees: int) -> QPixmap:
        degrees = int(degrees) % 360
        if pixmap is None or pixmap.isNull() or degrees == 0:
            return pixmap
        transform = QTransform()
        transform.rotate(degrees)
        return pixmap.transformed(transform, Qt.TransformationMode.SmoothTransformation)

    def _apply_pixmap_to_cell(self, idx: int, file_path: str, thumbnail) -> None:
        pixmap = _thumbnail_to_pixmap(thumbnail)
        if pixmap is None or pixmap.isNull():
            return
        cell = self._cells.get(idx)
        if cell is None:
            return
        rot = self._get_rotation_degrees_for_path(file_path)
        if rot:
            pixmap = self._rotate_pixmap(pixmap, rot)
        scaled = self._scale_pixmap_to_content(pixmap)
        if scaled is None or scaled.isNull() or scaled.height() <= 0:
            return
        cell.setPixmap(scaled)
        cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cell.setText("")
        cell._filmstrip_path = file_path

        new_aspect = scaled.width() / scaled.height()
        if idx < len(self._aspects):
            old = self._aspects[idx]
            if abs(new_aspect - old) > ASPECT_REBUILD_EPS:
                self._aspects[idx] = new_aspect
        width = self._cell_width_for_scaled(scaled)
        cell.setFixedSize(width, ROW_H)
        self._record_cell_width(file_path, idx, width)

    def _try_apply_cache_for_cell(self, idx: int, path: str, cell: QLabel) -> bool:
        viewer = self._viewer
        if viewer is None or not hasattr(viewer, "image_cache"):
            return False
        try:
            # In-memory tiers only: get_grid()/get_thumbnail() fall through to the
            # disk cache (SQLite read + JPEG decode) and this runs on the UI thread
            # for every visible cell on each navigation. Disk-tier misses return
            # False so the caller falls back to the async thumbnail load path.
            thumb = viewer.image_cache.get_grid_memory_only(path)
            if thumb is None:
                thumb = viewer.image_cache.get_thumbnail_memory_only(path)
        except Exception:
            return False
        if thumb is None:
            return False
        self._apply_pixmap_to_cell(idx, path, thumb)
        return True

    def _prepare_cell_for_path(self, cell: QLabel, idx: int, path: str) -> bool:
        """Bind cell to path; return True if async thumbnail load is still needed."""
        bound = getattr(cell, "_filmstrip_path", None)
        if bound != path:
            self._clear_cell_thumbnail(cell)
            cell._filmstrip_path = path
            if self._try_apply_cache_for_cell(idx, path, cell):
                return False
            cell.setText("…")
            return True
        if cell.pixmap() is not None and not cell.pixmap().isNull():
            width = self._cell_width_for_scaled(cell.pixmap())
            cell.setFixedSize(width, ROW_H)
            self._record_cell_width(path, idx, width)
            return False
        if self._try_apply_cache_for_cell(idx, path, cell):
            return False
        if cell.text() != "…":
            cell.setText("…")
        return True

    def move_selection(self, delta: int) -> None:
        if not self._files:
            return
        idx = self.current_index()
        if idx < 0:
            idx = 0
        idx = (idx + delta) % len(self._files)
        self._pending_index = idx
        self._update_selection_style()
        self._scroll_to_index_centered()
        self.selection_changed.emit(idx)
        self.commit_selection()

    def commit_selection(self) -> None:
        if not self._files or self._pending_index < 0:
            return
        self._current_index = self._pending_index
        self.committed.emit(self._current_index)

    def scroll_by_steps(self, steps: int) -> None:
        sb = self._scroll.horizontalScrollBar()
        sb.setValue(sb.value() - steps * max(80, ROW_H + MIN_GAP))

    def _warm_slot_widths_from_cache(self, *, center_index: Optional[int] = None) -> None:
        """Pre-measure slot widths for any thumbnails already in the global cache."""
        files = self._files
        if not files:
            return
        if center_index is not None and len(files) > _LAZY_WARM_THRESHOLD:
            idx = max(0, min(center_index, len(files) - 1))
            lo = max(0, idx - _WARM_BAND)
            hi = min(len(files), idx + _WARM_BAND + 1)
            paths = files[lo:hi]
        else:
            paths = files
        for path in paths:
            key = _path_key(path)
            if key in self._measured_widths:
                continue
            scaled = self._scaled_thumb_for_path(path)
            if scaled is not None and not scaled.isNull():
                self._measured_widths[key] = self._cell_width_for_scaled(scaled)

    def _fetch_missing_metadata(self) -> None:
        """Disabled to prevent expensive synchronous SQLite queries on the UI thread."""
        pass

    def _recompute_slot_starts(self) -> None:
        """Repack x positions after per-cell width tweaks (keep _widths/_aspects)."""
        n = len(self._files)
        if n == 0 or len(self._widths) != n:
            self._rebuild_layout_slots()
            return
        inner = SIDE_MARGIN * 2 + sum(self._widths) + (n - 1) * MIN_GAP
        vp_w = max(1, self._scroll.viewport().width())
        content_w = max(vp_w, inner)
        self._row_offset = max(0, (content_w - inner) // 2)
        x = self._row_offset + SIDE_MARGIN
        self._starts = []
        for w in self._widths:
            self._starts.append(x)
            x += w + MIN_GAP
        self._inner_width = inner
        h = max(ROW_H + 2, self._scroll.viewport().height())
        self._content.setFixedSize(max(content_w, 1), max(h, 1))
        for idx, cell in list(self._cells.items()):
            if idx < len(self._starts):
                self._place_cell_geometry(cell, idx)

    def _layout_width_for_index(self, path: str, idx: int) -> tuple[float, int]:
        """Return (aspect, cell_width) for layout packing."""
        width = self._slot_width_for_path(path, idx)
        ch = self._content_height()
        aspect = width / ch if ch > 0 else DEFAULT_ASPECT
        return aspect, width

    def _rebuild_layout_slots(self) -> None:
        n = len(self._files)
        self._starts = []
        self._widths = []
        self._aspects = []

        if n == 0:
            self._inner_width = 0
            self._row_offset = 0
            vp_w = max(1, self._scroll.viewport().width())
            self._content.setFixedSize(vp_w, max(ROW_H + 2, self._scroll.viewport().height()))
            return

        aspects = []
        widths = []
        for i, p in enumerate(self._files):
            aspect, width = self._layout_width_for_index(p, i)
            aspects.append(aspect)
            widths.append(width)

        inner = SIDE_MARGIN * 2 + sum(widths) + (n - 1) * MIN_GAP
        vp_w = max(1, self._scroll.viewport().width())
        content_w = max(vp_w, inner)
        self._row_offset = max(0, (content_w - inner) // 2)

        x = self._row_offset + SIDE_MARGIN
        for w in widths:
            self._starts.append(x)
            self._widths.append(w)
            x += w + MIN_GAP

        self._aspects = aspects
        self._inner_width = inner
        h = max(ROW_H + 2, self._scroll.viewport().height())
        self._content.setFixedSize(max(content_w, 1), max(h, 1))

        for idx, cell in list(self._cells.items()):
            if idx < len(self._starts):
                self._place_cell_geometry(cell, idx)

    def _place_cell_geometry(self, cell: QLabel, idx: int) -> None:
        if idx < 0 or idx >= len(self._starts):
            return
        w = self._widths[idx]
        x = self._starts[idx]
        y = max(0, (self._content.height() - ROW_H) // 2)
        cell.setFixedSize(w, ROW_H)
        cell.setGeometry(x, y, w, ROW_H)
        self._position_bookmark_badge(cell)

    def _on_scroll(self, _value: int) -> None:
        self._reload_timer.start(16)
        self._scroll_refresh_timer.start(250)

    def _on_scroll_settled(self) -> None:
        self.refresh_visible_thumbnails(refresh_cache=True)

    def _periodic_refresh(self) -> None:
        if self.isVisible() and self.isEnabled():
            self.refresh_visible_thumbnails(refresh_cache=True)

    def _index_at_content_x(self, x: float) -> int:
        if not self._starts:
            return -1
        i = bisect.bisect_right(self._starts, x) - 1
        return max(0, min(i, len(self._starts) - 1))

    def _visible_index_range(self) -> tuple:
        if not self._files or not self._starts:
            return 0, -1
        scroll_x = self._scroll.horizontalScrollBar().value()
        vp_w = max(1, self._scroll.viewport().width())
        x0 = scroll_x - BUFFER_PX
        x1 = scroll_x + vp_w + BUFFER_PX
        first = self._index_at_content_x(x0)
        last = self._index_at_content_x(x1)
        first = max(0, first - BUFFER_CELLS)
        last = min(len(self._files) - 1, last + BUFFER_CELLS)
        return first, last

    def _recycle_cell(self, index: int) -> None:
        cell = self._cells.pop(index, None)
        if cell is not None:
            path = self._files[index] if 0 <= index < len(self._files) else None
            if not path:
                path = getattr(cell, "_filmstrip_path", None)
            if path:
                self._active_paths.discard(path)
            self._clear_cell_thumbnail(cell)
            badge = getattr(cell, "_bookmark_badge", None)
            if badge is not None:
                badge.hide()
            cell.setObjectName("filmstrip_cell")
            cell.setStyleSheet("")
            cell.hide()
            self._cell_pool.append(cell)

    def _acquire_cell(self, index: int) -> QLabel:
        cell = self._cells.get(index)
        if cell is not None:
            return cell
        if self._cell_pool:
            cell = self._cell_pool.pop()
            self._clear_cell_thumbnail(cell)
            cell.setObjectName("filmstrip_cell")
            cell.setStyleSheet("")
        else:
            cell = FilmstripCell(self._content)
            cell.setObjectName("filmstrip_cell")
            cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cell.setScaledContents(False)
        self._cells[index] = cell
        return cell

    def _reload_visible(self) -> None:
        if not self._files:
            for idx in list(self._cells.keys()):
                self._recycle_cell(idx)
            return

        first, last = self._visible_index_range()
        if last < first:
            for idx in list(self._cells.keys()):
                self._recycle_cell(idx)
            return

        visible_set = set(range(first, last + 1))
        for idx in list(self._cells.keys()):
            if idx not in visible_set:
                self._recycle_cell(idx)

        needed_paths: List[str] = []
        relayout = False
        for idx in range(first, last + 1):
            path = self._files[idx]
            cell = self._acquire_cell(idx)
            cell.index = idx
            cell.file_path = path
            try:
                cell.clicked.disconnect()
            except Exception:
                pass
            cell.clicked.connect(self._on_cell_clicked)
            self._active_paths.add(path)
            self._thumb_gen[path] = self._generation
            if self._prepare_cell_for_path(cell, idx, path):
                needed_paths.append(path)
            # Width may have been measured from cache while preparing the cell.
            slot_w = self._slot_width_for_path(path, idx)
            if idx < len(self._widths) and self._widths[idx] != slot_w:
                self._widths[idx] = slot_w
                relayout = True
            self._place_cell_geometry(cell, idx)
            cell.show()
            cell.raise_()

        if relayout:
            self._recompute_slot_starts()

        self._update_selection_style()
        if needed_paths:
            self.thumbnails_needed.emit(needed_paths)

    def _on_cell_clicked(self, idx: int, event) -> None:
        self._pending_index = idx
        self._update_selection_style()
        self.committed.emit(idx)

    def _position_bookmark_badge(self, cell: QLabel) -> None:
        badge = getattr(cell, "_bookmark_badge", None)
        if badge is None or not badge.isVisible():
            return
        size = 14
        margin = 2
        badge.setFixedSize(size, size)
        badge.move(
            max(0, cell.width() - size - margin),
            max(0, cell.height() - size - margin),
        )
        badge.raise_()

    def _set_cell_bookmarked(self, cell: QLabel, bookmarked: bool) -> None:
        badge = getattr(cell, "_bookmark_badge", None)
        if bookmarked:
            if badge is None:
                badge = QLabel("★", cell)
                badge.setObjectName("filmstrip_batch_badge")
                badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cell._bookmark_badge = badge
            badge.show()
            self._position_bookmark_badge(cell)
        elif badge is not None:
            badge.hide()

    def set_bookmarked_norm_paths(self, norm_paths: Set[str]) -> None:
        """Show star badges on filmstrip cells that are bookmarked (norm paths)."""
        self._bookmarked_norm = set(norm_paths or set())
        self._update_selection_style()

    @staticmethod
    def _norm_path(path: str) -> str:
        return os.path.normcase(os.path.normpath(path or ""))

    def _update_selection_style(self) -> None:
        sel = self._pending_index
        for idx, cell in self._cells.items():
            path = self._files[idx] if 0 <= idx < len(self._files) else ""
            bookmarked = bool(
                path
                and self._bookmarked_norm
                and self._norm_path(path) in self._bookmarked_norm
            )
            if idx == sel:
                cell.setObjectName("filmstrip_cell_selected")
            else:
                cell.setObjectName("filmstrip_cell")
            cell.setStyleSheet("")
            cell.style().unpolish(cell)
            cell.style().polish(cell)
            self._set_cell_bookmarked(cell, bookmarked)

    def _scroll_to_index_centered(self) -> None:
        if not self._files or self._pending_index < 0 or not self._starts:
            return
        idx = self._pending_index
        if idx >= len(self._starts):
            return
        center_x = self._starts[idx] + self._widths[idx] // 2
        vp_w = max(1, self._scroll.viewport().width())
        target = center_x - vp_w // 2
        sb = self._scroll.horizontalScrollBar()
        sb.setValue(max(sb.minimum(), min(target, sb.maximum())))
        self._reload_timer.start(0)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y() or event.angleDelta().x()
        if delta:
            steps = delta // 120 if abs(delta) >= 120 else (1 if delta > 0 else -1)
            self.scroll_by_steps(steps)
            event.accept()
            return
        super().wheelEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key.Key_Left:
            self.move_selection(-1)
            event.accept()
            return
        if key == Qt.Key.Key_Right:
            self.move_selection(1)
            event.accept()
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.commit_selection()
            event.accept()
            return
        super().keyPressEvent(event)

    def enterEvent(self, event):
        super().enterEvent(event)
        parent = self.parent()
        while parent is not None:
            setter = getattr(parent, "set_filmstrip_pointer_active", None)
            if callable(setter):
                setter(True)
                break
            parent = parent.parent()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        parent = self.parent()
        while parent is not None:
            setter = getattr(parent, "set_filmstrip_pointer_active", None)
            if callable(setter):
                setter(False)
                break
            parent = parent.parent()
