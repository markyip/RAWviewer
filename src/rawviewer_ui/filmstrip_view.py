"""Bottom film-strip overlay for single-image view (virtualized justified thumbnails)."""

from __future__ import annotations

import bisect
import os
from typing import Dict, List, Optional, Set

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QTimer
from PyQt6.QtGui import QPixmap, QImage, QWheelEvent, QKeyEvent
from PyQt6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QWidget,
)

# Single justified row (same idea as gallery row: width = row_height * aspect)
ROW_H = 66
MIN_GAP = 2
SIDE_MARGIN = 4
MIN_CELL_W = 34
MAX_CELL_W = 140
DEFAULT_ASPECT = 1.5
BUFFER_PX = 120
BUFFER_CELLS = 2


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
        return QPixmap.fromImage(thumbnail) if not thumbnail.isNull() else None
    try:
        arr = np.ascontiguousarray(thumbnail)
        h, w = arr.shape[:2]
        if h <= 0 or w <= 0:
            return None
        channels = arr.shape[2] if len(arr.shape) > 2 else 1
        if channels == 1:
            qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        else:
            bpl = 3 * w
            qimg = QImage(arr.data.tobytes(), w, h, bpl, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)
    except Exception:
        return None


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
                background-color: rgba(24, 24, 24, 230);
                border-top: 1px solid #3A3A3A;
            }
            QLabel#filmstrip_cell {
                background-color: #2A2A2A;
                border: 1px solid transparent;
                border-radius: 2px;
                color: #888;
                font-size: 10px;
            }
            QLabel#filmstrip_cell_selected {
                background-color: #333333;
                border: 1px solid #6EA8D6;
                border-radius: 2px;
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
        return QSize(-1, ROW_H + 8)

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
        self._rebuild_path_index()
        if bulk_metadata:
            self._metadata_cache.update(bulk_metadata)
        self._fetch_missing_metadata()
        self._rebuild_layout_slots()
        if self._files:
            if select_index is not None and select_index >= 0:
                idx = max(0, min(select_index, len(self._files) - 1))
            else:
                idx = 0
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
            for idx, cell in list(self._cells.items()):
                if 0 <= idx < len(self._files):
                    path = self._files[idx]
                    if getattr(cell, "_filmstrip_path", None) == path:
                        self._try_apply_cache_for_cell(idx, path, cell)
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
            return
        if _path_key(getattr(cell, "_filmstrip_path", "")) != _path_key(file_path):
            return
        self._apply_pixmap_to_cell(idx, file_path, thumbnail)

    def _index_for_path(self, file_path: str) -> int:
        return self._path_to_index.get(_path_key(file_path), -1)

    def _rebuild_path_index(self) -> None:
        self._path_to_index = {_path_key(p): i for i, p in enumerate(self._files)}

    def _clear_cell_thumbnail(self, cell: QLabel) -> None:
        cell.setPixmap(QPixmap())
        cell.setText("")
        cell._filmstrip_path = None

    def _apply_pixmap_to_cell(self, idx: int, file_path: str, thumbnail) -> None:
        pixmap = _thumbnail_to_pixmap(thumbnail)
        if pixmap is None or pixmap.isNull():
            return
        cell = self._cells.get(idx)
        if cell is None:
            return
        if pixmap.height() > 0:
            new_aspect = pixmap.width() / pixmap.height()
            old = self._aspects[idx] if idx < len(self._aspects) else DEFAULT_ASPECT
            if abs(new_aspect - old) > 0.06:
                self._aspects[idx] = new_aspect
                self._rebuild_layout_slots()
        slot_w = self._widths[idx] if idx < len(self._widths) else MIN_CELL_W
        scaled = pixmap.scaled(
            slot_w,
            ROW_H,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        cell.setPixmap(scaled)
        cell.setText("")
        cell._filmstrip_path = file_path

    def _try_apply_cache_for_cell(self, idx: int, path: str, cell: QLabel) -> bool:
        viewer = self._viewer
        if viewer is None or not hasattr(viewer, "image_cache"):
            return False
        try:
            thumb = viewer.image_cache.get_thumbnail(path)
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

    def _fetch_missing_metadata(self) -> None:
        viewer = self._viewer
        if viewer is None or not hasattr(viewer, "image_cache"):
            return
        missing = [p for p in self._files if p not in self._metadata_cache]
        if not missing:
            return
        try:
            bulk = viewer.image_cache.get_multiple_exif(missing)
            if bulk:
                self._metadata_cache.update(bulk)
        except Exception:
            pass

    def _aspect_for_index(self, path: str, idx: int) -> float:
        cell = self._cells.get(idx)
        if cell is not None:
            px = cell.pixmap()
            if px is not None and not px.isNull() and px.height() > 0:
                return px.width() / px.height()

        meta = self._metadata_cache.get(path)
        if meta and meta.get("original_width") and meta.get("original_height"):
            w, h = meta["original_width"], meta["original_height"]
            if meta.get("orientation", 1) in (5, 6, 7, 8):
                w, h = h, w
            if h > 0:
                return w / h

        try:
            from common_image_loader import get_image_aspect_ratio
            return get_image_aspect_ratio(path)
        except Exception:
            return DEFAULT_ASPECT

    def _cell_width_for_aspect(self, aspect: float) -> int:
        aspect = max(0.35, min(3.5, aspect))
        return max(MIN_CELL_W, min(MAX_CELL_W, int(ROW_H * aspect)))

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

        aspects = [self._aspect_for_index(p, i) for i, p in enumerate(self._files)]
        widths = [self._cell_width_for_aspect(a) for a in aspects]

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
        cell.setGeometry(x, y, w, ROW_H)

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
            cell = QLabel(self._content)
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
        for idx in range(first, last + 1):
            path = self._files[idx]
            cell = self._acquire_cell(idx)
            self._place_cell_geometry(cell, idx)
            cell.show()
            cell.raise_()
            self._wire_cell(cell, idx)
            self._active_paths.add(path)
            self._thumb_gen[path] = self._generation
            if self._prepare_cell_for_path(cell, idx, path):
                needed_paths.append(path)

        self._update_selection_style()
        if needed_paths:
            self.thumbnails_needed.emit(needed_paths)

    def _wire_cell(self, cell: QLabel, index: int) -> None:
        def _on_click(event, i=index):
            if event.button() == Qt.MouseButton.LeftButton:
                self._pending_index = i
                self._update_selection_style()
                self.committed.emit(i)
            event.accept()

        cell.mousePressEvent = _on_click

    def _update_selection_style(self) -> None:
        sel = self._pending_index
        for idx, cell in self._cells.items():
            if idx == sel:
                cell.setObjectName("filmstrip_cell_selected")
            else:
                cell.setObjectName("filmstrip_cell")
            cell.setStyleSheet("")
            cell.style().unpolish(cell)
            cell.style().polish(cell)

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
