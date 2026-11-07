"""Gallery mixin part 7 of 7 for JustifiedGallery."""
from __future__ import annotations

import logging
import time

from PyQt6.QtCore import QRect, Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QLabel

from rawviewer_ui.widgets import ThumbnailLabel

from .helpers import _focus_gallery_switch_logs, _orient_debug_enabled

logger = logging.getLogger(__name__)

class GalleryPart7:

    def _rebuild_layout_from_image_index(
        self, start_image_idx: int, anchor_state: dict | None = None
    ) -> bool:
        """Recompute justified rows from *start_image_idx* onward (partial layout update)."""
        if (
            start_image_idx <= 0
            or not self.images
            or not self._gallery_layout_items
            or self._building
        ):
            return False

        viewport_width = self._get_viewport_width()
        left_margin = 24
        right_margin = 0
        net_width = viewport_width - left_margin - right_margin
        if net_width <= 0:
            return False

        start_path = self.images[start_image_idx]
        if not isinstance(start_path, str):
            return False
        layout_indices = self._path_to_indices.get(start_path) or []
        if not layout_indices:
            return False
        row_start_layout = self._layout_row_start_index(min(layout_indices))
        row_path = self._gallery_layout_items[row_start_layout].get("file_path")
        rebuild_from = start_image_idx
        if isinstance(row_path, str):
            row_image_idx = self._image_index_for_path(row_path)
            if row_image_idx >= 0:
                rebuild_from = row_image_idx

        current_y = self._gallery_layout_items[row_start_layout]["rect"].top()
        # Prune path_to_indices entries for the discarded tail *before*
        # truncating -- bounded by the tail's length, not the total item
        # count, unlike rebuilding the whole map afterward (see below; was
        # measured at ~6ms per rebuild on a 30k-item gallery for a 20-item
        # tail change, vs ~0.005ms for this incremental approach -- and this
        # rebuild can fire repeatedly as thumbnails/EXIF stream in during
        # initial population of a large folder).
        for discarded in self._gallery_layout_items[row_start_layout:]:
            p = discarded.get("file_path")
            indices = self._path_to_indices.get(p)
            if indices is None:
                continue
            kept = [idx for idx in indices if idx < row_start_layout]
            if kept:
                self._path_to_indices[p] = kept
            else:
                del self._path_to_indices[p]
        self._gallery_layout_items = self._gallery_layout_items[:row_start_layout]

        row: list = []
        aspect_sum = 0.0

        def commit_row(r, a_sum, is_last):
            nonlocal current_y
            if not r:
                return
            spacing = (len(r) - 1) * self.MIN_SPACING
            row_h = self.TARGET_ROW_HEIGHT if is_last else (net_width - spacing) / a_sum
            row_h = max(
                self.TARGET_ROW_HEIGHT * 0.4,
                min(self.TARGET_ROW_HEIGHT * 2.2, row_h),
            )

            curr_x = left_margin
            for i, (item, aspect) in enumerate(r):
                w = int(row_h * aspect)
                if not is_last and i == len(r) - 1:
                    w = net_width - (curr_x - left_margin)
                p = item if isinstance(item, str) else None
                self._gallery_layout_items.append(
                    {
                        "rect": QRect(curr_x, int(current_y), int(w), int(row_h)),
                        "file_path": p,
                        "aspect": aspect,
                    }
                )
                self._path_to_indices.setdefault(p, []).append(
                    len(self._gallery_layout_items) - 1
                )
                curr_x += w + self.MIN_SPACING
            current_y += row_h + self.MIN_SPACING

        for item in self.images[rebuild_from:]:
            aspect = 1.5
            if isinstance(item, str):
                aspect = self._layout_aspect_for_path(item)
            row.append((item, aspect))
            aspect_sum += aspect
            if (aspect_sum * self.TARGET_ROW_HEIGHT + (len(row) - 1) * self.MIN_SPACING) >= net_width:
                commit_row(row, aspect_sum, False)
                row, aspect_sum = [], 0.0

        if row:
            commit_row(row, aspect_sum, True)

        # path_to_indices is maintained incrementally above (pruned for the
        # discarded tail before truncation, appended to inside commit_row) --
        # no full rebuild needed here.

        self._total_content_height = int(current_y + 20)
        self._sync_content_geometry()
        scroll = self._scroll_area_widget()
        if scroll is not None:
            sb = scroll.verticalScrollBar()
            if sb.value() > sb.maximum():
                sb.setValue(max(sb.minimum(), sb.maximum()))
        self.update()
        self._last_build_ts = time.time()
        self._last_layout_content_height = self._total_content_height
        self._layout_image_sequence = tuple(self.images)
        self._last_force_build_ts = self._last_build_ts
        self._reposition_thumbnail_widgets()
        if anchor_state:
            self._apply_scroll_anchor_state(anchor_state)
        return True

    def _handle_metadata_rebuild(self):
        """Rebuild layout after metadata changes to settle aspect ratios."""
        large = len(self.images) > 800
        flip_count = len(getattr(self, "_orientation_flip_paths", None) or ())
        flip_pending = flip_count > 0
        if not self._metadata_changed_paths or self._building:
            return
        if self._is_scrolling_fast and not flip_pending:
            # Don't rebuild while scrolling as it blocks the UI thread
            if not self._metadata_rebuild_timer.isActive():
                self._metadata_rebuild_timer.start(1000)
            return
        time_since_last_change = time.time() - getattr(self, "_last_metadata_change_time", 0.0)
        if (
            large
            and not flip_pending
            and len(self._metadata_changed_paths) < 8
            and time_since_last_change < 1.5
        ):
            self._metadata_rebuild_timer.start(1200)
            return
        if not flip_pending and (time.time() - self._last_force_build_ts) < 0.8:
            self._metadata_rebuild_timer.start(400)
            return

        anchor = self._scroll_anchor_state_for_rebuild()
        changed_paths = set(self._metadata_changed_paths)
        flip_paths = set(getattr(self, "_orientation_flip_paths", None) or ())
        self._metadata_changed_paths.clear()
        if hasattr(self, "_orientation_flip_paths"):
            if _orient_debug_enabled() and flip_count:
                logger.info(
                    "[ORIENT] metadata rebuild starting (%d orientation flip(s))",
                    flip_count,
                )
            self._orientation_flip_paths.clear()
        if _focus_gallery_switch_logs():
            logger.debug("[GALLERY] metadata rebuild triggered")

        rebuild_t0 = time.time()
        partial_paths = flip_paths or changed_paths
        partial_start = self._partial_rebuild_start_image_index(partial_paths)
        use_partial = (
            partial_start is not None
            and partial_start > 0
            and len(partial_paths) <= max(64, len(self.images) // 50)
        )
        if use_partial:
            if not self._rebuild_layout_from_image_index(partial_start, anchor_state=anchor):
                self.build_gallery(bulk_metadata=None, force=True)
        else:
            self.build_gallery(bulk_metadata=None, force=True)
        rebuild_ms = (time.time() - rebuild_t0) * 1000.0
        if rebuild_ms > 250:
            # Main-thread rebuild cost is the prime suspect for periodic gallery stalls
            # on large folders — surface it so slow rebuilds are visible in logs.
            logger.info(
                "[GALLERY] metadata rebuild took %.0f ms (%d items)",
                rebuild_ms,
                len(self._gallery_layout_items),
            )

    def show_loading_message(self, message="Loading gallery..."):
        """Show loading message overlay - Simplified for better performance"""
        # Remove existing loading label if any
        if self._loading_label:
            # Update text if already visible
            self._loading_label.setText(message)
            self._loading_label.adjustSize()
            self._update_loading_label_geometry()
            return

        # Create loading label - smaller, bottom-right toast style
        self._loading_label = QLabel(message, self)
        self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._loading_label.setStyleSheet("""
            QLabel {
                background-color: rgba(20, 20, 20, 200);
                color: rgba(255, 255, 255, 220);
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 12px;
            }
        """)
        font = QFont()
        font.setPointSize(10)
        self._loading_label.setFont(font)
        self._loading_label.show()
        self._loading_label.raise_()  # Bring to front

        # Update geometry
        self._update_loading_label_geometry()

    def _update_loading_label_geometry(self):
        """Update loading label geometry - Bottom Center"""
        if self._loading_label and self.parent_viewer and self.parent_viewer.width() > 0:
            self._loading_label.adjustSize()
            w = self._loading_label.width()
            h = self._loading_label.height()

            # Position at bottom center of the viewport
            parent_scroll = self.parent_viewer.scroll_area if hasattr(self.parent_viewer, 'scroll_area') else None
            if parent_scroll:
                 # Calculate relative position in the viewport
                 viewport_h = parent_scroll.viewport().height()
                 scroll_y = parent_scroll.verticalScrollBar().value()

                 # Stick to bottom of viewport
                 y = scroll_y + viewport_h - h - 20
                 x = (self.width() - w) // 2

                 self._loading_label.move(x, int(y))
            else:
                 # Fallback
                 x = (self.width() - w) // 2
                 y = self.height() - h - 20
                 self._loading_label.move(x, y)

    def hide_loading_message(self):
        """Hide loading message overlay"""
        if self._loading_label:
            self._loading_label.hide()
            self._loading_label.deleteLater()
            self._loading_label = None

    def show_empty_message(self, message):
        """Show empty gallery message centered in the gallery frame."""
        self.hide_loading_message()

        style = """
            QLabel {
                color: #a8a8a8;
                font-size: 15px;
                background-color: transparent;
                padding: 24px;
            }
        """

        if hasattr(self, "_empty_label") and self._empty_label:
            self._empty_label.setText(message)
            self._empty_label.setAlignment(
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self._empty_label.setWordWrap(True)
            self._empty_label.setStyleSheet(style)
            self._update_empty_label_geometry()
            self._empty_label.show()
            self._empty_label.raise_()
            QTimer.singleShot(0, self._update_empty_label_geometry)
            return

        self._empty_label = QLabel(message, self)
        self._empty_label.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
        )
        self._empty_label.setWordWrap(True)
        self._empty_label.setStyleSheet(style)
        font = QFont()
        font.setPointSize(12)
        self._empty_label.setFont(font)
        self._empty_label.show()
        self._empty_label.raise_()
        self._update_empty_label_geometry()
        QTimer.singleShot(0, self._update_empty_label_geometry)

    def hide_empty_message(self):
        """Hide empty gallery message"""
        if hasattr(self, '_empty_label') and self._empty_label:
            self._empty_label.hide()
            self._empty_label.deleteLater()
            self._empty_label = None

    def clear_thumbnail_widgets(self, *, destroy: bool = False):
        """Hide visible thumbnails; pool widgets unless ``destroy`` (folder clear)."""
        try:
            if self.load_manager is not None:
                if hasattr(self.load_manager, "flush_queue"):
                    self.load_manager.flush_queue()
                else:
                    self.load_manager.cancel_all_tasks()
        except Exception:
            pass
        self._active_tasks.clear()
        self._requested_thumbnail_paths.clear()
        self._placeholder_thumb_paths.clear()

        for label in list(getattr(self, "_visible_widgets", {}).values()):
            try:
                label.hide()
                label.clear()
                label.setText("")
                label.file_path = None
                label.original_pixmap = None
                if hasattr(label, "set_gallery_selected"):
                    label.set_gallery_selected(False)
                if hasattr(label, "set_gallery_rating"):
                    label.set_gallery_rating(0)
                if destroy:
                    label.deleteLater()
                else:
                    self._widget_pool.append(label)
            except Exception:
                pass
        self._visible_widgets.clear()

        if destroy:
            for label in list(getattr(self, "_widget_pool", [])):
                try:
                    label.hide()
                    label.clear()
                    label.deleteLater()
                except Exception:
                    pass
            self._widget_pool = []
            try:
                for child in self.findChildren(ThumbnailLabel):
                    child.hide()
                    child.deleteLater()
            except Exception:
                pass

    def refresh_gallery_selection_visuals(self) -> None:
        """Sync selection border on visible pooled thumbnails."""
        pv = self.parent_viewer
        if pv is None or not hasattr(pv, "_is_gallery_path_selected"):
            return
        for w in self._visible_widgets.values():
            path = getattr(w, "file_path", None)
            if path and hasattr(w, "set_gallery_selected"):
                w.set_gallery_selected(pv._is_gallery_path_selected(path))

    def refresh_gallery_burst_visuals(self) -> None:
        """Sync burst stack count badges on visible thumbnails."""
        pv = self.parent_viewer
        if pv is None or not hasattr(pv, "_burst_stack_count_for_path"):
            return
        for w in self._visible_widgets.values():
            path = getattr(w, "file_path", None)
            if path and hasattr(w, "set_burst_stack_count"):
                w.set_burst_stack_count(pv._burst_stack_count_for_path(path))

    def refresh_gallery_edited_visuals(self) -> None:
        """Sync 'has saved RAW adjustments' badge on visible pooled thumbnails."""
        pv = self.parent_viewer
        if pv is None or not hasattr(pv, "_is_gallery_path_edited"):
            return
        for w in self._visible_widgets.values():
            path = getattr(w, "file_path", None)
            if path and hasattr(w, "set_gallery_edited"):
                w.set_gallery_edited(pv._is_gallery_path_edited(path))

    def refresh_gallery_rating_visuals(self) -> None:
        """Sync star rating badge on visible pooled thumbnails."""
        pv = self.parent_viewer
        if pv is None or not hasattr(pv, "_gallery_path_rating"):
            return
        for w in self._visible_widgets.values():
            path = getattr(w, "file_path", None)
            if path and hasattr(w, "set_gallery_rating"):
                w.set_gallery_rating(pv._gallery_path_rating(path))

    def _update_empty_label_geometry(self):
        """Lay out empty-state text across the justified gallery canvas (fills the frame when empty)."""
        if not getattr(self, "_empty_label", None):
            return
        margin_x = 32
        avail_w = max(120, int(self.width() - margin_x * 2))
        avail_h = max(120, self.height())

        label = self._empty_label
        label.setFixedWidth(avail_w)
        try:
            hint_h = int(label.heightForWidth(avail_w))
        except Exception:
            label.adjustSize()
            hint_h = int(label.sizeHint().height())
        label_h = min(max(hint_h, 48), avail_h - 16)
        y = max(8, (avail_h - label_h) // 2)
        x = margin_x
        label.setGeometry(int(x), int(y), avail_w, label_h)

    def _on_thumbnail_clicked(self, target_path, e):
        try:
            pv = self.parent_viewer
            if target_path and pv is not None:
                mods = (
                    pv._gallery_event_modifiers(e)
                    if hasattr(pv, "_gallery_event_modifiers")
                    else e.modifiers()
                )
                shift = (
                    hasattr(pv, "_gallery_has_shift_modifier")
                    and pv._gallery_has_shift_modifier(mods)
                )
                extend = (
                    hasattr(pv, "_gallery_has_extend_modifier")
                    and pv._gallery_has_extend_modifier(mods)
                )
                if shift and not extend:
                    if hasattr(pv, "_gallery_shift_click_selection"):
                        pv._gallery_shift_click_selection(target_path)
                    return
                if extend:
                    if hasattr(pv, "_gallery_toggle_path_selection"):
                        pv._gallery_toggle_path_selection(target_path)
                    return
                pv._gallery_item_clicked(target_path)
        except Exception:
            pass
