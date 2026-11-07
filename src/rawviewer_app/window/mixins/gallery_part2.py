"""gallery mixin, part 2 of 5 (17 methods, 872 lines)."""
from __future__ import annotations

import contextlib
import platform
import sys
from typing import Any, cast

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QHBoxLayout, QSizePolicy

from core import theme
from core.config import _norm_path
from rawviewer_app.signals import GalleryMetadataSignals

from .. import lazy
from ..lazy import ExternalJustifiedGallery, ThumbnailLabel
from ..qt_helpers import _get_bg_thread_pool

lazy.register(sys.modules[__name__])


class GalleryPart2:
    def _sync_gallery_scrollbar_policy(self):
        """Hide vertical scrollbar when gallery has nothing to scroll (e.g. no matches)."""
        gs = getattr(self, "gallery_scroll", None)
        if gs is None or getattr(self, "view_mode", "") != "gallery":
            return
        if not self.image_files:
            gs.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            gs.verticalScrollBar().setValue(gs.verticalScrollBar().minimum())
        else:
            gs.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    def _load_persisted_gallery_bookmarks(self, folder: str) -> set:
        """Restore bookmarked image norm paths for a folder from QSettings."""
        import json

        if not folder:
            return set()
        try:
            raw = self.get_settings().value(self._gallery_bookmarks_settings_key(folder), "")
            if not raw:
                return set()
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", errors="ignore")
            data = json.loads(raw)
            if not isinstance(data, list):
                return set()
            return {_norm_path(str(p)) for p in data if p}
        except Exception:
            return set()

    def _save_persisted_gallery_bookmarks(self) -> None:
        """Persist bookmarked image norm paths for the current folder."""
        import json

        folder = getattr(self, "current_folder", None)
        if not folder:
            return
        try:
            paths = sorted(getattr(self, "_gallery_bookmarked_paths", set()) or set())
            key = self._gallery_bookmarks_settings_key(folder)
            if paths:
                self.get_settings().setValue(
                    key,
                    json.dumps(paths, separators=(",", ":")),
                )
            else:
                self.get_settings().remove(key)
        except Exception:
            pass

    def _reconcile_gallery_bookmarks_with_folder(self) -> None:
        """Drop stale bookmark keys that are no longer in the current folder."""
        valid = {_norm_path(p) for p in (getattr(self, "image_files", None) or [])}
        bookmarked = getattr(self, "_gallery_bookmarked_paths", set()) or set()
        self._gallery_bookmarked_paths = bookmarked & valid
        self._apply_gallery_bookmark_visual_updates()

    def toggle_sort_by_newest(self):
        """Toggle to sort by newest (newest images first)"""
        settings = self.get_settings()
        settings.setValue("sort_by_newest", True)
        self.resort_current_folder()
        self._update_sort_button_text()

    def toggle_sort_by_oldest(self):
        """Toggle to sort by oldest (oldest images first)"""
        settings = self.get_settings()
        settings.setValue("sort_by_newest", False)
        self.resort_current_folder()
        self._update_sort_button_text()

    def toggle_sort_method(self):
        """Toggle between sort by newest and sort by oldest (by capture time)"""
        current_pref = self.get_sort_preference()
        if current_pref:
            # Currently sorting by newest, switch to oldest
            self.toggle_sort_by_oldest()
        else:
            # Currently sorting by oldest, switch to newest
            self.toggle_sort_by_newest()

    def _show_gallery_view(self):
        """Show gallery view - based on reference code"""
        self._gallery_entry_accept_embedded = False
        self._stop_animations()
        import logging
        import os
        import sys
        import time

        from PyQt6.QtCore import QTimer
        logger = logging.getLogger(__name__)
        self._suppress_single_manager_callbacks = True
        try:
            # Session restore may still be staging full decode / CuPy import;
            # abort that before raising gallery concurrency or tiles starve.
            self._abort_session_restore_heavy_for_gallery()
            self._defer_background_indexing_for_gallery()
            if hasattr(self, "image_manager") and self.image_manager is not None:
                mgr = self.image_manager
                if hasattr(mgr, "enter_gallery_warmup_throttle"):
                    mgr.enter_gallery_warmup_throttle()
            if hasattr(self, "_resume_indexing_timer") and self._resume_indexing_timer:
                self._resume_indexing_timer.stop()
            self._resume_indexing_timer = QTimer(self)
            self._resume_indexing_timer.setSingleShot(True)
            self._resume_indexing_timer.timeout.connect(self._resume_indexing_if_gallery_idle)
            file_count = len(getattr(self, "image_files", []) or [])
            resume_ms = min(30000, max(8000, 8000 + (file_count // 500) * 2000))
            self._resume_indexing_timer.start(resume_ms)
        except Exception:
            pass
        bar = self._filmstrip_bar()
        if bar is not None:
            bar.hide()
            bar.setEnabled(False)
        self._hide_filmstrip_chrome()
        logger.debug("[MODESWITCH] _show_gallery_view entered; files=%d", len(self.image_files))
        logger.info("[GALLERY] Showing gallery view")
        if hasattr(self.image_manager, "set_gallery_view_active"):
            self.image_manager.set_gallery_view_active(True)
        if getattr(self, "current_file_path", None):
            self._gallery_scroll_target_path = self.current_file_path
        # Only queue entry-scroll when the gallery still needs a fresh layout.
        # Re-entry with an intact layout must not set _pending_scroll_to_path
        # (that parked load_visible and left tiles blank for seconds).
        if getattr(self, "gallery_justified", None) is not None:
            gj = self.gallery_justified
            display_files = self._gallery_files_for_display()
            already_laid_out = bool(
                display_files
                and hasattr(gj, "_layout_matches_current_images")
                and gj._layout_matches_current_images(display_files)
                and getattr(gj, "_gallery_layout_items", None)
            )
            if not already_laid_out:
                self._scroll_gallery_to_current_file(backup=False)
        self._stop_slideshow()
        if hasattr(self, "loading_overlay"):
            self.loading_overlay.hide_loading()
        if hasattr(self, "_idle_display_prefetch_timer") and self._idle_display_prefetch_timer:
            self._idle_display_prefetch_timer.stop()

        self._cancel_filmstrip_prefetch_for_gallery()
        if hasattr(self, "_idle_display_prefetch_timer") and self._idle_display_prefetch_timer:
            self._idle_display_prefetch_timer.stop()

        # Entering gallery: drop leftover single-view decode/preload work so scrolling
        # thumbnails doesn't compete with stale full-image tasks.
        try:
            if hasattr(self, "image_manager") and self.image_manager is not None:
                mgr = self.image_manager
                if hasattr(mgr, "cancel_non_gallery_tasks"):
                    mgr.cancel_non_gallery_tasks()
                if hasattr(mgr, "cancel_queued_non_gallery_tasks"):
                    mgr.cancel_queued_non_gallery_tasks()
                if hasattr(mgr, "cancel_tasks_by_priority"):
                    from image_load_manager import Priority as _GalleryCancelPriority

                    for prio in (
                        _GalleryCancelPriority.PRELOAD_NEXT,
                        _GalleryCancelPriority.PRELOAD_PREV,
                        _GalleryCancelPriority.BACKGROUND,
                    ):
                        mgr.cancel_tasks_by_priority(prio)
                elif hasattr(mgr, "flush_queue"):
                    mgr.flush_queue()
                else:
                    mgr.cancel_all_tasks()
            if hasattr(self, "preload_manager") and self.preload_manager is not None:
                self.preload_manager.cancel_all_preloads()
        except Exception:
            pass

        # Track gallery loading start time for performance monitoring
        if hasattr(self, 'gallery_justified') and self.gallery_justified:
            gj = self.gallery_justified
            gj._is_scrolling_fast = False
            gj._current_scroll_speed = 0.0
            gj._first_thumb_ready_after_set = False
            if hasattr(gj, "_active_tasks"):
                gj._active_tasks.clear()
            if hasattr(gj, "_requested_thumbnail_paths"):
                gj._requested_thumbnail_paths.clear()
            if hasattr(gj, "_stop_layout_rebuild_timers"):
                gj._stop_layout_rebuild_timers()
            gj._gallery_load_start_time = time.time()
            gj._visible_images_to_load = 0
            gj._visible_images_loaded = 0
            logger.info(f"[GALLERY] Gallery load timing started at {gj._gallery_load_start_time:.3f}")

        # Update title / top metadata for gallery mode
        if hasattr(self, 'current_folder') and self.current_folder:
            folder_name = os.path.basename(self.current_folder)
            title = f"RAWviewer - {folder_name}"
        else:
            title = "RAWviewer"
        if platform.system() == "Darwin":
            self._apply_top_metadata_text(
                os.path.basename(str(self.current_folder or ""))
            )
        else:
            self.setWindowTitle(title)
            if hasattr(self, 'title_bar') and self.title_bar is not None:
                self.title_bar.set_title(title)

        # Create gallery widget if needed (at most one container in the layout)
        if not self._ensure_gallery_widget():
            return
        # Warmup caps (active_cap=12–18 for 1k+ folders) only apply after the widget
        # exists; calling begin_gallery_warmup earlier is a no-op on first gallery entry.
        self._arm_gallery_warmup()

        # Close the editor before leaving single view: _show_gallery_view only
        # ever hid single_view_container, never the Adjust panel itself, so a
        # still-open editor stayed visible over the gallery (splitter gives it
        # the space single_view_container gave up) and left
        # _adjust_overlay_visible stuck True -- which also broke the *next*
        # gallery->single click, since the app still believed it was editing.
        #
        # NOT _adjust_panel_active(): every caller here already flips
        # self.view_mode to "gallery" before calling _show_gallery_view(), and
        # _adjust_panel_active() gates on view_mode == "single" -- so that
        # check always evaluated False by the time it ran, silently no-op'ing
        # this entire fix (the bug it was meant to fix kept recurring).
        # Check the underlying overlay/panel state directly instead.
        panel = getattr(self, "single_image_adjust_panel", None)
        if getattr(self, "_adjust_overlay_visible", False) and panel is not None and panel.isVisible():
            self._close_adjust_panel_from_ui()

        # Hide single view before touching GPU/full-res buffers (Windows GL abort otherwise).
        if hasattr(self, 'single_view_container') and self.single_view_container:
            self.single_view_container.hide()
        else:
            self.scroll_area.hide()
        if hasattr(self, "compare_widget") and self.compare_widget:
            self.compare_widget.hide()
        # Let Qt finish hiding the single-view GL surface before teardown (Windows).
        release_delay_ms = 50 if sys.platform == "win32" else 0
        QTimer.singleShot(
            release_delay_ms, self._release_single_view_heavy_buffers_for_gallery
        )
        # Bottom chrome (sort/search/slideshow vs burst back-to-gallery) handled below.

        # In gallery mode: hide per-image metadata, but keep total count visible.
        if hasattr(self, 'status_bar'):
            self.status_bar.show()  # Keep status_bar visible to show sort button
            self.status_bar.showMessage("")  # Clear message
        if hasattr(self, 'status_metadata_label'):
            self._apply_top_metadata_text(self._gallery_top_metadata_label())
        if hasattr(self, 'status_counter_label'):
            self._update_gallery_counter()
        self._ensure_gallery_search_collapsed()
        if not self._is_gallery_search_indexing_active():
            self._clear_stale_gallery_search_index_progress()
        elif (
            getattr(self, "_search_panel_expanded", False)
            and getattr(self, "_gallery_search_show_index_progress", False)
        ):
            progress = (getattr(self, "_gallery_search_status_full", "") or "").strip()
            if progress:
                self._set_gallery_search_status(progress)
        if self._search_panel_expanded and hasattr(self, "search_expand_layout") and hasattr(
            self, "gallery_search_panel"
        ):
            self.search_expand_layout.setCurrentWidget(self.gallery_search_panel)
        # Show size slider in gallery mode; sort/search/slideshow vs burst back via chrome sync.
        if hasattr(self, 'size_slider'):
            self.size_slider.show()
        self._sync_burst_group_bottom_chrome()
        # Single-image actions stay in update_status_bar for single mode only; hide here
        # because we do not call update_status_bar() when entering gallery (counter text differs).
        if hasattr(self, "share_bottom_button"):
            self._sync_share_button_visibility()
        self._sync_bookmark_indicator()
        self._sync_slideshow_button_visibility()
        self._sync_burst_grouping_button_visibility()
        self._sync_compare_button_visibility()
        if hasattr(self, "rotate_bottom_button"):
            self.rotate_bottom_button.hide()
        if hasattr(self, "adjust_bottom_button"):
            self.adjust_bottom_button.hide()
        if hasattr(self, "raw_toggle_button"):
            self.raw_toggle_button.hide()
        if hasattr(self, "edr_toggle_button"):
            self.edr_toggle_button.hide()
        self._schedule_search_expand_overlay_sync()

        if hasattr(self, "gallery_scroll") and self.gallery_scroll is not None:
            self.gallery_scroll.setWidgetResizable(False)

        # Show gallery
        self.gallery_widget.show()
        self.gallery_widget.raise_()
        self._position_gallery_frame_grip()
        if hasattr(self, "gallery_justified") and self.gallery_justified:
            gj = self.gallery_justified
            if hasattr(gj, "prepare_gallery_display"):
                QTimer.singleShot(0, gj.prepare_gallery_display)

        # Update gallery content
        QTimer.singleShot(50, self._update_gallery_view)
        QTimer.singleShot(120, self._refresh_gallery_bookmark_visuals)
        QTimer.singleShot(0, self._restore_keyboard_focus)
        logger.debug("[MODESWITCH] _show_gallery_view scheduled gallery update")

    def _remove_orphan_gallery_containers(self) -> None:
        """Drop extra gallery shells left in the central layout (defensive)."""
        cw = self.centralWidget()
        if cw is None:
            return
        layout = cw.layout()
        if layout is None:
            return
        keep = getattr(self, "gallery_widget", None)
        for i in range(layout.count() - 1, -1, -1):
            item = layout.itemAt(i)
            if item is None:
                continue
            w = item.widget()
            if w is None or w is keep:
                continue
            if w.objectName() == "rawviewer_gallery_container":
                layout.removeWidget(w)
                w.deleteLater()

    def _ensure_gallery_widget(self) -> bool:
        """Create the gallery UI once; never stack duplicate containers."""
        self._remove_orphan_gallery_containers()
        existing = getattr(self, "gallery_widget", None)
        if existing is not None:
            try:
                layout = self.centralWidget().layout() if self.centralWidget() else None
                if layout is not None and layout.indexOf(existing) >= 0:
                    return True
            except RuntimeError:
                self.gallery_widget = None
                self.gallery_scroll = None
                self.gallery_frame_gutter = None
                self.gallery_justified = None
        self._create_gallery_widget()
        return bool(getattr(self, "gallery_widget", None))

    def _create_gallery_widget(self):
        """Create the gallery view widget - based on JustifiedGallery reference code"""
        if getattr(self, "gallery_widget", None) is not None:
            return
        from PyQt6.QtWidgets import (
            QLabel,
            QPushButton,
            QScrollArea,
            QVBoxLayout,
            QWidget,
        )

        # Create gallery widget container
        gallery_container = QWidget()
        gallery_container.setObjectName("rawviewer_gallery_container")
        gallery_container.setStyleSheet(f"""
            QWidget {{
                background-color: {theme.VOID};
            }}
        """)
        gallery_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        gallery_layout = QVBoxLayout(gallery_container)
        gallery_layout.setContentsMargins(0, 0, 0, 0)
        gallery_layout.setSpacing(0)

        # Batch-selection top bar (Select all · count · Clear · Delete · Discard).
        gallery_selection_bar = QWidget()
        gallery_selection_bar.setObjectName("gallery_selection_bar")
        gallery_selection_bar.setFixedHeight(34)
        bar_layout = QHBoxLayout(gallery_selection_bar)
        bar_layout.setContentsMargins(10, 4, 10, 4)
        bar_layout.setSpacing(8)
        _bar_btn_style = f"""
            QPushButton {{
                color: {theme.INK_MUTED}; background: transparent;
                border: 1px solid transparent; border-radius: 4px;
                padding: 2px 10px; font-size: 12px;
            }}
            QPushButton:hover {{ color: {theme.INK}; background: rgba(255, 255, 255, 0.05); }}
            QPushButton:pressed {{ color: {theme.INK}; background: rgba(255, 255, 255, 0.08); }}
            QPushButton:disabled {{ color: {theme.INK_FAINT}; }}
        """
        _bar_label_style = f"color: {theme.INK_MUTED}; font-size: 12px; background: transparent;"

        self.gallery_select_all_btn = QPushButton("Select all")
        self.gallery_select_all_btn.setStyleSheet(_bar_btn_style)
        self.gallery_select_all_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.gallery_select_all_btn.setToolTip("Select every image currently shown (Ctrl+A)")
        bar_layout.addWidget(self.gallery_select_all_btn)

        self.gallery_selection_count_label = QLabel("", gallery_selection_bar)
        self.gallery_selection_count_label.setStyleSheet(_bar_label_style)
        bar_layout.addWidget(self.gallery_selection_count_label)

        self.gallery_clear_selection_btn = QPushButton("Clear")
        self.gallery_clear_selection_btn.setStyleSheet(_bar_btn_style)
        self.gallery_clear_selection_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.gallery_clear_selection_btn.setToolTip("Clear selection (Esc)")
        bar_layout.addWidget(self.gallery_clear_selection_btn)

        bar_layout.addStretch(1)

        self.gallery_export_selection_btn = QPushButton("Export…")
        self.gallery_export_selection_btn.setStyleSheet(_bar_btn_style)
        self.gallery_export_selection_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.gallery_export_selection_btn.setToolTip(
            "Copy originals or export edited copies to a chosen folder"
        )
        bar_layout.addWidget(self.gallery_export_selection_btn)

        self.gallery_delete_selection_btn = QPushButton("Delete")
        self.gallery_delete_selection_btn.setStyleSheet(_bar_btn_style)
        self.gallery_delete_selection_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.gallery_delete_selection_btn.setToolTip("Move selected images to the Recycle Bin (Del)")
        bar_layout.addWidget(self.gallery_delete_selection_btn)

        self.gallery_discard_selection_btn = QPushButton("Discard")
        self.gallery_discard_selection_btn.setStyleSheet(_bar_btn_style)
        self.gallery_discard_selection_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.gallery_discard_selection_btn.setToolTip("Move selected images into a Discard folder")
        bar_layout.addWidget(self.gallery_discard_selection_btn)

        for _btn in (
            self.gallery_select_all_btn,
            self.gallery_clear_selection_btn,
            self.gallery_export_selection_btn,
            self.gallery_delete_selection_btn,
            self.gallery_discard_selection_btn,
        ):
            _btn.setEnabled(False)

        gallery_layout.addWidget(gallery_selection_bar)

        # Create scroll area for gallery
        gallery_scroll = QScrollArea()
        # False: widget must keep full content height so 1000s of rows scroll correctly.
        gallery_scroll.setWidgetResizable(False)
        gallery_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        gallery_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        gallery_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        gallery_scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: {theme.VOID};
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 24px;
                margin: 0px;
                border: none;
            }}
            QScrollBar::handle:vertical {{
                background: rgba(255, 255, 255, 0.2);
                min-height: 30px;
                border-radius: 6px;
                margin: 2px 6px 2px 6px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: rgba(255, 255, 255, 0.3);
            }}
            QScrollBar::handle:vertical:pressed {{
                background: rgba(255, 255, 255, 0.4);
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{
                height: 0px;
                width: 0px;
            }}
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {{
                background: transparent;
            }}
        """)

        # Create optimized justified gallery widget from rawviewer_ui module.
        justified_gallery = ExternalJustifiedGallery([], self)  # Empty list initially, will be populated

        # Restore saved row height from settings
        try:
            settings = self.get_settings()
            saved_row_height = int(settings.value("gallery_row_height", 220))
            saved_row_height = max(220, min(500, saved_row_height))
            justified_gallery.TARGET_ROW_HEIGHT = saved_row_height
        except Exception:
            pass

        # Capture dates for the scrollbar-drag date bubble. Injected because
        # the times live in this window's image cache, not in the gallery.
        with contextlib.suppress(Exception):
            justified_gallery.set_capture_date_provider(self._gallery_capture_date)

        justified_gallery.installEventFilter(self)
        gallery_scroll.setWidget(justified_gallery)
        gallery_layout.addWidget(gallery_scroll)

        # Overlay on the scrollbar's frame-side margin -- captures resize drags without
        # shrinking the scroll area (layout gutter was stealing 6px from the viewport).
        gallery_grip = QWidget(gallery_container)
        gallery_grip.setStyleSheet("background: transparent;")
        gallery_grip.setMouseTracking(True)
        gallery_grip.hide()

        # Insert gallery widget into main layout (after single-image row).
        # The anchor must be a DIRECT child of main_layout: since the Adjust
        # panel moved single_view_container inside single_view_splitter,
        # indexOf(single_view_container) returns -1, and insertWidget(-1 + 1)
        # placed the gallery at index 0 -- ABOVE the custom title bar. In
        # gallery mode the container then filled the window from the top and
        # the title bar (min/max/close) was pushed out of its expected spot:
        # "top bar missing, cannot close or maximize the app".
        main_layout = self.centralWidget().layout()
        scroll_index = -1
        for anchor in (
            getattr(self, "single_view_splitter", None),
            getattr(self, "single_view_container", None),
            getattr(self, "scroll_area", None),
        ):
            if anchor is not None:
                scroll_index = main_layout.indexOf(anchor)
                if scroll_index >= 0:
                    break
        if scroll_index < 0:
            # No known anchor found: append at the end rather than clobbering
            # index 0 (the title bar row).
            scroll_index = main_layout.count() - 1
        cast(Any, main_layout).insertWidget(scroll_index + 1, gallery_container)

        self.gallery_widget = gallery_container
        self.gallery_selection_bar = gallery_selection_bar
        self.gallery_select_all_btn.clicked.connect(self._gallery_select_all_displayed)
        self.gallery_clear_selection_btn.clicked.connect(self._clear_gallery_selection)
        self.gallery_export_selection_btn.clicked.connect(self._batch_export_selected)
        self.gallery_delete_selection_btn.clicked.connect(self.delete_gallery_selection)
        self.gallery_discard_selection_btn.clicked.connect(self.discard_gallery_selection)
        self.gallery_scroll = gallery_scroll
        self.gallery_frame_gutter = gallery_grip
        self.gallery_justified = justified_gallery
        if hasattr(self, "size_slider") and self.size_slider is not None:
            self.gallery_justified.TARGET_ROW_HEIGHT = self.size_slider.value()
        gallery_scroll.setMouseTracking(True)
        gallery_scroll.viewport().setMouseTracking(True)
        gallery_scroll.installEventFilter(self)
        gallery_scroll.viewport().installEventFilter(self)
        gallery_grip.installEventFilter(self)
        v_scrollbar = gallery_scroll.verticalScrollBar()
        v_scrollbar.setMouseTracking(True)
        v_scrollbar.installEventFilter(self)
        gallery_container.installEventFilter(self)
        self._position_gallery_frame_grip()
        self._setup_file_drop_targets()

        # Hide it initially - it will be shown by _show_gallery_view() when needed
        gallery_container.hide()

    def _rebuild_gallery_zoom(self):
        if hasattr(self, "gallery_justified") and self.gallery_justified:
            gj = self.gallery_justified
            value = self.size_slider.value()
            if not getattr(gj, "_is_zooming", False):
                gj.begin_zoom_session()
            gj.apply_gallery_zoom(value)
            gj.end_zoom_session()
            with contextlib.suppress(Exception):
                self.get_settings().setValue("gallery_row_height", value)

    def _gallery_layout_seed_paths(self, files: list, anchor_path: str | None) -> list:
        """Subset of paths for layout metadata pre-seed (avoid 5k+ SQLite lookups on gallery open)."""
        if not files:
            return []
        limit = self._GALLERY_LAYOUT_META_SEED_MAX
        if len(files) <= limit:
            return files
        if not anchor_path:
            return files[:limit]
        anchor_norm = _norm_path(anchor_path)
        center = 0
        for i, fp in enumerate(files):
            if _norm_path(fp) == anchor_norm:
                center = i
                break
        half = limit // 2
        start = max(0, center - half)
        end = min(len(files), start + limit)
        start = max(0, end - limit)
        return files[start:end]

    def _sync_filmstrip_files(self) -> None:
        """Push the current image_files into the filmstrip, if one is up."""
        try:
            bar = self._filmstrip_bar()
            if bar is not None and hasattr(bar, "set_files"):
                bar.set_files(
                    list(getattr(self, "image_files", []) or []),
                    select_index=getattr(self, "current_file_index", None),
                )
        except Exception:
            pass

    def _update_gallery_view(self):
            """Update gallery view - using JustifiedGallery"""
            import logging
            import time

            from PyQt6.QtCore import QTimer
            logger = logging.getLogger(__name__)
            if getattr(self, "view_mode", "") != "gallery":
                logger.debug(
                    "[GALLERY] _update_gallery_view skipped; view_mode=%s",
                    getattr(self, "view_mode", ""),
                )
                return
            logger.debug("[MODESWITCH] _update_gallery_view called; widget=%s justified=%s files=%d",
                           bool(self.gallery_widget), bool(self.gallery_justified), len(self.image_files))
            start_time = time.time()
            logger.info("[GALLERY] ========== _update_gallery_view() STARTED ==========")
            self._update_gallery_counter()
            self._sync_gallery_selection_chrome()

            if not self.gallery_widget or not self.gallery_justified:
                logger.info("[GALLERY] Gallery widget not available, returning")
                return
            self._sync_gallery_scrollbar_policy()
            if not self.image_files:
                logger.info("[GALLERY] No image files for gallery update, returning")
                query = getattr(self, "_last_semantic_query", "") or ""
                if query.strip():
                    bulk_meta = getattr(self, "_gallery_bulk_metadata", None)
                    self.gallery_justified.set_images(
                        [], bulk_meta if bulk_meta else None
                    )
                    self.gallery_justified.show_empty_message("No matching images")
                return

            # IMPORTANT: Switching to gallery should be instant.
            # Do NOT block the UI thread on metadata fetch for thousands of files.
            # 1) Show gallery immediately with a fast layout (default aspect ratios or cached).
            bulk_metadata = getattr(self, '_gallery_bulk_metadata', None) or {}
            try:
                # Pre-seed layout metadata from the fast semantic index if available
                # to prevent the gallery from flashing as async EXIF arrives
                try:
                    idx = self._get_semantic_index()
                    if idx and self.image_files:
                        layout_meta = idx.get_layout_metadata_for_paths(self.image_files)
                        if layout_meta:
                            for fp, meta in layout_meta.items():
                                if fp not in bulk_metadata:
                                    bulk_metadata[fp] = {}
                                # Don't overwrite if EXIF already arrived
                                if "original_width" not in bulk_metadata[fp] and (
                                    meta.get("original_width") or meta.get("width")
                                ):
                                    if meta.get("original_width"):
                                        bulk_metadata[fp]["original_width"] = int(
                                            meta["original_width"]
                                        )
                                        bulk_metadata[fp]["original_height"] = int(
                                            meta.get("original_height") or 0
                                        )
                                        bulk_metadata[fp]["orientation"] = int(
                                            meta.get("orientation") or 1
                                        )
                                    else:

                                        w = int(meta.get("width") or 0)
                                        h = int(meta.get("height") or 0)
                                        o = int(meta.get("orientation") or 1)
                                        bulk_metadata[fp]["original_width"] = w
                                        bulk_metadata[fp]["original_height"] = h
                                        bulk_metadata[fp]["orientation"] = o
                except Exception as ex:
                    logger.warning(f"[GALLERY] Could not pre-seed semantic metadata: {ex}")

                self._maybe_clear_empty_bookmark_filter()
                self._maybe_clear_empty_rating_filter()
                self._rebuild_burst_groups()
                display_files = self._gallery_files_for_display()
                self._prune_gallery_selection_to_displayed(display_files)
                filter_active = bool(
                    getattr(self, "_gallery_bookmark_filter_active", False)
                ) or int(getattr(self, "_gallery_rating_filter_min", 0) or 0) > 0
                if not display_files and filter_active:
                    self.gallery_justified.set_images(
                        [], bulk_metadata if bulk_metadata else None
                    )
                    self.gallery_justified.show_empty_message(
                        f"No {self._gallery_scope_filter_description()}"
                    )
                else:
                    if hasattr(self.gallery_justified, "hide_empty_message"):
                        self.gallery_justified.hide_empty_message()
                    gj = self.gallery_justified
                    display_files = self._gallery_files_for_display()
                    preserve_viewport = bool(
                        gj
                        and gj._layout_matches_current_images(display_files)
                        and getattr(gj, "_gallery_layout_items", None)
                    )
                    if not preserve_viewport:
                        self._scroll_gallery_to_current_file(backup=False)
                    self.gallery_justified.set_images(
                        display_files, bulk_metadata if bulk_metadata else None
                    )
                    # Preserve path: scroll position is already correct -- calling
                    # scroll_to_file() would reset attempt counters and park
                    # load_visible behind entry-scroll retries (blank gallery).
                    if not preserve_viewport:
                        self._scroll_gallery_to_current_file()
                # Avoid forced rebuild loops; set_images() already schedules build/layout.
                # Only nudge visible-load passes after initial layout is expected ready.
                self._warm_gallery_from_filmstrip_cache()
                if hasattr(self.gallery_justified, "sync_visual_rotations"):
                    # Fast path: layout already built (same folder/order). Full rebuild path
                    # syncs at end of build_gallery() once _path_to_indices exists.
                    self.gallery_justified.sync_visual_rotations()
                current_file = getattr(self, "current_file_path", None)
                if (
                    current_file
                    and hasattr(self.gallery_justified, "apply_visual_rotation_for_path")
                ):
                    self.gallery_justified.apply_visual_rotation_for_path(current_file)
                file_count = len(self.image_files or [])
                first_load_ms = min(200, max(80, 80 + file_count // 120))
                gj = self.gallery_justified
                if hasattr(gj, "_request_load_visible_images"):
                    gj._request_load_visible_images(0)
                    gj._request_load_visible_images(first_load_ms)
                    gj._request_load_visible_images(first_load_ms + 500)
                else:
                    QTimer.singleShot(first_load_ms, gj.load_visible_images)
                    QTimer.singleShot(first_load_ms + 500, gj.load_visible_images)
            except Exception as e:
                logger.exception(f"[GALLERY] set_images failed: {e}")
                self.show_error(
                    "Gallery Error",
                    f"Could not build the gallery view:\n{e}",
                )
                return

            # 2) If metadata is missing, fetch it in the background and refresh once ready.
            if not bulk_metadata and not getattr(self, "_gallery_metadata_fetch_in_progress", False):
                self._gallery_metadata_fetch_in_progress = True
                folder_at_request = getattr(self, "current_folder", None)
                files_snapshot = list(self.image_files)

                from PyQt6.QtCore import QRunnable

                # Create signal carrier for this specific fetch
                signals = GalleryMetadataSignals()
                signals.ready.connect(self._on_gallery_metadata_ready)

                # Store references to prevent garbage collection while thread is running
                self._active_metadata_signals = signals

                class _GalleryMetadataFetch(QRunnable):
                    def __init__(self_inner, files, signals, folder_path):
                        super().__init__()
                        self_inner.files = files
                        self_inner.signals = signals
                        # Ensure folder_path is string (PyQt signals are strict)
                        self_inner.folder_path = folder_path if folder_path is not None else ""

                    def run(self_inner):
                        try:
                            from image_cache import get_image_cache
                            cache = get_image_cache()
                            # OPTIMIZATION: In large folders, avoid hitting the disk for every file
                            # in a background thread while the gallery is also loading thumbnails.
                            # Just use what's in the SQLite cache (opportunistic like v1.6.0).
                            meta = cache.get_multiple_exif(self_inner.files, fast_mode=True)
                            try:
                                from PyQt6 import sip
                                if not sip.isdeleted(self_inner.signals):
                                    self_inner.signals.ready.emit(meta, self_inner.folder_path)
                            except Exception:
                                pass
                        except Exception as e:
                            import logging
                            logger = logging.getLogger(__name__)
                            logger.error(f"[GALLERY] Metadata fetch error: {e}")
                            # Emit empty meta but still valid folder path string
                            try:
                                from PyQt6 import sip
                                if not sip.isdeleted(self_inner.signals):
                                    self_inner.signals.ready.emit({}, self_inner.folder_path)
                            except Exception:
                                pass

                fetcher = _GalleryMetadataFetch(files_snapshot, signals, folder_at_request)
                self._active_metadata_fetcher = fetcher
                _get_bg_thread_pool().start(fetcher)

            total_time = time.time() - start_time
            logger.info(f"[GALLERY] ========== GALLERY LAYOUT COMPLETED in {total_time:.3f}s ==========")

    def _add_gallery_row(self, row_items, available_width, content_width, row_height, row_spacing):
            """Add a single row - based on reference code add_row method"""
            from PyQt6.QtWidgets import QHBoxLayout, QWidget

            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(0)

            # Remaining space after placing all thumbnails
            free_space = max(0, available_width - content_width)

            # Spacing distributed evenly between items
            gaps = len(row_items) - 1 if len(row_items) > 1 else 1
            # Limit extra padding to prevent excessive spacing
            # Use minimum of calculated padding or a maximum value (e.g., 8px)
            extra_padding = min(free_space // gaps, 8) if gaps > 0 else 0

            for i, (file_path, w) in enumerate(row_items):
                # Create thumbnail label (will be loaded async)
                thumb_label = ThumbnailLabel()
                thumb_label.file_path = file_path

                if not hasattr(self, '_gallery_thumb_labels'):
                    self._gallery_thumb_labels = {}
                self._gallery_thumb_labels[file_path] = thumb_label

                # Make clickable
                thumb_label.clicked.connect(self._gallery_item_clicked)

                # Load pixmap and scale it
                pixmap = self._get_gallery_pixmap(file_path)
                if pixmap and not pixmap.isNull():
                    # Scale to fixed height while preserving aspect ratio (like reference)
                    # Ensure dimensions are at least 1px to prevent crash
                    safe_width = max(1, w)
                    safe_height = max(1, row_height)
                    resized = pixmap.scaled(
                        QSize(safe_width, safe_height),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    thumb_label.setFixedSize(resized.size())  # CRITICAL: Like reference code
                    thumb_label.setPixmap(resized)
                    thumb_label.set_original_pixmap(pixmap)
                else:
                    # Placeholder size, will be updated when loaded
                    thumb_label.setFixedSize(w, row_height)
                    # Load asynchronously
                    from PyQt6.QtCore import QTimer
                    QTimer.singleShot(50, lambda fp=file_path, tl=thumb_label: self._load_gallery_thumbnail_simple(fp, tl))

                row_layout.addWidget(thumb_label)

                # Apply padding after each item (except last) - like reference code
                if i < len(row_items) - 1:
                    row_layout.addSpacing(row_spacing + extra_padding)

            self.gallery_content_layout.addWidget(row_widget)

    def _load_gallery_thumbnail_simple(self, file_path, thumb_label):
            """Load thumbnail simply - based on reference code"""

            pixmap = self._get_gallery_pixmap(file_path)
            if not pixmap or pixmap.isNull():
                # Try async load
                self._load_gallery_pixmap_async(file_path)
                return

            # Get aspect ratio
            aspect = pixmap.width() / pixmap.height() if pixmap.height() > 0 else 4.0 / 3.0
            thumb_width = int(self.gallery_row_height * aspect)

            # Scale to fixed height while preserving aspect ratio (like reference)
            # Ensure dimensions are at least 1px to prevent crash
            safe_width = max(1, thumb_width)
            safe_height = max(1, self.gallery_row_height)
            resized = pixmap.scaled(
                QSize(safe_width, safe_height),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            # CRITICAL: Like reference code - thumb.setFixedSize(resized.size())
            thumb_label.setFixedSize(resized.size())
            thumb_label.setPixmap(resized)
            thumb_label.set_original_pixmap(pixmap)

            if file_path in self._gallery_load_tracking:
                self._gallery_load_tracking[file_path]['loaded'] = True
                self._gallery_load_tracking[file_path]['displayed'] = True
