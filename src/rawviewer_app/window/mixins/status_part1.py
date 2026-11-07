"""status mixin, part 1 of 2 (20 methods, 227 lines)."""
from __future__ import annotations

import logging
import os
import platform
import sys

from PyQt6.QtCore import QPoint, QTimer

from core.config import _env_true, _norm_path

from .. import lazy
from ..lazy import use_progressive_raw_loading
from ..logging_setup import _share_log
from ..paths import _require_str_path
from ..qt_helpers import _qt_app

lazy.register(sys.modules[__name__])


class StatusPart1:
    def _notify_raw_recovery_zoom_blocked(self) -> None:
        self.status_bar.showMessage(
            "Recovery preview is fit-only (~half-res). Press P to return to full resolution.",
            4000,
        )

    def _show_edr_startup_status(self) -> None:
        from common_image_loader import summarize_macos_edr_startup

        msg = summarize_macos_edr_startup(gpu_view=getattr(self, "gpu_view", None))
        logging.getLogger(__name__).info("[EDR] %s", msg)
        if hasattr(self, "status_bar") and self.status_bar is not None:
            self.status_bar.showMessage(msg, 8000)

    def _burst_group_view_status_label(self) -> str:
        from core.burst_grouping import format_burst_group_status, sort_paths_by_capture_time

        members = self._burst_group_view_visible_members()
        if not members:
            return ""
        capture_times = self._capture_times_for_paths(members)
        ordered = sort_paths_by_capture_time(members, capture_times)
        return format_burst_group_status(ordered, capture_times)

    def _update_compare_status_bar(self) -> None:
        session = getattr(self, "_comparison_session", None)
        if session is None:
            return
        left = os.path.basename(session.select) if session.select else ""
        right = os.path.basename(session.candidate) if session.candidate else ""
        label = f"{left} / {right}".strip(" /")
        if hasattr(self, "status_metadata_label"):
            self._apply_top_metadata_text(label)
        if platform.system() != "Darwin" and hasattr(self, "title_bar") and self.title_bar:
            self.title_bar.set_title(f"RAWviewer - {label}")
        if hasattr(self, "shortcuts_hint_button"):
            self.shortcuts_hint_button.setToolTip(self._keyboard_shortcuts_tooltip())

    def _hide_all_loading_indicators(self):
        """Helper to hide all loading indicators across modes"""
        # Always clear gallery toast if the widget exists -- view_mode may have
        # switched to single during folder scan, leaving a stale "Scanning folder..." label.
        if self.gallery_justified:
            self.gallery_justified.hide_loading_message()
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.hide_loading()

    def on_manager_progress(self, file_path: str, status_message: str):
        """處理 ImageLoadManager 的進度更新信號"""
        import logging
        logging.getLogger(__name__)

        # Only handle current file's progress (normalize for Windows path format differences)
        if _norm_path(file_path) != _norm_path(getattr(self, "current_file_path", None)):
            return

        filename = os.path.basename(file_path)
        self.status_bar.showMessage(f"{filename}: {status_message}")

    def _sync_bookmark_indicator(self) -> None:
        self._sync_single_view_rating_stars()
        self._sync_gallery_rating_filter_stars()
        self._sync_gallery_clear_filter_button()
        # The gallery bookmark-filter star is retired: the rating filter
        # covers "narrow down the gallery". ↑ scrolls the gallery (or
        # promotes in Compare); it no longer toggles bookmarks.
        btn = getattr(self, "batch_mark_indicator", None)
        if btn is not None:
            btn.hide()

    def on_processing_progress(self, message):
        """Handle processing progress updates."""
        filename = os.path.basename(_require_str_path(self.current_file_path))
        self.status_bar.showMessage(f"{filename}: {message}")

    def _set_edit_loading_overlay(self, norm: str | None) -> None:
        """Show/hide the "applying saved edit" overlay over the photo.

        ``norm`` is the normalized path the overlay is waiting on (None to
        hide). Tracked so a late render for a file the user already
        navigated away from cannot dismiss a newer file's overlay.
        """
        self._adjust_loading_edit_norm = norm
        if norm is None:
            self._adjust_loading_edit_wants_db = False
            self._adjust_loading_edit_wants_heal = False
            # Also release the render suppressor here (panel close, edit-base
            # failure, navigating to an unedited file, render complete). A
            # stuck-True flag would silently suppress every future
            # full-quality render for this session.
            self._adjust_mask_load_pending = False
        overlay = getattr(self, "edit_loading_overlay", None)
        if overlay is None:
            return
        if norm is None:
            overlay.hide_loading()
            return
        overlay.setGeometry(self.rect())
        overlay.show_loading("Loading previous edit…")
        overlay.raise_()
        # Paint it now: the caller keeps running synchronous panel setup
        # (_begin_adjust_editing_session, filmstrip hide, ...) before it
        # yields to the event loop, so without this the overlay would not
        # appear until well after the work it is announcing has started.
        app = _qt_app()
        if app is not None:
            app.processEvents()

    def _on_adjust_ai_progress(self, generation: int, frac: float) -> None:
        if int(generation) != getattr(self, "_ai_preview_gen", None):
            return
        panel = getattr(self, "single_image_adjust_panel", None)
        if panel is not None and hasattr(panel, "set_ai_busy"):
            panel.set_ai_busy(True, frac)

    def _set_ai_preview_busy(self, busy: bool) -> None:
        panel = getattr(self, "single_image_adjust_panel", None)
        if panel is not None and hasattr(panel, "set_ai_busy"):
            panel.set_ai_busy(busy, 0.0)

    def _share_status_debug(self, msg: str, ms: int = 8000) -> None:
        """Surface share diagnostics in the status bar when debug is enabled."""
        _share_log(logging.INFO, "ui: %s", msg)
        if not _env_true("RAWVIEWER_SHARE_DEBUG"):
            return
        try:
            if hasattr(self, "status_bar") and self.status_bar:
                self.status_bar.showMessage(f"Share: {msg}", ms)
        except Exception:
            pass

    def _should_skip_loading_overlay(self) -> bool:
        """Evaluate if full-screen loading overlay should be skipped to prevent flashing."""
        return bool(
            (not self.fit_to_window and getattr(self, "_maintain_zoom_on_navigation", False))
            or getattr(self, "_navigation_in_progress", False)
            or getattr(self, "_loading_from_gallery", False)
            or getattr(self, "_loading_from_filmstrip", False)
            or getattr(self, "_gallery_preview_pending_full", False)
            or getattr(self, "_folder_fast_open_token", None) is not None
            or use_progressive_raw_loading()
        )

    def _update_gpu_status(self):
        gv = getattr(self, "gpu_view", None)
        if gv is None or not gv.has_pixmap():
            return
        try:
            if gv.is_fit_mode():
                self.update_status_bar()
            else:
                int(round(gv.current_scale() * 100))
                self.update_status_bar()
        except Exception:
            pass

    def _status_bar_corner_resize_at(self, pos: QPoint) -> bool:
        """Bottom corners over the status bar stay draggable on frameless windows."""
        sb = getattr(self, "status_bar", None)
        if sb is None or not sb.isVisible():
            return False
        status_h = sb.height()
        if status_h <= 0 or pos.y() < self.height() - status_h:
            return False
        border = self._frame_resize_border()
        if pos.x() <= border or pos.x() >= self.width() - border:
            return True
        return self._pointer_near_bottom_right_grip(pos)

    def _status_bar_blocks_resize_at(self, pos: QPoint) -> bool:
        """Bottom status row hosts controls -- block edge resize except corners."""
        sb = getattr(self, "status_bar", None)
        if sb is None or not sb.isVisible():
            return False
        status_h = sb.height()
        if status_h <= 0:
            return False
        if pos.y() < self.height() - status_h:
            return False
        return not self._status_bar_corner_resize_at(pos)

    def _is_status_bar_chrome_widget(self, obj) -> bool:
        from PyQt6.QtWidgets import QWidget

        sb = getattr(self, "status_bar", None)
        host = getattr(self, "status_bar_widget", None)
        w = obj if isinstance(obj, QWidget) else None
        while w is not None:
            if w is sb or w is host:
                return True
            w = w.parentWidget()
        return False

    def _apply_preview_first_status_bar(self, cached_exif: dict) -> None:
        """Lightweight HUD after first paint while full exiftool metadata loads."""
        if not self.current_file_path or getattr(self, "view_mode", "single") != "single":
            return
        file_label = self._metadata_file_label(self.current_file_path)
        ow = int(cached_exif.get("original_width") or 0)
        oh = int(cached_exif.get("original_height") or 0)
        gv = getattr(self, "gpu_view", None)
        if gv is not None and gv.has_pixmap():
            zoom_level = "Fit" if gv.is_fit_mode() else f"{int(round(gv.current_scale() * 100))}%"
        elif getattr(self, "fit_to_window", False):
            zoom_level = "Fit"
        else:
            zoom_level = f"{int(getattr(self, 'current_zoom_level', 1.0) * 100)}%"
        parts = []
        if ow > 0 and oh > 0:
            parts.append(f"{file_label} - {ow}x{oh}")
        else:
            parts.append(file_label)
        parts.append(zoom_level)
        for key in ("focal_length", "aperture", "iso"):
            val = cached_exif.get(key)
            if val:
                parts.append(str(val))
        self._apply_top_metadata_text(" - ".join(parts))
        if hasattr(self, "status_counter_label"):
            self._refresh_image_counter()

    def update_status_bar(self, width=None, height=None):
        """Debounced status-bar refresh.

        Navigation fires this 5-6 times in quick succession (EXIF-ready,
        thumbnail-ready, full-res-ready, zoom sync, ...), each execution doing
        EXIF lookups plus label/string work on the main thread. A 50ms
        trailing single-shot collapses each burst into one real refresh;
        the latest (width, height) passed during the burst wins.
        """
        self._status_bar_pending_dims = (width, height)
        t = getattr(self, "_status_bar_debounce_timer", None)
        if t is None:
            t = QTimer(self)
            t.setSingleShot(True)
            t.setInterval(50)
            t.timeout.connect(self._flush_status_bar_update)
            self._status_bar_debounce_timer = t
        if not t.isActive():
            t.start()

    def _flush_status_bar_update(self):
        w, h = getattr(self, "_status_bar_pending_dims", (None, None))
        self._update_status_bar_now(w, h)
