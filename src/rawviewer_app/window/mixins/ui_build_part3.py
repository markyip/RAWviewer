"""ui_build mixin, part 3 of 3 (8 methods, 229 lines)."""
from __future__ import annotations

import logging
import os
import sys
import time

from PyQt6.QtCore import QPoint
from PyQt6.QtGui import QAction, QKeySequence

from core import theme
from core.config import _env_int, _norm_path

from .. import lazy
from ..macos import _macos_try_force_dark_titlebar, _macos_try_force_dark_titlebar_for_window
from ..paths import _is_in_folder_scope
from ..qt_helpers import _qta_icon

lazy.register(sys.modules[__name__])

logger = logging.getLogger(__name__)


class UiBuildPart3:
    def _apply_macos_titlebar_appearance(self) -> None:
        """Request dark native title bar on macOS (system-drawn; not pure #000)."""
        _macos_try_force_dark_titlebar()
        _macos_try_force_dark_titlebar_for_window(self)

    def create_menu_bar(self):
        """Create the menu bar with File and Keyboard Shortcuts action"""
        menubar = self.menuBar()
        if menubar is None:
            return

        # File menu
        file_menu = menubar.addMenu('File')

        # Open action
        open_action = QAction('Open', self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.setStatusTip('Open a RAW image file')
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        # Open Folder action
        open_folder_action = QAction('Open Folder', self)
        open_folder_action.setShortcut('Ctrl+Shift+O')
        open_folder_action.setStatusTip('Open a folder of images')
        open_folder_action.triggered.connect(self.open_folder)
        file_menu.addAction(open_folder_action)

        file_menu.addSeparator()

        reveal_name = (
            "Reveal in Finder"
            if sys.platform == "darwin"
            else ("Show in File Explorer" if sys.platform == "win32" else "Open Folder in File Manager")
        )
        self.reveal_action = QAction(reveal_name, self)
        self.reveal_action.setStatusTip("Select this file in the system file manager")
        self.reveal_action.triggered.connect(self._reveal_current_file_in_os_file_manager)
        file_menu.addAction(self.reveal_action)

        file_menu.addSeparator()

        # Exit action
        exit_action = QAction('Exit', self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Keyboard Shortcuts action (direct in menu bar)
        shortcuts_action = QAction('Keyboard Shortcuts', self)
        shortcuts_action.setStatusTip('Show keyboard shortcuts')
        shortcuts_action.triggered.connect(self.show_keyboard_shortcuts)
        menubar.addAction(shortcuts_action)

        # Shortcuts still work when the menu bar is hidden (Windows frameless UI)
        self.addAction(open_action)
        self.addAction(open_folder_action)
        self.addAction(self.reveal_action)
        self.addAction(exit_action)
        self.addAction(shortcuts_action)

    def _show_created_file(self, path: str) -> None:
        """Open a file this app just created, in single view."""
        try:
            parent = getattr(self, "current_folder", None) or ""
            if parent and _is_in_folder_scope(
                os.path.dirname(os.path.abspath(path)), parent
            ):
                # Parent-fallback scope: the file may sit one level down; keep
                # the opened folder as the session scope instead of flipping to
                # the file's own directory.
                self.load_folder_images(parent, start_file=path, start_view="single")
            else:
                self.load_folder_images(
                    os.path.dirname(os.path.abspath(path)),
                    start_file=os.path.basename(path),
                    start_view="single",
                )
        except Exception:
            import logging

            logging.getLogger(__name__).warning(
                "[GALLERY] could not open %s", path, exc_info=True
            )

    def convert_widget_to_image_coords(self, widget_pos):
        """Convert widget coordinates to full-resolution image coordinates"""
        if not self.current_pixmap:
            return QPoint(0, 0)

        original_size = self.current_pixmap.size()
        if original_size.width() <= 0 or original_size.height() <= 0:
            return QPoint(0, 0)

        displayed_image = self.image_label.pixmap()
        if not displayed_image or displayed_image.isNull():
            return QPoint(
                original_size.width() // 2,
                original_size.height() // 2,
            )

        displayed_size = displayed_image.size()
        if displayed_size.width() <= 0 or displayed_size.height() <= 0:
            return QPoint(
                original_size.width() // 2,
                original_size.height() // 2,
            )

        # Image is centered in the label (same math as double-click zoom).
        label_size = self.image_label.size()
        image_x_offset = (label_size.width() - displayed_size.width()) / 2
        image_y_offset = (label_size.height() - displayed_size.height()) / 2
        adjusted_x = widget_pos.x() - image_x_offset
        adjusted_y = widget_pos.y() - image_y_offset

        click_ratio_x = adjusted_x / displayed_size.width()
        click_ratio_y = adjusted_y / displayed_size.height()
        image_x = int(click_ratio_x * original_size.width())
        image_y = int(click_ratio_y * original_size.height())
        image_x = max(0, min(image_x, original_size.width() - 1))
        image_y = max(0, min(image_y, original_size.height() - 1))
        return QPoint(image_x, image_y)

    def _setup_navigation_state(self, requested_file_path, force_reload):
        """Setup navigation state (pause preloading, indexing, etc.)"""
        try:
            if hasattr(self, 'gallery_view') and hasattr(self.gallery_view, 'pause_idle_preload'):
                self.gallery_view.pause_idle_preload()
                idle_timer = getattr(self.gallery_view, '_idle_preload_timer', None)
                if idle_timer is not None:
                    idle_timer.start(
                        _env_int("RAWVIEWER_SINGLE_VIEW_THUMB_PRELOAD_MS", 4000, minimum=1000)
                    )
        except Exception as e:
            logger.debug(f"[LOAD] Failed to pause gallery idle preload: {e}")

        # Stop metadata-index defer timer
        try:
            t = getattr(self, "_metadata_index_defer_timer", None)
            if t is not None and t.isActive():
                t.stop()
                logger.debug("[LOAD] Stopped _metadata_index_defer_timer for navigation")
        except Exception as e:
            logger.debug(f"[LOAD] Failed to stop metadata index timer: {e}")

        # Initialize single-view load tracking
        try:
            self._single_view_load_request_ts = float(time.time())
            self._single_view_load_request_path = str(requested_file_path or "")
            self._single_view_first_render_logged = False
            self._defer_sensor_full_decode_path = None
            self._sensor_full_decode_queued_norm = None
            self._defer_full_exif_path = None
            self._full_exif_queued_norm = None
            self._last_manager_exif_was_minimal = False

            # Restart the idle full-res timer for the new file
            try:
                it = getattr(self, "_full_res_idle_timer", None)
                if it is not None:
                    it.stop()
                self._full_res_idle_path = None
            except Exception:
                pass

            self._adjust_preview_base_rgb = None
            self._adjust_preview_base_rgb_fast = None
            self._adjust_edit_base_path = None
            self._adjust_edit_base_loading_norm = None
            self._adjust_panel_loaded_norm = None
        except Exception:
            pass

    def _sync_slideshow_button_icon(self, playing: bool):
        btn = getattr(self, "slideshow_bottom_button", None)
        if btn is None:
            return
        try:
            if playing:
                btn.setIcon(_qta_icon("fa5s.pause", color=theme.INK_MUTED))
            else:
                btn.setIcon(_qta_icon("fa5s.play", color=theme.INK_MUTED))
        except Exception:
            pass

    def _is_widget_in_main_window(self, obj) -> bool:
        from PyQt6.QtWidgets import QWidget

        if obj is self:
            return True
        w = obj if isinstance(obj, QWidget) else None
        if w is None:
            return False
        while w is not None:
            if w is self:
                return True
            if w.isWindow() and w is not self:
                return False
            w = w.parentWidget()
        return False

    def _build_quick_folder_image_list(self, folder_path: str, start_path: str | None = None):
        """Scandir + mtime sort only. Returns (image_files, file_stats, start_index)."""
        image_files = []
        file_stats = {}
        for ap, stat_info in self._scan_folder_image_paths(folder_path):
            image_files.append(ap)
            file_stats[ap] = (stat_info.st_size, stat_info.st_mtime)

        if not image_files:
            if start_path:
                return [os.path.abspath(start_path)], {}, 0
            return [], {}, 0

        image_files = self._mtime_sort_image_files(image_files, file_stats)
        start_idx = 0
        if start_path:
            start_abs = os.path.abspath(start_path)
            start_norm = _norm_path(start_abs)
            if not any(_norm_path(fp) == start_norm for fp in image_files) and self.is_image_file(start_abs) and os.path.isfile(start_abs):
                    image_files.append(start_abs)
                    if start_abs not in file_stats:
                        try:
                            st = os.stat(start_abs)
                            file_stats[start_abs] = (st.st_size, st.st_mtime)
                        except OSError:
                            file_stats[start_abs] = (0, 0)
                    image_files = self._mtime_sort_image_files(image_files, file_stats)
            for i, fp in enumerate(image_files):
                if _norm_path(fp) == start_norm:
                    start_idx = i
                    break
        return image_files, file_stats, start_idx
