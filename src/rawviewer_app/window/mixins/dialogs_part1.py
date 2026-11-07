
"""dialogs mixin, part 1 of 1 (40 methods, 852 lines)."""
from __future__ import annotations

import contextlib
import logging
import os
import sys
import time

from PyQt6.QtCore import QPoint, Qt, QTimer, QUrl
from PyQt6.QtGui import QCursor, QDesktopServices
from PyQt6.QtWidgets import QDialog, QFileDialog

from core.config import _env_true, _norm_path, safe_print
from rawviewer_app.core.widgets import CustomConfirmDialog, CustomWarningDialog
from rawviewer_app.signals import ReleaseUpdateCheckSignals
from rawviewer_app.workers import _ReleaseUpdateCheckWorker

from .. import lazy
from ..lazy import image_covers_sensor_resolution
from ..logging_setup import _share_log
from ..macos import (
    _activate_macos_foreground_app,
    _applescript_escape,
    _get_macos_share_picker_delegate_class,
    _macos_share_items_array,
    _run_applescript,
)
from ..qt_helpers import _get_bg_thread_pool, _qt_app

lazy.register(sys.modules[__name__])


class DialogsPart1:
    def _show_unsupported_file_dialog_if_still_failed(self, file_path: str) -> None:
        """Delayed second half of the 'clearly_unsupported' branch in
        on_manager_error: only pop the modal if the file is still current AND
        a genuine full-resolution fallback (e.g. embedded JPEG) hasn't landed
        in the grace period. A stale low-res thumbnail alone does NOT
        suppress the dialog -- a real decode failure with only a cached
        thumbnail should still be reported. See call site for why the delay
        exists.
        """
        if _norm_path(file_path) != _norm_path(getattr(self, "current_file_path", None)):
            return
        if self._single_view_pixels_on_screen(file_path):
            pm = getattr(self, "current_pixmap", None)
            if pm is not None and not pm.isNull():
                long_edge = max(int(pm.width()), int(pm.height()))
                # HE/HE* NEF and other LibRaw-unsupported RAW often settle on a
                # large embedded JPEG that does not quite cover sensor dims
                # (e.g. 8256 vs 8280). Treat a solid display-quality paint as
                # success so we do not flash a false "corrupt" modal.
                if long_edge >= 1400:
                    return
                exif = None
                with contextlib.suppress(Exception):
                    exif = self.image_cache.get_exif(file_path)
                if image_covers_sensor_resolution(pm.width(), pm.height(), exif):
                    return
                if self._nef_he_compressed(file_path) and long_edge >= 768:
                    return
        try:
            from raw_adjustments.file_extensions import get_supported_extensions
            exts = get_supported_extensions()
        except Exception:
            exts = ['.arw', '.cr2', '.nef', '.dng', '.jpg', '.jpeg', '.png']

        dialog = CustomWarningDialog(
            self,
            title="Unsupported File Format",
            message=(
                f"The file '{os.path.basename(file_path)}' is not supported by "
                f"RAWviewer or is corrupt."
            ),
            informative_text=(
                f"Check if the file is corrupt. Supported formats list: {', '.join(exts)}"
            ),
        )
        dialog.exec()

    def _run_release_update_check(self) -> None:
        if getattr(self, "_release_update_check_in_progress", False):
            return
        self._release_update_check_in_progress = True
        signals = ReleaseUpdateCheckSignals()
        self._release_update_check_signals = signals
        signals.finished.connect(self._on_release_update_check_finished)
        _get_bg_thread_pool().start(_ReleaseUpdateCheckWorker(signals))

    def _show_release_update_available_dialog(
        self, current: str, latest: str, release_url: str
    ) -> bool:
        """Return True when the user chose Not Now (or dismissed), False after Open Download Page."""
        from core.release_update import RELEASE_PAGE_URL
        from rawviewer_ui.dialogs.release_update_dialog import ReleaseUpdateDialog

        url = (release_url or RELEASE_PAGE_URL).strip() or RELEASE_PAGE_URL
        dialog = ReleaseUpdateDialog(
            self,
            current=current,
            latest=latest,
            release_url=url,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            QDesktopServices.openUrl(QUrl(dialog.release_url))
            return False
        return True

    def _on_release_update_check_finished(self, result: dict) -> None:
        self._release_update_check_in_progress = False
        result = result or {}
        if result.get("offline") or str(result.get("error") or "").strip():
            return

        if not result.get("update_available"):
            return

        current = str(result.get("current") or "").strip()
        latest = str(result.get("latest") or "").strip()
        if not latest:
            return

        settings = self.get_settings()
        snoozed_version = str(settings.value("update_check/snoozed_version", "") or "").strip()
        snooze_until = float(settings.value("update_check/snooze_until_epoch", 0) or 0)
        if snoozed_version and time.time() < snooze_until:
            from core.release_update import compare_version_labels

            if compare_version_labels(latest, snoozed_version) <= 0:
                return

        release_url = str(result.get("release_url") or "").strip()
        dismissed = self._show_release_update_available_dialog(current, latest, release_url)
        if dismissed:
            settings.setValue("update_check/snoozed_version", latest)
            settings.setValue(
                "update_check/snooze_until_epoch",
                int(time.time() + 14 * 86400),
            )

    def _prompt_mobileclip_download(self, corpus_files) -> bool:
        """Return True if download started or models already present; False if user declined."""
        if not self._mobileclip_models_missing():
            return True
        if getattr(self, "_mobileclip_download_dismissed_this_session", False):
            return False

        from rawviewer_ui.dialogs.mobileclip_download_dialog import MobileCLIPDownloadDialog

        dialog = MobileCLIPDownloadDialog(self)
        self._mobileclip_download_dialog = dialog
        dialog.download_requested.connect(
            lambda: self._start_semantic_asset_download_background(corpus_files)
        )
        dialog.exec()
        self._mobileclip_download_dialog = None
        if dialog.download_started:
            return True
        self._mobileclip_download_dismissed_this_session = True
        return False

    def _release_single_view_heavy_buffers_for_gallery(self) -> None:
        """Free GPU/RAM held by the open single-view full-res decode before gallery tiles load."""
        import logging
        import sys

        from PyQt6.QtGui import QPixmap

        logger = logging.getLogger(__name__)
        cur = getattr(self, "current_file_path", None)

        self._cancel_content_crossfade()
        self._full_resolution_loading = False

        gv = getattr(self, "gpu_view", None)
        gl_cleared = False
        gl_deferred = False
        gl_pending = False
        if gv is not None:
            try:
                if hasattr(gv, "release_for_gallery_entry"):
                    hidden = (
                        hasattr(gv, "_viewport_hidden_for_teardown")
                        and gv._viewport_hidden_for_teardown()
                    )
                    had_heavy = (
                        hasattr(gv, "has_heavy_pixmap") and gv.has_heavy_pixmap()
                    )
                    gv.release_for_gallery_entry()
                    if sys.platform == "win32" and gv.is_opengl_viewport():
                        if hidden and not had_heavy:
                            gl_cleared = True
                        elif hidden and had_heavy:
                            gl_cleared = not gv.has_heavy_pixmap()
                            gl_pending = gv.has_heavy_pixmap()
                        else:
                            gl_deferred = True
                            gl_pending = had_heavy
                    else:
                        gl_cleared = True
                else:
                    gv.clear()
                    gl_cleared = True
            except Exception:
                pass

        label = getattr(self, "image_label", None)
        if label is not None:
            with contextlib.suppress(Exception):
                label.setPixmap(QPixmap())
        self.current_pixmap = None
        self._manager_displayed_max_dim = 0
        self._gpu_last_path = None

        cache = getattr(self, "image_cache", None)
        if cur and cache is not None and hasattr(cache, "release_full_image_memory"):
            with contextlib.suppress(Exception):
                cache.release_full_image_memory(cur)

        logger.info(
            "[GALLERY] Released single-view full-res buffers for gallery entry (%s, gl_cleared=%s gl_deferred=%s gl_pending=%s)",
            os.path.basename(cur) if cur else "none",
            gl_cleared,
            gl_deferred,
            gl_pending,
        )

    def _confirm_bulk_deletion(self, count: int, *, discard: bool) -> bool:
        if count <= 0:
            return False
        if discard:
            title = "Confirm Move to Discard"
            verb = "move"
            detail = "They will be moved to a Discard folder next to the originals."
        else:
            title = "Confirm Delete"
            verb = "delete"
            detail = "They will be moved to the Recycle Bin."
        noun = "file" if count == 1 else "files"
        dialog = CustomConfirmDialog(
            parent=self,
            title=title,
            message=f"Are you sure you want to {verb} {count} {noun}?",
            informative_text=detail,
            confirm_action="Discard" if discard else "Delete",
        )
        self._is_delete_dialog_open = True
        try:
            dialog.exec()
            return dialog.result_value
        finally:
            self._is_delete_dialog_open = False

    def _sanitize_dialog_start_dir(self, last_dir) -> str:
        """Return an existing directory for QFileDialog, or empty for default location."""
        path = (last_dir or "").strip()
        if not path:
            return ""
        if os.path.isdir(path):
            return path
        parent = os.path.dirname(path)
        if parent and os.path.isdir(parent):
            return parent
        return ""

    def _use_qt_file_dialog(self) -> bool:
        """Use the native system file picker by default (Windows Explorer-style dialog).

        The cross-platform Qt (DontUseNativeDialog) picker is an OPT-IN fallback:
        set RAWVIEWER_QT_FILE_DIALOG=1 if the native Windows dialog ever crashes
        on a given machine (some shell preview/thumbnail handlers or cloud-storage
        overlays can abort it when it opens over the OpenGL view).
        """
        raw = os.environ.get("RAWVIEWER_QT_FILE_DIALOG", "").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _prepare_for_modal_dialog(self) -> None:
        """Bring the main window forward so native file dialogs are not hidden behind it."""
        try:
            _activate_macos_foreground_app()
            app = _qt_app()
            if app is not None:
                app.setActiveWindow(self)
            self.raise_()
            self.activateWindow()
            if app is not None:
                app.processEvents()
        except Exception:
            pass

    def _open_file_dialog_macos_applescript(self, last_dir: str) -> str:
        """Native macOS Finder file picker via AppleScript (works with Qt + Terminal)."""
        types = [
            ext.lstrip(".").lower()
            for ext in self.get_supported_extensions()
            if ext
        ]
        type_clause = ""
        if types:
            type_list = ", ".join(f'"{_applescript_escape(t)}"' for t in types)
            type_clause = f" of type {{{type_list}}}"

        start = self._sanitize_dialog_start_dir(last_dir)
        location_clause = ""
        if start:
            location_clause = (
                f' default location (POSIX file "{_applescript_escape(start)}")'
            )

        # A bare top-level "activate" (no tell-application context) does not
        # reliably bring this dialog forward -- confirmed live: the osascript
        # process was up and running with "activate" in the script, but the
        # picker stayed invisible (no crash, no timeout, just never seen).
        # Running "choose file" inside an explicit `tell application "Finder"`
        # block makes Finder itself own + activate the dialog, which is the
        # reliable form of this trick (Finder is always a proper foreground
        # GUI app, unlike the bare osascript process).
        script = (
            'tell application "Finder"\n'
            "    activate\n"
            f'    choose file{type_clause} with prompt "Open Image"{location_clause}\n'
            "end tell\n"
            "return POSIX path of result"
        )
        return _run_applescript(script)

    def _open_folder_dialog_macos_applescript(self, last_dir: str) -> str:
        start = self._sanitize_dialog_start_dir(last_dir)
        location_clause = ""
        if start:
            location_clause = (
                f' default location (POSIX file "{_applescript_escape(start)}")'
            )
        script = (
            'tell application "Finder"\n'
            "    activate\n"
            f'    choose folder with prompt "Open Folder"{location_clause}\n'
            "end tell\n"
            "return POSIX path of result"
        )
        return _run_applescript(script)

    def _open_file_dialog_macos_native(self, last_dir: str) -> str:
        """Prefer NSOpenPanel; fall back to AppleScript (true Finder UI under Qt)."""
        import logging

        logger = logging.getLogger(__name__)
        if not self._macos_nsopenpanel_reliable():
            logger.info(
                "Skipping NSOpenPanel (unbundled dev launch), using AppleScript choose file"
            )
            return self._open_file_dialog_macos_applescript(last_dir)
        try:
            return self._open_file_dialog_macos_nsopenpanel(last_dir)
        except Exception as exc:
            logger.warning(
                "NSOpenPanel unavailable (%s), using AppleScript choose file",
                exc,
            )
        return self._open_file_dialog_macos_applescript(last_dir)

    def _open_folder_dialog_macos_native(self, last_dir: str) -> str:
        import logging

        logger = logging.getLogger(__name__)
        if not self._macos_nsopenpanel_reliable():
            logger.info(
                "Skipping NSOpenPanel (unbundled dev launch), using AppleScript choose folder"
            )
            return self._open_folder_dialog_macos_applescript(last_dir)
        try:
            return self._open_folder_dialog_macos_nsopenpanel(last_dir)
        except Exception as exc:
            logger.warning(
                "NSOpenPanel unavailable (%s), using AppleScript choose folder",
                exc,
            )
        return self._open_folder_dialog_macos_applescript(last_dir)

    def _open_file_dialog_macos_nsopenpanel(self, last_dir: str) -> str:
        """macOS Finder open panel via AppKit (Qt native QFileDialog fails when run from Terminal)."""
        from AppKit import NSURL

        panel = self._create_macos_open_panel()
        panel.setTitle_("Open Image")
        panel.setPrompt_("Open")
        panel.setCanChooseFiles_(True)
        panel.setCanChooseDirectories_(False)
        panel.setAllowsMultipleSelection_(False)

        types = [ext.lstrip(".").lower() for ext in self.get_supported_extensions()]
        if types:
            panel.setAllowedFileTypes_(types)

        start = self._sanitize_dialog_start_dir(last_dir)
        if start:
            panel.setDirectoryURL_(NSURL.fileURLWithPath_(start))

        if panel.runModal() != 1:  # NSModalResponseOK
            return ""
        url = panel.URL()
        return url.path() if url is not None else ""

    def _open_folder_dialog_macos_nsopenpanel(self, last_dir: str) -> str:
        from AppKit import NSURL

        panel = self._create_macos_open_panel()
        panel.setTitle_("Open Folder")
        panel.setPrompt_("Open")
        panel.setCanChooseFiles_(False)
        panel.setCanChooseDirectories_(True)
        panel.setAllowsMultipleSelection_(False)

        start = self._sanitize_dialog_start_dir(last_dir)
        if start:
            panel.setDirectoryURL_(NSURL.fileURLWithPath_(start))

        if panel.runModal() != 1:
            return ""
        url = panel.URL()
        return url.path() if url is not None else ""

    def _save_file_dialog_macos_applescript(
        self, default_dir: str, default_name: str, prompt: str
    ) -> str:
        """Native macOS Finder save picker via AppleScript (works with Qt + Terminal)."""
        start = self._sanitize_dialog_start_dir(default_dir)
        location_clause = ""
        if start:
            location_clause = (
                f' default location (POSIX file "{_applescript_escape(start)}")'
            )
        name_clause = ""
        if default_name:
            name_clause = f' default name "{_applescript_escape(default_name)}"'
        script = (
            'tell application "Finder"\n'
            "    activate\n"
            f'    choose file name with prompt "{_applescript_escape(prompt)}"'
            f"{name_clause}{location_clause}\n"
            "end tell\n"
            "return POSIX path of result"
        )
        return _run_applescript(script)

    def _save_file_dialog_macos_nssavepanel(
        self,
        default_dir: str,
        default_name: str,
        prompt: str,
        file_types: list,
    ) -> str:
        """macOS Finder save panel via AppKit (Qt native QFileDialog fails when run from Terminal)."""
        from AppKit import NSURL

        panel = self._create_macos_save_panel()
        panel.setTitle_(prompt)
        panel.setPrompt_("Export")
        if default_name:
            panel.setNameFieldStringValue_(default_name)
        if file_types:
            with contextlib.suppress(Exception):
                panel.setAllowedFileTypes_(file_types)

        start = self._sanitize_dialog_start_dir(default_dir)
        if start:
            panel.setDirectoryURL_(NSURL.fileURLWithPath_(start))

        if panel.runModal() != 1:  # NSModalResponseOK
            return ""
        url = panel.URL()
        if url is None:
            return ""
        # Defensive: keep security-scoped access alive for the chosen file URL
        # (harmless when not sandboxed; required if TCC treats the pick specially).
        with contextlib.suppress(Exception):
            url.startAccessingSecurityScopedResource()
        try:
            return str(url.path() or "")
        finally:
            # Export writes asynchronously on a worker thread; do not stop
            # access here. macOS releases scope when the process exits; for
            # non-sandboxed apps this is a no-op either way.
            pass

    def _save_file_dialog_macos_native(
        self,
        default_dir: str,
        default_name: str,
        prompt: str,
        file_types: list,
    ) -> str:
        """Prefer NSSavePanel; fall back to AppleScript (true Finder UI under Qt)."""
        import logging

        logger = logging.getLogger(__name__)
        if not self._macos_nsopenpanel_reliable():
            logger.info(
                "Skipping NSSavePanel (unbundled dev launch), using AppleScript choose file name"
            )
            return self._save_file_dialog_macos_applescript(default_dir, default_name, prompt)
        try:
            return self._save_file_dialog_macos_nssavepanel(
                default_dir, default_name, prompt, file_types
            )
        except Exception as exc:
            logger.warning(
                "NSSavePanel unavailable (%s), using AppleScript choose file name",
                exc,
            )
        return self._save_file_dialog_macos_applescript(default_dir, default_name, prompt)

    def _choose_macos_editing_app_applescript(self, start_dir: str) -> str:
        """Native app picker via AppleScript (works when Qt owns NSApplication)."""
        start = self._sanitize_dialog_start_dir(start_dir) or "/Applications"
        location_clause = (
            f' default location (POSIX file "{_applescript_escape(start)}")'
        )
        script = (
            'tell application "Finder"\n'
            "    activate\n"
            '    choose file of type {"com.apple.application-bundle", "app"} '
            f'with prompt "Choose Editing App"{location_clause}\n'
            "end tell\n"
            "return POSIX path of result"
        )
        return _run_applescript(script)

    def _choose_macos_editing_app_nsopenpanel(self, start_dir: str) -> str:
        """macOS app bundle picker via AppKit."""
        from AppKit import NSURL

        panel = self._create_macos_open_panel()
        panel.setTitle_("Choose Editing App")
        panel.setPrompt_("Choose")
        panel.setCanChooseFiles_(True)
        panel.setCanChooseDirectories_(True)
        panel.setAllowsMultipleSelection_(False)
        panel.setTreatsFilePackagesAsDirectories_(True)
        with contextlib.suppress(Exception):
            panel.setAllowedFileTypes_(["app"])

        start = self._sanitize_dialog_start_dir(start_dir) or "/Applications"
        panel.setDirectoryURL_(NSURL.fileURLWithPath_(start))

        if panel.runModal() != 1:
            return ""
        url = panel.URL()
        return url.path() if url is not None else ""

    def _choose_macos_editing_app_dialog(self, start_dir: str = "/Applications") -> str:
        """Pick a .app bundle; Qt QFileDialog fails on macOS when launched from Terminal."""
        import logging

        logger = logging.getLogger(__name__)
        if self._use_qt_file_dialog():
            self._prepare_for_modal_dialog()
            app_path, _ = QFileDialog.getOpenFileName(
                self,
                "Choose Editing App",
                start_dir,
                "Applications (*.app);;All Files (*)",
                options=QFileDialog.Option.DontUseNativeDialog,
            )
            return app_path or ""
        if not self._macos_nsopenpanel_reliable():
            logger.info(
                "Skipping NSOpenPanel app chooser (unbundled dev launch), using AppleScript"
            )
            return self._choose_macos_editing_app_applescript(start_dir)
        try:
            return self._choose_macos_editing_app_nsopenpanel(start_dir)
        except Exception as exc:
            logger.warning(
                "NSOpenPanel app chooser unavailable (%s), using AppleScript",
                exc,
            )
        return self._choose_macos_editing_app_applescript(start_dir)

    def _pick_directory_dialog(self, title: str, start_dir: str = "") -> str:
        """Native Explorer-style folder picker (Qt + macOS fallbacks). '' on cancel.

        Same platform treatment open_folder used so opening a modal from a
        terminal-launched process on macOS never silently returns ''.
        """
        if self._use_qt_file_dialog():
            return QFileDialog.getExistingDirectory(
                self,
                title,
                start_dir or "",
                options=QFileDialog.Option.DontUseNativeDialog,
            )
        if sys.platform == "darwin":
            return self._open_folder_dialog_macos_native(start_dir or "")
        self._prepare_for_modal_dialog()
        return QFileDialog.getExistingDirectory(self, title, start_dir or "")

    def image_mouse_release_event(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self.panning = False
            if self.current_pixmap and not self.fit_to_window:
                # Update zoom_center_point to reflect current viewport center after panning
                # This ensures that when navigating, the zoom area is preserved correctly
                # Calculate current viewport center in image coordinates
                viewport_size = self.scroll_area.viewport().size()
                scroll_x = self.scroll_area.horizontalScrollBar().value()
                scroll_y = self.scroll_area.verticalScrollBar().value()
                # Viewport center in image coordinates
                viewport_center_x = scroll_x + viewport_size.width() // 2
                viewport_center_y = scroll_y + viewport_size.height() // 2
                # Update zoom_center_point to current viewport center
                self.zoom_center_point = QPoint(viewport_center_x, viewport_center_y)
                self.image_label.setCursor(
                    QCursor(Qt.CursorShape.OpenHandCursor))
            elif self.fit_to_window:
                self.image_label.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        # Always ensure main window has focus for keyboard events
        self.setFocus()

    def on_memory_warning(self, memory_percent):
        """Handle memory warning events."""
        safe_print(f"⚠️ Memory usage high: {memory_percent:.1f}%")

    def _wb_linear_sample_buffer(self, adj: dict | None):
        """As-shot linear RGB in the same spatial frame as the Adjust display.

        Edit base is pre-geometry; the on-screen preview is post-geometry
        (unless crop overlay is in uncropped preview). Sample from the
        geometry-applied linear buffer so click coords match what the user sees.
        """
        base = getattr(self, "_adjust_preview_base_rgb", None)
        if base is None or not hasattr(base, "shape"):
            return None
        from raw_adjustments.pipeline import _linear_float_from_buffer
        from raw_adjustments.transform import apply_geometry

        img = _linear_float_from_buffer(base)
        geo = dict(adj or {})
        if getattr(self, "_crop_preview_uncropped", False):
            # Crop overlay zeroes insets but still shows straighten/keystone.
            for k in ("CropLeft", "CropRight", "CropTop", "CropBottom"):
                geo[k] = 0.0
        return apply_geometry(img, geo)

    def _on_adjust_wb_picker_toggled(self, checked: bool) -> None:
        """Arm/cancel the WB dropper (GPU single-image view only, RAW files only)."""
        panel = getattr(self, "single_image_adjust_panel", None)
        gv = getattr(self, "gpu_view", None)
        if panel is None or gv is None:
            return
        if not checked:
            gv.set_color_pick_mode(False)
            return
        path = getattr(self, "current_file_path", None)
        from common_image_loader import is_raw_file

        base = getattr(self, "_adjust_preview_base_rgb", None)
        is_raw = bool(path and is_raw_file(path))
        has_base = base is not None and hasattr(base, "shape")
        if not path or not is_raw or not has_base:
            panel.set_wb_picker_active(False)
            if hasattr(self, "status_bar"):
                self.status_bar.showMessage(
                    "White balance dropper needs a decoded RAW edit base -- try again in a moment",
                    3500,
                )
            return
        # Settle to edit-base framing before the click so lite 640px coords
        # cannot be treated as half-res indices; also blocks lite paints
        # while armed via _should_defer_adjust_preview_paint.
        self._apply_adjust_panel_preview(full_quality=True)
        if getattr(self, "_adjust_hires_render", None) is not None:
            self._restore_adjust_hires_before_zoom()
        gv.set_color_pick_mode(True)
        if hasattr(self, "status_bar"):
            self.status_bar.showMessage(
                "Click a neutral gray/white area in the image to set white balance"
                " -- Esc to cancel",
                0,
            )

    def _cancel_wb_picker(self) -> None:
        panel = getattr(self, "single_image_adjust_panel", None)
        gv = getattr(self, "gpu_view", None)
        if gv is not None:
            gv.set_color_pick_mode(False)
        if panel is not None:
            panel.set_wb_picker_active(False)
        if hasattr(self, "status_bar"):
            self.status_bar.clearMessage()

    def _on_wb_color_picked(self, scene_pt) -> None:
        """Sample as-shot linear RGB under the click and solve Temperature/Tint.

        Uses the geometry-applied edit frame (same framing as the display),
        with display→buffer scale mapping so lite/settle size mismatches and
        crop insets do not pick the wrong pixel.
        """
        panel = getattr(self, "single_image_adjust_panel", None)
        if panel is not None:
            panel.set_wb_picker_active(False)
        if panel is None:
            return
        adj = panel.get_adjustments()
        sample = self._wb_linear_sample_buffer(adj)
        if sample is None or getattr(sample, "ndim", 0) != 3:
            if hasattr(self, "status_bar"):
                self.status_bar.showMessage(
                    "White balance dropper needs a decoded RAW edit base -- try again in a moment",
                    3500,
                )
            return
        h, w = int(sample.shape[0]), int(sample.shape[1])
        mx, my = self._map_adjust_display_point_to_buffer(scene_pt, w, h)
        cx = int(round(mx))
        cy = int(round(my))
        if not (0 <= cx < w and 0 <= cy < h):
            if hasattr(self, "status_bar"):
                self.status_bar.showMessage(
                    "White balance pick was outside the image -- try again",
                    3000,
                )
            return

        from raw_adjustments import solve_white_balance_from_sample, wb_reference_temperature

        radius = 3
        y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
        patch = sample[y0:y1, x0:x1]
        r, g, b = (float(patch[:, :, i].mean()) for i in range(3))

        ref_temp = wb_reference_temperature(adj)
        temperature, tint = solve_white_balance_from_sample(r, g, b, ref_temp)
        panel.apply_picked_white_balance(temperature, tint)
        if hasattr(self, "status_bar"):
            self.status_bar.showMessage(
                f"White balance set from picked pixel ({int(round(temperature))}K)", 3000
            )
        QTimer.singleShot(0, self._restore_keyboard_focus)

    def _on_auto_wb_requested(self) -> None:
        """Auto WB: robust gray-world estimate on the edit base, solved through
        the same Temperature/Tint inverse the WB dropper uses."""
        panel = getattr(self, "single_image_adjust_panel", None)
        base = getattr(self, "_adjust_preview_base_rgb", None)
        if panel is None or base is None or not hasattr(base, "shape"):
            if hasattr(self, "status_bar"):
                self.status_bar.showMessage("Auto WB needs a decoded RAW edit base -- try again in a moment", 3000)
            return
        from raw_adjustments import solve_white_balance_from_sample, wb_reference_temperature
        from raw_adjustments.auto_adjust import estimate_neutral_rgb
        from raw_adjustments.pipeline import _linear_float_from_buffer

        est = estimate_neutral_rgb(_linear_float_from_buffer(base))
        if est is None:
            if hasattr(self, "status_bar"):
                self.status_bar.showMessage("Auto WB: no reliable neutral estimate in this image", 3000)
            return
        r, g, b = est
        ref_temp = wb_reference_temperature(panel.get_adjustments())
        temperature, tint = solve_white_balance_from_sample(r, g, b, ref_temp)
        panel.apply_picked_white_balance(temperature, tint)
        if hasattr(self, "status_bar"):
            self.status_bar.showMessage(f"Auto white balance ({int(round(temperature))}K)", 3000)

    def _release_share_bottom_button(self) -> None:
        """Clear stuck pressed/highlight state after modal dialogs."""
        btn = getattr(self, "share_bottom_button", None)
        if btn is None:
            return
        try:
            btn.setDown(False)
            btn.clearFocus()
            btn.repaint()
        except Exception:
            pass

    def _open_macos_choose_editing_app(self, paths: list[str]) -> bool:
        app_path = self._choose_macos_editing_app_dialog("/Applications")
        if not app_path:
            return False
        return self._open_with_macos_app_path(paths, app_path)

    def _open_windows_choose_editing_app(self, paths: list[str]) -> bool:
        start_dir = os.environ.get("PROGRAMFILES", "C:\\Program Files")
        exe_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Editing App",
            start_dir,
            "Applications (*.exe);;All Files (*)",
        )
        if not exe_path:
            return False
        return self._open_with_windows_exe_path(paths, exe_path)

    def _macos_share_use_heavy_picker(self) -> bool:
        """Full v2.2 integration (filter pause, delegate). Default off -- caused empty spinner."""
        return _env_true("RAWVIEWER_SHARE_HEAVY_PICKER")

    def _macos_share_begin_picker_session(self) -> None:
        """Pause Qt input hooks during native picker (do not hide GPU image -- pixels live on gpu_view)."""
        self._pause_main_event_filters_for_share()
        self._macos_share_suspend_opengl_layers()

    def _macos_share_end_picker_session(self, reason: str = "unknown") -> None:
        """Restore Qt state after native picker dismisses."""
        self._resume_main_event_filters_for_share(reason)
        self._macos_share_restore_opengl_layers()
        self._macos_share_cached_services = None

    def _macos_share_try_native_picker(self) -> bool:
        return _env_true("RAWVIEWER_SHARE_TRY_NATIVE_PICKER") or _env_true(
            "RAWVIEWER_SHARE_NATIVE_PICKER"
        )

    def _share_macos_picker_ui_for_url(self, urls) -> None:
        """Show NSSharingServicePicker for prepared NSURL(s) (AirDrop-only experiments)."""
        items = _macos_share_items_array(urls)
        if items is None:
            _share_log(logging.WARNING, "AirDrop picker skipped: empty items array")
            return
        self._macos_share_begin_picker_session()
        try:
            from AppKit import NSSharingService, NSSharingServicePicker

            if not self._macos_share_ensure_appkit_context(require_window=True):
                self._macos_share_end_picker_session("picker_fail")
                return
            self._macos_share_cached_services = (
                NSSharingService.sharingServicesForItems_(items) or []
            )
            if not self._macos_share_on_main_thread():
                _share_log(logging.ERROR, "refusing AirDrop picker: not on Qt GUI thread")
                self._macos_share_end_picker_session("picker_fail")
                return
            view, rect, anchor_label = self._macos_share_picker_anchor_view_and_rect()
            if view is None or rect is None:
                _share_log(logging.WARNING, "no NSView anchor for AirDrop picker")
                self._macos_share_end_picker_session("picker_fail")
                return
            delegate_cls = _get_macos_share_picker_delegate_class()
            delegate = delegate_cls.alloc().initWithViewer_(self)
            picker = NSSharingServicePicker.alloc().initWithItems_(items)
            self._macos_share_picker_ref = picker
            self._macos_share_url_ref = urls
            self._macos_share_picker_delegate_ref = delegate
            if delegate is not None:
                picker.setDelegate_(delegate)
            preferred_edge = 3
            _share_log(
                logging.INFO,
                "AirDrop picker show anchor=%s edge=%s",
                anchor_label,
                preferred_edge,
            )
            picker.showRelativeToRect_ofView_preferredEdge_(rect, view, preferred_edge)
            self._macos_share_pump_events(1)
            QTimer.singleShot(
                120000,
                lambda: (
                    self._macos_share_end_picker_session("safety_timeout"),
                    self._macos_cleanup_share_temp(),
                ),
            )
        except Exception as exc:
            _share_log(logging.WARNING, "AirDrop picker failed: %s", exc, exc_info=True)
            self._macos_share_end_picker_session("picker_fail")
            self._macos_cleanup_share_temp()

    def _macos_share_picker_anchor_view_and_rect(self):
        """v2.1-style anchor on the window contentView or button (stable for popover content)."""
        import objc
        from AppKit import NSMakeRect

        if _env_true("RAWVIEWER_SHARE_ANCHOR_BUTTON"):
            btn = getattr(self, "share_bottom_button", None)
            if btn is not None and btn.isVisible():
                ptr = int(btn.winId())
                if ptr:
                    try:
                        view = objc.objc_object(c_void_p=ptr)
                        w = max(1, btn.width())
                        h = max(1, btn.height())
                        rect = NSMakeRect(0, 0, w, h)
                        return view, rect, "qt.share_button"
                    except Exception as exc:
                        _share_log(logging.DEBUG, "share button anchor: %s", exc)
        return self._macos_share_content_view_and_rect()

    def _share_macos_picker_ui(self, paths: list[str]) -> bool:
        """Native NSSharingServicePicker -- minimal v2.1 path by default."""
        if not self._macos_share_use_heavy_picker():
            if self._share_macos_minimal_v21(paths):
                self.status_bar.showMessage("Share", 1500)
                return True
            _share_log(logging.WARNING, "minimal v2.1 picker failed; trying heavy integration")

        self._macos_share_begin_picker_session()
        ok = self._share_macos(paths)
        if ok:
            _share_log(logging.INFO, "NSSharingServicePicker shown OK (heavy; session ends on dismiss)")
            self.status_bar.showMessage("Share", 1500)
            QTimer.singleShot(120000, lambda: (
                self._macos_share_end_picker_session("safety_timeout"),
                self._macos_cleanup_share_temp(),
            ))
            return True
        self._macos_share_end_picker_session("picker_fail")
        self._macos_cleanup_share_temp()
        _share_log(logging.WARNING, "NSSharingServicePicker failed for %s", paths)
        return False
