
"""external mixin, part 1 of 2 (48 methods, 882 lines)."""
from __future__ import annotations

import contextlib
import logging
import os
import sys
import time

from PyQt6.QtCore import QPoint, QTimer
from PyQt6.QtGui import QAction, QCursor, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import QApplication, QMenu

from core.config import _env_true, _find_file_index_in_list
from rawviewer_app.core.widgets import CustomWarningDialog

from .. import lazy
from ..logging_setup import _share_log
from ..macos import _macos_share_items_array
from ..windows_shell import (
    _open_windows_open_with_dialog,
    _share_windows_clipboard_cf_hdrop,
    _share_windows_clipboard_cf_hdrop_paths,
    _share_windows_clipboard_file_via_powershell,
    _share_windows_via_helper,
    _share_windows_via_winrt,
)

lazy.register(sys.modules[__name__])


class ExternalPart1:
    def _share_button_platform_enabled(self) -> bool:
        """Bottom action opens selected image(s) in another app on desktop platforms."""
        return sys.platform in ("darwin", "win32")

    def _sync_share_button_visibility(self) -> None:
        btn = getattr(self, "share_bottom_button", None)
        if btn is None:
            return
        if not self._share_button_platform_enabled():
            btn.hide()
            return
        if getattr(self, "view_mode", "single") == "compare":
            btn.hide()
            return
        if self._adjust_panel_active():
            btn.hide()
            return
        vis = bool(self._gallery_share_target_paths())
        if getattr(self, "view_mode", "single") == "single" and not getattr(
            self, "image_files", None
        ):
            vis = False
        if btn.isVisible() != vis:
            btn.setVisible(vis)

    def _gallery_share_target_paths(self) -> list[str]:
        """Paths to open/share: gallery multi-select, else all visible bookmarked when filter on; single view uses current file."""
        vm = getattr(self, "view_mode", "")
        if vm == "gallery":
            if self._gallery_has_selection():
                paths = self._gallery_selected_canonical_paths()
            elif getattr(self, "_gallery_bookmark_filter_active", False):
                paths = self._gallery_files_for_display()
            else:
                paths = []
        else:
            p = getattr(self, "current_file_path", None)
            paths = [p] if p else []
        return [
            os.path.abspath(p)
            for p in paths
            if p and os.path.isfile(p)
        ]

    def _open_folder_in_file_manager(self, folder: str) -> None:
        """Open a folder in the OS file manager ('Open destination when finished')."""
        if not folder or not os.path.isdir(folder):
            return
        import subprocess

        try:
            if sys.platform == "win32":
                os.startfile(folder)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", folder], check=False)
            else:
                subprocess.run(["xdg-open", folder], check=False)
        except Exception:
            pass

    def _create_macos_open_panel(self):
        """Return an NSOpenPanel instance (can fail when Qt owns NSApplication).

        A ``nil`` result here (not an exception) means the window-server /
        open-and-save-panel XPC service refused to hand back a panel --
        which it can do for a process that isn't NSApplicationActivationPolicy
        Regular / frontmost, the default state for a bare ``python3 main.py``
        Terminal launch. A forced-foreground retry was tried here and made
        things worse: forcing activation mid-attempt causes the retried
        alloc().init() to hang indefinitely (observed via py-spy: stuck
        inside this call after activation, never returning nil or raising)
        instead of the plain attempt's fast nil-return -- likely because
        forcing activation kicks off real window-server/XPC machinery that
        needs the run loop pumping, which nothing does on this synchronous
        Qt-thread call stack. Fail fast on the first nil instead so the
        (working) AppleScript fallback in ``_open_file_dialog_macos_native``
        takes over quickly.
        """
        import objc

        ns_open_panel = objc.lookUpClass("NSOpenPanel")
        panel = ns_open_panel.alloc().init()
        if panel is None:
            panel = ns_open_panel.openPanel()
        if panel is None:
            raise RuntimeError("NSOpenPanel could not be created")
        return panel

    def _accept_drag_urls(self, mime_data) -> bool:
        """True when mime data contains any local folder or file, permitting the drop to show warning pops."""
        if mime_data is None or not mime_data.hasUrls():
            return False
        try:
            from rawviewer_ui.widgets import is_rawviewer_export_drag

            if is_rawviewer_export_drag(mime_data):
                return False
        except Exception:
            pass
        return any(url.isLocalFile() for url in mime_data.urls())

    def _process_dropped_urls(self, mime_data) -> bool:
        """Open a dropped folder or image file. Shows a styled alert and stays on page if unsupported."""
        if mime_data is None or not mime_data.hasUrls():
            return False
        try:
            from rawviewer_ui.widgets import is_rawviewer_export_drag

            if is_rawviewer_export_drag(mime_data):
                return False
        except Exception:
            pass
        exts = self.get_supported_extensions()
        for url in mime_data.urls():
            if not url.isLocalFile():
                continue
            file_path = url.toLocalFile()

            if os.path.isdir(file_path):
                # One shared check (incl. the subfolder fallback) so this can
                # never disagree with the scan load_folder_images will run.
                has_supported_images = self._folder_has_supported_images(file_path)

                if has_supported_images:
                    self.load_folder_images(file_path)
                else:
                    dialog = CustomWarningDialog(
                        self,
                        title="No Supported Images",
                        message=f"The folder '{os.path.basename(file_path)}' does not contain any supported images.",
                        informative_text=f"Supported formats: {', '.join(exts)}"
                    )
                    dialog.exec()
                return True

            if os.path.isfile(file_path):
                ext = os.path.splitext(file_path)[1].lower()
                if ext in exts:
                    folder_path = os.path.dirname(file_path)
                    filename = os.path.basename(file_path)
                    self.load_folder_images(
                        folder_path, start_file=filename, start_view="single"
                    )
                    return True
                else:
                    dialog = CustomWarningDialog(
                        self,
                        title="Unsupported File Format",
                        message=f"The file '{os.path.basename(file_path)}' is not supported by RAWviewer.",
                        informative_text=f"Supported formats: {', '.join(exts)}"
                    )
                    dialog.exec()
                    return True
        return False

    def _register_file_drop_target(self, widget) -> None:
        """Enable drag-and-drop on a child widget and route events via eventFilter."""
        if widget is None:
            return
        targets = getattr(self, "_file_drop_targets", None)
        if targets is None:
            self._file_drop_targets = set()
            targets = self._file_drop_targets
        if widget in targets:
            return
        targets.add(widget)
        widget.setAcceptDrops(True)
        widget.installEventFilter(self)

    def _setup_file_drop_targets(self) -> None:
        """Child widgets stacked over the main window must accept drops themselves."""
        self._register_file_drop_target(self.centralWidget())
        self._register_file_drop_target(getattr(self, "single_view_container", None))
        self._register_file_drop_target(getattr(self, "scroll_area", None))
        sa = getattr(self, "scroll_area", None)
        if sa is not None:
            self._register_file_drop_target(sa.viewport())
        self._register_file_drop_target(getattr(self, "image_label", None))
        self._register_file_drop_target(getattr(self, "gpu_view", None))
        gv = getattr(self, "gpu_view", None)
        if gv is not None:
            self._register_file_drop_target(gv.viewport())
        gs = getattr(self, "gallery_scroll", None)
        if gs is not None:
            self._register_file_drop_target(gs)
            self._register_file_drop_target(gs.viewport())

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter events for file and folder dropping"""
        if self._accept_drag_urls(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        """Handle drop events for file and folder dropping"""
        if self._process_dropped_urls(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def _on_looks_drop_rejected(self, message: str) -> None:
        msg = (message or "").strip() or (
            "Only .cube and .xmp files can be added to Looks"
        )
        self._show_auto_dismiss_notice(
            "Unsupported file",
            detail=msg,
            duration_ms=3500,
        )
        if hasattr(self, "status_bar") and self.status_bar:
            self.status_bar.showMessage(msg, 4000)

    def _open_slideshow_image(self, path: str) -> None:
        idx = _find_file_index_in_list(self.image_files, path, default=-1)
        if idx >= 0:
            self.current_file_index = idx
        self.current_file_path = path
        self.load_raw_image(path)

    def _on_share_bottom_button_clicked(self):
        """Open/share gallery multi-selection, filtered bookmarks, or the current file in single view."""
        if getattr(self, "view_mode", "") == "compare":
            return
        try:
            paths = self._gallery_share_target_paths()
            _share_log(
                logging.INFO,
                "share button clicked view_mode=%s paths=%d visible=%s frozen=%s",
                getattr(self, "view_mode", "?"),
                len(paths),
                getattr(getattr(self, "share_bottom_button", None), "isVisible", lambda: None)(),
                getattr(sys, "frozen", False),
            )
            if paths:
                _share_log(logging.INFO, "targets: %s", paths[:3] if len(paths) > 3 else paths)
            if not paths:
                _share_log(logging.WARNING, "no share targets (gallery selection or bookmark filter)")
                self.status_bar.showMessage("No images selected to share", 2500)
                return
            self._dispatch_share_bottom(list(paths))
        finally:
            self._release_share_bottom_button()

    def _dispatch_share_bottom(self, paths: list[str]) -> None:
        _share_log(logging.INFO, "share dispatch paths=%d platform=%s", len(paths), sys.platform)
        valid = self._valid_open_target_paths(list(paths))
        if not valid:
            self.status_bar.showMessage("No file to share", 2000)
            return
        if len(valid) == 1 and sys.platform == "win32":
            self._open_image_with_app(valid[0])
            return
        self._show_gallery_open_share_menu(valid)

    def _exec_share_anchor_menu(self, menu) -> None:
        btn = getattr(self, "share_bottom_button", None)
        pos = QCursor.pos()
        if btn is not None and btn.isVisible():
            pos = btn.mapToGlobal(QPoint(0, btn.height()))
        menu.exec(pos)

    def _show_gallery_open_share_menu(self, paths: list[str]) -> None:
        """Open in another app and/or share for one or more gallery targets."""
        paths = list(paths)
        n = len(paths)
        menu = QMenu(self)

        if sys.platform == "win32":
            last_exe = str(
                self.get_settings().value(self._windows_editing_app_settings_key(), "") or ""
            )
            if last_exe and os.path.isfile(last_exe):
                last_action = QAction(
                    f"Open with {self._format_app_name(last_exe)}", self
                )
                last_action.triggered.connect(
                    lambda _checked=False, ps=list(paths), exe=last_exe: self._open_with_windows_exe_path(
                        ps, exe
                    )
                )
                menu.addAction(last_action)
            open_other = QAction("Open with other app...", self)
            open_other.triggered.connect(
                lambda _checked=False, ps=list(paths): self._open_windows_choose_editing_app(ps)
            )
            menu.addAction(open_other)
        elif sys.platform == "darwin":
            last_app = str(
                self.get_settings().value(self._macos_editing_app_settings_key(), "") or ""
            )
            if last_app and os.path.isdir(last_app):
                last_action = QAction(
                    f"Open with {self._format_app_name(last_app)}", self
                )
                last_action.triggered.connect(
                    lambda _checked=False, ps=list(paths), app=last_app: self._open_with_macos_app_path(
                        ps, app
                    )
                )
                menu.addAction(last_action)
            open_other = QAction("Open with other app...", self)
            open_other.triggered.connect(
                lambda _checked=False, ps=list(paths): self._open_macos_choose_editing_app(ps)
            )
            menu.addAction(open_other)
        else:
            open_other = QAction("Open with other app...", self)
            open_other.triggered.connect(
                lambda _checked=False, ps=list(paths): self._open_paths_in_editing_app(ps)
            )
            menu.addAction(open_other)

        menu.addSeparator()
        if sys.platform == "darwin":
            share_action = QAction("Share...", self)
            share_action.triggered.connect(
                lambda _checked=False, ps=list(paths): self._share_paths_os(ps)
            )
            menu.addAction(share_action)
        else:
            share_action = QAction("Copy for sharing (Mail, Teams, …)", self)
            share_action.triggered.connect(
                lambda _checked=False, ps=list(paths): self._share_paths_os(ps)
            )
            menu.addAction(share_action)
            if n > 1:
                copy_action = QAction("Copy file paths", self)
                copy_action.triggered.connect(
                    lambda _checked=False, ps=list(paths): (
                        self._copy_paths_to_clipboard(ps),
                        self.status_bar.showMessage("Paths copied to clipboard", 2500),
                    )
                )
                menu.addAction(copy_action)

        self._exec_share_anchor_menu(menu)

    def _valid_open_target_paths(self, paths: list[str]) -> list[str]:
        raw_in = list(paths)
        valid = [os.path.abspath(p) for p in raw_in if p and os.path.isfile(p)]
        if len(raw_in) != len(valid):
            for p in raw_in:
                if p and not os.path.isfile(os.path.abspath(p)):
                    _share_log(logging.WARNING, "skipped missing open-in-app file: %s", p)
        return valid

    def _open_paths_in_editing_app(self, paths: list[str]) -> None:
        """Open one or more RAW/JPEG originals in an external editing app."""
        valid = self._valid_open_target_paths(paths)
        if not valid:
            self.status_bar.showMessage("No file to open", 2000)
            return
        if sys.platform == "darwin":
            self._open_paths_in_macos_editing_app(valid)
        elif sys.platform == "win32":
            self._open_paths_in_windows_editing_app(valid)
        elif len(valid) == 1:
            self._open_image_with_default_app(valid[0])
        else:
            self._copy_paths_to_clipboard(valid)
            self.status_bar.showMessage("Paths copied to clipboard", 2500)

    def _open_image_with_app(self, path: str) -> None:
        """Show the platform Open With/default-app flow for one file."""
        if not path or not os.path.isfile(path):
            self.status_bar.showMessage("No file to open", 2000)
            return
        abs_path = os.path.abspath(path)

        def _show_open_with_dialog():
            if sys.platform == "win32":
                owner = self._share_windows_owner_hwnd()
                ok = _open_windows_open_with_dialog(abs_path, owner_hwnd=owner)
            elif sys.platform == "darwin":
                ok = self._open_macos_choose_editing_app([abs_path])
            else:
                ok = self._open_image_with_default_app(abs_path)
            if ok:
                self.status_bar.showMessage("Choose an app to open this file", 2500)
            else:
                self.status_bar.showMessage("Open With unavailable for this file", 3500)

        QTimer.singleShot(0, _show_open_with_dialog)

    def _copy_paths_to_clipboard(self, paths: list[str]) -> None:
        try:
            from PyQt6.QtCore import QMimeData, QUrl
            mime = QMimeData()
            text_val = "\n".join(os.path.abspath(p) for p in paths)
            mime.setText(text_val)
            urls = [QUrl.fromLocalFile(os.path.abspath(p)) for p in paths]
            mime.setUrls(urls)

            if sys.platform == "darwin":
                try:
                    from AppKit import NSURL, NSPasteboard
                    pb = NSPasteboard.generalPasteboard()
                    pb.clearContents()
                    ns_urls = [NSURL.fileURLWithPath_(os.path.abspath(p)) for p in paths]
                    pb.writeObjects_(ns_urls)
                except Exception:
                    pass

            QApplication.clipboard().setMimeData(mime)
        except Exception:
            with contextlib.suppress(Exception):
                QApplication.clipboard().setText("\n".join(os.path.abspath(p) for p in paths))

    def _open_image_with_default_app(self, path: str) -> bool:
        import subprocess

        try:
            if sys.platform == "darwin":
                subprocess.Popen(["/usr/bin/open", path])
            elif sys.platform == "win32":
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", path])
            return True
        except Exception:
            return False

    def _open_with_macos_app_path(self, paths: list[str], app_path: str) -> bool:
        import subprocess

        app_path = os.path.abspath(app_path)
        if not os.path.isdir(app_path) or not app_path.lower().endswith(".app"):
            return False
        try:
            subprocess.Popen(["/usr/bin/open", "-a", app_path, "--", *paths])
            self.get_settings().setValue(self._macos_editing_app_settings_key(), app_path)
            self.status_bar.showMessage(
                f"Opening {len(paths)} file{'s' if len(paths) != 1 else ''} in {self._format_app_name(app_path)}",
                2500,
            )
            return True
        except Exception as exc:
            _share_log(logging.WARNING, "macOS open in app failed: %s", exc)
            return False

    def _open_with_windows_exe_path(self, paths: list[str], exe_path: str) -> bool:
        import subprocess

        exe_path = os.path.abspath(exe_path)
        if not os.path.isfile(exe_path):
            return False
        try:
            subprocess.Popen([exe_path, *paths], cwd=os.path.dirname(exe_path) or None)
            self.get_settings().setValue(self._windows_editing_app_settings_key(), exe_path)
            self.status_bar.showMessage(
                f"Opening {len(paths)} file{'s' if len(paths) != 1 else ''} in {self._format_app_name(exe_path)}",
                2500,
            )
            return True
        except Exception as exc:
            _share_log(logging.WARNING, "Windows open in app failed: %s", exc)
            return False

    def _open_paths_in_macos_editing_app(self, paths: list[str]) -> None:
        """macOS: open in the last chosen editor, or pick one (no intermediate menu)."""
        last_app = str(self.get_settings().value(self._macos_editing_app_settings_key(), "") or "")
        if last_app and os.path.isdir(last_app):
            self._open_with_macos_app_path(paths, last_app)
            return
        ok = self._open_macos_choose_editing_app(paths)
        if not ok:
            self.status_bar.showMessage("No editing app selected", 2500)

    def _open_paths_in_windows_editing_app(self, paths: list[str]) -> None:
        """Windows: single file uses native Open With; multi-file shows open/share menu."""
        if len(paths) == 1:
            self._open_image_with_app(paths[0])
            return
        self._show_gallery_open_share_menu(paths)

    def _share_paths_os(self, paths: list[str]) -> None:
        """macOS share sheet or Windows share/clipboard for one or more files."""
        raw_in = list(paths)
        paths = [os.path.abspath(p) for p in paths if p and os.path.isfile(p)]
        _share_log(
            logging.INFO,
            "share_paths_os raw=%d valid_files=%d",
            len(raw_in),
            len(paths),
        )
        if len(raw_in) != len(paths):
            for p in raw_in:
                if p and not os.path.isfile(os.path.abspath(p)):
                    _share_log(logging.WARNING, "skipped missing file: %s", p)
        if not paths:
            self.status_bar.showMessage("No file to share", 2000)
            return
        n = len(paths)
        if sys.platform == "darwin":
            self._macos_share_pending_paths = list(paths)
            QTimer.singleShot(100, self._on_share_macos_deferred)
            return
        elif sys.platform == "win32":
            if n == 1:
                self._share_windows_ui_chain(paths[0])
                return
            if _share_windows_clipboard_cf_hdrop_paths(paths):
                self.status_bar.showMessage(
                    f"{n} files copied to clipboard -- paste into Mail, Teams, or other apps",
                    4500,
                )
                return
            if self._share_windows_ui_chain(paths[0]):
                return
        if n == 1:
            self._copy_file_path_to_clipboard(paths[0])
        else:
            joined = "\n".join(paths)
            with contextlib.suppress(Exception):
                QApplication.clipboard().setText(joined)
        hint = (
            "Share failed -- path(s) copied. "
            "Set RAWVIEWER_FILE_LOG=1 and retry; check logs for [SHARE]."
            if sys.platform == "darwin"
            else "Share unavailable -- path(s) copied to clipboard"
        )
        _share_log(logging.WARNING, "fallback clipboard (%s)", hint)
        self.status_bar.showMessage(hint, 6000)

    def _macos_share_suspend_opengl_layers(self) -> None:
        """Optional GPU tweak. Default off: hiding gpu_view blanks the image in GPU single-view mode."""
        if not _env_true("RAWVIEWER_SHARE_HIDE_GPU"):
            return
        gv = getattr(self, "gpu_view", None)
        self._macos_share_saved_gpu_visible = False
        if gv is not None and gv.isVisible():
            self._macos_share_saved_gpu_visible = True
            gv.hide()
            _share_log(logging.INFO, "hid gpu_view (RAWVIEWER_SHARE_HIDE_GPU=1)")
            app = QApplication.instance()
            if app is not None:
                app.processEvents()

    def _macos_share_restore_opengl_layers(self) -> None:
        if not getattr(self, "_macos_share_saved_gpu_visible", False):
            return
        gv = getattr(self, "gpu_view", None)
        if gv is not None:
            gv.show()
        container = getattr(self, "single_view_container", None)
        if container is not None and hasattr(container, "_raise_single_view_layers"):
            try:
                container._raise_single_view_layers()
            except Exception as exc:
                _share_log(logging.DEBUG, "raise single view layers: %s", exc)
        self._macos_share_saved_gpu_visible = False
        _share_log(logging.INFO, "restored gpu_view after share popover")

    def _macos_share_log_thread(self, label: str) -> None:
        import threading

        on_main = self._macos_share_on_main_thread()
        _share_log(
            logging.INFO,
            "%s: qt_gui_thread=%s python_thread=%s",
            label,
            on_main,
            threading.current_thread().name,
        )
        if not on_main:
            _share_log(
                logging.WARNING,
                "AppKit share must run on the Qt GUI thread or macOS may ignore the command",
            )

    def _macos_share_pump_events(self, passes: int = 1) -> None:
        """Light Qt event drain on the GUI thread (avoid nested NSRunLoop -- fights Qt Cocoa bridge)."""
        if not self._macos_share_on_main_thread():
            return
        app = QApplication.instance()
        for _ in range(max(1, passes)):
            if app is not None:
                app.processEvents()

    def _pause_main_event_filters_for_share(self) -> None:
        if getattr(self, "_share_paused_filter_widgets", None):
            _share_log(logging.INFO, "event filters already paused")
            return
        paused = []
        widgets = self._widgets_with_main_mouse_event_filter()
        _share_log(logging.INFO, "pausing event filters on %d widgets", len(widgets))
        for w in widgets:
            try:
                name = type(w).__name__
                w.removeEventFilter(self)
                paused.append(w)
                _share_log(logging.INFO, "paused event filter on widget: %s", name)
            except Exception as exc:
                _share_log(logging.WARNING, "failed to remove event filter: %s", exc)
        self._share_paused_filter_widgets = paused
        overlay_paused = []
        container = getattr(self, "single_view_container", None)
        if container is not None:
            ogl_gv = getattr(container, "_gpu_view", None)
            ogl_vp = ogl_gv.viewport() if ogl_gv is not None else None
            if ogl_vp is not None:
                try:
                    ogl_vp.removeEventFilter(container)
                    overlay_paused.append((container, ogl_vp))
                    _share_log(logging.INFO, "paused SingleImageViewOverlay filter on QOpenGLWidget viewport")
                except Exception as exc:
                    _share_log(logging.DEBUG, "overlay filter pause: %s", exc)
        self._share_paused_overlay_filters = overlay_paused
        _share_log(logging.INFO, "paused event filters on %d widget(s) total for share", len(paused))

    def _resume_main_event_filters_for_share(self, reason: str = "unknown") -> None:
        paused = getattr(self, "_share_paused_filter_widgets", None) or []
        _share_log(logging.INFO, "resuming event filters on %d widget(s) for share (reason: %s)", len(paused), reason)
        for w in paused:
            try:
                name = type(w).__name__
                w.installEventFilter(self)
                _share_log(logging.INFO, "resumed event filter on widget: %s", name)
            except Exception as exc:
                _share_log(logging.WARNING, "failed to install event filter: %s", exc)
        self._share_paused_filter_widgets = []
        for container, ogl_vp in getattr(self, "_share_paused_overlay_filters", None) or []:
            try:
                ogl_vp.installEventFilter(container)
                _share_log(logging.INFO, "resumed SingleImageViewOverlay filter on QOpenGLWidget viewport")
            except Exception as exc:
                _share_log(logging.DEBUG, "overlay filter resume: %s", exc)
        self._share_paused_overlay_filters = []
        _share_log(logging.INFO, "resumed all event filters for share")

    def _share_current_image_os(self):
        """Open the system share sheet (macOS / Windows) for the current file path."""
        p = getattr(self, "current_file_path", None)
        if not p or not os.path.isfile(p):
            self.status_bar.showMessage("No file to share", 2000)
            return
        path = os.path.abspath(p)
        now = time.monotonic()
        last = float(getattr(self, "_macos_share_last_request_ts", 0.0) or 0.0)
        if now - last < 0.45:
            _share_log(logging.INFO, "share request ignored (debounce %.0f ms)", (now - last) * 1000)
            return
        self._macos_share_last_request_ts = now
        self._macos_share_menu_fallback_used = False
        _share_log(logging.INFO, "share requested: %s", path)
        if sys.platform == "darwin":
            # Defer past press/release so AppKit is not invoked on mouseUp (see NSSharingServicePicker).
            self._macos_share_pending_path = path
            delay_ms = 100
            with contextlib.suppress(ValueError):
                delay_ms = max(50, int(os.environ.get("RAWVIEWER_SHARE_DELAY_MS", "100")))
            QTimer.singleShot(delay_ms, self._on_share_macos_deferred)
            return
        elif sys.platform == "win32":
            QTimer.singleShot(0, lambda fp=path: self._open_image_with_app(fp))
            return
        self._copy_current_file_path_to_clipboard()

    def _copy_file_path_to_clipboard(self, path: str) -> None:
        with contextlib.suppress(Exception):
            QApplication.clipboard().setText(os.path.abspath(path))

    def _share_windows_owner_hwnd(self) -> int:
        """Return a usable top-level HWND for anchoring the share UI."""
        try:
            self.raise_()
            self.activateWindow()
            app = QApplication.instance()
            if app is not None:
                app.processEvents()
        except Exception:
            pass
        wh = self.windowHandle()
        if wh is not None:
            try:
                owner = int(wh.winId())
                if owner:
                    return owner
            except Exception:
                pass
        try:
            return int(self.effectiveWinId())
        except Exception:
            return 0

    def _share_windows_ui_chain(self, path: str):
        """Open the Windows share sheet for a file path."""
        owner = self._share_windows_owner_hwnd()
        # Prefer the bundled .NET helper: it uses CsWinRT's MarshalInterface.FromAbi to
        # project the DataTransferManager, which the in-process Python WinRT path cannot do
        # on pywinrt 3.x (raw ABI pointers are not projectable from Python).
        if _share_windows_via_helper(path, owner):
            self.status_bar.showMessage("Share", 1500)
            return
        if _share_windows_via_winrt(path, owner):
            self.status_bar.showMessage("Share", 1500)
            return
        if self._share_windows_shell(path, owner):
            self.status_bar.showMessage("Share", 1500)
            return
        if _share_windows_clipboard_cf_hdrop(path):
            self.status_bar.showMessage(
                "File copied to clipboard -- paste into Mail, Teams, or other apps", 4500
            )
            return
        if _share_windows_clipboard_file_via_powershell(path):
            self.status_bar.showMessage(
                "File copied to clipboard -- paste into Mail, Teams, or other apps", 4500
            )
            return
        self._copy_current_file_path_to_clipboard()
        self.status_bar.showMessage("Share unavailable -- path copied to clipboard", 4000)

    def _macos_share_use_qt_menu(self) -> bool:
        """Qt menu listing NSSharingService targets (works in v2.2 Qt6 shell)."""
        if _env_true("RAWVIEWER_SHARE_TRY_NATIVE_PICKER"):
            return False
        if _env_true("RAWVIEWER_SHARE_NATIVE_PICKER"):
            return False
        return _env_true("RAWVIEWER_SHARE_MENU", default=True)

    def _macos_share_on_main_thread(self) -> bool:
        app = QApplication.instance()
        if app is None:
            return True
        try:
            from PyQt6.QtCore import QThread

            return QThread.currentThread() is app.thread()
        except Exception:
            return True

    def _macos_share_defer_to_main(self, callback) -> None:
        if self._macos_share_on_main_thread():
            callback()
            return
        _share_log(logging.WARNING, "share deferred to Qt main thread (was not on GUI thread)")
        QTimer.singleShot(0, callback)

    def _on_share_macos_deferred(self) -> None:
        """Queued on the MainWindow thread -- all AppKit share UI runs here."""
        paths = getattr(self, "_macos_share_pending_paths", None)
        if not paths:
            path = getattr(self, "_macos_share_pending_path", None)
            paths = [path] if path else []
        self._macos_share_pending_paths = None
        self._macos_share_pending_path = None
        if not paths:
            return
        self._macos_share_log_thread("share deferred slot")
        self._share_macos_ui_impl(paths)

    def _macos_share_log_runtime_context(self) -> None:
        """Diagnostics for sandbox / activation (dev python is usually unsandboxed)."""
        frozen = bool(getattr(sys, "frozen", False))
        _share_log(
            logging.INFO,
            "runtime: frozen=%s main_thread=%s (Terminal dev builds are not App Sandbox by default)",
            frozen,
            self._macos_share_on_main_thread(),
        )

    def _macos_share_validate_file(self, path: str) -> tuple[bool, str, str]:
        """Return (ok, absolute_path, reason). Rejects missing/empty/unreadable items."""
        if not path or not str(path).strip():
            return False, "", "empty_path"
        abs_path = os.path.abspath(path)
        if not os.path.isfile(abs_path):
            return False, abs_path, "not_a_file"
        try:
            size = os.path.getsize(abs_path)
        except OSError as exc:
            return False, abs_path, f"stat_failed:{exc}"
        if size <= 0:
            return False, abs_path, "zero_byte_file"
        if not os.access(abs_path, os.R_OK):
            return False, abs_path, "not_readable"
        return True, abs_path, ""

    def _macos_share_ensure_appkit_context(self, *, require_window: bool = False) -> bool:
        """Activate Qt + NSWindow before AppKit share UI (missing window → empty/spinning picker)."""
        if not self._macos_share_on_main_thread():
            _share_log(logging.WARNING, "AppKit share requires Qt main thread")
            return False
        try:
            self.raise_()
            self.activateWindow()
            wh = self.windowHandle()
            if wh is not None:
                wh.raise_()
        except Exception as exc:
            _share_log(logging.DEBUG, "Qt window activate: %s", exc)

        ns_win = None
        try:
            import objc

            ns_app = objc.lookUpClass("NSApplication").sharedApplication()
            if ns_app is not None:
                # Regular policy -- Terminal-launched Python is often Accessory/Prohibited.
                ns_app.setActivationPolicy_(0)
                ns_win = self._macos_share_ns_window()
                if ns_win is None:
                    ns_win = ns_app.mainWindow() or ns_app.keyWindow()
                if ns_win is not None:
                    ns_win.makeKeyAndOrderFront_(None)
        except Exception as exc:
            _share_log(logging.DEBUG, "NSApplication/window activate: %s", exc)

        app = QApplication.instance()
        if app is not None:
            app.processEvents()

        content_view, _rect, anchor = self._macos_share_content_view_and_rect()
        has_view = content_view is not None
        _share_log(
            logging.INFO,
            "AppKit context: ns_window=%s content_view=%s anchor=%s",
            ns_win is not None,
            has_view,
            anchor,
        )
        if require_window and ns_win is None and not has_view:
            _share_log(logging.WARNING, "missing active NSWindow or contentView for share")
            return False
        return True

    def _macos_share_prepare_url(self, path: str):
        """Return (NSURL, share_path) for AppKit; copies /Volumes/ files to a local temp file."""
        try:
            from AppKit import NSURL
        except Exception as exc:
            _share_log(logging.WARNING, "AppKit unavailable: %s", exc)
            return None, ""

        ok, abs_path, reason = self._macos_share_validate_file(path)
        if not ok:
            _share_log(logging.WARNING, "share file rejected (%s): %s", reason, path)
            return None, ""

        share_path = abs_path
        prev_temp = getattr(self, "_macos_share_temp_path", None)
        if prev_temp:
            try:
                if os.path.exists(prev_temp):
                    os.remove(prev_temp)
            except OSError:
                pass
            self._macos_share_temp_path = None

        if abs_path.startswith("/Volumes/"):
            try:
                import shutil
                import tempfile
                import time

                suffix = os.path.splitext(abs_path)[1] or ".jpg"
                fd, temp_path = tempfile.mkstemp(prefix="rawviewer_share_", suffix=suffix)
                os.close(fd)
                t0 = time.perf_counter()
                shutil.copy2(abs_path, temp_path)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                ok, share_path, reason = self._macos_share_validate_file(temp_path)
                if not ok:
                    _share_log(logging.WARNING, "temp share file invalid (%s): %s", reason, temp_path)
                    return None, ""
                self._macos_share_temp_path = temp_path
                _share_log(
                    logging.INFO,
                    "copied /Volumes/ file to local temp (%.0f ms, %d bytes): %s",
                    elapsed_ms,
                    os.path.getsize(temp_path),
                    temp_path,
                )
            except Exception as exc:
                _share_log(logging.WARNING, "failed to copy /Volumes/ file to temp path: %s", exc)
                return None, ""

        url = NSURL.fileURLWithPath_(share_path)
        if url is None:
            _share_log(logging.WARNING, "NSURL.fileURLWithPath_ returned None for %s", share_path)
            return None, ""
        try:
            if not bool(url.isFileURL()):
                _share_log(logging.WARNING, "share URL is not a file URL: %s", share_path)
                return None, ""
        except Exception:
            pass
        items = _macos_share_items_array(url)
        if items is None:
            _share_log(logging.WARNING, "unsupported or empty share items for %s", share_path)
            return None, ""
        return url, share_path

    def _macos_share_show_airdrop_in_menu(self) -> bool:
        """AirDrop perform UI does not work under Qt host; hidden from menu unless opted in."""
        return _env_true("RAWVIEWER_SHARE_SHOW_AIRDROP")

    def _macos_share_include_service_in_menu(self, service) -> bool:
        if self._macos_share_is_airdrop_service(service):
            return self._macos_share_show_airdrop_in_menu()
        return True

    def _macos_share_is_airdrop_service(self, service) -> bool:
        try:
            title = str(service.title() or "").lower()
            if "airdrop" in title:
                return True
        except Exception:
            pass
        try:
            ident = str(service.valueForKey_("NSSharingServiceName") or "")
            if "AirDrop" in ident or "airdrop" in ident.lower():
                return True
        except Exception:
            pass
        return False
