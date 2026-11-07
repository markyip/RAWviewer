
"""external mixin, part 2 of 2 (21 methods, 690 lines)."""
from __future__ import annotations

import contextlib
import logging
import os
import sys

from PyQt6.QtCore import QPoint, QRect, Qt, QTimer
from PyQt6.QtGui import QAction, QCursor
from PyQt6.QtWidgets import QApplication, QMenu

from core.config import _env_true, _norm_path

from .. import lazy
from ..logging_setup import _share_log
from ..macos import (
    _activate_macos_foreground_app,
    _activate_macos_foreground_for_share,
    _get_macos_share_picker_delegate_class,
    _get_macos_sharing_service_delegate_class,
    _macos_share_items_array,
)
from ..windows_shell import _windows_shell_verb_suggests_share

lazy.register(sys.modules[__name__])


class ExternalPart2:
    def _macos_share_airdrop_finder(self, share_paths) -> None:
        import subprocess

        paths = [share_paths] if isinstance(share_paths, str) else list(share_paths)

        try:
            for p in paths:
                subprocess.run(["open", "-R", os.path.abspath(p)], check=False)
            self.status_bar.showMessage(
                "AirDrop -- in Finder: right-click the file(s) → Share → AirDrop",
                7000,
            )
        except Exception as exc:
            _share_log(logging.WARNING, "Finder AirDrop failed: %s", exc)
            self.status_bar.showMessage(f"AirDrop failed: {exc}", 4000)
        QTimer.singleShot(600000, self._macos_cleanup_share_temp)

    def _macos_share_airdrop_perform_watchdog(self, share_paths) -> None:
        """If performWithItems returns OK but no delegate callback, open Finder."""
        if getattr(self, "_macos_share_airdrop_completed", False):
            return
        _share_log(
            logging.INFO,
            "AirDrop perform produced no panel (Qt host); opening Finder for Share → AirDrop",
        )
        self._macos_share_airdrop_finder(share_paths)

    def _macos_share_resolve_airdrop_service(self, menu_service):
        """Use the dedicated AirDrop send service (menu item object is often not perform-safe)."""
        try:
            from AppKit import NSSharingService

            named = NSSharingService.sharingServiceNamed_("com.apple.share.AirDrop.send")
            if named is not None:
                return named
        except Exception as exc:
            _share_log(logging.DEBUG, "AirDrop named service: %s", exc)
        return menu_service

    def _macos_share_airdrop_via_appkit(self, urls, share_paths) -> None:
        """AirDrop: Finder Share sheet is reliable under Qt; in-app perform is opt-in only."""
        paths = [share_paths] if isinstance(share_paths, str) else list(share_paths)

        if not _env_true("RAWVIEWER_AIRDROP_PERFORM"):
            _share_log(
                logging.INFO,
                "AirDrop: using Finder (NSSharingService.perform does not show UI in Qt host)",
            )
            self._macos_share_airdrop_finder(paths)
            return
        self._macos_share_airdrop_completed = False
        _activate_macos_foreground_for_share()
        try:
            self.raise_()
            self.activateWindow()
            app = QApplication.instance()
            if app is not None:
                app.processEvents()
        except Exception:
            pass
        service = self._macos_share_resolve_airdrop_service(None)
        if service is None or self._macos_share_ns_window() is None:
            self._macos_share_airdrop_finder(paths)
            return
        _share_log(logging.INFO, "AirDrop experimental perform paths=%s", paths)
        self._perform_macos_share_service(
            service, urls, is_airdrop=True, share_paths=paths
        )
        delay_ms = 1500
        with contextlib.suppress(ValueError):
            delay_ms = max(800, int(os.environ.get("RAWVIEWER_AIRDROP_WATCHDOG_MS", "1500")))
        QTimer.singleShot(
            delay_ms, lambda sps=paths: self._macos_share_airdrop_perform_watchdog(sps)
        )

    def _perform_macos_share_service(
        self, service, urls, *, is_airdrop: bool = False, share_paths = None
    ) -> None:
        def run() -> None:
            fallback_paths = [share_paths] if isinstance(share_paths, str) else list(share_paths or [])

            if not fallback_paths and urls:
                try:
                    if isinstance(urls, (list, tuple)):
                        fallback_paths = [str(u.path()) for u in urls]
                    else:
                        fallback_paths = [str(urls.path())]
                except Exception:
                    pass

            if not self._macos_share_ensure_appkit_context(require_window=is_airdrop):
                if is_airdrop and fallback_paths:
                    self._macos_share_airdrop_finder(fallback_paths)
                else:
                    self.status_bar.showMessage("Share unavailable -- no active window", 4000)
                return
            perform_items = _macos_share_items_array(urls)
            if perform_items is None:
                _share_log(logging.WARNING, "performWithItems skipped: empty items array")
                self.status_bar.showMessage("Share failed: no shareable file", 4000)
                if is_airdrop and fallback_paths:
                    self._macos_share_airdrop_finder(fallback_paths)
                else:
                    QTimer.singleShot(0, self._macos_cleanup_share_temp)
                return
            if is_airdrop:
                _activate_macos_foreground_for_share()
            src_win = self._macos_share_ns_window()
            if src_win is None and (is_airdrop or self._macos_share_is_airdrop_service(service)):
                _share_log(logging.WARNING, "perform without NSWindow")
                if is_airdrop and fallback_paths:
                    self._macos_share_airdrop_finder(fallback_paths)
                return
            try:
                delegate_cls = _get_macos_sharing_service_delegate_class()
                delegate = delegate_cls.alloc().initWithViewer_(self)
                self._macos_active_service_delegate = delegate
                if delegate is not None:
                    service.setDelegate_(delegate)
                try:
                    can = service.canPerformWithItems_(perform_items)
                except Exception:
                    can = True
                if not can:
                    _share_log(logging.WARNING, "AirDrop canPerformWithItems=False")
                    if is_airdrop and fallback_paths:
                        self._macos_share_airdrop_finder(fallback_paths)
                    return
                service.performWithItems_(perform_items)
                _share_log(logging.INFO, "performWithItems OK: %s", service)
                if is_airdrop:
                    self.status_bar.showMessage("AirDrop -- choose a recipient", 4000)
            except Exception as exc:
                _share_log(logging.WARNING, "performWithItems failed: %s", exc, exc_info=True)
                self.status_bar.showMessage(f"Share failed: {exc}", 4000)
                if is_airdrop and fallback_paths:
                    self._macos_share_airdrop_finder(fallback_paths)
                else:
                    QTimer.singleShot(0, self._macos_cleanup_share_temp)

        self._macos_share_defer_to_main(run)

    def _share_macos_menu_chose_service(self, service, urls, share_paths) -> None:
        try:
            title = str(service.title())
        except Exception:
            title = "?"
        _share_log(logging.INFO, "share menu chose: %s", title)
        if self._macos_share_is_airdrop_service(service):
            if _env_true("RAWVIEWER_AIRDROP_PICKER"):
                self._share_macos_picker_ui_for_url(urls)
                return
            self._macos_share_airdrop_via_appkit(urls, share_paths)
            return
        self._perform_macos_share_service(service, urls, share_paths=share_paths)

    def _share_macos_services_menu(self, paths: list[str]) -> bool:
        """List NSSharingService targets in a Qt menu (picker sheet often spins empty in v2.2)."""
        try:
            from AppKit import NSSharingService
        except Exception as exc:
            _share_log(logging.WARNING, "share menu unavailable: %s", exc)
            return False

        self._macos_share_log_runtime_context()
        if not self._macos_share_ensure_appkit_context(require_window=False):
            return False

        urls, share_paths = self._macos_share_prepare_urls(paths)
        if not urls:
            return False
        items = _macos_share_items_array(urls)
        if items is None:
            return False
        services = NSSharingService.sharingServicesForItems_(items) or []
        visible = [s for s in services if self._macos_share_include_service_in_menu(s)]
        hidden = len(services) - len(visible)
        if hidden:
            _share_log(
                logging.INFO,
                "share menu: %d service(s) (%d hidden, e.g. AirDrop -- RAWVIEWER_SHARE_SHOW_AIRDROP=1)",
                len(visible),
                hidden,
            )
        else:
            _share_log(logging.INFO, "share menu: %d service(s)", len(visible))
        if not visible:
            self.status_bar.showMessage(
                "No share services available for this file -- use Finder → Share",
                5000,
            )
            return False

        menu = QMenu(self)
        menu.setWindowFlag(Qt.WindowType.Popup, True)
        for svc in visible:
            try:
                title = str(svc.title() or "Share").strip()
            except Exception:
                title = "Share"
            action = QAction(title, self)
            action.triggered.connect(
                lambda _checked=False, s=svc, us=urls, sps=share_paths: self._share_macos_menu_chose_service(
                    s, us, sps
                )
            )
            menu.addAction(action)

        btn = getattr(self, "share_bottom_button", None)
        try:
            self.raise_()
            self.activateWindow()
        except Exception:
            pass
        _activate_macos_foreground_for_share()
        pos = btn.mapToGlobal(QPoint(0, btn.height())) if btn is not None and btn.isVisible() else QCursor.pos()
        # exec (not popup): popup often fails to appear above frameless Qt6 windows on macOS.
        menu.exec(pos)
        return True

    def _macos_share_content_view_and_rect(self):
        """Anchor on NSWindow contentView (Qt status-bar button winId breaks picker populate)."""
        import objc
        from AppKit import NSMakeRect

        btn = getattr(self, "share_bottom_button", None)
        if btn is None:
            return None, None, "no-button"

        def _view_from_ptr(ptr: int, label: str):
            if not ptr:
                return None
            try:
                return objc.objc_object(c_void_p=ptr)
            except Exception as exc:
                _share_log(logging.DEBUG, "%s ptr bridge: %s", label, exc)
                return None

        ns_win = None
        content = None
        label = ""
        for label, ptr in (
            ("qt.windowHandle", int(self.windowHandle().winId()) if self.windowHandle() else 0),
            ("qt.effectiveWinId", int(self.effectiveWinId())),
        ):
            if not ptr:
                continue
            view = _view_from_ptr(ptr, label)
            if view is None:
                continue
            try:
                ns_win = view.window()
            except Exception:
                ns_win = None
            if ns_win is not None:
                try:
                    content = ns_win.contentView()
                except Exception:
                    content = None
                if content is not None:
                    break

        if content is None:
            view = _view_from_ptr(int(btn.winId()), "qt.share_button")
            if view is None:
                return None, None, "no-view"
            rect = NSMakeRect(0, 0, max(1, btn.width()), max(1, btn.height()))
            return view, rect, "qt.share_button"

        try:
            bounds = content.bounds()
            ch = float(bounds.size.height)
        except Exception:
            ch = float(max(1, self.height()))

        pt = btn.mapTo(self, QPoint(0, btn.height()))
        bw = max(1, btn.width())
        bh = max(1, btn.height())
        x = float(pt.x())
        y = ch - float(pt.y())
        rect = NSMakeRect(x, y, bw, bh)
        return content, rect, "ns.contentView"

    def _share_macos_ui(self, path: str) -> None:
        """macOS share entry (prefer scheduling via _on_share_macos_deferred on GUI thread)."""
        self._macos_share_pending_paths = [path]

        def run() -> None:
            self._macos_share_log_thread("share macos ui")
            self._share_macos_ui_impl([path])

        self._macos_share_defer_to_main(run)

    def _share_macos_ui_impl(self, paths: list[str]) -> None:
        if self._macos_share_try_native_picker() and not self._macos_share_use_qt_menu():
            _share_log(
                logging.INFO,
                "trying native NSSharingServicePicker (popover content often spins in v2.2 Qt6 host)",
            )
            if self._share_macos_picker_ui(paths):
                if _env_true("RAWVIEWER_SHARE_AUTO_MENU_FALLBACK", default=True):
                    delay = 900
                    with contextlib.suppress(ValueError):
                        delay = max(400, int(os.environ.get("RAWVIEWER_SHARE_MENU_FALLBACK_MS", "900")))
                    QTimer.singleShot(delay, lambda fps=paths: self._share_macos_auto_menu_fallback(fps))
                return

        if self._macos_share_use_qt_menu():
            _share_log(logging.INFO, "using Qt share menu (same NSSharingService targets as system share)")
            if self._share_macos_services_menu(paths):
                self.status_bar.showMessage("Share", 1500)
                return
            _share_log(logging.WARNING, "share menu failed; trying native picker")
            if self._share_macos_picker_ui(paths):
                return

        self._macos_cleanup_share_temp()
        if paths:
            if len(paths) == 1:
                self._copy_file_path_to_clipboard(paths[0])
            else:
                self._copy_paths_to_clipboard(paths)
        self.status_bar.showMessage("Share unavailable -- path copied to clipboard", 4000)

    def _share_macos_auto_menu_fallback(self, paths: list[str]) -> None:
        """If the native popover spinner never populates, offer the working Qt service menu."""
        if getattr(self, "_macos_share_menu_fallback_used", False):
            return
        self._macos_share_menu_fallback_used = True
        _share_log(
            logging.INFO,
            "auto Qt share menu (NSSharingService list -- popover UI unavailable in this app)",
        )
        if self._share_macos_services_menu(paths):
            self.status_bar.showMessage("Share -- choose a target below", 3500)

    def _macos_share_prepare_urls(self, paths: list[str]):
        """Return (list of NSURL, list of share_paths) for AppKit; copies /Volumes/ files to local temp files."""
        try:
            from AppKit import NSURL
        except Exception as exc:
            _share_log(logging.WARNING, "AppKit unavailable: %s", exc)
            return [], []

        # Cleanup old temp files
        prev_temps = getattr(self, "_macos_share_temp_paths", None)
        if prev_temps:
            for temp in prev_temps:
                try:
                    if os.path.exists(temp):
                        os.remove(temp)
                except OSError:
                    pass
            self._macos_share_temp_paths = []
        else:
            self._macos_share_temp_paths = []

        # Keep original single path cleanup for compatibility
        prev_temp = getattr(self, "_macos_share_temp_path", None)
        if prev_temp:
            try:
                if os.path.exists(prev_temp):
                    os.remove(prev_temp)
            except OSError:
                pass
            self._macos_share_temp_path = None

        urls = []
        share_paths = []

        import shutil
        import tempfile
        import time

        for path in paths:
            ok, abs_path, reason = self._macos_share_validate_file(path)
            if not ok:
                _share_log(logging.WARNING, "share file rejected (%s): %s", reason, path)
                continue

            share_path = abs_path
            if abs_path.startswith("/Volumes/"):
                try:
                    suffix = os.path.splitext(abs_path)[1] or ".jpg"
                    fd, temp_path = tempfile.mkstemp(prefix="rawviewer_share_", suffix=suffix)
                    os.close(fd)
                    t0 = time.perf_counter()
                    shutil.copy2(abs_path, temp_path)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    ok, temp_share_path, reason = self._macos_share_validate_file(temp_path)
                    if not ok:
                        _share_log(logging.WARNING, "temp share file invalid (%s): %s", reason, temp_path)
                        with contextlib.suppress(OSError):
                            os.remove(temp_path)
                        continue
                    self._macos_share_temp_paths.append(temp_path)
                    share_path = temp_path
                    _share_log(
                        logging.INFO,
                        "copied /Volumes/ file to local temp (%.0f ms, %d bytes): %s",
                        elapsed_ms,
                        os.path.getsize(temp_path),
                        temp_path,
                    )
                except Exception as exc:
                    _share_log(logging.WARNING, "failed to copy /Volumes/ file to temp path: %s", exc)
                    continue

            url = NSURL.fileURLWithPath_(share_path)
            if url is not None:
                urls.append(url)
                share_paths.append(share_path)

        return urls, share_paths

    def _share_macos_minimal_v21(self, paths: list[str]) -> bool:
        """v2.1 picker: window contentView anchor, no event-filter pause, no picker delegate."""
        try:
            from AppKit import NSSharingServicePicker

            self._macos_share_log_thread("NSSharingServicePicker minimal v2.1")
            if not self._macos_share_on_main_thread():
                _share_log(logging.ERROR, "refusing minimal picker: not on Qt GUI thread")
                return False

            urls, share_paths = self._macos_share_prepare_urls(paths)
            if not urls:
                return False
            if _env_true("RAWVIEWER_MACOS_FORCE_FOREGROUND") or _env_true(
                "RAWVIEWER_SHARE_TRY_NATIVE_PICKER"
            ):
                _activate_macos_foreground_app()

            btn = getattr(self, "share_bottom_button", None)
            if btn is None:
                return False

            try:
                self.raise_()
                self.activateWindow()
            except Exception:
                pass

            view, rect, anchor_label = self._macos_share_picker_anchor_view_and_rect()
            if view is None or rect is None:
                return False

            items = _macos_share_items_array(urls)
            if items is None:
                return False

            picker = NSSharingServicePicker.alloc().initWithItems_(items)
            self._macos_share_picker_ref = picker
            self._macos_share_urls_ref = urls
            self._macos_share_picker_delegate_ref = None
            picker.showRelativeToRect_ofView_preferredEdge_(rect, view, 3)
            _share_log(
                logging.INFO,
                "minimal v2.1 picker shown anchor=%s paths=%s",
                anchor_label,
                share_paths,
            )
            QTimer.singleShot(300000, self._macos_cleanup_share_temp)
            return True
        except Exception as exc:
            self._macos_share_picker_ref = None
            self._macos_share_urls_ref = None
            _share_log(logging.WARNING, "minimal v2.1 picker failed: %s", exc, exc_info=True)
            return False

    def _macos_share_ns_window(self):
        """NSWindow for the Qt main window (for NSSharingServiceDelegate)."""
        import objc

        for label, ptr in (
            ("qt.windowHandle", int(self.windowHandle().winId()) if self.windowHandle() else 0),
            ("qt.effectiveWinId", int(self.effectiveWinId())),
        ):
            if not ptr:
                continue
            try:
                view = objc.objc_object(c_void_p=ptr)
                win = view.window()
                if win is not None:
                    return win
            except Exception as exc:
                _share_log(logging.DEBUG, "%s window: %s", label, exc)
        try:
            from AppKit import NSApplication

            app = NSApplication.sharedApplication()
            return app.mainWindow() or app.keyWindow()
        except Exception:
            return None

    def _share_macos(self, paths: list[str]) -> bool:
        """macOS NSSharingServicePicker -- must run on Qt GUI thread."""
        try:
            from AppKit import NSSharingService, NSSharingServicePicker

            self._macos_share_log_thread("NSSharingServicePicker show")
            if not self._macos_share_on_main_thread():
                _share_log(logging.ERROR, "refusing picker: not on Qt GUI thread")
                return False

            self._macos_share_log_runtime_context()
            if not self._macos_share_ensure_appkit_context(require_window=True):
                return False

            urls, share_paths = self._macos_share_prepare_urls(paths)
            if not urls:
                return False
            items = _macos_share_items_array(urls)
            if items is None:
                return False
            services = NSSharingService.sharingServicesForItems_(items) or []
            self._macos_share_cached_services = services
            _share_log(logging.INFO, "sharingServicesForItems count=%d paths=%s", len(services), share_paths)
            if not services:
                _share_log(logging.WARNING, "sharingServicesForItems returned no services (empty items?)")
                return False

            view, rect, anchor_label = self._macos_share_picker_anchor_view_and_rect()
            if view is None or rect is None:
                _share_log(logging.WARNING, "no NSView anchor for picker")
                return False

            delegate_cls = _get_macos_share_picker_delegate_class()
            delegate = delegate_cls.alloc().initWithViewer_(self)
            picker = NSSharingServicePicker.alloc().initWithItems_(items)
            self._macos_share_picker_ref = picker
            self._macos_share_urls_ref = urls
            self._macos_share_picker_delegate_ref = delegate
            if delegate is not None:
                picker.setDelegate_(delegate)

            # v2.1 / POC: edge 3 (NSMaxYEdge) -- popover above anchor.
            preferred_edge = 3
            _share_log(
                logging.INFO,
                "picker show anchor=%s edge=%s rect=(%.0f,%.0f,%.0f,%.0f)",
                anchor_label,
                preferred_edge,
                rect.origin.x,
                rect.origin.y,
                rect.size.width,
                rect.size.height,
            )
            picker.showRelativeToRect_ofView_preferredEdge_(rect, view, preferred_edge)
            self._macos_share_pump_events(1)
            return True
        except Exception as exc:
            self._macos_share_picker_ref = None
            self._macos_share_urls_ref = None
            self._macos_share_picker_delegate_ref = None
            self._macos_share_cached_services = None
            _share_log(logging.WARNING, "NSSharingServicePicker failed: %s", exc, exc_info=True)
            return False

    def _share_windows_shell(self, path: str, owner_hwnd: int = 0) -> bool:
        """Legacy Explorer shell verb fallback when WinRT share is unavailable."""
        abs_path = _norm_path(os.path.abspath(path))
        _ = owner_hwnd
        try:
            import win32com.client  # type: ignore

            folder = win32com.client.Dispatch("Shell.Application").Namespace(
                os.path.dirname(abs_path)
            )
            item = folder.ParseName(os.path.basename(abs_path))
            if item is None:
                return False
            for verb_key in ("share", "Windows.share", "Windows.Share"):
                try:
                    item.InvokeVerb(verb_key)
                    return True
                except Exception:
                    continue
            for verb in item.Verbs():
                try:
                    if _windows_shell_verb_suggests_share(verb.Name):
                        verb.DoIt()
                        return True
                except Exception:
                    continue
        except Exception:
            return False
        return False

    def _copy_current_file_path_to_clipboard(self):
        p = getattr(self, "current_file_path", None)
        if not p:
            self.status_bar.showMessage("No file open", 2000)
            return
        self._copy_paths_to_clipboard([p])
        self.status_bar.showMessage("Image copied to clipboard -- paste into messaging apps to send", 3500)

    def _reveal_current_file_in_os_file_manager(self):
        import subprocess

        p = getattr(self, "current_file_path", None)
        if not p or not os.path.isfile(p):
            self.status_bar.showMessage("No file to reveal", 2000)
            return
        try:
            if sys.platform == "darwin":
                subprocess.run(["/usr/bin/open", "-R", p], check=False)
            elif sys.platform == "win32":
                subprocess.run(
                    ["explorer", "/select,", os.path.normpath(p)], check=False
                )
            else:
                subprocess.run(["xdg-open", os.path.dirname(p)], check=False)
            self.status_bar.showMessage("Revealed in file manager", 2000)
        except Exception:
            self.status_bar.showMessage("Could not open file manager", 2000)

    def _continue_resize_drag(self, event) -> bool:
        if not self._resize_edge_active:
            return False
        current_pos = event.globalPosition().toPoint() if hasattr(event, "globalPosition") else event.globalPos()
        delta = current_pos - self._resize_start_pos
        start_geometry = self._resize_start_geometry
        if start_geometry is None:
            return False
        new_geometry = QRect(start_geometry)
        if "left" in self._resize_edge_active:
            new_geometry.setLeft(start_geometry.left() + delta.x())
        if "right" in self._resize_edge_active:
            new_geometry.setRight(start_geometry.right() + delta.x())
        if "top" in self._resize_edge_active:
            new_geometry.setTop(start_geometry.top() + delta.y())
        if "bottom" in self._resize_edge_active:
            new_geometry.setBottom(start_geometry.bottom() + delta.y())
        if (
            new_geometry.width() >= self.minimumWidth()
            and new_geometry.height() >= self.minimumHeight()
        ):
            self.setGeometry(new_geometry)
        return True

    def _finish_resize_drag(self) -> bool:
        if not self._resize_edge_active:
            return False
        self._resize_edge_active = None
        self._is_resizing = False
        self.unsetCursor()
        grip = getattr(self, "_resize_grip", None)
        if grip is not None:
            grip.hide()
        with contextlib.suppress(Exception):
            self.releaseMouse()
        if hasattr(self, "view_mode") and self.view_mode == "gallery":
            gallery = getattr(self, "gallery_justified", None)
            if gallery is not None:
                try:
                    gallery._ignore_resize_events = False
                    from PyQt6.QtCore import QTimer

                    QTimer.singleShot(100, gallery.force_layout_update)
                except Exception:
                    pass
        if hasattr(self, "view_mode") and self.view_mode == "single" and getattr(self, "current_pixmap", None) is not None and getattr(self, "fit_to_window", False):
                from PyQt6.QtCore import QTimer

                QTimer.singleShot(100, self.scale_image_to_fit)
        with contextlib.suppress(Exception):
            self._update_resize_edge_hover(self.mapFromGlobal(QCursor.pos()))
        return True

    def _should_fast_open_single_file(self, start_file, start_view) -> bool:
        """True when opening one file for single view (Explorer, drag-drop, Open File).

        Any ``start_file`` implies single view unless ``start_view="gallery"``.
        Do not require current ``view_mode == "single"``: opening a file while
        the gallery is showing must still fast-open (legacy switches to single).
        """
        if not start_file:
            return False
        return start_view != "gallery"
