"""Startup splash lifecycle helpers."""
from __future__ import annotations

import contextlib
import os
import sys

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QSplashScreen

from .bootstrap import _IS_GUI_MAIN_PROCESS
from .paths import _branded_icon_resource_path
from .platform_win import _is_windows_installed_app

__all__ = [
    "_should_show_startup_splash",
    "close_native_splash",
    "_load_branded_qicon",
    "_force_hide_startup_splash",
    "_dismiss_startup_splash",
    "_schedule_startup_splash_dismiss",
]

_startup_splash = None  # module-level reference, set by _schedule_startup_splash_dismiss


def _load_branded_qicon() -> QIcon | None:
    """Load multi-resolution branded icon for window / taskbar chrome."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QIcon, QPixmap

    path = _branded_icon_resource_path()
    if not path:
        return None
    icon = QIcon(path)
    # Ensure common taskbar sizes exist when the source is a single PNG.
    if path.lower().endswith(".png"):
        pix = QPixmap(path)
        if not pix.isNull():
            for edge in (16, 24, 32, 48, 64, 128, 256):
                icon.addPixmap(
                    pix.scaled(
                        edge,
                        edge,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
    if icon.isNull():
        return None
    return icon


def _should_show_startup_splash() -> bool:
    """Show Qt splash for packaged / installed launches (skip bare `python src/main.py`).

    RAWVIEWER_SHOW_SPLASH=0 (or false/no/off) is an explicit opt-out that wins
    even for frozen/installed launches -- otherwise it's opt-in-only for the
    bare dev-run path below.
    """
    if not _IS_GUI_MAIN_PROCESS:
        return False
    flag = os.environ.get("RAWVIEWER_SHOW_SPLASH", "").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return False
    if getattr(sys, "frozen", False):
        return True
    if _is_windows_installed_app():
        return True
    return flag in ("1", "true", "yes", "on")


def close_native_splash():
    try:
        import pyi_splash
        pyi_splash.close()
    except ImportError:
        pass


def _force_hide_startup_splash(app, main_window=None, splash=None) -> None:
    """Force-hide startup splash. macOS often keeps stay-on-top splash visible after finish()."""
    global _startup_splash
    close_native_splash()
    splash_widgets: list = []
    for candidate in (splash, _startup_splash):
        if candidate is not None and candidate not in splash_widgets:
            splash_widgets.append(candidate)
    try:
        for top_level in QApplication.topLevelWidgets():
            if isinstance(top_level, QSplashScreen) and top_level not in splash_widgets:
                splash_widgets.append(top_level)
    except Exception:
        pass
    for sp in splash_widgets:
        try:
            # Hide first. Never call setWindowFlags() while visible -- Qt recreates
            # the HWND and flashes a brief empty popup on Windows.
            sp.hide()
            if main_window is not None:
                with contextlib.suppress(Exception):
                    sp.finish(main_window)
            with contextlib.suppress(Exception):
                sp.close()
            with contextlib.suppress(Exception):
                sp.deleteLater()
        except Exception:
            pass
    _startup_splash = None
    if app is not None:
        with contextlib.suppress(Exception):
            app.processEvents()
    if main_window is not None:
        try:
            # Avoid repeated show/raise/activate on Windows (taskbar flicker).
            if not (main_window.isVisible() and main_window.isActiveWindow()):
                main_window.show()
                main_window.raise_()
                main_window.activateWindow()
                if app is not None:
                    app.processEvents()
        except Exception:
            pass
    if sys.platform == "darwin" and main_window is not None:
        try:
            import objc

            ns_app = objc.lookUpClass("NSApplication").sharedApplication()
            if ns_app is not None:
                ns_app.setActivationPolicy_(0)
                ns_app.activateIgnoringOtherApps_(True)
            wh = main_window.windowHandle()
            if wh is not None:
                view = objc.objc_object(c_void_p=int(wh.winId()))
                win = view.window()
                if win is not None:
                    win.makeKeyAndOrderFront_(None)
        except Exception:
            pass


def _dismiss_startup_splash(app, main_window=None, splash=None) -> None:
    """Alias used by fatal-error paths; same force-hide as the scheduled retries."""
    _force_hide_startup_splash(app, main_window, splash)


def _schedule_startup_splash_dismiss(app, main_window, splash) -> None:
    """Dismiss splash after the main window has been shown/painted."""
    _force_hide_startup_splash(app, main_window, splash)
    if sys.platform == "darwin":
        # macOS can lag QSplashScreen.finish(); retry a few times.
        for delay_ms in (0, 50, 150, 400, 800):
            QTimer.singleShot(
                delay_ms,
                lambda mw=main_window, a=app: _force_hide_startup_splash(a, mw, None),
            )
    else:
        # One deferred sweep only -- enough to catch a stuck splash without
        # repeatedly poking Z-order (which flashes on Windows).
        QTimer.singleShot(
            120,
            lambda mw=main_window, a=app: _force_hide_startup_splash(a, mw, None),
        )
