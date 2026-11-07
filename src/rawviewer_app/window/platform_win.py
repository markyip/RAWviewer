"""Windows platform integration (taskbar identity and native icons)."""
from __future__ import annotations

import contextlib
import ctypes
import os
import sys

from .paths import resource_path

__all__ = [
    "_WINDOWS_APP_USER_MODEL_ID",
    "_set_windows_app_user_model_id",
    "_is_windows_installed_app",
    "_apply_windows_native_window_icon",
]


_WINDOWS_APP_USER_MODEL_ID = "markyip.rawviewer.app.1"


def _set_windows_app_user_model_id() -> None:
    if sys.platform != "win32":
        return
    with contextlib.suppress(Exception):
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            _WINDOWS_APP_USER_MODEL_ID
        )


def _apply_windows_native_window_icon(widget, ico_path: str | None = None) -> None:
    """Push .ico into the HWND so the taskbar shows it on frameless windows."""
    if sys.platform != "win32" or widget is None:
        return
    path = ico_path or resource_path(os.path.join("icons", "appicon.ico"))
    if not path or not os.path.isfile(path):
        return
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        WM_SETICON = 0x0080
        ICON_SMALL = 0
        ICON_BIG = 1
        IMAGE_ICON = 1
        LR_LOADFROMFILE = 0x0010
        LR_DEFAULTSIZE = 0x0040

        hwnd = int(widget.winId())
        if not hwnd:
            return

        LoadImageW = user32.LoadImageW
        LoadImageW.argtypes = [
            wintypes.HINSTANCE,
            wintypes.LPCWSTR,
            wintypes.UINT,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        ]
        LoadImageW.restype = wintypes.HANDLE

        # Explicit small + large sizes; LR_DEFAULTSIZE alone is unreliable across DPI.
        hicon_small = LoadImageW(None, path, IMAGE_ICON, 16, 16, LR_LOADFROMFILE)
        hicon_big = LoadImageW(None, path, IMAGE_ICON, 256, 256, LR_LOADFROMFILE)
        if not hicon_big:
            hicon_big = LoadImageW(
                None, path, IMAGE_ICON, 0, 0, LR_LOADFROMFILE | LR_DEFAULTSIZE
            )
        if hicon_small:
            user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, hicon_small)
        if hicon_big:
            user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, hicon_big)
    except Exception:
        pass


def _is_windows_installed_app() -> bool:
    """True for the Windows installer layout (Pixi under _internal; not sys.frozen)."""
    if sys.platform != "win32":
        return False
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.isfile(os.path.join(root_dir, "_internal", "pixi", "pixi.exe"))
