
"""Windows shell integration: open-with and share helpers."""
from __future__ import annotations

import contextlib
import ctypes
import os
import struct
import subprocess
import sys

__all__ = [
    "_filmstrip_ui_env_int",
    "_windows_shell_verb_suggests_share",
    "_open_windows_open_with_dialog",
    "_share_windows_clipboard_cf_hdrop",
    "_share_windows_clipboard_cf_hdrop_paths",
    "_share_windows_clipboard_file_via_powershell",
    "_resolve_windows_share_helper_exe",
    "_share_windows_via_helper",
    "_share_windows_via_winrt",
]


def _filmstrip_ui_env_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default)).strip()))
    except (TypeError, ValueError):
        return default


def _windows_shell_verb_suggests_share(verb_name: object) -> bool:
    """Match Explorer context-menu verbs for 'Share' across English and common locales."""
    s = str(verb_name or "")
    low = s.replace("&", "").strip().lower()
    if "share" in low or "windows.share" in low:
        return True
    plain = s.replace("&", "")
    for token in (
        "\u5171\u7528",  # zh-TW: 共用
        "\u5206\u4eab",  # zh-CN: 分享
        "partage",
        "teilen",
        "condividi",
        "compartir",
        "delen",
    ):
        if token in plain or token in low:
            return True
    return False


def _open_windows_open_with_dialog(path: str, owner_hwnd: int = 0) -> bool:
    """Show the native Windows 'Open with' dialog for a file."""
    from ctypes import Structure, byref, c_long, wintypes

    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        return False

    hwnd = int(owner_hwnd or 0)
    OAIF_EXEC = 0x0004  # Required on Win10+ or Windows shows "go to Settings" instead.

    # Classic Open With picker (Unicode). Prefer this over SHOpenWithDialog on Win11.
    try:
        shell32 = ctypes.windll.shell32
        shell32.OpenAs_RunDLLW.argtypes = [
            wintypes.HWND,
            wintypes.HINSTANCE,
            wintypes.LPCWSTR,
            ctypes.c_int,
        ]
        shell32.OpenAs_RunDLLW.restype = None
        shell32.OpenAs_RunDLLW(hwnd, None, abs_path, 1)
        return True
    except Exception:
        pass

    # SHOpenWithDialog -- must pass OAIF_EXEC or user only sees the Settings redirect.
    try:
        class _OPENASINFO(Structure):
            _fields_ = [
                ("pcszFile", wintypes.LPCWSTR),
                ("pcszClass", wintypes.LPCWSTR),
                ("oaifInFlags", wintypes.DWORD),
            ]

        shell32 = ctypes.windll.shell32
        shell32.SHOpenWithDialog.argtypes = [wintypes.HWND, ctypes.POINTER(_OPENASINFO)]
        shell32.SHOpenWithDialog.restype = c_long
        info = _OPENASINFO(abs_path, None, OAIF_EXEC)
        hr = int(shell32.SHOpenWithDialog(hwnd, byref(info)))
        if hr >= 0:
            return True
    except Exception:
        pass

    # rundll32 fallback -- must use OpenAs_RunDLLW, not the ANSI OpenAs_RunDLL.
    try:
        subprocess.Popen(["rundll32.exe", "shell32.dll,OpenAs_RunDLLW", abs_path])
        return True
    except Exception:
        pass

    try:
        flags = 0
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            flags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        r = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Sta",
                "-Command",
                "Start-Process",
                "-FilePath",
                abs_path,
                "-Verb",
                "OpenAs",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=flags,
        )
        if r.returncode == 0:
            return True
    except Exception:
        pass

    try:
        import win32com.client  # type: ignore

        folder = win32com.client.Dispatch("Shell.Application").Namespace(
            os.path.dirname(abs_path)
        )
        item = folder.ParseName(os.path.basename(abs_path))
        if item is None:
            return False
        open_with_tokens = (
            "openas",
            "open with",
            "openwith",
            "開啟方式",
            "選擇開啟方式",
            "打开方式",
            "选择打开方式",
        )
        for verb_key in open_with_tokens:
            try:
                item.InvokeVerb(verb_key)
                return True
            except Exception:
                continue
        for verb in item.Verbs():
            try:
                name = str(getattr(verb, "Name", "") or "").replace("&", "").strip().lower()
                if any(token in name for token in open_with_tokens):
                    verb.DoIt()
                    return True
            except Exception:
                continue
    except Exception:
        return False
    return False


def _share_windows_clipboard_cf_hdrop(path: str) -> bool:
    """Place file(s) on clipboard as CF_HDROP (native Windows file clipboard)."""

    try:
        import win32clipboard  # type: ignore
        import win32con  # type: ignore
    except ImportError:
        return False

    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        return False
    paths = (abs_path + "\0").encode("utf-16le") + b"\x00\x00"
    dropfiles = struct.pack("<IIIII", 20, 0, 0, 0, 1)
    data = dropfiles + paths
    try:
        win32clipboard.OpenClipboard()
        try:
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32con.CF_HDROP, data)
        finally:
            win32clipboard.CloseClipboard()
        return True
    except Exception:
        with contextlib.suppress(Exception):
            win32clipboard.CloseClipboard()
        return False


def _share_windows_clipboard_cf_hdrop_paths(file_paths: list[str]) -> bool:
    """Place multiple files on the clipboard as CF_HDROP."""

    try:
        import win32clipboard  # type: ignore
        import win32con  # type: ignore
    except ImportError:
        return False

    abs_paths = [
        os.path.abspath(p) for p in file_paths if p and os.path.isfile(os.path.abspath(p))
    ]
    if not abs_paths:
        return False
    if len(abs_paths) == 1:
        return _share_windows_clipboard_cf_hdrop(abs_paths[0])
    paths_blob = "".join(p + "\0" for p in abs_paths).encode("utf-16le") + b"\x00\x00"
    dropfiles = struct.pack("<IIIII", 20, 0, 0, 0, len(abs_paths))
    data = dropfiles + paths_blob
    try:
        win32clipboard.OpenClipboard()
        try:
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32con.CF_HDROP, data)
        finally:
            win32clipboard.CloseClipboard()
        return True
    except Exception:
        with contextlib.suppress(Exception):
            win32clipboard.CloseClipboard()
        return False


def _share_windows_clipboard_file_via_powershell(path: str) -> bool:
    """Put the file object on the clipboard (Windows) so the user can paste into Mail, Teams, Explorer, etc."""

    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        return False
    flags = 0
    if sys.platform == "win32" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        flags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    try:
        r = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Sta",
                "-Command",
                "Set-Clipboard",
                "-LiteralPath",
                abs_path,
            ],
            capture_output=True,
            text=True,
            timeout=20,
            creationflags=flags,
        )
        return r.returncode == 0
    except Exception:
        return False


def _resolve_windows_share_helper_exe():
    """Locate built WindowsShareHelper.exe (dev tree or bundled install)."""
    rel = os.path.join(
        "windows_share_helper",
        "bin",
        "Release",
        "net8.0-windows10.0.19041.0",
        "WindowsShareHelper.exe",
    )
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.path.join(here, rel)]
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        candidates.append(os.path.join(exe_dir, "WindowsShareHelper.exe"))
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(os.path.join(meipass, "WindowsShareHelper.exe"))
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def _share_windows_via_helper(path: str, owner_hwnd: int) -> bool:
    """Launch the .NET helper that wraps IDataTransferManagerInterop."""
    import time

    helper = _resolve_windows_share_helper_exe()
    if not helper:
        return False
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        return False
    flags = 0
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        flags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    try:
        proc = subprocess.Popen(
            [helper, str(int(owner_hwnd or 0)), abs_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
        # Fail fast if the helper exits immediately, but never block the Qt UI
        # thread for hundreds of ms -- that prevented the share flyout on first click.
        app = None
        try:
            from PyQt6.QtWidgets import QApplication

            app = QApplication.instance()
        except Exception:
            app = None
        deadline = time.time() + 0.15
        while time.time() < deadline:
            rc = proc.poll()
            if rc is not None:
                return rc == 0
            if app is not None:
                app.processEvents()
            time.sleep(0.01)
        return True
    except Exception:
        return False


def _share_windows_via_winrt(path: str, owner_hwnd: int) -> bool:
    """In-process WinRT share sheet (DataTransferManager + interop)."""
    if not owner_hwnd:
        return False
    try:
        from platform.windows_share import share_file_windows

        return bool(share_file_windows(path, owner_hwnd))
    except Exception:
        return False
