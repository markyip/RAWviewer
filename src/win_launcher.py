"""Minimal Windows launcher stub — starts the installed Pixi app environment."""

from __future__ import annotations

import os
import subprocess
import sys


def _install_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _suppress_launcher_window() -> None:
    """Hide any console / bootloader window the short-lived stub may flash."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        hwnd = kernel32.GetConsoleWindow()
        if hwnd:
            user32.ShowWindow(hwnd, 0)  # SW_HIDE
    except Exception:
        pass


def _show_error(message: str) -> None:
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(  # type: ignore[attr-defined]
            0,
            message,
            "RAWviewer",
            0x00000010,  # MB_ICONERROR
        )
    except Exception:
        pass


def main() -> int:
    _suppress_launcher_window()
    install_dir = _install_dir()
    pixi_exe = os.path.join(install_dir, "_internal", "pixi", "pixi.exe")
    if not os.path.isfile(pixi_exe):
        _show_error(
            "RAWviewer could not find its runtime.\n\n"
            "Try running RAWviewer_Setup.exe again or reinstall from GitHub Releases."
        )
        return 1

    child_env = os.environ.copy()
    # Strip launcher-only flags; enable durable file logs for freeze diagnosis.
    passthrough_args: list[str] = []
    for arg in sys.argv[1:]:
        low = arg.strip().lower()
        if low in ("--file-log", "--debug-log", "/file-log", "/debug-log"):
            child_env["RAWVIEWER_FILE_LOG"] = "1"
            child_env["RAWVIEWER_REDIRECT_STDIO"] = "1"
            child_env["RAWVIEWER_VERBOSE_INFO_LOGS"] = "1"
            child_env["RAWVIEWER_FOCUS_GALLERY_SWITCH"] = "1"
            child_env.setdefault("RAWVIEWER_FATAL_DUMP", "1")
            child_env.setdefault("RAWVIEWER_DEBUG", "1")
            continue
        passthrough_args.append(arg)

    # Marker file also enables logging without a special bat.
    marker = os.path.join(
        os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"),
        "RAWviewer",
        "enable_debug_log",
    )
    if os.path.isfile(marker):
        child_env["RAWVIEWER_FILE_LOG"] = "1"
        child_env["RAWVIEWER_REDIRECT_STDIO"] = "1"
        child_env["RAWVIEWER_VERBOSE_INFO_LOGS"] = "1"
        child_env["RAWVIEWER_FOCUS_GALLERY_SWITCH"] = "1"
        child_env.setdefault("RAWVIEWER_FATAL_DUMP", "1")

    cmd = [pixi_exe, "run", "start-windowless"]
    if passthrough_args:
        cmd.append("--")
        cmd.extend(passthrough_args)

    # CREATE_NO_WINDOW only, never DETACHED_PROCESS: the two are mutually
    # exclusive (CREATE_NO_WINDOW is IGNORED when DETACHED_PROCESS is set),
    # and a detached pixi has no console for its console-subsystem children
    # (env-activation cmd.exe) to inherit -- Windows then allocates a fresh
    # VISIBLE console for them, which was the "unbranded window flashes
    # before the splash" bug. With CREATE_NO_WINDOW alone, pixi owns an
    # invisible console that every child inherits silently.
    creationflags = (
        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    )
    startupinfo = None
    try:
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0  # SW_HIDE
    except Exception:
        startupinfo = None

    try:
        subprocess.Popen(
            cmd,
            cwd=install_dir,
            env=child_env,
            creationflags=creationflags,
            close_fds=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            startupinfo=startupinfo,
        )
    except OSError as exc:
        _show_error(f"Could not start RAWviewer:\n\n{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
