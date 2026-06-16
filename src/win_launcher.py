"""Minimal Windows launcher stub — starts the installed Pixi app environment."""

from __future__ import annotations

import os
import subprocess
import sys


def _install_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
    install_dir = _install_dir()
    pixi_exe = os.path.join(install_dir, "_internal", "pixi", "pixi.exe")
    if not os.path.isfile(pixi_exe):
        _show_error(
            "RAWviewer could not find its runtime.\n\n"
            "Try running RAWviewer_Setup.exe again or reinstall from GitHub Releases."
        )
        return 1

    cmd = [pixi_exe, "run", "start-windowless"]
    file_args = [arg for arg in sys.argv[1:] if arg]
    if file_args:
        cmd.append("--")
        cmd.extend(file_args)

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        subprocess.Popen(
            cmd,
            cwd=install_dir,
            creationflags=creationflags,
        )
    except OSError as exc:
        _show_error(f"Could not start RAWviewer:\n\n{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
