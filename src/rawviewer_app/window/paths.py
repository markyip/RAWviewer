"""Filesystem resource and path helpers for the main window."""
from __future__ import annotations

import os
import platform
import sys

from core.config import _find_file_index_in_list, _norm_path

__all__ = [
    "resource_path",
    "_branded_icon_resource_path",
    "_is_in_folder_scope",
    "_norm_path",
    "_find_file_index_in_list",
    "_require_str_path",
    "_SUBFOLDER_FALLBACK_MAX_DIRS",
    "_SUBFOLDER_FALLBACK_SKIP",
]


def resource_path(relative_path):
    """Absolute path to a bundled resource (dev tree or PyInstaller _MEIPASS)."""
    try:
        base_path = getattr(sys, "_MEIPASS", None)
        if base_path is None:
            raise AttributeError
    except Exception:
        base_path = os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".."))
    return os.path.join(base_path, relative_path)


def _branded_icon_resource_path() -> str | None:
    """Bundled app icon for window chrome (platform format preferred)."""
    if platform.system() == "Windows":
        candidates = (
            os.path.join("icons", "appicon.ico"),
            os.path.join("icons", "appicon.png"),
        )
    elif platform.system() == "Darwin":
        candidates = (
            os.path.join("icons", "appicon.icns"),
            os.path.join("icons", "appicon.png"),
        )
    else:
        candidates = (os.path.join("icons", "appicon.png"),)
    for rel in candidates:
        path = resource_path(rel)
        if os.path.exists(path):
            return path
    return None


def _is_in_folder_scope(path: str, root: str) -> bool:
    """True when *path* (a dir) is at or inside *root* (dir); no symlink resolution."""
    try:
        p = os.path.normcase(os.path.normpath(os.path.abspath(path)))
        r = os.path.normcase(os.path.normpath(os.path.abspath(root)))
    except Exception:
        return False
    if not r or r == p:
        return True
    try:
        return os.path.commonpath((p, r)) == r
    except ValueError:
        # Mixed drives etc. cannot share a common ancestor.
        return False


_SUBFOLDER_FALLBACK_MAX_DIRS = 64


_SUBFOLDER_FALLBACK_SKIP = frozenset({"discard"})


def _require_str_path(path: object) -> str:
    return str(path) if path else ""
