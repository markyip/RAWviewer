"""Folder-root sandbox for MCP tools. Rejects path traversal."""

from __future__ import annotations

import os
from glob import glob as _glob


ENV_ROOT = "RAWVIEWER_MCP_ROOT"


class PathSandboxError(ValueError):
    """Raised when a path is missing, outside the rooted folder, or unsafe."""


def _norm_real(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def resolve_root(cli_root: str | None = None) -> str:
    raw = (cli_root or os.environ.get(ENV_ROOT) or "").strip()
    if not raw:
        raise PathSandboxError(
            f"Folder root is required (CLI --root or env {ENV_ROOT})"
        )
    root = _norm_real(raw)
    if not os.path.isdir(root):
        raise PathSandboxError(f"Folder root is not a directory: {raw}")
    return root


def is_within_root(root: str, path: str) -> bool:
    root_n = _norm_real(root)
    path_n = _norm_real(path)
    if path_n == root_n:
        return True
    prefix = root_n + os.sep
    return path_n.startswith(prefix)


def resolve_under_root(root: str, user_path: str, *, must_exist: bool = False) -> str:
    """Resolve ``user_path`` (relative or absolute) and require it stay under ``root``."""
    if user_path is None or not str(user_path).strip():
        raise PathSandboxError("Path is required")
    text = str(user_path).strip()
    if "\x00" in text:
        raise PathSandboxError("Invalid path")
    root_n = _norm_real(root)
    if os.path.isabs(text):
        candidate = text
    else:
        candidate = os.path.join(root_n, text)
    # Reject obvious traversal before symlink resolution.
    parts = os.path.normpath(candidate).split(os.sep)
    if ".." in parts:
        # Still allow if the resolved path lands inside the root.
        pass
    resolved = _norm_real(candidate)
    if not is_within_root(root_n, resolved):
        raise PathSandboxError("Path is outside the folder root")
    if must_exist and not os.path.lexists(resolved):
        raise PathSandboxError(f"Path does not exist: {text}")
    return resolved


def ensure_parent_under_root(root: str, dest_path: str) -> str:
    """Resolve a write destination; parent dir must already exist under root
    or be creatable under root (the dest file itself may be new)."""
    dest = resolve_under_root(root, dest_path, must_exist=False)
    parent = os.path.dirname(dest)
    if parent and not is_within_root(root, parent):
        raise PathSandboxError("Destination parent is outside the folder root")
    return dest


def list_photo_files(
    root: str,
    *,
    pattern: str | None = None,
    limit: int | None = None,
) -> list[str]:
    """Photos the app already treats as photos (RAW + standard images)."""
    from raw_file_extensions import get_supported_extensions

    exts = {e.lower() for e in get_supported_extensions()}
    root_n = _norm_real(root)
    out: list[str] = []

    if pattern:
        matches = _glob(os.path.join(root_n, pattern), recursive=True)
        candidates = []
        for m in matches:
            if os.path.isfile(m) and is_within_root(root_n, m):
                candidates.append(m)
        candidates.sort()
    else:
        candidates = []
        for dirpath, dirnames, filenames in os.walk(root_n, followlinks=False):
            if not is_within_root(root_n, dirpath):
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for name in filenames:
                candidates.append(os.path.join(dirpath, name))
        candidates.sort()

    for path in candidates:
        ext = os.path.splitext(path)[1].lower()
        if ext not in exts:
            continue
        if not os.path.isfile(path):
            continue
        if not is_within_root(root_n, path):
            continue
        out.append(path)
        if limit is not None and limit > 0 and len(out) >= int(limit):
            break
    return out
