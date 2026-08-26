"""Folder-root sandbox for MCP tools. Rejects path traversal."""

from __future__ import annotations

import heapq
import os
from glob import glob as _glob


ENV_ROOT = "RAWVIEWER_MCP_ROOT"


class PathSandboxError(ValueError):
    """Raised when a path is missing, outside the rooted folder, or unsafe."""


class _LexMax:
    """Max-heap wrapper so heapq can keep the N lexicographically smallest paths."""

    __slots__ = ("s",)

    def __init__(self, s: str) -> None:
        self.s = s

    def __lt__(self, other: "_LexMax") -> bool:
        return self.s > other.s


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


def _is_under_normalized_root(root_n: str, path: str) -> bool:
    """Like ``is_within_root`` but skips re-deriving ``root``'s realpath.

    ``root_n`` must already be the output of ``_norm_real``. Used in the
    per-file hot loop of ``list_photo_files`` so a large tree doesn't pay
    for re-resolving the (unchanging) root once per candidate file.
    """
    path_n = _norm_real(path)
    if path_n == root_n:
        return True
    return path_n.startswith(root_n + os.sep)


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
    cap = int(limit) if limit is not None and int(limit) > 0 else 0
    heap: list[_LexMax] = []
    all_paths: list[str] = []

    def _consider(path: str) -> None:
        if cap:
            if len(heap) < cap:
                heapq.heappush(heap, _LexMax(path))
            elif path < heap[0].s:
                heapq.heapreplace(heap, _LexMax(path))
        else:
            all_paths.append(path)

    if pattern:
        matches = _glob(os.path.join(root_n, pattern), recursive=True)
        for m in matches:
            ext = os.path.splitext(m)[1].lower()
            if ext not in exts:
                continue
            if os.path.isfile(m) and _is_under_normalized_root(root_n, m):
                _consider(m)
    else:
        # Manual scandir recursion: filter by extension (and skip hidden
        # dirs) before paying for any stat/realpath work, instead of
        # os.walk's collect-everything-then-filter pattern. With ``limit``,
        # only k paths are kept (heap) so RAM stays O(k); the tree is still
        # walked so "first N in sorted order" stays correct.
        def _walk(dirpath: str) -> None:
            try:
                entries = list(os.scandir(dirpath))
            except OSError:
                return
            for entry in entries:
                name = entry.name
                try:
                    is_dir = entry.is_dir(follow_symlinks=False)
                except OSError:
                    continue
                if is_dir:
                    if name.startswith("."):
                        continue
                    _walk(entry.path)
                    continue
                ext = os.path.splitext(name)[1].lower()
                if ext not in exts:
                    continue
                try:
                    if not entry.is_file():
                        continue
                except OSError:
                    continue
                if not _is_under_normalized_root(root_n, entry.path):
                    continue
                _consider(entry.path)

        _walk(root_n)

    if cap:
        return sorted(item.s for item in heap)
    all_paths.sort()
    return all_paths
