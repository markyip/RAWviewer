"""Shared env/path helpers for RAWviewer."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from typing import Any

# Set by main.py during bootstrap (pool workers vs GUI main).
_IS_GUI_MAIN_PROCESS = True


def configure_gui_process(is_main: bool) -> None:
    """Record whether the current process is the GUI main process.

    Args:
        is_main: True when running as the GUI main process.
    """
    global _IS_GUI_MAIN_PROCESS
    _IS_GUI_MAIN_PROCESS = bool(is_main)


def _env_true(name: str, default: bool = False) -> bool:
    """Read a boolean-valued environment variable.

    Args:
        name: Environment variable name.
        default: Fallback when the variable is unset or unparsable.

    Returns:
        Parsed boolean value.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    """Read an integer-valued environment variable, clamped to *minimum*.

    Args:
        name: Environment variable name.
        default: Fallback when the variable is unset or unparsable.
        minimum: Lower bound for the returned value.

    Returns:
        Clamped integer value.
    """
    try:
        return max(minimum, int(os.environ.get(name, str(default)).strip()))
    except (TypeError, ValueError):
        return default


def _is_primary_process() -> bool:
    """Return True when running as the GUI primary process."""
    return _IS_GUI_MAIN_PROCESS


_VERBOSE_CONSOLE = _env_true("RAWVIEWER_VERBOSE_CONSOLE", default=False)


def safe_print(*args: Any, **kwargs: Any) -> None:
    """Safely print to stdout, handling the None case in windowed builds.

    Args:
        args: Positional arguments forwarded to :func:`print`.
        kwargs: Keyword arguments forwarded to :func:`print`.
    """
    force = bool(kwargs.pop("force", False))
    if not force and (not _VERBOSE_CONSOLE or not _is_primary_process()):
        return
    if sys.stdout is not None:
        try:
            print(*args, **kwargs)
        except (OSError, AttributeError, UnicodeEncodeError):
            # Retry with error-character substitution (e.g. ❌ on cp1252)
            try:
                args2 = tuple(
                    a.encode(sys.stdout.encoding, "replace").decode(sys.stdout.encoding)
                    if isinstance(a, str) else a
                    for a in args
                )
                print(*args2, **kwargs)
            except (OSError, AttributeError, UnicodeEncodeError):
                pass


def safe_print_err(*args: Any, **kwargs: Any) -> None:
    """Safely print to stderr, handling the None case in windowed builds.

    Args:
        args: Positional arguments forwarded to :func:`print`.
        kwargs: Keyword arguments forwarded to :func:`print`.
    """
    _ = kwargs.pop("force", False)
    if sys.stderr is not None:
        with contextlib.suppress(OSError, AttributeError):
            print(*args, file=sys.stderr, **kwargs)


def _norm_path(p: object) -> str:
    """Normalize paths for reliable equality checks on Windows.

    Args:
        p: Path-like or string value.

    Returns:
        Normalized path string (empty string for falsy input).
    """
    try:
        if not p:
            return ""
        path = str(p)
        return os.path.normcase(os.path.normpath(path))
    except Exception:
        return str(p) if p else ""


def _find_file_index_in_list(files: list[Any], file_path: str, *, default: int = -1) -> int:
    """Return index of *file_path* in *files* (normcase/abspath/basename), or *default*.

    Args:
        files: Candidate file paths.
        file_path: Target file path.
        default: Fallback index when not found.

    Returns:
        Matching index or *default*.
    """
    if not files or not file_path:
        return default
    target_norm = _norm_path(file_path)
    try:
        target_abs = os.path.normcase(os.path.abspath(file_path))
    except (OSError, TypeError):
        target_abs = target_norm
    target_base = os.path.normcase(os.path.basename(file_path))
    for i, fp in enumerate(files):
        if not fp:
            continue
        if _norm_path(fp) == target_norm:
            return i
        try:
            if os.path.normcase(os.path.abspath(fp)) == target_abs:
                return i
        except (OSError, TypeError):
            pass
        if os.path.normcase(os.path.basename(fp)) == target_base:
            return i
    return default


def _share_logger() -> logging.Logger:
    """Return the shared RAWviewer share logger."""
    return logging.getLogger("rawviewer.share")
