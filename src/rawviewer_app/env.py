"""Shared env/path helpers."""
import logging
import os
import sys

# Set by main.py during bootstrap (pool workers vs GUI main).
_IS_GUI_MAIN_PROCESS = True


def configure_gui_process(is_main: bool) -> None:
    global _IS_GUI_MAIN_PROCESS
    _IS_GUI_MAIN_PROCESS = bool(is_main)


def _env_true(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default)).strip()))
    except (TypeError, ValueError):
        return default


def _is_primary_process() -> bool:
    return _IS_GUI_MAIN_PROCESS


_VERBOSE_CONSOLE = _env_true("RAWVIEWER_VERBOSE_CONSOLE", default=False)


def safe_print(*args, **kwargs):
    """Safely print to stdout, handling None case in windowed builds."""
    force = bool(kwargs.pop("force", False))
    if not force and (not _VERBOSE_CONSOLE or not _is_primary_process()):
        return
    if sys.stdout is not None:
        try:
            print(*args, **kwargs)
        except (OSError, AttributeError):
            pass


def safe_print_err(*args, **kwargs):
    """Safely print to stderr, handling None case in windowed builds."""
    _ = kwargs.pop("force", False)
    if sys.stderr is not None:
        try:
            print(*args, file=sys.stderr, **kwargs)
        except (OSError, AttributeError):
            pass


def _norm_path(p: object) -> str:
    """Normalize paths for reliable equality checks on Windows."""
    try:
        if not p:
            return ""
        path = str(p)
        return os.path.normcase(os.path.normpath(path))
    except Exception:
        return str(p) if p else ""


def _find_file_index_in_list(files, file_path, *, default: int = -1) -> int:
    """Return index of *file_path* in *files* (normcase/abspath/basename), or *default*."""
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
    return logging.getLogger("rawviewer.share")
