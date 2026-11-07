"""Early process bootstrap: log redirection and crash diagnostics."""
from __future__ import annotations

import os
import sys

__all__ = [
    "_runtime_log_dir",
    "_debug_log_marker_path",
    "_FILE_LOG_ARGV_FLAGS",
    "_consume_file_log_argv",
    "_file_logging_requested",
    "_redirect_stdio_to_file_if_needed",
    "_enable_fatal_crash_dump_if_needed",
]


def _is_multiprocessing_worker_process() -> bool:
    """Detect ProcessPool / spawn children before Qt runs (frozen .app re-exec)."""
    if any(
        arg == "--multiprocessing-fork" or arg.startswith("--multiprocessing-fork")
        for arg in sys.argv[1:]
    ):
        return True
    if len(sys.argv) >= 3 and sys.argv[1] == "-c" and "spawn_main" in sys.argv[2]:
        return True
    try:
        import multiprocessing as _mp
        if _mp.parent_process() is not None:
            return True
        # Spawn children can still report MainProcess until spawn_main renames them.
        if _mp.current_process().name != "MainProcess":
            return True
    except Exception:
        pass
    return False


def _bootstrap_multiprocessing_gui_main() -> bool:
    """True only for the real GUI process (not ProcessPoolExecutor workers)."""
    try:
        import multiprocessing as _mp
        _mp.freeze_support()
    except Exception:
        pass
    return not _is_multiprocessing_worker_process()


_IS_GUI_MAIN_PROCESS = _bootstrap_multiprocessing_gui_main()


def _runtime_log_dir() -> str:
    """
    Central runtime log directory.
    Use LocalAppData for both installed and dev runs to avoid packaging stale src/logs.
    """
    app_data = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~") if os.name == "nt" else os.path.expanduser("~")
    return os.path.join(app_data, "RAWviewer", "logs")


def _debug_log_marker_path() -> str:
    app_data = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~") if os.name == "nt" else os.path.expanduser("~")
    return os.path.join(app_data, "RAWviewer", "enable_debug_log")


_FILE_LOG_ARGV_FLAGS = frozenset(
    {"--file-log", "--debug-log", "/file-log", "/debug-log"}
)


def _consume_file_log_argv() -> bool:
    """Strip --file-log/--debug-log from argv; return True if present.

    Must run before startup treats argv[1] as an image/folder path.
    """
    if len(sys.argv) <= 1:
        return False
    kept = [sys.argv[0]]
    found = False
    for arg in sys.argv[1:]:
        if arg.strip().lower() in _FILE_LOG_ARGV_FLAGS:
            found = True
            continue
        kept.append(arg)
    if found:
        sys.argv[:] = kept
    return found


def _file_logging_requested() -> bool:
    """True when durable file logs should be written (explicit opt-in only).

    RAWVIEWER_DEBUG alone does not create log files -- that would grow disk use
    and persist paths/metadata. Use RAWVIEWER_FILE_LOG=1, --file-log, or the
    LocalAppData enable_debug_log marker when diagnosing freezes.
    """
    if os.environ.get("RAWVIEWER_FILE_LOG", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return True
    try:
        if os.path.isfile(_debug_log_marker_path()):
            return True
    except OSError:
        pass
    return False


def _redirect_stdio_to_file_if_needed():
    try:
        # Opt-in only: do not create log folders/files in normal releases.
        if os.environ.get("RAWVIEWER_REDIRECT_STDIO", "").strip() not in ("1", "true", "True", "YES", "yes"):
            return

        is_frozen = bool(getattr(sys, "frozen", False))
        if not is_frozen:
            return
        if sys.stdout is not None and sys.stderr is not None:
            return

        log_dir = _runtime_log_dir()
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError:
            return

        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(log_dir, f"rawviewer_console_{ts}.log")
        f = open(path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115  # Intentional: reassigned as sys.stdout
        if sys.stdout is None:
            sys.stdout = f
        if sys.stderr is None:
            sys.stderr = f
    except Exception:
        # Never fail startup due to debug output plumbing
        pass


def _enable_fatal_crash_dump_if_needed():
    """
    Enable low-level fatal crash dumps (access violation/segfault) very early.
    Opt-in only: normal launches must not open running dump files under
    LocalAppData (size + path leakage). Enable via RAWVIEWER_FATAL_DUMP=1,
    file-log env/marker, or --file-log on argv (checked before setup_logging).
    """
    try:
        dump_flag = os.environ.get("RAWVIEWER_FATAL_DUMP", "").strip().lower()
        if dump_flag in ("0", "false", "no", "off"):
            return
        argv_file_log = any(
            str(a).strip().lower() in _FILE_LOG_ARGV_FLAGS for a in sys.argv[1:]
        )
        enabled = (
            dump_flag in ("1", "true", "yes", "on")
            or _file_logging_requested()
            or argv_file_log
        )
        if not enabled:
            return
        import faulthandler
        import signal
        from datetime import datetime

        dump_dir = _runtime_log_dir()
        try:
            os.makedirs(dump_dir, exist_ok=True)
        except OSError:
            return

        is_dev = os.environ.get("RAWVIEWER_DEBUG", "").strip() in ("1", "true", "True", "YES", "yes")
        if is_dev:
            dump_path = os.path.join(dump_dir, "fatal_dump_dev.log")
            mode = "w"
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dump_path = os.path.join(dump_dir, f"fatal_dump_{ts}.log")
            mode = "a"

        f = open(dump_path, mode, encoding="utf-8", buffering=1)  # noqa: SIM115  # Intentional: for faulthandler
        # enable() already installs handlers for SIGSEGV/SIGFPE/SIGABRT/SIGBUS/SIGILL;
        # register()-ing SIGABRT again raises RuntimeError and was silently swallowed
        # by the except below, so SIGUSR1 registration below never actually ran.
        faulthandler.enable(file=f, all_threads=True)
        faulthandler.register(signal.SIGUSR1, file=f, all_threads=True)
        if os.environ.get("RAWVIEWER_SCROLL_DEBUG_DUMP", "").strip() == "1":
            faulthandler.dump_traceback_later(0.5, repeat=True, file=f, exit=False)
    except Exception:
        # Never block startup due to diagnostics setup.
        pass
