"""Logging configuration and uncaught-exception reporting."""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime

from core.config import _env_true, _share_logger, safe_print_err

from .bootstrap import (
    _consume_file_log_argv,
    _file_logging_requested,
    _runtime_log_dir,
)

__all__ = [
    "NoisyInfoFilter",
    "FocusGallerySwitchFilter",
    "setup_logging",
    "global_exception_handler",
]


def _share_log(
    level: int,
    msg: str,
    *args,
    exc_info: bool = False,
) -> None:
    """Dedicated share diagnostics (always prefixed; visible at INFO in console)."""
    _share_logger().log(level, "[SHARE] " + msg, *args, exc_info=exc_info)


class NoisyInfoFilter(logging.Filter):
    """Drop extremely chatty INFO logs unless explicitly enabled."""

    _noisy_prefixes = (
        "[RAW_PROC]",
        "[GALLERY]",
        "[DISPLAY]",
        "[DISPLAY_PIXMAP]",
        "[LOAD]",
        "[VIEW_MODE]",
        "[WINDOW_RESIZE]",
        "[PERF]",
    )

    def __init__(self, enabled: bool):
        super().__init__()
        self.enabled = enabled

    def filter(self, record: logging.LogRecord) -> bool:
        if self.enabled:
            return True
        if record.levelno != logging.INFO:
            return True
        msg = record.getMessage()
        return not msg.startswith(self._noisy_prefixes)


class FocusGallerySwitchFilter(logging.Filter):
    """Keep only gallery-switch related logs (plus warnings/errors)."""

    _allow_prefixes = (
        "[MODESWITCH]",
        "[VIEW_MODE]",
        "[GALLERY]",
        "[FOLDER]",
        "[MAIN]",
        "[ACCEL]",
        "[INDEX]",
        "[LOAD]",
        "[TTFR]",
        "[MANAGER]",
        "[SHARE]",
        "[CLOSE]",
        "[PROFILE]",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        # Always keep warnings/errors for troubleshooting.
        if record.levelno >= logging.WARNING:
            return True
        msg = record.getMessage()
        return msg.startswith(self._allow_prefixes)


def setup_logging():
    """Setup logging configuration with file and console handlers"""
    try:
        # Honor --file-log before anything treats argv as a photo path.
        if _consume_file_log_argv():
            os.environ["RAWVIEWER_FILE_LOG"] = "1"
            os.environ.setdefault("RAWVIEWER_VERBOSE_INFO_LOGS", "1")
            os.environ.setdefault("RAWVIEWER_FOCUS_GALLERY_SWITCH", "1")
            os.environ.setdefault("RAWVIEWER_REDIRECT_STDIO", "1")
            os.environ.setdefault("RAWVIEWER_FATAL_DUMP", "1")

        # Configure logging
        log_format = '%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'

        # Create formatters
        formatter = logging.Formatter(log_format, date_format)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # Clear any existing handlers to avoid duplicates
        root_logger.handlers.clear()

        # Always attach a console/stream handler when possible.
        stream = sys.stdout if sys.stdout is not None else getattr(sys, "__stdout__", None)
        focus_gallery_switch = _env_true("RAWVIEWER_FOCUS_GALLERY_SWITCH", default=False)
        is_dev = os.environ.get("RAWVIEWER_DEBUG", "").strip() in ("1", "true", "True", "YES", "yes")

        verbose_info = _env_true("RAWVIEWER_VERBOSE_INFO_LOGS", default=False) or focus_gallery_switch or is_dev

        if stream is not None:
            console_handler = logging.StreamHandler(stream)
            if is_dev:
                console_handler.setLevel(logging.DEBUG)
            else:
                console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            console_handler.addFilter(NoisyInfoFilter(verbose_info))
            if focus_gallery_switch:
                console_handler.addFilter(FocusGallerySwitchFilter())
            root_logger.addHandler(console_handler)
        else:
            # Windowed builds can have no stdout; keep logging silent unless file logging is enabled.
            root_logger.addHandler(logging.NullHandler())

        # File logging: explicit opt-in only (RAWVIEWER_FILE_LOG / --file-log /
        # enable_debug_log marker). Never auto-write session logs for privacy
        # and disk size -- console/stream handlers remain for interactive runs.
        enable_file_log = _file_logging_requested()

        if enable_file_log:
            log_dir = _runtime_log_dir()
            os.makedirs(log_dir, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(log_dir, f'rawviewer_{timestamp}.log')
            latest_file = os.path.join(log_dir, 'rawviewer_latest.log')

            file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            file_handler.addFilter(NoisyInfoFilter(verbose_info))
            if focus_gallery_switch:
                file_handler.addFilter(FocusGallerySwitchFilter())
            root_logger.addHandler(file_handler)

            # Easy-to-find alias for support: always the current session.
            try:
                latest_handler = logging.FileHandler(
                    latest_file, encoding='utf-8', mode='w'
                )
                latest_handler.setLevel(logging.DEBUG)
                latest_handler.setFormatter(formatter)
                latest_handler.addFilter(NoisyInfoFilter(verbose_info))
                if focus_gallery_switch:
                    latest_handler.addFilter(FocusGallerySwitchFilter())
                root_logger.addHandler(latest_handler)
            except OSError:
                latest_file = None

            root_logger.info("[LOG] File logging enabled: %s", log_file)
            if latest_file:
                root_logger.info("[LOG] Latest session alias: %s", latest_file)
            return log_file

        return None

    except Exception as e:
        # If logging setup fails, at least print to stderr
        error_msg = f"CRITICAL: Failed to setup logging: {e}"
        safe_print_err(error_msg)
        import traceback
        safe_print_err(f"Traceback: {traceback.format_exc()}")

        # Try to write to a fallback log file
        try:
            # Only attempt fallback file logging if explicitly enabled.
            if _file_logging_requested():
                fallback_log = os.path.join(_runtime_log_dir(), 'error.log')
                os.makedirs(os.path.dirname(fallback_log), exist_ok=True)
                with open(fallback_log, 'a', encoding='utf-8') as f:
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")
                    f.write(f"{traceback.format_exc()}\n")
        except Exception:
            pass  # If even fallback fails, we've done our best

        raise  # Re-raise to let caller handle it


def global_exception_handler(exc_type, exc_value, exc_traceback):
    """Handle uncaught exceptions and write them to a crash report log file."""
    import logging
    import traceback
    from datetime import datetime

    # Format the traceback
    try:
        tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    except Exception:
        tb_text = f"{exc_type}: {exc_value}\n(logging traceback failed)\n"

    # Try logging via the standard logger first (filters can recurse on some errors).
    try:
        logger = logging.getLogger(__name__)
        logger.critical("Uncaught Exception:\n" + tb_text)
    except Exception:
        safe_print_err("Uncaught Exception:\n" + tb_text)

    # Write to a persistent crash report file in LocalAppData
    try:
        log_dir = _runtime_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        crash_file = os.path.join(log_dir, f'crash_report_{timestamp}.txt')
        with open(crash_file, "w", encoding="utf-8") as f:
            f.write(f"RAWviewer Crash Report - {timestamp}\n")
            f.write("="*50 + "\n\n")
            f.write(tb_text)
    except Exception as e:
        safe_print_err(f"Could not write crash report file: {e}")

    # Fall back to default behavior
    sys.__excepthook__(exc_type, exc_value, exc_traceback)
