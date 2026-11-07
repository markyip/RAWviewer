# pyright: reportOptionalMemberAccess=false
import os
import sys

# Safety net only: the REAL fix is scripts/libraw/build_libraw_openmp.sh retargeting
# the custom LibRaw to torch's libomp so exactly one OpenMP runtime loads
# (two runtimes = OMP Error #15 abort, or with this flag, possible crashes /
# silently wrong results). This flag keeps envs that haven't re-run the build
# script bootable; a properly built env never exercises it.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Darkroom chrome palette (see theme.py). Zero-dependency module (no PyQt
# imports), safe this early -- needed before PyQt6 import below because the
# fallback splash-screen drawing code runs at module scope, ahead of any
# other local import in this file.

# GPU demosaic backends (CuPy / optional torch) are deferred until after the
# window is shown -- see torch_bootstrap.import_gpu_backend_on_main_thread().
# Background decode threads wait for that instead of importing on their own.
import logging
import platform
import traceback
from typing import Any

from PIL import Image

# Finite decompression-bomb cap (~200 MP). None disables Pillow's check.
Image.MAX_IMAGE_PIXELS = 200_000_000


import contextlib

from bootstrap_pkg.splash import make_splash_pixmap as _make_splash_pixmap
from bootstrap_pkg.splash import splash_show_status
from core.config import (
    _env_int,
    _env_true,
    _norm_path,
    safe_print,
    safe_print_err,
)
from rawviewer_app.window import lazy, state
from rawviewer_app.window.bootstrap import (
    _IS_GUI_MAIN_PROCESS,
    _enable_fatal_crash_dump_if_needed,
    _redirect_stdio_to_file_if_needed,
)
from rawviewer_app.window.lazy import (
    _lazy_import_heavy_modules,
    image_covers_sensor_resolution,
)
from rawviewer_app.window.logging_setup import (
    setup_logging,
)
from rawviewer_app.window.macos import (
    _disable_macos_app_nap,
    _run_applescript,
)
from rawviewer_app.window.paths import (
    _SUBFOLDER_FALLBACK_SKIP,
    _is_in_folder_scope,
    resource_path,
)
from rawviewer_app.window.platform_win import (
    _apply_windows_native_window_icon,
    _is_windows_installed_app,
    _set_windows_app_user_model_id,
)
from rawviewer_app.window.qt_helpers import (
    _qt_app,
)
from rawviewer_app.window.raw_image_viewer import RAWImageViewer
from rawviewer_app.window.splash import (
    _dismiss_startup_splash,
    _load_branded_qicon,
    _schedule_startup_splash_dismiss,
    _should_show_startup_splash,
    close_native_splash,
)

try:
    from rawviewer_app.env import configure_gui_process

    configure_gui_process(_IS_GUI_MAIN_PROCESS)
except Exception:
    pass

lazy.register(sys.modules[__name__])



if not _IS_GUI_MAIN_PROCESS:
    # Pool workers re-run the frozen binary; dismiss boot splash immediately, skip Qt.
    close_native_splash()

state._startup_splash = None  # set in ultra-fast splash block when frozen
RAWApplication = None

# Ultra Fast Splash: Qt only in the GUI main process (pool workers must not touch Cocoa).
if _IS_GUI_MAIN_PROCESS:
    try:
        from common_image_loader import raise_process_fd_limit

        raise_process_fd_limit()
    except Exception:
        pass
    try:
        from PyQt6.QtCore import QEvent, QLoggingCategory, Qt, QTimer
        from PyQt6.QtWidgets import QApplication, QSplashScreen

        # Silence qt.imageformats warnings, particularly for RAW files structured as TIFFs
        QLoggingCategory.setFilterRules(
            "qt.imageformats.warning=false\nqt.imageformats.tiff.warning=false"
        )

        class RAWApplication(QApplication):  # type: ignore[no-redef]
            """Custom QApplication to handle macOS FileOpen events"""
            def __init__(self, argv: list[str]) -> None:
                super().__init__(argv)
                # Set explicit AppUserModelID on Windows so the taskbar groups and shows the app icon correctly
                _set_windows_app_user_model_id()
                # Apply branded icon immediately (taskbar / Alt+Tab); window sets it again later.
                _early_icon = _load_branded_qicon()
                if _early_icon is not None:
                    self.setWindowIcon(_early_icon)
                self.viewer = None
                self.pending_files: list[str] = []

            def set_viewer(self, viewer: Any) -> None:
                """Set the main viewer window and load any pending files"""
                self.viewer = viewer
                for file_path in self.pending_files:
                    self._load_file(file_path)
                self.pending_files.clear()

            def event(self, event: Any) -> bool:
                """Intercept application-level events"""
                if event.type() == QEvent.Type.FileOpen:
                    file_path = event.file()
                    if file_path:
                        if self.viewer:
                            self._load_file(file_path)
                        else:
                            self.pending_files.append(file_path)
                    return True
                return super().event(event)  # type: ignore[no-any-return]

            def _load_file(self, path: str) -> None:
                """Load a file into the viewer"""
                if self.viewer is None:
                    return
                if os.path.isfile(path):
                    self.viewer.load_folder_images(
                        os.path.dirname(path),
                        start_file=os.path.basename(path),
                        start_view="single",
                    )
                elif os.path.isdir(path):
                    self.viewer.load_folder_images(path)

        # We need a temporary app just to show the splash
        _temp_app = QApplication.instance()
        if not _temp_app:
            _temp_app = RAWApplication(sys.argv)  # type: ignore[misc]

        # Compact branded splash (full-res PNG is far too large as QSplashScreen).
        _splash_pixmap = _make_splash_pixmap()

        if _should_show_startup_splash():
            # SplashScreen keeps it out of the taskbar; Tool avoids an Alt+Tab entry.
            _splash_flags = (
                Qt.WindowType.SplashScreen
                | Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.Tool
            )
            if sys.platform != "darwin":
                _splash_flags |= Qt.WindowType.WindowStaysOnTopHint
            state._startup_splash = QSplashScreen(_splash_pixmap, _splash_flags)
            state._startup_splash.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
            state._startup_splash.show()
            splash_show_status(state._startup_splash, "Starting…")
            _temp_app.processEvents()
        else:
            state._startup_splash = None

        close_native_splash()
    except Exception:
        _startup_splash = None

# Early diagnostics: redirect stdio and optionally install crash dumps.
_redirect_stdio_to_file_if_needed()
_enable_fatal_crash_dump_if_needed()



# Public facade: names historically importable from ``main``.
__all__ = [
    "RAWImageViewer",
    "main",
    "resource_path",
    "_norm_path",
    "_is_in_folder_scope",
    "_SUBFOLDER_FALLBACK_SKIP",
    "_run_applescript",
    "_env_int",
    "_env_true",
    "safe_print",
    "safe_print_err",
    "setup_logging",
]


def main() -> None:
    """Main function to run the application"""
    import traceback

    # GIL responsiveness: thumbnail/metadata workers run long pure-Python
    # stretches (TIFF preview scans, orientation probes) that hold the GIL for
    # the full 5ms default switch interval each; with 8+ workers the Qt event
    # loop gets starved and trackpad scrolling stutters (measured >1s main-thread
    # stalls). A shorter interval trades a little worker throughput for UI
    # latency. Override with RAWVIEWER_GIL_SWITCH_MS (0 disables).
    try:
        _gil_ms = float(os.environ.get("RAWVIEWER_GIL_SWITCH_MS", "2") or "2")
        if _gil_ms > 0:
            sys.setswitchinterval(_gil_ms / 1000.0)
    except Exception:
        pass

    # Cap OpenMP threads before rawpy/LibRaw loads. LibRaw decodes are
    # OpenMP-parallel on Windows (rawpy wheel ships vcomp140) and on macOS
    # via scripts/libraw/build_libraw_openmp.sh; with 2 concurrent RAW workers an
    # uncapped pool (all cores each) starves the GUI thread (observed as a
    # full app freeze on M1). Half the cores, clamped to 2..8, keeps most of
    # the per-decode speedup while leaving headroom for the GUI and the
    # second worker. Override by exporting OMP_NUM_THREADS.
    _cores = os.cpu_count() or 8
    os.environ.setdefault("OMP_NUM_THREADS", str(max(2, min(8, _cores // 2))))

    # macOS: opt out of App Nap before any pipeline timers are created --
    # throttled QTimers delay the staged full decode by 9-15s when the app
    # launches unfocused or sits idle.
    _disable_macos_app_nap()

    # Packaged or installed app: enable semantic search by default (dev uses launch_dev.sh / run_debug.bat).
    # Lite builds bake PROFILE=lite and skip AI/face via rawviewer_profile defaults.
    from core.profile import (
        apply_memory_tier_defaults,
        apply_profile_runtime_defaults,
        is_lite_build,
        memory_tier_startup_summary,
    )
    from core.upgrade_compat import apply_upgrade_compat_defaults

    apply_profile_runtime_defaults()
    tier = apply_memory_tier_defaults()
    compat = apply_upgrade_compat_defaults()
    logger = logging.getLogger(__name__)
    if tier != "disabled":
        logger.info("[PROFILE] %s (compat=%s)", memory_tier_startup_summary(), compat)
        safe_print(f"[PROFILE] {memory_tier_startup_summary()} (compat={compat})", flush=True)
    else:
        logger.info("[PROFILE] upgrade compat=%s", compat)
        safe_print(f"[PROFILE] upgrade compat={compat}", flush=True)

    # On Windows, the installed app runs inside a Pixi virtual environment (so sys.frozen is False),
    # but we can detect it by checking if "_internal/pixi/pixi.exe" exists in the root directory.
    is_installed = _is_windows_installed_app()

    if (getattr(sys, "frozen", False) or is_installed) and not is_lite_build():
        os.environ.setdefault("RAWVIEWER_ENABLE_SEMANTIC_SEARCH", "1")
        os.environ.setdefault("RAWVIEWER_ENABLE_FACE_SCAN", "1")
        os.environ.setdefault("RAWVIEWER_AUTO_METADATA_INDEX", "1")
        os.environ.setdefault("RAWVIEWER_PERF_V2", "1")
        os.environ.setdefault("RAWVIEWER_GPU_VIEW", "1")
        if sys.platform == "darwin":
            os.environ.setdefault("RAWVIEWER_PREFER_GPU_DECODE", "0")
    elif getattr(sys, "frozen", False) or is_installed:
        os.environ.setdefault("RAWVIEWER_GPU_VIEW", "1")
        os.environ.setdefault("RAWVIEWER_AUTO_METADATA_INDEX", "1")
        if sys.platform == "darwin":
            os.environ.setdefault("RAWVIEWER_PREFER_GPU_DECODE", "0")

    # One-line dump so packaged .app vs launch_dev.sh configs are easy to compare.
    try:
        from core.profile import edition_display_name, resolved_profile

        _cfg = (
            f"edition={edition_display_name()}({resolved_profile()}) "
            f"frozen={bool(getattr(sys, 'frozen', False))} "
            f"semantic={os.environ.get('RAWVIEWER_ENABLE_SEMANTIC_SEARCH', '0')} "
            f"face={os.environ.get('RAWVIEWER_ENABLE_FACE_SCAN', '1')} "
            f"gpu_decode={os.environ.get('RAWVIEWER_PREFER_GPU_DECODE', '') or 'auto'} "
            f"gpu_view={os.environ.get('RAWVIEWER_GPU_VIEW', '1')} "
            f"perf_v2={os.environ.get('RAWVIEWER_PERF_V2', '0')} "
            f"gallery_idle_batch={os.environ.get('RAWVIEWER_GALLERY_IDLE_PRELOAD_BATCH', 'default')} "
            f"load_workers={os.environ.get('RAWVIEWER_LOAD_MAX_WORKERS', 'default')}"
        )
        logger.info("[PROFILE] runtime config: %s", _cfg)
        safe_print(f"[PROFILE] runtime config: {_cfg}", flush=True)
    except Exception:
        pass

    # Print to console immediately (before logging might be ready)
    safe_print("main() function called", flush=True)

    # Logging should already be setup in if __name__ == '__main__'
    # But check if it's configured, if not, setup it
    logger = logging.getLogger(__name__)
    if not logger.handlers and not logging.getLogger().handlers:
        try:
            import platform as _platform_mod

            platform_system = _platform_mod.system()
            platform_release = _platform_mod.release()
            setup_logging()
            logger.info("=" * 80)
            logger.info("Application startup started")
            logger.info(f"Python version: {sys.version}")
            logger.info(f"Platform: {platform_system} {platform_release}")
            logger.info(f"Working directory: {os.getcwd()}")
            logger.info("=" * 80)
        except Exception as log_error:
            # If logging setup fails, at least print to stderr
            safe_print_err(f"ERROR: Failed to setup logging: {log_error}")
            safe_print_err(f"Traceback: {traceback.format_exc()}")
    else:
        # Logging already configured, just log startup
        safe_print("[MAIN] Logging already configured, logging startup info...", flush=True)
        logger.info("=" * 80)
        safe_print("[MAIN] Logger.info('=' * 80) called", flush=True)
        logger.info("Application startup started")
        safe_print("[MAIN] Logger.info('Application startup started') called", flush=True)
        logger.info(f"Python version: {sys.version}")
        safe_print(f"[MAIN] Python version logged: {sys.version}", flush=True)
        safe_print("[MAIN] Getting platform info...", flush=True)
        # Temporarily skip platform info to avoid potential blocking
        # Use hardcoded values for Windows
        safe_print("[MAIN] Using hardcoded platform info (Windows) to avoid blocking...", flush=True)
        platform_system = "Windows"
        platform_release = "10"
        try:
            # Try to get real platform info, but don't block if it fails
            import threading
            platform_result: list[Any] = [None, None]
            def get_platform_info() -> None:
                try:
                    platform_result[0] = platform.system()
                    platform_result[1] = platform.release()
                except Exception:
                    pass

            thread = threading.Thread(target=get_platform_info, daemon=True)
            thread.start()
            thread.join(timeout=0.1)  # Wait max 100ms
            if platform_result[0] and platform_result[1]:
                platform_system = platform_result[0]
                platform_release = platform_result[1]
                safe_print(f"[MAIN] Got platform info: {platform_system} {platform_release}", flush=True)
            else:
                safe_print(f"[MAIN] Using fallback platform info: {platform_system} {platform_release}", flush=True)
        except Exception as e:
            safe_print(f"[MAIN] Error getting platform info, using fallback: {e}", flush=True)

        safe_print("[MAIN] Calling logger.info for platform...", flush=True)
        logger.info(f"Platform: {platform_system} {platform_release}")
        safe_print(f"[MAIN] Platform logged: {platform_system} {platform_release}", flush=True)
        logger.info(f"Working directory: {os.getcwd()}")
        safe_print(f"[MAIN] Working directory logged: {os.getcwd()}", flush=True)
        logger.info("=" * 80)
        safe_print("[MAIN] All platform info logged, setting up Windows exception handler...", flush=True)

    # Set up Windows exception handler to catch access violations
    # Use platform_system variable to avoid calling platform.system() again
    is_windows = (platform_system == 'Windows')
    if is_windows:
        safe_print("  [Windows] Importing ctypes...", flush=True)
        safe_print("  [Windows] ctypes imported", flush=True)
        safe_print("  [Windows] Importing wintypes...", flush=True)
        safe_print("  [Windows] wintypes imported", flush=True)

        # Define exception handler function
        def exception_handler(exception_info: Any) -> int:
            """Handle Windows exceptions (access violations, etc.)"""
            exception_code = exception_info[0].ExceptionRecord[0].ExceptionCode
            exception_address = exception_info[0].ExceptionRecord[0].ExceptionAddress

            # 0xC0000005 is ACCESS_VIOLATION
            if exception_code == 0xC0000005:
                error_msg = f"Access Violation (0xC0000005) at address {exception_address}"
                logger.critical(f"Windows Access Violation: {error_msg}")
                logger.critical("This usually indicates accessing invalid memory (null pointer, freed object, etc.)")
                safe_print_err(f"\n{'='*80}")
                safe_print_err("WINDOWS ACCESS VIOLATION")
                safe_print_err(f"{'='*80}")
                safe_print_err(f"{error_msg}")
                safe_print_err(f"{'='*80}\n")
                return 1  # EXCEPTION_EXECUTE_HANDLER
            return 0  # EXCEPTION_CONTINUE_SEARCH

        # Note: Setting up structured exception handling in Python is complex
        # We'll rely on Python's exception handling and add more defensive checks
        safe_print("  [Windows] Exception handler setup complete", flush=True)
    else:
        safe_print("  [Non-Windows] Skipping Windows exception handler", flush=True)

    safe_print("Entering main try block...", flush=True)
    try:
        splash = state._startup_splash
        try:
            # 1. Use pre-created Application instance
            app = QApplication.instance()
            if not app:
                app = RAWApplication(sys.argv)  # type: ignore[misc]
            safe_print("Application instance ready", flush=True)

            # 2. Update existing splash if present
            if splash:
                splash_show_status(splash, "Starting…")
                safe_print("Splash screen updated", flush=True)
            elif _should_show_startup_splash():
                # Fallback splash if top-level creation failed (frozen or Windows installed).
                splash_pixmap = _make_splash_pixmap()
                splash = QSplashScreen(
                    splash_pixmap,
                    (
                        Qt.WindowType.SplashScreen
                        | Qt.WindowType.FramelessWindowHint
                        | Qt.WindowType.Tool
                    )
                    if sys.platform == "darwin"
                    else (
                        Qt.WindowType.SplashScreen
                        | Qt.WindowType.FramelessWindowHint
                        | Qt.WindowType.Tool
                        | Qt.WindowType.WindowStaysOnTopHint
                    ),
                )
                splash.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
                splash.show()
                splash_show_status(splash, "Starting…")
                safe_print("Fallback splash screen displayed", flush=True)
            else:
                splash = None

            # 3. Import heavy modules while splash is visible
            _lazy_import_heavy_modules(splash)

        finally:
            # 4. Continue with initialization
            if is_windows:
                # Keep the same AppUserModelID set at process start (do not
                # overwrite with a versioned ID -- that drops the taskbar icon).
                safe_print("  [Windows] Ensuring AppUserModelID...", flush=True)
                _set_windows_app_user_model_id()
                safe_print("  [Windows] AppUserModelID set", flush=True)

            # Set application properties
            from core.release_update import APP_VERSION

            app = _qt_app()
            app.setApplicationName("RAWviewer")
            app.setApplicationVersion(APP_VERSION)
            _brand_icon = _load_branded_qicon()
            if _brand_icon is not None:
                app.setWindowIcon(_brand_icon)

            # Create and show main window
            safe_print("Creating RAWImageViewer...", flush=True)
            viewer = RAWImageViewer()
            safe_print("RAWImageViewer created successfully", flush=True)

            # Connect viewer to application to handle macOS file open events
            app.set_viewer(viewer)

            # Check for file or folder argument
            if len(sys.argv) > 1:
                path = sys.argv[1]
                if os.path.isfile(path):
                    # If it's a file, load the folder containing that file
                    viewer.load_folder_images(
                        os.path.dirname(path),
                        start_file=os.path.basename(path),
                        start_view="single",
                    )
                elif os.path.isdir(path):
                    # If it's a folder, load the folder
                    viewer.load_folder_images(path)

            # Show main window and close splash screen
            viewer.show()
            if is_windows:
                _apply_windows_native_window_icon(viewer)
            # Paint the main window under the splash before dismissing, so the
            # splash does not uncover an empty desktop flash.
            with contextlib.suppress(Exception):
                app.processEvents()
            QTimer.singleShot(0, viewer._restore_keyboard_focus)
            _schedule_startup_splash_dismiss(app, viewer, splash)
            safe_print("Splash screen closed, main window displayed", flush=True)

            # Import torch/kornia on the main thread after first paint (see
            # torch_bootstrap.py). Must stay on the GUI thread: OpenMP first
            # init off-main hard-aborts on macOS. Prefer scheduling from
            # session-restore nav preload; this paint-based path is the
            # fallback when restore staging did not run (empty session, etc.).
            viewer._defer_until_first_paint(
                lambda: viewer._schedule_deferred_gpu_backend_import(
                    reason="after_first_paint"
                ),
                fallback_ms=_env_int(
                    "RAWVIEWER_GPU_IMPORT_DEFER_FALLBACK_MS", 8000, minimum=0
                ),
                label="gpu_backend_import",
            )

            if str(os.environ.get("RAWVIEWER_GALLERY_AUTOTEST", "0")).strip().lower() in ("1", "true", "yes"):
                from gallery_autotest import run_gallery_autotest

                run_gallery_autotest(viewer, safe_print=safe_print)

            if str(os.environ.get("RAWVIEWER_AUTOTEST", "0")).strip().lower() in ("1", "true", "yes"):
                from nav_autotest import run_nav_autotest

                run_nav_autotest(
                    viewer,
                    safe_print=safe_print,
                    image_covers_sensor_resolution=image_covers_sensor_resolution,
                )

        # Run application
        logger.info("[MAIN] Starting Qt event loop")
        exit_code = app.exec()

        # Check exit code for access violations
        if exit_code == -1073741819:  # 0xC0000005 - Access Violation
            logger.critical("[MAIN] Application crashed with Access Violation (0xC0000005)")
            logger.critical("[MAIN] This usually indicates:")
            logger.critical("[MAIN]   1. Accessing invalid memory (null pointer, freed object)")
            logger.critical("[MAIN]   2. Qt object accessed from wrong thread")
            logger.critical("[MAIN]   3. Memory corruption in rawpy/Qt")
            logger.critical("[MAIN]   4. Signal/slot connection to deleted object")
            safe_print_err(f"\n{'='*80}")
            safe_print_err("ACCESS VIOLATION DETECTED (0xC0000005)")
            safe_print_err(f"{'='*80}")
            safe_print_err("This error indicates the application tried to access invalid memory.")
            safe_print_err("Possible causes:")
            safe_print_err("  - Qt object accessed from wrong thread")
            safe_print_err("  - Accessing deleted/freed object")
            safe_print_err("  - Memory corruption in rawpy or Qt library")
            safe_print_err("  - Signal connected to deleted slot")
            safe_print_err("\nCheck the log file for detailed information.")
            safe_print_err(f"{'='*80}\n")

        logger.info(f"[MAIN] Application exited with code: {exit_code}")
        # Use os._exit to force kill any lingering background threads (like database connections or rawpy)
        # sys.exit() only raises SystemExit and waits for non-daemon threads
        logger.info(f"[MAIN] Force exiting process with os._exit({exit_code})")
        os._exit(exit_code)

    except KeyboardInterrupt:
        logger.info("Application interrupted by user (Ctrl+C)")
        safe_print("\n[INFO] Application interrupted by user (Ctrl+C)")
        sys.exit(0)
    except SystemExit as e:
        # Re-raise SystemExit to preserve exit code
        logger.info(f"SystemExit with code: {e.code}")
        raise
    except Exception as e:
        # Catch all other exceptions and log them before crashing
        error_msg = f"FATAL ERROR: {type(e).__name__}: {e}"
        error_traceback = traceback.format_exc()

        logger.critical(f"Fatal error: {error_msg}")
        logger.critical(f"Full traceback:\n{error_traceback}")

        # Also print to console/stderr so it's visible even if logging fails
        safe_print_err(f"\n{'='*80}")
        safe_print_err("FATAL ERROR")
        safe_print_err(f"{'='*80}")
        safe_print_err(f"{error_msg}")
        safe_print_err("\nFull traceback:")
        safe_print_err(f"{error_traceback}")
        safe_print_err(f"{'='*80}\n")

        # Try to show error dialog if possible
        try:
            app = QApplication.instance()
            _dismiss_startup_splash(app, None, None)

            if app is not None:
                from PyQt6.QtWidgets import QMessageBox
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Icon.Critical)
                msg.setWindowTitle("Fatal Error")
                msg.setText("The application encountered a fatal error and will now exit.")
                msg.setDetailedText(f"{error_msg}\n\n{error_traceback}")
                msg.exec()
        except Exception:
            pass  # If we cannot show dialog, at least we logged it

        sys.exit(1)


if __name__ == '__main__' and _IS_GUI_MAIN_PROCESS:
    # Print startup message to console immediately
    safe_print("=" * 80, flush=True)
    safe_print("RAWviewer Starting...", flush=True)
    safe_print("=" * 80, flush=True)

    try:
        from core.profile import apply_runtime_defaults, memory_tier_startup_summary

        tier = apply_runtime_defaults()
        if tier != "disabled":
            safe_print(f"[PROFILE] {memory_tier_startup_summary()}", flush=True)
    except Exception:
        pass

    # Setup logging before anything else
    try:
        safe_print("Setting up logging...", flush=True)
        setup_logging()
        safe_print("Logging setup complete.", flush=True)
    except Exception as e:
        safe_print_err(f"ERROR: Failed to setup logging: {e}", flush=True)
        import traceback
        safe_print_err(f"Traceback: {traceback.format_exc()}", flush=True)

    try:
        safe_print("Calling main()...", flush=True)
        main()
    except Exception as e:
        safe_print_err(f"\n{'='*80}", flush=True)
        safe_print_err(f"FATAL ERROR in main(): {type(e).__name__}: {e}", flush=True)
        import traceback
        safe_print_err(f"Traceback:\n{traceback.format_exc()}", flush=True)
        safe_print_err(f"{'='*80}\n", flush=True)
        raise
