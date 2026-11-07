"""lifecycle mixin, part 1 of 2 (12 methods, 825 lines)."""
from __future__ import annotations

import contextlib
import logging
import os
import platform
import sys
import threading
import time

from PyQt6.QtCore import QPoint, QRect, Qt, QTimer

from cache.memory import FullImageCache
from core.config import _env_int, _env_true, _norm_path, safe_print
from rawviewer_app.processing import RAWProcessor

from .. import lazy
from ..lazy import PreloadManager, get_image_cache, get_image_load_manager, np
from ..logging_setup import _share_log
from ..platform_win import _apply_windows_native_window_icon
from ..qt_helpers import _qt_app

lazy.register(sys.modules[__name__])


class LifecyclePart1:
    def __init__(self):
        safe_print("  [RAWImageViewer] Starting initialization...", flush=True)
        super().__init__()
        safe_print("  [RAWImageViewer] QMainWindow.__init__() completed", flush=True)

        # Configure vertical scrollbars to jump directly to absolute cursor position when clicked
        try:
            from PyQt6.QtWidgets import QProxyStyle, QStyle
            class ScrollBarJumpStyle(QProxyStyle):
                def styleHint(self, hint, option=None, widget=None, returnData=None):
                    if hint == QStyle.StyleHint.SH_ScrollBar_LeftClickAbsolutePosition:
                        return 1
                    return super().styleHint(hint, option, widget, returnData)
            app = _qt_app()
            if app:
                # Keep a persistent Python reference on the app object to prevent garbage collection!
                app._scrollbar_jump_style = ScrollBarJumpStyle()
                app.setStyle(app._scrollbar_jump_style)
        except Exception:
            pass

        self.current_image = None
        self.current_pixmap = None

        # Content cross-fade (thumbnail -> preview -> full upgrades, and navigation between files).
        # One-shot flag set by display_pixmap(); consumed by _set_single_view_pixmap().
        # Zoom/pan/resize paint through _set_single_view_pixmap directly without setting this
        # flag, so they never trigger a fade.
        self._crossfade_enabled = not _env_true("RAWVIEWER_DISABLE_CROSSFADE", default=False)
        self._crossfade_duration_ms = 160
        self._gallery_crossfade_duration_ms = 240
        self._resolution_crossfade_ms = _env_int(
            "RAWVIEWER_RESOLUTION_CROSSFADE_MS", 280, minimum=80
        )
        self._pending_resolution_crossfade = False
        self._crossfade_use_resolution_duration = False
        self._crossfade_next_paint = False
        self._crossfade_overlay = None
        self._crossfade_anim = None

        # Route B GPU view: track last displayed path to decide fit-vs-preserve on new pixmaps.
        self._gpu_last_path = None
        # Bumped on file switch to ignore stale deferred cross-fade apply callbacks.
        self._single_view_display_generation = 0

        # Enhanced zoom and pan state tracking
        # Note: Only using simple toggle between fit-to-window and 100% zoom
        self.current_zoom_level = 1.0  # Current zoom level (1.0 = 100%)
        self.fit_to_window = True  # Whether we're in fit-to-window mode
        self.zoom_center_point = None  # Store center point for zooming
        self._pending_zoom_focus_subject = False  # Space+focus box zoom waiting for full-res
        self._focus_zoom_anchor_active = False  # Keep focus anchor across buffer upgrades
        self._workflow_toggle_fit_until_display = False
        self._workflow_toggle_refit_timer_scheduled = False

        self._is_half_size_displayed = False  # Track if currently displaying half_size image
        self._full_resolution_loading = False  # Track if full resolution is being loaded
        self._suppress_single_manager_callbacks = False  # Hard gate for single-view callbacks
        self._original_image_size = None  # Store original image dimensions (width, height) from EXIF or RAW metadata
        # Panning state
        self.panning = False
        self.last_pan_point = QPoint()
        self.start_scroll_x = 0
        self.start_scroll_y = 0

        # Folder scanning and file list management
        self.current_folder = None
        self.image_files = []  # List of all image files in current folder
        self.current_file_index = -1  # Index of current file in the list
        self.current_file_path = None  # Path of currently loaded file
        # Highest max dimension already shown for current file (drops late stale thumbnails)
        self._manager_display_track_path = None
        self._manager_displayed_max_dim = 0
        # Path whose pixel data last written to ``current_pixmap`` (reject stale pixmap vs new file guards)
        self._displayed_content_path = None
        # End-to-end single-view timing: load request -> first visible paint.
        self._single_view_load_request_ts = 0.0
        self._single_view_load_request_path = None
        self._single_view_first_render_logged = False
        self._session_restore_defer_preload = False
        self._session_restore_prefetch_timer = None
        self._session_restore_staged_scheduled = False
        self._session_restore_full_decode_allowed = False
        self._user_zoom_interaction_ts = 0.0
        self._single_view_ttfr_watch_generation = 0
        # Simplified loading: a single idle timer requests full-res after the
        # user has stayed on one image. Replaces TTFR + full-res-upgrade watchdogs.
        self._full_res_idle_timer = QTimer(self)
        self._full_res_idle_timer.setSingleShot(True)
        self._full_res_idle_timer.timeout.connect(self._on_full_res_idle_timeout)
        self._full_res_idle_path = None
        self._on_manager_thumbnail_ready_in_progress = False
        self._on_manager_image_ready_in_progress = False
        self._on_single_view_content_displayed_in_progress = False
        self._defer_sensor_full_decode_path = None
        self._sensor_full_decode_queued_norm = None
        self._defer_full_exif_path = None
        self._full_exif_queued_norm = None
        self.thumbnail_cache = {}  # Cache for thumbnails
        self._histogram_user_hidden = True
        self._adjust_user_hidden = True
        self._adjust_overlay_visible = False
        # RAW view (toggle_raw_jpeg_workflow) is a persistent app-wide setting,
        # but it is a *session* choice in practice: full RAW decode costs a
        # multiple of the embedded-preview path on every navigation, and coming
        # back to a launch that silently starts in RAW mode reads as "the app is
        # slow" rather than "RAW mode is still on". Always start in the embedded-
        # preview workflow; the user re-enables RAW per session.
        with contextlib.suppress(Exception):
            self.get_settings().setValue("use_embedded_jpeg_workflow", True)
        self.thumbnail_threads = []  # Track running thumbnail threads

        # View mode: 'single' for single image view, 'gallery' for gallery view
        self.view_mode = 'single'  # Default to single image view
        self._filmstrip_pointer_active = False
        self._filmstrip_warmed_paths = set()
        # Gallery functionality enabled
        self.gallery_widget = None  # Gallery view widget
        self.gallery_justified = None  # JustifiedGallery widget
        self.gallery_scroll = None  # Gallery scroll area
        self.gallery_frame_gutter = None  # Overlay on gallery right edge for frame resize
        self.gallery_row_height = 200  # Fixed row height (matching reference)
        self.gallery_pixmaps = {}  # Cache for gallery pixmaps (store original pixmaps)
        self.gallery_aspect_cache = {}  # Cache aspect ratios
        self._gallery_thumb_labels = {}  # Store thumbnail labels
        self._gallery_load_tracking = {}  # Track loading status
        self._gallery_load_start_time = None  # Track loading start time
        self._loading_from_gallery = False  # Flag for gallery loading
        # Gallery→single: allow embedded JPEG until idle upgrade / next non-gallery load
        # (survives clearing _loading_from_gallery mid-load).
        self._gallery_entry_accept_embedded = False
        self._loading_from_filmstrip = False  # Flag for filmstrip loading
        self._gallery_selected_paths = set()  # normpath keys for multi-select in gallery
        self._gallery_selection_anchor = None  # normpath key for Shift+click range
        self._gallery_last_select_action = None  # "toggle" | "shift": gate shift-range extension
        # Burst grouping + compare (H)
        self._burst_groups = []
        self._burst_path_to_group = {}
        self._burst_group_covers = {}
        self._burst_stack_covers = {}
        self._burst_group_view_active = False
        self._burst_group_view_members = []
        self._burst_rejected_paths = set()
        self._burst_grouping_enabled = False
        self._comparison_session = None
        self._compare_sync_mode = "A"
        self._compare_sync_paused = False
        self._compare_sync_guard = False
        self._compare_focus_outline_active = False
        self.compare_widget = None
        self.compare_select_view = None
        self.compare_candidate_view = None
        self._gallery_bookmarked_paths = set()  # normpath keys for bookmarked share targets
        self._gallery_bookmark_filter_active = False  # gallery-only: show bookmarked images only
        self._gallery_rating_filter_min = 0  # gallery-only: 0=off, else show rating >= N
        self._gallery_rating_presence_epoch = 0
        self._gallery_any_rated_cache = None
        self._gallery_preview_pending_full = False  # Gallery thumb on screen; full-res pending
        self._gallery_instant_display_quality = False  # True when gallery→single already shows preview/full
        self._skip_resolution_crossfade_once = False
        self._gallery_crossfade_next = False

        # Background task tracking for stability
        self._active_metadata_fetcher = None  # Store QRunnable to prevent GC
        self._active_metadata_signals = None  # Store signals to prevent GC
        self._gallery_metadata_fetch_in_progress = False
        self._navigation_in_progress = False  # Flag to prevent overlapping navigations
        self._last_navigation_time = 0  # Timestamp of last navigation for rate limiting
        self._nav_coalesce_delta = 0
        self._nav_coalesce_timer = None
        self._nav_idle_full_generation = 0
        # Navigate while zoomed: request full decode first & apply one-step zoom restore
        self._preserve_nav_zoom_active = False
        # True while a low-res tier is shown at equivalent zoomed framing
        # (RAW-mode zoomed navigation); sub-sensor tiers are held back until
        # the sensor-res buffer clears it.
        self._nav_zoom_interim_active = False
        # Per-file zoom memory for zoomed navigation: norm_path ->
        # (frac_x, frac_y, sensor_level). Returning to a file restores the
        # framing the user last had *on that file* (fractions are buffer- and
        # orientation-independent), instead of carrying the neighbor's framing
        # across differently-oriented images.
        self._nav_zoom_memory = {}

        self._slideshow_timer = None
        self._slideshow_force_fit_next = False
        self._slideshow_scope_paths = None  # when set, slideshow cycles this list only
        # Non-destructive per-file visual rotation (clockwise degrees: 0/90/180/270)
        self._visual_rotation_degrees = {}
        self._load_persisted_visual_rotations()
        self._semantic_index = None
        self._pending_pause_semantic_index = False
        self._pending_abort_semantic_index = False
        self._semantic_search_backup_files = None
        self._semantic_search_result_paths = None
        # Full unfiltered file list for semantic index/search corpus in current folder.
        self._semantic_search_corpus_files = []
        self._semantic_index_active_token = None
        self._semantic_indexing_in_progress = False
        self._face_index_active_token = None
        self._face_indexing_in_progress = False
        self._face_index_signals = None
        self._semantic_asset_download_in_progress = False

        # Ensure main window can handle shortcuts immediately
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()
        self._semantic_asset_download_signals = None
        self._mobileclip_download_dismissed_this_session = False
        self._mobileclip_download_dialog = None
        self._last_semantic_query = ""
        self._semantic_index_progress_base = 0
        self._semantic_index_progress_total = 0
        self._semantic_coverage_cache = None
        self._semantic_coverage_cache_ts = 0.0
        # Gallery: user hid search strip during index/download -- skip auto-expand from status updates.
        self._gallery_search_user_collapsed_while_busy = False
        self._gallery_search_placeholder_saved = ""
        self._gallery_search_index_session_key = None

        # Focus-area dashed outline from EXIF / maker AF (toggle with F)
        self._focus_subject_outline_active = False
        # Composition guide overlay (toggle with G): off → 3×3 → diagonal → both → phi grid
        try:
            self._composition_grid_mode = str(self.get_settings().value("composition_grid_mode", "off"))
            if self._composition_grid_mode == "golden_spiral":
                self._composition_grid_mode = "golden"
        except Exception:
            self._composition_grid_mode = "off"
        # Highlight/shadow clipping overlay (toggle with J)
        self._clipping_overlay_active = False
        self._clipping_overlay_pm = None
        self._clipping_overlay_cache_key = None
        # RAW local shadow/highlight recovery (P -- session-only, single-view RAW flow)
        self._raw_recovery_active = False
        self._raw_recovery_cache = {}
        self._raw_recovery_skip_display_cache = False
        self._raw_recovery_worker_path = None
        from rawviewer_app.signals import RawRecoverySignals

        self._raw_recovery_signals = RawRecoverySignals()
        self._raw_recovery_signals.ready.connect(self._on_raw_recovery_ready)
        self._raw_recovery_signals.failed.connect(self._on_raw_recovery_failed)
        self._display_edr_mode = None  # None | "raw" | "hdr" when showing EDR content
        # EXIF SubjectArea / maker AF (pyexiv2/Exiv2) in current_pixmap coordinates
        self._focus_subject_rect_image: QRect | None = None
        # "makernote_af" (exifread Canon AF) vs "exif_subject" (SubjectArea/Location via Exiv2)
        self._focus_rect_source: str | None = None
        # After arrow-key nav while zoomed + focus outline: re-anchor on the new image's box.
        self._recenter_on_focus_after_nav = False

        # Resize event handling
        self._is_resizing = False  # Flag to track when window is being actively resized

        # Cleanup concurrency control
        self._cleanup_lock = threading.Lock()  # Lock to prevent multiple cleanup operations simultaneously
        self._cleanup_in_progress = False  # Flag to track if cleanup is in progress

        # Initialize enhanced performance components
        safe_print("  [RAWImageViewer] Initializing image cache...", flush=True)
        self.image_cache = get_image_cache()
        safe_print("  [RAWImageViewer] Image cache initialized", flush=True)
        # Initialize simplified full image cache (LRU 10)
        safe_print("  [RAWImageViewer] Initializing full image cache (LRU 10)...", flush=True)
        self.full_image_cache = FullImageCache(max_size=10)
        safe_print("  [RAWImageViewer] Full image cache initialized", flush=True)
        # Pass RAWProcessor class to PreloadManager for consistent processing (legacy support)
        safe_print("  [RAWImageViewer] Initializing PreloadManager...", flush=True)
        self.preload_manager = PreloadManager(max_preload_threads=8, processor_class=RAWProcessor)
        safe_print("  [RAWImageViewer] PreloadManager initialized", flush=True)
        self.current_processor = None  # Legacy support - will be phased out
        self._pending_thumbnail = None  # Store thumbnail when not immediately displayed
        self._preview_upgrade_attempted_paths: set[str] = set()
        self._raw_workflow_refused_tier_paths: set[str] = set()
        self._exif_data_ready = False  # Flag to track if EXIF data is available

        # Initialize new unified image load manager
        safe_print("  [RAWImageViewer] Initializing ImageLoadManager...", flush=True)
        self.image_manager = get_image_load_manager(max_workers=4)
        safe_print("  [RAWImageViewer] ImageLoadManager initialized", flush=True)
        safe_print("  [RAWImageViewer] Connecting ImageLoadManager signals...", flush=True)
        self._connect_image_manager_signals()
        self._save_session_debounce_timer = QTimer(self)
        self._save_session_debounce_timer.setSingleShot(True)
        self._save_session_debounce_timer.setInterval(420)
        self._save_session_debounce_timer.timeout.connect(self.save_session_state)

        self._map_overlay_visible = False
        self._map_user_hidden = True
        self._map_online = None
        self._map_gallery_title = None
        self._map_gallery_return_path = None
        self._map_sync_timer = QTimer(self)
        self._map_sync_timer.setSingleShot(True)
        self._map_sync_timer.timeout.connect(self._sync_location_map_now)
        self._defer_post_deletion_load_generation = 0

        # Deferred metadata indexing components
        self._metadata_index_defer_timer = QTimer(self)
        self._metadata_index_defer_timer.setSingleShot(True)
        self._metadata_index_defer_timer.timeout.connect(self._maybe_start_background_metadata_index)

        self._idle_display_prefetch_timer = QTimer(self)
        self._idle_display_prefetch_timer.setSingleShot(True)
        self._idle_display_prefetch_timer.timeout.connect(self._idle_display_prefetch_tick)
        self._idle_display_prefetch_cursor = 0
        self._memory_pressure_timer = QTimer(self)
        self._memory_pressure_timer.setInterval(4000)
        self._memory_pressure_timer.timeout.connect(self._on_memory_pressure_tick)
        QTimer.singleShot(5000, self._start_memory_pressure_monitor)
        # Safety net: re-arm the full-res upgrade watchdog if a preview is stuck
        # on screen and the normal paint-driven arming was missed (fast nav).
        if self._full_res_only_mode():
            safe_print("[FULL_RES_ONLY] Enabled - interim previews suppressed", flush=True)
        # Note: _full_res_upgrade_safety_timer was deprecated in Phase 1
        # and removed to simplify the TTFR + full-res upgrade system
        # The new idle timer system uses _full_res_idle_timer instead
        # _full_res_upgrade_safety_timer and _full_res_upgrade_safety_tick are no longer used
        # _full_res_upgrade_safety_tick functionality is covered by the new idle timer system
        self._metadata_index_run_done = False
        self._metadata_index_prep_complete = False
        self._metadata_index_extract_pending_count = 0
        self._metadata_index_deferred_corpus = None
        self._metadata_index_deferred_coverage = None
        self._metadata_index_deferred_pending = None
        self._last_single_view_interactivity_ts = 0.0
        self._single_view_index_defer_opened_ts = 0.0

        safe_print("  [RAWImageViewer] ImageLoadManager signals connected", flush=True)

        # Thumbnail display preferences
        # User preference: show thumbnails even at 100% zoom
        self.show_thumbnails_when_zoomed = False

        # Connect cache signals for performance monitoring
        safe_print("  [RAWImageViewer] Connecting cache signals...", flush=True)
        self.image_cache.cache_hit.connect(self.on_cache_hit)
        self.image_cache.memory_warning.connect(self.on_memory_warning)
        safe_print("  [RAWImageViewer] Cache signals connected", flush=True)

        safe_print("  [RAWImageViewer] Initializing UI...", flush=True)
        self.init_ui()
        safe_print("  [RAWImageViewer] UI initialized", flush=True)
        self._load_persisted_overlay_positions()

        # macOS native title bar tweaks disabled for stability.

        # Display cache initialization message
        safe_print("  [RAWImageViewer] Getting cache stats...", flush=True)
        try:
            cache_stats = self.image_cache.get_cache_stats()
            memory_info = cache_stats["memory_info"]
            safe_print("✓ Enhanced image cache initialized", flush=True)
            safe_print(f"  Cache budget: {cache_stats['cache_budget_mb']}MB", flush=True)
            safe_print(
                f"  Max full images: {cache_stats['full_image_cache']['max_size']}", flush=True)
            safe_print(
                f"  Max thumbnails: {cache_stats['thumbnail_cache']['max_size']}", flush=True)
            safe_print(
                f"  Available memory: {memory_info['system_available_gb']:.1f}GB", flush=True)
        except Exception as exc:
            safe_print(
                f"  [RAWImageViewer] Cache stats unavailable ({exc}); continuing startup",
                flush=True,
            )
        QTimer.singleShot(1000, self._cleanup_old_image_cache)

        # Restore last folder / file / view mode (opt-out: RAWVIEWER_DISABLE_SESSION_RESTORE=1)
        if os.environ.get("RAWVIEWER_DISABLE_SESSION_RESTORE", "0").strip() != "1":
            safe_print("  [RAWImageViewer] Restoring session state...", flush=True)
            if self.restore_session_state() and hasattr(self, "view_mode") and self.view_mode == "gallery":
                self._show_gallery_view()
        else:
            safe_print("  [RAWImageViewer] Session restore skipped (RAWVIEWER_DISABLE_SESSION_RESTORE)", flush=True)
        if os.environ.get("RAWVIEWER_UPDATE_CHECK_DELAY_MS"):
            update_delay_ms = _env_int("RAWVIEWER_UPDATE_CHECK_DELAY_MS", 12000, minimum=0)
        elif os.environ.get("RAWVIEWER_MOCK_UPDATE_VERSION", "").strip():
            update_delay_ms = 1500
        else:
            update_delay_ms = 12000
        QTimer.singleShot(update_delay_ms, self._check_for_updates_on_launch)
        safe_print("  [RAWImageViewer] Initialization complete!", flush=True)

    def _cleanup_old_image_cache(self):
        """Run persistent cache cleanup once after startup without delaying window creation."""
        try:
            if hasattr(self, 'image_cache') and self.image_cache:
                self.image_cache.cleanup_old_cache()
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(
                "Old image cache cleanup failed: %s", e, exc_info=True
            )
        self._enforce_disk_cache_budget_background()

    def _schedule_deferred_filmstrip_flush(self) -> None:
        """Spread the post-first-render filmstrip thumbnail flush over time.

        v2.5.0 dumped every deferred filmstrip path at once the instant the first image
        painted -- exactly when the user navigates to the second image -- flooding the worker
        pool and main thread. We now wait a short grace period and release in small chunks,
        backing off whenever the user is actively navigating.
        """
        if not (hasattr(self, "_deferred_filmstrip_paths") and self._deferred_filmstrip_paths):
            return
        grace_ms = _env_int("RAWVIEWER_FILMSTRIP_FLUSH_GRACE_MS", 1200, minimum=0)
        QTimer.singleShot(grace_ms, self._flush_deferred_filmstrip_chunked)

    def showEvent(self, event):
        super().showEvent(event)
        if platform.system() == "Darwin":
            self._apply_macos_titlebar_appearance()
        elif platform.system() == "Windows":
            # HWND is valid after show; frameless taskbar icon needs WM_SETICON.
            _apply_windows_native_window_icon(self)
        QTimer.singleShot(0, self._restore_keyboard_focus)

    def _schedule_deferred_sensor_full_after_first_paint(self, file_path: str) -> None:
        """Queue sensor full decode after preview TTFR (avoids RAW worker slot contention)."""
        from PyQt6.QtCore import QTimer

        if not file_path:
            return
        np = _norm_path(file_path)
        self._defer_sensor_full_decode_path = file_path
        if getattr(self, "_sensor_full_decode_queued_norm", None) != np:
            self._sensor_full_decode_queued_norm = None
        delay_ms = _env_int("RAWVIEWER_DEFER_FULL_FALLBACK_MS", 6000, minimum=2000)

        def _fallback(fp=file_path, norm=np) -> None:
            if _norm_path(getattr(self, "current_file_path", "") or "") != norm:
                return
            if getattr(self, "_single_view_first_render_logged", False):
                return
            import logging

            logging.getLogger(__name__).warning(
                "[LOAD] Preview first paint timed out -- queueing full decode for %s",
                os.path.basename(fp),
            )
            self._flush_deferred_sensor_full_decode(fp)

        QTimer.singleShot(delay_ms, _fallback)

    def _schedule_deferred_full_exif_after_first_paint(self, file_path: str) -> None:
        """Queue full exiftool EXIF after preview TTFR (avoids disk-cache EXIF before first paint)."""
        from PyQt6.QtCore import QTimer

        if not file_path:
            return
        np = _norm_path(file_path)
        self._defer_full_exif_path = file_path
        if getattr(self, "_full_exif_queued_norm", None) != np:
            self._full_exif_queued_norm = None
        delay_ms = _env_int("RAWVIEWER_DEFER_FULL_EXIF_FALLBACK_MS", 8000, minimum=3000)

        def _fallback(fp=file_path, norm=np) -> None:
            if _norm_path(getattr(self, "current_file_path", "") or "") != norm:
                return
            if getattr(self, "_single_view_first_render_logged", False):
                return
            import logging

            logging.getLogger(__name__).warning(
                "[LOAD] Preview first paint timed out -- queueing full EXIF for %s",
                os.path.basename(fp),
            )
            self._flush_deferred_full_exif_after_first_paint(fp)

        QTimer.singleShot(delay_ms, _fallback)

    def _cleanup_current_processing(self):
        """Clean up current processing threads and resources"""
        import logging
        import time
        import traceback
        logger = logging.getLogger(__name__)
        cleanup_start = time.time()

        # Prevent multiple cleanup operations from running simultaneously
        with self._cleanup_lock:
            if self._cleanup_in_progress:
                logger.debug("[CLEANUP] Cleanup already in progress, skipping duplicate call")
                return

            self._cleanup_in_progress = True
            logger.info(f"[CLEANUP] ========== _cleanup_current_processing() STARTED at {cleanup_start:.3f} ==========")

        try:
            self._cancel_display_pixmap_converter()

            # Stop any current processing
            if self.current_processor is not None:
                processor_type = type(self.current_processor).__name__
                is_running = self.current_processor.isRunning() if hasattr(self.current_processor, 'isRunning') else False
                logger.debug(f"Cleaning up processor: {processor_type}, is_running: {is_running}, file: {getattr(self.current_processor, 'file_path', 'unknown')}")

                try:
                    # Disconnect signals first to prevent access violations
                    logger.info("[CLEANUP] Disconnecting processor signals")
                    try:
                        if hasattr(self.current_processor, 'image_processed'):
                            self.current_processor.image_processed.disconnect()
                        if hasattr(self.current_processor, 'error_occurred'):
                            self.current_processor.error_occurred.disconnect()
                        if hasattr(self.current_processor, 'thumbnail_fallback_used'):
                            self.current_processor.thumbnail_fallback_used.disconnect()
                        if hasattr(self.current_processor, 'processing_progress'):
                            self.current_processor.processing_progress.disconnect()
                        if hasattr(self.current_processor, 'exif_data_ready'):
                            self.current_processor.exif_data_ready.disconnect()
                        logger.info("[CLEANUP] Processor signals disconnected")
                    except Exception as disconnect_error:
                        logger.warning(f"[CLEANUP] Error disconnecting signals (may be normal if already disconnected): {disconnect_error}")

                    # RAWProcessor uses cleanup(), EnhancedRAWProcessor uses stop_processing() and wait()
                    if hasattr(self.current_processor, 'cleanup'):
                        logger.info(f"[CLEANUP] Calling cleanup() on {processor_type}")
                        self.current_processor.cleanup()
                        logger.info(f"[CLEANUP] cleanup() completed for {processor_type}")
                    else:
                        # For EnhancedRAWProcessor, use stop_processing and wait
                        logger.info(f"[CLEANUP] Using stop_processing() for {processor_type}")
                        if hasattr(self.current_processor, 'stop_processing'):
                            self.current_processor.stop_processing()
                            logger.info(f"[CLEANUP] stop_processing() called for {processor_type}")

                        if hasattr(self.current_processor, 'isRunning'):
                            if self.current_processor.isRunning():
                                logger.info("[CLEANUP] Processor still running, calling quit() and wait()")
                                self.current_processor.quit()
                                wait_result = self.current_processor.wait(100)  # Wait up to 100ms
                                logger.info(f"[CLEANUP] wait() returned: {wait_result}, is_running: {self.current_processor.isRunning()}")
                                if not wait_result:
                                    logger.info("[CLEANUP] Processor did not stop gracefully, calling terminate()")
                                    self.current_processor.terminate()
                                    terminate_wait = self.current_processor.wait(50)  # Wait up to 50ms after terminate
                                    logger.info(f"[CLEANUP] After terminate(), wait() returned: {terminate_wait}, is_running: {self.current_processor.isRunning()}")
                            else:
                                logger.info("[CLEANUP] Processor not running, skip quit/wait")

                    # Clear processor reference after cleanup
                    self.current_processor = None
                    logger.info("[CLEANUP] Processor reference cleared")
                except Exception as cleanup_error:
                    logger.error(f"Error during processor cleanup: {cleanup_error}", exc_info=True)
                    logger.debug(f"Cleanup error traceback: {traceback.format_exc()}")
                    # Try to clear reference even if cleanup failed
                    with contextlib.suppress(Exception):
                        self.current_processor = None
            else:
                logger.debug("No current_processor to clean up")

            # Cancel preload threads (non-blocking)
            if hasattr(self, 'preload_manager'):
                logger.debug("Cancelling preload threads (non-blocking)")
                try:
                    self.preload_manager.cancel_all_preloads()
                    logger.debug("Preload threads cancelled")
                except Exception as preload_error:
                    logger.error(f"Error cancelling preload threads: {preload_error}", exc_info=True)
            else:
                logger.debug("No preload_manager available")

            cleanup_end = time.time()
            logger.info(f"[CLEANUP] _cleanup_current_processing completed successfully in {cleanup_end - cleanup_start:.3f}s")
        except Exception as e:
            cleanup_end = time.time()
            logger.error(f"[CLEANUP] ========== CRITICAL ERROR in _cleanup_current_processing "
                        f"(at {cleanup_end:.3f}, duration: {cleanup_end - cleanup_start:.3f}s) ==========")
            logger.error(f"[CLEANUP] Exception type: {type(e).__name__}, message: {e}", exc_info=True)
            logger.error(f"[CLEANUP] Full traceback:\n{traceback.format_exc()}")
            raise
        finally:
            # Always reset cleanup flag, even if an error occurred
            with self._cleanup_lock:
                self._cleanup_in_progress = False
                logger.debug("[CLEANUP] Cleanup flag reset, cleanup_in_progress=False")

    def _persist_editor_aligned_browse_caches(self, path: str) -> None:
        """Downscale the just-rendered Adjust frame into thumbnail/preview cache.

        Performance: uses the already-computed ``_adjust_last_render`` uint8
        buffer -- only a resize + JPEG encode (~5–30ms on typical preview
        sizes), not a full edit pipeline pass and not part of Export.
        Runs synchronously so gallery refresh after save sees the edited
        pixels immediately. Companion JPEGs are never written (RAW-only).
        Keeps ``SIDECAR_ADJUST`` / ``EDITED_PREVIEWS`` off by default.
        """
        from common_image_loader import is_raw_file
        from raw_adjustments import edited_previews_enabled

        if not path or not is_raw_file(path):
            return
        # When browse tiers re-apply sidecars themselves, baking edited pixels
        # here would double-apply on next delivery.
        if edited_previews_enabled():
            return
        last = getattr(self, "_adjust_last_render", None)
        if not last or not isinstance(last, tuple) or len(last) < 1:
            return
        array = last[0]
        if array is None or getattr(array, "ndim", 0) != 3 or array.dtype != np.uint8:
            return
        if _norm_path(getattr(self, "_adjust_last_render_norm", "") or "") != _norm_path(path):
            return
        try:
            import cv2

            from image_cache import (
                disk_preview_max_edge,
                get_image_cache,
                memory_preview_max_edge,
            )

            rgb = np.ascontiguousarray(array)
            cache = get_image_cache()
            h, w = rgb.shape[:2]
            thumb_cap = disk_preview_max_edge()
            if max(h, w) > thumb_cap:
                sc = thumb_cap / float(max(h, w))
                thumb = cv2.resize(
                    rgb,
                    (max(1, int(w * sc)), max(1, int(h * sc))),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                thumb = rgb
            ok, buf = cv2.imencode(
                ".jpg",
                cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 90],
            )
            jpeg = buf.tobytes() if ok else None
            cache.put_thumbnail(path, thumb, jpeg)
            # Gallery prefers the grid tier; put_grid rejects edges > 512.
            grid_cap = 512
            if max(thumb.shape[0], thumb.shape[1]) > grid_cap:
                sc = grid_cap / float(max(thumb.shape[0], thumb.shape[1]))
                grid = cv2.resize(
                    thumb,
                    (
                        max(1, int(thumb.shape[1] * sc)),
                        max(1, int(thumb.shape[0] * sc)),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
                okg, bufg = cv2.imencode(
                    ".jpg",
                    cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90],
                )
                cache.put_grid(path, grid, bufg.tobytes() if okg else None)
            else:
                cache.put_grid(path, thumb, jpeg)

            prev_cap = memory_preview_max_edge()
            if max(h, w) > prev_cap:
                sc = prev_cap / float(max(h, w))
                prev = cv2.resize(
                    rgb,
                    (max(1, int(w * sc)), max(1, int(h * sc))),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                prev = rgb
            disk_cap = disk_preview_max_edge()
            if max(prev.shape[0], prev.shape[1]) > disk_cap:
                sc = disk_cap / float(max(prev.shape[0], prev.shape[1]))
                disk_prev = cv2.resize(
                    prev,
                    (
                        max(1, int(prev.shape[1] * sc)),
                        max(1, int(prev.shape[0] * sc)),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                disk_prev = prev
            ok2, buf2 = cv2.imencode(
                ".jpg",
                cv2.cvtColor(disk_prev, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 92],
            )
            cache.put_preview(
                path,
                prev,
                jpeg_data=buf2.tobytes() if ok2 else None,
                libraw_rendered=True,
            )
        except Exception:
            logging.getLogger(__name__).debug(
                "editor-aligned browse cache bake failed", exc_info=True
            )

    def _macos_cleanup_share_temp(self) -> None:
        temps = getattr(self, "_macos_share_temp_paths", None)
        if temps:
            for temp in temps:
                try:
                    if os.path.exists(temp):
                        os.remove(temp)
                        _share_log(logging.DEBUG, "cleaned up share temp file: %s", temp)
                except OSError as exc:
                    _share_log(logging.WARNING, "failed to clean up share temp file %s: %s", temp, exc)
            self._macos_share_temp_paths = []

        temp = getattr(self, "_macos_share_temp_path", None)
        if temp:
            try:
                if os.path.exists(temp):
                    os.remove(temp)
                    _share_log(logging.DEBUG, "cleaned up share temp file: %s", temp)
            except OSError as exc:
                _share_log(logging.WARNING, "failed to clean up share temp file %s: %s", temp, exc)
            self._macos_share_temp_path = None

    def _defer_until_first_paint(
        self,
        callback,
        *,
        fallback_ms: int | None = None,
        poll_ms: int = 50,
        label: str = "",
    ) -> None:
        """Run *callback* after single-view TTFR, or after *fallback_ms* if paint is never logged.

        Always schedules via QTimer (never runs *callback* synchronously). Session
        restore can paint via processEvents() before app.exec(); a sync callback
        there (especially torch/kornia import) would block the event loop from
        starting and freeze every other post-TTFR timer for several seconds.
        """
        if fallback_ms is None:
            fallback_ms = _env_int("RAWVIEWER_DEFER_UNTIL_PAINT_MS", 2500, minimum=0)
        poll_ms = max(10, int(poll_ms))
        deadline = time.monotonic() + (fallback_ms / 1000.0)

        def _tick() -> None:
            if getattr(self, "_single_view_first_render_logged", False):
                callback()
                return
            if time.monotonic() >= deadline:
                callback()
                return
            QTimer.singleShot(poll_ms, _tick)

        # Even when TTFR already logged, queue for the next event-loop turn so
        # callers that run during pre-exec processEvents cannot stall app.exec().
        QTimer.singleShot(0 if getattr(self, "_single_view_first_render_logged", False) else poll_ms, _tick)

    def _schedule_deferred_gpu_backend_import(self, *, reason: str = "fallback") -> None:
        """Queue torch/kornia import after quiet period (main thread only).

        OpenMP forces the import onto the GUI thread; we only choose *when*
        so it cannot land on top of session-restore full decode or mid-nav.
        """
        try:
            from bootstrap_pkg import torch as torch_bootstrap

            if torch_bootstrap.gpu_backend_import_started():
                return
        except Exception:
            return
        # Prefer the post-restore schedule: after-paint is only a fallback when
        # session-restore staging never runs (no folder / empty restore).
        restore_busy = (
            getattr(self, "_session_restore_defer_preload", False)
            or self._should_defer_full_decode()
            or getattr(self, "_session_restore_staged_scheduled", False)
        )
        if reason == "after_first_paint" and restore_busy:
            # Safety net if nav-preload never fires (gallery stall, etc.).
            QTimer.singleShot(
                _env_int("RAWVIEWER_GPU_IMPORT_DEFER_FALLBACK_MS", 8000, minimum=0),
                lambda: self._schedule_deferred_gpu_backend_import(reason="fallback"),
            )
            return
        if reason == "post_session_restore":
            delay_ms = _env_int(
                "RAWVIEWER_GPU_IMPORT_AFTER_RESTORE_MS", 1500, minimum=0
            )
        else:
            delay_ms = _env_int(
                "RAWVIEWER_GPU_IMPORT_AFTER_PAINT_MS", 4000, minimum=0
            )
        import logging

        logging.getLogger(__name__).info(
            "[GPU] Scheduling backend import in %dms (%s)", delay_ms, reason
        )
        timer = getattr(self, "_gpu_backend_import_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._try_import_gpu_backend)
            self._gpu_backend_import_timer = timer
        timer.stop()
        timer.start(delay_ms)

    def _restore_keyboard_focus(self) -> None:
        """Return keyboard focus to the active view surface (gallery or single image)."""
        self.activateWindow()
        for view in (
            getattr(self, "compare_select_view", None),
            getattr(self, "compare_candidate_view", None),
        ):
            if view is None:
                continue
            if not view.isVisible() and view.hasFocus():
                view.clearFocus()
        vm = getattr(self, "view_mode", "single")
        if vm == "compare":
            gv = getattr(self, "compare_select_view", None)
            if gv is not None and gv.isVisible():
                gv.setFocus(Qt.FocusReason.OtherFocusReason)
                return
            self.setFocus(Qt.FocusReason.OtherFocusReason)
            return
        if vm == "gallery":
            gs = getattr(self, "gallery_scroll", None)
            if gs is not None and gs.isVisible():
                gs.setFocus(Qt.FocusReason.OtherFocusReason)
                gs.viewport().setFocus(Qt.FocusReason.OtherFocusReason)
                return
            self.setFocus(Qt.FocusReason.OtherFocusReason)
            return
        gv = getattr(self, "gpu_view", None)
        if gv is not None and gv.isVisible():
            gv.setFocus(Qt.FocusReason.OtherFocusReason)
            return
        self.setFocus(Qt.FocusReason.OtherFocusReason)
