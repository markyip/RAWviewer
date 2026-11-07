"""image_load mixin, part 2 of 3 (3 methods, 883 lines)."""
from __future__ import annotations

import contextlib
import os
import platform
import sys

from PyQt6.QtCore import Qt

from core.config import _env_int, _norm_path, safe_print
from rawviewer_app.processing import RAWProcessor

from .. import lazy
from ..lazy import (
    Priority,
    image_covers_sensor_resolution,
    is_raw_file,
    use_libraw_consistent_preview_first,
    use_progressive_raw_loading,
)

lazy.register(sys.modules[__name__])


class ImageLoadPart2:
    def load_raw_image(self, file_path, force_reload=False):
        import logging
        import time
        import traceback
        logger = logging.getLogger(__name__)

        load_start = time.time()
        logger.info(f"[LOAD] ========== load_raw_image() STARTED at {load_start:.3f} ==========")

        # Re-bake the leaving image's persisted preview only when browse
        # surfaces actually apply sidecar edits. load_adjustments_for_file()
        # always hits read_as_shot_temperature() (exifread/rawpy + global
        # lock) -- evaluating it as an argument blocked the UI thread for
        # ~17s on first nav while folder sort was probing thousands of
        # files on a USB volume (see [NAVTIME] 20s vs later ~0.5s).
        old_path = getattr(self, "current_file_path", None)
        if old_path and file_path != old_path:
            try:
                from raw_adjustments import sidecar_adjustments_enabled

                if sidecar_adjustments_enabled():
                    self._reencode_persisted_preview_for_sidecar(old_path, None)
            except Exception as e:
                logger.debug(f"[LOAD] Failed to re-encode preview for old path: {e}")
        safe_print(
            f"[LOAD] START {os.path.basename(file_path) if file_path else '<none>'}",
            flush=True,
        )
        logger.info(f"[LOAD] File path: {file_path}")
        logger.info(f"[LOAD] Previous file: {getattr(self, 'current_file_path', 'None')}")
        logger.info(f"[LOAD] Navigation state - in_progress: {getattr(self, '_navigation_in_progress', False)}")

        # Pause background gallery idle preloading when navigating to a new
        # image, then re-arm it with a settle delay: thumbnail warming keeps
        # running whenever the user idles in single view (instant later
        # gallery switch) but stays quiet during navigation bursts -- each
        # navigation restarts the single-shot timer, so it only fires after
        # ~4s without further navigation.
        try:
            if hasattr(self, 'gallery_view') and hasattr(self.gallery_view, 'pause_idle_preload'):
                self.gallery_view.pause_idle_preload()
                idle_timer = getattr(self.gallery_view, '_idle_preload_timer', None)
                if idle_timer is not None:
                    idle_timer.start(
                        _env_int("RAWVIEWER_SINGLE_VIEW_THUMB_PRELOAD_MS", 4000, minimum=1000)
                    )
        except Exception as e:
            logger.debug(f"[LOAD] Failed to pause gallery idle preload: {e}")

        # Stop the metadata-index defer timer so _SemanticIndexPrepWorker
        # is not triggered during navigation.  It will be restarted by
        # display_pixmap() after the new image is painted.
        try:
            t = getattr(self, "_metadata_index_defer_timer", None)
            if t is not None and t.isActive():
                t.stop()
                logger.debug("[LOAD] Stopped _metadata_index_defer_timer for navigation")
        except Exception as e:
            logger.debug(f"[LOAD] Failed to stop metadata index timer: {e}")

        try:
            self._single_view_load_request_ts = float(load_start)
            self._single_view_load_request_path = str(file_path or "")
            self._single_view_first_render_logged = False
            self._defer_sensor_full_decode_path = None
            self._sensor_full_decode_queued_norm = None
            self._defer_full_exif_path = None
            self._full_exif_queued_norm = None
            self._last_manager_exif_was_minimal = False
            # Restart the idle full-res timer for the new file.
            try:
                it = getattr(self, "_full_res_idle_timer", None)
                if it is not None:
                    it.stop()
                self._full_res_idle_path = None
            except Exception:
                pass
            self._adjust_preview_base_rgb = None
            self._adjust_preview_base_rgb_fast = None
            self._adjust_edit_base_path = None
            self._adjust_edit_base_loading_norm = None
            self._adjust_panel_loaded_norm = None
        except Exception:
            pass

        try:
            # Check if file exists
            if not os.path.exists(file_path):
                error_msg = f"The file {file_path} does not exist."
                logger.error(f"[LOAD] File not found: {file_path}")
                self.show_error("File not found", error_msg)
                if hasattr(self, "loading_overlay"):
                    self.loading_overlay.hide_loading()
                return

            # Stop any active animations
            self._stop_animations()

            # Handle animated images (GIF/WebP)
            if self.is_animated_image(file_path):
                self.current_file_path = file_path
                self._last_loaded_path = file_path
                if getattr(self, "view_mode", "single") == "single":
                    self._maybe_leave_burst_group_view_for_path(file_path)
                if self.image_files and file_path in self.image_files:
                    self.current_file_index = self.image_files.index(file_path)
                self._sync_filmstrip_index()

                started = self._load_and_start_animation(file_path)
                if started:
                    # Load EXIF in background for status HUD
                    if not self.image_cache.get_exif(file_path):
                        self.image_manager.load_image(
                            file_path,
                            priority=Priority.BACKGROUND,
                            cancel_existing=False,
                            use_full_resolution=False,
                            stages={"exif"},
                        )
                    self.update_status_bar()
                    self._start_preloading()
                    return

            # Restore GPU view if enabled for standard static images
            self._sync_gpu_single_view_visibility()

            # Store the requested file path for later comparison (after cleanup)
            # This allows us to detect if file changed during cleanup due to rapid navigation
            requested_file_path = file_path

            if getattr(self, "_loading_from_gallery", False):
                if (
                    getattr(self, "_gallery_instant_display_quality", False)
                    and _norm_path(getattr(self, "_displayed_content_path", ""))
                    == _norm_path(requested_file_path)
                    and self._single_view_pixels_on_screen(requested_file_path)
                ):
                    logger.info(
                        "[LOAD] Gallery→single: keeping on-screen display-quality buffer"
                    )
                    self._finish_gallery_to_single_without_redisplay(requested_file_path)
                    self._on_single_view_content_displayed()
                    return
            else:
                # Non-gallery loads (nav / filmstrip / Adjust) resume RAW JPEG suppression.
                self._gallery_entry_accept_embedded = False

            logger.info("[LOAD] File exists, proceeding with load")
            safe_print(f"[PERF] Loading image: {os.path.basename(requested_file_path)}")

            if (
                not force_reload
                and getattr(self, "view_mode", "single") == "single"
                and not getattr(self, "_navigation_in_progress", False)
                and not getattr(self, "_loading_from_gallery", False)
                and not getattr(self, "_loading_from_filmstrip", False)
                and not getattr(self, "_preserve_nav_zoom_active", False)
                and _norm_path(requested_file_path)
                == _norm_path(getattr(self, "current_file_path", "") or "")
                and getattr(self, "_single_view_first_render_logged", False)
                and self._single_view_pixels_on_screen(requested_file_path)
            ):
                logger.info(
                    "[LOAD] Skipping redundant load_raw_image for %s (already on screen)",
                    os.path.basename(requested_file_path),
                )
                if hasattr(self, "loading_overlay"):
                    self.loading_overlay.hide_loading()
                return

            # Note: Navigation concurrency is controlled by can_navigate() in navigation methods
            # We don't check _navigation_in_progress here because load_raw_image may be called
            # from other places (not just navigation), and the navigation methods already have
            # proper concurrency control

            # Reset flags when loading new image
            # Assume preview-first until a full-resolution buffer is confirmed on-screen, but ONLY for RAW files
            raw_extensions = {'.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2',
                             '.pef', '.srw', '.x3f', '.raf', '.3fr', '.fff', '.iiq',
                             '.cap', '.erf', '.mef', '.mos', '.nrw', '.rwl', '.srf'}
            is_raw = os.path.splitext(requested_file_path)[1].lower() in raw_extensions
            self._is_half_size_displayed = False
            self._full_resolution_loading = False
            self._smooth_zoom_full_request_sent = False
            self._raw_status_exif_refresh_path = None

            # Orientation defaults off for each new file; cache hits set this True again below.
            self._orientation_already_applied = False

            # Pause background indexing during active image load/navigation to eliminate resource contention
            try:
                from PyQt6.QtCore import QTimer
                self._pause_semantic_indexing_deferred()
                if hasattr(self, "_resume_indexing_timer") and self._resume_indexing_timer:
                    self._resume_indexing_timer.stop()

                self._resume_indexing_timer = QTimer(self)
                self._resume_indexing_timer.setSingleShot(True)
                self._resume_indexing_timer.timeout.connect(self._resume_indexing_if_single_view)
                self._resume_indexing_timer.start(5000)
            except Exception:
                pass

            logger.info(f"[LOAD] Flags reset - half_size: {self._is_half_size_displayed}, full_res_loading: {self._full_resolution_loading}, orientation_applied: {self._orientation_already_applied}")

            # Clear pending zoom restore when loading new image (will be set again if needed)
            if hasattr(self, '_pending_zoom_restore'):
                logger.debug("Clearing _pending_zoom_restore")
                delattr(self, '_pending_zoom_restore')
            # Keep deferred zoom when the same file is upgrading to full resolution.
            same_file_reload = _norm_path(requested_file_path) == _norm_path(
                getattr(self, "current_file_path", "") or ""
            )
            keep_pending_point_zoom = bool(
                getattr(self, "_pending_zoom", False) and same_file_reload
            )
            if not keep_pending_point_zoom:
                self._pending_zoom = False
                if hasattr(self, "_pending_zoom_center"):
                    logger.debug("Clearing _pending_zoom_center")
                    delattr(self, "_pending_zoom_center")
                if hasattr(self, "_pending_zoom_thumbnail_size"):
                    with contextlib.suppress(AttributeError):
                        delattr(self, "_pending_zoom_thumbnail_size")
            if hasattr(self, '_pending_zoom_level'):
                logger.debug("Clearing _pending_zoom_level")
                delattr(self, '_pending_zoom_level')
            # Keep user-requested pending zoom for the same file currently loading.
            pending_zoom_target = getattr(self, "_pending_zoom_target_path", None)
            same_pending_target = (
                bool(pending_zoom_target)
                and _norm_path(str(pending_zoom_target)) == _norm_path(str(requested_file_path))
            )
            if hasattr(self, "_pending_zoom_toggle") and not same_pending_target:
                self._pending_zoom_toggle = False
                if hasattr(self, "_pending_zoom_target_path"):
                    with contextlib.suppress(AttributeError):
                        delattr(self, "_pending_zoom_target_path")
            if hasattr(self, "_pending_zoom_focus_subject") and not (
                same_pending_target or keep_pending_point_zoom
            ):
                self._pending_zoom_focus_subject = False
            if hasattr(self, "_pending_gpu_zoom_point"):
                with contextlib.suppress(AttributeError):
                    delattr(self, "_pending_gpu_zoom_point")
            if not getattr(self, "_preserve_nav_zoom_active", False) and hasattr(self, "_maintain_zoom_on_navigation"):
                    with contextlib.suppress(AttributeError):
                        delattr(self, "_maintain_zoom_on_navigation")

            # Clean up current processing (simplified with new architecture)
            cleanup_start = time.time()
            logger.info("[LOAD] Starting cleanup of current processing (if any)")
            # Cancel in-flight loads only when switching files -- same-path reload must not cancel
            # the active task (e.g. duplicate load_raw_image after folder change).
            _prev_fp = getattr(self, "current_file_path", None)
            _same_path_reload = _prev_fp and _norm_path(_prev_fp) == _norm_path(requested_file_path)
            if force_reload:
                _same_path_reload = False
                self._manager_displayed_max_dim = 0
            if _prev_fp and not _same_path_reload:
                self._nav_idle_full_generation = int(
                    getattr(self, "_nav_idle_full_generation", 0)
                ) + 1
                # Drop CURRENT work for the file we left; keep PRELOAD/BACKGROUND prefetch.
                self.image_manager.cancel_current_priority_tasks(_prev_fp)
                self._single_view_display_generation = int(
                    getattr(self, "_single_view_display_generation", 0)
                ) + 1
                self._cancel_content_crossfade()
                # Keep the previous GPU/label pixels until the next pixmap lands (arrow-key nav).
            if _same_path_reload:
                # Same file (e.g. after on-disk rotation): drop "already displayed N px" gate so
                # thumbnail/image handlers don't skip updates and leave the loading overlay stuck.
                self._manager_displayed_max_dim = 0
            # Reset any stale deferred-sensor-decode gate from a previous file
            # that never completed. Without this, _sensor_full_decode_queued_norm
            # stays pinned to a dead path and blocks every future full decode.
            try:
                prev_norm = _norm_path(getattr(self, "_sensor_full_decode_queued_norm", None) or "")
                if prev_norm and prev_norm != _norm_path(requested_file_path):
                    self._sensor_full_decode_queued_norm = None
                    self._defer_sensor_full_decode_path = None
            except Exception:
                pass

            # Heavy RAW demosaics are serialized (_raw_load_limit, often 1). If a
            # neighbor prefetch still holds the slot, the on-screen image's own
            # sensor decode would wait behind it -- the "preview loads but the
            # full image never replaces it" symptom. Preempt non-current heavy
            # RAW work now so the CURRENT decode can claim the slot the moment
            # it is enqueued below.
            try:
                if hasattr(self.image_manager, "cancel_non_current_heavy_raw_except"):
                    preempted = self.image_manager.cancel_non_current_heavy_raw_except(
                        requested_file_path
                    )
                    if preempted:
                        logger.info(
                            "[LOAD] Preempted %d neighbor heavy RAW task(s) for %s",
                            preempted,
                            os.path.basename(requested_file_path),
                        )
            except Exception:
                pass
            # Drop phantom in-flight markers: a task that never finishes keeps
            # has_active_work_for_path() True forever, which blocks every later
            # full-decode request for this file. Leave prefetch in place (age only).
            try:
                gj = getattr(self, "gallery_justified", None)
                if gj is not None and hasattr(gj, "_evict_stale_active_tasks"):
                    gj._evict_stale_active_tasks(max_age_s=6.0, drop_prefetch=False)
            except Exception:
                pass
            # Legacy cleanup for old processor (if still exists)
            if self.current_processor:
                logger.info(f"[LOAD] Legacy processor cleanup: {type(self.current_processor).__name__}")
                self._cleanup_current_processing()
            cleanup_time = time.time() - cleanup_start
            logger.info(f"[LOAD] Cleanup completed in {cleanup_time:.3f}s")

            # Update current_file_path immediately after cleanup
            # This ensures we have the correct value for subsequent operations
            # NOTE: We removed the post-cleanup cancellation check because:
            # 1. can_navigate() / _navigation_in_progress already guard overlapping navigations
            # 2. The check was causing false cancellations during normal navigation
            # 3. current_file_path is always different from requested_file_path at this point
            #    (it's the old file), so the check would always cancel
            self.current_file_path = requested_file_path
            self._last_loaded_path = requested_file_path # Track for view switching optimizations
            if getattr(self, "view_mode", "single") == "single":
                self._maybe_leave_burst_group_view_for_path(requested_file_path)
            if self.image_files and requested_file_path in self.image_files:
                self.current_file_index = self.image_files.index(requested_file_path)
            self._sync_filmstrip_index()
            if self._adjust_panel_active():
                self._refresh_adjust_panel_for_navigation(requested_file_path)
            # Proactive HE/HE* handling: leave RAW mode before JPEG-interim
            # suppression can refuse the only available embed and leave the
            # previous LibRaw frame on screen.
            if (
                getattr(self, "view_mode", "single") == "single"
                and is_raw_file(requested_file_path)
                and self._nef_he_compressed(requested_file_path)
            ):
                self._auto_revert_to_embedded_jpeg_workflow(
                    requested_file_path, reason="proactive HE/HE* detect"
                )
            np_req = _norm_path(requested_file_path)
            if getattr(self, "_manager_display_track_path", None) != np_req:
                self._manager_display_track_path = np_req
                self._preview_upgrade_attempted_paths.discard(np_req)
                self._raw_workflow_refused_tier_paths.discard(np_req)
                # Always reset: this high-water mark is per *on-screen* file.
                # A warm 512px preview for the NEW path used to skip the reset
                # via _display_quality_buffer_cached, leaving the previous
                # file's 6k value in place -- then thumbnail_ready skipped the
                # only delivery as "stale" and the view never updated (HUD
                # changed, pixels stayed on the prior image).
                self._manager_displayed_max_dim = 0

            # Verify cleanup completed
            if self.current_processor is not None:
                logger.warning(f"[LOAD] WARNING: current_processor still exists after cleanup: {type(self.current_processor).__name__}")
                if hasattr(self.current_processor, 'isRunning') and self.current_processor.isRunning():
                    logger.warning("[LOAD] WARNING: current_processor is still running after cleanup!")
            else:
                logger.info("[LOAD] Cleanup verified: current_processor is None")

            # Note: current_file_path is now set above (after cleanup check)
            # to prevent false cancellations during normal navigation
            filename = os.path.basename(requested_file_path)
            app_title = "RAWviewer"
            # Windows: taskbar shows filename; centered EXIF HUD uses _apply_top_metadata_text.
            # macOS: window title is updated from _apply_top_metadata_text (same HUD string).
            if platform.system() != "Darwin":
                self.setWindowTitle(f"{filename} -- {app_title}")
                if hasattr(self, 'title_bar') and self.title_bar is not None:
                    self.title_bar.set_title(app_title)

            # Reset EXIF data ready flag for new image
            self._exif_data_ready = False

            _prev_norm = _norm_path(_prev_fp) if _prev_fp else ""
            _is_single_view_step = (
                getattr(self, "view_mode", "single") == "single"
                and not _same_path_reload
                and bool(_prev_norm)
                and _prev_norm != np_req
            )
            if _is_single_view_step and self._try_paint_navigation_cache_preview(
                requested_file_path
            ):
                logger.info(
                    "[LOAD] Single-view navigation: instant display from prefetch cache"
                )
                if not self.image_cache.get_exif(requested_file_path):
                    self.image_manager.load_image(
                        requested_file_path,
                        priority=Priority.BACKGROUND,
                        cancel_existing=False,
                        use_full_resolution=False,
                        stages={"exif"},
                    )
                if not (
                    use_libraw_consistent_preview_first(requested_file_path)
                    and is_raw_file(requested_file_path)
                ):
                    self._queue_single_view_embedded_preview_load(requested_file_path)
                elif getattr(self, "_nav_cache_paint_tier", None) != "full":
                    # RAW workflow revisit that painted only a persisted
                    # preview tier (512-1536px -- soft on a retina display):
                    # kick the CURRENT-priority LibRaw half decode NOW, the
                    # same request a cache-miss navigation would make.
                    # Without this, the persisted-preview paint (added for
                    # revisit color consistency) downgraded the cleanup from
                    # an immediate CURRENT decode to the 1.2s-deferred
                    # BACKGROUND sensor decode -- reported as "returns to a
                    # blurry image that takes a long time to sharpen". The
                    # manager won't short-circuit: a sub-sensor preview does
                    # not satisfy the libraw-first 'full' stage check.
                    self.image_manager.load_image(
                        requested_file_path,
                        priority=Priority.CURRENT,
                        cancel_existing=False,
                        use_full_resolution=False,
                        stages={"thumbnail", "exif", "full"},
                    )
                self._start_preloading()
                return

            rating = 0
            cached_exif = self.image_cache.get_exif(requested_file_path)
            if cached_exif:
                rating = cached_exif.get('rating', 0) or 0
            self._update_single_view_rating_display(rating)

            # Prefer display-quality preview before full image (faster first paint on warm cache).
            logger.info("[LOAD] Checking for cached preview/full image")
            cache_check_start = time.time()
            cache_check_time = 0.0

            # RAW workflow JPEG-interim suppression: decide BEFORE fetching --
            # get_preview_memory_only() avoids disk JPEG decode (~150ms, main thread)
            # that get_preview() would trigger on a memory miss.
            if self._suppress_jpeg_interim(
                requested_file_path
            ) and not self.image_cache.is_libraw_preview(requested_file_path):
                preview_cached = None
            else:
                preview_cached = self.image_cache.get_preview_memory_only(requested_file_path)
            if (
                not force_reload
                and preview_cached is not None
                and not self._preview_acceptable_for_single_view(
                preview_cached,
                file_path=requested_file_path,
                context="navigation" if _is_single_view_step else "display",
            )
            ):
                px = self._cache_buffer_max_dim(preview_cached)
                logger.info(
                    f"[LOAD] Ignoring low-res preview cache ({px}px) for {filename}; "
                    f"using embedded thumbnail pipeline"
                )
                preview_cached = None
            preserve_zoom_navigation = self._preserve_zoom_navigation_active()
            if (
                not force_reload
                and preview_cached is not None
                and not preserve_zoom_navigation
            ):
                cache_check_time = time.time() - cache_check_start
                logger.info(
                    f"[LOAD] Cache hit: display preview for {filename}, "
                    f"shape: {preview_cached.shape}"
                )
                safe_print(
                    f"[PERF] ✅ CACHE HIT: Preview loaded from cache in {cache_check_time*1000:.1f}ms"
                )
                try:
                    if self._display_cached_numpy_for_path(
                        requested_file_path, preview_cached, from_cache="preview"
                    ):
                        if not self.image_cache.get_exif(requested_file_path):
                            self.image_manager.load_image(
                                requested_file_path,
                                priority=Priority.BACKGROUND,
                                cancel_existing=False,
                                use_full_resolution=False,
                                stages={"exif"},
                            )
                        else:
                            try:
                                exif = self.image_cache.get_exif(requested_file_path)
                                if not image_covers_sensor_resolution(
                                    preview_cached.shape[1],
                                    preview_cached.shape[0],
                                    exif,
                                ):
                                    self._schedule_deferred_sensor_full_after_first_paint(
                                        requested_file_path
                                    )
                            except Exception:
                                pass
                        return
                except Exception as display_error:
                    logger.error(
                        f"[LOAD] Error displaying cached preview: {display_error}",
                        exc_info=True,
                    )
                    if hasattr(self, "loading_overlay"):
                        self.loading_overlay.hide_loading()

            cache_check_time = time.time() - cache_check_start
            cached_image = self.image_cache.get_full_image(requested_file_path)
            if not force_reload and cached_image is not None:
                logger.info(
                    f"[LOAD] Cache hit: full image found for {filename}, "
                    f"shape: {cached_image.shape}"
                )
                safe_print(
                    f"[PERF] ✅ CACHE HIT: Full image loaded from cache in "
                    f"{cache_check_time * 1000:.1f}ms"
                )
                try:
                    if self._display_cached_numpy_for_path(
                        requested_file_path, cached_image, from_cache="full_image"
                    ):
                        if not self.image_cache.get_exif(requested_file_path):
                            self.image_manager.load_image(
                                requested_file_path,
                                priority=Priority.BACKGROUND,
                                cancel_existing=False,
                                use_full_resolution=False,
                                stages={"exif"},
                            )
                        logger.info(
                            f"[LOAD] Successfully displayed cached full image for {filename} "
                            f"(total: {time.time() - load_start:.3f}s)"
                        )
                        return
                except Exception as display_error:
                    logger.error(
                        f"[LOAD] Error displaying cached image: {display_error}",
                        exc_info=True,
                    )
                    if hasattr(self, "loading_overlay"):
                        self.loading_overlay.hide_loading()

            # Check if we have a cached pixmap for non-RAW files ONLY
            # CRITICAL: Only check pixmap cache for non-RAW files to avoid loading JPEG when RAW is requested
            raw_extensions = {'.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2',
                             '.pef', '.srw', '.x3f', '.raf', '.3fr', '.fff', '.iiq',
                             '.cap', '.erf', '.mef', '.mos', '.nrw', '.rwl', '.srf'}
            file_ext = os.path.splitext(file_path)[1].lower()
            is_raw_ext = file_ext in raw_extensions

            if not is_raw_ext:
                # Only check pixmap cache for non-RAW files (JPEG, PNG, etc.)
                logger.info("[LOAD] Checking for cached pixmap (non-RAW file)")
                cache_check_start = time.time()
                cached_pixmap = self.image_cache.get_pixmap(requested_file_path)
                cache_check_time = time.time() - cache_check_start
                if cached_pixmap is not None:
                    logger.info(f"[LOAD] Cache hit: pixmap found for {filename}, size: {cached_pixmap.width()}x{cached_pixmap.height()}")
                    safe_print(f"[PERF] ✅ CACHE HIT: Pixmap loaded from cache in {cache_check_time*1000:.1f}ms")
                    self.status_bar.showMessage(f"Loaded {filename} from cache")
                    try:
                        # Cached pixmaps from load_pixmap_safe / load_raw_preview_pixmap already have orientation applied
                        # Set flag so display_pixmap doesn't apply it again
                        self._orientation_already_applied = True

                        logger.info("[LOAD] Displaying cached pixmap")
                        display_start = time.time()
                        self.display_pixmap(cached_pixmap)
                        self._on_single_view_content_displayed()
                        display_time = time.time() - display_start
                        logger.info(f"[LOAD] Cached pixmap displayed in {display_time:.3f}s")
                        self.setFocus()
                        self.schedule_save_session_state()
                        # Update index for preloading
                        try:
                            if self.image_files and requested_file_path in self.image_files:
                                self.current_file_index = self.image_files.index(requested_file_path)
                        except ValueError:
                            pass
                        self._start_preloading()

                        # Ensure EXIF is loaded even on cache hit
                        if not self.image_cache.get_exif(requested_file_path):
                            logger.info("[LOAD] Cache hit for pixmap, but missing EXIF. Requesting EXIF load.")
                            self.image_manager.load_image(
                                requested_file_path,
                                priority=Priority.BACKGROUND,
                                cancel_existing=False,
                                use_full_resolution=False,
                                stages={"exif"},
                            )

                        logger.info(f"[LOAD] Successfully displayed cached pixmap for {filename} (total: {time.time() - load_start:.3f}s)")
                        if hasattr(self, "loading_overlay"):
                            self.loading_overlay.hide_loading()
                        return
                    except Exception as display_error:
                        logger.error(f"[LOAD] Error displaying cached pixmap: {display_error}", exc_info=True)
                        logger.error(f"[LOAD] Display error traceback:\n{traceback.format_exc()}")
                        if hasattr(self, "loading_overlay"):
                            self.loading_overlay.hide_loading()
                        # Continue to process if display fails
            else:
                logger.info(f"[LOAD] Skipping pixmap cache check for RAW file: {filename}")

            # No cache hit, use new unified image load manager
            cache_miss_time = time.time() - load_start
            logger.info(f"[LOAD] No cache hit, starting unified image load manager (elapsed: {cache_miss_time:.3f}s)")
            safe_print(f"[PERF] ❌ CACHE MISS: Starting processing (cache check took {cache_miss_time*1000:.1f}ms)")
            self.status_bar.showMessage(f"Loading {filename}...")
            # Set loading message with proper alignment (centered both vertically and horizontally)
            # Ensure label fills the viewport for proper centering
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
            # Full-screen overlay: skip when arrow-navigating while zoomed in -- keeps prior image
            # visible until the next one is ready (avoids flashing "Loading Image..." every step).
            # Also skip for prev/next keyboard navigation and when opening from gallery: keep prior
            # pixels visible until the new decode lands instead of a blocking popup.
            if hasattr(self, "loading_overlay") and not self._should_skip_loading_overlay():
                self.loading_overlay.show_loading("Loading Image...")

            # Use new unified image load manager (non-blocking, thread pool based)
            manager_start = time.time()
            logger.info(f"[LOAD] Requesting image load via ImageLoadManager for: {requested_file_path}")
            try:
                # When navigating zoomed-in, decode full-resolution first and skip the thumbnail stage so we
                # never show a sharpened-zoom on a soft preview followed by another zoom swap.
                # DNG single-view path is also forced to full-resolution-first to keep
                # zoom logic deterministic (no preview/full handoff race on first zoom).
                preserve_zoom_navigation = self._preserve_zoom_navigation_active()
                workflow_toggle = getattr(self, "_workflow_toggle_reload", False)
                nav_single_step = (
                    getattr(self, "view_mode", "single") == "single"
                    and (
                        getattr(self, "_navigation_in_progress", False)
                        or getattr(self, "_loading_from_filmstrip", False)
                    )
                )
                nav_fit_browse = nav_single_step and not preserve_zoom_navigation
                is_dng_file = os.path.splitext(requested_file_path)[1].lower() == ".dng"
                from common_image_loader import dng_prefers_embedded_preview_first

                force_full_res_for_dng = (
                    is_dng_file
                    and getattr(self, "view_mode", "single") == "single"
                    and not dng_prefers_embedded_preview_first(requested_file_path)
                )
                from_gallery = getattr(self, "_loading_from_gallery", False)
                # Single-step navigation: only request full-res decode if navigating zoomed-in (preserve_zoom_navigation is active)
                # to prevent slow demosaicing lag when simply browsing fit-to-window previews.
                request_full_res = (
                    preserve_zoom_navigation
                    or force_full_res_for_dng
                    or (nav_single_step and preserve_zoom_navigation)
                )
                # Allow embedded JPEG first frame (libraw_fit = False) if:
                # 1. Progressive RAW load is explicitly set
                # 2. A fast-open token is active (cold opens should paint in milliseconds)
                # 3. Gallery → single (always embedded-first to cut demosaic pressure,
                #    even when RAW HQ or Adjust previously forced LibRaw)
                libraw_fit = (
                    (not from_gallery)
                    and use_libraw_consistent_preview_first(requested_file_path)
                    and is_raw_file(requested_file_path)
                    and not preserve_zoom_navigation
                    and not force_full_res_for_dng
                    and getattr(self, "_folder_fast_open_token", None) is None
                    and not use_progressive_raw_loading()
                    and (not nav_fit_browse or workflow_toggle)
                )
                libraw_nav_fit = (
                    (not from_gallery)
                    and use_libraw_consistent_preview_first(requested_file_path)
                    and is_raw
                    and nav_fit_browse
                    and not preserve_zoom_navigation
                    and not force_full_res_for_dng
                )
                quality_cached = self._display_quality_buffer_cached(requested_file_path)
                quality_on_screen = (
                    quality_cached
                    and self._single_view_pixels_on_screen(requested_file_path)
                )
                load_stages = (
                    {"exif", "full"}
                    if (
                        not is_raw
                        or preserve_zoom_navigation
                        or libraw_fit
                        or libraw_nav_fit
                        or force_full_res_for_dng
                        or (nav_single_step and preserve_zoom_navigation)
                        or quality_on_screen
                    )
                    else None
                )
                # Single-view RAW: preview + EXIF first; full decode in a separate task.
                # Gallery entry always takes this path for RAW (never LibRaw-first).
                preview_first_stages = (
                    is_raw
                    and getattr(self, "view_mode", "single") == "single"
                    and load_stages is None
                    and not request_full_res
                    and not preserve_zoom_navigation
                    and not libraw_fit
                    and not libraw_nav_fit
                    and not force_full_res_for_dng
                    and not (nav_single_step and preserve_zoom_navigation)
                    and not workflow_toggle
                ) or (
                    from_gallery
                    and is_raw
                    and not preserve_zoom_navigation
                    and not force_full_res_for_dng
                )
                if libraw_nav_fit and cached_exif:
                    load_stages = {"full"}
                    request_full_res = False
                if from_gallery and is_raw and not preserve_zoom_navigation:
                    # Non-raw first paint from gallery -- thumbnail/exif only.
                    load_stages = {"thumbnail", "exif"}
                    request_full_res = False
                    preview_first_stages = True
                if preview_first_stages and load_stages is None:
                    # Fast-open / cold single-view: paint embedded preview first;
                    # full exiftool metadata runs in the BACKGROUND exif task below.
                    load_stages = {"thumbnail"}
                    request_full_res = False

                logger.info(
                    f"[LOAD] ImageLoadManager -- use_full_resolution={request_full_res}, "
                    f"stages={load_stages or 'default'}, preserve_nav_zoom={preserve_zoom_navigation}, "
                    f"libraw_consistent_fit={libraw_fit}, force_dng_full_res={force_full_res_for_dng}, "
                    f"preview_first_stages={preview_first_stages}, from_gallery={from_gallery}"
                )

                if self._loading_from_gallery:
                    # Clear the flag after using it
                    self._loading_from_gallery = False
                if getattr(self, "_loading_from_filmstrip", False):
                    self._loading_from_filmstrip = False

                # Request image load with highest priority
                self.image_manager.load_image(
                    file_path=requested_file_path,
                    priority=Priority.CURRENT,
                    cancel_existing=True,
                    use_full_resolution=request_full_res,
                    stages=load_stages,
                    bypass_cache=bool(
                        force_reload or getattr(self, "_workflow_toggle_reload", False)
                    ),
                )
                if preview_first_stages:
                    # Gallery entry: idle full only -- do not kick sensor/LibRaw
                    # demosaic in the same breath as the first embedded paint.
                    if from_gallery:
                        self._schedule_idle_full_decode_after_nav_preview(
                            requested_file_path
                        )
                        self._schedule_deferred_full_exif_after_first_paint(
                            requested_file_path
                        )
                    else:
                        self._schedule_deferred_sensor_full_after_first_paint(
                            requested_file_path
                        )
                        self._schedule_deferred_full_exif_after_first_paint(
                            requested_file_path
                        )
                logger.info("[LOAD] Image load requested via ImageLoadManager")
            except Exception as manager_error:
                logger.error(f"[LOAD] Failed to request image load: {manager_error}", exc_info=True)
                logger.error(f"[LOAD] Manager error traceback:\n{traceback.format_exc()}")
                if hasattr(self, "loading_overlay"):
                    self.loading_overlay.hide_loading()
                # Fallback to legacy processor if manager fails
                logger.warning("[LOAD] Falling back to legacy RAWProcessor")
                try:
                    file_ext = os.path.splitext(requested_file_path)[1].lower()
                    is_raw = is_raw_file(requested_file_path)
                    fallback_full_res = is_raw and (file_ext == ".dng")
                    self.current_processor = RAWProcessor(
                        requested_file_path,
                        is_raw=is_raw,
                        use_full_resolution=fallback_full_res
                    )
                    self.current_processor.image_processed.connect(self.on_image_processed)
                    self.current_processor.error_occurred.connect(self.on_processing_error)
                    self.current_processor.thumbnail_fallback_used.connect(self.on_thumbnail_fallback)
                    self.current_processor.processing_progress.connect(self.on_processing_progress)
                    self.current_processor.exif_data_ready.connect(self.on_exif_data_ready)
                    self.current_processor.start()
                except Exception as fallback_error:
                    logger.error(f"[LOAD] Fallback processor also failed: {fallback_error}", exc_info=True)
                    raise

            manager_request_time = time.time() - manager_start
            logger.info(f"[LOAD] Image load requested in {manager_request_time:.3f}s")
            safe_print(f"[PERF] 📊 SETUP TIME: {manager_request_time*1000:.1f}ms (cleanup: {cleanup_time*1000:.1f}ms)")

            self.setFocus()
            # Save session state when image changes
            self.schedule_save_session_state()

            # Update index for preloading (but don't start yet - wait for image to display)
            # This matches RAWviewer-1.0 behavior and reduces resource competition
            try:
                if self.image_files and requested_file_path in self.image_files:
                    self.current_file_index = self.image_files.index(requested_file_path)
            except ValueError:
                pass
            # PERFORMANCE FIX: Don't preload immediately - wait for image to display first
            # Preloading will be triggered by on_manager_image_ready or on_image_processed
            # This reduces resource competition with current image loading

            total_time = time.time() - load_start
            logger.info(f"[LOAD] ========== load_raw_image() COMPLETED for {filename} in {total_time:.3f}s ==========")
            safe_print(f"[PERF] ✅ LOAD COMPLETE: {os.path.basename(requested_file_path)} in {total_time*1000:.1f}ms")
        except Exception as e:
            total_time = time.time() - load_start
            logger.error(f"[LOAD] ========== CRITICAL ERROR in load_raw_image (at {time.time():.3f}, duration: {total_time:.3f}s) ==========")
            logger.error(f"[LOAD] Exception type: {type(e).__name__}, message: {e}", exc_info=True)
            logger.error(f"[LOAD] Full traceback:\n{traceback.format_exc()}")
            requested_file_name = requested_file_path if 'requested_file_path' in locals() else (file_path if 'file_path' in locals() else 'unknown')
            safe_print(f"[PERF] ❌ LOAD ERROR: {os.path.basename(requested_file_name)} failed after {total_time*1000:.1f}ms - {type(e).__name__}: {e}")
            if hasattr(self, "loading_overlay"):
                self.loading_overlay.hide_loading()
            # Try to show error to user
            try:
                self.show_error("Load Error", f"Failed to load image: {str(e)}")
                # Graceful handling for ejected volumes or missing files
                target_path = requested_file_path if 'requested_file_path' in locals() else (file_path if 'file_path' in locals() else None)
                if target_path and not os.path.exists(target_path):
                    parent_dir = os.path.dirname(target_path)
                    if not os.path.exists(parent_dir):
                        self.reset_to_initial_state()
            except Exception:
                pass
            raise

    def _load_animated_image(self, requested_file_path):
        """Handle animated images (GIF/WebP)"""
        self.current_file_path = requested_file_path
        self._last_loaded_path = requested_file_path
        if getattr(self, "view_mode", "single") == "single":
            self._maybe_leave_burst_group_view_for_path(requested_file_path)
        if self.image_files and requested_file_path in self.image_files:
            self.current_file_index = self.image_files.index(requested_file_path)
        self._sync_filmstrip_index()

        started = self._load_and_start_animation(requested_file_path)
        if started:
            if not self.image_cache.get_exif(requested_file_path):
                self.image_manager.load_image(
                    requested_file_path,
                    priority=Priority.BACKGROUND,
                    cancel_existing=False,
                    use_full_resolution=False,
                    stages={"exif"},
                )
            self.update_status_bar()
            self._start_preloading()
            return True
        return False

    def _finalize_cache_load(self, requested_file_path):
        """Final cleanup after a cache load (hide overlay, etc.)"""
        if hasattr(self, "loading_overlay"):
            self.loading_overlay.hide_loading()
