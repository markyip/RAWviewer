"""image_load mixin, part 3 of 3 (25 methods, 747 lines)."""
from __future__ import annotations

import contextlib
import os
import sys

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QPixmap

from core.config import _env_int, _norm_path
from rawviewer_app.processing import PixmapConverter

from .. import lazy
from ..lazy import Priority, check_memory_cache_for_image, image_covers_sensor_resolution, process_file_from_path

lazy.register(sys.modules[__name__])


class ImageLoadPart3:
    def _calculate_load_stages(self, requested_file_path):
        """Calculate which stages to load for the given file and context"""
        is_raw = self._is_raw_file(requested_file_path)
        from_gallery = getattr(self, "_loading_from_gallery", False)
        preserve_zoom_navigation = self._preserve_zoom_navigation_active()
        nav_single_step = (
            getattr(self, "view_mode", "single") == "single"
            and getattr(self, "_navigation_in_progress", False)
            and not getattr(self, "_loading_from_filmstrip", False)
            and not getattr(self, "_preserve_nav_zoom_active", False)
        )

        # Gallery entry always takes thumbnail/exif path for RAW
        if from_gallery and is_raw and not preserve_zoom_navigation:
            return {"thumbnail", "exif"}

        # DNG specific handling
        is_dng_file = os.path.splitext(requested_file_path)[1].lower() == ".dng"
        if is_dng_file and getattr(self, "view_mode", "single") == "single":
            from common_image_loader import dng_prefers_embedded_preview_first
            if not dng_prefers_embedded_preview_first(requested_file_path):
                return {"full"}

        # Default behavior for non-RAW or when zoom/navigation is active
        if not is_raw or preserve_zoom_navigation or nav_single_step:
            return {"exif", "full"}

        # RAW preview-first workflow
        return None

    def _should_use_full_resolution(self, requested_file_path):
        """Determine if we should use full resolution for this load"""
        is_raw = self._is_raw_file(requested_file_path)
        preserve_zoom_navigation = self._preserve_zoom_navigation_active()

        # Use full resolution for zoom navigation, DNG (if applicable), etc.
        return preserve_zoom_navigation or not is_raw

    def on_cache_hit(self, file_path, cache_type):
        """Handle cache hit events for performance monitoring."""
        # Could be used for performance analytics
        pass

    def _restart_full_res_idle_timer(self, file_path=None) -> None:
        """Restart the 3s idle timer that triggers full-res decode."""
        try:
            fp = file_path or getattr(self, "current_file_path", None)
            if not fp:
                return
            it = getattr(self, "_full_res_idle_timer", None)
            if it is None:
                return
            self._full_res_idle_path = _norm_path(fp)
            it.stop()
            delay = _env_int("RAWVIEWER_FULL_RES_IDLE_MS", 3000, minimum=1000)
            it.start(delay)
        except Exception:
            pass

    def _on_full_res_idle_timeout(self) -> None:
        """Idle trigger: decode full resolution for the current image."""
        try:
            if getattr(self, "view_mode", "single") != "single":
                return
            fp = getattr(self, "current_file_path", None)
            if not fp or _norm_path(fp) != getattr(self, "_full_res_idle_path", None):
                return
            # Skip if already full-res on screen.
            try:
                from common_image_loader import is_raw_file
                if not is_raw_file(fp):
                    return
            except Exception:
                return
            # Request full decode (CURRENT priority).
            try:
                from image_load_manager import Priority as _Prio
                self.image_manager.load_image(
                    file_path=fp,
                    priority=_Prio.CURRENT,
                    cancel_existing=False,
                    use_full_resolution=True,
                    stages={"full"},
                    bypass_cache=False,
                )
            except Exception:
                pass
        except Exception:
            pass

    def _request_full_res_decode(self, file_path, *, priority_current=True, force=False, urgent=False) -> bool:
        """Request a full-resolution decode for file_path. Returns True."""
        try:
            from image_load_manager import Priority as _Prio
            self.image_manager.load_image(
                file_path=file_path,
                priority=_Prio.CURRENT if priority_current else _Prio.BACKGROUND,
                cancel_existing=False,
                use_full_resolution=True,
                stages={"full"},
                bypass_cache=bool(force),
            )
            return True
        except Exception:
            return False

    def _full_res_only_mode(self) -> bool:
        return str(os.environ.get("RAWVIEWER_FULL_RES_ONLY", "")).strip().lower() in (
            "1", "true", "yes", "on",
        )

    def _load_full_resolution_on_demand(self):
        """Load full resolution image when user zooms in (on-demand loading)"""
        import logging
        logger = logging.getLogger(__name__)

        if not self.current_file_path:
            return

        self._allow_full_decode_for_user_zoom()

        if self.current_pixmap and not self.current_pixmap.isNull():
            exif_now = None
            with contextlib.suppress(Exception):
                exif_now = self.image_cache.get_exif(self.current_file_path)
            if image_covers_sensor_resolution(
                self.current_pixmap.width(), self.current_pixmap.height(), exif_now
            ):
                self._is_half_size_displayed = False
                return

        exif_for_res = None
        with contextlib.suppress(Exception):
            exif_for_res = self.image_cache.get_exif(self.current_file_path)

        # Check if full resolution is already cached
        cached_full = self.image_cache.get_full_image(self.current_file_path)
        if cached_full is not None:
            ch, cw = cached_full.shape[0], cached_full.shape[1]
            if image_covers_sensor_resolution(cw, ch, exif_for_res):
                logger.info("Full resolution image already cached, loading...")
                # CRITICAL: UnifiedImageProcessor caches already-oriented images.
                # Mark as oriented to prevent double rotation in display_numpy_image.
                self._orientation_already_applied = True
                self.display_numpy_image(cached_full)
                self._is_half_size_displayed = False
                self._full_resolution_loading = False
                return
            with contextlib.suppress(Exception):
                self.image_cache.full_image_cache.remove(self.current_file_path)

        # Check if we're already loading full resolution
        if hasattr(self, '_full_resolution_loading') and self._full_resolution_loading:
            return
        norm = _norm_path(self.current_file_path)
        if getattr(self, "_sensor_full_decode_queued_norm", None) == norm:
            self._full_resolution_loading = True
            return

        logger.info("Loading full resolution image on-demand (user zoomed in)...")
        if self._request_full_res_decode(
            self.current_file_path, priority_current=True, urgent=True
        ):
            self._full_resolution_loading = True

    def _on_full_resolution_error(self, error_msg):
        """Handle full resolution loading error"""
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to load full resolution: {error_msg}")
        self._full_resolution_loading = False

    def _nav_preload_display_radius(self) -> int:
        return _env_int("RAWVIEWER_NAV_PRELOAD_RADIUS", 6, minimum=1)

    def _nav_preload_display_near_count(self) -> int:
        # In the no-JPEG RAW workflow, display buffers are the only tier that
        # can paint (thumbnail prefetch is useless there), and fresh decodes
        # cost ~0.2-0.5s each -- prefetch deeper so normal browsing pace stays
        # inside warm cache. Kept at 3 (not more): each buffered neighbor
        # costs ~50-100MB and 8GB machines hit memory pressure fast.
        default = 3 if self._suppress_jpeg_interim() else 2
        return _env_int("RAWVIEWER_NAV_PRELOAD_NEAR", default, minimum=1)

    def _should_defer_nav_preload(self) -> bool:
        """Hold neighbor prefetch during session restore until the staged nav phase."""
        if not self._session_restore_burst_active():
            return False
        flag = os.environ.get("RAWVIEWER_SESSION_RESTORE_DEFER_PRELOAD", "1").strip().lower()
        return flag not in ("0", "false", "no", "off")

    def _should_defer_full_decode(self) -> bool:
        """Hold sensor-res decode during session restore until the staged full-decode phase."""
        if not self._session_restore_burst_active():
            return False
        if getattr(self, "_session_restore_full_decode_allowed", False):
            return False
        flag = os.environ.get("RAWVIEWER_SESSION_RESTORE_DEFER_PRELOAD", "1").strip().lower()
        return flag not in ("0", "false", "no", "off")

    def _run_post_session_restore_full_decode(self) -> None:
        import logging

        logger = logging.getLogger(__name__)
        if not self._session_restore_burst_active():
            return
        if getattr(self, "view_mode", "single") == "gallery":
            # User left single view -- do not keep retrying a 30+ MP full decode
            # every 2s (that competed with gallery tile fill). Abort staging;
            # single-view will decode normally on the next gallery→single.
            self._abort_session_restore_heavy_for_gallery()
            return
        self._session_restore_full_decode_allowed = True
        logger.info(
            "[SESSION] Starting deferred full decode for current image "
            "(RAWVIEWER_SESSION_RESTORE_FULL_DECODE_DELAY_MS)"
        )
        self._flush_deferred_full_exif_after_first_paint()
        self._flush_deferred_sensor_full_decode(force=True)
        preload_delay_ms = _env_int(
            "RAWVIEWER_SESSION_RESTORE_PRELOAD_DELAY_MS", 200, minimum=0
        )
        QTimer.singleShot(preload_delay_ms, self._run_post_session_restore_nav_preload)

    def _run_post_session_restore_nav_preload(self) -> None:
        import logging

        logger = logging.getLogger(__name__)
        if not getattr(self, "_session_restore_defer_preload", False):
            return
        if getattr(self, "view_mode", "single") == "gallery":
            self._abort_session_restore_heavy_for_gallery()
            return
        self._session_restore_defer_preload = False
        self._session_restore_staged_scheduled = False
        logger.info("[SESSION] Starting deferred nav preload after session restore")
        self._start_preloading()
        self._schedule_idle_display_prefetch()
        with contextlib.suppress(Exception):
            self._queue_display_tier_prefetch_for_paths(
                self._paths_within_nav_prefetch_radius()
            )
        # Full decode + neighbor preload have landed -- now it's safe to pay
        # the main-thread CuPy/torch import without starving those timers.
        self._schedule_deferred_gpu_backend_import(reason="post_session_restore")

    def _preload_adjacent_display_buffers(self):
        """Prefetch adjacent display buffers (embedded preview / half-res) for navigation."""
        from image_load_manager import Priority

        flag = os.environ.get("RAWVIEWER_NAV_PRELOAD_DISPLAY", "1").strip().lower()
        if flag not in ("1", "true", "yes", "on"):
            return
        if not self.image_files or self.current_file_index < 0:
            return
        if len(self.image_files) <= 1:
            return

        zoomed = self._single_view_is_zoomed_in()
        radius = self._nav_preload_display_radius()
        near_count = self._nav_preload_display_near_count()
        n = len(self.image_files)
        for step in range(1, radius + 1):
            near = step <= near_count
            next_idx = (self.current_file_index + step) % n
            prev_idx = (self.current_file_index - step) % n
            prio_next = Priority.PRELOAD_NEXT if step == 1 else Priority.BACKGROUND
            prio_prev = Priority.PRELOAD_PREV if step == 1 else Priority.BACKGROUND
            self._request_display_buffer_prefetch(
                self.image_files[next_idx], priority=prio_next, zoomed=zoomed, near=near
            )
            self._request_display_buffer_prefetch(
                self.image_files[prev_idx], priority=prio_prev, zoomed=zoomed, near=near
            )

    def _reset_stuck_load_state(self) -> None:
        """Clear all state that can permanently block full-resolution loading."""
        import logging

        logger = logging.getLogger(__name__)
        logger.warning("[TTFR] Resetting stuck load state")
        # Clear sensor decode gate.
        self._sensor_full_decode_queued_norm = None
        self._defer_sensor_full_decode_path = None
        self._full_res_idle_path = None
        self._single_view_first_render_logged = False
        # Clear manager tasks and raw slots.
        try:
            if hasattr(self.image_manager, "cancel_non_current_heavy_raw_except"):
                self.image_manager.cancel_non_current_heavy_raw_except(None)
        except Exception:
            pass
        try:
            if hasattr(self.image_manager, "cancel_gallery_tasks"):
                self.image_manager.cancel_gallery_tasks()
        except Exception:
            pass
        # Purge orphaned raw slots.
        try:
            gj = getattr(self, "gallery_justified", None)
            if gj is not None and hasattr(gj, "_evict_stale_active_tasks"):
                gj._evict_stale_active_tasks(max_age_s=0.0, drop_prefetch=True)
        except Exception:
            pass
        logger.warning("[TTFR] Stuck load state reset complete")

    def _start_preloading(self):
        """Start preloading adjacent images for fast navigation using aggressive caching strategy."""
        from image_load_manager import Priority

        if not self.image_files or self.current_file_index < 0:
            return
        if len(self.image_files) <= 1:
            return
        if self._should_defer_nav_preload():
            return

        self._preload_adjacent_display_buffers()

        current_path = self.current_file_path
        zoomed = not getattr(self, "fit_to_window", True)
        near = self._nav_preload_display_near_count()
        next_count = min(self._PRELOAD_NEXT_COUNT, len(self.image_files) - 1)
        try:
            from core.memory_pressure import adjacent_preload_cap

            _cap = adjacent_preload_cap()
            if _cap is not None:
                next_count = min(next_count, _cap)
        except Exception:
            pass
        for i in range(1, next_count + 1):
            next_index = (self.current_file_index + i) % len(self.image_files)
            next_file = self.image_files[next_index]
            if _norm_path(next_file) == _norm_path(current_path):
                continue
            if i <= near:
                self._request_display_buffer_prefetch(
                    next_file, priority=Priority.PRELOAD_NEXT, zoomed=zoomed
                )
                continue
            if self.image_cache.get_thumbnail(next_file) is not None:
                continue
            cached_item, _cache_type = check_memory_cache_for_image(
                next_file, use_full_resolution=False
            )
            if cached_item is None:
                self.image_manager.load_image(
                    file_path=next_file,
                    priority=Priority.PRELOAD_NEXT,
                    cancel_existing=False,
                    use_full_resolution=False,
                    stages={"thumbnail", "exif"},
                )

        prev_count = min(self._PRELOAD_PREV_COUNT, len(self.image_files) - 1)
        try:
            from core.memory_pressure import adjacent_preload_cap

            _cap = adjacent_preload_cap()
            if _cap is not None:
                prev_count = min(prev_count, 1 if _cap <= 1 else max(1, _cap - 1))
        except Exception:
            pass
        for i in range(1, prev_count + 1):
            prev_index = (self.current_file_index - i) % len(self.image_files)
            prev_file = self.image_files[prev_index]
            if _norm_path(prev_file) == _norm_path(current_path):
                continue
            if i <= near:
                self._request_display_buffer_prefetch(
                    prev_file, priority=Priority.PRELOAD_PREV, zoomed=zoomed
                )
                continue
            if self.image_cache.get_thumbnail(prev_file) is not None:
                continue
            cached_item, _cache_type = check_memory_cache_for_image(
                prev_file, use_full_resolution=False
            )
            if cached_item is None:
                self.image_manager.load_image(
                    file_path=prev_file,
                    priority=Priority.PRELOAD_PREV,
                    cancel_existing=False,
                    use_full_resolution=False,
                    stages={"thumbnail", "exif"},
                )

        self._schedule_idle_display_prefetch()

    def _preload_next_image_full(self):
        """Aggressively preload next image's full version in background for instant display"""
        try:
            if not self.image_files or self.current_file_index < 0:
                return

            # Get next image
            next_index = (self.current_file_index + 1) % len(self.image_files)
            next_file = self.image_files[next_index]

            # Check if already fully cached (both numpy array and QPixmap)
            cached_image = self.image_cache.get_full_image(next_file)
            cached_pixmap = self.image_cache.get_pixmap(next_file)

            # If we have full image but no QPixmap, convert it
            if cached_image is not None and cached_pixmap is None:
                try:
                    converter = PixmapConverter(next_file, cached_image, self.image_cache)
                    self._track_pixmap_converter(converter)
                    converter.start()
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.debug(f"Failed to start pixmap converter for {next_file}: {e}")

            # If we don't have full image yet, preload via ImageLoadManager (same pipeline as navigation)
            elif cached_image is None:
                try:
                    mgr = getattr(self, "image_manager", None)
                    if mgr is not None:
                        mgr.load_image(
                            file_path=next_file,
                            priority=Priority.PRELOAD_NEXT,
                            cancel_existing=False,
                            use_full_resolution=False,
                            stages={"full"},
                        )
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.debug(f"Failed to queue ILM preload for {next_file}: {e}")
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Error in _preload_next_image_full: {e}")

    def _on_preload_error(self, file_path, error_msg):
        """Handle preload error (silent - preloading is background operation)"""
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Preload error for {os.path.basename(file_path)}: {error_msg}")

    def _on_preloaded_image_ready(self, file_path, rgb_image):
        """Handle preloaded image - convert to QPixmap in background"""
        try:
            if rgb_image is not None:
                # Convert to QPixmap in background
                converter = PixmapConverter(file_path, rgb_image, self.image_cache)
                self._track_pixmap_converter(converter)
                converter.start()
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Error in _on_preloaded_image_ready: {e}")

    def _gpu_request_full_resolution_if_needed(self):
        """When zooming in on the GPU view, request the full-res buffer if we only have a preview."""
        if self._raw_recovery_preview_active():
            return
        # Adjust panel paints its own half-res / settle preview tiers. Those
        # never cover EXIF sensor WxH, so treating every pinch/100% as a
        # browse-path sensor decode flooded the log with
        # "[ZOOM] Preview below sensor resolution" and fought the edit base.
        if self._adjust_panel_active():
            return
        with contextlib.suppress(Exception):
            self._maybe_request_full_res_for_smooth_zoom()

    def _request_sensor_dimensions_async(self, file_path, cached_exif) -> None:
        """Resolve a RAW file's true sensor dimensions on a worker thread.

        Reading sensor size needs rawpy.imread()/EXIF header parsing, which is slow on
        external/network drives. Doing it inline in update_status_bar() blocks the main
        thread on every navigation, so we offload it and refresh the HUD when done.
        """
        if not file_path:
            return
        try:
            from common_image_loader import is_raw_file as _is_raw
            if not _is_raw(file_path):
                return
        except Exception:
            pass
        if isinstance(cached_exif, dict):
            ow = cached_exif.get("original_width")
            oh = cached_exif.get("original_height")
            if ow and oh:
                if cached_exif.get("verified_orientation"):
                    return
                if max(int(ow), int(oh)) > 1920:
                    return
        # Attempt each file at most once per session: prevents both concurrent duplicate
        # probes and a re-probe loop when extraction fails (we then keep display dims).
        attempted = getattr(self, "_sensor_dim_probe_attempted", None)
        if attempted is None:
            attempted = set()
            self._sensor_dim_probe_attempted = attempted
        if file_path in attempted:
            return
        attempted.add(file_path)

        # Connect the completion signal once.
        if not getattr(self, "_sensor_dims_signal_connected", False):
            with contextlib.suppress(Exception):
                self._sensor_dims_resolved.connect(self._on_sensor_dims_resolved)
            self._sensor_dims_signal_connected = True

        def _probe(path=file_path):
            ow = oh = None
            try:
                import rawpy

                from enhanced_raw_processor import _rawpy_global_lock

                with _rawpy_global_lock, rawpy.imread(path) as raw:
                    ow = int(raw.sizes.width)
                    oh = int(raw.sizes.height)
            except Exception:
                ow = oh = None
            if not ow or not oh:
                try:
                    tags = process_file_from_path(path, details=False)
                    for w_tag in ("EXIF ExifImageWidth", "Image ImageWidth", "Image Width"):
                        if w_tag in tags:
                            ow = int(str(tags[w_tag]))
                            break
                    for h_tag in (
                        "EXIF ExifImageLength", "Image ImageLength", "Image Height",
                        "Image Length", "EXIF ExifImageHeight",
                    ):
                        if h_tag in tags:
                            oh = int(str(tags[h_tag]))
                            break
                except Exception:
                    ow = oh = None
            try:
                if ow and oh:
                    ce = self.image_cache.get_exif(path) or (cached_exif if isinstance(cached_exif, dict) else {})
                    if isinstance(ce, dict):
                        ce = dict(ce)
                        ce["original_width"] = ow
                        ce["original_height"] = oh
                        self.image_cache.put_exif(path, ce)
            except Exception:
                pass
            with contextlib.suppress(Exception):
                self._sensor_dims_resolved.emit(path)

        try:
            import threading
            threading.Thread(
                target=_probe, name="sensor-dim-probe", daemon=True
            ).start()
        except Exception:
            attempted.discard(file_path)

    def _on_folder_load_error(self, token, title, message):
        if token != getattr(self, "_folder_load_generation", None):
            return
        if getattr(self, "_folder_fast_open_token", None) == token:
            self._folder_fast_open_token = None
        self._active_folder_load_worker = None
        self._active_folder_load_signals = None
        self._hide_all_loading_indicators()
        self.show_error(title, message)

    def _on_folder_load_ready(self, token, image_files, bulk_metadata, file_stats,
                              folder_path, start_file, start_view, scan_time, sort_time):
        if token != getattr(self, "_folder_load_generation", None):
            return

        import logging
        import time
        logger = logging.getLogger(__name__)
        apply_start = time.time()
        self._active_folder_load_worker = None
        self._active_folder_load_signals = None
        was_fast_open = getattr(self, "_folder_fast_open_token", None) == token
        preserved_path = self.current_file_path if was_fast_open else None

        try:
            extensions = self.get_supported_extensions()
            if not image_files:
                self.show_no_images_message(extensions)
                self._hide_all_loading_indicators()
                self.current_folder = None
                self.image_files = []
                self._semantic_search_corpus_files = []
                self._gallery_search_index_session_key = None
                self.current_file_index = -1
                self.current_file_path = None
                self.update_status_bar()
                return

            self.current_folder = folder_path
            self._sync_burst_rejected_for_current_folder()
            self.image_files = image_files
            # Parent-fallback scope is all-or-nothing (worker yields child files
            # only when the opened folder has none): surface that to the user.
            if _norm_path(os.path.dirname(image_files[0])) != _norm_path(folder_path):
                logger.info(
                    "[FOLDER] Parent fallback: %d files from subfolders of %s",
                    len(image_files), folder_path,
                )
                if hasattr(self, "status_bar"):
                    self.status_bar.showMessage(
                        f"Showing {len(image_files)} photos from subfolders of "
                        f"{os.path.basename(folder_path)}",
                        5000,
                    )
            self._reconcile_burst_rejected_paths_with_folder()
            self._reconcile_gallery_bookmarks_with_folder()
            self._semantic_search_corpus_files = list(image_files)
            self._gallery_search_index_session_key = None
            self._filmstrip_warmed_paths = set()
            self._gallery_bulk_metadata = bulk_metadata
            self._bump_gallery_rating_presence_epoch()
            if not was_fast_open:
                self._folder_fast_open_preserved_path = None
                # Standard folder load already performed full capture-time sorting in background
                self._folder_sort_refinement_applied_token = token
                # Folder switched: invalidate render/task state from previous folder so
                # stale async callbacks cannot keep the old content visible.
                try:
                    if getattr(self, "image_manager", None) is not None:
                        self.image_manager.cancel_all_tasks()
                except Exception:
                    pass
                self._displayed_content_path = None
                self._manager_display_track_path = None
                self._manager_displayed_max_dim = 0
                self._last_loaded_path = None
                self._last_manager_exif_path = None
                self._last_manager_exif_ts = 0.0
                self._last_manager_exif_was_minimal = False
                # Clear previously displayed single-view content so the new folder does not
                # temporarily show stale dimensions/pixels from the old folder.
                self.current_image = None
                self.current_pixmap = None
                self._displayed_content_path = None
                self._manager_displayed_max_dim = 0
                self._is_half_size_displayed = False
                self._full_resolution_loading = False
                self._preserve_nav_zoom_active = False
                self._pending_zoom_restore = False
                self._restore_zoom_center = None
                self._restore_zoom_level = None
                if hasattr(self, '_maintain_zoom_on_navigation'):
                    with contextlib.suppress(AttributeError):
                        delattr(self, "_maintain_zoom_on_navigation")

                # Immediately clear the image view to prevent stale pixels
                self.image_label.setPixmap(QPixmap())
                self.image_label.setText("Loading folder...")
                self.image_label.adjustSize()
                self.scroll_area.updateGeometry()
                try:
                    if hasattr(self, "image_label") and self.image_label is not None:
                        self.image_label.clear()
                except Exception:
                    pass

            try:
                idx = 0
                if preserved_path:
                    preserved_norm = _norm_path(preserved_path)
                    for i, img_file in enumerate(self.image_files):
                        if _norm_path(img_file) == preserved_norm:
                            idx = i
                            break
                elif start_file:
                    start_file_path = None
                    # Windows-friendly matching for file-open/folder-switch flows:
                    # caller may pass full path, basename, or different casing/slashes.
                    start_file_norm = _norm_path(start_file)
                    start_file_base_norm = os.path.normcase(os.path.basename(start_file))
                    for img_file in self.image_files:
                        if (
                            _norm_path(img_file) == start_file_norm
                            or os.path.normcase(os.path.basename(img_file)) == start_file_base_norm
                        ):
                            start_file_path = img_file
                            break
                    if start_file_path is None:
                        logger.warning(
                            "[FOLDER] Requested start_file not found after scan: %s",
                            start_file,
                        )
                    idx = self.image_files.index(start_file_path) if start_file_path in self.image_files else 0
                self.current_file_index = idx
                self.current_file_path = self.image_files[idx]
            except Exception:
                logger.exception("Error determining start file")
                self.current_file_index = 0
                self.current_file_path = self.image_files[0] if self.image_files else None

            fast_preserved = getattr(self, "_folder_fast_open_preserved_path", None)
            effective_preserved = preserved_path or fast_preserved
            skip_single_reload = bool(
                effective_preserved
                and self.current_file_path
                and _norm_path(effective_preserved) == _norm_path(self.current_file_path)
                and (
                    was_fast_open
                    or getattr(self, "_single_view_first_render_logged", False)
                    or self._single_view_pixels_on_screen(self.current_file_path)
                )
            )

            if hasattr(self, 'view_mode') and self.view_mode == 'gallery' and getattr(self, 'gallery_justified', None):
                self.gallery_justified.show_loading_message("Preparing gallery...")

            if hasattr(self, 'view_mode') and self.view_mode == 'gallery':
                self._ensure_gallery_widget()
                self._show_gallery_view()
                # Force a second pass after layout settles to avoid "empty gallery"
                # race when switching folders quickly.
                QTimer.singleShot(0, self._update_gallery_view)
                QTimer.singleShot(120, self._update_gallery_view)
            elif skip_single_reload:
                self._folder_fast_open_preserved_path = None
                QTimer.singleShot(0, self._sync_filmstrip_to_folder)
                self.update_status_bar()
                logger.info(
                    "[FOLDER] Folder index ready (%d images); kept fast-opened image on screen",
                    len(self.image_files),
                )
                if len(self.image_files) > 1:
                    self._schedule_folder_sort_refinement(token, file_stats)
            else:
                self._show_single_view()
                QTimer.singleShot(0, self._sync_filmstrip_to_folder)
                # _show_single_view handles its own load_raw_image() call if current_file_path is set.
                if hasattr(self, 'gallery_justified') and self.gallery_justified:
                    self.gallery_justified._background_loading_active = False
                    if hasattr(self.gallery_justified, '_load_timer') and self.gallery_justified._load_timer:
                        self.gallery_justified._load_timer.stop()

            total_time = scan_time + sort_time + (time.time() - apply_start)
            logger.info(
                "[FOLDER] Background folder load applied in %.3fs (scan %.3fs, sort %.3fs)",
                total_time,
                scan_time,
                sort_time,
            )
            self._hide_all_loading_indicators()
            self.save_session_state()
            self._maybe_resume_semantic_indexing()
        except Exception as e:
            logger.error(f"Error updating gallery view for folder {folder_path}: {e}", exc_info=True)
            self._hide_all_loading_indicators()
            self.show_error("Gallery Update Error", f"Error updating gallery view:\n{str(e)}")
        finally:
            if getattr(self, "_folder_fast_open_token", None) == token:
                self._folder_fast_open_token = None
