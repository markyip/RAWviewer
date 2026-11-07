"""navigation mixin, part 1 of 2 (22 methods, 866 lines)."""
from __future__ import annotations

import contextlib
import logging
import os
import sys
import time

from PyQt6.QtCore import QRunnable, Qt, QTimer

from core.config import _env_int, _norm_path, safe_print
from rawviewer_app.signals import FolderResortSignals

from .. import lazy
from ..paths import _require_str_path
from ..qt_helpers import _get_bg_thread_pool

lazy.register(sys.modules[__name__])

logger = logging.getLogger(__name__)


class NavigationPart1:
    def _sync_burst_rejected_for_current_folder(self) -> None:
        folder = getattr(self, "current_folder", None)
        if folder:
            self._load_burst_rejected_paths(folder)

    def _reconcile_burst_rejected_paths_with_folder(self) -> None:
        """Un-reject any path the folder scan actually found.

        _burst_rejected_paths is a permanent per-folder blacklist (survives
        restarts via QSettings) with nothing that ever removes an entry --
        rejecting a photo moves it to the Discard subfolder (so it drops out
        of image_files naturally), but if the user manually moves it back
        into the main folder, the stale path string stays blacklisted
        forever and the photo never reappears in gallery view even though
        it's sitting right there on disk (single-image view isn't filtered
        through this set, which is why it opens fine there). A path's
        presence in the current folder scan is proof it's no longer
        sitting in Discard, so drop it from the blacklist.
        """
        rejected = getattr(self, "_burst_rejected_paths", set()) or set()
        if not rejected:
            return
        present = {_norm_path(p) for p in (getattr(self, "image_files", None) or [])}
        still_rejected = rejected - present
        if still_rejected != rejected:
            self._burst_rejected_paths = still_rejected
            self._save_burst_rejected_paths()

    def _navigation_files(self) -> list:
        """Files used for filmstrip and prev/next while browsing."""
        if self._is_semantic_search_filter_active():
            result = getattr(self, "_semantic_search_result_paths", None)
            if result is not None:
                files = [p for p in result if os.path.isfile(p)]
            else:
                files = list(getattr(self, "image_files", []) or [])
        else:
            files = list(getattr(self, "image_files", []) or [])
        scope = getattr(self, "_slideshow_scope_paths", None)
        if scope:
            return [p for p in scope if p and os.path.isfile(p)]
        if getattr(self, "_gallery_bookmark_filter_active", False):
            files = self._filter_paths_to_bookmarked(files)
        files = self._filter_paths_to_min_rating(
            files, int(getattr(self, "_gallery_rating_filter_min", 0) or 0)
        )
        return files

    def _in_nav_quiet_window(self) -> bool:
        """True if the user navigated very recently.

        Background prefetch (filmstrip warm-up) is held off during this window so the
        foreground image load keeps worker slots and the main thread stays responsive.
        """
        import time
        quiet_ms = _env_int("RAWVIEWER_NAV_PREFETCH_QUIET_MS", 450, minimum=0)
        if quiet_ms <= 0:
            return False
        last = float(getattr(self, "_last_navigation_time", 0.0) or 0.0)
        if last <= 0.0:
            return False
        return (time.time() - last) * 1000.0 < quiet_ms

    def _cancel_stale_folder_async_work(self, reason: str = "") -> None:
        """Stop background indexing/prefetch from a previous folder scope."""
        import logging

        logger = logging.getLogger(__name__)
        if reason:
            logger.info("[FOLDER] Cancelling stale async work (%s)", reason)
        self._abort_semantic_indexing_deferred()
        self._semantic_index_active_token = None
        self._semantic_indexing_in_progress = False
        self._face_index_active_token = None
        self._semantic_index_prep_in_progress = False
        self._release_indexing_load_throttle()
        try:
            mgr = getattr(self, "image_manager", None)
            if mgr is not None and hasattr(mgr, "cancel_all_tasks"):
                mgr.cancel_all_tasks()
        except Exception:
            pass
        try:
            pm = getattr(self, "preload_manager", None)
            if pm is not None and hasattr(pm, "cancel_all_preloads"):
                pm.cancel_all_preloads()
        except Exception:
            pass
        try:
            gj = getattr(self, "gallery_justified", None)
            if gj is not None and hasattr(gj, "prepare_for_folder_change"):
                gj.prepare_for_folder_change()
        except Exception:
            pass

    def _play_next_webp_frame(self):
        if not hasattr(self, "_current_webp_frames") or not self._current_webp_frames:
            return
        idx = self._current_webp_idx
        pixmap = self._current_webp_frames[idx]

        display_pixmap = pixmap
        if self.fit_to_window:
            v_size = self.scroll_area.viewport().size()
            display_pixmap = pixmap.scaled(v_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        self.image_label.setPixmap(display_pixmap)
        self.image_label.resize(display_pixmap.size())

        duration = self._current_webp_durations[idx]
        self._current_webp_idx = (idx + 1) % len(self._current_webp_frames)

        if hasattr(self, "_webp_timer") and self._webp_timer:
            self._webp_timer.start(duration)

    def resort_current_folder(self):
        """Resort the current folder with new sorting preference (sort runs off UI thread)."""
        if not self.current_folder or not self.image_files:
            return

        paths = list(self.image_files)
        token = getattr(self, "_folder_resort_generation", 0) + 1
        self._folder_resort_generation = token

        in_gallery = (
            self.view_mode == "gallery"
            and hasattr(self, "gallery_widget")
            and self.gallery_widget.isVisible()
        )
        anchor_file = None
        if in_gallery and hasattr(self, "gallery_justified") and self.gallery_justified:
            anchor_file = self.gallery_justified.get_scroll_anchor_path()

        self._folder_resort_context = {
            "token": token,
            "current_file": self.current_file_path,
            "old_index": self.current_file_index,
            "in_gallery": in_gallery,
            "anchor_file": anchor_file,
        }

        signals = getattr(self, "_folder_resort_signals", None)
        if signals is None:
            signals = FolderResortSignals()
            signals.ready.connect(self._apply_folder_resort)
            self._folder_resort_signals = signals

        viewer = self
        file_stats = getattr(self, "_quick_folder_file_stats", None)
        newest_first = self.get_sort_preference()
        folder_path = self.current_folder

        class _FolderResortWorker(QRunnable):
            def run(self_inner):
                try:
                    sorted_files, bulk_metadata = viewer.sort_files_by_capture_time(
                        paths,
                        newest_first=newest_first,
                        file_stats=file_stats,
                        folder_path=folder_path,
                    )
                except Exception as e:
                    import logging
                    import traceback

                    logging.getLogger(__name__).error(
                        "[SORT] Background folder resort failed: %s\n%s",
                        e,
                        traceback.format_exc(),
                    )
                    sorted_files, bulk_metadata = paths, {}
                try:
                    from PyQt6 import sip

                    if not sip.isdeleted(signals):
                        signals.ready.emit(token, sorted_files, bulk_metadata)
                except Exception:
                    pass

        _get_bg_thread_pool().start(_FolderResortWorker())

    def _apply_folder_resort(self, token: int, sorted_files: list, bulk_metadata: dict) -> None:
        """Apply background folder re-sort on the UI thread."""
        if token != getattr(self, "_folder_resort_generation", None):
            return
        ctx = getattr(self, "_folder_resort_context", None) or {}
        if ctx.get("token") != token:
            return
        if not self.current_folder or not sorted_files:
            return

        current_file = ctx.get("current_file")
        old_index = ctx.get("old_index", 0)
        in_gallery = ctx.get("in_gallery", False)
        anchor_file = ctx.get("anchor_file")

        self.image_files = sorted_files
        self._gallery_bulk_metadata = bulk_metadata
        self._bump_gallery_rating_presence_epoch()

        if current_file in self.image_files:
            self.current_file_index = self.image_files.index(current_file)
            self.current_file_path = current_file

            import logging

            logger = logging.getLogger(__name__)
            logger.debug(f"[SORT] Resorted folder: {os.path.basename(current_file or '')}")
            logger.debug(f"[SORT] Old index: {old_index}, New index: {self.current_file_index}")
            logger.debug(f"[SORT] Total files: {len(self.image_files)}")

            self.update_status_bar()

            if in_gallery:
                self._sync_gallery_to_folder_files(
                    bulk_metadata=bulk_metadata,
                    scroll_to_current=False,
                    scroll_anchor_path=anchor_file or current_file,
                    force=True,
                )

    def _mtime_birth_from_file_stats(file_stats, file_path: str) -> tuple:
        """Return (mtime, birthtime) from file_stats row (size, mtime) or (size, mtime, birth)."""
        if not file_stats:
            return 0.0, 0.0
        row = file_stats.get(file_path)
        if row is None:
            nk = _norm_path(file_path)
            for key, val in file_stats.items():
                if _norm_path(key) == nk:
                    row = val
                    break
        if row is None:
            return 0.0, 0.0
        if len(row) >= 3:
            return float(row[1]), float(row[2])
        if len(row) >= 2:
            return float(row[1]), 0.0
        return 0.0, 0.0

    def open_folder(self):
        settings = self.get_settings()
        last_dir = settings.value("last_opened_dir", "")
        folder_path = self._pick_directory_dialog("Open Folder", last_dir)
        if folder_path:
            self.load_folder_images(folder_path)
            settings.setValue("last_opened_dir", folder_path)

    def _prepare_file_for_load(self, requested_file_path, force_reload):
        """Prepare file for loading (reset flags, cleanup, etc.)"""
        # Reset flags when loading new image
        self._is_raw_file(requested_file_path)
        self._is_half_size_displayed = False
        self._full_resolution_loading = False
        self._smooth_zoom_full_request_sent = False
        self._raw_status_exif_refresh_path = None
        self._orientation_already_applied = False

        # Pause semantic indexing during active loading
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

        # Clean up current processing
        cleanup_start = time.time()
        logger.info("[LOAD] Starting cleanup of current processing (if any)")

        _prev_fp = getattr(self, "current_file_path", None)
        _same_path_reload = _prev_fp and _norm_path(_prev_fp) == _norm_path(requested_file_path)
        if force_reload:
            _same_path_reload = False
            self._manager_displayed_max_dim = 0

        if _prev_fp and not _same_path_reload:
            self._nav_idle_full_generation = int(getattr(self, "_nav_idle_full_generation", 0)) + 1
            self.image_manager.cancel_current_priority_tasks(_prev_fp)
            self._single_view_display_generation = int(getattr(self, "_single_view_display_generation", 0)) + 1
            self._cancel_content_crossfade()

        if _same_path_reload:
            self._manager_displayed_max_dim = 0

        try:
            prev_norm = _norm_path(getattr(self, "_sensor_full_decode_queued_norm", None) or "")
            if prev_norm and prev_norm != _norm_path(requested_file_path):
                self._sensor_full_decode_queued_norm = None
                self._defer_sensor_full_decode_path = None
        except Exception:
            pass

        # Preempt non-current heavy RAW work
        try:
            if hasattr(self.image_manager, "cancel_non_current_heavy_raw_except"):
                preempted = self.image_manager.cancel_non_current_heavy_raw_except(requested_file_path)
                if preempted:
                    logger.info(
                        "[LOAD] Preempted %d neighbor heavy RAW task(s) for %s",
                        preempted,
                        os.path.basename(requested_file_path),
                    )
        except Exception:
            pass

        # Evict stale active tasks from gallery justified
        try:
            gj = getattr(self, "gallery_justified", None)
            if gj is not None and hasattr(gj, "_evict_stale_active_tasks"):
                gj._evict_stale_active_tasks(max_age_s=6.0, drop_prefetch=False)
        except Exception:
            pass

        # Clean legacy processor if still exists
        if self.current_processor:
            logger.info(f"[LOAD] Legacy processor cleanup: {type(self.current_processor).__name__}")
            self._cleanup_current_processing()

        cleanup_time = time.time() - cleanup_start
        logger.info(f"[LOAD] Cleanup completed in {cleanup_time:.3f}s")

    def rate_current_image(self, rating: int):
        """Set or clear the rating of the active image and update caches."""
        if rating < 0 or rating > 5:
            return

        path = getattr(self, "current_file_path", None)
        if not path or not os.path.isfile(path):
            self.status_bar.showMessage("No active image to rate", 2000)
            return

        # 1. Update rating in memory/persistent EXIF cache
        exif_data = self.image_cache.get_exif(path)
        if exif_data is None:
            exif_data = {
                'orientation': 1,
                'camera_make': 'Unknown',
                'camera_model': 'Unknown',
                'exif_data': {},
                'capture_time': None,
                'original_width': None,
                'original_height': None
            }
        else:
            exif_data = exif_data.copy()

        exif_data['rating'] = rating
        if 'exif_data' in exif_data and isinstance(exif_data['exif_data'], dict):
            exif_data['exif_data'] = exif_data['exif_data'].copy()
            exif_data['exif_data']['rating'] = rating

        self.image_cache.put_exif(path, exif_data)

        # 1b. Persist to the XMP sidecar so the rating survives a restart
        # (in-memory/EXIF-cache updates above are lost when the app closes).
        try:
            from raw_adjustments import write_rating_for_file

            write_rating_for_file(path, rating)
        except Exception:
            logging.getLogger(__name__).warning(
                "Failed to write rating to XMP sidecar for %s", path, exc_info=True
            )
            self.status_bar.showMessage("Rating set, but couldn't save to XMP sidecar", 3000)

        # 2. Update local caches (bulk metadata and justified gallery)
        if hasattr(self, "_gallery_bulk_metadata") and self._gallery_bulk_metadata is not None:
            if path not in self._gallery_bulk_metadata:
                self._gallery_bulk_metadata[path] = {}
            self._gallery_bulk_metadata[path]['rating'] = rating
        self._bump_gallery_rating_presence_epoch()

        gj = getattr(self, "gallery_justified", None)
        if gj is not None and hasattr(gj, "_metadata_cache"):
            if path not in gj._metadata_cache:
                gj._metadata_cache[path] = {}
            gj._metadata_cache[path]['rating'] = rating
            if hasattr(gj, "refresh_visible_tile_for_path"):
                with contextlib.suppress(Exception):
                    gj.refresh_visible_tile_for_path(path)
            elif hasattr(gj, "update_visible_images"):
                with contextlib.suppress(Exception):
                    gj.update_visible_images()
            # refresh_visible_tile_for_path only repaints the pixmap (e.g.
            # after rotation); badges are a separate sync pass, same as
            # bookmark/burst/edited.
            if hasattr(gj, "refresh_gallery_rating_visuals"):
                with contextlib.suppress(Exception):
                    gj.refresh_gallery_rating_visuals()

        bar = self._filmstrip_bar()
        if bar is not None and hasattr(bar, "_metadata_cache"):
            if path not in bar._metadata_cache:
                bar._metadata_cache[path] = {}
            bar._metadata_cache[path]['rating'] = rating
            if hasattr(bar, "refresh_cell_for_path"):
                with contextlib.suppress(Exception):
                    bar.refresh_cell_for_path(path)
            elif hasattr(bar, "update"):
                bar.update()

        # 4. Update visual ratings display and status bar
        self._update_single_view_rating_display(rating)
        stars_str = "⭐" * rating if rating > 0 else "Cleared"
        self.status_bar.showMessage(f"Rating set to: {stars_str}", 2000)

    def can_navigate(self):
        """Check if navigation is allowed (prevents overlapping navigations and rate limiting)"""
        import logging
        import time
        logger = logging.getLogger(__name__)

        nav_in_progress = getattr(self, '_navigation_in_progress', False)
        last_nav_time = getattr(self, '_last_navigation_time', 0)
        current_time = time.time()
        time_since_last = current_time - last_nav_time if last_nav_time > 0 else float('inf')

        logger.debug(f"[NAV_CHECK] can_navigate() called - nav_in_progress={nav_in_progress}, "
                    f"last_nav_time={last_nav_time:.3f}, current_time={current_time:.3f}, "
                    f"time_since_last={time_since_last:.3f}s")

        # Check if navigation is in progress
        # Note: We removed the 100ms rate limiting cooldown because:
        # 1. The _navigation_in_progress flag already prevents overlapping navigations
        # 2. The rate limiting was causing navigation to be blocked unnecessarily
        # 3. Users should be able to navigate quickly if the previous navigation completed
        if nav_in_progress:
            logger.warning("[NAV_CHECK] Navigation BLOCKED: navigation already in progress")
            safe_print("[PERF] 🚫 NAVIGATION BLOCKED: Already in progress")
            return False

        logger.debug("[NAV_CHECK] Navigation ALLOWED")
        return True

    def _debounced_navigate(self, direction, from_slideshow=False):
        """Navigate on key/wheel; coalesce rapid steps when enabled."""
        if not from_slideshow:
            self._stop_slideshow()
        if from_slideshow:
            if direction == "prev":
                self.navigate_to_previous_image()
            elif direction == "next":
                self.navigate_to_next_image()
            return
        delta = -1 if direction == "prev" else 1
        self._schedule_coalesced_navigation(delta)

    def _schedule_coalesced_navigation(self, delta: int) -> None:
        from PyQt6.QtCore import QTimer

        self._nav_coalesce_delta = int(getattr(self, "_nav_coalesce_delta", 0)) + int(
            delta
        )
        ms = _env_int("RAWVIEWER_NAV_COALESCE_MS", 100, minimum=0)
        if ms <= 0:
            self._flush_coalesced_navigation()
            return
        timer = getattr(self, "_nav_coalesce_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._flush_coalesced_navigation)
            self._nav_coalesce_timer = timer
        timer.start(ms)

    def _flush_coalesced_navigation(self) -> None:
        import logging

        logger = logging.getLogger(__name__)
        delta = int(getattr(self, "_nav_coalesce_delta", 0))
        self._nav_coalesce_delta = 0
        if delta == 0 or not self.image_files or len(self.image_files) <= 1:
            return
        if not self.can_navigate():
            return
        n = len(self.image_files)
        old_index = int(self.current_file_index)
        new_index = (old_index + delta) % n
        if new_index == old_index:
            return
        logger.info(
            "[NAV] Coalesced %d step(s): index %d -> %d",
            abs(delta),
            old_index,
            new_index,
        )
        self._capture_nav_zoom_state_before_load()
        if delta > 0:
            self.current_file_index = (new_index - 1) % n
            self.navigate_to_next_image()
        else:
            self.current_file_index = (new_index + 1) % n
            self.navigate_to_previous_image()

    def start_navigation(self):
        """Mark navigation as started"""
        import logging
        import time
        logger = logging.getLogger(__name__)
        current_time = time.time()
        old_state = getattr(self, '_navigation_in_progress', False)
        logger.info(f"[NAV_START] start_navigation() called - old_state={old_state}, setting to True, "
                   f"time={current_time:.3f}")
        self._navigation_in_progress = True
        self._last_navigation_time = current_time
        # Anchor for the [NAVTIME] navigation->display latency metric.
        self._nav_instruction_ts = current_time
        self._pause_prefetch_for_navigation()
        self._cancel_content_crossfade()
        self._skip_resolution_crossfade_once = True
        self._clear_pending_resolution_crossfade()
        logger.debug(f"[NAV_START] Navigation flag set - _navigation_in_progress={self._navigation_in_progress}, "
                    f"_last_navigation_time={self._last_navigation_time:.3f}")

    def finish_navigation(self):
        """Mark navigation as finished"""
        import logging
        import time
        import traceback
        logger = logging.getLogger(__name__)
        old_state = getattr(self, '_navigation_in_progress', None)
        current_time = time.time()
        last_nav_time = getattr(self, '_last_navigation_time', 0)
        nav_duration = current_time - last_nav_time if last_nav_time > 0 else 0

        logger.info(f"[NAV_FINISH] finish_navigation() called - old_state={old_state}, will set to False, "
                   f"nav_duration={nav_duration:.3f}s, time={current_time:.3f}")
        logger.debug(f"[NAV_FINISH] Call stack:\n{traceback.format_stack()[-5:-1]}")

        self._navigation_in_progress = False
        self._last_navigation_finish_ts = current_time
        self._touch_single_view_interactivity()
        logger.debug(f"[NAV_FINISH] Navigation flag cleared - _navigation_in_progress={self._navigation_in_progress}")
        try:
            from PyQt6.QtCore import QTimer

            QTimer.singleShot(0, self._post_navigation_display_tick)
        except Exception:
            pass

    def navigate_to_previous_image(self):
        if getattr(self, '_is_delete_dialog_open', False):
            return
        import logging
        import os
        import time
        import traceback
        logger = logging.getLogger(__name__)

        nav_start_time = time.time()
        logger.info(f"[NAV_PREV] ========== navigate_to_previous_image() STARTED at {nav_start_time:.3f} ==========")
        logger.debug(f"[NAV_PREV] Current state - index: {self.current_file_index}, "
                    f"total_files: {len(self.image_files) if self.image_files else 0}, "
                    f"current_file: {os.path.basename(_require_str_path(self.current_file_path)) if self.current_file_path else 'None'}")

        # Check processor state
        processor_state = "None"
        if hasattr(self, 'current_processor') and self.current_processor:
            processor_state = f"Active (thread_id={self.current_processor.thread().currentThreadId() if hasattr(self.current_processor, 'thread') else 'N/A'})"
        logger.debug(f"[NAV_PREV] Current processor state: {processor_state}")

        # Clear any pending zoom requests for the current image
        if hasattr(self, '_pending_zoom'):
            self._pending_zoom = False
            if hasattr(self, '_pending_zoom_center'):
                delattr(self, '_pending_zoom_center')
            if hasattr(self, '_pending_zoom_thumbnail_size'):
                delattr(self, '_pending_zoom_thumbnail_size')

        # Check if navigation is allowed BEFORE starting navigation
        if not self.can_navigate():
            logger.warning("[NAV_PREV] Navigation BLOCKED by can_navigate() check")
            return

        # Mark navigation as started - must be done before any return statements
        self.start_navigation()

        try:
            try:
                if not self.image_files or len(self.image_files) <= 1:
                    logger.debug("Cannot navigate: no files or only one file")
                    return

                # Calculate previous index with wraparound
                old_index = self.current_file_index
                if self.current_file_index <= 0:
                    self.current_file_index = len(self.image_files) - 1
                else:
                    self.current_file_index -= 1
                # Editor mode: silently skip files the editor cannot open.
                self._advance_index_past_unsupported_for_editing(-1)

                logger.info(f"[NAV_PREV] Navigating to previous image - old_index: {old_index}, new_index: {self.current_file_index}")
                logger.info(f"[TRACK] User navigated to previous image (arrow key) - index: {self.current_file_index}")

                # Check if new index is valid
                if self.current_file_index < 0 or self.current_file_index >= len(self.image_files):
                    logger.error(f"[NAV_PREV] Invalid index after navigation: {self.current_file_index}, total_files: {len(self.image_files)}")
                    self.current_file_index = old_index  # Restore old index
                    return

                self._capture_nav_zoom_state_before_load()

                # Load the current image (at new_index)
                current_file = self.image_files[self.current_file_index]
                logger.debug(f"Loading file at index {self.current_file_index}: {os.path.basename(current_file)}")

                try:
                    self.load_raw_image(current_file)
                    logger.debug(f"Successfully called load_raw_image for: {os.path.basename(current_file)}")
                except Exception as load_error:
                    logger.error(f"Error in load_raw_image during navigation: {load_error}", exc_info=True)
                    logger.debug(f"Load error traceback: {traceback.format_exc()}")
                    # Restore previous index on error
                    self.current_file_index = old_index
                    raise

                try:
                    self.schedule_save_session_state()
                    logger.debug("Session state saved")
                except Exception as save_error:
                    logger.warning(f"Error saving session state: {save_error}")

            finally:
                # Always mark navigation as finished (inner try)
                inner_finally_time = time.time()
                logger.debug(f"[NAV_PREV] Inner finally block reached at {inner_finally_time:.3f} "
                           f"(nav duration: {inner_finally_time - nav_start_time:.3f}s)")
                self.finish_navigation()

        except Exception as e:
            error_time = time.time()
            logger.error(f"[NAV_PREV] ========== EXCEPTION in navigate_to_previous_image "
                        f"(at {error_time:.3f}, duration: {error_time - nav_start_time:.3f}s) ==========")
            logger.error(f"[NAV_PREV] Exception type: {type(e).__name__}, message: {e}", exc_info=True)
            logger.error(f"[NAV_PREV] Full traceback:\n{traceback.format_exc()}")
            # Ensure navigation is marked as finished even on error
            try:
                self.finish_navigation()
            except Exception as finish_error:
                logger.error(f"[NAV_PREV] Error in finish_navigation during exception handling: {finish_error}")
            raise
        finally:
            # Additional safety: ensure navigation is always marked as finished (outer finally)
            outer_finally_time = time.time()
            logger.debug(f"[NAV_PREV] Outer finally block reached at {outer_finally_time:.3f} "
                       f"(total duration: {outer_finally_time - nav_start_time:.3f}s)")
            if hasattr(self, '_navigation_in_progress') and self._navigation_in_progress:
                logger.warning("[NAV_PREV] Navigation flag still True in outer finally, clearing it")
                self.finish_navigation()
            logger.info(f"[NAV_PREV] ========== navigate_to_previous_image() COMPLETED "
                       f"(total duration: {outer_finally_time - nav_start_time:.3f}s) ==========")

    def navigate_to_next_image(self):
        if getattr(self, '_is_delete_dialog_open', False):
            return
        import logging
        import os
        import time
        import traceback
        logger = logging.getLogger(__name__)

        nav_start_time = time.time()
        # Store navigation start time immediately for tracking total time to display
        self._last_navigation_start = nav_start_time

        logger.info(f"[NAV_NEXT] ========== navigate_to_next_image() STARTED at {nav_start_time:.3f} ==========")
        logger.debug(f"[NAV_NEXT] Current state - index: {self.current_file_index}, "
                    f"total_files: {len(self.image_files) if self.image_files else 0}, "
                    f"current_file: {os.path.basename(_require_str_path(self.current_file_path)) if self.current_file_path else 'None'}")

        # Clear any pending zoom requests for the current image
        if hasattr(self, '_pending_zoom'):
            self._pending_zoom = False
            if hasattr(self, '_pending_zoom_center'):
                delattr(self, '_pending_zoom_center')
            if hasattr(self, '_pending_zoom_thumbnail_size'):
                delattr(self, '_pending_zoom_thumbnail_size')

        # Check processor state
        processor_state = "None"
        if hasattr(self, 'current_processor') and self.current_processor:
            processor_state = f"Active (thread_id={self.current_processor.thread().currentThreadId() if hasattr(self.current_processor, 'thread') else 'N/A'})"
        logger.debug(f"[NAV_NEXT] Current processor state: {processor_state}")

        # Check if navigation is allowed BEFORE starting navigation
        if not self.can_navigate():
            logger.warning("[NAV_NEXT] Navigation BLOCKED by can_navigate() check")
            return

        # Mark navigation as started - must be done before any return statements
        self.start_navigation()

        try:
            try:
                if not self.image_files or len(self.image_files) <= 1:
                    logger.debug("Cannot navigate: no files or only one file")
                    return

                # Calculate next index with wraparound
                old_index = self.current_file_index
                if self.current_file_index >= len(self.image_files) - 1:
                    self.current_file_index = 0
                else:
                    self.current_file_index += 1
                # Editor mode: silently skip files the editor cannot open.
                self._advance_index_past_unsupported_for_editing(+1)

                logger.info(f"[NAV_NEXT] Navigating to next image - old_index: {old_index}, new_index: {self.current_file_index}")
                logger.info(f"[TRACK] User navigated to next image (arrow key) - index: {self.current_file_index}")

                # Check if new index is valid
                if self.current_file_index < 0 or self.current_file_index >= len(self.image_files):
                    logger.error(f"[NAV_NEXT] Invalid index after navigation: {self.current_file_index}, total_files: {len(self.image_files)}")
                    self.current_file_index = old_index  # Restore old index
                    return

                # Only maintain zoom state if not in fit-to-window mode
                logger.info(f"[NAV_NEXT] Checking zoom state - fit_to_window: {self.fit_to_window}, "
                           f"zoom_level: {self.current_zoom_level}, "
                           f"zoom_center_point: {self.zoom_center_point}")
                self._capture_nav_zoom_state_before_load()

                # Load the current image (at new_index)
                current_file = self.image_files[self.current_file_index]
                load_start_time = time.time()
                logger.info(f"[NAV_NEXT] Loading file at index {self.current_file_index}: {os.path.basename(current_file)}")
                logger.debug(f"[NAV_NEXT] File path: {current_file}")
                logger.debug(f"[NAV_NEXT] Time since nav start: {load_start_time - nav_start_time:.3f}s")

                try:
                    self.load_raw_image(current_file)
                    load_end_time = time.time()
                    logger.info(f"[NAV_NEXT] Successfully called load_raw_image for: {os.path.basename(current_file)} "
                               f"(took {load_end_time - load_start_time:.3f}s)")
                except Exception as load_error:
                    load_end_time = time.time()
                    logger.error(f"[NAV_NEXT] ERROR in load_raw_image during navigation (took {load_end_time - load_start_time:.3f}s): "
                               f"{load_error}", exc_info=True)
                    logger.error(f"[NAV_NEXT] Load error traceback:\n{traceback.format_exc()}")
                    # Restore previous index on error
                    self.current_file_index = old_index
                    raise

                try:
                    self.schedule_save_session_state()
                    logger.debug("[NAV_NEXT] Session state saved")
                except Exception as save_error:
                    logger.warning(f"[NAV_NEXT] Error saving session state: {save_error}")

            finally:
                # Always mark navigation as finished (inner try)
                inner_finally_time = time.time()
                logger.debug(f"[NAV_NEXT] Inner finally block reached at {inner_finally_time:.3f} "
                           f"(nav duration: {inner_finally_time - nav_start_time:.3f}s)")
                self.finish_navigation()

        except Exception as e:
            error_time = time.time()
            logger.error(f"[NAV_NEXT] ========== EXCEPTION in navigate_to_next_image "
                        f"(at {error_time:.3f}, duration: {error_time - nav_start_time:.3f}s) ==========")
            logger.error(f"[NAV_NEXT] Exception type: {type(e).__name__}, message: {e}", exc_info=True)
            logger.error(f"[NAV_NEXT] Full traceback:\n{traceback.format_exc()}")
            # Ensure navigation is marked as finished even on error
            try:
                self.finish_navigation()
            except Exception as finish_error:
                logger.error(f"[NAV_NEXT] Error in finish_navigation during exception handling: {finish_error}")
            raise
        finally:
            # Additional safety: ensure navigation is always marked as finished (outer finally)
            outer_finally_time = time.time()
            logger.debug(f"[NAV_NEXT] Outer finally block reached at {outer_finally_time:.3f} "
                       f"(total duration: {outer_finally_time - nav_start_time:.3f}s)")
            if hasattr(self, '_navigation_in_progress') and self._navigation_in_progress:
                logger.warning("[NAV_NEXT] Navigation flag still True in outer finally, clearing it")
                self.finish_navigation()
            logger.info(f"[NAV_NEXT] ========== navigate_to_next_image() COMPLETED "
                       f"(total duration: {outer_finally_time - nav_start_time:.3f}s) ==========")

    def delete_current_image(self):
        if (not self.current_file_path or not os.path.exists(self.current_file_path)):
            self.show_error("Delete Error", "No image file to delete.")
            return
        self._stop_slideshow()

        if self.confirm_deletion():
            # Only maintain zoom state if not in fit-to-window mode
            if not self.fit_to_window:
                self._preserve_nav_zoom_active = True
                self._maintain_zoom_on_navigation = True
                self._restore_zoom_center = self._zoom_anchor_for_navigation_restore()
                self._restore_zoom_level = self.current_zoom_level
                if self.current_pixmap:
                    self._restore_pixmap_size = self.current_pixmap.size()
                # Save current scroll position instead of start_scroll_x/y
                try:
                    # Ensure scroll_area exists and is valid before accessing
                    if hasattr(self, 'scroll_area') and self.scroll_area is not None:
                        h_scroll = self.scroll_area.horizontalScrollBar()
                        v_scroll = self.scroll_area.verticalScrollBar()
                        if h_scroll is not None and v_scroll is not None:
                            self._restore_start_scroll_x = h_scroll.value()
                            self._restore_start_scroll_y = v_scroll.value()
                        else:
                            self._restore_start_scroll_x = 0
                            self._restore_start_scroll_y = 0
                    else:
                        self._restore_start_scroll_x = 0
                        self._restore_start_scroll_y = 0
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Error getting scroll position in delete_current_image: {e}", exc_info=True)
                    self._restore_start_scroll_x = 0
                    self._restore_start_scroll_y = 0
            else:
                self._preserve_nav_zoom_active = False
                if hasattr(self, '_maintain_zoom_on_navigation'):
                    delattr(self, '_maintain_zoom_on_navigation')
                self._restore_zoom_center = None
                self._restore_zoom_level = None
                self._restore_start_scroll_x = None
                self._restore_start_scroll_y = None
            self.perform_deletion()
        self.schedule_save_session_state()

    def handle_post_deletion_navigation(self):
        """Handle navigation after a file has been deleted"""
        if not self.image_files:
            # Semantic / gallery search narrows ``image_files``; discarding the last hit yields an
            # empty list while the folder may still contain other files -- restore corpus + gallery.
            had_semantic_scope = bool(
                self._semantic_search_backup_files or self._semantic_search_corpus_files
            )
            if had_semantic_scope:
                self._clear_semantic_search_results(silent=True, exit_to_gallery=True)
                inp = getattr(self, "gallery_search_input", None)
                if inp is not None:
                    inp.blockSignals(True)
                    try:
                        inp.clear()
                    finally:
                        inp.blockSignals(False)
                if self.image_files:
                    self.status_bar.showMessage("Search cleared -- showing full folder", 4500)
                    self.schedule_save_session_state()
                    return

            # No more images in folder (truly empty)
            self.current_file_path = None
            self.current_file_index = -1
            self.current_pixmap = None
            self._sync_single_image_histogram()
            self.image_label.setText(
                "No more images in this folder\n\n"
                "Use File > Open to load another image"
            )
            self.status_bar.showMessage("No images remaining in folder")
            self.setWindowTitle('RAWviewer')
            # Update custom title bar
            if hasattr(self, 'title_bar') and self.title_bar is not None:
                self.title_bar.set_title('RAWviewer')
            self.update_status_bar()
            return

        # Adjust current index if needed
        if self.current_file_index >= len(self.image_files):
            self.current_file_index = len(self.image_files) - 1

        # Defer decoding the next image so rapid discard bursts can process key events /
        # coalesced timer work before kicking off heavy load_raw_image.
        self._defer_post_deletion_load_generation += 1
        gen = self._defer_post_deletion_load_generation
        QTimer.singleShot(0, lambda g=gen: self._run_deferred_post_deletion_load(g))
