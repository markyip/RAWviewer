"""settings mixin, part 1 of 1 (21 methods, 474 lines)."""
from __future__ import annotations

import os
import sys

from PyQt6.QtCore import QSettings, QTimer

from core.config import _env_int, _env_true, _norm_path

from .. import lazy
from ..lazy import check_memory_cache_for_image, is_raw_file

lazy.register(sys.modules[__name__])


class SettingsPart1:
    def _load_burst_grouping_preference(self) -> None:
        try:
            from core.burst_grouping import burst_grouping_enabled_from_settings

            self._burst_grouping_enabled = burst_grouping_enabled_from_settings(
                self.get_settings()
            )
        except Exception:
            self._burst_grouping_enabled = False

    def _burst_rejected_settings_key(self, folder: str) -> str:
        return f"burst_rejected/{_norm_path(folder)}"

    def _filmstrip_prefetch_radius(self) -> int:
        try:
            return max(
                8,
                int(os.environ.get("RAWVIEWER_FILMSTRIP_PREFETCH_RADIUS", "28").strip()),
            )
        except (TypeError, ValueError):
            return 28

    def _prefetch_filmstrip_thumbnails(self) -> None:
        """Warm filmstrip thumbnails around the current image (before hover reveal)."""
        if getattr(self, "view_mode", "single") != "single":
            return

        # Defer filmstrip prefetch until the first image is actually on screen.
        if not getattr(self, "_single_view_first_render_logged", False):
            return

        # Hold off while the user is actively navigating -- retry once the quiet window passes.
        if self._in_nav_quiet_window():
            QTimer.singleShot(250, self._prefetch_filmstrip_thumbnails)
            return

        files = self._navigation_files()
        if len(files) <= 1:
            return
        idx = getattr(self, "current_file_index", -1)
        if idx < 0:
            return

        radius = self._filmstrip_prefetch_radius()
        lo = max(0, idx - radius)
        hi = min(len(files), idx + radius + 1)
        current = getattr(self, "current_file_path", None)
        # Order by distance from current so the nearest thumbnails warm first, then
        # hand off to the chunked enqueuer instead of dumping the whole radius at once.
        order = sorted(range(lo, hi), key=lambda i: abs(i - idx))
        pending = []
        for i in order:
            path = files[i]
            if current and _norm_path(path) == _norm_path(current):
                continue
            # Memory-only check: this loop is a prefetch *decision*, not a
            # display request, so it must not itself pay for a disk-tier
            # decode (see has_warm_thumbnail_in_memory's docstring).
            if self.image_cache.has_warm_thumbnail_in_memory(path):
                continue
            pending.append(path)
        self._enqueue_filmstrip_thumbnails_chunked(pending)

    def _pause_prefetch_for_navigation(self) -> None:
        """Drop queued low-priority neighbor/filmstrip prefetch on user navigation.

        Those tasks belong to the previous position and otherwise sit ahead of the new
        foreground load in the worker pool (and flood the main-thread signal queue),
        delaying the image the user asked for. Neighbors re-queue after the new image paints.
        """
        # Stop dripping filmstrip thumbnails; the new position will reseed the queue.
        self._filmstrip_prefetch_pending = []
        self._filmstrip_prefetch_draining = False
        try:
            lm = getattr(self, "image_manager", None)
            if lm is None:
                return
            from image_load_manager import Priority
            if hasattr(lm, "cancel_tasks_by_priority"):
                lm.cancel_tasks_by_priority(Priority.PRELOAD_NEXT)
                lm.cancel_tasks_by_priority(Priority.BACKGROUND)
        except Exception:
            pass

    def _drain_filmstrip_prefetch_chunk(self) -> None:
        pending = getattr(self, "_filmstrip_prefetch_pending", None)
        if not pending:
            self._filmstrip_prefetch_draining = False
            return
        # Yield to the user -- don't compete for I/O while a navigation is in flight.
        if self._in_nav_quiet_window():
            QTimer.singleShot(200, self._drain_filmstrip_prefetch_chunk)
            return
        from image_load_manager import Priority
        chunk = _env_int("RAWVIEWER_FILMSTRIP_FLUSH_CHUNK", 4, minimum=1)
        interval_ms = _env_int("RAWVIEWER_FILMSTRIP_FLUSH_INTERVAL_MS", 180, minimum=0)
        current = getattr(self, "current_file_path", None)
        sent = 0
        while pending and sent < chunk:
            path = pending.pop(0)
            if current and _norm_path(path) == _norm_path(current):
                continue
            # Memory-only check -- see has_warm_thumbnail_in_memory's docstring.
            if self.image_cache.has_warm_thumbnail_in_memory(path):
                continue
            # BACKGROUND priority so neighbor display prefetch (PRELOAD_NEXT) wins.
            # gallery_thumbnail=True: film strip now targets the same grid tier
            # (~512px) as gallery tiles, so the two share decodes instead of one
            # producing a 256px thumbnail-tier image the other cannot reuse.
            self.image_manager.load_image(
                file_path=path,
                priority=Priority.BACKGROUND,
                cancel_existing=False,
                use_full_resolution=False,
                stages={"thumbnail"},
                gallery_thumbnail=True,
            )
            sent += 1
        if pending:
            QTimer.singleShot(interval_ms, self._drain_filmstrip_prefetch_chunk)
        else:
            self._filmstrip_prefetch_draining = False

    def get_settings(self):
        return QSettings("RAWviewer", "RAWviewer")

    def _gallery_bookmarks_settings_key(self, folder: str) -> str:
        import hashlib

        norm = os.path.normcase(os.path.normpath(folder or ""))
        digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:24]
        return f"gallery_bookmarks/{digest}"

    def get_sort_preference(self):
        """Get user's preferred sorting method - Newest (True) or Oldest (False)"""
        settings = self.get_settings()
        # Default: oldest images first (capture time ascending)
        return settings.value("sort_by_newest", False, type=bool)

    def _cancel_filmstrip_prefetch_for_gallery(self) -> None:
        """Stop filmstrip/neighbor prefetch before gallery thumbnail loads."""
        self._filmstrip_prefetch_pending = []
        self._filmstrip_prefetch_draining = False
        if hasattr(self, "_deferred_filmstrip_paths") and self._deferred_filmstrip_paths:
            self._deferred_filmstrip_paths.clear()
        self._pause_prefetch_for_navigation()

    def _macos_editing_app_settings_key(self) -> str:
        return "externalEditor/macosAppPath"

    def _windows_editing_app_settings_key(self) -> str:
        return "externalEditor/windowsExePath"

    def _needs_display_buffer_prefetch(self, file_path: str, *, zoomed: bool) -> bool:
        # Failure memo: a file whose load keeps failing (e.g. undecodable NEF)
        # is never cached, so without this the idle prefetch loop re-requests
        # it every tick (~450ms) and each failure spams an error signal.
        fail_ts = getattr(self, "_prefetch_failure_ts", None)
        if fail_ts is not None:
            import time as _time

            ts = fail_ts.get(os.path.normcase(os.path.abspath(file_path)))
            if ts is not None and (_time.monotonic() - ts) < 300.0:
                return False
        min_dim = 2800 if zoomed else self._single_view_cache_min_dim(for_navigation=False)
        if self._display_quality_buffer_cached(file_path, min_dim=min_dim):
            return False
        cached_item, _cache_type = check_memory_cache_for_image(
            file_path, use_full_resolution=zoomed
        )
        return cached_item is None

    def _nav_prefetch_stages_for_path(
        self, file_path: str, *, zoomed: bool, near: bool = True
    ) -> tuple[set[str], bool]:
        """Stages for warming the next/prev image before navigation.

        Non-zoomed (fit-to-window) neighbors only get the light nav-preview
        tier -- the same tier the current image already shows on first
        display. Sensor-resolution full decode is deferred to
        _schedule_idle_full_decode_after_nav_preview once the user actually
        lands on and pauses on that image, exactly like the currently
        displayed image already works. This used to eagerly request
        stages={"full"} whenever use_libraw_consistent_preview_first() was on
        (the default), meaning every arrow-key press queued full-sensor
        LibRaw decodes for the next 2 neighbors regardless of whether the
        user paused anywhere -- measured as a "Full-resolution pixels loaded"
        line after nearly every navigation in a real browsing session, the
        exact waste PR-5 (thumbnail-cache-unification-plan.md) targets.
        Zoomed navigation keeps eager full: a user already at 100% zoom
        expects each new image at full resolution immediately.

        Exception, distinct from the reverted behavior above: when the
        embedded-JPEG workflow is on (use_full_embedded_raw_preview), the
        "full" stage's own first move is to return the camera's embedded
        JPEG if it covers sensor resolution (~38ms, processing.unified
        _try_full_embedded_raw_preview) -- only falling through to a LibRaw
        decode for images where that preview doesn't cover sensor res. That
        makes eager "full" cheap in the common case, unlike the LibRaw-first
        mode PR-5 was fixing, so near neighbors (within
        _nav_preload_display_near_count) get it eagerly instead of waiting
        for idle. Kept off for the wider background/idle radius: a "full"
        request still occupies a RAW-decode concurrency slot regardless of
        which branch it actually takes (image_load_manager's heavy-task
        throttle keys on 'full' in stages, not on measured cost), so this
        stays scoped to near neighbors rather than the full preload radius.

        (Previously reverted once after a hang was observed on a large
        folder; root-caused to an unrelated get_multiple_exif bug -- a
        synchronous rawpy.imread() per file on the main thread, fixed
        separately -- and confirmed not caused by this eager-full logic
        before reinstating it.)

        Second exception to the "full is cheap" assumption: RAF (Fujifilm)
        and 3FR (Hasselblad) camera firmware embeds a fixed-size preview that
        essentially never covers sensor resolution, verified against 36 real
        sample files -- so for these two formats "full" is *not* the ~38ms
        embedded-JPEG path, it's the real LibRaw demosaic (~20-35s on a
        100MP RAF, per nav_autotest.py's documented ceiling probing). Skip
        the eager add for them; idle/on-pause decode still gets there
        eventually, just not fired off automatically for every neighbor
        during ordinary arrow-key browsing. See
        common_image_loader.embedded_preview_rarely_covers_sensor for the
        full rationale.
        """
        if zoomed:
            return {"exif", "full"}, True
        stages = {"thumbnail"}
        if not self.image_cache.get_exif(file_path):
            stages = {"thumbnail", "exif"}
        if near:
            from common_image_loader import (
                embedded_preview_rarely_covers_sensor,
                use_full_embedded_raw_preview,
                use_libraw_consistent_preview_first,
            )

            if use_full_embedded_raw_preview() and not embedded_preview_rarely_covers_sensor(
                file_path
            ):
                stages = stages | {"full"}
            elif (
                use_libraw_consistent_preview_first(file_path)
                and not embedded_preview_rarely_covers_sensor(file_path)
            ):
                # RAW High Quality + GPU: warm ±1 sensor demosaic once CURRENT
                # full is already in flight (urgent prefetch gates on that).
                # Slot reservation keeps neighbors from starving the on-screen
                # decode; when it finishes the next demosaic starts immediately.
                try:
                    from processing.raw_decode.fast import prefer_gpu_decode_enabled

                    if prefer_gpu_decode_enabled():
                        stages = stages | {"full"}
                except Exception:
                    pass
        return stages, False

    def _request_display_buffer_prefetch(
        self,
        file_path: str,
        *,
        priority,
        zoomed: bool,
        near: bool = True,
    ) -> None:
        if not self._needs_display_buffer_prefetch(file_path, zoomed=zoomed):
            return
        stages, use_full = self._nav_prefetch_stages_for_path(
            file_path, zoomed=zoomed, near=near
        )
        self.image_manager.load_image(
            file_path=file_path,
            priority=priority,
            cancel_existing=False,
            use_full_resolution=use_full,
            stages=stages,
        )

    def _paths_within_nav_prefetch_radius(
        self, center_index: int | None = None
    ) -> list[str]:
        if not self.image_files or self.current_file_index < 0:
            return []
        idx = self.current_file_index if center_index is None else center_index
        if idx < 0 or idx >= len(self.image_files):
            return []
        radius = self._nav_preload_display_radius()
        n = len(self.image_files)
        out: list[str] = []
        seen = set()
        for step in range(1, radius + 1):
            for j in ((idx + step) % n, (idx - step) % n):
                key = _norm_path(self.image_files[j])
                if key not in seen:
                    seen.add(key)
                    out.append(self.image_files[j])
        return out

    def _schedule_post_session_restore_prefetch(self) -> None:
        """Staged heavy loads after session-restore TTFR: full decode, then nav preload."""
        if not self._should_defer_nav_preload():
            return
        if not getattr(self, "_single_view_first_render_logged", False):
            return
        if getattr(self, "_session_restore_staged_scheduled", False):
            return
        self._session_restore_staged_scheduled = True
        full_delay_ms = _env_int(
            # Was 2500ms: kept USB quiet after TTFR but made session restore
            # feel unfinished for ~3s+. ±1 neighbors already warm at TTFR
            # (_queue_urgent_neighbor_prefetch); a shorter pause is enough for
            # the first paint to settle before sensor full + wider preload.
            "RAWVIEWER_SESSION_RESTORE_FULL_DECODE_DELAY_MS", 800, minimum=200
        )
        QTimer.singleShot(full_delay_ms, self._run_post_session_restore_full_decode)

    def _queue_urgent_neighbor_prefetch(self) -> None:
        """Warm ±1 display-tier buffers ASAP (bypasses session-restore defer).

        Wider radius preload stays gated by ``_should_defer_nav_preload`` so
        TTFR is not starved; the immediate next/prev images are what make the
        first arrow-key press feel instant.

        When the current RAW still needs a sensor-covering full decode, defer
        this until that full is queued/in-flight -- otherwise neighbor ``full``
        PRELOAD tasks can occupy the only GPU raw_limit slot and the current
        image never reaches stages=['full'].
        """
        flag = os.environ.get("RAWVIEWER_NAV_PRELOAD_DISPLAY", "1").strip().lower()
        if flag not in ("1", "true", "yes", "on"):
            return
        if not getattr(self, "_single_view_first_render_logged", False):
            return
        if getattr(self, "view_mode", "single") != "single":
            return
        if not self.image_files or self.current_file_index < 0:
            return
        if len(self.image_files) <= 1:
            return
        cur = getattr(self, "current_file_path", None)
        if cur and self._needs_full_resolution_upgrade(cur):
            try:
                if not self.image_manager.has_active_full_stage_for_path(cur):
                    # Ensure CURRENT full is queued before neighbor PRELOAD full
                    # can occupy the only GPU raw_limit slot.
                    self._flush_deferred_sensor_full_decode(cur)
                    from PyQt6.QtCore import QTimer

                    QTimer.singleShot(200, self._queue_urgent_neighbor_prefetch)
                    return
            except Exception:
                pass
        from image_load_manager import Priority

        zoomed = self._single_view_is_zoomed_in()
        n = len(self.image_files)
        idx = self.current_file_index
        next_path = self.image_files[(idx + 1) % n]
        prev_path = self.image_files[(idx - 1) % n]
        import logging

        logging.getLogger(__name__).info(
            "[PRELOAD] Urgent ±1 neighbor display prefetch (%s, %s)",
            os.path.basename(next_path),
            os.path.basename(prev_path),
        )
        self._request_display_buffer_prefetch(
            next_path, priority=Priority.PRELOAD_NEXT, zoomed=zoomed, near=True
        )
        self._request_display_buffer_prefetch(
            prev_path, priority=Priority.PRELOAD_PREV, zoomed=zoomed, near=True
        )

    def _queue_display_tier_prefetch_for_paths(self, paths) -> None:
        """Background embedded-preview decode for navigation neighbors (not sensor full)."""
        if not getattr(self, "_single_view_first_render_logged", False):
            return

        if self._should_defer_nav_preload():
            return
        from image_load_manager import Priority

        for path in paths or []:
            if not path:
                continue
            try:
                if not is_raw_file(path):
                    continue
            except Exception:
                continue
            if self._display_quality_buffer_cached(
                path, min_dim=self._single_view_cache_min_dim(for_navigation=False)
            ):
                continue
            if self.image_manager.has_active_work_for_path(path):
                continue
            stages, use_full = self._nav_prefetch_stages_for_path(path, zoomed=False)
            self.image_manager.load_image(
                path,
                priority=Priority.BACKGROUND,
                cancel_existing=False,
                use_full_resolution=use_full,
                stages=stages,
            )

    def _schedule_idle_display_prefetch(self) -> None:
        if not _env_true("RAWVIEWER_IDLE_DISPLAY_PREFETCH", default=True):
            return
        if getattr(self, "view_mode", "single") != "single":
            return
        timer = getattr(self, "_idle_display_prefetch_timer", None)
        if timer is None:
            return
        delay_ms = _env_int("RAWVIEWER_IDLE_DISPLAY_PREFETCH_DELAY_MS", 650, minimum=100)
        timer.start(delay_ms)

    def _idle_display_prefetch_tick(self) -> None:
        """Background prefetch of display-quality buffers for upcoming navigation targets."""
        from image_load_manager import Priority

        if not _env_true("RAWVIEWER_IDLE_DISPLAY_PREFETCH", default=True):
            return
        if getattr(self, "view_mode", "single") != "single":
            return
        if not self.image_files or self.current_file_index < 0:
            return
        if getattr(self, "_semantic_index_prep_in_progress", False):
            self._schedule_idle_display_prefetch()
            return
        if getattr(self, "_semantic_indexing_in_progress", False):
            return
        if getattr(self, "_metadata_index_prep_complete", False) and not getattr(
            self, "_metadata_index_run_done", False
        ):
            self._schedule_idle_display_prefetch()
            return
        try:
            stats = self.image_manager.get_stats()
            queue_cap = _env_int("RAWVIEWER_IDLE_DISPLAY_PREFETCH_QUEUE_CAP", 10, minimum=2)
            if stats.get("queue_size", 0) >= queue_cap:
                self._schedule_idle_display_prefetch()
                return
        except Exception:
            pass

        zoomed = not getattr(self, "fit_to_window", True)
        radius = _env_int("RAWVIEWER_IDLE_DISPLAY_PREFETCH_RADIUS", 6, minimum=2)
        batch = _env_int("RAWVIEWER_IDLE_DISPLAY_PREFETCH_BATCH", 2, minimum=1)
        n = len(self.image_files)
        base = self.current_file_index
        candidates: list[tuple[int, str]] = []
        for step in range(1, radius + 1):
            for idx in (
                (base + step) % n,
                (base - step) % n,
            ):
                path = self.image_files[idx]
                if _norm_path(path) == _norm_path(getattr(self, "current_file_path", "")):
                    continue
                candidates.append((idx, path))

        if not candidates:
            return

        cursor = int(getattr(self, "_idle_display_prefetch_cursor", 0)) % len(candidates)
        loaded = 0
        for _ in range(len(candidates)):
            if loaded >= batch:
                break
            idx, path = candidates[cursor]
            cursor = (cursor + 1) % len(candidates)
            if not self._needs_display_buffer_prefetch(path, zoomed=zoomed):
                continue
            step_dist = min(abs(idx - base), n - abs(idx - base))
            priority = (
                Priority.PRELOAD_NEXT
                if step_dist == 1 and idx == (base + 1) % n
                else Priority.PRELOAD_PREV
                if step_dist == 1 and idx == (base - 1) % n
                else Priority.BACKGROUND
            )
            near = step_dist <= self._nav_preload_display_near_count()
            self._request_display_buffer_prefetch(
                path, priority=priority, zoomed=zoomed, near=near
            )
            loaded += 1

        self._idle_display_prefetch_cursor = cursor
        if loaded > 0 or any(
            self._needs_display_buffer_prefetch(p, zoomed=zoomed) for _, p in candidates
        ):
            interval_ms = _env_int("RAWVIEWER_IDLE_DISPLAY_PREFETCH_INTERVAL_MS", 450, minimum=150)
            self._idle_display_prefetch_timer.start(interval_ms)
