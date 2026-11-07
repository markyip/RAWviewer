"""ImageLoadManager part 3: cancellation, queue compaction, cache checks, stats."""

from __future__ import annotations

import contextlib
import os
import queue
from typing import Any

from PyQt6.QtCore import QSize, QTimer

from .helpers import (
    Priority,
    _gallery_memory_thumb_acceptable,
    _skip_low_res_memory_thumb_for_display_tier,
)
from .worker import ImageLoadTask, ImageLoadWorker


class ImageLoadManagerPart3:
    def cancel_all_tasks(self):
        """取消所有任務"""
        with self._queue_lock:
            for task in self._active_tasks.values():
                task.cancel()
                # Release rather than zeroing the counter: a task cancelled
                # mid-demosaic keeps running, and blindly resetting to 0 let
                # its later _task_finished decrement a counter that by then
                # described a different set of tasks.
                self._release_raw_slot_locked(task)
            self._active_tasks.clear()
            self._task_keys_by_path.clear()
            self._reconcile_raw_slots_locked()
            self._compact_work_queue()

    def cancel_all_tasks_except(self, keep_path: str | None) -> None:
        """Cancel every queued/active task except those for ``keep_path``.

        Used when switching gallery -> single view: dozens of gallery thumbnail
        decodes (all CURRENT priority) otherwise sit ahead of the foreground
        image in the queue and saturate the worker pool, delaying its first
        render. Dropping them frees worker slots for the image being opened.
        """
        with self._queue_lock:
            keep_keys = set()
            if keep_path:
                keep_keys = set(self._task_keys_by_path.get(keep_path, ()))
            for key, task in list(self._active_tasks.items()):
                if key in keep_keys:
                    continue
                task.cancel()
                self._release_raw_slot_locked(task)
                fp = getattr(task, "file_path", None)
                self._active_tasks.pop(key, None)
                if fp and fp in self._task_keys_by_path:
                    self._task_keys_by_path[fp].discard(key)
                    if not self._task_keys_by_path[fp]:
                        self._task_keys_by_path.pop(fp, None)
            # Reconcile alone is len(_raw_slot_holders), so cancelled heavies
            # stayed in the set and kept holding the limit until their
            # demosaic finished -- the CURRENT starvation the early release
            # exists to prevent. Releasing happens per task above.
            self._reconcile_raw_slots_locked()
            self._compact_work_queue()

    def reprioritise(self, distance_for) -> int:
        """Re-score queued work against the CURRENT viewport.

        ``distance_for(file_path)`` returns rows from the viewport; 0 means
        visible. A PriorityQueue cannot re-key in place, so the queue is
        drained, re-scored and rebuilt -- the same pattern _compact_work_queue
        and the scheduler's preempt block already use. Measured at ~2 ms for a
        few hundred tasks, which is nothing beside a decode, but it must be
        driven from the scroll THROTTLE and never per wheel event: a trackpad
        emits 60-120 of those a second.

        Returns the number of tasks re-scored.
        """
        rescored = 0
        with self._queue_lock:
            drained = []
            try:
                while True:
                    drained.append(self._work_queue.get_nowait())
            except queue.Empty:
                pass
            for task in drained:
                if task.is_cancelled():
                    continue
                try:
                    task.viewport_distance = int(distance_for(task.file_path))
                except Exception:
                    task.viewport_distance = 0
                rescored += 1
                self._work_queue.put(task)
        if rescored:
            self._schedule_next_task()
        return rescored

    def cancel_distant_queued_tasks(self, distance_for, max_distance: int) -> int:
        """Drop QUEUED work further than ``max_distance`` rows from view.

        Only queued tasks: running ones hold a slot and a decode that cannot
        be interrupted. Gallery thumbnails are CURRENT priority (so they beat
        prefetch), which used to exempt them from this prune -- they then sat
        in the queue until the visible-pass cancel, starving newly visible
        tiles. Far-off gallery thumbs are dropped; single-view CURRENT is not.
        """
        cancelled = 0
        with self._queue_lock:
            drained = []
            try:
                while True:
                    drained.append(self._work_queue.get_nowait())
            except queue.Empty:
                pass
            for task in drained:
                if task.is_cancelled():
                    continue
                try:
                    far = int(distance_for(task.file_path)) > max_distance
                except Exception:
                    far = False
                drop_queued = far and (
                    task.priority != Priority.CURRENT
                    or getattr(task, "gallery_thumbnail", False)
                )
                if drop_queued:
                    task.cancel()
                    self._release_raw_slot_locked(task)
                    key = getattr(task, "task_key", None)
                    if key in self._active_tasks and self._active_tasks[key] is task:
                        del self._active_tasks[key]
                        keys = self._task_keys_by_path.get(task.file_path)
                        if keys is not None:
                            keys.discard(key)
                            if not keys:
                                self._task_keys_by_path.pop(task.file_path, None)
                    cancelled += 1
                    continue
                self._work_queue.put(task)
        return cancelled

    def _compact_work_queue(self) -> None:
        """Re-enqueue only non-cancelled tasks still waiting in the priority queue."""
        if self._work_queue.empty():
            return
        kept: list = []
        while not self._work_queue.empty():
            try:
                task = self._work_queue.get_nowait()
            except queue.Empty:
                break
            if not task.is_cancelled():
                kept.append(task)
        for task in kept:
            self._work_queue.put(task)

    def flush_queue(self):
        """Aggressively drop pending tasks to prioritize new position (used for large scroll jumps)."""
        with self._queue_lock:
            for task in self._active_tasks.values():
                task.cancel()
                self._release_raw_slot_locked(task)
            self._active_tasks.clear()
            self._task_keys_by_path.clear()
            self._work_queue = queue.PriorityQueue()
            self._reconcile_raw_slots_locked()

    def shutdown(self):
        """關閉管理器並清理資源"""
        self._stopped = True
        watchdog = getattr(self, '_dispatch_watchdog', None)
        if watchdog is not None:
            with contextlib.suppress(Exception):
                watchdog.stop()
        if hasattr(self, '_process_pool') and self._process_pool:
            import atexit
            if hasattr(self, '_process_pool_shutdown_hook'):
                with contextlib.suppress(Exception):
                    atexit.unregister(self._process_pool_shutdown_hook)
            self._process_pool.shutdown(wait=False)
        for pool_name in ("_thread_pool", "_current_thread_pool"):
            pool = getattr(self, pool_name, None)
            if pool is not None:
                with contextlib.suppress(Exception):
                    pool.clear()
                with contextlib.suppress(Exception):
                    pool.waitForDone(1000)

    def _task_finished(self, task: ImageLoadTask, *, emit: bool = True):
        """Remove completed work from the active map and keep the queue moving.

        ``emit=False`` skips the signal but still does the bookkeeping. The
        worker used to gate this ENTIRE call on _safe_emit(), so a manager
        going away mid-flight meant the heavy RAW slot was never released --
        the holder set kept it forever. Releasing a slot is not an emission
        and must not depend on one.
        """
        lock = getattr(self, "_running_tasks_lock", None)
        if lock is not None:
            with lock:
                self._running_tasks.discard(task)
        with self._queue_lock:
            key = getattr(task, 'task_key', None)
            if key in self._active_tasks and self._active_tasks[key] is task:
                del self._active_tasks[key]
                self._task_keys_by_path[task.file_path].discard(key)
                if not self._task_keys_by_path[task.file_path]:
                    self._task_keys_by_path.pop(task.file_path, None)
            # Pairs with the claim through the holder set, so a task cancelled
            # while still running cannot decrement a second time.
            slot_before = int(self._active_raw_tasks or 0)
            self._release_raw_slot_locked(task)
            # Also purge any orphaned holders that were never released (a
            # cancelled demosaic stuck in an uninterruptible call). Without this
            # the holder set can grow forever and starve every later heavy decode.
            self._reconcile_raw_slots_locked()
            slot_after = int(self._active_raw_tasks or 0)
            if slot_before != slot_after:
                import logging

                logging.getLogger(__name__).info(
                    "[LOAD] _task_finished for %s: raw slots %d -> %d",
                    getattr(task, "file_path", "<unknown>"),
                    slot_before,
                    slot_after,
                )
            inflight = getattr(self, "_demosaic_in_flight", None)
            if inflight is not None:
                inflight.discard(task)
        if not emit:
            return
        self.task_completed.emit(task.file_path)
        self._schedule_next_task()

    def _full_stage_already_in_flight(
        self, file_path: str, use_full_resolution: bool, requesting_priority: Priority
    ) -> bool:
        """True if an active (queued/running), non-cancelled task for this file
        already includes the 'full' stage with the same use_full_resolution
        flag -- see the comment at its call site in load_image().

        A CURRENT request must NOT count an owner it is about to supersede: the
        supersede block in load_image() cancels every lower-priority task for
        this file, so deferring ('full'-stripping) to a PRELOAD_/BACKGROUND full
        owner and then cancelling that same owner would drop the full decode
        entirely (regression: pressing "next" onto a prefetched neighbor would
        cancel its in-flight prefetch full decode and never re-issue one). Only
        an owner that will survive the supersede is a safe one to defer to."""
        with self._queue_lock:
            for key in self._task_keys_by_path.get(file_path, ()):
                task = self._active_tasks.get(key)
                if task is None or task.is_cancelled():
                    continue
                if key[1] != use_full_resolution:
                    continue
                if 'full' not in key[2]:
                    continue
                if (
                    requesting_priority == Priority.CURRENT
                    and task.priority.value > requesting_priority.value
                ):
                    # This CURRENT request will cancel this owner below; don't
                    # rely on it to finish the full decode.
                    continue
                return True
        return False

    def _make_task_key(self, file_path: str, use_full_resolution: bool,
                       stages: set | None, thumbnail_target_size: QSize | None,
                       thumbnail_fit: str) -> tuple:
        """Build a stable key for de-duplicating queued/running work."""
        wanted = tuple(sorted(stages if stages is not None else {'thumbnail', 'exif', 'full'}))
        if thumbnail_target_size is not None and isinstance(thumbnail_target_size, QSize) and thumbnail_target_size.isValid():
            target_key = (thumbnail_target_size.width(), thumbnail_target_size.height(), thumbnail_fit)
        else:
            target_key = None
        return (file_path, use_full_resolution, wanted, target_key)

    @staticmethod
    def _emit_cached_result_later(signal, file_path: str, payload) -> None:
        """Defer cache-hit emissions to avoid re-entering PyQt slots synchronously."""
        QTimer.singleShot(0, lambda: signal.emit(file_path, payload))

    @staticmethod
    def _emit_cached_result_now(signal, file_path: str, payload) -> None:
        """Emit cache hits immediately (load_image runs on UI thread)."""
        signal.emit(file_path, payload)

    def _check_cache(
        self,
        file_path: str,
        use_full_resolution: bool,
        stages: set | None = None,
        *,
        gallery_thumbnail: bool = False,
    ) -> bool:
        """檢查快取，如果存在則直接發送信號（只檢查記憶體快取以避免阻塞 UI）"""
        from common_image_loader import (
            image_covers_sensor_resolution,
            is_raw_file,
            use_libraw_consistent_preview_first,
        )

        is_raw = is_raw_file(file_path)
        wanted = stages if stages is not None else {'thumbnail', 'exif', 'full'}
        # Only the 'full' stage consults this, and computing it costs a
        # sidecar probe (and used to cost a per-file EXIF parse) — a gallery
        # thumbnail-only request must never pay that on the UI thread.
        libraw_first = (
            use_libraw_consistent_preview_first(file_path)
            if 'full' in wanted
            else False
        )

        cache = self._cache
        preview_only = wanted == {"thumbnail"}

        # 1) Thumbnail stage: memory preview/thumbnail (display tier for preview-first)
        if "thumbnail" in wanted:

            def _try_emit_display_thumb(buf) -> bool:
                if buf is None:
                    return False
                if not _gallery_memory_thumb_acceptable(
                    buf, gallery_thumbnail=gallery_thumbnail
                ):
                    return False
                if _skip_low_res_memory_thumb_for_display_tier(
                    buf,
                    wanted,
                    use_full_resolution,
                    gallery_thumbnail=gallery_thumbnail,
                ):
                    return False
                # Gallery tiles: defer the emit. Synchronous emission runs the
                # whole tile-apply handler inline inside the gallery's
                # scheduling loop, stacking dozens of ~50-100ms applies into one
                # main-thread stall. Single-view (non-gallery) keeps the
                # immediate emit for instant nav display.
                emit = (
                    self._emit_cached_result_now
                    if preview_only and not gallery_thumbnail
                    else self._emit_cached_result_later
                )
                emit(self.thumbnail_ready, file_path, buf)
                if preview_only:
                    exif_data = cache.get_exif_memory_only(file_path)
                    if exif_data is not None:
                        emit(self.exif_data_ready, file_path, exif_data)
                return True

            if gallery_thumbnail:
                grid_buf = cache.get_grid_memory_only(file_path)
                if _try_emit_display_thumb(grid_buf) and preview_only:
                    return True

            # Must use *_memory_only/path_key: put_* stores under _path_key(),
            # so dict gets with raw paths miss after first load and re-decode
            # on every gallery scroll pass.
            preview_buf = cache.get_preview_memory_only(file_path)
            if _try_emit_display_thumb(preview_buf) and preview_only:
                return True
            thumb = cache.get_thumbnail_memory_only(file_path)
            if _try_emit_display_thumb(thumb) and preview_only:
                return True
                # Not terminal if callers also requested EXIF/full image work.

        # 2) EXIF stage: memory-only (this whole method's contract, see docstring).
        # cache.get_exif() also validates cached RAW orientation via
        # cached_raw_exif_orientation_trustworthy(), which does a synchronous
        # rawpy.imread() when the cached record lacks verified_orientation --
        # a real, sometimes multi-second file open that was firing here on
        # every prefetch-decision call, straight on the caller's thread
        # (confirmed via faulthandler stack dumps: rawpy.imread() on the Qt
        # main thread, called from here via _on_single_view_content_displayed
        # -> _queue_display_tier_prefetch_for_paths -> load_image). Let the
        # real load path do that verification when it actually decodes the
        # file; this is just a fast "is it already warm" check.
        if 'exif' in wanted:
            exif_data = cache.get_exif_memory_only(file_path)
            if exif_data is not None and not exif_data.get("minimal_preview_exif"):
                self._emit_cached_result_later(self.exif_data_ready, file_path, exif_data)
                # Note: EXIF hit alone doesn't terminate processing if pixels are also wanted

        # 3) Full stage: treat in-memory RAW preview as "full" when use_full_resolution=False
        if 'full' in wanted:
            if is_raw:
                if use_full_resolution:
                    full_img = cache.full_image_cache.get(file_path)
                    if full_img is not None:
                        self._emit_cached_result_later(self.image_ready, file_path, full_img)
                        return True
                    exif_data = cache.exif_cache.get(file_path)
                    preview = cache.preview_cache.get(file_path)
                    if preview is not None:
                        h, w = preview.shape[:2]
                        if image_covers_sensor_resolution(w, h, exif_data):
                            self._emit_cached_result_later(self.image_ready, file_path, preview)
                            return True
                else:
                    # Prefer any LibRaw buffer already in memory (fit ↔ zoom consistency)
                    if libraw_first:
                        exif_data = cache.exif_cache.get(file_path)
                        preview = cache.preview_cache.get(file_path)
                        if preview is not None:
                            if cache.is_libraw_preview(file_path):
                                self._emit_cached_result_later(
                                    self.image_ready, file_path, preview
                                )
                                return True
                            h, w = preview.shape[:2]
                            if image_covers_sensor_resolution(w, h, exif_data):
                                self._emit_cached_result_later(
                                    self.image_ready, file_path, preview
                                )
                                return True
                        full_img = cache.get_full_image(file_path)
                        # Only a sensor-covering buffer satisfies RAW-mode full.
                        # A bare max-edge >=1024 check previously treated 1920px
                        # nav previews as done, so load_image returned without
                        # queuing demosaic and autotest sat until the 45s ceiling.
                        if full_img is not None and hasattr(full_img, "shape"):
                            fh, fw = full_img.shape[:2]
                            if image_covers_sensor_resolution(fw, fh, exif_data):
                                self._emit_cached_result_later(
                                    self.image_ready, file_path, full_img
                                )
                                return True
                    else:
                        # Preview cache is memory-only LRU (embedded JPEG path)
                        preview = cache.preview_cache.get(file_path)
                        if preview is not None:
                            self._emit_cached_result_later(self.image_ready, file_path, preview)
                            return True
                # For RAW, do not use pixmap cache here
            else:
                pixmap = cache.pixmap_cache.get(file_path)
                if pixmap is not None and not pixmap.isNull():
                    self._emit_cached_result_later(self.pixmap_ready, file_path, pixmap)
                    return True

        def _memory_thumb_ok() -> bool:
            if "thumbnail" not in wanted:
                return True
            if gallery_thumbnail:
                grid_buf = cache.grid_cache.get(file_path)
                if _gallery_memory_thumb_acceptable(
                    grid_buf, gallery_thumbnail=True
                ):
                    return True
            preview_buf = cache.preview_cache.get(file_path)
            if preview_buf is not None and _gallery_memory_thumb_acceptable(
                preview_buf, gallery_thumbnail=gallery_thumbnail
            ) and not _skip_low_res_memory_thumb_for_display_tier(
                preview_buf,
                wanted,
                use_full_resolution,
                gallery_thumbnail=gallery_thumbnail,
            ):
                return True
            thumb = cache.thumbnail_cache.get(file_path)
            return thumb is not None and _gallery_memory_thumb_acceptable(
                thumb, gallery_thumbnail=gallery_thumbnail
            ) and not _skip_low_res_memory_thumb_for_display_tier(
                thumb,
                wanted,
                use_full_resolution,
                gallery_thumbnail=gallery_thumbnail,
            )

        thumb_ok = _memory_thumb_ok()
        exif_cached = cache.get_exif_memory_only(file_path)
        if "exif" not in wanted:
            exif_ok = True
        elif preview_only:
            exif_ok = exif_cached is not None
        else:
            exif_ok = (
                exif_cached is not None
                and not exif_cached.get("minimal_preview_exif")
            )
        full_ok = "full" not in wanted
        if "full" in wanted:
            if is_raw:
                if use_full_resolution:
                    full_img = cache.full_image_cache.get(file_path)
                    if full_img is None or not hasattr(full_img, "shape"):
                        full_ok = False
                    else:
                        fh, fw = full_img.shape[:2]
                        full_ok = image_covers_sensor_resolution(
                            fw, fh, cache.exif_cache.get(file_path)
                        )
                elif libraw_first:
                    full_img = cache.get_full_image(file_path)
                    if full_img is None or not hasattr(full_img, "shape"):
                        full_ok = False
                    else:
                        fh, fw = full_img.shape[:2]
                        full_ok = image_covers_sensor_resolution(
                            fw, fh, cache.exif_cache.get(file_path)
                        ) or bool(cache.is_libraw_preview(file_path))
                else:
                    full_ok = cache.preview_cache.get(file_path) is not None
            else:
                px = cache.pixmap_cache.get(file_path)
                full_ok = px is not None and not px.isNull()

        return thumb_ok and exif_ok and full_ok

    def _schedule_next_task(self):
        """Dispatch queued work onto the CURRENT or background pool.

        Policy runs under _queue_lock; tryStart runs outside so enqueue/cancel
        are not blocked while pool threads spin up.
        """
        if self._stopped:
            return

        from common_image_loader import io_pressure_active, is_raw_file
        pressure = io_pressure_active()

        # If CURRENT full is queued but neighbor heavies hold every raw slot,
        # cancel those neighbors so the on-screen demosaic can start.
        try:
            keep_path = None
            with self._queue_lock:
                # Reconcile leaked counters (cancelled workers that never finished).
                # From the holder set: _active_tasks excludes tasks cancelled
                # while their demosaic is still running, and counting it
                # dropped their slots on the floor.
                pre_count = int(self._active_raw_tasks or 0)
                self._reconcile_raw_slots_locked()
                post_count = int(self._active_raw_tasks or 0)
                if pre_count != post_count:
                    import logging

                    logging.getLogger(__name__).warning(
                        "[LOAD] Reconciled raw slots: %d -> %d (purged %d orphaned holders)",
                        pre_count,
                        post_count,
                        pre_count - post_count,
                    )
                if int(self._active_raw_tasks or 0) > 0:
                    tmp = []
                    while not self._work_queue.empty():
                        t = self._work_queue.get_nowait()
                        tmp.append(t)
                        if (
                            keep_path is None
                            and not t.is_cancelled()
                            and t.priority == Priority.CURRENT
                            and is_raw_file(t.file_path)
                            and "full" in (t.stages or set())
                        ):
                            keep_path = t.file_path
                    for t in tmp:
                        self._work_queue.put(t)
            if keep_path:
                n = self.cancel_non_current_heavy_raw_except(keep_path)
                if n:
                    import logging

                    logging.getLogger(__name__).info(
                        "[LOAD] Preempted %d neighbor heavy RAW task(s) for CURRENT full (%s)",
                        n,
                        os.path.basename(keep_path),
                    )
        except Exception:
            pass

        to_start: list = []  # (task, pool, is_heavy)
        deferred_put: list = []

        with self._queue_lock:
            if self._work_queue.empty():
                return
            try:
                # Provisional heavy slots for this scan while we still hold the lock.
                provisional_raw = int(self._active_raw_tasks or 0)
                while not self._work_queue.empty():
                    task = self._work_queue.get_nowait()

                    if task.is_cancelled():
                        key = getattr(task, 'task_key', None)
                        if key in self._active_tasks and self._active_tasks[key] is task:
                            del self._active_tasks[key]
                            self._task_keys_by_path[task.file_path].discard(key)
                            if not self._task_keys_by_path[task.file_path]:
                                self._task_keys_by_path.pop(task.file_path, None)
                        continue

                    pool = self._pool_for_task(task)

                    if pressure and task.priority != Priority.CURRENT:
                        active_prefetch = sum(
                            1 for t in self._active_tasks.values()
                            if getattr(t, "priority", None) != Priority.CURRENT and not t.is_cancelled()
                        )
                        if active_prefetch >= 1:
                            deferred_put.append(task)
                            continue

                    is_raw = is_raw_file(task.file_path)
                    is_heavy = is_raw and 'full' in (task.stages or set())

                    heavy_limit = self._raw_load_limit
                    if getattr(self, "_gallery_view_active", False):
                        # Gallery only needs thumbnails; don't let the GPU
                        # demosaic alignment (raw_limit=1 on MPS) serialize the
                        # shared pool while tiles are filling. Read-time
                        # override so warmup/indexing save/restore of
                        # _raw_load_limit cannot pin the GPU value.
                        pre_gpu = getattr(self, "_pre_gpu_raw_limit", None)
                        if pre_gpu:
                            heavy_limit = max(heavy_limit, int(pre_gpu))
                    if is_heavy and task.priority != Priority.CURRENT:
                        # Only reserve while CURRENT full is waiting or already
                        # occupying a heavy slot. Unconditional reserve at
                        # heavy_limit=1 would permanently block neighbor full
                        # (0 >= 0) after the on-screen decode finishes.
                        current_heavy_pending = False
                        for t in self._active_tasks.values():
                            if (
                                t is not None
                                and not t.is_cancelled()
                                and t.priority == Priority.CURRENT
                                and is_raw_file(t.file_path)
                                and "full" in (t.stages or set())
                            ):
                                current_heavy_pending = True
                                break
                        if not current_heavy_pending:
                            for deferred in deferred_put:
                                if (
                                    not deferred.is_cancelled()
                                    and deferred.priority == Priority.CURRENT
                                    and is_raw_file(deferred.file_path)
                                    and "full" in (deferred.stages or set())
                                ):
                                    current_heavy_pending = True
                                    break
                        if current_heavy_pending:
                            # Leave the last heavy slot for CURRENT. At
                            # heavy_limit=1 this becomes 0 so PRELOAD/BACKGROUND
                            # full cannot start until CURRENT's demosaic begins.
                            heavy_limit = max(0, int(heavy_limit) - 1)
                    if is_heavy and (
                        provisional_raw >= heavy_limit
                        or len(self._demosaic_in_flight) >= max(1, int(heavy_limit) or 1)
                    ):
                        deferred_put.append(task)
                        continue

                    if is_heavy:
                        # Claim at admission, under the lock, so a cancelled
                        # mid-demosaic task cannot be forgotten by a later
                        # recount of _active_tasks.
                        provisional_raw += 1
                        self._claim_raw_slot_locked(task)
                    to_start.append((task, pool, is_heavy))
            except queue.Empty:
                pass

            for deferred in deferred_put:
                self._work_queue.put(deferred)

        # Start outside the queue lock.
        requeue: list = []
        for task, pool, is_heavy in to_start:
            if self._stopped or task.is_cancelled():
                with self._queue_lock:
                    # The slot was claimed at admission. This task will never
                    # run, so _task_finished will never release it -- without
                    # this the holder set pins it forever and, at
                    # heavy_limit=1, every later full RAW decode is blocked.
                    if is_heavy:
                        self._release_raw_slot_locked(task)
                    key = getattr(task, "task_key", None)
                    if key in self._active_tasks and self._active_tasks[key] is task:
                        del self._active_tasks[key]
                        self._task_keys_by_path[task.file_path].discard(key)
                        if not self._task_keys_by_path[task.file_path]:
                            self._task_keys_by_path.pop(task.file_path, None)
                continue
            worker = ImageLoadWorker(task, self)
            if not pool.tryStart(worker):
                # Give the slot back: it was claimed at admission time.
                if is_heavy:
                    with self._queue_lock:
                        self._release_raw_slot_locked(task)
                requeue.append(task)
                continue
            if is_heavy:
                with self._queue_lock:
                    self._demosaic_in_flight.add(task)

        if requeue:
            with self._queue_lock:
                for task in requeue:
                    if not task.is_cancelled():
                        self._work_queue.put(task)

    def get_stats(self) -> dict[str, Any]:
        """獲取管理器統計信息"""
        with self._queue_lock:
            return {
                'queue_size': self._work_queue.qsize(),
                'active_tasks': len(self._active_tasks),
                'running_tasks': len(self._running_tasks),
                'active_threads': self._thread_pool.activeThreadCount(),
                'max_threads': self._thread_pool.maxThreadCount(),
                'current_active_threads': self._current_thread_pool.activeThreadCount(),
                'current_max_threads': self._current_thread_pool.maxThreadCount(),
            }

