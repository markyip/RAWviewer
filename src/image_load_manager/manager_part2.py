"""ImageLoadManager part 2: load scheduling, task registry, cancellation."""

from __future__ import annotations

import contextlib
import os
import queue
import time

from PyQt6.QtCore import QSize

from .helpers import Priority
from .worker import ImageLoadTask


class ImageLoadManagerPart2:
    def load_image(self, file_path: str, priority: Priority = Priority.CURRENT,
                   cancel_existing: bool = True, use_full_resolution: bool = False,
                   stages: set | None = None,
                   thumbnail_target_size: QSize | None = None,
                   thumbnail_fit: str = "crop",
                   bypass_cache: bool = False,
                   gallery_thumbnail: bool = False):
        """請求加載圖像"""
        # Don't accept new tasks if stopped
        if self._stopped:
            return

        _li_debug = os.environ.get("RAWVIEWER_SCROLL_PERF_DEBUG", "").strip() == "1"
        _li_t0 = time.perf_counter() if _li_debug else 0.0

        # 取消現有任務（如果需要）
        if cancel_existing:
            self.cancel_task(file_path)
        _li_t_cancel = time.perf_counter() if _li_debug else 0.0

        # 檢查快取（memory-only, stage-aware）
        if not bypass_cache and self._check_cache(
            file_path,
            use_full_resolution,
            stages=stages,
            gallery_thumbnail=gallery_thumbnail,
        ):
            return
        _li_t_cache = time.perf_counter() if _li_debug else 0.0

        # Dedup against an already in-flight 'full' decode for this exact file,
        # even if it was queued under a different stage set. _make_task_key()
        # keys on the *exact* stage set, so e.g. stages={'full'} (queued by
        # _maybe_queue_background_full_decode) and stages={'exif','full',
        # 'thumbnail'} (queued by the initial navigation load) for the same
        # file don't naturally dedupe against each other -- both would run a
        # full LibRaw/embedded-JPEG decode of the same image concurrently.
        # Strip 'full' from this request if another active task already owns
        # it; the in-flight task will populate the same full-image cache this
        # request would have populated. `priority` is passed so a CURRENT
        # request won't defer to an owner it is about to cancel in the
        # supersede block below (see _full_stage_already_in_flight).
        effective_stages = set(stages) if stages is not None else {'thumbnail', 'exif', 'full'}
        if 'full' in effective_stages and self._full_stage_already_in_flight(
            file_path, use_full_resolution, priority
        ):
            effective_stages.discard('full')
            if not effective_stages:
                return
            stages = effective_stages

        task_key = self._make_task_key(
            file_path,
            use_full_resolution,
            stages,
            thumbnail_target_size,
            thumbnail_fit,
        )
        with self._queue_lock:
            existing = self._active_tasks.get(task_key)
            if existing and not existing.is_cancelled():
                # Same work is already queued/running. Avoid filling the queue with duplicates.
                return

            if priority == Priority.CURRENT:
                # A visible tile should supersede lower-priority prefetch work for the same file.
                for key, task in list(self._active_tasks.items()):
                    if key and key[0] == file_path and task.priority.value > priority.value:
                        task.cancel()
                        del self._active_tasks[key]
                        self._task_keys_by_path[file_path].discard(key)
                        self._release_raw_slot_locked(task)
                if not self._task_keys_by_path[file_path]:
                    self._task_keys_by_path.pop(file_path, None)

        # 創建任務
        task = ImageLoadTask(
            file_path,
            priority,
            use_full_resolution,
            stages=stages,
            thumbnail_target_size=thumbnail_target_size,
            thumbnail_fit=thumbnail_fit,
            gallery_thumbnail=gallery_thumbnail,
        )
        task.task_key = task_key
        task._enqueue_ts = time.time()
        with self._queue_lock:
            existing = self._active_tasks.get(task_key)
            if existing and not existing.is_cancelled():
                return
            self._active_tasks[task_key] = task
            self._task_keys_by_path[file_path].add(task_key)

        # 添加到工作隊列
        _li_t_queue = time.perf_counter() if _li_debug else 0.0
        self._work_queue.put(task)
        self._schedule_next_task()
        if _li_debug:
            _li_end = time.perf_counter()
            total_ms = (_li_end - _li_t0) * 1000.0
            if total_ms > 50.0:
                import logging

                logging.getLogger(__name__).info(
                    "[SCROLL_PERF] load_image segments: cancel=%.0fms cache=%.0fms "
                    "keying=%.0fms schedule=%.0fms total=%.0fms",
                    (_li_t_cancel - _li_t0) * 1000.0,
                    (_li_t_cache - _li_t_cancel) * 1000.0,
                    (_li_t_queue - _li_t_cache) * 1000.0,
                    (_li_end - _li_t_queue) * 1000.0,
                    total_ms,
                )

    def has_active_work_for_path(self, file_path: str) -> bool:
        """True when a load task for this path is queued or running."""
        if not file_path:
            return False
        with self._queue_lock:
            return bool(self._task_keys_by_path.get(file_path))

    def has_active_full_stage_for_path(self, file_path: str) -> bool:
        """True when a non-cancelled task for this path includes ``full`` stage."""
        if not file_path:
            return False
        with self._queue_lock:
            for key in self._task_keys_by_path.get(file_path, ()):
                task = self._active_tasks.get(key)
                if (
                    task is not None
                    and not task.is_cancelled()
                    and "full" in (task.stages or set())
                ):
                    return True
        return False

    def has_current_priority_work(self, file_path: str) -> bool:
        """True when a CURRENT-priority task for this path is queued or running."""
        if not file_path:
            return False
        with self._queue_lock:
            for key in self._task_keys_by_path.get(file_path, ()):
                task = self._active_tasks.get(key)
                if (
                    task is not None
                    and task.priority == Priority.CURRENT
                    and not task.is_cancelled()
                ):
                    return True
        return False

    def cancel_non_current_heavy_raw_except(
        self, keep_path: str | None = None
    ) -> int:
        """Cancel PRELOAD/BACKGROUND RAW ``full`` tasks (free GPU/raw heavy slots).

        Used when the on-screen CURRENT image needs a sensor demosaic but the
        only heavy slot is held by neighbor prefetch. Returns how many tasks
        were cancelled.
        """
        from common_image_loader import is_raw_file

        keep_norm = None
        if keep_path:
            try:
                keep_norm = os.path.normcase(os.path.abspath(keep_path))
            except Exception:
                keep_norm = keep_path
        cancelled = 0
        with self._queue_lock:
            for key, task in list(self._active_tasks.items()):
                if task is None or task.is_cancelled():
                    continue
                if task.priority == Priority.CURRENT:
                    continue
                if "full" not in (task.stages or set()):
                    continue
                fp = getattr(task, "file_path", None) or ""
                if not is_raw_file(fp):
                    continue
                if keep_norm:
                    try:
                        if os.path.normcase(os.path.abspath(fp)) == keep_norm:
                            continue
                    except Exception:
                        if fp == keep_path:
                            continue
                task.cancel()
                self._active_tasks.pop(key, None)
                if fp and fp in self._task_keys_by_path:
                    self._task_keys_by_path[fp].discard(key)
                    if not self._task_keys_by_path[fp]:
                        self._task_keys_by_path.pop(fp, None)
                self._release_raw_slot_locked(task)
                cancelled += 1
            if cancelled:
                self._compact_work_queue()
        return cancelled

    def _claim_raw_slot_locked(self, task) -> None:
        """Record that ``task`` holds a heavy RAW slot. Under ``_queue_lock``.

        The holder SET is the source of truth, not the task's flag and not
        membership of _active_tasks. Slots used to be reconciled by counting
        _active_tasks, which silently forgot any task that had been cancelled
        while its demosaic was still running -- the counter dropped, new
        heavies were admitted, and then the old ones finished and decremented
        again, taking it below the real concurrency. On heavy_limit=1 that is
        exactly the thrash the limit exists to prevent.
        """
        if task is None or getattr(task, "_counted_raw_slot", False):
            return
        task._counted_raw_slot = True
        self._raw_slot_holders.add(task)
        self._active_raw_tasks = len(self._raw_slot_holders)

    def _reconcile_raw_slots_locked(self) -> None:
        """Recount from the holder set, purging orphaned holders. Under ``_queue_lock``.

        A holder that is no longer in ``_active_tasks`` (e.g. cancelled while
        its demosaic was stuck in an uninterruptible GPU/CPU call, or
        re-queued under a new task object) would otherwise keep its slot
        forever. ``_task_finished`` only releases slots for tasks that actually
        finish, so this purge is the only thing that unblocks every later heavy
        decode once a slot has leaked.
        """
        holders = getattr(self, "_raw_slot_holders", None)
        if holders is None:
            self._raw_slot_holders = set()
            holders = self._raw_slot_holders
        orphaned = set()
        _stale_threshold_s = 60.0
        _now = time.time()
        for t in list(holders):
            if t is None:
                orphaned.add(t)
                continue
            try:
                if getattr(t, "is_cancelled", lambda: False)():
                    orphaned.add(t)
                    continue
                # Keep only holders still tracked as active tasks.
                still_active = False
                for active in self._active_tasks.values():
                    if active is t:
                        still_active = True
                        break
                if not still_active:
                    orphaned.add(t)
                    continue
                # Stale-holder purge: a task that has been holding a slot for
                # far too long is almost certainly stuck in an uninterruptible
                # decode. Purge it (and cancel) so later heavies are not starved
                # forever. _task_finished would have released it on completion.
                enq = getattr(t, "_enqueue_ts", None)
                if enq is not None and (_now - float(enq)) > _stale_threshold_s:
                    with contextlib.suppress(Exception):
                        t.cancel()
                    orphaned.add(t)
            except Exception:
                orphaned.add(t)
        for t in orphaned:
            holders.discard(t)
            try:
                if getattr(t, "_counted_raw_slot", False):
                    t._counted_raw_slot = False
            except Exception:
                pass
            # Best-effort cancellation so a stuck worker stops chewing CPU/GPU.
            try:
                if t is not None and hasattr(t, "cancel"):
                    t.cancel()
            except Exception:
                pass
        self._active_raw_tasks = len(holders)

    def _release_raw_slot_locked(self, task: ImageLoadTask | None) -> None:
        """Drop a heavy RAW slot immediately when cancelling a counted task.

        Must run under ``_queue_lock``. Leaving the counter until ``_task_finished``
        starves every later ``full`` decode when a cancelled worker is stuck in
        an uninterruptible GPU/CPU demosaic (observed as 45s nav TIMEOUTs with
        no ``[PIPE_T] ... stages=['full']`` for the on-screen file).
        """
        if task is None:
            return
        if getattr(task, "_counted_raw_slot", False):
            task._counted_raw_slot = False
        holders = getattr(self, "_raw_slot_holders", None)
        if holders is None:
            self._raw_slot_holders = set()
            holders = self._raw_slot_holders
        holders.discard(task)
        self._active_raw_tasks = len(holders)

    def cancel_task(self, file_path: str):
        """取消任務（非阻塞）"""
        with self._queue_lock:
            keys = list(self._task_keys_by_path.get(file_path, ()))
            for key in keys:
                task = self._active_tasks.pop(key, None)
                if task is not None:
                    task.cancel()
                    self._release_raw_slot_locked(task)
            self._task_keys_by_path.pop(file_path, None)
            self._compact_work_queue()

    def cancel_current_priority_tasks(self, file_path: str) -> None:
        """Cancel only CURRENT-priority work; keep PRELOAD/BACKGROUND prefetch for that path."""
        if not file_path:
            return
        with self._queue_lock:
            keys = list(self._task_keys_by_path.get(file_path, ()))
            for key in keys:
                task = self._active_tasks.get(key)
                if task is None or task.priority != Priority.CURRENT:
                    continue
                task.cancel()
                self._active_tasks.pop(key, None)
                self._task_keys_by_path[file_path].discard(key)
                self._release_raw_slot_locked(task)
            if not self._task_keys_by_path.get(file_path):
                self._task_keys_by_path.pop(file_path, None)
            self._compact_work_queue()

    def cancel_non_gallery_tasks(self) -> None:
        """Cancel queued/active work that is not a gallery grid thumbnail decode."""
        with self._queue_lock:
            for key, task in list(self._active_tasks.items()):
                if getattr(task, "gallery_thumbnail", False):
                    continue
                task.cancel()
                fp = getattr(task, "file_path", None)
                self._active_tasks.pop(key, None)
                if fp and fp in self._task_keys_by_path:
                    self._task_keys_by_path[fp].discard(key)
                    if not self._task_keys_by_path[fp]:
                        self._task_keys_by_path.pop(fp, None)
                self._release_raw_slot_locked(task)
            self._reconcile_raw_slots_locked()
            self._compact_work_queue()

    def cancel_queued_non_gallery_tasks(self) -> None:
        """Drop queued (not yet running) work that is not a gallery thumbnail decode."""
        with self._queue_lock:
            if self._work_queue.empty():
                return
            kept: list = []
            while not self._work_queue.empty():
                try:
                    task = self._work_queue.get_nowait()
                except queue.Empty:
                    break
                if getattr(task, "gallery_thumbnail", False):
                    kept.append(task)
                    continue
                task.cancel()
                key = getattr(task, "task_key", None)
                if key in self._active_tasks and self._active_tasks[key] is task:
                    del self._active_tasks[key]
                    fp = getattr(task, "file_path", None)
                    if fp and fp in self._task_keys_by_path:
                        self._task_keys_by_path[fp].discard(key)
                        if not self._task_keys_by_path[fp]:
                            self._task_keys_by_path.pop(fp, None)
            for task in kept:
                self._work_queue.put(task)
            self._reconcile_raw_slots_locked()

    def cancel_gallery_tasks(self) -> None:
        """Cancel queued/active gallery thumbnail decodes (leaving single-view work)."""
        with self._queue_lock:
            for key, task in list(self._active_tasks.items()):
                if not getattr(task, "gallery_thumbnail", False):
                    continue
                task.cancel()
                fp = getattr(task, "file_path", None)
                self._active_tasks.pop(key, None)
                if fp and fp in self._task_keys_by_path:
                    self._task_keys_by_path[fp].discard(key)
                    if not self._task_keys_by_path[fp]:
                        self._task_keys_by_path.pop(fp, None)
            if not self._work_queue.empty():
                kept: list = []
                while not self._work_queue.empty():
                    try:
                        task = self._work_queue.get_nowait()
                    except queue.Empty:
                        break
                    if getattr(task, "gallery_thumbnail", False):
                        task.cancel()
                        key = getattr(task, "task_key", None)
                        if key in self._active_tasks and self._active_tasks[key] is task:
                            del self._active_tasks[key]
                        continue
                    kept.append(task)
                for task in kept:
                    self._work_queue.put(task)
            self._reconcile_raw_slots_locked()
            self._compact_work_queue()

    def cancel_tasks_by_priority(self, priority: Priority) -> None:
        """Cancel all tasks matching the specified priority."""
        with self._queue_lock:
            for key in list(self._active_tasks.keys()):
                task = self._active_tasks.get(key)
                if task is None or task.priority != priority:
                    continue
                task.cancel()
                self._release_raw_slot_locked(task)
                self._active_tasks.pop(key, None)
                file_path = getattr(task, 'file_path', None)
                if file_path and file_path in self._task_keys_by_path:
                    self._task_keys_by_path[file_path].discard(key)
                    if not self._task_keys_by_path.get(file_path):
                        self._task_keys_by_path.pop(file_path, None)
            self._compact_work_queue()


