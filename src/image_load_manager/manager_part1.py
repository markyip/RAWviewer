"""ImageLoadManager part 1: init, watchdog, throttling, GPU profile, probe."""
from __future__ import annotations

import concurrent.futures
import contextlib
import os
import queue
import threading
import time
from collections import defaultdict

from PyQt6.QtCore import QObject, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

from image_cache import get_image_cache

from .helpers import Priority, _env_int, _env_true
from .worker import ImageLoadTask


class ImageLoadManagerPart1(QObject):
    """統一管理所有圖像加載任務的工作隊列管理器（單例模式）"""

    # 信號定義
    # NOTE: Can emit np.ndarray (base thumbnail) or QImage (pre-scaled/cropped) for smooth UI.
    thumbnail_ready = pyqtSignal(str, object)  # file_path, thumbnail (np.ndarray or QImage)
    image_ready = pyqtSignal(str, object)  # file_path, full_image (ndarray or DeviceRgb)
    pixmap_ready = pyqtSignal(str, QPixmap)  # file_path, pixmap
    # Non-RAW full buffers arrive as QImage: workers cannot legally build a
    # QPixmap. Converted to one on the GUI thread by the receiving slot.
    qimage_ready = pyqtSignal(str, QImage)  # file_path, image
    error_occurred = pyqtSignal(str, str)  # file_path, error_message
    task_completed = pyqtSignal(str)  # file_path
    progress_updated = pyqtSignal(str, str)  # file_path, status_message
    exif_data_ready = pyqtSignal(str, dict)  # file_path, exif_data

    def __init__(self, max_workers: int = 4):
        """初始化管理器"""
        # CRITICAL: 對於 QObject 子類，必須在最開始就調用 super().__init__()
        # 不能在調用 super().__init__() 之前訪問任何實例屬性（包括 hasattr）
        import sys
        verbose_init = _env_true("RAWVIEWER_VERBOSE_MANAGER_INIT", default=False)
        if verbose_init:
            print("[ImageLoadManager.__init__] Starting initialization...", flush=True)

        # Ensure QApplication exists before initializing QObject
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            print("[ImageLoadManager.__init__] ERROR: QApplication instance not found!", file=sys.stderr, flush=True)
            raise RuntimeError("QApplication must be created before ImageLoadManager")

        if verbose_init:
            print("[ImageLoadManager.__init__] Calling super().__init__(app)...", flush=True)
        try:
            # Root the manager in the application's lifecycle to prevent premature deletion.
            super().__init__(app)
            if verbose_init:
                print("[ImageLoadManager.__init__] super().__init__(app) completed", flush=True)
        except Exception as e:
            print(f"[ImageLoadManager.__init__] ERROR in super().__init__(): {e}", file=sys.stderr, flush=True)
            import traceback
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            raise

        # 避免重複初始化實例變量
        if hasattr(self, '_initialized') and self._initialized:
            if verbose_init:
                print("[ImageLoadManager.__init__] Already initialized, skipping instance variables", flush=True)
            return

        self._work_queue = queue.PriorityQueue()
        # Background / prefetch lane (PRELOAD_*, BACKGROUND).
        self._thread_pool = QThreadPool()
        # Interactive CURRENT lane: can start even when the background pool is full,
        # so non-CURRENT sleepers (or long prefetch) never starve visible loads.
        self._current_thread_pool = QThreadPool()
        # Tasks currently inside ImageLoadWorker.run() (running-only yield checks).
        self._running_tasks: set = set()
        # Tasks currently holding a heavy RAW slot. Source of truth for
        # _active_raw_tasks; see _claim_raw_slot_locked.
        self._raw_slot_holders: set = set()
        self._demosaic_in_flight: set = set()
        self._running_tasks_lock = threading.Lock()

        self._stopped = False  # Flag to stop scheduling new tasks
        self._gallery_warmup_throttle_depth = 0
        self._io_pressure_throttle_depth = 0
        self._indexing_throttle_depth = 0
        # Semantic indexing throttle is toggled from the background index worker
        # thread as well as the UI thread (face scan), so guard the depth counter.
        # ONE lock for every piece of throttle state: the io-pressure and
        # gallery-warmup depths, the saved baselines, the setMaxThreadCount
        # save/restore, the semantic-indexing depth, and _raw_load_limit.
        #
        # These used to be two locks over one piece of state:
        # update_volume_throttling read _indexing_throttle_depth while holding
        # only _throttle_lock, and wrote _raw_load_limit outside both. The
        # interleaving that bit: indexing enter saves raw_limit=4 and sets 1,
        # the volume probe then writes 8, indexing exit restores 4 -- and the
        # new volume's limit is silently lost.
        #
        # Defined BEFORE the alias below: assigning the alias first raised
        # AttributeError and no ImageLoadManager could be constructed at all.
        # An RLock so the existing nesting cannot deadlock.
        self._throttle_lock = threading.RLock()
        self._indexing_throttle_lock = self._throttle_lock

        # Throttled workers on macOS external drives to prevent IO overload crashes
        self.update_volume_throttling(os.getcwd())
        if max_workers != 4:
            self._thread_pool.setMaxThreadCount(max_workers)
            self._apply_current_pool_limit(max_workers)

        # PROCESS POOL (optional): on Windows debug/startup paths, process spawn can
        # re-import heavy modules and hurt first-load latency. Keep it opt-in.
        # PROCESS POOL: LibRaw postprocess in worker processes (multi-core on Windows).
        from common_image_loader import (
            process_pool_worker_count,
            use_raw_process_pool,
        )

        use_process_pool = use_raw_process_pool()
        self._process_pool = None
        if use_process_pool:
            import atexit
            import logging

            from common_image_loader import process_pool_mp_context

            logging.getLogger(__name__).info(
                "[LOAD] LibRaw process pool enabled (%d workers)",
                process_pool_worker_count(),
            )
            self._process_pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=process_pool_worker_count(),
                mp_context=process_pool_mp_context(),
            )
            # Keep a reference to the bound method so we can unregister it cleanly
            self._process_pool_shutdown_hook = lambda: self._process_pool.shutdown(wait=False)
            atexit.register(self._process_pool_shutdown_hook)
        else:
            import logging
            logging.getLogger(__name__).info("[LOAD] LibRaw process pool disabled")

        self._active_tasks: dict[tuple, ImageLoadTask] = {}
        self._task_keys_by_path = defaultdict(set)
        self._cache = get_image_cache()
        self._queue_lock = threading.Lock()

        # RAW throttling: Limit concurrent heavy RAW decodes (set by update_volume_throttling)
        self._active_raw_tasks = 0

        # Watchdog pump: dispatch is normally triggered by enqueues and completions, but
        # several stall variants have been observed where the queue sits non-empty with
        # idle workers (yield livelock, throttle transitions, missed completion events)
        # until some unrelated event kicks the pool — seen as multi-second gallery
        # freezes. This periodic pump guarantees any queued work is (re)dispatched
        # within one tick regardless of the trigger. It is a cheap no-op when the
        # queue is empty.
        self._dispatch_watchdog = QTimer(self)
        self._dispatch_watchdog.setInterval(
            _env_int("RAWVIEWER_DISPATCH_WATCHDOG_MS", 750, minimum=100)
        )
        self._dispatch_watchdog.timeout.connect(self._watchdog_pump)
        self._dispatch_watchdog.start()

        self._initialized = True

    def _watchdog_pump(self) -> None:
        if self._stopped:
            return
        try:
            if not self._work_queue.empty():
                self._schedule_next_task()
        except Exception:
            pass

    def update_volume_throttling(self, folder_path: str | None) -> None:
        """Dynamically update maximum threads and raw load limit for a new folder path."""
        from common_image_loader import (
            is_external_or_network_volume,
            raw_concurrent_load_limit,
            volume_speed_tier,
        )
        core_count = os.cpu_count() or 4
        default_workers = max(16, core_count * 2)

        env_workers = os.environ.get("RAWVIEWER_LOAD_MAX_WORKERS", "").strip()
        if env_workers:
            with contextlib.suppress(ValueError):
                default_workers = max(4, int(env_workers))
        elif is_external_or_network_volume(folder_path):
            # Gallery thumbnail decodes are NOT gated by raw_load_limit (only heavy
            # full-res decodes are), so the worker pool size is the real lever for
            # how many simultaneous reads hit an external/USB/network drive. A full
            # cpu*2 pool (~40 on a many-core box) saturates the drive and the LibRaw
            # process pool, which can crash the app on Windows. Confirmed-slow drives
            # are capped on every platform; the moderate cap for fast external drives
            # is Windows-only (macOS handled them at full speed without crashing).
            from common_image_loader import moderate_external_cap_enabled

            cap = None
            if volume_speed_tier(folder_path) == "slow":
                cap = _env_int("RAWVIEWER_SLOW_VOLUME_MAX_WORKERS", 12, minimum=2)
            elif moderate_external_cap_enabled():
                cap = _env_int("RAWVIEWER_EXTERNAL_VOLUME_MAX_WORKERS", 16, minimum=2)
            if cap is not None:
                default_workers = min(default_workers, cap)

        # CURRENT pool: small default keeps nav from saturating host; raise via
        # RAWVIEWER_CURRENT_MAX_WORKERS or gallery warmup throttle (still ≤ bg).
        current_default = max(2, min(4, default_workers // 4 if default_workers >= 8 else 2))
        current_workers = _env_int(
            "RAWVIEWER_CURRENT_MAX_WORKERS", current_default, minimum=1
        )
        # Cap wells short of unbounded: CURRENT pool never exceeds bg baseline.
        current_workers = max(1, min(current_workers, default_workers))

        # If no throttle is currently active, apply directly. Guarded by
        # _throttle_lock: this runs on the VolumeSpeedProbe daemon thread and
        # races enter/exit throttle from worker threads.
        with self._throttle_lock:
            is_throttled = (
                (getattr(self, "_io_pressure_throttle_depth", 0) or 0) > 0
                or (getattr(self, "_gallery_warmup_throttle_depth", 0) or 0) > 0
                or (getattr(self, "_indexing_throttle_depth", 0) or 0) > 0
            )
            if not is_throttled:
                self._thread_pool.setMaxThreadCount(default_workers)
                self._apply_current_pool_limit(current_workers)

            # Save baseline to be restored when throttles exit
            self._pressure_saved_max_threads = default_workers
            self._warmup_saved_max_threads = default_workers
            self._indexing_saved_max_threads = default_workers
            self._pressure_saved_current_threads = current_workers
            self._warmup_saved_current_threads = current_workers
            self._indexing_saved_current_threads = current_workers
            self._current_pool_baseline = current_workers

            from common_image_loader import raw_load_limit_aligned_to_gpu

            # Inside the lock: the indexing throttle saves and restores this
            # around its own enter/exit, so an unlocked write here was lost
            # the moment indexing exited.
            self._raw_load_limit = raw_load_limit_aligned_to_gpu(
                raw_concurrent_load_limit(folder_path)
            )

    def apply_gpu_decode_profile(self) -> None:
        """After GPU backend import: drop process pool conflict; align RAW slots."""
        import logging

        from common_image_loader import (
            _gpu_demosaic_backend_in_use,
            raw_load_limit_aligned_to_gpu,
            use_raw_process_pool,
        )

        log = logging.getLogger(__name__)
        try:
            import sys as _sys

            mod = _sys.modules.get("gpu_raw_processor")
            backend = mod.detect_gpu_backend() if mod is not None else "cpu_only"
        except Exception:
            backend = "cpu_only"

        # Keep both limits: GPU demosaic wants heavy decodes serialized, but that
        # throttle starves gallery tile fill, so gallery view swaps back to the
        # pre-GPU limit (set_gallery_view_active) and single view re-applies it.
        pre_gpu = int(getattr(self, "_raw_load_limit", 4) or 4)
        self._pre_gpu_raw_limit = pre_gpu
        self._gpu_aligned_raw_limit = raw_load_limit_aligned_to_gpu(pre_gpu)
        if not getattr(self, "_gallery_view_active", False):
            self._raw_load_limit = self._gpu_aligned_raw_limit

        # Tear down process pool only when GPU demosaic is both available AND
        # actually enabled (RAWVIEWER_PREFER_GPU_DECODE) -- not merely present.
        # A present-but-unused GPU backend (the shipping default) must not
        # cost the LibRaw process pool's out-of-process parallelism.
        force_on = os.environ.get("RAWVIEWER_USE_PROCESS_POOL", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
        if (
            _gpu_demosaic_backend_in_use()
            and not force_on
            and getattr(self, "_process_pool", None) is not None
        ):
            import atexit

            pool = self._process_pool
            self._process_pool = None
            if hasattr(self, "_process_pool_shutdown_hook"):
                with contextlib.suppress(Exception):
                    atexit.unregister(self._process_pool_shutdown_hook)
            with contextlib.suppress(Exception):
                pool.shutdown(wait=False)
            log.info(
                "[LOAD] LibRaw process pool disabled (GPU demosaic=%s; RAWVIEWER_USE_PROCESS_POOL=1 to force)",
                backend,
            )
        elif not use_raw_process_pool() and getattr(self, "_process_pool", None) is not None:
            pass  # already handled above
        log.info(
            "[LOAD] GPU decode profile: backend=%s raw_limit=%s process_pool=%s",
            backend,
            getattr(self, "_raw_load_limit", "?"),
            getattr(self, "_process_pool", None) is not None,
        )

    def set_gallery_view_active(self, active: bool) -> None:
        """Suspend the GPU-decode raw throttle while gallery view is showing.

        GPU demosaic serializes heavy RAW decodes (raw_limit=1 on MPS), which
        starves gallery tile fill through the shared pool. Gallery only needs
        thumbnails, so restore the volume-probed limit there and re-apply the
        GPU-aligned limit when returning to single view.
        """
        self._gallery_view_active = bool(active)
        # The effective limit is resolved at dispatch time in the scheduler
        # (heavy_limit override): mutating _raw_load_limit here raced with the
        # warmup/indexing throttles' save/restore and got pinned back to the
        # GPU value. This setter only records the mode.
        gpu_limit = getattr(self, "_gpu_aligned_raw_limit", None)
        pre_gpu = getattr(self, "_pre_gpu_raw_limit", None)
        if gpu_limit is None or pre_gpu is None or gpu_limit == pre_gpu:
            return  # GPU profile not applied (or it changed nothing)
        import logging

        logging.getLogger(__name__).info(
            "[LOAD] Gallery-aware GPU throttle: raw_limit=%d (gallery_active=%s)",
            pre_gpu if active else gpu_limit,
            active,
        )

    def _apply_current_pool_limit(self, max_workers: int) -> None:
        pool = getattr(self, "_current_thread_pool", None)
        if pool is not None:
            pool.setMaxThreadCount(max(1, int(max_workers)))

    def _pool_for_task(self, task: ImageLoadTask) -> QThreadPool:
        # Gallery viewport tiles are queued as Priority.CURRENT (beat PRELOAD in
        # the priority queue) but must not share the small interactive CURRENT
        # pool (default 2–4): after warmup, scroll re-requests dozens of thumbs
        # and a tiny CURRENT pool stalls the gallery. Single-view CURRENT stays
        # on _current_thread_pool so it can still preempt a full bg pool.
        if getattr(task, "gallery_thumbnail", False):
            return self._thread_pool
        if getattr(task, "priority", None) == Priority.CURRENT:
            return self._current_thread_pool
        return self._thread_pool

    def _mark_task_running(self, task: ImageLoadTask) -> None:
        lock = getattr(self, "_running_tasks_lock", None)
        if lock is not None:
            with lock:
                self._running_tasks.add(task)

    def _mark_task_not_running(self, task: ImageLoadTask) -> None:
        lock = getattr(self, "_running_tasks_lock", None)
        if lock is not None:
            with lock:
                self._running_tasks.discard(task)

    def prime_volume_speed_async(
        self, folder_path: str | None, sample_path: str | None = None
    ) -> None:
        """Run one lightweight read-speed probe for *folder_path*'s volume off the
        main thread, then re-apply throttling from the measured speed.

        Cheap and idempotent: only external/network volumes are probed, the probe
        reads a few MB from a single file, and the result is cached per mount so
        repeated folder opens on the same drive do no extra work.
        """
        from common_image_loader import is_external_or_network_volume

        if not folder_path or not is_external_or_network_volume(folder_path):
            return

        def _worker() -> None:
            from common_image_loader import (
                _windows_drive_is_fixed_local,
                invalidate_volume_speed_cache,
                probe_volume_speed,
                record_volume_speed,
                volume_speed_tier,
            )

            # Let fast-open / sort probes finish so we measure disk not page cache.
            time.sleep(2.0)

            candidates = self._pick_probe_samples(
                folder_path, exclude=sample_path, limit=8
            )
            if not candidates and sample_path and os.path.isfile(sample_path):
                candidates = [sample_path]

            best_mbps: float | None = None
            probe_path = None
            for cand in candidates:
                mbps = probe_volume_speed(cand)
                if mbps is None:
                    continue
                probe_path = cand
                if best_mbps is None or mbps > best_mbps:
                    best_mbps = mbps

            # Contended first probe on a fixed secondary SSD can read artificially low.
            if (
                best_mbps is not None
                and best_mbps < 120.0
                and _windows_drive_is_fixed_local(folder_path or probe_path)
            ):
                time.sleep(3.0)
                invalidate_volume_speed_cache(probe_path or folder_path)
                for cand in candidates[:3]:
                    mbps = probe_volume_speed(
                        cand, sample_bytes=12 * 1024 * 1024, timeout_s=4.0
                    )
                    if mbps is None:
                        continue
                    probe_path = cand
                    if best_mbps is None or mbps > best_mbps:
                        best_mbps = mbps

            if probe_path is None:
                return
            if best_mbps is not None:
                record_volume_speed(probe_path, best_mbps)
            with contextlib.suppress(Exception):
                self.update_volume_throttling(folder_path)
            try:
                import logging

                logging.getLogger(__name__).info(
                    "[LOAD] Volume speed probe: %s -> %s MB/s (%s, raw_limit=%s)",
                    folder_path,
                    f"{best_mbps:.1f}" if best_mbps else "n/a",
                    volume_speed_tier(probe_path),
                    getattr(self, "_raw_load_limit", "?"),
                )
            except Exception:
                pass

        threading.Thread(
            target=_worker, name="VolumeSpeedProbe", daemon=True
        ).start()

    @staticmethod
    def _pick_probe_samples(
        folder_path: str | None,
        *,
        exclude: str | None = None,
        limit: int = 5,
    ) -> list:
        """Pick representative files in *folder_path* to probe (cheap scandir).

        Excludes *exclude* (e.g. the just-opened file whose bytes are already
        cached) and prefers RAW files, which are large enough for a meaningful
        random-offset read. Returns up to *limit* paths.
        """
        if not folder_path or not os.path.isdir(folder_path):
            return []
        raw_exts = {".arw", ".cr2", ".cr3", ".nef", ".dng", ".raf", ".orf", ".rw2"}
        other_exts = {".jpg", ".jpeg", ".heic", ".png", ".tif", ".tiff"}
        exclude_norm = None
        if exclude:
            try:
                exclude_norm = os.path.normcase(os.path.normpath(os.path.abspath(exclude)))
            except Exception:
                exclude_norm = None
        raw_hits: list = []
        other_hits: list = []
        try:
            with os.scandir(folder_path) as it:
                for entry in it:
                    try:
                        if not entry.is_file():
                            continue
                    except OSError:
                        continue
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext not in raw_exts and ext not in other_exts:
                        continue
                    if exclude_norm is not None:
                        try:
                            if os.path.normcase(os.path.normpath(entry.path)) == exclude_norm:
                                continue
                        except Exception:
                            pass
                    (raw_hits if ext in raw_exts else other_hits).append(entry.path)
                    if len(raw_hits) >= limit:
                        break
        except Exception:
            return []
        return (raw_hits + other_hits)[:limit]

    @classmethod
    def _pick_probe_sample(cls, folder_path: str | None) -> str | None:
        """Single representative file in *folder_path* (legacy helper)."""
        picks = cls._pick_probe_samples(folder_path, limit=1)
        return picks[0] if picks else None

    def enter_io_pressure_throttle(self) -> None:
        """Cut worker/RAW concurrency after EMFILE so open fds can drain."""
        self._io_pressure_throttle_depth = (
            int(getattr(self, "_io_pressure_throttle_depth", 0) or 0) + 1
        )
        if self._io_pressure_throttle_depth != 1:
            return
        self._pressure_saved_max_threads = self._thread_pool.maxThreadCount()
        self._pressure_saved_current_threads = self._current_thread_pool.maxThreadCount()
        pressure_max = _env_int("RAWVIEWER_IO_PRESSURE_MAX_WORKERS", 6, minimum=2)
        self._thread_pool.setMaxThreadCount(
            min(pressure_max, self._pressure_saved_max_threads)
        )
        # Keep a small CURRENT lane open so UI never hard-blocks under pressure.
        self._apply_current_pool_limit(
            min(max(2, pressure_max // 2), self._pressure_saved_current_threads)
        )
        self._pressure_saved_raw_limit = self._raw_load_limit
        self._raw_load_limit = min(2, int(self._raw_load_limit or 2))

    def exit_io_pressure_throttle(self) -> None:
        depth = int(getattr(self, "_io_pressure_throttle_depth", 0) or 0)
        if depth <= 0:
            return
        self._io_pressure_throttle_depth = depth - 1
        if self._io_pressure_throttle_depth > 0:
            return
        saved_threads = getattr(self, "_pressure_saved_max_threads", None)
        if saved_threads is not None:
            self._thread_pool.setMaxThreadCount(saved_threads)
        saved_current = getattr(self, "_pressure_saved_current_threads", None)
        if saved_current is not None:
            self._apply_current_pool_limit(saved_current)
        saved_raw = getattr(self, "_pressure_saved_raw_limit", None)
        if saved_raw is not None:
            self._raw_load_limit = saved_raw

    def enter_gallery_warmup_throttle(self) -> None:
        """Raise worker/RAW concurrency while gallery tiles are first painting.

        Single-view GPU demosaic alignment often leaves ``_raw_load_limit`` at
        1–3. The previous ``min(6, _raw_load_limit)`` therefore *kept* that
        GPU cap during gallery entry instead of restoring volume parallelism —
        a regression when switching single→gallery with CUDA demosaic on.
        """
        self._gallery_warmup_throttle_depth = (
            int(getattr(self, "_gallery_warmup_throttle_depth", 0) or 0) + 1
        )
        if self._gallery_warmup_throttle_depth != 1:
            return
        self._warmup_saved_max_threads = self._thread_pool.maxThreadCount()
        self._warmup_saved_current_threads = self._current_thread_pool.maxThreadCount()
        warmed_max = _env_int("RAWVIEWER_GALLERY_WARMUP_MAX_WORKERS", 24, minimum=2)
        self._thread_pool.setMaxThreadCount(
            min(warmed_max, max(self._warmup_saved_max_threads, warmed_max))
        )
        # Warmup is CURRENT-heavy: raise CURRENT pool toward the warmup budget
        # (baseline CURRENT is intentionally small for single-view).
        bg_cap = self._thread_pool.maxThreadCount()
        self._apply_current_pool_limit(min(warmed_max, bg_cap))
        self._warmup_saved_raw_limit = self._raw_load_limit
        pre_gpu = getattr(self, "_pre_gpu_raw_limit", None)
        try:
            pre_gpu_n = int(pre_gpu) if pre_gpu is not None else 0
        except (TypeError, ValueError):
            pre_gpu_n = 0
        # Prefer the pre-GPU volume limit (often cpu_count-based) over the
        # GPU-aligned single-view cap so thumbnail admission is not serialized.
        base = pre_gpu_n if pre_gpu_n > 0 else int(self._raw_load_limit or 6)
        self._raw_load_limit = max(6, min(32, base))
        import logging

        logging.getLogger(__name__).info(
            "[LOAD] Gallery warmup throttle ON (workers=%d current=%d raw_limit=%d; was raw_limit=%s pre_gpu=%s)",
            self._thread_pool.maxThreadCount(),
            self._current_thread_pool.maxThreadCount(),
            self._raw_load_limit,
            self._warmup_saved_raw_limit,
            pre_gpu_n or "n/a",
        )
    def exit_gallery_warmup_throttle(self) -> None:
        depth = int(getattr(self, "_gallery_warmup_throttle_depth", 0) or 0)
        if depth <= 0:
            return
        self._gallery_warmup_throttle_depth = depth - 1
        if self._gallery_warmup_throttle_depth > 0:
            return
        saved_threads = getattr(self, "_warmup_saved_max_threads", None)
        if saved_threads is not None:
            self._thread_pool.setMaxThreadCount(saved_threads)
        saved_current = getattr(self, "_warmup_saved_current_threads", None)
        if saved_current is not None:
            self._apply_current_pool_limit(saved_current)
        saved_raw = getattr(self, "_warmup_saved_raw_limit", None)
        if saved_raw is not None:
            self._raw_load_limit = saved_raw

    def indexing_throttle_active(self) -> bool:
        """True while semantic/face heavy work has engaged the ILM throttle."""
        return int(getattr(self, "_indexing_throttle_depth", 0) or 0) > 0

    def enter_semantic_indexing_throttle(self) -> None:
        """Lower gallery/preload concurrency while semantic or face indexing runs."""
        lock = getattr(self, "_indexing_throttle_lock", None)
        if lock is not None:
            lock.acquire()
        try:
            self._indexing_throttle_depth = (
                int(getattr(self, "_indexing_throttle_depth", 0) or 0) + 1
            )
            if self._indexing_throttle_depth != 1:
                return
            self._indexing_saved_max_threads = self._thread_pool.maxThreadCount()
            self._indexing_saved_current_threads = self._current_thread_pool.maxThreadCount()
            indexing_max = _env_int("RAWVIEWER_INDEXING_MAX_WORKERS", 6, minimum=2)
            self._thread_pool.setMaxThreadCount(
                min(indexing_max, self._indexing_saved_max_threads)
            )
            self._apply_current_pool_limit(
                min(indexing_max, self._indexing_saved_current_threads)
            )
            self._indexing_saved_raw_limit = self._raw_load_limit
            self._raw_load_limit = min(1, int(self._raw_load_limit or 1))
            import logging

            logging.getLogger(__name__).info(
                "[LOAD] Semantic indexing throttle ON (max_workers=%d current=%d raw_limit=%d)",
                self._thread_pool.maxThreadCount(),
                self._current_thread_pool.maxThreadCount(),
                self._raw_load_limit,
            )
        finally:
            if lock is not None:
                lock.release()

    def exit_semantic_indexing_throttle(self) -> None:
        lock = getattr(self, "_indexing_throttle_lock", None)
        if lock is not None:
            lock.acquire()
        try:
            depth = int(getattr(self, "_indexing_throttle_depth", 0) or 0)
            if depth <= 0:
                return
            self._indexing_throttle_depth = depth - 1
            if self._indexing_throttle_depth > 0:
                return
            saved_threads = getattr(self, "_indexing_saved_max_threads", None)
            if saved_threads is not None:
                self._thread_pool.setMaxThreadCount(saved_threads)
            saved_current = getattr(self, "_indexing_saved_current_threads", None)
            if saved_current is not None:
                self._apply_current_pool_limit(saved_current)
            saved_raw = getattr(self, "_indexing_saved_raw_limit", None)
            if saved_raw is not None:
                self._raw_load_limit = saved_raw
            import logging

            logging.getLogger(__name__).info("[LOAD] Semantic indexing throttle OFF")
        finally:
            if lock is not None:
                lock.release()

    def force_exit_semantic_indexing_throttle(self) -> None:
        lock = getattr(self, "_indexing_throttle_lock", None)
        if lock is not None:
            lock.acquire()
        try:
            depth = int(getattr(self, "_indexing_throttle_depth", 0) or 0)
            if depth <= 0:
                return
            self._indexing_throttle_depth = 0
            saved_threads = getattr(self, "_indexing_saved_max_threads", None)
            if saved_threads is not None:
                self._thread_pool.setMaxThreadCount(saved_threads)
            saved_current = getattr(self, "_indexing_saved_current_threads", None)
            if saved_current is not None:
                self._apply_current_pool_limit(saved_current)
            saved_raw = getattr(self, "_indexing_saved_raw_limit", None)
            if saved_raw is not None:
                self._raw_load_limit = saved_raw
            import logging

            logging.getLogger(__name__).info("[LOAD] Semantic indexing throttle FORCE OFF")
        finally:
            if lock is not None:
                lock.release()


