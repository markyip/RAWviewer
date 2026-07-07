"""
統一圖像加載管理器 - 基於工作隊列和線程池的架構

這個模組實現了重構提案中的 ImageLoadManager，使用工作隊列和線程池
來管理所有圖像加載任務，避免頻繁創建/銷毀線程的開銷。
"""

import os
import threading
import queue
import time
from collections import defaultdict
from enum import Enum
from typing import Optional, Dict, Any, Tuple
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, QRunnable, QThreadPool, QSize, Qt, QTimer, QLoggingCategory
from PyQt6.QtGui import QPixmap, QImage
# Silence qt.imageformats warnings (e.g. missing TIFF tag warnings on RAW files)
QLoggingCategory.setFilterRules("qt.imageformats.warning=false\nqt.imageformats.tiff.warning=false")

import concurrent.futures
from image_cache import get_image_cache
# UnifiedImageProcessor will be imported lazily to avoid circular import issues

# Thread-local storage to track if current thread is executing a CURRENT priority task
worker_thread_local = threading.local()

def yield_if_current_task_active() -> None:
    """Yield execution if another thread is running a high priority task.

    BOUNDED: this must never sleep indefinitely. _active_tasks includes CURRENT
    tasks that are merely QUEUED (registered at enqueue, before dispatch), and
    _schedule_next_task only starts tasks while activeThreadCount < maxThreadCount.
    So if the pool is full of non-CURRENT workers all sleeping here, the queued
    CURRENT tasks can never start and the sleepers never wake — a livelock that
    froze gallery entry for ~20s (pending>0, zero decodes) until an unrelated event
    kicked the pool. A bounded yield keeps the intent (give CURRENT work a head
    start at yield points) while guaranteeing pool threads are eventually released
    so the queued CURRENT work can actually run. The worker's own task is also
    checked so cancelled workers exit immediately instead of sleeping.
    """
    import time
    if threading.current_thread() is threading.main_thread():
        return
    current_priority = getattr(worker_thread_local, 'priority', None)
    if current_priority == Priority.CURRENT:
        return

    manager = _global_manager
    if not manager:
        return

    own_task = getattr(worker_thread_local, 'task', None)
    max_wait_s = _env_int("RAWVIEWER_YIELD_MAX_MS", 1500, minimum=100) / 1000.0
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        if own_task is not None and own_task.is_cancelled():
            return
        has_current = False
        with manager._queue_lock:
            for t in manager._active_tasks.values():
                if getattr(t, "priority", None) == Priority.CURRENT and not t.is_cancelled():
                    has_current = True
                    break
        if not has_current:
            break
        time.sleep(0.05)


def _env_true(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return max(minimum, default)
    try:
        return max(minimum, int(str(raw).strip()))
    except (TypeError, ValueError):
        return max(minimum, default)


def _buffer_max_dim(buf) -> int:
    try:
        if hasattr(buf, "shape"):
            return max(int(buf.shape[0]), int(buf.shape[1]))
        if hasattr(buf, "width") and hasattr(buf, "height"):
            return max(int(buf.width()), int(buf.height()))
    except Exception:
        pass
    return 0


def _display_preview_min_dim() -> int:
    from image_cache import memory_preview_max_edge

    return int(memory_preview_max_edge() * 0.85)


def _min_acceptable_preview_dim(file_path: str) -> int:
    from common_image_loader import dng_prefers_embedded_preview_first

    if dng_prefers_embedded_preview_first(file_path):
        return 1024
    return _display_preview_min_dim()


def _gallery_grid_min_dim() -> int:
    from image_cache import disk_preview_max_edge

    return int(disk_preview_max_edge() * 0.85)


def _gallery_memory_thumb_acceptable(thumb, *, gallery_thumbnail: bool) -> bool:
    """Gallery tiles need grid-tier (~512px), not 256px thumbnail tier."""
    if not gallery_thumbnail:
        return True
    dim = _buffer_max_dim(thumb)
    if dim <= 0:
        return False
    return dim >= _gallery_grid_min_dim()


def _skip_low_res_memory_thumb_for_display_tier(
    thumb,
    wanted: set,
    use_full_resolution: bool,
    *,
    gallery_thumbnail: bool = False,
) -> bool:
    """Do not emit 512px grid-tier thumbs for preview-first single-view loads."""
    if gallery_thumbnail:
        return False
    if use_full_resolution or "full" in wanted:
        return False
    if wanted == {"thumbnail"}:
        return False
    if "thumbnail" not in wanted or "exif" not in wanted:
        return False
    min_dim = _env_int("RAWVIEWER_GALLERY_INSTANT_MIN_DIM", 1400, minimum=400)
    px = _buffer_max_dim(thumb)
    return 0 < px < min_dim


class Priority(Enum):
    """任務優先級"""
    CURRENT = 0  # 當前圖像（最高優先級）
    PRELOAD_NEXT = 1  # 下一個圖像預載入
    PRELOAD_PREV = 2  # 上一個圖像預載入
    BACKGROUND = 3  # 背景任務（最低優先級）


class ImageLoadTask:
    """圖像加載任務"""
    
    def __init__(self, file_path: str, priority: Priority = Priority.CURRENT, 
                 use_full_resolution: bool = False,
                 stages: Optional[set] = None,
                 thumbnail_target_size: Optional[QSize] = None,
                 thumbnail_fit: str = "crop",
                 gallery_thumbnail: bool = False):
        self.file_path = file_path
        self.priority = priority
        self.use_full_resolution = use_full_resolution
        self.gallery_thumbnail = gallery_thumbnail
        # Stages: {'thumbnail', 'exif', 'full'}.
        # Gallery should typically request only {'thumbnail'} (and optionally 'exif').
        self.stages = stages if stages is not None else {'thumbnail', 'exif', 'full'}
        # If provided, the thumbnail will be scaled/cropped in worker thread to this size.
        # This keeps UI thread light for smooth pixel-level scrolling.
        self.thumbnail_target_size = thumbnail_target_size
        self.thumbnail_fit = thumbnail_fit
        self.task_key = None
        self._counted_raw_slot = False
        self._cancelled = False
        self._lock = threading.Lock()
    
    def cancel(self):
        """標記任務為已取消（非阻塞）"""
        with self._lock:
            self._cancelled = True
    
    def is_cancelled(self) -> bool:
        """檢查任務是否已取消"""
        with self._lock:
            return self._cancelled
    
    def __lt__(self, other):
        """Priority queue ordering: enum priority, then lighter thumbnail work first."""
        if not isinstance(other, ImageLoadTask):
            return NotImplemented
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return _task_queue_weight(self) < _task_queue_weight(other)


def _task_queue_weight(task: ImageLoadTask) -> int:
    """Lower weight = scheduled sooner within the same Priority band."""
    stages = task.stages or set()
    if stages == {"thumbnail"}:
        return 0
    if stages <= {"thumbnail", "exif"}:
        return 1
    if "full" in stages:
        return 3
    return 2


class ImageLoadWorker(QRunnable):
    """可重用的工作線程"""
    
    def __init__(self, task: ImageLoadTask, manager: 'ImageLoadManager'):
        super().__init__()
        self.task = task
        self.manager = manager
        self._processor = None  # 延遲初始化，避免導入時創建

    def _has_active_current_task_elsewhere(self) -> bool:
        try:
            from image_load_manager import Priority
            with self.manager._queue_lock:
                for t in self.manager._active_tasks.values():
                    if (
                        getattr(t, "priority", None) == Priority.CURRENT
                        and t is not self.task
                        and not t.is_cancelled()
                    ):
                        return True
        except Exception:
            pass
        return False

    def _yield_if_needed(self) -> None:
        from image_load_manager import Priority
        if self.task.priority == Priority.CURRENT:
            return
        import time
        while self._has_active_current_task_elsewhere():
            if self.task.is_cancelled():
                break
            time.sleep(0.05)


    @staticmethod
    def _uses_display_preview_tier(task: ImageLoadTask) -> bool:
        """CURRENT single-view loads use ~1920px embedded preview, not 512px gallery grid."""
        if getattr(task, "gallery_thumbnail", False):
            return False
        if task.priority != Priority.CURRENT:
            return False
        if task.thumbnail_target_size is not None:
            return False
        stages = task.stages or set()
        if "full" in stages:
            return True
        if "thumbnail" not in stages:
            return False
        from common_image_loader import is_raw_file

        return is_raw_file(task.file_path)
    
    def _get_processor(self):
        """獲取處理器實例（延遲初始化）"""
        if self._processor is None:
            # Lazy import to avoid circular import issues
            from unified_image_processor import UnifiedImageProcessor
            self._processor = UnifiedImageProcessor()
        return self._processor
    
    def _emit_preview_failure(self, file_path: str, message: str) -> None:
        """Skip preview-only errors when a full decode stage will run in the same task."""
        if "full" in (self.task.stages or set()):
            return
        if self._safe_emit():
            self.manager.error_occurred.emit(file_path, message)

    def run(self):
        """執行任務"""
        worker_thread_local.priority = getattr(self.task, 'priority', None)
        worker_thread_local.task = self.task
        try:
            if self.task.priority == Priority.CURRENT:
                import logging

                enq = float(getattr(self.task, "_enqueue_ts", 0.0) or 0.0)
                if enq > 0.0:
                    logging.getLogger(__name__).info(
                        "[PIPE_T] queue-wait %.0fms %s stages=%s",
                        (time.time() - enq) * 1000.0,
                        os.path.basename(self.task.file_path),
                        sorted(self.task.stages or set()),
                    )
            if self.task.is_cancelled():
                if self._safe_emit():
                    self.manager._task_finished(self.task)
                return
            
            file_path = self.task.file_path
            self._yield_if_needed()
            stages = self.task.stages or set()
            if self._safe_emit() and not self.task.is_cancelled():
                self.manager.progress_updated.emit(file_path, "Loading image...")
            
            processor = self._get_processor()
            
            # COMBINED OPTIMIZATION: If both exif and thumbnail are needed, do them in one pass.
            if 'exif' in stages and 'thumbnail' in stages and not self.task.is_cancelled():
                self._yield_if_needed()
                if self._safe_emit():
                    self.manager.progress_updated.emit(file_path, "Reading metadata & preview...")
                
                allow_heavy_fallback = self._uses_display_preview_tier(self.task)
                exif_data, thumbnail = processor.process_metadata_and_thumbnail(
                    file_path,
                    allow_heavy_fallback=allow_heavy_fallback,
                    target_size=self.task.thumbnail_target_size
                )


                preview_tier = allow_heavy_fallback
                min_preview_dim = _min_acceptable_preview_dim(file_path)
                if preview_tier and thumbnail is not None and not self.task.is_cancelled():
                    if processor._is_raw_file(file_path):
                        if (
                            processor._preview_buffer_max_dim(thumbnail)
                            < min_preview_dim
                        ):
                            thumbnail = processor.ensure_display_tier_preview(
                                file_path, thumbnail
                            )
                        if (
                            processor._preview_buffer_max_dim(thumbnail)
                            < min_preview_dim
                            and not processor.is_libraw_unsupported(file_path)
                        ):
                            thumbnail = None
                if thumbnail is not None and not self.task.is_cancelled():
                    self._cache_gallery_thumbnail_for_indexing(
                        processor, file_path, thumbnail, exif_data
                    )
                if preview_tier:
                    if thumbnail is not None and not self.task.is_cancelled():
                        self._handle_thumbnail_result(file_path, thumbnail)
                    elif not self.task.is_cancelled():
                        self._emit_preview_failure(
                            file_path,
                            "Display-tier preview extraction failed",
                        )
                elif not self.task.is_cancelled():
                    if thumbnail is not None:
                        self._handle_thumbnail_result(file_path, thumbnail)
                    elif self._safe_emit():
                        self._emit_preview_failure(
                            file_path, "Thumbnail extraction returned None"
                        )

                if exif_data is not None and not self.task.is_cancelled():
                    if self._safe_emit():
                        self.manager.exif_data_ready.emit(file_path, exif_data)
                elif exif_data is None and not self.task.is_cancelled():
                    if self._safe_emit():
                        self.manager.exif_data_ready.emit(file_path, {})
            else:
                # Sequential or partial stages
                if 'exif' in stages and not self.task.is_cancelled():
                    self._yield_if_needed()
                    if self._safe_emit():
                        self.manager.progress_updated.emit(file_path, "Reading metadata...")
                    exif_data = processor.process_exif(file_path)
                    if exif_data is not None and not self.task.is_cancelled():
                        if self._safe_emit():
                            self.manager.exif_data_ready.emit(file_path, exif_data)
                    elif exif_data is None and not self.task.is_cancelled():
                        if self._safe_emit():
                            self.manager.exif_data_ready.emit(file_path, {})
                
                if 'thumbnail' in stages and not self.task.is_cancelled():
                    self._yield_if_needed()
                    if self._safe_emit():
                        self.manager.progress_updated.emit(file_path, "Extracting preview...")

                    allow_heavy_fallback = self._uses_display_preview_tier(self.task)
                    thumbnail = None
                    exif_data = None
                    if allow_heavy_fallback and "exif" not in stages:
                        cache = processor.cache
                        cached_exif = cache.get_exif(file_path)
                        cached_thumb = cache.get_preview(file_path)
                        if cached_thumb is None:
                            cached_thumb = cache.get_thumbnail(file_path)
                        fast_pair = processor._extract_raw_preview_before_full_exiftool(
                            file_path, cached_exif, cached_thumb
                        )
                        if fast_pair is not None:
                            exif_data, thumbnail = fast_pair
                    if thumbnail is None:
                        thumbnail = processor.process_thumbnail(
                            file_path,
                            allow_heavy_fallback=allow_heavy_fallback,
                            target_size=self.task.thumbnail_target_size,
                        )
                    if allow_heavy_fallback and "exif" not in stages and exif_data is None:
                        deferred = processor.cache.get_exif(file_path)
                        if deferred and deferred.get("minimal_preview_exif"):
                            exif_data = deferred
                    if (
                        allow_heavy_fallback
                        and exif_data is not None
                        and not self.task.is_cancelled()
                    ):
                        if self._safe_emit():
                            self.manager.exif_data_ready.emit(file_path, exif_data)
                    if allow_heavy_fallback and processor._is_raw_file(file_path):
                        min_preview_dim = _min_acceptable_preview_dim(file_path)
                        if (
                            thumbnail is not None
                            and processor._preview_buffer_max_dim(thumbnail)
                            < min_preview_dim
                        ):
                            thumbnail = processor.ensure_display_tier_preview(
                                file_path, thumbnail
                            )
                        if (
                            thumbnail is not None
                            and processor._preview_buffer_max_dim(thumbnail)
                            < min_preview_dim
                            and not processor.is_libraw_unsupported(file_path)
                        ):
                            thumbnail = None
                    if thumbnail is not None and not self.task.is_cancelled():
                        cache_exif = exif_data or processor.cache.get_exif(file_path)
                        self._cache_gallery_thumbnail_for_indexing(
                            processor, file_path, thumbnail, cache_exif
                        )
                        self._handle_thumbnail_result(file_path, thumbnail)
                    elif not self.task.is_cancelled():
                        self._emit_preview_failure(
                            file_path, "Thumbnail extraction returned None"
                        )
            
            # 處理完整圖像（只在需要時）
            if 'full' in stages and not self.task.is_cancelled():
                self._yield_if_needed()
                if self.task.use_full_resolution:
                    if self._safe_emit():
                        self.manager.progress_updated.emit(file_path, "Loading full resolution...")
                else:
                    if self._safe_emit():
                        self.manager.progress_updated.emit(file_path, "Processing image...")

                result = processor.process_full_image(
                    file_path,
                    use_full_resolution=self.task.use_full_resolution,
                    executor=self.manager._process_pool if self._safe_emit() else None
                )
                if result is not None and not self.task.is_cancelled():
                    if self._safe_emit():
                        if isinstance(result, np.ndarray):
                            self.manager.image_ready.emit(file_path, result)
                        elif isinstance(result, QPixmap):
                            self.manager.pixmap_ready.emit(file_path, result)
                elif (
                    "full" in stages
                    and not self.task.is_cancelled()
                    and self.task.priority == Priority.CURRENT
                ):
                    if self._safe_emit():
                        from common_image_loader import is_raw_file

                        if is_raw_file(file_path):
                            msg = (
                                "Could not decode image data (unsupported or corrupt RAW, "
                                "or no usable embedded JPEG)."
                            )
                        else:
                            msg = (
                                "Could not decode image file "
                                "(file may be corrupt or too large to load at full resolution)."
                            )
                        self.manager.error_occurred.emit(file_path, msg)

            # 發送完成信號
            if self._safe_emit() and not self.task.is_cancelled():
                self.manager.progress_updated.emit(file_path, "Processing complete")
                    
        except Exception as e:
            if self._safe_emit():
                self.manager.error_occurred.emit(file_path, str(e))
        finally:
            if self._safe_emit():
                self.manager._task_finished(self.task)
            worker_thread_local.priority = None
            worker_thread_local.task = None

    def _handle_thumbnail_result(self, file_path, thumbnail):
        """Internal helper to process and emit thumbnail results."""
        tgt = self.task.thumbnail_target_size
        if tgt is not None and isinstance(tgt, QSize) and tgt.isValid():
            try:
                if isinstance(thumbnail, QImage):
                    qimg = thumbnail
                else:
                    arr = np.ascontiguousarray(thumbnail)
                    h, w = arr.shape[:2]
                    bpl = arr.strides[0]
                    qimg = QImage(arr.data, w, h, bpl, QImage.Format.Format_RGB888).copy()
                if qimg.isNull():
                    if self._safe_emit():
                        self.manager.error_occurred.emit(file_path, "Created QImage was null")
                    return
                
                if self.task.thumbnail_fit == "crop":
                    scaled = qimg.scaled(tgt, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
                    x = max(0, (scaled.width() - tgt.width()) // 2)
                    y = max(0, (scaled.height() - tgt.height()) // 2)
                    qimg_out = scaled.copy(x, y, tgt.width(), tgt.height())
                else:
                    qimg_out = qimg.scaled(tgt, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                
                if self._safe_emit():
                    self.manager.thumbnail_ready.emit(file_path, qimg_out)
                    return
            except Exception:
                pass

        if self._safe_emit():
            self.manager.thumbnail_ready.emit(file_path, thumbnail)

    def _cache_gallery_thumbnail_for_indexing(
        self, processor, file_path: str, thumbnail, exif_data
    ) -> None:
        """Publish gallery extractions to ImageCache for semantic/face indexing reuse."""
        if thumbnail is None:
            return
        stages = self.task.stages if self.task.stages is not None else set()
        if "thumbnail" not in stages:
            return
        processor.cache_index_source_mipmap_tiers(
            file_path, thumbnail, exif_data=exif_data
        )

    def _maybe_cache_preview_first_warm(
        self, processor, file_path: str, thumbnail, exif_data
    ) -> None:
        """Backward-compatible alias for gallery → global cache publish."""
        self._cache_gallery_thumbnail_for_indexing(
            processor, file_path, thumbnail, exif_data
        )

    def _safe_emit(self) -> bool:
        """Verify the manager still exists before emitting signals from background thread."""
        try:
            from PyQt6 import sip
            if self.manager is None or sip.isdeleted(self.manager):
                return False
            return True
        except (ImportError, AttributeError):
            # Fallback if sip is not available or manager is None
            return self.manager is not None


class ImageLoadManager(QObject):
    """統一管理所有圖像加載任務的工作隊列管理器（單例模式）"""
    
    # 信號定義
    # NOTE: Can emit np.ndarray (base thumbnail) or QImage (pre-scaled/cropped) for smooth UI.
    thumbnail_ready = pyqtSignal(str, object)  # file_path, thumbnail (np.ndarray or QImage)
    image_ready = pyqtSignal(str, np.ndarray)  # file_path, full_image
    pixmap_ready = pyqtSignal(str, QPixmap)  # file_path, pixmap
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
        self._thread_pool = QThreadPool()
        
        self._stopped = False  # Flag to stop scheduling new tasks
        self._gallery_warmup_throttle_depth = 0
        self._io_pressure_throttle_depth = 0
        self._indexing_throttle_depth = 0
        # Semantic indexing throttle is toggled from the background index worker
        # thread as well as the UI thread (face scan), so guard the depth counter.
        self._indexing_throttle_lock = threading.Lock()
        
        # Throttled workers on macOS external drives to prevent IO overload crashes
        self.update_volume_throttling(os.getcwd())
        if max_workers != 4:
            self._thread_pool.setMaxThreadCount(max_workers)

        # PROCESS POOL (optional): on Windows debug/startup paths, process spawn can
        # re-import heavy modules and hurt first-load latency. Keep it opt-in.
        # PROCESS POOL: LibRaw postprocess in worker processes (multi-core on Windows).
        from common_image_loader import (
            process_pool_worker_count,
            raw_concurrent_load_limit,
            use_raw_process_pool,
        )

        use_process_pool = use_raw_process_pool()
        self._process_pool = None
        if use_process_pool:
            import logging
            logging.getLogger(__name__).info(
                "[LOAD] LibRaw process pool enabled (%d workers)",
                process_pool_worker_count(),
            )
            self._process_pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=process_pool_worker_count()
            )
        else:
            import logging
            logging.getLogger(__name__).info("[LOAD] LibRaw process pool disabled")
        
        self._active_tasks: Dict[Tuple, ImageLoadTask] = {}
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

    def update_volume_throttling(self, folder_path: Optional[str]) -> None:
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
            try:
                default_workers = max(4, int(env_workers))
            except ValueError:
                pass
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

        # If no throttle is currently active, apply directly
        is_throttled = (
            (getattr(self, "_io_pressure_throttle_depth", 0) or 0) > 0
            or (getattr(self, "_gallery_warmup_throttle_depth", 0) or 0) > 0
            or (getattr(self, "_indexing_throttle_depth", 0) or 0) > 0
        )
        if not is_throttled:
            self._thread_pool.setMaxThreadCount(default_workers)
        
        # Save baseline to be restored when throttles exit
        self._pressure_saved_max_threads = default_workers
        self._warmup_saved_max_threads = default_workers
        self._indexing_saved_max_threads = default_workers
        
        self._raw_load_limit = raw_concurrent_load_limit(folder_path)

    def prime_volume_speed_async(
        self, folder_path: Optional[str], sample_path: Optional[str] = None
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
                invalidate_volume_speed_cache,
                probe_volume_speed,
                record_volume_speed,
                volume_speed_tier,
                _windows_drive_is_fixed_local,
            )

            # Let fast-open / sort probes finish so we measure disk not page cache.
            time.sleep(2.0)

            candidates = self._pick_probe_samples(
                folder_path, exclude=sample_path, limit=8
            )
            if not candidates and sample_path and os.path.isfile(sample_path):
                candidates = [sample_path]

            best_mbps: Optional[float] = None
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
            try:
                self.update_volume_throttling(folder_path)
            except Exception:
                pass
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
        folder_path: Optional[str],
        *,
        exclude: Optional[str] = None,
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
    def _pick_probe_sample(cls, folder_path: Optional[str]) -> Optional[str]:
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
        pressure_max = _env_int("RAWVIEWER_IO_PRESSURE_MAX_WORKERS", 6, minimum=2)
        self._thread_pool.setMaxThreadCount(
            min(pressure_max, self._pressure_saved_max_threads)
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
        saved_raw = getattr(self, "_pressure_saved_raw_limit", None)
        if saved_raw is not None:
            self._raw_load_limit = saved_raw

    def enter_gallery_warmup_throttle(self) -> None:
        """Lower worker/RAW concurrency while gallery tiles are first painting."""
        self._gallery_warmup_throttle_depth = (
            int(getattr(self, "_gallery_warmup_throttle_depth", 0) or 0) + 1
        )
        if self._gallery_warmup_throttle_depth != 1:
            return
        self._warmup_saved_max_threads = self._thread_pool.maxThreadCount()
        warmed_max = _env_int("RAWVIEWER_GALLERY_WARMUP_MAX_WORKERS", 24, minimum=2)
        self._thread_pool.setMaxThreadCount(
            min(warmed_max, self._warmup_saved_max_threads)
        )
        self._warmup_saved_raw_limit = self._raw_load_limit
        self._raw_load_limit = min(6, int(self._raw_load_limit or 6))

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
        saved_raw = getattr(self, "_warmup_saved_raw_limit", None)
        if saved_raw is not None:
            self._raw_load_limit = saved_raw

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
            indexing_max = _env_int("RAWVIEWER_INDEXING_MAX_WORKERS", 6, minimum=2)
            self._thread_pool.setMaxThreadCount(
                min(indexing_max, self._indexing_saved_max_threads)
            )
            self._indexing_saved_raw_limit = self._raw_load_limit
            self._raw_load_limit = min(1, int(self._raw_load_limit or 1))
            import logging

            logging.getLogger(__name__).info(
                "[LOAD] Semantic indexing throttle ON (max_workers=%d raw_limit=%d)",
                self._thread_pool.maxThreadCount(),
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
            saved_raw = getattr(self, "_indexing_saved_raw_limit", None)
            if saved_raw is not None:
                self._raw_load_limit = saved_raw
            import logging

            logging.getLogger(__name__).info("[LOAD] Semantic indexing throttle FORCE OFF")
        finally:
            if lock is not None:
                lock.release()

    def load_image(self, file_path: str, priority: Priority = Priority.CURRENT,
                   cancel_existing: bool = True, use_full_resolution: bool = False,
                   stages: Optional[set] = None,
                   thumbnail_target_size: Optional[QSize] = None,
                   thumbnail_fit: str = "crop",
                   bypass_cache: bool = False,
                   gallery_thumbnail: bool = False):
        """請求加載圖像"""
        # Don't accept new tasks if stopped
        if self._stopped:
            return
            
        # 取消現有任務（如果需要）
        if cancel_existing:
            self.cancel_task(file_path)
        
        # 檢查快取（memory-only, stage-aware）
        if not bypass_cache and self._check_cache(
            file_path,
            use_full_resolution,
            stages=stages,
            gallery_thumbnail=gallery_thumbnail,
        ):
            return

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
        self._work_queue.put(task)
        self._schedule_next_task()
    
    def has_active_work_for_path(self, file_path: str) -> bool:
        """True when a load task for this path is queued or running."""
        if not file_path:
            return False
        with self._queue_lock:
            return bool(self._task_keys_by_path.get(file_path))

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

    def cancel_task(self, file_path: str):
        """取消任務（非阻塞）"""
        with self._queue_lock:
            keys = list(self._task_keys_by_path.get(file_path, ()))
            for key in keys:
                task = self._active_tasks.pop(key, None)
                if task is not None:
                    task.cancel()
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
            self._active_raw_tasks = sum(
                1
                for t in self._active_tasks.values()
                if getattr(t, "_counted_raw_slot", False)
            )
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
            self._active_raw_tasks = sum(
                1
                for t in self._active_tasks.values()
                if getattr(t, "_counted_raw_slot", False)
            )

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
            self._active_raw_tasks = sum(
                1
                for t in self._active_tasks.values()
                if getattr(t, "_counted_raw_slot", False)
            )
            self._compact_work_queue()

    def cancel_tasks_by_priority(self, priority: Priority) -> None:
        """Cancel all tasks matching the specified priority."""
        with self._queue_lock:
            for key in list(self._active_tasks.keys()):
                task = self._active_tasks.get(key)
                if task is None or task.priority != priority:
                    continue
                task.cancel()
                self._active_tasks.pop(key, None)
                file_path = getattr(task, 'file_path', None)
                if file_path and file_path in self._task_keys_by_path:
                    self._task_keys_by_path[file_path].discard(key)
                    if not self._task_keys_by_path.get(file_path):
                        self._task_keys_by_path.pop(file_path, None)
            self._compact_work_queue()
    
    def cancel_all_tasks(self):
        """取消所有任務"""
        with self._queue_lock:
            for task in self._active_tasks.values():
                task.cancel()
            self._active_tasks.clear()
            self._task_keys_by_path.clear()
            # Reset RAW slot counter when dropping all tracked tasks; otherwise old
            # running tasks can leak the counter and block future RAW scheduling.
            self._active_raw_tasks = 0
            self._compact_work_queue()

    def cancel_all_tasks_except(self, keep_path: Optional[str]) -> None:
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
                fp = getattr(task, "file_path", None)
                self._active_tasks.pop(key, None)
                if fp and fp in self._task_keys_by_path:
                    self._task_keys_by_path[fp].discard(key)
                    if not self._task_keys_by_path[fp]:
                        self._task_keys_by_path.pop(fp, None)
            # In-flight RAW slots for kept tasks stay counted; everything else is gone.
            self._active_raw_tasks = sum(
                1
                for t in self._active_tasks.values()
                if getattr(t, "_counted_raw_slot", False)
            )
            self._compact_work_queue()

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
            self._active_tasks.clear()
            self._task_keys_by_path.clear()
            self._work_queue = queue.PriorityQueue()
            self._active_raw_tasks = 0

    def shutdown(self):
        """關閉管理器並清理資源"""
        self._stopped = True
        watchdog = getattr(self, '_dispatch_watchdog', None)
        if watchdog is not None:
            try:
                watchdog.stop()
            except Exception:
                pass
        if hasattr(self, '_process_pool') and self._process_pool:
            self._process_pool.shutdown(wait=False)

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
                       stages: Optional[set], thumbnail_target_size: Optional[QSize],
                       thumbnail_fit: str) -> Tuple:
        """Build a stable key for de-duplicating queued/running work."""
        wanted = tuple(sorted(stages if stages is not None else {'thumbnail', 'exif', 'full'}))
        if thumbnail_target_size is not None and isinstance(thumbnail_target_size, QSize) and thumbnail_target_size.isValid():
            target_key = (thumbnail_target_size.width(), thumbnail_target_size.height(), thumbnail_fit)
        else:
            target_key = None
        return (file_path, use_full_resolution, wanted, target_key)

    def _task_finished(self, task: ImageLoadTask):
        """Remove completed/cancelled work from active map and keep the queue moving."""
        with self._queue_lock:
            key = getattr(task, 'task_key', None)
            if key in self._active_tasks and self._active_tasks[key] is task:
                del self._active_tasks[key]
                self._task_keys_by_path[task.file_path].discard(key)
                if not self._task_keys_by_path[task.file_path]:
                    self._task_keys_by_path.pop(task.file_path, None)
            if getattr(task, "_counted_raw_slot", False):
                self._active_raw_tasks = max(0, self._active_raw_tasks - 1)
                task._counted_raw_slot = False
        self.task_completed.emit(task.file_path)
        self._schedule_next_task()

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
        stages: Optional[set] = None,
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
        libraw_first = use_libraw_consistent_preview_first()
        wanted = stages if stages is not None else {'thumbnail', 'exif', 'full'}

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
                emit = (
                    self._emit_cached_result_now
                    if preview_only
                    else self._emit_cached_result_later
                )
                emit(self.thumbnail_ready, file_path, buf)
                if preview_only:
                    exif_data = cache.get_exif(file_path)
                    if exif_data is not None:
                        emit(self.exif_data_ready, file_path, exif_data)
                return True

            if gallery_thumbnail:
                grid_buf = cache.grid_cache.get(file_path)
                if _try_emit_display_thumb(grid_buf):
                    if preview_only:
                        return True

            preview_buf = cache.preview_cache.get(file_path)
            if _try_emit_display_thumb(preview_buf):
                if preview_only:
                    return True
            thumb = cache.thumbnail_cache.get(file_path)
            if _try_emit_display_thumb(thumb):
                if preview_only:
                    return True
                # Not terminal if callers also requested EXIF/full image work.

        # 2) EXIF stage: check memory + persistent cache and emit if found
        if 'exif' in wanted:
            exif_data = cache.get_exif(file_path)
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
                            h, w = preview.shape[:2]
                            if image_covers_sensor_resolution(w, h, exif_data):
                                self._emit_cached_result_later(
                                    self.image_ready, file_path, preview
                                )
                                return True
                        full_img = cache.get_full_image(file_path)
                        if full_img is not None:
                            self._emit_cached_result_later(self.image_ready, file_path, full_img)
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
        exif_cached = cache.get_exif(file_path)
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
                    full_ok = cache.full_image_cache.get(file_path) is not None
                elif libraw_first:
                    full_ok = cache.get_full_image(file_path) is not None
                else:
                    full_ok = cache.preview_cache.get(file_path) is not None
            else:
                px = cache.pixmap_cache.get(file_path)
                full_ok = px is not None and not px.isNull()

        return thumb_ok and exif_ok and full_ok
    
    def _schedule_next_task(self):
        """調度下一個任務到線程池，實現 RAW 限制"""
        if self._stopped:
            return
            
        from common_image_loader import io_pressure_active, is_raw_file
        pressure = io_pressure_active()
        with self._queue_lock:
            if self._work_queue.empty():
                return
            
            try:
                # We want to fill the thread pool, but limit heavy RAW tasks
                # JPEGs can always proceed if threads are free.
                deferred_raw_tasks = []
                deferred_prefetch_tasks = []
                while (
                    not self._work_queue.empty()
                    and self._thread_pool.activeThreadCount() < self._thread_pool.maxThreadCount()
                ):
                    task = self._work_queue.get_nowait()

                    if task.is_cancelled():
                        key = getattr(task, 'task_key', None)
                        if key in self._active_tasks and self._active_tasks[key] is task:
                            del self._active_tasks[key]
                            self._task_keys_by_path[task.file_path].discard(key)
                            if not self._task_keys_by_path[task.file_path]:
                                self._task_keys_by_path.pop(task.file_path, None)
                        continue

                    if pressure and task.priority != Priority.CURRENT:
                        active_prefetch = sum(
                            1 for t in self._active_tasks.values()
                            if getattr(t, "priority", None) != Priority.CURRENT and not t.is_cancelled()
                        )
                        if active_prefetch >= 1:
                            deferred_prefetch_tasks.append(task)
                            continue

                    is_raw = is_raw_file(task.file_path)
                    # HEAVY TASK CHECK: Only throttle RAW tasks that perform full-resolution processing.
                    # Metadata and thumbnail extraction (extract_thumb) are lightweight enough to bypass
                    # the 4-slot limit, preventing gallery starvation in mixed folders.
                    is_heavy = is_raw and 'full' in (task.stages or set())
                    
                    heavy_limit = self._raw_load_limit
                    if is_heavy and task.priority != Priority.CURRENT:
                        # Reserve one RAW slot for the user's current image:
                        # an in-flight rawpy decode cannot be aborted, so a
                        # fully prefetch-occupied pool adds up to a whole
                        # decode (~0.5s) to every navigation's latency.
                        heavy_limit = max(1, heavy_limit - 1)
                    if is_heavy and self._active_raw_tasks >= heavy_limit:
                        # Keep throttled heavy RAW aside for now and continue scanning queue.
                        deferred_raw_tasks.append(task)
                        continue
                    
                    if is_heavy:
                        self._active_raw_tasks += 1
                        task._counted_raw_slot = True

                    worker = ImageLoadWorker(task, self)
                    self._thread_pool.start(worker)
                for deferred in deferred_prefetch_tasks:
                    self._work_queue.put(deferred)
                for deferred in deferred_raw_tasks:
                    self._work_queue.put(deferred)
            except queue.Empty:
                pass
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取管理器統計信息"""
        with self._queue_lock:
            return {
                'queue_size': self._work_queue.qsize(),
                'active_tasks': len(self._active_tasks),
                'active_threads': self._thread_pool.activeThreadCount(),
                'max_threads': self._thread_pool.maxThreadCount()
            }


# 全局單例實例（使用函數級單例，類似 ImageCache）
_global_manager: Optional[ImageLoadManager] = None
_manager_lock = threading.Lock()


def get_image_load_manager(max_workers: int = 4) -> ImageLoadManager:
    """獲取全局圖像加載管理器實例（函數級單例模式）"""
    global _global_manager
    if _global_manager is None:
        with _manager_lock:
            if _global_manager is None:
                _global_manager = ImageLoadManager(max_workers)
    return _global_manager

