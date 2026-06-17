"""
統一圖像加載管理器 - 基於工作隊列和線程池的架構

這個模組實現了重構提案中的 ImageLoadManager，使用工作隊列和線程池
來管理所有圖像加載任務，避免頻繁創建/銷毀線程的開銷。
"""

import os
import threading
import queue
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
        if self.task.is_cancelled():
            if self._safe_emit():
                self.manager._task_finished(self.task)
            return
        
        file_path = self.task.file_path
        try:
            stages = self.task.stages or set()
            if self._safe_emit() and not self.task.is_cancelled():
                self.manager.progress_updated.emit(file_path, "Loading image...")
            
            processor = self._get_processor()
            
            # COMBINED OPTIMIZATION: If both exif and thumbnail are needed, do them in one pass.
            if 'exif' in stages and 'thumbnail' in stages and not self.task.is_cancelled():
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
            # 任務完成，調度下一個
            if self._safe_emit():
                self.manager._task_finished(self.task)

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
        
        # INCREASED CONCURRENCY: Scale with CPU cores (embedded JPEG thumbnails are I/O-light).
        core_count = os.cpu_count() or 4
        default_workers = max(16, core_count * 2)
        env_workers = os.environ.get("RAWVIEWER_LOAD_MAX_WORKERS", "").strip()
        if env_workers:
            try:
                default_workers = max(4, int(env_workers))
            except ValueError:
                pass
        if max_workers == 4:  # If default was used, upgrade it
            max_workers = default_workers
            
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
        
        # RAW throttling: Limit concurrent heavy RAW decodes
        self._raw_load_limit = raw_concurrent_load_limit()
        self._active_raw_tasks = 0
        
        self._stopped = False  # Flag to stop scheduling new tasks
        self._gallery_warmup_throttle_depth = 0
        self._io_pressure_throttle_depth = 0
        self._initialized = True

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
        warmed_max = _env_int("RAWVIEWER_GALLERY_WARMUP_MAX_WORKERS", 8, minimum=2)
        self._thread_pool.setMaxThreadCount(
            min(warmed_max, self._warmup_saved_max_threads)
        )
        self._warmup_saved_raw_limit = self._raw_load_limit
        self._raw_load_limit = min(2, int(self._raw_load_limit or 2))

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
        
        # 清空工作隊列
        while not self._work_queue.empty():
            try:
                self._work_queue.get_nowait()
            except queue.Empty:
                break
        
        # NOTE: Do not shutdown pools here. This is used for folder switches and UI navigation.
        # Use shutdown() only when the application is closing.

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
        if hasattr(self, '_process_pool') and self._process_pool:
            self._process_pool.shutdown(wait=False)

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
            preview_buf = cache.preview_cache.get(file_path)
            if preview_buf is not None and not _skip_low_res_memory_thumb_for_display_tier(
                preview_buf,
                wanted,
                use_full_resolution,
                gallery_thumbnail=gallery_thumbnail,
            ):
                return True
            thumb = cache.thumbnail_cache.get(file_path)
            return thumb is not None and not _skip_low_res_memory_thumb_for_display_tier(
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
                    
                    if is_heavy and self._active_raw_tasks >= self._raw_load_limit:
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

