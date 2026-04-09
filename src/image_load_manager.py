"""
統一圖像加載管理器 - 基於工作隊列和線程池的架構

這個模組實現了重構提案中的 ImageLoadManager，使用工作隊列和線程池
來管理所有圖像加載任務，避免頻繁創建/銷毀線程的開銷。
"""

import os
import threading
import queue
from enum import Enum
from typing import Optional, Dict, Any, Tuple
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, QRunnable, QThreadPool, QSize, Qt
from PyQt6.QtGui import QPixmap, QImage

import concurrent.futures
from image_cache import get_image_cache
# UnifiedImageProcessor will be imported lazily to avoid circular import issues


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
                 thumbnail_fit: str = "crop"):
        self.file_path = file_path
        self.priority = priority
        self.use_full_resolution = use_full_resolution
        # Stages: {'thumbnail', 'exif', 'full'}.
        # Gallery should typically request only {'thumbnail'} (and optionally 'exif').
        self.stages = stages if stages is not None else {'thumbnail', 'exif', 'full'}
        # If provided, the thumbnail will be scaled/cropped in worker thread to this size.
        # This keeps UI thread light for smooth pixel-level scrolling.
        self.thumbnail_target_size = thumbnail_target_size
        self.thumbnail_fit = thumbnail_fit
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
        """用於優先級隊列排序"""
        if not isinstance(other, ImageLoadTask):
            return NotImplemented
        return self.priority.value < other.priority.value


class ImageLoadWorker(QRunnable):
    """可重用的工作線程"""
    
    def __init__(self, task: ImageLoadTask, manager: 'ImageLoadManager'):
        super().__init__()
        self.task = task
        self.manager = manager
        self._processor = None  # 延遲初始化，避免導入時創建
    
    def _get_processor(self):
        """獲取處理器實例（延遲初始化）"""
        if self._processor is None:
            # Lazy import to avoid circular import issues
            from unified_image_processor import UnifiedImageProcessor
            self._processor = UnifiedImageProcessor()
        return self._processor
    
    def run(self):
        """執行任務"""
        if self.task.is_cancelled():
            return
        
        file_path = self.task.file_path
        
        try:
            stages = self.task.stages or set()
            # 發送進度更新
            if self._safe_emit() and not self.task.is_cancelled():
                self.manager.progress_updated.emit(file_path, "Loading image...")
            
            # 獲取處理器（延遲初始化）
            processor = self._get_processor()
            
            # 處理縮圖
            if 'thumbnail' in stages and not self.task.is_cancelled():
                if self._safe_emit():
                    self.manager.progress_updated.emit(file_path, "Extracting preview...")
                thumbnail = processor.process_thumbnail(file_path)
                if thumbnail is not None and not self.task.is_cancelled():
                    # Optional: pre-scale/crop thumbnail in worker thread (emit QImage)
                    tgt = self.task.thumbnail_target_size
                    if tgt is not None and isinstance(tgt, QSize) and tgt.isValid():
                        try:
                            arr = np.ascontiguousarray(thumbnail)
                            if arr is not None and hasattr(arr, 'shape'):
                                h, w = arr.shape[:2]
                                bpl = arr.strides[0]
                                qimg = QImage(arr.data, w, h, bpl, QImage.Format.Format_RGB888).copy()
                            else:
                                raise AttributeError("Invalid thumbnail array structure")
                            if self.task.thumbnail_fit == "crop":
                                scaled = qimg.scaled(
                                    tgt,
                                    aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                    transformMode=Qt.TransformationMode.SmoothTransformation,
                                )
                                x = max(0, (scaled.width() - tgt.width()) // 2)
                                y = max(0, (scaled.height() - tgt.height()) // 2)
                                qimg_out = scaled.copy(x, y, tgt.width(), tgt.height())
                            else:
                                qimg_out = qimg.scaled(
                                    tgt,
                                    aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio,
                                    transformMode=Qt.TransformationMode.SmoothTransformation,
                                )
                            if self._safe_emit():
                                self.manager.thumbnail_ready.emit(file_path, qimg_out)
                        except Exception:
                            # Fallback: emit numpy thumbnail
                            if self._safe_emit():
                                self.manager.thumbnail_ready.emit(file_path, thumbnail)
                    else:
                        if self._safe_emit():
                            self.manager.thumbnail_ready.emit(file_path, thumbnail)
            
            # 處理 EXIF 數據
            if 'exif' in stages and not self.task.is_cancelled():
                if self._safe_emit():
                    self.manager.progress_updated.emit(file_path, "Reading metadata...")
                exif_data = processor.process_exif(file_path)
                if exif_data and not self.task.is_cancelled():
                    if self._safe_emit():
                        self.manager.exif_data_ready.emit(file_path, exif_data)
            
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
                        self.manager.error_occurred.emit(
                            file_path,
                            "Could not decode image data (unsupported or corrupt RAW, "
                            "or no usable embedded JPEG).",
                        )

            # 發送完成信號
            if self._safe_emit() and not self.task.is_cancelled():
                self.manager.progress_updated.emit(file_path, "Processing complete")
                    
        except Exception as e:
            if self._safe_emit():
                self.manager.error_occurred.emit(file_path, str(e))
        finally:
            # 任務完成，調度下一個
            if self._safe_emit():
                self.manager._schedule_next_task()

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
    progress_updated = pyqtSignal(str, str)  # file_path, status_message
    exif_data_ready = pyqtSignal(str, dict)  # file_path, exif_data
    
    def __init__(self, max_workers: int = 4):
        """初始化管理器"""
        # CRITICAL: 對於 QObject 子類，必須在最開始就調用 super().__init__()
        # 不能在調用 super().__init__() 之前訪問任何實例屬性（包括 hasattr）
        import sys
        print("[ImageLoadManager.__init__] Starting initialization...", flush=True)
        
        # Ensure QApplication exists before initializing QObject
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            print("[ImageLoadManager.__init__] ERROR: QApplication instance not found!", file=sys.stderr, flush=True)
            raise RuntimeError("QApplication must be created before ImageLoadManager")
        
        print("[ImageLoadManager.__init__] Calling super().__init__(app)...", flush=True)
        try:
            # Root the manager in the application's lifecycle to prevent premature deletion.
            super().__init__(app)
            print("[ImageLoadManager.__init__] super().__init__(app) completed", flush=True)
        except Exception as e:
            print(f"[ImageLoadManager.__init__] ERROR in super().__init__(): {e}", file=sys.stderr, flush=True)
            import traceback
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            raise
        
        # 避免重複初始化實例變量
        if hasattr(self, '_initialized') and self._initialized:
            print("[ImageLoadManager.__init__] Already initialized, skipping instance variables", flush=True)
            return
        
        self._work_queue = queue.PriorityQueue()
        self._thread_pool = QThreadPool()
        
        # INCREASED CONCURRENCY: Scale with CPU cores
        # For I/O bound tasks (thumbnails), we can have many threads
        core_count = os.cpu_count() or 4
        default_workers = max(12, core_count * 2) 
        if max_workers == 4: # If default was used, upgrade it
            max_workers = default_workers
            
        self._thread_pool.setMaxThreadCount(max_workers)
        
        # PROCESS POOL: For heavy RAW processing to bypass GIL
        self._process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=max(2, core_count // 2))
        
        self._active_tasks: Dict[str, ImageLoadTask] = {}
        self._cache = get_image_cache()
        self._queue_lock = threading.Lock()
        self._stopped = False  # Flag to stop scheduling new tasks
        self._initialized = True
    
    def load_image(self, file_path: str, priority: Priority = Priority.CURRENT,
                   cancel_existing: bool = True, use_full_resolution: bool = False,
                   stages: Optional[set] = None,
                   thumbnail_target_size: Optional[QSize] = None,
                   thumbnail_fit: str = "crop"):
        """請求加載圖像"""
        # Don't accept new tasks if stopped
        if self._stopped:
            return
            
        # 取消現有任務（如果需要）
        if cancel_existing:
            self.cancel_task(file_path)
        
        # 檢查快取（memory-only, stage-aware）
        if self._check_cache(file_path, use_full_resolution, stages=stages):
            return
        
        # 創建任務
        task = ImageLoadTask(
            file_path,
            priority,
            use_full_resolution,
            stages=stages,
            thumbnail_target_size=thumbnail_target_size,
            thumbnail_fit=thumbnail_fit
        )
        with self._queue_lock:
            self._active_tasks[file_path] = task
        
        # 添加到工作隊列
        self._work_queue.put(task)
        self._schedule_next_task()
    
    def cancel_task(self, file_path: str):
        """取消任務（非阻塞）"""
        with self._queue_lock:
            if file_path in self._active_tasks:
                self._active_tasks[file_path].cancel()
                del self._active_tasks[file_path]
    
    def cancel_all_tasks(self):
        """取消所有任務"""
        with self._queue_lock:
            for task in self._active_tasks.values():
                task.cancel()
            self._active_tasks.clear()
        
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
            self._work_queue = queue.PriorityQueue()

    def shutdown(self):
        """關閉管理器並清理資源"""
        self._stopped = True
        if hasattr(self, '_process_pool') and self._process_pool:
            self._process_pool.shutdown(wait=False)
    
    def _check_cache(self, file_path: str, use_full_resolution: bool, stages: Optional[set] = None) -> bool:
        """檢查快取，如果存在則直接發送信號（只檢查記憶體快取以避免阻塞 UI）"""
        from common_image_loader import is_raw_file

        is_raw = is_raw_file(file_path)
        wanted = stages if stages is not None else {'thumbnail', 'exif', 'full'}

        cache = self._cache
        any_terminal_hit = False

        # 1) Thumbnail stage: memory-only thumbnail cache (numpy)
        if 'thumbnail' in wanted:
            thumb = cache.thumbnail_cache.get(file_path)
            if thumb is not None:
                self.thumbnail_ready.emit(file_path, thumb)
                # Not terminal; we may still want 'full'

        # 2) EXIF stage: EXIF is persisted; but this check must remain memory-only.
        # We skip emitting EXIF from cache here to avoid potential disk I/O.

        # 3) Full stage: treat RAW preview (1920) as "full" when use_full_resolution=False
        if 'full' in wanted:
            if is_raw:
                if use_full_resolution:
                    full_img = cache.full_image_cache.get(file_path)
                    if full_img is not None:
                        self.image_ready.emit(file_path, full_img)
                        return True
                else:
                    # Preview cache is memory-only LRU
                    preview = cache.preview_cache.get(file_path)
                    if preview is not None:
                        self.image_ready.emit(file_path, preview)
                        return True
                # For RAW, do not use pixmap cache here
            else:
                pixmap = cache.pixmap_cache.get(file_path)
                if pixmap is not None and not pixmap.isNull():
                    self.pixmap_ready.emit(file_path, pixmap)
                    return True

        return any_terminal_hit
    
    def _schedule_next_task(self):
        """調度下一個任務到線程池"""
        # Don't schedule new tasks if stopped
        if self._stopped:
            return
            
        with self._queue_lock:
            if self._work_queue.empty():
                return
            
            try:
                # Fill available thread pool capacity, skipping cancelled tasks.
                while (
                    not self._work_queue.empty()
                    and self._thread_pool.activeThreadCount() < self._thread_pool.maxThreadCount()
                ):
                    task = self._work_queue.get_nowait()

                    # Skip cancelled tasks quickly (they still sit in PriorityQueue)
                    if task.is_cancelled():
                        if task.file_path in self._active_tasks:
                            del self._active_tasks[task.file_path]
                        continue

                    worker = ImageLoadWorker(task, self)
                    self._thread_pool.start(worker)
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

