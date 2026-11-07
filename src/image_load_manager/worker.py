
"""ImageLoadWorker: reusable QRunnable that executes one ImageLoadTask."""
from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from image_load_manager import ImageLoadManager
from PyQt6.QtCore import QRunnable, QSize, Qt
from PyQt6.QtGui import QImage, QPixmap

from .helpers import (
    Priority,
    _env_int,
    _has_running_current_task,
    _min_acceptable_preview_dim,
    _preview_is_best_possible,
    _task_queue_weight,
    worker_thread_local,
)


class ImageLoadTask:
    """圖像加載任務"""

    def __init__(self, file_path: str, priority: Priority = Priority.CURRENT,
                 use_full_resolution: bool = False,
                 stages: set | None = None,
                 thumbnail_target_size: QSize | None = None,
                 thumbnail_fit: str = "crop",
                 gallery_thumbnail: bool = False):
        self.file_path = file_path
        self.priority = priority
        self.use_full_resolution = use_full_resolution
        self.gallery_thumbnail = gallery_thumbnail
        # Stages: {'thumbnail', 'exif', 'full'}.
        # Gallery should typically request only {'thumbnail'} (and optionally 'exif').
        self.stages = stages if stages is not None else {'thumbnail', 'exif', 'full'}
        # Rows between this file and the visible viewport, as of the last
        # reprioritise() call. Priority used to be fixed at enqueue: a tile
        # kept the urgency it had when requested, so after a scroll or a jump
        # the queue was ordered by where the user USED to be. 0 = on screen.
        self.viewport_distance = 0
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
        # Distance before stage weight: a thumbnail two screens away is worth
        # less than the full decode of what is on screen right now.
        if self.viewport_distance != other.viewport_distance:
            return self.viewport_distance < other.viewport_distance
        return _task_queue_weight(self) < _task_queue_weight(other)

class ImageLoadWorker(QRunnable):
    """可重用的工作線程"""

    def __init__(self, task: ImageLoadTask, manager: ImageLoadManager):
        super().__init__()
        self.task = task
        self.manager = manager
        self._processor = None  # 延遲初始化，避免導入時創建

    def _yield_if_needed(self) -> None:
        """Bounded cooperative yield while another CURRENT task is running."""
        if self.task.priority == Priority.CURRENT:
            return
        if self.task.is_cancelled():
            return
        max_wait_s = _env_int("RAWVIEWER_YIELD_MAX_MS", 1500, minimum=100) / 1000.0
        deadline = time.time() + max_wait_s
        while time.time() < deadline:
            if self.task.is_cancelled():
                return
            if not _has_running_current_task(self.manager, exclude_task=self.task):
                return
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
            from processing.unified import UnifiedImageProcessor
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
        self.manager._mark_task_running(self.task)
        file_path = self.task.file_path
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
                # No explicit finish here: the `finally` below owns the
                # bookkeeping and runs on this return too. Calling it in both
                # places emitted task_completed twice for one task -- so the
                # gallery kicked _request_load_visible_images twice -- and ran
                # _schedule_next_task twice under exactly the churn that
                # produces cancelled-at-start tasks.
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
                if preview_tier and thumbnail is not None and not self.task.is_cancelled() and processor._is_raw_file(file_path):
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
                        and not _preview_is_best_possible(processor, file_path)
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
                elif exif_data is None and not self.task.is_cancelled() and self._safe_emit():
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
                    elif exif_data is None and not self.task.is_cancelled() and self._safe_emit():
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
                        # verify=True: this record feeds preview extraction /
                        # orientation decisions on a worker thread, where the
                        # synchronous trustworthiness probe is acceptable.
                        cached_exif = cache.get_exif(file_path, verify=True)
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
                        deferred = processor.cache.get_exif(file_path, verify=True)
                        if deferred and deferred.get("minimal_preview_exif"):
                            exif_data = deferred
                    if (
                        allow_heavy_fallback
                        and exif_data is not None
                        and not self.task.is_cancelled()
                    ) and self._safe_emit():
                        self.manager.exif_data_ready.emit(file_path, exif_data)
                    if allow_heavy_fallback and processor._is_raw_file(file_path):
                        min_preview_dim = _min_acceptable_preview_dim(file_path)
                        _d0 = processor._preview_buffer_max_dim(thumbnail) if thumbnail is not None else -1
                        if (
                            thumbnail is not None
                            and processor._preview_buffer_max_dim(thumbnail)
                            < min_preview_dim
                        ):
                            thumbnail = processor.ensure_display_tier_preview(
                                file_path, thumbnail
                            )
                        if os.environ.get("RAWVIEWER_DEBUG_THUMB"):
                            import logging

                            logging.getLogger(__name__).warning(
                                "[DBGTHUMB] %s heavy=%s first=%s after_ensure=%s min=%s best_possible=%s",
                                os.path.basename(file_path), allow_heavy_fallback, _d0,
                                processor._preview_buffer_max_dim(thumbnail) if thumbnail is not None else None,
                                min_preview_dim,
                                _preview_is_best_possible(processor, file_path),
                            )
                        if (
                            thumbnail is not None
                            and processor._preview_buffer_max_dim(thumbnail)
                            < min_preview_dim
                            and not _preview_is_best_possible(processor, file_path)
                        ):
                            thumbnail = None
                    if thumbnail is not None and not self.task.is_cancelled():
                        cache_exif = exif_data or processor.cache.get_exif(
                            file_path, verify=True
                        )
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

                from raw_adjustments import (
                    is_default_adjustments,
                    load_adjustments_for_file,
                    sidecar_adjustments_enabled,
                )

                # Sidecar edits on the full-res tier are only worth paying for
                # the image actually on screen: the gamma-path apply costs
                # seconds at 30+MP, and letting PRELOAD/BACKGROUND tasks run it
                # for neighbors monopolized the worker pool -- gallery
                # thumbnail queue-waits climbed past 5s with repeated
                # sidecar_apply ms=3300-7200 entries for files nobody was
                # looking at ("gallery loading is bottlenecked" report). The
                # CURRENT task still applies, so the displayed image shows its
                # edits; a prefetched neighbor gets them applied when it
                # becomes current (memoized in _apply_sidecar_if_needed).
                #
                # Progressive path (CURRENT + non-default XMP only): decode
                # unadjusted first (still caches the unadjusted base), emit a
                # preview-sized edited interim so demosaic-accurate edits
                # paint before the multi-second full apply finishes, then
                # emit the full adjusted buffer. Prefetch stays sidecar-free.
                apply_sidecar = (
                    sidecar_adjustments_enabled()
                    and self.task.priority == Priority.CURRENT
                )
                result = processor.process_full_image(
                    file_path,
                    use_full_resolution=self.task.use_full_resolution,
                    executor=self.manager._process_pool if self._safe_emit() else None,
                    apply_sidecar_adjustments=False,
                )

                def _emit_image_buffer(buf) -> None:
                    if buf is None or self.task.is_cancelled() or not self._safe_emit():
                        return
                    from processing.raw_decode.gpu_bridge import DeviceRgb as _DeviceRgb

                    if isinstance(buf, (np.ndarray, _DeviceRgb)):
                        self.manager.image_ready.emit(file_path, buf)
                    elif isinstance(buf, QImage):
                        # Built on this worker because QPixmap cannot be. The
                        # signal is queued, so the conversion below lands on
                        # the GUI thread, which is the only place it is legal.
                        self.manager.qimage_ready.emit(file_path, buf)
                    elif isinstance(buf, QPixmap):
                        self.manager.pixmap_ready.emit(file_path, buf)

                if (
                    result is not None
                    and apply_sidecar
                    and not self.task.is_cancelled()
                ):
                    needs_apply = False
                    try:
                        adj = load_adjustments_for_file(file_path)
                        needs_apply = not is_default_adjustments(adj)
                    except Exception:
                        needs_apply = False
                    if needs_apply:
                        result = processor.apply_sidecar_progressive(
                            file_path,
                            result,
                            cancel_check=self.task.is_cancelled,
                            on_interim=_emit_image_buffer,
                        )

                if result is not None and not self.task.is_cancelled():
                    _emit_image_buffer(result)
                elif (
                    "full" in stages
                    and not self.task.is_cancelled()
                    and self.task.priority == Priority.CURRENT
                ):
                    # Suppress the failure dialog when a usable buffer for this
                    # file is already cached: a concurrent task produced one.
                    # Most visible for HE/HE* NEF -- LibRaw can't demosaic it, so
                    # a full-decode task that raced ahead of the embedded-JPEG
                    # routing returns None while another task already displayed
                    # the embedded preview. Erroring here pops an "unsupported or
                    # corrupt" dialog for a file that is already on screen.
                    already_displayable = False
                    try:
                        buf = processor.cache.get_full_image(file_path)
                        if buf is None:
                            buf = processor.cache.get_preview(file_path)
                        if (
                            buf is not None
                            and hasattr(buf, "shape")
                            and getattr(buf, "ndim", 0) >= 2
                            and max(buf.shape[0], buf.shape[1]) >= 1024
                        ):
                            already_displayable = True
                    except Exception:
                        already_displayable = False

                    if not already_displayable and self._safe_emit():
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
            self.manager._mark_task_not_running(self.task)
            # Always: the bookkeeping (active map, heavy slot) has to happen
            # even when the manager is tearing down. Only the signal is gated.
            self.manager._task_finished(self.task, emit=self._safe_emit())
            worker_thread_local.priority = None
            worker_thread_local.task = None

    def _handle_thumbnail_result(self, file_path, thumbnail):
        """Internal helper to process and emit thumbnail results."""
        # Saved XMP edits on the fit/preview tier (see apply_saved_edits_for_
        # display: no-op unless the file actually has non-default edits).
        # Worker thread: the ~215ms apply for a 2304px preview of an edited
        # file must never land on the UI thread. Applied at delivery only --
        # the caches upstream keep the unadjusted base.
        try:
            from raw_adjustments import apply_saved_edits_for_display

            thumbnail = apply_saved_edits_for_display(file_path, thumbnail)
        except Exception:
            pass
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
            return not (self.manager is None or sip.isdeleted(self.manager))
        except (ImportError, AttributeError):
            # Fallback if sip is not available or manager is None
            return self.manager is not None

