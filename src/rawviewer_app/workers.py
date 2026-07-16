"""QRunnable background workers."""
import logging, threading, os, sys, time, traceback
import numpy as np
from PyQt6.QtCore import QRunnable, QObject, pyqtSignal
from PyQt6.QtGui import QImage
from rawviewer_app.signals import ReleaseUpdateCheckSignals, SemanticIndexPrepSignals, RawRecoverySignals

class _ReleaseUpdateCheckWorker(QRunnable):
    """Fetch latest GitHub release tag and compare to the running app version."""

    def __init__(self, signals: ReleaseUpdateCheckSignals, *, timeout: float = 5.0):
        super().__init__()
        self.signals = signals
        self.timeout = float(timeout)

    def run(self) -> None:
        from release_update import check_for_update

        try:
            result = check_for_update(timeout=self.timeout)
        except Exception:
            result = {
                "current": "",
                "latest": "",
                "is_latest": True,
                "update_available": False,
                "release_url": "",
                "release_name": "",
                "published_at": "",
                "offline": True,
                "error": "",
            }
        try:
            from PyQt6 import sip
            if not sip.isdeleted(self.signals):
                self.signals.finished.emit(result)
        except Exception:
            pass

class _SemanticIndexPrepWorker(QRunnable):
    """Background task to prepare semantic index build (coverage & pending scans)."""
    def __init__(self, corpus_files, index, signals):
        super().__init__()
        self.corpus_files = corpus_files
        self.index = index
        self.signals = signals

    def run(self):
        try:
            coverage = self.index.get_index_coverage(self.corpus_files)
            pending = self.index.get_pending_paths(self.corpus_files)
            face_pending = 0
            if not pending:
                face_pending = self.index.get_face_pending_count(self.corpus_files)
            try:
                from PyQt6 import sip
                if not sip.isdeleted(self.signals):
                    self.signals.done.emit(coverage, pending, face_pending)
            except Exception:
                pass
        except Exception as e:
            try:
                from PyQt6 import sip
                if not sip.isdeleted(self.signals):
                    self.signals.error.emit(str(e))
            except Exception:
                pass

class SemanticAssetDownloadSignals(QObject):
    """Signal carrier for background semantic backend asset download."""
    progress = pyqtSignal(object, int, str)         # token, pct, short status
    done = pyqtSignal(object, str, object)        # token, asset path, corpus files
    error = pyqtSignal(object, str)               # token, error


# -----------------------------
# Worker to load images in background
# -----------------------------
class ImageLoadTask(QRunnable):
    """Background task to load and scale images"""
    def __init__(self, index, file_path, target_width, target_height, signal, parent_viewer=None, generation=0):
        super().__init__()
        self.index = index
        self.file_path = file_path
        self.target_width = target_width
        self.target_height = target_height
        self.signal = signal
        self.parent_viewer = parent_viewer
        self.generation = generation  # Track which folder generation this task belongs to
        self._cancelled = False
        self._lock = threading.Lock()

    def cancel(self):
        """Cancel the task"""
        with self._lock:
            self._cancelled = True

    def is_cancelled(self):
        """Check if task is cancelled"""
        with self._lock:
            return self._cancelled

    
    def run(self):
        """Load and scale image in worker thread - returns QImage, not QPixmap"""
        import os
        import logging
        import time
        logger = logging.getLogger(__name__)
        
        if self.is_cancelled():
            return
            
        task_start = time.time()
        file_basename = os.path.basename(self.file_path) if self.file_path else 'unknown'
        
        try:
            from PyQt6.QtGui import QImageReader, QImage
            from PyQt6.QtCore import QSize, Qt
            
            # Check if this is a RAW file
            file_ext = os.path.splitext(self.file_path)[1].lower()
            raw_extensions = ['.arw', '.cr2', '.nef', '.raf', '.orf', '.dng', '.cr3', '.rw2', '.rwl', '.srw', 
                             '.pef', '.x3f', '.3fr', '.fff', '.iiq', '.cap', '.erf', '.mef', '.mos', '.nrw', '.srf']
            is_raw = file_ext in raw_extensions
            
            # For RAW files, try to extract embedded JPEG thumbnail first
            if is_raw:
                raw_start = time.time()
                try:
                    import rawpy
                    import numpy as np
                    from image_cache import get_image_cache
                    
                    # Check disk cache first (much faster than extracting from RAW)
                    cache = get_image_cache()
                    disk_cache_start = time.time()
                    jpeg_data = cache.disk_thumbnail_cache.get(self.file_path)
                    if jpeg_data is not None:
                        disk_cache_time = time.time() - disk_cache_start
                        logger.info(f"[IMAGE_LOAD_TASK] Disk cache hit in {disk_cache_time:.3f}s: {file_basename}")
                        try:
                            from io import BytesIO
                            from PIL import Image, ImageOps
                            
                            # Load JPEG from disk cache
                            pil_image = Image.open(BytesIO(jpeg_data))
                            # Apply EXIF orientation correction
                            pil_image = ImageOps.exif_transpose(pil_image)
                            # Convert to RGB if needed
                            if pil_image.mode != 'RGB':
                                pil_image = pil_image.convert('RGB')
                            
                            # Convert PIL image to QImage
                            width, height = pil_image.size
                            image_bytes = pil_image.tobytes('raw', 'RGB')
                            bytes_per_line = 3 * width
                            qimage = QImage(image_bytes, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                            
                            if not qimage.isNull():
                                # Scale to target size
                                aspect = qimage.width() / qimage.height() if qimage.height() > 0 else 1.0
                                scaled_width = int(self.target_height * aspect)
                                scaled_height = self.target_height
                                
                                # Ensure we don't exceed target width
                                if scaled_width > self.target_width:
                                    scaled_width = self.target_width
                                    scaled_height = int(self.target_width / aspect) if aspect > 0 else self.target_height
                                
                                # Ensure dimensions are at least 1px to prevent crash in SmoothTransformation
                                scaled_width = max(1, scaled_width)
                                scaled_height = max(1, scaled_height)
                                
                                scaled_image = qimage.scaled(
                                    scaled_width, 
                                    scaled_height,
                                    Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation
                                )
                                
                                if not scaled_image.isNull():
                                    total_time = time.time() - task_start
                                    logger.info(f"[IMAGE_LOAD_TASK] Loaded thumbnail from disk cache for {file_basename} in {total_time:.3f}s (size: {scaled_width}x{scaled_height})")
                                    self.signal.loaded.emit(self.index, scaled_image, self.generation)
                                    return
                                else:
                                    logger.warning(f"[IMAGE_LOAD_TASK] Failed to scale QImage from disk cache for {file_basename}")
                            else:
                                logger.warning(f"[IMAGE_LOAD_TASK] Failed to create QImage from disk cache JPEG for {file_basename}")
                        except Exception as e:
                            logger.warning(f"[IMAGE_LOAD_TASK] Failed to load from disk cache, will extract from RAW: {e}")
                            # Remove invalid cache entry
                            cache.disk_thumbnail_cache.remove(self.file_path)
                    
                    if self.is_cancelled(): return
                    
                    # Disk cache miss, extract from RAW
                    raw_open_start = time.time()
                    with rawpy.imread(self.file_path) as raw:
                        if self.is_cancelled(): return
                        raw_open_time = time.time() - raw_open_start
                        logger.debug(f"[IMAGE_LOAD_TASK] RAW file opened in {raw_open_time:.3f}s: {file_basename}")
                        # Try to extract embedded JPEG thumbnail
                        try:
                            # Extract embedded JPEG preview (usually much smaller than full RAW)
                            thumb_extract_start = time.time()
                            thumb = raw.extract_thumb()
                            thumb_extract_time = time.time() - thumb_extract_start
                            logger.debug(f"[IMAGE_LOAD_TASK] Thumbnail extracted in {thumb_extract_time:.3f}s: {file_basename}")
                            
                            if thumb.format == rawpy.ThumbFormat.JPEG:
                                # Thumbnail is JPEG - load it directly
                                from io import BytesIO
                                from PIL import Image, ImageOps
                                jpeg_data = thumb.data
                                
                                # Save to disk cache for future use
                                try:
                                    cache.disk_thumbnail_cache.put(self.file_path, jpeg_data)
                                    logger.debug(f"[IMAGE_LOAD_TASK] Saved thumbnail to disk cache: {file_basename}")
                                except Exception as e:
                                    logger.debug(f"[IMAGE_LOAD_TASK] Failed to save to disk cache: {e}")
                                
                                # Use PIL to load JPEG and apply EXIF orientation
                                pil_image = Image.open(BytesIO(jpeg_data))
                                # Apply EXIF orientation correction
                                pil_image = ImageOps.exif_transpose(pil_image)
                                # Convert to RGB if needed
                                if pil_image.mode != 'RGB':
                                    pil_image = pil_image.convert('RGB')
                                
                                # Convert PIL image to QImage
                                width, height = pil_image.size
                                image_bytes = pil_image.tobytes('raw', 'RGB')
                                bytes_per_line = 3 * width
                                qimage = QImage(image_bytes, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                                
                                if not qimage.isNull():
                                    # Scale to target size
                                    aspect = qimage.width() / qimage.height() if qimage.height() > 0 else 1.0
                                    scaled_width = int(self.target_height * aspect)
                                    scaled_height = self.target_height
                                    
                                    # Ensure we don't exceed target width
                                    if scaled_width > self.target_width:
                                        scaled_width = self.target_width
                                        scaled_height = int(self.target_width / aspect) if aspect > 0 else self.target_height
                                    
                                    # Ensure dimensions are at least 1px to prevent crash in SmoothTransformation
                                    scaled_width = max(1, scaled_width)
                                    scaled_height = max(1, scaled_height)
                                    
                                    scaled_image = qimage.scaled(
                                        scaled_width, 
                                        scaled_height,
                                        Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation
                                    )
                                    
                                    if not scaled_image.isNull():
                                        raw_time = time.time() - raw_start
                                        total_time = time.time() - task_start
                                        logger.debug(f"[IMAGE_LOAD_TASK] Loaded embedded JPEG thumbnail for {file_basename} in {raw_time:.3f}s (total: {total_time:.3f}s)")
                                        self.signal.loaded.emit(self.index, scaled_image, self.generation)
                                        return
                            
                            elif thumb.format == rawpy.ThumbFormat.BITMAP:
                                # Thumbnail is bitmap - convert to QImage
                                bitmap = thumb.data
                                if bitmap is not None and len(bitmap.shape) >= 2:
                                    # Convert numpy array to QImage
                                    height, width = bitmap.shape[:2]
                                    
                                    # Ensure contiguous array
                                    if not bitmap.flags['C_CONTIGUOUS']:
                                        bitmap = np.ascontiguousarray(bitmap)
                                    
                                    if len(bitmap.shape) == 3:
                                        # Color image (RGB)
                                        if bitmap.shape[2] == 3:
                                            # Convert BGR to RGB (rawpy returns BGR)
                                            rgb = np.flip(bitmap, axis=2)  # BGR to RGB
                                            # Ensure uint8
                                            if rgb.dtype != np.uint8:
                                                rgb = rgb.astype(np.uint8)
                                            qimage = QImage(rgb.data, width, height, width * 3, QImage.Format.Format_RGB888)
                                        else:
                                            # Other color formats - try direct conversion
                                            if bitmap.dtype != np.uint8:
                                                bitmap = bitmap.astype(np.uint8)
                                            qimage = QImage(bitmap.data, width, height, width * bitmap.shape[2], QImage.Format.Format_RGB888)
                                    else:
                                        # Grayscale
                                        if bitmap.dtype != np.uint8:
                                            bitmap = bitmap.astype(np.uint8)
                                        qimage = QImage(bitmap.data, width, height, width, QImage.Format.Format_Grayscale8)
                                    
                                    if not qimage.isNull():
                                        # Scale to target size
                                        aspect = qimage.width() / qimage.height() if qimage.height() > 0 else 1.0
                                        scaled_width = int(self.target_height * aspect)
                                        scaled_height = self.target_height
                                        
                                        if scaled_width > self.target_width:
                                            scaled_width = self.target_width
                                            scaled_height = int(self.target_width / aspect) if aspect > 0 else self.target_height
                                        
                                        # Ensure dimensions are at least 1px to prevent crash in SmoothTransformation
                                        scaled_width = max(1, scaled_width)
                                        scaled_height = max(1, scaled_height)
                                        
                                        scaled_image = qimage.scaled(
                                            scaled_width, 
                                            scaled_height,
                                            Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation
                                        )
                                        
                                        if not scaled_image.isNull():
                                            raw_time = time.time() - raw_start
                                            total_time = time.time() - task_start
                                            logger.debug(f"[IMAGE_LOAD_TASK] Loaded embedded bitmap thumbnail for {file_basename} in {raw_time:.3f}s (total: {total_time:.3f}s)")
                                            self.signal.loaded.emit(self.index, scaled_image, self.generation)
                                            return
                        except Exception as thumb_error:
                            logger.debug(f"[IMAGE_LOAD_TASK] Could not extract thumbnail from RAW file {os.path.basename(self.file_path)}: {thumb_error}")
                            if self.is_cancelled(): return
                            # Fall through to regular loading
                
                except Exception as raw_error:
                    logger.debug(f"[IMAGE_LOAD_TASK] Error processing RAW file {os.path.basename(self.file_path)}: {raw_error}")
                    # Fall through to regular loading
            
            # Use QImageReader with setScaledSize to load already-scaled image
            # This avoids loading full resolution and then scaling
            reader = QImageReader(self.file_path)
            reader.setAutoTransform(True)  # CRITICAL: Handle EXIF orientation BEFORE getting size
            
            # Calculate scaled size maintaining aspect ratio
            original_size = reader.size()
            if not original_size.isValid():
                # FALLBACK path (existing)...
                pass # This chunk only shows the start of the fix
            
            aspect = original_size.width() / original_size.height() if original_size.height() > 0 else 1.0
            scaled_width = int(self.target_height * aspect)
            scaled_height = self.target_height
            
            # Ensure we don't exceed target width
            if scaled_width > self.target_width:
                scaled_width = self.target_width
                scaled_height = int(self.target_width / aspect) if aspect > 0 else self.target_height
            
            # Ensure dimensions are at least 1px to prevent crash
            scaled_width = max(1, scaled_width)
            scaled_height = max(1, scaled_height)
            
            # Set scaled size - this makes QImageReader decode at target size directly
            reader.setScaledSize(QSize(scaled_width, scaled_height))
            # reader.setAutoTransform(True) - MOVED UP
            
            # Read the already-scaled image (very cheap, no full decode)
            read_start = time.time()
            scaled_image = reader.read()
            read_time = time.time() - read_start
            
            if scaled_image.isNull():
                # FALLBACK: Try PIL if QImageReader fails to read
                try:
                    from PIL import Image, ImageOps
                    with Image.open(self.file_path) as img:
                        img = ImageOps.exif_transpose(img)
                        w, h = img.size
                        aspect = w / h if h > 0 else 1.0
                        sw = int(self.target_height * aspect)
                        sh = self.target_height
                        if sw > self.target_width:
                            sw = self.target_width
                            sh = int(self.target_width / aspect) if aspect > 0 else self.target_height
                        
                        # Ensure dimensions are at least 1px to prevent crash
                        sw = max(1, sw)
                        sh = max(1, sh)
                        
                        img = img.resize((sw, sh), Image.Resampling.HAMMING)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Convert to QImage
                        image_bytes = img.tobytes('raw', 'RGB')
                        scaled_image = QImage(image_bytes, sw, sh, sw * 3, QImage.Format.Format_RGB888)
                        
                        if not scaled_image.isNull():
                            logger.debug(f"[IMAGE_LOAD_TASK] Loaded via PIL fallback (read failed): {os.path.basename(self.file_path)}")
                except Exception as pil_err:
                    logger.debug(f"[IMAGE_LOAD_TASK] PIL fallback failed for {os.path.basename(self.file_path)}: {pil_err}")

            if self.is_cancelled():
                return
                
            # Emit QImage to UI thread (will convert to QPixmap there)
            total_time = time.time() - task_start
            if is_raw:
                # For RAW files that fall through to non-RAW path, read_time should be defined
                if 'read_time' in locals():
                    logger.debug(f"[IMAGE_LOAD_TASK] Loaded non-RAW fallback for {file_basename} in {total_time:.3f}s (read: {read_time:.3f}s)")
                else:
                    logger.debug(f"[IMAGE_LOAD_TASK] Loaded non-RAW fallback for {file_basename} in {total_time:.3f}s")
            else:
                logger.debug(f"[IMAGE_LOAD_TASK] Loaded {file_basename} in {total_time:.3f}s (read: {read_time:.3f}s)")
            try:
                self.signal.loaded.emit(self.index, scaled_image, self.generation)
            except RuntimeError:
                # This happens if the JustifiedGallery or its signal carrier was deleted
                # while this background task was still running.
                logger.debug(f"[IMAGE_LOAD_TASK] Signal carrier deleted, ignoring result for: {file_basename}")
            
        except Exception as e:
            logger.error(f"[IMAGE_LOAD_TASK] Error loading image {os.path.basename(self.file_path) if self.file_path else 'unknown'}: {e}", exc_info=True)


class RawRecoveryWorker(QRunnable):
    """Decode RAW linear preview off the UI thread and apply local tone recovery."""

    def __init__(
        self,
        file_path: str,
        signals: RawRecoverySignals,
        *,
        exif_data: dict | None = None,
    ):
        super().__init__()
        self.file_path = file_path
        self.signals = signals
        self.exif_data = exif_data or {}

    def run(self) -> None:
        from raw_tone_recovery import decode_and_recover_raw

        logger = logging.getLogger(__name__)
        logger.info(
            "RAW recovery worker started for %s",
            os.path.basename(self.file_path),
        )
        try:
            from unified_image_processor import UnifiedImageProcessor

            processor = UnifiedImageProcessor()

            def _orient(rgb, orientation, exif):
                return processor._apply_orientation_correction(rgb, orientation, exif)

            rgb = decode_and_recover_raw(
                self.file_path,
                apply_orientation=_orient,
                exif_data=self.exif_data,
            )
            if rgb is None:
                raise RuntimeError("decode returned empty buffer")
            logger.info(
                "RAW recovery worker finished for %s shape=%s",
                os.path.basename(self.file_path),
                getattr(rgb, "shape", None),
            )
            try:
                from PyQt6 import sip

                if not sip.isdeleted(self.signals):
                    self.signals.ready.emit(self.file_path, rgb)
            except Exception:
                pass
        except Exception as exc:
            try:
                from PyQt6 import sip

                if not sip.isdeleted(self.signals):
                    self.signals.failed.emit(self.file_path, str(exc))
            except Exception:
                pass

