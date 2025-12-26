"""
Enhanced RAW processor with smart thumbnail utilization and caching.

This module provides optimized RAW processing with immediate thumbnail display,
intelligent caching, and progressive image loading for maximum performance.
"""

import os
import time
import threading
import warnings
import numpy as np
import rawpy
import exifread
from typing import Optional, Dict, Any, Tuple, Union
from PyQt6.QtCore import QThread, pyqtSignal, QObject
from PyQt6.QtGui import QPixmap, QImage
from PIL import Image
import io

# Suppress exifread warnings for unsupported file formats (e.g., video files)
warnings.filterwarnings('ignore', category=UserWarning, module='exifread')

from image_cache import get_image_cache


class ThumbnailExtractor(QObject):
    """Fast thumbnail extractor for immediate display."""

    def __init__(self):
        super().__init__()

    def extract_thumbnail_from_raw(self, file_path: str) -> Optional[np.ndarray]:
        """Extract embedded thumbnail from RAW file."""
        try:
            with rawpy.imread(file_path) as raw:
                thumb = raw.extract_thumb()

                if thumb.format == rawpy.ThumbFormat.JPEG:
                    # Convert JPEG bytes to numpy array
                    jpeg_image = Image.open(io.BytesIO(thumb.data))
                    if jpeg_image.mode != 'RGB':
                        jpeg_image = jpeg_image.convert('RGB')
                    return np.array(jpeg_image)
                elif thumb.format == rawpy.ThumbFormat.BITMAP:
                    # Already a numpy array
                    return thumb.data

        except Exception as e:
            # Log the error for debugging
            # print(f"Thumbnail extraction error: {str(e)}")
            pass

        return None

    def extract_thumbnail_from_image(self, file_path: str, max_size: int = 512) -> Optional[np.ndarray]:
        """Extract thumbnail from regular image file."""
        try:
            with Image.open(file_path) as img:
                # Calculate thumbnail size maintaining aspect ratio
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                return np.array(img)

        except Exception:
            pass

        return None


class EXIFExtractor(QObject):
    """Fast EXIF data extractor with caching."""

    def __init__(self):
        super().__init__()
        self.cache = get_image_cache()

    def extract_exif_data(self, file_path: str) -> Dict[str, Any]:
        """Extract EXIF data with caching."""
        # Check cache first
        cached_exif = self.cache.get_exif(file_path)
        if cached_exif is not None:
            return cached_exif

        # Extract fresh EXIF data
        exif_data = self._extract_exif_from_file(file_path)

        # Cache the result
        if exif_data:
            self.cache.put_exif(file_path, exif_data)

        return exif_data

    def _extract_exif_from_file(self, file_path: str) -> Dict[str, Any]:
        """Extract EXIF data from file."""
        try:
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)

                # Extract key information
                orientation = 1
                camera_make = ''
                camera_model = ''

                # Get orientation
                orientation_tag = tags.get('Image Orientation')
                if orientation_tag:
                    orientation_str = str(orientation_tag)
                    orientation_map = {
                        'Horizontal (normal)': 1,
                        'Mirrored horizontal': 2,
                        'Rotated 180': 3,
                        'Mirrored vertical': 4,
                        'Mirrored horizontal then rotated 90 CCW': 5,
                        'Rotated 90 CW': 6,
                        'Mirrored horizontal then rotated 90 CW': 7,
                        'Rotated 90 CCW': 8
                    }
                    orientation = orientation_map.get(orientation_str, 1)

                # Get camera info
                make_tag = tags.get('Image Make')
                if make_tag:
                    camera_make = str(make_tag).strip()

                model_tag = tags.get('Image Model')
                if model_tag:
                    camera_model = str(model_tag).strip()

                # Convert all tags to serializable format
                exif_dict = {}
                for key, value in tags.items():
                    try:
                        exif_dict[key] = str(value)
                    except:
                        pass

                return {
                    'orientation': orientation,
                    'camera_make': camera_make,
                    'camera_model': camera_model,
                    'exif_data': exif_dict
                }

        except Exception:
            pass

        return {
            'orientation': 1,
            'camera_make': '',
            'camera_model': '',
            'exif_data': {}
        }


class OptimizedRAWProcessor(QObject):
    """Optimized RAW processor with format-specific optimizations."""

    def __init__(self):
        super().__init__()

    def is_canon_camera(self, file_path: str, exif_data: Dict[str, Any] = None) -> bool:
        """Check if this is a Canon camera."""
        # Check file extension first (faster)
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in ['.cr2', '.cr3']:
            return True

        # Check EXIF if available
        if exif_data and exif_data.get('camera_make'):
            return 'CANON' in exif_data['camera_make'].upper()

        return False

    def is_fujifilm_camera(self, file_path: str, exif_data: Dict[str, Any] = None) -> bool:
        """Check if this is a Fujifilm camera."""
        # Check file extension first (faster)
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in ['.raf']:
            return True

        # Check EXIF if available
        if exif_data and exif_data.get('camera_make'):
            make = exif_data['camera_make'].upper()
            return 'FUJIFILM' in make or 'FUJI' in make

        return False

    def is_sony_camera(self, file_path: str, exif_data: Dict[str, Any] = None) -> bool:
        """Check if this is a Sony camera."""
        # Check file extension first (faster)
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in ['.arw']:
            return True

        # Check EXIF if available
        if exif_data and exif_data.get('camera_make'):
            return 'SONY' in exif_data['camera_make'].upper()

        return False

    def get_optimized_processing_params(self, file_path: str, exif_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get optimized processing parameters based on camera and file."""
        # Default fast processing parameters
        params = {
            'use_camera_wb': True,
            'use_auto_wb': False,
            'output_bps': 8,  # 8-bit for speed
            'no_auto_bright': False,
            'gamma': (2.222, 4.5),  # Standard sRGB gamma
            'bright': 1.0,
            # 'highlight': 0,  # Removed for rawpy 0.25.0 compatibility
            # 'shadow': 0,  # Removed for rawpy 0.25.0 compatibility
        }

        # Get file size for processing decisions
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        except:
            file_size_mb = 50  # Default assumption

        # Adjust for large files (use faster processing)
        if file_size_mb > 80:
            params.update({
                'half_size': True,  # Process at half resolution for speed
                'use_camera_wb': True,
                'no_auto_bright': True
            })

        # Camera-specific optimizations
        if self.is_canon_camera(file_path, exif_data):
            # Canon cameras benefit from camera white balance
            params.update({
                'use_camera_wb': True,
                'use_auto_wb': False
            })
        elif self.is_fujifilm_camera(file_path, exif_data):
            # Fujifilm X-Trans sensors
            params.update({
                'use_camera_wb': True,
                'dcb_enhance': True  # Better demosaicing for X-Trans
            })
        elif self.is_sony_camera(file_path, exif_data):
            # Sony cameras
            params.update({
                'use_camera_wb': True,
                'use_auto_wb': False
            })

        return params

    def process_raw_fast(self, file_path: str, exif_data: Dict[str, Any] = None) -> Optional[np.ndarray]:
        """Process RAW file with optimized parameters for speed."""
        try:
            with rawpy.imread(file_path) as raw:
                params = self.get_optimized_processing_params(
                    file_path, exif_data)

                # Process with optimized parameters
                rgb_image = raw.postprocess(**params)

                return rgb_image

        except Exception as e:
            # Log the actual error for debugging
            # print(f"RAW processing error in process_raw_fast: {str(e)}")
            return None

    def process_raw_quality(self, file_path: str, exif_data: Dict[str, Any] = None) -> Optional[np.ndarray]:
        """Process RAW file with quality parameters (slower)."""
        try:
            with rawpy.imread(file_path) as raw:
                # Quality processing parameters
                params = {
                    'use_camera_wb': True,
                    'output_bps': 16,  # 16-bit for quality
                    'gamma': (1, 1),   # Linear gamma for better quality
                    'no_auto_bright': True,
                    'bright': 1.0
                }

                # Camera-specific quality adjustments
                if self.is_canon_camera(file_path, exif_data):
                    params.update({
                        'use_camera_wb': True,
                        # 'highlight': 1,  # Removed for rawpy 0.25.0 compatibility
                    })
                elif self.is_fujifilm_camera(file_path, exif_data):
                    params.update({
                        'dcb_enhance': True,
                        'fbdd_noise_reduction': rawpy.FBDDNoiseReductionMode.Full
                    })

                rgb_image = raw.postprocess(**params)

                # Convert 16-bit to 8-bit for display
                if rgb_image.dtype == np.uint16:
                    rgb_image = (rgb_image / 256).astype(np.uint8)

                return rgb_image

        except Exception as e:
            # Log the actual error for debugging
            # print(f"RAW processing error in process_raw_quality: {str(e)}")
            return None


class EnhancedRAWProcessor(QThread):
    """Enhanced RAW processor with progressive loading and caching."""

    # Signals
    thumbnail_ready = pyqtSignal(object)  # np.ndarray thumbnail
    image_processed = pyqtSignal(object)  # np.ndarray full image
    error_occurred = pyqtSignal(str)
    processing_progress = pyqtSignal(str)  # Status message
    exif_data_ready = pyqtSignal(dict)  # EXIF data dictionary

    def __init__(self, file_path: str, use_quality_processing: bool = False):
        super().__init__()
        self.file_path = file_path
        self.use_quality_processing = use_quality_processing
        self.cache = get_image_cache()
        self.thumbnail_extractor = ThumbnailExtractor()
        self.exif_extractor = EXIFExtractor()
        self.raw_processor = OptimizedRAWProcessor()

        # Determine if this is a RAW file
        self.is_raw_file = self._is_raw_file(file_path)

        # Processing state
        self._should_stop = False

    def _is_raw_file(self, file_path: str) -> bool:
        """Check if file is a RAW format."""
        raw_exts = {
            '.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef',
            '.srw', '.x3f', '.raf', '.3fr', '.fff', '.iiq', '.cap', '.erf',
            '.mef', '.mos', '.nrw', '.rwl', '.srf'
        }
        ext = os.path.splitext(file_path)[1].lower()
        return ext in raw_exts

    def stop_processing(self):
        """Stop the processing thread."""
        self._should_stop = True

    def run(self):
        """Main processing loop with progressive loading."""
        try:
            if not os.path.exists(self.file_path):
                self.error_occurred.emit(f"File not found: {self.file_path}")
                return

            # Step 1: Check cache for full image first
            cached_image = self.cache.get_full_image(self.file_path)
            if cached_image is not None and not self._should_stop:
                self.processing_progress.emit("Loaded from cache")
                self.image_processed.emit(cached_image)
                return

            # Step 2: Extract and emit thumbnail immediately
            if not self._should_stop:
                self._process_thumbnail()

            # Step 3: Extract EXIF data
            exif_data = None
            if not self._should_stop:
                self.processing_progress.emit("Reading metadata...")
                exif_data = self.exif_extractor.extract_exif_data(
                    self.file_path)

                # Emit EXIF data ready signal for immediate status bar update
                if exif_data:
                    self.exif_data_ready.emit(exif_data)

            # Step 4: Process full image
            if not self._should_stop:
                self._process_full_image(exif_data)

        except Exception as e:
            if not self._should_stop:
                self.error_occurred.emit(f"Processing error: {str(e)}")

    def _process_thumbnail(self):
        """Process and emit thumbnail."""
        # Check thumbnail cache first
        cached_thumbnail = self.cache.get_thumbnail(self.file_path)
        if cached_thumbnail is not None:
            self.thumbnail_ready.emit(cached_thumbnail)
            return

        self.processing_progress.emit("Extracting preview...")

        # Extract thumbnail
        thumbnail = None
        if self.is_raw_file:
            thumbnail = self.thumbnail_extractor.extract_thumbnail_from_raw(
                self.file_path)
        else:
            thumbnail = self.thumbnail_extractor.extract_thumbnail_from_image(
                self.file_path)

        if thumbnail is not None:
            # Cache the thumbnail
            self.cache.put_thumbnail(self.file_path, thumbnail)

            # Emit thumbnail
            self.thumbnail_ready.emit(thumbnail)
        else:
            # No thumbnail available, will wait for full processing
            pass

    def _process_full_image(self, exif_data: Dict[str, Any]):
        """Process full resolution image."""
        if self.is_raw_file:
            self._process_raw_image(exif_data)
        else:
            self._process_regular_image(exif_data)

    def _process_raw_image(self, exif_data: Dict[str, Any]):
        """Process RAW image file."""
        self.processing_progress.emit("Processing RAW image...")

        # Try to extract thumbnail as fallback first
        thumbnail_fallback = None
        try:
            thumbnail_fallback = self.thumbnail_extractor.extract_thumbnail_from_raw(
                self.file_path)
        except Exception:
            pass

        # Choose processing method based on quality setting
        try:
            if self.use_quality_processing:
                rgb_image = self.raw_processor.process_raw_quality(
                    self.file_path, exif_data)
            else:
                rgb_image = self.raw_processor.process_raw_fast(
                    self.file_path, exif_data)

            if rgb_image is not None:
                # Apply orientation correction
                orientation = exif_data.get(
                    'orientation', 1) if exif_data else 1
                rgb_image = self._apply_orientation_correction(
                    rgb_image, orientation, exif_data)

                # Cache the processed image
                self.cache.put_full_image(self.file_path, rgb_image)

                # Emit the result
                self.processing_progress.emit("Processing complete")
                self.image_processed.emit(rgb_image)
            else:
                # RAW processing failed, try fallback
                if thumbnail_fallback is not None:
                    # Use thumbnail as fallback
                    orientation = exif_data.get(
                        'orientation', 1) if exif_data else 1
                    thumbnail_fallback = self._apply_orientation_correction(
                        thumbnail_fallback, orientation, exif_data)

                    self.processing_progress.emit(
                        "Using embedded thumbnail (RAW processing failed)")
                    self.image_processed.emit(thumbnail_fallback)
                else:
                    self.error_occurred.emit("Failed to process RAW file")

        except Exception as e:
            # Provide more specific error messages based on the error type
            error_msg = str(e)
            if "data corrupted" in error_msg.lower():
                error_msg = f"RAW processing failed due to LibRaw compatibility issue.\n\nThis is a known issue with LibRaw and certain RAW files.\nTry using a different RAW processor or contact the developer for updates.\n\nOriginal error: {error_msg}"
            elif "unsupported file format" in error_msg.lower():
                error_msg = f"This RAW file format may not be supported by your LibRaw version.\n\nOriginal error: {error_msg}"
            elif "input/output error" in error_msg.lower():
                error_msg = f"Cannot read the file. It may be corrupted or in use by another program.\n\nOriginal error: {error_msg}"
            elif "cannot allocate memory" in error_msg.lower():
                error_msg = f"Not enough memory to process this large RAW file.\n\nOriginal error: {error_msg}"

            # Try fallback if available
            if thumbnail_fallback is not None:
                orientation = exif_data.get(
                    'orientation', 1) if exif_data else 1
                thumbnail_fallback = self._apply_orientation_correction(
                    thumbnail_fallback, orientation, exif_data)

                self.processing_progress.emit(
                    "Using embedded thumbnail (RAW processing failed)")
                self.image_processed.emit(thumbnail_fallback)
            else:
                self.error_occurred.emit(
                    f"Error processing RAW file:\n{error_msg}")

    def _process_regular_image(self, exif_data: Dict[str, Any]):
        """Process regular image file (JPEG, TIFF, etc.)."""
        self.processing_progress.emit("Loading image...")

        # For non-RAW files, we emit None to let the main thread handle with QPixmap
        # This is more efficient for regular image formats
        self.image_processed.emit(None)

    def _apply_orientation_correction(self, image_array: np.ndarray, orientation: int, exif_data: Dict[str, Any] = None) -> np.ndarray:
        """Apply orientation correction to image array."""
        # Check if camera stores data pre-rotated
        if self._is_camera_pre_rotated(exif_data):
            return image_array

        if orientation == 1:
            return image_array
        elif orientation == 2:
            return np.fliplr(image_array)
        elif orientation == 3:
            return np.rot90(image_array, 2)
        elif orientation == 4:
            return np.flipud(image_array)
        elif orientation == 5:
            return np.rot90(np.fliplr(image_array), 1)
        elif orientation == 6:
            # Rotated 90 CW - need to rotate 90 CW to correct (same as QPixmap transform.rotate(90))
            # np.rot90 with k=3 rotates 270° CCW = 90° CW
            return np.rot90(image_array, 3)
        elif orientation == 7:
            return np.rot90(np.fliplr(image_array), -1)
        elif orientation == 8:
            # Rotated 90 CCW - need to rotate 90 CW (270 CCW) to correct
            # np.rot90 with k=3 rotates 270° CCW = 90° CW
            return np.rot90(image_array, 3)
        else:
            return image_array

    def _is_camera_pre_rotated(self, exif_data: Dict[str, Any] = None) -> bool:
        """Check if camera stores data pre-rotated."""
        if not exif_data or not exif_data.get('camera_make'):
            return False

        make = exif_data['camera_make'].upper()
        # Sony, Leica, and Hasselblad often store RAW data pre-rotated
        return any(brand in make for brand in ['SONY', 'LEICA', 'HASSELBLAD'])


class PreloadManager(QObject):
    """Manages preloading of adjacent images for fast navigation using RAWProcessor (v0.5 style)."""

    def __init__(self, max_preload_threads: int = 2, processor_class=None):
        super().__init__()
        self.max_preload_threads = max_preload_threads
        self.active_threads = {}
        self.cache = get_image_cache()
        # Accept processor class as parameter to avoid circular imports
        # If None, will use EnhancedRAWProcessor as fallback
        self.processor_class = processor_class

    def _is_raw_file(self, file_path: str) -> bool:
        """Check if file is a RAW format."""
        raw_exts = {
            '.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef',
            '.srw', '.x3f', '.raf', '.3fr', '.fff', '.iiq', '.cap', '.erf',
            '.mef', '.mos', '.nrw', '.rwl', '.srf'
        }
        ext = os.path.splitext(file_path)[1].lower()
        return ext in raw_exts

    def preload_images(self, file_paths: list, priority_order: list = None):
        """Preload images in background threads using RAWProcessor (v0.5 style)."""
        if priority_order is None:
            priority_order = file_paths

        # Cancel existing preload threads
        self.cancel_all_preloads()

        # Use provided processor class or fallback to EnhancedRAWProcessor
        ProcessorClass = self.processor_class if self.processor_class else EnhancedRAWProcessor
        use_raw_processor = (self.processor_class is not None)

        # Start preloading up to max_preload_threads images
        for i, file_path in enumerate(priority_order[:self.max_preload_threads]):
            if file_path not in self.active_threads:
                # Check if already cached
                if self.cache.get_full_image(file_path) is not None:
                    continue

                # Start preload thread
                if use_raw_processor:
                    # Use RAWProcessor (v0.5 style)
                    is_raw = self._is_raw_file(file_path)
                    thread = ProcessorClass(
                        file_path, is_raw=is_raw, use_full_resolution=False)
                    # Connect signals - RAWProcessor uses image_processed signal
                    thread.image_processed.connect(
                        lambda img, fp=file_path: self._on_preload_complete(fp, img))
                    thread.error_occurred.connect(
                        lambda err, fp=file_path: self._on_preload_error(fp, err))
                else:
                    # Fallback to EnhancedRAWProcessor
                    thread = ProcessorClass(
                        file_path, use_quality_processing=False)
                    thread.image_processed.connect(
                        lambda img, fp=file_path: self._on_preload_complete(fp, img))
                
                thread.finished.connect(
                    lambda fp=file_path: self._cleanup_thread(fp))
                
                self.active_threads[file_path] = thread
                thread.start()

    def _on_preload_complete(self, file_path: str, rgb_image):
        """Handle preloaded image completion."""
        # Image is automatically cached by RAWProcessor/EnhancedRAWProcessor
        # Just clean up the thread reference
        if file_path in self.active_threads:
            # Thread will be cleaned up by finished signal
            pass

    def _on_preload_error(self, file_path: str, error_msg: str):
        """Handle preload error."""
        # Log error but don't show to user (preloading is background operation)
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Preload error for {os.path.basename(file_path)}: {error_msg}")
        # Clean up thread on error
        if file_path in self.active_threads:
            self._cleanup_thread(file_path)

    def cancel_all_preloads(self):
        """Cancel all active preload threads."""
        import logging
        logger = logging.getLogger(__name__)
        
        threads_to_cleanup = list(self.active_threads.items())
        logger.debug(f"cancel_all_preloads(): Cancelling {len(threads_to_cleanup)} preload threads")
        
        for file_path, thread in threads_to_cleanup:
            try:
                file_basename = os.path.basename(file_path)
                # Use cleanup() if available (RAWProcessor), otherwise use stop_processing()
                if hasattr(thread, 'cleanup'):
                    # cleanup() will handle thread stopping and rawpy handle closing safely
                    logger.debug(f"cancel_all_preloads(): Calling cleanup() for {file_basename}")
                    thread.cleanup()
                else:
                    # For EnhancedRAWProcessor, use stop_processing and wait
                    logger.debug(f"cancel_all_preloads(): Using stop_processing() for {file_basename}")
                    if hasattr(thread, 'stop_processing'):
                        thread.stop_processing()
                    
                    # OPTIMIZED: Use shorter initial wait, but allow longer if needed
                    # Preload threads are lower priority, so we can be more aggressive
                    if hasattr(thread, 'wait'):
                        wait_result = thread.wait(100)  # Initial 100ms wait (fast path)
                        logger.debug(f"cancel_all_preloads(): wait(100) returned {wait_result} for {file_basename}")
                        if not wait_result and hasattr(thread, 'isRunning') and thread.isRunning():
                            # Thread still running, wait longer for rawpy operations
                            additional_wait = thread.wait(200)  # Additional 200ms for rawpy operations
                            logger.debug(f"cancel_all_preloads(): Additional wait(200) returned {additional_wait} for {file_basename}")
                            if not additional_wait and thread.isRunning():
                                logger.debug(f"cancel_all_preloads(): Terminating thread for {file_basename}")
                                thread.terminate()
                                thread.wait(50)  # Short wait after terminate
            except Exception as e:
                logger.warning(f"cancel_all_preloads(): Error cleaning up thread for {os.path.basename(file_path)}: {e}", exc_info=True)
        
        self.active_threads.clear()
        logger.debug(f"cancel_all_preloads(): All preload threads cancelled")

    def _cleanup_thread(self, file_path: str):
        """Clean up finished thread."""
        if file_path in self.active_threads:
            del self.active_threads[file_path]
