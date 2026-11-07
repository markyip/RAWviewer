"""Background RAW / pixmap processing."""
import contextlib
import logging
import os
import threading
import traceback

import rawpy
from PyQt6.QtCore import QThread, pyqtSignal

from enhanced_raw_processor.thumbnail_extractor import ThumbnailExtractor
from metadata.backend import process_file_from_path


class RAWProcessorPart1(QThread):
    """Thread for processing RAW images to avoid UI blocking (part 1)."""
    image_processed = pyqtSignal(object)  # Accepts np.ndarray or None
    error_occurred = pyqtSignal(str)
    # Signal when thumbnail fallback is used
    thumbnail_fallback_used = pyqtSignal(str)
    # Progress and metadata signals (improved from EnhancedRAWProcessor)
    processing_progress = pyqtSignal(str)  # Status message for progress updates
    exif_data_ready = pyqtSignal(dict)  # EXIF data dictionary when ready

    def __init__(self, file_path, is_raw, use_full_resolution=False):
        super().__init__()
        self.file_path = file_path
        self.is_raw = is_raw
        self.use_full_resolution = use_full_resolution  # Force full resolution when True
        self._should_stop = False
        self._raw_handle = None  # Track rawpy handle for cleanup
        self._raw_handle_lock = threading.Lock()  # Lock for rawpy handle access
        self._use_fast_processing = None  # Store processing mode for logging
        # Use ThumbnailExtractor for cleaner thumbnail extraction (following 複製 version pattern)
        self.thumbnail_extractor = ThumbnailExtractor()

    def stop_processing(self):
        """Request processing to stop"""
        self._should_stop = True

    def cleanup(self):
        """Clean up the thread gracefully - optimized for fast navigation"""
        logger = logging.getLogger(__name__)

        file_basename = os.path.basename(self.file_path) if hasattr(self, 'file_path') else 'unknown'
        logger.debug(f"RAWProcessor.cleanup() called for: {file_basename}")

        try:
            # CRITICAL: Stop processing first, but DO NOT close rawpy handle yet
            # The thread might still be using it in raw.postprocess()
            logger.debug(f"Calling stop_processing() for: {file_basename}")
            self.stop_processing()
            logger.debug(f"stop_processing() completed for: {file_basename}")

            # Wait for thread to finish gracefully BEFORE closing rawpy handle
            # This ensures any ongoing rawpy operations complete safely
            # OPTIMIZATION: Check if thread is already finished first to avoid unnecessary waits
            is_running = self.isRunning()
            logger.debug(f"Thread is_running: {is_running} for: {file_basename}")

            if is_running:
                logger.debug(f"Thread is running, calling quit() for: {file_basename}")
                self.quit()

                # OPTIMIZED: Use shorter initial wait, but allow longer if needed
                # Most threads will stop quickly if they check _should_stop flag
                wait_result = self.wait(100)  # Initial 100ms wait (fast path for most cases)
                logger.debug(f"wait(100) returned: {wait_result}, is_running: {self.isRunning()} for: {file_basename}")

                if not wait_result and self.isRunning():
                    # Thread is still running, likely in rawpy operation
                    # Wait longer to allow rawpy operations to complete safely
                    logger.debug(f"Thread still running after initial wait, waiting additional 300ms for rawpy operations: {file_basename}")
                    additional_wait = self.wait(300)  # Additional 300ms for rawpy operations
                    logger.debug(f"Additional wait(300) returned: {additional_wait}, is_running: {self.isRunning()} for: {file_basename}")

                    if not additional_wait and self.isRunning():
                        # Thread still running after all waits - likely stuck or in long operation
                        # Terminate it, but this should be rare
                        logger.debug(f"Thread still running after all waits, calling terminate() for: {file_basename}")
                        self.terminate()
                        terminate_wait = self.wait(50)  # Short wait after terminate
                        logger.debug(f"After terminate(), wait(50) returned: {terminate_wait}, is_running: {self.isRunning()} for: {file_basename}")
                else:
                    logger.debug(f"Thread stopped gracefully for: {file_basename}")
            else:
                logger.debug(f"Thread not running, skip quit/wait for: {file_basename}")

            # NOW it's safe to close rawpy handle - thread has finished or been terminated
            logger.debug(f"Attempting to close rawpy handle for: {file_basename}")
            with self._raw_handle_lock:
                if self._raw_handle is not None:
                    try:
                        logger.debug(f"Closing rawpy handle for: {file_basename}")
                        self._raw_handle.close()
                        logger.debug(f"rawpy handle closed successfully for: {file_basename}")
                    except Exception as close_error:
                        error_str = str(close_error)
                        error_type = type(close_error).__name__
                        is_cancellation = (
                            self._should_stop or
                            'OutOfOrderCall' in error_type or
                            'LibRaw' in error_type or
                            'Out of order' in error_str or
                            'out of order' in error_str.lower()
                        )
                        if not is_cancellation:
                            logger.warning(f"Error closing rawpy handle for {file_basename}: {close_error}")
                        else:
                            logger.debug(f"Expected cancellation error when closing handle for {file_basename}: {close_error}")
                    finally:
                        self._raw_handle = None
                        logger.debug(f"rawpy handle reference cleared for: {file_basename}")
                else:
                    logger.debug(f"No rawpy handle to close for: {file_basename}")

            logger.debug(f"RAWProcessor.cleanup() completed successfully for: {file_basename}")
        except Exception as e:
            logger.error(f"Error in RAWProcessor.cleanup() for {file_basename}: {e}", exc_info=True)
            logger.debug(f"Cleanup error traceback: {traceback.format_exc()}")
            # Try to clear handle even on error
            try:
                with self._raw_handle_lock:
                    self._raw_handle = None
            except Exception:
                pass

    def get_orientation_from_exif(self, file_path):
        """Extract orientation from EXIF data - optimized for minimal logging"""
        try:
            tags = process_file_from_path(
                file_path, details=False, stop_tag="Image Orientation"
            )

            # Check for orientation tag
            orientation_tag = tags.get("Image Orientation")
            if orientation_tag:
                orientation_str = str(orientation_tag)

                # Map orientation descriptions to numeric values
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

                return orientation_map.get(orientation_str, 1)

            return 1  # Default orientation (no rotation needed)
        except Exception:
            return 1  # Default orientation if EXIF reading fails

    def apply_orientation_correction(self, image_array, orientation):
        """Apply orientation correction to numpy array, with safety guards to prevent double rotation"""
        if self.is_raw_data_pre_rotated():
            return image_array

        from common_image_loader import apply_container_orientation_to_array
        from image_cache import get_image_cache

        cache = get_image_cache()
        exif_data = cache.get_exif(self.file_path) if cache else None

        return apply_container_orientation_to_array(image_array, orientation, exif_data)

    def is_raw_data_pre_rotated(self):
        """Check if this camera/file stores RAW data pre-rotated - optimized"""
        # CRITICAL: Even for SONY/Leica/Hasselblad cameras, we should apply orientation correction
        # based on EXIF orientation tag, as the RAW data may not always be pre-rotated correctly
        # Only skip orientation correction if we're certain the data is already correctly oriented
        # For now, we'll apply orientation correction for all RAW files to ensure correctness
        return False  # Always apply orientation correction for RAW files

        # OLD CODE (disabled): Some cameras may store pre-rotated data, but it's safer to apply correction
        # try:
        #     # Read camera make from EXIF - only extract what we need
        #     with open(self.file_path, 'rb') as f:
        #         tags = exifread.process_file(f, details=False, stop_tag='Image Make')
        #         make = tags.get('Image Make')
        #
        #         if make:
        #             make_str = str(make).upper()
        #             # Sony cameras often store RAW data pre-rotated
        #             if 'SONY' in make_str:
        #                 return True
        #
        #             # Leica cameras also store RAW data pre-rotated
        #             if 'LEICA' in make_str:
        #                 return True
        #
        #             # Hasselblad cameras also store RAW data pre-rotated
        #             if 'HASSELBLAD' in make_str:
        #                 return True
        #
        # except Exception:
        #     pass
        #
        # return False

    def is_canon_camera(self):
        """Check if this is a Canon camera that needs special white balance processing"""
        try:
            # First try to detect by file extension (more reliable for CR3)
            file_ext = os.path.splitext(self.file_path)[1].lower()
            if file_ext in ['.cr2', '.cr3']:
                return True

            # Fallback to EXIF detection for other formats - only read Image Make tag
            tags = process_file_from_path(
                self.file_path, details=False, stop_tag="Image Make"
            )
            make = tags.get("Image Make")

            if make:
                make_str = str(make).upper()
                # Canon cameras need special white balance processing
                if "CANON" in make_str:
                    return True

        except Exception:
            pass

        return False

    def is_fujifilm_camera(self):
        """Check if this is a Fujifilm camera that needs special white balance processing"""
        try:
            # First try to detect by file extension (more reliable for RAF)
            file_ext = os.path.splitext(self.file_path)[1].lower()
            if file_ext in ['.raf']:
                return True

            # Fallback to EXIF detection for other formats - only read Image Make tag
            tags = process_file_from_path(
                self.file_path, details=False, stop_tag="Image Make"
            )
            make = tags.get("Image Make")

            if make:
                make_str = str(make).upper()
                # Fujifilm cameras need special white balance processing
                if "FUJIFILM" in make_str or "FUJI" in make_str:
                    return True

        except Exception:
            pass

        return False

    def _check_available_memory(self):
        """Check if there's enough memory to process the file"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024 ** 3)

            # Check file size
            file_size_mb = os.path.getsize(self.file_path) / (1024 * 1024)

            # Estimate memory needed (conservative: 3x file size for processing)
            estimated_needed_gb = (file_size_mb * 3) / 1024

            # Need at least 2x estimated memory available
            if available_gb < (estimated_needed_gb * 2):
                return False, available_gb, estimated_needed_gb

            return True, available_gb, estimated_needed_gb
        except Exception:
            # If psutil not available or error, assume we can proceed
            return True, 0, 0

    def process_raw_with_camera_specific_settings(self, raw):
        """Process RAW data with camera-specific settings with improved memory management - thread-safe"""
        logger = logging.getLogger(__name__)
        try:
            # Check if we should stop before processing
            if self._should_stop:
                return None

            # Verify raw handle is still valid (thread-safe)
            with self._raw_handle_lock:
                if self._raw_handle is None or self._raw_handle != raw:
                    return None  # Handle was closed, stop processing

            # Check available memory before processing
            has_memory, available_gb, needed_gb = self._check_available_memory()
            if not has_memory:
                raise MemoryError(
                    f"Insufficient memory: {available_gb:.1f}GB available, "
                    f"estimated {needed_gb:.1f}GB needed. "
                    f"Try closing other applications or use a smaller image."
                )

            # Check if we should stop after memory check
            if self._should_stop:
                return None

            # Check if we should force full resolution (on-demand loading when user zooms)
            if self.use_full_resolution:
                use_fast_processing = False  # Force full resolution
                use_auto_bright = False  # Disable auto-brightness to preserve original RAW colors
                logger.debug("Loading full resolution on-demand (user zoomed in)")
            else:
                # Check file size to determine if we should use faster processing
                # Use half_size by default for fast loading (<0.5s target)
                # Full resolution will be loaded on-demand when user zooms in
                file_size_mb = os.path.getsize(self.file_path) / (1024 * 1024)
                # OPTIMIZATION: Lower threshold to 20MB for faster loading on more files
                use_fast_processing = file_size_mb > 20  # Use fast processing (half_size) for files > 20MB

                # For very large files (>80MB), force half_size for memory efficiency
                if file_size_mb > 80:
                    use_fast_processing = True
                    logger.debug(f"Very large file detected ({file_size_mb:.1f}MB), using half_size for memory efficiency")

                # CRITICAL: Disable auto-brightness to show original RAW colors
                # Auto-brightness applies exposure compensation which changes the original RAW appearance
                use_auto_bright = False  # Disable auto-brightness to preserve original RAW colors

            # Store processing mode for logging
            self._use_fast_processing = use_fast_processing

            # Check if we should stop before camera-specific processing
            if self._should_stop:
                return None
            # Check if this is a Canon camera
            if self.is_canon_camera():
                # Canon cameras (especially CR3) need proper white balance correction
                # to avoid red hue issues. Try camera white balance first.
                logger.debug("Applying Canon-specific white balance correction...")
                try:
                    if self._should_stop:
                        return None
                    # Verify handle is still valid before postprocess
                    with self._raw_handle_lock:
                        if self._raw_handle is None or self._raw_handle != raw:
                            return None
                    # Optimized processing parameters for faster loading
                    # Use auto-brightness for full resolution to match initial display brightness
                    # Performance optimizations: use camera WB (faster), 8-bit output, fast gamma
                    postprocess_params = {
                        'use_camera_wb': True,
                        'half_size': use_fast_processing,
                        'output_bps': 8,  # Use 8-bit for faster processing
                        'no_auto_bright': not use_auto_bright,  # Use auto-brightness for full resolution
                        'gamma': (2.222, 4.5),  # Standard sRGB gamma
                        'user_flip': 0
                    }
                    # Add performance optimizations if available
                    with contextlib.suppress(AttributeError, TypeError):
                        # Use fastest demosaicing algorithm for speed (if supported)
                        postprocess_params['demosaic_algorithm'] = rawpy.DemosaicAlgorithm.LINEAR
                    rgb_image = raw.postprocess(**postprocess_params)
                    # Check again after postprocess
                    if self._should_stop:
                        return None
                    return rgb_image
                except Exception:
                    # If camera WB fails, try auto white balance
                    try:
                        if self._should_stop:
                            return None
                        with self._raw_handle_lock:
                            if self._raw_handle is None or self._raw_handle != raw:
                                return None
                        # Optimized processing parameters for faster loading
                        # Use auto-brightness for full resolution to match initial display brightness
                        # Performance optimizations: use auto WB (fallback), 8-bit output, fast gamma
                        postprocess_params = {
                            'use_auto_wb': True,
                            'half_size': use_fast_processing,
                            'output_bps': 8,  # Use 8-bit for faster processing
                            'no_auto_bright': not use_auto_bright,  # Use auto-brightness for full resolution
                            'gamma': (2.222, 4.5),  # Standard sRGB gamma
                        'user_flip': 0
                        }
                        # Add performance optimizations if available
                        with contextlib.suppress(AttributeError, TypeError):
                            postprocess_params['demosaic_algorithm'] = rawpy.DemosaicAlgorithm.LINEAR
                        rgb_image = raw.postprocess(**postprocess_params)
                        if self._should_stop:
                            return None
                        return rgb_image
                    except Exception:
                        # If both fail, use default processing
                        if self._should_stop:
                            return None
                        with self._raw_handle_lock:
                            if self._raw_handle is None or self._raw_handle != raw:
                                return None
                        # Optimized processing parameters for faster loading
                        # Use auto-brightness for full resolution to match initial display brightness
                        # Performance optimizations: default processing, 8-bit output, fast gamma
                        postprocess_params = {
                            'half_size': use_fast_processing,
                            'output_bps': 8,  # Use 8-bit for faster processing
                            'no_auto_bright': not use_auto_bright,  # Use auto-brightness for full resolution
                            'gamma': (2.222, 4.5),  # Standard sRGB gamma
                        'user_flip': 0
                        }
                        # Add performance optimizations if available
                        with contextlib.suppress(AttributeError, TypeError):
                            postprocess_params['demosaic_algorithm'] = rawpy.DemosaicAlgorithm.LINEAR
                        rgb_image = raw.postprocess(**postprocess_params)
                        if self._should_stop:
                            return None
                        return rgb_image
            # Check if this is a Fujifilm camera
            elif self.is_fujifilm_camera():
                # Fujifilm cameras (especially RAF) need proper white balance correction
                # to avoid green hue issues and improve processing speed
                if use_fast_processing:
                    logger.debug(f"Applying Fujifilm-specific processing with fast mode for large file ({file_size_mb:.1f}MB)...")
                else:
                    logger.debug("Applying Fujifilm-specific white balance correction...")
                try:
                    if self._should_stop:
                        return None
                    with self._raw_handle_lock:
                        if self._raw_handle is None or self._raw_handle != raw:
                            return None
                    # Optimized processing parameters for faster loading
                    # Use auto-brightness for full resolution to match initial display brightness
                    # Performance optimizations: use camera WB (faster), 8-bit output, fast gamma
                    postprocess_params = {
                        'use_camera_wb': True,
                        'half_size': use_fast_processing,
                        'output_bps': 8,  # Use 8-bit for faster processing
                        'no_auto_bright': not use_auto_bright,  # Use auto-brightness for full resolution
                        'gamma': (2.222, 4.5),  # Standard sRGB gamma
                        'user_flip': 0
                    }
                    # Add performance optimizations if available
                    with contextlib.suppress(AttributeError, TypeError):
                        # Use fastest demosaicing algorithm for speed (if supported)
                        postprocess_params['demosaic_algorithm'] = rawpy.DemosaicAlgorithm.LINEAR
                    rgb_image = raw.postprocess(**postprocess_params)
                    if self._should_stop:
                        return None
                    return rgb_image
                except Exception:
                    # If camera WB fails, try auto white balance
                    try:
                        if self._should_stop:
                            return None
                        with self._raw_handle_lock:
                            if self._raw_handle is None or self._raw_handle != raw:
                                return None
                        # Optimized processing parameters for faster loading
                        # Use auto-brightness for full resolution to match initial display brightness
                        # Performance optimizations: use auto WB (fallback), 8-bit output, fast gamma
                        postprocess_params = {
                            'use_auto_wb': True,
                            'half_size': use_fast_processing,
                            'output_bps': 8,  # Use 8-bit for faster processing
                            'no_auto_bright': not use_auto_bright,  # Use auto-brightness for full resolution
                            'gamma': (2.222, 4.5),  # Standard sRGB gamma
                        'user_flip': 0
                        }
                        # Add performance optimizations if available
                        with contextlib.suppress(AttributeError, TypeError):
                            postprocess_params['demosaic_algorithm'] = rawpy.DemosaicAlgorithm.LINEAR
                        rgb_image = raw.postprocess(**postprocess_params)
                        if self._should_stop:
                            return None
                        return rgb_image
                    except Exception:
                        # If both fail, use default processing
                        if self._should_stop:
                            return None
                        with self._raw_handle_lock:
                            if self._raw_handle is None or self._raw_handle != raw:
                                return None
                        # Optimized processing parameters for faster loading
                        # Use auto-brightness for full resolution to match initial display brightness
                        # Performance optimizations: default processing, 8-bit output, fast gamma
                        postprocess_params = {
                            'half_size': use_fast_processing,
                            'output_bps': 8,  # Use 8-bit for faster processing
                            'no_auto_bright': not use_auto_bright,  # Use auto-brightness for full resolution
                            'gamma': (2.222, 4.5),  # Standard sRGB gamma
                        'user_flip': 0
                        }
                        # Add performance optimizations if available
                        with contextlib.suppress(AttributeError, TypeError):
                            postprocess_params['demosaic_algorithm'] = rawpy.DemosaicAlgorithm.LINEAR
                        rgb_image = raw.postprocess(**postprocess_params)
                        if self._should_stop:
                            return None
                        return rgb_image
            else:
                # For other cameras, use default processing
                if self._should_stop:
                    return None
                with self._raw_handle_lock:
                    if self._raw_handle is None or self._raw_handle != raw:
                        return None
                # Optimized processing parameters for faster loading
                # Use auto-brightness for full resolution to match initial display brightness
                # Performance optimizations: default processing, 8-bit output, fast gamma
                postprocess_params = {
                    'half_size': use_fast_processing,
                    'output_bps': 8,  # Use 8-bit for faster processing
                    'no_auto_bright': not use_auto_bright,  # Use auto-brightness for full resolution
                    'gamma': (2.222, 4.5),  # Standard sRGB gamma
                    'user_flip': 0
                }
                # Add performance optimizations if available
                with contextlib.suppress(AttributeError, TypeError):
                    postprocess_params['demosaic_algorithm'] = rawpy.DemosaicAlgorithm.LINEAR
                rgb_image = raw.postprocess(**postprocess_params)
                if self._should_stop:
                    return None
                return rgb_image
        except Exception:
            # Fallback to default processing if anything fails
            if self._should_stop:
                return None
            with self._raw_handle_lock:
                if self._raw_handle is None or self._raw_handle != raw:
                    return None
            # Fallback with optimized parameters for speed (keep LibRaw auto-brightness off)
            postprocess_params = {
                'output_bps': 8,  # Use 8-bit for faster processing
                'no_auto_bright': True,
                'gamma': (2.222, 4.5),  # Standard sRGB gamma
                'user_flip': 0
            }
            # Add performance optimizations if available
            with contextlib.suppress(AttributeError, TypeError):
                postprocess_params['demosaic_algorithm'] = rawpy.DemosaicAlgorithm.LINEAR
            rgb_image = raw.postprocess(**postprocess_params)
            if self._should_stop:
                return None
            return rgb_image

