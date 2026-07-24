"""Background RAW / pixmap processing."""
import logging, threading, os, time, traceback
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from image_load_manager import yield_if_current_task_active

class RAWProcessor(QThread):
    """Thread for processing RAW images to avoid UI blocking"""
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
        import logging
        import traceback
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

    def _postprocess_with_wb_fallback(self, raw, use_fast_processing, use_auto_bright):
        """Postprocess trying camera WB, then auto WB, then default settings.

        Each attempt is a full LibRaw demosaic; parameters are identical to the
        previous per-camera code paths (8-bit output, fast gamma, linear demosaic).
        """
        wb_attempts = ({'use_camera_wb': True}, {'use_auto_wb': True}, {})
        last_attempt = len(wb_attempts) - 1
        for i, wb_params in enumerate(wb_attempts):
            try:
                if self._should_stop:
                    return None
                # Verify handle is still valid before postprocess
                with self._raw_handle_lock:
                    if self._raw_handle is None or self._raw_handle != raw:
                        return None
                # Optimized processing parameters for faster loading
                # Use auto-brightness for full resolution to match initial display brightness
                # Performance optimizations: 8-bit output, fast gamma
                postprocess_params = {
                    'half_size': use_fast_processing,
                    'output_bps': 8,  # Use 8-bit for faster processing
                    'no_auto_bright': not use_auto_bright,  # Use auto-brightness for full resolution
                    'gamma': (2.222, 4.5),  # Standard sRGB gamma
                    'user_flip': 0
                }
                postprocess_params.update(wb_params)
                # Add performance optimizations if available
                try:
                    # Use fastest demosaicing algorithm for speed (if supported)
                    postprocess_params['demosaic_algorithm'] = rawpy.DemosaicAlgorithm.LINEAR
                except (AttributeError, TypeError):
                    pass  # Parameter not available in this rawpy version
                rgb_image = raw.postprocess(**postprocess_params)
                # Check again after postprocess
                if self._should_stop:
                    return None
                return rgb_image
            except Exception:
                # On last attempt (default params), let the error propagate to
                # the outer fallback in process_raw_with_camera_specific_settings
                if i == last_attempt:
                    raise
        return None

    def process_raw_with_camera_specific_settings(self, raw):
        """Process RAW data with camera-specific settings with improved memory management - thread-safe"""
        import logging
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
                logger.debug(f"Applying Canon-specific white balance correction...")
                return self._postprocess_with_wb_fallback(raw, use_fast_processing, use_auto_bright)
            # Check if this is a Fujifilm camera
            elif self.is_fujifilm_camera():
                # Fujifilm cameras (especially RAF) need proper white balance correction
                # to avoid green hue issues and improve processing speed
                if use_fast_processing:
                    logger.debug(f"Applying Fujifilm-specific processing with fast mode for large file ({file_size_mb:.1f}MB)...")
                else:
                    logger.debug(f"Applying Fujifilm-specific white balance correction...")
                return self._postprocess_with_wb_fallback(raw, use_fast_processing, use_auto_bright)
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
                try:
                    postprocess_params['demosaic_algorithm'] = rawpy.DemosaicAlgorithm.LINEAR
                except (AttributeError, TypeError):
                    pass
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
            try:
                postprocess_params['demosaic_algorithm'] = rawpy.DemosaicAlgorithm.LINEAR
            except (AttributeError, TypeError):
                pass
            rgb_image = raw.postprocess(**postprocess_params)
            if self._should_stop:
                return None
            return rgb_image

    def run(self):
        """Main processing method with improved error handling and resource management"""
        import logging
        logger = logging.getLogger(__name__)
        try:
            if self.is_raw:
                # Emit initial progress signal
                filename = os.path.basename(self.file_path)
                logger.info(f"[RAW_PROC] ========== RAWProcessor.run() STARTED for {filename} ==========")
                self.processing_progress.emit(f"Loading {filename}...")
                
                # Get orientation from EXIF data
                # Check if we should stop before starting
                if self._should_stop:
                    logger.info(f"[RAW_PROC] Processing stopped before starting for: {filename}")
                    return
                
                # OPTIMIZATION: Check thumbnail cache FIRST before opening RAW file
                # This avoids opening RAW file if thumbnail is already cached
                from image_cache import get_image_cache
                cache = get_image_cache()
                thumbnail_data = cache.get_thumbnail(self.file_path)
                
                # OPTIMIZATION: Check EXIF cache before opening RAW file
                # This allows us to emit EXIF data immediately if cached
                cached_exif = cache.get_exif(self.file_path)
                exif_data = None
                original_width = None
                original_height = None
                
                if cached_exif:
                    # Use cached EXIF data
                    exif_data = cached_exif
                    original_width = cached_exif.get('original_width')
                    original_height = cached_exif.get('original_height')
                    logger.debug(f"[RAW_PROC] EXIF data found in cache, original dimensions: {original_width}x{original_height}")
                    
                    # Only emit EXIF data if original dimensions are also in cache
                    # Otherwise, wait until we extract dimensions from RAW file
                    if original_width and original_height:
                        # Emit EXIF data immediately if available with dimensions
                        if not self._should_stop:
                            logger.info(f"[RAW_PROC] Emitting cached exif_data_ready signal with dimensions: {original_width}x{original_height}")
                            self.exif_data_ready.emit(exif_data)
                            logger.info(f"[RAW_PROC] Cached exif_data_ready signal emitted")
                    else:
                        logger.debug(f"[RAW_PROC] Cached EXIF data found but missing original dimensions, will emit after RAW file is opened")
                
                if thumbnail_data is not None:
                    logger.info(f"[RAW_PROC] Thumbnail found in cache: {os.path.basename(self.file_path)} ({thumbnail_data.shape[1]}x{thumbnail_data.shape[0]})")
                    # Emit cached thumbnail immediately for fast display
                    if not self.use_full_resolution:
                        logger.info(f"[RAW_PROC] Emitting cached thumbnail immediately")
                        self.thumbnail_fallback_used.emit("Loading thumbnail...")
                        self.image_processed.emit(thumbnail_data)
                        logger.info(f"[RAW_PROC] Cached thumbnail emitted successfully")
                else:
                    if use_libraw_consistent_preview_first(self.file_path):
                        logger.debug(
                            f"[RAW_PROC] LibRaw-consistent preview: skipping embedded JPEG for "
                            f"{os.path.basename(self.file_path)}"
                        )
                    else:
                        logger.debug(f"Thumbnail not in cache, will try embedded JPEG first: {os.path.basename(self.file_path)}")
                        # OPTIMIZATION: Try to extract embedded JPEG thumbnail BEFORE opening RAW file
                        # This is much faster than opening the entire RAW file
                        try:
                            import rawpy
                            import time
                            embedded_start = time.time()
                            # Quick check: try to extract embedded thumbnail without full RAW processing
                            with rawpy.imread(self.file_path) as raw_quick:
                                thumb = raw_quick.extract_thumb()
                                if thumb is not None and thumb.format == rawpy.ThumbFormat.JPEG:
                                    # Successfully extracted embedded JPEG thumbnail
                                    embedded_time = time.time() - embedded_start
                                    logger.info(f"[RAW_PROC] ⚡ FAST: Extracted embedded JPEG thumbnail in {embedded_time*1000:.1f}ms")
                                    safe_print(f"[PERF] ⚡ FAST THUMBNAIL: Embedded JPEG extracted in {embedded_time*1000:.1f}ms")
                                    
                                    # Convert JPEG bytes to numpy array
                                    from io import BytesIO
                                    from PIL import Image, ImageOps
                                    jpeg_data = thumb.data
                                    
                                    # Save to disk cache for future use (much faster than extracting again)
                                    try:
                                        cache.disk_thumbnail_cache.put(self.file_path, jpeg_data)
                                        logger.debug(f"[RAW_PROC] Saved embedded JPEG to disk cache")
                                    except Exception as cache_error:
                                        logger.debug(f"[RAW_PROC] Failed to save to disk cache: {cache_error}")
                                    
                                    # Embedded JPEG: segment EXIF transpose + container EXIF (once, shared helper)
                                    from common_image_loader import (
                                        decode_embedded_jpeg_bytes,
                                        apply_container_orientation_to_array,
                                    )

                                    thumbnail_data = decode_embedded_jpeg_bytes(
                                        jpeg_data, 0
                                    )
                                    if thumbnail_data is not None:
                                        orientation = self.get_orientation_from_exif(
                                            self.file_path
                                        )
                                        if orientation != 1:
                                            exif_for_o = cached_exif or {}
                                            if not exif_for_o.get("orientation"):
                                                exif_for_o = dict(exif_for_o)
                                                exif_for_o["orientation"] = orientation
                                            thumbnail_data = apply_container_orientation_to_array(
                                                thumbnail_data,
                                                orientation,
                                                exif_for_o or None,
                                            )

                                        cache.put_thumbnail(self.file_path, thumbnail_data)
                                        logger.info(
                                            "[RAW_PROC] Embedded JPEG thumbnail cached: %s (%dx%d)",
                                            os.path.basename(self.file_path),
                                            thumbnail_data.shape[1],
                                            thumbnail_data.shape[0],
                                        )

                                        if not self.use_full_resolution:
                                            logger.info(
                                                "[RAW_PROC] Emitting embedded JPEG thumbnail immediately"
                                            )
                                            self.thumbnail_fallback_used.emit(
                                                "Loading thumbnail..."
                                            )
                                            self._orientation_already_applied = True
                                            self.image_processed.emit(thumbnail_data)
                                            logger.info(
                                                "[RAW_PROC] Embedded JPEG thumbnail emitted successfully"
                                            )
                                else:
                                    logger.debug(f"[RAW_PROC] No embedded JPEG thumbnail found, will process RAW")
                        except Exception as embedded_error:
                            # If embedded extraction fails, continue with normal RAW processing
                            logger.debug(f"[RAW_PROC] Embedded JPEG extraction failed (will process RAW): {embedded_error}")
                
                # Get orientation from cached EXIF if available, otherwise extract
                # CRITICAL: Always extract orientation from file to ensure accuracy
                # Cached orientation may be incorrect or outdated
                orientation = self.get_orientation_from_exif(self.file_path)
                logger.debug(f"Extracted orientation from file: {orientation}")
                
                # Update cache with correct orientation
                if cached_exif:
                    cached_exif['orientation'] = orientation
                    cache.put_exif(self.file_path, cached_exif)
                    logger.debug(f"Updated cached orientation: {orientation}")

                logger.debug(f"Image orientation: {orientation}")
                
                # Check if we should stop after EXIF reading
                if self._should_stop:
                    logger.debug(f"Processing stopped after EXIF reading for: {self.file_path}")
                    return
                
                # OPTIMIZATION: Check if full image is already cached before opening RAW file
                # If both thumbnail and full image are cached, we can skip RAW file processing entirely
                cached_full_image = cache.get_full_image(self.file_path)
                if cached_full_image is not None:
                    logger.info(f"[RAW_PROC] Full image found in cache: {os.path.basename(self.file_path)} ({cached_full_image.shape[1]}x{cached_full_image.shape[0]})")
                    # If we're only loading full resolution (on-demand zoom), emit cached full image immediately
                    if self.use_full_resolution:
                        logger.info(f"[RAW_PROC] Emitting cached full image immediately (on-demand zoom)")
                        self.processing_progress.emit("Loading full resolution...")
                        # Apply orientation correction to cached image
                        cached_full_image = self.apply_orientation_correction(cached_full_image, orientation)
                        self.image_processed.emit(cached_full_image)
                        logger.info(f"[RAW_PROC] Cached full image emitted successfully")
                        return  # Skip RAW file processing entirely
                    # If thumbnail is cached and full image is also cached, we can skip processing
                    # unless user explicitly needs full resolution
                    elif thumbnail_data is not None:
                        logger.info(f"[RAW_PROC] Both thumbnail and full image cached, skipping RAW processing")
                        # Don't emit full image yet - wait for user to zoom in or request it
                        return  # Skip RAW file processing entirely
                
                # Only open RAW file if we need to:
                # 1. Thumbnail is not cached (need to extract/generate it)
                # 2. Full image is not cached (need to process it)
                # 3. User explicitly requested full resolution
                needs_raw_file = (
                    thumbnail_data is None or  # Need to extract/generate thumbnail
                    cached_full_image is None or  # Need to process full image
                    self.use_full_resolution  # User requested full resolution
                )
                
                if not needs_raw_file:
                    logger.info(f"[RAW_PROC] Skipping RAW file processing - all data cached: {os.path.basename(self.file_path)}")
                    return
                
                # Open RAW file (needed for thumbnail extraction or full image processing)
                try:
                    # First try to open the RAW file
                    # Store handle for potential cleanup
                    logger.info(f"[RAW_PROC] Opening RAW file: {os.path.basename(self.file_path)}")
                    raw = rawpy.imread(self.file_path)
                    logger.info(f"[RAW_PROC] RAW file opened successfully")
                    with self._raw_handle_lock:
                        self._raw_handle = raw
                    logger.info(f"[RAW_PROC] RAW handle stored and locked")
                    
                    # Extract and store original image dimensions from RAW metadata FIRST
                    # This will be used in status bar to show original size instead of processed size
                    # We do this BEFORE extracting EXIF data to ensure it's not overwritten
                    try:
                        original_width = raw.sizes.width
                        original_height = raw.sizes.height
                        # Store in cache for later retrieval
                        from image_cache import get_image_cache
                        cache = get_image_cache()
                        # Get existing EXIF cache or create new dict
                        cached_exif = cache.get_exif(self.file_path) or {}
                        # Store original dimensions in EXIF cache
                        cached_exif['original_width'] = original_width
                        cached_exif['original_height'] = original_height
                        cache.put_exif(self.file_path, cached_exif)
                        logger.debug(f"Original image dimensions stored: {original_width}x{original_height}")
                    except Exception as dim_error:
                        logger.debug(f"Could not extract original dimensions from RAW: {dim_error}")
                    
                    # OPTIMIZATION: Only extract EXIF if not already cached
                    # This avoids redundant EXIF extraction when data is already available
                    if not exif_data:
                        # Extract and cache full EXIF data for metadata display
                        self.processing_progress.emit("Reading metadata...")
                        from enhanced_raw_processor import EXIFExtractor
                        exif_extractor = EXIFExtractor()
                        exif_data = exif_extractor.extract_exif_data(self.file_path)
                        logger.debug(f"[RAW_PROC] EXIF data extracted from file")
                    else:
                        logger.debug(f"[RAW_PROC] Using cached EXIF data, skipping extraction")
                    
                    # CRITICAL: Ensure original dimensions are ALWAYS preserved in the EXIF cache
                    # (EXIFExtractor might have overwritten the cache, or cache might have been from previous session)
                    cached_exif = cache.get_exif(self.file_path) or {}
                    # Force update original dimensions - they come from RAW metadata which is authoritative
                    if original_width and original_height:
                        cached_exif['original_width'] = original_width
                        cached_exif['original_height'] = original_height
                        cache.put_exif(self.file_path, cached_exif)
                        logger.info(f"[RAW_PROC] Final stored original dimensions: {original_width}x{original_height}")
                    
                    # Emit EXIF data ready signal if not already emitted (from cache check above)
                    # Always emit if we extracted new EXIF data, or if we have EXIF data and haven't emitted yet
                    if exif_data and not self._should_stop:
                        # Emit if:
                        # 1. We extracted new EXIF data (not from cache), OR
                        # 2. We have cached EXIF but didn't emit it earlier (because dimensions were missing)
                        should_emit = False
                        if not cached_exif or cached_exif.get('original_width') is None:
                            # New EXIF data extracted
                            should_emit = True
                        elif original_width and original_height:
                            # We now have dimensions, check if we already emitted
                            # If cached_exif had dimensions, we would have emitted earlier
                            # So if we're here, we need to emit now
                            should_emit = True
                        
                        if should_emit:
                            # Ensure exif_data includes original dimensions
                            if original_width and original_height:
                                exif_data['original_width'] = original_width
                                exif_data['original_height'] = original_height
                            logger.info(f"[RAW_PROC] Emitting exif_data_ready signal with dimensions: {original_width}x{original_height}")
                            self.exif_data_ready.emit(exif_data)
                            logger.info(f"[RAW_PROC] exif_data_ready signal emitted")
                    
                    try:
                        # Check if we should stop before processing
                        if self._should_stop:
                            logger.debug(f"Processing stopped before thumbnail extraction for: {self.file_path}")
                            return
                        
                        # Skip thumbnail generation if we're only loading full resolution (on-demand zoom)
                        if self.use_full_resolution:
                            logger.info(f"[RAW_PROC] Skipping thumbnail generation - loading full resolution only: {os.path.basename(self.file_path)}")
                            thumbnail_data = None  # Skip thumbnail, go straight to full resolution
                        # Check thumbnail cache again (in case it was added by another thread)
                        # But if we already emitted cached thumbnail above, skip extraction
                        # Also skip if we already extracted embedded JPEG thumbnail above
                        elif thumbnail_data is None:
                            libraw_consistent = use_libraw_consistent_preview_first(self.file_path)
                            # OPTIMIZATION: Try to extract thumbnail using already-opened raw handle first
                            # This avoids reopening the file, which is much faster
                            logger.debug(f"Extracting thumbnail: {os.path.basename(self.file_path)}")
                            try:
                                if self._should_stop:
                                    return
                                
                                # OPTIMIZATION: Use already-opened raw handle if available (faster)
                                extracted_thumbnail = None
                                if not libraw_consistent:
                                    with self._raw_handle_lock:
                                        if self._raw_handle is not None and self._raw_handle == raw:
                                            # Use existing raw handle to extract thumbnail (much faster)
                                            try:
                                                thumb = raw.extract_thumb()
                                                if thumb is not None:
                                                    if thumb.format == rawpy.ThumbFormat.JPEG:
                                                        import io
                                                        from PIL import Image
                                                        jpeg_image = Image.open(io.BytesIO(thumb.data))
                                                        if jpeg_image.mode != 'RGB':
                                                            jpeg_image = jpeg_image.convert('RGB')
                                                        extracted_thumbnail = np.array(jpeg_image)
                                                    elif thumb.format == rawpy.ThumbFormat.BITMAP:
                                                        extracted_thumbnail = thumb.data
                                                    thumb_size = f"{extracted_thumbnail.shape[1]}x{extracted_thumbnail.shape[0]}" if extracted_thumbnail is not None else 'N/A'
                                                    logger.debug(f"Thumbnail extracted using existing raw handle: {thumb_size}")
                                                    safe_print(f"[PERF] ⚡ FAST THUMBNAIL: Extracted using existing raw handle ({thumb_size})")
                                            except Exception as thumb_extract_error:
                                                logger.debug(f"Failed to extract thumbnail from raw handle: {thumb_extract_error}")
                                                safe_print(f"[PERF] ⚠️  Raw handle extraction failed, falling back")
                                    
                                    # Fallback: Use ThumbnailExtractor if raw handle extraction failed
                                    if extracted_thumbnail is None:
                                        logger.debug(f"Falling back to ThumbnailExtractor for thumbnail extraction")
                                        fallback_start = time.time()
                                        extracted_thumbnail = self.thumbnail_extractor.extract_thumbnail_from_raw(self.file_path)
                                        fallback_time = time.time() - fallback_start
                                        if extracted_thumbnail is not None:
                                            safe_print(f"[PERF] 🔄 FALLBACK THUMBNAIL: Extracted via ThumbnailExtractor in {fallback_time*1000:.1f}ms")
                                
                                if self._should_stop:
                                    return
                                
                                if extracted_thumbnail is not None:
                                    logger.debug(f"Thumbnail extracted successfully: {extracted_thumbnail.shape[1]}x{extracted_thumbnail.shape[0]}")
                                    thumbnail_data = extracted_thumbnail
                                    
                                    # Resize thumbnail if too large (optimize for display and memory)
                                    # Use dynamic sizing based on typical display needs
                                    max_thumb_size = 1024  # Maximum thumbnail dimension for high-quality preview
                                    if len(thumbnail_data.shape) >= 2:
                                        h, w = thumbnail_data.shape[0], thumbnail_data.shape[1]
                                        if h > max_thumb_size or w > max_thumb_size:
                                            from PIL import Image
                                            logger.debug(f"Resizing thumbnail from {w}x{h} to max {max_thumb_size}x{max_thumb_size}")
                                            # Convert to PIL Image for resizing
                                            if len(thumbnail_data.shape) == 3:
                                                pil_image = Image.fromarray(thumbnail_data)
                                                pil_image.thumbnail((max_thumb_size, max_thumb_size), Image.Resampling.HAMMING)
                                                thumbnail_data = np.array(pil_image, dtype=np.uint8)
                                                logger.debug(f"Thumbnail resized to: {thumbnail_data.shape[1]}x{thumbnail_data.shape[0]}")
                                    
                                    # Apply orientation correction
                                    thumbnail_data = self.apply_orientation_correction(thumbnail_data, orientation)
                                    
                                    # Cache the thumbnail
                                    cache.put_thumbnail(self.file_path, thumbnail_data)
                                    logger.info(f"Thumbnail extracted and cached: {os.path.basename(self.file_path)} ({thumbnail_data.shape[1]}x{thumbnail_data.shape[0]})")
                                    
                                    # Emit thumbnail immediately for fast display (only if not loading full resolution only)
                                    if not self.use_full_resolution:
                                        logger.info(f"[RAW_PROC] Emitting thumbnail_fallback_used signal")
                                        self.thumbnail_fallback_used.emit("Loading thumbnail...")
                                        logger.info(f"[RAW_PROC] Emitting image_processed signal with thumbnail data: {thumbnail_data.shape}")
                                        self.image_processed.emit(thumbnail_data)
                                        logger.info(f"[RAW_PROC] Thumbnail signals emitted successfully")
                                    
                                    # Mark that we have thumbnail, skip RAW processing for thumbnail
                                    thumb = None  # No embedded thumb object needed
                                else:
                                    logger.debug("No embedded thumbnail found, will process RAW for thumbnail")
                                    thumb = None
                                    
                            except Exception as thumb_error:
                                # Handle thumbnail extraction errors gracefully
                                error_str = str(thumb_error)
                                error_type = type(thumb_error).__name__
                                is_cancellation = (
                                    self._should_stop or 
                                    'OutOfOrderCall' in error_type or 
                                    'LibRaw' in error_type or
                                    'Out of order' in error_str or
                                    'out of order' in error_str.lower()
                                )
                                if is_cancellation:
                                    logger.debug(f"Thumbnail extraction cancelled for {os.path.basename(self.file_path)}")
                                    return
                                # For other errors, log but continue (we can still try RAW processing)
                                logger.debug(f"Thumbnail extraction failed, will process RAW: {thumb_error}")
                                thumbnail_data = None
                                thumb = None
                            
                            # Only process RAW for thumbnail if embedded thumbnail is not available
                            if thumbnail_data is None and thumb is None:
                                # Emit progress signal for thumbnail generation
                                if not self.use_full_resolution:
                                    self.processing_progress.emit("Extracting preview...")
                                # Generate thumbnail from RAW processing (no_auto_bright)
                                # This ensures thumbnails match processed images exactly
                                logger.debug(f"Generating thumbnail from RAW processing (no embedded thumbnail): {os.path.basename(self.file_path)}")
                                
                                try:
                                    if self._should_stop:
                                        return
                                    
                                    # Check if handle is still valid
                                    with self._raw_handle_lock:
                                        if self._raw_handle is None or self._raw_handle != raw:
                                            logger.debug("RAW handle invalidated before thumbnail generation")
                                            return
                                    
                                    # OPTIMIZATION: Use smaller processing size for faster thumbnail generation
                                    # Process at quarter size (faster than half_size) and resize to 1024px
                                    # This is much faster than half_size for large files
                                    thumbnail_rgb = raw.postprocess(
                                        half_size=True,  # Use half_size (faster than full), will resize to 1024px below
                                        output_bps=8,    # 8-bit for speed
                                        no_auto_bright=True,  # Disable auto-brightness to preserve original RAW colors
                                        gamma=(2.222, 4.5),  # Standard sRGB gamma
                                        # Performance optimizations for speed
                                        use_camera_wb=True,  # Faster than auto WB
                                        demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR  # Fastest demosaicing
                                    )
                                    
                                    if self._should_stop:
                                        return
                                    
                                    if thumbnail_rgb is not None:
                                        # Resize to max 1024px for thumbnail
                                        from PIL import Image
                                        import numpy as np
                                        
                                        # Convert to PIL Image for resizing
                                        pil_image = Image.fromarray(thumbnail_rgb)
                                        max_thumb_size = 1024
                                        if pil_image.width > max_thumb_size or pil_image.height > max_thumb_size:
                                            logger.debug(f"Resizing processed thumbnail from {pil_image.size} to max {max_thumb_size}x{max_thumb_size}")
                                            pil_image.thumbnail((max_thumb_size, max_thumb_size), Image.Resampling.HAMMING)
                                        
                                        # Convert back to numpy array
                                        thumbnail_data = np.array(pil_image, dtype=np.uint8)
                                        logger.debug(f"Generated thumbnail from RAW processing: {thumbnail_data.shape[1]}x{thumbnail_data.shape[0]}")
                                        
                                        # Apply orientation correction
                                        thumbnail_data = self.apply_orientation_correction(
                                            thumbnail_data, orientation)
                                        
                                        # Cache the thumbnail
                                        cache.put_thumbnail(self.file_path, thumbnail_data)
                                        logger.info(f"Thumbnail generated from RAW (no auto-brightness): {os.path.basename(self.file_path)} ({thumbnail_data.shape[1]}x{thumbnail_data.shape[0]})")
                                        
                                        # Emit thumbnail immediately for fast display (only if not loading full resolution only)
                                        if not self.use_full_resolution:
                                            logger.info(f"[RAW_PROC] Emitting thumbnail_fallback_used signal")
                                            self.thumbnail_fallback_used.emit("Loading thumbnail...")
                                            logger.info(f"[RAW_PROC] Emitting image_processed signal with thumbnail data: {thumbnail_data.shape}")
                                            self.image_processed.emit(thumbnail_data)
                                            logger.info(f"[RAW_PROC] Thumbnail signals emitted successfully")
                                        
                                        # Continue to full processing below
                                        thumb = None  # Mark that we used processed thumbnail
                                    
                                except Exception as thumb_error:
                                    # If processing fails, fall back to embedded thumbnail
                                    error_str = str(thumb_error)
                                    error_type = type(thumb_error).__name__
                                    is_cancellation = (
                                        self._should_stop or 
                                        'OutOfOrderCall' in error_type or 
                                        'LibRaw' in error_type or
                                        'Out of order' in error_str or
                                        'out of order' in error_str.lower()
                                    )
                                    if is_cancellation:
                                        logger.debug(f"Thumbnail generation cancelled for {os.path.basename(self.file_path)}")
                                        return
                                    # For other errors, log but continue (embedded thumbnail may still be processed below)
                                    logger.debug(f"RAW thumbnail generation failed, will try embedded thumbnail: {thumb_error}")
                                    thumbnail_data = None
                                    thumb = None  # Ensure thumb is None so embedded thumbnail section can try
                            
                            # Note: Embedded thumbnail processing is now handled by ThumbnailExtractor above
                            # This section is kept for backward compatibility but should not be reached
                            # if ThumbnailExtractor successfully extracted the thumbnail
                        else:
                            # Thumbnail was already loaded from cache above
                            # OPTIMIZATION: Check if full image is already cached before processing
                            cached_full = cache.get_full_image(self.file_path)
                            if cached_full is not None:
                                logger.info(f"[RAW_PROC] Full image already cached, skipping processing: {os.path.basename(self.file_path)}")
                                # Don't emit full image yet - wait for user to zoom in or request it
                                return
                            
                            # Skip full processing if we're only loading full resolution (on-demand zoom)
                            # and it's not cached (we already checked above)
                            if not self.use_full_resolution:
                                logger.debug(f"Using cached thumbnail, will process full image in background (lazy loading)")

                        # Check if we should stop before full processing
                        if self._should_stop:
                            logger.debug(f"Processing stopped before full image processing for: {self.file_path}")
                            return
                        
                        # OPTIMIZATION: Only process full image if:
                        # 1. User explicitly requested full resolution (on-demand zoom), OR
                        # 2. Full image is not cached (need to generate it)
                        cached_full = cache.get_full_image(self.file_path)
                        if cached_full is not None and not self.use_full_resolution:
                            logger.info(f"[RAW_PROC] Full image already cached, skipping processing (lazy loading): {os.path.basename(self.file_path)}")
                            return
                        
                        # Now try full RAW processing in background
                        try:
                            import time
                            processing_start = time.time()
                            # Emit progress signal for full image processing
                            if self.use_full_resolution:
                                self.processing_progress.emit("Loading full resolution on-demand (user zoomed in)")
                            else:
                                self.processing_progress.emit("Processing RAW image...")
                            logger.debug(f"Starting full RAW image processing: {os.path.basename(self.file_path)}")
                            rgb_image = self.process_raw_with_camera_specific_settings(
                                raw)
                            # Apply orientation correction to processed RAW image
                            
                            # Check if processing was stopped or failed
                            if self._should_stop or rgb_image is None:
                                logger.debug(f"Processing stopped or cancelled during RAW processing for: {self.file_path}")
                                return
                            
                            processing_time = time.time() - processing_start
                            logger.info(f"RAW processing completed in {processing_time:.3f}s, shape: {rgb_image.shape}, dtype: {rgb_image.dtype}")
                            
                            if True:  # Always apply correction (user_flip=0 forced)
                                logger.info(f"[RAW_PROC] Applying orientation correction: {orientation}")
                                rgb_image = self.apply_orientation_correction(
                                    rgb_image, orientation)
                            
                            # Cache the full image
                            # Check if we should stop after orientation correction
                            if self._should_stop:
                                logger.debug(f"Processing stopped after orientation correction for: {self.file_path}")
                                return
                            
                            # Cache the full image (only if not stopping)
                            cache.put_full_image(self.file_path, rgb_image)
                            
                            # Mark if this is half_size or full resolution
                            is_half_size = self._use_fast_processing if hasattr(self, '_use_fast_processing') else False
                            logger.info(f"Full image processed and cached: {os.path.basename(self.file_path)} ({rgb_image.shape[1]}x{rgb_image.shape[0]}) {'[half_size]' if is_half_size else '[full_resolution]'}")
                            # Emit the full quality image (only once)
                            if not self._should_stop:
                                logger.info(f"[RAW_PROC] Emitting processing_progress signal: Processing complete")
                                self.processing_progress.emit("Processing complete")
                                logger.info(f"[RAW_PROC] Emitting image_processed signal with full image: {rgb_image.shape}")
                                self.image_processed.emit(rgb_image)
                                logger.info(f"[RAW_PROC] Full image signals emitted successfully for: {os.path.basename(self.file_path)}")
                            
                        except MemoryError as mem_error:
                            # Memory error - provide helpful message
                            error_msg = str(mem_error)
                            logger.error(f"Memory error processing RAW file {os.path.basename(self.file_path)}: {error_msg}", exc_info=True)
                            if not self._should_stop:
                                self.error_occurred.emit(
                                    f"Memory error processing RAW file:\n{error_msg}\n\n"
                                    f"Try:\n"
                                    f"- Closing other applications\n"
                                    f"- Processing smaller images\n"
                                    f"- Restarting the application"
                                )
                            return
                        except Exception as processing_error:
                            # If RAW processing fails, we already have thumbnail displayed
                            # Check if this is a cancellation error (LibRawOutOfOrderCallError)
                            # This happens when processing is stopped during RAW processing
                            error_type = type(processing_error).__name__
                            error_str = str(processing_error)
                            
                            # Check if this is a cancellation-related error
                            is_cancellation = (
                                self._should_stop or 
                                'OutOfOrderCall' in error_type or 
                                'LibRaw' in error_type or
                                'Out of order' in error_str or
                                'out of order' in error_str.lower()
                            )
                            
                            if is_cancellation:
                                # This is expected when processing is cancelled - just log as debug
                                logger.debug(f"RAW processing cancelled for {os.path.basename(self.file_path)}: {processing_error}")
                                return  # Normal cancellation, exit gracefully
                            
                            # If RAW processing fails for other reasons, we already have thumbnail displayed
                            logger.error(f"Full RAW processing failed for {os.path.basename(self.file_path)}: {processing_error}", exc_info=True)
                            if not self._should_stop:
                                # Only emit error if we don't have a thumbnail
                                if thumbnail_data is None:
                                    self.error_occurred.emit(
                                        f"Failed to process RAW file: {str(processing_error)}"
                                    )
                                else:
                                    # We have thumbnail, so just log the error
                                    logger.warning(f"Full RAW processing failed but thumbnail is available, continuing with thumbnail display")
                            
                    
                    finally:
                        # Ensure raw handle is closed (thread-safe)
                        # Only close if it hasn't been closed by cleanup logic
                        with self._raw_handle_lock:
                            if self._raw_handle is not None and self._raw_handle == raw:
                                # Handle is still valid and matches - safe to close
                                try:
                                    raw.close()
                                    logger.debug(f"RAW file handle closed: {os.path.basename(self.file_path)}")
                                except Exception as close_error:
                                    # Check if this is a cancellation error
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
                                        logger.warning(f"Error closing RAW file handle: {close_error}")
                                self._raw_handle = None
                            elif self._raw_handle is None:
                                # Handle was already closed by cleanup - just log
                                logger.debug(f"RAW file handle already closed by cleanup: {os.path.basename(self.file_path)}")
                            # If handle doesn't match, it was replaced - don't close
                except Exception as e:
                    # Handle file opening errors
                    logger.error(f"Error opening RAW file {os.path.basename(self.file_path)}: {e}", exc_info=True)
                    # Don't raise - just return gracefully to prevent crashes
                    if not self._should_stop:
                        self.error_occurred.emit(f"Failed to open RAW file: {str(e)}")
                    return
            else:
                # For non-RAW files (JPEG, PNG, WebP, etc.), emit None to let main thread handle with QPixmap
                filename = os.path.basename(self.file_path)
                logger.info(f"[RAW_PROC] ========== RAWProcessor.run() STARTED for {filename} (non-RAW file) ==========")
                logger.info(f"[RAW_PROC] Non-RAW file detected, emitting None signal for QPixmap handling")
                
                # Extract EXIF data for non-RAW files
                try:
                    self.processing_progress.emit("Reading metadata...")
                    from enhanced_raw_processor import EXIFExtractor
                    exif_extractor = EXIFExtractor()
                    exif_data = exif_extractor.extract_exif_data(self.file_path)
                    
                    # Emit EXIF data ready signal for immediate status bar update
                    if exif_data and not self._should_stop:
                        logger.info(f"[RAW_PROC] Emitting exif_data_ready signal for non-RAW file")
                        self.exif_data_ready.emit(exif_data)
                        logger.info(f"[RAW_PROC] exif_data_ready signal emitted")
                except Exception as exif_error:
                    logger.debug(f"Error extracting EXIF from non-RAW file: {exif_error}")
                
                # Emit None signal to indicate this is a non-RAW file (main thread will use QPixmap)
                if not self._should_stop:
                    logger.info(f"[RAW_PROC] Emitting image_processed signal with None for non-RAW file")
                    self.image_processed.emit(None)
                    logger.info(f"[RAW_PROC] Signal emitted successfully for non-RAW file")
        except Exception as e:
            # Provide more specific error messages
            error_msg = str(e)
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Unhandled exception in RAWProcessor for {os.path.basename(self.file_path) if hasattr(self, 'file_path') else 'unknown'}: {e}", exc_info=True)
            
            if "data corrupted" in error_msg.lower():
                error_msg = f"RAW processing failed due to LibRaw compatibility issue.\n\nThis is a known issue with LibRaw 0.21.3 and certain NEF files.\nTry using a different RAW processor or contact the developer for updates.\n\nOriginal error: {error_msg}"
            elif "unsupported file format" in error_msg.lower():
                error_msg = f"This RAW file format may not be supported by your LibRaw version.\n\nOriginal error: {error_msg}"
            elif "input/output error" in error_msg.lower():
                error_msg = f"Cannot read the file. It may be corrupted or in use by another program.\n\nOriginal error: {error_msg}"
            elif "cannot allocate memory" in error_msg.lower():
                error_msg = f"Not enough memory to process this large RAW file.\n\nOriginal error: {error_msg}"

            self.error_occurred.emit(error_msg)


class PixmapConverter(QThread):
    """Background thread to convert numpy array to QPixmap and cache it"""
    pixmap_ready = pyqtSignal(str, QPixmap)  # file_path, pixmap
    
    def __init__(self, file_path, rgb_image, image_cache):
        super().__init__()
        self.file_path = file_path
        # Not copied here: this class is a QThread, but __init__ runs
        # synchronously on whatever thread constructs it (always the main/
        # GUI thread for this class -- see on_manager_image_ready). A
        # full-resolution RAW buffer can be 100+MB; copying it here blocked
        # the main event loop for the copy's full duration on every image
        # display, confirmed via a live SIGUSR1/faulthandler stack dump
        # during a multi-second UI stall (main thread parked inside this
        # __init__). The copy still happens -- just moved to the start of
        # run(), on the background thread, which is what QThread is for.
        self.rgb_image = rgb_image
        self.image_cache = image_cache
        self._should_stop = False

    def stop_processing(self):
        """Request processing to stop"""
        self._should_stop = True

    def run(self):
        """Convert numpy array to QPixmap in background"""
        try:
            if self._should_stop:
                return

            # Copy here (background thread), not in __init__ (main thread).
            # See __init__ docstring comment for why this moved.
            if hasattr(self.rgb_image, 'copy'):
                self.rgb_image = self.rgb_image.copy()

            if not hasattr(self.rgb_image, 'shape'):
                if hasattr(self.rgb_image, 'width') and hasattr(self.rgb_image, 'height'):
                    height, width = self.rgb_image.height(), self.rgb_image.width()
                    channels = 3
                else:
                    return
            else:
                shape = self.rgb_image.shape
                height, width = shape[0], shape[1]
                channels = shape[2] if len(shape) > 2 else 1
                if self.rgb_image.dtype == np.uint16:
                    self.rgb_image = (self.rgb_image / 257.0).astype(np.uint8)
                elif np.issubdtype(self.rgb_image.dtype, np.floating):
                    from raw_tone_recovery import _encode_srgb8
                    self.rgb_image = _encode_srgb8(
                        np.clip(self.rgb_image.astype(np.float32), 0.0, None)
                    )
                elif self.rgb_image.dtype != np.uint8:
                    self.rgb_image = np.clip(self.rgb_image, 0, 255).astype(np.uint8)

            bytes_per_line = channels * width
            
            # Ensure the data is contiguous
            if not self.rgb_image.flags['C_CONTIGUOUS']:
                self.rgb_image = np.ascontiguousarray(self.rgb_image)
            
            if self._should_stop:
                return
            
            # Create QImage directly from numpy array without copying memory
            q_format = QImage.Format.Format_RGB888
            if channels == 1:
                q_format = QImage.Format.Format_Grayscale8
            elif channels == 4:
                q_format = QImage.Format.Format_RGBA8888

            q_image = QImage(self.rgb_image.data, width, height,
                             bytes_per_line, q_format)
            q_image.ndarr = self.rgb_image  # Keep numpy array alive
            pixmap = QPixmap.fromImage(q_image)
            
            if self._should_stop:
                return
            
            # Cache the pixmap
            if self.image_cache:
                self.image_cache.put_pixmap(self.file_path, pixmap)
            
            # Emit signal if not stopped
            if not self._should_stop:
                self.pixmap_ready.emit(self.file_path, pixmap)
                
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Error in PixmapConverter for {self.file_path}: {e}")


class RawEdrPixmapConverter(QThread):
    """Background thread: linear RAW decode → macOS EDR RGBX64 QPixmap."""

    pixmap_ready = pyqtSignal(str, QPixmap)

    def __init__(
        self,
        file_path: str,
        *,
        max_edge: int = 0,
        exif_data=None,
    ):
        super().__init__()
        self.file_path = file_path
        self.max_edge = int(max_edge or 0)
        self.exif_data = exif_data
        self._should_stop = False

    def stop_processing(self):
        self._should_stop = True

    def run(self):
        try:
            if self._should_stop:
                return
            from common_image_loader import decode_raw_edr_pixmap

            pixmap = decode_raw_edr_pixmap(
                self.file_path,
                max_edge=self.max_edge,
                exif_data=self.exif_data,
            )
            if self._should_stop:
                return
            if pixmap is None or pixmap.isNull():
                self.pixmap_ready.emit(self.file_path, QPixmap())
                return
            self.pixmap_ready.emit(self.file_path, pixmap)
        except Exception as e:
            import logging

            logging.getLogger(__name__).debug(
                "Error in RawEdrPixmapConverter for %s: %s",
                self.file_path,
                e,
            )
            if not self._should_stop:
                self.pixmap_ready.emit(self.file_path, QPixmap())

