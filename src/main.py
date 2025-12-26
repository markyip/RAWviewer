import sys
import os

# Force unbuffered output for Windows console
# Note: In PyInstaller --windowed builds, sys.stdout/stderr may be None
if sys.platform == 'win32':
    if sys.stdout is not None:
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, OSError):
            pass  # stdout may not support reconfigure in some environments
    if sys.stderr is not None:
        try:
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, OSError):
            pass  # stderr may not support reconfigure in some environments

# Safe print function for PyInstaller --windowed builds
# In windowed mode, sys.stdout/stderr may be None
def safe_print(*args, **kwargs):
    """Safely print to stdout, handling None case in windowed builds"""
    if sys.stdout is not None:
        try:
            print(*args, **kwargs)
        except (OSError, AttributeError):
            pass  # stdout may not be available

def safe_print_err(*args, **kwargs):
    """Safely print to stderr, handling None case in windowed builds"""
    if sys.stderr is not None:
        try:
            print(*args, file=sys.stderr, **kwargs)
        except (OSError, AttributeError):
            pass  # stderr may not be available

# Print immediately to verify script is running
safe_print("=" * 80, flush=True)
safe_print("RAWviewer: Starting imports...", flush=True)
safe_print(f"Python: {sys.version}", flush=True)
safe_print(f"Working directory: {os.getcwd()}", flush=True)
safe_print("=" * 80, flush=True)

import logging
import traceback
import threading
import warnings

# Suppress exifread warnings for unsupported file formats (e.g., video files)
# This prevents spam in logs when scanning folders with mixed content
warnings.filterwarnings('ignore', category=UserWarning, module='exifread')

print("Basic imports done, importing PyQt6...", flush=True)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QFileDialog,
                             QMessageBox, QScrollArea, QSizePolicy, QPushButton, QFrame,
                             QGridLayout, QScrollBar, QDialog, QSplashScreen)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPoint, QEvent, QSettings, QSize, QRect, QObject, QRunnable, QThreadPool
from PyQt6.QtGui import (QPixmap, QImage, QAction, QKeySequence,
                         QDragEnterEvent, QDropEvent, QCursor, QIcon,
                         QTransform, QRegion, QPainterPath, QPainter, QColor, QPen)
print("PyQt6 imported successfully", flush=True)

print("Importing third-party libraries...", flush=True)
try:
    import rawpy
    print("  - rawpy: OK", flush=True)
except Exception as e:
    print(f"  - rawpy: ERROR - {e}", file=sys.stderr, flush=True)
    raise

try:
    import numpy as np
    print("  - numpy: OK", flush=True)
except Exception as e:
    print(f"  - numpy: ERROR - {e}", file=sys.stderr, flush=True)
    raise

# natsort and send2trash will be imported lazily when needed
# These modules can cause import delays or hangs, so we import them on-demand
print("  - natsort: Will be imported on demand", flush=True)
print("  - send2trash: Will be imported on demand", flush=True)

try:
    import exifread
    print("  - exifread: OK", flush=True)
except Exception as e:
    print(f"  - exifread: ERROR - {e}", file=sys.stderr, flush=True)
    raise

from datetime import datetime
import platform
import ctypes
import time

print("Third-party imports done, importing local modules...", flush=True)

# Import enhanced performance modules
try:
    from image_cache import get_image_cache, initialize_cache
    print("  - image_cache: OK", flush=True)
except Exception as e:
    print(f"  - image_cache: ERROR - {e}", file=sys.stderr, flush=True)
    import traceback
    print(traceback.format_exc(), file=sys.stderr, flush=True)
    raise

try:
    from enhanced_raw_processor import EnhancedRAWProcessor, PreloadManager, ThumbnailExtractor
    print("  - enhanced_raw_processor: OK", flush=True)
except Exception as e:
    print(f"  - enhanced_raw_processor: ERROR - {e}", file=sys.stderr, flush=True)
    import traceback
    print(traceback.format_exc(), file=sys.stderr, flush=True)
    raise

print("Enhanced modules imported, importing new unified architecture...", flush=True)

# Import new unified architecture
try:
    from image_load_manager import get_image_load_manager, Priority
    print("  - image_load_manager: OK", flush=True)
except Exception as e:
    print(f"  - image_load_manager: ERROR - {e}", file=sys.stderr, flush=True)
    import traceback
    print(traceback.format_exc(), file=sys.stderr, flush=True)
    raise

try:
    from common_image_loader import check_cache_for_image, is_raw_file, load_pixmap_safe
    print("  - common_image_loader: OK", flush=True)
except Exception as e:
    print(f"  - common_image_loader: ERROR - {e}", file=sys.stderr, flush=True)
    import traceback
    print(traceback.format_exc(), file=sys.stderr, flush=True)
    raise

print("All imports completed successfully!", flush=True)


def setup_logging():
    """Setup logging configuration with file and console handlers"""
    try:
        # Create logs directory if it doesn't exist
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Verify directory was created
        if not os.path.exists(log_dir):
            raise OSError(f"Failed to create log directory: {log_dir}")
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'rawviewer_{timestamp}.log')
        
        # Configure logging
        log_format = '%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        
        # Create formatters
        formatter = logging.Formatter(log_format, date_format)
        
        # File handler - log everything
        try:
            file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
        except Exception as e:
            raise OSError(f"Failed to create log file handler for {log_file}: {e}")
        
        # Console handler - log INFO and above
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers to avoid duplicates
        root_logger.handlers.clear()
        
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Verify log file was created and is writable
        if not os.path.exists(log_file):
            raise OSError(f"Log file was not created: {log_file}")
        
        # Test write to log file
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write("")
        except Exception as e:
            raise OSError(f"Log file is not writable: {log_file}, error: {e}")
        
        # Log the log file location
        logger = logging.getLogger(__name__)
        logger.info(f"Log file created successfully: {log_file}")
        logger.info(f"Log directory: {log_dir}")
        
        return log_file
        
    except Exception as e:
        # If logging setup fails, at least print to stderr
        error_msg = f"CRITICAL: Failed to setup logging: {e}"
        print(error_msg, file=sys.stderr)
        import traceback
        print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
        
        # Try to write to a fallback log file
        try:
            fallback_log = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'error.log')
            os.makedirs(os.path.dirname(fallback_log), exist_ok=True)
            with open(fallback_log, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")
                f.write(f"{traceback.format_exc()}\n")
        except:
            pass  # If even fallback fails, we've done our best
        
        raise  # Re-raise to let caller handle it


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # The script is in src/, so we go one level up to the project root
        base_path = os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".."))
    return os.path.join(base_path, relative_path)


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
            except:
                pass

    def get_orientation_from_exif(self, file_path):
        """Extract orientation from EXIF data - optimized for minimal logging"""
        try:
            with open(file_path, 'rb') as f:
                # Only extract essential tags to reduce processing time and logging
                tags = exifread.process_file(f, details=False, stop_tag='Image Orientation')

                # Check for orientation tag
                orientation_tag = tags.get('Image Orientation')
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
        """Apply orientation correction to numpy array"""
        # Check if this is a camera that stores RAW data pre-rotated
        # Some cameras (like Sony) store RAW data in the correct orientation
        # and the EXIF orientation tag may be misleading
        if self.is_raw_data_pre_rotated():
            return image_array

        if orientation == 1:
            # Normal orientation, no changes needed
            return image_array
        elif orientation == 2:
            # Mirrored horizontal
            return np.fliplr(image_array)
        elif orientation == 3:
            # Rotated 180 degrees
            return np.rot90(image_array, 2)
        elif orientation == 4:
            # Mirrored vertical
            return np.flipud(image_array)
        elif orientation == 5:
            # Mirrored horizontal then rotated 90 CCW
            return np.rot90(np.fliplr(image_array), 1)
        elif orientation == 6:
            # Rotated 90 CW - need to rotate 90 CW to correct (same as QPixmap transform.rotate(90))
            # np.rot90 with k=3 rotates 270° CCW = 90° CW
            return np.rot90(image_array, 3)
        elif orientation == 7:
            # Mirrored horizontal then rotated 90 CW
            return np.rot90(np.fliplr(image_array), -1)
        elif orientation == 8:
            # Rotated 90 CCW - need to rotate 90 CW (270 CCW) to correct
            # np.rot90 with k=3 rotates 270° CCW = 90° CW
            return np.rot90(image_array, 3)
        else:
            return image_array

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

            # Fallback to EXIF detection for other formats
            # Fallback to EXIF detection for other formats - only read Image Make tag
            with open(self.file_path, 'rb') as f:
                tags = exifread.process_file(f, details=False, stop_tag='Image Make')
                make = tags.get('Image Make')

                if make:
                    make_str = str(make).upper()
                    # Canon cameras need special white balance processing
                    if 'CANON' in make_str:
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

            # Fallback to EXIF detection for other formats
            # Fallback to EXIF detection for other formats - only read Image Make tag
            with open(self.file_path, 'rb') as f:
                tags = exifread.process_file(f, details=False, stop_tag='Image Make')
                make = tags.get('Image Make')

                if make:
                    make_str = str(make).upper()
                    # Fujifilm cameras need special white balance processing
                    if 'FUJIFILM' in make_str or 'FUJI' in make_str:
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
                        'gamma': (2.222, 4.5)  # Standard sRGB gamma
                    }
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
                            'gamma': (2.222, 4.5)  # Standard sRGB gamma
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
                            'gamma': (2.222, 4.5)  # Standard sRGB gamma
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
            # Check if this is a Fujifilm camera
            elif self.is_fujifilm_camera():
                # Fujifilm cameras (especially RAF) need proper white balance correction
                # to avoid green hue issues and improve processing speed
                if use_fast_processing:
                    logger.debug(f"Applying Fujifilm-specific processing with fast mode for large file ({file_size_mb:.1f}MB)...")
                else:
                    logger.debug(f"Applying Fujifilm-specific white balance correction...")
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
                        'gamma': (2.222, 4.5)  # Standard sRGB gamma
                    }
                    # Add performance optimizations if available
                    try:
                        # Use fastest demosaicing algorithm for speed (if supported)
                        postprocess_params['demosaic_algorithm'] = rawpy.DemosaicAlgorithm.LINEAR
                    except (AttributeError, TypeError):
                        pass  # Parameter not available in this rawpy version
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
                            'gamma': (2.222, 4.5)  # Standard sRGB gamma
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
                            'gamma': (2.222, 4.5)  # Standard sRGB gamma
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
                    'gamma': (2.222, 4.5)  # Standard sRGB gamma
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
            # Fallback with optimized parameters for speed
            # Use auto-brightness by default (True) to match initial display brightness
            use_auto_bright_fallback = True
            postprocess_params = {
                'output_bps': 8,  # Use 8-bit for faster processing
                'no_auto_bright': not use_auto_bright_fallback,  # Use auto-brightness for full resolution
                'gamma': (2.222, 4.5)  # Standard sRGB gamma
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
                    logger.debug(f"Thumbnail not in cache, will generate: {os.path.basename(self.file_path)}")
                
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
                
                # Always open RAW file (needed for full image processing, or if thumbnail not cached)
                # Even if thumbnail is cached, we still need to process full image in background
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
                        elif thumbnail_data is None:
                            # OPTIMIZATION: Try to extract thumbnail using already-opened raw handle first
                            # This avoids reopening the file, which is much faster
                            logger.debug(f"Extracting thumbnail: {os.path.basename(self.file_path)}")
                            try:
                                if self._should_stop:
                                    return
                                
                                # OPTIMIZATION: Use already-opened raw handle if available (faster)
                                extracted_thumbnail = None
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
                                                print(f"[PERF] ⚡ FAST THUMBNAIL: Extracted using existing raw handle ({thumb_size})")
                                        except Exception as thumb_extract_error:
                                            logger.debug(f"Failed to extract thumbnail from raw handle: {thumb_extract_error}")
                                            print(f"[PERF] ⚠️  Raw handle extraction failed, falling back")
                                
                                # Fallback: Use ThumbnailExtractor if raw handle extraction failed
                                if extracted_thumbnail is None:
                                    logger.debug(f"Falling back to ThumbnailExtractor for thumbnail extraction")
                                    fallback_start = time.time()
                                    extracted_thumbnail = self.thumbnail_extractor.extract_thumbnail_from_raw(self.file_path)
                                    fallback_time = time.time() - fallback_start
                                    if extracted_thumbnail is not None:
                                        print(f"[PERF] 🔄 FALLBACK THUMBNAIL: Extracted via ThumbnailExtractor in {fallback_time*1000:.1f}ms")
                                
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
                                                pil_image.thumbnail((max_thumb_size, max_thumb_size), Image.Resampling.LANCZOS)
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
                                # Generate thumbnail from RAW processing with auto-brightness
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
                                            pil_image.thumbnail((max_thumb_size, max_thumb_size), Image.Resampling.LANCZOS)
                                        
                                        # Convert back to numpy array
                                        thumbnail_data = np.array(pil_image, dtype=np.uint8)
                                        logger.debug(f"Generated thumbnail from RAW processing: {thumbnail_data.shape[1]}x{thumbnail_data.shape[0]}")
                                        
                                        # Apply orientation correction
                                        thumbnail_data = self.apply_orientation_correction(
                                            thumbnail_data, orientation)
                                        
                                        # Cache the thumbnail
                                        cache.put_thumbnail(self.file_path, thumbnail_data)
                                        logger.info(f"Thumbnail generated from RAW with auto-brightness: {os.path.basename(self.file_path)} ({thumbnail_data.shape[1]}x{thumbnail_data.shape[0]})")
                                        
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
                            # Thumbnail was already loaded from cache above, but we still need to process full image
                            # Skip this if we're only loading full resolution (on-demand zoom)
                            if not self.use_full_resolution:
                                logger.debug(f"Using cached thumbnail, proceeding to full image processing")

                        # Check if we should stop before full processing
                        if self._should_stop:
                            logger.debug(f"Processing stopped before full image processing for: {self.file_path}")
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
        self.rgb_image = rgb_image.copy()  # Make a copy to avoid issues
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
            
            height, width, channels = self.rgb_image.shape
            bytes_per_line = channels * width
            
            # Ensure the data is contiguous
            if not self.rgb_image.flags['C_CONTIGUOUS']:
                self.rgb_image = np.ascontiguousarray(self.rgb_image)
            
            if self._should_stop:
                return
            
            # Convert to bytes for PyQt6 compatibility
            image_data = self.rgb_image.data.tobytes() if hasattr(
                self.rgb_image.data, 'tobytes') else bytes(self.rgb_image.data)
            
            if self._should_stop:
                return
            
            # Create QImage and QPixmap
            q_image = QImage(image_data, width, height,
                             bytes_per_line, QImage.Format.Format_RGB888)
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


class ThumbnailLabel(QLabel):
    """
    Thumbnail widget - keeps original pixmap and rescales cleanly.
    Based on reference implementation: simple and reliable.
    """
    def __init__(self, pixmap=None):
        super().__init__()
        self.original_pixmap = pixmap
        if pixmap:
            self.setPixmap(pixmap)
        else:
            self.setText("Loading…")
        # Use setScaledContents(False) - like reference code for JustifiedGallery
        self.setScaledContents(False)
        # Use Fixed size policy - prevents layout from resizing
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
    def set_original_pixmap(self, pixmap):
        """Store the original pixmap for rescaling"""
        self.original_pixmap = pixmap
    
    def get_original_pixmap(self):
        """Get the original pixmap"""
        return self.original_pixmap


# -----------------------------
# Signal carrier (thread → UI)
# -----------------------------
class ImageLoaded(QObject):
    """Signal carrier for image loading - thread to UI communication"""
    loaded = pyqtSignal(int, object)  # index, QImage (convert to QPixmap in UI thread)


# -----------------------------
# Worker to load images in background
# -----------------------------
class ImageLoadTask(QRunnable):
    """Background task to load and scale images"""
    def __init__(self, index, file_path, target_width, target_height, signal, parent_viewer=None):
        super().__init__()
        self.index = index
        self.file_path = file_path
        self.target_width = target_width
        self.target_height = target_height
        self.signal = signal
        self.parent_viewer = parent_viewer
    
    def run(self):
        """Load and scale image in worker thread - returns QImage, not QPixmap"""
        import os
        import logging
        logger = logging.getLogger(__name__)
        
        # Check file extension - use PIL directly for TIFF files to avoid Qt warnings
        file_ext = os.path.splitext(self.file_path)[1].lower()
        is_tiff = file_ext in ('.tiff', '.tif')
        
        # RAW file extensions - these use TIFF structure internally and may trigger TIFF warnings
        raw_extensions = ['.arw', '.cr2', '.nef', '.raf', '.orf', '.dng', '.cr3', '.rw2', '.rwl', '.srw', 
                         '.pef', '.x3f', '.3fr', '.fff', '.iiq', '.cap', '.erf', '.mef', '.mos', '.nrw', '.srf']
        is_raw = file_ext in raw_extensions
        
        # Also check file content to detect TIFF files with wrong extension
        # This prevents QImageReader from triggering warnings for TIFF files
        if not is_tiff and not is_raw:
            try:
                from PIL import Image
                with Image.open(self.file_path) as test_img:
                    if test_img.format in ('TIFF', 'TIF'):
                        is_tiff = True
                        logger.debug(f"[IMAGE_LOAD_TASK] Detected TIFF file by content (not extension): {os.path.basename(self.file_path)}")
            except:
                pass  # Not a PIL-readable file or not TIFF, continue with normal loading
        
        # For TIFF files or RAW files (which use TIFF structure), use PIL directly to avoid Qt TIFF plugin warnings
        # For RAW files in gallery view, use embedded JPEG preview for fast loading
        if is_tiff or is_raw:
            if is_raw:
                # For RAW files in gallery, try to extract embedded JPEG preview (fast)
                if self.parent_viewer and hasattr(self.parent_viewer, '_extract_embedded_preview'):
                    try:
                        pixmap = self.parent_viewer._extract_embedded_preview(self.file_path)
                        if pixmap and not pixmap.isNull():
                            # Convert QPixmap to QImage for consistency with other load paths
                            qimage = pixmap.toImage()
                            # Scale to target size if needed
                            if qimage.width() != self.target_width or qimage.height() != self.target_height:
                                from PyQt6.QtCore import Qt
                                scaled = pixmap.scaled(
                                    self.target_width,
                                    self.target_height,
                                    Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation
                                )
                                qimage = scaled.toImage()
                            logger.debug(f"[IMAGE_LOAD_TASK] Extracted embedded JPEG preview for RAW: {os.path.basename(self.file_path)}")
                            self.signal.loaded.emit(self.index, qimage)
                            return
                    except Exception as e:
                        logger.debug(f"[IMAGE_LOAD_TASK] Failed to extract embedded preview for RAW: {os.path.basename(self.file_path)}: {e}")
                
                # If embedded preview extraction failed, return without loading
                # RAW files should be handled by RAWProcessor for full processing
                logger.debug(f"[IMAGE_LOAD_TASK] RAW file detected, skipping QImageReader to avoid TIFF warnings: {os.path.basename(self.file_path)}")
                return
            try:
                self._try_fallback_loading(logger)
                return
            except Exception as e:
                logger.debug(f"[IMAGE_LOAD_TASK] PIL fallback failed for TIFF: {os.path.basename(self.file_path)}: {e}")
                # For TIFF files, never use QImageReader as it triggers warnings
                # Return without emitting signal (image will remain as placeholder)
                return
        
        try:
            from PyQt6.QtGui import QImageReader, QImage
            from PyQt6.QtCore import QSize
            
            # Double-check: never use QImageReader for TIFF files or RAW files (even if previous check failed)
            # This check must be BEFORE creating QImageReader, as QImageReader creation triggers warnings for TIFF
            if is_tiff or is_raw:
                logger.debug(f"[IMAGE_LOAD_TASK] Skipping QImageReader for {'TIFF' if is_tiff else 'RAW'}: {os.path.basename(self.file_path)}")
                # Try PIL fallback instead (only for TIFF, not RAW)
                if is_tiff:
                    try:
                        self._try_fallback_loading(logger)
                    except:
                        pass
                return
            
            # Try QImageReader first (fastest for most formats)
            # NOTE: QImageReader creation may trigger warnings for problematic TIFF files or RAW files
            # So we must check is_tiff and is_raw BEFORE this line
            reader = QImageReader(self.file_path)
            
            # Check if file can be read
            if not reader.canRead():
                logger.debug(f"[IMAGE_LOAD_TASK] Cannot read: {os.path.basename(self.file_path)}")
                # Try fallback method
                self._try_fallback_loading(logger)
                return
            
            # Calculate scaled size maintaining aspect ratio
            original_size = reader.size()
            if not original_size.isValid():
                # Try to get size from error message or use fallback
                logger.debug(f"[IMAGE_LOAD_TASK] Invalid size for: {os.path.basename(self.file_path)}, trying fallback")
                self._try_fallback_loading(logger)
                return
            
            aspect = original_size.width() / original_size.height() if original_size.height() > 0 else 1.0
            scaled_width = int(self.target_height * aspect)
            scaled_height = self.target_height
            
            # Ensure we don't exceed target width
            if scaled_width > self.target_width:
                scaled_width = self.target_width
                scaled_height = int(self.target_width / aspect) if aspect > 0 else self.target_height
            
            # Set scaled size - this makes QImageReader decode at target size directly
            reader.setScaledSize(QSize(scaled_width, scaled_height))
            reader.setAutoTransform(True)  # Handle EXIF orientation
            
            # Read the already-scaled image (very cheap, no full decode)
            scaled_image = reader.read()
            
            if scaled_image.isNull():
                # Check for error message
                error_string = reader.errorString()
                if error_string:
                    logger.debug(f"[IMAGE_LOAD_TASK] QImageReader error for {os.path.basename(self.file_path)}: {error_string}, trying fallback")
                else:
                    logger.debug(f"[IMAGE_LOAD_TASK] Failed to load: {os.path.basename(self.file_path)}, trying fallback")
                # Try fallback method
                self._try_fallback_loading(logger)
                return
            
            # Emit QImage to UI thread (will convert to QPixmap there)
            self.signal.loaded.emit(self.index, scaled_image)
            
        except Exception as e:
            logger.debug(f"[IMAGE_LOAD_TASK] Exception loading with QImageReader: {e}, trying fallback")
            # Try fallback method
            try:
                self._try_fallback_loading(logger)
            except Exception as fallback_error:
                logger.error(f"[IMAGE_LOAD_TASK] Both QImageReader and fallback failed for {os.path.basename(self.file_path) if self.file_path else 'unknown'}: {fallback_error}", exc_info=True)
    
    def _try_fallback_loading(self, logger):
        """Fallback method using PIL/Pillow for problematic images"""
        import os
        try:
            from PIL import Image
            from PyQt6.QtGui import QImage
            from PyQt6.QtCore import QSize
            
            # Open image with PIL (handles problematic TIFF files better)
            with Image.open(self.file_path) as pil_image:
                # Convert to RGB if necessary (handles RGBA, P mode, etc.)
                if pil_image.mode not in ('RGB', 'L'):
                    pil_image = pil_image.convert('RGB')
                
                # Calculate aspect ratio and scaled size
                original_width, original_height = pil_image.size
                aspect = original_width / original_height if original_height > 0 else 1.0
                scaled_width = int(self.target_height * aspect)
                scaled_height = self.target_height
                
                # Ensure we don't exceed target width
                if scaled_width > self.target_width:
                    scaled_width = self.target_width
                    scaled_height = int(self.target_width / aspect) if aspect > 0 else self.target_height
                
                # Resize image
                resized_pil = pil_image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
                
                # Convert to QImage
                # PIL image to QImage: convert to bytes
                if resized_pil.mode == 'RGB':
                    qimage = QImage(resized_pil.tobytes('raw', 'RGB'), scaled_width, scaled_height, QImage.Format.Format_RGB888)
                elif resized_pil.mode == 'L':
                    qimage = QImage(resized_pil.tobytes('raw', 'L'), scaled_width, scaled_height, QImage.Format.Format_Grayscale8)
                else:
                    # Convert to RGB as fallback
                    rgb_pil = resized_pil.convert('RGB')
                    qimage = QImage(rgb_pil.tobytes('raw', 'RGB'), scaled_width, scaled_height, QImage.Format.Format_RGB888)
                
                if qimage.isNull():
                    logger.debug(f"[IMAGE_LOAD_TASK] Fallback method failed to create QImage for: {os.path.basename(self.file_path)}")
                    return
                
                # Emit QImage to UI thread
                self.signal.loaded.emit(self.index, qimage)
                
        except ImportError:
            logger.debug(f"[IMAGE_LOAD_TASK] PIL not available for fallback loading: {os.path.basename(self.file_path)}")
        except Exception as e:
            logger.debug(f"[IMAGE_LOAD_TASK] Fallback loading failed for {os.path.basename(self.file_path) if self.file_path else 'unknown'}: {e}")


# ============================================================================
# GALLERY FUNCTIONALITY - COMMENTED OUT FOR STABLE RELEASE
# ============================================================================
# Gallery view functionality has been temporarily disabled to focus on
# stable single image viewing. Uncomment when ready to resume gallery development.
# ============================================================================

if False:  # Gallery functionality disabled
    class JustifiedGallery(QWidget):
        """
        Adaptive justified gallery layout - based on reference code.
        Uses adaptive row height with ±25% tolerance for better space utilization.
        Stores original pixmaps and rebuilds layout on resize.
        Implements lazy loading for faster initial display.
        """
    TARGET_ROW_HEIGHT = 200
    HEIGHT_TOLERANCE = 0.25  # allow ±25% adjustment
    MIN_SPACING = 4
    
    def __init__(self, images, parent=None):
        super().__init__(parent)
        import logging
        self.logger = logging.getLogger(__name__)
        self.parent_viewer = parent  # Reference to RAWImageViewer for loading images
        
        # Store original pixmaps (never scale twice)
        self.images = images  # List of file paths or pixmaps
        
        # Recursion protection
        self._building = False
        self._resize_in_progress = False
        self._last_viewport_width = None  # Track viewport width for resize detection
        
        # Thread pool for background image loading
        from PyQt6.QtCore import QThreadPool
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(4)  # Limit concurrent threads
        
        # Signal for image loading
        self.loader_signal = ImageLoaded()
        self.loader_signal.loaded.connect(self.apply_thumbnail)
        
        # Store tiles (labels) for lazy loading
        # List of (label, file_path, target_width, target_height) tuples
        self.tiles = []
        
        # Track rendering progress
        self._render_start_time = None
        self._images_loaded_count = 0
        self._total_images_to_load = 0
        
        # Debounce for load_visible_images
        self._load_timer = None
        self._resize_timer = None
        self._loading_tiles = set()  # Track tiles currently being loaded (by file_path)
        
        # Rate limiting for image loading
        self._load_queue = []  # Queue of (index, file_path, target_width, target_height) tuples
        self._load_rate_timer = None
        self._loads_per_second = 8  # Limit to 8 images per second
        self._batch_size = 2  # Load 2 images per batch
        
        # Cache scaled thumbnails per row-height bucket
        self._thumbnail_cache = {}  # {(file_path, row_height): QPixmap}
        self._row_height_buckets = [160, 200, 240]  # Cache at these heights
        
        # Vertical container = rows of images
        self.container = QVBoxLayout(self)
        self.container.setSpacing(self.MIN_SPACING)
        self.container.setContentsMargins(8, 8, 8, 8)
        
        # Don't build immediately - wait for widget to have proper size
        # Use QTimer to delay initial build
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(100, self._delayed_build)
    
    def _delayed_build(self):
        """Delayed initial build to ensure widget has proper size"""
        if self.width() > 0:
            self.build_gallery()
        else:
            # If still no width, try again
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(100, self._delayed_build)
    
    def build_gallery(self):
        """Build adaptive justified rows with lazy loading"""
        import logging
        import time
        import os
        logger = logging.getLogger(__name__)
        
        # Prevent recursive calls
        if self._building:
            logger.warning("[JUSTIFIED_GALLERY] build_gallery() called while already building, skipping")
            return
        
        self._building = True
        start_time = time.time()
        logger.info(f"[JUSTIFIED_GALLERY] ========== build_gallery() STARTED ==========")
        logger.info(f"[JUSTIFIED_GALLERY] Widget size: {self.width()}x{self.height()}")
        logger.info(f"[JUSTIFIED_GALLERY] Image count: {len(self.images)}")
        
        try:
            # Clear tiles list before rebuilding (old labels may have been deleted)
            self.tiles = []
            # Clear loading tracking to avoid stale references
            self._loading_tiles.clear()
            self._load_queue = []
            
            # Get viewport width from parent scroll area if available
            viewport_width = max(300, self.width() - 16)
            from PyQt6.QtWidgets import QScrollArea
            if self.parent():
                # Try to get width from scroll area viewport
                scroll_area = None
                parent = self.parent()
                while parent:
                    if isinstance(parent, QScrollArea):
                        scroll_area = parent
                        break
                    parent = parent.parent()
                
                if scroll_area:
                    viewport_width = max(300, scroll_area.viewport().width() - 16)
            
            logger.info(f"[JUSTIFIED_GALLERY] Viewport width: {viewport_width}")
            
            if viewport_width <= 0:
                logger.warning(f"[JUSTIFIED_GALLERY] Invalid viewport width: {viewport_width}, skipping build")
                self._building = False
                return
            
            row = []
            aspect_sum = 0
            images_processed = 0
            
            min_h = self.TARGET_ROW_HEIGHT * (1 - self.HEIGHT_TOLERANCE)
            max_h = self.TARGET_ROW_HEIGHT * (1 + self.HEIGHT_TOLERANCE)
            
            # Build rows using aspect ratios (fast, no image loading)
            for idx, item in enumerate(self.images):
                images_processed += 1
                
                # Get aspect ratio without loading full image
                if isinstance(item, str):
                    # File path - get aspect ratio from cache or EXIF
                    aspect = self._get_aspect_ratio_for_path(item)
                    if aspect <= 0:
                        continue
                else:
                    # Already a QPixmap
                    if item.isNull():
                        continue
                    aspect = item.width() / item.height() if item.height() > 0 else 1.0
                
                row.append((item, aspect))
                aspect_sum += aspect
                
                # Compute required row height to fill width
                row_height = (viewport_width - (len(row) - 1) * self.MIN_SPACING) / aspect_sum
                
                # If height would shrink too much → wrap to next row
                if row_height < min_h:
                    # Remove last image & draw current row
                    last = row.pop()
                    aspect_sum -= last[1]
                    
                    logger.debug(f"[JUSTIFIED_GALLERY] Row height too small ({row_height:.1f} < {min_h:.1f}), wrapping. Row has {len(row)} images")
                    self.render_row_lazy(row, viewport_width, aspect_sum)
                    row = [last]
                    aspect_sum = last[1]
                
                # Log progress every 50 images
                if (idx + 1) % 50 == 0:
                    logger.info(f"[JUSTIFIED_GALLERY] Layout progress: {idx+1}/{len(self.images)} images processed")
            
            # Render final row (not stretched)
            if row:
                logger.debug(f"[JUSTIFIED_GALLERY] Adding final row with {len(row)} images")
                self.render_row_lazy(row, viewport_width, aspect_sum, stretch_last_row=False)
            
            total_time = time.time() - start_time
            logger.info(f"[JUSTIFIED_GALLERY] ========== build_gallery() COMPLETED in {total_time:.3f}s ==========")
            logger.info(f"[JUSTIFIED_GALLERY] Layout created for {images_processed} images, lazy loading will continue")
            
            # Count how many images need to be loaded (file paths, not pixmaps)
            self._total_images_to_load = sum(1 for item in self.images if isinstance(item, str))
            self._images_loaded_count = 0
            self._render_start_time = time.time()
            logger.info(f"[JUSTIFIED_GALLERY] Starting lazy load for {self._total_images_to_load} images")
            print(f"[GALLERY] Starting lazy load for {self._total_images_to_load} images")
            
        except Exception as e:
            logger.error(f"[JUSTIFIED_GALLERY] Error in build_gallery(): {e}", exc_info=True)
        finally:
            self._building = False
            # Ensure visible images are loaded after build completes
            # Use a longer delay to allow layout geometry to be calculated by Qt
            # render_row_lazy() already triggers loading, but we add this as backup
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(300, self.load_visible_images)  # Increased delay for geometry update
    
    def _get_aspect_ratio_for_path(self, file_path):
        """Get aspect ratio for file path without loading full image"""
        if self.parent_viewer:
            # Use parent viewer's aspect ratio cache
            return self.parent_viewer._get_gallery_aspect_ratio(file_path)
        else:
            # Fallback: try to get from EXIF or load minimal image
            import os
            file_ext = os.path.splitext(file_path)[1].lower()
            is_tiff = file_ext in ('.tiff', '.tif')
            
            # Also check file content to detect TIFF files with wrong extension
            if not is_tiff:
                try:
                    from PIL import Image
                    with Image.open(file_path) as test_img:
                        if test_img.format in ('TIFF', 'TIF'):
                            is_tiff = True
                except:
                    pass  # Not a PIL-readable file or not TIFF
            
            # For TIFF files, use PIL to avoid Qt warnings
            if is_tiff:
                try:
                    from PIL import Image
                    with Image.open(file_path) as img:
                        width, height = img.size
                        if height > 0:
                            return width / height
                except:
                    pass
            else:
                # For other formats, try QImageReader (but not for TIFF or RAW to avoid warnings)
                # RAW files use TIFF structure internally and may trigger TIFF warnings
                raw_extensions = ['.arw', '.cr2', '.nef', '.raf', '.orf', '.dng', '.cr3', '.rw2', '.rwl', '.srw', 
                                 '.pef', '.x3f', '.3fr', '.fff', '.iiq', '.cap', '.erf', '.mef', '.mos', '.nrw', '.srf']
                is_raw = file_ext in raw_extensions
                
                if not is_raw:
                    try:
                        from PyQt6.QtGui import QImageReader
                        reader = QImageReader(file_path)
                        size = reader.size()
                        if size.isValid():
                            return size.width() / size.height() if size.height() > 0 else 1.0
                    except:
                        pass
            return 1.0  # Default aspect ratio
    
    def _get_pixmap_for_path(self, file_path):
        """Get pixmap for file path - uses parent viewer's cache if available"""
        import os
        import logging
        logger = logging.getLogger(__name__)
        
        if self.parent_viewer:
            # Use parent viewer's pixmap cache
            pixmap = self.parent_viewer._get_gallery_pixmap(file_path)
            if pixmap and not pixmap.isNull():
                logger.debug(f"[JUSTIFIED_GALLERY] Loaded pixmap from cache: {os.path.basename(file_path)}, size: {pixmap.width()}x{pixmap.height()}")
            else:
                logger.debug(f"[JUSTIFIED_GALLERY] Failed to load pixmap from cache: {os.path.basename(file_path)}")
            return pixmap
        else:
            # Fallback: load directly (use parent viewer's safe loader if available)
            if self.parent_viewer and hasattr(self.parent_viewer, '_load_pixmap_safe'):
                pixmap = self.parent_viewer._load_pixmap_safe(file_path)
            else:
                from PyQt6.QtGui import QPixmap
                import os
                file_ext = os.path.splitext(file_path)[1].lower()
                is_tiff = file_ext in ('.tiff', '.tif')
                if is_tiff:
                    # Use PIL for TIFF to avoid Qt warnings
                    try:
                        from PIL import Image
                        from PyQt6.QtGui import QImage
                        with Image.open(file_path) as pil_image:
                            if pil_image.mode not in ('RGB', 'L'):
                                pil_image = pil_image.convert('RGB')
                            width, height = pil_image.size
                            if pil_image.mode == 'RGB':
                                qimage = QImage(pil_image.tobytes('raw', 'RGB'), width, height, QImage.Format.Format_RGB888)
                            else:
                                qimage = QImage(pil_image.tobytes('raw', 'L'), width, height, QImage.Format.Format_Grayscale8)
                            pixmap = QPixmap.fromImage(qimage) if not qimage.isNull() else QPixmap()
                        # For TIFF files, never use QPixmap(file_path) as it triggers warnings
                        # If PIL fails, return empty QPixmap
                    except:
                        # For TIFF files, never use QPixmap(file_path) as it triggers warnings
                        pixmap = QPixmap()
                else:
                    pixmap = QPixmap(file_path)
            if pixmap and not pixmap.isNull():
                logger.debug(f"[JUSTIFIED_GALLERY] Loaded pixmap directly: {os.path.basename(file_path)}, size: {pixmap.width()}x{pixmap.height()}")
            else:
                logger.debug(f"[JUSTIFIED_GALLERY] Failed to load pixmap directly: {os.path.basename(file_path)}")
            return pixmap
    
    def render_row_lazy(self, row, viewport_width, aspect_sum, stretch_last_row=True):
        """Render one adaptive justified row with lazy loading"""
        from PyQt6.QtWidgets import QHBoxLayout
        
        row_layout = QHBoxLayout()
        row_layout.setSpacing(self.MIN_SPACING)
        row_layout.setContentsMargins(0, 0, 0, 0)
        
        # Compute height that fills width
        if stretch_last_row:
            row_height = (viewport_width - (len(row) - 1) * self.MIN_SPACING) / aspect_sum
        else:
            # Last row stays near target height
            row_height = self.TARGET_ROW_HEIGHT
        
        # Add each image widget with lazy loading
        for item, aspect in row:
            target_width = int(row_height * aspect)
            target_height = int(row_height)
            
            label = ThumbnailLabel()  # Will show "Loading…" initially
            
            # Set fixed size based on calculated dimensions
            label.setFixedSize(target_width, target_height)
            label.setScaledContents(False)
            
            # Make clickable if we have file path
            file_path = item if isinstance(item, str) else None
            if file_path and self.parent_viewer:
                label.file_path = file_path
                label.mousePressEvent = lambda e, fp=file_path: self.parent_viewer._gallery_item_clicked(fp)
            
            # Store tile info for lazy loading
            if isinstance(item, str):
                # File path - will be loaded lazily
                self.tiles.append((label, file_path, target_width, target_height))
            else:
                # Already a QPixmap - load immediately
                scaled = item.scaled(
                    target_width,
                    target_height,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                label.setPixmap(scaled)
                label.setFixedSize(scaled.size())
                label.set_original_pixmap(item)
                label.setText("")  # Clear "Loading…" text
                self.tiles.append((label, None, target_width, target_height))  # No file path needed
            
            row_layout.addWidget(label)
        
        self.container.addLayout(row_layout)
        
        # Don't trigger loading here - wait for build_gallery() to complete
        # Loading will be triggered by build_gallery() after all rows are built
    
    def load_visible_images(self):
        """Load images that are currently visible in viewport with debouncing and prefetch"""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info("[JUSTIFIED_GALLERY] load_visible_images() called")
        
        # Cancel pending load timer (debounce)
        if self._load_timer:
            self._load_timer.stop()
            self._load_timer = None
        
        # Ensure tiles list exists
        if not hasattr(self, 'tiles') or not self.tiles:
            logger.warning("[JUSTIFIED_GALLERY] No tiles list, returning early")
            return
        
        logger.info(f"[JUSTIFIED_GALLERY] Tiles count: {len(self.tiles)}")
        
        # Get visible region from parent scroll area
        visible_rect = None
        from PyQt6.QtWidgets import QScrollArea
        scroll_area = None
        if self.parent():
            parent = self.parent()
            while parent:
                if isinstance(parent, QScrollArea):
                    scroll_area = parent
                    break
                parent = parent.parent()
        
        if scroll_area:
            logger.info(f"[JUSTIFIED_GALLERY] Found scroll area, isVisible: {scroll_area.isVisible()}")
            if scroll_area.isVisible():
                try:
                    viewport = scroll_area.viewport()
                    if viewport:
                        # Get scroll position
                        scroll_y = scroll_area.verticalScrollBar().value()
                        scroll_x = scroll_area.horizontalScrollBar().value()
                        
                        # Get viewport size
                        viewport_width = viewport.width()
                        viewport_height = viewport.height()
                        
                        logger.info(f"[JUSTIFIED_GALLERY] Viewport size: {viewport_width}x{viewport_height}, scroll: ({scroll_x}, {scroll_y})")
                        
                        if viewport_width > 0 and viewport_height > 0:
                            # Calculate visible region in gallery widget coordinates
                            visible_rect = QRect(
                                scroll_x,
                                scroll_y,
                                viewport_width,
                                viewport_height
                            )
                            logger.info(f"[JUSTIFIED_GALLERY] Created visible_rect: {visible_rect}")
                except Exception as e:
                    logger.warning(f"[JUSTIFIED_GALLERY] Error calculating visible rect: {e}", exc_info=True)
                    visible_rect = None
        else:
            logger.info("[JUSTIFIED_GALLERY] No scroll area found")
        
        # If no scroll area or error, load just first row
        if not visible_rect or visible_rect.isEmpty():
            logger.info("[JUSTIFIED_GALLERY] No visible rect, loading first row only")
            # Use simple approach: load first 15 tiles (estimate of first row)
            # Don't rely on geometry which may not be set yet
            max_first_row = 15
            tiles_added = 0
            
            for index in range(min(max_first_row, len(self.tiles))):
                try:
                    label, file_path, target_width, target_height = self.tiles[index]
                    if not file_path:
                        continue  # Already a pixmap, not a file path
                    if file_path in self._loading_tiles:
                        continue  # Already loading
                    
                    # Check if already loaded
                    try:
                        if label and hasattr(label, 'pixmap'):
                            pixmap = label.pixmap()
                            if pixmap and not pixmap.isNull():
                                continue  # Already loaded
                    except RuntimeError:
                        continue
                    
                    self._load_queue.append((index, file_path, target_width, target_height, True))
                    tiles_added += 1
                except (RuntimeError, AttributeError, IndexError) as e:
                    logger.debug(f"[JUSTIFIED_GALLERY] Error adding tile {index} to queue: {e}")
                    continue
            
            logger.info(f"[JUSTIFIED_GALLERY] Added {tiles_added} tiles from first row to load queue")
            if tiles_added > 0:
                self._process_load_queue()
            else:
                logger.warning("[JUSTIFIED_GALLERY] No tiles added to load queue from first row")
            return
        
        # Load visible tiles + prefetch next 1-2 rows
        visible_indices = []
        prefetch_indices = []
        last_visible_y = None
        
        logger.info("[JUSTIFIED_GALLERY] Checking layout geometry validity...")
        
        # Check if layout geometry is valid (widgets may not have positions yet)
        # When layout is first built, all widgets may be at (0,0) or have invalid positions
        geometry_valid = False
        if len(self.tiles) > 0:
            try:
                # Check if we have multiple rows with different y positions
                # If all tiles are at y=0 or same y, layout may not be positioned yet
                y_positions = set()
                valid_geometries = 0
                
                # Sample first 10 tiles to check layout positioning
                sample_size = min(10, len(self.tiles))
                logger.info(f"[JUSTIFIED_GALLERY] Sampling {sample_size} tiles to check geometry...")
                for i in range(sample_size):
                    label = self.tiles[i][0]
                    if label and hasattr(label, 'geometry'):
                        try:
                            rect = label.geometry()
                            logger.debug(f"[JUSTIFIED_GALLERY] Tile {i}: geometry={rect}, width={rect.width()}, height={rect.height()}, y={rect.y()}")
                            if rect.width() > 0 and rect.height() > 0:
                                valid_geometries += 1
                                y_positions.add(rect.y())
                        except (RuntimeError, AttributeError) as e:
                            logger.debug(f"[JUSTIFIED_GALLERY] Error getting geometry for tile {i}: {e}")
                            continue
                
                logger.info(f"[JUSTIFIED_GALLERY] Geometry check: {valid_geometries} valid tiles, {len(y_positions)} different y positions: {sorted(y_positions)}")
                
                # Layout is valid if:
                # 1. At least 3 tiles have valid geometry
                # 2. We have at least 2 different y positions (multiple rows)
                if valid_geometries >= 3 and len(y_positions) >= 2:
                    geometry_valid = True
                    logger.info(f"[JUSTIFIED_GALLERY] Layout geometry VALID: {valid_geometries} valid tiles, {len(y_positions)} rows")
                else:
                    logger.info(f"[JUSTIFIED_GALLERY] Layout geometry NOT VALID: {valid_geometries} valid tiles, {len(y_positions)} rows")
            except Exception as e:
                logger.warning(f"[JUSTIFIED_GALLERY] Error checking geometry validity: {e}", exc_info=True)
                pass
        
        # If geometry not valid yet, only load first row
        if not geometry_valid:
            logger.info("[JUSTIFIED_GALLERY] Layout geometry not valid yet, loading first row only")
            # Use a simpler approach: just load first 10-15 tiles (roughly first row)
            # This avoids relying on geometry which may not be set yet
            first_row_count = 0
            max_first_row = 15  # Estimate: first row typically has 10-15 images
            
            for index, (label, file_path, target_width, target_height) in enumerate(self.tiles):
                if not file_path:
                    continue  # Skip already-loaded pixmaps
                if index >= max_first_row:
                    break  # Limit to first row estimate
                try:
                    if not label:
                        continue
                    first_row_count += 1
                except (RuntimeError, AttributeError):
                    continue
            
            logger.info(f"[JUSTIFIED_GALLERY] Estimated first row size: {first_row_count} tiles")
            
            # Load first row only
            tiles_added = 0
            for index in range(min(first_row_count, len(self.tiles))):
                try:
                    label, file_path, target_width, target_height = self.tiles[index]
                    if not file_path:
                        continue  # Already a pixmap, not a file path
                    if file_path in self._loading_tiles:
                        continue  # Already loading
                    try:
                        pixmap = label.pixmap()
                        if pixmap and not pixmap.isNull():
                            continue  # Already loaded
                    except RuntimeError:
                        continue
                    self._load_queue.append((index, file_path, target_width, target_height, True))
                    tiles_added += 1
                except (RuntimeError, AttributeError, IndexError) as e:
                    logger.debug(f"[JUSTIFIED_GALLERY] Error adding tile {index} to queue: {e}")
                    continue
            
            logger.info(f"[JUSTIFIED_GALLERY] Added {tiles_added} tiles from first row ({first_row_count} tiles in row) to load queue")
            if tiles_added > 0:
                self._process_load_queue()
            else:
                logger.warning("[JUSTIFIED_GALLERY] No tiles added to load queue from first row")
            return
        
        for index, (label, file_path, target_width, target_height) in enumerate(self.tiles):
            if not file_path:
                continue  # Already loaded
            
            # Check if label still exists (might have been deleted during resize)
            try:
                if not label or not hasattr(label, 'geometry'):
                    continue
                
                # Check if already loaded or loading
                try:
                    pixmap = label.pixmap()
                    if pixmap and not pixmap.isNull():
                        continue
                except RuntimeError:
                    # Label was deleted, skip
                    continue
                
                if file_path in self._loading_tiles:
                    continue
                
                label_rect = label.geometry()
                
                # Validate geometry - skip if invalid
                if label_rect.width() <= 0 or label_rect.height() <= 0:
                    continue
                
                is_visible = visible_rect.intersects(label_rect)
                
                if is_visible:
                    visible_indices.append((index, file_path, target_width, target_height))
                    last_visible_y = label_rect.bottom()
                elif last_visible_y is not None:
                    # Prefetch: tiles in next 1-2 rows below visible area
                    if label_rect.top() <= last_visible_y + (self.TARGET_ROW_HEIGHT * 2.5):
                        prefetch_indices.append((index, file_path, target_width, target_height))
            except (RuntimeError, AttributeError) as e:
                # Label was deleted or invalid, skip
                logger.debug(f"[JUSTIFIED_GALLERY] Label at index {index} is invalid (deleted?): {e}")
                continue
            except Exception as e:
                logger.debug(f"[JUSTIFIED_GALLERY] Error checking visibility for tile {index}: {e}")
        
        # Log how many tiles are visible
        if visible_indices or prefetch_indices:
            logger.info(f"[JUSTIFIED_GALLERY] Found {len(visible_indices)} visible tiles, {len(prefetch_indices)} prefetch tiles")
            if len(visible_indices) > 50:
                logger.warning(f"[JUSTIFIED_GALLERY] WARNING: Too many visible tiles ({len(visible_indices)}), geometry may not be valid yet!")
                # Limit to first 20 visible tiles as safety measure
                visible_indices = visible_indices[:20]
                logger.info(f"[JUSTIFIED_GALLERY] Limited to first 20 visible tiles")
        
        # Add visible tiles to queue (priority)
        for index, file_path, target_width, target_height in visible_indices:
            if file_path not in self._loading_tiles:
                self._load_queue.append((index, file_path, target_width, target_height, True))  # True = priority
        
        # Add prefetch tiles to queue (lower priority)
        prefetch_limit = min(10, len(prefetch_indices))
        for index, file_path, target_width, target_height in prefetch_indices[:prefetch_limit]:
            if file_path not in self._loading_tiles:
                self._load_queue.append((index, file_path, target_width, target_height, False))  # False = prefetch
        
        # Start rate-limited loading
        self._process_load_queue()
    
    def _process_load_queue(self):
        """Process load queue with rate limiting"""
        import logging
        logger = logging.getLogger(__name__)
        
        if not self._load_queue:
            logger.debug("[JUSTIFIED_GALLERY] Load queue is empty")
            return
        
        logger.info(f"[JUSTIFIED_GALLERY] Processing load queue: {len(self._load_queue)} items")
        
        # Sort queue: priority items first
        self._load_queue.sort(key=lambda x: not x[4])  # True (priority) comes first
        
        # Process a small batch
        batch = self._load_queue[:self._batch_size]
        self._load_queue = self._load_queue[self._batch_size:]
        
        for index, file_path, target_width, target_height, is_priority in batch:
            if file_path in self._loading_tiles:
                continue  # Already loading
            
            # Check cache first (for row height buckets)
            cache_key = self._get_cache_key(file_path, target_height)
            if cache_key in self._thumbnail_cache:
                # Use cached thumbnail
                cached_pixmap = self._thumbnail_cache[cache_key]
                # Scale to exact size if needed
                if cached_pixmap.height() != target_height:
                    from PyQt6.QtGui import QPixmap
                    from PyQt6.QtCore import Qt
                    scaled = cached_pixmap.scaled(
                        target_width,
                        target_height,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    self.apply_thumbnail(index, scaled.toImage())
                else:
                    self.apply_thumbnail(index, cached_pixmap.toImage())
                continue
            
            # Mark as loading
            self._loading_tiles.add(file_path)
            
            # Create and start load task
            task = ImageLoadTask(
                index=index,
                file_path=file_path,
                target_width=target_width,
                target_height=target_height,
                signal=self.loader_signal,
                parent_viewer=self.parent_viewer
            )
            self.thread_pool.start(task)
        
        # Schedule next batch if queue not empty
        if self._load_queue:
            from PyQt6.QtCore import QTimer
            interval_ms = int(1000 / self._loads_per_second)  # Convert to milliseconds
            if not self._load_rate_timer:
                self._load_rate_timer = QTimer()
                self._load_rate_timer.setSingleShot(True)
                self._load_rate_timer.timeout.connect(self._process_load_queue)
            self._load_rate_timer.start(interval_ms)
    
    def _get_cache_key(self, file_path, row_height):
        """Get cache key for thumbnail, using closest row height bucket"""
        if not file_path:
            return None
        
        # Find closest bucket
        closest_bucket = min(self._row_height_buckets, key=lambda x: abs(x - row_height))
        return (file_path, closest_bucket)
    
    def apply_thumbnail(self, index, image):
        """UI update happens here - called from worker thread via signal"""
        import logging
        import time
        logger = logging.getLogger(__name__)
        
        try:
            # Ensure tiles list exists
            if not hasattr(self, 'tiles'):
                logger.warning(f"[JUSTIFIED_GALLERY] tiles list not found when applying thumbnail at index {index}")
                return
            
            # Check if index is still valid (tiles might have been rebuilt)
            if index >= len(self.tiles):
                logger.debug(f"[JUSTIFIED_GALLERY] Index {index} out of range (tiles count: {len(self.tiles)})")
                return
            
            label, file_path, target_width, target_height = self.tiles[index]
            
            # Check if label still exists (might have been deleted during resize)
            try:
                if not label or not hasattr(label, 'setPixmap'):
                    logger.debug(f"[JUSTIFIED_GALLERY] Label at index {index} is invalid (deleted?)")
                    return
            except RuntimeError:
                logger.debug(f"[JUSTIFIED_GALLERY] Label at index {index} was deleted")
                return
            
            # Check if this tile still needs this image (might have been rebuilt)
            if not file_path:
                return  # This tile was a pre-loaded pixmap, skip
            
            # Remove from loading set
            if file_path in self._loading_tiles:
                self._loading_tiles.remove(file_path)
            
            # Check if label was already updated (prevent duplicate updates)
            try:
                pixmap = label.pixmap()
                if pixmap and not pixmap.isNull():
                    # Already loaded, skip to prevent double counting
                    return
            except RuntimeError:
                # Label was deleted, skip
                logger.debug(f"[JUSTIFIED_GALLERY] Label at index {index} was deleted during update")
                return
            
            # Convert QImage to QPixmap in UI thread (required by Qt)
            from PyQt6.QtGui import QPixmap
            pixmap = QPixmap.fromImage(image)
            
            if pixmap.isNull():
                logger.debug(f"[JUSTIFIED_GALLERY] Failed to convert QImage to QPixmap for index {index}")
                return
            
            # Cache thumbnail at closest row height bucket
            cache_key = self._get_cache_key(file_path, target_height)
            if cache_key:
                # Store in cache (will reuse on resize)
                self._thumbnail_cache[cache_key] = pixmap
                # Limit cache size (keep last 500)
                if len(self._thumbnail_cache) > 500:
                    # Remove oldest entries (simple FIFO)
                    keys_to_remove = list(self._thumbnail_cache.keys())[:100]
                    for key in keys_to_remove:
                        del self._thumbnail_cache[key]
            
            # Update label
            label.setPixmap(pixmap)
            label.setFixedSize(pixmap.size())
            label.setText("")  # Clear "Loading…" text
            
            # Store original pixmap if we have file path
            if file_path and self.parent_viewer:
                original_pixmap = self.parent_viewer._get_gallery_pixmap(file_path)
                if original_pixmap:
                    label.set_original_pixmap(original_pixmap)
            
            # Track rendering progress
            if file_path:  # Only count file paths, not pre-loaded pixmaps
                # Only increment if this is the first time loading
                if hasattr(self, '_images_loaded_count') and hasattr(self, '_total_images_to_load'):
                    self._images_loaded_count += 1
                    
                    # Prevent overflow - cap at total
                    if self._total_images_to_load > 0 and self._images_loaded_count > self._total_images_to_load:
                        self._images_loaded_count = self._total_images_to_load
                    
                    # Log progress every 10 images or when complete
                    if self._total_images_to_load > 0:
                        if self._images_loaded_count % 10 == 0 or self._images_loaded_count == self._total_images_to_load:
                            elapsed = time.time() - self._render_start_time if self._render_start_time else 0
                            progress_pct = (self._images_loaded_count / self._total_images_to_load * 100)
                            logger.info(f"[JUSTIFIED_GALLERY] Rendering progress: {self._images_loaded_count}/{self._total_images_to_load} images ({progress_pct:.1f}%) - elapsed: {elapsed:.2f}s")
                            
                            # Log completion
                            if self._images_loaded_count == self._total_images_to_load:
                                total_render_time = time.time() - self._render_start_time if self._render_start_time else 0
                                logger.info(f"[JUSTIFIED_GALLERY] ========== ALL IMAGES RENDERED in {total_render_time:.3f}s ==========")
                                print(f"[GALLERY] All {self._total_images_to_load} images rendered in {total_render_time:.3f}s")
                
        except Exception as e:
            logger.error(f"[JUSTIFIED_GALLERY] Error applying thumbnail at index {index}: {e}", exc_info=True)
    
    def resizeEvent(self, event):
        """Re-layout when window resizes"""
        import logging
        logger = logging.getLogger(__name__)
        
        # Call super first to ensure widget size is updated
        super().resizeEvent(event)
        
        # Prevent recursive resize events
        if self._resize_in_progress:
            logger.debug("[JUSTIFIED_GALLERY] resizeEvent() called while resize in progress, skipping")
            return
        
        # Prevent building during resize
        if self._building:
            logger.debug("[JUSTIFIED_GALLERY] resizeEvent() called while building, will rebuild after build completes")
            # Schedule rebuild after current build completes
            from PyQt6.QtCore import QTimer
            def retry_resize():
                if not self._building:
                    self.resizeEvent(event)
                else:
                    QTimer.singleShot(100, retry_resize)
            QTimer.singleShot(100, retry_resize)
            return
        
        # Get actual viewport width (same logic as build_gallery)
        new_viewport_width = max(300, self.width() - 16)
        from PyQt6.QtWidgets import QScrollArea
        if self.parent():
            # Try to get width from scroll area viewport
            parent = self.parent()
            while parent:
                if isinstance(parent, QScrollArea):
                    viewport = parent.viewport()
                    if viewport:
                        new_viewport_width = max(300, viewport.width() - 16)
                    break
                parent = parent.parent()
        
        # Store old viewport width if not set
        if not hasattr(self, '_last_viewport_width') or self._last_viewport_width is None:
            self._last_viewport_width = new_viewport_width
            # First time, just trigger visible image loading
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(50, self.load_visible_images)
            return
        
        old_viewport_width = self._last_viewport_width
        
        # Only rebuild if viewport width actually changed significantly (avoid flicker)
        width_diff = abs(old_viewport_width - new_viewport_width)
        if width_diff < 20:  # Increased threshold to prevent rapid rebuilds
            logger.debug(f"[JUSTIFIED_GALLERY] Viewport width change too small ({old_viewport_width} -> {new_viewport_width}), skipping rebuild")
            # Still trigger visible image loading in case scroll position changed
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(50, self.load_visible_images)
            return
        
        logger.info(f"[JUSTIFIED_GALLERY] resizeEvent() - viewport width changed: {old_viewport_width} -> {new_viewport_width}")
        
        # Update stored viewport width
        self._last_viewport_width = new_viewport_width
        
        # Debounce resize: cancel previous timer and start new one
        if self._resize_timer:
            self._resize_timer.stop()
        
        from PyQt6.QtCore import QTimer
        self._resize_timer = QTimer()
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._handle_resize_rebuild)
        self._resize_timer.start(200)  # 200ms debounce as recommended
    
    def _handle_resize_rebuild(self):
        """Handle resize rebuild - recompute layout without clearing tiles"""
        import logging
        logger = logging.getLogger(__name__)
        
        if self._resize_in_progress:
            return
        
        self._resize_in_progress = True
        
        try:
            # Clear only the layout, keep tiles and cached thumbnails
            while self.container.count():
                item = self.container.takeAt(0)
                if item.layout():
                    layout = item.layout()
                    # Remove all widgets from layout first
                    while layout.count():
                        layout_item = layout.takeAt(0)
                        if layout_item.widget():
                            widget = layout_item.widget()
                            widget.hide()
                            widget.deleteLater()
                    layout.deleteLater()
                elif item.widget():
                    widget = item.widget()
                    widget.hide()
                    widget.deleteLater()
            
            # Force immediate update
            self.update()
            self.repaint()
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()
            
            # Rebuild gallery (will reuse cached thumbnails via _process_load_queue)
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(10, self.build_gallery)
            
        except Exception as e:
            logger.error(f"[JUSTIFIED_GALLERY] Error in _handle_resize_rebuild(): {e}", exc_info=True)
        finally:
            self._resize_in_progress = False
    
    def wheelEvent(self, event):
        """Trigger loading of visible images when scrolling with debouncing"""
        # Debounce: cancel previous timer and start new one
        if self._load_timer:
            self._load_timer.stop()
        
        from PyQt6.QtCore import QTimer
        self._load_timer = QTimer()
        self._load_timer.setSingleShot(True)
        self._load_timer.timeout.connect(self.load_visible_images)
        self._load_timer.start(100)  # 100ms debounce
        
        super().wheelEvent(event)
    
    def set_images(self, images):
        """Update the images list and rebuild"""
        import logging
        import time
        logger = logging.getLogger(__name__)
        start_time = time.time()
        logger.info(f"[JUSTIFIED_GALLERY] ========== set_images() STARTED ==========")
        logger.info(f"[JUSTIFIED_GALLERY] New image count: {len(images)}")
        
        # Check if images list has actually changed (optimization: avoid unnecessary rebuilds)
        if hasattr(self, 'images') and self.images == images:
            logger.info(f"[JUSTIFIED_GALLERY] Image list unchanged, skipping rebuild (preserving {len(self.tiles)} tiles)")
            # Just trigger visible image loading in case scroll position changed
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(50, self.load_visible_images)
            return
        
        # Prevent building if already building
        if self._building:
            logger.warning("[JUSTIFIED_GALLERY] set_images() called while building, will rebuild after current build completes")
            # Use QTimer to retry after current build
            from PyQt6.QtCore import QTimer
            def retry_set_images():
                if not self._building:
                    self.set_images(images)
                else:
                    QTimer.singleShot(100, retry_set_images)
            QTimer.singleShot(100, retry_set_images)
            return
        
        self.images = images
        
        # Reset rendering progress tracking
        self._images_loaded_count = 0
        self._render_start_time = None
        
        # Clear loading tracking
        self._loading_tiles.clear()
        
        # Clear tiles list
        self.tiles = []
        
        # Clear layout safely and immediately update display
        clear_start = time.time()
        while self.container.count():
            item = self.container.takeAt(0)
            if item.layout():
                layout = item.layout()
                # Remove all widgets from layout first
                while layout.count():
                    layout_item = layout.takeAt(0)
                    if layout_item.widget():
                        widget = layout_item.widget()
                        widget.hide()  # Hide immediately
                        widget.deleteLater()
                layout.deleteLater()
            elif item.widget():
                widget = item.widget()
                widget.hide()  # Hide immediately
                widget.deleteLater()
        clear_time = time.time() - clear_start
        logger.debug(f"[JUSTIFIED_GALLERY] Layout cleared in {clear_time:.3f}s")
        
        # Force immediate update to clear old layout
        self.update()
        self.repaint()
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()
        
        # Rebuild - but only if widget has proper size
        if self.width() > 0:
            self.build_gallery()
        else:
            logger.warning("[JUSTIFIED_GALLERY] Widget has no width yet, delaying build")
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(100, self._delayed_build)
        
        total_time = time.time() - start_time
        logger.info(f"[JUSTIFIED_GALLERY] ========== set_images() COMPLETED in {total_time:.3f}s ==========")


class CustomTitleBar(QFrame):
    """Material Design 3 style custom title bar for frameless window."""
    def __init__(self, parent=None, title="RAW Image Viewer"):
        super().__init__(parent)
        self.parent = parent
        self.setFixedHeight(40)  # Smaller height
        
        # Use the same background color as image viewing area (#1E1E1E)
        self.setStyleSheet("""
            QFrame {
                background-color: #1E1E1E;
                border-bottom: 1px solid #2E2E2E;
            }
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 6, 0)
        layout.setSpacing(8)
        
        # Logo Icon (Favicon)
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(24, 24)  # Smaller icon
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setStyleSheet("background-color: transparent; border: none;")
        
        # Load favicon - try multiple paths
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))
        
        icon_paths = [
            os.path.join(base_path, "icons", "favicon.ico"),
            os.path.join(base_path, "favicon.ico"),
            os.path.join(os.getcwd(), "icons", "favicon.ico"),
            os.path.join(os.getcwd(), "favicon.ico"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons", "favicon.ico"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "favicon.ico"),
            "icons/favicon.ico",
            "favicon.ico"
        ]
        
        icon_loaded = False
        for icon_path in icon_paths:
            if os.path.exists(icon_path):
                try:
                    icon = QIcon(icon_path)
                    pixmap = icon.pixmap(24, 24)
                    if not pixmap.isNull():
                        self.icon_label.setPixmap(pixmap)
                        icon_loaded = True
                        break
                except Exception:
                    continue
        
        if not icon_loaded:
            # Fallback to 'R' if favicon not found
            self.icon_label.setText("R")
            self.icon_label.setStyleSheet("""
                background-color: #4A4A4A;
                color: #E0E0E0;
                border-radius: 12px;
                font-weight: bold;
                font-size: 14px;
            """)
        layout.addWidget(self.icon_label)
        
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("""
            color: #E0E0E0;
            font-size: 13px;
            font-weight: 500;
            font-family: 'Roboto', 'Segoe UI', sans-serif;
        """)
        layout.addWidget(self.title_label)
        
        layout.addStretch()
        
        # Window Controls - Smaller buttons
        control_btn_style = """
            QPushButton {
                background-color: transparent;
                color: #E0E0E0;
                border: none;
                width: 36px;
                height: 28px;
                font-size: 14px;
            }
            QPushButton:hover { 
                background-color: rgba(255, 255, 255, 0.1); 
            }
        """
        
        self.min_btn = QPushButton("—")
        self.min_btn.setStyleSheet(control_btn_style)
        self.min_btn.clicked.connect(self.parent.showMinimized)
        layout.addWidget(self.min_btn)
        
        self.max_btn = QPushButton("⬜")
        self.max_btn.setStyleSheet(control_btn_style)
        self.max_btn.clicked.connect(self._toggle_maximize)
        layout.addWidget(self.max_btn)
        
        self.close_btn = QPushButton("✕")
        self.close_btn.setStyleSheet(control_btn_style + "QPushButton:hover { background-color: #f44336; color: white; }")
        self.close_btn.clicked.connect(self.parent.close)
        layout.addWidget(self.close_btn)
        
        self._is_maximized = False
        self._dragging = False
        self._drag_pos = None

    def _toggle_maximize(self):
        if self._is_maximized:
            self.parent.showNormal()
            self.max_btn.setText("⬜")
        else:
            self.parent.showMaximized()
            self.max_btn.setText("❐")
        self._is_maximized = not self._is_maximized
        # Update title bar state
        if hasattr(self.parent, 'title_bar'):
            self.parent.title_bar._is_maximized = self._is_maximized

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_pos = event.globalPosition().toPoint() - self.parent.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging and not self._is_maximized:
            self.parent.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._toggle_maximize()
            event.accept()
    
    def set_title(self, title):
        """Update the title text"""
        self.title_label.setText(title)


class CustomConfirmDialog(QDialog):
    """Material Design 3 style confirmation dialog with custom title bar."""
    def __init__(self, parent=None, title="Confirm Delete", message="", informative_text=""):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setModal(True)
        
        # Main container with rounded corners and shadow effect
        self.container = QWidget(self)
        self.container.setStyleSheet("""
            QWidget {
                background-color: #1E1E1E;
                border-radius: 12px;
            }
        """)
        
        # Main layout
        main_layout = QVBoxLayout(self.container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Custom title bar
        self.title_bar = CustomTitleBar(self.container, title=title)
        # Remove minimize and maximize buttons for dialog
        self.title_bar.min_btn.hide()
        self.title_bar.max_btn.hide()
        # Update close button to reject dialog
        self.title_bar.close_btn.clicked.disconnect()
        self.title_bar.close_btn.clicked.connect(self.reject)
        main_layout.addWidget(self.title_bar)
        
        # Content area
        content_widget = QWidget()
        content_widget.setStyleSheet("""
            QWidget {
                background-color: #1E1E1E;
            }
        """)
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(24, 20, 24, 24)
        content_layout.setSpacing(16)
        
        # Icon and message
        icon_layout = QHBoxLayout()
        icon_layout.setContentsMargins(0, 0, 0, 0)
        icon_layout.setSpacing(12)
        
        # Warning icon (using emoji or text)
        icon_label = QLabel("⚠")
        icon_label.setFixedSize(48, 48)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 152, 0, 0.1);
                border-radius: 24px;
                color: #FF9800;
                font-size: 28px;
            }
        """)
        icon_layout.addWidget(icon_label)
        
        # Message text
        message_layout = QVBoxLayout()
        message_layout.setContentsMargins(0, 0, 0, 0)
        message_layout.setSpacing(8)
        
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setStyleSheet("""
            QLabel {
                color: #E0E0E0;
                font-size: 16px;
                font-weight: 500;
                font-family: 'Roboto', 'Segoe UI', sans-serif;
            }
        """)
        message_layout.addWidget(message_label)
        
        if informative_text:
            info_label = QLabel(informative_text)
            info_label.setWordWrap(True)
            info_label.setStyleSheet("""
                QLabel {
                    color: #B0B0B0;
                    font-size: 14px;
                    font-family: 'Roboto', 'Segoe UI', sans-serif;
                    line-height: 1.5;
                }
            """)
            message_layout.addWidget(info_label)
        
        icon_layout.addLayout(message_layout)
        icon_layout.addStretch()
        content_layout.addLayout(icon_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 8, 0, 0)
        button_layout.setSpacing(12)
        button_layout.addStretch()
        
        # Cancel button (MD3 style - outlined)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedHeight(40)
        cancel_btn.setMinimumWidth(100)
        cancel_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #E0E0E0;
                border: 1px solid #4A4A4A;
                border-radius: 20px;
                font-size: 14px;
                font-weight: 500;
                font-family: 'Roboto', 'Segoe UI', sans-serif;
                padding: 0px 24px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.05);
                border-color: #5A5A5A;
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 0.1);
            }
        """)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        # Delete button (MD3 style - filled, with warning color)
        delete_btn = QPushButton("Delete")
        delete_btn.setFixedHeight(40)
        delete_btn.setMinimumWidth(100)
        delete_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF5252;
                color: white;
                border: none;
                border-radius: 20px;
                font-size: 14px;
                font-weight: 500;
                font-family: 'Roboto', 'Segoe UI', sans-serif;
                padding: 0px 24px;
            }
            QPushButton:hover {
                background-color: #FF6B6B;
            }
            QPushButton:pressed {
                background-color: #FF4444;
            }
        """)
        delete_btn.clicked.connect(self.accept)
        button_layout.addWidget(delete_btn)
        
        content_layout.addLayout(button_layout)
        main_layout.addWidget(content_widget)
        
        # Set container size and position
        self.container.setFixedSize(420, 220)
        container_layout = QVBoxLayout(self)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addWidget(self.container)
        
        # Set dialog size (slightly larger for shadow effect)
        self.setFixedSize(420, 220)
        
        # Center on parent
        if parent:
            parent_geometry = parent.geometry()
            dialog_x = parent_geometry.x() + (parent_geometry.width() - self.width()) // 2
            dialog_y = parent_geometry.y() + (parent_geometry.height() - self.height()) // 2
            self.move(dialog_x, dialog_y)
        
        # Store result
        self.result_value = False
    
    def accept(self):
        self.result_value = True
        super().accept()
    
    def reject(self):
        self.result_value = False
        super().reject()
    
    def mousePressEvent(self, event):
        """Allow dragging the dialog"""
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle dragging"""
        if hasattr(self, '_dragging') and self._dragging:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
            return
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Stop dragging"""
        if hasattr(self, '_dragging'):
            self._dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)


class RAWImageViewer(QMainWindow):
    def _load_pixmap_safe(self, file_path):
        """Safely load QPixmap, using PIL for TIFF files to avoid Qt warnings"""
        import os
        from PyQt6.QtGui import QPixmap, QImage
        
        file_ext = os.path.splitext(file_path)[1].lower()
        is_tiff = file_ext in ('.tiff', '.tif')
        
        # Also check file content to detect TIFF files with wrong extension
        if not is_tiff:
            try:
                from PIL import Image
                with Image.open(file_path) as test_img:
                    if test_img.format in ('TIFF', 'TIF'):
                        is_tiff = True
            except:
                pass  # Not a PIL-readable file or not TIFF
        
        # For TIFF files, use PIL to avoid Qt TIFF plugin warnings
        if is_tiff:
            try:
                from PIL import Image
                with Image.open(file_path) as pil_image:
                    # Convert to RGB if necessary
                    if pil_image.mode not in ('RGB', 'L'):
                        pil_image = pil_image.convert('RGB')
                    
                    width, height = pil_image.size
                    if pil_image.mode == 'RGB':
                        qimage = QImage(pil_image.tobytes('raw', 'RGB'), width, height, QImage.Format.Format_RGB888)
                    elif pil_image.mode == 'L':
                        qimage = QImage(pil_image.tobytes('raw', 'L'), width, height, QImage.Format.Format_Grayscale8)
                    else:
                        rgb_pil = pil_image.convert('RGB')
                        qimage = QImage(rgb_pil.tobytes('raw', 'RGB'), width, height, QImage.Format.Format_RGB888)
                    
                    if not qimage.isNull():
                        return QPixmap.fromImage(qimage)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"[LOAD] PIL fallback failed for TIFF: {os.path.basename(file_path)}: {e}")
                # For TIFF files, never use QPixmap(file_path) as it triggers warnings
                # Return empty QPixmap instead
                return QPixmap()
        
        # For other formats, use QPixmap directly
        return QPixmap(file_path)
    
    def __init__(self):
        print("  [RAWImageViewer] Starting initialization...", flush=True)
        super().__init__()
        print("  [RAWImageViewer] QMainWindow.__init__() completed", flush=True)
        self.current_image = None
        self.current_pixmap = None

        # Enhanced zoom and pan state tracking
        # Note: Only using simple toggle between fit-to-window and 100% zoom
        self.current_zoom_level = 1.0  # Current zoom level (1.0 = 100%)
        self.fit_to_window = True  # Whether we're in fit-to-window mode
        self.zoom_center_point = None  # Store center point for zooming

        self._is_half_size_displayed = False  # Track if currently displaying half_size image
        self._full_resolution_loading = False  # Track if full resolution is being loaded
        self._original_image_size = None  # Store original image dimensions (width, height) from EXIF or RAW metadata
        # Panning state
        self.panning = False
        self.last_pan_point = QPoint()
        self.start_scroll_x = 0
        self.start_scroll_y = 0

        # Folder scanning and file list management
        self.current_folder = None
        self.image_files = []  # List of all image files in current folder
        self.current_file_index = -1  # Index of current file in the list
        self.current_file_path = None  # Path of currently loaded file
        self.thumbnail_cache = {}  # Cache for thumbnails
        self.film_strip_visible = False
        self.thumbnail_threads = []  # Track running thumbnail threads
        
        # View mode: 'single' for single image view, 'gallery' for gallery view
        self.view_mode = 'single'  # Default to single image view
        # GALLERY FUNCTIONALITY COMMENTED OUT
        # self.gallery_widget = None  # Gallery view widget
        # self.gallery_justified = None  # JustifiedGallery widget
        # self.gallery_row_height = 200  # Fixed row height (matching reference)
        # self.gallery_pixmaps = {}  # Cache for gallery pixmaps (store original pixmaps)
        # self.gallery_aspect_cache = {}  # Cache aspect ratios
        # self._gallery_thumb_labels = {}  # Store thumbnail labels
        self._gallery_load_tracking = {}  # Track loading status
        self._gallery_load_start_time = None  # Track loading start time

        # Navigation rate limiting and concurrency control
        self._navigation_in_progress = False  # Flag to prevent overlapping navigations
        self._last_navigation_time = 0  # Timestamp of last navigation for rate limiting
        self._pending_navigation = None  # Store pending navigation request (file_path) for debouncing
        self._navigation_timer = None  # QTimer for debouncing rapid navigation
        
        # Cleanup concurrency control
        self._cleanup_lock = threading.Lock()  # Lock to prevent multiple cleanup operations simultaneously
        self._cleanup_in_progress = False  # Flag to track if cleanup is in progress

        # Initialize enhanced performance components
        print("  [RAWImageViewer] Initializing image cache...", flush=True)
        self.image_cache = get_image_cache()
        print("  [RAWImageViewer] Image cache initialized", flush=True)
        # Pass RAWProcessor class to PreloadManager for consistent processing (legacy support)
        print("  [RAWImageViewer] Initializing PreloadManager...", flush=True)
        self.preload_manager = PreloadManager(max_preload_threads=8, processor_class=RAWProcessor)
        print("  [RAWImageViewer] PreloadManager initialized", flush=True)
        self.current_processor = None  # Legacy support - will be phased out
        self._pending_thumbnail = None  # Store thumbnail when not immediately displayed
        self._exif_data_ready = False  # Flag to track if EXIF data is available
        
        # Initialize new unified image load manager
        print("  [RAWImageViewer] Initializing ImageLoadManager...", flush=True)
        self.image_manager = get_image_load_manager(max_workers=4)
        print("  [RAWImageViewer] ImageLoadManager initialized", flush=True)
        print("  [RAWImageViewer] Connecting ImageLoadManager signals...", flush=True)
        self._connect_image_manager_signals()
        print("  [RAWImageViewer] ImageLoadManager signals connected", flush=True)

        # Thumbnail display preferences
        # User preference: show thumbnails even at 100% zoom
        self.show_thumbnails_when_zoomed = False

        # Connect cache signals for performance monitoring
        print("  [RAWImageViewer] Connecting cache signals...", flush=True)
        self.image_cache.cache_hit.connect(self.on_cache_hit)
        self.image_cache.memory_warning.connect(self.on_memory_warning)
        print("  [RAWImageViewer] Cache signals connected", flush=True)

        print("  [RAWImageViewer] Initializing UI...", flush=True)
        self.init_ui()
        print("  [RAWImageViewer] UI initialized", flush=True)

        # Display cache initialization message
        print("  [RAWImageViewer] Getting cache stats...", flush=True)
        cache_stats = self.image_cache.get_cache_stats()
        memory_info = cache_stats['memory_info']
        print(f"✓ Enhanced image cache initialized", flush=True)
        print(f"  Cache budget: {cache_stats['cache_budget_mb']}MB", flush=True)
        print(
            f"  Max full images: {cache_stats['full_image_cache']['max_size']}", flush=True)
        print(
            f"  Max thumbnails: {cache_stats['thumbnail_cache']['max_size']}", flush=True)
        print(
            f"  Available memory: {memory_info['system_available_gb']:.1f}GB", flush=True)

        # Try to restore previous session
        print("  [RAWImageViewer] Restoring session state...", flush=True)
        if not self.restore_session_state():
            # If no session, show default message
            pass
        print("  [RAWImageViewer] Initialization complete!", flush=True)

    def _connect_image_manager_signals(self):
        """連接 ImageLoadManager 的永久信號（事件驅動架構）"""
        # 縮圖就緒
        self.image_manager.thumbnail_ready.connect(self.on_manager_thumbnail_ready)
        # 完整圖像就緒
        self.image_manager.image_ready.connect(self.on_manager_image_ready)
        # QPixmap 就緒（非 RAW 文件）
        self.image_manager.pixmap_ready.connect(self.on_manager_pixmap_ready)
        # EXIF 數據就緒
        self.image_manager.exif_data_ready.connect(self.on_manager_exif_ready)
        # 錯誤處理
        self.image_manager.error_occurred.connect(self.on_manager_error)
        # 進度更新
        self.image_manager.progress_updated.connect(self.on_manager_progress)

    def on_manager_thumbnail_ready(self, file_path: str, thumbnail):
        """處理 ImageLoadManager 的縮圖就緒信號"""
        import logging
        logger = logging.getLogger(__name__)
        
        # 只處理當前文件的縮圖
        if file_path != self.current_file_path:
            logger.debug(f"[MANAGER] Thumbnail for different file: {os.path.basename(file_path)}")
            return
        
        logger.info(f"[MANAGER] Thumbnail ready for {os.path.basename(file_path)}")
        # Mark that orientation is already applied (UnifiedImageProcessor applies it to thumbnails)
        logger.debug(f"[MANAGER] Setting _orientation_already_applied = True before display_numpy_image (thumbnail)")
        self._orientation_already_applied = True
        if self._should_show_thumbnail():
            self.display_numpy_image(thumbnail)
            self.status_bar.showMessage("Preview loaded - processing full image...")
        else:
            self._pending_thumbnail = thumbnail
            self.status_bar.showMessage("Processing full image for quality evaluation...")
        logger.debug(f"[MANAGER] Resetting _orientation_already_applied = False after display_numpy_image (thumbnail)")
        self._orientation_already_applied = False  # Reset flag

    def on_manager_image_ready(self, file_path: str, image):
        """處理 ImageLoadManager 的完整圖像就緒信號"""
        import logging
        logger = logging.getLogger(__name__)
        
        # 只處理當前文件的圖像
        if file_path != self.current_file_path:
            logger.debug(f"[MANAGER] Image for different file: {os.path.basename(file_path)}")
            return
        
        logger.info(f"[MANAGER] Full image ready for {os.path.basename(file_path)}")
        # Mark that orientation is already applied (UnifiedImageProcessor applies it)
        logger.debug(f"[MANAGER] Setting _orientation_already_applied = True before display_numpy_image")
        self._orientation_already_applied = True
        self.display_numpy_image(image)
        logger.debug(f"[MANAGER] Resetting _orientation_already_applied = False after display_numpy_image")
        self._orientation_already_applied = False  # Reset flag
        self.status_bar.showMessage(f"Loaded {os.path.basename(file_path)}")
        self.setFocus()
        self.save_session_state()
        self._start_preloading()

    def on_manager_pixmap_ready(self, file_path: str, pixmap):
        """處理 ImageLoadManager 的 QPixmap 就緒信號（非 RAW 文件）"""
        import logging
        logger = logging.getLogger(__name__)
        
        # 只處理當前文件的 pixmap
        if file_path != self.current_file_path:
            logger.debug(f"[MANAGER] Pixmap for different file: {os.path.basename(file_path)}")
            return
        
        logger.info(f"[MANAGER] Pixmap ready for {os.path.basename(file_path)}")
        # 應用方向校正
        orientation = self.get_orientation_from_exif(file_path)
        pixmap = self.apply_orientation_to_pixmap(pixmap, orientation)
        self.display_pixmap(pixmap)
        self.status_bar.showMessage(f"Loaded {os.path.basename(file_path)}")
        self.setFocus()
        self.save_session_state()
        self._start_preloading()

    def on_manager_exif_ready(self, file_path: str, exif_data: dict):
        """處理 ImageLoadManager 的 EXIF 數據就緒信號"""
        import logging
        logger = logging.getLogger(__name__)
        
        # 只處理當前文件的 EXIF
        if file_path != self.current_file_path:
            logger.debug(f"[MANAGER] EXIF for different file: {os.path.basename(file_path)}")
            return
        
        logger.info(f"[MANAGER] EXIF data ready for {os.path.basename(file_path)}")
        self._exif_data_ready = True
        self.update_status_bar()

    def on_manager_error(self, file_path: str, error_message: str):
        """處理 ImageLoadManager 的錯誤信號"""
        import logging
        logger = logging.getLogger(__name__)
        
        # 只處理當前文件的錯誤
        if file_path != self.current_file_path:
            logger.debug(f"[MANAGER] Error for different file: {os.path.basename(file_path)}")
            return
        
        logger.error(f"[MANAGER] Error loading {os.path.basename(file_path)}: {error_message}")
        self.show_error("Load Error", f"Failed to load image: {error_message}")

    def on_manager_progress(self, file_path: str, status_message: str):
        """處理 ImageLoadManager 的進度更新信號"""
        import logging
        logger = logging.getLogger(__name__)
        
        # 只處理當前文件的進度
        if file_path != self.current_file_path:
            return
        
        filename = os.path.basename(file_path)
        self.status_bar.showMessage(f"{filename}: {status_message}")

    def get_orientation_from_exif(self, file_path):
        """Extract orientation from EXIF data for non-RAW files"""
        try:
            # Suppress exifread warnings for unsupported file formats
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, module='exifread')
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)

                # Debug: Print EXIF orientation, make, and model
                orientation_tag = tags.get('Image Orientation')
                make_tag = tags.get('Image Make')
                model_tag = tags.get('Image Model')
                print(f"[DEBUG] EXIF Orientation: {orientation_tag}")
                print(f"[DEBUG] EXIF Make: {make_tag}")
                print(f"[DEBUG] EXIF Model: {model_tag}")

                # Check for orientation tag
                orientation_tag = tags.get('Image Orientation')
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

    def apply_orientation_to_pixmap(self, pixmap, orientation):
        """Apply orientation correction to QPixmap"""
        # Check if this is a camera that stores image data pre-rotated
        # Some cameras (like Sony, Leica) store image data in the correct orientation
        # and the EXIF orientation tag may be misleading
        if self.is_camera_pre_rotated():
            return pixmap

        if orientation == 1:
            # Normal orientation, no changes needed
            return pixmap

        transform = QTransform()

        if orientation == 2:
            # Mirrored horizontal
            transform.scale(-1, 1)
        elif orientation == 3:
            # Rotated 180 degrees
            transform.rotate(180)
        elif orientation == 4:
            # Mirrored vertical
            transform.scale(1, -1)
        elif orientation == 5:
            # Mirrored horizontal then rotated 90 CCW
            transform.scale(-1, 1)
            transform.rotate(-90)
        elif orientation == 6:
            # Rotated 90 CW - need to rotate 90 CCW to correct
            transform.rotate(90)
        elif orientation == 7:
            # Mirrored horizontal then rotated 90 CW
            transform.scale(-1, 1)
            transform.rotate(90)
        elif orientation == 8:
            # Rotated 90 CCW - need to rotate 90 CW to correct
            transform.rotate(-90)

        return pixmap.transformed(transform)

    def is_camera_pre_rotated(self):
        """Check if this camera stores image data pre-rotated for non-RAW files"""
        # CRITICAL: Only skip orientation correction for RAW files, not JPEG files
        # JPEG files always need orientation correction based on EXIF orientation tag
        # RAW files from certain cameras may be pre-rotated, but JPEG files are not
        try:
            # Check if this is a RAW file
            raw_extensions = {'.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2', 
                             '.pef', '.srw', '.x3f', '.raf', '.3fr', '.fff', '.iiq', 
                             '.cap', '.erf', '.mef', '.mos', '.nrw', '.rwl', '.srf'}
            file_ext = os.path.splitext(self.current_file_path)[1].lower()
            
            # For JPEG and other non-RAW files, always apply orientation correction
            if file_ext not in raw_extensions:
                return False
            
            # For RAW files, check camera make
            # Read camera make from EXIF
            with open(self.current_file_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)
                make = tags.get('Image Make')

                if make:
                    make_str = str(make).upper()
                    # Sony cameras often store RAW data pre-rotated (but not JPEG)
                    if 'SONY' in make_str:
                        return True

                    # Leica cameras also store RAW data pre-rotated (but not JPEG)
                    if 'LEICA' in make_str:
                        return True

                    # Hasselblad cameras also store RAW data pre-rotated (but not JPEG)
                    if 'HASSELBLAD' in make_str:
                        return True

        except Exception:
            pass

        return False

    def init_ui(self):
        """Initialize the user interface"""
        # Set window to frameless for custom title bar
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle('RAW Image Viewer')
        
        # Set simple background style (no rounded corners - simplifies window resizing)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1E1E1E;
            }
        """)
        
        # Set icon based on platform and available files
        icon_path = None
        # Use resource_path to find icons, ensuring it works when bundled
        ico_path = resource_path(os.path.join('icons', 'appicon.ico'))
        icns_path = resource_path(os.path.join('icons', 'appicon.icns'))
        png_path = resource_path(os.path.join('icons', 'appicon.png'))

        if platform.system() == 'Windows' and os.path.exists(ico_path):
            icon_path = ico_path
        elif platform.system() == 'Darwin' and os.path.exists(icns_path):
            icon_path = icns_path
        elif os.path.exists(png_path):
            icon_path = png_path

        if icon_path:
            self.setWindowIcon(QIcon(icon_path))

        # Calculate minimum width for 5 images at 4:3 aspect ratio
        # Each image: height=200px, width=200*(4/3)=267px
        # 5 images: 5*267 = 1335px
        # Spacing: 4 gaps * 4px = 16px
        # Margins: 8px * 2 = 16px
        # Total: 1335 + 16 + 16 = 1367px, round up to 1400px for comfortable display
        self.setGeometry(100, 100, 1400, 800)
        self.setMinimumSize(800, 600)
        self.setAcceptDrops(True)
        
        # Initialize resize tracking for frameless window edge resizing
        self._resize_edge_active = None
        self._resize_start_pos = None
        self._resize_start_geometry = None
        
        # Enable native window resizing for Windows (allows edge dragging)
        if platform.system() == 'Windows':
            self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
            # Ensure no mask is set (allows mouse events at edges)
            self.setMask(QRegion())
            # Don't use translucent background (simplifies event handling)
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        
        # Create custom title bar
        self.title_bar = CustomTitleBar(self, title="RAW Image Viewer")
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        # Reduce padding for image view
        main_layout.setContentsMargins(0, 0, 0, 0)  # Set padding to 0
        main_layout.setSpacing(0)  # Remove spacing
        
        # Add title bar first
        main_layout.addWidget(self.title_bar)
        # Menu bar removed as requested
        self.scroll_area = QScrollArea()
        # Key: allow scrolling when image is larger
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Disable scrollbars completely - user can pan by dragging
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Apply Material Design 3 scrollbar styling
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #1E1E1E;
            }
            QScrollArea > QWidget > QWidget {
                border: none;
                background-color: #1E1E1E;
            }
            /* Material Design 3 Scrollbar Styling */
            QScrollBar:vertical {
                background: transparent;
                width: 12px;
                margin: 0px;
                border: none;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 0.2);
                min-height: 30px;
                border-radius: 6px;
                margin: 2px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(255, 255, 255, 0.3);
            }
            QScrollBar::handle:vertical:pressed {
                background: rgba(255, 255, 255, 0.4);
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
                width: 0px;
            }
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: transparent;
            }
            QScrollBar:horizontal {
                background: transparent;
                height: 12px;
                margin: 0px;
                border: none;
            }
            QScrollBar::handle:horizontal {
                background: rgba(255, 255, 255, 0.2);
                min-width: 30px;
                border-radius: 6px;
                margin: 2px;
            }
            QScrollBar::handle:horizontal:hover {
                background: rgba(255, 255, 255, 0.3);
            }
            QScrollBar::handle:horizontal:pressed {
                background: rgba(255, 255, 255, 0.4);
            }
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {
                height: 0px;
                width: 0px;
            }
            QScrollBar::add-page:horizontal,
            QScrollBar::sub-page:horizontal {
                background: transparent;
            }
        """)
        self.image_label = QLabel()
        # Center the label in viewport, but left-align the text content
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.image_label.setText(
            "No image loaded\n\n"
            "Click 🗁 or drag and drop a folder or image to load it\n"
            "Press Space to toggle between fit-to-window and 100% zoom\n"
            "Double-click image to zoom in/out\n"
            "Click and drag to pan when zoomed\n"
            "Use Left/Right arrow keys to navigate between images (preserves zoom if zoomed in)\n"
            "Press Down Arrow to move the current image to Discard folder\n"
            "Press Delete to remove the current image"
        )
        self.image_label.setStyleSheet(
            "QLabel { color: #666; font-size: 14px; }")
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self.image_mouse_press_event
        self.image_label.mouseMoveEvent = self.image_mouse_move_event
        self.image_label.mouseReleaseEvent = self.image_mouse_release_event
        self.image_label.mouseDoubleClickEvent = self.image_double_click_event
        self.scroll_area.setWidget(self.image_label)
        main_layout.addWidget(self.scroll_area)
        # --- Status bar with Material Design 3 styling ---
        # Material Design 3 color scheme:
        # - Surface: #1E1E1E (dark background)
        # - On Surface: #E0E0E0 (primary text)
        # - Surface Variant: #2A2A2A (elevated surface)
        # - Outline: #2E2E2E (borders)
        # - Secondary: #B0B0B0 (secondary text)
        # Create status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("")  # Empty message when no image loaded
        # Disable resize grip (dotted triangle in bottom-right corner)
        self.status_bar.setSizeGripEnabled(False)
        # Set simple status bar style (no rounded corners)
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #1E1E1E !important;
                color: #E0E0E0;
                border-top: 1px solid #2E2E2E;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: 400;
            }
        """)
        
        # Create custom status bar widget with horizontal layout
        status_widget = QWidget()
        status_widget.setObjectName("status_widget")  # Set object name for finding it later
        status_layout = QHBoxLayout(status_widget)
        # Add left padding to balance with right padding of counter (12px)
        status_layout.setContentsMargins(12, 0, 0, 0)
        status_layout.setSpacing(20)
        
        # Left side buttons container
        left_buttons_widget = QWidget()
        left_buttons_layout = QHBoxLayout(left_buttons_widget)
        left_buttons_layout.setContentsMargins(0, 0, 0, 0)
        # Add 12px spacing between buttons (🗁 and Gallery)
        left_buttons_layout.setSpacing(12)
        
        # Open button (left side) - Material Design 3 text button style with folder icon
        self.open_button = QPushButton("🗁")  # U+1F5C1 folder icon
        self.open_button.setFlat(True)
        self.open_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.open_button.clicked.connect(self.open_file)
        self.open_button.setStyleSheet("""
            QPushButton {
                color: #B0B0B0;
                font-size: 13px;
                font-weight: 700;
                padding: 0px;
                border: none;
                background: transparent;
                text-align: center;
                letter-spacing: 0.25px;
            }
            QPushButton:hover {
                color: #E0E0E0;
                background-color: rgba(255, 255, 255, 0.05);
                border-radius: 4px;
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 0.1);
            }
        """)
        left_buttons_layout.addWidget(self.open_button, 0, alignment=Qt.AlignmentFlag.AlignVCenter)
        
        # Sort toggle button (left side) - Material Design 3 text button style
        # Hidden by default in single view, shown in gallery view
        self.sort_toggle_button = QPushButton()
        self.sort_toggle_button.setFlat(True)
        self.sort_toggle_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._update_sort_button_text()
        self.sort_toggle_button.clicked.connect(self.toggle_sort_method)
        self.sort_toggle_button.setStyleSheet("""
            QPushButton {
                color: #B0B0B0;
                font-size: 13px;
                font-weight: 500;
                padding: 6px 12px;
                border: none;
                background: transparent;
                text-align: left;
                letter-spacing: 0.25px;
            }
            QPushButton:hover {
                color: #E0E0E0;
                background-color: rgba(255, 255, 255, 0.05);
                border-radius: 4px;
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 0.1);
            }
        """)
        left_buttons_layout.addWidget(self.sort_toggle_button)
        self.sort_toggle_button.hide()  # Hidden by default (single view)
        
        # GALLERY FUNCTIONALITY COMMENTED OUT
        # View mode toggle button (single/gallery view)
        # self.view_mode_button = QPushButton("Gallery")
        # self.view_mode_button.setFlat(True)
        # self.view_mode_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        # self.view_mode_button.clicked.connect(self.toggle_view_mode)
        # self.view_mode_button.setStyleSheet("""
        #     QPushButton {
        #         color: #B0B0B0;
        #         font-size: 13px;
        #         font-weight: 500;
        #         padding: 0px;
        #         border: none;
        #         background: transparent;
        #         text-align: center;
        #         letter-spacing: 0.25px;
        #     }
        #     QPushButton:hover {
        #         color: #E0E0E0;
        #         background-color: rgba(255, 255, 255, 0.05);
        #         border-radius: 4px;
        #     }
        #     QPushButton:pressed {
        #         background-color: rgba(255, 255, 255, 0.1);
        #     }
        # """)
        # left_buttons_layout.addWidget(self.view_mode_button, 0, alignment=Qt.AlignmentFlag.AlignVCenter)
        
        # Add left buttons to main layout
        status_layout.addWidget(left_buttons_widget)
        
        # Calculate widths for proper centering
        # Measure counter width for balancing
        counter_placeholder = QLabel("999/999")
        counter_placeholder.setStyleSheet("""
            QLabel {
                color: #B0B0B0;
                font-size: 13px;
                font-weight: 400;
                padding: 4px 12px 4px 0px;
                letter-spacing: 0.25px;
            }
        """)
        counter_placeholder.adjustSize()
        counter_width = counter_placeholder.width()
        
        # Measure actual left buttons width for accurate centering
        # Create temporary buttons to measure their actual widths
        open_placeholder = QPushButton("🗁")  # U+1F5C1 folder icon
        open_placeholder.setStyleSheet("""
            QPushButton {
                color: #B0B0B0;
                font-size: 13px;
                font-weight: 700;
                padding: 0px;
                border: none;
                background: transparent;
            }
        """)
        open_placeholder.adjustSize()
        open_width = open_placeholder.width()
        
        view_placeholder = QPushButton("Gallery")
        view_placeholder.setStyleSheet("""
            QPushButton {
                color: #B0B0B0;
                font-size: 13px;
                font-weight: 500;
                padding: 0px;
                border: none;
                background: transparent;
            }
        """)
        view_placeholder.adjustSize()
        view_width = view_placeholder.width()
        
        # Calculate total left side width: Open button + spacing + View mode button
        # Note: sort button is hidden in single mode, so we don't include it
        left_buttons_spacing = 12  # 12px spacing between buttons (🗁 and Gallery)
        left_buttons_width = open_width + left_buttons_spacing + view_width
        
        # Left spacer - accounts for left buttons width to center metadata
        left_spacer = QWidget()
        left_spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        left_spacer.setMinimumWidth(left_buttons_width)  # Match left side width
        status_layout.addWidget(left_spacer)
        
        # Center metadata label - text center aligns with window center
        # Text should be centered, so the center of the text aligns with window center
        self.status_metadata_label = QLabel("")
        self.status_metadata_label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        self.status_metadata_label.setStyleSheet("""
            QLabel {
                color: #E0E0E0;
                font-size: 15px;
                font-weight: 500;
                padding: 6px 0px;
                letter-spacing: 0.15px;
            }
        """)
        # Add with stretch=0 so it doesn't expand, and center alignment
        # This ensures the label itself is centered, and text within label is also centered
        status_layout.addWidget(self.status_metadata_label, 0, alignment=Qt.AlignmentFlag.AlignHCenter)
        
        # Right spacer - balances left spacer and accounts for counter width
        right_spacer = QWidget()
        right_spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        right_spacer.setMinimumWidth(counter_width)  # Match counter width
        status_layout.addWidget(right_spacer)
        
        # Image counter label (right-aligned) - Material Design 3 style
        self.status_counter_label = QLabel("")
        self.status_counter_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.status_counter_label.setStyleSheet("""
            QLabel {
                color: #B0B0B0;
                font-size: 13px;
                font-weight: 400;
                padding: 4px 12px 4px 0px;
                letter-spacing: 0.25px;
            }
        """)
        status_layout.addWidget(self.status_counter_label)
        
        # Add custom widget to status bar
        self.status_bar.addPermanentWidget(status_widget, 1)
        
        # Status bar created - no rounded corners to update
        
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Install event filter to intercept arrow keys
        self.scroll_area.installEventFilter(self)
        self.image_label.installEventFilter(self)

    def create_menu_bar(self):
        """Create the menu bar with File and Keyboard Shortcuts action"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('File')

        # Open action
        open_action = QAction('Open', self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.setStatusTip('Open a RAW image file')
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        # Open Folder action
        open_folder_action = QAction('Open Folder', self)
        open_folder_action.setShortcut('Ctrl+Shift+O')
        open_folder_action.setStatusTip('Open a folder of images')
        open_folder_action.triggered.connect(self.open_folder)
        file_menu.addAction(open_folder_action)

        file_menu.addSeparator()

        # Exit action
        exit_action = QAction('Exit', self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Keyboard Shortcuts action (direct in menu bar)
        shortcuts_action = QAction('Keyboard Shortcuts', self)
        shortcuts_action.setStatusTip('Show keyboard shortcuts')
        shortcuts_action.triggered.connect(self.show_keyboard_shortcuts)
        menubar.addAction(shortcuts_action)

    def get_settings(self):
        return QSettings("RAWviewer", "RAWviewer")
    
    def get_sort_preference(self):
        """Get user's preferred sorting method - Newest (True) or Oldest (False)"""
        settings = self.get_settings()
        # Default to Newest (True) - newest images first
        return settings.value("sort_by_newest", True, type=bool)
    
    def toggle_sort_by_newest(self):
        """Toggle to sort by newest (newest images first)"""
        settings = self.get_settings()
        settings.setValue("sort_by_newest", True)
        self.resort_current_folder()
        self._update_sort_button_text()
    
    def toggle_sort_by_oldest(self):
        """Toggle to sort by oldest (oldest images first)"""
        settings = self.get_settings()
        settings.setValue("sort_by_newest", False)
        self.resort_current_folder()
        self._update_sort_button_text()
    
    def toggle_sort_method(self):
        """Toggle between sort by newest and sort by oldest (by capture time)"""
        current_pref = self.get_sort_preference()
        if current_pref:
            # Currently sorting by newest, switch to oldest
            self.toggle_sort_by_oldest()
        else:
            # Currently sorting by oldest, switch to newest
            self.toggle_sort_by_newest()
    
    # GALLERY FUNCTIONALITY COMMENTED OUT
    # def toggle_view_mode(self):
    #     """Toggle between single image view and gallery view"""
    #     import logging
    #     import time
    #     logger = logging.getLogger(__name__)
    #     start_time = time.time()
    #     logger.info(f"[VIEW_MODE] ========== toggle_view_mode() STARTED at {start_time} ==========")
    #     
    #     if self.view_mode == 'single':
    #         logger.info(f"[VIEW_MODE] Switching from single to gallery mode")
    #         switch_start = time.time()
    #         self.view_mode = 'gallery'
    #         self.view_mode_button.setText("Single")
    #         logger.info(f"[VIEW_MODE] Mode changed, calling _show_gallery_view() (elapsed: {time.time() - switch_start:.3f}s)")
    #         self._show_gallery_view()
    #         logger.info(f"[VIEW_MODE] _show_gallery_view() completed (elapsed: {time.time() - switch_start:.3f}s)")
    #     else:
    #         logger.info(f"[VIEW_MODE] Switching from gallery to single mode")
    #         switch_start = time.time()
    #         self.view_mode = 'single'
    #         self.view_mode_button.setText("Gallery")
    #         logger.info(f"[VIEW_MODE] Mode changed, calling _show_single_view() (elapsed: {time.time() - switch_start:.3f}s)")
    #         self._show_single_view()
    #         logger.info(f"[VIEW_MODE] _show_single_view() completed (elapsed: {time.time() - switch_start:.3f}s)")
    #     
    #     total_time = time.time() - start_time
    #     logger.info(f"[VIEW_MODE] ========== toggle_view_mode() COMPLETED in {total_time:.3f}s ==========")
    
    def _show_single_view(self):
        """Show single image view"""
        import logging
        import time
        import os
        logger = logging.getLogger(__name__)
        start_time = time.time()
        logger.info(f"[VIEW_MODE] ========== _show_single_view() STARTED at {start_time} ==========")
        
        # Step 1: Hide gallery widget (GALLERY FUNCTIONALITY COMMENTED OUT)
        # hide_start = time.time()
        # if self.gallery_widget:
        #     self.gallery_widget.hide()
        #     hide_time = time.time() - hide_start
        #     logger.info(f"[VIEW_MODE] Step 1: Gallery widget hidden (elapsed: {hide_time:.3f}s)")
        
        # Step 2: Show scroll area
        show_start = time.time()
        self.scroll_area.show()
        show_time = time.time() - show_start
        logger.info(f"[VIEW_MODE] Step 2: Scroll area shown (elapsed: {show_time:.3f}s)")
        
        # Step 3: Show UI elements
        ui_start = time.time()
        # Show status bar footer (metadata and image counter)
        if hasattr(self, 'status_bar'):
            self.status_bar.show()
        if hasattr(self, 'status_metadata_label'):
            self.status_metadata_label.show()
        if hasattr(self, 'status_counter_label'):
            self.status_counter_label.show()
        
        # In single view mode: hide sort button, show Gallery button (GALLERY FUNCTIONALITY COMMENTED OUT)
        if hasattr(self, 'sort_toggle_button'):
            self.sort_toggle_button.hide()
        # if hasattr(self, 'view_mode_button'):
        #     self.view_mode_button.show()
        
        ui_time = time.time() - ui_start
        logger.info(f"[VIEW_MODE] Step 3: UI elements shown (elapsed: {ui_time:.3f}s)")
        
        # Step 4: Reload current image if available
        # First try to use cached pixmap from gallery view for instant display
        if self.current_file_path:
            load_start = time.time()
            logger.info(f"[VIEW_MODE] Step 4: Starting image reload: {os.path.basename(self.current_file_path)}")
            
            # Try to use cached pixmap from gallery or image cache for instant display
            cached_pixmap = None
            try:
                # GALLERY FUNCTIONALITY COMMENTED OUT
                # First check gallery cache (might have embedded preview)
                # if hasattr(self, 'gallery_pixmaps') and self.current_file_path in self.gallery_pixmaps:
                #     cached_pixmap = self.gallery_pixmaps[self.current_file_path]
                #     if cached_pixmap and not cached_pixmap.isNull():
                #         logger.info(f"[VIEW_MODE] Using cached pixmap from gallery for instant display")
                #         # Display cached pixmap immediately for smooth transition
                #         self.display_pixmap(cached_pixmap)
                #         # Also ensure it's in global cache
                #         try:
                #             self.image_cache.put_pixmap(self.current_file_path, cached_pixmap)
                #         except:
                #             pass
                # Also check global image cache
                if (not cached_pixmap or cached_pixmap.isNull()) and hasattr(self, 'image_cache'):
                    cached_pixmap = self.image_cache.get_pixmap(self.current_file_path)
                    if cached_pixmap and not cached_pixmap.isNull():
                        logger.info(f"[VIEW_MODE] Using cached pixmap from image cache for instant display")
                        self.display_pixmap(cached_pixmap)
            except Exception as e:
                logger.debug(f"[VIEW_MODE] Error using cached pixmap: {e}")
            
            # Only load full image if we don't have a good cached version
            # For RAW files, we might want to load full resolution, but for non-RAW, cached is fine
            if not cached_pixmap or cached_pixmap.isNull():
                # Load full image in background (will replace cached preview if different)
                # This allows smooth transition: cached preview shows immediately, full image loads in background
                self.load_raw_image(self.current_file_path)
            else:
                # We have a cached version, but for RAW files we might want to load full resolution
                # Check if it's a RAW file and if we should load full resolution
                import os
                file_ext = os.path.splitext(self.current_file_path)[1].lower()
                raw_extensions = ['.arw', '.cr2', '.nef', '.raf', '.orf', '.dng', '.cr3', '.rw2', '.rwl', '.srw']
                if file_ext in raw_extensions:
                    # For RAW files, load full resolution in background (cached preview is just embedded JPEG)
                    self.load_raw_image(self.current_file_path)
                else:
                    # For non-RAW files, cached version is good enough
                    logger.info(f"[VIEW_MODE] Using cached pixmap, skipping reload for non-RAW file")
            load_time = time.time() - load_start
            logger.info(f"[VIEW_MODE] Step 4: Image reload completed (elapsed: {load_time:.3f}s)")
        else:
            # Update status bar to show metadata even if no image is loaded
            self.update_status_bar()
            logger.info(f"[VIEW_MODE] Step 4: No image to reload, status bar updated")
        
        total_time = time.time() - start_time
        logger.info(f"[VIEW_MODE] ========== TIMING BREAKDOWN ==========")
        # GALLERY FUNCTIONALITY COMMENTED OUT
        # if self.gallery_widget:
        #     logger.info(f"[VIEW_MODE] Hide gallery widget: {hide_time:.3f}s")
        logger.info(f"[VIEW_MODE] Show scroll area: {show_time:.3f}s")
        logger.info(f"[VIEW_MODE] Show UI elements: {ui_time:.3f}s")
        if self.current_file_path:
            logger.info(f"[VIEW_MODE] Image reload: {load_time:.3f}s")
        logger.info(f"[VIEW_MODE] ========== SINGLE VIEW RENDERING COMPLETED in {total_time:.3f}s ==========")
    
    # GALLERY FUNCTIONALITY COMMENTED OUT
    if False:  # Gallery functionality disabled
        def _show_gallery_view(self):
            """Show gallery view - based on reference code"""
        import logging
        from PyQt6.QtCore import QTimer
        logger = logging.getLogger(__name__)
        logger.info(f"[GALLERY] Showing gallery view")
        
        # Create gallery widget if needed
        if not self.gallery_widget:
            self._create_gallery_widget()
        
        # Hide single view elements
        self.scroll_area.hide()
        
        # Hide view mode button in gallery mode (it shows "Single" which is confusing)
        if hasattr(self, 'view_mode_button'):
            self.view_mode_button.hide()
        
        # In gallery mode: hide metadata and counter, but keep status_bar visible for sort button
        if hasattr(self, 'status_bar'):
            self.status_bar.show()  # Keep status_bar visible to show sort button
            self.status_bar.showMessage("")  # Clear message
        # Hide metadata and counter labels
        if hasattr(self, 'status_metadata_label'):
            self.status_metadata_label.hide()
        if hasattr(self, 'status_counter_label'):
            self.status_counter_label.hide()
        # Show sort button in gallery mode
        if hasattr(self, 'sort_toggle_button'):
            self.sort_toggle_button.show()
        
        # Show gallery
        self.gallery_widget.show()
        self.gallery_widget.raise_()
        
        # Update gallery content
        QTimer.singleShot(50, self._update_gallery_view)
    
        def _create_gallery_widget(self):
            """Create the gallery view widget - based on JustifiedGallery reference code"""
            from PyQt6.QtWidgets import QWidget, QVBoxLayout, QScrollArea
            
            # Create gallery widget container
            gallery_container = QWidget()
            gallery_container.setStyleSheet("""
                QWidget {
                    background-color: #1E1E1E;
                }
            """)
            gallery_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            gallery_layout = QVBoxLayout(gallery_container)
            gallery_layout.setContentsMargins(0, 0, 0, 0)
            gallery_layout.setSpacing(0)
            
            # Create scroll area for gallery
            gallery_scroll = QScrollArea()
            gallery_scroll.setWidgetResizable(True)
            gallery_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            gallery_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            gallery_scroll.setStyleSheet("""
                QScrollArea {
                    border: none;
                    background-color: #1E1E1E;
                }
                QScrollBar:vertical {
                    background: transparent;
                    width: 12px;
                    margin: 0px;
                    border: none;
                }
                QScrollBar::handle:vertical {
                    background: rgba(255, 255, 255, 0.2);
                    min-height: 30px;
                    border-radius: 6px;
                    margin: 2px;
                }
                QScrollBar::handle:vertical:hover {
                    background: rgba(255, 255, 255, 0.3);
                }
                QScrollBar::handle:vertical:pressed {
                    background: rgba(255, 255, 255, 0.4);
                }
                QScrollBar::add-line:vertical,
                QScrollBar::sub-line:vertical {
                    height: 0px;
                    width: 0px;
                }
                QScrollBar::add-page:vertical,
                QScrollBar::sub-page:vertical {
                    background: transparent;
                }
            """)
            
            # Create JustifiedGallery widget
            justified_gallery = JustifiedGallery([], self)  # Empty list initially, will be populated
            gallery_scroll.setWidget(justified_gallery)
            gallery_layout.addWidget(gallery_scroll)
            
            # Insert gallery widget into main layout
            main_layout = self.centralWidget().layout()
            scroll_index = main_layout.indexOf(self.scroll_area)
            main_layout.insertWidget(scroll_index + 1, gallery_container)
            
            self.gallery_widget = gallery_container
            self.gallery_scroll = gallery_scroll
            self.gallery_justified = justified_gallery
        
        def _update_gallery_view(self):
            """Update gallery view - using JustifiedGallery"""
            import logging
            import time
            logger = logging.getLogger(__name__)
            start_time = time.time()
            logger.info(f"[GALLERY] ========== _update_gallery_view() STARTED ==========")
            
            if not self.gallery_widget or not self.gallery_justified or not self.image_files:
                logger.info(f"[GALLERY] Gallery widget or image files not available, returning")
                return
            
            # Update JustifiedGallery with file paths
            # The gallery will load pixmaps as needed
            self.gallery_justified.set_images(self.image_files)
            
            total_time = time.time() - start_time
            logger.info(f"[GALLERY] ========== GALLERY LAYOUT COMPLETED in {total_time:.3f}s ==========")
        
        def _add_gallery_row(self, row_items, available_width, content_width, row_height, row_spacing):
            """Add a single row - based on reference code add_row method"""
            from PyQt6.QtWidgets import QWidget, QHBoxLayout
            
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(0)
            
            # Remaining space after placing all thumbnails
            free_space = max(0, available_width - content_width)
            
            # Spacing distributed evenly between items
            gaps = len(row_items) - 1 if len(row_items) > 1 else 1
            # Limit extra padding to prevent excessive spacing
            # Use minimum of calculated padding or a maximum value (e.g., 8px)
            extra_padding = min(free_space // gaps, 8) if gaps > 0 else 0
            
            for i, (file_path, w) in enumerate(row_items):
                # Create thumbnail label (will be loaded async)
                thumb_label = ThumbnailLabel()
                thumb_label.file_path = file_path
                
                if not hasattr(self, '_gallery_thumb_labels'):
                    self._gallery_thumb_labels = {}
                self._gallery_thumb_labels[file_path] = thumb_label
                
                # Make clickable
                thumb_label.mousePressEvent = lambda e, fp=file_path: self._gallery_item_clicked(fp)
                
                # Load pixmap and scale it
                pixmap = self._get_gallery_pixmap(file_path)
                if pixmap and not pixmap.isNull():
                    # Scale to fixed height while preserving aspect ratio (like reference)
                    resized = pixmap.scaled(
                        QSize(w, row_height),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    thumb_label.setFixedSize(resized.size())  # CRITICAL: Like reference code
                    thumb_label.setPixmap(resized)
                    thumb_label.set_original_pixmap(pixmap)
                else:
                    # Placeholder size, will be updated when loaded
                    thumb_label.setFixedSize(w, row_height)
                    # Load asynchronously
                    from PyQt6.QtCore import QTimer
                    QTimer.singleShot(50, lambda fp=file_path, tl=thumb_label: self._load_gallery_thumbnail_simple(fp, tl))
                
                row_layout.addWidget(thumb_label)
                
                # Apply padding after each item (except last) - like reference code
                if i < len(row_items) - 1:
                    row_layout.addSpacing(row_spacing + extra_padding)
            
            self.gallery_content_layout.addWidget(row_widget)
        
        def _load_gallery_thumbnail_simple(self, file_path, thumb_label):
            """Load thumbnail simply - based on reference code"""
            from PyQt6.QtWidgets import QApplication
            from PyQt6.QtCore import QTimer
            QApplication.processEvents()
            
            pixmap = self._get_gallery_pixmap(file_path)
            if not pixmap or pixmap.isNull():
                # Try async load
                self._load_gallery_pixmap_async(file_path)
                return
            
            # Get aspect ratio
            aspect = pixmap.width() / pixmap.height() if pixmap.height() > 0 else 4.0 / 3.0
            thumb_width = int(self.gallery_row_height * aspect)
            
            # Scale to fixed height while preserving aspect ratio (like reference)
            resized = pixmap.scaled(
                QSize(thumb_width, self.gallery_row_height),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            # CRITICAL: Like reference code - thumb.setFixedSize(resized.size())
            thumb_label.setFixedSize(resized.size())
            thumb_label.setPixmap(resized)
            thumb_label.set_original_pixmap(pixmap)
            
            if file_path in self._gallery_load_tracking:
                self._gallery_load_tracking[file_path]['loaded'] = True
                self._gallery_load_tracking[file_path]['displayed'] = True
        
        def _get_gallery_aspect_ratio(self, file_path):
            """Get aspect ratio for gallery layout - optimized to avoid loading full pixmaps"""
            # Check cache first (fastest)
            if file_path in self.gallery_aspect_cache:
                return self.gallery_aspect_cache[file_path]
            
            # Try to get from EXIF cache (fast, no file I/O)
            try:
                exif_data = self.image_cache.get_exif(file_path)
                if exif_data:
                    original_width = exif_data.get('original_width')
                    original_height = exif_data.get('original_height')
                    if original_width and original_height and original_height > 0:
                        aspect = original_width / original_height
                        self.gallery_aspect_cache[file_path] = aspect
                        return aspect
            except:
                pass
            
            # Try to extract from EXIF tags directly (for files not yet in cache)
            try:
                import exifread
                with open(file_path, 'rb') as f:
                    tags = exifread.process_file(f, details=False)
                    if 'EXIF ExifImageWidth' in tags and 'EXIF ExifImageLength' in tags:
                        original_width = int(str(tags['EXIF ExifImageWidth']))
                        original_height = int(str(tags['EXIF ExifImageLength']))
                        if original_width and original_height and original_height > 0:
                            aspect = original_width / original_height
                            self.gallery_aspect_cache[file_path] = aspect
                            # Also update EXIF cache for future use
                            try:
                                cached_exif = self.image_cache.get_exif(file_path) or {}
                                cached_exif['original_width'] = original_width
                                cached_exif['original_height'] = original_height
                                self.image_cache.put_exif(file_path, cached_exif)
                            except:
                                pass
                            return aspect
            except:
                pass
            
            # For non-RAW files, try to get dimensions without loading full image
            # For TIFF files, use PIL to avoid Qt warnings
            import os
            file_ext = os.path.splitext(file_path)[1].lower()
            is_tiff = file_ext in ('.tiff', '.tif')
            
            # Also check file content to detect TIFF files with wrong extension
            if not is_tiff:
                try:
                    from PIL import Image
                    with Image.open(file_path) as test_img:
                        if test_img.format in ('TIFF', 'TIF'):
                            is_tiff = True
                except:
                    pass  # Not a PIL-readable file or not TIFF
            
            if is_tiff:
                # Use PIL for TIFF files to avoid Qt TIFF plugin warnings
                try:
                    from PIL import Image
                    with Image.open(file_path) as img:
                        width, height = img.size
                        if height > 0:
                            aspect = width / height
                            self.gallery_aspect_cache[file_path] = aspect
                            return aspect
                except:
                    pass
            
            # For other formats, try QImageReader (but not for TIFF to avoid warnings)
            try:
                if file_ext not in ['.arw', '.cr2', '.nef', '.raf', '.orf', '.dng', '.cr3', '.rw2', '.rwl', '.srw'] and not is_tiff:
                    # Try to read just the image size without loading full image
                    from PyQt6.QtGui import QImageReader
                    reader = QImageReader(file_path)
                    size = reader.size()
                    if size.isValid() and size.height() > 0:
                        aspect = size.width() / size.height()
                        self.gallery_aspect_cache[file_path] = aspect
                        # Also update EXIF cache for future use
                        try:
                            cached_exif = self.image_cache.get_exif(file_path) or {}
                            cached_exif['original_width'] = size.width()
                            cached_exif['original_height'] = size.height()
                            self.image_cache.put_exif(file_path, cached_exif)
                        except:
                            pass
                        return aspect
            except:
                pass
            
            # CRITICAL OPTIMIZATION: Don't load pixmaps during layout calculation!
            # Only check already-loaded pixmaps (if any), but don't trigger new loads
            # This prevents blocking on file I/O during layout
            if file_path in self.gallery_pixmaps:
                pixmap = self.gallery_pixmaps[file_path]
                if pixmap and not pixmap.isNull():
                    aspect = pixmap.width() / pixmap.height() if pixmap.height() > 0 else 4.0 / 3.0
                    self.gallery_aspect_cache[file_path] = aspect
                    return aspect
            
            # Fallback: use default aspect ratio (4:3) only as last resort
            # This allows layout to proceed instantly without blocking on file I/O
            # Images will be loaded asynchronously later and layout can be refined if needed
            default_aspect = 4.0 / 3.0
            self.gallery_aspect_cache[file_path] = default_aspect
            return default_aspect
    
    def _numpy_to_qpixmap(self, numpy_array):
        """
        Convert numpy array to QPixmap safely.
        Handles all edge cases and ensures correct format.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            if numpy_array is None:
                return None
            
            # Get shape
            if len(numpy_array.shape) != 3:
                logger.warning(f"[GALLERY] Invalid numpy array shape: {numpy_array.shape}, expected 3D (H, W, C)")
                return None
            
            height, width, channels = numpy_array.shape
            
            if channels != 3:
                logger.warning(f"[GALLERY] Invalid channel count: {channels}, expected 3 (RGB)")
                return None
            
            if width <= 0 or height <= 0:
                logger.warning(f"[GALLERY] Invalid dimensions: {width}x{height}")
                return None
            
            # Ensure contiguous and uint8
            if not numpy_array.flags['C_CONTIGUOUS']:
                numpy_array = np.ascontiguousarray(numpy_array)
            
            if numpy_array.dtype != np.uint8:
                numpy_array = numpy_array.astype(np.uint8)
            
            # Calculate bytes per line (must be exact, no padding)
            bytes_per_line = channels * width
            
            # Convert to bytes
            image_data = numpy_array.tobytes()
            
            # Create QImage
            qimage = QImage(image_data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            
            if qimage.isNull():
                logger.warning(f"[GALLERY] Failed to create QImage from numpy array {width}x{height}")
                return None
            
            # Convert to QPixmap
            pixmap = QPixmap.fromImage(qimage)
            
            if pixmap.isNull():
                logger.warning(f"[GALLERY] Failed to create QPixmap from QImage")
                return None
            
            return pixmap
            
        except Exception as e:
            logger.warning(f"[GALLERY] Error converting numpy to QPixmap: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _extract_embedded_preview(self, file_path):
        """Extract embedded JPEG preview from RAW file for gallery display"""
        import os
        import logging
        import io
        logger = logging.getLogger(__name__)
        
        try:
            with rawpy.imread(file_path) as raw:
                thumb = raw.extract_thumb()
                
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    from PIL import Image
                    pil_image = Image.open(io.BytesIO(thumb.data))
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    
                    # Convert PIL Image to QPixmap
                    from PyQt6.QtGui import QImage, QPixmap
                    width, height = pil_image.size
                    image_bytes = pil_image.tobytes('raw', 'RGB')
                    bytes_per_line = 3 * width  # RGB = 3 channels
                    qimage = QImage(image_bytes, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                    
                    if not qimage.isNull():
                        pixmap = QPixmap.fromImage(qimage)
                        logger.debug(f"[GALLERY] Extracted embedded JPEG preview from RAW: {os.path.basename(file_path)}, size: {width}x{height}")
                        return pixmap
                
                # Fallback: decode RAW (expensive, but better than nothing)
                # Only use this if embedded JPEG is not available
                logger.debug(f"[GALLERY] No embedded JPEG found, decoding RAW for thumbnail: {os.path.basename(file_path)}")
                rgb = raw.postprocess(
                    half_size=True,  # Use half size for speed
                    output_bps=8,
                    use_camera_wb=True,
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR
                )
                
                # Convert numpy array to QPixmap
                from PyQt6.QtGui import QImage, QPixmap
                height, width, channels = rgb.shape
                bytes_per_line = channels * width
                qimage = QImage(rgb.data.tobytes(), width, height, bytes_per_line, QImage.Format.Format_RGB888)
                
                if not qimage.isNull():
                    pixmap = QPixmap.fromImage(qimage)
                    logger.debug(f"[GALLERY] Decoded RAW for thumbnail: {os.path.basename(file_path)}, size: {width}x{height}")
                    return pixmap
                    
        except Exception as e:
            logger.debug(f"[GALLERY] Failed to extract embedded preview from RAW: {os.path.basename(file_path)}: {e}")
        
        return None
    
    # GALLERY FUNCTIONALITY COMMENTED OUT
    # def _get_gallery_pixmap(self, file_path):
    #     """Get pixmap for gallery view, loading if necessary - optimized for performance"""
    #     pass  # Gallery functionality disabled
        import os
        import logging
        logger = logging.getLogger(__name__)
        
        # Check cache first (fastest)
        if file_path in self.gallery_pixmaps:
            pixmap = self.gallery_pixmaps[file_path]
            if pixmap and not pixmap.isNull():
                return pixmap
        
        # Try to load from image cache (fast, already processed)
        # This includes pixmaps from single view mode, enabling smooth switching
        try:
            cached_pixmap = self.image_cache.get_pixmap(file_path)
            if cached_pixmap and not cached_pixmap.isNull():
                # Also store in gallery_pixmaps for faster subsequent access
                self.gallery_pixmaps[file_path] = cached_pixmap
                # Update aspect cache
                aspect = cached_pixmap.width() / cached_pixmap.height() if cached_pixmap.height() > 0 else 4.0 / 3.0
                self.gallery_aspect_cache[file_path] = aspect
                logger.debug(f"[GALLERY] Using cached pixmap from single view: {os.path.basename(file_path)}")
                return cached_pixmap
        except Exception as e:
            logger.debug(f"Error getting pixmap from cache for {file_path}: {e}")
        
        # Try to get thumbnail from thumbnail cache (faster than extracting)
        try:
            thumbnail_data = self.image_cache.get_thumbnail(file_path)
            if thumbnail_data is not None:
                # Use unified conversion method
                    pixmap = self._numpy_to_qpixmap(thumbnail_data)
                    if pixmap and not pixmap.isNull():
                        self.gallery_pixmaps[file_path] = pixmap
                        # Also cache in global image cache for smooth switching between views
                        try:
                            self.image_cache.put_pixmap(file_path, pixmap)
                        except:
                            pass  # Cache might be full, continue anyway
                        # Update aspect cache
                        aspect = pixmap.width() / pixmap.height() if pixmap.height() > 0 else 4.0 / 3.0
                        self.gallery_aspect_cache[file_path] = aspect
                        return pixmap
        except Exception as e:
            logger.warning(f"Error getting thumbnail from cache for {file_path}: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
        
        # For RAW files, try to extract embedded JPEG preview (fast)
        try:
            if os.path.exists(file_path):
                file_ext = os.path.splitext(file_path)[1].lower()
                raw_extensions = ['.arw', '.cr2', '.nef', '.raf', '.orf', '.dng', '.cr3', '.rw2', '.rwl', '.srw']
                if file_ext in raw_extensions:
                    pixmap = self._extract_embedded_preview(file_path)
                    if pixmap and not pixmap.isNull():
                        self.gallery_pixmaps[file_path] = pixmap
                        # Also cache in global image cache for smooth switching between views
                        try:
                            self.image_cache.put_pixmap(file_path, pixmap)
                        except:
                            pass  # Cache might be full, continue anyway
                        # Update aspect cache
                        aspect = pixmap.width() / pixmap.height() if pixmap.height() > 0 else 4.0 / 3.0
                        self.gallery_aspect_cache[file_path] = aspect
                        logger.debug(f"[GALLERY] Using embedded preview for gallery: {os.path.basename(file_path)}")
                        return pixmap
        except Exception as e:
            logger.debug(f"Error extracting embedded preview from RAW {file_path}: {e}")
        
        # For non-RAW files, try direct QPixmap load (fast for JPEG/PNG)
        try:
            if os.path.exists(file_path):
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext not in ['.arw', '.cr2', '.nef', '.raf', '.orf', '.dng', '.cr3', '.rw2', '.rwl', '.srw']:
                    pixmap = self._load_pixmap_safe(file_path)
                    if not pixmap.isNull():
                        self.gallery_pixmaps[file_path] = pixmap
                        # Also cache in global image cache for smooth switching between views
                        try:
                            self.image_cache.put_pixmap(file_path, pixmap)
                        except:
                            pass  # Cache might be full, continue anyway
                        # Update aspect cache
                        aspect = pixmap.width() / pixmap.height() if pixmap.height() > 0 else 4.0 / 3.0
                        self.gallery_aspect_cache[file_path] = aspect
                        return pixmap
        except Exception as e:
            logger.debug(f"Error loading pixmap from file {file_path}: {e}")
        
        # Return None - will be loaded asynchronously
        # This allows layout to proceed without blocking
        return None
    
    # GALLERY FUNCTIONALITY COMMENTED OUT
    # def _add_grid_row(self, row_files, available_width, row_height, item_spacing, row_index=0, start_index=0):
    #     """Create a simple grid row with fixed height - basic gallery layout"""
    #     pass  # Gallery functionality disabled
        
        # Create row widget with fixed height
        row_widget = QWidget()
        row_widget.setFixedHeight(row_height)
        row_widget.setFixedWidth(available_width)
        row_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        row_widget.setStyleSheet("""
            QWidget {
                background-color: #1E1E1E;
                margin: 0px;
                padding: 0px;
            }
        """)
        
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(item_spacing)
        
        # Calculate item width (distribute available space evenly)
        num_items = len(row_files)
        total_spacing = item_spacing * (num_items - 1) if num_items > 1 else 0
        available_for_items = available_width - total_spacing
        item_width = int(available_for_items / num_items) if num_items > 0 else available_width
        
        for item_index, file_path in enumerate(row_files):
            # Create thumbnail label
            thumb_label = ThumbnailLabel()
            thumb_label.setFixedSize(item_width, row_height)
            thumb_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            thumb_label.setStyleSheet("""
                QLabel {
                    background-color: #2A2A2A;
                    border: none;
                    margin: 0px;
                    padding: 0px;
                }
            """)
            thumb_label.file_path = file_path
            
            # Store reference
            if not hasattr(self, '_gallery_thumb_labels'):
                self._gallery_thumb_labels = {}
            self._gallery_thumb_labels[file_path] = thumb_label
            
            # Make clickable
            thumb_label.mousePressEvent = lambda e, fp=file_path: self._gallery_item_clicked(fp)
            
            # Load thumbnail asynchronously with staggered delay to avoid blocking UI
            # Each item gets a slightly longer delay to spread out the loading
            global_index = start_index + item_index
            delay = 10 + (global_index * 5)  # Stagger delays: 10ms, 15ms, 20ms, etc.
            self._load_gallery_thumbnail_async_justified(file_path, thumb_label, item_width, row_height, delay)
            
            row_layout.addWidget(thumb_label)
        
        # Add row to gallery
        self.gallery_content_layout.addWidget(row_widget)
    
    # GALLERY FUNCTIONALITY COMMENTED OUT
    # def _add_justified_row(self, row_items, row_width, available_width, row_height, min_spacing, row_index=0):
    #     """Create a row of thumbnails with equal horizontal spacing between images"""
    #     pass  # Gallery functionality disabled
        # Calculate total width needed for all images at base scale
        total_images_width = sum(base_width for _, base_width in row_items)
        
        if total_images_width <= 0 or len(row_items) == 0:
            return
        
        # Calculate number of gaps between images (n-1 gaps for n images)
        num_gaps = len(row_items) - 1
        
        # Calculate equal spacing between images
        # Formula: available_width = scaled_total_width + (num_gaps * spacing_between)
        # We want to ensure spacing is at least min_spacing, then scale images to fit
        if num_gaps > 0:
            # Calculate maximum spacing we can use
            max_spacing = (available_width - total_images_width) / num_gaps if num_gaps > 0 else 0
            
            # Use the larger of min_spacing or calculated spacing
            # If we have enough space, use min_spacing and scale images up
            # If we don't have enough space, use calculated spacing (which may be less than min_spacing)
            if max_spacing >= min_spacing:
                # We have enough space, use min_spacing and scale images to fill remaining space
                total_spacing_width = min_spacing * num_gaps
                available_for_images = available_width - total_spacing_width
                scale_factor = available_for_images / total_images_width if total_images_width > 0 else 1.0
                spacing_between = min_spacing
            else:
                # Not enough space, use all available space for spacing and scale images down
                scale_factor = 1.0
                spacing_between = max_spacing if max_spacing > 0 else 0
        else:
            # Only one image, scale it to fit available width
            scale_factor = available_width / total_images_width if total_images_width > 0 else 1.0
            spacing_between = 0
        
        # Calculate row height: thumbnail height only (no histogram)
        # All items in the row should have the same total height
        first_item_height = int(row_height * scale_factor) if row_items else row_height
        total_row_height = first_item_height
        
        # Create row widget with fixed height and width to ensure no vertical spacing and no horizontal scroll
        row_widget = QWidget()
        row_widget.setFixedHeight(total_row_height)  # CRITICAL: Set fixed height to prevent spacing
        row_widget.setFixedWidth(available_width)  # CRITICAL: Set fixed width to prevent horizontal scroll
        row_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)  # Prevent resizing
        row_widget.setStyleSheet("""
            QWidget {
                background-color: #1E1E1E;
                margin: 0px;
                padding: 0px;
            }
        """)
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)  # No margins
        row_layout.setSpacing(0)  # No spacing between items (we add spacers manually)
        
        for i, (file_path, base_width) in enumerate(row_items):
            # CRITICAL OPTIMIZATION: Don't load pixmap here during layout!
            # This is called for every image during layout calculation
            # Loading images synchronously here would block for 30+ seconds
            # Instead, create placeholder and load asynchronously
            
            # Compute final scaled size (maintain aspect ratio, no stretching)
            new_width = int(base_width * scale_factor)
            new_height = int(row_height * scale_factor)
            
            # Create thumbnail label
            thumb_label = ThumbnailLabel()
            # Set fixed size - this is the target size for the thumbnail
            thumb_label.setFixedSize(new_width, new_height)
            # Ensure size policy is Fixed (already set in ThumbnailLabel.__init__, but double-check)
            thumb_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            thumb_label.setStyleSheet("""
                QLabel {
                    background-color: #2A2A2A;
                    border: none;
                    margin: 0px;
                    padding: 0px;
                }
            """)
            # setScaledContents(False) is set in ThumbnailLabel.__init__
            # We will manually scale pixmap to exact size to prevent stretching
            
            # Store file path for async loading
            thumb_label.file_path = file_path
            # Store reference to thumb_label for later updates
            if not hasattr(self, '_gallery_thumb_labels'):
                self._gallery_thumb_labels = {}
            self._gallery_thumb_labels[file_path] = thumb_label
            
            # Make clickable
            thumb_label.mousePressEvent = lambda e, fp=file_path: self._gallery_item_clicked(fp)
            
            # Create item widget - set fixed height to match row
            item_widget = QWidget()
            item_widget.setFixedHeight(total_row_height)  # CRITICAL: Match row height exactly
            item_widget.setStyleSheet("""
                QWidget {
                    background-color: #1E1E1E;
                    margin: 0px;
                    padding: 0px;
                }
            """)
            item_layout = QVBoxLayout(item_widget)
            item_layout.setContentsMargins(0, 0, 0, 0)  # No margins
            item_layout.setSpacing(0)  # No spacing
            # CRITICAL: Align label to center to prevent stretching
            item_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            item_layout.addWidget(thumb_label, alignment=Qt.AlignmentFlag.AlignCenter)
            
            # Load thumbnail asynchronously
            self._load_gallery_thumbnail_async_justified(file_path, thumb_label, new_width, new_height)
            
            row_layout.addWidget(item_widget)
            
            # Add equal spacing after each image (except the last one)
            if i < len(row_items) - 1 and spacing_between > 0:
                spacer = QWidget()
                spacer.setFixedWidth(int(spacing_between))
                spacer.setFixedHeight(total_row_height)  # Match row height
                spacer.setStyleSheet("background-color: transparent; margin: 0px; padding: 0px;")
                row_layout.addWidget(spacer)
        
        # Add row widget with no spacing (vertical spacing is already 0 in gallery_content_layout)
        self.gallery_content_layout.addWidget(row_widget)
        # Store row widget reference
        self._gallery_row_widgets[row_index] = row_widget
        
        # Ensure no additional spacing is added
        self.gallery_content_layout.setSpacing(0)
        self.gallery_content_layout.setContentsMargins(0, 0, 0, 0)
    
    # GALLERY FUNCTIONALITY COMMENTED OUT
    # def _load_gallery_thumbnail_async_justified(self, file_path, thumb_label, target_width, target_height, delay=10):
    #     """Load thumbnail for justified gallery item asynchronously"""
    #     pass  # Gallery functionality disabled
        import time
        import logging
        import os
        from PyQt6.QtCore import QTimer
        logger = logging.getLogger(__name__)
        
        # Validate inputs
        if not file_path or not thumb_label:
            logger.warning(f"[GALLERY] Invalid inputs for thumbnail load: file_path={file_path}, thumb_label={thumb_label}")
            return
        
        # Track loading start time
        if file_path in self._gallery_load_tracking:
            self._gallery_load_tracking[file_path]['start_time'] = time.time()
        else:
            logger.warning(f"[GALLERY] File {os.path.basename(file_path)} not in load tracking")
        
        # Store target dimensions in local variables to ensure they're available in closure
        final_target_width = target_width
        final_target_height = target_height
        
        def load_thumbnail():
            # Process events at start to keep UI responsive
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()
            load_start = time.time()
            # Use local variables for target dimensions
            local_target_width = final_target_width
            local_target_height = final_target_height
            
            try:
                logger.debug(f"[GALLERY] Starting load for {os.path.basename(file_path)}")
                # Get or load pixmap
                pixmap = self._get_gallery_pixmap(file_path)
                if not pixmap or pixmap.isNull():
                    # Try to load asynchronously if not in cache
                    logger.debug(f"[GALLERY] Pixmap not in cache for {os.path.basename(file_path)}, loading async")
                    if file_path in self._gallery_load_tracking:
                        self._gallery_load_tracking[file_path]['loaded'] = False
                    # Load pixmap asynchronously - it will update the label when ready
                    self._load_gallery_pixmap_async(file_path)
                    # Return early - label will be updated by _load_gallery_pixmap_async
                    return
                
                # Mark as loaded
                load_time = time.time() - load_start
                if file_path in self._gallery_load_tracking:
                    self._gallery_load_tracking[file_path]['loaded'] = True
                    self._gallery_load_tracking[file_path]['load_time'] = load_time
                logger.debug(f"[GALLERY] [PIXMAP] {os.path.basename(file_path)} - Loaded from cache in {load_time:.3f}s")
                
                # Get actual image dimensions
                pixmap_width = pixmap.width()
                pixmap_height = pixmap.height()
                if pixmap_width <= 0 or pixmap_height <= 0:
                    logger.warning(f"[GALLERY] Invalid pixmap size for {os.path.basename(file_path)}: {pixmap_width}x{pixmap_height}")
                    return
                
                # Update aspect cache with actual image dimensions
                actual_aspect = pixmap_width / pixmap_height
                self.gallery_aspect_cache[file_path] = actual_aspect
                
                # Log original image dimensions and aspect ratio
                logger.info(f"[GALLERY] [THUMBNAIL] {os.path.basename(file_path)} - Original: {pixmap_width}x{pixmap_height}, Aspect: {actual_aspect:.3f}")
                
                # For fixed height gallery: scale to fixed height while preserving aspect ratio
                # Reference code: resized = pixmap.scaled(QSize(w, ROW_HEIGHT), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                # We use the target_height as the fixed constraint (this is the row_height from layout)
                target_height = local_target_height if local_target_height > 0 else self.gallery_row_height
                
                # Get label size for logging
                label_width = thumb_label.width()
                label_height = thumb_label.height()
                label_aspect = label_width / label_height if label_height > 0 else 0
                logger.info(f"[GALLERY] [THUMBNAIL] {os.path.basename(file_path)} - Label: {label_width}x{label_height}, Label Aspect: {label_aspect:.3f}")
                
                # Calculate width based on actual image aspect ratio and fixed height
                scaled_width = int(target_height * actual_aspect)
                scaled_height = target_height
                scaled_aspect = scaled_width / scaled_height if scaled_height > 0 else 0
                logger.info(f"[GALLERY] [THUMBNAIL] {os.path.basename(file_path)} - Scaled Target: {scaled_width}x{scaled_height}, Scaled Aspect: {scaled_aspect:.3f}")
                
                # Scale pixmap to fixed height while preserving aspect ratio
                scaled_pixmap = pixmap.scaled(
                    QSize(scaled_width, scaled_height),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                
                if scaled_pixmap.isNull():
                    logger.warning(f"[GALLERY] Failed to scale pixmap for {os.path.basename(file_path)}")
                    return
                
                # Log actual scaled pixmap dimensions
                actual_scaled_width = scaled_pixmap.width()
                actual_scaled_height = scaled_pixmap.height()
                actual_scaled_aspect = actual_scaled_width / actual_scaled_height if actual_scaled_height > 0 else 0
                logger.info(f"[GALLERY] [THUMBNAIL] {os.path.basename(file_path)} - Scaled Result: {actual_scaled_width}x{actual_scaled_height}, Result Aspect: {actual_scaled_aspect:.3f}")
                
                # CRITICAL: Resize label to match scaled pixmap size (like reference code: thumb.setFixedSize(resized.size()))
                # This ensures the label matches the pixmap dimensions exactly, preventing compression
                # The label size must match the pixmap size to avoid stretching/compression
                display_start = time.time()
                thumb_label.setFixedSize(actual_scaled_width, actual_scaled_height)
                thumb_label.setPixmap(scaled_pixmap)
                thumb_label.set_original_pixmap(pixmap)
                
                # Log final state
                final_label_width = thumb_label.width()
                final_label_height = thumb_label.height()
                final_pixmap_width = scaled_pixmap.width()
                final_pixmap_height = scaled_pixmap.height()
                logger.info(f"[GALLERY] [THUMBNAIL] {os.path.basename(file_path)} - Final Label: {final_label_width}x{final_label_height}, Pixmap: {final_pixmap_width}x{final_pixmap_height}")
                display_time = time.time() - display_start
                
                # Mark as displayed
                if file_path in self._gallery_load_tracking:
                    self._gallery_load_tracking[file_path]['displayed'] = True
                    self._gallery_load_tracking[file_path]['display_time'] = display_time
                    total_time = time.time() - self._gallery_load_tracking[file_path]['start_time']
                    logger.info(f"[GALLERY] [IMAGE] {os.path.basename(file_path)} - Loaded in {load_time:.3f}s, Displayed in {display_time:.3f}s, Total: {total_time:.3f}s")
            except Exception as e:
                logger.warning(f"Error loading justified thumbnail for {os.path.basename(file_path)}: {e}")
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")
                if file_path in self._gallery_load_tracking:
                    self._gallery_load_tracking[file_path]['loaded'] = False
                    self._gallery_load_tracking[file_path]['displayed'] = False
        
        # Use a small delay to allow UI to update first
        QTimer.singleShot(10, load_thumbnail)
    
    # GALLERY FUNCTIONALITY COMMENTED OUT
    # def _load_gallery_pixmap_async(self, file_path):
    #     """Load pixmap asynchronously and update specific thumbnail when ready"""
    #     pass  # Gallery functionality disabled
        import time
        import logging
        from PyQt6.QtCore import QTimer
        from PIL import Image
        logger = logging.getLogger(__name__)
        
        # Track loading start time if not already tracked
        if file_path not in self._gallery_load_tracking:
            self._gallery_load_tracking[file_path] = {
                'start_time': time.time(),
                'loaded': False,
                'displayed': False,
                'load_time': None,
                'display_time': None
            }
        elif self._gallery_load_tracking[file_path]['start_time'] is None:
            self._gallery_load_tracking[file_path]['start_time'] = time.time()
        
        def load_pixmap():
            load_start = time.time()
            try:
                # Check if already loaded by another thread/method (avoid duplicate loading)
                if file_path in self.gallery_pixmaps:
                    pixmap = self.gallery_pixmaps[file_path]
                    if pixmap and not pixmap.isNull():
                        # Already loaded, skip reloading but still update label if needed
                        load_time = time.time() - load_start
                        if file_path in self._gallery_load_tracking:
                            if not self._gallery_load_tracking[file_path]['loaded']:
                                self._gallery_load_tracking[file_path]['loaded'] = True
                                self._gallery_load_tracking[file_path]['load_time'] = load_time
                        # Continue to update label below
                        # (label update logic is at the end of this function)
                    else:
                        # Pixmap exists but is null, try to reload
                        pixmap = None
                
                # Try to load from cache or file if not already loaded
                if not pixmap or pixmap.isNull():
                    pixmap = self._get_gallery_pixmap(file_path)
                
                # If not in cache, try to load directly
                # CRITICAL: Process events periodically during file I/O to keep UI responsive
                from PyQt6.QtWidgets import QApplication
                
                if not pixmap or pixmap.isNull():
                    if os.path.exists(file_path):
                        file_ext = os.path.splitext(file_path)[1].lower()
                        raw_extensions = ['.arw', '.cr2', '.nef', '.raf', '.orf', '.dng', '.cr3', '.rw2', '.rwl', '.srw']
                        
                        # For RAW files, try to extract embedded JPEG preview (fast)
                        if file_ext in raw_extensions:
                            # Process events before file I/O to keep UI responsive
                            QApplication.processEvents()
                            try:
                                pixmap = self._extract_embedded_preview(file_path)
                                if pixmap and not pixmap.isNull():
                                    logger.debug(f"[GALLERY] Loaded embedded preview for gallery: {os.path.basename(file_path)}")
                            except Exception as raw_error:
                                logger.debug(f"Error extracting embedded preview for {os.path.basename(file_path)}: {raw_error}")
                                pixmap = None
                        
                        # For non-RAW files, try direct QPixmap load
                        if (not pixmap or pixmap.isNull()) and file_ext not in raw_extensions:
                            # Process events before file I/O to keep UI responsive
                            QApplication.processEvents()
                            pixmap = self._load_pixmap_safe(file_path)
                            if pixmap.isNull():
                                # If QPixmap fails, try PIL Image
                                try:
                                    # Process events before PIL operations
                                    QApplication.processEvents()
                                    pil_image = Image.open(file_path)
                                    if pil_image.mode != 'RGB':
                                        pil_image = pil_image.convert('RGB')
                                    # Convert PIL Image to QPixmap
                                    width, height = pil_image.size
                                    image_bytes = pil_image.tobytes('raw', 'RGB')
                                    # CRITICAL: Calculate bytes_per_line for PIL Image
                                    bytes_per_line = 3 * width  # RGB = 3 channels
                                    qimage = QImage(image_bytes, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                                    if not qimage.isNull():
                                        pixmap = QPixmap.fromImage(qimage)
                                    else:
                                        pixmap = None
                                except Exception as pil_error:
                                    logger.debug(f"Error loading with PIL for {os.path.basename(file_path)}: {pil_error}")
                                    pixmap = None
                        
                        # If still no pixmap, try to get thumbnail from image cache
                        if (not pixmap or pixmap.isNull()) and hasattr(self, 'image_cache'):
                            try:
                                # Process events before cache access
                                QApplication.processEvents()
                                thumbnail_data = self.image_cache.get_thumbnail(file_path)
                                if thumbnail_data is not None:
                                    # Use unified conversion method
                                    pixmap = self._numpy_to_qpixmap(thumbnail_data)
                                    # Process events after conversion
                                    QApplication.processEvents()
                            except Exception as thumb_error:
                                logger.debug(f"Error getting thumbnail from cache for {os.path.basename(file_path)}: {thumb_error}")
                                import traceback
                                logger.debug(f"Traceback: {traceback.format_exc()}")
                
                if pixmap and not pixmap.isNull():
                    # Store in gallery cache
                    self.gallery_pixmaps[file_path] = pixmap
                    # Also cache in global image cache for smooth switching between views
                    try:
                        self.image_cache.put_pixmap(file_path, pixmap)
                    except:
                        pass  # Cache might be full, continue anyway
                    # Update aspect cache
                    aspect = pixmap.width() / pixmap.height() if pixmap.height() > 0 else 4.0 / 3.0
                    self.gallery_aspect_cache[file_path] = aspect
                    load_time = time.time() - load_start
                    
                    # Mark as loaded
                    if file_path in self._gallery_load_tracking:
                        self._gallery_load_tracking[file_path]['loaded'] = True
                        self._gallery_load_tracking[file_path]['load_time'] = load_time
                        logger.debug(f"[GALLERY] [PIXMAP] {os.path.basename(file_path)} - Loaded pixmap in {load_time:.3f}s")
                    
                    # Update the specific thumbnail label directly instead of refreshing entire gallery
                    if hasattr(self, '_gallery_thumb_labels') and file_path in self._gallery_thumb_labels:
                        thumb_label = self._gallery_thumb_labels[file_path]
                        if thumb_label:
                            # Get actual image dimensions
                            pixmap_width = pixmap.width()
                            pixmap_height = pixmap.height()
                            if pixmap_width <= 0 or pixmap_height <= 0:
                                return
                            
                            # Update aspect cache with actual image dimensions
                            actual_aspect = pixmap_width / pixmap_height
                            self.gallery_aspect_cache[file_path] = actual_aspect
                            
                            # Log original image dimensions and aspect ratio
                            logger.info(f"[GALLERY] [THUMBNAIL] {os.path.basename(file_path)} - Original: {pixmap_width}x{pixmap_height}, Aspect: {actual_aspect:.3f}")
                            
                            # For fixed height gallery: scale to fixed height while preserving aspect ratio
                            # Reference code: resized = pixmap.scaled(QSize(w, ROW_HEIGHT), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                            # Get label's fixed height (row_height)
                            label_height = thumb_label.height()
                            if label_height <= 0:
                                label_height = self.gallery_row_height
                            
                            label_width = thumb_label.width()
                            label_aspect = label_width / label_height if label_height > 0 else 0
                            logger.info(f"[GALLERY] [THUMBNAIL] {os.path.basename(file_path)} - Label: {label_width}x{label_height}, Label Aspect: {label_aspect:.3f}")
                            
                            # Calculate width based on actual image aspect ratio and fixed height
                            scaled_width = int(label_height * actual_aspect)
                            scaled_height = label_height
                            scaled_aspect = scaled_width / scaled_height if scaled_height > 0 else 0
                            logger.info(f"[GALLERY] [THUMBNAIL] {os.path.basename(file_path)} - Scaled Target: {scaled_width}x{scaled_height}, Scaled Aspect: {scaled_aspect:.3f}")
                            
                            # Scale pixmap to fixed height while preserving aspect ratio
                            scaled_pixmap = pixmap.scaled(
                                QSize(scaled_width, scaled_height),
                                Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation
                            )
                            
                            if scaled_pixmap.isNull():
                                logger.warning(f"[GALLERY] Failed to scale pixmap for {os.path.basename(file_path)}")
                                return
                            
                            # Log actual scaled pixmap dimensions
                            actual_scaled_width = scaled_pixmap.width()
                            actual_scaled_height = scaled_pixmap.height()
                            actual_scaled_aspect = actual_scaled_width / actual_scaled_height if actual_scaled_height > 0 else 0
                            logger.info(f"[GALLERY] [THUMBNAIL] {os.path.basename(file_path)} - Scaled Result: {actual_scaled_width}x{actual_scaled_height}, Result Aspect: {actual_scaled_aspect:.3f}")
                            
                            # CRITICAL: Resize label to match scaled pixmap size (like reference code: thumb.setFixedSize(resized.size()))
                            # This ensures the label matches the pixmap dimensions exactly, preventing compression
                            # The label size must match the pixmap size to avoid stretching/compression
                            display_start = time.time()
                            thumb_label.setFixedSize(actual_scaled_width, actual_scaled_height)
                            thumb_label.setPixmap(scaled_pixmap)
                            thumb_label.set_original_pixmap(pixmap)
                            
                            # Log final state
                            final_label_width = thumb_label.width()
                            final_label_height = thumb_label.height()
                            final_pixmap_width = scaled_pixmap.width()
                            final_pixmap_height = scaled_pixmap.height()
                            logger.info(f"[GALLERY] [THUMBNAIL] {os.path.basename(file_path)} - Final Label: {final_label_width}x{final_label_height}, Pixmap: {final_pixmap_width}x{final_pixmap_height}")
                            display_time = time.time() - display_start
                            
                            # Mark as displayed
                            if file_path in self._gallery_load_tracking:
                                self._gallery_load_tracking[file_path]['displayed'] = True
                                self._gallery_load_tracking[file_path]['display_time'] = display_time
                                total_time = time.time() - self._gallery_load_tracking[file_path]['start_time']
                                logger.debug(f"[GALLERY] [IMAGE] {os.path.basename(file_path)} - Displayed in {display_time:.3f}s, Total: {total_time:.3f}s")
                else:
                    # Failed to load
                    logger.warning(f"[GALLERY] [PIXMAP] Failed to load pixmap for {os.path.basename(file_path)}")
                    if file_path in self._gallery_load_tracking:
                        self._gallery_load_tracking[file_path]['loaded'] = False
                        self._gallery_load_tracking[file_path]['displayed'] = False
            except Exception as e:
                logger.warning(f"Error loading pixmap async for {os.path.basename(file_path)}: {e}")
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")
                if file_path in self._gallery_load_tracking:
                    self._gallery_load_tracking[file_path]['loaded'] = False
                    self._gallery_load_tracking[file_path]['displayed'] = False
        
        QTimer.singleShot(50, load_pixmap)
    
    # GALLERY FUNCTIONALITY COMMENTED OUT
    # def _create_gallery_item(self, file_path, thumb_width=200):
    #     """Create a single gallery item with thumbnail"""
    #     pass  # Gallery functionality disabled
        from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
        from PyQt6.QtCore import Qt
        
        item = QWidget()
        item.setStyleSheet("""
            QWidget {
                background-color: #1E1E1E;
            }
        """)
        item_layout = QVBoxLayout(item)
        item_layout.setContentsMargins(0, 0, 0, 0)  # No margins for tight packing
        item_layout.setSpacing(0)  # No spacing
        
        # Thumbnail label
        thumb_label = QLabel()
        thumb_height = int(thumb_width * 0.75)  # 4:3 aspect ratio
        thumb_label.setFixedSize(thumb_width, thumb_height)
        thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb_label.setStyleSheet("""
            QLabel {
                background-color: #2A2A2A;
                border: none;
            }
        """)
        # CRITICAL: Use setScaledContents(False) to prevent stretching
        thumb_label.setScaledContents(False)
        thumb_label.setText("")  # No placeholder text for cleaner look
        
        item_layout.addWidget(thumb_label)
        
        # Make item clickable
        thumb_label.mousePressEvent = lambda e, fp=file_path: self._gallery_item_clicked(fp)
        
        # Store references
        item.thumb_label = thumb_label
        item.file_path = file_path
        
        return item
    
    # GALLERY FUNCTIONALITY COMMENTED OUT
    # def _load_gallery_thumbnail_async(self, file_path, item_widget):
    #     """Load thumbnail for gallery item asynchronously (non-blocking)"""
    #     pass  # Gallery functionality disabled
        # Use QThread or simple delayed loading to avoid blocking UI
        from PyQt6.QtCore import QTimer
        
        def load_thumbnail():
            try:
                self._load_gallery_thumbnail(file_path, item_widget.thumb_label)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Error in async thumbnail load: {e}")
        
        # Use a small delay to allow UI to update first
        QTimer.singleShot(10, load_thumbnail)
    
        def _gallery_item_clicked(self, file_path):
            """Handle gallery item click - switch to single view and load image"""
            self.view_mode = 'single'
            self.view_mode_button.setText("Gallery")
            self._show_single_view()
            self.load_raw_image(file_path)
        
        def _load_gallery_thumbnail(self, file_path, label):
            """Load thumbnail for gallery item asynchronously"""
            # Try to get thumbnail from cache first
            try:
                # Check if we have a cached pixmap
                cached_pixmap = self.image_cache.get_pixmap(file_path)
                if cached_pixmap:
                    # Scale to fit thumbnail size
                    scaled_pixmap = cached_pixmap.scaled(200, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    label.setPixmap(scaled_pixmap)
                    return
                
                # Try to load from file
                if os.path.exists(file_path):
                    # For non-RAW files, use safe loader (handles TIFF properly)
                    file_ext = os.path.splitext(file_path)[1].lower()
                    if file_ext not in ['.arw', '.cr2', '.nef', '.raf', '.orf', '.dng']:
                        pixmap = self._load_pixmap_safe(file_path)
                        if not pixmap.isNull():
                            scaled_pixmap = pixmap.scaled(200, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                            label.setPixmap(scaled_pixmap)
                            return
                    
                    # For RAW files, we'll need to extract thumbnail
                    # This is a placeholder - full implementation would use ThumbnailExtractor
                    label.setText("RAW\nLoading...")
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Error loading gallery thumbnail for {file_path}: {e}")
                label.setText("Error")
    
    def _update_sort_button_text(self):
        """Update the sort toggle button text based on current preference"""
        if hasattr(self, 'sort_toggle_button'):
            if self.get_sort_preference():
                self.sort_toggle_button.setText("⇅ Newest")
            else:
                self.sort_toggle_button.setText("⇅ Oldest")
    
    def resort_current_folder(self):
        """Resort the current folder with new sorting preference"""
        if self.current_folder and self.image_files:
            # Store current file path
            current_file = self.current_file_path
            old_index = self.current_file_index
            
            # Resort the files
            self.image_files = self.sort_image_files(self.image_files)
            
            # Find the current file in the new order
            if current_file in self.image_files:
                self.current_file_index = self.image_files.index(current_file)
                self.current_file_path = current_file
                
                # Debug logging
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"[SORT] Resorted folder: {os.path.basename(current_file)}")
                logger.debug(f"[SORT] Old index: {old_index}, New index: {self.current_file_index}")
                logger.debug(f"[SORT] Total files: {len(self.image_files)}")
                logger.debug(f"[SORT] Old position: {old_index + 1}/{len(self.image_files)}")
                logger.debug(f"[SORT] New position: {self.current_file_index + 1}/{len(self.image_files)}")
                
                # Update status bar to reflect new position
                self.update_status_bar()
                
                # GALLERY FUNCTIONALITY COMMENTED OUT
                # Update gallery view if in gallery mode
                # if self.view_mode == 'gallery' and self.gallery_widget and self.gallery_widget.isVisible():
                #     self._gallery_update_needed = True
                #     self._update_gallery_view()
    
    def get_image_capture_time(self, file_path):
        """Extract image capture time from EXIF data (DateTimeOriginal)"""
        try:
            # Try to get from cache first
            from image_cache import get_image_cache
            cache = get_image_cache()
            cached_exif = cache.get_exif(file_path)
            
            if cached_exif and 'capture_time' in cached_exif and cached_exif['capture_time']:
                # Parse cached capture time string (format: "HH:MM:SS YYYY-MM-DD")
                try:
                    time_str = cached_exif['capture_time']
                    dt = datetime.strptime(time_str, "%H:%M:%S %Y-%m-%d")
                    return dt.timestamp()
                except (ValueError, AttributeError):
                    pass
            
            # If not in cache, try to extract from EXIF
            try:
                # Suppress exifread warnings for unsupported file formats
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning, module='exifread')
                    with open(file_path, 'rb') as f:
                        tags = exifread.process_file(f, details=False)
                    
                        # Try different datetime tags in order of preference
                        datetime_tags = ['EXIF DateTimeOriginal', 'Image DateTime', 'EXIF DateTime']
                        for tag_name in datetime_tags:
                            if tag_name in tags:
                                datetime_raw = tags[tag_name]
                                try:
                                    datetime_str = str(datetime_raw)
                                    # Parse datetime string (format: "YYYY:MM:DD HH:MM:SS")
                                    dt = datetime.strptime(datetime_str, "%Y:%m:%d %H:%M:%S")
                                    return dt.timestamp()
                                except (ValueError, AttributeError):
                                    continue
            except Exception:
                pass
            
            # Fallback to file modification time if EXIF extraction fails
            try:
                return os.path.getmtime(file_path)
            except (OSError, AttributeError):
                # Last resort: use 0 (will sort to beginning/end depending on order)
                return 0
        except Exception:
            # If all else fails, use file modification time
            try:
                return os.path.getmtime(file_path)
            except (OSError, AttributeError):
                return 0
    
    def sort_files_by_capture_time(self, file_paths, newest_first=True):
        """Sort files by image capture time (EXIF DateTimeOriginal)"""
        import time
        import logging
        logger = logging.getLogger(__name__)
        
        # OPTIMIZATION: Try to get capture times from cache first to avoid reading EXIF
        from image_cache import get_image_cache
        cache = get_image_cache()
        
        def get_capture_time(file_path):
            # Try cache first (much faster)
            cached_exif = cache.get_exif(file_path)
            if cached_exif and 'capture_time' in cached_exif and cached_exif['capture_time']:
                try:
                    time_str = cached_exif['capture_time']
                    from datetime import datetime
                    dt = datetime.strptime(time_str, "%H:%M:%S %Y-%m-%d")
                    return dt.timestamp()
                except (ValueError, AttributeError):
                    pass
            
            # Fallback to reading EXIF (slower)
            return self.get_image_capture_time(file_path)
        
        sort_start = time.time()
        # Use stable sort with secondary key (filename) to ensure consistent ordering
        # when multiple files have the same capture time
        sorted_files = sorted(
            file_paths, 
            key=lambda fp: (get_capture_time(fp), os.path.basename(fp).lower()),
            reverse=newest_first
        )
        sort_time = time.time() - sort_start
        if len(file_paths) > 100:
            logger.info(f"[SORT] Sorted {len(file_paths)} files in {sort_time:.3f}s")
            print(f"[PERF] 🔄 Sorted {len(file_paths)} files in {sort_time*1000:.1f}ms")
        return sorted_files
    
    def sort_image_files(self, file_paths):
        """Sort files by capture time according to user preference (Newest/Oldest)"""
        newest_first = self.get_sort_preference()  # True = Newest first, False = Oldest first
        return self.sort_files_by_capture_time(file_paths, newest_first=newest_first)

    def open_file(self):
        """Open a folder containing images"""
        settings = self.get_settings()
        last_dir = settings.value("last_opened_dir", "")
        folder_path = QFileDialog.getExistingDirectory(
            self, "Open Folder", last_dir)
        if folder_path:
            self.load_folder_images(folder_path)
            settings.setValue("last_opened_dir", folder_path)

    def open_folder(self):
        settings = self.get_settings()
        last_dir = settings.value("last_opened_dir", "")
        folder_path = QFileDialog.getExistingDirectory(
            self, "Open Folder", last_dir)
        if folder_path:
            self.load_folder_images(folder_path)
            settings.setValue("last_opened_dir", folder_path)

    def show_keyboard_shortcuts(self):
        """Show keyboard shortcuts dialog"""
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setWindowTitle("Keyboard Shortcuts")
        msg_box.setText("Available Keyboard Shortcuts:")
        msg_box.setInformativeText(
            "- Ctrl+O - 🗁 Open image file\n"
            "- Ctrl+Shift+O - 🗁 Open folder of images\n"
            "- Space - Toggle between fit-to-window and 100% zoom\n"
            "- Double-click - Toggle between fit-to-window and 100% zoom\n"
            "- Click and drag - Pan around zoomed image\n"
            "- Left Arrow - Previous image (preserves zoom if zoomed in)\n"
            "- Right Arrow - Next image (preserves zoom if zoomed in)\n"
            "- Down Arrow - Move current image to Discard folder\n"
            "- Delete - Delete current image\n"
            "- Ctrl+Q - Exit application\n\n"
            "You can also drag and drop image files onto the window."
        )
        msg_box.exec()

    def image_mouse_press_event(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.current_pixmap:
            if not self.fit_to_window and self._can_pan():
                self.panning = True
                self.last_pan_point = event.pos()
                self.start_scroll_x = self.scroll_area.horizontalScrollBar().value()
                self.start_scroll_y = self.scroll_area.verticalScrollBar().value()
                self.image_label.setCursor(
                    QCursor(Qt.CursorShape.ClosedHandCursor))

    def image_mouse_move_event(self, event):
        if self.panning and self.current_pixmap and self._can_pan():
            delta = event.pos() - self.last_pan_point
            self.last_pan_point = event.pos()
            h_scroll = self.scroll_area.horizontalScrollBar()
            v_scroll = self.scroll_area.verticalScrollBar()
            new_x = h_scroll.value() - delta.x()
            new_y = v_scroll.value() - delta.y()
            h_scroll.setValue(max(0, min(new_x, h_scroll.maximum())))
            v_scroll.setValue(max(0, min(new_y, v_scroll.maximum())))
        elif self.current_pixmap and not self.fit_to_window and self._can_pan():
            self.image_label.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        else:
            self.image_label.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def image_mouse_release_event(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.panning = False
            if self.current_pixmap and not self.fit_to_window:
                # Update zoom_center_point to reflect current viewport center after panning
                # This ensures that when navigating, the zoom area is preserved correctly
                # Calculate current viewport center in image coordinates
                viewport_size = self.scroll_area.viewport().size()
                scroll_x = self.scroll_area.horizontalScrollBar().value()
                scroll_y = self.scroll_area.verticalScrollBar().value()
                # Viewport center in image coordinates
                viewport_center_x = scroll_x + viewport_size.width() // 2
                viewport_center_y = scroll_y + viewport_size.height() // 2
                # Update zoom_center_point to current viewport center
                self.zoom_center_point = QPoint(viewport_center_x, viewport_center_y)
                self.image_label.setCursor(
                    QCursor(Qt.CursorShape.OpenHandCursor))
            elif self.fit_to_window:
                self.image_label.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        # Always ensure main window has focus for keyboard events
        self.setFocus()

    def image_double_click_event(self, event):
        if not self.current_pixmap:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if self.fit_to_window:
                # Zooming in from fit-to-window mode
                click_pos = event.pos()
                displayed_pixmap = self.image_label.pixmap()

                if displayed_pixmap:
                    # Calculate where the click occurred relative to the displayed image
                    label_size = self.image_label.size()
                    displayed_size = displayed_pixmap.size()

                    # Calculate image offset within the label (image is centered in label)
                    image_x_offset = (label_size.width() -
                                      displayed_size.width()) / 2
                    image_y_offset = (label_size.height() -
                                      displayed_size.height()) / 2

                    # Adjust click position relative to the displayed image
                    adjusted_click_x = click_pos.x() - image_x_offset
                    adjusted_click_y = click_pos.y() - image_y_offset

                    # Check if click is within the displayed image bounds
                    if (0 <= adjusted_click_x < displayed_size.width() and
                            0 <= adjusted_click_y < displayed_size.height()):

                        # Calculate the ratio of the click position within the displayed image
                        click_ratio_x = adjusted_click_x / displayed_size.width()
                        click_ratio_y = adjusted_click_y / displayed_size.height()

                        # Map this ratio to the full-size image coordinates
                        full_size = self.current_pixmap.size()
                        image_click_x = int(click_ratio_x * full_size.width())
                        image_click_y = int(click_ratio_y * full_size.height())

                        # Clamp to valid coordinates
                        image_click_x = max(
                            0, min(image_click_x, full_size.width() - 1))
                        image_click_y = max(
                            0, min(image_click_y, full_size.height() - 1))

                        self.zoom_center_point = QPoint(
                            image_click_x, image_click_y)
                    else:
                        # Click outside image, center on image center
                        self.zoom_center_point = QPoint(
                            self.current_pixmap.width() // 2,
                            self.current_pixmap.height() // 2)
                else:
                    # No displayed pixmap, center on image center
                    self.zoom_center_point = QPoint(
                        self.current_pixmap.width() // 2,
                        self.current_pixmap.height() // 2)

                # Switch to 100% zoom mode
                # If currently displaying half_size and user zooms in, load full resolution FIRST
                if hasattr(self, '_is_half_size_displayed') and self._is_half_size_displayed:
                    if not hasattr(self, '_full_resolution_loading') or not self._full_resolution_loading:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.info("User double-clicked to zoom in - triggering full resolution load")
                        # Check if full resolution is already cached - if so, load it immediately
                        cached_full = self.image_cache.get_full_image(self.current_file_path)
                        if cached_full is not None:
                            cached_max_dim = max(cached_full.shape[1], cached_full.shape[0])
                            if cached_max_dim > 5000:
                                logger.info("Full resolution image already cached, loading immediately...")
                                self._full_resolution_loading = True
                                # Set flag to prevent display_pixmap from resetting fit_to_window
                                # We're about to zoom in, so we want to preserve that intent
                                self._maintain_zoom_on_navigation = True
                                self.display_numpy_image(cached_full)
                                self._is_half_size_displayed = False
                                self._full_resolution_loading = False
                                # Update pixmap reference for zoom calculation
                                self.current_pixmap = self.image_label.pixmap()
                            # Recalculate zoom center point for full resolution image
                            if self.current_pixmap:
                                # Scale the zoom center point from half_size to full resolution
                                scale_x = self.current_pixmap.width() / displayed_pixmap.width() if displayed_pixmap else 1.0
                                scale_y = self.current_pixmap.height() / displayed_pixmap.height() if displayed_pixmap else 1.0
                                if hasattr(self, 'zoom_center_point') and self.zoom_center_point:
                                    self.zoom_center_point = QPoint(
                                        int(self.zoom_center_point.x() * scale_x),
                                        int(self.zoom_center_point.y() * scale_y)
                                    )
                            # Clear the flag after display
                            if hasattr(self, '_maintain_zoom_on_navigation'):
                                delattr(self, '_maintain_zoom_on_navigation')
                            # Now execute the zoom after displaying full resolution
                            self.fit_to_window = False
                            self.current_zoom_level = 1.0
                            self.zoom_to_point()
                            return  # Exit early since we've handled the zoom
                        else:
                            # Start loading full resolution in background
                            self._load_full_resolution_on_demand()
                            # Store zoom intent for when full resolution is ready
                            self._pending_zoom = True
                            # Store the calculated zoom center point
                            self._pending_zoom_center = self.zoom_center_point if hasattr(self, 'zoom_center_point') else None
                            return  # Don't zoom yet - wait for full resolution

                self.fit_to_window = False
                self.current_zoom_level = 1.0
                self.zoom_to_point()
            else:
                # Zooming out to fit-to-window mode
                self.fit_to_window = True
                self.current_zoom_level = 1.0
                self.zoom_center_point = None
                self.scale_image_to_fit()
                self.image_label.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

            self.update_status_bar()
            self.setFocus()

    def zoom_to_point(self):
        if not self.current_pixmap:
            return
        self.image_label.setPixmap(self.current_pixmap)
        self.image_label.resize(self.current_pixmap.size())
        self.image_label.adjustSize()  # Ensure label is resized to pixmap
        self.scroll_area.widget().adjustSize()  # Force scroll area to update
        self.scroll_area.updateGeometry()
        self.image_label.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))

        # Actually center the view on the zoom point
        self._complete_zoom_to_point()

    def _complete_zoom_to_point(self):
        if self.zoom_center_point:
            viewport_size = self.scroll_area.viewport().size()
            image_size = self.current_pixmap.size()
            # Center the zoom point in the viewport
            target_scroll_x = self.zoom_center_point.x() - (viewport_size.width() // 2)
            target_scroll_y = self.zoom_center_point.y() - (viewport_size.height() // 2)
            max_scroll_x = max(0, image_size.width() - viewport_size.width())
            max_scroll_y = max(0, image_size.height() - viewport_size.height())
            final_scroll_x = max(0, min(target_scroll_x, max_scroll_x))
            final_scroll_y = max(0, min(target_scroll_y, max_scroll_y))
            self.scroll_area.horizontalScrollBar().setValue(final_scroll_x)
            self.scroll_area.verticalScrollBar().setValue(final_scroll_y)
        else:
            self.center_image_in_scroll_area()

    def convert_widget_to_image_coords(self, widget_pos):
        """Convert widget coordinates to full-resolution image coordinates"""
        if not self.current_pixmap:
            return QPoint(0, 0)

        # Get current displayed image size
        displayed_image = self.image_label.pixmap()
        if not displayed_image:
            return QPoint(0, 0)

        # Calculate scaling factor from displayed to original
        original_size = self.current_pixmap.size()
        displayed_size = displayed_image.size()

        scale_x = original_size.width() / displayed_size.width()
        scale_y = original_size.height() / displayed_size.height()

        # Convert widget coordinates to image coordinates
        image_x = int(widget_pos.x() * scale_x)
        image_y = int(widget_pos.y() * scale_y)

        return QPoint(image_x, image_y)

    def apply_pan_offset(self):
        # Deprecated: direct panning is now handled in mouse events
        pass

    def apply_zoom_and_pan_simple(self):
        """Simple zoom and pan that centers on the clicked point"""
        if not self.current_pixmap:
            return

        # Set the image at 100% zoom
        self.image_label.setPixmap(self.current_pixmap)
        self.image_label.resize(self.current_pixmap.size())

        # If we have a zoom center point, center the scroll area on it
        if self.zoom_center_point:
            # Calculate the position in the full-size image
            # The zoom_center_point is in scroll area coordinates
            # We need to convert it to full-size image coordinates

            # Get the scaling factor from the fit-to-window to full size
            scroll_area_size = self.scroll_area.size()
            image_size = self.current_pixmap.size()

            # Calculate what proportion of the scroll area the click was at
            click_x_ratio = self.zoom_center_point.x() / scroll_area_size.width()
            click_y_ratio = self.zoom_center_point.y() / scroll_area_size.height()

            # Calculate the corresponding position in the full-size image
            target_x = int(click_x_ratio * image_size.width())
            target_y = int(click_y_ratio * image_size.height())

            # Center the scroll area on this point
            scroll_x = target_x - round(scroll_area_size.width() / 2)
            scroll_y = target_y - round(scroll_area_size.height() / 2)

            # Clamp to valid range
            max_scroll_x = max(0, image_size.width() -
                               scroll_area_size.width())
            max_scroll_y = max(0, image_size.height() -
                               scroll_area_size.height())

            scroll_x = max(0, min(scroll_x, max_scroll_x))
            scroll_y = max(0, min(scroll_y, max_scroll_y))

            # Set scroll position
            self.scroll_area.horizontalScrollBar().setValue(scroll_x)
            self.scroll_area.verticalScrollBar().setValue(scroll_y)
        else:
            # Center the image
            self.center_image_in_scroll_area()

        # Update cursor
        self.image_label.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))

    def apply_zoom_and_pan(self):
        """Apply current zoom level and pan offset to the image"""
        if not self.current_pixmap:
            return

        # Calculate scaled size
        original_size = self.current_pixmap.size()
        scaled_width = int(original_size.width() * self.current_zoom_level)
        scaled_height = int(original_size.height() * self.current_zoom_level)

        # Scale the pixmap
        scaled_pixmap = self.current_pixmap.scaled(
            scaled_width, scaled_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        # Set the scaled pixmap
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.resize(scaled_pixmap.size())

        # Apply zoom center point and panning
        if self.zoom_center_point:
            # Check if we have restored scroll positions from navigation
            if hasattr(self, '_restore_start_scroll_x') and self._restore_start_scroll_x is not None:
                # Use the restored scroll positions directly
                scroll_x = self._restore_start_scroll_x
                scroll_y = self._restore_start_scroll_y
                
                # Clamp scroll positions to valid range
                viewport_size = self.scroll_area.viewport().size()
                image_size = scaled_pixmap.size()
                max_scroll_x = max(0, image_size.width() - viewport_size.width())
                max_scroll_y = max(0, image_size.height() - viewport_size.height())
                
                scroll_x = max(0, min(scroll_x, max_scroll_x))
                scroll_y = max(0, min(scroll_y, max_scroll_y))
                
                # Set scroll position
                self.scroll_area.horizontalScrollBar().setValue(scroll_x)
                self.scroll_area.verticalScrollBar().setValue(scroll_y)
                
                # Clear the restored scroll positions after use
                self._restore_start_scroll_x = None
                self._restore_start_scroll_y = None
            else:
                # Calculate the position to center the zoom point
                viewport_size = self.scroll_area.viewport().size()
                image_size = scaled_pixmap.size()

                # Convert image coordinates to scaled coordinates
                center_x = int(self.zoom_center_point.x()
                               * self.current_zoom_level)
                center_y = int(self.zoom_center_point.y()
                               * self.current_zoom_level)

                # Calculate scroll position to center the zoom point
                scroll_x = center_x - round(viewport_size.width() / 2)
                scroll_y = center_y - round(viewport_size.height() / 2)

                # Clamp scroll positions to valid range
                max_scroll_x = max(0, image_size.width() - viewport_size.width())
                max_scroll_y = max(0, image_size.height() - viewport_size.height())

                scroll_x = max(0, min(scroll_x, max_scroll_x))
                scroll_y = max(0, min(scroll_y, max_scroll_y))

                # Set scroll position
                self.scroll_area.horizontalScrollBar().setValue(scroll_x)
                self.scroll_area.verticalScrollBar().setValue(scroll_y)
        else:
            # Center the image if no zoom center point is set
            self.center_image_in_scroll_area()

        # Update cursor
        if not self.fit_to_window:
            self.image_label.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        else:
            self.image_label.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter events for file and folder dropping"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            for url in urls:
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    # Check if it's a folder
                    if os.path.isdir(file_path):
                        event.acceptProposedAction()
                        return
                    # Check if it's a supported image file
                    file_ext = os.path.splitext(file_path)[1].lower()
                    if file_ext in self.get_supported_extensions():
                        event.acceptProposedAction()
                        return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        """Handle drop events for file and folder dropping"""
        urls = event.mimeData().urls()
        for url in urls:
            if url.isLocalFile():
                file_path = url.toLocalFile()
                # Check if it's a folder
                if os.path.isdir(file_path):
                    # Load folder images
                    self.load_folder_images(file_path)
                    event.acceptProposedAction()
                    return
                # Check if it's a supported image file
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext in self.get_supported_extensions():
                    # If it's an image file, load the folder containing it
                    # This matches the behavior when opening a file from command line
                    folder_path = os.path.dirname(file_path)
                    filename = os.path.basename(file_path)
                    self.load_folder_images(folder_path, start_file=filename)
                    event.acceptProposedAction()
                    return
        event.ignore()

    def _cleanup_current_processing(self):
        """Clean up current processing threads and resources"""
        import logging
        import traceback
        import time
        logger = logging.getLogger(__name__)
        cleanup_start = time.time()
        
        # Prevent multiple cleanup operations from running simultaneously
        with self._cleanup_lock:
            if self._cleanup_in_progress:
                logger.debug(f"[CLEANUP] Cleanup already in progress, skipping duplicate call")
                return
            
            self._cleanup_in_progress = True
            logger.info(f"[CLEANUP] ========== _cleanup_current_processing() STARTED at {cleanup_start:.3f} ==========")
        
        try:
            # Stop any current processing
            if self.current_processor is not None:
                processor_type = type(self.current_processor).__name__
                is_running = self.current_processor.isRunning() if hasattr(self.current_processor, 'isRunning') else False
                logger.debug(f"Cleaning up processor: {processor_type}, is_running: {is_running}, file: {getattr(self.current_processor, 'file_path', 'unknown')}")
                
                try:
                    # Disconnect signals first to prevent access violations
                    logger.info(f"[CLEANUP] Disconnecting processor signals")
                    try:
                        if hasattr(self.current_processor, 'image_processed'):
                            self.current_processor.image_processed.disconnect()
                        if hasattr(self.current_processor, 'error_occurred'):
                            self.current_processor.error_occurred.disconnect()
                        if hasattr(self.current_processor, 'thumbnail_fallback_used'):
                            self.current_processor.thumbnail_fallback_used.disconnect()
                        if hasattr(self.current_processor, 'processing_progress'):
                            self.current_processor.processing_progress.disconnect()
                        if hasattr(self.current_processor, 'exif_data_ready'):
                            self.current_processor.exif_data_ready.disconnect()
                        logger.info(f"[CLEANUP] Processor signals disconnected")
                    except Exception as disconnect_error:
                        logger.warning(f"[CLEANUP] Error disconnecting signals (may be normal if already disconnected): {disconnect_error}")
                    
                    # RAWProcessor uses cleanup(), EnhancedRAWProcessor uses stop_processing() and wait()
                    if hasattr(self.current_processor, 'cleanup'):
                        logger.info(f"[CLEANUP] Calling cleanup() on {processor_type}")
                        self.current_processor.cleanup()
                        logger.info(f"[CLEANUP] cleanup() completed for {processor_type}")
                    else:
                        # For EnhancedRAWProcessor, use stop_processing and wait
                        logger.info(f"[CLEANUP] Using stop_processing() for {processor_type}")
                        if hasattr(self.current_processor, 'stop_processing'):
                            self.current_processor.stop_processing()
                            logger.info(f"[CLEANUP] stop_processing() called for {processor_type}")
                        
                        if hasattr(self.current_processor, 'isRunning'):
                            if self.current_processor.isRunning():
                                logger.info(f"[CLEANUP] Processor still running, calling quit() and wait()")
                                self.current_processor.quit()
                                wait_result = self.current_processor.wait(100)  # Wait up to 100ms
                                logger.info(f"[CLEANUP] wait() returned: {wait_result}, is_running: {self.current_processor.isRunning()}")
                                if not wait_result:
                                    logger.info(f"[CLEANUP] Processor did not stop gracefully, calling terminate()")
                                    self.current_processor.terminate()
                                    terminate_wait = self.current_processor.wait(50)  # Wait up to 50ms after terminate
                                    logger.info(f"[CLEANUP] After terminate(), wait() returned: {terminate_wait}, is_running: {self.current_processor.isRunning()}")
                            else:
                                logger.info(f"[CLEANUP] Processor not running, skip quit/wait")
                    
                    # Clear processor reference after cleanup
                    self.current_processor = None
                    logger.info(f"[CLEANUP] Processor reference cleared")
                except Exception as cleanup_error:
                    logger.error(f"Error during processor cleanup: {cleanup_error}", exc_info=True)
                    logger.debug(f"Cleanup error traceback: {traceback.format_exc()}")
                    # Try to clear reference even if cleanup failed
                    try:
                        self.current_processor = None
                    except:
                        pass
            else:
                logger.debug("No current_processor to clean up")
            
            # Cancel preload threads (non-blocking)
            if hasattr(self, 'preload_manager'):
                logger.debug("Cancelling preload threads (non-blocking)")
                try:
                    self.preload_manager.cancel_all_preloads()
                    logger.debug("Preload threads cancelled")
                except Exception as preload_error:
                    logger.error(f"Error cancelling preload threads: {preload_error}", exc_info=True)
            else:
                logger.debug("No preload_manager available")
            
            cleanup_end = time.time()
            logger.info(f"[CLEANUP] _cleanup_current_processing completed successfully in {cleanup_end - cleanup_start:.3f}s")
        except Exception as e:
            cleanup_end = time.time()
            logger.error(f"[CLEANUP] ========== CRITICAL ERROR in _cleanup_current_processing "
                        f"(at {cleanup_end:.3f}, duration: {cleanup_end - cleanup_start:.3f}s) ==========")
            logger.error(f"[CLEANUP] Exception type: {type(e).__name__}, message: {e}", exc_info=True)
            logger.error(f"[CLEANUP] Full traceback:\n{traceback.format_exc()}")
            raise
        finally:
            # Always reset cleanup flag, even if an error occurred
            with self._cleanup_lock:
                self._cleanup_in_progress = False
                logger.debug(f"[CLEANUP] Cleanup flag reset, cleanup_in_progress=False")

    def load_raw_image(self, file_path):
        import time
        import logging
        import traceback
        logger = logging.getLogger(__name__)
        load_start = time.time()
        logger.info(f"[LOAD] ========== load_raw_image() STARTED at {load_start:.3f} ==========")
        logger.info(f"[LOAD] File path: {file_path}")
        logger.info(f"[LOAD] Previous file: {getattr(self, 'current_file_path', 'None')}")
        logger.info(f"[LOAD] Navigation state - in_progress: {getattr(self, '_navigation_in_progress', False)}")
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                error_msg = f"The file {file_path} does not exist."
                logger.error(f"[LOAD] File not found: {file_path}")
                self.show_error("File not found", error_msg)
                return

            # Store the requested file path for later comparison (after cleanup)
            # This allows us to detect if file changed during cleanup due to rapid navigation
            requested_file_path = file_path
            
            logger.info(f"[LOAD] File exists, proceeding with load")
            print(f"[PERF] Loading image: {os.path.basename(requested_file_path)}")
            
            # Note: Navigation concurrency is controlled by can_navigate() in navigation methods
            # We don't check _navigation_in_progress here because load_raw_image may be called
            # from other places (not just navigation), and the navigation methods already have
            # proper concurrency control
            
            # Reset half_size flag when loading new image
            logger.info(f"[LOAD] Resetting flags - half_size: {getattr(self, '_is_half_size_displayed', False)}, full_res_loading: {getattr(self, '_full_resolution_loading', False)}")
            self._is_half_size_displayed = False
            self._full_resolution_loading = False
            logger.info(f"[LOAD] Flags reset - half_size: {self._is_half_size_displayed}, full_res_loading: {self._full_resolution_loading}")
            
            # Clear pending zoom restore when loading new image (will be set again if needed)
            if hasattr(self, '_pending_zoom_restore'):
                logger.debug("Clearing _pending_zoom_restore")
                delattr(self, '_pending_zoom_restore')
            if hasattr(self, '_pending_zoom_center'):
                logger.debug("Clearing _pending_zoom_center")
                delattr(self, '_pending_zoom_center')
            if hasattr(self, '_pending_zoom_level'):
                logger.debug("Clearing _pending_zoom_level")
                delattr(self, '_pending_zoom_level')

            # Clean up current processing (simplified with new architecture)
            cleanup_start = time.time()
            logger.info(f"[LOAD] Starting cleanup of current processing (if any)")
            # Cancel any pending image load tasks (non-blocking)
            if self.current_file_path:
                self.image_manager.cancel_task(self.current_file_path)
            # Legacy cleanup for old processor (if still exists)
            if self.current_processor:
                logger.info(f"[LOAD] Legacy processor cleanup: {type(self.current_processor).__name__}")
                self._cleanup_current_processing()
            cleanup_time = time.time() - cleanup_start
            logger.info(f"[LOAD] Cleanup completed in {cleanup_time:.3f}s")
            
            # Update current_file_path immediately after cleanup
            # This ensures we have the correct value for subsequent operations
            # NOTE: We removed the post-cleanup cancellation check because:
            # 1. The debounce mechanism already handles rapid navigation
            # 2. The check was causing false cancellations during normal navigation
            # 3. current_file_path is always different from requested_file_path at this point
            #    (it's the old file), so the check would always cancel
            self.current_file_path = requested_file_path
            
            # Verify cleanup completed
            if self.current_processor is not None:
                logger.warning(f"[LOAD] WARNING: current_processor still exists after cleanup: {type(self.current_processor).__name__}")
                if hasattr(self.current_processor, 'isRunning') and self.current_processor.isRunning():
                    logger.warning(f"[LOAD] WARNING: current_processor is still running after cleanup!")
            else:
                logger.info(f"[LOAD] Cleanup verified: current_processor is None")

            # Note: current_file_path is now set above (after cleanup check)
            # to prevent false cancellations during normal navigation
            filename = os.path.basename(requested_file_path)
            logger.debug(f"Setting window title to: {filename}")
            self.setWindowTitle(f"RAW Image Viewer - {filename}")
            # Update custom title bar
            if hasattr(self, 'title_bar'):
                self.title_bar.set_title(f"RAW Image Viewer - {filename}")

            # Reset EXIF data ready flag for new image
            self._exif_data_ready = False

            # CRITICAL: For RAW files, skip full image cache to ensure fresh processing
            # This prevents showing stale cached images with incorrect orientation
            # Only use cache for non-RAW files
            raw_extensions = {'.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2', 
                             '.pef', '.srw', '.x3f', '.raf', '.3fr', '.fff', '.iiq', 
                             '.cap', '.erf', '.mef', '.mos', '.nrw', '.rwl', '.srf'}
            file_ext = os.path.splitext(requested_file_path)[1].lower()
            is_raw_file = file_ext in raw_extensions
            
            if not is_raw_file:
                # Only check full image cache for non-RAW files
                logger.info(f"[LOAD] Checking for cached full image (non-RAW file)")
                cache_check_start = time.time()
                cached_image = self.image_cache.get_full_image(requested_file_path)
                cache_check_time = time.time() - cache_check_start
                if cached_image is not None:
                    logger.info(f"[LOAD] Cache hit: full image found for {filename}, shape: {cached_image.shape}")
                    print(f"[PERF] ✅ CACHE HIT: Full image loaded from cache in {cache_check_time*1000:.1f}ms")
                    self.status_bar.showMessage(f"Loaded {filename} from cache")
                    try:
                        logger.info(f"[LOAD] Displaying cached full image")
                        display_start = time.time()
                        self.display_numpy_image(cached_image)
                        display_time = time.time() - display_start
                        logger.info(f"[LOAD] Cached image displayed in {display_time:.3f}s")
                        self.setFocus()
                        self.save_session_state()
                        self._start_preloading()
                        logger.info(f"[LOAD] Successfully displayed cached full image for {filename} (total: {time.time() - load_start:.3f}s)")
                        return
                    except Exception as display_error:
                        logger.error(f"[LOAD] Error displaying cached image: {display_error}", exc_info=True)
                        logger.error(f"[LOAD] Display error traceback:\n{traceback.format_exc()}")
                        # Continue to process if display fails
            else:
                logger.info(f"[LOAD] Skipping full image cache check for RAW file: {filename}")

            # Check if we have a cached pixmap for non-RAW files ONLY
            # CRITICAL: Only check pixmap cache for non-RAW files to avoid loading JPEG when RAW is requested
            raw_extensions = {'.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2', 
                             '.pef', '.srw', '.x3f', '.raf', '.3fr', '.fff', '.iiq', 
                             '.cap', '.erf', '.mef', '.mos', '.nrw', '.rwl', '.srf'}
            file_ext = os.path.splitext(file_path)[1].lower()
            is_raw_file = file_ext in raw_extensions
            
            if not is_raw_file:
                # Only check pixmap cache for non-RAW files (JPEG, PNG, etc.)
                logger.info(f"[LOAD] Checking for cached pixmap (non-RAW file)")
                cache_check_start = time.time()
                cached_pixmap = self.image_cache.get_pixmap(requested_file_path)
                cache_check_time = time.time() - cache_check_start
                if cached_pixmap is not None:
                    logger.info(f"[LOAD] Cache hit: pixmap found for {filename}, size: {cached_pixmap.width()}x{cached_pixmap.height()}")
                    print(f"[PERF] ✅ CACHE HIT: Pixmap loaded from cache in {cache_check_time*1000:.1f}ms")
                    self.status_bar.showMessage(f"Loaded {filename} from cache")
                    try:
                        # CRITICAL: Apply orientation correction to cached pixmap
                        # Cached pixmaps should already have orientation applied, but we apply it again
                        # to ensure consistency, especially if orientation was cached incorrectly
                        orientation = self.get_orientation_from_exif(requested_file_path)
                        cached_pixmap = self.apply_orientation_to_pixmap(cached_pixmap, orientation)
                        logger.info(f"[LOAD] Applied orientation correction to cached pixmap: {orientation}")
                        
                        logger.info(f"[LOAD] Displaying cached pixmap")
                        display_start = time.time()
                        self.display_pixmap(cached_pixmap)
                        display_time = time.time() - display_start
                        logger.info(f"[LOAD] Cached pixmap displayed in {display_time:.3f}s")
                        self.setFocus()
                        self.save_session_state()
                        self._start_preloading()
                        logger.info(f"[LOAD] Successfully displayed cached pixmap for {filename} (total: {time.time() - load_start:.3f}s)")
                        return
                    except Exception as display_error:
                        logger.error(f"[LOAD] Error displaying cached pixmap: {display_error}", exc_info=True)
                        logger.error(f"[LOAD] Display error traceback:\n{traceback.format_exc()}")
                        # Continue to process if display fails
            else:
                logger.info(f"[LOAD] Skipping pixmap cache check for RAW file: {filename}")

            # No cache hit, use new unified image load manager
            cache_miss_time = time.time() - load_start
            logger.info(f"[LOAD] No cache hit, starting unified image load manager (elapsed: {cache_miss_time:.3f}s)")
            print(f"[PERF] ❌ CACHE MISS: Starting processing (cache check took {cache_miss_time*1000:.1f}ms)")
            self.status_bar.showMessage(f"Loading {filename}...")
            # Set loading message with proper alignment (centered both vertically and horizontally)
            # Ensure label fills the viewport for proper centering
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
            self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.image_label.setText("Loading preview...\nPlease wait...")

            # Use new unified image load manager (non-blocking, thread pool based)
            manager_start = time.time()
            logger.info(f"[LOAD] Requesting image load via ImageLoadManager for: {requested_file_path}")
            try:
                # Request image load with highest priority
                self.image_manager.load_image(
                    file_path=requested_file_path,
                    priority=Priority.CURRENT,
                    cancel_existing=True,
                    use_full_resolution=False
                )
                logger.info(f"[LOAD] Image load requested via ImageLoadManager")
            except Exception as manager_error:
                logger.error(f"[LOAD] Failed to request image load: {manager_error}", exc_info=True)
                logger.error(f"[LOAD] Manager error traceback:\n{traceback.format_exc()}")
                # Fallback to legacy processor if manager fails
                logger.warning(f"[LOAD] Falling back to legacy RAWProcessor")
                try:
                    file_ext = os.path.splitext(requested_file_path)[1].lower()
                    is_raw = is_raw_file(requested_file_path)
                    self.current_processor = RAWProcessor(requested_file_path, is_raw=is_raw, use_full_resolution=False)
                    self.current_processor.image_processed.connect(self.on_image_processed)
                    self.current_processor.error_occurred.connect(self.on_processing_error)
                    self.current_processor.thumbnail_fallback_used.connect(self.on_thumbnail_fallback)
                    self.current_processor.processing_progress.connect(self.on_processing_progress)
                    self.current_processor.exif_data_ready.connect(self.on_exif_data_ready)
                    self.current_processor.start()
                except Exception as fallback_error:
                    logger.error(f"[LOAD] Fallback processor also failed: {fallback_error}", exc_info=True)
                    raise
            
            manager_request_time = time.time() - manager_start
            logger.info(f"[LOAD] Image load requested in {manager_request_time:.3f}s")
            print(f"[PERF] 📊 SETUP TIME: {manager_request_time*1000:.1f}ms (cleanup: {cleanup_time*1000:.1f}ms)")

            self.setFocus()
            # Save session state when image changes
            self.save_session_state()
            total_time = time.time() - load_start
            logger.info(f"[LOAD] ========== load_raw_image() COMPLETED for {filename} in {total_time:.3f}s ==========")
            print(f"[PERF] ✅ LOAD COMPLETE: {os.path.basename(requested_file_path)} in {total_time*1000:.1f}ms")
        except Exception as e:
            total_time = time.time() - load_start
            logger.error(f"[LOAD] ========== CRITICAL ERROR in load_raw_image (at {time.time():.3f}, duration: {total_time:.3f}s) ==========")
            logger.error(f"[LOAD] Exception type: {type(e).__name__}, message: {e}", exc_info=True)
            logger.error(f"[LOAD] Full traceback:\n{traceback.format_exc()}")
            requested_file_name = requested_file_path if 'requested_file_path' in locals() else (file_path if 'file_path' in locals() else 'unknown')
            print(f"[PERF] ❌ LOAD ERROR: {os.path.basename(requested_file_name)} failed after {total_time*1000:.1f}ms - {type(e).__name__}: {e}")
            # Try to show error to user
            try:
                self.show_error("Load Error", f"Failed to load image: {str(e)}")
            except:
                pass
            raise

    def on_thumbnail_fallback(self, message):
        """Handle when thumbnail fallback is used"""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Thumbnail fallback: Loading thumbnail...")
        self.status_bar.showMessage(
            f"⚠️ {message} - Image quality may be reduced")

    def on_thumbnail_ready(self, thumbnail):
        """Handle when thumbnail is ready for immediate display."""
        if thumbnail is not None:
            # Smart thumbnail display: only show thumbnail if it makes sense
            if self._should_show_thumbnail():
                self.display_numpy_image(thumbnail)
                self.status_bar.showMessage(
                    "Preview loaded - processing full image...")
            else:
                # Store thumbnail but don't display it yet
                self._pending_thumbnail = thumbnail
                self.status_bar.showMessage(
                    "Processing full image for quality evaluation...")

    def on_image_processed_enhanced(self, rgb_image):
        """Handle enhanced image processing results."""
        try:
            if rgb_image is None:
                # Non-RAW file - load with QPixmap and cache it
                pixmap = self._load_pixmap_safe(self.current_file_path)
                if pixmap.isNull():
                    self.show_error("Display Error",
                                    "Could not load image file.")
                    return

                # Apply orientation correction for non-RAW files
                cached_exif = self.image_cache.get_exif(self.current_file_path)
                orientation = cached_exif.get(
                    'orientation', 1) if cached_exif else 1
                pixmap = self.apply_orientation_to_pixmap(pixmap, orientation)

                # Cache the pixmap
                self.image_cache.put_pixmap(self.current_file_path, pixmap)

                self.display_pixmap(pixmap)
            else:
                # RAW file - processed numpy array
                self.display_numpy_image(rgb_image)

            # Update UI state
            if self.current_file_path:
                self.scan_folder_for_images(self.current_file_path)

            # Update status bar with EXIF data instead of just showing "Loaded"
            self.update_status_bar()

            # Start preloading adjacent images
            self._start_preloading()

            # Clear any pending thumbnail since we now have the full image
            self._pending_thumbnail = None

        except Exception as e:
            error_msg = f"Error displaying image: {str(e)}"
            self.show_error("Display Error", error_msg)

        self.setFocus()

    def on_processing_progress(self, message):
        """Handle processing progress updates."""
        filename = os.path.basename(self.current_file_path)
        self.status_bar.showMessage(f"{filename}: {message}")

    def on_exif_data_ready(self, exif_data):
        """Handle when EXIF data becomes available."""
        # Update status bar immediately with EXIF data when it becomes available
        # This ensures EXIF data is shown even in fit-to-window mode
        import logging
        logger = logging.getLogger(__name__)
        if self.current_file_path:
            # Set a flag to indicate EXIF data is ready
            self._exif_data_ready = True
            # Always update status bar when EXIF data is ready, even if pixmap is not yet available
            # This ensures original resolution is shown immediately
            logger.info(f"[EXIF] EXIF data ready, updating status bar for {os.path.basename(self.current_file_path)}")
            self.update_status_bar()
            logger.info(f"[EXIF] Status bar updated")

    def on_cache_hit(self, file_path, cache_type):
        """Handle cache hit events for performance monitoring."""
        # Could be used for performance analytics
        pass

    def on_memory_warning(self, memory_percent):
        """Handle memory warning events."""
        print(f"⚠️ Memory usage high: {memory_percent:.1f}%")

    def _should_show_thumbnail(self):
        """Determine if we should show thumbnail immediately or wait for full image."""
        # If user explicitly wants thumbnails even when zoomed, always show
        if self.show_thumbnails_when_zoomed:
            return True

        # Don't show thumbnail if user is in 100% zoom mode (checking sharpness)
        if not self.fit_to_window:
            return False

        # Don't show thumbnail if we're maintaining zoom state from navigation
        # (user was previously at 100% zoom checking sharpness)
        if hasattr(self, '_maintain_zoom_on_navigation'):
            return False

        # Don't show thumbnail if we're restoring zoom state to 100%
        if (hasattr(self, '_restore_zoom_center') and
                self._restore_zoom_center is not None):
            return False

        # Show thumbnail in fit-to-window mode for quick overview
        return True

    def display_numpy_image(self, rgb_image):
        """Display a numpy image array."""
        import logging
        import time
        logger = logging.getLogger(__name__)
        display_start = time.time()
        
        try:
            height, width, channels = rgb_image.shape
            logger.info(f"[DISPLAY] ========== display_numpy_image() STARTED at {display_start:.3f} ==========")
            logger.info(f"[DISPLAY] Image dimensions: {width}x{height}, channels: {channels}")
            
            # Check if we have a cached pixmap first (fastest path)
            # BUT: if this is a full resolution image (max dimension > 5000px), don't use cached half_size pixmap
            if hasattr(self, 'current_file_path') and self.current_file_path:
                height, width, channels = rgb_image.shape
                # Use max dimension to handle both portrait and landscape orientations
                max_dimension = max(width, height)
                is_full_resolution = max_dimension > 5000
                logger.info(f"[DISPLAY] Full resolution check: {is_full_resolution} (max dimension: {max_dimension}, {width}x{height})")
                
                if is_full_resolution:
                    # For full resolution images, always convert and display the new image
                    # Don't use cached pixmap which might be half_size
                    logger.info(f"[DISPLAY] Full resolution image ({width}x{height}), converting and displaying")
                else:
                    # For thumbnails or half_size, check cache first
                    logger.info(f"[DISPLAY] Checking for cached pixmap")
                    cached_pixmap = self.image_cache.get_pixmap(self.current_file_path)
                    if cached_pixmap is not None:
                        logger.info(f"[DISPLAY] Using cached pixmap for {width}x{height} image")
                        self.display_pixmap(cached_pixmap)
                        return
                    else:
                        logger.info(f"[DISPLAY] No cached pixmap found, will convert")
            
            # Convert numpy array to QPixmap
            bytes_per_line = channels * width
            logger.info(f"[DISPLAY] Converting numpy array to QPixmap - bytes_per_line: {bytes_per_line}")

            # Ensure the data is contiguous
            if not rgb_image.flags['C_CONTIGUOUS']:
                logger.info(f"[DISPLAY] Making array contiguous")
                rgb_image = np.ascontiguousarray(rgb_image)

            # Convert to bytes for PyQt6 compatibility
            conversion_start = time.time()
            logger.info(f"[DISPLAY] Converting to bytes")
            image_data = rgb_image.data.tobytes() if hasattr(
                rgb_image.data, 'tobytes') else bytes(rgb_image.data)
            logger.info(f"[DISPLAY] Bytes conversion completed, creating QImage")

            q_image = QImage(image_data, width, height,
                             bytes_per_line, QImage.Format.Format_RGB888)
            logger.info(f"[DISPLAY] QImage created, creating QPixmap")
            pixmap = QPixmap.fromImage(q_image)
            conversion_time = time.time() - conversion_start
            logger.info(f"[DISPLAY] QImage/QPixmap conversion completed in {conversion_time:.3f}s")

            # CRITICAL: Apply orientation correction only if not already applied
            # Images from UnifiedImageProcessor (via ImageLoadManager) already have orientation applied
            # Images from old RAWProcessor path need orientation correction here
            if hasattr(self, 'current_file_path') and self.current_file_path:
                # Check if orientation was already applied (e.g., by UnifiedImageProcessor)
                if not getattr(self, '_orientation_already_applied', False):
                    orientation = self.get_orientation_from_exif(self.current_file_path)
                    if orientation != 1:
                        logger.info(f"[DISPLAY] Applying orientation correction: {orientation}")
                        pixmap = self.apply_orientation_to_pixmap(pixmap, orientation)
                    else:
                        logger.debug(f"[DISPLAY] Orientation is 1 (normal), no correction needed")
                else:
                    logger.debug(f"[DISPLAY] Orientation already applied by processor, skipping")
            
            # Cache the pixmap for future use (after orientation correction)
            if hasattr(self, 'current_file_path') and self.current_file_path:
                logger.info(f"[DISPLAY] Caching pixmap")
                self.image_cache.put_pixmap(self.current_file_path, pixmap)
            
            # Set _is_half_size_displayed flag based on image dimensions
            # This is important for zoom detection - if user zooms in, we need to load full resolution
            max_dimension = max(width, height)
            if max_dimension < 5000:
                self._is_half_size_displayed = True
                logger.debug(f"[DISPLAY] Setting _is_half_size_displayed=True for thumbnail/half_size image ({width}x{height}, max: {max_dimension})")
            else:
                self._is_half_size_displayed = False
                logger.debug(f"[DISPLAY] Setting _is_half_size_displayed=False for full resolution image ({width}x{height}, max: {max_dimension})")
            
            pixmap_display_start = time.time()
            logger.info(f"[DISPLAY] Calling display_pixmap()")
            self.display_pixmap(pixmap)
            pixmap_display_time = time.time() - pixmap_display_start
            total_time = time.time() - display_start
            logger.info(f"[DISPLAY] RAW image displayed successfully: {width}x{height} (pixmap display: {pixmap_display_time:.3f}s, total: {total_time:.3f}s)")
            print(f"[PERF] 🖼️  DISPLAY COMPLETE: {width}x{height} (pixmap: {pixmap_display_time*1000:.1f}ms, total: {total_time*1000:.1f}ms)")

        except Exception as e:
            total_time = time.time() - display_start
            logger.error(f"[DISPLAY] ========== ERROR in display_numpy_image (at {time.time():.3f}, duration: {total_time:.3f}s) ==========")
            logger.error(f"[DISPLAY] Exception type: {type(e).__name__}, message: {e}", exc_info=True)
            error_msg = f"Error displaying numpy image: {str(e)}"
            self.show_error("Display Error", error_msg)

    def display_pixmap(self, pixmap):
        """Display a QPixmap."""
        import logging
        import time
        logger = logging.getLogger(__name__)
        display_start = time.time()
        
        # Get current file name for logging context
        current_file = os.path.basename(self.current_file_path) if hasattr(self, 'current_file_path') and self.current_file_path else "Unknown"
        logger.info(f"[DISPLAY_PIXMAP] ========== display_pixmap() STARTED at {display_start:.3f} ==========")
        logger.info(f"[DISPLAY_PIXMAP] File: {current_file}, Pixmap size: {pixmap.width()}x{pixmap.height()}")
        
        self.current_pixmap = pixmap

        # Check if we're restoring zoom state - if so, don't reset fit_to_window
        if hasattr(self, '_restore_zoom_center') and self._restore_zoom_center is not None:
            # Zoom restoration will be handled below, don't reset fit_to_window here
            logger.debug(f"display_pixmap: Zoom restoration pending, preserving zoom state")
        elif not hasattr(self, '_maintain_zoom_on_navigation'):
            # CRITICAL: Check current fit_to_window state before resetting
            # If user has zoomed in (fit_to_window = False), preserve that state
            # This prevents zoom state from being lost when navigating from a zoomed image
            if self.fit_to_window:
                # User is in fit-to-window mode, safe to reset
                logger.debug(f"display_pixmap: fit_to_window=True, resetting to fit-to-window")
                self.current_zoom_level = 1.0
                self.zoom_center_point = None
                self.scale_image_to_fit()
            else:
                # User has zoomed in (fit_to_window = False), preserve zoom state
                # This happens when navigating from a zoomed image - the zoom state
                # will be saved by navigate_to_next_image() after load_raw_image() completes
                logger.debug(f"display_pixmap: fit_to_window=False, preserving zoom state (zoom_level={self.current_zoom_level})")
                # Don't reset zoom state - just apply current zoom
                self.apply_zoom_and_pan()
        else:
            if self.fit_to_window:
                self.scale_image_to_fit()
            else:
                self.apply_zoom_and_pan()
            delattr(self, '_maintain_zoom_on_navigation')

        # Handle zoom restoration
        if hasattr(self, '_restore_zoom_center') and self._restore_zoom_center is not None:
            self.fit_to_window = False
            logger.debug(f"display_pixmap: Checking zoom restoration - half_size={getattr(self, '_is_half_size_displayed', False)}, pixmap_size={pixmap.width()}x{pixmap.height()}")
            # If restoring zoom and currently displaying half_size, load full resolution FIRST
            # Also check if the pixmap itself is half_size (for thumbnails that are being replaced)
            pixmap_max_dim = max(pixmap.width(), pixmap.height())
            is_pixmap_half_size = pixmap_max_dim < 5000
            if (hasattr(self, '_is_half_size_displayed') and self._is_half_size_displayed) or is_pixmap_half_size:
                # Update _is_half_size_displayed if pixmap is half_size
                if is_pixmap_half_size:
                    self._is_half_size_displayed = True
                if not hasattr(self, '_full_resolution_loading') or not self._full_resolution_loading:
                    # Check if full resolution is already cached - if so, load it immediately
                    cached_full = self.image_cache.get_full_image(self.current_file_path)
                    if cached_full is not None:
                        cached_max_dim = max(cached_full.shape[1], cached_full.shape[0])
                        if cached_max_dim > 5000:
                            logger.debug("Full resolution image already cached, loading immediately for zoom restoration...")
                            self._full_resolution_loading = True
                            self.display_numpy_image(cached_full)
                            self._is_half_size_displayed = False
                            self._full_resolution_loading = False
                            # Update pixmap reference for zoom calculation
                            self.current_pixmap = self.image_label.pixmap()
                        else:
                            # Start loading full resolution in background
                            self._load_full_resolution_on_demand()
                            # Store zoom restoration intent for when full resolution is ready
                            self._pending_zoom_restore = True
                            self._pending_zoom_center = self._restore_zoom_center
                            self._pending_zoom_level = self._restore_zoom_level
                        self._restore_zoom_center = None
                        self._restore_zoom_level = None
                        return  # Don't restore zoom yet - wait for full resolution
            
            # Restore zoom state
            self.current_zoom_level = self._restore_zoom_level or 1.0
            self.zoom_center_point = self._restore_zoom_center
            # Don't set start_scroll_x/y here - they should be set when panning starts
            self.apply_zoom_and_pan()
            self._restore_zoom_center = None
            self._restore_zoom_level = None


        # Update status bar immediately with EXIF data
        # Don't pass dimensions - let update_status_bar use original dimensions from cache
        self.update_status_bar()

    def _load_full_resolution_on_demand(self):
        """Load full resolution image when user zooms in (on-demand loading)"""
        import logging
        logger = logging.getLogger(__name__)
        
        if not self.current_file_path:
            return
        
        # Check if full resolution is already cached
        cached_full = self.image_cache.get_full_image(self.current_file_path)
        if cached_full is not None:
            # Check if cached image is full resolution (width > 5000px)
            cached_max_dim = max(cached_full.shape[1], cached_full.shape[0])
            if cached_max_dim > 5000:
                logger.info("Full resolution image already cached, loading...")
                self._full_resolution_loading = True
                # Display the full resolution image
                self.display_numpy_image(cached_full)
                self._is_half_size_displayed = False
                self._full_resolution_loading = False
                return
        
        # Check if we're already loading full resolution
        if hasattr(self, '_full_resolution_loading') and self._full_resolution_loading:
            return
        
        logger.info("Loading full resolution image on-demand (user zoomed in)...")
        self._full_resolution_loading = True
        
        # Determine if this is a RAW file based on extension
        raw_extensions = {'.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2', 
                         '.pef', '.srw', '.x3f', '.raf', '.3fr', '.fff', '.iiq', 
                         '.cap', '.erf', '.mef', '.mos', '.nrw', '.rwl', '.srf'}
        file_ext = os.path.splitext(self.current_file_path)[1].lower()
        is_raw = file_ext in raw_extensions
        
        # Start processing full resolution image in background
        processor = RAWProcessor(self.current_file_path, is_raw=is_raw, use_full_resolution=True)
        processor.image_processed.connect(self._on_full_resolution_ready)
        processor.error_occurred.connect(self._on_full_resolution_error)
        processor.start()
        
        # Store reference to prevent garbage collection
        if not hasattr(self, '_full_res_processors'):
            self._full_res_processors = []
        self._full_res_processors.append(processor)
    
    def _on_full_resolution_ready(self, rgb_image):
        """Handle full resolution image ready"""
        import logging
        logger = logging.getLogger(__name__)
        
        if rgb_image is not None:
            # Only process if this is actually a full resolution image (width > 5000px)
            # Ignore thumbnails or half_size images that might be emitted
            # Check if this is full resolution by checking the maximum dimension
            max_dim = max(rgb_image.shape[1], rgb_image.shape[0])
            if max_dim > 5000:
                logger.info(f"Full resolution image ready: {rgb_image.shape[1]}x{rgb_image.shape[0]} (max: {max_dim})")
                # Set flag to prevent display_pixmap from resetting zoom state
                # We're about to apply zoom, so we want to preserve that intent
                self._maintain_zoom_on_navigation = True
                # Set fit_to_window to False and zoom_level to 1.0 before displaying
                # so display_pixmap displays at 100% zoom, then we'll apply zoom after display
                self.fit_to_window = False
                self.current_zoom_level = 1.0
                self.display_numpy_image(rgb_image)
                self._is_half_size_displayed = False
                # Clear the flag after display
                if hasattr(self, '_maintain_zoom_on_navigation'):
                    delattr(self, '_maintain_zoom_on_navigation')
                
                # If there's a pending zoom, execute it now
                if hasattr(self, '_pending_zoom') and self._pending_zoom:
                    logger.info("Executing pending zoom after full resolution image loaded")
                    self._pending_zoom = False
                    # Set up zoom parameters for full resolution image
                    self.fit_to_window = False
                    self.current_zoom_level = 1.0
                    # Use stored zoom center point if available (from double-click), otherwise center on image
                    if hasattr(self, '_pending_zoom_center') and self._pending_zoom_center:
                        # Convert click position to full resolution coordinates
                        old_pixmap = self.image_label.pixmap()
                        if old_pixmap:
                            # Calculate scale factor between old and new image
                            scale_x = self.current_pixmap.width() / old_pixmap.width()
                            scale_y = self.current_pixmap.height() / old_pixmap.height()
                            # Adjust zoom center point
                            self.zoom_center_point = QPoint(
                                int(self._pending_zoom_center.x() * scale_x),
                                int(self._pending_zoom_center.y() * scale_y)
                            )
                            delattr(self, '_pending_zoom_center')
                        else:
                            # Fallback to image center
                            image_center_x = self.current_pixmap.width() // 2
                            image_center_y = self.current_pixmap.height() // 2
                            self.zoom_center_point = QPoint(image_center_x, image_center_y)
                    else:
                        # Center on image center (from space bar)
                        image_center_x = self.current_pixmap.width() // 2
                        image_center_y = self.current_pixmap.height() // 2
                        self.zoom_center_point = QPoint(image_center_x, image_center_y)
                    self.zoom_to_point()
                
                # If there's a pending zoom restoration (from navigation), execute it now
                if hasattr(self, '_pending_zoom_restore') and self._pending_zoom_restore:
                    logger.info("Executing pending zoom restoration after full resolution image loaded")
                    self._pending_zoom_restore = False
                    # Restore zoom state
                    self.fit_to_window = False
                    self.current_zoom_level = self._pending_zoom_level or 1.0
                    self.zoom_center_point = self._pending_zoom_center
                    # Restore scroll position if available
                    if hasattr(self, '_restore_start_scroll_x') and hasattr(self, '_restore_start_scroll_y'):
                        self.start_scroll_x = self._restore_start_scroll_x
                        self.start_scroll_y = self._restore_start_scroll_y
                    # Use zoom_to_point to ensure proper centering
                    if self.zoom_center_point:
                        self.zoom_to_point()
                    else:
                        self.apply_zoom_and_pan()
                    # Clean up
                    if hasattr(self, '_pending_zoom_center'):
                        delattr(self, '_pending_zoom_center')
                    if hasattr(self, '_pending_zoom_level'):
                        delattr(self, '_pending_zoom_level')
                
                # Update status bar once at the end (after all zoom operations)
                self.update_status_bar()
            else:
                logger.debug(f"Ignoring non-full-resolution image in _on_full_resolution_ready: {rgb_image.shape[1]}x{rgb_image.shape[0]}")
                return  # Don't mark as finished if this wasn't the full resolution image
        
        self._full_resolution_loading = False
    
    def _on_full_resolution_error(self, error_msg):
        """Handle full resolution loading error"""
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to load full resolution: {error_msg}")
        self._full_resolution_loading = False

    def _start_preloading(self):
        """Start preloading adjacent images for fast navigation using new architecture."""
        if not self.image_files or self.current_file_index < 0:
            return

        # Use new ImageLoadManager for preloading
        # Next images (higher priority)
        for i in range(1, 4):  # Preload next 3 images
            next_index = (self.current_file_index + i) % len(self.image_files)
            next_file = self.image_files[next_index]
            # Check if already cached
            cached_item, cache_type = check_cache_for_image(next_file, use_full_resolution=False)
            if cached_item is None:
                self.image_manager.load_image(
                    file_path=next_file,
                    priority=Priority.PRELOAD_NEXT,
                    cancel_existing=False,
                    use_full_resolution=False
                )

        # Previous images (lower priority)
        for i in range(1, 3):  # Preload previous 2 images
            prev_index = (self.current_file_index - i) % len(self.image_files)
            prev_file = self.image_files[prev_index]
            # Check if already cached
            cached_item, cache_type = check_cache_for_image(prev_file, use_full_resolution=False)
            if cached_item is None:
                self.image_manager.load_image(
                    file_path=prev_file,
                    priority=Priority.PRELOAD_PREV,
                    cancel_existing=False,
                    use_full_resolution=False
                )
        
        # Legacy preload manager (for backward compatibility)
        # self.preload_manager.preload_images(preload_files, preload_files[:2])

    def _preload_next_image_full(self):
        """Aggressively preload next image's full version in background for instant display"""
        try:
            if not self.image_files or self.current_file_index < 0:
                return
            
            # Get next image
            next_index = (self.current_file_index + 1) % len(self.image_files)
            next_file = self.image_files[next_index]
            
            # Check if already fully cached (both numpy array and QPixmap)
            cached_image = self.image_cache.get_full_image(next_file)
            cached_pixmap = self.image_cache.get_pixmap(next_file)
            
            # If we have full image but no QPixmap, convert it
            if cached_image is not None and cached_pixmap is None:
                try:
                    converter = PixmapConverter(next_file, cached_image, self.image_cache)
                    converter.start()
                    if not hasattr(self, '_pixmap_converters'):
                        self._pixmap_converters = []
                    self._pixmap_converters.append(converter)
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.debug(f"Failed to start pixmap converter for {next_file}: {e}")
            
            # If we don't have full image yet, start processing it in background
            elif cached_image is None:
                try:
                    # Use RAWProcessor (v0.5 style) to preload full image for consistency
                    # Determine if this is a RAW file
                    raw_extensions = {'.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2', 
                                     '.pef', '.srw', '.x3f', '.raf', '.3fr', '.fff', '.iiq', 
                                     '.cap', '.erf', '.mef', '.mos', '.nrw', '.rwl', '.srf'}
                    file_ext = os.path.splitext(next_file)[1].lower()
                    is_raw = file_ext in raw_extensions
                    
                    preload_processor = RAWProcessor(next_file, is_raw=is_raw, use_full_resolution=False)
                    preload_processor.image_processed.connect(
                        lambda img, fp=next_file: self._on_preloaded_image_ready(fp, img))
                    preload_processor.error_occurred.connect(
                        lambda err, fp=next_file: self._on_preload_error(fp, err))
                    preload_processor.start()
                    
                    # Store reference to prevent garbage collection
                    if not hasattr(self, '_preload_processors'):
                        self._preload_processors = []
                    self._preload_processors.append(preload_processor)
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.debug(f"Failed to start preload processor for {next_file}: {e}")
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Error in _preload_next_image_full: {e}")
    
    def _on_preload_error(self, file_path, error_msg):
        """Handle preload error (silent - preloading is background operation)"""
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Preload error for {os.path.basename(file_path)}: {error_msg}")
    
    def _on_preloaded_image_ready(self, file_path, rgb_image):
        """Handle preloaded image - convert to QPixmap in background"""
        try:
            if rgb_image is not None:
                # Convert to QPixmap in background
                converter = PixmapConverter(file_path, rgb_image, self.image_cache)
                converter.start()
                if not hasattr(self, '_pixmap_converters'):
                    self._pixmap_converters = []
                self._pixmap_converters.append(converter)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Error in _on_preloaded_image_ready: {e}")

    def on_image_processed(self, rgb_image):
        import logging
        import traceback
        import time
        logger = logging.getLogger(__name__)
        
        process_start = time.time()
        
        # Defensive check: ensure we're still valid
        try:
            # Check if object is still valid
            if not hasattr(self, 'current_file_path'):
                logger.warning(f"[PROCESS] on_image_processed called but object may be invalid")
                return
        except:
            logger.error(f"[PROCESS] on_image_processed called but object is invalid (access violation risk)")
            return
        
        # Get current file info for logging
        current_file = getattr(self, 'current_file_path', None)
        current_file_basename = os.path.basename(current_file) if current_file else 'N/A'
        
        logger.info(f"[PROCESS] ========== on_image_processed() STARTED at {process_start:.3f} ==========")
        logger.info(f"[PROCESS] Current file: {current_file_basename}")
        
        # Log image info
        if rgb_image is not None:
            image_shape = rgb_image.shape if hasattr(rgb_image, 'shape') else 'unknown'
            image_dtype = rgb_image.dtype if hasattr(rgb_image, 'dtype') else 'unknown'
            logger.info(f"[PROCESS] Image data - shape: {image_shape}, dtype: {image_dtype}")
        else:
            logger.info(f"[PROCESS] Image data is None")
        
        # Check if this signal is for the current file (important for rapid navigation)
        if current_file:
            # Try to get the file path from the processor if available
            processor_file = None
            if hasattr(self, 'current_processor') and self.current_processor is not None:
                processor_file = getattr(self.current_processor, 'file_path', None)
                logger.info(f"[PROCESS] Processor file: {os.path.basename(processor_file) if processor_file else 'None'}")
            
            if processor_file and processor_file != current_file:
                logger.warning(f"[PROCESS] Signal mismatch: processor file ({os.path.basename(processor_file)}) != current file ({current_file_basename}). Skipping processing to avoid displaying wrong image.")
                print(f"[PERF] ⚠️  SKIP PROCESSING: File changed (processor: {os.path.basename(processor_file)}, current: {current_file_basename})")
                # Skip processing - this image is no longer relevant
                return
        
        try:
            if rgb_image is None:
                # Check if this is a RAW file - QPixmap cannot load RAW files directly
                raw_extensions = {'.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2', 
                                 '.pef', '.srw', '.x3f', '.raf', '.3fr', '.fff', '.iiq', 
                                 '.cap', '.erf', '.mef', '.mos', '.nrw', '.rwl', '.srf'}
                file_ext = os.path.splitext(self.current_file_path)[1].lower()
                if file_ext in raw_extensions:
                    # This is expected behavior: RAW processing returned None (possibly cancelled or failed)
                    # QPixmap cannot load RAW files directly, so we skip the fallback
                    # Error handling is done via error_occurred signal, so this is just informational
                    logger.debug(f"RAW file processing returned None, cannot use QPixmap fallback: {os.path.basename(self.current_file_path)}")
                    return
                
                # Non-RAW file: load with safe loader (handles TIFF properly)
                pixmap = self._load_pixmap_safe(self.current_file_path)
                if pixmap.isNull():
                    self.show_error("Display Error",
                                    "Could not load image file.")
                    return

                # Apply orientation correction for non-RAW files
                # CRITICAL: Always get orientation from EXIF, even if cached
                # This ensures orientation is correct on first load and after zoom
                orientation = self.get_orientation_from_exif(
                    self.current_file_path)
                logger.debug(f"[PROCESS] Applying orientation correction to JPEG: {orientation}")
                pixmap = self.apply_orientation_to_pixmap(pixmap, orientation)

                self.current_pixmap = pixmap
                
                # CRITICAL: Check for zoom restoration FIRST before resetting fit_to_window
                # This ensures zoom state is preserved when navigating from a zoomed image
                has_restore_zoom = hasattr(self, '_restore_zoom_center') and self._restore_zoom_center is not None
                has_maintain_zoom = hasattr(self, '_maintain_zoom_on_navigation')
                logger.debug(f"on_image_processed (QPixmap): has_restore_zoom={has_restore_zoom}, has_maintain_zoom={has_maintain_zoom}, current fit_to_window={self.fit_to_window}")
                
                if has_restore_zoom:
                    # Zoom restoration needed - don't reset fit_to_window
                    logger.info(f"on_image_processed (QPixmap): Restoring zoom state - center={self._restore_zoom_center}, level={getattr(self, '_restore_zoom_level', 'N/A')}")
                    self.fit_to_window = False
                    self.current_zoom_level = self._restore_zoom_level or 1.0
                    self.zoom_center_point = self._restore_zoom_center
                    self.start_scroll_x = self.scroll_area.horizontalScrollBar().value()
                    self.start_scroll_y = self.scroll_area.verticalScrollBar().value()
                    self.apply_zoom_and_pan()
                    self._restore_zoom_center = None
                    self._restore_zoom_level = None
                    self._restore_start_scroll_x = None
                    self._restore_start_scroll_y = None
                    # Clean up _maintain_zoom_on_navigation if it exists
                    if has_maintain_zoom:
                        delattr(self, '_maintain_zoom_on_navigation')
                elif not has_maintain_zoom:
                    # No zoom restoration needed and not maintaining zoom - reset to fit-to-window
                    logger.debug(f"on_image_processed (QPixmap): No zoom state to restore, resetting to fit-to-window")
                    self.fit_to_window = True
                    self.current_zoom_level = 1.0
                    self.zoom_center_point = None
                    self.scale_image_to_fit()
                else:
                    # Maintaining zoom state from navigation
                    logger.debug(f"on_image_processed (QPixmap): Maintaining zoom state from navigation, fit_to_window={self.fit_to_window}")
                    if self.fit_to_window:
                        self.scale_image_to_fit()
                    else:
                        self.apply_zoom_and_pan()
                    delattr(self, '_maintain_zoom_on_navigation')
                
                if self.current_file_path:
                    self.scan_folder_for_images(self.current_file_path)
                # Don't pass dimensions - let update_status_bar use original dimensions from cache
                self.update_status_bar()
            else:
                # RAW: successful processing with numpy array
                try:
                    height, width, channels = rgb_image.shape
                    bytes_per_line = channels * width

                    # Ensure the data is contiguous and convert to bytes for PyQt6 compatibility
                    if not rgb_image.flags['C_CONTIGUOUS']:
                        rgb_image = np.ascontiguousarray(rgb_image)

                    # Convert to bytes if needed (PyQt6 compatibility)
                    # Check if this is half_size image (for on-demand full resolution loading)
                    # Detect half_size by checking if dimensions are approximately half of expected full resolution
                    # Typical full resolution: 6000-8000px in largest dimension, half_size: 3000-4000px
                    # Check both width and height to handle both portrait and landscape orientations
                    max_dimension = max(width, height)
                    is_half_size = max_dimension < 5000  # Assume full resolution is typically >5000px in largest dimension
                    self._is_half_size_displayed = is_half_size
                    import logging
                    logger = logging.getLogger(__name__)
                    if is_half_size:
                        logger.debug(f"Detected half_size image: {width}x{height} (max: {max_dimension}), will load full resolution on zoom")
                    else:
                        logger.info(f"Detected full resolution image: {width}x{height} (max: {max_dimension})")
                    
                    # Check if we need to restore zoom - if so, skip half_size display and load full resolution directly
                    if hasattr(self, '_restore_zoom_center') and self._restore_zoom_center is not None:
                        if is_half_size:
                            logger.debug(f"Zoom restoration needed: center={self._restore_zoom_center}, level={getattr(self, '_restore_zoom_level', 'N/A')}, skipping half_size display")
                            if not hasattr(self, '_full_resolution_loading') or not self._full_resolution_loading:
                                # Check if full resolution is already cached
                                cached_full = self.image_cache.get_full_image(self.current_file_path)
                                if cached_full is not None:
                                    cached_max_dim = max(cached_full.shape[1], cached_full.shape[0])
                                    if cached_max_dim > 5000:
                                        logger.debug("Full resolution image already cached, loading immediately for zoom restoration...")
                                        self._full_resolution_loading = True
                                        self.display_numpy_image(cached_full)
                                        self._is_half_size_displayed = False
                                        self._full_resolution_loading = False
                                        return
                                else:
                                    # Start loading full resolution in background
                                    logger.debug("Starting full resolution load for zoom restoration - skipping half_size display")
                                    self._load_full_resolution_on_demand()
                                    # Store zoom restoration intent
                                    self._pending_zoom_restore = True
                                    self._pending_zoom_center = self._restore_zoom_center
                                    self._pending_zoom_level = self._restore_zoom_level
                                    self._restore_zoom_center = None
                                    self._restore_zoom_level = None
                                    # Don't display half_size image - wait for full resolution
                                    return
                    
                    # Also check if we're already loading full resolution - if so, don't display half_size
                    if hasattr(self, '_full_resolution_loading') and self._full_resolution_loading:
                        logger.debug("Full resolution loading in progress, skipping half_size display")
                        return
                    
                    # Also check if we have pending zoom restore - if so, don't display half_size
                    if hasattr(self, '_pending_zoom_restore') and self._pending_zoom_restore:
                        logger.debug("Pending zoom restore, skipping half_size display")
                        return
                    
                    # Check if we have a cached pixmap first (faster path)
                    # BUT: if this is a full resolution image, don't use cached half_size pixmap
                    cached_pixmap = self.image_cache.get_pixmap(self.current_file_path)
                    if cached_pixmap is not None:
                        # Check if cached pixmap matches the current image size
                        # If current image is full resolution but cached pixmap is small, convert new one
                        # Check if cached pixmap is smaller than current image (use max dimension)
                        cached_max_dim = max(cached_pixmap.width(), cached_pixmap.height())
                        current_max_dim = max(width, height)
                        if not is_half_size and cached_max_dim < current_max_dim:
                            logger.info(f"[PROCESS] Cached pixmap is small ({cached_pixmap.width()}x{cached_pixmap.height()}, max: {cached_max_dim}) but current image is full resolution ({width}x{height}, max: {current_max_dim}), converting new pixmap")
                            # Convert to QPixmap for full resolution image
                            image_data = rgb_image.data.tobytes() if hasattr(
                                rgb_image.data, 'tobytes') else bytes(rgb_image.data)

                            q_image = QImage(image_data, width, height,
                                             bytes_per_line, QImage.Format.Format_RGB888)
                            pixmap = QPixmap.fromImage(q_image)
                            self.current_pixmap = pixmap
                            
                            # Cache the new full resolution pixmap (replace the old small one)
                            self.image_cache.put_pixmap(self.current_file_path, pixmap)
                            logger.info(f"[PROCESS] Full resolution pixmap cached: {pixmap.width()}x{pixmap.height()}")
                        else:
                            logger.debug("Using cached QPixmap for faster display")
                            pixmap = cached_pixmap
                            # Check if cached pixmap is half_size
                            pixmap_max_dim = max(pixmap.width(), pixmap.height())
                            if pixmap_max_dim < 5000:
                                logger.debug(f"Cached pixmap is half_size: {pixmap.width()}x{pixmap.height()}, will load full resolution on zoom")
                            self.current_pixmap = pixmap
                    else:
                        # Convert to QPixmap
                        logger.info(f"[PROCESS] Converting full resolution image to QPixmap: {width}x{height}")
                        image_data = rgb_image.data.tobytes() if hasattr(
                            rgb_image.data, 'tobytes') else bytes(rgb_image.data)

                        q_image = QImage(image_data, width, height,
                                         bytes_per_line, QImage.Format.Format_RGB888)
                        pixmap = QPixmap.fromImage(q_image)
                        
                        # CRITICAL: Apply orientation correction to pixmap before caching
                        # This ensures cached pixmaps have correct orientation
                        orientation = self.get_orientation_from_exif(self.current_file_path)
                        logger.info(f"[PROCESS] Applying orientation correction to full resolution pixmap: {orientation}")
                        pixmap = self.apply_orientation_to_pixmap(pixmap, orientation)
                        
                        self.current_pixmap = pixmap
                        
                        # Cache the pixmap for future use (faster than numpy->QPixmap conversion)
                        # This ensures next time we load this image, we can skip conversion
                        self.image_cache.put_pixmap(self.current_file_path, pixmap)
                        logger.info(f"[PROCESS] Full resolution pixmap cached: {pixmap.width()}x{pixmap.height()}")
                    
                    # Check if we need to restore zoom - if so, skip half_size display and load full resolution directly
                    if hasattr(self, '_restore_zoom_center') and self._restore_zoom_center is not None:
                        if is_half_size:
                            logger.debug(f"Zoom restoration needed: skipping half_size display (converted from numpy)")
                            if not hasattr(self, '_full_resolution_loading') or not self._full_resolution_loading:
                                # Check if full resolution is already cached
                                cached_full = self.image_cache.get_full_image(self.current_file_path)
                                if cached_full is not None:
                                    cached_max_dim = max(cached_full.shape[1], cached_full.shape[0])
                                    if cached_max_dim > 5000:
                                        logger.debug("Full resolution image already cached, loading immediately for zoom restoration...")
                                        self._full_resolution_loading = True
                                        self.display_numpy_image(cached_full)
                                        self._is_half_size_displayed = False
                                        self._full_resolution_loading = False
                                        return
                                else:
                                    # Start loading full resolution in background
                                    logger.debug("Starting full resolution load for zoom restoration - skipping half_size display")
                                    self._load_full_resolution_on_demand()
                                    # Store zoom restoration intent
                                    self._pending_zoom_restore = True
                                    self._pending_zoom_center = self._restore_zoom_center
                                    self._pending_zoom_level = self._restore_zoom_level
                                    self._restore_zoom_center = None
                                    self._restore_zoom_level = None
                                    # Don't display half_size image - wait for full resolution
                                    return
                    
                    # Use display_pixmap to handle zoom restoration if needed
                    self.display_pixmap(pixmap)
                    
                    # Start aggressive preloading: pre-process next image's full version in background
                    # This ensures next image is ready when user navigates
                    from PyQt6.QtCore import QTimer
                    QTimer.singleShot(100, lambda: self._preload_next_image_full())
                    
                    # Calculate total time from navigation to display
                    # Update status bar to show original dimensions from cache
                    self.update_status_bar()
                    
                    if hasattr(self, '_last_navigation_start'):
                        total_time = time.time() - self._last_navigation_start
                        logger.info(f"RAW image displayed successfully: {width}x{height} (total from navigation: {total_time:.3f}s)")
                        print(f"[PERF] 🖼️  IMAGE DISPLAYED: {width}x{height} (total navigation time: {total_time*1000:.1f}ms)")
                    else:
                        logger.info(f"RAW image displayed successfully: {width}x{height}")
                        print(f"[PERF] 🖼️  IMAGE DISPLAYED: {width}x{height}")
                except Exception as e:
                    import logging
                    import traceback
                    logger = logging.getLogger(__name__)
                    logger.error(f"Error processing RAW image in on_image_processed: {e}", exc_info=True)
                    logger.debug(f"RAW image processing error traceback: {traceback.format_exc()}")
                    # Try to show error to user
                    try:
                        self.show_error("Display Error", f"Error processing RAW image: {str(e)}")
                    except Exception as show_error_ex:
                        logger.error(f"Error showing error message: {show_error_ex}")
        except Exception as e:
            import traceback
            logger.error(f"Critical error in on_image_processed: {e}", exc_info=True)
            logger.debug(f"on_image_processed error traceback: {traceback.format_exc()}")
            error_msg = f"Error displaying image: {str(e)}"
            try:
                self.show_error("Display Error", error_msg)
            except Exception as show_error_ex:
                logger.error(f"Error showing error message: {show_error_ex}")
        finally:
            try:
                self.setFocus()
            except Exception as focus_error:
                logger.warning(f"Error setting focus: {focus_error}")

    def on_processing_error(self, error_message):
        """Handle RAW processing errors"""
        import logging
        import traceback
        logger = logging.getLogger(__name__)
        
        current_file = getattr(self, 'current_file_path', 'unknown')
        logger.error(f"on_processing_error called for: {os.path.basename(current_file)}, error: {error_message}")
        
        # Check if this error is for the current file (important for rapid navigation)
        if hasattr(self, 'current_processor') and self.current_processor is not None:
            processor_file = getattr(self.current_processor, 'file_path', None)
            if processor_file and processor_file != current_file:
                logger.warning(f"Error signal mismatch: processor file ({os.path.basename(processor_file)}) != current file ({os.path.basename(current_file)}). This may indicate rapid navigation.")
        
        try:
            # If we have a pending thumbnail and full processing failed, show it as fallback
            if hasattr(self, '_pending_thumbnail') and self._pending_thumbnail is not None:
                logger.debug("Using pending thumbnail as fallback")
                try:
                    self.display_numpy_image(self._pending_thumbnail)
                    self.status_bar.showMessage(
                        "⚠️ Using preview - full processing failed")
                    self._pending_thumbnail = None
                    return
                except Exception as display_error:
                    logger.error(f"Error displaying pending thumbnail: {display_error}", exc_info=True)

            error_msg = f"Error processing RAW file:\n{error_message}"
            logger.debug(f"Showing error message to user: {error_msg}")
            try:
                self.show_error("RAW Processing Error", error_msg)
            except Exception as show_error_ex:
                logger.error(f"Error showing error dialog: {show_error_ex}")
            
            try:
                self.image_label.setText(
                    "Error loading image\n\nPlease try a different RAW file"
                )
                self.status_bar.showMessage("Error loading image")
                # Reset window title on error
                self.setWindowTitle('RAW Image Viewer')
                # Update custom title bar
                if hasattr(self, 'title_bar'):
                    self.title_bar.set_title('RAW Image Viewer')
            except Exception as ui_error:
                logger.error(f"Error updating UI on processing error: {ui_error}")
        except Exception as e:
            logger.error(f"Critical error in on_processing_error: {e}", exc_info=True)
            logger.debug(f"on_processing_error error traceback: {traceback.format_exc()}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.toggle_zoom()
            event.accept()  # Mark event as handled
        elif event.key() == Qt.Key.Key_Left:
            self._debounced_navigate('prev')
            event.accept()  # Mark event as handled
        elif event.key() == Qt.Key.Key_Right:
            self._debounced_navigate('next')
            event.accept()  # Mark event as handled
        elif event.key() == Qt.Key.Key_Down:
            self.move_current_image_to_discard()
            event.accept()  # Mark event as handled
        elif event.key() == Qt.Key.Key_Up:
            # Prevent up arrow from moving the scroll area
            event.accept()
        elif event.key() == Qt.Key.Key_Delete:
            self.delete_current_image()
            event.accept()  # Mark event as handled
        else:
            super().keyPressEvent(event)

    def can_navigate(self):
        """Check if navigation is allowed (prevents overlapping navigations and rate limiting)"""
        import logging
        import time
        logger = logging.getLogger(__name__)
        
        nav_in_progress = getattr(self, '_navigation_in_progress', False)
        last_nav_time = getattr(self, '_last_navigation_time', 0)
        current_time = time.time()
        time_since_last = current_time - last_nav_time if last_nav_time > 0 else float('inf')
        
        logger.debug(f"[NAV_CHECK] can_navigate() called - nav_in_progress={nav_in_progress}, "
                    f"last_nav_time={last_nav_time:.3f}, current_time={current_time:.3f}, "
                    f"time_since_last={time_since_last:.3f}s")
        
        # Check if navigation is in progress
        # Note: We removed the 100ms rate limiting cooldown because:
        # 1. The _navigation_in_progress flag already prevents overlapping navigations
        # 2. The rate limiting was causing navigation to be blocked unnecessarily
        # 3. Users should be able to navigate quickly if the previous navigation completed
        if nav_in_progress:
            logger.warning(f"[NAV_CHECK] Navigation BLOCKED: navigation already in progress")
            print(f"[PERF] 🚫 NAVIGATION BLOCKED: Already in progress")
            return False
        
        logger.debug(f"[NAV_CHECK] Navigation ALLOWED")
        return True
    
    def _debounced_navigate(self, direction):
        """Debounced navigation to handle rapid key presses efficiently"""
        import logging
        from PyQt6.QtCore import QTimer
        logger = logging.getLogger(__name__)
        
        # Store the navigation direction
        had_pending = self._pending_navigation is not None
        self._pending_navigation = direction
        
        # Cancel any existing timer
        if self._navigation_timer is not None:
            self._navigation_timer.stop()
            if had_pending:
                print(f"[PERF] 🔄 DEBOUNCE: Cancelled previous navigation, queued new {direction} request")
        
        # Create a new timer with short delay (50ms) to batch rapid key presses
        # This allows users to press keys rapidly, but only the last navigation within 50ms will execute
        self._navigation_timer = QTimer()
        self._navigation_timer.setSingleShot(True)
        self._navigation_timer.timeout.connect(lambda: self._execute_pending_navigation())
        self._navigation_timer.start(50)  # 50ms debounce delay
        
        logger.debug(f"[NAV_DEBOUNCE] Navigation request queued: {direction}")
        if not had_pending:
            print(f"[PERF] ⏱️  DEBOUNCE: Navigation {direction} queued (50ms delay)")
    
    def _execute_pending_navigation(self):
        """Execute the pending navigation after debounce delay"""
        import logging
        import time
        logger = logging.getLogger(__name__)
        
        if self._pending_navigation is None:
            return
        
        direction = self._pending_navigation
        self._pending_navigation = None
        self._navigation_timer = None
        
        logger.debug(f"[NAV_DEBOUNCE] Executing pending navigation: {direction}")
        nav_start = time.time()
        print(f"[PERF] ▶️  EXECUTING: Navigation {direction} (after debounce)")
        
        if direction == 'prev':
            self.navigate_to_previous_image()
        elif direction == 'next':
            self.navigate_to_next_image()
        
        nav_time = time.time() - nav_start
        print(f"[PERF] ✅ NAVIGATION COMPLETE: {direction} took {nav_time*1000:.1f}ms")
    
    def start_navigation(self):
        """Mark navigation as started"""
        import time
        import logging
        logger = logging.getLogger(__name__)
        current_time = time.time()
        old_state = getattr(self, '_navigation_in_progress', False)
        logger.info(f"[NAV_START] start_navigation() called - old_state={old_state}, setting to True, "
                   f"time={current_time:.3f}")
        self._navigation_in_progress = True
        self._last_navigation_time = current_time
        logger.debug(f"[NAV_START] Navigation flag set - _navigation_in_progress={self._navigation_in_progress}, "
                    f"_last_navigation_time={self._last_navigation_time:.3f}")
    
    def finish_navigation(self):
        """Mark navigation as finished"""
        import logging
        import time
        import traceback
        logger = logging.getLogger(__name__)
        old_state = getattr(self, '_navigation_in_progress', None)
        current_time = time.time()
        last_nav_time = getattr(self, '_last_navigation_time', 0)
        nav_duration = current_time - last_nav_time if last_nav_time > 0 else 0
        
        logger.info(f"[NAV_FINISH] finish_navigation() called - old_state={old_state}, will set to False, "
                   f"nav_duration={nav_duration:.3f}s, time={current_time:.3f}")
        logger.debug(f"[NAV_FINISH] Call stack:\n{traceback.format_stack()[-5:-1]}")
        
        self._navigation_in_progress = False
        logger.debug(f"[NAV_FINISH] Navigation flag cleared - _navigation_in_progress={self._navigation_in_progress}")

    def navigate_to_previous_image(self):
        import logging
        import traceback
        import time
        import os
        logger = logging.getLogger(__name__)
        
        nav_start_time = time.time()
        logger.info(f"[NAV_PREV] ========== navigate_to_previous_image() STARTED at {nav_start_time:.3f} ==========")
        logger.debug(f"[NAV_PREV] Current state - index: {self.current_file_index}, "
                    f"total_files: {len(self.image_files) if self.image_files else 0}, "
                    f"current_file: {os.path.basename(self.current_file_path) if self.current_file_path else 'None'}")
        
        # Check processor state
        processor_state = "None"
        if hasattr(self, 'current_processor') and self.current_processor:
            processor_state = f"Active (thread_id={self.current_processor.thread().currentThreadId() if hasattr(self.current_processor, 'thread') else 'N/A'})"
        logger.debug(f"[NAV_PREV] Current processor state: {processor_state}")
        
        # Check if navigation is allowed BEFORE starting navigation
        if not self.can_navigate():
            logger.warning(f"[NAV_PREV] Navigation BLOCKED by can_navigate() check")
            return
        
        # Mark navigation as started - must be done before any return statements
        self.start_navigation()
        
        try:
            try:
                if not self.image_files or len(self.image_files) <= 1:
                    logger.debug("Cannot navigate: no files or only one file")
                    return

                # Calculate previous index with wraparound
                old_index = self.current_file_index
                if self.current_file_index <= 0:
                    self.current_file_index = len(self.image_files) - 1
                else:
                    self.current_file_index -= 1

                logger.info(f"[NAV_PREV] Navigating to previous image - old_index: {old_index}, new_index: {self.current_file_index}")
                
                # Check if new index is valid
                if self.current_file_index < 0 or self.current_file_index >= len(self.image_files):
                    logger.error(f"[NAV_PREV] Invalid index after navigation: {self.current_file_index}, total_files: {len(self.image_files)}")
                    self.current_file_index = old_index  # Restore old index
                    return
                
                # Only maintain zoom state if not in fit-to-window mode
                if not self.fit_to_window:
                    logger.debug("Maintaining zoom state for navigation")
                    self._maintain_zoom_on_navigation = True
                    self._restore_zoom_center = self.zoom_center_point
                    self._restore_zoom_level = self.current_zoom_level
                    # Save current scroll position instead of start_scroll_x/y
                    try:
                        current_scroll_x = self.scroll_area.horizontalScrollBar().value()
                        current_scroll_y = self.scroll_area.verticalScrollBar().value()
                        self._restore_start_scroll_x = current_scroll_x
                        self._restore_start_scroll_y = current_scroll_y
                        logger.debug(f"Saved scroll position: x={current_scroll_x}, y={current_scroll_y}")
                    except Exception as scroll_error:
                        logger.warning(f"Error getting scroll position: {scroll_error}")
                        self._restore_start_scroll_x = 0
                        self._restore_start_scroll_y = 0
                else:
                    logger.debug("Not maintaining zoom state (fit-to-window mode)")
                    if hasattr(self, '_maintain_zoom_on_navigation'):
                        delattr(self, '_maintain_zoom_on_navigation')
                    self._restore_zoom_center = None
                    self._restore_zoom_level = None
                    self._restore_start_scroll_x = None
                    self._restore_start_scroll_y = None

                # Load the current image (at new_index)
                current_file = self.image_files[self.current_file_index]
                logger.debug(f"Loading file at index {self.current_file_index}: {os.path.basename(current_file)}")
                
                try:
                    self.load_raw_image(current_file)
                    logger.debug(f"Successfully called load_raw_image for: {os.path.basename(current_file)}")
                except Exception as load_error:
                    logger.error(f"Error in load_raw_image during navigation: {load_error}", exc_info=True)
                    logger.debug(f"Load error traceback: {traceback.format_exc()}")
                    # Restore previous index on error
                    self.current_file_index = old_index
                    raise
                
                try:
                    self.save_session_state()
                    logger.debug("Session state saved")
                except Exception as save_error:
                    logger.warning(f"Error saving session state: {save_error}")
                
            finally:
                # Always mark navigation as finished (inner try)
                inner_finally_time = time.time()
                logger.debug(f"[NAV_PREV] Inner finally block reached at {inner_finally_time:.3f} "
                           f"(nav duration: {inner_finally_time - nav_start_time:.3f}s)")
                self.finish_navigation()
                
        except Exception as e:
            error_time = time.time()
            logger.error(f"[NAV_PREV] ========== EXCEPTION in navigate_to_previous_image "
                        f"(at {error_time:.3f}, duration: {error_time - nav_start_time:.3f}s) ==========")
            logger.error(f"[NAV_PREV] Exception type: {type(e).__name__}, message: {e}", exc_info=True)
            logger.error(f"[NAV_PREV] Full traceback:\n{traceback.format_exc()}")
            # Ensure navigation is marked as finished even on error
            try:
                self.finish_navigation()
            except Exception as finish_error:
                logger.error(f"[NAV_PREV] Error in finish_navigation during exception handling: {finish_error}")
            raise
        finally:
            # Additional safety: ensure navigation is always marked as finished (outer finally)
            outer_finally_time = time.time()
            logger.debug(f"[NAV_PREV] Outer finally block reached at {outer_finally_time:.3f} "
                       f"(total duration: {outer_finally_time - nav_start_time:.3f}s)")
            if hasattr(self, '_navigation_in_progress') and self._navigation_in_progress:
                logger.warning(f"[NAV_PREV] Navigation flag still True in outer finally, clearing it")
                self.finish_navigation()
            logger.info(f"[NAV_PREV] ========== navigate_to_previous_image() COMPLETED "
                       f"(total duration: {outer_finally_time - nav_start_time:.3f}s) ==========")

    def navigate_to_next_image(self):
        import logging
        import time
        import traceback
        import os
        logger = logging.getLogger(__name__)
        
        nav_start_time = time.time()
        # Store navigation start time immediately for tracking total time to display
        self._last_navigation_start = nav_start_time
        
        logger.info(f"[NAV_NEXT] ========== navigate_to_next_image() STARTED at {nav_start_time:.3f} ==========")
        logger.debug(f"[NAV_NEXT] Current state - index: {self.current_file_index}, "
                    f"total_files: {len(self.image_files) if self.image_files else 0}, "
                    f"current_file: {os.path.basename(self.current_file_path) if self.current_file_path else 'None'}")
        
        # Check processor state
        processor_state = "None"
        if hasattr(self, 'current_processor') and self.current_processor:
            processor_state = f"Active (thread_id={self.current_processor.thread().currentThreadId() if hasattr(self.current_processor, 'thread') else 'N/A'})"
        logger.debug(f"[NAV_NEXT] Current processor state: {processor_state}")
        
        # Check if navigation is allowed BEFORE starting navigation
        if not self.can_navigate():
            logger.warning(f"[NAV_NEXT] Navigation BLOCKED by can_navigate() check")
            return
        
        # Mark navigation as started - must be done before any return statements
        self.start_navigation()
        
        try:
            try:
                if not self.image_files or len(self.image_files) <= 1:
                    logger.debug("Cannot navigate: no files or only one file")
                    return

                # Calculate next index with wraparound
                old_index = self.current_file_index
                if self.current_file_index >= len(self.image_files) - 1:
                    self.current_file_index = 0
                else:
                    self.current_file_index += 1

                logger.info(f"[NAV_NEXT] Navigating to next image - old_index: {old_index}, new_index: {self.current_file_index}")
                
                # Check if new index is valid
                if self.current_file_index < 0 or self.current_file_index >= len(self.image_files):
                    logger.error(f"[NAV_NEXT] Invalid index after navigation: {self.current_file_index}, total_files: {len(self.image_files)}")
                    self.current_file_index = old_index  # Restore old index
                    return
                
                # Only maintain zoom state if not in fit-to-window mode
                if not self.fit_to_window:
                    logger.debug("Maintaining zoom state for navigation")
                    self._maintain_zoom_on_navigation = True
                    self._restore_zoom_center = self.zoom_center_point
                    self._restore_zoom_level = self.current_zoom_level
                    # Save current scroll position instead of start_scroll_x/y
                    try:
                        current_scroll_x = self.scroll_area.horizontalScrollBar().value()
                        current_scroll_y = self.scroll_area.verticalScrollBar().value()
                        self._restore_start_scroll_x = current_scroll_x
                        self._restore_start_scroll_y = current_scroll_y
                        logger.debug(f"Saved scroll position: x={current_scroll_x}, y={current_scroll_y}")
                    except Exception as scroll_error:
                        logger.warning(f"Error getting scroll position: {scroll_error}")
                        self._restore_start_scroll_x = 0
                        self._restore_start_scroll_y = 0
                else:
                    logger.debug("Not maintaining zoom state (fit-to-window mode)")
                    if hasattr(self, '_maintain_zoom_on_navigation'):
                        delattr(self, '_maintain_zoom_on_navigation')
                    self._restore_zoom_center = None
                    self._restore_zoom_level = None
                    self._restore_start_scroll_x = None
                    self._restore_start_scroll_y = None

                # Load the current image (at new_index)
                current_file = self.image_files[self.current_file_index]
                load_start_time = time.time()
                logger.info(f"[NAV_NEXT] Loading file at index {self.current_file_index}: {os.path.basename(current_file)}")
                logger.debug(f"[NAV_NEXT] File path: {current_file}")
                logger.debug(f"[NAV_NEXT] Time since nav start: {load_start_time - nav_start_time:.3f}s")
                
                try:
                    self.load_raw_image(current_file)
                    load_end_time = time.time()
                    logger.info(f"[NAV_NEXT] Successfully called load_raw_image for: {os.path.basename(current_file)} "
                               f"(took {load_end_time - load_start_time:.3f}s)")
                except Exception as load_error:
                    load_end_time = time.time()
                    logger.error(f"[NAV_NEXT] ERROR in load_raw_image during navigation (took {load_end_time - load_start_time:.3f}s): "
                               f"{load_error}", exc_info=True)
                    logger.error(f"[NAV_NEXT] Load error traceback:\n{traceback.format_exc()}")
                    # Restore previous index on error
                    self.current_file_index = old_index
                    raise
                
                try:
                    self.save_session_state()
                    logger.debug("[NAV_NEXT] Session state saved")
                except Exception as save_error:
                    logger.warning(f"[NAV_NEXT] Error saving session state: {save_error}")
                
            finally:
                # Always mark navigation as finished (inner try)
                inner_finally_time = time.time()
                logger.debug(f"[NAV_NEXT] Inner finally block reached at {inner_finally_time:.3f} "
                           f"(nav duration: {inner_finally_time - nav_start_time:.3f}s)")
                self.finish_navigation()
                
        except Exception as e:
            error_time = time.time()
            logger.error(f"[NAV_NEXT] ========== EXCEPTION in navigate_to_next_image "
                        f"(at {error_time:.3f}, duration: {error_time - nav_start_time:.3f}s) ==========")
            logger.error(f"[NAV_NEXT] Exception type: {type(e).__name__}, message: {e}", exc_info=True)
            logger.error(f"[NAV_NEXT] Full traceback:\n{traceback.format_exc()}")
            # Ensure navigation is marked as finished even on error
            try:
                self.finish_navigation()
            except Exception as finish_error:
                logger.error(f"[NAV_NEXT] Error in finish_navigation during exception handling: {finish_error}")
            raise
        finally:
            # Additional safety: ensure navigation is always marked as finished (outer finally)
            outer_finally_time = time.time()
            logger.debug(f"[NAV_NEXT] Outer finally block reached at {outer_finally_time:.3f} "
                       f"(total duration: {outer_finally_time - nav_start_time:.3f}s)")
            if hasattr(self, '_navigation_in_progress') and self._navigation_in_progress:
                logger.warning(f"[NAV_NEXT] Navigation flag still True in outer finally, clearing it")
                self.finish_navigation()
            logger.info(f"[NAV_NEXT] ========== navigate_to_next_image() COMPLETED "
                       f"(total duration: {outer_finally_time - nav_start_time:.3f}s) ==========")

    def delete_current_image(self):
        if (not self.current_file_path or not os.path.exists(self.current_file_path)):
            self.show_error("Delete Error", "No image file to delete.")
            return

        if self.confirm_deletion():
            # Only maintain zoom state if not in fit-to-window mode
            if not self.fit_to_window:
                self._maintain_zoom_on_navigation = True
                self._restore_zoom_center = self.zoom_center_point
                self._restore_zoom_level = self.current_zoom_level
                # Save current scroll position instead of start_scroll_x/y
                self._restore_start_scroll_x = self.scroll_area.horizontalScrollBar().value()
                self._restore_start_scroll_y = self.scroll_area.verticalScrollBar().value()
            else:
                if hasattr(self, '_maintain_zoom_on_navigation'):
                    delattr(self, '_maintain_zoom_on_navigation')
                self._restore_zoom_center = None
                self._restore_zoom_level = None
                self._restore_start_scroll_x = None
                self._restore_start_scroll_y = None
            self.perform_deletion()
        self.save_session_state()

    def confirm_deletion(self):
        """Show confirmation dialog for file deletion with custom MD3 design"""
        filename = os.path.basename(self.current_file_path)
        
        dialog = CustomConfirmDialog(
            parent=self,
            title="Confirm Delete",
            message="Are you sure you want to delete this file?",
            informative_text=f"File: {filename}\n\nThis will move the file to the Recycle Bin."
        )
        
        result = dialog.exec()
        return dialog.result_value

    def perform_deletion(self):
        """Perform the actual file deletion"""
        try:
            file_to_delete = self.current_file_path
            filename = os.path.basename(file_to_delete)

            # Normalize the file path to handle UNC paths and other issues
            normalized_path = os.path.normpath(file_to_delete)

            # Before deleting, cancel any preload tasks for this file and clear cache
            if hasattr(self, 'preload_manager'):
                # Cancel preload for this specific file if it's in the active threads
                if file_to_delete in self.preload_manager.active_threads:
                    thread = self.preload_manager.active_threads[file_to_delete]
                    try:
                        if hasattr(thread, 'cleanup'):
                            thread.cleanup()
                        elif hasattr(thread, 'stop_processing'):
                            thread.stop_processing()
                            thread.quit()
                            thread.wait(100)
                    except Exception as e:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.debug(f"Error cancelling preload for {filename}: {e}")
                    finally:
                        if file_to_delete in self.preload_manager.active_threads:
                            del self.preload_manager.active_threads[file_to_delete]
            
            # Clear cache for this file
            from image_cache import get_image_cache
            cache = get_image_cache()
            cache.invalidate_file(file_to_delete)

            # Move file to trash using send2trash
            # Lazy import send2trash to avoid import delays
            from send2trash import send2trash
            send2trash(normalized_path)

            # Remove from image files list
            if file_to_delete in self.image_files:
                self.image_files.remove(file_to_delete)

            # Update status
            self.status_bar.showMessage(f"Deleted: {filename}")

            # Handle navigation after deletion
            self.handle_post_deletion_navigation()

        except Exception as e:
            error_msg = f"Could not delete file:\n{str(e)}"
            self.show_error("Delete Error", error_msg)

    def handle_post_deletion_navigation(self):
        """Handle navigation after a file has been deleted"""
        if not self.image_files:
            # No more images in folder
            self.current_file_path = None
            self.current_file_index = -1
            self.current_pixmap = None
            self.image_label.setText(
                "No more images in this folder\n\n"
                "Use File > Open to load another image"
            )
            self.status_bar.showMessage("No images remaining in folder")
            self.setWindowTitle('RAW Image Viewer')
            # Update custom title bar
            if hasattr(self, 'title_bar'):
                self.title_bar.set_title('RAW Image Viewer')
            return

        # Adjust current index if needed
        if self.current_file_index >= len(self.image_files):
            self.current_file_index = len(self.image_files) - 1

        # Load the next image (or previous if we were at the end)
        if self.current_file_index >= 0:
            self.load_raw_image(self.image_files[self.current_file_index])

    def toggle_zoom(self):
        """Toggle between fit-to-window and 100% zoom modes"""
        import logging
        logger = logging.getLogger(__name__)
        if not self.current_pixmap:
            return
        if self.fit_to_window:
            # Switch to 100% zoom mode - center on image center
            self.fit_to_window = False
            
            # Check if we're displaying a thumbnail or half_size image
            # Compare current pixmap size with original dimensions from cache
            should_load_full_resolution = False
            if self.current_pixmap:
                # Get original dimensions from cache
                cached_exif = self.image_cache.get_exif(self.current_file_path)
                if cached_exif and cached_exif.get('original_width') and cached_exif.get('original_height'):
                    original_width = cached_exif['original_width']
                    original_height = cached_exif['original_height']
                    current_max = max(self.current_pixmap.width(), self.current_pixmap.height())
                    original_max = max(original_width, original_height)
                    # If current display is significantly smaller than original, load full resolution
                    if current_max < original_max * 0.8:
                        should_load_full_resolution = True
                        logger.info(f"User zoomed in - current display ({self.current_pixmap.width()}x{self.current_pixmap.height()}) is smaller than original ({original_width}x{original_height}), loading full resolution")
                elif hasattr(self, '_is_half_size_displayed') and self._is_half_size_displayed:
                    # Fallback: use the flag if original dimensions not available
                    should_load_full_resolution = True
                    logger.info("User zoomed in - _is_half_size_displayed is True, loading full resolution")
            
            # If currently displaying thumbnail/half_size and user zooms in, load full resolution FIRST
            if should_load_full_resolution:
                if not hasattr(self, '_full_resolution_loading') or not self._full_resolution_loading:
                    # Check if full resolution is already cached - if so, load it immediately
                    cached_full = self.image_cache.get_full_image(self.current_file_path)
                    if cached_full is not None:
                        cached_max_dim = max(cached_full.shape[1], cached_full.shape[0])
                        if cached_max_dim > 5000:
                            logger.info("Full resolution image already cached, loading immediately...")
                            self._full_resolution_loading = True
                            # Set flag to prevent display_pixmap from resetting fit_to_window
                            # We're about to zoom in, so we want to preserve that intent
                            self._maintain_zoom_on_navigation = True
                            self.display_numpy_image(cached_full)
                            self._is_half_size_displayed = False
                            self._full_resolution_loading = False
                            # Update pixmap reference for zoom calculation
                            self.current_pixmap = self.image_label.pixmap()
                            # Clear the flag after display
                            if hasattr(self, '_maintain_zoom_on_navigation'):
                                delattr(self, '_maintain_zoom_on_navigation')
                            # Continue to zoom setup below
                        else:
                            # Cached image is also half_size, need to process full resolution
                            logger.info("Cached image is half_size, processing full resolution...")
                            self._load_full_resolution_on_demand()
                            # Don't zoom yet - wait for full resolution to load
                            # Store zoom intent for when full resolution is ready
                            self._pending_zoom = True
                            logger.debug("Stored pending zoom - will execute when full resolution is ready")
                            return
                    else:
                        # Start loading full resolution in background
                        self._load_full_resolution_on_demand()
                        # Don't zoom yet - wait for full resolution to load
                        # Store zoom intent for when full resolution is ready
                        self._pending_zoom = True
                        logger.debug("Stored pending zoom - will execute when full resolution is ready")
                        return
            
            # Set up zoom parameters
            self.current_zoom_level = 1.0
            # Always center on image center when using space bar
            image_center_x = self.current_pixmap.width() // 2
            image_center_y = self.current_pixmap.height() // 2
            self.zoom_center_point = QPoint(image_center_x, image_center_y)
            self.zoom_to_point()
        else:
            # Switch to fit-to-window mode
            self.fit_to_window = True
            self.current_zoom_level = 1.0
            self.zoom_center_point = None
            self.scale_image_to_fit()
            self.image_label.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        self.update_status_bar()
        self.setFocus()

    def scale_image_to_fit(self):
        """Scale image to fit the current window size while maintaining aspect ratio"""
        if not self.current_pixmap:
            return

        available_size = self.scroll_area.size()
        margin = 20
        max_width = available_size.width() - margin
        max_height = available_size.height() - margin
        scaled_pixmap = self.current_pixmap.scaled(
            max_width, max_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.resize(scaled_pixmap.size())
        self.image_label.adjustSize()  # Ensure label is resized to pixmap
        self.scroll_area.widget().adjustSize()  # Force scroll area to update
        self.scroll_area.updateGeometry()
        self.setFocus()

    def scale_image_to_100_percent(self):
        """Display image at 100% zoom (actual pixel size)"""
        if not self.current_pixmap:
            return

        # Set original pixmap without scaling
        self.image_label.setPixmap(self.current_pixmap)
        self.image_label.resize(self.current_pixmap.size())

        # Center the image in the scroll area
        self.center_image_in_scroll_area()

        # Ensure main window retains focus for keyboard events
        self.setFocus()

    def center_image_in_scroll_area(self):
        """Center the zoomed image in the scroll area"""
        if not self.current_pixmap:
            return

        # Get viewport size (actual visible area)
        viewport_size = self.scroll_area.viewport().size()
        image_size = self.current_pixmap.size()

        # Calculate center position with proper rounding
        center_x = max(0, (image_size.width() - viewport_size.width()) // 2)
        center_y = max(0, (image_size.height() - viewport_size.height()) // 2)

        # Set scroll position to center the image
        self.scroll_area.horizontalScrollBar().setValue(center_x)
        self.scroll_area.verticalScrollBar().setValue(center_y)

    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        # No rounded corners to update
        # Rescale image when window is resized, but only in fit-to-window mode
        if self.current_pixmap and self.fit_to_window:
            self.scale_image_to_fit()
        # GALLERY FUNCTIONALITY COMMENTED OUT
        # Update gallery layout if in gallery view mode (justified layout rebuilds automatically on resize)
        # JustifiedGallery handles its own resizeEvent, so we don't need to manually update
        # if self.view_mode == 'gallery' and self.gallery_justified and self.gallery_justified.isVisible():
        #     # JustifiedGallery will handle resize in its own resizeEvent
        #     pass
    
    def mousePressEvent(self, event):
        """Handle mouse press for window resizing"""
        if event.button() == Qt.MouseButton.LeftButton:
            # Check if mouse is near window edge (but not on title bar)
            pos = event.position().toPoint()
            
            # Don't resize if clicking on title bar area
            if hasattr(self, 'title_bar') and pos.y() < self.title_bar.height():
                super().mousePressEvent(event)
                return
            
            # Don't resize if clicking on scrollbar area (but allow in status bar)
            if hasattr(self, 'scroll_area'):
                v_scrollbar = self.scroll_area.verticalScrollBar()
                if v_scrollbar.isVisible():
                    scrollbar_width = v_scrollbar.width()
                    # Get status bar height
                    status_bar_height = 0
                    if hasattr(self, 'status_bar'):
                        status_bar_height = self.status_bar.height()
                    # If mouse is in scrollbar area, don't start resize
                    # UNLESS we're in the status bar area (bottom)
                    if pos.x() >= self.width() - scrollbar_width and pos.y() < self.height() - status_bar_height:
                        super().mousePressEvent(event)
                        return
            
            edge = self._get_resize_edge(pos)
            if edge:
                self._resize_edge_active = edge
                self._resize_start_pos = event.globalPosition().toPoint()
                self._resize_start_geometry = self.geometry()
                event.accept()
                return
        
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for window resizing and cursor updates"""
        # Update cursor based on edge position
        pos = event.position().toPoint()
        
        # Don't show resize cursor on title bar
        if hasattr(self, 'title_bar') and pos.y() < self.title_bar.height():
            self.unsetCursor()
            super().mouseMoveEvent(event)
            return
        
        # Get status bar height (footer area where resizing should still work)
        status_bar_height = 0
        if hasattr(self, 'status_bar'):
            status_bar_height = self.status_bar.height()
        
        # Check if mouse is over scrollbar area - if so, don't show resize cursor
        # BUT allow resizing in status bar area (bottom edge)
        scrollbar_width = 0
        if hasattr(self, 'scroll_area'):
            v_scrollbar = self.scroll_area.verticalScrollBar()
            if v_scrollbar.isVisible():
                scrollbar_width = v_scrollbar.width()
                # If mouse is in scrollbar area (rightmost scrollbar_width pixels), don't resize
                # UNLESS we're in the status bar area (bottom)
                if pos.x() >= self.width() - scrollbar_width and pos.y() < self.height() - status_bar_height:
                    self.setCursor(Qt.CursorShape.ArrowCursor)
                    super().mouseMoveEvent(event)
                    return
        
        # Thickness of resize zone (in pixels)
        border = 10  # Increased area to make resizing easier
        w = self.width()
        h = self.height()
        
        # Adjust right border to exclude scrollbar area (but allow in status bar)
        right_border_start = w - border - scrollbar_width
        
        # --- Corner cursors ---
        if pos.x() <= border and pos.y() <= border:
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif pos.x() >= right_border_start and pos.x() < w - scrollbar_width and pos.y() <= border:
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif pos.x() <= border and pos.y() >= h - border:
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif pos.x() >= right_border_start and pos.x() < w - scrollbar_width and pos.y() >= h - border:
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        # --- Edge cursors ---
        elif pos.x() <= border:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif pos.x() >= right_border_start and pos.x() < w - scrollbar_width:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif pos.y() <= border:
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        elif pos.y() >= h - border:
            # Bottom edge: always allow resizing (including in status bar area)
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        # --- Normal area ---
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        
        # Handle window resizing if active
        if self._resize_edge_active:
            current_pos = event.globalPosition().toPoint()
            delta = current_pos - self._resize_start_pos
            new_geometry = QRect(self._resize_start_geometry)
            
            if 'left' in self._resize_edge_active:
                new_geometry.setLeft(self._resize_start_geometry.left() + delta.x())
            if 'right' in self._resize_edge_active:
                new_geometry.setRight(self._resize_start_geometry.right() + delta.x())
            if 'top' in self._resize_edge_active:
                new_geometry.setTop(self._resize_start_geometry.top() + delta.y())
            if 'bottom' in self._resize_edge_active:
                new_geometry.setBottom(self._resize_start_geometry.bottom() + delta.y())
            
            # Ensure minimum size
            if new_geometry.width() >= self.minimumWidth() and new_geometry.height() >= self.minimumHeight():
                self.setGeometry(new_geometry)
            event.accept()
            return
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release to stop window resizing"""
        if self._resize_edge_active:
            self._resize_edge_active = None
            self.unsetCursor()
            event.accept()
            return
        
        super().mouseReleaseEvent(event)
    
    def _get_resize_edge(self, pos):
        """Determine which edge the mouse is near"""
        if self.isMaximized():
            return None
        
        # Thickness of resize zone (in pixels)
        border = 10  # Increased area to make resizing easier
        w = self.width()
        h = self.height()
        x, y = pos.x(), pos.y()
        
        # Get status bar height (footer area where resizing should still work)
        status_bar_height = 0
        if hasattr(self, 'status_bar'):
            status_bar_height = self.status_bar.height()
        
        # Check if vertical scrollbar is visible and get its width
        scrollbar_width = 0
        if hasattr(self, 'scroll_area'):
            v_scrollbar = self.scroll_area.verticalScrollBar()
            if v_scrollbar.isVisible():
                scrollbar_width = v_scrollbar.width()
        
        # Adjust right border to exclude scrollbar area (but allow in status bar)
        right_border_start = w - border - scrollbar_width
        
        # --- Corner resize zones (larger area helps accuracy) ---
        if x <= border and y <= border:
            return 'top_left'
        # Top-right corner: exclude scrollbar area
        if x >= right_border_start and x < w - scrollbar_width and y <= border:
            return 'top_right'
        if x <= border and y >= h - border:
            return 'bottom_left'
        # Bottom-right corner: exclude scrollbar area, but allow in status bar
        if x >= right_border_start and x < w - scrollbar_width and y >= h - border:
            return 'bottom_right'
        
        # --- Edge resize zones ---
        if x <= border:
            return 'left'
        # Right edge: exclude scrollbar area (but allow in status bar)
        if x >= right_border_start and x < w - scrollbar_width:
            return 'right'
        if y <= border:
            return 'top'
        # Bottom edge: always allow resizing (including in status bar area)
        if y >= h - border:
            return 'bottom'
        
        return None
    

    def get_supported_extensions(self):
        """Get list of supported image file extensions"""
        return [
            # RAW formats
            '.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef',
            '.srw', '.x3f', '.raf', '.3fr', '.fff', '.iiq', '.cap', '.erf',
            '.mef', '.mos', '.nrw', '.rwl', '.srf',
            # Standard image formats
            '.jpeg', '.jpg', '.png', '.webp', '.heif'
        ]

    def scan_folder_for_images(self, file_path):
        """Scan the folder containing the given file for all image files"""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Get the folder path
            folder_path = os.path.dirname(file_path)
            
            # Validate folder
            if not os.path.isdir(folder_path):
                logger.warning(f"Folder does not exist: {folder_path}")
                self.show_error("Folder Not Found", 
                              f"The folder does not exist:\n{folder_path}")
                return
            
            self.current_folder = folder_path

            # Get supported extensions
            supported_extensions = self.get_supported_extensions()

            # Scan folder for image files (recursively including subfolders)
            image_files = []

            try:
                # Use os.walk() to recursively scan all subfolders
                for root, dirs, files in os.walk(folder_path):
                    # Skip "Discard" folder and its subfolders
                    if 'Discard' in dirs:
                        dirs.remove('Discard')
                    
                    for filename in files:
                        # Skip macOS resource fork files (._filename)
                        if filename.startswith('._'):
                            continue
                        file_ext = os.path.splitext(filename)[1].lower()
                        if file_ext in supported_extensions:
                            full_path = os.path.join(root, filename)
                            if os.path.isfile(full_path):
                                image_files.append(full_path)
            except OSError as e:
                error_msg = f"Cannot read folder contents:\n{str(e)}"
                logger.error(f"Error reading folder {folder_path}: {e}")
                self.show_error("Folder Access Error", error_msg)
                return

            # Check if any images were found
            if not image_files:
                logger.warning(f"No supported images found in folder: {folder_path}")
                # Display message in main viewing area instead of popup
                self.show_no_images_message(supported_extensions)
                # Reset state
                self.image_files = []
                self.current_file_index = -1
                self.current_file_path = None
                self.current_folder = None
                return

            # Sort files according to user preference
            self.image_files = self.sort_image_files(image_files)

            # Find current file index
            self.current_file_index = -1

            # Normalize paths for comparison by converting to absolute paths
            # This handles both forward/backward slashes and case differences
            try:
                normalized_target = os.path.abspath(file_path)

                for i, img_file in enumerate(self.image_files):
                    normalized_img_file = os.path.abspath(img_file)
                    if normalized_target.lower() == normalized_img_file.lower():
                        self.current_file_index = i
                        break
            except Exception:
                # Fallback to original logic
                if file_path in self.image_files:
                    self.current_file_index = self.image_files.index(file_path)

            # Update status bar after scanning
            self.update_status_bar()

        except Exception as e:
            import traceback
            logger.error(f"Error scanning folder: {e}", exc_info=True)
            error_msg = f"Error scanning folder:\n{str(e)}"
            self.show_error("Folder Scan Error", error_msg)

    def show_no_images_message(self, supported_extensions):
        """Display 'No images found' message in the main viewing area"""
        # Format supported extensions for display
        formats_text = ', '.join(supported_extensions)
        message = f"No images found.\n\nSupported formats: {formats_text}"
        
        # Clear any existing pixmap
        self.current_pixmap = None
        self.current_image = None
        
        # Display message in main viewing area
        self.image_label.setText(message)
        self.image_label.setStyleSheet(
            "QLabel { color: #B0B0B0; font-size: 14px; }")
        
        # Hide metadata, image counter, and sort button when no files
        if hasattr(self, 'status_metadata_label'):
            # Only hide in gallery mode, show in single view mode
            if self.view_mode == 'single':
                self.status_metadata_label.setVisible(True)
                self.status_metadata_label.setText("")
            else:
                self.status_metadata_label.setVisible(False)
        if hasattr(self, 'status_counter_label'):
            self.status_counter_label.setVisible(False)
        if hasattr(self, 'sort_toggle_button'):
            self.sort_toggle_button.setVisible(False)
        
        # Update status bar
        self.status_bar.showMessage("No images found")
        
        # Reset window title
        self.setWindowTitle('RAW Image Viewer')
        # Update custom title bar
        if hasattr(self, 'title_bar'):
            self.title_bar.set_title('RAW Image Viewer')

    def show_error(self, title, message):
        """Show error message dialog"""
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec()

    def extract_exif_data(self, file_path):
        """Extract EXIF data from image file"""
        exif_data = {
            'focal_length': None,
            'aperture': None,
            'iso': None,
            'capture_time': None
        }

        try:
            # Suppress exifread warnings for unsupported file formats
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, module='exifread')
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)

                # Extract focal length
                if 'EXIF FocalLength' in tags:
                    focal_length_raw = tags['EXIF FocalLength']
                    try:
                        # Handle different focal length formats
                        focal_str = str(focal_length_raw)
                        if '/' in focal_str:
                            # Handle fraction format (e.g., "24/1")
                            num, den = focal_str.split('/')
                            focal_length = round(float(num) / float(den))
                        else:
                            # Handle decimal format
                            focal_length = round(float(focal_str))
                        exif_data['focal_length'] = f"{focal_length}mm"
                    except (ValueError, AttributeError, ZeroDivisionError):
                        pass

                # Extract aperture
                if 'EXIF FNumber' in tags:
                    aperture_raw = tags['EXIF FNumber']
                    try:
                        # Handle different aperture formats
                        aperture_str = str(aperture_raw)
                        if '/' in aperture_str:
                            # Handle fraction format (e.g., "28/10")
                            num, den = aperture_str.split('/')
                            aperture = float(num) / float(den)
                        else:
                            # Handle decimal format
                            aperture = float(aperture_str)
                        exif_data['aperture'] = f"f/{aperture:.1f}"
                    except (ValueError, AttributeError, ZeroDivisionError):
                        pass

                # Extract ISO
                if 'EXIF ISOSpeedRatings' in tags:
                    iso_raw = tags['EXIF ISOSpeedRatings']
                    try:
                        iso = int(str(iso_raw))
                        exif_data['iso'] = f"ISO {iso}"
                    except (ValueError, AttributeError):
                        pass

                # Extract capture time
                datetime_tags = ['EXIF DateTimeOriginal',
                                 'Image DateTime', 'EXIF DateTime']
                for tag_name in datetime_tags:
                    if tag_name in tags:
                        datetime_raw = tags[tag_name]
                        try:
                            datetime_str = str(datetime_raw)
                            # Parse datetime string (format: "YYYY:MM:DD HH:MM:SS")
                            dt = datetime.strptime(
                                datetime_str, "%Y:%m:%d %H:%M:%S")
                            # Format as "HH:MM:SS YYYY-MM-DD"
                            exif_data['capture_time'] = dt.strftime(
                                "%H:%M:%S %Y-%m-%d")
                            break  # Use first available datetime
                        except (ValueError, AttributeError):
                            continue

        except Exception:
            # If any error occurs during EXIF extraction, return empty data
            pass

        return exif_data

    def update_status_bar(self, width=None, height=None):
        """Update status bar with comprehensive information including EXIF data"""
        if not hasattr(self, 'status_metadata_label') or not hasattr(self, 'status_counter_label'):
            # Fallback to old method if UI components not initialized
            if not self.current_file_path:
                self.status_bar.showMessage("")  # Empty message when no image loaded
                return
            self.status_bar.showMessage("")
            return
        
        # Show metadata, counter, and sort button when there are files
        # Only show metadata in single view mode, hide in gallery mode
        if hasattr(self, 'status_metadata_label'):
            if self.view_mode == 'single':
                self.status_metadata_label.setVisible(True)  # Show metadata in single view
            else:
                self.status_metadata_label.setVisible(False)  # Hide metadata in gallery view
        if hasattr(self, 'status_counter_label'):
            if self.view_mode == 'single':
                self.status_counter_label.setVisible(True)  # Show counter in single view
            else:
                self.status_counter_label.setVisible(False)  # Hide counter in gallery view
        # Show/hide sort button based on view mode
        if hasattr(self, 'sort_toggle_button'):
            if self.view_mode == 'gallery':
                self.sort_toggle_button.setVisible(True)  # Show in gallery mode
            else:
                self.sort_toggle_button.setVisible(False)  # Hide in single mode
        
        if not self.current_file_path:
            # Hide metadata when no image is loaded
            if hasattr(self, 'status_metadata_label'):
                if self.view_mode == 'single':
                    self.status_metadata_label.setVisible(True)
                    self.status_metadata_label.setText("")
                else:
                    self.status_metadata_label.setVisible(False)
            if hasattr(self, 'status_counter_label'):
                self.status_counter_label.setText("")
            return

        # Get filename
        filename = os.path.basename(self.current_file_path)

        # Get image dimensions - ALWAYS prefer original dimensions from cache
        # This ensures we show the original resolution even when displaying a thumbnail
        original_width = None
        original_height = None
        display_width = None
        display_height = None
        
            # Get original dimensions from cache (authoritative source)
        cached_exif = self.image_cache.get_exif(self.current_file_path)
        if cached_exif:
            # CRITICAL: Always check for original_width and original_height in cache
            # These are stored when RAW file is opened, before processing
            original_width = cached_exif.get('original_width')
            original_height = cached_exif.get('original_height')
            # Log for debugging
            import logging
            logger = logging.getLogger(__name__)
            if original_width and original_height:
                logger.info(f"[STATUS] Found original dimensions in cache: {original_width}x{original_height}")
            else:
                logger.warning(f"[STATUS] No original dimensions in cache for {os.path.basename(self.current_file_path)}")
                # Try to extract from EXIF if not in cache
                if not original_width or not original_height:
                    try:
                        import exifread
                        # Suppress exifread warnings for unsupported file formats
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning, module='exifread')
                            with open(self.current_file_path, 'rb') as f:
                                tags = exifread.process_file(f, details=False)
                                if 'EXIF ExifImageWidth' in tags:
                                    original_width = int(str(tags['EXIF ExifImageWidth']))
                                if 'EXIF ExifImageLength' in tags:
                                    original_height = int(str(tags['EXIF ExifImageLength']))
                                if original_width and original_height:
                                    logger.info(f"[STATUS] Extracted original dimensions from EXIF: {original_width}x{original_height}")
                                    # Update cache
                                    cached_exif['original_width'] = original_width
                                    cached_exif['original_height'] = original_height
                                    self.image_cache.put_exif(self.current_file_path, cached_exif)
                    except Exception as e:
                        logger.debug(f"[STATUS] Could not extract dimensions from EXIF: {e}")
        
        # Get current display dimensions (from pixmap)
        if self.current_pixmap:
            display_width = self.current_pixmap.width()
            display_height = self.current_pixmap.height()
        
        # Use provided dimensions if available, otherwise use original dimensions
        # CRITICAL: Always prioritize original_width/original_height over display dimensions
        if width is None or height is None:
            if original_width and original_height:
                width = original_width
                height = original_height
            elif display_width and display_height:
                width = display_width
                height = display_height
            else:
                width = height = 0
        
        # Determine if we're displaying a thumbnail (for status bar indication)
        is_displaying_thumbnail = False
        if original_width and original_height and display_width and display_height:
            # Check if display size is significantly smaller than original
            original_max = max(original_width, original_height)
            display_max = max(display_width, display_height)
            # Consider it a thumbnail if display is less than 80% of original
            if display_max < original_max * 0.8:
                is_displaying_thumbnail = True

        # Get zoom level
        if self.fit_to_window:
            zoom_level = "Fit"
        else:
            zoom_level = f"{int(self.current_zoom_level * 100)}%"

        # Try to get EXIF data from cache first (faster)
        exif_info = []
        cached_exif = self.image_cache.get_exif(self.current_file_path)
        if cached_exif and cached_exif.get('exif_data'):
            # Use cached EXIF data to build info string
            exif_tags = cached_exif['exif_data']

            # Extract focal length
            if 'EXIF FocalLength' in exif_tags:
                focal_length_raw = exif_tags['EXIF FocalLength']
                try:
                    focal_str = str(focal_length_raw)
                    if '/' in focal_str:
                        num, den = focal_str.split('/')
                        focal_length = round(float(num) / float(den))
                    else:
                        focal_length = round(float(focal_str))
                    exif_info.append(f"{focal_length}mm")
                except (ValueError, AttributeError, ZeroDivisionError):
                    pass

            # Extract aperture
            if 'EXIF FNumber' in exif_tags:
                aperture_raw = exif_tags['EXIF FNumber']
                try:
                    aperture_str = str(aperture_raw)
                    if '/' in aperture_str:
                        num, den = aperture_str.split('/')
                        aperture = float(num) / float(den)
                    else:
                        aperture = float(aperture_str)
                    exif_info.append(f"f/{aperture:.1f}")
                except (ValueError, AttributeError, ZeroDivisionError):
                    pass

            # Extract ISO
            if 'EXIF ISOSpeedRatings' in exif_tags:
                iso_raw = exif_tags['EXIF ISOSpeedRatings']
                try:
                    iso = int(str(iso_raw))
                    exif_info.append(f"ISO {iso}")
                except (ValueError, AttributeError):
                    pass

            # Extract capture time
            datetime_tags = ['EXIF DateTimeOriginal',
                             'Image DateTime', 'EXIF DateTime']
            for tag_name in datetime_tags:
                if tag_name in exif_tags:
                    datetime_raw = exif_tags[tag_name]
                    try:
                        datetime_str = str(datetime_raw)
                        from datetime import datetime
                        dt = datetime.strptime(
                            datetime_str, "%Y:%m:%d %H:%M:%S")
                        exif_info.append(dt.strftime("%H:%M:%S %Y-%m-%d"))
                        break
                    except (ValueError, AttributeError):
                        continue
        else:
            # Fallback to direct EXIF extraction (slower, but ensures data is available)
            exif_data = self.extract_exif_data(self.current_file_path)
            if exif_data['focal_length']:
                exif_info.append(exif_data['focal_length'])
            if exif_data['aperture']:
                exif_info.append(exif_data['aperture'])
            if exif_data['iso']:
                exif_info.append(exif_data['iso'])
            if exif_data['capture_time']:
                exif_info.append(exif_data['capture_time'])

        # Construct metadata text (center label)
        metadata_parts = []

        # Add filename and dimensions - ALWAYS show original resolution
        # If displaying thumbnail, indicate it but still show original resolution
        # CRITICAL: Use original_width and original_height if available, not display dimensions
        display_width_final = original_width if original_width else width
        display_height_final = original_height if original_height else height
        
        if display_width_final > 0 and display_height_final > 0:
            # Show original resolution, and optionally indicate if displaying thumbnail
            # Show original resolution (no thumbnail indicator needed)
            metadata_parts.append(f"{filename} - {display_width_final}x{display_height_final}")
        else:
            metadata_parts.append(filename)

        # Add zoom level
        metadata_parts.append(zoom_level)

        # Add EXIF info if available
        if exif_info:
            metadata_parts.extend(exif_info)

        # Join all parts with separator
        metadata_text = " - ".join(metadata_parts)
        # Show and set metadata text in single view mode
        if hasattr(self, 'status_metadata_label'):
            if self.view_mode == 'single':
                self.status_metadata_label.setVisible(True)
                self.status_metadata_label.setText(metadata_text)
            else:
                self.status_metadata_label.setVisible(False)

        # Update image counter (right label)
        if self.image_files and self.current_file_index >= 0:
            total_files = len(self.image_files)
            current_pos = self.current_file_index + 1
            self.status_counter_label.setText(f"{current_pos} / {total_files}")
        else:
            self.status_counter_label.setText("")

    def _can_pan(self):
        # Only allow panning if the image is larger than the viewport
        if not self.current_pixmap:
            return False
        pixmap_size = self.image_label.pixmap().size()
        viewport_size = self.scroll_area.viewport().size()
        return pixmap_size.width() > viewport_size.width() or pixmap_size.height() > viewport_size.height()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key == Qt.Key.Key_Left:
                self.navigate_to_previous_image()
                return True
            elif key == Qt.Key.Key_Right:
                self.navigate_to_next_image()
                return True
            elif key == Qt.Key.Key_Up or key == Qt.Key.Key_Down:
                # Ignore up/down arrows to prevent panning
                return True
        return super().eventFilter(obj, event)

    def move_current_image_to_discard(self):
        """Move the current image to a 'Discard' folder in the same directory"""
        if not self.current_file_path or not os.path.exists(self.current_file_path):
            self.show_error("Discard Error", "No image file to move.")
            return
        try:
            file_to_move = self.current_file_path
            folder_path = os.path.dirname(file_to_move)
            discard_folder = os.path.join(folder_path, "Discard")
            os.makedirs(discard_folder, exist_ok=True)
            filename = os.path.basename(file_to_move)
            target_path = os.path.join(discard_folder, filename)
            # If file with same name exists in Discard, add a suffix
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(target_path):
                target_path = os.path.join(
                    discard_folder, f"{base}_discarded_{counter}{ext}")
                counter += 1
            
            # Before moving, cancel any preload tasks for this file and clear cache
            if hasattr(self, 'preload_manager'):
                # Cancel preload for this specific file if it's in the active threads
                if file_to_move in self.preload_manager.active_threads:
                    thread = self.preload_manager.active_threads[file_to_move]
                    try:
                        if hasattr(thread, 'cleanup'):
                            thread.cleanup()
                        elif hasattr(thread, 'stop_processing'):
                            thread.stop_processing()
                            thread.quit()
                            thread.wait(100)
                    except Exception as e:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.debug(f"Error cancelling preload for {filename}: {e}")
                    finally:
                        if file_to_move in self.preload_manager.active_threads:
                            del self.preload_manager.active_threads[file_to_move]
            
            # Clear cache for this file
            from image_cache import get_image_cache
            cache = get_image_cache()
            cache.invalidate_file(file_to_move)
            
            # Now move the file
            os.rename(file_to_move, target_path)
            
            # Remove from image files list
            if file_to_move in self.image_files:
                self.image_files.remove(file_to_move)
            self.status_bar.showMessage(f"Moved to Discard: {filename}")
            # --- Preserve zoom/pan state for next image (like navigation/discard) ---
            self._maintain_zoom_on_navigation = True
            if not self.fit_to_window:
                self._restore_zoom_center = self.zoom_center_point
                self._restore_zoom_level = self.current_zoom_level
                # Save current scroll position instead of start_scroll_x/y
                self._restore_start_scroll_x = self.scroll_area.horizontalScrollBar().value()
                self._restore_start_scroll_y = self.scroll_area.verticalScrollBar().value()
            else:
                self._restore_zoom_center = None
                self._restore_zoom_level = None
                self._restore_start_scroll_x = None
                self._restore_start_scroll_y = None
            self.handle_post_deletion_navigation()
            self.save_session_state()
        except Exception as e:
            error_msg = f"Could not move file to Discard folder:\n{str(e)}"
            self.show_error("Discard Error", error_msg)

    def load_folder_images(self, folder_path, start_file=None):
        """Load images from a folder"""
        import logging
        import time
        logger = logging.getLogger(__name__)
        load_start = time.time()
        logger.info(f"[FOLDER] ========== load_folder_images() STARTED at {load_start:.3f} ==========")
        print(f"[PERF] 📁 Loading folder: {os.path.basename(folder_path)}")
        
        # Validate folder path
        if not folder_path:
            self.show_error("Invalid Folder", "No folder path provided")
            return
        
        if not os.path.exists(folder_path):
            self.show_error("Folder Not Found", 
                          f"The folder does not exist:\n{folder_path}")
            return
        
        if not os.path.isdir(folder_path):
            self.show_error("Invalid Path", 
                          f"The path is not a folder:\n{folder_path}")
            return
        
        # Scan for images in the folder and all subfolders (recursive)
        scan_start = time.time()
        extensions = self.get_supported_extensions()
        # Get all image files with full paths
        image_files = []
        
        try:
            # Use os.walk() to recursively scan all subfolders
            for root, dirs, files in os.walk(folder_path):
                # Skip "Discard" folder and its subfolders
                if 'Discard' in dirs:
                    dirs.remove('Discard')
                
                for filename in files:
                    # Skip macOS resource fork files (._filename)
                    if filename.startswith('._'):
                        continue
                    file_ext = os.path.splitext(filename)[1].lower()
                    if file_ext in extensions:
                        full_path = os.path.join(root, filename)
                        if os.path.isfile(full_path):
                            image_files.append(full_path)
        except OSError as e:
            error_msg = f"Cannot read folder contents:\n{str(e)}"
            logger.error(f"Error reading folder {folder_path}: {e}")
            self.show_error("Folder Access Error", error_msg)
            return
        except Exception as e:
            error_msg = f"Unexpected error while scanning folder:\n{str(e)}"
            logger.error(f"Unexpected error scanning folder {folder_path}: {e}", exc_info=True)
            self.show_error("Scan Error", error_msg)
            return
        
        scan_time = time.time() - scan_start
        logger.info(f"[FOLDER] Scanned {len(image_files)} images in {scan_time:.3f}s")
        print(f"[PERF] 📁 Scanned {len(image_files)} images in {scan_time*1000:.1f}ms")
        
        if not image_files:
            # Display message in main viewing area instead of popup
            self.show_no_images_message(extensions)
            # Reset folder state
            self.current_folder = None
            self.image_files = []
            self.current_file_index = -1
            self.current_file_path = None
            return
        
        # Sort files according to user preference
        sort_start = time.time()
        self.current_folder = folder_path
        self.image_files = self.sort_image_files(image_files)
        sort_time = time.time() - sort_start
        logger.info(f"[FOLDER] Sorted {len(image_files)} images in {sort_time:.3f}s")
        print(f"[PERF] 📁 Sorted {len(image_files)} images in {sort_time*1000:.1f}ms")
        
        # Determine which image to start with
        if start_file:
            # Find the full path of start_file
            start_file_path = None
            for img_file in self.image_files:
                if os.path.basename(img_file) == start_file or img_file == start_file:
                    start_file_path = img_file
                    break
            if start_file_path and start_file_path in self.image_files:
                idx = self.image_files.index(start_file_path)
            else:
                idx = 0
        else:
            idx = 0
            
        self.current_file_index = idx
        self.current_file_path = self.image_files[idx]
        
        total_time = time.time() - load_start
        logger.info(f"[FOLDER] ========== load_folder_images() COMPLETED in {total_time:.3f}s ==========")
        print(f"[PERF] 📁 Folder loaded in {total_time*1000:.1f}ms (scan: {scan_time*1000:.1f}ms, sort: {sort_time*1000:.1f}ms)")
        
        # Mark gallery as needing update if in gallery mode
        if hasattr(self, 'view_mode') and self.view_mode == 'gallery':
            self._gallery_update_needed = True
            # Update gallery view if it's currently visible
            if self.gallery_widget and self.gallery_widget.isVisible():
                self._update_gallery_view()
                self._gallery_update_needed = False
        else:
            # Load the first image in single view mode
            self.load_raw_image(self.current_file_path)
        self.save_session_state()

    def save_session_state(self):
        settings = self.get_settings()
        if self.current_folder and self.current_file_index >= 0 and self.image_files:
            filename = os.path.basename(
                self.image_files[self.current_file_index])
            settings.setValue("last_session_folder", self.current_folder)
            settings.setValue("last_session_file", filename)
        else:
            settings.remove("last_session_folder")
            settings.remove("last_session_file")

    def restore_session_state(self):
        settings = self.get_settings()
        folder = settings.value("last_session_folder", None)
        file = settings.value("last_session_file", None)
        if folder and file and os.path.isdir(folder):
            # Lazy import natsort to avoid import delays
            from natsort import natsorted
            files = [f for f in natsorted(os.listdir(folder))
                     if os.path.splitext(f)[1].lower() in self.get_supported_extensions()]
            if file in files:
                self.load_folder_images(folder, start_file=file)
            return True
        return False

    def changeEvent(self, event):
        """Handle window state changes"""
        if event.type() == QEvent.Type.WindowStateChange:
            # Update title bar's internal maximized state
            if hasattr(self, 'title_bar'):
                self.title_bar._is_maximized = self.isMaximized()
        super().changeEvent(event)
    
    
    def closeEvent(self, event):
        self.save_session_state()
        super().closeEvent(event)
    

def main():
    """Main function to run the application"""
    import logging
    import traceback
    
    # Print to console immediately (before logging might be ready)
    print("main() function called", flush=True)
    
    # Logging should already be setup in if __name__ == '__main__'
    # But check if it's configured, if not, setup it
    logger = logging.getLogger(__name__)
    if not logger.handlers and not logging.getLogger().handlers:
        try:
            log_file = setup_logging()
            logger.info("=" * 80)
            logger.info("Application startup started")
            logger.info(f"Python version: {sys.version}")
            logger.info(f"Platform: {platform.system()} {platform.release()}")
            logger.info(f"Working directory: {os.getcwd()}")
            logger.info("=" * 80)
        except Exception as log_error:
            # If logging setup fails, at least print to stderr
            print(f"ERROR: Failed to setup logging: {log_error}", file=sys.stderr)
            print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
    else:
        # Logging already configured, just log startup
        print("[MAIN] Logging already configured, logging startup info...", flush=True)
        logger.info("=" * 80)
        print("[MAIN] Logger.info('=' * 80) called", flush=True)
        logger.info("Application startup started")
        print("[MAIN] Logger.info('Application startup started') called", flush=True)
        logger.info(f"Python version: {sys.version}")
        print(f"[MAIN] Python version logged: {sys.version}", flush=True)
        print("[MAIN] Getting platform info...", flush=True)
        # Temporarily skip platform info to avoid potential blocking
        # Use hardcoded values for Windows
        print("[MAIN] Using hardcoded platform info (Windows) to avoid blocking...", flush=True)
        platform_system = "Windows"
        platform_release = "10"
        try:
            # Try to get real platform info, but don't block if it fails
            import threading
            platform_result = [None, None]
            def get_platform_info():
                try:
                    platform_result[0] = platform.system()
                    platform_result[1] = platform.release()
                except:
                    pass
            
            thread = threading.Thread(target=get_platform_info, daemon=True)
            thread.start()
            thread.join(timeout=0.1)  # Wait max 100ms
            if platform_result[0] and platform_result[1]:
                platform_system = platform_result[0]
                platform_release = platform_result[1]
                print(f"[MAIN] Got platform info: {platform_system} {platform_release}", flush=True)
            else:
                print(f"[MAIN] Using fallback platform info: {platform_system} {platform_release}", flush=True)
        except Exception as e:
            print(f"[MAIN] Error getting platform info, using fallback: {e}", flush=True)
        
        print("[MAIN] Calling logger.info for platform...", flush=True)
        logger.info(f"Platform: {platform_system} {platform_release}")
        print(f"[MAIN] Platform logged: {platform_system} {platform_release}", flush=True)
        logger.info(f"Working directory: {os.getcwd()}")
        print(f"[MAIN] Working directory logged: {os.getcwd()}", flush=True)
        logger.info("=" * 80)
        print("[MAIN] All platform info logged, setting up Windows exception handler...", flush=True)
    
    # Set up Windows exception handler to catch access violations
    # Use platform_system variable to avoid calling platform.system() again
    is_windows = (platform_system == 'Windows')
    if is_windows:
        print("  [Windows] Importing ctypes...", flush=True)
        import ctypes
        print("  [Windows] ctypes imported", flush=True)
        print("  [Windows] Importing wintypes...", flush=True)
        from ctypes import wintypes
        print("  [Windows] wintypes imported", flush=True)
        
        # Define exception handler function
        def exception_handler(exception_info):
            """Handle Windows exceptions (access violations, etc.)"""
            exception_code = exception_info[0].ExceptionRecord[0].ExceptionCode
            exception_address = exception_info[0].ExceptionRecord[0].ExceptionAddress
            
            # 0xC0000005 is ACCESS_VIOLATION
            if exception_code == 0xC0000005:
                error_msg = f"Access Violation (0xC0000005) at address {exception_address}"
                logger.critical(f"Windows Access Violation: {error_msg}")
                logger.critical(f"This usually indicates accessing invalid memory (null pointer, freed object, etc.)")
                print(f"\n{'='*80}", file=sys.stderr)
                print(f"WINDOWS ACCESS VIOLATION", file=sys.stderr)
                print(f"{'='*80}", file=sys.stderr)
                print(f"{error_msg}", file=sys.stderr)
                print(f"{'='*80}\n", file=sys.stderr)
                return 1  # EXCEPTION_EXECUTE_HANDLER
            return 0  # EXCEPTION_CONTINUE_SEARCH
        
        # Note: Setting up structured exception handling in Python is complex
        # We'll rely on Python's exception handling and add more defensive checks
        print("  [Windows] Exception handler setup complete", flush=True)
    else:
        print("  [Non-Windows] Skipping Windows exception handler", flush=True)
    
    print("Entering main try block...", flush=True)
    try:
        print("Creating QApplication...", flush=True)
        # Set AppUserModelID to ensure the icon is displayed correctly on the taskbar on Windows
        # Use is_windows variable to avoid calling platform.system() again
        if is_windows:
            print("  [Windows] Setting AppUserModelID...", flush=True)
            myappid = 'RAWviewer.1.0'
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
            print("  [Windows] AppUserModelID set", flush=True)

        app = QApplication(sys.argv)
        print("QApplication created successfully", flush=True)

        # Set application properties
        app.setApplicationName("RAW Image Viewer")
        app.setApplicationVersion("1.0")

        # Create and show splash screen
        print("Creating splash screen...", flush=True)
        splash_pixmap = None
        # Use resource_path to find icon, ensuring it works when bundled
        splash_path = resource_path(os.path.join('icons', 'appicon.png'))
        if os.path.exists(splash_path):
            splash_pixmap = QPixmap(splash_path)
            # Scale to a reasonable size for splash screen (e.g., 400x400)
            if not splash_pixmap.isNull():
                splash_pixmap = splash_pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        else:
            # Create a simple splash screen if icon not found
            splash_pixmap = QPixmap(400, 400)
            splash_pixmap.fill(QColor(173, 216, 230))  # Light blue background
            painter = QPainter(splash_pixmap)
            painter.setPen(QPen(QColor(70, 130, 180), 4))  # Darker blue
            font = painter.font()
            font.setPointSize(48)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(splash_pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "RAW")
            painter.end()
        
        splash = QSplashScreen(splash_pixmap, Qt.WindowType.WindowStaysOnTopHint)
        splash.show()
        app.processEvents()  # Process events to show splash screen immediately
        print("Splash screen displayed", flush=True)

        # Create and show main window
        print("Creating RAWImageViewer...", flush=True)
        viewer = RAWImageViewer()
        print("RAWImageViewer created successfully", flush=True)
        # Check for file or folder argument
        if len(sys.argv) > 1:
            path = sys.argv[1]
            if os.path.isfile(path):
                # If it's a file, load the folder containing that file
                viewer.load_folder_images(os.path.dirname(path), start_file=os.path.basename(path))
            elif os.path.isdir(path):
                # If it's a folder, load the folder
                viewer.load_folder_images(path)
        
        # Show main window and close splash screen
        viewer.show()
        splash.finish(viewer)  # Close splash screen when main window is ready
        print("Splash screen closed, main window displayed", flush=True)

        # Run application
        logger.info(f"[MAIN] Starting Qt event loop")
        exit_code = app.exec()
        
        # Check exit code for access violations
        if exit_code == -1073741819:  # 0xC0000005 - Access Violation
            logger.critical(f"[MAIN] Application crashed with Access Violation (0xC0000005)")
            logger.critical(f"[MAIN] This usually indicates:")
            logger.critical(f"[MAIN]   1. Accessing invalid memory (null pointer, freed object)")
            logger.critical(f"[MAIN]   2. Qt object accessed from wrong thread")
            logger.critical(f"[MAIN]   3. Memory corruption in rawpy/Qt")
            logger.critical(f"[MAIN]   4. Signal/slot connection to deleted object")
            print(f"\n{'='*80}", file=sys.stderr)
            print(f"ACCESS VIOLATION DETECTED (0xC0000005)", file=sys.stderr)
            print(f"{'='*80}", file=sys.stderr)
            print(f"This error indicates the application tried to access invalid memory.", file=sys.stderr)
            print(f"Possible causes:", file=sys.stderr)
            print(f"  - Qt object accessed from wrong thread", file=sys.stderr)
            print(f"  - Accessing deleted/freed object", file=sys.stderr)
            print(f"  - Memory corruption in rawpy or Qt library", file=sys.stderr)
            print(f"  - Signal connected to deleted slot", file=sys.stderr)
            print(f"\nCheck the log file for detailed information.", file=sys.stderr)
            print(f"{'='*80}\n", file=sys.stderr)
        
        logger.info(f"[MAIN] Application exited with code: {exit_code}")
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user (Ctrl+C)")
        print("\n[INFO] Application interrupted by user (Ctrl+C)")
        sys.exit(0)
    except SystemExit as e:
        # Re-raise SystemExit to preserve exit code
        logger.info(f"SystemExit with code: {e.code}")
        raise
    except Exception as e:
        # Catch all other exceptions and log them before crashing
        error_msg = f"FATAL ERROR: {type(e).__name__}: {e}"
        error_traceback = traceback.format_exc()
        
        logger.critical(f"Fatal error: {error_msg}")
        logger.critical(f"Full traceback:\n{error_traceback}")
        
        # Also print to console/stderr so it's visible even if logging fails
        print(f"\n{'='*80}", file=sys.stderr)
        print(f"FATAL ERROR", file=sys.stderr)
        print(f"{'='*80}", file=sys.stderr)
        print(f"{error_msg}", file=sys.stderr)
        print(f"\nFull traceback:", file=sys.stderr)
        print(f"{error_traceback}", file=sys.stderr)
        print(f"{'='*80}\n", file=sys.stderr)
        
        # Try to show error dialog if possible
        try:
            app = QApplication.instance()
            if app is not None:
                from PyQt6.QtWidgets import QMessageBox
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Icon.Critical)
                msg.setWindowTitle("Fatal Error")
                msg.setText("The application encountered a fatal error and will now exit.")
                msg.setDetailedText(f"{error_msg}\n\n{error_traceback}")
                msg.exec()
        except:
            pass  # If we can't show dialog, at least we logged it
        
        sys.exit(1)


if __name__ == '__main__':
    # Print startup message to console immediately
    print("=" * 80, flush=True)
    print("RAWviewer Starting...", flush=True)
    print("=" * 80, flush=True)
    
    # Setup logging before anything else
    try:
        print("Setting up logging...", flush=True)
        setup_logging()
        print("Logging setup complete.", flush=True)
    except Exception as e:
        print(f"ERROR: Failed to setup logging: {e}", file=sys.stderr, flush=True)
        import traceback
        print(f"Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
    
    try:
        print("Calling main()...", flush=True)
        main()
    except Exception as e:
        print(f"\n{'='*80}", file=sys.stderr, flush=True)
        print(f"FATAL ERROR in main(): {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        print(f"{'='*80}\n", file=sys.stderr, flush=True)
        raise
