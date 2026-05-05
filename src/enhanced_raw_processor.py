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
from PyQt6.QtCore import QThread, pyqtSignal, QObject, QSize
from PyQt6.QtGui import QPixmap, QImage, QImageReader
from PIL import Image
import io

# Suppress exifread warnings for unsupported file formats (e.g., video files)
warnings.filterwarnings('ignore', category=UserWarning, module='exifread')

from image_cache import get_image_cache
import metadata_backend
from raw_file_extensions import RAW_FILE_EXTENSIONS

# Cached EXIF rows without this version used embedded-JPEG dimensions as "original" (e.g. 1920×1080).
# Cached EXIF rows without this version used buggy orientation logic (e.g. LibRaw 5 mis-mapped, Sony MakerNote missing, or Silent Failures).
RAW_EXIF_SENSOR_META_VER = 6


def _qimage_to_rgb_array(image: QImage) -> Optional[np.ndarray]:
    """Convert a QImage into a contiguous RGB numpy array without creating a QPixmap."""
    img = image.convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    if w < 1 or h < 1:
        return None
    bpl = img.bytesPerLine()
    nbytes = h * bpl
    bits = img.constBits()
    if bits is None:
        return None
    try:
        if hasattr(bits, "asstring"):
            raw = bits.asstring(nbytes)
        else:
            raw = bytes(memoryview(bits)[:nbytes])
    except (BufferError, TypeError, AttributeError):
        out = np.empty((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                c = img.pixel(x, y)
                out[y, x, 0] = (c >> 16) & 0xFF
                out[y, x, 1] = (c >> 8) & 0xFF
                out[y, x, 2] = c & 0xFF
        return out
    arr = np.frombuffer(bytearray(raw), dtype=np.uint8).reshape(h, bpl)
    return np.ascontiguousarray(arr[:, : w * 3].reshape(h, w, 3))


_embedded_scan_miss_cache = set()
_embedded_scan_miss_lock = threading.Lock()
_embedded_scan_miss_cache_max = 4096


def extract_embedded_jpeg_by_scan(file_path: str, max_size: int) -> Optional[np.ndarray]:
    """
    When LibRaw cannot open a file, scan for embedded JPEG (SOI … EOI) and decode with PIL.
    Many damaged or newer ARW/RAW containers still carry a JPEG preview readable without rawpy.
    Picks the largest successfully decoded JPEG (by pixel area), skips tiny icons.
    """
    try:
        stat = os.stat(file_path)
        size = stat.st_size
        miss_key = (file_path, size, stat.st_mtime, max_size)
        with _embedded_scan_miss_lock:
            if miss_key in _embedded_scan_miss_cache:
                return None

        if max_size <= 512:
            read_limit = 32 * 1024 * 1024
        elif max_size <= 2048:
            read_limit = 64 * 1024 * 1024
        else:
            read_limit = 120 * 1024 * 1024
        to_read = min(size, read_limit)
        with open(file_path, "rb") as f:
            blob = f.read(to_read)
    except OSError:
        return None

    best_arr = None
    best_area = 0
    start = 0
    while True:
        idx = blob.find(b"\xff\xd8\xff", start)
        if idx < 0:
            break
        end_marker = blob.find(b"\xff\xd9", idx + 3)
        segments = []
        if end_marker >= 0:
            segments.append(blob[idx : end_marker + 2])
        segments.append(blob[idx:])

        for segment in segments:
            try:
                im = Image.open(io.BytesIO(segment))
                im.load()
                if im.mode != "RGB":
                    im = im.convert("RGB")
                w, h = im.size
                if w < 32 or h < 32:
                    continue
                area = w * h
                if area <= best_area:
                    continue
                best_area = area
                work = im
                if w > max_size or h > max_size:
                    work = im.copy()
                    work.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)
                best_arr = np.array(work)
            except Exception:
                continue
        start = idx + 3

    if best_arr is None:
        with _embedded_scan_miss_lock:
            if len(_embedded_scan_miss_cache) >= _embedded_scan_miss_cache_max:
                _embedded_scan_miss_cache.clear()
            _embedded_scan_miss_cache.add(miss_key)

    return best_arr


class ThumbnailExtractor(QObject):
    """Fast thumbnail extractor for immediate display."""

    def __init__(self):
        super().__init__()

    def extract_thumbnail_from_raw(self, file_path: str, max_size: int = 512,
                                   allow_scan_fallback: bool = True,
                                   raw_object: Optional[rawpy.RawPy] = None) -> Optional[np.ndarray]:
        """Extract embedded thumbnail from RAW file and resize to max_size."""
        try:
            if raw_object is not None:
                return self._extract_from_raw_obj(raw_object, file_path, max_size)
                
            with rawpy.imread(file_path) as raw:
                return self._extract_from_raw_obj(raw, file_path, max_size)
        except Exception:
            if not allow_scan_fallback:
                return None
            return extract_embedded_jpeg_by_scan(file_path, max_size)

    def _extract_from_raw_obj(self, raw, file_path, max_size):
        """Internal helper to extract thumb from an open rawpy object."""
        try:
            thumb = raw.extract_thumb()
            
            if thumb.format == rawpy.ThumbFormat.JPEG:
                from PIL import Image
                jpeg_image = Image.open(io.BytesIO(thumb.data))
                
                if jpeg_image.mode != 'RGB':
                    jpeg_image = jpeg_image.convert('RGB')
                    
                w, h = jpeg_image.size
                if w > max_size or h > max_size:
                    jpeg_image.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)
                    
                return np.array(jpeg_image)
                
            elif thumb.format == rawpy.ThumbFormat.BITMAP:
                thumb_array = thumb.data.copy()
                if thumb_array is None or not hasattr(thumb_array, 'shape'):
                    return None
                
                h, w = thumb_array.shape[:2]
                if w > max_size or h > max_size:
                     from PIL import Image
                     pil_thumb = Image.fromarray(thumb_array)
                     pil_thumb.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)
                     return np.array(pil_thumb)
                return thumb_array
            
            return None
        except Exception:
            return None

    def extract_preview_from_raw(self, file_path: str, max_size: int = 2048,
                                 allow_scan_fallback: bool = True) -> Optional[np.ndarray]:
        """Extract high-quality preview from RAW file (embedded JPEG)."""
        # Reuse thumbnail logic but with larger size
        return self.extract_thumbnail_from_raw(
            file_path,
            max_size=max_size,
            allow_scan_fallback=allow_scan_fallback,
        )

    def extract_thumbnail_from_image(self, file_path: str, max_size: int = 512,
                                     target_size: Optional[QSize] = None) -> Optional[Union[np.ndarray, QImage]]:
        """Extract thumbnail from regular image file. Returns QImage (preferred) or np.ndarray."""
        try:
            reader = QImageReader(file_path)
            reader.setAutoTransform(True)
            size = reader.size()
            if size.isValid() and size.width() > 0 and size.height() > 0:
                w, h = size.width(), size.height()
                
                # OPTIMIZATION: If we have a target_size, scale directly to it during decode.
                # This is much faster than decoding to 512px and then scaling again.
                if target_size is not None and isinstance(target_size, QSize) and target_size.isValid():
                    reader.setScaledSize(target_size)
                elif w > max_size or h > max_size:
                    scale = min(max_size / w, max_size / h)
                    reader.setScaledSize(QSize(max(1, int(w * scale)), max(1, int(h * scale))))
            
            image = reader.read()
            if not image.isNull():
                # OPTIMIZATION: Return QImage directly to avoid Numpy conversion
                return image
        except Exception:
            pass

        try:
            with Image.open(file_path) as img:
                if target_size is not None and isinstance(target_size, QSize):
                    img.thumbnail((target_size.width(), target_size.height()), Image.Resampling.BILINEAR)
                else:
                    img.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)
                
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

    def extract_exif_data(self, file_path: str, raw_object: Optional[rawpy.RawPy] = None) -> Optional[Dict[str, Any]]:
        """Extract EXIF data from image file with RAW-specific orientation fallbacks."""
        # Check SQLite cache first
        cached = self.cache.get_exif(file_path)
        if cached:
            # Check if cache version matches current logic
            cached_ver = cached.get('raw_exif_sensor_meta_ver', 0)
            if cached_ver < RAW_EXIF_SENSOR_META_VER:
                if os.environ.get("RAWVIEWER_VERBOSE_ORIENTATION_LOGS") == "1":
                    print(f"[ORIENTATION] EXIFExtractor: Stale cache version ({cached_ver} < {RAW_EXIF_SENSOR_META_VER}) for {os.path.basename(file_path)}, forcing re-extraction...")
            else:
                if os.environ.get("RAWVIEWER_VERBOSE_ORIENTATION_LOGS") == "1":
                    print(f"[ORIENTATION] EXIFExtractor: Found valid cached orientation={cached.get('orientation')} for {os.path.basename(file_path)}")
                return cached

        try:
            # First pass: standard exifread (works for JPEGs and many RAW containers)
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)
            
            # Standard orientation tags
            orientation = 1
            orientation_tag_found = None
            
            # DEBUG: Log all potential orientation tags to help troubleshoot
            if os.environ.get("RAWVIEWER_VERBOSE_ORIENTATION_LOGS") == "1":
                orient_tags = {k: v for k, v in tags.items() if 'orient' in k.lower()}
                if orient_tags:
                    print(f"[ORIENTATION] EXIFExtractor debug for {os.path.basename(file_path)}: Found orientation-like tags: {orient_tags}")

            for tag_name in ('Image Orientation', 'EXIF Orientation', 'Orientation', 'MakerNote Orientation', 'EXIF SceneType', 'Sony Orientation', 'Sony Orientation 2'):
                tag = tags.get(tag_name)
                if tag:
                    # Try to get the numeric value directly
                    try:
                        if hasattr(tag, 'values') and tag.values:
                            val = tag.values[0]
                            if isinstance(val, int):
                                orientation = val
                                orientation_tag_found = tag_name
                                break
                    except: pass
                    
                    # Fallback to string mapping (same as main.py)
                    orientation_str = str(tag).strip()
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
                    if orientation_str in orientation_map:
                        orientation = orientation_map[orientation_str]
                        orientation_tag_found = tag_name
                        break
            
            if os.environ.get("RAWVIEWER_VERBOSE_ORIENTATION_LOGS") == "1" and orientation != 1:
                print(f"[ORIENTATION] EXIFExtractor: Found orientation={orientation} via tag '{orientation_tag_found}' for {os.path.basename(file_path)}")
            
            camera_make = str(tags.get('Image Make', '')).strip()
            camera_model = str(tags.get('Image Model', '')).strip()
            
            exif_dict = {k: str(v) for k, v in tags.items()}
            
            # Dimensions
            original_width = 0
            original_height = 0
            for tag in ('EXIF ExifImageWidth', 'Image ImageWidth'):
                if tag in tags:
                    try: 
                        original_width = int(tags[tag].values[0])
                        break
                    except: pass
            for tag in ('EXIF ExifImageLength', 'Image ImageLength'):
                if tag in tags:
                    try:
                        original_height = int(tags[tag].values[0])
                        break
                    except: pass

            # Second pass: If it's a RAW file, use rawpy to verify dimensions and orientation (flip)
            if self._is_raw_file(file_path):
                try:
                    if raw_object is not None:
                        sizes = raw_object.sizes
                        original_width = sizes.width
                        original_height = sizes.height
                        if orientation == 1 and sizes.flip != 0:
                            # Map LibRaw flip codes to EXIF Orientation
                            # LibRaw flip: 0=0, 3=180, 5=90CCW, 6=90CW
                            # EXIF Orient: 1=0, 3=180, 8=90CCW, 6=90CW
                            flip_map = {0: 1, 3: 3, 5: 8, 6: 6}
                            orientation = flip_map.get(sizes.flip, sizes.flip)
                            if os.environ.get("RAWVIEWER_VERBOSE_ORIENTATION_LOGS") == "1":
                                print(f"[ORIENTATION] EXIFExtractor: Falling back to LibRaw flip={sizes.flip} -> Orientation {orientation} for {os.path.basename(file_path)}")
                    else:
                        with rawpy.imread(file_path) as raw:
                            sizes = raw.sizes
                            original_width = sizes.width
                            original_height = sizes.height
                            if orientation == 1 and sizes.flip != 0:
                                flip_map = {0: 1, 3: 3, 5: 8, 6: 6}
                                orientation = flip_map.get(sizes.flip, sizes.flip)
                except Exception:
                    pass

            # Extract technical metadata for top-level cache columns
            focal_length = None
            aperture = None
            iso = None
            
            # Focal Length
            for tag in ('EXIF FocalLength', 'EXIF FocalLengthIn35mmFilm'):
                if tag in exif_dict:
                    try:
                        val = exif_dict[tag]
                        if '/' in val:
                            num, den = val.split('/')
                            focal_length = f"{round(float(num) / float(den))}mm"
                        else:
                            focal_length = f"{round(float(val))}mm"
                        break
                    except: pass
            
            # Aperture
            for tag in ('EXIF FNumber', 'EXIF ApertureValue'):
                if tag in exif_dict:
                    try:
                        val = exif_dict[tag]
                        if '/' in val:
                            num, den = val.split('/')
                            aperture = f"f/{float(num) / float(den):.1f}"
                        else:
                            aperture = f"f/{float(val):.1f}"
                        break
                    except: pass
            
            # ISO
            for tag in ('EXIF ISOSpeedRatings', 'EXIF ISO', 'EXIF PhotographicSensitivity'):
                if tag in exif_dict:
                    try:
                        val = exif_dict[tag]
                        if '/' in val:
                            num, den = val.split('/')
                            iso = f"ISO {int(float(num) / float(den))}"
                        else:
                            iso = f"ISO {val}"
                        break
                    except: pass

            # Build return dict
            result = {
                'orientation': orientation,
                'camera_make': camera_make,
                'camera_model': camera_model,
                'exif_data': exif_dict,
                'original_width': original_width,
                'original_height': original_height,
                'capture_time': exif_dict.get('EXIF DateTimeOriginal') or exif_dict.get('Image DateTime'),
                'focal_length': focal_length,
                'aperture': aperture,
                'iso': iso,
                'verified_orientation': True,
                'raw_exif_sensor_meta_ver': RAW_EXIF_SENSOR_META_VER,
            }
            
            if os.environ.get("RAWVIEWER_VERBOSE_ORIENTATION_LOGS") == "1":
                print(f"[ORIENTATION] EXIFExtractor: Successfully returning metadata with orientation={orientation} for {os.path.basename(file_path)}")
            
            return result
            
        except Exception as e:
            if os.environ.get("RAWVIEWER_VERBOSE_ORIENTATION_LOGS") == "1":
                print(f"[ORIENTATION] EXIFExtractor error for {os.path.basename(file_path)}: {e}")
            pass

        return {
            'orientation': orientation if 'orientation' in locals() else 1,
            'camera_make': camera_make if 'camera_make' in locals() else '',
            'camera_model': camera_model if 'camera_model' in locals() else '',
            'exif_data': {},
            'verified_orientation': True
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
            'no_auto_bright': True,
            'gamma': (2.222, 4.5),  # Standard sRGB gamma
            'bright': 1.0,
            # 'highlight': 0,  # Removed for rawpy 0.25.0 compatibility
            # 'shadow': 0,  # Removed for rawpy 0.25.0 compatibility
            'user_flip': 0,  # Force rawpy to ignore EXIF orientation
            'demosaic_algorithm': rawpy.DemosaicAlgorithm.LINEAR, # MUCH faster than default for fast previews
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
            # DCB is high quality but slower - disable for fast processing
            params.update({
                'use_camera_wb': True,
                'dcb_enhance': False if file_size_mb > 20 else True 
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
                if rgb_image is not None and hasattr(rgb_image, 'dtype') and rgb_image.dtype == np.uint16:
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
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        return ext in RAW_FILE_EXTENSIONS

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
            self.cache.put_thumbnail(self.file_path, thumbnail, None)

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
                if rgb_image is not None:
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
        if image_array is None:
            return None
            
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
            # Mirrored horizontal + Rotated 270° CW (k=1 CCW)
            return np.rot90(np.fliplr(image_array), 1)
        elif orientation == 6:
            # Orientation 6: Image is rotated 90° CW.
            # We need to rotate it 90° CW (k=3) to fix it.
            return np.rot90(image_array, 3)
        elif orientation == 7:
            # Mirror LR + rotate 90° CW
            return np.rot90(np.fliplr(image_array), 3)
        elif orientation == 8:
            # Orientation 8: Image is rotated 270° CW (90° CCW).
            # We need to rotate it 90° CCW (k=1) to fix it.
            return np.rot90(image_array, 1)
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
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        return ext in RAW_FILE_EXTENSIONS

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
