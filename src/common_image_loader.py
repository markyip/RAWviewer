"""
共用圖像載入函數 - 統一處理所有圖像類型的載入邏輯

這個模組提供統一的圖像載入函數，避免重複代碼。
"""

import io
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

_io_pressure_until = 0.0


def raise_process_fd_limit(soft_target: int = 4096) -> None:
    """Raise the soft RLIMIT_NOFILE cap (macOS default 256 is too low for gallery prefetch)."""
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard == resource.RLIM_INFINITY:
            target = soft_target
        else:
            target = min(int(hard), soft_target)
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except Exception:
        pass


def is_emfile_error(exc: BaseException) -> bool:
    errno = getattr(exc, "errno", None)
    if errno == 24:
        return True
    return "too many open files" in str(exc).lower()


def note_emfile_pressure(duration_s: float = 12.0) -> None:
    """Back off concurrent I/O after EMFILE so the process can recover."""
    global _io_pressure_until
    _io_pressure_until = max(_io_pressure_until, time.time() + max(5.0, duration_s))


def io_pressure_active() -> bool:
    return time.time() < _io_pressure_until

import numpy as np

from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QLoggingCategory
# Silence qt.imageformats warnings (e.g. missing TIFF tag warnings on RAW files)
QLoggingCategory.setFilterRules("qt.imageformats.warning=false\nqt.imageformats.tiff.warning=false")
# PIL Image will be imported lazily to avoid import delays

from image_cache import get_image_cache
from raw_file_extensions import RAW_FILE_EXTENSIONS


_CAPTURE_TIME_FORMATS = (
    "%H:%M:%S %Y-%m-%d",  # normalized display / cache
    "%Y:%m:%d %H:%M:%S",  # EXIF / exifread default
    "%Y-%m-%d %H:%M:%S",
)


def parse_capture_time_to_timestamp(capture_time: Optional[str]) -> float:
    """Parse EXIF-style capture time strings to a Unix timestamp (0 if unknown)."""
    if not capture_time:
        return 0.0
    s = str(capture_time).strip()
    if not s:
        return 0.0
    for fmt in _CAPTURE_TIME_FORMATS:
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    return 0.0


def decode_embedded_jpeg_bytes(
    jpeg_bytes: bytes,
    max_size: int = 0,
) -> Optional[np.ndarray]:
    """Decode embedded JPEG to RGB; applies the JPEG segment's own EXIF orientation."""
    import logging
    logger = logging.getLogger(__name__)
    try:
        from PIL import Image, ImageOps, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        im = Image.open(io.BytesIO(jpeg_bytes))
        im = ImageOps.exif_transpose(im)
        if im.mode != "RGB":
            im = im.convert("RGB")
        w, h = im.size
        if max_size > 0 and (w > max_size or h > max_size):
            im.thumbnail((max_size, max_size), Image.Resampling.HAMMING)
        return np.array(im)
    except Exception as e:
        logger.error("[DECODE] Error decoding embedded JPEG bytes: %s", e)
        return None


def orientation_from_embedded_jpeg_bytes(jpeg_bytes: bytes) -> int:
    """EXIF Orientation (1–8) from an embedded JPEG; portrait pixels → 6 if tag missing."""
    try:
        from PIL import Image

        im = Image.open(io.BytesIO(jpeg_bytes))
        exif = im.getexif()
        if exif:
            o = exif.get(274)
            if o is None:
                o = exif.get(0x0112)
            if isinstance(o, int) and 1 <= o <= 8:
                return o
        w, h = im.size
        if h > w:
            return 6
    except Exception:
        pass
    return 1


def normalize_capture_time_string(capture_time: Optional[str]) -> Optional[str]:
    """Canonical form used for sorting and status display."""
    ts = parse_capture_time_to_timestamp(capture_time)
    if ts <= 0:
        return None
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S %Y-%m-%d")


def probe_capture_timestamp_from_file(file_path: str) -> float:
    """Lightweight EXIF DateTime probe when cache has no capture_time."""
    try:
        import metadata_backend

        tags = metadata_backend.process_file_from_path(file_path, details=False)
        for tag_name in (
            "EXIF DateTimeOriginal",
            "EXIF DateTimeDigitized",
            "Image DateTime",
            "EXIF DateTime",
        ):
            if tag_name in tags:
                ts = parse_capture_time_to_timestamp(str(tags[tag_name]))
                if ts > 0:
                    return ts
    except Exception:
        pass
    return 0.0


def capture_timestamp_for_sort(
    file_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    file_mtime: float = 0.0,
    *,
    file_birthtime: float = 0.0,
    probe_file: bool = True,
    semantic_capture_time: Optional[str] = None,
) -> float:
    """Timestamp for folder/gallery sort (see resolve_folder_sort_timestamp)."""
    _has_capture, ts, _source = resolve_folder_sort_timestamp(
        file_path,
        metadata,
        file_mtime,
        file_birthtime=file_birthtime,
        probe_file=probe_file,
        semantic_capture_time=semantic_capture_time,
    )
    return ts


def resolve_folder_sort_timestamp(
    file_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    file_mtime: float = 0.0,
    *,
    file_birthtime: float = 0.0,
    probe_file: bool = True,
    semantic_capture_time: Optional[str] = None,
    probed_capture_timestamp: float = 0.0,
) -> Tuple[bool, float, str]:
    """
    Returns (has_capture_time, sort_timestamp, sort_source).

    Priority for sort_timestamp:
      1. EXIF / semantic / probe capture time
      2. Filesystem birth time (creation on Windows/macOS), when no capture time
      3. File modification time (mtime)

    All files share one timeline (camera + AI exports interleave by these rules).
    """
    if metadata:
        ts = parse_capture_time_to_timestamp(metadata.get("capture_time"))
        if ts > 0:
            return True, ts, "cache"
    if semantic_capture_time:
        ts = parse_capture_time_to_timestamp(semantic_capture_time)
        if ts > 0:
            return True, ts, "semantic"
    if probed_capture_timestamp > 0:
        return True, float(probed_capture_timestamp), "probe"
    if probe_file and file_path:
        ts = probe_capture_timestamp_from_file(file_path)
        if ts > 0:
            return True, ts, "probe"

    birth = float(file_birthtime or 0.0)
    mtime = float(file_mtime or 0.0)
    if file_path:
        try:
            st = os.stat(file_path)
            if birth <= 0:
                birth = float(getattr(st, "st_birthtime", st.st_mtime))
            if mtime <= 0:
                mtime = float(st.st_mtime)
        except OSError:
            pass
    if birth > 0:
        return False, birth, "birth"
    if mtime > 0:
        return False, mtime, "mtime"
    return False, 0.0, "unknown"


def folder_sort_key_tuple(
    file_path: str,
    timestamp: float,
    newest_first: bool,
    *,
    has_capture_time: bool = True,
) -> tuple:
    """Stable folder/gallery ordering on one chronological axis."""
    del has_capture_time  # kept for call-site compatibility
    base_name = os.path.basename(file_path).lower()
    stem = os.path.splitext(base_name)[0]
    ext = os.path.splitext(base_name)[1]
    raw_rank = 1 if is_raw_file(file_path) else 0
    primary_ts = -timestamp if newest_first else timestamp
    return (primary_ts, stem, raw_rank, ext, base_name)


def use_libraw_consistent_preview_first() -> bool:
    """
    When True (default), single-image RAW avoids embedded-JPEG preview paths so fit and zoom
    share the same LibRaw postprocess look. Set RAWVIEWER_LIBRAW_CONSISTENT_PREVIEW=0 to restore
    the faster embedded-preview first paint.

    Full-resolution embedded JPEGs (see use_full_embedded_raw_preview) bypass LibRaw even when
    this flag is on.
    """
    from PyQt6.QtCore import QSettings
    settings = QSettings("RAWviewer", "RAWviewer")
    if settings.contains("use_embedded_jpeg_workflow"):
        use_embedded = settings.value("use_embedded_jpeg_workflow", True, type=bool)
        if not use_embedded:
            return True

    v = os.environ.get("RAWVIEWER_LIBRAW_CONSISTENT_PREVIEW", "0").strip().lower()
    return v not in ("0", "false", "no", "off")


# Embedded JPEG long edge must reach this fraction of sensor long edge to count as "full size".
FULL_EMBEDDED_SENSOR_COVERAGE = 0.92


def use_full_embedded_raw_preview() -> bool:
    """
    When True (default), use camera-embedded JPEG for fit/zoom when it covers sensor resolution,
    avoiding LibRaw demosaic. Set RAWVIEWER_USE_FULL_EMBEDDED_JPEG=0 to always prefer LibRaw.
    """
    from PyQt6.QtCore import QSettings
    settings = QSettings("RAWviewer", "RAWviewer")
    if settings.contains("use_embedded_jpeg_workflow"):
        return settings.value("use_embedded_jpeg_workflow", True, type=bool)

    v = os.environ.get("RAWVIEWER_USE_FULL_EMBEDDED_JPEG", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def sensor_pixel_dimensions(exif_data: Optional[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
    """Sensor / full-frame pixel size from EXIF cache, if known."""
    if not exif_data:
        return None
    ow = exif_data.get("original_width")
    oh = exif_data.get("original_height")
    try:
        w, h = int(ow), int(oh)
    except (TypeError, ValueError):
        return None
    if w > 0 and h > 0:
        return w, h
    return None


def image_covers_sensor_resolution(
    img_width: int,
    img_height: int,
    exif_data: Optional[Dict[str, Any]],
    *,
    coverage: float = FULL_EMBEDDED_SENSOR_COVERAGE,
) -> bool:
    """
    True when displayed/embedded pixels are large enough to treat as full resolution for zoom
    (no on-demand LibRaw full decode required).
    """
    if img_width <= 0 or img_height <= 0:
        return False
    sensor = sensor_pixel_dimensions(exif_data)
    if not sensor:
        return max(img_width, img_height) >= 3000
    sw, sh = sensor
    sensor_long = max(sw, sh)
    img_long = max(img_width, img_height)
    return img_long >= sensor_long * coverage


def check_cache_for_image(file_path: str, use_full_resolution: bool = False) -> Tuple[Optional[Any], Optional[str]]:
    """
    檢查快取中是否存在圖像或相關數據。
    
    返回: (數據, 快取類型) 或 (None, None)
    類型包括: 'full_image', 'pixmap', 'thumbnail', 'exif'

    注意：若只有 EXIF 也會視為命中；單圖「預載像素」請用 check_memory_cache_for_image，避免誤跳過預載。
    """
    cache = get_image_cache()
    
    # 1. 如果請求全解析度，優先檢查全圖像快取
    if use_full_resolution:
        full_image = cache.get_full_image(file_path)
        if full_image is not None:
            return full_image, 'full_image'
        if is_raw_file(file_path) and use_full_embedded_raw_preview():
            exif_data = cache.get_exif(file_path)
            preview = cache.get_preview(file_path)
            if preview is not None:
                h, w = preview.shape[:2]
                if image_covers_sensor_resolution(w, h, exif_data):
                    return preview, 'full_image'
            
    # 2. 檢查 Pixmap 快取 (常用於 QSS/一般顯示)
    pixmap = cache.get_pixmap(file_path)
    if pixmap is not None and not pixmap.isNull():
        return pixmap, 'pixmap'
        
    # 3. 如果不是必須全解析度，檢查記憶體縮圖快取
    if not use_full_resolution:
        thumbnail = cache.get_thumbnail(file_path)
        if thumbnail is not None:
            return thumbnail, 'thumbnail'
            
    # 4. 檢查 EXIF 元數據快取
    exif_data = cache.get_exif(file_path)
    if exif_data:
        return exif_data, 'exif'
        
    return None, None

def check_memory_cache_for_image(file_path: str, use_full_resolution: bool = False) -> Tuple[Optional[Any], Optional[str]]:
    """
    檢查記憶體快取中是否存在圖像 (非阻塞，不讀取磁盤)。
    
    返回: (數據, 快取類型) 或 (None, None)
    類型包括: 'full_image', 'pixmap', 'thumbnail'
    """
    cache = get_image_cache()
    
    # 1. 如果請求全解析度，優先檢查全圖像快取
    if use_full_resolution:
        # 直接訪問記憶體 LRUCache，避免觸發磁盤讀取
        full_image = cache.full_image_cache.get(file_path)
        if full_image is not None:
            return full_image, 'full_image'
        if is_raw_file(file_path) and use_full_embedded_raw_preview():
            exif_data = cache.exif_cache.get(file_path)
            preview = cache.preview_cache.get(file_path)
            if preview is not None:
                h, w = preview.shape[:2]
                if image_covers_sensor_resolution(w, h, exif_data):
                    return preview, 'full_image'
            
    # 2. 檢查 Pixmap 快取
    pixmap = cache.pixmap_cache.get(file_path)
    if pixmap is not None and not pixmap.isNull():
        return pixmap, 'pixmap'
        
    # 3. 如果不是必須全解析度，檢查記憶體縮圖／RAW 預覽快取（不觸發磁碟）
    if not use_full_resolution:
        thumbnail = cache.thumbnail_cache.get(file_path)
        if thumbnail is not None:
            return thumbnail, 'thumbnail'
        # 單圖導覽常以 preview_cache 為載入結果；全尺寸內嵌 JPEG 時等同完整像素
        if is_raw_file(file_path):
            preview = cache.preview_cache.get(file_path)
            if preview is not None:
                exif_data = cache.exif_cache.get(file_path)
                if image_covers_sensor_resolution(
                    preview.shape[1], preview.shape[0], exif_data
                ):
                    return preview, 'preview'
                if not use_libraw_consistent_preview_first():
                    return preview, 'preview'

    return None, None

def is_raw_file(file_path: str) -> bool:
    """檢查是否為 RAW 文件（與 semantic `format:raw` 使用同一附檔名集合）。"""
    ext = os.path.splitext(file_path)[1].lower().lstrip(".")
    return ext in RAW_FILE_EXTENSIONS


def dng_prefers_embedded_preview_first(file_path: str) -> bool:
    """Composite/HDR DNGs often fail LibRaw; use embedded JPEG before sensor decode."""
    if os.path.splitext(file_path)[1].lower() != ".dng":
        return False
    return "-hdr" in os.path.basename(file_path).lower()


def is_tiff_file(file_path: str) -> bool:
    """檢查是否為 TIFF 文件（包括錯誤擴展名的情況）"""
    file_ext = os.path.splitext(file_path)[1].lower()
    is_tiff = file_ext in ('.tiff', '.tif')
    
    # 檢查文件內容以檢測錯誤擴展名的 TIFF 文件
    if not is_tiff:
        try:
            from PIL import Image
            with Image.open(file_path) as test_img:
                if test_img.format in ('TIFF', 'TIF'):
                    is_tiff = True
        except:
            pass  # 不是 PIL 可讀文件或不是 TIFF
    
    return is_tiff


def _regular_image_max_edge() -> int:
    """Safety cap when full-resolution Qt/PIL decode fails (e.g. huge panorama JPEG)."""
    raw = os.environ.get("RAWVIEWER_REGULAR_IMAGE_MAX_EDGE", "16384").strip()
    try:
        v = int(raw)
    except Exception:
        v = 16384
    return max(2048, min(v, 65536))


def _pil_file_to_qpixmap(file_path: str, max_edge: int = 0) -> QPixmap:
    """Load JPEG/PNG/WebP via PIL with EXIF orientation; optional downscale."""
    import logging

    logger = logging.getLogger(__name__)
    try:
        from PIL import Image, ImageOps, ImageFile

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(file_path) as pil_image:
            pil_image = ImageOps.exif_transpose(pil_image)
            if max_edge > 0:
                w, h = pil_image.size
                if max(w, h) > max_edge:
                    pil_image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            if pil_image.mode not in ("RGB", "L"):
                pil_image = pil_image.convert("RGB")
            if pil_image.mode == "L":
                pil_image = pil_image.convert("RGB")
            width, height = pil_image.size
            qimage = QImage(
                pil_image.tobytes("raw", "RGB"),
                width,
                height,
                QImage.Format.Format_RGB888,
            )
            if qimage.isNull():
                return QPixmap()
            return QPixmap.fromImage(qimage)
    except Exception as e:
        logger.debug(
            "PIL load failed for %s: %s", os.path.basename(file_path), e
        )
        return QPixmap()


def load_pixmap_safe(file_path: str, max_edge: int = 0) -> QPixmap:
    """安全載入 QPixmap，對 TIFF 文件使用 PIL 以避免 Qt 警告"""
    cache = get_image_cache()
    
    # 檢查快取
    cached_pixmap = cache.get_pixmap(file_path)
    if cached_pixmap is not None and not cached_pixmap.isNull():
        return cached_pixmap

    if is_raw_file(file_path):
        preview_max = max_edge if max_edge > 0 else 2048
        return load_raw_preview_pixmap(file_path, max_size=preview_max)
    
    # 對於 TIFF 文件，使用 PIL 避免 Qt 警告
    if is_tiff_file(file_path):
        try:
            from PIL import Image, ImageOps
            with Image.open(file_path) as pil_image:
                # Apply EXIF orientation correction
                pil_image = ImageOps.exif_transpose(pil_image)
                
                if pil_image.mode not in ('RGB', 'L'):
                    pil_image = pil_image.convert('RGB')
                if pil_image.mode == 'L':
                    pil_image = pil_image.convert('RGB')
                
                width, height = pil_image.size
                qimage = QImage(pil_image.tobytes('raw', 'RGB'), 
                               width, height, 
                               QImage.Format.Format_RGB888)
                
                if not qimage.isNull():
                    pixmap = QPixmap.fromImage(qimage)
                    cache.put_pixmap(file_path, pixmap)
                    return pixmap
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"PIL fallback failed for TIFF: {os.path.basename(file_path)}: {e}")
            # 對於 TIFF 文件，永遠不要使用 QPixmap(file_path)，因為會觸發警告
            return QPixmap()
    
    # 對於其他格式（JPEG, PNG, etc.），使用 QImageReader 並啟用自動方向轉換
    try:
        from PyQt6.QtCore import QSize
        from PyQt6.QtGui import QImageReader

        reader = QImageReader(file_path)
        reader.setAutoTransform(True)  # 自動應用 EXIF 方向
        if max_edge > 0:
            original_size = reader.size()
            if original_size.isValid():
                w, h = original_size.width(), original_size.height()
                if max(w, h) > max_edge:
                    if w >= h:
                        sw = max_edge
                        sh = max(1, int(h * max_edge / max(w, 1)))
                    else:
                        sh = max_edge
                        sw = max(1, int(w * max_edge / max(h, 1)))
                    reader.setScaledSize(QSize(sw, sh))
        pixmap = QPixmap.fromImageReader(reader)
        if not pixmap.isNull():
            cache.put_pixmap(file_path, pixmap)
            return pixmap
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"QImageReader failed for {os.path.basename(file_path)}: {e}")

    pixmap = _pil_file_to_qpixmap(file_path, max_edge=max_edge)
    if not pixmap.isNull():
        cache.put_pixmap(file_path, pixmap)
        return pixmap

    # 回退到直接使用 QPixmap（不會應用方向，但比沒有好）
    pixmap = QPixmap(file_path)
    if not pixmap.isNull():
        cache.put_pixmap(file_path, pixmap)
        return pixmap

    if max_edge <= 0:
        safety_edge = _regular_image_max_edge()
        pixmap = _pil_file_to_qpixmap(file_path, max_edge=safety_edge)
        if not pixmap.isNull():
            cache.put_pixmap(file_path, pixmap)
    return pixmap


def exif_display_dimensions(
    width: int,
    height: int,
    orientation: int,
) -> Tuple[int, int]:
    """Pixel width/height after EXIF orientation is applied (display size).

    Sensor dimensions are often stored landscape (w > h) while orientation 6/8
  means the image should display portrait — swap only when w > h.
    """
    w, h = int(width or 0), int(height or 0)
    if w <= 0 or h <= 0:
        return w, h
    o = int(orientation or 1)
    if o in (5, 6, 7, 8) and w > h:
        return h, w
    return w, h


def exif_display_aspect_ratio(
    width: int,
    height: int,
    orientation: int,
) -> float:
    """Width/height ratio for layout after EXIF orientation."""
    dw, dh = exif_display_dimensions(width, height, orientation)
    if dh <= 0:
        return 1.5
    return dw / dh


def pixmap_matches_exif_display(
    pixmap_width: int,
    pixmap_height: int,
    original_width: int,
    original_height: int,
    orientation: int,
    *,
    tolerance: float = 0.14,
) -> bool:
    """True when pixmap pixels already match EXIF display orientation (no extra rotation)."""
    pw, ph = int(pixmap_width or 0), int(pixmap_height or 0)
    if pw <= 0 or ph <= 0:
        return False
    dw, dh = exif_display_dimensions(original_width, original_height, orientation)
    if dw <= 0 or dh <= 0:
        return False
    if (ph > pw) != (dh > dw):
        return False
    exp_ar = dw / dh
    act_ar = pw / ph
    if exp_ar <= 0 or act_ar <= 0:
        return False
    ratio = exp_ar / act_ar if exp_ar >= act_ar else act_ar / exp_ar
    return ratio <= (1.0 + tolerance)


def array_matches_exif_display(
    pixel_width: int,
    pixel_height: int,
    exif_data: Optional[Dict[str, Any]],
    *,
    tolerance: float = 0.14,
) -> bool:
    """True when RGB/thumbnail pixels already match container EXIF display orientation."""
    if not exif_data:
        return True
    ow = int(exif_data.get("original_width") or 0)
    oh = int(exif_data.get("original_height") or 0)
    if ow <= 0 or oh <= 0:
        return True
    return pixmap_matches_exif_display(
        pixel_width,
        pixel_height,
        ow,
        oh,
        int(exif_data.get("orientation", 1) or 1),
        tolerance=tolerance,
    )


def apply_container_orientation_to_array(
    image_array: np.ndarray,
    orientation: int,
    exif_data: Optional[Dict[str, Any]] = None,
) -> Optional[np.ndarray]:
    """Apply container EXIF orientation once; skip when pixels already match display."""
    if image_array is None:
        return None
    o = int(orientation or 1)
    if o == 1:
        return image_array

    h, w = image_array.shape[:2]
    
    # Safety guard: Determine expected vs actual display orientation to avoid double rotation.
    # If original dimensions are not available, we assume the camera sensor is landscape (e.g. 3x2).
    ow = int((exif_data or {}).get("original_width") or 0)
    oh = int((exif_data or {}).get("original_height") or 0)
    if ow <= 0 or oh <= 0:
        ow, oh = 3, 2  # Default to landscape sensor aspect ratio
    
    dw, dh = exif_display_dimensions(ow, oh, o)
    if (dh > dw) == (h > w):
        import logging
        logging.getLogger(__name__).debug(
            f"apply_container_orientation_to_array: skipping orientation {o} "
            f"because array shape {w}x{h} already matches expected display orientation."
        )
        return image_array

    if o == 2:
        return np.fliplr(image_array)
    if o == 3:
        return np.rot90(image_array, 2)
    if o == 4:
        return np.flipud(image_array)
    if o == 5:
        return np.rot90(np.fliplr(image_array), 1)
    if o == 6:
        return np.rot90(image_array, 3)
    if o == 7:
        return np.rot90(np.fliplr(image_array), 3)
    if o == 8:
        return np.rot90(image_array, 1)
    return image_array


def rgb_array_to_qpixmap(rgb_image: np.ndarray) -> QPixmap:
    """Convert oriented uint8 RGB/grayscale ndarray to QPixmap."""
    if rgb_image is None or not hasattr(rgb_image, "shape"):
        return QPixmap()
    if rgb_image.dtype != np.uint8:
        rgb_image = rgb_image.astype(np.uint8)
    h, w = rgb_image.shape[:2]
    if len(rgb_image.shape) == 2:
        qimg = QImage(
            rgb_image.tobytes(), w, h, w, QImage.Format.Format_Grayscale8
        )
    elif rgb_image.shape[2] >= 3:
        channels = int(rgb_image.shape[2])
        if not rgb_image.flags["C_CONTIGUOUS"]:
            rgb_image = np.ascontiguousarray(rgb_image)
        q_format = (
            QImage.Format.Format_RGB888
            if channels == 3
            else QImage.Format.Format_RGBA8888
        )
        qimg = QImage(
            rgb_image.tobytes(), w, h, channels * w, q_format
        )
    else:
        return QPixmap()
    return QPixmap.fromImage(qimg) if not qimg.isNull() else QPixmap()


def load_raw_preview_array(
    file_path: str,
    max_size: int = 2048,
    *,
    apply_container_orientation: bool = True,
) -> Optional[np.ndarray]:
    """Single RAW preview path: cache → ThumbnailExtractor → container EXIF (once)."""
    cache = get_image_cache()
    for getter in (cache.get_thumbnail, cache.get_preview):
        try:
            cached = getter(file_path)
        except Exception:
            cached = None
        if cached is not None and hasattr(cached, "shape"):
            return np.asarray(cached, dtype=np.uint8)

    from enhanced_raw_processor import ThumbnailExtractor

    thumb = ThumbnailExtractor().extract_thumbnail_from_raw(
        file_path, max_size=max_size, allow_scan_fallback=True
    )
    if thumb is None:
        return None

    if hasattr(thumb, "width"):
        from enhanced_raw_processor import _qimage_to_rgb_array

        thumb = _qimage_to_rgb_array(thumb)
    if thumb is None or not isinstance(thumb, np.ndarray):
        return None

    if not apply_container_orientation:
        return thumb

    exif_data = cache.get_exif(file_path)
    if not exif_data:
        try:
            from enhanced_raw_processor import EXIFExtractor

            exif_data = EXIFExtractor().extract_exif_data(file_path)
        except Exception:
            exif_data = None
    orientation = int((exif_data or {}).get("orientation", 1) or 1)
    if orientation != 1:
        thumb = apply_container_orientation_to_array(thumb, orientation, exif_data)
    return thumb


def load_raw_preview_pixmap(file_path: str, max_size: int = 2048) -> QPixmap:
    """Oriented RAW preview as QPixmap (shared by single view fallbacks and gallery)."""
    cache = get_image_cache()
    cached_pixmap = cache.get_pixmap(file_path)
    if cached_pixmap is not None and not cached_pixmap.isNull():
        return cached_pixmap

    arr = load_raw_preview_array(file_path, max_size=max_size)
    if arr is None:
        return QPixmap()
    pixmap = rgb_array_to_qpixmap(arr)
    if not pixmap.isNull():
        cache.put_pixmap(file_path, pixmap)
    return pixmap


def exif_rotation_degrees_for_pixmap(
    pixmap_width: int,
    pixmap_height: int,
    original_width: int,
    original_height: int,
    orientation: int,
) -> int:
    """Clockwise rotation (0/90/180/270) to align pixmap with EXIF display orientation."""
    pw, ph = int(pixmap_width or 0), int(pixmap_height or 0)
    if pw <= 0 or ph <= 0:
        return 0
    o = int(orientation or 1)
    ow, oh = int(original_width or 0), int(original_height or 0)
    act_portrait = ph > pw

    if ow > 0 and oh > 0:
        dw, dh = exif_display_dimensions(ow, oh, o)
        exp_portrait = dh > dw
        if exp_portrait == act_portrait:
            return 0
        if not exp_portrait and act_portrait:
            # Thumbnail already portrait; metadata still landscape — trust pixels.
            return 0
        if o == 3:
            return 180
        if o == 6:
            return 90
        if o == 8:
            return 270
        if o <= 1 and oh > ow:
            return 90
        return 0

    if act_portrait and o in (6, 8):
        return 0
    if o == 3:
        return 180
    if o == 6:
        return 90
    if o == 8:
        return 270
    return 0


def get_image_aspect_ratio(file_path: str) -> float:
    """獲取圖像寬高比（不載入完整圖像）"""
    cache = get_image_cache()
    
    # 嘗試從 EXIF 快取獲取尺寸
    exif_data = cache.get_exif(file_path)
    if exif_data:
        original_width = exif_data.get('original_width')
        original_height = exif_data.get('original_height')
        orientation = exif_data.get('orientation', 1)
        
        if original_width and original_height and original_height > 0:
            return exif_display_aspect_ratio(
                int(original_width), int(original_height), int(orientation or 1)
            )
    
    # 對於 RAW 文件，嘗試從快取獲取尺寸，否則使用 rawpy 快速獲取
    if is_raw_file(file_path):
        # 優先嘗試 EXIF 快取（最快且包含方向）
        if exif_data:
            orig_w = exif_data.get('original_width')
            orig_h = exif_data.get('original_height')
            orient = exif_data.get('orientation', 1)
            if orig_w and orig_h and orig_h > 0:
                return exif_display_aspect_ratio(
                    int(orig_w), int(orig_h), int(orient or 1)
                )
                
        try:
            import rawpy
            # 只讀取 Metadata，不處理圖像
            with rawpy.imread(file_path) as raw:
                sizes = raw.sizes
                # Use iwidth and iheight as they already account for orientation (rotation)
                # sizes.iwidth and iheight are internal dimensions after rotation
                # HOWEVER, for absolute safety with all rawpy versions, 
                # let's check flip and swap sensor dimensions manually if needed.
                w = sizes.width
                h = sizes.height
                flip = sizes.flip
                
                if flip in (5, 6, 7, 8):
                    w, h = h, w
                
                if h > 0:
                    return w / h
                return 1.333
        except:
             pass

    # 對於 TIFF 文件，使用 PIL
    if is_tiff_file(file_path):
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                width, height = img.size
                if height > 0:
                    return width / height
        except:
            pass
    
    # 對於其他格式，嘗試 QImageReader（但不包括 RAW 和 TIFF）
    if not is_raw_file(file_path):
        try:
            from PyQt6.QtGui import QImageReader
            reader = QImageReader(file_path)
            # Enable automatic EXIF orientation transformation
            reader.setAutoTransform(True)
            size = reader.size()
            
            if size.isValid() and size.height() > 0:
                width = size.width()
                height = size.height()
                
                # Check transformation to see if dimensions need swapping
                # QImageReader.size() returns the size *before* transformation in some versions,
                # so we need to manually swap if the transformation involves 90 degree rotation.
                transformation = reader.transformation()
                from PyQt6.QtGui import QImageIOHandler
                
                # Check for 90 or 270 degree rotation
                # QImageIOHandler.Transformation.TransformationRotate90 = 2
                # QImageIOHandler.Transformation.TransformationRotate270 = 4
                # We also need to consider mirrored versions if they involve 90 deg rotation
                # TransformationMirrorAndRotate90 = 6, TransformationMirrorAndRotate270 = 8 is not a standard enum value in older Qt?
                # Let's stick to the basic check:
                
                # Simple check: if orientation is 5, 6, 7, 8 (which correspond to rotations), we swap.
                # However, QImageReader.transformation() returns a Transformation enum.
                # TransformationRotate90 (2), TransformationRotate270 (4), 
                # TransformationMirrorAndRotate90 (6), TransformationFlipAndRotate90 (6?)
                
                t_val = transformation.value
                
                # Qt::ImageTransformation:
                # 0: None
                # 1: Mirror
                # 2: Flip
                # 3: Rotate180
                # 4: Rotate90
                # 5: MirrorAndRotate90
                # 6: FlipAndRotate90
                # 7: Rotate270
                
                # So 4, 5, 6, 7 imply 90 degree component (swapped dimensions)
                if t_val >= 4:
                    width, height = height, width
                    
                return width / height
        except Exception as e:
            # import logging
            # logging.getLogger(__name__).debug(f"Aspect ratio check failed: {e}")
            pass
    
    return 1.333  # 默認寬高比 (4:3)


def use_progressive_raw_loading() -> bool:
    """
    When True, show embedded preview/thumbnail first and upgrade to LibRaw half-res in
    the background (may cause a visible color shift). Default: off; use
    RAWVIEWER_LIBRAW_CONSISTENT_PREVIEW=1 for a single LibRaw pipeline instead.
    """
    v = os.environ.get("RAWVIEWER_PROGRESSIVE_RAW_LOAD") or os.environ.get("SkySpotter_PROGRESSIVE_RAW_LOAD", "")
    v = v.strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return False


def pil_downscale_resample():
    """CPU downscale filter for thumbnails — HAMMING is faster than LANCZOS for shrink."""
    from PIL import Image

    return Image.Resampling.HAMMING


def metadata_index_idle_delay_ms() -> int:
    """Idle delay after first single-view paint before background metadata indexing starts."""
    raw = os.environ.get("RAWVIEWER_AUTO_METADATA_INDEX_IDLE_MS", "5000").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 5000


def use_raw_process_pool() -> bool:
    """
    Offload LibRaw postprocess to a process pool (multi-core). Opt-out with
    SkySpotter_USE_PROCESS_POOL=0. Default: on when CPU count >= 4 (Windows/Linux).
    macOS default off: spawn re-executes the .app (extra PIDs / splash). Opt in only via
    RAWVIEWER_USE_PROCESS_POOL=1 (SkySpotter_* is ignored on macOS).
    """
    import os as _os
    import sys as _sys

    if _sys.platform == "darwin":
        raw = os.environ.get("RAWVIEWER_USE_PROCESS_POOL", "").strip().lower()
        if raw in ("0", "false", "no", "off"):
            return False
        if raw in ("1", "true", "yes", "on"):
            return True
        return False

    raw = os.environ.get("RAWVIEWER_USE_PROCESS_POOL") or os.environ.get(
        "SkySpotter_USE_PROCESS_POOL", ""
    )
    raw = raw.strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return (_os.cpu_count() or 0) >= 4


def _env_int_bounded(name: str, default: int, *, minimum: int = 1, maximum: int = 64) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, min(maximum, int(raw)))
    except ValueError:
        return default


def is_slow_storage_path(path: str) -> bool:
    """
    Heuristic for paths where aggressive parallel reads hurt more than help
    (UNC shares, user-configured prefixes). Safe on Windows and macOS.
    """
    if not path:
        return False
    norm = os.path.normpath(path)
    if norm.startswith("\\\\"):
        return True
    prefixes = os.environ.get("RAWVIEWER_SLOW_STORAGE_PREFIXES", "").strip()
    if prefixes:
        for prefix in prefixes.split(","):
            p = prefix.strip()
            if p and norm.lower().startswith(os.path.normpath(p).lower()):
                return True
    if os.name == "nt" and len(norm) >= 2 and norm[1] == ":":
        try:
            import ctypes

            root = norm[:2] + "\\"
            # DRIVE_REMOTE=4, DRIVE_CDROM=5 — prefer scan-first embedded JPEG extraction
            if ctypes.windll.kernel32.GetDriveTypeW(root) in (4, 5):
                return True
        except Exception:
            pass
    return False


def sort_probe_worker_count(
    sample_path: Optional[str] = None,
    *,
    conservative: bool = False,
) -> int:
    """
    Thread pool size for cold EXIF header probes during folder sort.

    Override: RAWVIEWER_SORT_PROBE_WORKERS (1–32).
    Conservative mode (fast-open window): RAWVIEWER_SORT_PROBE_WORKERS_CONSERVATIVE or min(3, default).
    Slow storage (UNC / RAWVIEWER_SLOW_STORAGE_PREFIXES): capped at 3.
  """
    override = os.environ.get("RAWVIEWER_SORT_PROBE_WORKERS", "").strip()
    if override:
        return _env_int_bounded("RAWVIEWER_SORT_PROBE_WORKERS", 4, minimum=1, maximum=32)

    if io_pressure_active():
        return 2

    cpu = os.cpu_count() or 4
    if conservative:
        cons = os.environ.get("RAWVIEWER_SORT_PROBE_WORKERS_CONSERVATIVE", "").strip()
        if cons:
            return _env_int_bounded(
                "RAWVIEWER_SORT_PROBE_WORKERS_CONSERVATIVE", 3, minimum=1, maximum=8
            )
        return min(3, max(2, cpu))

    if sample_path and is_slow_storage_path(sample_path):
        # Network/UNC paths have high latency. We need *more* workers to hide the I/O wait.
        return max(8, cpu * 2)

    # Local SSD/NVMe: I/O-bound header reads; scale past 3 workers (old hard cap under-used CPU).
    return min(12, max(4, cpu - 1))


def index_metadata_worker_count(total_files: int) -> int:
    """
    Parallel metadata extraction during semantic index build.

    Override: RAWVIEWER_INDEX_METADATA_WORKERS.
    Large folders (>2000) use a lower default to reduce SQLite EXIF cache lock contention.
    """
    override = os.environ.get("RAWVIEWER_INDEX_METADATA_WORKERS", "").strip()
    if override:
        return _env_int_bounded("RAWVIEWER_INDEX_METADATA_WORKERS", 2, minimum=1, maximum=16)

    cpu = os.cpu_count() or 4
    if total_files > 2000:
        return min(3, max(2, cpu // 2))
    return min(6, max(2, cpu - 1))


def raw_concurrent_load_limit() -> int:
    """Max concurrent LibRaw full/preview decodes in ImageLoadManager."""
    cpu = os.cpu_count() or 4
    default = max(4, cpu)
    return _env_int_bounded("RAWVIEWER_RAW_LOAD_LIMIT", default, minimum=1, maximum=32)


def process_pool_worker_count() -> int:
    """LibRaw postprocess process-pool size when RAWVIEWER_USE_PROCESS_POOL is on."""
    cpu = os.cpu_count() or 4
    default = max(2, cpu - 1)
    return _env_int_bounded("RAWVIEWER_PROCESS_POOL_WORKERS", default, minimum=1, maximum=32)


