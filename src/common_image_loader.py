"""
共用圖像載入函數 - 統一處理所有圖像類型的載入邏輯

這個模組提供統一的圖像載入函數，避免重複代碼。
"""

import io
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

_io_pressure_until = 0.0

# One-shot per-volume read-speed probe cache: {volume_root: mbps_or_-1.0}.
# Populated lazily by probe_volume_speed() so we only measure each mount once.
_volume_speed_cache: Dict[str, float] = {}
_volume_speed_lock = threading.Lock()

# Readings above this are the OS page cache (RAM), not a real disk read, e.g. when
# probing a file that was just opened/decoded. They must not classify a slow drive
# as "fast". Real external interfaces (USB3/SATA/USB4/TB NVMe) stay well below this.
_VOLUME_SPEED_CACHE_CEILING_MBPS = 3000.0


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
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


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
        # Fast path: JPEG DCT-domain scaled decode (1/2, 1/4, 1/8). Decoding a
        # 30MP+ embedded preview at reduced scale instead of full size then
        # downsizing dominates both gallery (~512) and single-view display-tier
        # (~1920–2304) cost. draft() never upsizes and keeps pixel orientation
        # identical to a full decode. Kill-switch: RAWVIEWER_JPEG_DRAFT_DECODE=0.
        draft_cap = 1024
        try:
            from image_cache import memory_preview_max_edge

            draft_cap = max(1024, int(memory_preview_max_edge()))
        except Exception:
            draft_cap = 2304
        if (
            0 < max_size <= draft_cap
            and im.format == "JPEG"
            and max(im.size) > max_size * 2
            and os.environ.get("RAWVIEWER_JPEG_DRAFT_DECODE", "1").strip().lower()
            not in {"0", "false", "no", "off"}
        ):
            try:
                im.draft("RGB", (max_size, max_size))
            except Exception:
                pass
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


# Memoized QSettings read for the embedded-JPEG workflow toggle. Invalidated
# via invalidate_libraw_consistent_preview_settings() when the user flips the
# workflow control so load paths don't construct QSettings on every call.
# Sentinel distinguishes "key missing" (None) from "not yet read".
_EMBEDDED_JPEG_WORKFLOW_UNSET = object()
_EMBEDDED_JPEG_WORKFLOW_MEMO: Any = _EMBEDDED_JPEG_WORKFLOW_UNSET


def invalidate_libraw_consistent_preview_settings() -> None:
    """Clear the memoized ``use_embedded_jpeg_workflow`` QSettings value."""
    global _EMBEDDED_JPEG_WORKFLOW_MEMO
    _EMBEDDED_JPEG_WORKFLOW_MEMO = _EMBEDDED_JPEG_WORKFLOW_UNSET


def _embedded_jpeg_workflow_setting() -> Optional[bool]:
    """Cached QSettings read: True/False when set, None when key absent."""
    global _EMBEDDED_JPEG_WORKFLOW_MEMO
    if _EMBEDDED_JPEG_WORKFLOW_MEMO is not _EMBEDDED_JPEG_WORKFLOW_UNSET:
        return _EMBEDDED_JPEG_WORKFLOW_MEMO  # type: ignore[return-value]
    try:
        from PyQt6.QtCore import QSettings

        settings = QSettings("RAWviewer", "RAWviewer")
        if not settings.contains("use_embedded_jpeg_workflow"):
            _EMBEDDED_JPEG_WORKFLOW_MEMO = None
            return None
        use_embedded = bool(
            settings.value("use_embedded_jpeg_workflow", True, type=bool)
        )
        _EMBEDDED_JPEG_WORKFLOW_MEMO = use_embedded
        return use_embedded
    except Exception:
        _EMBEDDED_JPEG_WORKFLOW_MEMO = None
        return None


def use_libraw_consistent_preview_first(file_path: Optional[str] = None) -> bool:
    """
    When True (default), single-image RAW avoids embedded-JPEG preview paths so fit and zoom
    share the same LibRaw postprocess look. Set RAWVIEWER_LIBRAW_CONSISTENT_PREVIEW=0 to restore
    the faster embedded-preview first paint.

    Full-resolution embedded JPEGs (see use_full_embedded_raw_preview) bypass LibRaw even when
    this flag is on.
    """
    if file_path:
        try:
            # This escalation exists so DISPLAYED edits look the same at fit
            # and zoom (LibRaw both ways). With sidecar display off (the
            # default: edits render only in the Adjust panel) it must not
            # fire at all — a mere xmp:Rating sidecar was flipping rated
            # files into the RAW workflow, showing LibRaw renders in non-RAW
            # view and blanking gallery→single landings while the interim
            # JPEG was refused. Existence probe only (never
            # load_adjustments_for_file: its as-shot EXIF parse cost ~0.5s
            # per ARW on the UI thread via _check_cache).
            from raw_adjustments import resolve_xmp_path, sidecar_adjustments_enabled

            if sidecar_adjustments_enabled():
                xmp = resolve_xmp_path(file_path)
                if xmp and os.path.isfile(xmp):
                    return True
        except Exception:
            pass

    use_embedded = _embedded_jpeg_workflow_setting()
    if use_embedded is False:
        return True

    v = os.environ.get("RAWVIEWER_LIBRAW_CONSISTENT_PREVIEW", "0").strip().lower()
    return v not in ("0", "false", "no", "off")


# Embedded JPEG long edge must reach this fraction of sensor long edge to count as "full size".
FULL_EMBEDDED_SENSOR_COVERAGE = 0.92

# RAF (Fujifilm) and 3FR (Hasselblad) camera firmware embeds a fixed-size JPEG
# preview capped well below sensor resolution regardless of RAW megapixels --
# a manufacturer convention, not a coverage edge case, and NOT tied to X-Trans
# vs Bayer CFA (GFX medium-format RAF and 3FR are both Bayer and still miss).
# Verified empirically: 36/36 sample RAF+3FR files -- X-Trans APS-C through
# 100MP Bayer medium format -- all failed FULL_EMBEDDED_SENSOR_COVERAGE (best
# case ratio 0.71, most 0.33-0.57; direct byte-scan of one file confirmed
# LibRaw's extract_thumb() already returns the largest embedded JPEG present,
# nothing bigger is hiding in the container). Checking real coverage requires
# opening the file with rawpy -- the exact cost callers use this predicate to
# avoid -- so treat these two formats as a known miss for prefetch decisions
# rather than eagerly requesting a "full" stage that's actually a ~20-35s
# LibRaw demosaic in disguise.
_EAGER_FULL_UNRELIABLE_EXTS = frozenset({".raf", ".3fr"})


def embedded_preview_rarely_covers_sensor(file_path: str) -> bool:
    """Cheap (extension-only) predicate for formats whose embedded preview is
    known to almost never satisfy FULL_EMBEDDED_SENSOR_COVERAGE. See
    _EAGER_FULL_UNRELIABLE_EXTS for the rationale and verification.
    """
    ext = os.path.splitext(file_path)[1].lower()
    return ext in _EAGER_FULL_UNRELIABLE_EXTS


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


# RAW sensor dims below this are almost always a preview/thumbnail IFD
# (e.g. 160x120) wrongly cached as original_width/height — treating them as
# sensor size makes a 512px gallery buffer look "full-res", which then races
# RAW-mode JPEG-interim refusal into a permanent stuck-on-previous-file view.
_MIN_TRUSTED_SENSOR_LONG_EDGE = 1920


def sensor_pixel_dimensions(exif_data: Optional[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
    """Sensor / full-frame pixel size from EXIF cache, if known and trustworthy."""
    if not exif_data:
        return None
    if exif_data.get("minimal_preview_exif"):
        return None
    ow = exif_data.get("original_width")
    oh = exif_data.get("original_height")
    try:
        w, h = int(ow), int(oh)
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    if max(w, h) < _MIN_TRUSTED_SENSOR_LONG_EDGE:
        return None
    return w, h


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
                if not use_libraw_consistent_preview_first(file_path):
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
        except Exception:
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
            import numpy as np
            arr = np.asarray(pil_image)
            qimage = QImage(
                arr.data,
                width,
                height,
                width * 3,
                QImage.Format.Format_RGB888,
            )
            qimage.ndarr = arr  # Keep array alive
            if qimage.isNull():
                return QPixmap()
            return QPixmap.fromImage(qimage)
    except Exception as e:
        logger.debug(
            "PIL load failed for %s: %s", os.path.basename(file_path), e
        )
        return QPixmap()


_last_load_used_edr = False


def take_last_load_edr_flag() -> bool:
    """True when the most recent load_pixmap_safe() used macOS HDR/EDR output."""
    global _last_load_used_edr
    used = bool(_last_load_used_edr)
    _last_load_used_edr = False
    return used


def pixmap_covers_requested_edge(file_path: str, pixmap, max_edge: int) -> bool:
    """True when a cached ``pixmap`` satisfies a load request of ``max_edge``.

    ``max_edge <= 0`` means "native resolution": the pixmap must reach the
    file's header-probed dimensions (compared sorted, so an EXIF-rotated
    pixmap still matches). A pixmap at/above the app's own safety cap
    (_regular_image_max_edge) is always accepted -- it is the largest this
    app will ever produce, so rejecting it would re-run an expensive decode
    into the same result. Unknown native size accepts the cache.

    Without this check, the fit view's preview-capped pixmap (same cache key)
    satisfied every later full-resolution request: a 32888x8470 panorama
    cached at 2304px was re-delivered forever, so zooming to 100% showed a
    blurry upscale and the display never upgraded.
    """
    try:
        have = sorted((int(pixmap.width()), int(pixmap.height())))
        if have[1] >= _regular_image_max_edge():
            return True
        if max_edge > 0 and have[1] >= max_edge - 2:
            # Fast accept without a header probe -- the common (gallery/fit)
            # case stays as cheap as the unconditional hit it replaces.
            return True
        # Cached pixmap is smaller than the request: it still satisfies it if
        # the FILE is just natively small (never force a redecode that cannot
        # produce more pixels). One header-only probe, no pixel decode.
        from PyQt6.QtGui import QImageReader

        sz = QImageReader(file_path).size()
        if not sz.isValid() or sz.width() <= 0 or sz.height() <= 0:
            return True
        need = sorted((int(sz.width()), int(sz.height())))
        if max_edge > 0:
            scale = min(1.0, max_edge / max(need[1], 1))
            need = [int(need[0] * scale), int(need[1] * scale)]
        return have[0] >= need[0] - 2 and have[1] >= need[1] - 2
    except Exception:
        return True


def load_pixmap_safe(file_path: str, max_edge: int = 0) -> QPixmap:
    """安全載入 QPixmap，對 TIFF 文件使用 PIL 以避免 Qt 警告"""
    global _last_load_used_edr
    _last_load_used_edr = False
    cache = get_image_cache()

    # 檢查快取 (size-aware: a preview-capped entry must not satisfy a
    # larger/native-resolution request -- see pixmap_covers_requested_edge)
    cached_pixmap = cache.get_pixmap(file_path)
    if cached_pixmap is not None and not cached_pixmap.isNull():
        if pixmap_covers_requested_edge(file_path, cached_pixmap, max_edge):
            return cached_pixmap

    if is_raw_file(file_path):
        preview_max = max_edge if max_edge > 0 else 2048
        return load_raw_preview_pixmap(file_path, max_size=preview_max)
    
    # Check if this is an HDR HEIF/AVIF or TIFF
    hdr_pixmap = try_load_hdr_image_pixmap(file_path, max_edge=max_edge)
    if hdr_pixmap is not None:
        _last_load_used_edr = is_macos_edr_enabled()
        cache.put_pixmap(file_path, hdr_pixmap)
        return hdr_pixmap
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
                import numpy as np
                arr = np.asarray(pil_image)
                qimage = QImage(arr.data,
                               width, height, width * 3,
                               QImage.Format.Format_RGB888)
                qimage.ndarr = arr  # Keep array alive

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
            try:
                cached_exif = cache.get_exif(file_path)
                if cached_exif and int(cached_exif.get("orientation", 1) or 1) != 1:
                    cache.put_exif(file_path, mark_exif_pixels_display_oriented(cached_exif))
            except Exception:
                pass
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


def exif_pixels_display_oriented(exif_data: Optional[Dict[str, Any]]) -> bool:
    """True when decode already baked EXIF orientation into pixel data."""
    return bool((exif_data or {}).get("pixels_display_oriented"))


def mark_exif_pixels_display_oriented(exif_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Record that container orientation is already applied to cached pixels."""
    data = dict(exif_data or {})
    data["orientation"] = 1
    data["pixels_display_oriented"] = True
    return data


def pixmap_matches_exif_display(
    pixmap_width: int,
    pixmap_height: int,
    original_width: int,
    original_height: int,
    orientation: int,
    *,
    tolerance: float = 0.14,
    pixels_display_oriented: bool = False,
) -> bool:
    """True when pixmap pixels already match EXIF display orientation (no extra rotation)."""
    if pixels_display_oriented:
        return True
    pw, ph = int(pixmap_width or 0), int(pixmap_height or 0)
    if pw <= 0 or ph <= 0:
        return False
    o = int(orientation or 1)
    # 180° / mirror tags do not change aspect ratio — never infer correctness from AR alone.
    if o in (2, 3, 4):
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
    if exif_pixels_display_oriented(exif_data):
        return True
    o = int(exif_data.get("orientation", 1) or 1)
    if o in (2, 3, 4):
        return False
    return pixmap_matches_exif_display(
        pixel_width,
        pixel_height,
        ow,
        oh,
        o,
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

    # Make-independent double-rotation guard (single source of truth for orientation):
    # MEASURE the pixels' actual portrait/landscape against the EXIF display orientation
    # and rotate only if they disagree. This is idempotent — calling it twice, or on an
    # embedded preview that is already display-oriented (Sony/Leica pre-rotate; many other
    # cameras' JPEG previews are display-oriented too), never double-rotates. We deliberately
    # do NOT branch on camera make: make allowlists are wrong for the many cameras that
    # pre-rotate their preview, which is what produced the "portrait shown sideways" bug.
    # exif_display_dimensions only ever swaps landscape->portrait, so this stays correct
    # whether original_width/height were stored as sensor dims or already-display dims.
    ow = int((exif_data or {}).get("original_width") or 0)
    oh = int((exif_data or {}).get("original_height") or 0)
    if ow <= 0 or oh <= 0:
        ow, oh = 3, 2  # Default to landscape sensor aspect ratio (matches most cameras)

    dw, dh = exif_display_dimensions(ow, oh, o)
    if (dh > dw) == (h > w):
        if o not in (2, 3, 4) or exif_pixels_display_oriented(exif_data):
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
    """Convert oriented uint8 RGB/grayscale ndarray to QPixmap.

    Builds the QImage from the array's own buffer (no .tobytes() copy):
    QPixmap.fromImage() below deep-copies pixel data into the QPixmap's own
    native storage before this function returns, so the QImage never
    outlives rgb_image's scope -- safe to reference directly rather than
    paying for an extra full-frame copy on every conversion (this path runs
    on every gallery/thumbnail/display repaint).
    """
    if rgb_image is None or not hasattr(rgb_image, "shape"):
        return QPixmap()
    if rgb_image.dtype != np.uint8:
        rgb_image = rgb_image.astype(np.uint8, copy=False)
    if not rgb_image.flags["C_CONTIGUOUS"]:
        rgb_image = np.ascontiguousarray(rgb_image)
    h, w = rgb_image.shape[:2]
    if len(rgb_image.shape) == 2:
        qimg = QImage(
            rgb_image.data, w, h, w, QImage.Format.Format_Grayscale8
        )
    elif rgb_image.shape[2] >= 3:
        channels = int(rgb_image.shape[2])
        q_format = (
            QImage.Format.Format_RGB888
            if channels == 3
            else QImage.Format.Format_RGBA8888
        )
        qimg = QImage(
            rgb_image.data, w, h, channels * w, q_format
        )
    else:
        return QPixmap()

    qimg.ndarr = rgb_image  # Keep array alive explicitly
    return QPixmap.fromImage(qimg) if not qimg.isNull() else QPixmap()


def apply_container_orientation_to_pixmap(
    pixmap: QPixmap,
    exif_data: Optional[Dict[str, Any]],
) -> QPixmap:
    """Apply container EXIF orientation to a QPixmap (idempotent with display-oriented pixels)."""
    from PyQt6.QtGui import QTransform

    if pixmap is None or pixmap.isNull() or not exif_data:
        return pixmap
    if exif_pixels_display_oriented(exif_data):
        return pixmap
    orientation = int(exif_data.get("orientation", 1) or 1)
    if orientation == 1:
        return pixmap

    ow = int(exif_data.get("original_width") or 0)
    oh = int(exif_data.get("original_height") or 0)
    if ow > 0 and oh > 0 and pixmap_matches_exif_display(
        pixmap.width(),
        pixmap.height(),
        ow,
        oh,
        orientation,
        pixels_display_oriented=exif_pixels_display_oriented(exif_data),
    ):
        return pixmap

    transform = QTransform()
    if orientation == 2:
        transform.scale(-1, 1)
    elif orientation == 3:
        transform.rotate(180)
    elif orientation == 4:
        transform.scale(1, -1)
    elif orientation == 5:
        transform.scale(-1, 1)
        transform.rotate(-90)
    elif orientation == 6:
        transform.rotate(90)
    elif orientation == 7:
        transform.scale(-1, 1)
        transform.rotate(90)
    elif orientation == 8:
        transform.rotate(-90)
    else:
        return pixmap
    return pixmap.transformed(transform)


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
            arr = np.asarray(cached, dtype=np.uint8)
            if is_raw_file(file_path):
                return finalize_index_thumbnail_array(file_path, arr, cache=cache)
            return arr

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

    return finalize_index_thumbnail_array(file_path, thumb, cache=cache)


def resolve_container_exif_for_file(
    file_path: str,
    *,
    cache=None,
) -> tuple[Optional[Dict[str, Any]], int]:
    """Return (exif_data, orientation) from ImageCache or container EXIF on disk.

    verify=True: this is the orientation authority for pixel rotation
    (_heal_raw_orientation / finalize_index_thumbnail_array) -- an
    untrustworthy cached orientation here rotates pixels wrongly, so the
    synchronous probe stays. get_exif memoizes the successful probe to disk,
    so the cost is once per file ever, not per call.
    """
    cache = cache or get_image_cache()
    exif_data = cache.get_exif(file_path, verify=True)
    if not exif_data:
        try:
            from enhanced_raw_processor import EXIFExtractor

            exif_data = EXIFExtractor().extract_exif_data(file_path)
        except Exception:
            exif_data = None
    orientation = int((exif_data or {}).get("orientation", 1) or 1)
    return exif_data, orientation


def index_thumbnail_needs_orient_fix(
    file_path: str,
    arr: np.ndarray,
    *,
    cache=None,
) -> bool:
    """True when cached index pixels don't match container EXIF display orientation."""
    if arr is None:
        return True
    exif_data, _ = resolve_container_exif_for_file(file_path, cache=cache)
    if not exif_data:
        return False
    h, w = arr.shape[:2]
    return not array_matches_exif_display(w, h, exif_data)


def finalize_index_thumbnail_array(
    file_path: str,
    arr: np.ndarray,
    *,
    cache=None,
) -> np.ndarray:
    """Apply container EXIF orientation so semantic warm-up matches gallery thumbnails."""
    if arr is None:
        return arr
    exif_data, orientation = resolve_container_exif_for_file(file_path, cache=cache)
    if os.environ.get("RAWVIEWER_ORIENT_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            import logging as _lg
            _h, _w = arr.shape[:2]
            _ow = int((exif_data or {}).get("original_width") or 0)
            _oh = int((exif_data or {}).get("original_height") or 0)
            _src = "cache" if (cache is not None and cache.get_exif(file_path)) else "file"
            _lg.getLogger(__name__).info(
                "[ORIENT] finalize file=%s in=%dx%d(%s) resolved(o=%s ow=%d oh=%d src=%s)",
                os.path.basename(file_path), _w, _h,
                "P" if _h > _w else "L", orientation, _ow, _oh, _src,
            )
        except Exception:
            pass
    if orientation != 1:
        oriented = apply_container_orientation_to_array(arr, orientation, exif_data)
        if oriented is not None:
            return oriented
    return arr


def _mipmap_publish_debug_enabled() -> bool:
    return os.environ.get("RAWVIEWER_MIPMAP_PUBLISH_DEBUG", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


_TIER_CACHE_ATTRS = {
    "preview": ("preview_cache", "disk_preview_cache"),
    "grid": ("grid_cache", "disk_grid_cache"),
    "thumbnail": ("thumbnail_cache", "disk_thumbnail_cache"),
}


def _tile_cache_encode_format() -> str:
    """Output format for grid/thumbnail/preview disk-cache tiles.

    JPEG by default. WebP is ~25-35% smaller at equivalent quality with the
    same encode cost, but PIL's WebP *decode* is ~5-6x slower than JPEG at
    these tile sizes (measured directly: ~3ms vs ~0.6ms for a 512x384 tile,
    both synthetic-gradient and photo-like content) -- and this cache tier is
    read synchronously on the gallery's main-thread scroll path
    (get_grid() -> disk_grid_cache), every cache-miss-in-memory tile decode
    happening inline during a scroll frame. That decode cost, not the disk
    footprint, is what mattered here, so WebP was a net loss for this
    specific hot path. RAWVIEWER_TILE_CACHE_FORMAT=webp opts back in.
    Deliberately NOT used for PR-7's raw embedded-JPEG-bytes cache (a
    verbatim byte-slice from the RAW file, not a re-encode -- converting that
    would force a decode+encode pass the whole point of that cache is to
    avoid). Read side is format-agnostic: PIL.Image.open() sniffs format from
    content, so existing WebP entries keep working and this can be toggled
    freely without a migration.
    """
    raw = os.environ.get("RAWVIEWER_TILE_CACHE_FORMAT", "jpeg").strip().lower()
    return "WEBP" if raw == "webp" else "JPEG"


def encode_tile_bytes(pil_image, *, quality: int = 85) -> bytes:
    """Encode a PIL image for a grid/thumbnail/preview disk-cache tile."""
    import io

    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    fmt = _tile_cache_encode_format()
    buf = io.BytesIO()
    try:
        pil_image.save(buf, format=fmt, quality=quality)
    except Exception:
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _bump_mipmap_stat(cache, key: str) -> None:
    """Debug-only counters surfaced via ImageCache.get_cache_stats()['request_stats']."""
    try:
        cache.stats[key] = int(cache.stats.get(key, 0)) + 1
    except Exception:
        pass


def _tier_really_present(cache, file_path: str, tier: str) -> bool:
    """Direct presence check for ONE tier, bypassing get_preview()/get_grid()'s
    cross-tier "dynamic mipmap fallback" (which synthesizes a tier by up/down-sampling
    a different tier and PERSISTS it as a side effect of merely reading). Calling the
    public get_X() here would let a presence check silently manufacture and cache a
    blurry upsampled tier, defeating the "never upsample" guarantee below -- so tier
    existence must be checked against the memory/disk stores directly.
    """
    mem_attr, disk_attr = _TIER_CACHE_ATTRS[tier]
    try:
        mem_cache = getattr(cache, mem_attr)
        if mem_cache.get(cache._path_key(file_path)) is not None:
            return True
    except Exception:
        pass
    try:
        disk_cache = getattr(cache, disk_attr)
        if hasattr(disk_cache, "has_valid") and disk_cache.has_valid(file_path):
            return True
    except Exception:
        pass
    return False


def publish_mipmap_tiers(
    file_path: str,
    native_array,
    *,
    exif_data: Optional[Dict[str, Any]] = None,
    cache=None,
    source: str = "decode",
) -> str:
    """Backfill missing preview/grid/thumbnail ImageCache tiers from one decoded array.

    Single write point for "I just decoded this file, publish it so every other
    consumer (gallery, film strip, semantic/face indexing) can reuse the pixels
    instead of re-extracting/re-decoding the same RAW." Each tier is filled
    independently and NEVER upsampled: a small decode (e.g. a 256px face-scan
    thumbnail) only satisfies the tiers it's large enough for without fabricating
    a fake higher-resolution tier. Tiers that already exist are left untouched.

    Returns the highest tier now present for this path: "preview" > "grid" >
    "thumbnail" > "" (nothing present/written).
    """
    if native_array is None:
        return ""

    if isinstance(native_array, QImage):
        from enhanced_raw_processor import _qimage_to_rgb_array

        native_array = _qimage_to_rgb_array(native_array)
    if native_array is None or not isinstance(native_array, np.ndarray):
        return ""

    try:
        h, w = native_array.shape[:2]
    except Exception:
        return ""
    if h <= 0 or w <= 0:
        return ""
    long_edge = max(h, w)

    cache = cache or get_image_cache()
    _bump_mipmap_stat(cache, "mipmap_publish_count")

    have_preview = _tier_really_present(cache, file_path, "preview")
    have_grid = _tier_really_present(cache, file_path, "grid")
    have_thumb = _tier_really_present(cache, file_path, "thumbnail")

    if have_preview:
        highest = "preview"
    elif have_grid:
        highest = "grid"
    elif have_thumb:
        highest = "thumbnail"
    else:
        highest = ""

    from PIL import Image
    import io as _io

    arr = native_array.astype(np.uint8) if native_array.dtype != np.uint8 else native_array
    _pil_cache = []

    def _as_pil():
        if not _pil_cache:
            if len(arr.shape) == 2:
                _pil_cache.append(Image.fromarray(arr, "L").convert("RGB"))
            elif len(arr.shape) == 3:
                _pil_cache.append(Image.fromarray(arr, "RGB"))
            else:
                _pil_cache.append(None)
        return _pil_cache[0]

    def _resized(img, target: int):
        if w <= target and h <= target:
            return img
        scale = min(target / w, target / h)
        return img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.HAMMING
        )

    # A tier is only written when the source is at least ~85% of its target long
    # edge -- the same tolerance used elsewhere (pixmap_matches_exif_display) for
    # "close enough, don't bother re-deriving." Anything smaller is left for a real
    # decode later rather than silently upsampled (upsampled tiers look soft and
    # would otherwise get cached as if they were native-resolution).
    from image_cache import disk_preview_max_edge

    pv_max = disk_preview_max_edge()
    if not have_preview and long_edge >= int(pv_max * 0.85):
        try:
            img = _as_pil()
            if img is not None:
                cache.put_preview(file_path, np.array(_resized(img, pv_max)))
                _bump_mipmap_stat(cache, "tier_backfill_count")
                if highest in ("", "grid", "thumbnail"):
                    highest = "preview"
        except Exception:
            pass

    grid_max = disk_preview_max_edge()
    if not have_grid and long_edge >= int(grid_max * 0.85):
        try:
            img = _as_pil()
            if img is not None:
                grid_pil = _resized(img, grid_max)
                cache.put_grid(file_path, np.array(grid_pil), encode_tile_bytes(grid_pil))
                _bump_mipmap_stat(cache, "tier_backfill_count")
                if highest == "":
                    highest = "grid"
        except Exception:
            pass

    if not have_thumb and long_edge >= int(256 * 0.85):
        try:
            img = _as_pil()
            if img is not None:
                thumb_pil = _resized(img, 256)
                cache.put_thumbnail(file_path, np.array(thumb_pil), encode_tile_bytes(thumb_pil))
                _bump_mipmap_stat(cache, "tier_backfill_count")
                if highest == "":
                    highest = "thumbnail"
        except Exception:
            pass

    if exif_data:
        try:
            cur = cache.get_exif(file_path)
            if cur is None or cur.get("minimal_preview_exif"):
                cache.put_exif(
                    file_path,
                    exif_data,
                    persist_disk=not bool(exif_data.get("minimal_preview_exif")),
                )
        except Exception:
            pass

    if _mipmap_publish_debug_enabled():
        try:
            import logging

            logging.getLogger(__name__).info(
                "[MIPMAP] publish file=%s source=%s in=%dx%d "
                "had(preview=%s,grid=%s,thumb=%s) -> highest=%s",
                os.path.basename(file_path),
                source,
                w,
                h,
                have_preview,
                have_grid,
                have_thumb,
                highest,
            )
        except Exception:
            pass

    return highest


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
            if o in (2, 3, 4):
                if o == 3:
                    return 180
                return 0
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
        if o <= 1:
            if dh > dw and not act_portrait:
                return 90
            if dw > dh and act_portrait:
                return 270
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
        except Exception:
             pass

    # 對於 TIFF 文件，使用 PIL
    if is_tiff_file(file_path):
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                width, height = img.size
                if height > 0:
                    return width / height
        except Exception:
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
    macOS default off: spawn re-executes the .app (extra PIDs / splash). Opt in via
    RAWVIEWER_USE_PROCESS_POOL=1. For frozen macOS builds, also set
    RAWVIEWER_PROCESS_POOL_PYTHON to a plain interpreter (see libraw_pool_worker.py)
    so workers do not re-launch the GUI .app.

    When an in-process GPU demosaic backend is already loaded (cuda/mps/cupy),
    default off so process-pool LibRaw does not compete with GPU VRAM/host slots.
    Explicit RAWVIEWER_USE_PROCESS_POOL=1 still forces the pool on.
    """
    import os as _os
    import sys as _sys

    if _sys.platform == "darwin":
        raw = os.environ.get("RAWVIEWER_USE_PROCESS_POOL", "").strip().lower()
        if raw in ("0", "false", "no", "off"):
            return False
        if raw in ("1", "true", "yes", "on"):
            # Frozen .app without a plain-python helper would spawn the GUI
            # binary again — refuse unless the helper is configured.
            if getattr(_sys, "frozen", False) and not process_pool_python_executable():
                return False
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
    # Prefer GPU/in-process fast_raw when that stack is available AND actually
    # in use (late probe: gpu_raw_processor only after main-thread bootstrap;
    # until then pool may start, then ImageLoadManager.apply_gpu_decode_profile()
    # tears it down). GPU decode now defaults ON for CUDA and OFF for MPS/CPU
    # (see fast_raw_decode.prefer_gpu_decode_enabled), so a merely-present GPU
    # backend must not disable the process pool: on MPS -- where the default is
    # still off -- that would throw away LibRaw's out-of-process parallelism to
    # serve a GPU decode path nobody is exercising. Gate on in_use, not
    # available.
    if _gpu_demosaic_backend_in_use():
        return False
    return (_os.cpu_count() or 0) >= 4


def process_pool_python_executable() -> str:
    """Optional plain-Python interpreter for ProcessPool workers (macOS .app safe)."""
    return (os.environ.get("RAWVIEWER_PROCESS_POOL_PYTHON") or "").strip()


def process_pool_mp_context():
    """Spawn context; retarget executable when RAWVIEWER_PROCESS_POOL_PYTHON is set."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    helper = process_pool_python_executable()
    if helper and os.path.isfile(helper) and os.access(helper, os.X_OK):
        try:
            ctx.set_executable(helper)
        except Exception:
            pass
    return ctx


def _gpu_demosaic_backend_available() -> bool:
    """True when gpu_raw_processor is loaded and reports a non-CPU backend."""
    try:
        import sys as _sys

        mod = _sys.modules.get("gpu_raw_processor")
        if mod is None:
            return False
        backend = mod.detect_gpu_backend()
        return backend in ("pytorch_cuda", "pytorch_mps", "cupy")
    except Exception:
        return False


def _gpu_demosaic_backend_in_use() -> bool:
    """True when a GPU backend is available AND GPU decode is actually enabled.

    ``_gpu_demosaic_backend_available()`` alone answers "is a GPU present",
    not "will decodes use it" -- the default is per-backend (ON for CUDA, OFF
    for MPS; see fast_raw_decode.prefer_gpu_decode_enabled), and either can be
    forced by RAWVIEWER_PREFER_GPU_DECODE. On a machine where GPU decode is NOT
    in use (Apple MPS at its default, or an explicit =0), callers that gated on
    availability alone were tearing down the LibRaw process pool and capping RAW
    concurrency to GPU semaphore slots (1-3) for a GPU path that never actually
    decodes anything -- a pure throughput regression vs CPU-only hardware.

    On CUDA, where GPU decode is now on by default, that pool teardown is the
    INTENDED trade: decodes run on the GPU (measured 2.35x on full sensor-res
    demosaic), so LibRaw's CPU-side parallelism is worth less than the GPU path
    it would compete with for the same files.
    """
    if not _gpu_demosaic_backend_available():
        return False
    try:
        from fast_raw_decode import prefer_gpu_decode_enabled

        return prefer_gpu_decode_enabled()
    except Exception:
        return False


def raw_load_limit_aligned_to_gpu(base_limit: int) -> int:
    """Cap concurrent heavy RAW work to GPU demosaic concurrency when active."""
    limit = max(1, int(base_limit or 1))
    try:
        import sys as _sys

        mod = _sys.modules.get("gpu_raw_processor")
        if mod is None:
            return limit
        if not _gpu_demosaic_backend_in_use():
            return limit
        gpu_n = int(mod.gpu_decode_concurrency())
        return max(1, min(limit, gpu_n))
    except Exception:
        return limit


def _env_int_bounded(name: str, default: int, *, minimum: int = 1, maximum: int = 64) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, min(maximum, int(raw)))
    except ValueError:
        return default


def _windows_drive_type(path: str) -> Optional[int]:
    """Windows GetDriveTypeW for the volume containing *path* (None if unavailable)."""
    if os.name != "nt":
        return None
    norm = os.path.normpath(path or "")
    if len(norm) < 2 or norm[1] != ":":
        return None
    try:
        import ctypes

        return int(ctypes.windll.kernel32.GetDriveTypeW(norm[:2] + "\\"))
    except Exception:
        return None


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
        drive_type = _windows_drive_type(norm)
        # DRIVE_REMOTE=4, DRIVE_CDROM=5 — prefer scan-first embedded JPEG extraction
        if drive_type in (4, 5):
            return True
    return False


def _windows_system_drive() -> str:
    """Boot/system volume letter (e.g. ``C:``) for probe heuristics."""
    drive = os.environ.get("SystemDrive", "C:").strip()
    if len(drive) >= 2 and drive[1] == ":":
        return drive[:2].upper()
    return "C:"


def invalidate_volume_speed_cache(path: Optional[str]) -> None:
    """Drop cached probe result for the volume containing *path* (e.g. before retry)."""
    root = _volume_root_for(path)
    if not root:
        return
    with _volume_speed_lock:
        _volume_speed_cache.pop(root, None)


def record_volume_speed(path: Optional[str], mbps: float) -> None:
    """Store a probed throughput for the volume containing *path*."""
    root = _volume_root_for(path)
    if not root or mbps <= 0:
        return
    with _volume_speed_lock:
        _volume_speed_cache[root] = float(mbps)


def _windows_drive_is_fixed_local(path: Optional[str]) -> bool:
    """True for non-system fixed volumes (secondary SSD/NVMe) on Windows."""
    if not path or os.name != "nt":
        return False
    norm = os.path.normpath(path)
    if len(norm) < 2 or norm[1] != ":":
        return False
    letter = norm[:2].upper()
    if letter == _windows_system_drive():
        return False
    return _windows_drive_type(norm) == 3


def is_external_or_network_volume(path: Optional[str] = None) -> bool:
    """True when *path* is on an external or network volume (not the boot fixed disk).

    macOS: paths under ``/Volumes/`` (the boot volume is ``/``, not under Volumes).
    Windows: UNC shares; removable USB/SD; mapped network drives; and any
    **non-system** fixed volume (``D:\\``, ``I:\\``, …). USB HDD enclosures often
    report as DRIVE_FIXED, so we probe every letter except ``%SystemDrive%`` once
    per session — fast secondary SSDs stay unthrottled after the probe.
    """
    import sys

    if path:
        norm = os.path.normpath(path)
    else:
        try:
            norm = os.path.normpath(os.getcwd())
        except Exception:
            return False

    if sys.platform == "darwin":
        return norm.startswith("/Volumes/")

    if norm.startswith("\\\\"):
        return True

    if os.name == "nt" and len(norm) >= 2 and norm[1] == ":":
        letter = norm[:2].upper()
        drive_type = _windows_drive_type(norm)
        if drive_type in (2, 4, 5):
            return True
        if drive_type == 3 and letter != _windows_system_drive():
            return True
    return False


def sort_probe_worker_count(
    sample_path: Optional[str] = None,
    *,
    conservative: bool = False,
) -> int:
    """
    Thread pool size for cold EXIF header probes during folder sort.

    Override: RAWVIEWER_SORT_PROBE_WORKERS (1–32).
    Conservative mode (fast-open window): RAWVIEWER_SORT_PROBE_WORKERS_CONSERVATIVE or
    half the machine's cores (was a flat 3 regardless of core count -- on an
    8-core machine that's less than half of even the *conservative* budget a
    proportional formula gives, and the flat cap doesn't back off further on
    a weak machine either. Caller triggers conservative mode for any folder
    over 400 uncached files, not just briefly at fast-open -- on a 6880-file
    folder this flat cap measured a 23s sort at 3 workers).
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
        return min(6, max(2, cpu // 2))

    if sample_path and is_slow_storage_path(sample_path):
        # Network/UNC paths have high latency. We need *more* workers to hide the I/O wait.
        return max(8, cpu * 2)

    # Local SSD/NVMe: I/O-bound header reads; scale past 3 workers (old hard cap under-used CPU).
    return min(12, max(4, cpu - 1))


def index_metadata_worker_count(total_files: int, sample_path: Optional[str] = None) -> int:
    """
    Parallel metadata extraction during semantic index build.

    Override: RAWVIEWER_INDEX_METADATA_WORKERS.
    Large folders (>2000) use a lower default to reduce SQLite EXIF cache lock contention.
    External/USB/network volumes — and any confirmed-slow volume — use a smaller
    pool so background indexing does not saturate the disk that gallery thumbnail
    decodes are also reading from.
    """
    override = os.environ.get("RAWVIEWER_INDEX_METADATA_WORKERS", "").strip()
    if override:
        return _env_int_bounded("RAWVIEWER_INDEX_METADATA_WORKERS", 2, minimum=1, maximum=16)

    cpu = os.cpu_count() or 4

    if total_files > 2000:
        count = min(3, max(2, cpu // 2))
    else:
        count = min(6, max(2, cpu - 1))

    if sample_path is not None:
        try:
            if volume_speed_tier(sample_path) == "slow":
                count = min(count, _env_int_bounded(
                    "RAWVIEWER_SLOW_VOLUME_INDEX_WORKERS", 2, minimum=1, maximum=8
                ))
            elif is_external_or_network_volume(sample_path):
                count = min(count, _env_int_bounded(
                    "RAWVIEWER_EXTERNAL_VOLUME_INDEX_WORKERS", 4, minimum=1, maximum=8
                ))
        except Exception:
            pass
    return count


def _volume_root_for(path: Optional[str]) -> Optional[str]:
    """Stable cache key for the volume/mount that contains *path*."""
    if not path:
        return None
    try:
        norm = os.path.abspath(path)
    except Exception:
        return None
    import sys

    if sys.platform == "darwin" and norm.startswith("/Volumes/"):
        parts = norm.split(os.sep)
        if len(parts) >= 3:
            return os.sep.join(parts[:3])  # /Volumes/<name>
        return norm
    drive, _ = os.path.splitdrive(norm)
    if drive:
        return drive + os.sep
    return os.sep


def _measure_read_mbps(
    path: Optional[str],
    sample_bytes: int,
    timeout_s: float,
) -> Optional[float]:
    """One cold-ish read-throughput measurement (MB/s) for *path*; no caching.

    Reads ``sample_bytes`` starting from a pseudo-random offset rather than the
    file head. Thumbnail/preview extraction (and a prior fast-open) caches the
    head/embedded-JPEG region, so reading the head would report RAM speed; a
    random mid-file offset is far more likely to hit the disk.
    """
    try:
        if not (path and os.path.isfile(path)):
            return None
        size = os.path.getsize(path)
        to_read = min(sample_bytes, size)
        if to_read <= 0:
            return None
        max_offset = max(0, size - to_read)
        offset = 0
        if max_offset > 0:
            try:
                offset = (os.getpid() * 2654435761 + time.monotonic_ns()) % (max_offset + 1)
            except Exception:
                offset = max_offset // 2
        read_total = 0
        start = time.monotonic()
        with open(path, "rb", buffering=0) as fh:
            if offset:
                fh.seek(offset)
            while read_total < to_read:
                chunk = fh.read(min(1024 * 1024, to_read - read_total))
                if not chunk:
                    break
                read_total += len(chunk)
                if time.monotonic() - start > timeout_s:
                    break
        elapsed = max(1e-4, time.monotonic() - start)
        # Require a meaningful sample (>=512 KB) before trusting the number.
        if read_total >= 512 * 1024:
            return (read_total / (1024 * 1024)) / elapsed
    except Exception:
        return None
    return None


def probe_volume_speed(
    sample_path: Optional[str],
    *,
    sample_bytes: int = 8 * 1024 * 1024,
    timeout_s: float = 3.0,
) -> Optional[float]:
    """Very lightweight one-shot read-speed probe for the volume holding *sample_path*.

    Reads up to ``sample_bytes`` (default 4 MB) from a single file and estimates
    throughput in MB/s. The result is cached per volume root and reused for the
    whole session, so this measures each mount at most once. Designed to run off
    the main thread (see ImageLoadManager.prime_volume_speed_async).

    Page-cache hits (implausibly high readings, e.g. probing a just-opened file)
    are rejected and NOT cached, so the caller can retry with a colder file.

    Returns the measured MB/s, or ``None`` if it could not be measured/trusted.
    """
    root = _volume_root_for(sample_path)
    if not root:
        return None
    with _volume_speed_lock:
        cached = _volume_speed_cache.get(root)
    if cached is not None:
        return cached if cached >= 0 else None

    mbps = _measure_read_mbps(sample_path, sample_bytes, timeout_s)

    if mbps is not None and mbps > _VOLUME_SPEED_CACHE_CEILING_MBPS:
        # RAM/page-cache read, not the disk. Do not trust or cache it — leave the
        # volume unprobed so prime_volume_speed_async can try a colder file.
        return None

    with _volume_speed_lock:
        _volume_speed_cache[root] = mbps if mbps is not None else -1.0
    return mbps


def volume_speed_tier(sample_path: Optional[str]) -> str:
    """Classify a probed volume as ``"fast"``, ``"slow"`` or ``"unknown"``.

    A volume is only treated as ``"slow"`` once it has actually been probed and
    measured below RAWVIEWER_VOLUME_SLOW_MBPS (default 120 MB/s) — fast external
    interfaces (Thunderbolt 3/4, USB4, NVMe enclosures) stay ``"fast"`` and are
    never throttled. Unprobed/local volumes return ``"unknown"`` (no throttle).
    """
    root = _volume_root_for(sample_path)
    if not root:
        return "unknown"
    with _volume_speed_lock:
        cached = _volume_speed_cache.get(root)
    if cached is None or cached < 0:
        return "unknown"
    try:
        slow_threshold = float(
            os.environ.get("RAWVIEWER_VOLUME_SLOW_MBPS", "").strip() or 120.0
        )
    except ValueError:
        slow_threshold = 120.0
    return "slow" if cached < slow_threshold else "fast"


def moderate_external_cap_enabled() -> bool:
    """Whether fast (not just confirmed-slow) external volumes get a moderate cap.

    Windows-only by default: the native gallery crash (high concurrent LibRaw +
    GDI/handle pressure) only reproduces on Windows. macOS handled fast external
    (Thunderbolt/USB4) drives at full concurrency without crashing, so we keep its
    original behaviour — only confirmed-slow volumes are throttled there.
    Override with RAWVIEWER_MODERATE_EXTERNAL_CAP=1/0.
    """
    import sys

    raw = os.environ.get("RAWVIEWER_MODERATE_EXTERNAL_CAP", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return sys.platform == "win32"


def raw_concurrent_load_limit(sample_path: Optional[str] = None) -> int:
    """Max concurrent LibRaw full/preview decodes in ImageLoadManager.

    An explicit RAWVIEWER_RAW_LOAD_LIMIT always wins. Otherwise concurrency is
    reduced for confirmed-slow external/network volumes (all platforms). On
    Windows, every external volume also gets a moderate cap (see
    moderate_external_cap_enabled) to avoid the native gallery crash; macOS keeps
    full concurrency on fast external drives.
    """
    cpu = os.cpu_count() or 4
    default = max(4, cpu)
    limit = _env_int_bounded(
        "RAWVIEWER_RAW_LOAD_LIMIT", default, minimum=1, maximum=32
    )

    if not os.environ.get("RAWVIEWER_RAW_LOAD_LIMIT", "").strip():
        if is_external_or_network_volume(sample_path):
            if volume_speed_tier(sample_path) == "slow":
                limit = min(
                    limit,
                    _env_int_bounded(
                        "RAWVIEWER_SLOW_VOLUME_RAW_LIMIT", 3, minimum=1, maximum=8
                    ),
                )
            elif moderate_external_cap_enabled():
                limit = min(
                    limit,
                    _env_int_bounded(
                        "RAWVIEWER_EXTERNAL_VOLUME_RAW_LIMIT", 8, minimum=1, maximum=32
                    ),
                )
    return raw_load_limit_aligned_to_gpu(limit)


def process_pool_worker_count() -> int:
    """LibRaw postprocess process-pool size when RAWVIEWER_USE_PROCESS_POOL is on."""
    cpu = os.cpu_count() or 4
    default = max(2, cpu - 1)
    return _env_int_bounded("RAWVIEWER_PROCESS_POOL_WORKERS", default, minimum=1, maximum=32)


def is_macos_edr_enabled() -> bool:
    # EDR support is removed at this stage because the updated image loading pipeline is not compatible with EDR, causing slow loads.
    return False


def format_edr_status_label(
    *,
    display_mode: str | None = None,
    gpu_view=None,
) -> str | None:
    """Short label for metadata / status bar."""
    return None


def summarize_macos_edr_startup(*, gpu_view=None) -> str:
    """One-line startup hint for logs / status bar."""
    import platform

    if platform.system() != "Darwin":
        return "Display: SDR (not macOS)"
    if not is_macos_edr_enabled():
        return "Display: SDR (RAWVIEWER_DISABLE_EDR=1)"
    parts = ["macOS EDR viewport enabled"]
    if use_raw_edr_for_display():
        parts.append("RAW EDR active")
    elif use_raw_edr_display():
        parts.append("RAW EDR on hold (embedded JPEG workflow)")
    else:
        parts.append("RAW EDR off (RAWVIEWER_RAW_EDR=0)")
    if gpu_view is not None and not getattr(gpu_view, "_edr_initialized", False):
        parts.append("viewport initializing…")
    return " · ".join(parts)


def use_raw_edr_display() -> bool:
    """macOS RAW EDR single-view path (on by default; RAWVIEWER_RAW_EDR=0 to disable).

    Also user-toggleable at runtime via the bottom-bar EDR button
    (QSettings ``raw_edr_display_enabled``) -- EDR display re-decodes each RAW
    with LibRaw highlight reconstruction (~12x the cost of a plain decode), so
    turning it off trades highlight headroom for much faster RAW browsing.
    An explicit RAWVIEWER_RAW_EDR=0 env still hard-disables regardless.

    This setting only gates *whether EDR applies at all*; the dispatch point
    (``RAWviewerMainWindow._should_use_raw_edr_display`` in main.py) further
    idle-defers the actual decode so rapid navigation doesn't pay the 12x
    cost on every frame -- see ``_maybe_arm_raw_edr_after_idle``.
    """
    if not is_macos_edr_enabled():
        return False
    v = os.environ.get("RAWVIEWER_RAW_EDR", "1").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    from PyQt6.QtCore import QSettings

    settings = QSettings("RAWviewer", "RAWviewer")
    return settings.value("raw_edr_display_enabled", True, type=bool)


def use_raw_edr_for_display() -> bool:
    """True when macOS RAW EDR should replace the normal display path (RAW workflow only)."""
    if not use_raw_edr_display():
        return False
    # Embedded-JPEG workflow must keep camera preview pixels (SDR); do not re-decode via LibRaw.
    return use_libraw_consistent_preview_first()


# Windows 10-bit / HDR display notes (Qt 6 / PyQt6, investigated 2025):
# - QImage exposes Format_RGB30 / Format_BGR30 (x-10-10-10) but Qt docs warn against
#   QPainter rendering to them; QGraphicsView / GpuImageView uses 8-bit or RGBX64 paths.
# - Windows HDR output for custom content needs QRhi swapchain formats
#   (HDRExtendedSrgbLinear scRGB or HDR10 PQ) on a QWindow — unlike macOS CALayer EDR
#   (setWantsExtendedDynamicRangeContent_) there is no one-line widget hook.
# - try_load_hdr_image_pixmap() tone-maps to SDR uint8 on non-macOS; RAW stays RGB888.
# - A future Windows path would probe DXGI OUTPUT_DESC1 + optional QRhi viewport (large scope).
def is_windows_hdr_display_available() -> bool:
    """Best-effort HDR display probe for future Windows 10-bit work (not wired yet)."""
    import sys

    if sys.platform != "win32":
        return False
    return False


def linear_edr_float_to_qpixmap(
    rgb_edr: np.ndarray,
    *,
    peak_display: float = 4.0,
    max_edge: int = 0,
) -> QPixmap:
    """Convert extended display-referred float RGB to macOS EDR RGBX64 QPixmap."""
    from PyQt6.QtGui import QColorSpace
    from PyQt6.QtCore import Qt, QSize

    from raw_tone_recovery import encode_edr_rgbx64

    if rgb_edr is None or rgb_edr.size == 0:
        return QPixmap()
    arr_64 = encode_edr_rgbx64(rgb_edr, peak_display=peak_display)
    h, w = arr_64.shape[:2]
    qimage = QImage(arr_64.data, w, h, w * 8, QImage.Format.Format_RGBX64)
    qimage.ndarr = arr_64  # Keep array alive
    qimage.setColorSpace(QColorSpace(QColorSpace.NamedColorSpace.SRgb))
    if max_edge > 0 and max(w, h) > max_edge:
        if w >= h:
            sw = max_edge
            sh = max(1, int(h * max_edge / max(w, 1)))
        else:
            sh = max_edge
            sw = max(1, int(w * max_edge / max(h, 1)))
        qimage = qimage.scaled(
            QSize(sw, sh),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    return QPixmap.fromImage(qimage) if not qimage.isNull() else QPixmap()


def decode_raw_edr_pixmap(
    file_path: str,
    *,
    max_edge: int = 0,
    exif_data: Optional[Dict[str, Any]] = None,
) -> Optional[QPixmap]:
    """Decode RAW with linear 16-bit pipeline and build macOS EDR QPixmap."""
    if not use_raw_edr_for_display():
        return None
    from raw_tone_recovery import decode_raw_for_edr_rgb

    apply_orientation = apply_container_orientation_to_array
    rgb_edr = decode_raw_for_edr_rgb(
        file_path,
        max_edge=max_edge,
        apply_orientation=apply_orientation,
        exif_data=exif_data,
    )
    if rgb_edr is None:
        return None
    pixmap = linear_edr_float_to_qpixmap(rgb_edr, max_edge=max_edge)
    return pixmap if not pixmap.isNull() else None


def pq_to_linear(v: np.ndarray) -> np.ndarray:
    m1 = 2610.0 / 16384.0
    m2 = (2523.0 / 32.0) * 128.0
    c1 = 3424.0 / 4096.0
    c2 = (2413.0 / 4096.0) * 32.0
    c3 = (2392.0 / 4096.0) * 32.0
    
    v_pow = np.power(v, 1.0 / m2)
    numerator = np.maximum(v_pow - c1, 0.0)
    denominator = c2 - c3 * v_pow
    # Avoid divide by zero
    denominator = np.where(denominator == 0.0, 1e-8, denominator)
    linear = np.power(numerator / denominator, 1.0 / m1)
    return linear


def hlg_to_linear(v: np.ndarray) -> np.ndarray:
    a = 0.17883277
    b = 0.28466892
    c = 0.55991073
    
    linear = np.zeros_like(v, dtype=np.float32)
    mask_low = v <= 0.5
    linear[mask_low] = (v[mask_low] ** 2) / 3.0
    
    mask_high = ~mask_low
    # Avoid domain error in exp
    exponent = np.clip((v[mask_high] - c) / a, -80.0, 80.0)
    linear[mask_high] = (np.exp(exponent) + b) / 12.0
    return linear


def linear_to_srgb(l: np.ndarray) -> np.ndarray:
    srgb = np.zeros_like(l, dtype=np.float32)
    mask_low = l <= 0.0031308
    srgb[mask_low] = 12.92 * l[mask_low]
    mask_high = ~mask_low
    srgb[mask_high] = 1.055 * np.power(np.maximum(l[mask_high], 0.0), 1.0 / 2.4) - 0.055
    return np.clip(srgb, 0.0, 1.0)


def tone_map_reinhard(img_float: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    img_exposed = img_float * exposure
    mapped = img_exposed / (1.0 + img_exposed)
    return np.clip(mapped, 0.0, 1.0)


def try_load_hdr_image_pixmap(file_path: str, max_edge: int = 0) -> Optional[QPixmap]:
    """Try to decode HEIF/AVIF/TIFF HDR images with EDR on macOS or SDR tone-mapping."""
    import os
    from PIL import Image
    
    ext = file_path.lower()
    
    is_heif_or_avif = ext.endswith(('.heic', '.heif', '.avif'))
    is_tiff = ext.endswith(('.tif', '.tiff'))
    
    if not (is_heif_or_avif or is_tiff):
        return None

    try:
        bit_depth = 8
        transfer = 2
        icc_profile = b""
        arr = None
        channels = 3
        h, w = 0, 0

        if is_heif_or_avif:
            import pillow_heif
            heif_file = pillow_heif.read_heif(file_path)
            bit_depth = heif_file.info.get("bit_depth", 8)
            nclx = heif_file.info.get("nclx", {})
            if nclx:
                transfer = nclx.get("transfer_characteristics", 2)
            icc_profile = heif_file.info.get("icc_profile", b"")
            w, h = heif_file.size
            channels = 4 if heif_file.mode == 'RGBA' else 3
            
            # Check if HDR
            is_hdr = (bit_depth > 8) or (transfer in (16, 18))
            if not is_hdr:
                return None
                
            if bit_depth > 8:
                arr = np.frombuffer(heif_file.data, dtype=np.uint16)
                arr = arr.reshape((h, w, channels))
            else:
                arr = np.frombuffer(heif_file.data, dtype=np.uint8)
                arr = arr.reshape((h, w, channels))

        elif is_tiff:
            with Image.open(file_path) as pil_img:
                # Check BitsPerSample tag 258
                bits = pil_img.tag.get(258)
                is_16bit = False
                if bits:
                    if isinstance(bits, tuple):
                        is_16bit = any(b > 8 for b in bits)
                    else:
                        is_16bit = bits > 8
                if not is_16bit:
                    is_16bit = pil_img.mode in ("I;16", "I;16L", "I;16B", "RGBA;16", "RGB;16")
                
                if not is_16bit:
                    return None
                icc_profile = pil_img.info.get("icc_profile", b"")
                w, h = pil_img.size
                
            import cv2
            arr = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if arr is None:
                return None
            bit_depth = 16
            
            if len(arr.shape) == 3:
                channels = arr.shape[2]
                if channels == 3:
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                elif channels == 4:
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
            else:
                channels = 1

        if arr is None:
            return None

        # Determine the [0,1] float denominator from the FULL-resolution
        # array before any downsizing. INTER_AREA block-averaging can only
        # ever pull arr.max() down (never up), so computing this heuristic
        # AFTER a downsize can misjudge a true-16-bit image (native peak just
        # above 4095) as 10/12-bit if its bright content is sparse enough to
        # get averaged below the threshold -- picking a denominator dozens of
        # times too small and rendering the preview badly over-bright.
        if arr.dtype == np.uint16:
            max_val = 65535.0 if arr.max() > 4095 else ((2 ** bit_depth) - 1)
        else:
            max_val = 255.0

        # Perform downsizing early to save memory and processing time
        if max_edge > 0 and max(w, h) > max_edge:
            import cv2
            if w >= h:
                sw = max_edge
                sh = max(1, int(h * max_edge / max(w, 1)))
            else:
                sh = max_edge
                sw = max(1, int(w * max_edge / max(h, 1)))
            arr = cv2.resize(arr, (sw, sh), interpolation=cv2.INTER_AREA)
            w, h = sw, sh
            if len(arr.shape) == 2:
                arr = arr.reshape((h, w, 1))

        # Determine float representation [0.0, 1.0]
        img_float = arr.astype(np.float32) / max_val

        from PyQt6.QtGui import QColorSpace
        from PyQt6.QtCore import Qt
        
        # 1. macOS Native EDR path
        if is_macos_edr_enabled():
            # Pad to 4 channels for QImage RGBX64 or RGBA64
            if channels == 3:
                padded = np.empty((h, w, 4), dtype=np.uint16)
                if arr.dtype == np.uint16:
                    padded[:, :, :3] = arr
                else:
                    padded[:, :, :3] = (img_float[:, :, :3] * 65535.0).astype(np.uint16)
                padded[:, :, 3] = 65535
                arr_64 = padded
                q_format = QImage.Format.Format_RGBX64
            elif channels == 4:
                if arr.dtype == np.uint16:
                    arr_64 = arr
                else:
                    arr_64 = (img_float * 65535.0).astype(np.uint16)
                q_format = QImage.Format.Format_RGBA64
            else:  # Grayscale -> RGBX64
                padded = np.empty((h, w, 4), dtype=np.uint16)
                val_16 = (img_float[:, :, 0] * 65535.0).astype(np.uint16) if len(img_float.shape) == 3 else (img_float * 65535.0).astype(np.uint16)
                padded[:, :, 0] = val_16
                padded[:, :, 1] = val_16
                padded[:, :, 2] = val_16
                padded[:, :, 3] = 65535
                arr_64 = padded
                q_format = QImage.Format.Format_RGBX64

            arr_64 = np.ascontiguousarray(arr_64)
            qimage = QImage(
                arr_64.data,
                w,
                h,
                w * 8,
                q_format
            )
            qimage.ndarr = arr_64  # Keep array alive

            # Set the color space so Quartz/Metal rendering will treat values correctly
            if icc_profile:
                cs = QColorSpace.fromIccProfile(icc_profile)
                if cs.isValid():
                    qimage.setColorSpace(cs)
            else:
                # Fallback: assume standard PQ/HLG space
                if transfer == 16:
                    # PQ
                    qimage.setColorSpace(QColorSpace(QColorSpace.NamedColorSpace.SRgb))
                else:
                    # default sRGB
                    qimage.setColorSpace(QColorSpace(QColorSpace.NamedColorSpace.SRgb))

        # 2. Windows / SDR Tone-mapping path
        else:
            rgb_float = img_float[:, :, :3] if channels >= 3 else (np.stack([img_float]*3, axis=-1) if len(img_float.shape) == 2 else np.stack([img_float[:, :, 0]]*3, axis=-1))
            
            # Linearize
            if transfer == 16:
                linear = pq_to_linear(rgb_float)
            elif transfer == 18:
                linear = hlg_to_linear(rgb_float)
            else:
                linear = np.power(rgb_float, 2.2)
                
            # Tone-map using Reinhard
            sdr_linear = tone_map_reinhard(linear, exposure=1.0)
            # Gamma correct to sRGB
            sdr_rgb = linear_to_srgb(sdr_linear)
            sdr_uint8 = (sdr_rgb * 255.0).astype(np.uint8)

            if channels == 4:
                sdr_rgba = np.empty((h, w, 4), dtype=np.uint8)
                sdr_rgba[:, :, :3] = sdr_uint8
                sdr_rgba[:, :, 3] = (img_float[:, :, 3] * 255.0).astype(np.uint8)
                out_arr = sdr_rgba
                q_format = QImage.Format.Format_RGBA8888
            else:
                out_arr = sdr_uint8
                q_format = QImage.Format.Format_RGB888

            out_arr = np.ascontiguousarray(out_arr)
            qimage = QImage(out_arr.data, w, h, w * out_arr.shape[2], q_format)
            qimage.ndarr = out_arr  # Keep array alive
            qimage.setColorSpace(QColorSpace(QColorSpace.NamedColorSpace.SRgb))

        return QPixmap.fromImage(qimage)

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to load HDR image {os.path.basename(file_path)}: {e}")
        return None



