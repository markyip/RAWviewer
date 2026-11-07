"""Module-level helpers and state for the common image loader package."""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any

from image_cache import get_image_cache
from raw_adjustments.file_extensions import RAW_FILE_EXTENSIONS

_io_pressure_until = 0.0
_CAPTURE_TIME_FORMATS = (
    "%H:%M:%S %Y-%m-%d",  # normalized display / cache
    "%Y:%m:%d %H:%M:%S",  # EXIF / exifread default
    "%Y-%m-%d %H:%M:%S",
)
_EMBEDDED_JPEG_WORKFLOW_UNSET = object()
_EMBEDDED_JPEG_WORKFLOW_MEMO: Any = _EMBEDDED_JPEG_WORKFLOW_UNSET
FULL_EMBEDDED_SENSOR_COVERAGE = 0.92
_EAGER_FULL_UNRELIABLE_EXTS = frozenset({".raf", ".3fr"})
_MIN_TRUSTED_SENSOR_LONG_EDGE = 1920
def raise_process_fd_limit(soft_target: int = 4096) -> None:
    """Raise the soft RLIMIT_NOFILE cap (macOS default 256 is too low for gallery prefetch)."""
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = soft_target if hard == resource.RLIM_INFINITY else min(int(hard), soft_target)
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
def parse_capture_time_to_timestamp(capture_time: str | None) -> float:
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
def normalize_capture_time_string(capture_time: str | None) -> str | None:
    """Canonical form used for sorting and status display."""
    ts = parse_capture_time_to_timestamp(capture_time)
    if ts <= 0:
        return None
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S %Y-%m-%d")
def probe_capture_timestamp_from_file(file_path: str) -> float:
    """Lightweight EXIF DateTime probe when cache has no capture_time."""
    try:
        from metadata import backend as metadata_backend

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
    metadata: dict[str, Any] | None = None,
    file_mtime: float = 0.0,
    *,
    file_birthtime: float = 0.0,
    probe_file: bool = True,
    semantic_capture_time: str | None = None,
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
    metadata: dict[str, Any] | None = None,
    file_mtime: float = 0.0,
    *,
    file_birthtime: float = 0.0,
    probe_file: bool = True,
    semantic_capture_time: str | None = None,
    probed_capture_timestamp: float = 0.0,
) -> tuple[bool, float, str]:
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
) -> tuple[Any, ...]:
    """Stable folder/gallery ordering on one chronological axis."""
    del has_capture_time  # kept for call-site compatibility
    base_name = os.path.basename(file_path).lower()
    stem = os.path.splitext(base_name)[0]
    ext = os.path.splitext(base_name)[1]
    raw_rank = 1 if is_raw_file(file_path) else 0
    primary_ts = -timestamp if newest_first else timestamp
    return (primary_ts, stem, raw_rank, ext, base_name)
def invalidate_libraw_consistent_preview_settings() -> None:
    """Clear the memoized ``use_embedded_jpeg_workflow`` QSettings value."""
    global _EMBEDDED_JPEG_WORKFLOW_MEMO
    _EMBEDDED_JPEG_WORKFLOW_MEMO = _EMBEDDED_JPEG_WORKFLOW_UNSET
def _embedded_jpeg_workflow_setting() -> bool | None:
    """Cached QSettings read: True/False when set, None when key absent."""
    global _EMBEDDED_JPEG_WORKFLOW_MEMO
    if _EMBEDDED_JPEG_WORKFLOW_MEMO is not _EMBEDDED_JPEG_WORKFLOW_UNSET:
        return _EMBEDDED_JPEG_WORKFLOW_MEMO
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
def use_libraw_consistent_preview_first(file_path: str | None = None) -> bool:
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
            # fire at all ΓÇö a mere xmp:Rating sidecar was flipping rated
            # files into the RAW workflow, showing LibRaw renders in non-RAW
            # view and blanking galleryΓåÆsingle landings while the interim
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
def sensor_pixel_dimensions(exif_data: dict[str, Any] | None) -> tuple[int, int] | None:
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
    exif_data: dict[str, Any] | None,
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
def check_cache_for_image(file_path: str, use_full_resolution: bool = False) -> tuple[Any | None, str | None]:
    """
    µ¬óµƒÑσ┐½σÅûΣ╕¡µÿ»σÉªσ¡ÿσ£¿σ£ûσâÅµêûτ¢╕Θù£µò╕µôÜπÇé

    Φ┐öσ¢₧: (µò╕µôÜ, σ┐½σÅûΘí₧σ₧ï) µêû (None, None)
    Θí₧σ₧ïσîàµï¼: 'full_image', 'pixmap', 'thumbnail', 'exif'

    µ│¿µäÅ∩╝ÜΦïÑσÅ¬µ£ë EXIF Σ╣ƒµ£âΦªûτé║σæ╜Σ╕¡∩╝¢σû«σ£ûπÇîΘáÉΦ╝ëσâÅτ┤áπÇìΦ½ïτö¿ check_memory_cache_for_image∩╝îΘü┐σàìΦ¬ñΦ╖│ΘüÄΘáÉΦ╝ëπÇé
    """
    cache = get_image_cache()

    # 1. σªéµ₧£Φ½ïµ▒éσà¿Φºúµ₧Éσ║ª∩╝îσä¬σàêµ¬óµƒÑσà¿σ£ûσâÅσ┐½σÅû
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

    # 2. µ¬óµƒÑ Pixmap σ┐½σÅû (σ╕╕τö¿µû╝ QSS/Σ╕ÇΦê¼Θí»τñ║)
    pixmap = cache.get_pixmap(file_path)
    if pixmap is not None and not pixmap.isNull():
        return pixmap, 'pixmap'

    # 3. σªéµ₧£Σ╕ìµÿ»σ┐àΘáêσà¿Φºúµ₧Éσ║ª∩╝îµ¬óµƒÑΦ¿ÿµå╢Θ½öτ╕«σ£ûσ┐½σÅû
    if not use_full_resolution:
        thumbnail = cache.get_thumbnail(file_path)
        if thumbnail is not None:
            return thumbnail, 'thumbnail'

    # 4. µ¬óµƒÑ EXIF σàâµò╕µôÜσ┐½σÅû
    exif_data = cache.get_exif(file_path)
    if exif_data:
        return exif_data, 'exif'

    return None, None
def check_memory_cache_for_image(file_path: str, use_full_resolution: bool = False) -> tuple[Any | None, str | None]:
    """
    µ¬óµƒÑΦ¿ÿµå╢Θ½öσ┐½σÅûΣ╕¡µÿ»σÉªσ¡ÿσ£¿σ£ûσâÅ (Θ¥₧Θÿ╗σí₧∩╝îΣ╕ìΦ«ÇσÅûτúüτ¢ñ)πÇé

    Φ┐öσ¢₧: (µò╕µôÜ, σ┐½σÅûΘí₧σ₧ï) µêû (None, None)
    Θí₧σ₧ïσîàµï¼: 'full_image', 'pixmap', 'thumbnail'
    """
    cache = get_image_cache()

    # 1. σªéµ₧£Φ½ïµ▒éσà¿Φºúµ₧Éσ║ª∩╝îσä¬σàêµ¬óµƒÑσà¿σ£ûσâÅσ┐½σÅû
    if use_full_resolution:
        # τ¢┤µÄÑΦ¿¬σòÅΦ¿ÿµå╢Θ½ö LRUCache∩╝îΘü┐σàìΦº╕τÖ╝τúüτ¢ñΦ«ÇσÅû
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

    # 2. µ¬óµƒÑ Pixmap σ┐½σÅû
    pixmap = cache.pixmap_cache.get(file_path)
    if pixmap is not None and not pixmap.isNull():
        return pixmap, 'pixmap'

    # 3. σªéµ₧£Σ╕ìµÿ»σ┐àΘáêσà¿Φºúµ₧Éσ║ª∩╝îµ¬óµƒÑΦ¿ÿµå╢Θ½öτ╕«σ£û∩╝ÅRAW ΘáÉΦª╜σ┐½σÅû∩╝êΣ╕ìΦº╕τÖ╝τúüτóƒ∩╝ë
    if not use_full_resolution:
        thumbnail = cache.thumbnail_cache.get(file_path)
        if thumbnail is not None:
            return thumbnail, 'thumbnail'
        # σû«σ£ûσ░ÄΦª╜σ╕╕Σ╗Ñ preview_cache τé║Φ╝ëσàÑτ╡Éµ₧£∩╝¢σà¿σ░║σ»╕σàºσ╡î JPEG µÖéτ¡ëσÉîσ«îµò┤σâÅτ┤á
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
    """µ¬óµƒÑµÿ»σÉªτé║ RAW µûçΣ╗╢∩╝êΦêç semantic `format:raw` Σ╜┐τö¿σÉîΣ╕ÇΘÖäµ¬öσÉìΘ¢åσÉê∩╝ëπÇé"""
    ext = os.path.splitext(file_path)[1].lower().lstrip(".")
    return ext in RAW_FILE_EXTENSIONS
def dng_prefers_embedded_preview_first(file_path: str) -> bool:
    """Composite/HDR DNGs often fail LibRaw; use embedded JPEG before sensor decode."""
    if os.path.splitext(file_path)[1].lower() != ".dng":
        return False
    return "-hdr" in os.path.basename(file_path).lower()
def is_tiff_file(file_path: str) -> bool:
    """µ¬óµƒÑµÿ»σÉªτé║ TIFF µûçΣ╗╢∩╝êσîàµï¼Θî»Φ¬ñµô┤σ▒òσÉìτÜäµâàµ│ü∩╝ë"""
    file_ext = os.path.splitext(file_path)[1].lower()
    is_tiff = file_ext in ('.tiff', '.tif')

    # µ¬óµƒÑµûçΣ╗╢σàºσ«╣Σ╗Ñµ¬óµ╕¼Θî»Φ¬ñµô┤σ▒òσÉìτÜä TIFF µûçΣ╗╢
    if not is_tiff:
        try:
            from PIL import Image
            with Image.open(file_path) as test_img:
                if test_img.format in ('TIFF', 'TIF'):
                    is_tiff = True
        except Exception:
            pass  # Σ╕ìµÿ» PIL σÅ»Φ«ÇµûçΣ╗╢µêûΣ╕ìµÿ» TIFF

    return is_tiff
def _regular_image_max_edge() -> int:
    """Safety cap when full-resolution Qt/PIL decode fails (e.g. huge panorama JPEG)."""
    raw = os.environ.get("RAWVIEWER_REGULAR_IMAGE_MAX_EDGE", "16384").strip()
    try:
        v = int(raw)
    except Exception:
        v = 16384
    return max(2048, min(v, 65536))

