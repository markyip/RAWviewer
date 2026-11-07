"""
Enhanced RAW processor with smart thumbnail utilization and caching.

This module provides optimized RAW processing with immediate thumbnail display,
intelligent caching, and progressive image loading for maximum performance.
"""

from .enhanced_processor import EnhancedRAWProcessor
from .exif_extractor import EXIFExtractor
from .extraction import (
    _BYTESCAN_FIRST_EXTS,
    RAW_EXIF_SENSOR_META_VER,
    _bytescan_first_enabled,
    _detect_nef_he_compression,
    _extract_bmff_uuid_embedded_jpeg,
    _extract_embedded_jpeg_by_scan_impl,
    _heavy_fallback_semaphore,
    _largest_jpeg_from_blob,
    _orientation_from_embedded_preview,
    _qimage_to_rgb_array,
    _rawpy_global_lock,
    _rawpy_unpack_concurrency,
    _thumbnail_via_qimage_reader,
    cached_raw_exif_orientation_trustworthy,
    extract_embedded_jpeg_by_scan,
    extract_previews_via_tiff_parse,
    get_jpeg_dimensions,
    probe_effective_raw_orientation,
    set_unpack_concurrency_cap,
)
from .optimized_processor import OptimizedRAWProcessor
from .preload_manager import PreloadManager
from .thumbnail_extractor import ThumbnailExtractor

# Preserve original module path for existing imports
__all__ = [
    "EnhancedRAWProcessor",
    "EXIFExtractor",
    "OptimizedRAWProcessor",
    "PreloadManager",
    "ThumbnailExtractor",
    # Module-level functions
    "_BYTESCAN_FIRST_EXTS",
    "_bytescan_first_enabled",
    "_detect_nef_he_compression",
    "_extract_bmff_uuid_embedded_jpeg",
    "_extract_embedded_jpeg_by_scan_impl",
    "_heavy_fallback_semaphore",
    "_largest_jpeg_from_blob",
    "_orientation_from_embedded_preview",
    "_qimage_to_rgb_array",
    "_rawpy_global_lock",
    "_rawpy_unpack_concurrency",
    "_thumbnail_via_qimage_reader",
    "cached_raw_exif_orientation_trustworthy",
    "extract_embedded_jpeg_by_scan",
    "extract_previews_via_tiff_parse",
    "get_jpeg_dimensions",
    "probe_effective_raw_orientation",
    "set_unpack_concurrency_cap",
    # Module-level state
    "RAW_EXIF_SENSOR_META_VER",
]
