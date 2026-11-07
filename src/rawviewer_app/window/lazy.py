"""Deferred heavy imports shared by every main-window module.

The GUI keeps startup fast by importing numpy/rawpy/metadata backends only
after the splash is visible. Modules that reference those names import the
placeholders from here and call :func:`register`; :func:`_publish` then pushes
the real objects into every registered module namespace.
"""

from __future__ import annotations

import contextlib
import sys
from typing import Any

LAZY_NAMES = (
    "rawpy",
    "np",
    "exifread",
    "qta",
    "SemanticImageIndex",
    "get_image_cache",
    "initialize_cache",
    "EnhancedRAWProcessor",
    "PreloadManager",
    "ThumbnailExtractor",
    "get_image_load_manager",
    "Priority",
    "is_raw_file",
    "load_pixmap_safe",
    "check_memory_cache_for_image",
    "use_libraw_consistent_preview_first",
    "image_covers_sensor_resolution",
    "use_progressive_raw_loading",
    "metadata_index_idle_delay_ms",
    "ImageHistogramWidget",
    "ThumbnailLabel",
    "ExternalJustifiedGallery",
    "ImageLocationMapWidget",
    "ImageAdjustPanelWidget",
    "exif_backend_mode",
    "exif_orientation_after_cw90",
    "has_pyexiv2",
    "process_file_from_path",
)

_BINDINGS: dict[str, Any] = dict.fromkeys(LAZY_NAMES)
_CONSUMERS: list[Any] = []

globals().update(_BINDINGS)


def register(module: Any) -> None:
    """Push the current bindings into ``module`` and track it for updates."""
    for name in LAZY_NAMES:
        setattr(module, name, _BINDINGS[name])
    _CONSUMERS.append(module)


def _publish(values: dict[str, Any]) -> None:
    """Store resolved objects and update every registered consumer module."""
    for name in LAZY_NAMES:
        if name not in values:
            continue
        value = values[name]
        _BINDINGS[name] = value
        setattr(sys.modules[__name__], name, value)
        for module in _CONSUMERS:
            setattr(module, name, value)


def _lazy_import_heavy_modules(splash: Any = None) -> None:
    """Import heavy modules while the splash screen is visible."""
    from bootstrap_pkg.splash import splash_show_status

    def _update_splash(msg: str) -> None:
        splash_show_status(splash, msg)

    _update_splash("Loading RAW processor...")
    import rawpy as _rawpy
    rawpy = _rawpy

    _update_splash("Loading math libraries...")
    import numpy as _np
    np = _np

    _update_splash("Loading metadata engine...")
    import exifread as _exifread
    exifread = _exifread

    _update_splash("Loading iconography...")
    try:
        import qtawesome as _qta
        qta = _qta
    except Exception:
        qta = None

    _update_splash("Loading metadata backend...")
    from metadata.backend import exif_backend_mode as _ebm
    from metadata.backend import exif_orientation_after_cw90 as _eocw
    from metadata.backend import has_pyexiv2 as _hp2
    from metadata.backend import process_file_from_path as _pfp
    exif_backend_mode = _ebm
    exif_orientation_after_cw90 = _eocw
    has_pyexiv2 = _hp2
    process_file_from_path = _pfp

    with contextlib.suppress(Exception):
        _update_splash(f"Metadata engine: {'pyexiv2' if has_pyexiv2() else 'exifread'}")

    _update_splash("Loading core architecture...")
    from image_cache import get_image_cache as _get_image_cache
    from image_cache import initialize_cache as _initialize_cache
    get_image_cache = _get_image_cache
    initialize_cache = _initialize_cache

    from enhanced_raw_processor import EnhancedRAWProcessor as _EnhancedRAWProcessor
    from enhanced_raw_processor import PreloadManager as _PreloadManager
    from enhanced_raw_processor import ThumbnailExtractor as _ThumbnailExtractor
    EnhancedRAWProcessor = _EnhancedRAWProcessor
    PreloadManager = _PreloadManager
    ThumbnailExtractor = _ThumbnailExtractor

    from image_load_manager import Priority as _Priority
    from image_load_manager import get_image_load_manager as _get_image_load_manager
    get_image_load_manager = _get_image_load_manager
    Priority = _Priority

    from common_image_loader import check_memory_cache_for_image as _check_memory_cache_for_image
    from common_image_loader import is_raw_file as _is_raw_file
    from common_image_loader import load_pixmap_safe as _load_pixmap_safe
    is_raw_file = _is_raw_file
    load_pixmap_safe = _load_pixmap_safe
    check_memory_cache_for_image = _check_memory_cache_for_image

    from common_image_loader import use_libraw_consistent_preview_first as _use_libraw_consistent_preview_first
    use_libraw_consistent_preview_first = _use_libraw_consistent_preview_first
    from common_image_loader import image_covers_sensor_resolution as _image_covers_sensor_resolution
    image_covers_sensor_resolution = _image_covers_sensor_resolution
    from common_image_loader import (
        metadata_index_idle_delay_ms as _metadata_index_idle_delay_ms,
    )
    from common_image_loader import (
        use_progressive_raw_loading as _use_progressive_raw_loading,
    )
    use_progressive_raw_loading = _use_progressive_raw_loading
    metadata_index_idle_delay_ms = _metadata_index_idle_delay_ms

    from metadata.histogram import ImageHistogramWidget as _ImageHistogramWidget
    ImageHistogramWidget = _ImageHistogramWidget

    _update_splash("Loading UI components...")
    from rawviewer_ui.overlays.location_map_overlay import ImageLocationMapWidget as _ImageLocationMapWidget
    from rawviewer_ui.widgets import ThumbnailLabel as _ThumbnailLabel
    from rawviewer_ui.widgets.gallery import JustifiedGallery as _ExternalJustifiedGallery
    ThumbnailLabel = _ThumbnailLabel
    ExternalJustifiedGallery = _ExternalJustifiedGallery
    ImageLocationMapWidget = _ImageLocationMapWidget
    from raw_adjustments import editing_features_enabled as _editing_features_enabled

    if _editing_features_enabled():
        from rawviewer_ui.widgets.adjust import ImageAdjustPanelWidget as _ImageAdjustPanelWidget

        ImageAdjustPanelWidget = _ImageAdjustPanelWidget
    else:
        ImageAdjustPanelWidget = None

    _publish(locals())
    _update_splash("Startup complete")


__all__ = ["LAZY_NAMES", "register", "_lazy_import_heavy_modules"]
