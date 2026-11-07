"""Main image cache composing memory + persistent tiers (singleton accessor)."""
from __future__ import annotations

import atexit
import contextlib
import logging
import os
from typing import Any

import numpy as np

from cache.image_cache.core import ImageCacheCore
from cache.image_cache.full_exif_util import ImageCacheFullExifUtil
from cache.image_cache.thumb_grid_preview import ImageCacheThumbGridPreview

logger = logging.getLogger(__name__)


class ImageCache(ImageCacheCore, ImageCacheThumbGridPreview, ImageCacheFullExifUtil):
    """Main image caching system with thumbnails, full images, and metadata.

    Composite class assembling core lifecycle, thumbnail/grid/preview operations,
    full image/EXIF/pixmap operations, and utility methods via cooperative
    multiple inheritance (mixin parts).
    """


# ========== Module-level singleton ==========

_global_cache: ImageCache | None = None
_cache_initialized = False
_legacy_cache_cleanup_done = False
_exif_cache_atexit_registered = False


def _cleanup_legacy_disk_cache_once() -> None:
    """One-time cleanup of legacy cache format (e.g., old .npy files)."""
    global _legacy_cache_cleanup_done
    if _legacy_cache_cleanup_done:
        return
    _legacy_cache_cleanup_done = True
    # Removed: no longer needed. Kept as empty function for callers.
    pass


def _register_exif_cache_atexit() -> None:
    """Register atexit handler for EXIF cache flush."""
    global _exif_cache_atexit_registered
    if _exif_cache_atexit_registered:
        return
    _exif_cache_atexit_registered = True

    def _flush_exif_on_exit():
        global _global_cache
        if _global_cache is not None:
            with contextlib.suppress(Exception):
                _global_cache.flush_exif_disk_cache()
    atexit.register(_flush_exif_on_exit)


def get_image_cache() -> ImageCache:
    """Get the singleton ImageCache instance, creating it if needed."""
    global _global_cache
    if _global_cache is None:
        _global_cache = ImageCache(
            cache_dir=os.environ.get("RAWVIEWER_CACHE_DIR", None),
            persistent_cache_enabled=bool(
                os.environ.get("RAWVIEWER_PERSISTENT_CACHE", "1")
                not in ("0", "false", "no")
            ),
        )
        _register_exif_cache_atexit()
        _cleanup_legacy_disk_cache_once()
    return _global_cache


def initialize_cache(cache_dir: str = None, persistent_cache_enabled: bool = False) -> ImageCache:
    """Initialize the image cache explicitly with given parameters."""
    global _global_cache
    if _global_cache is not None:
        _global_cache.close()
    _global_cache = ImageCache(
        cache_dir=cache_dir,
        persistent_cache_enabled=persistent_cache_enabled,
    )
    _register_exif_cache_atexit()
    return _global_cache
