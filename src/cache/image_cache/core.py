"""Core lifecycle, purge helpers, and memory management for ImageCache."""
from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from collections import OrderedDict

from PyQt6.QtCore import QObject, pyqtSignal

from cache.disk import PersistentGridCache, PersistentPreviewCache, PersistentThumbnailCache
from cache.keys import (
    _cache_path_key,
    _env_int,
    libraw_runtime_stamp,
)
from cache.memory import LRUCache, MemoryMonitor, MemoryOnlyPersistentCache
from cache.metadata import PersistentEXIFCache

logger = logging.getLogger(__name__)


class ImageCacheCore(QObject):
    """Core lifecycle, purge helpers, and memory management."""

    # Signals for cache events
    cache_hit = pyqtSignal(str, str)  # file_path, cache_type
    cache_miss = pyqtSignal(str, str)  # file_path, cache_type
    memory_warning = pyqtSignal(float)  # memory_usage_percent

    def __init__(self, cache_dir: str = None, persistent_cache_enabled: bool = False):
        super().__init__()

        # Initialize cache directory
        self.persistent_cache_enabled = persistent_cache_enabled
        if self.persistent_cache_enabled:
            if cache_dir is None:
                cache_dir = os.path.expanduser("~/.rawviewer_cache")
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_dir = cache_dir
        else:
            self.cache_dir = None

        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor()
        cache_sizes = self.memory_monitor.get_recommended_cache_sizes()

        # Initialize caches
        self.thumbnail_cache = LRUCache(max_size=cache_sizes['thumbnails'])
        self.grid_cache = LRUCache(max_size=cache_sizes['thumbnails'] // 2)
        def _note_preview_evicted(key):
            self.libraw_preview_paths.discard(key)

        self.preview_cache = LRUCache(
            max_size=cache_sizes['preview_images'],
            on_evict=_note_preview_evicted,
        )
        self._full_cache_reasons: OrderedDict[str, str] = OrderedDict()

        def _note_full_evicted(key):
            self._note_full_cache_reason(key, "evicted (capacity 10)")

        self.full_image_cache = LRUCache(
            max_size=10, on_evict=_note_full_evicted
        )
        self.pixmap_cache = LRUCache(max_size=cache_sizes['full_images'])
        self.libraw_preview_paths: set[str] = set()
        if self.cache_dir:
            self._libraw_preview_paths_file = os.path.join(self.cache_dir, "libraw_preview_paths.json")
            self._load_libraw_preview_paths()
        else:
            self._libraw_preview_paths_file = None
        self.full_image_embedded_jpeg_paths: set[str] = set()
        if self.persistent_cache_enabled:
            self.exif_cache = PersistentEXIFCache(cache_dir)
            self.disk_thumbnail_cache = PersistentThumbnailCache(cache_dir)
            self.disk_grid_cache = PersistentGridCache(cache_dir)
            self.disk_preview_cache = PersistentPreviewCache(cache_dir)
            self._purge_stale_oriented_pixel_caches(cache_dir)
            self._purge_stale_edit_baked_pixel_caches(cache_dir)
            self._purge_stale_libraw_pixel_caches(cache_dir)
        else:
            self.exif_cache = MemoryOnlyPersistentCache()
            self.disk_thumbnail_cache = MemoryOnlyPersistentCache()
            self.disk_grid_cache = MemoryOnlyPersistentCache()
            self.disk_preview_cache = MemoryOnlyPersistentCache()

        # In-memory EXIF cache
        self.exif_memory_cache = LRUCache(max_size=cache_sizes['thumbnails'])

        # Cache statistics
        self.stats = {
            'thumbnail_requests': 0,
            'grid_requests': 0,
            'preview_requests': 0,
            'full_image_requests': 0,
            'pixmap_requests': 0,
            'exif_requests': 0,
        }

        # Memory management
        self.max_memory_mb = cache_sizes['cache_budget_mb']
        self._tier_max_memory_mb = int(self.max_memory_mb)
        self.memory_check_interval = 5
        self.last_memory_check = 0
        self.last_aggressive_purge = 0.0
        self.last_reduce_cache = 0.0

        self._mem_poll_lock = threading.Lock()
        self._mem_poll_info: dict[str, float] | None = None
        self._mem_poll_pressure = 'ok'
        self._mem_poll_thread: threading.Thread | None = None

        self._orient_verify_lock = threading.Lock()
        self._orient_verify_attempted: set[str] = set()
        self._orient_verify_attempted_max = 4096
        self._orient_verify_queue: list[str] = []
        self._orient_verify_queue_max = 256
        self._orient_verify_thread: threading.Thread | None = None
        self._closing = False

    def _purge_stale_oriented_pixel_caches(self, cache_dir: str) -> None:
        """One-time purge of persisted PIXEL caches when orientation logic version bumps."""
        try:
            from enhanced_raw_processor import RAW_EXIF_SENSOR_META_VER

            marker = os.path.join(cache_dir, "orient_pixel_ver.txt")
            stored = None
            try:
                with open(marker, encoding="utf-8") as fh:
                    stored = int(fh.read().strip() or 0)
            except Exception:
                stored = None
            if stored == int(RAW_EXIF_SENSOR_META_VER):
                return
            print(
                "[CACHE] Orientation version changed "
                f"({stored} -> {RAW_EXIF_SENSOR_META_VER}); purging persisted pixel caches"
            )
            for cache in (
                self.disk_thumbnail_cache,
                self.disk_grid_cache,
                self.disk_preview_cache,
            ):
                with contextlib.suppress(Exception):
                    cache.clear()
            with open(marker, "w", encoding="utf-8") as fh:
                fh.write(str(int(RAW_EXIF_SENSOR_META_VER)))
        except Exception:
            pass

    EDIT_BAKE_PIXEL_VER = 1

    def _purge_stale_edit_baked_pixel_caches(self, cache_dir: str) -> None:
        """One-time purge of persisted pixel tiers when the edit-bake hygiene version bumps."""
        try:
            marker = os.path.join(cache_dir, "edit_bake_ver.txt")
            stored = None
            try:
                with open(marker, encoding="utf-8") as fh:
                    stored = int(fh.read().strip() or 0)
            except Exception:
                stored = None
            if stored == int(self.EDIT_BAKE_PIXEL_VER):
                return
            print(
                "[CACHE] Edit-bake hygiene version changed "
                f"({stored} -> {self.EDIT_BAKE_PIXEL_VER}); purging persisted pixel caches"
            )
            for cache in (
                self.disk_thumbnail_cache,
                self.disk_grid_cache,
                self.disk_preview_cache,
            ):
                with contextlib.suppress(Exception):
                    cache.clear()
            with open(marker, "w", encoding="utf-8") as fh:
                fh.write(str(int(self.EDIT_BAKE_PIXEL_VER)))
        except Exception:
            pass

    def _purge_stale_libraw_pixel_caches(self, cache_dir: str) -> None:
        """Purge demosaic/preview pixel tiers when LibRaw binary changes."""
        try:
            stamp = libraw_runtime_stamp()
            if not stamp:
                return
            marker = os.path.join(cache_dir, "libraw_pixel_ver.txt")
            stored = None
            try:
                with open(marker, encoding="utf-8") as fh:
                    stored = (fh.read() or "").strip() or None
            except Exception:
                stored = None
            if stored == stamp:
                return
            print(
                "[CACHE] LibRaw runtime changed "
                f"({stored!r} -> {stamp!r}); purging persisted pixel caches"
            )
            for cache in (
                self.disk_thumbnail_cache,
                self.disk_grid_cache,
                self.disk_preview_cache,
            ):
                with contextlib.suppress(Exception):
                    cache.clear()
            try:
                from processing.unified import (
                    prune_libraw_unsupported_paths,
                    release_edit_base_cache,
                )
            except Exception:
                prune_libraw_unsupported_paths = None
                release_edit_base_cache = None
            if release_edit_base_cache is not None:
                with contextlib.suppress(Exception):
                    release_edit_base_cache()
            if prune_libraw_unsupported_paths is not None:
                with contextlib.suppress(Exception):
                    prune_libraw_unsupported_paths(clear_all=True)
            try:
                self.libraw_preview_paths.clear()
                self._save_libraw_preview_paths()
            except Exception:
                pass
            with open(marker, "w", encoding="utf-8") as fh:
                fh.write(stamp)
        except Exception:
            pass

    @staticmethod
    def _path_key(file_path: str) -> str:
        return _cache_path_key(file_path)

    def _ensure_mem_poll_thread(self) -> None:
        """Start background memory-pressure poller."""
        if getattr(self, "_closing", False):
            return
        with self._mem_poll_lock:
            if self._mem_poll_thread is not None and self._mem_poll_thread.is_alive():
                return
            self._mem_poll_thread = threading.Thread(
                target=self._mem_poll_worker,
                name="mem-pressure-poll",
                daemon=True,
            )
            self._mem_poll_thread.start()

    def _mem_poll_worker(self) -> None:
        """Refresh cached memory-info snapshot periodically."""
        while not getattr(self, "_closing", False):
            try:
                info = self.memory_monitor.get_memory_info()
                level = self.memory_monitor.system_memory_pressure_level(info)
            except Exception:
                info, level = None, None
            if info is not None:
                with self._mem_poll_lock:
                    self._mem_poll_info = info
                    self._mem_poll_pressure = level
            time.sleep(self.memory_check_interval)

    def _check_memory_pressure(self) -> bool:
        """Check memory pressure from cached snapshot."""
        current_time = time.time()
        if current_time - self.last_memory_check < self.memory_check_interval:
            return False
        self.last_memory_check = current_time
        self._ensure_mem_poll_thread()

        with self._mem_poll_lock:
            memory_info = self._mem_poll_info
            pressure_level = self._mem_poll_pressure

        if memory_info is None:
            return False

        percent_used = memory_info['system_percent_used']
        available_gb = memory_info['system_available_gb']
        rss_mb = memory_info['process_rss_mb']

        if pressure_level == 'critical':
            if current_time - getattr(self, 'last_aggressive_purge', 0.0) < 60.0:
                return False
            self.last_aggressive_purge = current_time
            logger.warning(
                "CRITICAL system memory pressure (system: %.1f%% used, %.1f GB free, process RSS: %.1f MB). Aggressively purging caches.",
                percent_used, available_gb, rss_mb,
            )
            self.memory_warning.emit(percent_used)
            self._aggressive_purge()
            return True

        if pressure_level == 'elevated':
            if current_time - getattr(self, 'last_reduce_cache', 0.0) < 20.0:
                return False
            self.last_reduce_cache = current_time
            logger.warning(
                "Elevated system memory pressure (system: %.1f%% used, %.1f GB free, process RSS: %.1f MB). Reducing cache sizes.",
                percent_used, available_gb, rss_mb,
            )
            self.memory_warning.emit(percent_used)
            self._reduce_cache_sizes()
            return True

        return False

    def _aggressive_purge(self) -> None:
        """Clear heavy caches aggressively under critical pressure."""
        logger.info("Performing aggressive cache purge...")
        for _k in list(self.full_image_cache.cache.keys()):
            self._note_full_cache_reason(_k, "dropped by memory-pressure purge")
        self.full_image_cache.clear()
        self.pixmap_cache.clear()
        try:
            from processing.unified import (
                release_adjusted_full_cache,
                release_edit_base_cache,
                release_unpacked_raw_cache,
            )
            freed = (
                release_edit_base_cache()
                + release_adjusted_full_cache()
                + release_unpacked_raw_cache()
            )
            if freed:
                import logging
                logging.getLogger(__name__).info(
                    "[MEMORY] released %.0f MB of decode/edit buffers", freed / 1e6
                )
        except Exception:
            pass
        self.full_image_cache.shrink_to_size(3)
        self.pixmap_cache.shrink_to_size(3)
        new_thumbnail_size = max(10, int(self.thumbnail_cache.max_size * 0.5))
        new_grid_size = max(5, int(self.grid_cache.max_size * 0.5))
        new_preview_size = max(4, int(self.preview_cache.max_size * 0.5))
        self.thumbnail_cache.shrink_to_size(new_thumbnail_size)
        self.grid_cache.shrink_to_size(new_grid_size)
        self.preview_cache.shrink_to_size(new_preview_size)
        self._release_gpu_memory()
        import gc
        gc.collect()

    def _reduce_cache_sizes(self) -> None:
        """Reduce cache sizes under elevated memory pressure."""
        new_full_size = max(5, int(self.full_image_cache.max_size * 0.75))
        new_pixmap_size = max(5, int(self.pixmap_cache.max_size * 0.75))
        new_thumbnail_size = max(20, int(self.thumbnail_cache.max_size * 0.75))
        new_grid_size = max(10, int(self.grid_cache.max_size * 0.75))
        min_preview = _env_int(
            "RAWVIEWER_PREVIEW_CACHE_ITEMS_MIN", 6, minimum=4, maximum=32
        )
        new_preview_size = max(
            min_preview, int(self.preview_cache.max_size * 0.75)
        )
        self.full_image_cache.shrink_to_size(new_full_size)
        self.pixmap_cache.shrink_to_size(new_pixmap_size)
        self.thumbnail_cache.shrink_to_size(new_thumbnail_size)
        self.grid_cache.shrink_to_size(new_grid_size)
        self.preview_cache.shrink_to_size(new_preview_size)
        self._release_gpu_memory()

    def _release_gpu_memory(self) -> None:
        """Release GPU/unified-memory caching allocator's pool."""
        try:
            from processing.raw_decode.gpu_processor import release_cached_gpu_memory
            release_cached_gpu_memory()
        except Exception:
            logger.debug("GPU memory release failed", exc_info=True)
