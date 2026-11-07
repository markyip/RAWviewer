"""In-memory cache tiers for RAWviewer (LRU + full-resolution + memory monitor)."""

from __future__ import annotations

import contextlib
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import psutil

from cache.keys import (
    _env_float,
    _env_int,
    _safe_virtual_memory,
    recommended_preview_cache_items,
)

logger = logging.getLogger(__name__)


class FullImageCache:
    """Simplified LRU(10) cache for full-resolution images.

    Replaces the adaptive full_image_cache with byte-budget enforcement
    with a simple fixed-size LRU cache that stores up to 10 full-resolution images.
    """

    def __init__(self, max_size: int = 10) -> None:
        self.max_size = max_size
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def __contains__(self, key: str) -> bool:
        with self.lock:
            return key in self.cache

    def get(self, key: str) -> Any | None:
        """Get cached image, promoting it to MRU."""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None

    def put(self, key: str, value: Any) -> None:
        """Store image in cache with LRU eviction."""
        with self.lock:
            if key in self.cache:
                self.cache[key] = value
                self.cache.move_to_end(key)
            else:
                self.cache[key] = value
                if len(self.cache) > self.max_size:
                    self.cache.popitem(last=False)

    def remove(self, key: str) -> bool:
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

    def get_stats(self) -> dict[str, Any]:
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
            }

    # Alias methods for compatibility with existing code
    def get_full_image(self, file_path: str) -> Any | None:
        """Alias for get() to maintain compatibility."""
        return self.get(file_path)

    def put_full_image(self, file_path: str, image: Any, *, copy: bool = True) -> None:
        """Alias for put() to maintain compatibility."""
        if image is not None:
            import numpy as np

            stored = np.ascontiguousarray(image) if not copy else image.copy()
            self.put(file_path, stored)

    def peek(self, key: str) -> Any | None:
        """Non-mutating lookup for compatibility."""
        with self.lock:
            return self.cache.get(key)


class LRUCache:
    """Thread-safe LRU cache implementation."""

    def __init__(self, max_size: int = 20, *, on_evict: Callable[[Any], None] | None = None) -> None:
        self.max_size = max_size
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.lock = threading.RLock()
        # Called with the key dropped to make room. Diagnostics only: a miss
        # says nothing about whether an entry was never stored, evicted for
        # capacity, or explicitly invalidated, and those want different fixes.
        self.on_evict = on_evict
        self.hits = 0
        self.misses = 0

    def shrink_to_size(self, new_max_size: int) -> None:
        """Update max_size and remove excess items thread-safely."""
        with self.lock:
            self.max_size = new_max_size
            while len(self.cache) > self.max_size:
                evicted_key, _ = self.cache.popitem(last=False)
                if self.on_evict is not None:
                    with contextlib.suppress(Exception):
                        self.on_evict(evicted_key)

    def __contains__(self, key: str) -> bool:
        with self.lock:
            return key in self.cache

    def get(self, key: str) -> Any | None:
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None

    def peek(self, key: str) -> Any | None:
        """Non-mutating lookup: no recency promotion, no hit/miss accounting.

        For "is this warm?" scans (e.g. filmstrip prefetch decisions) that
        must not distort LRU eviction order -- get() would promote entries
        that are merely checked, never displayed, pushing genuinely
        recently-used entries out first.
        """
        with self.lock:
            return self.cache.get(key)

    def put(self, key: str, value: Any) -> None:
        with self.lock:
            if key in self.cache:
                # Update existing item
                self.cache[key] = value
                self.cache.move_to_end(key)
            else:
                # Add new item
                self.cache[key] = value
                if len(self.cache) > self.max_size:
                    # Remove least recently used item
                    evicted_key, _ = self.cache.popitem(last=False)
                    if self.on_evict is not None:
                        with contextlib.suppress(Exception):
                            self.on_evict(evicted_key)

    def remove(self, key: str) -> bool:
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    def remove_keys_for_file_path(self, file_path: str) -> int:
        """Remove entries keyed by file_path (tuple first element or string key)."""
        n = 0
        with self.lock:
            for k in list(self.cache.keys()):
                if k == file_path or (
                    isinstance(k, tuple) and len(k) > 0 and k[0] == file_path
                ):
                    del self.cache[k]
                    n += 1
        return n

    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

    def get_stats(self) -> dict[str, Any]:
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
            }


class MemoryOnlyPersistentCache:
    """No-op persistent cache used when disk/SQLite caching is disabled."""

    def get(self, file_path: str) -> Any | None:
        return None

    def get_multiple(
        self,
        file_paths: list[str],
        file_stats: dict[str, tuple[int, float]] | None = None,
        fast_mode: bool = True,
    ) -> dict[str, dict[str, Any]]:
        return {}

    def get_capture_times_bulk(self, file_paths: list[str]) -> dict[str, str]:
        return {}

    def get_capture_times_for_folder(self, folder_path: str) -> dict[str, str]:
        return {}

    def put(self, file_path: str, value: Any, *args: Any, **kwargs: Any) -> bool:
        # Signature-tolerant: the real persistent caches grow keyword args
        # (file_size, mtime, ...) over time and callers pass them through;
        # a no-op stand-in must absorb whatever the real one accepts, or
        # memory-only mode crashes on every EXIF put (seen with file_size).
        return False

    def has_valid(self, file_path: str) -> bool:
        return False

    def remove(self, file_path: str) -> bool:
        return False

    def cleanup_old_entries(self, max_age_days: int = 30) -> None:
        return

    def clear(self) -> None:
        return

    def close(self) -> None:
        return


class MemoryMonitor:
    """Monitor system memory usage and provide cache sizing recommendations."""

    def __init__(self) -> None:
        self.process = psutil.Process()

    def get_memory_info(self) -> dict[str, float]:
        """Get current memory usage information."""
        system_memory = _safe_virtual_memory()
        try:
            process_memory = self.process.memory_info()
        except Exception as exc:
            logger.warning("process.memory_info() failed (%s)", exc)
            from types import SimpleNamespace

            process_memory = SimpleNamespace(rss=0, vms=0)

        return {
            "system_total_gb": system_memory.total / (1024**3),
            "system_available_gb": system_memory.available / (1024**3),
            "system_percent_used": system_memory.percent,
            "process_rss_mb": process_memory.rss / (1024**2),
            "process_vms_mb": process_memory.vms / (1024**2),
        }

    def system_memory_pressure_level(
        self, memory_info: dict[str, float] | None = None
    ) -> str:
        """Classify system RAM pressure ('critical', 'elevated', 'ok').

        Uses system-wide percent used and available GB only — not process RSS.
        Large semantic-model footprints are normal on high-RAM machines and must
        not trigger cache purges while the OS still has headroom.
        """
        if memory_info is None:
            memory_info = self.get_memory_info()
        percent_used = memory_info["system_percent_used"]
        available_gb = memory_info["system_available_gb"]

        crit_pct = _env_float(
            "RAWVIEWER_MEMORY_PRESSURE_PERCENT_CRITICAL", 90.0, minimum=50.0, maximum=99.0
        )
        crit_avail_gb = _env_float(
            "RAWVIEWER_MEMORY_PRESSURE_AVAILABLE_GB_CRITICAL", 2.0, minimum=0.25, maximum=32.0
        )
        elev_pct = _env_float(
            "RAWVIEWER_MEMORY_PRESSURE_PERCENT_ELEVATED", 90.0, minimum=50.0, maximum=98.0
        )
        elev_avail_gb = _env_float(
            "RAWVIEWER_MEMORY_PRESSURE_AVAILABLE_GB_ELEVATED", 2.0, minimum=0.5, maximum=64.0
        )

        if percent_used >= crit_pct or available_gb <= crit_avail_gb:
            return "critical"
        if percent_used >= elev_pct or available_gb <= elev_avail_gb:
            return "elevated"
        return "ok"

    def get_recommended_cache_sizes(self) -> dict[str, int]:
        """Get recommended cache sizes based on available memory."""
        memory_info = self.get_memory_info()
        available_gb = memory_info["system_available_gb"]

        # Use up to 40% of available memory for caching (was 25%), higher cap
        # for modern systems with 16GB+ RAM. The byte budget (_enforce_numpy_cache_byte_budget)
        # still provides the real limit; this just raises the count ceiling.
        cache_budget_gb = min(available_gb * 0.40, 8.0)

        # Estimate memory per image (conservative)
        # Full image: ~100MB, Thumbnail: ~1MB
        full_image_mb = 100
        thumbnail_mb = 1

        max_full_images = max(8, int((cache_budget_gb * 1024 * 0.8) / full_image_mb))
        max_thumbnails = max(100, int((cache_budget_gb * 1024 * 0.2) / thumbnail_mb))

        preview_items = recommended_preview_cache_items(available_gb)

        return {
            # Opt-in override for zoomed-culling workflows: retaining more
            # sensor-res buffers makes zoomed A<->B revisits instant, at
            # ~100-200MB per slot.
            #
            # This was min(max_full_images, 8). The 8 was a second, cruder
            # limit on top of the byte budget that already governs this cache
            # (_enforce_numpy_cache_byte_budget, 55% of max_memory_mb), and it
            # only ever bound when the byte budget said there was room:
            # measured on a 17GB machine, bytes allowed 13 entries at 24MP and
            # 9 at 33MP while the count cap allowed 8. At 61MP the byte budget
            # binds first (5) and correctly, which is the case the 8 was meant
            # to protect and does not need it.
            #
            # So the RAM-derived figure stands on its own now, with maximum=64
            # as a sanity bound rather than the operative one. Memory is still
            # bounded by bytes; it is simply bounded ONCE, by the limit that
            # knows how large these buffers actually are.
            "full_images": _env_int(
                "RAWVIEWER_FULL_IMAGE_CACHE_ITEMS",
                max_full_images,
                minimum=4,
                maximum=64,
            ),
            "thumbnails": min(max_thumbnails, 2000),
            "preview_images": min(preview_items, 32),
            "cache_budget_mb": int(cache_budget_gb * 1024),
        }
