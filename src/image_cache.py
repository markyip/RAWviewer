"""
High-performance image caching system for RAWviewer.

This module provides intelligent caching for thumbnails, full images, and metadata
to dramatically improve browsing performance.
"""

import os
import time
import threading
import sqlite3
import hashlib
import pickle
import numpy as np
from collections import OrderedDict
from typing import Optional, Tuple, Dict, Any, Union
import psutil
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QObject, pyqtSignal


class LRUCache:
    """Thread-safe LRU cache implementation."""

    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None

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

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate
            }


class PersistentEXIFCache:
    """Persistent cache for EXIF data using SQLite."""

    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.rawviewer_cache")
        os.makedirs(cache_dir, exist_ok=True)

        self.db_path = os.path.join(cache_dir, "exif_cache.db")
        self.lock = threading.RLock()
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS exif_cache (
                    file_path TEXT PRIMARY KEY,
                    file_size INTEGER,
                    file_mtime REAL,
                    orientation INTEGER,
                    camera_make TEXT,
                    camera_model TEXT,
                    exif_data BLOB,
                    cached_time REAL
                )
            """)
            conn.commit()
            conn.close()

    def _get_file_hash(self, file_path: str) -> Tuple[int, float]:
        """Get file size and modification time for cache validation."""
        try:
            stat = os.stat(file_path)
            return stat.st_size, stat.st_mtime
        except OSError:
            return 0, 0

    def get(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get cached EXIF data if still valid."""
        file_size, file_mtime = self._get_file_hash(file_path)
        if file_size == 0:
            return None

        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.execute(
                    "SELECT file_size, file_mtime, orientation, camera_make, camera_model, exif_data "
                    "FROM exif_cache WHERE file_path = ?",
                    (file_path,)
                )
                row = cursor.fetchone()
                conn.close()

                if row and row[0] == file_size and row[1] == file_mtime:
                    # Cache is valid
                    exif_data = pickle.loads(row[5]) if row[5] else {}
                    return {
                        'orientation': row[2],
                        'camera_make': row[3],
                        'camera_model': row[4],
                        'exif_data': exif_data
                    }
            except Exception:
                pass

        return None

    def remove(self, file_path: str) -> bool:
        """Remove cached EXIF data for a file."""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.execute("DELETE FROM exif_cache WHERE file_path = ?", (file_path,))
                conn.commit()
                conn.close()
                return cursor.rowcount > 0
            except Exception:
                return False

    def put(self, file_path: str, exif_info: Dict[str, Any]) -> None:
        """Cache EXIF data."""
        file_size, file_mtime = self._get_file_hash(file_path)
        if file_size == 0:
            return

        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                exif_blob = pickle.dumps(exif_info.get('exif_data', {}))
                conn.execute(
                    "INSERT OR REPLACE INTO exif_cache "
                    "(file_path, file_size, file_mtime, orientation, camera_make, camera_model, exif_data, cached_time) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        file_path,
                        file_size,
                        file_mtime,
                        exif_info.get('orientation', 1),
                        exif_info.get('camera_make', ''),
                        exif_info.get('camera_model', ''),
                        exif_blob,
                        time.time()
                    )
                )
                conn.commit()
                conn.close()
            except Exception:
                pass

    def cleanup_old_entries(self, max_age_days: int = 30) -> None:
        """Remove cache entries older than specified days."""
        cutoff_time = time.time() - (max_age_days * 24 * 3600)

        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.execute(
                    "DELETE FROM exif_cache WHERE cached_time < ?", (cutoff_time,))
                conn.commit()
                conn.close()
            except Exception:
                pass


class MemoryMonitor:
    """Monitor system memory usage and provide cache sizing recommendations."""

    def __init__(self):
        self.process = psutil.Process()

    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information."""
        system_memory = psutil.virtual_memory()
        process_memory = self.process.memory_info()

        return {
            'system_total_gb': system_memory.total / (1024**3),
            'system_available_gb': system_memory.available / (1024**3),
            'system_percent_used': system_memory.percent,
            'process_rss_mb': process_memory.rss / (1024**2),
            'process_vms_mb': process_memory.vms / (1024**2)
        }

    def get_recommended_cache_sizes(self) -> Dict[str, int]:
        """Get recommended cache sizes based on available memory."""
        memory_info = self.get_memory_info()
        available_gb = memory_info['system_available_gb']

        # Conservative approach: use max 25% of available memory for caching
        cache_budget_gb = min(available_gb * 0.25, 4.0)  # Cap at 4GB

        # Estimate memory per image (conservative)
        # Full image: ~100MB, Thumbnail: ~1MB
        full_image_mb = 100
        thumbnail_mb = 1

        max_full_images = max(
            5, int((cache_budget_gb * 1024 * 0.8) / full_image_mb))
        max_thumbnails = max(
            50, int((cache_budget_gb * 1024 * 0.2) / thumbnail_mb))

        return {
            'full_images': min(max_full_images, 50),  # Cap at 50 full images
            'thumbnails': min(max_thumbnails, 500),   # Cap at 500 thumbnails
            'cache_budget_mb': int(cache_budget_gb * 1024)
        }


class ImageCache(QObject):
    """Main image caching system with thumbnails, full images, and metadata."""

    # Signals for cache events
    cache_hit = pyqtSignal(str, str)  # file_path, cache_type
    cache_miss = pyqtSignal(str, str)  # file_path, cache_type
    memory_warning = pyqtSignal(float)  # memory_usage_percent

    def __init__(self, cache_dir: str = None):
        super().__init__()

        # Initialize cache directory
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.rawviewer_cache")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor()
        cache_sizes = self.memory_monitor.get_recommended_cache_sizes()

        # Initialize caches
        self.thumbnail_cache = LRUCache(max_size=cache_sizes['thumbnails'])
        self.full_image_cache = LRUCache(max_size=cache_sizes['full_images'])
        self.pixmap_cache = LRUCache(max_size=cache_sizes['full_images'])
        self.exif_cache = PersistentEXIFCache(cache_dir)

        # Cache statistics
        self.stats = {
            'thumbnail_requests': 0,
            'full_image_requests': 0,
            'pixmap_requests': 0,
            'exif_requests': 0
        }

        # Memory management
        self.max_memory_mb = cache_sizes['cache_budget_mb']
        self.memory_check_interval = 60  # seconds
        self.last_memory_check = 0

    def _check_memory_pressure(self) -> bool:
        """Check if we're under memory pressure and need to reduce cache sizes."""
        current_time = time.time()
        if current_time - self.last_memory_check < self.memory_check_interval:
            return False

        self.last_memory_check = current_time
        memory_info = self.memory_monitor.get_memory_info()

        # Emit warning if memory usage is high
        if memory_info['system_percent_used'] > 85:
            self.memory_warning.emit(memory_info['system_percent_used'])

            # Reduce cache sizes under memory pressure
            self._reduce_cache_sizes()
            return True

        return False

    def _reduce_cache_sizes(self) -> None:
        """Reduce cache sizes when under memory pressure."""
        # Reduce by 25%
        new_full_size = max(5, int(self.full_image_cache.max_size * 0.75))
        new_thumbnail_size = max(20, int(self.thumbnail_cache.max_size * 0.75))

        # Clear excess items
        while len(self.full_image_cache.cache) > new_full_size:
            self.full_image_cache.cache.popitem(last=False)

        while len(self.thumbnail_cache.cache) > new_thumbnail_size:
            self.thumbnail_cache.cache.popitem(last=False)

        # Update max sizes
        self.full_image_cache.max_size = new_full_size
        self.thumbnail_cache.max_size = new_thumbnail_size

    def get_thumbnail(self, file_path: str) -> Optional[np.ndarray]:
        """Get cached thumbnail or return None if not cached."""
        self.stats['thumbnail_requests'] += 1
        self._check_memory_pressure()

        thumbnail = self.thumbnail_cache.get(file_path)
        if thumbnail is not None:
            self.cache_hit.emit(file_path, 'thumbnail')
            return thumbnail
        else:
            self.cache_miss.emit(file_path, 'thumbnail')
            return None

    def put_thumbnail(self, file_path: str, thumbnail: np.ndarray) -> None:
        """Cache a thumbnail image."""
        if thumbnail is not None:
            # Ensure thumbnail is reasonable size (max 512x512)
            if thumbnail.shape[0] > 512 or thumbnail.shape[1] > 512:
                # This should be handled by the caller, but safety check
                return
            self.thumbnail_cache.put(file_path, thumbnail.copy())

    def get_full_image(self, file_path: str) -> Optional[np.ndarray]:
        """Get cached full image or return None if not cached."""
        self.stats['full_image_requests'] += 1
        self._check_memory_pressure()

        image = self.full_image_cache.get(file_path)
        if image is not None:
            self.cache_hit.emit(file_path, 'full_image')
            return image
        else:
            self.cache_miss.emit(file_path, 'full_image')
            return None

    def put_full_image(self, file_path: str, image: np.ndarray) -> None:
        """Cache a full processed image."""
        if image is not None:
            self.full_image_cache.put(file_path, image.copy())

    def get_pixmap(self, file_path: str) -> Optional[QPixmap]:
        """Get cached QPixmap or return None if not cached."""
        self.stats['pixmap_requests'] += 1

        pixmap = self.pixmap_cache.get(file_path)
        if pixmap is not None:
            self.cache_hit.emit(file_path, 'pixmap')
            return pixmap
        else:
            self.cache_miss.emit(file_path, 'pixmap')
            return None

    def put_pixmap(self, file_path: str, pixmap: QPixmap) -> None:
        """Cache a QPixmap."""
        if pixmap is not None and not pixmap.isNull():
            self.pixmap_cache.put(file_path, pixmap)

    def get_exif(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get cached EXIF data or return None if not cached."""
        self.stats['exif_requests'] += 1

        exif_data = self.exif_cache.get(file_path)
        if exif_data is not None:
            self.cache_hit.emit(file_path, 'exif')
            return exif_data
        else:
            self.cache_miss.emit(file_path, 'exif')
            return None

    def put_exif(self, file_path: str, exif_data: Dict[str, Any]) -> None:
        """Cache EXIF data."""
        if exif_data:
            self.exif_cache.put(file_path, exif_data)

    def invalidate_file(self, file_path: str) -> None:
        """Remove all cached data for a specific file."""
        self.thumbnail_cache.remove(file_path)
        self.full_image_cache.remove(file_path)
        self.pixmap_cache.remove(file_path)
        # Note: EXIF cache handles file modification time automatically

    def clear_all(self) -> None:
        """Clear all caches."""
        self.thumbnail_cache.clear()
        self.full_image_cache.clear()
        self.pixmap_cache.clear()
        # Don't clear persistent EXIF cache

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        memory_info = self.memory_monitor.get_memory_info()

        return {
            'thumbnail_cache': self.thumbnail_cache.get_stats(),
            'full_image_cache': self.full_image_cache.get_stats(),
            'pixmap_cache': self.pixmap_cache.get_stats(),
            'memory_info': memory_info,
            'request_stats': self.stats.copy(),
            'cache_budget_mb': self.max_memory_mb
        }

    def cleanup_old_cache(self) -> None:
        """Clean up old cache entries."""
        self.exif_cache.cleanup_old_entries()


# Global cache instance
_global_cache: Optional[ImageCache] = None


def get_image_cache() -> ImageCache:
    """Get the global image cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = ImageCache()
    return _global_cache


def initialize_cache(cache_dir: str = None) -> ImageCache:
    """Initialize the global image cache with custom settings."""
    global _global_cache
    _global_cache = ImageCache(cache_dir)
    return _global_cache
