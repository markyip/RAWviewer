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

    def __contains__(self, key: str) -> bool:
        with self.lock:
            return key in self.cache

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
                    cached_time REAL,
                    capture_time TEXT,
                    original_width INTEGER,
                    original_height INTEGER
                )
            """)
            
            # Migration: Check if new columns exist, add if not
            try:
                cursor = conn.execute("PRAGMA table_info(exif_cache)")
                columns = [info[1] for info in cursor.fetchall()]
                
                if 'capture_time' not in columns:
                    print("[CACHE] Migrating database: adding capture_time column")
                    conn.execute("ALTER TABLE exif_cache ADD COLUMN capture_time TEXT")
                    
                if 'original_width' not in columns:
                    print("[CACHE] Migrating database: adding original_width column")
                    conn.execute("ALTER TABLE exif_cache ADD COLUMN original_width INTEGER")
                    
                if 'original_height' not in columns:
                    print("[CACHE] Migrating database: adding original_height column")
                    conn.execute("ALTER TABLE exif_cache ADD COLUMN original_height INTEGER")
                    
            except Exception as e:
                print(f"[CACHE] Error checking/migrating schema: {e}")
                
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
                    "SELECT file_size, file_mtime, orientation, camera_make, camera_model, exif_data, "
                    "capture_time, original_width, original_height "
                    "FROM exif_cache WHERE file_path = ?",
                    (file_path,)
                )
                row = cursor.fetchone()
                conn.close()

                if row and row[0] == file_size and row[1] == file_mtime:
                    # Cache is valid
                    exif_data = pickle.loads(row[5]) if row[5] else {}
                    
                    # Ensure consistent data (DB columns override/augment blob)
                    result_capture_time = row[6]
                    if result_capture_time and 'capture_time' not in exif_data:
                         exif_data['capture_time'] = result_capture_time
                    
                    result_width = row[7]
                    result_height = row[8]
                    if result_width: exif_data['original_width'] = result_width
                    if result_height: exif_data['original_height'] = result_height
                    
                    return {
                        'orientation': row[2],
                        'camera_make': row[3],
                        'camera_model': row[4],
                        'exif_data': exif_data,
                        'capture_time': result_capture_time,
                        'original_width': result_width if result_width else exif_data.get('original_width'),
                        'original_height': result_height if result_height else exif_data.get('original_height')
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
                
                # Extract capture_time for separate column
                capture_time = exif_info.get('capture_time')
                if not capture_time:
                    # Try to extract from exif_data dict if not at top level
                    exif_dict = exif_info.get('exif_data', {})
                    if isinstance(exif_dict, dict):
                         capture_time = exif_dict.get('capture_time')
                
                # Extract dimensions for separate columns
                width = exif_info.get('original_width')
                height = exif_info.get('original_height')
                if not width or not height:
                    exif_dict = exif_info.get('exif_data', {})
                    if isinstance(exif_dict, dict):
                         if not width: width = exif_dict.get('original_width')
                         if not height: height = exif_dict.get('original_height')
                
                conn.execute(
                    "INSERT OR REPLACE INTO exif_cache "
                    "(file_path, file_size, file_mtime, orientation, camera_make, camera_model, exif_data, "
                    "cached_time, capture_time, original_width, original_height) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        file_path,
                        file_size,
                        file_mtime,
                        exif_info.get('orientation', 1),
                        exif_info.get('camera_make', ''),
                        exif_info.get('camera_model', ''),
                        exif_blob,
                        time.time(),
                        capture_time,
                        width,
                        height
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

    def get_multiple(self, file_paths: list, file_stats: Optional[Dict[str, Tuple[int, float]]] = None, fast_mode: bool = True) -> Dict[str, Dict[str, Any]]:
        """Get cached EXIF data for multiple files at once.
        
        Args:
            fast_mode: If True, tries to skip unpickling the full EXIF blob if columns are available.
        """
        if not file_paths:
            return {}

        results = {}
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                # sqlite has a limit on the number of variables in a query
                # Process in chunks of 500
                for i in range(0, len(file_paths), 500):
                    chunk = file_paths[i:i+500]
                    placeholders = ','.join(['?'] * len(chunk))
                    
                    # Fetch all optimized columns
                    query = f"SELECT file_path, file_size, file_mtime, orientation, camera_make, camera_model, " \
                            f"exif_data, capture_time, original_width, original_height " \
                            f"FROM exif_cache WHERE file_path IN ({placeholders})"
                            
                    cursor = conn.execute(query, chunk)
                    
                    for row in cursor.fetchall():
                        path = row[0]
                        # Use provided stats if available, otherwise fetch from disk
                        if file_stats and path in file_stats:
                            file_size, file_mtime = file_stats[path]
                        else:
                            file_size, file_mtime = self._get_file_hash(path)
                        
                        if row[1] == file_size and row[2] == file_mtime:
                            # Cache is valid
                            
                            # Fast Mode Optimization: Skip unpickling if we have all needed data
                            # We need capture_time, width, height, orientation
                            has_fast_data = row[7] is not None and row[8] is not None and row[9] is not None
                            
                            if fast_mode and has_fast_data:
                                # FAST PATH: Skip unpickling!
                                exif_data = {} # Empty dict as placeholder
                                result_capture_time = row[7]
                                result_width = row[8]
                                result_height = row[9]
                            else:
                                # SLOW PATH: Unpickle blob
                                exif_data = pickle.loads(row[6]) if row[6] else {}
                                result_capture_time = row[7] if row[7] else exif_data.get('capture_time')
                                
                                # Width fallback
                                result_width = row[8]
                                if result_width is None:
                                    result_width = exif_data.get('original_width')
                                    if result_width is None:
                                        # Fallback to standard EXIF tags
                                        for tag in ['Image ImageWidth', 'EXIF ExifImageWidth', 'Image Width']:
                                            val = exif_data.get(tag)
                                            if val:
                                                try:
                                                    result_width = int(str(val))
                                                    break
                                                except:
                                                    pass

                                # Height fallback
                                result_height = row[9]
                                if result_height is None:
                                    result_height = exif_data.get('original_height')
                                    if result_height is None:
                                         # Fallback to standard EXIF tags
                                        for tag in ['Image ImageLength', 'EXIF ExifImageLength', 'Image Height', 'Image Length']:
                                            val = exif_data.get(tag)
                                            if val:
                                                try:
                                                    result_height = int(str(val))
                                                    break
                                                except:
                                                    pass
                            
                            results[path] = {
                                'orientation': row[3],
                                'camera_make': row[4],
                                'camera_model': row[5],
                                'exif_data': exif_data,
                                'capture_time': result_capture_time,
                                'original_width': result_width,
                                'original_height': result_height
                            }
                conn.close()
            except Exception:
                pass
        return results

        return results


class PersistentPreviewCache(PersistentThumbnailCache):
    """Persistent disk cache for larger JPEG previews (e.g., 2048px)."""
    
    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.rawviewer_cache")
        # Use 'previews' subdirectory
        self.base_cache_dir = cache_dir 
        super().__init__(cache_dir) # Init with base dir, will setup 'thumbnails'
        
        # Override cache_dir and db_path for previews
        self.cache_dir = os.path.join(self.base_cache_dir, "previews")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.db_path = os.path.join(self.base_cache_dir, "preview_cache.db")
        self._init_db() # Re-init DB with new path
        
    def _init_db(self):
        """Initialize preview cache database."""
        with self.lock:
            try:
                # Reset thread-local connection for new DB path
                if hasattr(self._local, 'conn'):
                    del self._local.conn
                    
                conn = self._get_connection()
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS thumbnail_cache (
                        file_path TEXT PRIMARY KEY,
                        file_size INTEGER,
                        file_mtime REAL,
                        cache_file TEXT,
                        cached_time REAL
                    )
                """)
                # Create index for faster lookups
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_file_path ON thumbnail_cache(file_path)
                """)
                conn.commit()
            except Exception:
                pass


class PersistentThumbnailCache:
    """Persistent disk cache for JPEG thumbnails extracted from RAW files."""
    
    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.rawviewer_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        self.cache_dir = os.path.join(cache_dir, "thumbnails")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.lock = threading.RLock()
        
        # SQLite database to track cache entries
        self.db_path = os.path.join(cache_dir, "thumbnail_cache.db")
        
        # Use thread-local storage for database connections to avoid connection overhead
        self._local = threading.local()
        
        # Cache for file stats to avoid repeated os.stat() calls
        self._file_stats_cache = {}
        self._file_stats_cache_lock = threading.RLock()
        self._file_stats_cache_max_size = 1000
        
        self._init_db()
    
    def _get_connection(self):
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn'):
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            # Enable WAL mode for better concurrent performance
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            self._local.conn = conn
        return self._local.conn
    
    def _init_db(self):
        """Initialize the SQLite database for tracking cache entries."""
        with self.lock:
            try:
                conn = self._get_connection()
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS thumbnail_cache (
                        file_path TEXT PRIMARY KEY,
                        file_size INTEGER,
                        file_mtime REAL,
                        cache_file TEXT,
                        cached_time REAL
                    )
                """)
                # Create index for faster lookups
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_file_path ON thumbnail_cache(file_path)
                """)
                conn.commit()
            except Exception:
                pass
    
    def _get_file_hash(self, file_path: str) -> Tuple[int, float]:
        """Get file size and modification time for cache validation."""
        # Check cache first to avoid repeated os.stat() calls
        with self._file_stats_cache_lock:
            if file_path in self._file_stats_cache:
                return self._file_stats_cache[file_path]
        
        try:
            stat = os.stat(file_path)
            result = (stat.st_size, stat.st_mtime)
            
            # Cache the result
            with self._file_stats_cache_lock:
                if len(self._file_stats_cache) >= self._file_stats_cache_max_size:
                    # Clear half of the cache (simple FIFO-like eviction)
                    keys_to_remove = list(self._file_stats_cache.keys())[:self._file_stats_cache_max_size // 2]
                    for key in keys_to_remove:
                        del self._file_stats_cache[key]
                self._file_stats_cache[file_path] = result
            
            return result
        except OSError:
            return 0, 0
    
    def _get_cache_file_path(self, file_path: str) -> str:
        """Generate cache file path from source file path."""
        # Use hash of file path to avoid filesystem issues with long paths
        path_hash = hashlib.md5(file_path.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"{path_hash}.jpg")
    
    def get(self, file_path: str) -> Optional[bytes]:
        """Get cached JPEG thumbnail if still valid."""
        file_size, file_mtime = self._get_file_hash(file_path)
        if file_size == 0:
            return None
        
        try:
            # Use persistent connection instead of creating new one each time
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT file_size, file_mtime, cache_file FROM thumbnail_cache WHERE file_path = ?",
                (file_path,)
            )
            row = cursor.fetchone()
            
            if row and row[0] == file_size and row[1] == file_mtime:
                # Cache is valid, check if file exists
                cache_file = row[2]
                if os.path.exists(cache_file):
                    with open(cache_file, 'rb') as f:
                        return f.read()
                else:
                    # Cache file missing, remove from database
                    self.remove(file_path)
        except Exception:
            pass
        
        return None
    
    def put(self, file_path: str, jpeg_data: bytes) -> bool:
        """Cache JPEG thumbnail to disk."""
        file_size, file_mtime = self._get_file_hash(file_path)
        if file_size == 0:
            return False
        
        cache_file = self._get_cache_file_path(file_path)
        
        with self.lock:
            try:
                # Write JPEG data to disk
                with open(cache_file, 'wb') as f:
                    f.write(jpeg_data)
                
                # Update database using persistent connection
                conn = self._get_connection()
                conn.execute(
                    "INSERT OR REPLACE INTO thumbnail_cache "
                    "(file_path, file_size, file_mtime, cache_file, cached_time) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (file_path, file_size, file_mtime, cache_file, time.time())
                )
                conn.commit()
                return True
            except Exception as e:
                # Clean up on error
                try:
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
                except Exception:
                    pass
                return False
    
    def remove(self, file_path: str) -> bool:
        """Remove cached thumbnail for a file."""
        with self.lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute(
                    "SELECT cache_file FROM thumbnail_cache WHERE file_path = ?",
                    (file_path,)
                )
                row = cursor.fetchone()
                
                if row:
                    cache_file = row[0]
                    # Delete cache file
                    try:
                        if os.path.exists(cache_file):
                            os.remove(cache_file)
                    except Exception:
                        pass
                
                # Remove from database
                cursor = conn.execute("DELETE FROM thumbnail_cache WHERE file_path = ?", (file_path,))
                conn.commit()
                return cursor.rowcount > 0
            except Exception:
                return False
    
    def cleanup_old_entries(self, max_age_days: int = 30) -> None:
        """Remove cache entries older than specified days."""
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        
        with self.lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute(
                    "SELECT file_path, cache_file FROM thumbnail_cache WHERE cached_time < ?",
                    (cutoff_time,)
                )
                rows = cursor.fetchall()
                
                # Delete cache files
                for row in rows:
                    try:
                        if os.path.exists(row[1]):
                            os.remove(row[1])
                    except Exception:
                        pass
                
                # Remove from database
                conn.execute("DELETE FROM thumbnail_cache WHERE cached_time < ?", (cutoff_time,))
                conn.commit()
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
        # Initialize caches
        self.thumbnail_cache = LRUCache(max_size=cache_sizes['thumbnails'])
        self.preview_cache = LRUCache(max_size=10) # Keep a few high-res previews in memory
        self.full_image_cache = LRUCache(max_size=cache_sizes['full_images'])
        self.pixmap_cache = LRUCache(max_size=cache_sizes['full_images'])
        self.exif_cache = PersistentEXIFCache(cache_dir)
        self.disk_thumbnail_cache = PersistentThumbnailCache(cache_dir)
        self.disk_preview_cache = PersistentPreviewCache(cache_dir)

        # Cache statistics
        self.stats = {
            'thumbnail_requests': 0,
            'preview_requests': 0,
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

        # Check in-memory cache first
        thumbnail = self.thumbnail_cache.get(file_path)
        if thumbnail is not None:
            self.cache_hit.emit(file_path, 'thumbnail')
            return thumbnail
        
        # Check disk cache for JPEG thumbnails
        jpeg_data = self.disk_thumbnail_cache.get(file_path)
        if jpeg_data is not None:
            try:
                from PIL import Image
                import io
                # Load JPEG from bytes
                pil_image = Image.open(io.BytesIO(jpeg_data))
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                thumbnail = np.array(pil_image)
                # Also cache in memory for faster subsequent access
                self.thumbnail_cache.put(file_path, thumbnail.copy())
                self.cache_hit.emit(file_path, 'thumbnail')
                return thumbnail
            except Exception:
                # If loading from disk cache fails, remove it
                self.disk_thumbnail_cache.remove(file_path)
        
        self.cache_miss.emit(file_path, 'thumbnail')
        return None

    def put_thumbnail(self, file_path: str, thumbnail: np.ndarray, jpeg_data: bytes = None) -> None:
        """Cache a thumbnail image."""
        if thumbnail is not None:
            # Ensure thumbnail is reasonable size (max 512x512)
            if thumbnail.shape[0] > 512 or thumbnail.shape[1] > 512:
                # This should be handled by the caller, but safety check
                return
            # Cache in memory
            self.thumbnail_cache.put(file_path, thumbnail.copy())
            
            # If JPEG data is provided, also cache to disk
            if jpeg_data is not None:
                self.disk_thumbnail_cache.put(file_path, jpeg_data)

    def get_preview(self, file_path: str) -> Optional[np.ndarray]:
        """Get cached preview (screen size) or return None."""
        self.stats['preview_requests'] += 1
        self._check_memory_pressure()

        # Check in-memory cache
        preview = self.preview_cache.get(file_path)
        if preview is not None:
            self.cache_hit.emit(file_path, 'preview')
            return preview
        
        # Check disk cache
        jpeg_data = self.disk_preview_cache.get(file_path)
        if jpeg_data is not None:
            try:
                from PIL import Image
                import io
                pil_image = Image.open(io.BytesIO(jpeg_data))
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                preview = np.array(pil_image)
                # Cache in RAM
                self.preview_cache.put(file_path, preview.copy())
                self.cache_hit.emit(file_path, 'preview')
                return preview
            except Exception:
                self.disk_preview_cache.remove(file_path)
        
        self.cache_miss.emit(file_path, 'preview')
        return None

    def put_preview(self, file_path: str, preview: np.ndarray, jpeg_data: bytes = None) -> None:
        """Cache a preview image."""
        if preview is not None:
            self.preview_cache.put(file_path, preview.copy())
            if jpeg_data is not None:
                self.disk_preview_cache.put(file_path, jpeg_data)

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

    def get_multiple_exif(self, file_paths: list, file_stats: Optional[Dict[str, Tuple[int, float]]] = None, fast_mode: bool = True) -> Dict[str, Dict[str, Any]]:
        """Get cached EXIF data for multiple files at once."""
        self.stats['exif_requests'] += len(file_paths)
        return self.exif_cache.get_multiple(file_paths, file_stats, fast_mode)

    def invalidate_file(self, file_path: str) -> None:
        """Remove all cached data for a specific file."""
        self.thumbnail_cache.remove(file_path)
        self.full_image_cache.remove(file_path)
        self.pixmap_cache.remove(file_path)
        self.disk_thumbnail_cache.remove(file_path)
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
        self.disk_thumbnail_cache.cleanup_old_entries()


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
