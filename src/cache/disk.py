"""Persistent SQLite disk caches (thumbnails, grid, previews) for RAWviewer."""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import threading
import time

from cache.keys import (
    _cache_path_key,
    _safe_path_under_root,
)

logger = logging.getLogger(__name__)


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

    def _get_file_hash(self, file_path: str) -> tuple[int, float]:
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
        # Use hash of normalized path so case variants share one disk entry on Windows.
        path_hash = hashlib.md5(_cache_path_key(file_path).encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"{path_hash}.jpg")

    def _jailed_cache_file(self, cache_file: str | None) -> str | None:
        """Allow disk ops only for paths under this cache directory."""
        return _safe_path_under_root(cache_file, getattr(self, "cache_dir", None))

    def has_valid(self, file_path: str) -> bool:
        """True if a valid on-disk entry exists (metadata + file path only, no JPEG read)."""
        key = _cache_path_key(file_path)
        file_size, file_mtime = self._get_file_hash(file_path)
        if file_size == 0:
            return False
        try:
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT file_size, file_mtime, cache_file FROM thumbnail_cache WHERE file_path = ?",
                (key,),
            )
            row = cursor.fetchone()
            if row and row[0] == file_size and row[1] == file_mtime:
                cache_file = self._jailed_cache_file(row[2])
                return bool(cache_file) and os.path.exists(cache_file)
        except Exception:
            pass
        return False

    def get(self, file_path: str) -> bytes | None:
        """Get cached JPEG thumbnail if still valid."""
        key = _cache_path_key(file_path)
        file_size, file_mtime = self._get_file_hash(file_path)
        if file_size == 0:
            return None

        try:
            # Use persistent connection instead of creating new one each time
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT file_size, file_mtime, cache_file FROM thumbnail_cache WHERE file_path = ?",
                (key,)
            )
            row = cursor.fetchone()

            if row and row[0] == file_size and row[1] == file_mtime:
                # Cache is valid, check if file exists (path-jailed)
                cache_file = self._jailed_cache_file(row[2])
                if cache_file and os.path.exists(cache_file):
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
        key = _cache_path_key(file_path)
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
                    (key, file_size, file_mtime, cache_file, time.time())
                )
                conn.commit()
                return True
            except Exception:
                # Clean up on error
                try:
                    safe = self._jailed_cache_file(cache_file)
                    if safe and os.path.exists(safe):
                        os.remove(safe)
                except Exception:
                    pass
                return False

    def remove(self, file_path: str) -> bool:
        """Remove cached thumbnail for a file."""
        key = _cache_path_key(file_path)
        with self.lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute(
                    "SELECT cache_file FROM thumbnail_cache WHERE file_path = ?",
                    (key,)
                )
                row = cursor.fetchone()

                if row:
                    cache_file = self._jailed_cache_file(row[0])
                    # Delete cache file only inside this cache root
                    try:
                        if cache_file and os.path.exists(cache_file):
                            os.remove(cache_file)
                    except Exception:
                        pass

                # Remove from database
                cursor = conn.execute("DELETE FROM thumbnail_cache WHERE file_path = ?", (key,))
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

                # Delete cache files (path-jailed)
                for row in rows:
                    try:
                        cache_file = self._jailed_cache_file(row[1])
                        if cache_file and os.path.exists(cache_file):
                            os.remove(cache_file)
                    except Exception:
                        pass

                # Remove from database
                conn.execute("DELETE FROM thumbnail_cache WHERE cached_time < ?", (cutoff_time,))
                conn.commit()
            except Exception:
                pass

    def get_disk_usage_bytes(self) -> int:
        """Best-effort total bytes of on-disk cache files. Stat-based (no size column
        is tracked in the DB), so only call from a background/startup pass, never a
        per-request hot path."""
        total = 0
        with self.lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute("SELECT cache_file FROM thumbnail_cache")
                for (cache_file,) in cursor.fetchall():
                    try:
                        safe = self._jailed_cache_file(cache_file)
                        if safe:
                            total += os.path.getsize(safe)
                    except OSError:
                        pass
            except Exception:
                pass
        return total

    def evict_lru(self, bytes_to_free: int) -> int:
        """Evict oldest-cached entries (by cached_time) until bytes_to_free is freed
        or the cache is exhausted. Returns bytes actually freed."""
        if bytes_to_free <= 0:
            return 0
        freed = 0
        with self.lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute(
                    "SELECT file_path, cache_file FROM thumbnail_cache ORDER BY cached_time ASC"
                )
                rows = cursor.fetchall()
                evicted_paths: list[str] = []
                for file_path, cache_file in rows:
                    if freed >= bytes_to_free:
                        break
                    safe = self._jailed_cache_file(cache_file)
                    try:
                        if safe:
                            freed += os.path.getsize(safe)
                    except OSError:
                        pass
                    try:
                        if safe and os.path.exists(safe):
                            os.remove(safe)
                    except Exception:
                        pass
                    evicted_paths.append(file_path)
                if evicted_paths:
                    placeholders = ",".join(["?"] * len(evicted_paths))
                    conn.execute(
                        f"DELETE FROM thumbnail_cache WHERE file_path IN ({placeholders})",
                        evicted_paths,
                    )
                    conn.commit()
            except Exception:
                pass
        return freed

    def clear(self) -> None:
        """Remove all disk thumbnail cache files and database rows."""
        with self.lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute("SELECT cache_file FROM thumbnail_cache")
                rows = cursor.fetchall()
                for row in rows:
                    try:
                        cache_file = self._jailed_cache_file(row[0])
                        if cache_file and os.path.exists(cache_file):
                            os.remove(cache_file)
                    except Exception:
                        pass
                conn.execute("DELETE FROM thumbnail_cache")
                conn.commit()
            except Exception:
                pass

        with self._file_stats_cache_lock:
            self._file_stats_cache.clear()

    def close(self):
        """Close the database connection for the current thread."""
        try:
            if hasattr(self, '_local') and hasattr(self._local, 'conn'):
                self._local.conn.close()
                del self._local.conn
        except Exception:
            pass


class PersistentGridCache(PersistentThumbnailCache):
    """Persistent disk cache for medium-res grid tiles (e.g., 512px)."""

    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.rawviewer_cache")
        # Use 'grid' subdirectory
        self.base_cache_dir = cache_dir
        super().__init__(cache_dir) # Init with base dir, will setup 'thumbnails'

        # Override cache_dir and db_path for grid
        self.cache_dir = os.path.join(self.base_cache_dir, "grid")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.db_path = os.path.join(self.base_cache_dir, "grid_cache.db")
        self._init_db() # Re-init DB with new path

    def _init_db(self):
        """Initialize grid cache database."""
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
                conn.execute("PRAGMA journal_mode=WAL")
                conn.commit()
            except Exception:
                pass


class PersistentPreviewCache(PersistentThumbnailCache):
    """Persistent disk cache for medium JPEG previews (default max edge 512px)."""

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
