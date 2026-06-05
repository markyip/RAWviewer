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
import shutil
import numpy as np
from collections import OrderedDict
from typing import Optional, Tuple, Dict, Any, Union, List
import psutil
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QObject, pyqtSignal


def disk_preview_max_edge() -> int:
    """Max long edge for JPEG files under ~/.rawviewer_cache/previews (grid tier is separate)."""
    raw = os.environ.get("RAWVIEWER_DISK_PREVIEW_MAX", "512").strip()
    try:
        v = int(raw)
    except Exception:
        v = 512
    return max(256, min(v, 1024))


def memory_preview_max_edge() -> int:
    """Max in-memory preview for progressive single-image RAW display (not disk size)."""
    raw = os.environ.get("RAWVIEWER_MEMORY_PREVIEW_MAX", "1920").strip()
    try:
        v = int(raw)
    except Exception:
        v = 1920
    return max(512, min(v, 4096))


def _jpeg_bytes_max_edge(jpeg_data: bytes, max_edge: int) -> bytes:
    """Downscale JPEG bytes so disk cache entries stay thumbnail-sized."""
    if not jpeg_data or max_edge <= 0:
        return jpeg_data
    try:
        from PIL import Image, ImageOps
        import io

        pil_image = Image.open(io.BytesIO(jpeg_data))
        pil_image = ImageOps.exif_transpose(pil_image)
        w, h = pil_image.size
        if max(w, h) <= max_edge:
            # We must re-save to strip EXIF orientation and ensure it stays rotated.
            out = io.BytesIO()
            pil_image.save(out, format="JPEG", quality=85)
            return out.getvalue()
        scale = max_edge / float(max(w, h))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = pil_image.resize((new_w, new_h), Image.Resampling.HAMMING)
        if resized.mode != "RGB":
            resized = resized.convert("RGB")
        out = io.BytesIO()
        resized.save(out, format="JPEG", quality=85)
        return out.getvalue()
    except Exception:
        return jpeg_data


def _exif_cache_path_key(file_path: str) -> str:
    """Normalized path for exif_cache row keys and lookups (case- and slash-tolerant on Windows)."""
    if not file_path:
        return ""
    try:
        p = file_path if os.path.isabs(file_path) else os.path.abspath(file_path)
        key = os.path.normcase(os.path.normpath(p))
    except OSError:
        key = os.path.normcase(os.path.normpath(file_path))
    if os.sep == "\\":
        key = key.replace("/", "\\")
    return key


def _exif_sql_path_expr(column: str = "file_path") -> str:
    """SQL expression that normalizes slashes for path comparisons."""
    if os.sep == "\\":
        return f"lower(replace({column}, '/', '\\\\'))"
    return f"lower(replace({column}, '\\\\', '/'))"


def _exif_sql_folder_like_pattern(folder_path: str) -> str:
    """LIKE prefix for all files under folder_path (must match _exif_sql_path_expr slashes)."""
    root = _exif_cache_path_key(os.path.abspath(folder_path)).rstrip("\\/")
    root_lower = root.lower()
    if os.sep == "\\":
        return root_lower + "\\%"
    return root_lower.replace("\\", "/") + "/%"


def _exif_sql_folder_like_patterns(folder_path: str) -> List[str]:
    """Candidate LIKE prefixes (drive letter vs UNC realpath may differ in the DB)."""
    patterns: List[str] = []
    try:
        primary = _exif_sql_folder_like_pattern(folder_path)
        if primary:
            patterns.append(primary)
        try:
            real_root = os.path.realpath(os.path.abspath(folder_path))
        except OSError:
            real_root = None
        if real_root:
            alt = _exif_sql_folder_like_pattern(real_root)
            if alt and alt not in patterns:
                patterns.append(alt)
    except OSError:
        pass
    return patterns


def _mtime_matches(cached_mtime: float, current_mtime: float) -> bool:
    """Allow sub-second drift (network shares / FAT timestamp granularity)."""
    try:
        return abs(float(cached_mtime) - float(current_mtime)) < 1.0
    except (TypeError, ValueError):
        return cached_mtime == current_mtime


def _capture_time_from_exif_row(row, exif_data: Optional[dict] = None) -> Optional[str]:
    ct = row[7] if len(row) > 7 else None
    if ct and str(ct).strip():
        return str(ct)
    if exif_data and exif_data.get("capture_time"):
        return str(exif_data["capture_time"])
    return None


def _file_stats_row(
    file_stats: Optional[Dict[str, Tuple[int, float]]], *paths: str
) -> Optional[Tuple[int, float]]:
    if not file_stats:
        return None
    for path in paths:
        if not path:
            continue
        if path in file_stats:
            return file_stats[path]
        nk = _exif_cache_path_key(path)
        for key, row in file_stats.items():
            if _exif_cache_path_key(key) == nk:
                return row
    return None


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


class MemoryOnlyPersistentCache:
    """No-op persistent cache used when disk/SQLite caching is disabled."""

    def get(self, file_path: str) -> Optional[Any]:
        return None

    def get_multiple(self, file_paths: list, file_stats: Optional[Dict[str, Tuple[int, float]]] = None,
                     fast_mode: bool = True) -> Dict[str, Dict[str, Any]]:
        return {}

    def get_capture_times_bulk(self, file_paths: list) -> Dict[str, str]:
        return {}

    def get_capture_times_for_folder(self, folder_path: str) -> Dict[str, str]:
        return {}

    def put(self, file_path: str, value: Any) -> bool:
        return False

    def remove(self, file_path: str) -> bool:
        return False

    def cleanup_old_entries(self, max_age_days: int = 30) -> None:
        return

    def clear(self) -> None:
        return

    def close(self) -> None:
        return


class PersistentEXIFCache:
    """Persistent cache for EXIF data using SQLite."""

    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.rawviewer_cache")
        os.makedirs(cache_dir, exist_ok=True)

        self.db_path = os.path.join(cache_dir, "exif_cache.db")
        self.lock = threading.RLock()
        
        # Use thread-local storage for database connections
        self._local = threading.local()
        
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
            conn.execute("PRAGMA cache_size=1000")
            self._local.conn = conn
        return self._local.conn

    def _init_db(self):
        """Initialize the SQLite database."""
        with self.lock:
            conn = self._get_connection()
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

                if 'sensor_meta_ver' not in columns:
                    print("[CACHE] Migrating database: adding sensor_meta_ver column")
                    conn.execute(
                        "ALTER TABLE exif_cache ADD COLUMN sensor_meta_ver INTEGER DEFAULT 0"
                    )
                    
            except Exception as e:
                print(f"[CACHE] Error checking/migrating schema: {e}")
                
            conn.commit()

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
        cache_key = _exif_cache_path_key(file_path)

        with self.lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute(
                    "SELECT file_size, file_mtime, orientation, camera_make, camera_model, exif_data, "
                    "capture_time, original_width, original_height, sensor_meta_ver "
                    "FROM exif_cache WHERE lower(file_path) = lower(?)",
                    (cache_key,),
                )
                row = cursor.fetchone()

                if row and row[0] == file_size and row[1] == file_mtime:
                    # Cache is valid
                    exif_data = pickle.loads(row[5]) if row[5] else {}
                    
                    # Ensure consistent data (DB columns override/augment blob)
                    result_capture_time = row[6]
                    if result_capture_time and 'capture_time' not in exif_data:
                         exif_data['capture_time'] = result_capture_time
                    
                    result_width = row[7]
                    result_height = row[8]
                    sensor_meta_ver = row[9] if len(row) > 9 else 0
                    if sensor_meta_ver is None:
                        sensor_meta_ver = 0
                    if result_width: exif_data['original_width'] = result_width
                    if result_height: exif_data['original_height'] = result_height
                    
                    return {
                        'orientation': row[2],
                        'camera_make': row[3],
                        'camera_model': row[4],
                        'exif_data': exif_data,
                        'capture_time': result_capture_time,
                        'original_width': result_width if result_width else exif_data.get('original_width'),
                        'original_height': result_height if result_height else exif_data.get('original_height'),
                        'raw_exif_sensor_meta_ver': int(sensor_meta_ver),
                    }
            except Exception:
                pass

        return None

    def remove(self, file_path: str) -> bool:
        """Remove cached EXIF data for a file."""
        cache_key = _exif_cache_path_key(file_path)
        with self.lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute(
                    "DELETE FROM exif_cache WHERE lower(file_path) = lower(?)",
                    (cache_key,),
                )
                conn.commit()
                return cursor.rowcount > 0
            except Exception:
                return False

    def put(
        self,
        file_path: str,
        exif_info: Dict[str, Any],
        *,
        file_size: Optional[int] = None,
        file_mtime: Optional[float] = None,
    ) -> None:
        """Cache EXIF data. Pass file_size/file_mtime when already known (avoids stat under lock)."""
        if file_size is None or file_mtime is None:
            file_size, file_mtime = self._get_file_hash(file_path)
        if file_size == 0:
            return

        storage_path = _exif_cache_path_key(file_path)

        with self.lock:
            try:
                conn = self._get_connection()
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

                sensor_meta_ver = exif_info.get('raw_exif_sensor_meta_ver')
                if sensor_meta_ver is None:
                    sensor_meta_ver = 0
                
                conn.execute(
                    "INSERT OR REPLACE INTO exif_cache "
                    "(file_path, file_size, file_mtime, orientation, camera_make, camera_model, exif_data, "
                    "cached_time, capture_time, original_width, original_height, sensor_meta_ver) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        storage_path,
                        file_size,
                        file_mtime,
                        exif_info.get('orientation', 1),
                        exif_info.get('camera_make', ''),
                        exif_info.get('camera_model', ''),
                        exif_blob,
                        time.time(),
                        capture_time,
                        width,
                        height,
                        int(sensor_meta_ver),
                    )
                )
                conn.commit()
            except Exception:
                pass

    def cleanup_old_entries(self, max_age_days: int = 30) -> None:
        """Remove cache entries older than specified days."""
        cutoff_time = time.time() - (max_age_days * 24 * 3600)

        with self.lock:
            try:
                conn = self._get_connection()
                conn.execute(
                    "DELETE FROM exif_cache WHERE cached_time < ?", (cutoff_time,))
                conn.commit()
            except Exception:
                pass

    def clear(self) -> None:
        """Remove all cached EXIF entries."""
        with self.lock:
            try:
                conn = self._get_connection()
                conn.execute("DELETE FROM exif_cache")
                conn.commit()
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
        try:
            conn = self._get_connection()
            # sqlite has a limit on the number of variables in a query
            # Process in chunks of 500; do not hold lock across entire folder (6889+ files).
            for i in range(0, len(file_paths), 450):
                chunk = file_paths[i : i + 450]
                lookup: Dict[str, str] = {}
                norm_keys: List[str] = []
                for p in chunk:
                    nk = _exif_cache_path_key(p)
                    lookup[nk] = p
                    norm_keys.append(nk)
                placeholders = ",".join(["?"] * len(norm_keys))

                path_expr = _exif_sql_path_expr("file_path")
                query = (
                    f"SELECT file_path, file_size, file_mtime, orientation, camera_make, camera_model, "
                    f"exif_data, capture_time, original_width, original_height, sensor_meta_ver "
                    f"FROM exif_cache WHERE {path_expr} IN ({placeholders})"
                )
                with self.lock:
                    cursor = conn.execute(query, norm_keys)
                    rows = cursor.fetchall()

                for row in rows:
                    db_path = row[0]
                    req_path = lookup.get(_exif_cache_path_key(db_path))
                    if not req_path:
                        continue
                    st = _file_stats_row(file_stats, req_path, db_path)
                    if st is not None:
                        file_size = int(st[0])
                        file_mtime = float(st[1]) if len(st) > 1 else 0.0
                    elif fast_mode:
                        file_size, file_mtime = int(row[1]), float(row[2])
                    else:
                        file_size, file_mtime = self._get_file_hash(req_path)

                    mtime_ok = int(row[1]) == int(file_size) and _mtime_matches(
                        row[2], file_mtime
                    )

                    exif_data: Dict[str, Any] = {}
                    result_capture_time = _capture_time_from_exif_row(row)
                    if not result_capture_time and row[6]:
                        try:
                            exif_data = pickle.loads(row[6]) or {}
                            result_capture_time = _capture_time_from_exif_row(row, exif_data)
                        except Exception:
                            exif_data = {}

                    # For folder sort: trust DB capture_time even when size/mtime drift (K: shares).
                    if not mtime_ok and not result_capture_time:
                        continue

                    sensor_meta_ver = int(row[10] or 0) if len(row) > 10 else 0
                    has_fast_data = (
                        row[7] is not None and row[8] is not None and row[9] is not None
                    )

                    if mtime_ok and fast_mode and has_fast_data:
                        exif_data = {}
                        result_capture_time = row[7]
                        result_width = row[8]
                        result_height = row[9]
                    elif mtime_ok:
                        if not exif_data and row[6]:
                            try:
                                exif_data = pickle.loads(row[6]) or {}
                            except Exception:
                                exif_data = {}
                        if not result_capture_time:
                            result_capture_time = exif_data.get("capture_time")

                        result_width = row[8]
                        if result_width is None:
                            result_width = exif_data.get("original_width")
                            if result_width is None:
                                for tag in [
                                    "Image ImageWidth",
                                    "EXIF ExifImageWidth",
                                    "Image Width",
                                ]:
                                    val = exif_data.get(tag)
                                    if val:
                                        try:
                                            result_width = int(str(val))
                                            break
                                        except Exception:
                                            pass

                        result_height = row[9]
                        if result_height is None:
                            result_height = exif_data.get("original_height")
                            if result_height is None:
                                for tag in [
                                    "Image ImageLength",
                                    "EXIF ExifImageLength",
                                    "Image Height",
                                    "Image Length",
                                ]:
                                    val = exif_data.get(tag)
                                    if val:
                                        try:
                                            result_height = int(str(val))
                                            break
                                        except Exception:
                                            pass
                    else:
                        # sort_only_stale: capture_time for ordering only
                        result_width = row[8]
                        result_height = row[9]

                    results[req_path] = {
                        "orientation": row[3] if row[3] is not None else 1,
                        "camera_make": row[4] or "",
                        "camera_model": row[5] or "",
                        "exif_data": exif_data,
                        "capture_time": result_capture_time,
                        "original_width": result_width,
                        "original_height": result_height,
                        "raw_exif_sensor_meta_ver": sensor_meta_ver,
                    }
        except Exception:
            pass
        return results

    def get_capture_times_bulk(self, file_paths: list) -> Dict[str, str]:
        """Read capture_time for sorting only — no size/mtime validation (fast, K:-safe)."""
        if not file_paths:
            return {}
        out: Dict[str, str] = {}
        try:
            conn = self._get_connection()
            path_expr = _exif_sql_path_expr("file_path")
            # SQLite default max bind variables is 999; one key per path only.
            for i in range(0, len(file_paths), 450):
                chunk = file_paths[i : i + 450]
                lookup: Dict[str, str] = {}
                query_keys: List[str] = []
                for p in chunk:
                    nk = _exif_cache_path_key(p)
                    lookup[nk] = p
                    query_keys.append(nk)
                placeholders = ",".join(["?"] * len(query_keys))
                query = (
                    f"SELECT file_path, capture_time, exif_data FROM exif_cache "
                    f"WHERE {path_expr} IN ({placeholders})"
                )
                with self.lock:
                    rows = conn.execute(query, query_keys).fetchall()
                for db_path, ct_col, blob in rows:
                    req = lookup.get(_exif_cache_path_key(db_path))
                    if not req or req in out:
                        continue
                    ct = ct_col if ct_col and str(ct_col).strip() else None
                    if not ct and blob:
                        try:
                            data = pickle.loads(blob) or {}
                            if isinstance(data, dict):
                                ct = data.get("capture_time")
                        except Exception:
                            pass
                    if ct and str(ct).strip():
                        out[req] = str(ct)
        except Exception:
            pass
        return out

    def get_capture_times_for_folder(self, folder_path: str) -> Dict[str, str]:
        """All cached capture_time entries under folder_path (slash-agnostic LIKE)."""
        if not folder_path:
            return {}
        patterns = _exif_sql_folder_like_patterns(folder_path)
        if not patterns:
            return {}
        path_expr = _exif_sql_path_expr("file_path")
        out: Dict[str, str] = {}
        try:
            conn = self._get_connection()
            for pattern in patterns:
                with self.lock:
                    rows = conn.execute(
                        f"SELECT file_path, capture_time FROM exif_cache "
                        f"WHERE {path_expr} LIKE ? "
                        f"AND capture_time IS NOT NULL AND capture_time != ''",
                        (pattern,),
                    ).fetchall()
                for db_path, ct_col in rows:
                    nk = _exif_cache_path_key(db_path)
                    if nk in out:
                        continue
                    if ct_col and str(ct_col).strip():
                        out[nk] = str(ct_col)
            if not out:
                # Rare: capture_time only inside pickled blob
                for pattern in patterns:
                    with self.lock:
                        rows = conn.execute(
                            f"SELECT file_path, exif_data FROM exif_cache "
                            f"WHERE {path_expr} LIKE ? AND exif_data IS NOT NULL",
                            (pattern,),
                        ).fetchall()
                    for db_path, blob in rows:
                        nk = _exif_cache_path_key(db_path)
                        if nk in out or not blob:
                            continue
                        try:
                            data = pickle.loads(blob) or {}
                            if isinstance(data, dict):
                                ct = data.get("capture_time")
                                if ct and str(ct).strip():
                                    out[nk] = str(ct)
                        except Exception:
                            pass
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning(
                "[CACHE] get_capture_times_for_folder failed for %r (patterns=%s): %s",
                folder_path,
                patterns,
                exc,
                exc_info=True,
            )
        return out
    
    def close(self):
        """Close the database connection for the current thread."""
        try:
            if hasattr(self, '_local') and hasattr(self._local, 'conn'):
                self._local.conn.close()
                del self._local.conn
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

    def clear(self) -> None:
        """Remove all disk thumbnail cache files and database rows."""
        with self.lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute("SELECT cache_file FROM thumbnail_cache")
                rows = cursor.fetchall()
                for row in rows:
                    try:
                        cache_file = row[0]
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

        # Conservative approach: use up to 25% of available memory for caching
        cache_budget_gb = min(available_gb * 0.25, 4.0)

        # Estimate memory per image (conservative)
        # Full image: ~100MB, Thumbnail: ~1MB
        full_image_mb = 100
        thumbnail_mb = 1

        max_full_images = max(
            5, int((cache_budget_gb * 1024 * 0.8) / full_image_mb))
        max_thumbnails = max(
            50, int((cache_budget_gb * 1024 * 0.2) / thumbnail_mb))

        return {
            'full_images': min(max_full_images, 100),
            'thumbnails': min(max_thumbnails, 2000),
            'cache_budget_mb': int(cache_budget_gb * 1024)
        }


class ImageCache(QObject):
    """Main image caching system with thumbnails, full images, and metadata."""

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
        # Initialize caches
        self.thumbnail_cache = LRUCache(max_size=cache_sizes['thumbnails'])
        self.grid_cache = LRUCache(max_size=cache_sizes['thumbnails'] // 2)
        self.preview_cache = LRUCache(max_size=10) # Keep a few high-res previews in memory
        self.full_image_cache = LRUCache(max_size=cache_sizes['full_images'])
        self.pixmap_cache = LRUCache(max_size=cache_sizes['full_images'])
        if self.persistent_cache_enabled:
            self.exif_cache = PersistentEXIFCache(cache_dir)
            self.disk_thumbnail_cache = PersistentThumbnailCache(cache_dir)
            self.disk_grid_cache = PersistentGridCache(cache_dir)
            self.disk_preview_cache = PersistentPreviewCache(cache_dir)
        else:
            # Trial mode: keep all acceleration in RAM and never create/write local cache files.
            self.exif_cache = MemoryOnlyPersistentCache()
            self.disk_thumbnail_cache = MemoryOnlyPersistentCache()
            self.disk_grid_cache = MemoryOnlyPersistentCache()
            self.disk_preview_cache = MemoryOnlyPersistentCache()

        # In-memory EXIF cache (used even if persistent cache is enabled for speed)
        self.exif_memory_cache = LRUCache(max_size=cache_sizes['thumbnails'])

        # Cache statistics
        self.stats = {
            'thumbnail_requests': 0,
            'grid_requests': 0,
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
                from common_image_loader import decode_embedded_jpeg_bytes

                thumbnail = decode_embedded_jpeg_bytes(jpeg_data, max_size=0)
                if thumbnail is None:
                    raise ValueError("decode_embedded_jpeg_bytes failed")
                # Also cache in memory for faster subsequent access
                self.thumbnail_cache.put(file_path, thumbnail.copy())
                self.cache_hit.emit(file_path, 'thumbnail')
                return thumbnail
            except Exception:
                # If loading from disk cache fails, remove it
                self.disk_thumbnail_cache.remove(file_path)
        
        # --- Dynamic Mipmap Fallback ---
        # 1. Downsample from Grid cache (512px) if available
        grid = self.grid_cache.get(file_path)
        if grid is None:
            grid_jpeg = self.disk_grid_cache.get(file_path)
            if grid_jpeg is not None:
                try:
                    from PIL import Image, ImageOps
                    import io
                    pil_image = Image.open(io.BytesIO(grid_jpeg))
                    pil_image = ImageOps.exif_transpose(pil_image)
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    grid = np.array(pil_image)
                except Exception:
                    pass
        
        if grid is not None:
            try:
                from PIL import Image
                import io
                if grid.dtype != np.uint8:
                    grid = grid.astype(np.uint8)
                pil_img = Image.fromarray(grid)
                w, h = pil_img.size
                scale = min(256 / w, 256 / h)
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                thumb_pil = pil_img.resize((new_w, new_h), Image.Resampling.HAMMING)
                thumbnail = np.array(thumb_pil)
                
                # Compress and store in disk/memory caches
                buffer = io.BytesIO()
                thumb_pil.save(buffer, format='JPEG', quality=85)
                t_jpeg = buffer.getvalue()
                self.put_thumbnail(file_path, thumbnail, t_jpeg)
                return thumbnail
            except Exception:
                pass
                
        # 2. Downsample from preview cache (memory or disk, up to ~512px) if available
        preview = self.preview_cache.get(file_path)
        if preview is None:
            preview_jpeg = self.disk_preview_cache.get(file_path)
            if preview_jpeg is not None:
                try:
                    from PIL import Image, ImageOps
                    import io
                    pil_image = Image.open(io.BytesIO(preview_jpeg))
                    pil_image = ImageOps.exif_transpose(pil_image)
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    preview = np.array(pil_image)
                except Exception:
                    pass
                    
        if preview is not None:
            try:
                from PIL import Image
                import io
                if preview.dtype != np.uint8:
                    preview = preview.astype(np.uint8)
                pil_img = Image.fromarray(preview)
                w, h = pil_img.size
                scale = min(256 / w, 256 / h)
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                thumb_pil = pil_img.resize((new_w, new_h), Image.Resampling.HAMMING)
                thumbnail = np.array(thumb_pil)
                
                # Compress and store in disk/memory caches
                buffer = io.BytesIO()
                thumb_pil.save(buffer, format='JPEG', quality=85)
                t_jpeg = buffer.getvalue()
                self.put_thumbnail(file_path, thumbnail, t_jpeg)
                return thumbnail
            except Exception:
                pass

        self.cache_miss.emit(file_path, 'thumbnail')
        return None

    def put_thumbnail(self, file_path: str, thumbnail: np.ndarray, jpeg_data: bytes = None) -> None:
        """Cache a thumbnail image."""
        if thumbnail is not None:
            # Ensure thumbnail is reasonable size (max 1024x1024)
            if thumbnail.shape[0] > 1024 or thumbnail.shape[1] > 1024:
                # This should be handled by the caller, but safety check
                return
            # Cache in memory
            self.thumbnail_cache.put(file_path, thumbnail.copy())
            
            # If JPEG data is provided, also cache to disk
            if jpeg_data is not None:
                self.disk_thumbnail_cache.put(file_path, jpeg_data)

    def get_grid(self, file_path: str) -> Optional[np.ndarray]:
        """Get cached grid image (max 512px) or return None if not cached."""
        if 'grid_requests' not in self.stats:
            self.stats['grid_requests'] = 0
        self.stats['grid_requests'] += 1
        self._check_memory_pressure()

        # 1. Check in-memory grid cache
        grid = self.grid_cache.get(file_path)
        if grid is not None:
            self.cache_hit.emit(file_path, 'grid')
            return grid

        # 2. Check disk grid cache
        jpeg_data = self.disk_grid_cache.get(file_path)
        if jpeg_data is not None:
            try:
                from PIL import Image, ImageOps
                import io
                pil_image = Image.open(io.BytesIO(jpeg_data))
                pil_image = ImageOps.exif_transpose(pil_image)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                grid = np.array(pil_image)
                self.grid_cache.put(file_path, grid.copy())
                self.cache_hit.emit(file_path, 'grid')
                return grid
            except Exception:
                self.disk_grid_cache.remove(file_path)

        # 3. Dynamic Mipmap Fallback 1: Downsample from preview tier
        preview = self.preview_cache.get(file_path)
        if preview is None:
            preview_jpeg = self.disk_preview_cache.get(file_path)
            if preview_jpeg is not None:
                try:
                    from PIL import Image, ImageOps
                    import io
                    pil_image = Image.open(io.BytesIO(preview_jpeg))
                    pil_image = ImageOps.exif_transpose(pil_image)
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    preview = np.array(pil_image)
                except Exception:
                    pass
        
        if preview is not None:
            try:
                from PIL import Image
                import io
                if preview.dtype != np.uint8:
                    preview = preview.astype(np.uint8)
                pil_img = Image.fromarray(preview)
                w, h = pil_img.size
                scale = min(512 / w, 512 / h)
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                grid_pil = pil_img.resize((new_w, new_h), Image.Resampling.HAMMING)
                grid = np.array(grid_pil)
                
                # Compress and store in disk/memory caches
                buffer = io.BytesIO()
                grid_pil.save(buffer, format='JPEG', quality=85)
                g_jpeg = buffer.getvalue()
                self.put_grid(file_path, grid, g_jpeg)
                return grid
            except Exception:
                pass

        # 4. Dynamic Mipmap Fallback 2: Upsample from Thumbnail (256px)
        thumb = self.thumbnail_cache.get(file_path)
        if thumb is None:
            thumb_jpeg = self.disk_thumbnail_cache.get(file_path)
            if thumb_jpeg is not None:
                try:
                    from PIL import Image, ImageOps
                    import io
                    pil_image = Image.open(io.BytesIO(thumb_jpeg))
                    pil_image = ImageOps.exif_transpose(pil_image)
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    thumb = np.array(pil_image)
                except Exception:
                    pass
        
        if thumb is not None:
            try:
                from PIL import Image
                if thumb.dtype != np.uint8:
                    thumb = thumb.astype(np.uint8)
                pil_img = Image.fromarray(thumb)
                w, h = pil_img.size
                scale = min(512 / w, 512 / h)
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                grid_pil = pil_img.resize((new_w, new_h), Image.Resampling.BICUBIC)
                grid = np.array(grid_pil)
                self.grid_cache.put(file_path, grid.copy())
                return grid
            except Exception:
                pass

        self.cache_miss.emit(file_path, 'grid')
        return None

    def put_grid(self, file_path: str, grid: np.ndarray, jpeg_data: bytes = None) -> None:
        """Cache a grid image (max 512px)."""
        if grid is not None:
            # Ensure grid image is reasonable size (max 512x512)
            if grid.shape[0] > 512 or grid.shape[1] > 512:
                return
            # Cache in memory
            self.grid_cache.put(file_path, grid.copy())
            
            # Cache on disk
            if self.persistent_cache_enabled:
                if jpeg_data is not None:
                    self.disk_grid_cache.put(file_path, jpeg_data)
                else:
                    try:
                        from PIL import Image
                        import io
                        if grid.dtype != np.uint8:
                            grid = grid.astype(np.uint8)
                        pil_image = Image.fromarray(grid)
                        buffer = io.BytesIO()
                        pil_image.save(buffer, format='JPEG', quality=85)
                        self.disk_grid_cache.put(file_path, buffer.getvalue())
                    except Exception:
                        pass

    def get_preview(self, file_path: str) -> Optional[np.ndarray]:
        """Get cached preview (screen size) or return None."""
        self.stats['preview_requests'] += 1
        self._check_memory_pressure()

        # Check in-memory cache first
        preview = self.preview_cache.get(file_path)
        if preview is not None:
            self.cache_hit.emit(file_path, 'preview')
            return preview

        # Check disk cache for previews
        jpeg_data = self.disk_preview_cache.get(file_path)
        if jpeg_data is None:
            jpeg_data = self.disk_grid_cache.get(file_path)
        if jpeg_data is not None:
             try:
                from PIL import Image, ImageOps
                import io
                pil_image = Image.open(io.BytesIO(jpeg_data))
                pil_image = ImageOps.exif_transpose(pil_image)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                preview = np.array(pil_image)
                # Also cache in memory
                self.preview_cache.put(file_path, preview.copy())
                self.cache_hit.emit(file_path, 'preview')
                return preview
             except Exception:
                self.disk_preview_cache.remove(file_path)
        
        self.cache_miss.emit(file_path, 'preview')
        return None

    def put_preview(self, file_path: str, preview: np.ndarray, jpeg_data: bytes = None) -> None:
        """Cache a preview image (memory); optional disk JPEG clamped to disk_preview_max_edge()."""
        if preview is not None:
            self.preview_cache.put(file_path, preview.copy())
            if jpeg_data is not None:
                cap = disk_preview_max_edge()
                jpeg_data = _jpeg_bytes_max_edge(jpeg_data, cap)
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

    def get_exif_memory_only(self, file_path: str) -> Optional[Dict[str, Any]]:
        """In-memory EXIF only (no SQLite) — for cache-hit emit during preview-first."""
        return self.exif_memory_cache.get(file_path)

    def get_exif(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get cached EXIF data or return None if not cached."""
        self.stats['exif_requests'] += 1

        # 1. Check in-memory cache first
        exif_data = self.exif_memory_cache.get(file_path)
        if exif_data is None:
            # 2. Check persistent cache
            exif_data = self.exif_cache.get(file_path)
            if exif_data is not None:
                self.exif_memory_cache.put(file_path, exif_data)

        if exif_data is not None:
            self.cache_hit.emit(file_path, 'exif')
            return exif_data
        else:
            self.cache_miss.emit(file_path, 'exif')
            return None

    def put_exif(
        self,
        file_path: str,
        exif_data: Dict[str, Any],
        *,
        file_size: Optional[int] = None,
        file_mtime: Optional[float] = None,
        persist_disk: bool = True,
    ) -> None:
        """Cache EXIF data (memory always; disk optional for preview-first minimal stubs)."""
        if exif_data:
            self.exif_memory_cache.put(file_path, exif_data)
            if persist_disk:
                self.exif_cache.put(
                    file_path,
                    exif_data,
                    file_size=file_size,
                    file_mtime=file_mtime,
                )

    def get_capture_times_for_sort(self, file_paths: list) -> Dict[str, str]:
        """capture_time only, for folder sort (ignores stale mtime on network folders)."""
        return self.exif_cache.get_capture_times_bulk(file_paths)

    def get_capture_times_for_folder_sort(self, folder_path: str) -> Dict[str, str]:
        """All capture_time rows under folder_path (normalized path keys)."""
        return self.exif_cache.get_capture_times_for_folder(folder_path)

    def get_multiple_exif(self, file_paths: list, file_stats: Optional[Dict[str, Tuple[int, float]]] = None, fast_mode: bool = True) -> Dict[str, Dict[str, Any]]:
        """Get cached EXIF data for multiple files at once."""
        self.stats['exif_requests'] += len(file_paths)
        
        results = {}
        missing_paths = []
        
        # Check memory cache first
        for path in file_paths:
            exif = self.exif_memory_cache.get(path)
            if exif:
                results[path] = exif
            else:
                missing_paths.append(path)
        
        if missing_paths:
            # Fetch missing from persistent cache
            db_results = self.exif_cache.get_multiple(missing_paths, file_stats, fast_mode)
            for path, exif in db_results.items():
                self.exif_memory_cache.put(path, exif)
                results[path] = exif
                
        return results

    def invalidate_file(self, file_path: str) -> None:
        """Remove all cached data for a specific file."""
        self.thumbnail_cache.remove(file_path)
        self.grid_cache.remove(file_path)
        self.preview_cache.remove(file_path)
        self.full_image_cache.remove(file_path)
        self.pixmap_cache.remove(file_path)
        self.disk_thumbnail_cache.remove(file_path)
        self.disk_grid_cache.remove(file_path)
        self.disk_preview_cache.remove(file_path)
        self.exif_memory_cache.remove(file_path)
        # Note: EXIF persistent cache handles file modification time automatically

    def clear_all(self) -> None:
        """Clear all caches."""
        self.thumbnail_cache.clear()
        self.grid_cache.clear()
        self.preview_cache.clear()
        self.full_image_cache.clear()
        self.pixmap_cache.clear()
        self.exif_cache.clear()
        self.exif_memory_cache.clear()
        self.disk_thumbnail_cache.clear()
        self.disk_grid_cache.clear()
        self.disk_preview_cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        memory_info = self.memory_monitor.get_memory_info()

        return {
            'thumbnail_cache': self.thumbnail_cache.get_stats(),
            'grid_cache': self.grid_cache.get_stats(),
            'full_image_cache': self.full_image_cache.get_stats(),
            'pixmap_cache': self.pixmap_cache.get_stats(),
            'memory_info': memory_info,
            'request_stats': self.stats.copy(),
            'cache_budget_mb': self.max_memory_mb,
            'persistent_cache_enabled': self.persistent_cache_enabled
        }

    def cleanup_old_cache(self) -> None:
        """Clean up old cache entries."""
        self.exif_cache.cleanup_old_entries()
        self.disk_thumbnail_cache.cleanup_old_entries()
        self.disk_grid_cache.cleanup_old_entries()
        self.disk_preview_cache.cleanup_old_entries()

    def close(self):
        """Close all persistent cache connections."""
        if hasattr(self, 'exif_cache'):
            self.exif_cache.close()
        if hasattr(self, 'disk_thumbnail_cache'):
            self.disk_thumbnail_cache.close()
        if hasattr(self, 'disk_grid_cache'):
            self.disk_grid_cache.close()
        if hasattr(self, 'disk_preview_cache'):
            self.disk_preview_cache.close()


# Global cache instance
_global_cache: Optional[ImageCache] = None
_legacy_cache_cleanup_done = False


def _cleanup_legacy_disk_cache_once() -> None:
    """Remove legacy image caches while preserving semantic search assets."""
    global _legacy_cache_cleanup_done
    if _legacy_cache_cleanup_done:
        return
    _legacy_cache_cleanup_done = True

    legacy_dir = os.path.expanduser("~/.rawviewer_cache")
    if not os.path.isdir(legacy_dir):
        return

    # Keep semantic_index.db, WAL/SHM files, MobileCLIP assets, and any future
    # non-image assets in this cache root. Memory-only mode should only clear
    # obsolete image/EXIF cache stores from older builds.
    obsolete_entries = (
        "exif_cache.db",
        "exif_cache.db-shm",
        "exif_cache.db-wal",
        "thumbnail_cache.db",
        "thumbnail_cache.db-shm",
        "thumbnail_cache.db-wal",
        "grid_cache.db",
        "grid_cache.db-shm",
        "grid_cache.db-wal",
        "preview_cache.db",
        "preview_cache.db-shm",
        "preview_cache.db-wal",
        "thumbnails",
        "grid",
        "previews",
        "images",
    )
    for name in obsolete_entries:
        path = os.path.join(legacy_dir, name)
        if not os.path.exists(path):
            continue
        try:
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        except OSError:
            # Best effort cleanup; app should continue even if deletion fails.
            pass


def get_image_cache() -> ImageCache:
    """Get the global image cache instance."""
    global _global_cache
    if _global_cache is None:
        persistent = os.environ.get("RAWVIEWER_PERSISTENT_CACHE", "1").lower() in {"1", "true", "yes", "on"}
        if not persistent:
            _cleanup_legacy_disk_cache_once()
        _global_cache = ImageCache(persistent_cache_enabled=persistent)
    return _global_cache


def initialize_cache(cache_dir: str = None, persistent_cache_enabled: bool = False) -> ImageCache:
    """Initialize the global image cache with custom settings."""
    global _global_cache
    _global_cache = ImageCache(cache_dir, persistent_cache_enabled=persistent_cache_enabled)
    return _global_cache
