"""
High-performance image caching system for RAWviewer.

This module provides intelligent caching for thumbnails, full images, and metadata
to dramatically improve browsing performance.
"""

import os
import logging
import platform
import subprocess
import time
import threading
import sqlite3
import hashlib
import pickle
import shutil
import numpy as np
from collections import OrderedDict
from typing import Optional, Tuple, Dict, Any, Union, List, Sequence
import psutil
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

_psutil_vm_fallback_logged = False


def _sysctl_hw_memsize_bytes() -> Optional[int]:
    """Physical RAM via sysctl (macOS fallback when psutil virtual_memory fails)."""
    if platform.system() != "Darwin":
        return None
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"],
            text=True,
            timeout=2,
        ).strip()
        return int(out)
    except Exception:
        return None


def _darwin_page_size_bytes() -> int:
    if platform.system() != "Darwin":
        return 4096
    try:
        out = subprocess.check_output(["pagesize"], text=True, timeout=2).strip()
        return max(4096, int(out))
    except Exception:
        return 4096


def _parse_vm_stat_page_count(vm_stat_output: str, label: str) -> int:
    prefix = f"{label}:"
    for line in vm_stat_output.splitlines():
        if not line.startswith(prefix):
            continue
        raw = line.split(":", 1)[1].strip().rstrip(".")
        try:
            return int(raw.replace(",", ""))
        except ValueError:
            return 0
    return 0


def _darwin_memory_stats_from_vm_stat() -> Optional[Tuple[int, int, float]]:
    """Installed RAM, available bytes, and percent used via sysctl + vm_stat."""
    if platform.system() != "Darwin":
        return None
    total = _sysctl_hw_memsize_bytes()
    if not total:
        return None
    try:
        vm_out = subprocess.check_output(["vm_stat"], text=True, timeout=2)
    except Exception:
        return None

    page_size = _darwin_page_size_bytes()
    free_pages = _parse_vm_stat_page_count(vm_out, "Pages free")
    inactive_pages = _parse_vm_stat_page_count(vm_out, "Pages inactive")
    speculative_pages = _parse_vm_stat_page_count(vm_out, "Pages speculative")
    available_pages = free_pages + inactive_pages + speculative_pages
    available = min(total, available_pages * page_size)
    used = max(0, total - available)
    percent = min(100.0, max(0.0, (used / total) * 100.0))
    return total, available, percent


def system_memory_total_bytes() -> Optional[int]:
    """Total installed RAM in bytes (sysctl on macOS, then psutil)."""
    if platform.system() == "Darwin":
        total = _sysctl_hw_memsize_bytes()
        if total:
            return int(total)
    try:
        return int(psutil.virtual_memory().total)
    except Exception:
        return _sysctl_hw_memsize_bytes()


def _safe_virtual_memory():
    """Memory stats with macOS sysctl/vm_stat first (avoids psutil HOST_VM_INFO64 issues)."""
    global _psutil_vm_fallback_logged
    if platform.system() == "Darwin":
        fallback = _darwin_memory_stats_from_vm_stat()
        if fallback is not None:
            from types import SimpleNamespace

            total, available, percent = fallback
            return SimpleNamespace(total=total, available=available, percent=percent)

    try:
        return psutil.virtual_memory()
    except Exception as exc:
        if not _psutil_vm_fallback_logged:
            logger.debug(
                "psutil.virtual_memory() unavailable (%s); using fallback memory stats",
                exc,
            )
            _psutil_vm_fallback_logged = True
        fallback = _darwin_memory_stats_from_vm_stat()
        from types import SimpleNamespace

        if fallback is not None:
            total, available, percent = fallback
            return SimpleNamespace(total=total, available=available, percent=percent)

        total = _sysctl_hw_memsize_bytes() or (8 * 1024**3)
        available = total // 4
        return SimpleNamespace(
            total=total,
            available=available,
            percent=min(100.0, max(0.0, ((total - available) / total) * 100.0)),
        )


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


def _cache_path_key(file_path: str) -> str:
    """Normalized path key for thumbnail/grid/preview caches (matches semantic canonical paths on Windows)."""
    return _exif_cache_path_key(file_path)


def _exif_sql_path_expr(column: str = "file_path") -> str:
    """SQL expression that normalizes slashes for path comparisons."""
    if os.sep == "\\":
        return f"lower(replace({column}, '/', '\\\\'))"
    return f"lower(replace({column}, '\\\\', '/'))"


def _exif_sql_folder_like_pattern(folder_path: str) -> str:
    """LIKE prefix for all files under folder_path (matches stored canonical keys)."""
    root = _exif_cache_path_key(os.path.abspath(folder_path)).rstrip("\\/")
    if os.sep == "\\":
        return root + "\\%"
    return root.replace("\\", "/") + "/%"


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

    def shrink_to_size(self, new_max_size: int) -> None:
        """Update max_size and remove excess items thread-safely."""
        with self.lock:
            self.max_size = new_max_size
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

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

    def peek(self, key: str) -> Optional[Any]:
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

    def put(self, file_path: str, value: Any, *args, **kwargs) -> bool:
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


class PersistentEXIFCache:
    """Persistent cache for EXIF data using SQLite."""

    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.rawviewer_cache")
        os.makedirs(cache_dir, exist_ok=True)

        self.db_path = os.path.join(cache_dir, "exif_cache.db")
        self.lock = threading.RLock()
        
        # Reads: thread-local connections. Writes: single writer connection (WAL).
        self._local = threading.local()
        self._writer_conn: Optional[sqlite3.Connection] = None
        self._pending_writes: List[tuple] = []
        self._commit_every = max(
            1,
            int(os.environ.get("RAWVIEWER_EXIF_CACHE_COMMIT_EVERY", "40")),
        )
        
        self._init_db()

    def _get_connection(self):
        """Get thread-local database connection (read-optimized)."""
        if not hasattr(self._local, 'conn'):
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            # Enable WAL mode for better concurrent performance
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-65536")
            self._local.conn = conn
        return self._local.conn

    def _get_writer_connection(self) -> sqlite3.Connection:
        """Dedicated writer connection for index/backfill batches (separate from readers)."""
        if self._writer_conn is None:
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0,
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-65536")
            self._writer_conn = conn
        return self._writer_conn

    def _prepare_put_row(
        self,
        file_path: str,
        exif_info: Dict[str, Any],
        *,
        file_size: Optional[int] = None,
        file_mtime: Optional[float] = None,
    ) -> Optional[tuple]:
        if file_size is None or file_mtime is None:
            file_size, file_mtime = self._get_file_hash(file_path)
        if file_size == 0:
            return None

        storage_path = _exif_cache_path_key(file_path)
        exif_blob = pickle.dumps(exif_info.get('exif_data', {}))

        capture_time = exif_info.get('capture_time')
        if not capture_time:
            exif_dict = exif_info.get('exif_data', {})
            if isinstance(exif_dict, dict):
                capture_time = exif_dict.get('capture_time')

        width = exif_info.get('original_width')
        height = exif_info.get('original_height')
        if not width or not height:
            exif_dict = exif_info.get('exif_data', {})
            if isinstance(exif_dict, dict):
                if not width:
                    width = exif_dict.get('original_width')
                if not height:
                    height = exif_dict.get('original_height')

        sensor_meta_ver = exif_info.get('raw_exif_sensor_meta_ver')
        if sensor_meta_ver is None:
            sensor_meta_ver = 0

        return (
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

    def _flush_pending_writes(self) -> None:
        if not self._pending_writes:
            return
        writer = self._get_writer_connection()
        writer.executemany(
            "INSERT OR REPLACE INTO exif_cache "
            "(file_path, file_size, file_mtime, orientation, camera_make, camera_model, exif_data, "
            "cached_time, capture_time, original_width, original_height, sensor_meta_ver) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            self._pending_writes,
        )
        writer.commit()
        self._pending_writes.clear()

    def flush(self) -> None:
        """Commit any batched EXIF writes (call after indexing batches)."""
        with self.lock:
            self._flush_pending_writes()

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
        """Get cached EXIF data if still valid.

        Deliberately NOT under self.lock: this reads via the calling
        thread's own thread-local connection (_get_connection()), and
        SQLite WAL mode already lets multiple readers proceed concurrently
        without blocking each other or a writer -- that's the entire reason
        get_multiple() has a connection-per-thread design in the first
        place. Wrapping this in the same process-wide lock used by
        put()/flush() (which do need it, for the shared writer connection
        and _pending_writes list) serialized every single-file EXIF read
        across every thread in the app -- including background semantic
        indexing's per-file extraction racing single-view navigation for
        the same lock, which measured as multi-second navigation stalls on
        real folders. timeout=30.0 on connection open already covers the
        rare genuine SQLite-level WAL contention case.
        """
        file_size, file_mtime = self._get_file_hash(file_path)
        if file_size == 0:
            return None
        cache_key = _exif_cache_path_key(file_path)

        try:
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT file_size, file_mtime, orientation, camera_make, camera_model, exif_data, "
                "capture_time, original_width, original_height, sensor_meta_ver "
                "FROM exif_cache WHERE file_path = ?",
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
                    # rating is stored inside the pickled exif_data blob (see
                    # _prepare_put_row/rate_current_image), not its own column --
                    # promote it to the top level so callers doing exif.get('rating')
                    # (the same shape put_exif() was called with) see it. Without
                    # this a rating survived to disk fine but read back as 0 on
                    # every app restart, since only the in-memory cache (which
                    # stores the dict as-passed) ever had the top-level key.
                    'rating': exif_data.get('rating', 0),
                }
        except Exception:
            pass

        return None

    def remove(self, file_path: str) -> bool:
        """Remove cached EXIF data for a file."""
        cache_key = _exif_cache_path_key(file_path)
        with self.lock:
            try:
                self._flush_pending_writes()
                writer = self._get_writer_connection()
                cursor = writer.execute(
                    "DELETE FROM exif_cache WHERE file_path = ?",
                    (cache_key,),
                )
                writer.commit()
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
        commit: bool = False,
    ) -> None:
        """Cache EXIF data. Pass file_size/file_mtime when already known (avoids stat under lock)."""
        row = self._prepare_put_row(
            file_path, exif_info, file_size=file_size, file_mtime=file_mtime
        )
        if row is None:
            return

        with self.lock:
            try:
                self._pending_writes.append(row)
                if commit or len(self._pending_writes) >= self._commit_every:
                    self._flush_pending_writes()
            except Exception:
                pass

    def put_many(
        self,
        items: Sequence[tuple[str, Dict[str, Any], Optional[int], Optional[float]]],
        *,
        commit: bool = True,
    ) -> None:
        """Batch EXIF writes (file_path, exif_info, file_size, file_mtime)."""
        rows: List[tuple] = []
        for file_path, exif_info, file_size, file_mtime in items:
            row = self._prepare_put_row(
                file_path,
                exif_info,
                file_size=file_size,
                file_mtime=file_mtime,
            )
            if row is not None:
                rows.append(row)
        if not rows:
            return
        with self.lock:
            try:
                self._pending_writes.extend(rows)
                if commit or len(self._pending_writes) >= self._commit_every:
                    self._flush_pending_writes()
            except Exception:
                pass

    def cleanup_old_entries(self, max_age_days: int = 30) -> None:
        """Remove cache entries older than specified days."""
        cutoff_time = time.time() - (max_age_days * 24 * 3600)

        with self.lock:
            try:
                self._flush_pending_writes()
                writer = self._get_writer_connection()
                writer.execute(
                    "DELETE FROM exif_cache WHERE cached_time < ?", (cutoff_time,)
                )
                writer.commit()
            except Exception:
                pass

    def clear(self) -> None:
        """Remove all cached EXIF entries."""
        with self.lock:
            try:
                self._pending_writes.clear()
                writer = self._get_writer_connection()
                writer.execute("DELETE FROM exif_cache")
                writer.commit()
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
            # sqlite has a limit on the number of variables in a query.
            # Process in chunks of 450; no lock needed here (see get()'s
            # docstring) -- this is a read-only query on the calling
            # thread's own connection, and SQLite WAL mode already permits
            # concurrent readers.
            for i in range(0, len(file_paths), 450):
                chunk = file_paths[i : i + 450]
                lookup: Dict[str, str] = {}
                norm_keys: List[str] = []
                for p in chunk:
                    nk = _exif_cache_path_key(p)
                    lookup[nk] = p
                    norm_keys.append(nk)
                placeholders = ",".join(["?"] * len(norm_keys))

                query = (
                    f"SELECT file_path, file_size, file_mtime, orientation, camera_make, camera_model, "
                    f"exif_data, capture_time, original_width, original_height, sensor_meta_ver "
                    f"FROM exif_cache WHERE file_path IN ({placeholders})"
                )
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
                        # See get()'s 'rating' comment -- only recoverable when the
                        # blob was actually unpickled (not the fast_mode/has_fast_data
                        # shortcut above, which skips it for bulk-scan speed).
                        "rating": exif_data.get("rating", 0) if exif_data else 0,
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
                    f"WHERE file_path IN ({placeholders})"
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

    def update_capture_time_bulk(self, timestamps: Dict[str, str]) -> None:
        """Persist probed capture_time values without touching any other cached
        EXIF field. UPDATE-only for rows that already have one (never overwrites
        an existing value); a bare row is INSERTed only for paths with no row at
        all yet, so a later full EXIF extraction's INSERT OR REPLACE still wins.
        This lets the sort-time capture-time probe (main.py's
        _parallel_probe_capture_times, which uses a cheap exifread-only read, not
        a RAW decode) survive to the next folder open instead of re-probing every
        uncached file every time — without any risk of clobbering width/height/
        orientation data a row might hold under a stale sensor_meta_ver.
        """
        if not timestamps:
            return
        rows = [
            (str(ct), _exif_cache_path_key(p)) for p, ct in timestamps.items() if ct
        ]
        if not rows:
            return
        try:
            writer = self._get_writer_connection()
            with self.lock:
                for i in range(0, len(rows), 450):
                    chunk = rows[i : i + 450]
                    writer.executemany(
                        "UPDATE exif_cache SET capture_time = ? "
                        "WHERE file_path = ? AND (capture_time IS NULL OR capture_time = '')",
                        chunk,
                    )
                    path_keys = [pk for _, pk in chunk]
                    placeholders = ",".join(["?"] * len(path_keys))
                    cursor = writer.execute(
                        f"SELECT file_path FROM exif_cache WHERE file_path IN ({placeholders})",
                        path_keys,
                    )
                    existing = {row[0] for row in cursor.fetchall()}
                    insert_rows = [
                        (pk, None, None, 1, "", "", pickle.dumps({}), time.time(), ct, None, None, 0)
                        for ct, pk in chunk
                        if pk not in existing
                    ]
                    if insert_rows:
                        writer.executemany(
                            "INSERT OR IGNORE INTO exif_cache "
                            "(file_path, file_size, file_mtime, orientation, camera_make, camera_model, "
                            "exif_data, cached_time, capture_time, original_width, original_height, sensor_meta_ver) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            insert_rows,
                        )
                writer.commit()
        except Exception:
            pass

    def get_capture_times_for_folder(self, folder_path: str) -> Dict[str, str]:
        """All cached capture_time entries under folder_path (slash-agnostic LIKE)."""
        if not folder_path:
            return {}
        patterns = _exif_sql_folder_like_patterns(folder_path)
        if not patterns:
            return {}
        out: Dict[str, str] = {}
        try:
            conn = self._get_connection()
            for pattern in patterns:
                with self.lock:
                    rows = conn.execute(
                        "SELECT file_path, capture_time FROM exif_cache "
                        "WHERE file_path LIKE ? "
                        "AND capture_time IS NOT NULL AND capture_time != ''",
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
                            "SELECT file_path, exif_data FROM exif_cache "
                            "WHERE file_path LIKE ? AND exif_data IS NOT NULL",
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
        """Close the database connection for the current thread.

        Flushes batched writes first -- without this, any EXIF update that
        didn't happen to hit the 40-write commit threshold (e.g. a single
        star rating set near the end of a session) sat in _pending_writes
        and was silently dropped on quit, never reaching disk.
        """
        try:
            self.flush()
        except Exception:
            pass
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
        # Use hash of normalized path so case variants share one disk entry on Windows.
        path_hash = hashlib.md5(_cache_path_key(file_path).encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"{path_hash}.jpg")
    
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
                cache_file = row[2]
                return bool(cache_file) and os.path.exists(cache_file)
        except Exception:
            pass
        return False

    def get(self, file_path: str) -> Optional[bytes]:
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
                    cache_file = row[0]
                    # Delete cache file
                    try:
                        if os.path.exists(cache_file):
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
                        total += os.path.getsize(cache_file)
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
                evicted_paths: List[str] = []
                for file_path, cache_file in rows:
                    if freed >= bytes_to_free:
                        break
                    try:
                        freed += os.path.getsize(cache_file)
                    except OSError:
                        pass
                    try:
                        if os.path.exists(cache_file):
                            os.remove(cache_file)
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



def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, "").strip()
    if raw:
        try:
            return max(minimum, min(int(raw), maximum))
        except ValueError:
            pass
    return max(minimum, min(default, maximum))


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name, "").strip()
    if raw:
        try:
            return max(minimum, min(float(raw), maximum))
        except ValueError:
            pass
    return max(minimum, min(default, maximum))


def _preview_cache_adaptive_enabled() -> bool:
    raw = os.environ.get("RAWVIEWER_PREVIEW_CACHE_ADAPTIVE", "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


def recommended_preview_cache_items(available_gb: float | None = None) -> int:
    """Preview LRU capacity from free RAM, preview resolution, and build profile."""
    explicit = os.environ.get("RAWVIEWER_PREVIEW_CACHE_ITEMS", "").strip()
    if explicit:
        return _env_int("RAWVIEWER_PREVIEW_CACHE_ITEMS", 10, minimum=6, maximum=64)

    if not _preview_cache_adaptive_enabled():
        return 10

    if available_gb is None:
        try:
            available_gb = float(_safe_virtual_memory().available) / (1024**3)
        except Exception:
            return 10

    edge = memory_preview_max_edge()
    preview_mb = max(6.0, (edge / 1920.0) ** 2 * 12.0)

    budget_fraction = 0.08
    budget_cap_mb = 1200.0
    try:
        from rawviewer_profile import indexing_loads_compete, is_lite_build

        if not indexing_loads_compete():
            budget_fraction = 0.12
            budget_cap_mb = 1600.0
        if is_lite_build():
            budget_fraction = max(budget_fraction, 0.14)
            budget_cap_mb = 1800.0
    except Exception:
        pass

    budget_mb = min(float(available_gb) * 1024.0 * budget_fraction, budget_cap_mb)
    items = int(budget_mb / preview_mb)

    if available_gb >= 24:
        floor = 20
    elif available_gb >= 16:
        floor = 16
    elif available_gb >= 10:
        floor = 14
    elif available_gb >= 6:
        floor = 12
    elif available_gb >= 3:
        floor = 8
    else:
        floor = 6

    min_items = _env_int("RAWVIEWER_PREVIEW_CACHE_ITEMS_MIN", 6, minimum=4, maximum=32)
    max_items = _env_int("RAWVIEWER_PREVIEW_CACHE_ITEMS_MAX", 64, minimum=8, maximum=128)
    return max(min_items, min(max(floor, items), max_items))


class MemoryMonitor:
    """Monitor system memory usage and provide cache sizing recommendations."""

    def __init__(self):
        self.process = psutil.Process()

    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information."""
        system_memory = _safe_virtual_memory()
        try:
            process_memory = self.process.memory_info()
        except Exception as exc:
            logger.warning("process.memory_info() failed (%s)", exc)
            from types import SimpleNamespace

            process_memory = SimpleNamespace(rss=0, vms=0)

        return {
            'system_total_gb': system_memory.total / (1024**3),
            'system_available_gb': system_memory.available / (1024**3),
            'system_percent_used': system_memory.percent,
            'process_rss_mb': process_memory.rss / (1024**2),
            'process_vms_mb': process_memory.vms / (1024**2)
        }

    def system_memory_pressure_level(
        self, memory_info: Dict[str, float] | None = None
    ) -> str:
        """Classify system RAM pressure ('critical', 'elevated', 'ok').

        Uses system-wide percent used and available GB only — not process RSS.
        Large semantic-model footprints are normal on high-RAM machines and must
        not trigger cache purges while the OS still has headroom.
        """
        if memory_info is None:
            memory_info = self.get_memory_info()
        percent_used = memory_info['system_percent_used']
        available_gb = memory_info['system_available_gb']

        crit_pct = _env_float(
            "RAWVIEWER_MEMORY_PRESSURE_PERCENT_CRITICAL", 90.0, minimum=50.0, maximum=99.0
        )
        crit_avail_gb = _env_float(
            "RAWVIEWER_MEMORY_PRESSURE_AVAILABLE_GB_CRITICAL", 2.0, minimum=0.25, maximum=32.0
        )
        elev_pct = _env_float(
            "RAWVIEWER_MEMORY_PRESSURE_PERCENT_ELEVATED", 85.0, minimum=50.0, maximum=98.0
        )
        elev_avail_gb = _env_float(
            "RAWVIEWER_MEMORY_PRESSURE_AVAILABLE_GB_ELEVATED", 4.0, minimum=0.5, maximum=64.0
        )

        if percent_used >= crit_pct or available_gb <= crit_avail_gb:
            return 'critical'
        if percent_used >= elev_pct or available_gb <= elev_avail_gb:
            return 'elevated'
        return 'ok'

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

        preview_items = recommended_preview_cache_items(available_gb)

        return {
            # Opt-in override for zoomed-culling workflows: retaining more
            # sensor-res buffers makes zoomed A<->B revisits instant, at
            # ~100-200MB per slot. Default stays conservative for 8GB machines.
            'full_images': _env_int(
                "RAWVIEWER_FULL_IMAGE_CACHE_ITEMS",
                min(max_full_images, 8),
                minimum=2,
                maximum=32,
            ),
            'thumbnails': min(max_thumbnails, 1000),
            'preview_images': min(preview_items, 16),
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
        self.preview_cache = LRUCache(
            max_size=cache_sizes['preview_images']
        )  # High-res preview working set (RAM-adaptive)
        self.full_image_cache = LRUCache(max_size=cache_sizes['full_images'])
        self.pixmap_cache = LRUCache(max_size=cache_sizes['full_images'])
        self.libraw_preview_paths: set = set()
        if self.cache_dir:
            self._libraw_preview_paths_file = os.path.join(self.cache_dir, "libraw_preview_paths.json")
            self._load_libraw_preview_paths()
        else:
            self._libraw_preview_paths_file = None
        # Session-scoped provenance: full_image_cache keys currently holding an
        # embedded-JPEG stand-in rather than the app's own RAW-decoded pixels
        # (see mark_full_image_source). Many camera bodies (Canon CR3, Sony
        # ARW, ...) embed a JPEG preview whose pixel dimensions reach/exceed
        # FULL_EMBEDDED_SENSOR_COVERAGE of the true sensor resolution -- by
        # dimensions alone that stand-in is indistinguishable from a true
        # decode, which previously caused every "does this already cover
        # sensor resolution" check (the fit-view display, the cached-full
        # check, and process_full_image's own cache short-circuit) to
        # conclude no further work was needed, permanently blocking the real
        # RAW pipeline from ever running for that file during normal browsing.
        self.full_image_embedded_jpeg_paths: set = set()
        if self.persistent_cache_enabled:
            self.exif_cache = PersistentEXIFCache(cache_dir)
            self.disk_thumbnail_cache = PersistentThumbnailCache(cache_dir)
            self.disk_grid_cache = PersistentGridCache(cache_dir)
            self.disk_preview_cache = PersistentPreviewCache(cache_dir)
            self._purge_stale_oriented_pixel_caches(cache_dir)
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
        self.memory_check_interval = 5  # seconds
        self.last_memory_check = 0
        self.last_aggressive_purge = 0.0
        self.last_reduce_cache = 0.0

        # Background memory-pressure polling: get_memory_info() spawns a
        # `vm_stat` subprocess on macOS (see _darwin_memory_stats_from_vm_stat),
        # tens of ms per call. _check_memory_pressure() used to call it inline
        # on whatever thread touched the cache -- during gallery scrolling
        # that's the main/GUI thread, so every memory_check_interval seconds
        # of active scrolling paid a synchronous subprocess-spawn cost right
        # in the scroll path. A lazily-started daemon thread now refreshes a
        # cached snapshot on that same interval; _check_memory_pressure only
        # ever reads it.
        self._mem_poll_lock = threading.Lock()
        self._mem_poll_info: Optional[Dict[str, float]] = None
        self._mem_poll_pressure = 'ok'
        self._mem_poll_thread: Optional[threading.Thread] = None

        # Background RAW-orientation verification (see get_exif): one short
        # queue drained by a single lazily-started daemon thread, one attempt
        # per path per session.
        self._orient_verify_lock = threading.Lock()
        self._orient_verify_attempted: set = set()
        # Dedup-only set (never holds a value, just "already tried"), so
        # capping it is cheap to reason about: a path evicted here that
        # comes up again just gets re-queued for one more probe, and a
        # record that already probed clean short-circuits before ever
        # reaching this set again (get_exif/get_multiple_exif only call
        # _schedule_orientation_verify when verified_orientation is falsy).
        # Uncapped, a very large library browsed in one long session grows
        # this set forever; same clear-when-full convention as
        # enhanced_raw_processor._embedded_scan_miss_cache.
        self._orient_verify_attempted_max = 4096
        self._orient_verify_queue: list = []
        self._orient_verify_thread: Optional[threading.Thread] = None
        self._closing = False

    def _purge_stale_oriented_pixel_caches(self, cache_dir: str) -> None:
        """One-time purge of persisted PIXEL caches when the orientation logic version bumps.

        RAW_EXIF_SENSOR_META_VER invalidates cached EXIF ROWS, but the disk
        thumbnail/grid/preview JPEGs persist the decoded PIXELS. A version with buggy
        rotation can write content-rotated pixels at plausible dimensions — invisible to
        every dimension-based self-heal (they compare width/height, not content). Stamp
        the orientation version in the cache dir and drop all persisted pixel tiers when
        it changes, so upgraded users never see stale sideways thumbnails.
        """
        try:
            from enhanced_raw_processor import RAW_EXIF_SENSOR_META_VER

            marker = os.path.join(cache_dir, "orient_pixel_ver.txt")
            stored = None
            try:
                with open(marker, "r", encoding="utf-8") as fh:
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
                try:
                    cache.clear()
                except Exception:
                    pass
            with open(marker, "w", encoding="utf-8") as fh:
                fh.write(str(int(RAW_EXIF_SENSOR_META_VER)))
        except Exception:
            pass

    @staticmethod
    def _path_key(file_path: str) -> str:
        return _cache_path_key(file_path)

    def _ensure_mem_poll_thread(self) -> None:
        """Start the background memory-pressure poller if it isn't already running."""
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
        """Refresh the cached memory-info/pressure snapshot every
        memory_check_interval seconds, off the main/calling thread.
        """
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
        """Check if we're under memory pressure and need to reduce cache sizes.

        Reads the snapshot the background poller last collected rather than
        collecting it inline -- get_memory_info() spawns a `vm_stat`
        subprocess on macOS, and this is called from cache-read paths
        (get_grid/get_thumbnail/etc.) that run on whatever thread touches
        the cache, including the main/GUI thread during gallery scrolling.
        """
        current_time = time.time()
        if current_time - self.last_memory_check < self.memory_check_interval:
            return False
        self.last_memory_check = current_time
        self._ensure_mem_poll_thread()

        with self._mem_poll_lock:
            memory_info = self._mem_poll_info
            pressure_level = self._mem_poll_pressure

        if memory_info is None:
            # No snapshot yet (first check(s) after startup) -- the poller
            # just started; nothing to act on until it reports back.
            return False

        percent_used = memory_info['system_percent_used']
        available_gb = memory_info['system_available_gb']
        rss_mb = memory_info['process_rss_mb']

        if pressure_level == 'critical':
            if current_time - getattr(self, 'last_aggressive_purge', 0.0) < 60.0:
                # Under cooldown to prevent app freeze / infinite loop of purging
                return False
            self.last_aggressive_purge = current_time
            logger.warning(
                "CRITICAL system memory pressure (system: %.1f%% used, %.1f GB free, process RSS: %.1f MB). Aggressively purging caches.",
                percent_used,
                available_gb,
                rss_mb,
            )
            self.memory_warning.emit(percent_used)
            self._aggressive_purge()
            return True

        if pressure_level == 'elevated':
            if current_time - getattr(self, 'last_reduce_cache', 0.0) < 20.0:
                # Under cooldown
                return False
            self.last_reduce_cache = current_time
            logger.warning(
                "Elevated system memory pressure (system: %.1f%% used, %.1f GB free, process RSS: %.1f MB). Reducing cache sizes.",
                percent_used,
                available_gb,
                rss_mb,
            )
            self.memory_warning.emit(percent_used)
            self._reduce_cache_sizes()
            return True

        return False

    def _aggressive_purge(self) -> None:
        """Aggressively clear heavy caches and drop their capacities to a minimum under critical pressure."""
        logger.info("Performing aggressive cache purge...")
        
        # Clear heavy caches entirely
        self.full_image_cache.clear()
        self.pixmap_cache.clear()
        
        # Aggressively drop capacities to 3 thread-safely
        self.full_image_cache.shrink_to_size(3)
        self.pixmap_cache.shrink_to_size(3)
        
        # Reduce and clear other caches aggressively thread-safely
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
        """Reduce cache sizes when under memory pressure."""
        # Reduce by 25%
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

        # Clear excess items thread-safely and update max sizes
        self.full_image_cache.shrink_to_size(new_full_size)
        self.pixmap_cache.shrink_to_size(new_pixmap_size)
        self.thumbnail_cache.shrink_to_size(new_thumbnail_size)
        self.grid_cache.shrink_to_size(new_grid_size)
        self.preview_cache.shrink_to_size(new_preview_size)

        self._release_gpu_memory()

    def _release_gpu_memory(self) -> None:
        """Return the GPU/unified-memory caching allocator's pool to the OS.

        On Apple Silicon, PyTorch's MPS allocator caches freed device buffers
        in UNIFIED memory (shared with system RAM) rather than releasing them
        -- that pool is invisible to gc/tracemalloc but shows up directly as
        process RSS, and nothing in this codebase ever called
        torch.mps.empty_cache() before this. v2.5 had no GPU decode path at
        all and doesn't have this cost; this build's GPU-accelerated decode
        (gpu_raw_processor.py) does. Piggyback on the existing cache-pressure
        cadence rather than calling it every decode (that would defeat the
        allocator's whole point -- reuse without re-requesting from the OS).
        """
        try:
            from gpu_raw_processor import release_cached_gpu_memory

            release_cached_gpu_memory()
        except Exception:
            logger.debug("GPU memory release failed", exc_info=True)

    def _heal_raw_orientation(self, file_path: str, arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Self-heal: return arr already in EXIF display orientation, re-orienting if not.

        Every cache tier (thumbnail, grid, preview) must return display-oriented pixels so
        the gallery/single-view never render a stale, wrongly-rotated buffer. Uses the same
        make-independent authority as extraction (finalize_index_thumbnail_array), which is
        idempotent, so calling this on already-correct pixels is a no-op.
        """
        if arr is None:
            return None
        try:
            from common_image_loader import (
                finalize_index_thumbnail_array,
                index_thumbnail_needs_orient_fix,
                is_raw_file,
            )

            if not is_raw_file(file_path):
                return arr
            if not index_thumbnail_needs_orient_fix(file_path, arr, cache=self):
                return arr
            repaired = finalize_index_thumbnail_array(file_path, arr, cache=self)
            return repaired if repaired is not None else arr
        except Exception:
            return arr

    def get_thumbnail(self, file_path: str) -> Optional[np.ndarray]:
        """Get cached thumbnail or return None if not cached."""
        self.stats['thumbnail_requests'] += 1
        self._check_memory_pressure()

        key = self._path_key(file_path)

        def _validate_raw_thumbnail(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
            return self._heal_raw_orientation(file_path, arr)

        # Check in-memory cache first
        thumbnail = self.thumbnail_cache.get(key)
        if thumbnail is not None:
            validated = _validate_raw_thumbnail(thumbnail)
            if validated is not thumbnail:
                self.thumbnail_cache.put(key, validated.copy())
                self.disk_thumbnail_cache.remove(file_path)
            self.cache_hit.emit(file_path, 'thumbnail')
            return validated
        
        # Check disk cache for JPEG thumbnails
        jpeg_data = self.disk_thumbnail_cache.get(file_path)
        if jpeg_data is not None:
            try:
                from common_image_loader import decode_embedded_jpeg_bytes

                thumbnail = decode_embedded_jpeg_bytes(jpeg_data, max_size=0)
                if thumbnail is None:
                    raise ValueError("decode_embedded_jpeg_bytes failed")
                thumbnail = _validate_raw_thumbnail(thumbnail)
                if thumbnail is None:
                    raise ValueError("thumbnail orientation validation failed")
                # Also cache in memory for faster subsequent access
                self.thumbnail_cache.put(key, thumbnail.copy())
                self.cache_hit.emit(file_path, 'thumbnail')
                return thumbnail
            except Exception:
                # If loading from disk cache fails, remove it
                self.disk_thumbnail_cache.remove(file_path)
        
        # --- Dynamic Mipmap Fallback ---
        # 1. Downsample from Grid cache (512px) if available
        grid = self.grid_cache.get(key)
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
                from common_image_loader import encode_tile_bytes

                t_jpeg = encode_tile_bytes(thumb_pil)
                self.put_thumbnail(file_path, thumbnail, t_jpeg)
                return thumbnail
            except Exception:
                pass

        # 2. Downsample from preview cache (memory or disk, up to ~512px) if available
        preview = self.preview_cache.get(key)
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
                from common_image_loader import encode_tile_bytes

                t_jpeg = encode_tile_bytes(thumb_pil)
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
            key = self._path_key(file_path)
            # Cache in memory
            self.thumbnail_cache.put(key, thumbnail.copy())
            
            # If JPEG data is provided, also cache to disk
            if jpeg_data is not None:
                self.disk_thumbnail_cache.put(file_path, jpeg_data)

    def get_preview_memory_only(self, file_path: str) -> Optional[np.ndarray]:
        """In-memory preview only -- never decodes disk tier on the calling thread."""
        key = self._path_key(file_path)
        return self.preview_cache.peek(key)

    def get_grid_memory_only(self, file_path: str) -> Optional[np.ndarray]:
        """In-memory grid only -- never decodes disk tier on the calling thread."""
        key = self._path_key(file_path)
        return self.grid_cache.peek(key)

    def get_thumbnail_memory_only(self, file_path: str) -> Optional[np.ndarray]:
        """In-memory thumbnail only -- never decodes disk tier on the calling thread."""
        key = self._path_key(file_path)
        return self.thumbnail_cache.peek(key)

    def has_warm_thumbnail_in_memory(self, file_path: str) -> bool:
        """True if a grid/thumbnail tier is already in the fast in-memory
        cache -- no disk read, no decode, unlike get_grid()/get_thumbnail().

        Those two fall through to a disk-cache read + synchronous PIL decode
        + EXIF transpose on an in-memory miss. Callers that only want to know
        "is this warm, should I skip prefetching it" (filmstrip prefetch scan
        over dozens of neighbor files) were using get_grid()/get_thumbnail()
        for that check and paying the decode cost as a side effect -- on a
        long session, once the small in-memory tier evicts entries the disk
        tier still has, every such "check" becomes a real synchronous decode.
        Measured via faulthandler stack dumps during a 250+ navigation
        stress test: multi-second main-thread stalls with the current frame
        inside PIL's WebP/JPEG decode, called from exactly this "just
        checking" path.
        """
        key = self._path_key(file_path)
        # peek(): a warmth CHECK must not promote never-displayed entries to MRU.
        return (
            self.grid_cache.peek(key) is not None
            or self.thumbnail_cache.peek(key) is not None
        )

    def get_grid(self, file_path: str) -> Optional[np.ndarray]:
        """Get cached grid image (max 512px) or return None if not cached."""
        if 'grid_requests' not in self.stats:
            self.stats['grid_requests'] = 0
        self.stats['grid_requests'] += 1
        self._check_memory_pressure()

        key = self._path_key(file_path)
        # 1. Check in-memory grid cache
        grid = self.grid_cache.get(key)
        if grid is not None:
            healed = self._heal_raw_orientation(file_path, grid)
            if healed is not grid:
                self.grid_cache.put(key, healed.copy())
                self.disk_grid_cache.remove(file_path)
            self.cache_hit.emit(file_path, 'grid')
            return healed

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
                grid = self._heal_raw_orientation(file_path, grid)
                self.grid_cache.put(key, grid.copy())
                self.cache_hit.emit(file_path, 'grid')
                return grid
            except Exception:
                self.disk_grid_cache.remove(file_path)

        # 3. Dynamic Mipmap Fallback 1: Downsample from preview tier
        preview = self.preview_cache.get(key)
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
                from common_image_loader import encode_tile_bytes

                g_jpeg = encode_tile_bytes(grid_pil)
                self.put_grid(file_path, grid, g_jpeg)
                return grid
            except Exception:
                pass

        # 4. Dynamic Mipmap Fallback 2: Upsample from Thumbnail (256px)
        thumb = self.thumbnail_cache.get(key)
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
                self.grid_cache.put(key, grid.copy())
                return grid
            except Exception:
                pass

        self.cache_miss.emit(file_path, 'grid')
        return None

    def put_grid(self, file_path: str, grid: np.ndarray, jpeg_data: bytes = None) -> None:
        """Cache a grid image (max 512px)."""
        if grid is not None:
            key = self._path_key(file_path)
            # Ensure grid image is reasonable size (max 512x512)
            if grid.shape[0] > 512 or grid.shape[1] > 512:
                return
            # Cache in memory
            self.grid_cache.put(key, grid.copy())
            
            # Cache on disk
            if self.persistent_cache_enabled:
                if jpeg_data is not None:
                    self.disk_grid_cache.put(file_path, jpeg_data)
                else:
                    try:
                        from PIL import Image
                        from common_image_loader import encode_tile_bytes

                        if grid.dtype != np.uint8:
                            grid = grid.astype(np.uint8)
                        pil_image = Image.fromarray(grid)
                        self.disk_grid_cache.put(file_path, encode_tile_bytes(pil_image))
                    except Exception:
                        pass

    def get_preview(self, file_path: str) -> Optional[np.ndarray]:
        """Get cached preview (screen size) or return None."""
        self.stats['preview_requests'] += 1
        self._check_memory_pressure()

        key = self._path_key(file_path)
        # Check in-memory cache first
        preview = self.preview_cache.get(key)
        if preview is not None:
            self.cache_hit.emit(file_path, 'preview')
            return preview

        # Check disk cache for previews
        jpeg_data = self.disk_preview_cache.get(file_path)
        if jpeg_data is None:
            jpeg_data = self.disk_grid_cache.get(file_path)
        if jpeg_data is not None:
             try:
                # cv2.imdecode over PIL: several times faster on the
                # 1280-1536px entries here, and this path runs on the
                # CALLING thread -- navigation paints call get_preview()
                # from the UI thread, so every ms is a paint stall. EXIF
                # transpose is intentionally absent: disk preview entries
                # are re-encoded from already-oriented pixels with EXIF
                # stripped (_jpeg_bytes_max_edge), and grid tiles are
                # encoded from oriented arrays.
                import cv2

                arr = cv2.imdecode(
                    np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR
                )
                if arr is None:
                    raise ValueError("imdecode failed")
                preview = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                # Also cache in memory
                self.preview_cache.put(key, preview)
                self.cache_hit.emit(file_path, 'preview')
                return preview
             except Exception:
                self.disk_preview_cache.remove(file_path)
        
        self.cache_miss.emit(file_path, 'preview')
        return None

    def put_preview(
        self,
        file_path: str,
        preview: np.ndarray,
        jpeg_data: bytes = None,
        *,
        libraw_rendered: bool = False,
    ) -> None:
        """Cache a preview image (memory); optional disk JPEG clamped to disk_preview_max_edge().

        ``libraw_rendered`` marks the preview as LibRaw-pipeline pixels (vs
        embedded-JPEG-derived) so the RAW workflow can refuse to paint
        camera-JPEG interim tiers (session-scoped provenance).
        """
        if preview is not None:
            key = self._path_key(file_path)
            self.preview_cache.put(key, preview.copy())
            if libraw_rendered:
                self.libraw_preview_paths.add(key)
                self._prune_libraw_preview_paths()
                self._save_libraw_preview_paths()
            else:
                if key in self.libraw_preview_paths:
                    self.libraw_preview_paths.discard(key)
                    self._save_libraw_preview_paths()
            if jpeg_data is not None:
                cap = disk_preview_max_edge()
                jpeg_data = _jpeg_bytes_max_edge(jpeg_data, cap)
                self.disk_preview_cache.put(file_path, jpeg_data)
            self._enforce_numpy_cache_byte_budget(
                self.preview_cache,
                fraction=0.25,
                min_keep=2,
                label="preview",
            )

    def is_libraw_preview(self, file_path: str) -> bool:
        """True when the cached preview holds LibRaw-rendered pixels (this session)."""
        return self._path_key(file_path) in self.libraw_preview_paths

    def _prune_libraw_preview_paths(self, max_paths: int = 2048) -> None:
        """Bound session provenance set (paths only)."""
        n = len(self.libraw_preview_paths)
        if n <= max_paths:
            return
        # Discard arbitrary surplus (set has no order); prefer keeping keys still in preview.
        excess = n - max_paths
        drop = []
        for k in self.libraw_preview_paths:
            if self.preview_cache.peek(k) is None:
                drop.append(k)
            if len(drop) >= excess:
                break
        for k in drop:
            self.libraw_preview_paths.discard(k)
        while len(self.libraw_preview_paths) > max_paths:
            self.libraw_preview_paths.pop()

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
            img_copy = image.copy()
            self.full_image_cache.put(file_path, img_copy)
            self._enforce_numpy_cache_byte_budget(
                self.full_image_cache,
                fraction=0.55,
                min_keep=2,
                label="full-image",
            )

    def mark_full_image_source(self, file_path: str, *, is_embedded_jpeg: bool) -> None:
        """Track whether the cached full_image entry is an embedded-JPEG
        stand-in rather than the app's own RAW-decoded pixels -- see
        full_image_embedded_jpeg_paths. Call alongside every put_full_image()
        so the flag never goes stale: mark True from the embedded-preview
        shortcut, False once a true decode overwrites that same key.
        """
        if not file_path:
            return
        key = self._path_key(file_path)
        if is_embedded_jpeg:
            self.full_image_embedded_jpeg_paths.add(key)
            self._prune_full_image_embedded_jpeg_paths()
        else:
            self.full_image_embedded_jpeg_paths.discard(key)

    def full_image_is_embedded_jpeg(self, file_path: str) -> bool:
        """True when the currently cached full_image for this path is an
        embedded-JPEG stand-in (see mark_full_image_source)."""
        if not file_path:
            return False
        return self._path_key(file_path) in self.full_image_embedded_jpeg_paths

    def _prune_full_image_embedded_jpeg_paths(self, max_paths: int = 2048) -> None:
        """Bound session provenance set (paths only); same shape as
        _prune_libraw_preview_paths."""
        n = len(self.full_image_embedded_jpeg_paths)
        if n <= max_paths:
            return
        excess = n - max_paths
        drop = []
        for k in self.full_image_embedded_jpeg_paths:
            if self.full_image_cache.peek(k) is None:
                drop.append(k)
            if len(drop) >= excess:
                break
        for k in drop:
            self.full_image_embedded_jpeg_paths.discard(k)
        while len(self.full_image_embedded_jpeg_paths) > max_paths:
            self.full_image_embedded_jpeg_paths.pop()

    def _estimate_entry_bytes(self, value: Any) -> int:
        try:
            if hasattr(value, "nbytes"):
                return int(value.nbytes)
            # QPixmap / QImage
            if hasattr(value, "width") and hasattr(value, "height") and callable(value.width):
                w, h = int(value.width()), int(value.height())
                # Assume up to 4 bytes/pixel (RGBA) for budget purposes.
                return max(0, w * h * 4)
        except Exception:
            pass
        return 0

    def _enforce_numpy_cache_byte_budget(
        self,
        lru: "LRUCache",
        *,
        fraction: float,
        min_keep: int = 2,
        label: str = "cache",
    ) -> None:
        """Evict LRU entries until sum(nbytes) fits fraction of max_memory_mb."""
        try:
            budget_bytes = float(self.max_memory_mb) * 1024 * 1024 * float(fraction)
            cache = lru.cache
            with lru.lock:
                total = sum(self._estimate_entry_bytes(v) for v in cache.values())
                if total <= budget_bytes:
                    return
                evicted = 0
                while total > budget_bytes and len(cache) > max(1, min_keep):
                    _, oldest = cache.popitem(last=False)
                    total -= self._estimate_entry_bytes(oldest)
                    evicted += 1
            if evicted:
                logger.info(
                    "%s cache: evicted %d oldest (budget ~%.0fMB, now ~%.0fMB, %d left)",
                    label,
                    evicted,
                    budget_bytes / 1e6,
                    total / 1e6,
                    len(cache),
                )
        except Exception:
            logger.warning("%s byte-budget enforcement failed", label, exc_info=True)

    def _enforce_full_image_byte_budget(self) -> None:
        """Backward-compatible alias for full-image RAM budget."""
        self._enforce_numpy_cache_byte_budget(
            self.full_image_cache,
            fraction=0.55,
            min_keep=2,
            label="full-image",
        )

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
            self._enforce_numpy_cache_byte_budget(
                self.pixmap_cache,
                fraction=0.20,
                min_keep=2,
                label="pixmap",
            )

    def clear_heavy_memory_tiers(self, keep_path: Optional[str] = None) -> None:
        """Drop full/preview/pixmap RAM tiers (keep grid/thumb for gallery scroll).

        Call on folder change. Optional keep_path retains that file's heavy entries
        when still navigating within open flow.
        """
        keep_key = self._path_key(keep_path) if keep_path else None
        keep_full = None
        keep_preview = None
        keep_pixmap = None
        keep_full_is_embedded_jpeg = False
        if keep_path:
            keep_full = self.full_image_cache.get(keep_path)
            keep_preview = self.preview_cache.get(keep_key)
            keep_pixmap = self.pixmap_cache.get(keep_path)
            keep_full_is_embedded_jpeg = self.full_image_is_embedded_jpeg(keep_path)

        self.full_image_cache.clear()
        self.preview_cache.clear()
        self.pixmap_cache.clear()
        self.libraw_preview_paths.clear()
        self.full_image_embedded_jpeg_paths.clear()

        if keep_path and keep_full is not None:
            self.full_image_cache.put(keep_path, keep_full)
            # Preserve provenance across the clear -- otherwise a kept
            # embedded-JPEG stand-in would look like a true decode afterward
            # and never get upgraded to the real RAW pixels.
            self.mark_full_image_source(keep_path, is_embedded_jpeg=keep_full_is_embedded_jpeg)
        if keep_path and keep_preview is not None and keep_key:
            self.preview_cache.put(keep_key, keep_preview)
        if keep_path and keep_pixmap is not None:
            self.pixmap_cache.put(keep_path, keep_pixmap)

    def on_folder_changed(self, keep_path: Optional[str] = None) -> None:
        """Folder scope changed: free heavy RAM; leave disk thumbs for warm reopen."""
        self.clear_heavy_memory_tiers(keep_path=keep_path)
        try:
            from unified_image_processor import prune_libraw_unsupported_paths

            prune_libraw_unsupported_paths(clear_all=False, max_keep=512)
        except Exception:
            pass

    def get_exif_memory_only(self, file_path: str) -> Optional[Dict[str, Any]]:
        """In-memory EXIF only (no SQLite) — for cache-hit emit during preview-first."""
        return self.exif_memory_cache.get(file_path)

    def get_exif_for_ui(self, file_path: str) -> Optional[Dict[str, Any]]:
        """UI hot path: never blocks (see get_exif's verify=False default)."""
        return self.get_exif(file_path)

    def get_exif(
        self, file_path: str, *, verify: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Get cached EXIF data or return None if not cached.

        RAW orientation trustworthiness: records lacking ``verified_orientation``
        used to be re-checked synchronously on EVERY call via
        cached_raw_exif_orientation_trustworthy() -> rawpy.imread() -- a real
        file open that stalled whatever thread called this (including the Qt
        main thread: status bar, HUD, prefetch decisions), and whose
        successful result was never memoized, so the same file re-paid the
        probe on every subsequent call too.

        Now: ``verify=False`` (default) returns the record optimistically and
        schedules the probe on a background thread; every pixel-rotation path
        re-derives orientation at decode time, so a briefly-unverified record
        can't rotate pixels wrongly, and the background result either stamps
        verified_orientation=True or evicts the bad record so the next fetch
        re-extracts fresh. ``verify=True`` (decode paths that gate orientation
        decisions on this record, e.g. image_load_manager's deferred-EXIF
        emits) keeps the synchronous probe but now memoizes success.
        """
        self.stats['exif_requests'] += 1

        # 1. Check in-memory cache first
        exif_data = self.exif_memory_cache.get(file_path)
        if exif_data is None:
            # 2. Check persistent cache
            exif_data = self.exif_cache.get(file_path)
            if exif_data is not None:
                self.exif_memory_cache.put(file_path, exif_data)

        if exif_data is None:
            self.cache_miss.emit(file_path, 'exif')
            return None

        try:
            from enhanced_raw_processor import (
                RAW_EXIF_SENSOR_META_VER,
                cached_raw_exif_orientation_trustworthy,
            )
            cached_ver = exif_data.get('raw_exif_sensor_meta_ver', 0)
            if cached_ver < RAW_EXIF_SENSOR_META_VER:
                self.cache_miss.emit(file_path, 'exif')
                return None
            if not exif_data.get("verified_orientation"):
                if verify:
                    if not cached_raw_exif_orientation_trustworthy(
                        file_path, exif_data
                    ):
                        self.cache_miss.emit(file_path, 'exif')
                        return None
                    # Memoize the successful probe so no caller ever re-pays it.
                    exif_data = dict(exif_data)
                    exif_data["verified_orientation"] = True
                    self.put_exif(file_path, exif_data)
                else:
                    self._schedule_orientation_verify(file_path, exif_data)
        except Exception:
            pass

        self.cache_hit.emit(file_path, 'exif')
        return exif_data

    def _schedule_orientation_verify(
        self, file_path: str, exif_data: Dict[str, Any]
    ) -> None:
        """Queue a background trustworthiness probe for an unverified RAW record.

        One attempt per path per session; a single lazily-started daemon
        thread drains the queue (probes serialize on the rawpy global lock
        anyway, so more workers would only pile up blocked threads).
        """
        try:
            import common_image_loader

            if not common_image_loader.is_raw_file(file_path):
                return
        except Exception:
            return
        with self._orient_verify_lock:
            if getattr(self, "_closing", False):
                return
            if file_path in self._orient_verify_attempted:
                return
            if len(self._orient_verify_attempted) >= self._orient_verify_attempted_max:
                self._orient_verify_attempted.clear()
            self._orient_verify_attempted.add(file_path)
            self._orient_verify_queue.append((file_path, dict(exif_data)))
            if (
                self._orient_verify_thread is None
                or not self._orient_verify_thread.is_alive()
            ):
                self._orient_verify_thread = threading.Thread(
                    target=self._orientation_verify_worker,
                    name="exif-orient-verify",
                    daemon=True,
                )
                self._orient_verify_thread.start()

    def _orientation_verify_worker(self) -> None:
        while True:
            with self._orient_verify_lock:
                if not self._orient_verify_queue or getattr(self, "_closing", False):
                    self._orient_verify_thread = None
                    return
                path, cached = self._orient_verify_queue.pop(0)
            try:
                from enhanced_raw_processor import (
                    cached_raw_exif_orientation_trustworthy,
                )

                if cached_raw_exif_orientation_trustworthy(path, cached):
                    cached["verified_orientation"] = True
                    if not getattr(self, "_closing", False):
                        self.put_exif(path, cached)
                else:
                    # Cached orientation is wrong: drop the record so the next
                    # fetch re-extracts fresh (extraction writes a verified one).
                    if not getattr(self, "_closing", False):
                        self.exif_memory_cache.remove(path)
                    try:
                        self.exif_cache.remove(path)
                    except Exception:
                        pass
                    self.cache_miss.emit(path, 'exif')
            except Exception:
                pass

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

    def flush_exif_disk_cache(self) -> None:
        """Commit batched EXIF disk writes (call after indexing batches)."""
        if hasattr(self.exif_cache, "flush"):
            self.exif_cache.flush()

    def get_capture_times_for_sort(self, file_paths: list) -> Dict[str, str]:
        """capture_time only, for folder sort (ignores stale mtime on network folders)."""
        return self.exif_cache.get_capture_times_bulk(file_paths)

    def record_probed_capture_times(self, timestamps: Dict[str, float]) -> None:
        """Persist sort-probe capture times so the next folder open skips re-probing.

        timestamps values are epoch floats (from probe_capture_timestamp_from_file);
        stored as ISO strings to match the capture_time column's existing format.
        """
        if not timestamps or not hasattr(self.exif_cache, "update_capture_time_bulk"):
            return
        import datetime

        iso_times: Dict[str, str] = {}
        for path, ts in timestamps.items():
            if not ts or ts <= 0:
                continue
            try:
                iso_times[path] = datetime.datetime.fromtimestamp(ts).strftime(
                    "%Y:%m:%d %H:%M:%S"
                )
            except Exception:
                continue
        self.exif_cache.update_capture_time_bulk(iso_times)

    def get_capture_times_for_folder_sort(self, folder_path: str) -> Dict[str, str]:
        """All capture_time rows under folder_path (normalized path keys)."""
        return self.exif_cache.get_capture_times_for_folder(folder_path)

    def get_multiple_exif(self, file_paths: list, file_stats: Optional[Dict[str, Tuple[int, float]]] = None, fast_mode: bool = True) -> Dict[str, Dict[str, Any]]:
        """Get cached EXIF data for multiple files at once.

        Orientation trust for unverified RAW records is optimistic + deferred
        to a background probe, same as get_exif's verify=False default --
        NOT the old synchronous cached_raw_exif_orientation_trustworthy() ->
        rawpy.imread() call per file. That call, run for every file in a
        folder on the caller's thread (e.g. the main-thread instant-sort
        fast path in main.py's _on_quick_folder_index_ready), turned a large
        folder's sort into a serial rawpy-open storm that froze the UI for
        the whole scan -- get_exif itself was already fixed for this; this
        bulk sibling had the same synchronous check duplicated inline and
        was missed.
        """
        self.stats['exif_requests'] += len(file_paths)

        results = {}
        missing_paths = []

        try:
            from enhanced_raw_processor import RAW_EXIF_SENSOR_META_VER
            has_ver = True
        except Exception:
            has_ver = False
            RAW_EXIF_SENSOR_META_VER = 0

        # Check memory cache first
        for path in file_paths:
            exif = self.exif_memory_cache.get(path)
            if exif:
                if has_ver:
                    cached_ver = exif.get('raw_exif_sensor_meta_ver', 0)
                    if cached_ver < RAW_EXIF_SENSOR_META_VER:
                        missing_paths.append(path)
                        continue
                    if not exif.get('verified_orientation'):
                        self._schedule_orientation_verify(path, exif)
                results[path] = exif
            else:
                missing_paths.append(path)

        if missing_paths:
            # Fetch missing from persistent cache
            db_results = self.exif_cache.get_multiple(missing_paths, file_stats, fast_mode)
            for path, exif in db_results.items():
                if has_ver:
                    cached_ver = exif.get('raw_exif_sensor_meta_ver', 0)
                    if cached_ver < RAW_EXIF_SENSOR_META_VER:
                        continue
                    if not exif.get('verified_orientation'):
                        self._schedule_orientation_verify(path, exif)
                self.exif_memory_cache.put(path, exif)
                results[path] = exif

        return results

    def release_full_image_memory(self, file_path: Optional[str]) -> None:
        """Drop in-memory full-res / pixmap tiers; keep preview/grid/thumbnail on disk."""
        if not file_path:
            return
        self.full_image_cache.remove(file_path)
        self.pixmap_cache.remove(file_path)

    def invalidate_file(self, file_path: str) -> None:
        """Remove all cached data for a specific file."""
        key = self._path_key(file_path)
        self.thumbnail_cache.remove(key)
        self.grid_cache.remove(key)
        self.preview_cache.remove(key)
        self.full_image_cache.remove(file_path)
        self.pixmap_cache.remove(file_path)
        self.disk_thumbnail_cache.remove(file_path)
        self.disk_grid_cache.remove(file_path)
        self.disk_preview_cache.remove(file_path)
        self.exif_memory_cache.remove(file_path)
        self.exif_cache.remove(file_path)
        self.full_image_embedded_jpeg_paths.discard(key)

    def clear_all(self) -> None:
        """Clear all caches."""
        self.thumbnail_cache.clear()
        self.grid_cache.clear()
        self.preview_cache.clear()
        self.full_image_cache.clear()
        self.pixmap_cache.clear()
        self.libraw_preview_paths.clear()
        if self._libraw_preview_paths_file and os.path.isfile(self._libraw_preview_paths_file):
            try:
                os.remove(self._libraw_preview_paths_file)
            except Exception:
                pass
        self.full_image_embedded_jpeg_paths.clear()
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

    def enforce_disk_cache_budget(self, max_mb: Optional[int] = None) -> Dict[str, float]:
        """Evict oldest-cached thumbnail/grid/preview disk entries when total usage
        exceeds a budget (RAWVIEWER_DISK_CACHE_MAX_MB, default 4096 MB).

        Age-based cleanup_old_cache() only removes entries older than 30 days --
        nothing previously capped total disk usage, so a long-lived install could
        grow unbounded across every folder ever browsed. This is stat-based (no
        size column is tracked in the DB) and evicts down to 90% of budget
        (hysteresis) so it doesn't re-trigger on every startup right at the
        boundary -- call from a background/startup pass only, never a hot path.
        """
        import logging

        logger = logging.getLogger(__name__)
        if max_mb is None:
            max_mb = _env_int(
                "RAWVIEWER_DISK_CACHE_MAX_MB", 4096, minimum=256, maximum=1024 * 1024
            )
        max_bytes = max_mb * 1024 * 1024
        target_bytes = int(max_bytes * 0.9)

        tiers = {
            "thumbnail": self.disk_thumbnail_cache,
            "grid": self.disk_grid_cache,
            "preview": self.disk_preview_cache,
        }
        usage: Dict[str, int] = {}
        for name, cache in tiers.items():
            try:
                usage[name] = cache.get_disk_usage_bytes()
            except Exception:
                usage[name] = 0

        total = sum(usage.values())
        usage_mb = {f"{k}_mb": v / (1024 * 1024) for k, v in usage.items()}
        if total <= max_bytes:
            return usage_mb

        to_free = total - target_bytes
        freed_total = 0
        for name, cache in tiers.items():
            tier_usage = usage.get(name, 0)
            if tier_usage <= 0 or total <= 0:
                continue
            # Evict proportionally to this tier's share of total usage.
            share = int(to_free * (tier_usage / total))
            try:
                freed_total += cache.evict_lru(share)
            except Exception:
                pass

        logger.info(
            "[CACHE] Disk cache budget exceeded (%.1f MB > %d MB); evicted %.1f MB "
            "(thumbnail=%.1fMB grid=%.1fMB preview=%.1fMB before eviction)",
            total / (1024 * 1024),
            max_mb,
            freed_total / (1024 * 1024),
            usage_mb.get("thumbnail_mb", 0.0),
            usage_mb.get("grid_mb", 0.0),
            usage_mb.get("preview_mb", 0.0),
        )
        return usage_mb

    def _load_libraw_preview_paths(self) -> None:
        try:
            if self._libraw_preview_paths_file and os.path.isfile(self._libraw_preview_paths_file):
                import json
                with open(self._libraw_preview_paths_file, "r", encoding="utf-8") as f:
                    self.libraw_preview_paths = set(json.load(f))
        except Exception:
            self.libraw_preview_paths = set()

    def _save_libraw_preview_paths(self) -> None:
        try:
            if self._libraw_preview_paths_file:
                import json
                with open(self._libraw_preview_paths_file, "w", encoding="utf-8") as f:
                    json.dump(list(self.libraw_preview_paths), f)
        except Exception:
            pass

    def close(self):
        """Close all persistent cache connections."""
        self._closing = True
        self._save_libraw_preview_paths()
        
        with self._orient_verify_lock:
            self._orient_verify_queue.clear()
        if self._orient_verify_thread is not None and self._orient_verify_thread.is_alive():
            self._orient_verify_thread.join(timeout=2.0)
            
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


_exif_cache_atexit_registered = False


def _register_exif_cache_atexit() -> None:
    global _exif_cache_atexit_registered
    if _exif_cache_atexit_registered:
        return
    _exif_cache_atexit_registered = True
    import atexit

    def _flush_exif_on_exit() -> None:
        try:
            global _global_cache
            if _global_cache is not None and hasattr(_global_cache, "flush_exif_disk_cache"):
                _global_cache.flush_exif_disk_cache()
        except Exception:
            pass

    atexit.register(_flush_exif_on_exit)


def get_image_cache() -> ImageCache:
    """Get the global image cache instance."""
    global _global_cache
    if _global_cache is None:
        persistent = os.environ.get("RAWVIEWER_PERSISTENT_CACHE", "1").lower() in {"1", "true", "yes", "on"}
        if not persistent:
            _cleanup_legacy_disk_cache_once()
        _global_cache = ImageCache(persistent_cache_enabled=persistent)
        _register_exif_cache_atexit()
    return _global_cache


def initialize_cache(cache_dir: str = None, persistent_cache_enabled: bool = False) -> ImageCache:
    """Initialize the global image cache with custom settings."""
    global _global_cache
    _global_cache = ImageCache(cache_dir, persistent_cache_enabled=persistent_cache_enabled)
    return _global_cache
