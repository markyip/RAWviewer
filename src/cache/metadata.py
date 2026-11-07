"""Persistent SQLite EXIF metadata cache for RAWviewer."""

from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Sequence
from typing import Any

from cache.keys import (
    _capture_time_from_exif_row,
    _decode_exif_blob,
    _encode_exif_blob,
    _exif_cache_path_key,
    _exif_sql_folder_like_patterns,
    _file_stats_row,
    _mtime_matches,
)

logger = logging.getLogger(__name__)


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
        self._writer_conn: sqlite3.Connection | None = None
        self._pending_writes: list[tuple[str, Any]] = []
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
        exif_info: dict[str, Any],
        *,
        file_size: int | None = None,
        file_mtime: float | None = None,
    ) -> tuple[Any, ...] | None:
        if file_size is None or file_mtime is None:
            file_size, file_mtime = self._get_file_hash(file_path)
        if file_size == 0:
            return None

        storage_path = _exif_cache_path_key(file_path)
        exif_blob = _encode_exif_blob(exif_info.get("exif_data", {}))

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

        # Never persist preview/thumbnail IFD sizes as sensor dimensions for RAW —
        # those (often 160x120) poison image_covers_sensor_resolution and stall
        # single-view upgrades on the previous file's pixels.
        try:
            from common_image_loader import (
                _MIN_TRUSTED_SENSOR_LONG_EDGE,
                is_raw_file,
            )

            if is_raw_file(file_path) and width and height and max(int(width), int(height)) < _MIN_TRUSTED_SENSOR_LONG_EDGE:
                width = None
                height = None
        except Exception:
            pass

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

    def _get_file_hash(self, file_path: str) -> tuple[int, float]:
        """Get file size and modification time for cache validation."""
        try:
            stat = os.stat(file_path)
            return stat.st_size, stat.st_mtime
        except OSError:
            return 0, 0

    def get(self, file_path: str) -> dict[str, Any] | None:
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
                exif_data = _decode_exif_blob(row[5])

                # Ensure consistent data (DB columns override/augment blob)
                result_capture_time = row[6]
                if result_capture_time and 'capture_time' not in exif_data:
                     exif_data['capture_time'] = result_capture_time

                result_width = row[7]
                result_height = row[8]
                sensor_meta_ver = row[9] if len(row) > 9 else 0
                if sensor_meta_ver is None:
                    sensor_meta_ver = 0
                if result_width:
                    exif_data['original_width'] = result_width
                if result_height:
                    exif_data['original_height'] = result_height

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
        exif_info: dict[str, Any],
        *,
        file_size: int | None = None,
        file_mtime: float | None = None,
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
        items: Sequence[tuple[str, dict[str, Any], int | None, float | None]],
        *,
        commit: bool = True,
    ) -> None:
        """Batch EXIF writes (file_path, exif_info, file_size, file_mtime)."""
        rows: list[tuple[str, Any]] = []
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

    def get_multiple(self, file_paths: list[str], file_stats: dict[str, tuple[int, float]] | None = None, fast_mode: bool = True) -> dict[str, dict[str, Any]]:
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
                lookup: dict[str, str] = {}
                norm_keys: list[str] = []
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

                    exif_data: dict[str, Any] = {}
                    result_capture_time = _capture_time_from_exif_row(row)
                    if not result_capture_time and row[6]:
                        try:
                            exif_data = _decode_exif_blob(row[6])
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
                                exif_data = _decode_exif_blob(row[6])
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

    def get_capture_times_bulk(self, file_paths: list[str]) -> dict[str, str]:
        """Read capture_time for sorting only — no size/mtime validation (fast, K:-safe)."""
        if not file_paths:
            return {}
        out: dict[str, str] = {}
        try:
            conn = self._get_connection()
            # SQLite default max bind variables is 999; one key per path only.
            for i in range(0, len(file_paths), 450):
                chunk = file_paths[i : i + 450]
                lookup: dict[str, str] = {}
                query_keys: list[str] = []
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
                            data = _decode_exif_blob(blob)
                            if isinstance(data, dict):
                                ct = data.get("capture_time")
                        except Exception:
                            pass
                    if ct and str(ct).strip():
                        out[req] = str(ct)
        except Exception:
            pass
        return out

    def update_capture_time_bulk(self, timestamps: dict[str, str]) -> None:
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
                        (pk, None, None, 1, "", "", _encode_exif_blob({}), time.time(), ct, None, None, 0)
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

    def get_capture_times_for_folder(self, folder_path: str) -> dict[str, str]:
        """All cached capture_time entries under folder_path (slash-agnostic LIKE)."""
        if not folder_path:
            return {}
        patterns = _exif_sql_folder_like_patterns(folder_path)
        if not patterns:
            return {}
        out: dict[str, str] = {}
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
                            data = _decode_exif_blob(blob)
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
        with contextlib.suppress(Exception):
            self.flush()
        try:
            if hasattr(self, '_local') and hasattr(self._local, 'conn'):
                self._local.conn.close()
                del self._local.conn
        except Exception:
            pass
