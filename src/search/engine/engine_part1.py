"""Engine mixin part 1 of 7 for SemanticImageIndex."""
from __future__ import annotations

import contextlib
import hashlib
import os
import re
import sqlite3
import sys
import threading
import time
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from PIL import Image

from core.errors import IndexAborted
from raw_adjustments.file_extensions import RAW_FILE_EXTENSIONS
from search.backends import resolve_mobileclip_backend
from search.indexer import _bundled_resource_path, _coerce_float, semantic_embeddings_enabled
from search.queries import CustomReverseGeocoder

try:
    import pycountry
except ImportError:
    pycountry = None

class EnginePart1:

    _RAW_FILE_EXTENSIONS = RAW_FILE_EXTENSIONS

    _FORMAT_HINT_FILENAME_TOKENS: frozenset[str] = (
        frozenset(
            {
                "jpeg",
                "jpg",
                "jpe",
                "tif",
                "tiff",
                "png",
                "gif",
                "bmp",
                "webp",
                "heic",
                "heif",
                "avif",
                "raw",
            }
        )
        | _RAW_FILE_EXTENSIONS
    )

    _FACE_COUNT_POSITIVE_TOKENS = frozenset(
        {
            "has:face",
            "has:faces",
            "has:people",
            "has:person",
            "face",
            "faces",
            "person",
            "people",
            "human",
            "humans",
            "portrait",
        }
    )

    _FACE_COUNT_NEGATIVE_TOKENS = frozenset(
        {
            "no:face",
            "no:faces",
            "no:people",
            "no:person",
            "no:human",
            "no:humans",
        }
    )

    def __init__(self, db_path: str | None = None, model_name: str | None = None):
        if db_path is None:
            cache_dir = os.path.expanduser("~/.rawviewer_cache")
            os.makedirs(cache_dir, exist_ok=True)
            db_path = os.path.join(cache_dir, "semantic_index.db")
        self.db_path = db_path
        self._mobileclip_backend = None # Lazy load
        self._model_name = model_name
        self._index_conn = None
        self._rg_lock = threading.Lock()
        self._paused = False
        self._abort_requested = False
        self._pause_lock = threading.Lock()
        self._metadata_backfill_lock = threading.Lock()
        self._metadata_backfill_paths: set[str] = set()
        self._metadata_backfill_geocode: set[str] = set()
        self._metadata_backfill_running = False
        # ImageLoadManager throttle hooks. Acquired only around heavy work (neural
        # pass / thumbnail warm / face scan) and released while paused, so that a
        # background index sitting idle in the gallery never starves thumbnail decode.
        self._load_throttle_enter = None
        self._load_throttle_exit = None
        self._load_throttle_held = False
        self._load_throttle_lock = threading.Lock()
        self._init_db_if_needed()

    def set_load_throttle_hooks(self, enter, exit_) -> None:
        """Wire callables that raise/lower ImageLoadManager concurrency.

        Called by the UI before a background index run. ``enter``/``exit_`` are the
        ImageLoadManager throttle toggles; they are thread-safe and depth-counted.
        """
        with self._load_throttle_lock:
            self._load_throttle_enter = enter
            self._load_throttle_exit = exit_

    def _acquire_load_throttle(self) -> None:
        """Engage the ILM indexing throttle for the duration of heavy work."""
        with self._load_throttle_lock:
            if self._load_throttle_held:
                return
            fn = self._load_throttle_enter
            self._load_throttle_held = True
        if fn is not None:
            with contextlib.suppress(Exception):
                fn()

    def _release_load_throttle(self) -> None:
        """Release the ILM indexing throttle (idempotent)."""
        with self._load_throttle_lock:
            if not self._load_throttle_held:
                return
            fn = self._load_throttle_exit
            self._load_throttle_held = False
        if fn is not None:
            with contextlib.suppress(Exception):
                fn()

    def pause_indexing(self, pause: bool) -> None:
        """Pause or resume the background indexing loops."""
        with self._pause_lock:
            self._paused = pause
            if not pause:
                self._abort_requested = False

    def request_abort_indexing(self) -> None:
        """Stop in-flight indexing from a previous folder scope."""
        with self._pause_lock:
            self._abort_requested = True
            self._paused = True

    def _wait_if_paused(self) -> None:
        """Block while paused; raise IndexAborted when folder scope is invalidated.

        While blocked we drop any ImageLoadManager throttle we are holding so a
        paused index never serializes gallery thumbnail decode, then re-acquire it
        on resume only if it was held when the pause began.
        """
        flag = os.environ.get("RAWVIEWER_INDEX_PAUSE_IN_GALLERY", "1").strip().lower()
        throttle_dropped = False
        while True:
            with self._pause_lock:
                if self._abort_requested:
                    # Tearing down this run; leave the throttle released.
                    raise IndexAborted()
                if flag not in ("1", "true", "yes", "on") or not self._paused:
                    if throttle_dropped:
                        self._acquire_load_throttle()
                    return
            if self._load_throttle_held and not throttle_dropped:
                self._release_load_throttle()
                throttle_dropped = True
            time.sleep(0.5)

    def _mark_metadata_ready_without_embedding(
        self,
        canonical_fp: str,
        st: os.stat_result,
        conn: sqlite3.Connection,
    ) -> None:
        """Mark file as metadata ready but without semantic embeddings (dim = 0)."""
        conn.execute(
            """
            UPDATE semantic_index
            SET semantic_ready = 1, dim = 0, embedding = ?, updated_at = ?
            WHERE file_path = ? AND model_name = ? AND file_size = ? AND mtime_ns = ?
            """,
            (
                b"",
                float(time.time()),
                canonical_fp,
                self.model_name,
                int(st.st_size),
                self._mtime_ns_from_stat(st),
            ),
        )

    def backend(self):
        if self._mobileclip_backend is None:
            self._mobileclip_backend = resolve_mobileclip_backend()
        return self._mobileclip_backend

    def release_mobileclip_gpu_memory(self) -> None:
        """Free ONNX GPU memory after semantic indexing, before GPU face detection."""
        import logging

        backend = self._mobileclip_backend
        if backend is None:
            return
        if hasattr(backend, "release_gpu_sessions"):
            try:
                backend.release_gpu_sessions()
                logging.getLogger(__name__).info(
                    "[INDEX][FACE] Released MobileCLIP ONNX sessions before face scan"
                )
            except Exception as exc:
                logging.getLogger(__name__).debug(
                    "[INDEX][FACE] MobileCLIP GPU release skipped: %s", exc
                )

    def model_name(self):
        if self._model_name is None:
            if hasattr(self.backend, "MODEL_ID"):
                self._model_name = self.backend.MODEL_ID
            else:
                self._model_name = "unknown"
        return self._model_name

    def _init_db_if_needed(self):
        self._model = None
        self._reverse_geocoder: CustomReverseGeocoder | Literal[False] | None = None
        self._conn = sqlite3.connect(
            self.db_path, check_same_thread=False, timeout=60.0
        )
        self._conn.isolation_level = None
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=60000")
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_index (
                file_path TEXT PRIMARY KEY,
                file_name TEXT,
                file_signature TEXT,
                file_size INTEGER NOT NULL,
                file_mtime REAL NOT NULL,
                mtime_ns INTEGER,
                semantic_ready INTEGER DEFAULT 1,
                model_name TEXT NOT NULL,
                dim INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                capture_time TEXT,
                camera_model TEXT,
                lens_model TEXT,
                iso INTEGER,
                width INTEGER,
                height INTEGER,
                gps_lat REAL,
                gps_lon REAL,
                gps_raw TEXT,
                city TEXT,
                landmark TEXT,
                admin1 TEXT,
                country TEXT,
                country_code TEXT,
                face_count INTEGER,
                orientation INTEGER,
                updated_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_semantic_capture_time ON semantic_index(capture_time)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_semantic_camera_model ON semantic_index(camera_model)"
        )
        # Backward-compatible migration for existing DBs created before GPS fields.
        cols = {
            r[1]
            for r in self._conn.execute("PRAGMA table_info(semantic_index)").fetchall()
        }
        if "file_name" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN file_name TEXT")
        if "file_signature" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN file_signature TEXT")
        if "mtime_ns" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN mtime_ns INTEGER")
        if "semantic_ready" not in cols:
            self._conn.execute(
                "ALTER TABLE semantic_index ADD COLUMN semantic_ready INTEGER DEFAULT 1"
            )
            self._conn.execute(
                "UPDATE semantic_index SET semantic_ready = 1 WHERE semantic_ready IS NULL"
            )
        if "gps_lat" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN gps_lat REAL")
        if "gps_lon" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN gps_lon REAL")
        if "gps_raw" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN gps_raw TEXT")
        if "city" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN city TEXT")
        if "landmark" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN landmark TEXT")
        if "admin1" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN admin1 TEXT")
        if "country" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN country TEXT")
        if "country_code" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN country_code TEXT")
        if "face_count" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN face_count INTEGER")
        if "orientation" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN orientation INTEGER")
        self._conn.commit()
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_semantic_signature_model ON semantic_index(file_signature, model_name)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_semantic_ready_model ON semantic_index(semantic_ready, model_name)"
        )
        # Must run after the migration block above adds gps_lat/gps_lon -- creating
        # this index earlier crashes _init_db() on any pre-GPS-feature (pre-2.5.0)
        # database with "no such column: gps_lat" (found while verifying upgrade
        # compatibility from older versions).
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_semantic_gps_latlon ON semantic_index(gps_lat, gps_lon)"
        )
        self._conn.commit()
        self._backfill_file_signatures()

    def _ensure_model(self):
        # Legacy SentenceTransformer path removed — product builds use MobileCLIP
        # (ONNX / Core ML) only. Keep a clear error if model_name is non-mobileclip.
        raise RuntimeError(
            f"Semantic search model {self.model_name!r} is not supported. "
            "Install MobileCLIP ONNX/Core ML assets (Plus edition)."
        )

    @staticmethod
    def semantic_backend_available() -> bool:
        if not semantic_embeddings_enabled():
            return False
        return resolve_mobileclip_backend().available()

    def semantic_backend_error(self) -> str:
        if self.model_name.startswith("mobileclip-"):
            backend = self._mobileclip_backend or resolve_mobileclip_backend()
            return backend.availability_error()
        return (
            f"Semantic search model {self.model_name!r} is not supported. "
            "Plus edition requires MobileCLIP ONNX/Core ML assets."
        )

    def mobileclip_supports_hub_download(self) -> bool:
        if not self.model_name.startswith("mobileclip-"):
            return False
        backend = self.backend
        return bool(getattr(backend, "SUPPORTS_HUB_DOWNLOAD", False))

    def download_semantic_backend_assets(
        self, progress_callback: Callable[[str], None] | None = None
    ) -> str:
        if self.model_name.startswith("mobileclip-"):
            backend = self._mobileclip_backend or resolve_mobileclip_backend()
            path = backend.download_assets(progress_callback=progress_callback)
            self._mobileclip_backend = backend
            return path
        raise RuntimeError("This semantic backend has no downloadable assets")

    def _ensure_reverse_geocoder(self) -> CustomReverseGeocoder | Literal[False]:
        if self._reverse_geocoder is not None:
            return self._reverse_geocoder
        with self._rg_lock:
            if self._reverse_geocoder is not None:
                return self._reverse_geocoder
            try:
                import csv
                import gzip
                import logging

                db_path = _bundled_resource_path(os.path.join("icons", "cities500.csv.gz"))
                if not os.path.isfile(db_path):
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    db_path = os.path.join(base_dir, "icons", "cities500.csv.gz")
                if not os.path.isfile(db_path):
                    db_path = os.path.join("src", "icons", "cities500.csv.gz")

                logging.getLogger(__name__).info("[VISION] Loading custom reverse geocoder database from %s...", db_path)
                cities = []
                with gzip.open(db_path, "rt", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        cities.append((
                            float(row[0]),
                            float(row[1]),
                            row[2],
                            row[3],
                            row[4],
                            row[5] if len(row) > 5 else ""
                        ))

                # Load landmarks
                landmarks_path = _bundled_resource_path(os.path.join("icons", "landmarks.csv.gz"))
                if not os.path.isfile(landmarks_path):
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    landmarks_path = os.path.join(base_dir, "icons", "landmarks.csv.gz")
                if not os.path.isfile(landmarks_path):
                    landmarks_path = os.path.join("src", "icons", "landmarks.csv.gz")

                landmarks = []
                if os.path.isfile(landmarks_path):
                    logging.getLogger(__name__).info("[VISION] Loading custom landmarks database from %s...", landmarks_path)
                    try:
                        with gzip.open(landmarks_path, "rt", encoding="utf-8") as f:
                            reader = csv.reader(f)
                            for row in reader:
                                landmarks.append((
                                    float(row[0]),
                                    float(row[1]),
                                    row[2],
                                    row[3]
                                ))
                    except Exception as e:
                        logging.getLogger(__name__).warning("[VISION] Failed to parse landmarks file: %s", e)

                self._reverse_geocoder = CustomReverseGeocoder(cities, landmarks)
                logging.getLogger(__name__).info(
                    "[VISION] Place name index ready: %d exact tokens",
                    len(self._reverse_geocoder._place_names),
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("[VISION] Failed to load custom reverse geocoder: %s", e)
                self._reverse_geocoder = False
            return self._reverse_geocoder

    @staticmethod
    def _country_name_from_code(code: str) -> str:
        cc = (code or "").strip().upper()
        if not cc:
            return ""

        # Hardcoded fallbacks for common regions to ensure search reliability
        _FALLBACKS = {
            "JP": "Japan",
            "US": "United States",
            "GB": "United Kingdom",
            "UK": "United Kingdom",
            "TW": "Taiwan",
        }
        if cc in _FALLBACKS:
            return _FALLBACKS[cc]

        if pycountry is not None:
            try:
                c = pycountry.countries.get(alpha_2=cc)
                if c is not None:
                    name = str(getattr(c, "name", "") or "")
                    # Keep UI-friendly labels for search/filter display.
                    if name == "Taiwan, Province of China":
                        return "Taiwan"
                    return name
            except Exception:
                pass
        return cc

    @staticmethod
    def _to_blob(vec: np.ndarray) -> bytes:
        arr = np.asarray(vec, dtype=np.float32)
        return arr.tobytes()

    @staticmethod
    def _from_blob(blob: bytes, dim: int) -> np.ndarray:
        arr = np.frombuffer(blob, dtype=np.float32)
        if dim > 0 and arr.size != dim:
            arr = arr[:dim]
        return arr

    @staticmethod
    def _safe_int(v) -> int:
        try:
            from search.engine.semantic_index import SemanticImageIndex

            values = getattr(v, "values", None)
            if values:
                return SemanticImageIndex._safe_int(values[0])
            ratio_value = SemanticImageIndex._ratio_to_float(v)
            if ratio_value is not None:
                return int(round(ratio_value))
            text = str(v or "").strip()
            if not text:
                return 0
            # exifread may stringify numeric tags as "[800]" or "800 800".
            m = re.search(r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", text)
            if not m:
                return 0
            token = m.group(0)
            if "/" in token:
                num, den = token.split("/", 1)
                den_f = float(den) if float(den) != 0 else 1.0
                return int(round(float(num) / den_f))
            return int(round(float(token)))
        except Exception:
            return 0

    @staticmethod
    def _tag_text(tags: dict[str, object], *names: str) -> str:
        for name in names:
            value = tags.get(name)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    @classmethod
    def _tag_int(cls, tags: dict[str, object], *names: str) -> int:
        for name in names:
            if name not in tags:
                continue
            value = cls._safe_int(tags.get(name))
            if value:
                return value
        return 0

    @staticmethod
    def _pil_dimensions(file_path: str) -> tuple[int, int]:
        """Fallback dimensions when EXIF tags omit width/height.

        For RAW, prefer rawpy header read only (PIL often fails on ARW). On network
        drives this can still take seconds per file under parallel indexing.
        """
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        if ext in RAW_FILE_EXTENSIONS:
            try:
                import rawpy

                with rawpy.imread(file_path) as raw:
                    w = int(raw.sizes.width)
                    h = int(raw.sizes.height)
                    if w > 0 and h > 0:
                        return w, h
            except Exception:
                pass
            return 0, 0

        try:
            with Image.open(file_path) as im:
                return int(im.width), int(im.height)
        except Exception:
            pass
        return 0, 0

    @staticmethod
    def _clean_join(*parts: str) -> str:
        out: list[str] = []
        for part in parts:
            p = (part or "").strip()
            if p and p not in out:
                out.append(p)
        return " ".join(out).strip()

    @staticmethod
    def _legacy_int(v) -> int:
        try:
            return int(v)
        except Exception:
            return 0

    @staticmethod
    def _mtime_matches(stored_mtime: float, st: os.stat_result) -> bool:
        """
        Compare mtime with tolerance to avoid cross-session float precision drift.
        """
        try:
            current = float(st.st_mtime)
            return abs(float(stored_mtime) - current) <= 1e-3
        except Exception:
            return False

    def _canonical_path(self, file_path: str) -> str:
        if not file_path:
            return ""
        try:
            # OPTIMIZATION: If already absolute, skip abspath() which can hit disk/slow down on Windows.
            # Pure string normalization is much faster for thousands of paths on the UI thread.
            if os.path.isabs(file_path):
                ret = os.path.normpath(file_path)
            else:
                ret = os.path.normpath(os.path.abspath(file_path))
            if sys.platform.startswith("win"):
                ret = ret.lower()
            return ret
        except Exception:
            if sys.platform.startswith("win"):
                return file_path.lower()
            return file_path

    @staticmethod
    def _path_aliases(file_path: str) -> list[str]:
        # Fast string-based aliases only.
        ap = os.path.abspath(file_path)
        np = os.path.normpath(ap)
        aliases = [file_path]
        if ap != file_path:
            aliases.append(ap)
        if np not in aliases:
            aliases.append(np)
        return aliases

    def _upsert_metadata(self, canonical_fp: str, st: os.stat_result, meta: dict[str, Any], conn: sqlite3.Connection | None = None) -> None:
        """Helper to insert or update metadata in the index."""
        db = conn if conn is not None else self._conn
        file_name = os.path.basename(canonical_fp)
        file_signature = self._file_signature_from_stat(canonical_fp, st)
        mtime_ns = self._mtime_ns_from_stat(st)

        db.execute(
            """
            INSERT INTO semantic_index (
                file_path, file_name, file_signature, file_size, file_mtime, mtime_ns,
                semantic_ready,
                model_name, dim, embedding,
                capture_time, camera_model, lens_model, iso, width, height,
                gps_lat, gps_lon, gps_raw, city, landmark, admin1, country, country_code, face_count, orientation, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                file_name=excluded.file_name,
                file_signature=excluded.file_signature,
                file_size=excluded.file_size,
                file_mtime=excluded.file_mtime,
                mtime_ns=excluded.mtime_ns,
                semantic_ready=excluded.semantic_ready,
                model_name=excluded.model_name,
                dim=excluded.dim,
                embedding=excluded.embedding,
                capture_time=excluded.capture_time,
                camera_model=excluded.camera_model,
                lens_model=excluded.lens_model,
                iso=excluded.iso,
                width=excluded.width,
                height=excluded.height,
                gps_lat=excluded.gps_lat,
                gps_lon=excluded.gps_lon,
                gps_raw=excluded.gps_raw,
                city=excluded.city,
                landmark=excluded.landmark,
                admin1=excluded.admin1,
                country=excluded.country,
                country_code=excluded.country_code,
                face_count=excluded.face_count,
                orientation=excluded.orientation,
                updated_at=excluded.updated_at
            """,
            (
                canonical_fp,
                file_name,
                file_signature,
                int(st.st_size),
                float(st.st_mtime),
                mtime_ns,
                0,  # semantic_ready = 0 until Phase 2
                self.model_name,
                0,
                b"",
                str(meta.get("capture_time") or ""),
                str(meta.get("camera_model") or ""),
                str(meta.get("lens_model") or ""),
                int(meta.get("iso") or 0),
                int(meta.get("width") or 0),
                int(meta.get("height") or 0),
                _coerce_float(meta.get("gps_lat")),
                _coerce_float(meta.get("gps_lon")),
                str(meta.get("gps_raw") or ""),
                str(meta.get("city") or ""),
                str(meta.get("landmark") or ""),
                str(meta.get("admin1") or ""),
                str(meta.get("country") or ""),
                str(meta.get("country_code") or ""),
                int(meta["face_count"]) if meta.get("face_count") is not None else None,
                int(meta["orientation"]) if meta.get("orientation") is not None else None,
                float(time.time()),
            ),
        )

    @staticmethod
    def _mtime_ns_from_stat(st: os.stat_result) -> int:
        """Extract nanosecond mtime if available, otherwise fallback to float."""
        if hasattr(st, "st_mtime_ns"):
            return int(st.st_mtime_ns)
        return int(st.st_mtime * 1e9)

    @classmethod
    def _file_signature_from_stat(cls, file_path: str, st: os.stat_result) -> str:
        """
        Stable per-file identity that survives path alias changes.
        """
        name = os.path.basename(file_path)
        raw = f"{name}\0{int(st.st_size)}\0{cls._mtime_ns_from_stat(st)}"
        return hashlib.sha1(raw.encode("utf-8", errors="surrogateescape")).hexdigest()

    def _backfill_file_signatures(self) -> None:
        """
        Populate signature columns for existing databases created before stable identity.
        """
        try:
            rows = self._conn.execute(
                """
                SELECT file_path
                FROM semantic_index
                WHERE file_signature IS NULL OR file_signature = '' OR mtime_ns IS NULL OR file_name IS NULL
                """
            ).fetchall()
            for (fp,) in rows:
                try:
                    canonical = self._canonical_path(str(fp))
                    if not os.path.isfile(canonical):
                        continue
                    st = os.stat(canonical)
                    self._conn.execute(
                        """
                        UPDATE semantic_index
                        SET file_name = ?, file_signature = ?, mtime_ns = ?, file_path = ?
                        WHERE file_path = ?
                        """,
                        (
                            os.path.basename(canonical),
                            self._file_signature_from_stat(canonical, st),
                            self._mtime_ns_from_stat(st),
                            canonical,
                            fp,
                        ),
                    )
                except Exception:
                    continue
            self._conn.commit()
        except Exception:
            pass

    @staticmethod
    def _ratio_to_float(v) -> float | None:
        try:
            # exifread.utils.Ratio supports num/den
            if hasattr(v, "num") and hasattr(v, "den"):
                den = float(v.den) if float(v.den) != 0 else 1.0
                return float(v.num) / den
            return float(v)
        except Exception:
            return None
