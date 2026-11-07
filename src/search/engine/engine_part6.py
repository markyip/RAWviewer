"""Engine mixin part 6 of 7 for SemanticImageIndex."""
from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Sequence
from typing import Any

from search.indexer import _coerce_float, _coerce_int
from search.types import MetadataRow, Row


class EnginePart6:
    # Type annotations for attributes inherited from other EnginePart siblings
    _canonical_path: Any
    _conn: Any
    _mtime_ns_from_stat: Any
    _mtime_matches: Any
    _filter_duplicate_raw_companions: Any
    _normalize_loose_metadata_query: Any
    _auto_match_metadata_keywords: Any
    _FORMAT_HINT_FILENAME_TOKENS: frozenset[str]
    _FACE_COUNT_POSITIVE_TOKENS: frozenset[str]
    _FACE_COUNT_NEGATIVE_TOKENS: frozenset[str]
    _RAW_FILE_EXTENSIONS: frozenset[str]
    model_name: str

    def get_pending_paths(self, file_paths: Sequence[str]) -> list[str]:
        """
        Return paths that need indexing (missing row, semantic_ready=0, or file changed on disk).
        Completed work is persisted in semantic_index.db and survives app restarts.
        """
        if not file_paths:
            return []

        import time
        t0 = time.time()

        indexable_paths, _ = self._filter_duplicate_raw_companions(file_paths)
        if not indexable_paths:
            return []

        # Bulk fetch rows in one (or a few) database queries!
        rows = self._fetch_rows_for_paths(indexable_paths)

        # Build map of canonical path to row
        row_map = {}
        for r in rows:
            fp = r['file_path']
            row_map[self._canonical_path(fp)] = r

        pending: list[str] = []

        canonical_map = {self._canonical_path(p): p for p in indexable_paths if p}
        for cp, original in canonical_map.items():
            try:
                st = os.stat(cp)
            except OSError:
                pending.append(original)
                continue

            row = row_map.get(cp)
            sr = _coerce_int(self._row_value(row, 'semantic_ready')) if row else 0
            dim = _coerce_int(self._row_value(row, 'dim')) if row else 0
            if (not row) or sr == 0 or (sr == 1 and dim == 0):
                pending.append(original)
                continue

            # Check if file changed on disk (mtime or size mismatch)
            row_size = self._row_value(row, 'file_size', None)
            if row_size is None or _coerce_int(row_size) != int(st.st_size):
                pending.append(original)
                continue

            row_mtime_ns = self._row_value(row, 'mtime_ns', None)
            if row_mtime_ns is not None:
                try:
                    if _coerce_int(row_mtime_ns) != self._mtime_ns_from_stat(st):
                        pending.append(original)
                        continue
                except Exception:
                    pass
            else:
                row_mtime = self._row_value(row, 'file_mtime', None)
                mtime = _coerce_float(row_mtime)
                if mtime is None or not self._mtime_matches(mtime, st):
                    pending.append(original)
                    continue

        import logging
        logger = logging.getLogger(__name__)
        logger.debug(
            "[INDEX] Pending scan: %d pending of %d paths in %.4fs (bulk)",
            len(pending),
            len(file_paths),
            time.time() - t0
        )
        return pending

    def get_metadata_extraction_pending_paths(self, file_paths: Sequence[str]) -> list[str]:
        """Paths that need on-disk EXIF/metadata extraction (no row or file changed).

        Narrower than :meth:`get_pending_paths`, which also lists files awaiting
        semantic embeddings. Use this to skip ``build_index`` when the silent
        metadata pass has nothing to read from disk.
        """
        if not file_paths:
            return []

        indexable_paths, _ = self._filter_duplicate_raw_companions(file_paths)
        if not indexable_paths:
            return []

        rows = self._fetch_rows_for_paths(indexable_paths)
        row_map = {}
        for r in rows:
            row_map[self._canonical_path(r["file_path"])] = r

        pending: list[str] = []
        canonical_map = {self._canonical_path(p): p for p in indexable_paths if p}
        for cp, original in canonical_map.items():
            try:
                st = os.stat(cp)
            except OSError:
                pending.append(original)
                continue

            row = row_map.get(cp)
            if not row:
                pending.append(original)
                continue

            row_size = self._row_value(row, "file_size", None)
            if row_size is None or _coerce_int(row_size) != int(st.st_size):
                pending.append(original)
                continue

            row_mtime_ns = self._row_value(row, "mtime_ns", None)
            if row_mtime_ns is not None:
                try:
                    if _coerce_int(row_mtime_ns) != self._mtime_ns_from_stat(st):
                        pending.append(original)
                except Exception:
                    pending.append(original)
                continue

            row_mtime = self._row_value(row, "file_mtime", None)
            if row_mtime is None or not self._mtime_matches(_coerce_float(row_mtime) or 0.0, st):
                pending.append(original)

        return pending

    def get_pending_embedding_paths(self, file_paths: Sequence[str]) -> list[str]:
        """
        Return paths that have metadata in database but are missing semantic embeddings (dim = 0).
        """
        if not file_paths:
            return []

        indexable_paths, _ = self._filter_duplicate_raw_companions(file_paths)
        if not indexable_paths:
            return []

        rows = self._fetch_rows_for_paths(indexable_paths)
        row_map = {}
        for r in rows:
            fp = r['file_path']
            row_map[self._canonical_path(fp)] = r

        pending: list[str] = []
        canonical_map = {self._canonical_path(p): p for p in indexable_paths if p}
        for cp, original in canonical_map.items():
            row = row_map.get(cp)
            sr = _coerce_int(self._row_value(row, "semantic_ready")) if row else 0
            dim = _coerce_int(self._row_value(row, "dim")) if row else 0
            # Match get_pending_paths: missing row, metadata-only (sr=0), or no embedding yet.
            if (not row) or sr == 0 or (sr == 1 and dim == 0):
                pending.append(original)

        return pending

    def _fetch_rows_for_paths(self, paths: Sequence[str]) -> list[MetadataRow]:
        """Bulk fetch rows for a list of paths using optimized batch queries."""
        if not paths:
            return []

        # 1. Deduplicate and canonicalize
        canonical_to_original = {}
        unique_canonical = []
        for p in paths:
            if not p:
                continue
            cp = self._canonical_path(p)
            if cp not in canonical_to_original:
                canonical_to_original[cp] = p
                unique_canonical.append(cp)

        if not unique_canonical:
            return []

        # 2. Bulk fetch by file_path
        self._conn.row_factory = sqlite3.Row

        found_map = {}
        # SQLite parameter limit is usually 999; use 500 to be safe
        chunk_size = 500
        for i in range(0, len(unique_canonical), chunk_size):
            chunk = unique_canonical[i:i+chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            # IMPORTANT: Do NOT filter by model_name here.
            # Metadata fields (face_count, gps_lat/lon, camera_model, city, iso,
            # capture_time, etc.) are model-independent and must be accessible
            # regardless of which AI/embedding model is currently active.
            # Only `embedding` and `semantic_ready` are model-specific.
            # If multiple rows exist for the same file_path (different model_names),
            # prefer the current model's row so that semantic_ready/embedding are
            # correct; otherwise fall back to any row for the path.
            query = f"""
                SELECT file_path, file_name, file_signature, dim, embedding, capture_time,
                       camera_model, lens_model, iso, gps_lat, gps_lon, width, height,
                       city, landmark, admin1, country, country_code, face_count, semantic_ready,
                       file_size, file_mtime, mtime_ns, model_name
                FROM semantic_index
                WHERE file_path IN ({placeholders})
                ORDER BY
                    CASE WHEN model_name = ? THEN 0 ELSE 1 END,
                    rowid DESC
            """
            rows = self._conn.execute(query, [*chunk, self.model_name]).fetchall()
            for r in rows:
                fp = r['file_path']
                if fp not in found_map:
                    # First match wins: preferred model first (ORDER BY above), then any row
                    found_map[fp] = r

        # 3. Assemble results in original order, injecting mock rows for missing files
        # This ensures that newly added files appear in search results (as unindexed)
        # instead of being filtered out.
        results = []
        for cp in unique_canonical:
            if cp in found_map:
                results.append(found_map[cp])
            else:
                # Create a mock row for files not yet indexed
                full_path = canonical_to_original[cp]
                file_name = os.path.basename(full_path)

                # Use a dictionary as a proxy for sqlite3.Row
                # RAWviewer's UI components are hardened to handle missing metadata
                mock_row = {
                    'file_path': cp,
                    'file_name': file_name,
                    'file_signature': "",
                    'dim': 0,
                    'embedding': None,
                    'capture_time': "",
                    'camera_model': "",
                    'lens_model': "",
                    'iso': 0,
                    'gps_lat': None,
                    'gps_lon': None,
                    'width': 0,
                    'height': 0,
                    'city': "",
                    'landmark': "",
                    'admin1': "",
                    'country': "",
                    'country_code': "",
                    'face_count': 0,
                    'semantic_ready': 0,
                    'file_size': 0,
                    'file_mtime': 0,
                    'mtime_ns': 0,
                    'model_name': self.model_name
                }
                results.append(mock_row)

        return results

    def get_layout_metadata_for_paths(self, paths: Sequence[str]) -> dict[str, dict[str, Any]]:
        """
        Fast synchronous fetch of layout metadata (width, height, orientation, capture_time)
        for pre-seeding the gallery layout to avoid flashing on first render.
        """
        if not paths:
            return {}

        canonical_to_original = {}
        unique_canonical = []
        for p in paths:
            if not p:
                continue
            cp = self._canonical_path(p)
            if cp not in canonical_to_original:
                canonical_to_original[cp] = p
                unique_canonical.append(cp)

        if not unique_canonical:
            return {}

        # Bulk fetch by file_path
        self._conn.row_factory = sqlite3.Row

        result_map = {}
        chunk_size = 500
        for i in range(0, len(unique_canonical), chunk_size):
            chunk = unique_canonical[i:i+chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            query = f"""
                SELECT file_path, width, height, orientation, capture_time
                FROM semantic_index
                WHERE file_path IN ({placeholders})
                ORDER BY
                    CASE WHEN model_name = ? THEN 0 ELSE 1 END,
                    rowid DESC
            """
            rows = self._conn.execute(query, [*chunk, self.model_name]).fetchall()
            for r in rows:
                fp = r['file_path']
                if fp not in result_map:
                    orig_path = canonical_to_original[fp]
                    w = _coerce_int(r["width"])
                    h = _coerce_int(r["height"])
                    o = _coerce_int(r["orientation"], 1)
                    result_map[orig_path] = {
                        "original_width": w,
                        "original_height": h,
                        "orientation": o,
                        "capture_time": r["capture_time"] or "",
                    }

        return result_map

    @staticmethod
    def _parse_capture_year(capture_time: str) -> int:
        # EXIF style: "YYYY:MM:DD HH:MM:SS"
        try:
            return int((capture_time or "")[:4])
        except Exception:
            return 0

    @staticmethod
    def _parse_capture_month(capture_time: str) -> int:
        try:
            return int((capture_time or "")[5:7])
        except Exception:
            return 0

    @staticmethod
    def _row_value(row: MetadataRow | None, key: str, default: object = "") -> object:
        if row is None:
            return default
        try:
            return row[key]
        except Exception:
            try:
                if isinstance(row, dict):
                    return row.get(key, default)
            except Exception:
                pass
            return default

    @staticmethod
    def _to_int(value: object, default: int = 0) -> int:
        return _coerce_int(value, default)

    @staticmethod
    def _to_float(value: object) -> float | None:
        return _coerce_float(value)

    @staticmethod
    def _normalize_date_value(value: str) -> str:
        return (value or "").strip().replace("-", ":").replace("/", ":")

    @staticmethod
    def _row_file_extension(row: Row | dict[str, object] | None) -> str:
        from search.engine.semantic_index import SemanticImageIndex

        fp = str(SemanticImageIndex._row_value(row, "file_path") or "")
        fn = str(SemanticImageIndex._row_value(row, "file_name") or "")
        # Prefer path basename (primary key truth); fallback to indexed file_name.
        for base in (os.path.basename(fp), fn):
            ext = os.path.splitext(base)[1].lower().lstrip(".")
            if ext:
                return ext
        return ""

    def _needle_is_solitary_format_token(self, needle: str) -> bool:
        n = (needle or "").strip().lower().replace("_", "")
        if len(n) < 2 or len(n) > 12:
            return False
        if not re.fullmatch(r"[a-z0-9]+", n):
            return False
        return n in self._FORMAT_HINT_FILENAME_TOKENS

    @classmethod
    def _format_specs_to_accept_set(cls, spec: str) -> frozenset[str]:
        """Map one user-facing format keyword to a set of lowercase extensions."""
        s = (spec or "").strip().lower().lstrip(".")
        if not s:
            return frozenset()
        if s in ("jpeg", "jpg", "jpe"):
            return frozenset({"jpg", "jpeg", "jpe"})
        if s in ("tif", "tiff"):
            return frozenset({"tif", "tiff"})
        if s in ("heic", "heif"):
            return frozenset({"heic", "heif"})
        if s == "raw":
            return cls._RAW_FILE_EXTENSIONS
        return frozenset({s})

    @classmethod
    def _row_matches_format_specs(cls, row: Row | dict[str, object] | None, specs: Sequence[str]) -> bool:
        fe = cls._row_file_extension(row)
        for spec in specs:
            accepted = cls._format_specs_to_accept_set(spec)
            if fe in accepted:
                return True
        return False

    def _capture_time_matches_date(cls, capture_time: str, value: str) -> bool:
        normalized = cls._normalize_date_value(value)
        if not normalized:
            return False
        capture = str(capture_time or "")
        return capture.startswith(normalized)

    @staticmethod
    def _contains_loose(haystack: str, needle: str) -> bool:
        h = str(haystack or "").strip().lower()
        n = str(needle or "").strip().lower()
        if not n or not h:
            return False
        variants = {n, n.replace("_", " "), n.replace(" ", "_")}
        if any(v and v in h for v in variants):
            return True
        if len(h) >= 4:
            for v in variants:
                if h in v:
                    return True
        return False

    @staticmethod
    def _contains_location_place(haystack: str, needle: str) -> bool:
        """Location field match — forward substring with word boundaries only."""
        h = str(haystack or "").strip().lower()
        n = str(needle or "").strip().lower()
        if not n or not h:
            return False
        variants = {n, n.replace("_", " "), n.replace(" ", "_")}
        for v in variants:
            if not v:
                continue
            if h == v:
                return True
            start = 0
            while True:
                idx = h.find(v, start)
                if idx < 0:
                    break
                before_ok = idx == 0 or not h[idx - 1].isalnum()
                end = idx + len(v)
                after_ok = end == len(h) or not h[end].isalnum()
                if before_ok and after_ok:
                    return True
                start = idx + 1
        return False

    def _apply_filters(
        self, rows: Sequence[MetadataRow], query_text: str
    ) -> tuple[list[MetadataRow], str]:
        """
        Parse a mixed query and filter rows.

        Supported filter tokens:
        - camera:<text>
        - lens:<text>
        - city:<text>
        - admin:<text>
        - country:<text>          (country name or code)
        - date:<prefix>          (e.g. date:2026:05, date:2026-05, date:2026-05-01)
        - filename:<text> / file:<text> / name:<text>
        - format:<ext> / type:<ext> / ext:<ext>
          Comma-separated OR: format:jpg,png. Leading dot optional. Synonyms:
          jpg/jpeg/jpe, tif/tiff, heif/heic, raw=<common RAW extensions>.
        - iso<800 / iso<=800 / iso=400 / iso>=200 / iso>100
        - year=2026 / year>=2024 / year<2020
        - month=5 / month>=6 / month<3
        - width>=3000 / height<2000
        - has:gps / no:gps
        - has:face / no:face — also shorthand face, faces, people, person, human(s),
          has:people (uses indexed Vision face counts, not CLIP similarity)
        """
        raw = (query_text or "").strip()
        if not raw:
            return list(rows), ""

        # Normalize numeric filters so both styles work:
        # - iso<800
        # - iso < 800
        raw = re.sub(
            r"\b(iso|year|month|width|height)\s*(<=|>=|=|<|>)\s*(\d+)\b",
            lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}",
            raw,
            flags=re.I,
        )
        raw = re.sub(
            r"\b(iso|year|month|width|height)\s*:\s*(\d+)\b",
            lambda m: f"{m.group(1)}={m.group(2)}",
            raw,
            flags=re.I,
        )
        raw = self._normalize_loose_metadata_query(raw)

        parts = [p for p in raw.split() if p.strip()]
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[SEARCH] Raw query: '{query_text}' -> Normalized: '{raw}'")

        semantic_terms: list[str] = []
        filtered = list(rows)

        num_pat = re.compile(r"^(iso|year|month|width|height)\s*(<=|>=|=|<|>)\s*(\d+)$", re.I)
        date_like_pat = re.compile(r"^\d{4}(?:[:-]\d{1,2}){0,2}$")

        for token in parts:
            t = token.strip()
            low = t.lower()
            matched = False

            if low.startswith("camera:"):
                matched = True
                needle = t.split(":", 1)[1].strip().lower()
                if needle:
                    filtered = [r for r in filtered if self._contains_loose(str(r["camera_model"] or ""), needle)]
                continue

            if low.startswith("lens:"):
                matched = True
                needle = t.split(":", 1)[1].strip().lower()
                if needle:
                    filtered = [r for r in filtered if self._contains_loose(str(r["lens_model"] or ""), needle)]
                continue

            if low.startswith("date:"):
                matched = True
                pref = t.split(":", 1)[1].strip()
                if pref:
                    filtered = [
                        r
                        for r in filtered
                        if self._capture_time_matches_date(str(self._row_value(r, "capture_time") or ""), pref)
                    ]
                continue

            if low.startswith(("filename:", "file:", "name:")):
                matched = True
                needle = t.split(":", 1)[1].strip().lower()
                if needle:
                    if self._needle_is_solitary_format_token(needle):
                        specs = [needle]
                        filtered = [r for r in filtered if self._row_matches_format_specs(r, specs)]
                    else:
                        filtered = [
                            r
                            for r in filtered
                            if self._contains_loose(
                                str(self._row_value(r, "file_name") or os.path.basename(str(self._row_value(r, "file_path") or ""))),
                                needle,
                            )
                        ]
                continue

            if low.startswith(("format:", "type:", "ext:")):
                matched = True
                rest = t.split(":", 1)[1].strip()
                if rest:
                    specs = [s.strip().lower().lstrip(".") for s in re.split(r"[,;/|]", rest) if s.strip()]
                    if specs:
                        filtered = [r for r in filtered if self._row_matches_format_specs(r, specs)]
                continue

            if low.startswith("city:") or low.startswith("landmark:"):
                matched = True
                needle = t.split(":", 1)[1].strip().lower()
                if needle:
                    filtered = [
                        r for r in filtered
                        if self._contains_location_place(str(r["city"] or ""), needle)
                        or self._contains_location_place(str(self._row_value(r, "landmark") or ""), needle)
                    ]
                continue

            if low.startswith("admin:"):
                matched = True
                needle = t.split(":", 1)[1].strip().lower()
                if needle:
                    filtered = [
                        r for r in filtered
                        if self._contains_location_place(str(r["admin1"] or ""), needle)
                    ]
                continue

            if low.startswith("country:"):
                matched = True
                needle = t.split(":", 1)[1].strip().lower()
                if needle:
                    filtered = [
                        r
                        for r in filtered
                        if (
                            self._contains_location_place(str(r["country"] or ""), needle)
                            or self._contains_location_place(str(r["country_code"] or ""), needle)
                        )
                    ]
                continue

            if low == "has:gps":
                matched = True
                filtered = [
                    r
                    for r in filtered
                    if r["gps_lat"] is not None and r["gps_lon"] is not None
                ]
                continue

            if low == "no:gps":
                matched = True
                filtered = [
                    r
                    for r in filtered
                    if r["gps_lat"] is None or r["gps_lon"] is None
                ]
                continue

            if low in self._FACE_COUNT_POSITIVE_TOKENS:
                matched = True
                filtered = [r for r in filtered if _coerce_int(r["face_count"]) > 0]
                continue

            if low in self._FACE_COUNT_NEGATIVE_TOKENS:
                matched = True
                filtered = [r for r in filtered if _coerce_int(r["face_count"]) <= 0]
                continue

            if low == "has:gps":
                matched = True
                filtered = [r for r in filtered if r["gps_lat"] is not None]
                continue

            if low == "no:gps":
                matched = True
                filtered = [r for r in filtered if r["gps_lat"] is None]
                continue

            if low.startswith("gps:"):
                matched = True
                try:
                    parts = low[4:].split(",")
                    if len(parts) == 2:
                        target_lat = float(parts[0])
                        target_lon = float(parts[1])
                        # Filter by images within ~50km radius (approx 0.5 degrees)
                        filtered = [
                            r for r in filtered
                            if r["gps_lat"] is not None and
                            abs((_coerce_float(r["gps_lat"]) or 0.0) - target_lat) < 0.5 and
                            abs((_coerce_float(r["gps_lon"]) or 0.0) - target_lon) < 0.5
                        ]
                except Exception:
                    pass
                continue

            m = num_pat.match(low)
            if m:
                matched = True
                key, op, val_str = m.group(1), m.group(2), m.group(3)
                val = int(val_str)

                def _ok(x: int, op: str = op, val: int = val) -> bool:
                    if x <= 0:
                        if op == "=":
                            return val == 0
                        if op == "<=":
                            return val == 0
                        return False
                    if op == "<":
                        return x < val
                    if op == "<=":
                        return x <= val
                    if op == "=":
                        return x == val
                    if op == ">=":
                        return x >= val
                    return x > val

                if key == "iso":
                    filtered = [r for r in filtered if _ok(_coerce_int(self._row_value(r, "iso")))]
                elif key == "year":
                    filtered = [r for r in filtered if _ok(self._parse_capture_year(str(self._row_value(r, "capture_time") or "")))]
                elif key == "month":
                    filtered = [r for r in filtered if _ok(self._parse_capture_month(str(self._row_value(r, "capture_time") or "")))]
                elif key == "width":
                    filtered = [r for r in filtered if _ok(_coerce_int(self._row_value(r, "width")))]
                elif key == "height":
                    filtered = [r for r in filtered if _ok(_coerce_int(self._row_value(r, "height")))]
                continue

            if date_like_pat.match(low):
                matched = True
                filtered = [
                    r
                    for r in filtered
                    if self._capture_time_matches_date(str(self._row_value(r, "capture_time") or ""), low)
                ]
                continue

            if not matched:
                bare = low.strip().lstrip(".")
                if (
                    re.fullmatch(r"[a-z0-9]{2,15}", bare or "")
                    and bare in self._FORMAT_HINT_FILENAME_TOKENS
                ):
                    matched = True
                    filtered = [r for r in filtered if self._row_matches_format_specs(r, [bare])]
                    continue
                bare.replace("_", " ")
                semantic_terms.append(t)

        filtered, semantic_terms = self._auto_match_metadata_keywords(filtered, semantic_terms)
        metadata_stopwords = {
            "a",
            "an",
            "the",
            "of",
            "with",
            "for",
            "photo",
            "photos",
            "image",
            "images",
            "picture",
            "pictures",
            "shot",
            "shots",
            "taken",
            "exif",
            "metadata",
        }
        semantic_terms = [
            t for t in semantic_terms if (t or "").strip().lower() not in metadata_stopwords
        ]
        return filtered, " ".join(semantic_terms).strip()
