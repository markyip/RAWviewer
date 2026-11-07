"""Engine mixin part 7 of 7 for SemanticImageIndex."""
from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Sequence
from typing import cast

from search.indexer import _coerce_float, _coerce_int, semantic_embeddings_enabled
from search.queries import KNOWN_LOCATION_NAMES, CustomReverseGeocoder, _normalize_location_term, _pycountry_name_index
from search.types import MetadataRow, SearchHit


class EnginePart7:




    def _normalize_loose_metadata_query(self, raw: str) -> str:
        """
        Accept light natural-language metadata filters without an LLM.
        Examples:
        - iso under 800 -> iso<800
        - in tokyo -> city:tokyo
        - from japan -> country:japan
        """
        text = f" {raw.strip()} "

        text = re.sub(
            r"\b(iso|year|month|width|height)\s+(\d+)\b",
            r"\1=\2",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\b(camera|lens|city|admin|country|date|filename|file|name|format|type|ext)\s*=\s*([^\s]+)\b",
            r"\1:\2",
            text,
            flags=re.I,
        )

        numeric_phrases = [
            (r"\b(iso|year|month|width|height)\s+(?:under|below|less\s+than|smaller\s+than)\s+(\d+)\b", r"\1<\2"),
            (r"\b(iso|year|month|width|height)\s+(?:at\s+most|up\s+to|no\s+more\s+than|less\s+than\s+or\s+equal\s+to)\s+(\d+)\b", r"\1<=\2"),
            (r"\b(iso|year|month|width|height)\s+(?:over|above|more\s+than|greater\s+than|larger\s+than)\s+(\d+)\b", r"\1>\2"),
            (r"\b(iso|year|month|width|height)\s+(?:at\s+least|no\s+less\s+than|greater\s+than\s+or\s+equal\s+to)\s+(\d+)\b", r"\1>=\2"),
            (r"\b(iso|year|month|width|height)\s+(?:is|equals?|equal\s+to)\s+(\d+)\b", r"\1=\2"),
        ]
        for pattern, replacement in numeric_phrases:
            text = re.sub(pattern, replacement, text, flags=re.I)

        # File-format phrases MUST run before loose "in <place>" / "from <place>" patterns,
        # otherwise "in jpeg", "near raw", "from raw" become city:/country: filters.
        text = re.sub(
            r"\b(format|type|ext)\s+(jpeg|jpe|jpg|tif|tiff|png|gif|bmp|webp|heic|heif|avif|raw"
            r"|cr2|cr3|arw|nef|dng|orf|raf|rw2|pef|srw|rwl|erf|x3f|3fr)\b",
            lambda m: f"{m.group(1).lower()}:{m.group(2).lower()}",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\b(?:file|file\s+name|name)\s+(jpeg|jpe|jpg|tif|tiff|png|gif|bmp|webp|heic|heif|avif|raw)\b",
            lambda m: f"format:{m.group(1).lower()}",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\b(?:file|file\s+name|name)\s+(cr2|cr3|arw|nef|dng|orf|raf|rw2|pef|srw|rwl|erf|x3f|3fr)\b",
            lambda m: f"format:{m.group(1).lower()}",
            text,
            flags=re.I,
        )

        # Multi-word location detection pass
        # This joins known multi-word places into single tokens to prevent splitting
        _MULTI_WORD_LOCATIONS = [
            "hong kong", "new york", "san francisco", "los angeles", "united states",
            "united kingdom", "south korea", "north korea", "saudi arabia", "south africa",
            "kuala Lumpur", "ho chi minh", "san diego", "las vegas", "new zealand",
            "buenos aires", "rio de janeiro", "mexico city", "cape town", "saint petersburg",
            "san jose", "nha trang", "da nang", "koh samui", "koh phangan", "bora bora",
            "puerto rico", "costa rica", "el salvador", "gran canaria", "san sebastian"
        ]
        for loc in _MULTI_WORD_LOCATIONS:
            # Match word boundaries to avoid partial matches (e.g., "Hong Kong" vs "Hong Konger")
            pattern = rf"\b{re.escape(loc)}\b"
            if re.search(pattern, text, re.I):
                # Convert to underscore-joined city:token
                token = "_".join(loc.split())
                text = re.sub(pattern, f"city:{token}", text, flags=re.I)

        fmt_hints = self._FORMAT_HINT_FILENAME_TOKENS

        @staticmethod
        def _loose_place_to_city(m):
            place = m.group(1).strip()
            if not re.search(r"\s", place):
                token = place.lower().lstrip(".")
                if token in fmt_hints:
                    return f"format:{token}"
            return f"city:{'_'.join(place.split())}"

        @staticmethod
        def _loose_place_to_country(m):
            place = m.group(1).strip()
            if not re.search(r"\s", place):
                token = place.lower().lstrip(".")
                if token in fmt_hints:
                    return f"format:{token}"
            return f"country:{'_'.join(place.split())}"

        text = re.sub(
            r"\b(?:in|at|near)\s+([A-Za-z][\w\-]*(?:\s+[A-Za-z][\w\-]*){0,2})\b",
            _loose_place_to_city,
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\b(?:from|country)\s+([A-Za-z][\w\-]*(?:\s+[A-Za-z][\w\-]*){0,2})\b",
            _loose_place_to_country,
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\bcamera\s+([A-Za-z0-9][\w\-.]*(?:\s+[A-Za-z0-9][\w\-.]*){0,2})\b",
            lambda m: f"camera:{'_'.join(m.group(1).split())}",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\blens\s+([A-Za-z0-9][\w\-.]*(?:\s+[A-Za-z0-9][\w\-.]*){0,3})\b",
            lambda m: f"lens:{'_'.join(m.group(1).split())}",
            text,
            flags=re.I,
        )

        text = re.sub(
            r"\b(?:filename|file\s+name|file|name)\s+([A-Za-z0-9][\w\-.]*(?:\s+[A-Za-z0-9][\w\-.]*){0,2})\b",
            lambda m: f"filename:{'_'.join(m.group(1).split())}",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\bdate\s+(\d{4}(?:[-:/]\d{1,2}){0,2})\b",
            r"date:\1",
            text,
            flags=re.I,
        )

        # GPS / Location shortcuts
        text = re.sub(r"\b(?:with|has|include|containing|containing\s+exif)\s+gps\b", "has:gps", text, flags=re.I)
        text = re.sub(r"\b(?:no|without|missing|missing\s+exif)\s+gps\b", "no:gps", text, flags=re.I)
        text = re.sub(r"\bcoords?:\s*(-?\d+(?:\.\d+)?)\s*[,/ ]\s*(-?\d+(?:\.\d+)?)\b", r"gps:\1,\2", text, flags=re.I)

        return text.strip()

    def _is_strictly_location_name(self, term: str) -> bool:
        """
        Check if a term is a known country or city/place name.
        Uses static lists plus the bundled GeoNames place index (O(1) lookup).
        """
        needle = _normalize_location_term(term)
        if len(needle) < 3:
            return False

        if needle in KNOWN_LOCATION_NAMES:
            return True

        if needle in _pycountry_name_index():
            return True

        geo = self._reverse_geocoder
        if not isinstance(geo, CustomReverseGeocoder):
            ensured = self._ensure_reverse_geocoder()
            if isinstance(ensured, CustomReverseGeocoder):
                geo = ensured
        if isinstance(geo, CustomReverseGeocoder):
            return geo.has_place_name(needle)

        return False

    def _row_matches_location_place(self, row: MetadataRow, needle: str) -> bool:
        place = (needle or "").strip().lower()
        if not place:
            return False
        return (
            self._contains_location_place(str(row["city"] or ""), place)
            or self._contains_location_place(str(self._row_value(row, "landmark") or ""), place)
            or self._contains_location_place(str(row["admin1"] or ""), place)
            or self._contains_location_place(str(row["country"] or ""), place)
            or self._contains_location_place(str(row["country_code"] or ""), place)
        )

    def _row_has_resolved_location(self, row: MetadataRow) -> bool:
        return bool(
            str(row["city"] or "").strip()
            or str(self._row_value(row, "landmark") or "").strip()
            or str(row["country"] or "").strip()
        )

    def _auto_match_metadata_keywords(
        self, rows: Sequence[MetadataRow], semantic_terms: Sequence[str]
    ) -> tuple[list[MetadataRow], list[str]]:
        """
        If a free-text token appears in metadata fields, use it as a metadata filter
        and remove it from the semantic text query.
        """
        filtered = list(rows)
        remaining: list[str] = []
        metadata_fields = ("city", "landmark", "admin1", "country", "country_code", "camera_model", "lens_model", "file_name")

        import logging
        logger = logging.getLogger(__name__)

        for term in semantic_terms:
            needle = (term or "").strip().lower()
            if len(needle) < 3:
                remaining.append(term)
                continue
            if re.fullmatch(r"\d{4}(?:[:-]\d{1,2}){0,2}", needle):
                matched_rows = [
                    r
                    for r in filtered
                    if self._capture_time_matches_date(str(self._row_value(r, "capture_time") or ""), needle)
                ]
                if matched_rows:
                    filtered = matched_rows
                    continue
            is_location = self._is_strictly_location_name(needle)
            if is_location:
                matched_rows = [
                    r for r in filtered if self._row_matches_location_place(r, needle)
                ]
            else:
                matched_rows = [
                    r
                    for r in filtered
                    if any(
                        self._contains_loose(str(self._row_value(r, field) or ""), needle)
                        for field in metadata_fields
                    )
                ]
            if matched_rows:
                logger.info(f"[SEARCH] Metadata match for '{needle}' found in {len(matched_rows)} image(s)")
                filtered = matched_rows
            else:
                # CONTRADICTION FILTER: If this is a location name (e.g. "Korea") but
                # doesn't match any image in the folder, check if any images HAVE
                # verified location metadata. If an image says "Japan", and we search
                # "Korea", we should exclude it rather than letting AI "guess".
                if is_location and not semantic_embeddings_enabled():
                    new_filtered = []
                    contradiction_count = 0
                    for r in filtered:
                        if self._row_has_resolved_location(r):
                            if self._row_matches_location_place(r, needle):
                                new_filtered.append(r)
                            else:
                                # Contradiction! (Has location, but it's different)
                                contradiction_count += 1
                    if contradiction_count > 0:
                        logger.warning(f"[SEARCH] Contradiction Filter: Excluded {contradiction_count} image(s) from '{needle}' due to conflicting GPS metadata")
                    filtered = new_filtered
                elif is_location and semantic_embeddings_enabled():
                    # Ambiguous bare token that is also a GeoNames place (e.g. "train"
                    # the town in Germany vs trains the vehicle). With no GPS hits in this
                    # folder, prefer semantic CLIP rather than returning zero results.
                    logger.debug(
                        "[SEARCH] No GPS match for place name '%s'; passing to semantic AI",
                        needle,
                    )
                    remaining.append(term)
                else:
                    if semantic_embeddings_enabled():
                        logger.debug(f"[SEARCH] No metadata match for '{needle}', passing to semantic AI")
                        remaining.append(term)
                    else:
                        # Metadata-only build: unresolved free-text is not a match.
                        filtered = []

        return filtered, remaining

    def search_text(
        self, query: str, candidate_paths: Sequence[str], top_k: int = 200, min_score: float = 0.0
    ) -> list[SearchHit]:
        if not semantic_embeddings_enabled():
            return []
        raw_query = (query or "").strip()
        if not raw_query:
            return []
        canonical_to_original = {}
        signature_to_original = {}
        for p in candidate_paths:
            if not p or not os.path.isfile(p):
                continue
            try:
                canonical = self._canonical_path(p)
                st = os.stat(canonical)
                canonical_to_original[canonical] = p
                signature_to_original[self._file_signature_from_stat(canonical, st)] = p
            except Exception:
                canonical_to_original[self._canonical_path(p)] = p
        rows = self._fetch_rows_for_paths(candidate_paths)
        if not rows:
            return []
        rows, semantic_query = self._apply_filters(rows, raw_query)
        if not rows:
            return []

        # Filter-only query: return deterministic list without semantic ranking.
        if not semantic_query:
            hits = [
                SearchHit(
                    file_path=str(
                        signature_to_original.get(str(r["file_signature"] or ""))
                        or canonical_to_original.get(self._canonical_path(str(r["file_path"])))
                        or str(r["file_path"])
                    ),
                    score=1.0,
                    file_name=str(r["file_name"] or os.path.basename(str(r["file_path"]))),
                    capture_time=str(r["capture_time"] or ""),
                    camera_model=str(r["camera_model"] or ""),
                    lens_model=str(r["lens_model"] or ""),
                    iso=_coerce_int(r["iso"]),
                    gps_lat=_coerce_float(r["gps_lat"]),
                    gps_lon=_coerce_float(r["gps_lon"]),
                    city=str(r["city"] or ""),
                    landmark=str(r["landmark"] or ""),
                    admin1=str(r["admin1"] or ""),
                    country=str(r["country"] or ""),
                    country_code=str(r["country_code"] or ""),
                    face_count=_coerce_int(r["face_count"]),
                )
                for r in rows
            ]
            hits.sort(key=lambda h: h.capture_time, reverse=True)
            return hits[: max(1, int(top_k))]

        query_vecs = self._encode_semantic_query(semantic_query)
        if not query_vecs:
            return []

        scores: list[SearchHit] = []
        for r in rows:
            if not self._row_semantic_ready(r):
                continue
            vec = self._from_blob(cast(bytes, r["embedding"]), _coerce_int(r["dim"]))
            if vec.size == 0:
                continue
            score = self._semantic_similarity(query_vecs, vec)
            # Semantic query should filter out clearly non-matching images.
            # Keep strict filtering (not just ranking) by removing sub-threshold scores.
            if score <= float(min_score):
                continue
            scores.append(
                SearchHit(
                    file_path=str(
                        signature_to_original.get(str(r["file_signature"] or ""))
                        or canonical_to_original.get(self._canonical_path(str(r["file_path"])))
                        or str(r["file_path"])
                    ),
                    score=score,
                    file_name=str(r["file_name"] or os.path.basename(str(r["file_path"]))),
                    capture_time=str(r["capture_time"] or ""),
                    camera_model=str(r["camera_model"] or ""),
                    lens_model=str(r["lens_model"] or ""),
                    iso=_coerce_int(r["iso"]),
                    gps_lat=_coerce_float(r["gps_lat"]),
                    gps_lon=_coerce_float(r["gps_lon"]),
                    city=str(r["city"] or ""),
                    landmark=str(r["landmark"] or ""),
                    admin1=str(r["admin1"] or ""),
                    country=str(r["country"] or ""),
                    country_code=str(r["country_code"] or ""),
                    face_count=_coerce_int(r["face_count"]),
                )
            )
        scores.sort(key=lambda x: x.score, reverse=True)
        return scores[: max(1, int(top_k))]

    def search_metadata_text(
        self, query: str, candidate_paths: Sequence[str], top_k: int = 500, sort_newest: bool = True
    ) -> tuple[list[SearchHit], str]:
        """
        Metadata/EXIF-only search that works without any embedding backend.

        Returns (hits, remaining_semantic_query). If remaining_semantic_query is empty,
        the query was fully satisfied by metadata filters/auto-matches.
        """
        raw_query = (query or "").strip()
        if not raw_query:
            return [], ""

        needs_face = self._query_needs_face_detection(raw_query)
        rows = self._metadata_rows_for_search(candidate_paths, needs_face=needs_face)

        filtered, semantic_query = self._apply_filters(rows, raw_query)
        hits = [
            SearchHit(
                file_path=str(r["file_path"]),
                score=1.0,
                file_name=str(self._row_value(r, "file_name") or os.path.basename(str(r["file_path"]))),
                capture_time=str(r["capture_time"] or ""),
                camera_model=str(r["camera_model"] or ""),
                lens_model=str(r["lens_model"] or ""),
                iso=_coerce_int(r["iso"]),
                gps_lat=_coerce_float(r["gps_lat"]),
                gps_lon=_coerce_float(r["gps_lon"]),
                city=str(r["city"] or ""),
                landmark=str(r["landmark"] or ""),
                admin1=str(r["admin1"] or ""),
                country=str(r["country"] or ""),
                country_code=str(r["country_code"] or ""),
                face_count=_coerce_int(r["face_count"]),
            )
            for r in filtered
        ]
        hits.sort(key=lambda h: h.capture_time, reverse=sort_newest)
        return hits[: max(1, int(top_k))], semantic_query

    def _metadata_rows_for_search(
        self, candidate_paths: Sequence[str], needs_face: bool = False
    ) -> list[dict[str, object]]:
        """Metadata rows for search, with DB-first lookup and fallback EXIF extraction."""
        if not candidate_paths:
            return []

        # Build a canonical→original map so DB rows can be resolved back to the
        # original-case path that self.image_files uses. On Windows the DB stores
        # lowercase paths (_canonical_path lowercases), but the UI file list keeps
        # the original case from os.listdir(). Returning lowercase paths from search
        # causes a mismatch where the gallery cannot locate the files.
        canonical_to_original: dict[str, str] = {}
        for p in candidate_paths:
            if p:
                canonical_to_original[self._canonical_path(p)] = p

        # Bulk fetch all rows that exist in the index
        rows: list[dict[str, object]] = []
        db_rows = self._fetch_rows_for_paths(candidate_paths)
        found_paths: set[str] = set()

        # Rows present in DB (fast path).
        for row in db_rows:
            db_path = str(row["file_path"])
            # Resolve back to original-case path from candidate_paths; fall back to
            # the DB path (already canonical/lowercase) if not found.
            canonical = self._canonical_path(db_path)
            original = canonical_to_original.get(canonical, db_path)
            found_paths.add(self._canonical_path(original))

            face_count = row.get("face_count", None)
            # Never run YuNet/OpenCV on the UI thread during gallery search — use
            # indexed face_count only (NULL → 0). Live detection belongs in indexing.
            if face_count is None:
                face_count = 0

            city = str(row["city"] or "")
            landmark = str(row["landmark"] or "")
            admin1 = str(row["admin1"] or "")
            country = str(row["country"] or "")
            gps_lat = row["gps_lat"]
            gps_lon = row["gps_lon"]

            try:
                from core.upgrade_compat import legacy_search_metadata_sync

                sync_search = legacy_search_metadata_sync()
            except Exception:
                sync_search = True

            if sync_search and not city and gps_lat is not None and gps_lon is not None:
                geo = self._ensure_reverse_geocoder()
                if isinstance(geo, CustomReverseGeocoder):
                    try:
                        with self._rg_lock:
                            recs = geo.search(
                                [
                                    (
                                        _coerce_float(gps_lat) or 0.0,
                                        _coerce_float(gps_lon) or 0.0,
                                    )
                                ],
                                mode=1,
                            )
                        if recs:
                            rec = recs[0] or {}
                            city_name = str(rec.get("name", "") or "")
                            alternates = str(rec.get("alternates", "") or "")
                            city = f"{city_name}, {alternates}" if alternates else city_name
                            landmark = str(rec.get("landmark", "") or "")
                            admin1 = str(rec.get("admin1", "") or "")
                            cc = str(rec.get("cc", "") or "").upper()
                            country = self._country_name_from_code(cc)
                            self._conn.execute(
                                "UPDATE semantic_index SET city=?, landmark=?, admin1=?, country=?, country_code=? WHERE file_path=?",
                                (city, landmark, admin1, country, cc, original),
                            )
                            self._conn.commit()
                    except Exception:
                        pass
            elif not sync_search and not city and gps_lat is not None and gps_lon is not None:
                self._enqueue_metadata_backfill([], geocode_paths=[original])

            rows.append(
                {
                    "file_path": original,
                    "file_name": str(row["file_name"] or os.path.basename(original)),
                    "capture_time": str(row["capture_time"] or ""),
                    "camera_model": str(row["camera_model"] or ""),
                    "lens_model": str(row["lens_model"] or ""),
                    "iso": _coerce_int(row["iso"]),
                    "width": _coerce_int(row["width"]),
                    "height": _coerce_int(row["height"]),
                    "gps_lat": gps_lat,
                    "gps_lon": gps_lon,
                    "city": city,
                    "landmark": landmark,
                    "admin1": admin1,
                    "country": country,
                    "country_code": str(row["country_code"] or ""),
                    "face_count": _coerce_int(face_count),
                }
            )
        try:
            from core.upgrade_compat import legacy_search_metadata_sync

            sync_search = legacy_search_metadata_sync()
        except Exception:
            sync_search = True

        missing_paths = [
            p
            for p in candidate_paths
            if p
            and os.path.isfile(p)
            and self._canonical_path(p) not in found_paths
        ]

        if sync_search:
            for p in missing_paths:
                meta = self._extract_exif_brief(p, include_face=False)
                rows.append(
                    {
                        "file_path": p,
                        "file_name": os.path.basename(p),
                        "capture_time": str(meta.get("capture_time") or ""),
                        "camera_model": str(meta.get("camera_model") or ""),
                        "lens_model": str(meta.get("lens_model") or ""),
                        "iso": self._to_int(meta.get("iso")),
                        "width": self._to_int(meta.get("width")),
                        "height": self._to_int(meta.get("height")),
                        "gps_lat": meta.get("gps_lat"),
                        "gps_lon": meta.get("gps_lon"),
                        "city": str(meta.get("city") or ""),
                        "landmark": str(meta.get("landmark") or ""),
                        "admin1": str(meta.get("admin1") or ""),
                        "country": str(meta.get("country") or ""),
                        "country_code": str(meta.get("country_code") or ""),
                        "face_count": self._to_int(meta.get("face_count")),
                    }
                )
        else:
            if missing_paths:
                cached_rows = self._metadata_rows_from_exif_cache(missing_paths)
                rows.extend(cached_rows)
                cached_canonical = {
                    self._canonical_path(str(r["file_path"])) for r in cached_rows
                }
                still_missing = [
                    p
                    for p in missing_paths
                    if self._canonical_path(p) not in cached_canonical
                ]
                if still_missing:
                    self._enqueue_metadata_backfill(still_missing)
        return rows

    def _metadata_rows_from_exif_cache(
        self, file_paths: Sequence[str]
    ) -> list[dict[str, object]]:
        """Read-only EXIF cache rows for search (no disk extract on UI thread)."""
        out: list[dict[str, object]] = []
        if not file_paths:
            return out
        try:
            from image_cache import get_image_cache

            cache = get_image_cache()
            cached = cache.get_multiple_exif(list(file_paths), fast_mode=True)
        except Exception:
            return out
        for p in file_paths:
            exif = cached.get(p)
            if not isinstance(exif, dict):
                continue
            raw_exif_data = exif.get("exif_data")
            exif_data: dict[str, object] = (
                raw_exif_data if isinstance(raw_exif_data, dict) else {}
            )
            out.append(
                {
                    "file_path": p,
                    "file_name": os.path.basename(p),
                    "capture_time": str(exif.get("capture_time") or exif_data.get("capture_time") or ""),
                    "camera_model": str(exif.get("camera_model") or exif_data.get("camera_model") or ""),
                    "lens_model": str(exif_data.get("lens_model") or ""),
                    "iso": self._to_int(exif_data.get("iso")),
                    "width": self._to_int(exif.get("original_width") or exif_data.get("original_width")),
                    "height": self._to_int(exif.get("original_height") or exif_data.get("original_height")),
                    "gps_lat": exif_data.get("gps_lat"),
                    "gps_lon": exif_data.get("gps_lon"),
                    "city": str(exif_data.get("city") or ""),
                    "landmark": str(exif_data.get("landmark") or ""),
                    "admin1": str(exif_data.get("admin1") or ""),
                    "country": str(exif_data.get("country") or ""),
                    "country_code": str(exif_data.get("country_code") or ""),
                    "face_count": 0,
                }
            )
        return out

    def _store_face_count(
        self,
        file_path: str,
        face_count: int,
        conn: sqlite3.Connection | None = None,
        *,
        commit: bool = True,
    ) -> None:
        try:
            db = conn if conn is not None else self._conn
            canonical = self._canonical_path(file_path)
            st = os.stat(canonical)
            signature = self._file_signature_from_stat(canonical, st)
            aliases = self._path_aliases(canonical)
            placeholders = ",".join(["?"] * len(aliases))
            db.execute(
                f"""
                UPDATE semantic_index
                SET face_count = ?
                WHERE (file_signature = ? AND model_name = ?)
                   OR file_path IN ({placeholders})
                """,
                [int(face_count), signature, self.model_name, *aliases],
            )
            if commit:
                db.commit()
        except Exception:
            pass

    @staticmethod
    def _query_needs_face_detection(query: str) -> bool:
        return bool(
            re.search(
                r"\b(?:has:faces?|has:people|has:person|has:humans?"
                r"|no:faces?|no:people|no:person|no:humans?"
                r"|faces?|people|person|humans?)\b",
                query or "",
                flags=re.I,
            )
        )
