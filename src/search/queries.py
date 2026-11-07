"""Semantic query: capture times, location, reverse geocoding, text search."""

from __future__ import annotations

import contextlib
import os
import sqlite3
import threading
from collections.abc import Sequence

import numpy as np
import pycountry

_semantic_sort_read_local = threading.local()


def _semantic_sort_read_conn(db_path: str) -> sqlite3.Connection:
    """Per-thread read connection for capture_time lookups during background folder sort."""
    conn = getattr(_semantic_sort_read_local, "conn", None)
    if conn is None or getattr(_semantic_sort_read_local, "db_path", None) != db_path:
        conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=60000")
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute("PRAGMA query_only=ON")
        _semantic_sort_read_local.conn = conn
        _semantic_sort_read_local.db_path = db_path
    return conn


def get_semantic_capture_times_for_paths(
    paths: Sequence[str],
    db_path: str | None = None,
) -> dict[str, str]:
    """
    Thread-safe bulk read of capture_time from semantic_index.db.

    Used by folder-load / refinement workers (not only the UI thread) so navigation
    order matches the semantic index instead of degrading to mtime when EXIF cache is cold.
    """
    if not paths:
        return {}
    from search.engine import SemanticImageIndex
    if db_path is None:
        cache_dir = os.path.expanduser("~/.rawviewer_cache")
        db_path = os.path.join(cache_dir, "semantic_index.db")
    if not os.path.isfile(db_path):
        return {}

    canonical_to_original: dict[str, str] = {}
    unique_canonical: list[str] = []
    for p in paths:
        if not p:
            continue
        cp = SemanticImageIndex._canonical_path(p)
        if cp not in canonical_to_original:
            canonical_to_original[cp] = p
            unique_canonical.append(cp)
    if not unique_canonical:
        return {}

    conn = _semantic_sort_read_conn(db_path)
    result_map: dict[str, str] = {}
    chunk_size = 500
    for i in range(0, len(unique_canonical), chunk_size):
        chunk = unique_canonical[i : i + chunk_size]
        placeholders = ",".join(["?"] * len(chunk))
        rows = conn.execute(
            f"""
            SELECT file_path, capture_time
            FROM semantic_index
            WHERE lower(file_path) IN ({placeholders})
              AND capture_time IS NOT NULL AND capture_time != ''
            ORDER BY rowid DESC
            """,
            chunk,
        ).fetchall()
        for fp, ct in rows:
            if fp in result_map:
                continue
            orig = canonical_to_original.get(fp)
            if orig and ct:
                result_map[orig] = str(ct)
    return result_map


def get_semantic_capture_times_for_folder(
    folder_path: str,
    db_path: str | None = None,
) -> dict[str, str]:
    """Bulk read capture_time for all indexed files under folder_path (any thread)."""
    from search.engine import SemanticImageIndex
    if not folder_path:
        return {}
    if db_path is None:
        cache_dir = os.path.expanduser("~/.rawviewer_cache")
        db_path = os.path.join(cache_dir, "semantic_index.db")
    if not os.path.isfile(db_path):
        return {}
    try:
        root = SemanticImageIndex._canonical_path(
            os.path.abspath(folder_path)
        )
    except OSError:
        return {}
    if not root:
        return {}
    root_lower = root.rstrip("\\/").lower()
    prefix = root_lower + "\\%" if os.sep == "\\" else root_lower.replace("\\", "/") + "/%"
    path_expr = (
        "lower(replace(file_path, '/', '\\\\'))"
        if os.sep == "\\"
        else "lower(replace(file_path, '\\\\', '/'))"
    )
    conn = _semantic_sort_read_conn(db_path)
    result_map: dict[str, str] = {}
    rows = conn.execute(
        f"""
        SELECT file_path, capture_time
        FROM semantic_index
        WHERE {path_expr} LIKE ?
          AND capture_time IS NOT NULL AND capture_time != ''
        ORDER BY rowid DESC
        """,
        (prefix,),
    ).fetchall()
    for fp, ct in rows:
        if fp in result_map:
            continue
        if ct:
            result_map[fp] = str(ct)
    return result_map


def _normalize_location_term(term: str) -> str:
    return " ".join((term or "").strip().lower().split())


def _semantic_text_variants(text: str) -> list[str]:
    """CLIP query variants for a semantic term (singular/plural + short prompt)."""
    raw = (text or "").strip()
    if not raw:
        return []
    seen: set[str] = set()
    out: list[str] = []

    def add(s: str) -> None:
        s = (s or "").strip()
        if not s:
            return
        key = s.casefold()
        if key in seen:
            return
        seen.add(key)
        out.append(s)

    add(raw)
    tokens = raw.split()
    if len(tokens) != 1:
        return out
    word = tokens[0]
    low = word.lower()
    if len(low) < 3 or not low.isalpha():
        return out
    if low.endswith("s") and len(low) > 3 and not low.endswith("ss"):
        add(word[:-1])
    elif not low.endswith("s"):
        add(word + "s")
    add(f"a photo of {word}")
    if low.endswith("s") and len(low) > 3 and not low.endswith("ss"):
        add(f"a photo of {word[:-1]}")
    elif not low.endswith("s"):
        add(f"a photo of {word}s")
    return out


# Quick O(1) checks before any geocoder CSV load (contradiction filter).
KNOWN_LOCATION_NAMES: frozenset[str] = frozenset({
    # Countries
    "japan", "korea", "china", "taiwan", "usa", "uk", "canada", "france",
    "germany", "italy", "spain", "russia", "australia", "india", "brazil",
    "mexico", "thailand", "vietnam", "singapore", "malaysia", "indonesia",
    "philippines", "switzerland", "austria", "netherlands", "greece", "turkey",
    "egypt", "south africa", "new zealand", "united states", "united kingdom",
    "south korea", "north korea",
    # Major cities & capitals
    "tokyo", "seoul", "london", "paris", "berlin", "new york", "los angeles",
    "san francisco", "hong kong", "beijing", "shanghai", "bangkok", "sydney", "melbourne", "toronto", "vancouver", "rome", "milan", "madrid",
    "barcelona", "amsterdam", "vienna", "zurich", "geneva", "mumbai", "delhi",
    "bangalore", "jakarta", "manila", "ho chi minh", "hanoi", "taipei", "macao",
    "moscow", "istanbul", "dubai", "abu dhabi", "riyadh", "cairo", "nairobi",
    "chicago", "boston", "seattle", "miami", "munich", "frankfurt", "lyon",
    "prague", "budapest", "warsaw", "stockholm", "oslo", "copenhagen", "helsinki",
    "lisbon", "athens", "dublin", "tel aviv", "mexico city", "buenos aires",
    "sao paulo", "rio de janeiro", "santiago", "lima", "bogota",
    "manchester", "birmingham", "liverpool", "edinburgh", "glasgow", "leeds",
    "bristol", "cardiff", "belfast", "nottingham", "sheffield", "newcastle",
    # Popular travel destinations
    "santorini", "mykonos", "bali", "phuket", "kyoto", "osaka", "nara", "hokkaido",
    "jeju", "busan", "boracay", "cebu", "nha trang", "da nang", "koh samui",
    "krabi", "chiang mai", "luang prabang", "angkor wat", "siem reap", "halong bay",
    "maldives", "fiji", "bora bora", "tahiti", "maui", "honolulu", "oahu", "kauai",
    "ibiza", "mallorca", "tenerife", "capri", "amalfi", "positano", "venice",
    "florence", "tuscany", "provence", "cannes", "nice", "monaco", "st moritz",
    "zermatt", "interlaken", "hallstatt", "salzburg", "innsbruck", "banff",
    "whistler", "yellowstone", "yosemite", "grand canyon", "sedona", "reykjavik",
    "blue lagoon", "cappadocia", "petra", "machu picchu", "cusco", "uyuni",
    "patagonia", "queenstown", "rotorua", "milford sound",
})

_PYCOUNTRY_NAME_INDEX: frozenset[str] | None = None


def _pycountry_name_index() -> frozenset[str]:
    global _PYCOUNTRY_NAME_INDEX
    if _PYCOUNTRY_NAME_INDEX is not None:
        return _PYCOUNTRY_NAME_INDEX
    names: set[str] = set()
    if pycountry is not None:
        try:
            for c in pycountry.countries:
                for attr in ("name", "common_name", "official_name"):
                    raw = getattr(c, attr, None)
                    if raw:
                        tok = _normalize_location_term(str(raw))
                        if len(tok) >= 3:
                            names.add(tok)
        except Exception:
            pass
    _PYCOUNTRY_NAME_INDEX = frozenset(names)
    return _PYCOUNTRY_NAME_INDEX

class CustomReverseGeocoder:
    """GeoNames + Wikidata reverse geocoder with O(log n) nearest-neighbor lookup."""

    _LANDMARK_DIST_SQ = 0.0182  # ~15 km squared degree distance

    def __init__(self, cities: list, landmarks: list | None = None):
        self.cities = cities
        self.landmarks = landmarks or []
        self._city_coords = (
            np.asarray([(float(c[0]), float(c[1])) for c in self.cities], dtype=np.float64)
            if self.cities
            else None
        )
        self._landmark_coords = (
            np.asarray([(float(lm[0]), float(lm[1])) for lm in self.landmarks], dtype=np.float64)
            if self.landmarks
            else None
        )
        self._place_names = self._build_place_name_index()

    @staticmethod
    def _build_place_name_index_from_rows(
        cities: list,
        landmarks: list,
    ) -> frozenset[str]:
        names: set[str] = set()
        for city in cities:
            primary = _normalize_location_term(str(city[2]))
            if len(primary) >= 3:
                names.add(primary)
            alternates = city[5] if len(city) > 5 else ""
            if alternates:
                for alt in str(alternates).split(","):
                    tok = _normalize_location_term(alt)
                    if len(tok) >= 3:
                        names.add(tok)
        for lm in landmarks:
            tok = _normalize_location_term(str(lm[2]))
            if len(tok) >= 3:
                names.add(tok)
        return frozenset(names)

    def _build_place_name_index(self) -> frozenset[str]:
        return self._build_place_name_index_from_rows(self.cities, self.landmarks)

    def has_place_name(self, term: str) -> bool:
        """O(1) exact match against indexed city / alternate / landmark names."""
        needle = _normalize_location_term(term)
        return len(needle) >= 3 and needle in self._place_names

    @staticmethod
    def _nearest_index(
        coords: np.ndarray,
        target_lat: float,
        target_lon: float,
    ) -> tuple[int, float]:
        """Nearest-neighbor by squared degree distance, vectorized over numpy
        rather than a scipy.spatial.cKDTree -- avoids a ~77MB scipy dependency
        for a lookup against a few hundred thousand rows at most (cities500 +
        landmarks), called once per GPS-tagged photo, not in a hot loop."""
        if coords.size == 0:
            return 0, float("inf")
        dist_sq = (coords[:, 0] - target_lat) ** 2 + (coords[:, 1] - target_lon) ** 2
        idx = int(np.argmin(dist_sq))
        return idx, float(dist_sq[idx])

    def search(self, coords: list[tuple[float, float]], mode: int = 1) -> list[dict]:
        if not coords or not self.cities:
            return []
        target_lat, target_lon = float(coords[0][0]), float(coords[0][1])

        city_idx, _ = self._nearest_index(
            self._city_coords if self._city_coords is not None else np.empty((0, 2)),
            target_lat,
            target_lon,
        )
        lat, lon, name, admin1, cc = self.cities[city_idx][:5]
        alternates = self.cities[city_idx][5] if len(self.cities[city_idx]) > 5 else ""
        best_city = {"name": name, "admin1": admin1, "cc": cc, "alternates": alternates}

        landmark_name = ""
        if self.landmarks and self._landmark_coords is not None:
            lm_idx, lm_dist_sq = self._nearest_index(
                self._landmark_coords, target_lat, target_lon
            )
            if lm_dist_sq < self._LANDMARK_DIST_SQ:
                landmark_name = str(self.landmarks[lm_idx][2])

        return [{
            "name": best_city["name"],
            "admin1": best_city["admin1"],
            "cc": best_city["cc"],
            "alternates": best_city["alternates"],
            "landmark": landmark_name,
        }]

