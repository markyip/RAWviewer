"""Phase 1 item 2: EXIF SQLite cache_size + canonical path exact lookups."""
from __future__ import annotations

import os
import re
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from image_cache import (
    PersistentEXIFCache,
    _exif_cache_path_key,
    _exif_sql_folder_like_pattern,
)


def test_no_lower_file_path_queries_in_source() -> None:
    path = os.path.join(os.path.dirname(__file__), "..", "src", "image_cache.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    assert "lower(file_path)" not in src, "EXIF queries must not use lower(file_path) scans"
    assert "PRAGMA cache_size=-65536" in src


def test_exact_path_get_put_remove() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache = PersistentEXIFCache(cache_dir=tmp)
        sample = os.path.join(tmp, "Sample.CR3")
        with open(sample, "wb") as f:
            f.write(b"raw")

        exif = {
            "orientation": 1,
            "camera_make": "Canon",
            "camera_model": "R5",
            "exif_data": {"capture_time": "2020:07:06 11:07:16"},
            "capture_time": "2020:07:06 11:07:16",
            "original_width": 5496,
            "original_height": 3670,
        }
        cache.put(sample, exif, commit=True)

        # Lookup with different casing should miss unless normcase maps it (macOS preserves case).
        key = _exif_cache_path_key(sample)
        got = cache.get(sample)
        assert got is not None
        assert got["camera_model"] == "R5"

        bulk = cache.get_multiple([sample])
        assert sample in bulk or key in bulk or len(bulk) == 1

        times = cache.get_capture_times_bulk([sample])
        assert times, "bulk capture_time lookup failed"

        folder_times = cache.get_capture_times_for_folder(tmp)
        assert folder_times, "folder LIKE lookup failed"

        assert cache.remove(sample)
        assert cache.get(sample) is None


def test_folder_like_pattern_matches_storage_key() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        folder = os.path.join(tmp, "Canon_Sample")
        os.makedirs(folder)
        sample = os.path.join(folder, "IMG_0001.CR3")
        with open(sample, "wb") as f:
            f.write(b"x")
        key = _exif_cache_path_key(sample)
        pattern = _exif_sql_folder_like_pattern(folder)
        assert key.startswith(pattern[:-1]) or pattern.rstrip("%") in key


if __name__ == "__main__":
    test_no_lower_file_path_queries_in_source()
    test_exact_path_get_put_remove()
    test_folder_like_pattern_matches_storage_key()
    print("OK: Phase 1 item 2 (EXIF SQLite) tests passed")
