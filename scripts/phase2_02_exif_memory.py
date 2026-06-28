"""Phase 2 item 2: EXIF verified_orientation fast path + UI memory-first reads."""
from __future__ import annotations

import os
import sys
import tempfile
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PyQt6.QtCore import QCoreApplication
from PyQt6.QtWidgets import QApplication

if QCoreApplication.instance() is None:
    QApplication([])

from enhanced_raw_processor import (
    RAW_EXIF_SENSOR_META_VER,
    cached_raw_exif_orientation_trustworthy,
)
from image_cache import ImageCache


def test_trustworthy_skips_probe_when_verified() -> None:
    raw_path = "/tmp/phase2_test.CR3"
    cached = {
        "orientation": 6,
        "verified_orientation": True,
        "raw_exif_sensor_meta_ver": RAW_EXIF_SENSOR_META_VER,
    }
    with patch(
        "enhanced_raw_processor.probe_effective_raw_orientation",
        side_effect=AssertionError("probe must not run when verified_orientation=True"),
    ):
        assert cached_raw_exif_orientation_trustworthy(raw_path, cached) is True


def test_trustworthy_calls_probe_when_unverified() -> None:
    raw_path = "/tmp/phase2_test.CR3"
    cached = {
        "orientation": 1,
        "raw_exif_sensor_meta_ver": RAW_EXIF_SENSOR_META_VER,
    }
    calls = []

    def _probe(path, raw_object=None):
        calls.append(path)
        return 1

    with patch("enhanced_raw_processor.probe_effective_raw_orientation", side_effect=_probe):
        with patch("common_image_loader.is_raw_file", return_value=True):
            assert cached_raw_exif_orientation_trustworthy(raw_path, cached) is True
    assert calls == [raw_path]


def test_get_exif_for_ui_memory_and_verified_disk() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache = ImageCache(cache_dir=tmp, persistent_cache_enabled=True)
        sample = os.path.join(tmp, "IMG_0001.CR3")
        with open(sample, "wb") as f:
            f.write(b"raw")

        verified = {
            "orientation": 1,
            "verified_orientation": True,
            "raw_exif_sensor_meta_ver": RAW_EXIF_SENSOR_META_VER,
            "exif_data": {"EXIF FocalLength": "50"},
            "original_width": 6000,
            "original_height": 4000,
        }
        cache.put_exif(sample, verified, persist_disk=True)
        cache.flush_exif_disk_cache()

        # Memory hit — no probe
        with patch(
            "enhanced_raw_processor.probe_effective_raw_orientation",
            side_effect=AssertionError("probe must not run for verified UI read"),
        ):
            got = cache.get_exif_for_ui(sample)
        assert got is not None
        assert got.get("original_width") == 6000

        # Memory miss, disk hit (verified_orientation is memory-only; may probe once)
        cache.exif_memory_cache.remove(sample)
        with patch("enhanced_raw_processor.probe_effective_raw_orientation", return_value=1):
            with patch("common_image_loader.is_raw_file", return_value=True):
                got = cache.get_exif_for_ui(sample)
        assert got is not None
        assert got.get("original_width") == 6000


def test_get_exif_memory_only_no_sqlite() -> None:
    cache = ImageCache()
    sample = "/tmp/only_mem.CR3"
    stub = {
        "orientation": 1,
        "verified_orientation": True,
        "raw_exif_sensor_meta_ver": RAW_EXIF_SENSOR_META_VER,
    }
    cache.exif_memory_cache.put(sample, stub)
    assert cache.get_exif_memory_only(sample) == stub
    assert cache.get_exif(sample) is not None


if __name__ == "__main__":
    test_trustworthy_skips_probe_when_verified()
    test_trustworthy_calls_probe_when_unverified()
    test_get_exif_for_ui_memory_and_verified_disk()
    test_get_exif_memory_only_no_sqlite()
    print("OK: Phase 2 item 2 (EXIF memory fast path) tests passed")
