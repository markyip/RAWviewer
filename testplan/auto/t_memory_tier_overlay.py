#!/usr/bin/env python3
"""RAM-tier overlay must beat shared scroll prefetch on low-end machines.

Shared defaults set idle-preload 120 / 12 preload threads for snappy Plus
gallery fill. Those used to be applied with setdefault *before* the RAM tier,
so an 8 GB Air inherited the 24 GB numbers and the tier caps never stuck.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


_KEYS = (
    "RAWVIEWER_GALLERY_IDLE_PRELOAD_BATCH",
    "RAWVIEWER_GALLERY_IDLE_PRELOAD_MS",
    "RAWVIEWER_PRELOAD_THREADS",
    "RAWVIEWER_GALLERY_HOVER_PREFETCH",
    "RAWVIEWER_MEMORY_PREVIEW_MAX",
    "RAWVIEWER_MEMORY_TIER_AUTO",
    "RAWVIEWER_EDIT_BASE_MAX_EDGE",
    "RAWVIEWER_ADJUST_FAST_PREVIEW_MAX_EDGE",
    "RAWVIEWER_EDIT_BASE_LRU_MAX",
    "RAWVIEWER_BG_POOL_THREADS",
)


def _clear():
    import rawviewer_profile as rp

    rp._USER_RAWVIEWER_ENV = None
    for key in _KEYS:
        os.environ.pop(key, None)


def main() -> int:
    import rawviewer_profile as rp

    _clear()
    os.environ["RAWVIEWER_MEMORY_TIER_AUTO"] = "1"
    rp.apply_profile_runtime_defaults()
    check(
        "shared prefetch still sets the fast-machine idle batch",
        os.environ.get("RAWVIEWER_GALLERY_IDLE_PRELOAD_BATCH") == "120",
        os.environ.get("RAWVIEWER_GALLERY_IDLE_PRELOAD_BATCH"),
    )

    orig_ram = rp.system_total_ram_gb
    orig_note = rp._write_memory_tier_note
    rp.system_total_ram_gb = lambda: 8.0
    rp._write_memory_tier_note = lambda *a, **k: None
    try:
        tier = rp.apply_memory_tier_defaults()
        check("8 GB classifies as low", tier == "low", tier)
        check(
            "low tier overlays idle batch (not left at 120)",
            os.environ.get("RAWVIEWER_GALLERY_IDLE_PRELOAD_BATCH") == "16",
            os.environ.get("RAWVIEWER_GALLERY_IDLE_PRELOAD_BATCH"),
        )
        check(
            "low tier overlays preload threads (not left at 12)",
            os.environ.get("RAWVIEWER_PRELOAD_THREADS") == "4",
            os.environ.get("RAWVIEWER_PRELOAD_THREADS"),
        )
        check(
            "low tier turns off hover full-res decode",
            os.environ.get("RAWVIEWER_GALLERY_HOVER_PREFETCH") == "0",
            os.environ.get("RAWVIEWER_GALLERY_HOVER_PREFETCH"),
        )
        check(
            "low preview max overlays the Standard 2304 default",
            os.environ.get("RAWVIEWER_MEMORY_PREVIEW_MAX") == "1280",
            os.environ.get("RAWVIEWER_MEMORY_PREVIEW_MAX"),
        )
        check(
            "low tier caps the working Adjust edit base",
            os.environ.get("RAWVIEWER_EDIT_BASE_MAX_EDGE") == "1600",
            os.environ.get("RAWVIEWER_EDIT_BASE_MAX_EDGE"),
        )
        check(
            "low tier shrinks live-drag preview edge",
            os.environ.get("RAWVIEWER_ADJUST_FAST_PREVIEW_MAX_EDGE") == "480",
            os.environ.get("RAWVIEWER_ADJUST_FAST_PREVIEW_MAX_EDGE"),
        )
    finally:
        rp.system_total_ram_gb = orig_ram
        rp._write_memory_tier_note = orig_note

    _clear()
    os.environ["RAWVIEWER_MEMORY_TIER_AUTO"] = "1"
    os.environ["RAWVIEWER_GALLERY_IDLE_PRELOAD_BATCH"] = "99"
    rp._USER_RAWVIEWER_ENV = None
    rp.system_total_ram_gb = lambda: 8.0
    rp._write_memory_tier_note = lambda *a, **k: None
    try:
        rp.apply_profile_runtime_defaults()
        rp.apply_memory_tier_defaults()
        check(
            "a user-exported idle batch is not overwritten",
            os.environ.get("RAWVIEWER_GALLERY_IDLE_PRELOAD_BATCH") == "99",
            os.environ.get("RAWVIEWER_GALLERY_IDLE_PRELOAD_BATCH"),
        )
    finally:
        rp.system_total_ram_gb = orig_ram
        rp._write_memory_tier_note = orig_note
        _clear()

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
