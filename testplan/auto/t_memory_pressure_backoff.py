#!/usr/bin/env python3
"""Memory-pressure backoff is separate from installed-RAM tiering.

Mocks OS samples. Does not need a real low-RAM machine.
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


def _sample(*, available_gb, percent, total_gb=16.0, win_load=None, mac_mb=None):
    import memory_pressure as mp

    avail = int(available_gb * (1024**3))
    total = int(total_gb * (1024**3))
    mac = None if mac_mb is None else int(mac_mb * 1024 * 1024)
    return mp.MemorySample(
        total_bytes=total,
        available_bytes=avail,
        percent_used=float(percent),
        windows_memory_load=win_load,
        macos_available_bytes=mac,
    )


def main() -> int:
    import memory_pressure as mp
    import rawviewer_profile as rp

    saved = {
        k: os.environ.get(k)
        for k in list(os.environ)
        if k.startswith("RAWVIEWER_MEMORY_PRESSURE")
        or k in (
            "RAWVIEWER_GALLERY_IDLE_PRELOAD_BATCH",
            "RAWVIEWER_UNPACK_CONCURRENCY",
            "RAWVIEWER_EDIT_BASE_MAX_EDGE",
            "RAWVIEWER_EDIT_BASE_LRU_MAX",
            "RAWVIEWER_MEMORY_PRESSURE_BACKOFF",
            "RAWVIEWER_MEMORY_PRESSURE_DEBOUNCE_SAMPLES",
        )
    }

    def _restore():
        mp.reset_for_tests()
        for key, val in saved.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
        for key in (
            "RAWVIEWER_GALLERY_IDLE_PRELOAD_BATCH",
            "RAWVIEWER_UNPACK_CONCURRENCY",
            "RAWVIEWER_EDIT_BASE_MAX_EDGE",
            "RAWVIEWER_EDIT_BASE_LRU_MAX",
            "RAWVIEWER_PRELOAD_THREADS",
            "RAWVIEWER_GALLERY_HOVER_PREFETCH",
            "RAWVIEWER_IDLE_DISPLAY_PREFETCH",
        ):
            if key not in saved:
                os.environ.pop(key, None)

    try:
        os.environ["RAWVIEWER_MEMORY_PRESSURE_DEBOUNCE_SAMPLES"] = "3"
        os.environ["RAWVIEWER_MEMORY_PRESSURE_BACKOFF"] = "1"
        mp.reset_for_tests()

        none = _sample(available_gb=8.0, percent=50.0)
        warn = _sample(available_gb=3.0, percent=86.0)
        crit = _sample(available_gb=1.0, percent=93.0)

        check("ample RAM is none", mp.classify_sample(none) == "none", mp.classify_sample(none))
        check("low available is warn", mp.classify_sample(warn) == "warn", mp.classify_sample(warn))
        check("very low available is critical", mp.classify_sample(crit) == "critical", mp.classify_sample(crit))

        mp.reset_for_tests()
        mp.observe(warn)
        mp.observe(warn)
        check(
            "two warn samples do not latch (debounce)",
            mp.latched_level() == "none",
            mp.latched_level(),
        )
        mp.observe(warn)
        check(
            "three warn samples latch warn",
            mp.latched_level() == "warn",
            mp.latched_level(),
        )

        mp.observe(none)
        mp.observe(none)
        mp.observe(none)
        check(
            "recovery samples do not unlatch mid-session",
            mp.latched_level() == "warn",
            mp.latched_level(),
        )

        mp.observe(crit)
        mp.observe(crit)
        check(
            "two critical samples do not yet raise latch",
            mp.latched_level() == "warn",
            mp.latched_level(),
        )
        mp.observe(crit)
        check(
            "three critical samples latch critical",
            mp.latched_level() == "critical",
            mp.latched_level(),
        )
        mp.observe(none)
        mp.observe(none)
        mp.observe(none)
        check(
            "critical stays latched after plenty of none samples",
            mp.latched_level() == "critical",
            mp.latched_level(),
        )

        mp.reset_for_tests()
        os.environ["RAWVIEWER_EDIT_BASE_MAX_EDGE"] = "2560"
        os.environ["RAWVIEWER_GALLERY_IDLE_PRELOAD_BATCH"] = "48"
        os.environ["RAWVIEWER_UNPACK_CONCURRENCY"] = "3"
        os.environ["RAWVIEWER_EDIT_BASE_LRU_MAX"] = "2"
        for _ in range(3):
            mp.observe(warn)
        applied = mp.apply_backoff("warn")
        check("apply_backoff returns warn", applied == "warn", applied)
        check(
            "warn shrinks idle preload batch",
            int(os.environ["RAWVIEWER_GALLERY_IDLE_PRELOAD_BATCH"]) <= 16,
            os.environ.get("RAWVIEWER_GALLERY_IDLE_PRELOAD_BATCH"),
        )
        check(
            "warn caps unpack concurrency at 1",
            os.environ.get("RAWVIEWER_UNPACK_CONCURRENCY") == "1",
            os.environ.get("RAWVIEWER_UNPACK_CONCURRENCY"),
        )
        check(
            "warn does not change EDIT_BASE_MAX_EDGE",
            os.environ.get("RAWVIEWER_EDIT_BASE_MAX_EDGE") == "2560",
            os.environ.get("RAWVIEWER_EDIT_BASE_MAX_EDGE"),
        )
        check("warn halves thumb budget scale", mp.thumb_budget_scale() == 0.5, str(mp.thumb_budget_scale()))
        check(
            "warn halves full-image budget scale",
            mp.full_image_budget_scale() == 0.5,
            str(mp.full_image_budget_scale()),
        )
        check(
            "scale_thumb_budget_bytes shrinks 384 MiB to 192 MiB",
            mp.scale_thumb_budget_bytes(384 * 1024 * 1024) == 192 * 1024 * 1024,
            str(mp.scale_thumb_budget_bytes(384 * 1024 * 1024)),
        )

        before_batch = os.environ.get("RAWVIEWER_GALLERY_IDLE_PRELOAD_BATCH")
        mp.apply_backoff("warn")
        check(
            "second warn apply is a no-op (no flap)",
            os.environ.get("RAWVIEWER_GALLERY_IDLE_PRELOAD_BATCH") == before_batch,
            os.environ.get("RAWVIEWER_GALLERY_IDLE_PRELOAD_BATCH"),
        )

        mp.apply_backoff("critical")
        check("critical further shrinks thumb scale", mp.thumb_budget_scale() == 0.25, str(mp.thumb_budget_scale()))
        check(
            "critical still leaves EDIT_BASE_MAX_EDGE alone",
            os.environ.get("RAWVIEWER_EDIT_BASE_MAX_EDGE") == "2560",
            os.environ.get("RAWVIEWER_EDIT_BASE_MAX_EDGE"),
        )
        check(
            "critical idle batch is <= 8",
            int(os.environ.get("RAWVIEWER_GALLERY_IDLE_PRELOAD_BATCH", "99")) <= 8,
            os.environ.get("RAWVIEWER_GALLERY_IDLE_PRELOAD_BATCH"),
        )

        mp.reset_for_tests()
        check(
            "installed 32 GB still classifies as ultra after backoff machinery",
            rp.classify_memory_tier(32.0) == "ultra",
            rp.classify_memory_tier(32.0),
        )
        check(
            "installed 8 GB still classifies as low",
            rp.classify_memory_tier(8.0) == "low",
            rp.classify_memory_tier(8.0),
        )
        check(
            "classify_memory_tier ignores available-RAM kwargs (call signature)",
            rp.classify_memory_tier(24.0) == "high",
            rp.classify_memory_tier(24.0),
        )

        os.environ["RAWVIEWER_MEMORY_PRESSURE_BACKOFF"] = "0"
        mp.reset_for_tests()
        for _ in range(5):
            mp.observe(crit)
        check(
            "kill switch never latches",
            mp.latched_level() == "none",
            mp.latched_level(),
        )
        os.environ["RAWVIEWER_EDIT_BASE_MAX_EDGE"] = "2048"
        os.environ["RAWVIEWER_GALLERY_IDLE_PRELOAD_BATCH"] = "48"
        mp.apply_backoff("critical")
        check(
            "kill switch apply_backoff does not cap idle batch",
            os.environ.get("RAWVIEWER_GALLERY_IDLE_PRELOAD_BATCH") == "48",
            os.environ.get("RAWVIEWER_GALLERY_IDLE_PRELOAD_BATCH"),
        )
        check(
            "kill switch leaves EDIT_BASE_MAX_EDGE",
            os.environ.get("RAWVIEWER_EDIT_BASE_MAX_EDGE") == "2048",
            os.environ.get("RAWVIEWER_EDIT_BASE_MAX_EDGE"),
        )

        os.environ["RAWVIEWER_MEMORY_PRESSURE_BACKOFF"] = "1"
        mp.reset_for_tests()
        win_crit = _sample(available_gb=8.0, percent=50.0, win_load=95)
        check(
            "Windows dwMemoryLoad 95 is critical",
            mp.classify_sample(win_crit) == "critical",
            mp.classify_sample(win_crit),
        )
        mac_warn = _sample(available_gb=8.0, percent=50.0, mac_mb=800)
        check(
            "macOS os_proc_available_memory 800 MiB is warn",
            mp.classify_sample(mac_warn) == "warn",
            mp.classify_sample(mac_warn),
        )

        class _DummyCache:
            def __init__(self):
                self.max_memory_mb = 800
                self._tier_max_memory_mb = 800

                class _LRU:
                    def __init__(self):
                        self.max_size = 16

                    def shrink_to_size(self, n):
                        self.max_size = n

                self.full_image_cache = _LRU()

            def _enforce_full_image_byte_budget(self):
                self.enforced = True

            def apply_pressure_budget_scale(self, scale, *, min_mb=128):
                original = self._tier_max_memory_mb
                new_mb = max(int(min_mb), int(original * float(scale)))
                self.max_memory_mb = min(int(self.max_memory_mb), new_mb)
                self.full_image_cache.shrink_to_size(
                    min(self.full_image_cache.max_size, max(2, int(self.full_image_cache.max_size * float(scale))))
                )
                self._enforce_full_image_byte_budget()
                return int(self.max_memory_mb)

        dummy = _DummyCache()
        dummy.apply_pressure_budget_scale(0.5)
        check("dummy cache budget shrinks", dummy.max_memory_mb == 400, str(dummy.max_memory_mb))
        dummy.apply_pressure_budget_scale(0.25)
        check(
            "dummy cache budget never rises",
            dummy.max_memory_mb == 200,
            str(dummy.max_memory_mb),
        )
        dummy.apply_pressure_budget_scale(1.0)
        check(
            "scale 1.0 still cannot raise a already-shrunk budget",
            dummy.max_memory_mb == 200,
            str(dummy.max_memory_mb),
        )

        from unified_image_processor import trim_edit_base_lru, _EDIT_BASE_LRU, _EDIT_BASE_LRU_LOCK

        with _EDIT_BASE_LRU_LOCK:
            _EDIT_BASE_LRU.clear()
            _EDIT_BASE_LRU["a"] = (type("A", (), {"nbytes": 10})(),)
            _EDIT_BASE_LRU["b"] = (type("B", (), {"nbytes": 20})(),)
        freed = trim_edit_base_lru(1)
        with _EDIT_BASE_LRU_LOCK:
            left = list(_EDIT_BASE_LRU.keys())
            _EDIT_BASE_LRU.clear()
        check("trim_edit_base_lru keeps one newest entry", left == ["b"], str(left))
        check("trim_edit_base_lru reports bytes", freed == 10, str(freed))

    finally:
        _restore()

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
