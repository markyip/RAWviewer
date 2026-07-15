#!/usr/bin/env python3
"""Cold gallery-thumbnail speed guard.

Two fixes went into the embedded-preview grid-thumbnail path:
  1. extract_previews_via_tiff_parse(path, max_size) now decodes the
     smallest available embedded preview that's still big enough for the
     request, instead of always the largest -- some RAW formats (e.g. Sony
     ARW) embed a near-sensor-resolution JPEG (20-40MP) alongside a much
     smaller one, and decoding the huge one for a 512px gallery tile wastes
     ~10x the CPU for the same visual result (it gets downsampled either
     way).
  2. enhanced_raw_processor.extract_exif_data no longer opens rawpy.imread()
     twice per file (once for dimensions/flip, once more for the embedded-
     preview orientation fallback) -- it now reuses the same open RawPy
     object for both, since each open is a full LibRaw parse.

Uses the same real golden ARW already relied on elsewhere in this suite
(scripts/fast_raw_decode_parity_gate.py) so this only runs where that file
is actually present; skips (not fails) otherwise, matching this repo's
convention for machine-local golden fixtures.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []
GOLDEN_ARW = os.environ.get("RAWVIEWER_TEST_ASSETS_ARW", "/tmp/RAW_Sample/DSC01089.ARW")


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    if not os.path.isfile(GOLDEN_ARW):
        print(f"SKIP  golden file not present on this machine: {GOLDEN_ARW}")
        return 0

    from enhanced_raw_processor import extract_previews_via_tiff_parse, get_jpeg_dimensions

    full = extract_previews_via_tiff_parse(GOLDEN_ARW, 0)
    small = extract_previews_via_tiff_parse(GOLDEN_ARW, 512)
    check("full-quality request (max_size=0) returns exactly one candidate", len(full) == 1)
    check("gallery-tile request (max_size=512) returns exactly one candidate", len(small) == 1)
    if full and small:
        full_dims = get_jpeg_dimensions(full[0])
        small_dims = get_jpeg_dimensions(small[0])
        check(
            "gallery-tile request picks a smaller source JPEG than full-quality",
            small_dims is not None and full_dims is not None and (small_dims[0] * small_dims[1]) < (full_dims[0] * full_dims[1]),
            f"full={full_dims} small={small_dims}",
        )
        check(
            "gallery-tile source is still >= the requested 512px long edge",
            small_dims is not None and max(small_dims) >= 512,
            f"small={small_dims}",
        )

    # Cold-path timing guard: clear this file's cached tiers so the timing
    # reflects a real cold decode, not a cache hit. Generous threshold (not
    # a tight perf-baseline-style regression gate) since this also covers
    # disk I/O variance -- it exists to catch a gross regression (e.g. the
    # duplicate-rawpy.imread bug coming back), not to enforce a tight bound.
    from image_cache import get_image_cache
    cache = get_image_cache()
    try:
        cache.exif_cache.remove(GOLDEN_ARW)
        cache.exif_memory_cache.remove(GOLDEN_ARW)
    except Exception:
        pass
    for tier_remove in ("thumbnail_cache", "grid_cache", "preview_cache"):
        tier = getattr(cache, tier_remove, None)
        if tier is not None:
            try:
                tier.remove(GOLDEN_ARW)
            except Exception:
                pass

    from common_image_loader import load_raw_preview_array

    t0 = time.perf_counter()
    arr = load_raw_preview_array(GOLDEN_ARW, max_size=512)
    elapsed_ms = 1000 * (time.perf_counter() - t0)
    check("cold thumbnail decode succeeds", arr is not None)
    check(
        "cold thumbnail decode completes well under the pre-fix ~180ms/file baseline",
        elapsed_ms < 150.0,
        f"{elapsed_ms:.1f}ms",
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
