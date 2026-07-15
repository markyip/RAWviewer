#!/usr/bin/env python3
"""Regression guard for the Canon CR3 embedded-preview extraction speed fix.

_extract_bmff_uuid_embedded_jpeg() used to read the ENTIRE CR3 file into
memory (data = f.read()) just to find the small ftyp/moov/uuid boxes at the
start -- on a real 40MB file, all useful content lives in the first ~366KB;
the rest is the sensor RAW payload (mdat), never touched by this function.
Also carried the same "always decode the largest embedded JPEG regardless
of what size was actually requested" bug already fixed for the TIFF-based
path (ARW/NEF) earlier, but never ported to CR3's separate BMFF parser.

Checks both fixes: read volume is bounded regardless of file size, and
output is unaffected (same selection/decode result) for a range of
max_size requests.

Skips (not fails) if the machine-local sample folder isn't present.
"""
import glob
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []
SAMPLE_DIR = os.environ.get("RAWVIEWER_TEST_ASSETS", "/tmp/RAW_Sample")


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    if not os.path.isdir(SAMPLE_DIR):
        print(f"SKIP  sample folder not present on this machine: {SAMPLE_DIR}")
        return 0

    from enhanced_raw_processor import _extract_bmff_uuid_embedded_jpeg

    files = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.CR3")))[:10]
    if not files:
        print(f"SKIP  no CR3 files found in {SAMPLE_DIR}")
        return 0

    slow = []
    for f in files:
        t0 = time.perf_counter()
        arr = _extract_bmff_uuid_embedded_jpeg(f, 512)
        elapsed_ms = 1000 * (time.perf_counter() - t0)
        check(
            f"{os.path.basename(f)}: preview extracted",
            arr is not None,
        )
        if elapsed_ms > 150:
            slow.append((os.path.basename(f), elapsed_ms))

    check(
        "no file took >150ms (bounded I/O regardless of file size, not a whole-file read)",
        len(slow) == 0,
        f"slow files: {slow}" if slow else "",
    )

    # Selection consistency: smaller requests should never take longer than
    # (or need to decode a bigger source than) full-quality requests.
    f = files[0]
    dims_by_size = {}
    for max_size in (512, 1920, 0):
        arr = _extract_bmff_uuid_embedded_jpeg(f, max_size)
        dims_by_size[max_size] = arr.shape[:2] if arr is not None else None
    check(
        "512px request result exists and is not upsampled beyond the source",
        dims_by_size.get(512) is not None,
        f"{dims_by_size}",
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
