#!/usr/bin/env python3
"""WB embedded-JPEG sanity: model verdict cache + misparse correction (golden files)."""
import glob
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

GOOD_A = "/Volumes/Development/Development/Canon_Sample/025A2279.CR3"
GOOD_B = "/Volumes/Development/Development/Canon_Sample/025A2705.CR3"
BAD = sorted(glob.glob("/Volumes/Development/Development/Canon_Sample/683A*.CR3"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    if not (os.path.isfile(GOOD_A) and os.path.isfile(GOOD_B) and BAD):
        print("SKIP  golden RAW files not available on this machine")
        return 0

    os.environ["RAWVIEWER_WB_SANITY"] = "1"
    import fast_raw_decode as frd

    frd._WB_CORRECTION_CACHE.clear()
    frd._WB_MODEL_VERDICT.clear()

    # Clean model: first file measures, verdict recorded, no correction
    frd.unpack_raw(GOOD_A)
    check("clean file uncorrected", frd.get_corrected_camera_wb(GOOD_A) is None)
    check("clean model verdict recorded", False in frd._WB_MODEL_VERDICT.values())

    # Second file of the clean model: skipped (near-zero overhead)
    t0 = time.perf_counter()
    frd.unpack_raw(GOOD_B)
    with_verdict = time.perf_counter() - t0
    check("second clean file uncorrected", frd.get_corrected_camera_wb(GOOD_B) is None)
    print(f"INFO  second-file unpack with verdict skip: {with_verdict*1000:.0f}ms")

    # Misparsed model: every file corrected, per-file values
    for p in BAD:
        frd.unpack_raw(p)
        corrected = frd.get_corrected_camera_wb(p)
        check(f"misparse corrected {os.path.basename(p)}", corrected is not None)
    check("misparsed model verdict True", True in frd._WB_MODEL_VERDICT.values())

    # Model-key disambiguation regression: the misparsed body shares its
    # matrix with a clean body (LibRaw aliasing) -- both verdicts must coexist.
    check(
        "aliased-matrix models disambiguated",
        True in frd._WB_MODEL_VERDICT.values() and False in frd._WB_MODEL_VERDICT.values(),
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
