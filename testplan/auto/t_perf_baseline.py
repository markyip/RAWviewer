#!/usr/bin/env python3
"""Perf regression: decode-stage benchmarks vs per-machine stored baseline.

First run (or --rebaseline) writes testplan/baselines/perf_baseline.json.
Later runs fail when any metric's min-of-3 regresses more than TOLERANCE.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

GOLDEN = "/Volumes/Development/Development/Canon_Sample/1G9A0419.CR3"
BASELINE = os.path.join(os.path.dirname(__file__), "..", "baselines", "perf_baseline.json")
TOLERANCE = 1.25  # fail if slower than baseline * 1.25


def bench(fn, n=3) -> float:
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0


def main() -> int:
    if not os.path.isfile(GOLDEN):
        print("SKIP  golden RAW not available on this machine")
        return 0

    os.environ.setdefault("RAWVIEWER_WB_SANITY", "0")
    os.environ.setdefault("RAWVIEWER_PERF", "0")
    import numpy as np
    import fast_raw_decode as frd

    results = {}
    results["unpack_ms"] = bench(lambda: frd.unpack_raw(GOLDEN))
    u = frd.unpack_raw(GOLDEN)
    results["decode_half_ms"] = bench(lambda: frd.decode_half_from_unpacked(u))
    results["decode_full_cpu_ms"] = bench(lambda: frd.finish_full_decode(u))

    from raw_adjustments import apply_adjustments_to_rgb

    rgb = (np.random.RandomState(1).rand(2000, 3000, 3) * 255).astype(np.uint8)
    adj = {"Exposure2012": 0.5, "Shadows2012": 40.0}
    results["sidecar_apply_6mp_ms"] = bench(lambda: apply_adjustments_to_rgb(rgb, adj), n=2)

    rebaseline = "--rebaseline" in sys.argv or not os.path.isfile(BASELINE)
    if rebaseline:
        os.makedirs(os.path.dirname(BASELINE), exist_ok=True)
        with open(BASELINE, "w") as f:
            json.dump(results, f, indent=2)
        print("BASELINE written:")
        for k, v in results.items():
            print(f"  {k}: {v:.0f}ms")
        return 0

    with open(BASELINE) as f:
        base = json.load(f)
    failures = 0
    for k, v in results.items():
        b = base.get(k)
        if b is None:
            print(f"NEW   {k}: {v:.0f}ms (no baseline; rerun with --rebaseline to record)")
            continue
        ok = v <= b * TOLERANCE
        print(f"{'PASS' if ok else 'FAIL'}  {k}: {v:.0f}ms vs baseline {b:.0f}ms "
              f"({(v - b) / b * 100.0:+.0f}%, limit +{(TOLERANCE - 1) * 100:.0f}%)")
        if not ok:
            failures += 1
    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
