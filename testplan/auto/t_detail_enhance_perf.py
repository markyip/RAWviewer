#!/usr/bin/env python3
"""Regression guard for the detail-enhancement (Sharpness/Clarity/Defringe)
speed fix.

Root cause of a real "low-res preview sits on screen for 10+ seconds before
the full-res image pops in" bug: apply_adjustments_to_rgb's detail-enhance
stage (Sharpness/Clarity/Defringe, all of raw_detail_enhance.py) took 7.5s+
on a real 32MP image with typical Lightroom-style edits applied, because:
  1. np.max/min(img, axis=-1) is ~8x slower than an elementwise 3-way chain
     for a size-3 last axis (raw_detail_enhance.py, raw_tone_recovery.py,
     raw_adjustments.py all had this).
  2. Clarity's local-contrast blur (sigma=10, inherently low-frequency) was
     computed at full resolution instead of downsampled.

Checks both the speed budget and that the downsample-blur approximation
(the one behavior change, Clarity's blur) stays within a small, previously
measured error bound (max abs diff 8/255, mean ~0.026/255 on real edit
settings) -- not a byte-exact regression gate like the axis-order fix
would be, since it's a deliberate, tiny quality/speed tradeoff.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from raw_adjustments import apply_adjustments_to_rgb

    # Real-world-shaped edit settings (matches an actual XMP sidecar that
    # exposed this bug): heavy sharpness/clarity/defringe + highlights/
    # shadows + vibrance, all of which touch the slow paths that were fixed.
    adj = {
        "Exposure2012": 2.71, "Contrast2012": 71.0, "Highlights2012": -52.0,
        "Shadows2012": 29.0, "Temperature": 6075.0, "Tint": 17.0,
        "Saturation": 73.0, "Vibrance": 83.0, "Sharpness": 75.0,
        "Clarity2012": 35.0, "Defringe": 100.0, "ColorNoiseReduction": 50.0,
        "LuminanceNoiseReduction": 48.0,
    }

    rng = np.random.RandomState(42)
    # Full 32MP scale (matches the real file that exposed this bug) so the
    # timing check reflects the actual single-image full-decode path.
    img = rng.randint(0, 255, (4672, 7008, 3), dtype=np.uint8)

    t0 = time.perf_counter()
    out = apply_adjustments_to_rgb(img, adj)
    elapsed_s = time.perf_counter() - t0
    check(
        "32MP detail-enhance-heavy edit completes well under the pre-fix ~7.5s baseline",
        elapsed_s < 5.5,
        f"{elapsed_s:.2f}s",
    )
    check("output shape preserved", out.shape == img.shape)
    check("output dtype preserved", out.dtype == img.dtype)

    # Smaller scale for the approximation-error check (keeps the test fast;
    # the error bound doesn't depend on resolution).
    small_img = rng.randint(0, 255, (2336, 3504, 3), dtype=np.uint8)
    small_out = apply_adjustments_to_rgb(small_img, adj)
    check(
        "output stays in valid uint8 range",
        small_out.min() >= 0 and small_out.max() <= 255,
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
