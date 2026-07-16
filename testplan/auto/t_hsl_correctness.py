#!/usr/bin/env python3
"""HSL float32 HSV scale + proportional slider regression tests."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from raw_hsl import _rgb_to_hsv, apply_hsl_adjustments

    rgb = np.zeros((1, 3, 3), np.float32)
    rgb[0, 0] = [1, 0, 0]
    rgb[0, 1] = [0, 1, 0]
    rgb[0, 2] = [0, 0, 1]
    h, s, v = _rgb_to_hsv(rgb)
    check("float H in degrees (~0/120/240)", abs(float(h[0, 0]) - 0) < 1 and abs(float(h[0, 1]) - 120) < 1)
    check("float S/V in [0,1]", float(s.max()) <= 1.01 and float(v.max()) <= 1.01)

    base = np.full((16, 16, 3), 0.0, np.float32)
    base[:, :, 0] = 0.65
    base[:, :, 1] = 0.22
    base[:, :, 2] = 0.22
    d20 = float(np.mean(np.abs(apply_hsl_adjustments(base, {"SaturationAdjustmentRed": 20}) - base)))
    d40 = float(np.mean(np.abs(apply_hsl_adjustments(base, {"SaturationAdjustmentRed": 40}) - base)))
    ratio = d40 / max(d20, 1e-9)
    check("sat slider roughly proportional (not on/off)", 1.6 < ratio < 2.4, f"ratio={ratio:.3f}")

    # Old uint8-mistaken path crushed S≈1/255 so ±ds dominated → near-identical
    # huge jumps for any non-zero sat. Guard that mid sat leaves room to grow.
    mid = apply_hsl_adjustments(base, {"SaturationAdjustmentRed": 25})
    hard = apply_hsl_adjustments(base, {"SaturationAdjustmentRed": 100})
    check(
        "sat+25 milder than sat+100",
        float(np.mean(np.abs(mid - base))) < float(np.mean(np.abs(hard - base))) * 0.6,
    )

    out_hue = apply_hsl_adjustments(base, {"HueAdjustmentRed": 40})
    h0, _, _ = _rgb_to_hsv(base)
    h1, _, _ = _rgb_to_hsv(out_hue)
    check("hue slider shifts mean hue", abs(float(h1.mean() - h0.mean())) > 5.0)

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
