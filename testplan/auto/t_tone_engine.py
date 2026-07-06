#!/usr/bin/env python3
"""Tone-engine invariants: identity, monotonicity, black anchor, recovery floor."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from raw_edit_pipeline import linear_to_display_uint8, process_linear_edit_buffer

    def render(img, adj):
        return linear_to_display_uint8(process_linear_edit_buffer(img, adj, preview=True), adj)

    # 1. Identity: defaults / zeroed sliders are an exact no-op
    img = (np.random.RandomState(7).rand(64, 64, 3).astype(np.float32)) ** 2
    a0 = render(img, {})
    for adj in ({"Exposure2012": 0.0}, {"Shadows2012": 0.0}, {"Blacks2012": 0.0}):
        d = np.abs(a0.astype(int) - render(img, adj).astype(int)).max()
        check(f"identity {list(adj)[0]}=0", d == 0, f"maxdiff={d}")

    # 2. Monotonicity at slider extremes (no tone-curve inversion/banding)
    xs = np.linspace(1e-5, 1.2, 800, dtype=np.float32)
    ramp = np.repeat(xs[None, :, None], 3, axis=2).reshape(1, -1, 3)
    for adj in (
        {"Shadows2012": 100.0}, {"Shadows2012": -100.0},
        {"Blacks2012": 100.0}, {"Blacks2012": -100.0},
        {"Whites2012": 100.0}, {"Highlights2012": -100.0},
        {"Shadows2012": 100.0, "Blacks2012": 100.0},
    ):
        out = render(ramp, adj)[0, :, 0].astype(int)
        mono = bool(np.all(np.diff(out) >= -1))
        check(f"monotonic {adj}", mono)

    # 3. Black anchor: absolute black stays 0 under max lift
    black = np.zeros((4, 4, 3), dtype=np.float32)
    v = int(render(black, {"Shadows2012": 100.0, "Blacks2012": 100.0})[0, 0, 0])
    check("absolute black pinned", v == 0, f"out={v}")

    # 4. Toe shape: noise floor barely lifts, detail band lifts strongly
    def lift(scene, adj):
        img2 = np.full((4, 4, 3), scene, dtype=np.float32)
        return int(render(img2, adj)[0, 0, 0])

    sb = {"Shadows2012": 100.0, "Blacks2012": 100.0}
    check("noise floor (8.5 stops under) stays dark", lift(0.0005, sb) <= 6,
          f"out={lift(0.0005, sb)}")
    check("detail band (5.2 stops under) recovers", lift(0.005, sb) >= 60,
          f"out={lift(0.005, sb)}")

    # 5. Combined >= individual (regression: chroma damp once cancelled lift)
    s_only = lift(0.01, {"Shadows2012": 100.0})
    b_only = lift(0.01, {"Blacks2012": 100.0})
    both = lift(0.01, sb)
    check("shadows+blacks >= max(individual)", both >= max(s_only, b_only) - 1,
          f"s={s_only} b={b_only} both={both}")

    # 6. Recovery floor: >= 2x the old 3.0-ratio-cap engine's reach
    base = lift(0.01, {})
    check("recovery strength floor", both >= base * 3, f"base={base} both={both}")

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
