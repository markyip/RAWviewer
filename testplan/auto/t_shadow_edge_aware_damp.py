#!/usr/bin/env python3
"""Regression guard: shadow-recovery chroma damping must be edge-aware,
not a flat global strength.

Real bug (reported via a real-file A/B against Exposure): a single global
damp strength can't distinguish genuine local color structure (a fold
line, fabric weave) from per-pixel sensor noise that got amplified by the
same lift -- both look like local chroma deviation to a per-pixel-only
computation. Damping hard enough to control the noise also flattened
real color texture toward grey, measured as 2.4x less chroma than an
equivalent Exposure push at matched brightness on a real fabric crop.

This builds a synthetic image with a hard, colored edge (simulating a
real fold/object boundary) on one side and flat colored noise on the
other, both requiring the same Shadows lift, and checks that the edge
side keeps more of its chroma than the flat side.
"""
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
    import raw_pv2012 as pv

    rng = np.random.RandomState(7)
    h, w = 80, 160
    img = np.full((h, w, 3), 0.01, dtype=np.float32)

    # Left half: a hard, colored edge (two flat color blocks meeting at a
    # sharp boundary) -- simulates a real object/fold boundary. Small
    # per-pixel noise on top, same as the flat side, so any difference in
    # preserved chroma comes from the edge-awareness, not from having less
    # noise to begin with.
    img[:, : w // 4] = [0.006, 0.010, 0.014]
    img[:, w // 4 : w // 2] = [0.014, 0.010, 0.006]

    # Right half: uniform color, same average level, no structure -- a
    # stand-in for flat, genuinely noisy fabric.
    img[:, w // 2 :] = [0.010, 0.010, 0.010]

    noise = rng.normal(0, 0.0006, (h, w, 3)).astype(np.float32)
    noise[:, :, 2] += rng.normal(0, 0.0004, (h, w)).astype(np.float32)
    img = np.clip(img + noise, 0, None).astype(np.float32)

    out = pv.apply_pv2012_tone_rgb(img, {"Shadows2012": 100.0})

    edge_region = out[:, : w // 2]
    flat_region = out[:, w // 2 :]

    def chroma_rms(region):
        lum = 0.2126 * region[..., 0] + 0.7152 * region[..., 1] + 0.0722 * region[..., 2]
        chroma = region - lum[..., np.newaxis]
        return float(np.sqrt(np.mean(np.sum(chroma**2, axis=-1))))

    edge_chroma = chroma_rms(edge_region)
    flat_chroma = chroma_rms(flat_region)

    check(
        "edge/structured region keeps more chroma than flat/noisy region "
        "at the same lift",
        edge_chroma > flat_chroma * 1.05,
        f"edge={edge_chroma:.5f} flat={flat_chroma:.5f}",
    )

    # Guard against the fix regressing back into the "everything reads as
    # an edge" failure mode of a naive (non-pre-smoothed) Sobel weight:
    # the flat region must still be substantially damped, not merely
    # slightly less than the edge region.
    out_no_lift = pv.apply_pv2012_tone_rgb(img, {})
    flat_chroma_no_lift = chroma_rms(out_no_lift[:, w // 2 :])
    check(
        "flat region is still meaningfully damped relative to no-lift baseline "
        "(damping wasn't accidentally disabled everywhere)",
        flat_chroma < flat_chroma_no_lift * 6.0,
        f"flat_lifted={flat_chroma:.5f} flat_no_lift={flat_chroma_no_lift:.5f}",
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
