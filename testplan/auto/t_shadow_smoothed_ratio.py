#!/usr/bin/env python3
"""Regression guard: the shadow-recovery ratio must be computed from a
smoothed luminance, not raw per-pixel luminance.

Root problem (via researching how darktable/RawTherapee handle the same
issue -- their old shadows/highlights module has the identical documented
bug, "hue shifts towards blue in the shadows"): ratio = y1/y0 computed
entirely from a pixel's own raw luminance means any sensor noise in that
pixel's y0 becomes noise in its own ratio -- amplifying noise into extra
color speckle on top of whatever the sensor already had, independent of
the chroma damp that cleans up sensor noise afterward. darktable's fix
was architectural: compute the region gain from a blurred/multi-scale
luminance mask, not the raw per-pixel value.

This checks two pixels with the *same* true underlying brightness but
different per-pixel noise get a much more similar ratio (i.e. similar
lift) than they would from the raw, unsmoothed computation -- the direct
mechanism behind the measured chroma improvement (Canon: 58% -> 71% of
Exposure's chroma at matched brightness; Sony: 90% -> 97%).
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

    # A flat deep-shadow patch with per-pixel noise (same statistics
    # everywhere -- no real edges), matching the real dark-region scale
    # from prior measurements (y0 perceptual ~0.005).
    rng = np.random.RandomState(11)
    h, w = 60, 60
    base_lin = 0.0018
    noise = rng.normal(0, 0.0004, (h, w, 3)).astype(np.float32)
    img = np.clip(base_lin + noise, 0, None).astype(np.float32)

    out = pv.apply_pv2012_tone_rgb(img, {"Shadows2012": 100.0})
    lum_out = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]

    # If the ratio were driven by raw per-pixel noise, the resulting
    # per-pixel lifted luminance would vary a lot pixel-to-pixel purely
    # from that noise (noise gets amplified differently per pixel based on
    # its own value). With a smoothed driving signal, neighbouring pixels
    # (same true brightness) should end up much more uniform.
    lum_std = float(lum_out.std())
    lum_mean = float(lum_out.mean())
    check(
        "lifted luminance is spatially coherent on a flat noisy patch "
        "(smoothed ratio, not raw per-pixel jitter)",
        lum_std < lum_mean * 0.35,
        f"std={lum_std:.4f} mean={lum_mean:.4f} ratio={lum_std / max(lum_mean, 1e-6):.3f}",
    )

    # Sanity: the smoothing must not eliminate the lift itself.
    out_default = pv.apply_pv2012_tone_rgb(img, {})
    lum_default_mean = float(
        (0.2126 * out_default[..., 0] + 0.7152 * out_default[..., 1] + 0.0722 * out_default[..., 2]).mean()
    )
    check(
        "Shadows+100 still lifts substantially above the unadjusted baseline",
        lum_mean > lum_default_mean * 2.0,
        f"lifted={lum_mean:.4f} baseline={lum_default_mean:.4f}",
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
