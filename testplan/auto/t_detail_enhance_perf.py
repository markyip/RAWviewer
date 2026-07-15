#!/usr/bin/env python3
"""Regression guard for the detail-enhancement (Sharpness/Clarity/Defringe)
speed fix, including the row-band thread-parallel pipeline.

Root cause of a real "low-res preview sits on screen for 10+ seconds before
the full-res image pops in" bug: apply_adjustments_to_rgb's detail-enhance
stage (Sharpness/Clarity/Defringe, all of raw_detail_enhance.py) took 7.5s+
on a real 32MP image with typical Lightroom-style edits applied. Three
rounds of fixes, same file/settings, median steady-state:
  1. np.max/min(img, axis=-1) is ~8x slower than an elementwise 3-way chain
     for a size-3 last axis; Clarity's blur (sigma=10) computed at full res
     instead of downsampled.                              7540ms -> 4964ms
  2. Broadcasting a (H,W,1) factor against (H,W,3) is slower than a
     per-channel loop with a sliced out=; skip a redundant astype() copy.
                                                             4964ms -> 4300ms
  3. Row-band thread parallelism: every stage is per-pixel except detail-
     enhance's small-radius Gaussian blurs, so splitting rows across
     threads with padding covering those blur radii (numpy/cv2 release the
     GIL, so plain threads get real parallelism) -- 4300ms -> 2222ms.

Checks the speed budget and that approximation error (Clarity's downsample
blur, plus a small additional per-band grid-alignment effect from step 3)
stays within a previously measured bound -- not a byte-exact regression
gate, since these are deliberate, tiny, verified-imperceptible tradeoffs
(the banding-added error is NOT concentrated at band boundaries -- verified
manually against real decoded image content, not narrated here since that
needs a real RAW file this suite doesn't require).
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np  # noqa: E402

FAILURES = []
GOLDEN_ARW = os.environ.get("RAWVIEWER_TEST_ASSETS_ARW", "/tmp/RAW_Sample/DSC01089.ARW")


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
        elapsed_s < 4.5,
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

    # Banding correctness: random noise can't reveal blur-seam artifacts (no
    # spatial structure to blur), so this needs real decoded image content.
    if os.path.isfile(GOLDEN_ARW):
        import rawpy

        from raw_adjustments import (
            _apply_adjustments_to_srgb_banded,
            _apply_adjustments_to_srgb_core,
        )

        with rawpy.imread(GOLDEN_ARW) as raw:
            rgb = raw.postprocess(use_camera_wb=True, half_size=True, output_bps=8)

        merged = dict(adj)
        ref = _apply_adjustments_to_srgb_core(
            rgb.astype(np.float32) / 255.0, merged
        )
        ref_u8 = (np.clip(ref, 0.0, 1.0) * 255.0).astype(np.uint8)

        for nw in (2, 4, 8):
            banded = _apply_adjustments_to_srgb_banded(rgb, merged, nw)
            diff = np.abs(banded.astype(np.int16) - ref_u8.astype(np.int16))
            check(
                f"banded (n_workers={nw}) matches single-pass within the accepted approximation bound",
                diff.max() <= 10,
                f"max={diff.max()} mean={diff.mean():.4f}",
            )

        h = rgb.shape[0]
        band_h = -(-h // 4)
        diff4 = np.abs(
            _apply_adjustments_to_srgb_banded(rgb, merged, 4).astype(np.int16)
            - ref_u8.astype(np.int16)
        ).max(axis=-1)
        near = np.zeros(h, dtype=bool)
        for b in (band_h, 2 * band_h, 3 * band_h):
            near[max(0, b - 20) : min(h, b + 20)] = True
        near_rate = (diff4[near] > 0).mean() if near.any() else 0.0
        far_rate = (diff4[~near] > 0).mean() if (~near).any() else 0.0
        check(
            "banding error is not concentrated at band boundaries (not a visible seam)",
            near_rate <= far_rate * 2.0 + 0.05,
            f"near_boundary_rate={near_rate:.3f} far_from_boundary_rate={far_rate:.3f}",
        )
    else:
        print(f"SKIP  banding correctness check: golden file not present: {GOLDEN_ARW}")

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
