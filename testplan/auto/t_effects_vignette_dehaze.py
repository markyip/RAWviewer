#!/usr/bin/env python3
"""Vignette / dehaze effect smoke tests."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from raw_effects import apply_dehaze, apply_vignette
    from raw_edit_pipeline import _apply_display_stage, _apply_display_stage_banded
    from raw_adjustments import DEFAULT_ADJUSTMENTS

    img = np.full((200, 300, 3), 0.45, dtype=np.float32)
    check("vignette 0 is identity object", apply_vignette(img, 0.0) is img)
    dark = apply_vignette(img, -80.0)
    check(
        "negative vignette darkens corners (LR)",
        float(dark[2, 2, 0]) < float(dark[100, 150, 0]),
    )
    bright = apply_vignette(img, 80.0)
    check(
        "positive vignette brightens corners (LR)",
        float(bright[2, 2, 0]) > float(bright[100, 150, 0]),
    )
    # Amount must be monotonic / near-linear (no direction reversals).
    corners = [
        float(apply_vignette(img, float(a))[2, 2, 0]) for a in range(-100, 101, 10)
    ]
    diffs = [corners[i + 1] - corners[i] for i in range(len(corners) - 1)]
    check(
        "corner brightness rises monotonically with Amount",
        all(d >= -1e-6 for d in diffs),
        f"diffs={diffs}",
    )
    # Equal Amount steps → roughly equal corner deltas (Paint Overlay).
    mid_diffs = diffs[4:16]  # around -60…+50
    span = max(mid_diffs) - min(mid_diffs)
    check(
        "corner response near-linear in Amount",
        span < 0.08,
        f"span={span:.4f} mid_diffs={[round(d,4) for d in mid_diffs]}",
    )
    # At Amount=-100 corners approach black; +100 approach white.
    check(
        "Amount -100 corners near black",
        float(apply_vignette(img, -100.0)[2, 2, 0]) < 0.05,
    )
    check(
        "Amount +100 corners near white",
        float(apply_vignette(img, 100.0)[2, 2, 0]) > 0.95,
    )

    # Midpoint: lower = larger vignette (more center darkening at mid-radius).
    # Sample a point ~halfway from center to corner.
    mid_y, mid_x = 40, 60
    low_mid = apply_vignette(img, -100.0, midpoint=10.0)
    high_mid = apply_vignette(img, -100.0, midpoint=90.0)
    check(
        "low Midpoint reaches further toward center",
        float(low_mid[mid_y, mid_x, 0]) < float(high_mid[mid_y, mid_x, 0]),
        f"low={low_mid[mid_y, mid_x, 0]:.3f} high={high_mid[mid_y, mid_x, 0]:.3f}",
    )
    check(
        "high Midpoint leaves near-center untouched",
        abs(float(high_mid[100, 150, 0]) - 0.45) < 1e-3,
    )
    # Midpoint=100 must stay a tight corner-only effect (not a wide rim).
    tight = apply_vignette(img, -100.0, midpoint=100.0)
    # ~70% of the way from center to corner should stay nearly untouched.
    y70 = int(round(99 + 0.70 * (2 - 99)))
    x70 = int(round(149 + 0.70 * (2 - 149)))
    check(
        "Midpoint 100 barely affects mid-frame",
        float(tight[y70, x70, 0]) > 0.40,
        f"at70%={tight[y70, x70, 0]:.3f}",
    )
    check(
        "Midpoint 100 still darkens true corners",
        float(tight[0, 0, 0]) < 0.05,
    )

    # Soft haze field: brighter, washed midtones
    yy, xx = np.mgrid[0:240, 0:320]
    haze = np.clip(0.35 + 0.25 * (yy / 240.0), 0, 1).astype(np.float32)
    haze_img = np.stack([haze, haze * 0.98, haze * 0.95], axis=-1)
    cleared = apply_dehaze(haze_img, 70.0, preview=True)
    check(
        "dehaze increases local contrast (std)",
        float(cleared.std()) > float(haze_img.std()) * 0.95,
        f"in={haze_img.std():.4f} out={cleared.std():.4f}",
    )
    hazed = apply_dehaze(img, -60.0, preview=True)
    check(
        "negative dehaze pulls toward atmosphere",
        float(hazed.mean()) != float(img.mean()),
    )
    check("dehaze 0 identity", apply_dehaze(img, 0.0) is img)

    # Banded display stage must not re-center vignette per row strip.
    lin = np.full((512, 768, 3), 0.35, dtype=np.float32)
    adj = dict(DEFAULT_ADJUSTMENTS)
    adj["PostCropVignetteAmount"] = 90.0
    full = _apply_display_stage(lin, adj)
    banded = _apply_display_stage_banded(lin, adj, n_workers=8)
    max_diff = float(np.max(np.abs(full.astype(np.float64) - banded.astype(np.float64))))
    check(
        "banded display matches full-frame vignette",
        max_diff < 1e-5,
        f"max_diff={max_diff:.6g}",
    )
    # Positive Amount lightens corners (LR) — center stays darker than corner.
    check(
        "positive vignette keeps frame center darker than corner",
        float(full[256, 384, 0]) < float(full[4, 4, 0]),
    )

    # uint8 browse path must use the real pipeline (not legacy gamma core).
    from raw_adjustments import DEFAULT_ADJUSTMENTS, apply_adjustments_to_rgb

    u8 = np.full((128, 192, 3), 120, dtype=np.uint8)
    adj_u8 = dict(DEFAULT_ADJUSTMENTS)
    adj_u8["PostCropVignetteAmount"] = -100.0
    adj_u8["AsShotTemperature"] = 5500.0
    out_u8 = apply_adjustments_to_rgb(u8, adj_u8)
    check("uint8 vignette returns ndarray", isinstance(out_u8, np.ndarray))
    check(
        "uint8 negative vignette darkens corners vs center",
        int(out_u8[2, 2, 0]) < int(out_u8[64, 96, 0]),
        f"corner={out_u8[2, 2, 0]} center={out_u8[64, 96, 0]}",
    )

    if FAILURES:
        print(f"FAILED: {FAILURES}")
        return 1
    print("PASS t_effects_vignette_dehaze")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
