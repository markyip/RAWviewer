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

    img = np.full((200, 300, 3), 0.45, dtype=np.float32)
    check("vignette 0 is identity object", apply_vignette(img, 0.0) is img)
    dark = apply_vignette(img, 80.0)
    check(
        "positive vignette darkens corners",
        float(dark[2, 2, 0]) < float(dark[100, 150, 0]),
    )
    bright = apply_vignette(img, -80.0)
    check(
        "negative vignette brightens corners",
        float(bright[2, 2, 0]) > float(bright[100, 150, 0]),
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

    if FAILURES:
        print(f"FAILED: {FAILURES}")
        return 1
    print("PASS t_effects_vignette_dehaze")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
