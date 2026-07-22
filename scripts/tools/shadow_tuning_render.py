#!/usr/bin/env python3
"""Render Shadows/Blacks recovery crops from real RAW files for visual
inspection -- a repeatable harness for the raw_pv2012.py grey-casting /
blue-speckle tuning loop, since numeric proxies alone weren't a reliable
enough guide to the perceptual result.

Usage: pixi run python3 scripts/tools/shadow_tuning_render.py [out_dir]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

FILES = [f for f in sys.argv[1:] if os.path.isfile(f)]

SETTINGS = [
    ("default", {}),
    ("exposure_p2", {"Exposure2012": 2.0}),
    ("shadows_50", {"Shadows2012": 50.0}),
    ("shadows_100", {"Shadows2012": 100.0}),
    ("shadows_100_blacks_m50", {"Shadows2012": 100.0, "Blacks2012": -50.0}),
]

CROP_SIZE = 420


def darkest_crop_box(lum: np.ndarray, size: int) -> tuple[int, int]:
    h, w = lum.shape
    size = min(size, h, w)
    block = max(16, size // 4)
    best_score = None
    best_yx = (0, 0)
    for y in range(0, h - size + 1, block):
        for x in range(0, w - size + 1, block):
            score = float(lum[y : y + size, x : x + size].mean())
            if best_score is None or score < best_score:
                best_score = score
                best_yx = (y, x)
    return best_yx


def main() -> None:
    from unified_image_processor import UnifiedImageProcessor
    from raw_adjustments import DEFAULT_ADJUSTMENTS
    from raw_edit_pipeline import process_linear_edit_buffer, linear_to_display_uint8
    import raw_pv2012 as pv

    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/shadow_tuning"
    os.makedirs(out_dir, exist_ok=True)

    proc = UnifiedImageProcessor()
    for path in FILES:
        stem = os.path.splitext(os.path.basename(path))[0]
        base = proc.decode_raw_edit_base(path, use_full_resolution=False)
        lum = pv._luminance(base)
        y0, x0 = darkest_crop_box(lum, CROP_SIZE)
        size = min(CROP_SIZE, base.shape[0], base.shape[1])
        crop = base[y0 : y0 + size, x0 : x0 + size]

        for label, overrides in SETTINGS:
            merged = dict(DEFAULT_ADJUSTMENTS)
            merged.update(overrides)
            processed = process_linear_edit_buffer(crop, merged, preview=True)
            out = linear_to_display_uint8(processed, merged)
            out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            out_path = os.path.join(out_dir, f"{stem}__{label}.png")
            cv2.imwrite(out_path, out_bgr)
            print("wrote", out_path)


if __name__ == "__main__":
    main()
