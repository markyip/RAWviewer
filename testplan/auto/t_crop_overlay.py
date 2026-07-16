#!/usr/bin/env python3
"""Crop insets + overlay kernel smoke tests."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
os.environ["RAWVIEWER_ENABLE_EDITING"] = "1"

import numpy as np

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from raw_adjustments import DEFAULT_ADJUSTMENTS
    from raw_transform import apply_geometry, has_geometry

    for k in ("CropLeft", "CropRight", "CropTop", "CropBottom"):
        check(f"DEFAULT has {k}", k in DEFAULT_ADJUSTMENTS)

    img = np.full((100, 200, 3), 40, dtype=np.uint8)
    adj = dict(DEFAULT_ADJUSTMENTS)
    check("no crop is identity object", apply_geometry(img, adj) is img)

    adj["CropLeft"] = 0.1
    adj["CropRight"] = 0.1
    adj["CropTop"] = 0.05
    adj["CropBottom"] = 0.05
    check("has_geometry with crop", has_geometry(adj))
    out = apply_geometry(img, adj)
    check("crop shrinks width", out.shape[1] < img.shape[1], f"{out.shape}")
    check("crop shrinks height", out.shape[0] < img.shape[0], f"{out.shape}")

    from rawviewer_ui.crop_overlay import CropOverlayItem

    item = CropOverlayItem()
    item.set_image_size(200, 100)
    item.set_insets(0.1, 0.1, 0.1, 0.1)
    r = item.crop_rect()
    check("crop_rect size", abs(r.width() - 160) < 1 and abs(r.height() - 80) < 1, str(r))
    item.set_aspect_ratio(1.0)
    r2 = item.crop_rect()
    check(
        "1:1 aspect roughly square in pixels",
        abs(r2.width() - r2.height()) < 2.0,
        f"{r2.width():.1f}x{r2.height():.1f}",
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
