#!/usr/bin/env python3
"""WB dropper: display→buffer mapping + crop-framed linear sample."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from raw_transform import apply_geometry, map_display_point_to_buffer

    # Lite display (640) → half-res buffer (3514): center must land near center.
    bx, by = map_display_point_to_buffer(
        320.0, 213.5, display_w=640, display_h=427, buffer_w=3514, buffer_h=2344
    )
    check(
        "lite center maps near half-res center",
        abs(bx - 1757.0) < 2.0 and abs(by - 1172.0) < 2.0,
        f"got ({bx:.1f},{by:.1f})",
    )

    # Crop insets: sample buffer is post-geometry; a red patch only in the
    # cropped region must be what display-center hits.
    h, w = 200, 300
    base = np.zeros((h, w, 3), dtype=np.float32)
    base[:, :] = (0.2, 0.2, 0.8)  # cool blue elsewhere
    # After 20% left/right crop, the surviving center band is painted warm.
    x0, x1 = int(w * 0.20), w - int(w * 0.20)
    base[:, x0:x1] = (0.8, 0.5, 0.3)
    adj = {"CropLeft": 0.20, "CropRight": 0.20, "CropTop": 0.0, "CropBottom": 0.0}
    framed = apply_geometry(base, adj)
    check("crop shrinks width", framed.shape[1] < w, f"shape={framed.shape}")

    # Simulate a further lite downscale of the cropped frame.
    import cv2

    lite = cv2.resize(framed, (640, int(640 * framed.shape[0] / framed.shape[1])), interpolation=cv2.INTER_AREA)
    cx = lite.shape[1] / 2.0
    cy = lite.shape[0] / 2.0
    mx, my = map_display_point_to_buffer(
        cx,
        cy,
        display_w=lite.shape[1],
        display_h=lite.shape[0],
        buffer_w=framed.shape[1],
        buffer_h=framed.shape[0],
    )
    ix, iy = int(round(mx)), int(round(my))
    sample = framed[iy, ix]
    check(
        "cropped lite center samples warm patch (not blue margins)",
        float(sample[0]) > 0.6 and float(sample[2]) < 0.5,
        f"rgb={sample} at ({ix},{iy}) framed={framed.shape}",
    )

    # Bug-shaped control: indexing display coords into the pre-crop base
    # (no geometry, no scale) is the old WB path and misses the warm band.
    bad = base[h // 2, w // 2]
    # Full-frame center is still in the warm painted band here; use a point
    # that lands in the blue margin under naive lite→base indexing.
    naive_x = int(cx)  # ~320 on a 300-wide base → OOB / wrong
    naive_y = min(h - 1, int(cy))
    if 0 <= naive_x < w:
        bad = base[naive_y, naive_x]
        check(
            "naive lite-x into pre-crop base is NOT the warm center",
            float(bad[0]) < 0.5 or float(bad[2]) > 0.5,
            f"rgb={bad} at ({naive_x},{naive_y})",
        )
    else:
        check(
            "naive lite-x into pre-crop base is out of range (old silent fail)",
            True,
            f"x={naive_x} w={w}",
        )

    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
