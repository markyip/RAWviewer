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

    # Non-interactive preview must not claim edge drags -- otherwise Masks /
    # gradient placement near the photo boundary silently crops and looks
    # like the image itself is resizing.
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([])
    item.set_image_size(400, 300)
    item.set_insets(0.0, 0.0, 0.0, 0.0)
    item.set_interactive(False)
    check("preview mode reports non-interactive", not item.is_interactive())
    check(
        "preview mode accepts no mouse buttons",
        item.acceptedMouseButtons() == Qt.MouseButton.NoButton,
    )
    item.set_interactive(True)
    check("crop mode is interactive again", item.is_interactive())
    check(
        "crop mode accepts left button",
        bool(item.acceptedMouseButtons() & Qt.MouseButton.LeftButton),
    )
    # Edge hit still resolves, but only interactive mode will accept a press.
    check("edge hit test still finds the border", item._hit_test(QPointF(0, 150)) == "l")

    # --- the passive geometry preview must not draw a thirds grid ---------
    #
    # This item is reused for the geometry-overlay preview, which stays on for
    # as long as straighten/perspective is non-zero. Drawing the rule-of-thirds
    # guides unconditionally left a 3x3 grid over the photo from the moment the
    # frame was straightened until the rotation was zeroed again, with nothing
    # on screen explaining it or offering to turn it off.
    #
    # Painted rather than asserted on a flag: the bug was the ORDER of two
    # statements inside paint(), which no amount of state inspection would
    # catch.
    from PyQt6.QtGui import QImage, QPainter

    def _interior_ink(interactive: bool) -> int:
        """Non-transparent pixels strictly inside the crop, ignoring the frame."""
        item.set_image_size(300, 300)
        item.set_insets(0.0, 0.0, 0.0, 0.0)
        item.set_aspect_ratio(None)
        item.set_interactive(interactive)
        img = QImage(300, 300, QImage.Format.Format_ARGB32)
        img.fill(0)
        p = QPainter(img)
        item.paint(p, None, None)
        p.end()
        # Inset well past the border frame and the corner handles.
        margin = 40
        return sum(
            1
            for y in range(margin, 300 - margin)
            for x in range(margin, 300 - margin)
            if (img.pixel(x, y) >> 24) != 0
        )

    ink_preview = _interior_ink(False)
    ink_crop = _interior_ink(True)
    check(
        "passive preview draws no thirds grid",
        ink_preview == 0,
        f"{ink_preview} interior pixels painted",
    )
    check(
        "explicit crop mode still draws the thirds grid",
        ink_crop > 0,
        f"{ink_crop} interior pixels painted",
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
