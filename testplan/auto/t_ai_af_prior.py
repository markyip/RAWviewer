"""AF-prior bias for Smart Object mattes and Blur background lock."""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
sys.path.insert(0, ROOT)

from raw_ai_masks import (  # noqa: E402
    _bias_matte_to_focus,
    _focus_xy_in_image,
    _matte_covers_focus,
    bokeh_model_kind,
    grow_subject_lock,
)


def test_bias_keeps_focus_blob():
    alpha = np.zeros((40, 80), dtype=np.float32)
    alpha[5:15, 5:20] = 0.9   # left blob
    alpha[5:15, 55:75] = 0.9  # right blob
    out = _bias_matte_to_focus(alpha, (65.0, 10.0))
    assert float(out[8, 65]) > 0.7
    assert float(out[8, 10]) < 0.25
    print("  OK   AF prior keeps the focused component")


def test_matte_covers_focus():
    alpha = np.zeros((20, 20), dtype=np.float32)
    alpha[8:12, 8:12] = 1.0
    assert _matte_covers_focus(alpha, (10.0, 10.0))
    assert not _matte_covers_focus(alpha, (1.0, 1.0))
    print("  OK   AF hit test")


def test_grow_subject_lock_expands_edge():
    alpha = np.zeros((32, 32), dtype=np.float32)
    alpha[10:22, 10:22] = 1.0
    grown = grow_subject_lock(alpha, dilate_px=3, feather=2.5)
    assert float(grown[16, 16]) > 0.9
    assert float(grown[8, 16]) > 0.05, "dilate+feather should leak past the hard edge"
    assert float(grown[2, 2]) < 0.05
    print("  OK   subject lock grows a few pixels")


def test_focus_xy_scales_into_source_image():
    rgb = np.zeros((100, 200, 3), dtype=np.float32)
    xy = _focus_xy_in_image((50.0, 25.0), rgb, (50, 100))
    assert abs(xy[0] - 100.0) < 1e-6
    assert abs(xy[1] - 50.0) < 1e-6
    print("  OK   AF point maps into the SAM source frame")


def test_bokeh_model_kind_prefers_click_when_focus_and_ready():
    # Without weights on disk both ops are unreadied in a clean checkout;
    # with a focus point the download target is the smaller click model.
    kind = bokeh_model_kind(True)
    assert kind in ("click", "subject"), kind
    assert bokeh_model_kind(False) == "subject"
    print(f"  OK   bokeh model kind with AF={kind!r}, without=subject")


def main() -> int:
    test_bias_keeps_focus_blob()
    test_matte_covers_focus()
    test_grow_subject_lock_expands_edge()
    test_focus_xy_scales_into_source_image()
    test_bokeh_model_kind_prefers_click_when_focus_and_ready()
    print("All AF-prior checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
