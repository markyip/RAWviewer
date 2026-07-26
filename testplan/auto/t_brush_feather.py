#!/usr/bin/env python3
"""Brush Feather -- headless.

The brush profile was a fixed Hann window from centre to radius, i.e.
permanently at maximum feather: only 4.2% of the disc reached 0.9 strength and
the disc averaged 0.298, so a stamp deposited under a third of its nominal
strength and no combination of Size and Flow could produce a crisp edge.

Feather is now a two-radius profile -- full strength out to (1-feather)*radius,
then a Hann fall to zero -- shared by every brush in the app: dodge/burn, heal
and mask-layer paint all call circular_brush_falloff.

It is session tool state, not an adjustment key: it describes the brush, not
the photo. That is asserted, because "make it persist like everything else"
looks like an obvious improvement and would put a tool setting in the sidecar
of every image.
"""
import inspect
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import numpy as np  # noqa: E402

from raw_dodge_burn import (  # noqa: E402
    DEFAULT_BRUSH_FEATHER,
    DodgeBurnMask,
    circular_brush_falloff,
    erase_brush,
    stamp_brush,
)

R = 100.0
DISC = np.pi * R * R


def _profile(feather):
    return circular_brush_falloff(0, 201, 0, 201, 100.0, 100.0, R, feather)


def _solid_fraction(feather):
    return float(np.count_nonzero(_profile(feather) >= 0.9)) / DISC


def test_feather_controls_the_solid_core():
    hard = _solid_fraction(0.0)
    mid = _solid_fraction(0.5)
    soft = _solid_fraction(1.0)
    assert hard > 0.95, f"feather=0 should be a near-solid disc, got {hard:.1%}"
    assert 0.2 < mid < 0.8, f"feather=0.5 should be part solid, got {mid:.1%}"
    assert soft < 0.10, f"feather=1 should be almost entirely falloff, got {soft:.1%}"
    assert hard > mid > soft, "solid core must shrink monotonically with feather"
    print(f"  OK   solid core: {hard:.0%} -> {mid:.0%} -> {soft:.0%} across the range")


def test_feather_1_reproduces_the_old_fixed_profile():
    """Existing behaviour must still be reachable, not merely approximated."""
    p = _profile(1.0)
    dist = np.hypot(*np.mgrid[0:201, 0:201].astype(np.float32) - 100.0)
    t = np.clip(dist / R, 0.0, 1.0)
    old = (0.5 * (1.0 + np.cos(np.pi * t))).astype(np.float32)
    old[dist > R] = 0.0
    assert np.allclose(p, old, atol=1e-6), "feather=1 is not the original curve"
    print("  OK   feather=100 is exactly the old fixed profile")


def test_hard_edge_has_no_transition_band():
    p = _profile(0.0)
    interior = p[100, 100 - int(R) + 2 : 100 + int(R) - 2]
    assert np.all(interior >= 0.999), "feather=0 is not flat inside the radius"
    # A corner of the patch is ~141px from centre, comfortably outside R=100.
    assert float(p[0, 0]) == 0.0, "feather=0 leaks past the radius"
    print("  OK   feather=0 is flat inside and zero outside")


def test_nothing_ever_exceeds_the_radius():
    for feather in (0.0, 0.3, DEFAULT_BRUSH_FEATHER, 1.0):
        p = _profile(feather)
        dist = np.hypot(*np.mgrid[0:201, 0:201].astype(np.float32) - 100.0)
        assert float(p[dist > R].max(initial=0.0)) == 0.0, (
            f"feather={feather} paints outside the brush radius"
        )
    print("  OK   no feather setting paints outside the radius")


def test_profile_is_monotonic_outward():
    """A non-monotonic profile shows as rings when stamps overlap."""
    for feather in (0.2, DEFAULT_BRUSH_FEATHER, 1.0):
        row = _profile(feather)[100, 100:201]
        assert np.all(np.diff(row) <= 1e-6), f"feather={feather} profile is not monotonic"
    print("  OK   the profile falls monotonically outward")


def test_out_of_range_feather_is_clamped():
    assert np.allclose(_profile(-5.0), _profile(0.0))
    assert np.allclose(_profile(9.0), _profile(1.0))
    print("  OK   out-of-range feather clamps instead of misbehaving")


def test_default_is_usable_not_the_old_extreme():
    assert 0.0 < DEFAULT_BRUSH_FEATHER < 1.0, "default is still an extreme"
    assert _solid_fraction(DEFAULT_BRUSH_FEATHER) > 4.0 * _solid_fraction(1.0), (
        "the default barely improves on the old fixed profile"
    )
    print(
        f"  OK   default {DEFAULT_BRUSH_FEATHER:.2f} gives a "
        f"{_solid_fraction(DEFAULT_BRUSH_FEATHER):.0%} solid core "
        f"(was {_solid_fraction(1.0):.0%})"
    )


def test_every_brush_accepts_feather():
    """One profile function, six brushes -- none may be left behind."""
    import raw_mask_layers as ml
    import raw_spot_heal as sh

    for fn in (
        stamp_brush,
        erase_brush,
        sh.stamp_heal_brush,
        sh.erase_heal_brush,
        ml.stamp_mask_layer_brush,
        ml.erase_mask_layer_brush,
    ):
        params = inspect.signature(fn).parameters
        assert "feather" in params, f"{fn.__name__} does not take feather"
        assert params["feather"].default == DEFAULT_BRUSH_FEATHER, (
            f"{fn.__name__} defaults to {params['feather'].default}, not the shared default"
        )
    print("  OK   all six brushes take feather with the shared default")


def test_painting_honours_feather():
    solid = {}
    for feather in (0.0, DEFAULT_BRUSH_FEATHER, 1.0):
        mask = DodgeBurnMask.empty(200, 200)
        stamp_brush(mask, 100, 100, 60, 1.0, dodge=True, edge_assist=False, feather=feather)
        solid[feather] = int(np.count_nonzero(mask.data >= 0.9))
    assert solid[0.0] > solid[DEFAULT_BRUSH_FEATHER] > solid[1.0], (
        f"a real stroke ignored feather: {solid}"
    )
    print(f"  OK   a stroke's solid area tracks feather ({solid[0.0]} -> {solid[1.0]})")


def test_erasing_honours_feather():
    base = DodgeBurnMask.empty(200, 200)
    stamp_brush(base, 100, 100, 80, 1.0, dodge=True, edge_assist=False, feather=0.0)
    remaining = {}
    for feather in (0.0, 1.0):
        work = DodgeBurnMask(base.data.copy())
        erase_brush(work, 100, 100, 50, 1.0, edge_assist=False, feather=feather)
        remaining[feather] = float(work.data.sum())
    assert remaining[0.0] < remaining[1.0], (
        f"a hard eraser removed no more than a soft one: {remaining}"
    )
    print("  OK   a hard eraser removes more than a soft one")


def test_feather_is_not_an_adjustment_key():
    """Tool state, not an image edit -- it must not reach the sidecar."""
    from raw_adjustments import DEFAULT_ADJUSTMENTS

    for key in DEFAULT_ADJUSTMENTS:
        assert "feather" not in key.lower(), f"{key} put brush feather in the sidecar"
    print("  OK   feather is session tool state, not a persisted adjustment")


def test_panel_exposes_the_slider_and_the_view_tracks_it():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])  # noqa: F841
    from rawviewer_ui.adjust_panel import ImageAdjustPanelWidget
    from rawviewer_ui.gpu_image_view import GpuImageView

    p = ImageAdjustPanelWidget()
    p.show()
    assert abs(p.dodge_burn_brush_feather() - DEFAULT_BRUSH_FEATHER) < 0.011, (
        f"panel default is {p.dodge_burn_brush_feather()}"
    )
    p._db_feather_slider.setValue(0)
    assert p.dodge_burn_brush_feather() == 0.0
    p._db_feather_slider.setValue(100)
    assert p.dodge_burn_brush_feather() == 1.0

    # The cursor preview must rebuild when feather changes, or the ring keeps
    # describing a brush that no longer exists.
    v = GpuImageView()
    v.set_dodge_burn_brush_feather(0.2)
    assert abs(v._dodge_burn_brush_feather - 0.2) < 1e-6
    src = inspect.getsource(type(v)._ensure_brush_cursor_pixmap)
    assert "feather" in src, "cursor preview ignores feather"
    assert "_brush_cursor_pixmap_feather" in src, "feather is not in the preview cache key"
    print("  OK   panel slider drives it; the cursor preview tracks it")


def main() -> int:
    test_feather_controls_the_solid_core()
    test_feather_1_reproduces_the_old_fixed_profile()
    test_hard_edge_has_no_transition_band()
    test_nothing_ever_exceeds_the_radius()
    test_profile_is_monotonic_outward()
    test_out_of_range_feather_is_clamped()
    test_default_is_usable_not_the_old_extreme()
    test_every_brush_accepts_feather()
    test_painting_honours_feather()
    test_erasing_honours_feather()
    test_feather_is_not_an_adjustment_key()
    test_panel_exposes_the_slider_and_the_view_tracks_it()
    print("\nPASS t_brush_feather")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
