"""Brush arming model + eraser correctness.

Arming. A brush hotkey is momentary: hold D/B/X/H and sweep to paint with
no mouse button, release to close the stroke AND disarm the tool, so the
key press is the whole interaction. Previously the hotkey only selected the
tool and a mouse button still had to be held to paint at all. The panel's
tool buttons remain the click-to-arm-and-stay alternative.

The latch tests below drive that same paint gate directly. Latching is what
the opt-out path (RAWVIEWER_HOLD_TO_PAINT=0) uses -- tap on, tap off -- and
the held key opens the identical gate, so exercising it covers both.

Viewing. The mask overlay toggle needs no armed tool. It was disabled
without one, which put the answer to "is anything masked?" behind arming a
tool, and would now grey out the instant a stroke ended.

Erasing, two compounding defects:

1. Edge assist gated the eraser, so mask on the far side of a luminance
   edge could not be removed at all.
2. The end-of-stroke edge snap ran on erase strokes too, smearing
   surrounding mask back into the hole the moment the brush was released
   (+35% of the erased area) -- the eraser visibly ADDED coverage.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "rawviewer_ui"))

import numpy as np  # noqa: E402
from PyQt6.QtCore import QPointF, Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])


class _MoveEvent:
    """Minimal stand-in for QMouseEvent with no button held."""

    def __init__(self, buttons=Qt.MouseButton.NoButton):
        self._buttons = buttons

    def buttons(self):
        return self._buttons

    def position(self):
        return QPointF(10.0, 20.0)

    def accept(self):
        pass


def _make_view(on_image: bool = True):
    from rawviewer_ui.gpu_image_view import GpuImageView

    v = GpuImageView()
    v._has_pixmap = True
    v._dodge_burn_mode = True
    v._view_pos_on_image = lambda pos=None: on_image
    v._clamped_scene_point = lambda pos: QPointF(10.0, 20.0)
    v._place_brush_cursor = lambda pt: None
    strokes: list = []
    v.dodgeBurnStroke.connect(
        lambda pt, pressure, is_end: strokes.append((pt.x(), pt.y(), pressure, is_end))
    )
    return v, strokes


# ---------------------------------------------------------------- tap latch


def test_latch_survives_key_release():
    """The whole point: no key held, no button down, still painting."""
    v, _ = _make_view()
    v.begin_latched_paint()
    assert v.is_paint_latched() is True
    assert v._db_key_held is True
    print("  OK   tap latches the paint gate open")


def test_latched_move_paints_without_any_button():
    v, strokes = _make_view(on_image=False)
    v.begin_latched_paint()
    assert strokes == [], "off-image tap must not stamp"
    v._view_pos_on_image = lambda pos=None: True
    v.mouseMoveEvent(_MoveEvent())  # NoButton
    assert strokes, "a latched pointer move with no button held must paint"
    assert strokes[-1][3] is False, "move should be a stamp, not a stroke end"
    print("  OK   pointer move paints with no mouse button")


def test_end_key_paint_clears_the_latch():
    v, _ = _make_view()
    v.begin_latched_paint()
    v.end_key_paint()
    assert v.is_paint_latched() is False
    assert v._db_key_held is False
    print("  OK   disarming clears the latch")


def test_move_after_unlatch_does_not_paint():
    """A cleared latch must actually stop stamping, not just flip a flag."""
    v, strokes = _make_view()
    v.begin_latched_paint()
    v.end_key_paint()
    strokes.clear()
    try:
        v.mouseMoveEvent(_MoveEvent())
    except TypeError:
        # Reaching QGraphicsView.mouseMoveEvent means the handler fell past
        # every paint branch, which is exactly the outcome under test -- the
        # real QMouseEvent it wants cannot be built for a stubbed view.
        pass
    assert strokes == [], f"painted after disarm: {strokes}"
    print("  OK   no painting once disarmed")


def test_latch_is_idempotent():
    v, _ = _make_view()
    v.begin_latched_paint()
    v.begin_latched_paint()
    v.end_key_paint()
    v.end_key_paint()
    assert v.is_paint_latched() is False
    print("  OK   double latch / double release is safe")


# ------------------------------------------------------------- eraser reach


def _edge_scene():
    """200x200 with a hard luminance edge at x=100, dodged across all of it."""
    from raw_dodge_burn import DodgeBurnMask, stamp_brush

    h = w = 200
    luma = np.zeros((h, w), np.float32)
    luma[:, 100:] = 0.8
    mask = DodgeBurnMask.empty(h, w)
    stamp_brush(mask, 100, 100, 80, 0.9, dodge=True, luminance=luma, edge_assist=False)
    return mask, luma


def test_eraser_crosses_a_luminance_edge():
    from raw_dodge_burn import DodgeBurnMask, erase_brush

    mask, luma = _edge_scene()
    far_before = float(mask.data[:, 100:].mean())
    assert far_before > 0.05, "fixture did not paint the far side"

    work = DodgeBurnMask(mask.data.copy())
    for _ in range(12):
        # Brush centred on the DARK side, reaching across the edge.
        erase_brush(work, 60, 100, 80, 1.0, luminance=luma, edge_assist=True)
    far_after = float(work.data[:, 100:].mean())
    assert far_after < far_before * 0.75, (
        f"eraser could not reach across the edge: {far_before:.3f} -> {far_after:.3f}"
    )
    print(f"  OK   eraser crosses the edge ({far_before:.3f} -> {far_after:.3f})")


def test_edge_assist_flag_no_longer_changes_erasing():
    from raw_dodge_burn import DodgeBurnMask, erase_brush

    mask, luma = _edge_scene()
    results = []
    for edge_assist in (True, False):
        work = DodgeBurnMask(mask.data.copy())
        for _ in range(12):
            erase_brush(work, 60, 100, 80, 1.0, luminance=luma, edge_assist=edge_assist)
        results.append(work.data.copy())
    assert np.allclose(results[0], results[1]), "edge_assist still gates the eraser"
    print("  OK   edge_assist is inert for the eraser")


def test_painting_still_respects_edge_assist():
    """The fix must not disarm edge assist for the tools it was built for."""
    from raw_dodge_burn import DodgeBurnMask, stamp_brush

    h = w = 200
    luma = np.zeros((h, w), np.float32)
    luma[:, 100:] = 0.8
    mask = DodgeBurnMask.empty(h, w)
    stamp_brush(mask, 60, 100, 80, 0.9, dodge=True, luminance=luma, edge_assist=True)
    far = float(np.abs(mask.data[:, 100:]).mean())
    near = float(np.abs(mask.data[:, :100]).mean())
    assert near > 0.05, "paint did not land on the seed side"
    assert far < 0.01, f"paint bled across the edge: {far:.4f}"
    print(f"  OK   paint still blocked at the edge (near {near:.3f}, far {far:.4f})")


def test_eraser_removes_the_effect_not_just_coverage():
    """Mask value *is* the effect, so a cleared area must apply no gain."""
    from raw_dodge_burn import DodgeBurnMask, apply_dodge_burn, erase_brush, stamp_brush

    h = w = 120
    mask = DodgeBurnMask.empty(h, w)
    stamp_brush(mask, 60, 60, 40, 1.0, dodge=True, edge_assist=False)
    img = np.full((h, w, 3), 0.25, np.float32)
    lifted = apply_dodge_burn(img.copy(), mask, 1.75)
    assert lifted[60, 60, 0] > 0.30, "fixture did not brighten the centre"

    for _ in range(10):
        erase_brush(mask, 60, 60, 40, 1.0, edge_assist=False)
    restored = apply_dodge_burn(img.copy(), mask, 1.75)
    assert abs(float(restored[60, 60, 0]) - 0.25) < 0.005, (
        f"dodge effect survived the erase: {restored[60, 60, 0]:.4f}"
    )
    print("  OK   erasing removes the dodge effect, not just the coverage")


def test_release_must_not_put_erased_mask_back():
    """The end-of-stroke edge snap must not run on an erase stroke.

    Edge snap is a guided filter that pulls mask edges onto image edges --
    a tidy-up for paint. Run over a just-cleared region it smears the
    surrounding mask back into the hole, so releasing the eraser ADDED
    coverage. Measured at +35% of the erased area returning on release.
    """
    from raw_dodge_burn import DodgeBurnMask, edge_snap_region, erase_brush, stamp_brush

    h = w = 240
    luma = np.full((h, w), 0.45, np.float32)
    mask = DodgeBurnMask.empty(h, w)
    stamp_brush(mask, 120, 120, 90, 1.0, dodge=True, edge_assist=False)

    bbox = None
    for _ in range(8):
        bbox = erase_brush(mask, 120, 120, 40, 1.0, edge_assist=False)
    erased_total = float(mask.data.sum())

    # What the host does at is_end for a PAINT stroke. It must never be
    # reached for an erase stroke -- this asserts the damage it would do,
    # which is what the mode guard in _on_dodge_burn_stroke prevents.
    snapped = DodgeBurnMask(mask.data.copy())
    edge_snap_region(snapped, luma, bbox)
    assert float(snapped.data.sum()) > erased_total * 1.05, (
        "fixture no longer reproduces the regrowth this guard exists to prevent"
    )

    # The guard itself: main.py must gate the snap on the stroke mode.
    src = (REPO / "src" / "main.py").read_text(encoding="utf-8")
    assert 'mode != "erase"' in src and "edge_snap_region(mask, luminance, " in src, (
        "edge_snap_region is no longer guarded against erase strokes"
    )
    print(f"  OK   erase strokes skip the release edge-snap (would regrow "
          f"{100 * (float(snapped.data.sum()) / erased_total - 1):.0f}%)")


def test_erasing_only_ever_reduces_the_mask():
    """Whatever the brush does, |mask| must be monotonically non-increasing."""
    from raw_dodge_burn import DodgeBurnMask, erase_brush, stamp_brush

    h = w = 160
    mask = DodgeBurnMask.empty(h, w)
    stamp_brush(mask, 80, 80, 60, 1.0, dodge=True, edge_assist=False)
    stamp_brush(mask, 40, 40, 30, 0.8, dodge=False, edge_assist=False)

    luma = np.zeros((h, w), np.float32)
    luma[:, 80:] = 0.9
    total = float(np.abs(mask.data).sum())
    for step in range(10):
        erase_brush(mask, 70 + step, 80, 35, 0.6, luminance=luma, edge_assist=True)
        now = float(np.abs(mask.data).sum())
        assert now <= total + 1e-4, f"erase step {step} grew the mask: {total} -> {now}"
        total = now
    print("  OK   erasing never increases mask magnitude")


# ------------------------------------------------- release disarms the tool


class _StubPanel:
    """Only the surface _handle_brush_key_up / _abort_hold_to_paint touch."""

    def __init__(self):
        self.mode = None
        self.disarm_calls = 0

    def set_dodge_burn_mode(self, mode, toggle=True):
        self.mode = mode

    def dodge_burn_mode(self):
        return self.mode

    def disarm_dodge_burn(self):
        self.disarm_calls += 1
        self.mode = None

    def mask_layer_mode(self):
        return None

    def disarm_mask_layer_tools(self):
        pass


class _StubView:
    def __init__(self):
        self.began = 0
        self.ended = 0

    def begin_key_paint(self):
        self.began += 1

    def end_key_paint(self):
        self.ended += 1

    def begin_latched_paint(self):
        self.began += 1


def _host():
    """A bare object carrying just the host methods under test."""
    import main

    class _Host:
        _handle_brush_key_down = main.RAWImageViewer._handle_brush_key_down
        _handle_brush_key_up = main.RAWImageViewer._handle_brush_key_up
        _abort_hold_to_paint = main.RAWImageViewer._abort_hold_to_paint

    h = _Host()
    h.single_image_adjust_panel = _StubPanel()
    h.gpu_view = _StubView()
    h._brush_key_held = None
    return h, main


def test_hold_to_paint_is_the_default():
    import main

    assert main.HOLD_TO_PAINT is True, "hold-to-paint must be the shipped default"
    print("  OK   hold-to-paint is the default model")


def test_key_release_disarms_the_tool():
    h, _ = _host()
    assert h._handle_brush_key_down("dodge") is True
    assert h.single_image_adjust_panel.mode == "dodge"
    assert h.gpu_view.began == 1

    assert h._handle_brush_key_up("dodge") is True
    assert h.gpu_view.ended == 1, "stroke was not closed"
    assert h.single_image_adjust_panel.disarm_calls == 1, "tool was not disarmed"
    assert h.single_image_adjust_panel.mode is None
    assert h._brush_key_held is None
    print("  OK   releasing the hotkey disarms the tool")


def test_superseded_key_release_does_not_disarm_the_new_tool():
    """D held, then B pressed, then D released -- B must survive."""
    h, _ = _host()
    h._handle_brush_key_down("dodge")
    h._handle_brush_key_down("burn")
    assert h.single_image_adjust_panel.mode == "burn"

    assert h._handle_brush_key_up("dodge") is False, "stale key claimed the release"
    assert h.single_image_adjust_panel.mode == "burn", "released stale key disarmed Burn"
    assert h.single_image_adjust_panel.disarm_calls == 0
    print("  OK   a superseded key's release leaves the new tool armed")


def test_focus_loss_abort_also_disarms():
    """A swallowed key-up must not leave a live tool behind."""
    h, _ = _host()
    h._handle_brush_key_down("heal")
    h._abort_hold_to_paint()
    assert h.gpu_view.ended == 1
    assert h.single_image_adjust_panel.disarm_calls == 1
    assert h._brush_key_held is None
    print("  OK   focus-loss abort disarms too")


# ------------------------------------------------ mask toggle without a tool


def test_mask_overlay_togglable_with_nothing_armed():
    from rawviewer_ui.adjust_panel import ImageAdjustPanelWidget

    p = ImageAdjustPanelWidget()
    p.show()
    assert p.dodge_burn_mode() is None, "fixture should start with no tool armed"
    assert p._db_show_mask_btn.isEnabled(), "mask toggle disabled with no tool armed"

    seen = []
    p.dodgeBurnMaskToggled.connect(seen.append)
    p.toggle_dodge_burn_show_mask()
    assert p._db_show_mask_btn.isChecked() is True, "M did not turn the overlay on"
    p.toggle_dodge_burn_show_mask()
    assert p._db_show_mask_btn.isChecked() is False, "M did not turn the overlay off"
    assert seen == [True, False], f"overlay signal not emitted both ways: {seen}"
    print("  OK   mask overlay toggles with no brush armed")


def test_mask_toggle_stays_enabled_after_disarm():
    """Release-disarms must not grey out the overlay the moment a stroke ends."""
    from rawviewer_ui.adjust_panel import ImageAdjustPanelWidget

    p = ImageAdjustPanelWidget()
    p.show()
    p.set_dodge_burn_mode("dodge")
    assert p._db_show_mask_btn.isEnabled()
    p.disarm_dodge_burn()
    assert p._db_show_mask_btn.isEnabled(), "mask toggle greyed out on disarm"
    print("  OK   mask toggle survives disarming")


# ------------------------------------------------- release must not print a box


def _smooth_luma(h, w, seed=0):
    import cv2

    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.random((h, w), dtype=np.float32), (0, 0), 12.0) * 0.2 + 0.4


def test_release_snap_leaves_no_rectangular_edge():
    """The end-of-stroke snap must not print its own bbox into the mask.

    It used to paste the guided-filter result over the padded bbox. The filter
    legitimately lowers values inside (measured 0.563 -> 0.488) while the pixel
    just outside kept the unfiltered value, so the rectangle's straight edge
    landed in the mask as a step ~19x the local gradient -- a visible square
    around wherever the brush stopped.
    """
    from raw_dodge_burn import DodgeBurnMask, edge_snap_region, stamp_brush

    h = w = 400
    luma = _smooth_luma(h, w)
    for strength, label in ((0.10, "soft"), (0.35, "medium")):
        mask = DodgeBurnMask.empty(h, w)
        bbox = None
        for x in range(80, 320, 8):
            bbox = stamp_brush(
                mask, x, 200, 45, strength, dodge=True, luminance=luma, edge_assist=False
            )
        before = mask.data.copy()
        edge_snap_region(mask, luma, bbox)

        x0, _y0, _x1, _y1 = bbox
        left = max(0, x0 - 16)  # the padded rect's left edge
        row = 200
        outside = float(mask.data[row, left - 3 : left].mean())
        inside = float(mask.data[row, left : left + 3].mean())
        step = abs(inside - outside)
        local = float(np.abs(np.diff(before[row, left + 10 : left + 60])).mean())
        # Two ways to pass, because a solid stroke has no local gradient to
        # compare against and the ratio then divides by ~0: either the step is
        # in scale with its surroundings, or it is too small to see at all.
        # The bug this guards produced a step of 0.074 -- three orders of
        # magnitude above this floor.
        assert step < 1e-3 or step < 5.0 * max(local, 1e-6), (
            f"{label}: rectangular edge at x={left}: step {step:.5f} vs local "
            f"gradient {local:.5f} ({step / max(local, 1e-9):.1f}x)"
        )
    print("  OK   release snap leaves no rectangular edge")


def test_release_snap_still_snaps_to_real_edges():
    """Feathering must not defeat the point of the snap."""
    from raw_dodge_burn import DodgeBurnMask, edge_snap_region, stamp_brush

    h = w = 300
    luma = np.zeros((h, w), np.float32)
    luma[:, 150:] = 0.9
    mask = DodgeBurnMask.empty(h, w)
    bbox = stamp_brush(mask, 150, 150, 70, 0.8, dodge=True, luminance=luma, edge_assist=False)
    before = mask.data.copy()
    edge_snap_region(mask, luma, bbox)

    core = slice(120, 180)

    def jump(a):
        return abs(float(a[core, 148:150].mean()) - float(a[core, 150:152].mean()))

    assert jump(mask.data) > jump(before) * 2.0, (
        f"snap no longer aligns the mask to the luminance edge: "
        f"{jump(before):.4f} -> {jump(mask.data):.4f}"
    )
    print(f"  OK   snap still pulls onto real edges ({jump(before):.4f} -> {jump(mask.data):.4f})")


def test_feather_window_shape_and_bounds():
    from raw_dodge_burn import _feather_window

    win = _feather_window(80, 80, 16, feather_top=True, feather_bottom=True,
                          feather_left=True, feather_right=True)
    assert win.shape == (80, 80)
    assert 0.0 < win.min() < 0.2, f"border weight should be near 0, got {win.min()}"
    assert abs(win.max() - 1.0) < 1e-6, "interior weight should reach 1"
    assert abs(win[40, 40] - 1.0) < 1e-6, "centre must be fully snapped"

    # A side on the image border is not feathered -- nothing outside to blend.
    edge = _feather_window(80, 80, 16, feather_top=False, feather_bottom=True,
                           feather_left=False, feather_right=True)
    assert abs(edge[0, 0] - 1.0) < 1e-6, "un-feathered corner should stay at 1"

    # Narrower than 2*margin: still monotone up to the middle, never re-crossing.
    tiny = _feather_window(9, 9, 16, feather_top=True, feather_bottom=True,
                           feather_left=True, feather_right=True)
    mid = tiny[4, :5]
    assert np.all(np.diff(mid) >= -1e-6), f"ramp not monotone on a narrow region: {mid}"
    print("  OK   feather window is bounded, monotone, and border-aware")


# --------------------------------------- eraser clears both brush systems


def test_eraser_hotkey_arms_like_the_others():
    """X must arm through the same path as D/B/H, not a special case."""
    for mode in ("dodge", "burn", "erase", "heal"):
        h, _ = _host()
        assert h._handle_brush_key_down(mode) is True, f"{mode} hotkey not handled"
        assert h.single_image_adjust_panel.mode == mode, f"{mode} did not arm"
        assert h.gpu_view.began == 1, f"{mode} did not open the paint gate"
        assert h._handle_brush_key_up(mode) is True
        assert h.single_image_adjust_panel.mode is None, f"{mode} did not disarm"
    print("  OK   D / B / X / H all arm and disarm identically")


def test_eraser_clears_heal_coverage_too():
    """One eraser for both systems: dodge/burn mask AND heal coverage."""
    from raw_dodge_burn import DodgeBurnMask, erase_brush, stamp_brush
    from raw_spot_heal import HealMask, erase_heal_brush, stamp_heal_brush

    h = w = 160
    db = DodgeBurnMask.empty(h, w)
    stamp_brush(db, 80, 80, 40, 1.0, dodge=True, edge_assist=False)
    heal = HealMask.empty(h, w)
    stamp_heal_brush(heal, 80, 80, 40, 1.0)

    db_before = float(np.abs(db.data).sum())
    heal_before = float(np.abs(heal.data).sum())
    assert db_before > 0 and heal_before > 0, "fixture painted nothing"

    # What the host does for one erase stamp -- both systems, same brush.
    for _ in range(10):
        erase_brush(db, 80, 80, 40, 1.0, edge_assist=False)
        erase_heal_brush(heal, 80, 80, 40, 1.0)

    db_after = float(np.abs(db.data).sum())
    heal_after = float(np.abs(heal.data).sum())
    assert db_after < db_before * 0.05, f"dodge/burn not cleared: {db_before} -> {db_after}"
    assert heal_after < heal_before * 0.05, f"heal not cleared: {heal_before} -> {heal_after}"
    print("  OK   eraser clears dodge/burn and heal coverage alike")


def test_host_erase_branch_touches_both_masks():
    """Guard the host wiring, not just the primitives."""
    src = (REPO / "src" / "main.py").read_text(encoding="utf-8")
    # Scope to the stroke handler; "mode == 'erase'" appears in other handlers.
    start = src.index("def _on_dodge_burn_stroke")
    end = src.index("\n    def ", start + 1)
    body = src[start:end]
    i = body.index('if mode == "erase":')
    branch = body[i : i + 2000]
    assert "erase_brush(" in branch, "erase branch no longer erases the dodge/burn mask"
    assert "erase_heal_brush(" in branch, "erase branch no longer erases heal coverage"
    print("  OK   host erase branch drives both masks")


def test_whole_stroke_is_snapped_not_the_last_stamp():
    """Snapping only the final stamp cut a straight edge through solid paint."""
    src = (REPO / "src" / "main.py").read_text(encoding="utf-8")
    assert "edge_snap_region(mask, luminance, stroke_region)" in src, (
        "release snap no longer uses the accumulated stroke region"
    )
    assert "_dodge_burn_stroke_dirty" in src
    print("  OK   release snaps the accumulated stroke region")


def test_whole_stroke_snap_has_no_straight_cut():
    """End to end: a long stroke snapped over its full region stays smooth."""
    from raw_dodge_burn import DodgeBurnMask, edge_snap_region, stamp_brush

    h = w = 400
    luma = _smooth_luma(h, w)
    mask = DodgeBurnMask.empty(h, w)
    acc = None
    for x in range(120, 300, 8):
        b = stamp_brush(mask, x, 200, 45, 0.12, dodge=True, luminance=luma, edge_assist=False)
        acc = b if acc is None else (
            min(acc[0], b[0]), min(acc[1], b[1]), max(acc[2], b[2]), max(acc[3], b[3])
        )
    edge_snap_region(mask, luma, acc)

    # Inside the painted band, no column-to-column jump should tower over the
    # rest: a pasted rectangle showed up as a >15x outlier.
    band = mask.data[170:230, :]
    steps = np.abs(np.diff(band, axis=1))
    ratio = steps.max() / max(steps.mean(), 1e-9)
    assert ratio < 10.0, f"straight cut in the stroke: worst step {ratio:.1f}x the mean"
    print(f"  OK   whole-stroke snap stays smooth (worst step {ratio:.1f}x mean)")


# ------------------------------------------- mask brushes are momentary too


class _MaskHost:
    """Just the host methods the mask-brush keys touch."""

    def __init__(self, panel):
        import main

        self._m = main
        self._handle_brush_key_down = main.RAWImageViewer._handle_brush_key_down.__get__(self)
        self._handle_brush_key_up = main.RAWImageViewer._handle_brush_key_up.__get__(self)
        self._handle_mask_brush_key_down = main.RAWImageViewer._handle_mask_brush_key_down.__get__(self)
        self._masks_tab_is_forward = main.RAWImageViewer._masks_tab_is_forward.__get__(self)
        self.single_image_adjust_panel = panel
        self.gpu_view = _StubView()
        self._brush_key_held = None
        self.status = []

    def ensure_mask_layer_for_painting(self):
        return self.single_image_adjust_panel.active_mask_index() is not None

    def _show_status(self, msg, *a, **k):
        self.status.append(msg)


class _StubView:
    def __init__(self):
        self.began = 0
        self.ended = 0

    def begin_key_paint(self):
        self.began += 1

    def end_key_paint(self):
        self.ended += 1


def _mask_panel_and_host():
    from raw_mask_layers import MaskLayer, MaskLayerStack
    from rawviewer_ui.adjust_panel import ImageAdjustPanelWidget, reset_section_expanded_session

    reset_section_expanded_session()
    p = ImageAdjustPanelWidget()
    p.show()
    layer = MaskLayer.empty(64, 64, name="M1")
    layer.alpha[10:40, 10:40] = 1.0
    layer.touch()
    p.set_mask_layer_stack(MaskLayerStack([layer]))
    p._panel_tabs.set_current(1)
    return p, _MaskHost(p)


def test_p_holds_to_paint_the_mask():
    p, h = _mask_panel_and_host()
    assert h._handle_brush_key_down("mask_paint") is True
    assert p.mask_layer_mode() == "paint", "P did not arm the mask brush"
    assert h.gpu_view.began == 1, "P did not open the paint gate"

    assert h._handle_brush_key_up("mask_paint") is True
    assert p.mask_layer_mode() is None, "releasing P left the mask brush armed"
    assert h.gpu_view.ended == 1
    print("  OK   P holds to paint the mask, release puts it away")


def test_x_erases_whichever_system_is_in_play():
    p, h = _mask_panel_and_host()

    # Masks tab forward -> X is the mask eraser.
    assert h._masks_tab_is_forward()
    h._handle_brush_key_down("mask_erase")
    assert p.mask_layer_mode() == "erase"
    assert p.dodge_burn_mode() is None, "X armed dodge/burn while masking"
    h._handle_brush_key_up("mask_erase")
    assert p.mask_layer_mode() is None

    # Global tab -> X is the dodge/burn eraser, as it always was.
    p._panel_tabs.set_current(0)
    assert not h._masks_tab_is_forward()
    h._handle_brush_key_down("erase")
    assert p.dodge_burn_mode() == "erase"
    assert p.mask_layer_mode() is None
    h._handle_brush_key_up("erase")
    assert p.dodge_burn_mode() is None
    print("  OK   X erases in whichever system is forward")


def test_erase_key_release_follows_the_press_not_the_tab():
    """Switching tabs mid-hold must not orphan the stroke."""
    src = (REPO / "src" / "main.py").read_text(encoding="utf-8")
    assert 'held == "mask_erase"' in src, (
        "the X release maps from the current tab rather than what the press armed"
    )
    print("  OK   the release matches what the press armed")


def test_erasing_with_no_mask_says_so():
    from rawviewer_ui.adjust_panel import ImageAdjustPanelWidget, reset_section_expanded_session

    reset_section_expanded_session()
    p = ImageAdjustPanelWidget()
    p.show()
    p._panel_tabs.set_current(1)
    h = _MaskHost(p)
    assert h._handle_brush_key_down("mask_erase") is False, (
        "X armed an eraser with no mask to erase"
    )
    assert h.status and "paint one first" in h.status[-1], h.status
    assert h.gpu_view.began == 0, "the paint gate opened with nothing to paint"
    print("  OK   X with no mask explains instead of arming nothing")


def main() -> int:
    test_latch_survives_key_release()
    test_latched_move_paints_without_any_button()
    test_end_key_paint_clears_the_latch()
    test_move_after_unlatch_does_not_paint()
    test_latch_is_idempotent()
    test_eraser_crosses_a_luminance_edge()
    test_edge_assist_flag_no_longer_changes_erasing()
    test_painting_still_respects_edge_assist()
    test_eraser_removes_the_effect_not_just_coverage()
    test_release_must_not_put_erased_mask_back()
    test_erasing_only_ever_reduces_the_mask()
    test_hold_to_paint_is_the_default()
    test_key_release_disarms_the_tool()
    test_superseded_key_release_does_not_disarm_the_new_tool()
    test_focus_loss_abort_also_disarms()
    test_mask_overlay_togglable_with_nothing_armed()
    test_mask_toggle_stays_enabled_after_disarm()
    test_p_holds_to_paint_the_mask()
    test_x_erases_whichever_system_is_in_play()
    test_erase_key_release_follows_the_press_not_the_tab()
    test_erasing_with_no_mask_says_so()
    test_release_snap_leaves_no_rectangular_edge()
    test_release_snap_still_snaps_to_real_edges()
    test_feather_window_shape_and_bounds()
    test_eraser_hotkey_arms_like_the_others()
    test_eraser_clears_heal_coverage_too()
    test_host_erase_branch_touches_both_masks()
    test_whole_stroke_is_snapped_not_the_last_stamp()
    test_whole_stroke_snap_has_no_straight_cut()
    print("\nPASS t_tap_to_paint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
