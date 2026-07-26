"""Tap-to-paint latch + eraser correctness.

Three behaviours the brush was expected to have and did not:

1. A brush hotkey tap should arm the tool *and* start painting on pointer
   movement. Previously the tap only selected the tool and a mouse button
   still had to be held down -- arming the same tool twice for one decision.
2. The eraser should remove mask wherever it is, including across the
   luminance boundary edge assist protects. Edge assist gating the eraser
   made painted mask on the far side of a hard edge unreachable.
3. The eraser should never ADD coverage. It did: the end-of-stroke edge
   snap ran on erase strokes too, smearing surrounding mask back into the
   hole the moment the brush was released (+35% of the erased area).

(2) and (3) compounded -- the eraser could not reach across an edge, and
gave back a third of what it did manage to remove.
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
    assert 'mode != "erase"' in src and "edge_snap_region(mask, luminance, bbox)" in src, (
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
    print("\nPASS t_tap_to_paint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
