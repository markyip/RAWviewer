"""Tap-to-paint latch + eraser reach.

Two behaviours the brush was expected to have and did not:

1. A brush hotkey tap should arm the tool *and* start painting on pointer
   movement. Previously the tap only selected the tool and a mouse button
   still had to be held down -- arming the same tool twice for one decision.
2. The eraser should remove mask wherever it is, including across the
   luminance boundary edge assist protects. Edge assist gating the eraser
   made painted mask on the far side of a hard edge unreachable.
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
    print("\nPASS t_tap_to_paint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
