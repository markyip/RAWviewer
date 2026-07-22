"""Hold-to-paint brush control: paint gate decoupled from brush context.

Exercises the view-side key-paint state machine and the panel force-arm path
without a full app. The host wiring (key routing, focus-loss abort) is thin
glue over these; the tricky invariants -- one stroke per hold, multi-key
last-wins, idempotent end -- live here.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "rawviewer_ui"))

from PyQt6.QtCore import QPointF  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])


def _make_view(on_image: bool = True):
    from rawviewer_ui.gpu_image_view import GpuImageView

    v = GpuImageView()
    v._has_pixmap = True
    v._dodge_burn_mode = True  # brush context armed
    # Stub geometry so the state machine runs without a real transform/pixmap.
    v._view_pos_on_image = lambda pos=None: on_image
    v._clamped_scene_point = lambda pos: QPointF(10.0, 20.0)
    v._place_brush_cursor = lambda pt: None
    strokes: list = []
    v.dodgeBurnStroke.connect(lambda pt, pressure, is_end: strokes.append((pt.x(), pt.y(), pressure, is_end)))
    return v, strokes


def test_tap_over_image_is_single_dab() -> None:
    """Key down then up with no movement -> exactly one start + one end."""
    v, strokes = _make_view(on_image=True)
    v.begin_key_paint()
    v.end_key_paint()
    assert len(strokes) == 2, strokes
    assert strokes[0][3] is False and strokes[-1][3] is True
    assert v._db_key_held is False


def test_hold_off_image_then_enter_starts_stroke() -> None:
    """Key down off-image stamps nothing; the first on-image move starts it."""
    v, strokes = _make_view(on_image=False)
    v.begin_key_paint()
    assert strokes == [], "off-image keydown must not stamp"
    assert v._db_key_held is True
    # Pointer enters the image while the key is still held.
    v._view_pos_on_image = lambda pos=None: True

    class _Ev:
        def buttons(self):
            from PyQt6.QtCore import Qt

            return Qt.MouseButton.NoButton

        def position(self):
            return QPointF(5, 5)

        def accept(self):
            pass

    v.mouseMoveEvent(_Ev())
    assert len(strokes) == 1 and strokes[0][3] is False, strokes
    v.end_key_paint()
    assert strokes[-1][3] is True


def test_end_key_paint_is_idempotent() -> None:
    """The focus-loss safety net calls end unconditionally; must not double-end."""
    v, strokes = _make_view(on_image=True)
    v.begin_key_paint()
    v.end_key_paint()
    n = len(strokes)
    v.end_key_paint()  # already closed
    assert len(strokes) == n, "second end_key_paint emitted a spurious stroke"


def test_disarming_context_closes_hold() -> None:
    """Turning the context off mid-hold must not leave a dangling stroke."""
    v, strokes = _make_view(on_image=True)
    v.begin_key_paint()
    assert v._dodge_burn_painting is True
    v.set_dodge_burn_mode(False)
    assert strokes[-1][3] is True
    assert v._db_key_held is False


def test_panel_force_arm_does_not_toggle_off() -> None:
    """toggle=False must re-arm the same tool (hold-to-paint), not turn it off."""
    from rawviewer_ui.adjust_panel import ImageAdjustPanelWidget

    p = ImageAdjustPanelWidget()
    p.set_dodge_burn_mode("dodge", toggle=False)
    assert p.dodge_burn_mode() == "dodge"
    p.set_dodge_burn_mode("dodge", toggle=False)  # same again
    assert p.dodge_burn_mode() == "dodge", "force-arm wrongly toggled off"
    # Legacy default still toggles.
    p.set_dodge_burn_mode("dodge")
    assert p.dodge_burn_mode() is None, "default toggle behaviour regressed"


def main() -> int:
    test_tap_over_image_is_single_dab()
    test_hold_off_image_then_enter_starts_stroke()
    test_end_key_paint_is_idempotent()
    test_disarming_context_closes_hold()
    test_panel_force_arm_does_not_toggle_off()
    print("PASS t_hold_to_paint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
