"""Mid-stroke two-finger brush resize: suspend stamping, resize, resume.

Drives synthetic wheel events with macOS-style scroll phases against the view
to check the suspend/resume state machine: while the resize gesture is active a
hold-to-paint stroke must stop stamping WITHOUT ending (one undo unit), resize
must still fire, and lifting the fingers must resume and reset the stamp anchor.
Momentum after lift must not keep resizing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "rawviewer_ui"))

from PyQt6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QWheelEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])


def _make_view():
    from rawviewer_ui.gpu_image_view import GpuImageView

    v = GpuImageView()
    v._has_pixmap = True
    v._dodge_burn_mode = True
    v._view_pos_on_image = lambda pos=None: True
    v._clamped_scene_point = lambda pos: QPointF(10.0, 20.0)
    v._place_brush_cursor = lambda pt: None
    strokes: list = []
    resizes: list = []
    resumes: list = []
    v.dodgeBurnStroke.connect(lambda pt, p, e: strokes.append((pt.x(), pt.y(), p, e)))
    v.dodgeBurnBrushSizeWheel.connect(lambda d: resizes.append(d))
    v.dodgeBurnResumeAfterResize.connect(lambda: resumes.append(True))
    return v, strokes, resizes, resumes


def _wheel(phase: Qt.ScrollPhase, dy: int = 30) -> QWheelEvent:
    return QWheelEvent(
        QPointF(50, 50),
        QPointF(50, 50),
        QPoint(0, 0),
        QPoint(0, dy),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        phase,
        False,
    )


class _Move:
    def buttons(self):
        return Qt.MouseButton.NoButton

    def position(self):
        return QPointF(60, 60)

    def accept(self):
        pass


def test_resize_gesture_suspends_and_resumes_stroke() -> None:
    v, strokes, resizes, resumes = _make_view()
    # Start a hold-to-paint stroke (key down, pointer on image).
    v.begin_key_paint()
    assert len(strokes) == 1 and strokes[0][3] is False  # opening stamp

    # Second finger lands: ScrollBegin -> suspend.
    v.wheelEvent(_wheel(Qt.ScrollPhase.ScrollBegin))
    assert v._brush_resizing is True
    n_before = len(strokes)

    # Resize updates fire while suspended; a pointer move must NOT stamp.
    v.wheelEvent(_wheel(Qt.ScrollPhase.ScrollUpdate))
    v.mouseMoveEvent(_Move())
    assert len(strokes) == n_before, "stamped while resizing"
    assert len(resizes) >= 2, "resize did not fire during gesture"
    # Crucially the stroke never ended (still one undo unit).
    assert not any(s[3] for s in strokes), "stroke ended during resize"

    # Fingers lift: ScrollEnd -> resume + anchor reset, no resize on the end tick.
    resizes_at_end = len(resizes)
    v.wheelEvent(_wheel(Qt.ScrollPhase.ScrollEnd, dy=0))
    assert v._brush_resizing is False
    assert resumes == [True]
    assert len(resizes) == resizes_at_end, "ScrollEnd wrongly emitted a resize"

    # Painting resumes.
    v.mouseMoveEvent(_Move())
    assert len(strokes) == n_before + 1 and strokes[-1][3] is False

    # Key release closes the single stroke.
    v.end_key_paint()
    assert strokes[-1][3] is True


def test_momentum_does_not_resize() -> None:
    v, _s, resizes, _r = _make_view()
    v.begin_key_paint()
    v.wheelEvent(_wheel(Qt.ScrollPhase.ScrollBegin))
    v.wheelEvent(_wheel(Qt.ScrollPhase.ScrollEnd, dy=0))
    n = len(resizes)
    # Inertial momentum after the fingers lifted must be ignored.
    v.wheelEvent(_wheel(Qt.ScrollPhase.ScrollMomentum))
    assert len(resizes) == n, "momentum kept resizing the brush"


def test_no_phase_falls_back_to_plain_resize() -> None:
    """Mouse wheel / platforms without phase: resize per tick, no suspend."""
    v, _s, resizes, _r = _make_view()
    v.begin_key_paint()
    v.wheelEvent(_wheel(Qt.ScrollPhase.NoScrollPhase))
    assert v._brush_resizing is False
    assert len(resizes) == 1


def test_key_release_clears_resize_state() -> None:
    v, _s, _rz, _r = _make_view()
    v.begin_key_paint()
    v.wheelEvent(_wheel(Qt.ScrollPhase.ScrollBegin))
    assert v._brush_resizing is True
    v.end_key_paint()  # lift the hotkey mid-resize
    assert v._brush_resizing is False


def main() -> int:
    test_resize_gesture_suspends_and_resumes_stroke()
    test_momentum_does_not_resize()
    test_no_phase_falls_back_to_plain_resize()
    test_key_release_clears_resize_state()
    print("PASS t_brush_resize_gesture")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
