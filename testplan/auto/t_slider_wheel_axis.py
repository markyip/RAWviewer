#!/usr/bin/env python3
"""Adjust-panel slider wheel axis -- headless.

The panel's sliders are horizontal, so on a trackpad only a horizontal
two-finger swipe should move them. Vertical used to work too, which meant
scrolling the panel snagged on whichever slider the pointer happened to be
over and silently edited it -- and asked a left-to-right control to respond
to an up-down gesture.

A real mouse wheel stays exempt: it reports vertical only, so the same rule
would leave those users unable to move any slider at all. That exemption is
the part most at risk of being "cleaned up" later, so it is asserted here.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from PyQt6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QWheelEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from rawviewer_ui.adjust_panel import AdjustSlider  # noqa: E402

TRACKPAD = Qt.ScrollPhase.ScrollUpdate
WHEEL = Qt.ScrollPhase.NoScrollPhase


def _slider():
    s = AdjustSlider(Qt.Orientation.Horizontal)
    s.setRange(-100, 100)
    s.setSingleStep(1)
    s.setValue(0)
    return s


def _wheel(pixel=(0, 0), angle=(0, 0), phase=WHEEL):
    return QWheelEvent(
        QPointF(10, 10),
        QPointF(10, 10),
        QPoint(*pixel),
        QPoint(*angle),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        phase,
        False,
    )


def _send(**kw):
    """Returns (resulting value, whether the slider consumed the event)."""
    s = _slider()
    ev = _wheel(**kw)
    s.wheelEvent(ev)
    return s.value(), ev.isAccepted()


def test_trackpad_horizontal_moves_the_slider():
    right, accepted = _send(pixel=(12, 0), phase=TRACKPAD)
    assert right > 0 and accepted, f"swipe right did nothing (value {right})"
    left, accepted = _send(pixel=(-12, 0), phase=TRACKPAD)
    assert left < 0 and accepted, f"swipe left did nothing (value {left})"
    print("  OK   trackpad horizontal moves the slider both ways")


def test_trackpad_vertical_is_left_for_the_scroll_area():
    for dy in (12, -12):
        value, accepted = _send(pixel=(0, dy), phase=TRACKPAD)
        assert value == 0, f"vertical swipe changed the slider to {value}"
        assert not accepted, "vertical swipe was consumed; the panel cannot scroll"
    print("  OK   trackpad vertical is ignored and propagates")


def test_diagonal_locks_to_the_dominant_axis():
    value, accepted = _send(pixel=(12, 4), phase=TRACKPAD)
    assert value > 0 and accepted, "mostly-horizontal swipe should move the slider"
    value, accepted = _send(pixel=(4, 12), phase=TRACKPAD)
    assert value == 0 and not accepted, "mostly-vertical swipe should scroll instead"
    print("  OK   diagonal swipes lock to the dominant axis")


def test_mouse_wheel_still_works_vertically():
    """The exemption. A wheel has no horizontal axis to offer."""
    up, accepted = _send(angle=(0, 120), phase=WHEEL)
    assert up > 0 and accepted, f"mouse wheel up no longer adjusts (value {up})"
    down, accepted = _send(angle=(0, -120), phase=WHEEL)
    assert down < 0 and accepted, f"mouse wheel down no longer adjusts (value {down})"
    print("  OK   mouse wheel still adjusts vertically")


def test_one_step_per_notch():
    """Qt's default is singleStep * wheelScrollLines, which overshoots."""
    s = _slider()
    s.setSingleStep(1)
    s.wheelEvent(_wheel(angle=(0, 120)))
    assert s.value() == 1, f"expected exactly one step, got {s.value()}"
    print("  OK   one step per notch")


def test_wheel_adjustment_persists():
    """Wheel changes bypass sliderReleased, so the slider re-emits it."""
    s = _slider()
    seen = []
    s.sliderReleased.connect(lambda: seen.append(True))
    s.wheelEvent(_wheel(pixel=(12, 0), phase=TRACKPAD))
    assert seen, "sliderReleased not emitted; the edit would never be saved"
    print("  OK   wheel edits emit sliderReleased so they persist")


def test_empty_delta_is_ignored():
    value, accepted = _send(pixel=(0, 0), angle=(0, 0), phase=TRACKPAD)
    assert value == 0 and not accepted
    print("  OK   a zero-delta event is ignored")


def main() -> int:
    test_trackpad_horizontal_moves_the_slider()
    test_trackpad_vertical_is_left_for_the_scroll_area()
    test_diagonal_locks_to_the_dominant_axis()
    test_mouse_wheel_still_works_vertically()
    test_one_step_per_notch()
    test_wheel_adjustment_persists()
    test_empty_delta_is_ignored()
    print("\nPASS t_slider_wheel_axis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
