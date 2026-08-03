"""Brush Size set from the Masks tab reaches the brush, not just the number.

The Masks tab mirrors Brush Size / Flow / Feather from the Local section so
they are to hand while masking. The mirror copies the value to its twin
under blockSignals, to stop a drag ping-ponging between the two.

That suppressed the twin's valueChanged -- and for these three sliders
valueChanged is the ONLY thing that emits dodge_burn_brush_changed, which is
what resizes the brush ring on the canvas. So dragging Brush Size in the
Masks tab moved the number in both places and never told the brush. The ring
stayed exactly as it was, which is indistinguishable from a dead control,
and the Masks tab is the one place you would be using it from.

It tried to compensate by emitting sliderReleased on the twin. Nothing
connects sliderReleased on _db_size/_db_strength/_db_feather, so that
reached nothing at all -- the test below would have passed against
sliderReleased while the app stayed broken, which is why it asserts on the
signal the cursor actually listens to.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


_app = QApplication.instance() or QApplication([])


def main() -> int:
    from rawviewer_ui.adjust_panel import ImageAdjustPanelWidget

    p = ImageAdjustPanelWidget()
    p.show()

    pairs = (
        ("Brush Size", "_db_size_slider"),
        ("Brush Flow", "_db_strength_slider"),
        ("Brush Feather", "_db_feather_slider"),
    )

    for label, attr in pairs:
        source = getattr(p, attr)
        mirror = p._mask_brush_sliders[label]
        check(
            f"{label}: the two sliders share a range",
            (mirror.minimum(), mirror.maximum())
            == (source.minimum(), source.maximum()),
            f"{(mirror.minimum(), mirror.maximum())} vs "
            f"{(source.minimum(), source.maximum())}",
        )

        fired = []
        conn = p.dodge_burn_brush_changed.connect(lambda: fired.append(1))

        # --- Masks tab -> Local, the direction that was broken ---
        target = min(source.maximum(), source.value() + 40)
        mirror.setValue(target)
        check(f"{label}: the Masks tab writes through", source.value() == target)
        check(
            f"{label}: and the brush is told",
            len(fired) >= 1,
            "without this the ring on screen keeps the old size",
        )

        # --- Local -> Masks tab, which already worked ---
        fired.clear()
        back = max(source.minimum(), source.value() - 20)
        source.setValue(back)
        check(f"{label}: the Local section writes through", mirror.value() == back)
        check(f"{label}: and the brush is told", len(fired) >= 1)

        # --- no echo: one move must not bounce between the two ---
        fired.clear()
        mirror.setValue(min(source.maximum(), mirror.value() + 5))
        check(
            f"{label}: one move emits once, not repeatedly",
            len(fired) == 1,
            f"{len(fired)} emissions -- a ping-pong would spam the cursor",
        )
        p.dodge_burn_brush_changed.disconnect(conn)

    # --- the value the brush actually reads follows the Masks tab ---
    p._mask_brush_sliders["Brush Size"].setValue(240)
    check(
        "brush_size_value() reports what the Masks tab set",
        float(p._db_size_slider.value()) == 240.0,
        str(p._db_size_slider.value()),
    )

    # --- setting the same value again is not an event ---
    fired = []
    p.dodge_burn_brush_changed.connect(lambda: fired.append(1))
    p._mask_brush_sliders["Brush Size"].setValue(240)
    check("re-setting the same value is silent", len(fired) == 0, str(len(fired)))

    # --- Edge Assist: same shared toggle, reachable while masking ---
    edge = p._db_edge_btn
    mirror_edge = p._mask_edge_btn
    check("Edge Assist mirror exists", mirror_edge is not None and mirror_edge.isCheckable())
    mirror_edge.setChecked(False)
    check("Masks Edge Assist writes through", edge.isChecked() is False)
    check(
        "dodge_burn_edge_assist follows the Masks toggle",
        p.dodge_burn_edge_assist() is False,
    )
    edge.setChecked(True)
    check("Local Edge Assist writes through", mirror_edge.isChecked() is True)
    check(
        "dodge_burn_edge_assist follows the Local toggle",
        p.dodge_burn_edge_assist() is True,
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
