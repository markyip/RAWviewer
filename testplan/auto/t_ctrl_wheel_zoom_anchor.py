#!/usr/bin/env python3
"""Ctrl+scroll zoom keeps the image point under the pointer fixed."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []
# Qt scroll positions are integers, so up to ~1 device px of rounding is expected.
TOL_PX = 2.0


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def ctrl_wheel(view, pos, notches_up=True, count=1):
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QWheelEvent

    for _ in range(count):
        delta = 120 if notches_up else -120
        ev = QWheelEvent(
            QPointF(pos),
            QPointF(view.viewport().mapToGlobal(pos)),
            QPoint(0, 0),
            QPoint(0, delta),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.ControlModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        view.wheelEvent(ev)


def drift_px(view, pos, before_scene):
    after = view.mapToScene(pos)
    dx = after.x() - before_scene.x()
    dy = after.y() - before_scene.y()
    return (dx * dx + dy * dy) ** 0.5 * view.current_scale()


def main() -> int:
    from PyQt6.QtCore import QPoint
    from PyQt6.QtGui import QColor, QPixmap
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841
    from rawviewer_ui.widgets.gpu import GpuImageView

    view = GpuImageView()
    view.resize(800, 600)
    view.show()
    app.processEvents()

    pix = QPixmap(4000, 3000)
    pix.fill(QColor(120, 120, 120))
    view.set_pixmap(pix)
    app.processEvents()

    positions = {
        "centre": QPoint(400, 300),
        "upper-left quadrant": QPoint(220, 170),
        "lower-right quadrant": QPoint(580, 430),
    }
    for label, pos in positions.items():
        view.fit_to_window()
        app.processEvents()
        ctrl_wheel(view, pos, count=3)
        app.processEvents()
        before = view.mapToScene(pos)
        ctrl_wheel(view, pos, count=1)
        app.processEvents()
        drift = drift_px(view, pos, before)
        check(f"ctrl+wheel keeps point fixed ({label})", drift <= TOL_PX, f"drift={drift:.2f}px")

    view.fit_to_window()
    app.processEvents()
    pos = QPoint(300, 260)
    ctrl_wheel(view, pos, count=3)
    app.processEvents()
    anchor_scene = view.mapToScene(pos)
    old_frac = (anchor_scene.x() / 4000.0, anchor_scene.y() / 3000.0)
    full = QPixmap(8000, 6000)
    full.fill(QColor(110, 110, 110))
    view.set_pixmap(full, preserve_view=True)
    app.processEvents()
    new_scene = view.mapToScene(pos)
    new_frac = (new_scene.x() / 8000.0, new_scene.y() / 6000.0)
    drift = max(
        abs(new_frac[0] - old_frac[0]) * 8000.0,
        abs(new_frac[1] - old_frac[1]) * 6000.0,
    ) * view.current_scale()
    check("full-res swap keeps pointer point fixed", drift <= TOL_PX, f"drift={drift:.2f}px")

    view.fit_to_window()
    app.processEvents()
    check("fit clears stored anchor", getattr(view, "_zoom_anchor", None) is None)

    view.close()
    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
