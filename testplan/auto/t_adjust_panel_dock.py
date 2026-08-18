#!/usr/bin/env python3
"""Adjust panel left/right dock (QSplitter), without a full viewer window."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication, QSplitter, QWidget  # noqa: E402

_app = QApplication.instance() or QApplication([])

from rawviewer_ui.adjust_panel import (  # noqa: E402
    ADJUST_DOCK_DEFAULT,
    ADJUST_DOCK_SETTINGS_KEY,
    ImageAdjustPanelWidget,
    apply_adjust_splitter_dock,
    normalize_adjust_dock,
)

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def test_normalize_dock():
    check("default dock is right", normalize_adjust_dock(None) == ADJUST_DOCK_DEFAULT)
    check("left is accepted", normalize_adjust_dock("left") == "left")
    check("LEFT is accepted", normalize_adjust_dock("LEFT") == "left")
    check("junk falls back to right", normalize_adjust_dock("floating") == "right")
    check("settings key is adjust/dock", ADJUST_DOCK_SETTINGS_KEY == "adjust/dock")


def _shown_splitter(side_first="right"):
    splitter = QSplitter(Qt.Orientation.Horizontal)
    image = QWidget()
    panel = QWidget()
    panel.setMinimumWidth(440)
    splitter.addWidget(image)
    splitter.addWidget(panel)
    panel.show()
    image.show()
    splitter.resize(1200, 800)
    splitter.show()
    _app.processEvents()
    apply_adjust_splitter_dock(splitter, image, panel, side_first)
    _app.processEvents()
    return splitter, image, panel


def test_splitter_left_flips_x():
    splitter, image, panel = _shown_splitter("right")
    apply_adjust_splitter_dock(splitter, image, panel, "right")
    _app.processEvents()
    right_panel_x = panel.x()
    right_image_x = image.x()
    check("default order is image then panel", splitter.widget(0) is image and splitter.widget(1) is panel)
    check("right dock: panel x > image x", right_panel_x > right_image_x)

    apply_adjust_splitter_dock(splitter, image, panel, "left")
    _app.processEvents()
    check("left order is panel then image", splitter.widget(0) is panel and splitter.widget(1) is image)
    check("left dock: panel x < image x", panel.x() < image.x())
    check("left dock moved panel x left of the right-dock x", panel.x() < right_panel_x)


def test_header_button_emits_opposite_side():
    p = ImageAdjustPanelWidget()
    p.set_dock_side("right")
    received = []
    p.dock_side_requested.connect(received.append)
    btn = p._header_dock_btn
    check("dock button exists on header", btn is not None)
    btn.click()
    check("from right, button requests left", received == ["left"])
    p.set_dock_side("left")
    received.clear()
    btn.click()
    check("from left, button requests right", received == ["right"])
    check("tooltip names the destination", "right" in (btn.toolTip() or "").lower())


def main() -> int:
    test_normalize_dock()
    test_splitter_left_flips_x()
    test_header_button_emits_opposite_side()
    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
