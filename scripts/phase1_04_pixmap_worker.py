"""Phase 1 item 4: background QPixmap conversion for full-resolution display."""
from __future__ import annotations

import os
import sys
import time

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PyQt6.QtCore import QCoreApplication
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication

app = QApplication([]) if QCoreApplication.instance() is None else QApplication.instance()

from rawviewer_app.processing import PixmapConverter


def test_main_py_wiring() -> None:
    main_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
    with open(main_path, encoding="utf-8") as f:
        src = f.read()
    assert "def _cancel_display_pixmap_converter" in src
    assert "def _start_display_numpy_pixmap_worker" in src
    assert "def _on_display_numpy_pixmap_ready" in src
    assert '"orientation_already_applied"' in src.split("def _start_display_numpy_pixmap_worker")[1].split("def _on_display_numpy_pixmap_ready")[0]
    assert "ctx.get(\"orientation_already_applied\")" in src
    block = src.split("def display_numpy_image")[1].split("def display_pixmap")[0]
    assert "_start_display_numpy_pixmap_worker" in block
    assert "preview / sub-full-res path stays on UI thread" in block
    cleanup = src.split("def _cleanup_current_processing")[1].split("def load_raw_image")[0]
    assert "_cancel_display_pixmap_converter()" in cleanup


def test_pixmap_converter_emits() -> None:
    rgb = np.zeros((3200, 2400, 3), dtype=np.uint8)
    rgb[:, :, 1] = 180
    path = "/tmp/test_full_res.cr3"
    received = []

    converter = PixmapConverter(path, rgb, None)
    converter.pixmap_ready.connect(
        lambda fp, pm: received.append((fp, pm.width(), pm.height()))
    )
    converter.start()
    deadline = time.time() + 5.0
    while not received and time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
    assert received, "PixmapConverter did not emit"
    assert received[0][0] == path
    assert received[0][1] == 2400
    assert received[0][2] == 3200
    assert isinstance(converter, PixmapConverter)


if __name__ == "__main__":
    test_main_py_wiring()
    test_pixmap_converter_emits()
    print("OK: Phase 1 item 4 (background QPixmap conversion) tests passed")
