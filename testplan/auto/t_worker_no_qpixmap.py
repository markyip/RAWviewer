"""Workers must not construct QPixmap: it is GUI-thread-only in Qt.

ImageLoadWorker.run() reaches load_pixmap_safe through
process_full_image -> _process_regular_image for every non-RAW file, so a
QPixmap was being built on a QThreadPool thread for every JPEG, PNG, WebP
and TIFF, cached from there, and handed across via pixmap_ready. QPixmap is
backed by platform paint resources; doing this can corrupt the paint engine,
macOS especially.

Off the GUI thread the processor now returns a QImage -- plain pixel data,
legal to build anywhere -- and the GUI slot converts it.
"""

import os
import sys
import tempfile
import threading

import numpy as np
from PyQt6.QtCore import QThread
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


_app = QApplication.instance() or QApplication([])


def main() -> int:
    import cv2

    import unified_image_processor as uip
    from common_image_loader import load_qimage_safe

    with tempfile.TemporaryDirectory() as d:
        jpg = os.path.join(d, "shot.jpg")
        cv2.imwrite(jpg, np.full((240, 320, 3), (30, 150, 210), np.uint8))
        png = os.path.join(d, "shot.png")
        cv2.imwrite(png, np.full((180, 260, 3), (200, 40, 40), np.uint8))

        # --- the QImage loader itself ---
        for path, wh in ((jpg, (320, 240)), (png, (260, 180))):
            img = load_qimage_safe(path)
            check(
                f"load_qimage_safe reads {os.path.basename(path)}",
                not img.isNull() and (img.width(), img.height()) == wh,
                f"{img.width()}x{img.height()}",
            )
        small = load_qimage_safe(jpg, max_edge=100)
        check("max_edge is honoured", max(small.width(), small.height()) == 100,
              f"{small.width()}x{small.height()}")
        check("missing file yields a null QImage", load_qimage_safe(os.path.join(d, "no.jpg")).isNull())

        # --- the thread predicate ---
        check("main thread reports as the GUI thread", uip._on_gui_thread() is True)
        seen = {}

        def probe():
            seen["gui"] = uip._on_gui_thread()

        t = threading.Thread(target=probe)
        t.start()
        t.join()
        check("a worker thread reports as NOT the GUI thread", seen.get("gui") is False)

        # --- the actual contract: what comes back off-thread ---
        proc = uip.UnifiedImageProcessor()
        on_gui = proc._process_regular_image(jpg, use_full_resolution=True)
        check(
            "GUI thread still gets a QPixmap",
            isinstance(on_gui, QPixmap) and not on_gui.isNull(),
            type(on_gui).__name__,
        )

        result = {}

        def worker():
            try:
                result["buf"] = proc._process_regular_image(jpg, use_full_resolution=True)
            except Exception as exc:  # noqa: BLE001
                result["err"] = exc

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        check("worker thread did not raise", "err" not in result, str(result.get("err")))
        buf = result.get("buf")
        check(
            "worker gets a QImage, NOT a QPixmap",
            isinstance(buf, QImage) and not isinstance(buf, QPixmap),
            type(buf).__name__,
        )
        check("worker QImage carries the pixels", buf is not None and not buf.isNull()
              and (buf.width(), buf.height()) == (320, 240),
              f"{buf.width()}x{buf.height()}" if buf is not None else "None")

    # --- the delivery path exists on both ends ---
    import image_load_manager as ilm
    import main as mainmod

    check("manager exposes qimage_ready", hasattr(ilm.ImageLoadManager, "qimage_ready"))
    check("host has the converting slot", hasattr(mainmod.RAWImageViewer, "on_manager_qimage_ready"))
    src = open(os.path.join(os.path.dirname(__file__), "..", "..", "src", "image_load_manager.py")).read()
    check("the worker emits QImage buffers rather than dropping them", "qimage_ready.emit" in src)

    # --- the conversion must not be paid for a photo already navigated away ---
    import inspect

    slot_src = inspect.getsource(mainmod.RAWImageViewer.on_manager_qimage_ready)
    guard_at = min(
        slot_src.find("view_mode"),
        slot_src.find("_norm_path"),
    )
    convert_at = slot_src.find("fromImage")
    check(
        "staleness is checked BEFORE the QPixmap conversion",
        guard_at != -1 and convert_at != -1 and guard_at < convert_at,
        "converting first costs ~18ms of GUI thread for a 33MP frame that is then dropped",
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
