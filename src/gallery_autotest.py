"""Gallery scroll benchmark harness (RAWVIEWER_GALLERY_AUTOTEST=1 only).

Flow (mirrors the manual repro: open file -> gallery button -> hold Down):
  1. Wait for the single-view first render (the point the gallery toggle is
     usable), plus a short settle.
  2. Toggle to gallery view.
  3. Simulate holding the Down key at RAWVIEWER_GALLERY_AUTOTEST_KEY_HZ
     (default 30, matching macOS key-repeat) for
     RAWVIEWER_GALLERY_AUTOTEST_SECONDS (default 20).
  4. Poll visible tiles at 10Hz and count distinct paths that received real
     pixels; report [GALLERYTEST] summary lines and quit the app.

Metric: rendered_tiles = distinct images that were on screen WITH pixels at
some poll instant. Recycled-away tiles that never got pixels don't count, so
maximizing this is exactly "process more images in the 20s window".
"""

import os
import time


def run_gallery_autotest(viewer, *, safe_print) -> None:
    from PyQt6.QtCore import QTimer

    duration_s = float(os.environ.get("RAWVIEWER_GALLERY_AUTOTEST_SECONDS", "20"))
    repeat_hz = float(os.environ.get("RAWVIEWER_GALLERY_AUTOTEST_KEY_HZ", "30"))
    settle_s = float(os.environ.get("RAWVIEWER_GALLERY_AUTOTEST_SETTLE", "3.0"))

    state = {
        "started": False,
        "t0": 0.0,
        "rendered": set(),
        "presses": 0,
        "polls": 0,
        "samples": [],  # (t_offset, cumulative_rendered)
    }

    key_timer = QTimer(viewer)
    key_timer.setInterval(max(1, int(1000.0 / repeat_hz)))
    poll_timer = QTimer(viewer)
    poll_timer.setInterval(100)

    def _gallery():
        return getattr(viewer, "gallery_justified", None)

    def _poll():
        g = _gallery()
        if g is None:
            return
        state["polls"] += 1
        widgets = getattr(g, "_visible_widgets", {})
        for w in list(widgets.values()):
            try:
                pm = w.pixmap()
                if pm is not None and not pm.isNull() and w.file_path:
                    state["rendered"].add(w.file_path)
            except Exception:
                pass
        if state["polls"] % 10 == 0:
            state["samples"].append(
                (round(time.time() - state["t0"], 1), len(state["rendered"]))
            )

    def _finish():
        _poll()
        poll_timer.stop()
        n = len(state["rendered"])
        safe_print(
            f"[GALLERYTEST] DONE presses={state['presses']} rendered_tiles={n} "
            f"duration_s={duration_s} key_hz={repeat_hz}",
            flush=True,
            force=True,
        )
        safe_print(
            f"[GALLERYTEST] timeline={state['samples']}", flush=True, force=True
        )
        from PyQt6.QtWidgets import QApplication

        viewer.close()
        app = QApplication.instance()
        if app is not None:
            QTimer.singleShot(500, app.quit)

    def _press():
        if time.time() - state["t0"] >= duration_s:
            key_timer.stop()
            # Let in-flight tiles land before the final count.
            QTimer.singleShot(2000, _finish)
            return
        try:
            viewer._shortcut_activate_gallery_down()
            state["presses"] += 1
        except Exception as e:  # pragma: no cover
            safe_print(f"[GALLERYTEST] key press failed: {e}", flush=True, force=True)

    def _begin():
        if state["started"]:
            return
        state["started"] = True
        try:
            if getattr(viewer, "view_mode", "single") != "gallery":
                viewer.toggle_view_mode()
        except Exception as e:
            safe_print(f"[GALLERYTEST] gallery toggle failed: {e}", flush=True, force=True)
            return
        state["t0"] = time.time()
        key_timer.timeout.connect(_press)
        poll_timer.timeout.connect(_poll)
        key_timer.start()
        poll_timer.start()
        safe_print(
            f"[GALLERYTEST] STARTED duration_s={duration_s} key_hz={repeat_hz}",
            flush=True,
            force=True,
        )

    def _wait_ready():
        if getattr(viewer, "_single_view_first_render_logged", False):
            QTimer.singleShot(500, _begin)
        else:
            QTimer.singleShot(200, _wait_ready)

    QTimer.singleShot(int(settle_s * 1000), _wait_ready)
