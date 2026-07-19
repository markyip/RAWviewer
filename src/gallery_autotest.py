"""Gallery scroll benchmark harness (RAWVIEWER_GALLERY_AUTOTEST=1 only).

Flow (mirrors the manual repro: open file -> gallery button -> hold Up/Down):
  1. Wait for the single-view first render.
  2. Wait until gallery entry is allowed (v2.5 blocks on EXIF sort; v3 enters
     on provisional order). Toggle and verify view_mode == gallery.
  3. Simulate holding Up or Down at RAWVIEWER_GALLERY_AUTOTEST_KEY_HZ
     (default 30) for RAWVIEWER_GALLERY_AUTOTEST_SECONDS (default 20).
  4. Poll visible tiles at 10Hz and count distinct paths that received real
     pixels; report [GALLERYTEST] summary lines and quit the app.

Env:
  RAWVIEWER_GALLERY_AUTOTEST_DIRECTION  up|down (default: up)
  RAWVIEWER_GALLERY_AUTOTEST_GALLERY_WAIT  max seconds waiting for gallery
     entry after first render (default 600). Needed for large cold folders
     on builds that gate the Gallery button on EXIF sort (v2.5).

Metrics:
  rendered_tiles = distinct images that were on screen WITH pixels at some
  poll instant.
  first_render_s / gallery_ready_s = seconds from harness arm to single-view
  first paint / confirmed gallery mode (not merely a toggle attempt).
"""

import os
import time


def run_gallery_autotest(viewer, *, safe_print) -> None:
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication

    duration_s = float(os.environ.get("RAWVIEWER_GALLERY_AUTOTEST_SECONDS", "20"))
    repeat_hz = float(os.environ.get("RAWVIEWER_GALLERY_AUTOTEST_KEY_HZ", "30"))
    settle_s = float(os.environ.get("RAWVIEWER_GALLERY_AUTOTEST_SETTLE", "3.0"))
    gallery_wait_s = float(
        os.environ.get("RAWVIEWER_GALLERY_AUTOTEST_GALLERY_WAIT", "600")
    )
    direction = (
        os.environ.get("RAWVIEWER_GALLERY_AUTOTEST_DIRECTION", "up").strip().lower()
        or "up"
    )
    if direction not in ("up", "down"):
        direction = "up"

    harness_t0 = time.time()
    state = {
        "started": False,
        "t0": 0.0,
        "rendered": set(),
        "presses": 0,
        "polls": 0,
        "samples": [],  # (t_offset, cumulative_rendered)
        "first_render_s": None,
        "gallery_ready_s": None,
        "enter_attempts": 0,
    }

    key_timer = QTimer(viewer)
    key_timer.setInterval(max(1, int(1000.0 / repeat_hz)))
    poll_timer = QTimer(viewer)
    poll_timer.setInterval(100)

    def _quit_app(reason: str) -> None:
        safe_print(f"[GALLERYTEST] quit: {reason}", flush=True, force=True)
        try:
            viewer.close()
        except Exception:
            pass
        app = QApplication.instance()
        if app is not None:
            QTimer.singleShot(500, app.quit)

    def _gallery():
        return getattr(viewer, "gallery_justified", None)

    def _in_gallery() -> bool:
        return getattr(viewer, "view_mode", "single") == "gallery"

    def _exif_sort_ready() -> bool:
        fn = getattr(viewer, "_is_exif_sort_ready", None)
        if not callable(fn):
            return True
        try:
            return bool(fn())
        except Exception:
            return False

    def _press_key():
        # Guard: Up in single view toggles bookmark; never press until gallery.
        if not _in_gallery():
            safe_print(
                "[GALLERYTEST] abort key-hold: still in "
                f"{getattr(viewer, 'view_mode', '?')} (would bookmark, not scroll)",
                flush=True,
                force=True,
            )
            key_timer.stop()
            _quit_app("not_in_gallery_during_keyhold")
            return
        if direction == "up":
            viewer._shortcut_activate_gallery_up()
        else:
            viewer._shortcut_activate_gallery_down()

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
            f"duration_s={duration_s} key_hz={repeat_hz} direction={direction} "
            f"first_render_s={state['first_render_s']} "
            f"gallery_ready_s={state['gallery_ready_s']} "
            f"view_mode={getattr(viewer, 'view_mode', None)}",
            flush=True,
            force=True,
        )
        safe_print(
            f"[GALLERYTEST] timeline={state['samples']}", flush=True, force=True
        )
        _quit_app("done")

    def _press():
        if time.time() - state["t0"] >= duration_s:
            key_timer.stop()
            # Let in-flight tiles land before the final count.
            QTimer.singleShot(2000, _finish)
            return
        try:
            _press_key()
            state["presses"] += 1
        except Exception as e:  # pragma: no cover
            safe_print(f"[GALLERYTEST] key press failed: {e}", flush=True, force=True)

    def _start_keyhold():
        if state["started"]:
            return
        if not _in_gallery():
            safe_print(
                "[GALLERYTEST] refuse key-hold: view_mode="
                f"{getattr(viewer, 'view_mode', None)}",
                flush=True,
                force=True,
            )
            _quit_app("gallery_not_active")
            return
        state["started"] = True
        state["gallery_ready_s"] = round(time.time() - harness_t0, 3)
        state["t0"] = time.time()
        key_timer.timeout.connect(_press)
        poll_timer.timeout.connect(_poll)
        key_timer.start()
        poll_timer.start()
        safe_print(
            f"[GALLERYTEST] STARTED duration_s={duration_s} key_hz={repeat_hz} "
            f"direction={direction} first_render_s={state['first_render_s']} "
            f"gallery_ready_s={state['gallery_ready_s']} "
            f"files={len(getattr(viewer, 'image_files', []) or [])}",
            flush=True,
            force=True,
        )

    def _try_enter_gallery():
        """Wait for folder scan + EXIF-sort gate (v2.5), then toggle into gallery."""
        if _in_gallery():
            _start_keyhold()
            return

        elapsed = time.time() - harness_t0
        first_s = state["first_render_s"] or 0.0
        waited = elapsed - first_s
        if waited > gallery_wait_s:
            safe_print(
                f"[GALLERYTEST] TIMEOUT waiting for gallery entry "
                f"({gallery_wait_s}s after first render); "
                f"exif_sort_ready={_exif_sort_ready()} "
                f"files={len(getattr(viewer, 'image_files', []) or [])}",
                flush=True,
                force=True,
            )
            _quit_app("gallery_entry_timeout")
            return

        n_files = len(getattr(viewer, "image_files", []) or [])
        # Fast-open starts with 1 file; _is_exif_sort_ready() is True while
        # len<=1, which would enter a 1-tile gallery. Wait for the folder scan.
        if n_files <= 1:
            if state["enter_attempts"] % 25 == 0:
                safe_print(
                    f"[GALLERYTEST] waiting for folder scan (files={n_files} "
                    f"elapsed_s={waited:.1f})",
                    flush=True,
                    force=True,
                )
            state["enter_attempts"] += 1
            QTimer.singleShot(200, _try_enter_gallery)
            return

        sort_ready = _exif_sort_ready()
        if not sort_ready:
            # v2.5: toggle is a no-op until capture-time sort finishes.
            # v3: still enters on provisional order ??try toggle below anyway.
            if state["enter_attempts"] % 25 == 0:
                safe_print(
                    f"[GALLERYTEST] waiting for EXIF sort (files={n_files} "
                    f"elapsed_s={waited:.1f})",
                    flush=True,
                    force=True,
                )

        state["enter_attempts"] += 1
        try:
            viewer.toggle_view_mode()
        except Exception as e:
            safe_print(f"[GALLERYTEST] gallery toggle failed: {e}", flush=True, force=True)
            _quit_app("toggle_exception")
            return

        if _in_gallery():
            QTimer.singleShot(500, _start_keyhold)
            return

        # Still blocked (typical v2.5 while sort runs). Keep polling.
        if sort_ready and state["enter_attempts"] % 10 == 0:
            safe_print(
                "[GALLERYTEST] toggle did not enter gallery despite sort ready; "
                f"retrying (attempt={state['enter_attempts']})",
                flush=True,
                force=True,
            )
        QTimer.singleShot(200, _try_enter_gallery)

    def _wait_first_render():
        if getattr(viewer, "_single_view_first_render_logged", False):
            if state["first_render_s"] is None:
                state["first_render_s"] = round(time.time() - harness_t0, 3)
                safe_print(
                    f"[GALLERYTEST] FIRST_RENDER s={state['first_render_s']}",
                    flush=True,
                    force=True,
                )
            QTimer.singleShot(200, _try_enter_gallery)
        elif time.time() - harness_t0 > float(os.environ.get("RAWVIEWER_GALLERY_AUTOTEST_FIRST_RENDER_WAIT", "180")):
            safe_print(
                "[GALLERYTEST] TIMEOUT waiting for first render",
                flush=True,
                force=True,
            )
            _quit_app("first_render_timeout")
        else:
            QTimer.singleShot(200, _wait_first_render)

    safe_print(
        f"[GALLERYTEST] armed settle_s={settle_s} direction={direction} "
        f"gallery_wait_s={gallery_wait_s}",
        flush=True,
        force=True,
    )
    QTimer.singleShot(int(settle_s * 1000), _wait_first_render)
