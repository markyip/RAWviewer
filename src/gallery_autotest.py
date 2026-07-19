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
  RAWVIEWER_GALLERY_AUTOTEST_INPUT  keys|trackpad (default: keys). trackpad
     synthesizes native-style QWheelEvents on the gallery viewport at 120Hz:
     repeated gestures of ScrollBegin -> drag ScrollUpdates -> ScrollMomentum
     decay -> ScrollEnd, plus a main-thread stall monitor. Extra summary line:
     [GALLERYTEST] STALLS count/max/p95/total (gaps >25ms in an 8ms heartbeat)
     and per-sample visible-blank-tile counts in the timeline.
  RAWVIEWER_GALLERY_AUTOTEST_DRAG_PX_S  drag speed for trackpad mode
     (default 2500 px/s).

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
    input_mode = (
        os.environ.get("RAWVIEWER_GALLERY_AUTOTEST_INPUT", "keys").strip().lower()
        or "keys"
    )
    drag_px_s = float(os.environ.get("RAWVIEWER_GALLERY_AUTOTEST_DRAG_PX_S", "2500"))

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
        "stall_gaps": [],  # (t_offset, gap_ms) for heartbeat gaps > 25ms
        "hb_last": 0.0,
        "wheel_phase": "idle",  # idle | drag | momentum
        "wheel_phase_t0": 0.0,
        "wheel_velocity": 0.0,  # px/s, momentum decay state
        "wheel_last_t": 0.0,
        "gestures": 0,
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
        blank_now = 0
        for w in list(widgets.values()):
            try:
                pm = w.pixmap()
                if pm is not None and not pm.isNull() and w.file_path:
                    state["rendered"].add(w.file_path)
                elif w.file_path:
                    blank_now += 1
            except Exception:
                pass
        if state["polls"] % 10 == 0:
            state["samples"].append(
                (
                    round(time.time() - state["t0"], 1),
                    len(state["rendered"]),
                    blank_now,
                )
            )

    def _send_wheel(dy: float, phase) -> None:
        from PyQt6.QtCore import QPoint, QPointF, Qt
        from PyQt6.QtGui import QWheelEvent

        g = _gallery()
        sa = getattr(g, "_scroll_area", None) if g is not None else None
        vp = sa.viewport() if sa is not None else None
        if vp is None:
            return
        pos = QPointF(vp.width() / 2.0, vp.height() / 2.0)
        gpos = QPointF(vp.mapToGlobal(QPoint(int(pos.x()), int(pos.y()))))
        d = int(round(dy))
        ev = QWheelEvent(
            pos,
            gpos,
            QPoint(0, d),
            QPoint(0, d),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            phase,
            False,
            Qt.MouseEventSource.MouseEventSynthesizedByApplication,
        )
        QApplication.sendEvent(vp, ev)

    def _wheel_tick():
        """120Hz synthetic trackpad: drag 0.6s -> momentum decay -> idle 0.5s."""
        from PyQt6.QtCore import Qt

        if not _in_gallery():
            key_timer.stop()
            _quit_app("not_in_gallery_during_wheel")
            return
        now = time.time()
        if now - state["t0"] >= duration_s and state["wheel_phase"] == "idle":
            key_timer.stop()
            QTimer.singleShot(2000, _finish)
            return
        dt = min(0.05, max(0.001, now - (state["wheel_last_t"] or now)))
        state["wheel_last_t"] = now
        phase = state["wheel_phase"]
        phase_elapsed = now - state["wheel_phase_t0"]
        sign = 1.0 if direction == "up" else -1.0
        if phase == "idle":
            if phase_elapsed >= 0.5:
                state["wheel_phase"] = "drag"
                state["wheel_phase_t0"] = now
                state["gestures"] += 1
                _send_wheel(0, Qt.ScrollPhase.ScrollBegin)
        elif phase == "drag":
            _send_wheel(sign * drag_px_s * dt, Qt.ScrollPhase.ScrollUpdate)
            state["presses"] += 1
            if phase_elapsed >= 0.6:
                state["wheel_phase"] = "momentum"
                state["wheel_phase_t0"] = now
                state["wheel_velocity"] = drag_px_s
        elif phase == "momentum":
            state["wheel_velocity"] *= 0.955  # ~2.3x decay per 0.15s at 120Hz
            dy = sign * state["wheel_velocity"] * dt
            if abs(dy) < 1.0:
                _send_wheel(0, Qt.ScrollPhase.ScrollEnd)
                state["wheel_phase"] = "idle"
                state["wheel_phase_t0"] = now
            else:
                _send_wheel(dy, Qt.ScrollPhase.ScrollMomentum)
                state["presses"] += 1

    def _heartbeat():
        now = time.time()
        last = state["hb_last"]
        state["hb_last"] = now
        if last <= 0.0 or state["t0"] <= 0.0:
            return
        gap_ms = (now - last) * 1000.0
        if gap_ms > 25.0:
            state["stall_gaps"].append(
                (round(now - state["t0"], 2), round(gap_ms, 1))
            )

    def _finish():
        _poll()
        poll_timer.stop()
        gaps = sorted(g for _, g in state["stall_gaps"])
        if input_mode == "trackpad":
            total = sum(gaps)
            p95 = gaps[int(len(gaps) * 0.95)] if gaps else 0.0
            worst = state["stall_gaps"] and max(state["stall_gaps"], key=lambda x: x[1])
            safe_print(
                f"[GALLERYTEST] STALLS count={len(gaps)} max_ms={gaps[-1] if gaps else 0} "
                f"p95_ms={p95} total_ms={round(total)} worst_at={worst} "
                f"gestures={state['gestures']}",
                flush=True,
                force=True,
            )
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
        if input_mode == "trackpad":
            from PyQt6.QtCore import Qt as _Qt

            # Periodic all-thread stack dumps: when the main thread is stalled,
            # the dump names the blocking frame (py-spy needs root on macOS).
            dump_path = os.environ.get("RAWVIEWER_GALLERY_AUTOTEST_STACK_DUMP", "")
            if dump_path:
                import faulthandler

                state["stack_dump_fh"] = open(dump_path, "w")
                faulthandler.dump_traceback_later(
                    0.4, repeat=True, file=state["stack_dump_fh"]
                )
            state["wheel_phase_t0"] = state["t0"]
            wheel_hz = float(
                os.environ.get("RAWVIEWER_GALLERY_AUTOTEST_WHEEL_HZ", "120")
            )
            key_timer.setInterval(max(1, int(round(1000.0 / max(1.0, wheel_hz)))))
            key_timer.setTimerType(_Qt.TimerType.PreciseTimer)
            key_timer.timeout.connect(_wheel_tick)
            hb_timer = QTimer(viewer)
            hb_timer.setInterval(8)
            hb_timer.setTimerType(_Qt.TimerType.PreciseTimer)
            hb_timer.timeout.connect(_heartbeat)
            hb_timer.start()
            state["hb_timer"] = hb_timer  # keep alive
        else:
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
