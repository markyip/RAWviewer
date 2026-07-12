"""Navigation-performance benchmark harness (RAWVIEWER_AUTOTEST=1 only).

Drives single-view "next image" navigation on a timer and logs per-image
true on-screen full-resolution paint latency, then quits. Diagnostic tool
for navigation-perf work -- imported by main.py only when the env var is
set, so it costs nothing in normal operation.

Hooks display_pixmap() itself (the actual paint call), not the image_ready
decode-complete signal -- the numpy->QPixmap conversion for full-res buffers
runs on a background QThread, so "decoded" and "painted on screen" are
measurably different instants. Every display_pixmap() call during a
navigation is recorded (not just the final full-res one) so
paints-per-navigation quantifies low-res->high-res flashing.

Env vars:
  RAWVIEWER_AUTOTEST_COUNT   default: full folder length (one complete
                             loop back to the starting image).
  RAWVIEWER_AUTOTEST_REVISIT default 15: extra navigations run *after*
                             the full loop closes, re-visiting the first
                             N files a second time so cold (first-visit)
                             vs warm (post-loop revisit) timing can be
                             compared directly.
  RAWVIEWER_AUTOTEST_SETTLE  default 3.0s before the first nav.
  RAWVIEWER_AUTOTEST_TIMEOUT default 45.0s per-image hard ceiling (a 100MP
                             RAF sensor decode legitimately takes 20-35s on
                             an M-series laptop, more under GPU contention;
                             censoring those as <timeout> hides the true
                             cost -- validated: at 30s DSCF0311.RAF still
                             raced the ceiling, at 34.3s DSCF0363.RAF
                             measured cleanly).
  RAWVIEWER_AUTOTEST_SETTLE_IDLE default 2.5s: after a below-sensor-res
                             paint, if no further paint arrives for this
                             long AND the load manager has no active task
                             for the file, the pipeline is done -- the file's
                             final form simply doesn't cover sensor
                             resolution (embedded-preview-only formats).
                             Accept the last paint as final (timed at the
                             paint, not at detection) instead of burning the
                             hard ceiling.
  RAWVIEWER_AUTOTEST_HEARTBEAT set 1 for a 100ms main-thread liveness probe
                             (gaps >150ms logged) + SIGUSR1 faulthandler
                             stack dumps.
  RAWVIEWER_AUTOTEST_MEMPROF set 1 to enable periodic RSS + gc object-count
                             + tracemalloc snapshots -- off by default since
                             gc.get_objects() over the whole heap isn't free.
  RAWVIEWER_AUTOTEST_MEMPROF_EVERY sample every N navigations (default 15).
  RAWVIEWER_AUTOTEST_RAW_MODE set 1 to benchmark the RAW (High Quality)
                             workflow -- display driven by the LibRaw/GPU
                             sensor decode -- instead of the default
                             embedded-JPEG workflow. Flips the persistent
                             use_embedded_jpeg_workflow setting off for the
                             run and restores the user's original values
                             (incl. raw_edr_display_enabled, defaulted off
                             on RAW entry same as the UI toggle does) at
                             the end.
"""

from __future__ import annotations

import os
import time

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication


def run_nav_autotest(viewer, *, safe_print, image_covers_sensor_resolution):
    """Install and start the harness on ``viewer`` (a RAWImageViewer).

    ``safe_print`` and ``image_covers_sensor_resolution`` are injected by
    main.py (they live there / in common_image_loader) to avoid a circular
    import of the 29k-line main module from here.
    """
    count_env = os.environ.get("RAWVIEWER_AUTOTEST_COUNT")
    revisit = int(os.environ.get("RAWVIEWER_AUTOTEST_REVISIT", "15"))
    settle_s = float(os.environ.get("RAWVIEWER_AUTOTEST_SETTLE", "3.0"))
    timeout_s = float(os.environ.get("RAWVIEWER_AUTOTEST_TIMEOUT", "45.0"))
    settle_idle_s = float(os.environ.get("RAWVIEWER_AUTOTEST_SETTLE_IDLE", "2.5"))
    memprof = str(os.environ.get("RAWVIEWER_AUTOTEST_MEMPROF", "0")).strip().lower() in (
        "1",
        "true",
        "yes",
    )
    memprof_every = int(os.environ.get("RAWVIEWER_AUTOTEST_MEMPROF_EVERY", "15"))
    raw_mode = str(os.environ.get("RAWVIEWER_AUTOTEST_RAW_MODE", "0")).strip().lower() in (
        "1",
        "true",
        "yes",
    )

    restore_settings = {}
    if raw_mode:
        try:
            settings = viewer.get_settings()
            restore_settings["use_embedded_jpeg_workflow"] = settings.value(
                "use_embedded_jpeg_workflow", True, type=bool
            )
            restore_settings["raw_edr_display_enabled"] = settings.value(
                "raw_edr_display_enabled", False, type=bool
            )
        except Exception as e:
            safe_print(f"[AUTOTEST] RAW mode setup failed: {e}", flush=True, force=True)
            raw_mode = False

    def _enter_raw_mode() -> bool:
        """Switch to the RAW (High Quality) workflow via the app's OWN toggle.

        toggle_raw_jpeg_workflow() is exactly what the UI button does: flips
        the setting, updates the button state, defaults EDR off, invalidates
        the workflow pixel caches, and reloads the current image. Poking the
        QSettings key directly (the first version of this) drove the same
        decode pipeline but left the UI claiming preview mode and skipped the
        cache invalidation -- not the real user path this benchmark should
        measure. Returns True if a toggle happened (caller allows settle time).
        """
        try:
            settings = viewer.get_settings()
            if not settings.value("use_embedded_jpeg_workflow", True, type=bool):
                safe_print(
                    "[AUTOTEST] workflow: RAW (High Quality) -- already active",
                    flush=True,
                    force=True,
                )
                return False
            viewer.toggle_raw_jpeg_workflow()
            now_embedded = settings.value("use_embedded_jpeg_workflow", True, type=bool)
            safe_print(
                "[AUTOTEST] workflow: RAW (High Quality) -- toggled via "
                f"toggle_raw_jpeg_workflow(), use_embedded_jpeg_workflow={now_embedded}",
                flush=True,
                force=True,
            )
            return True
        except Exception as e:
            safe_print(f"[AUTOTEST] RAW mode toggle failed: {e}", flush=True, force=True)
            return False

    def _restore_workflow_settings():
        if not restore_settings:
            return
        try:
            settings = viewer.get_settings()
            for k, v in restore_settings.items():
                settings.setValue(k, v)
        except Exception:
            pass
    # Background folder load (full scan + EXIF sort) can take several
    # seconds after the window shows a fast single-file open -- image_files
    # has length 1 until it lands. Wait for it to stabilize (unchanged
    # across two 300ms polls) instead of a fixed delay, or this undercounts
    # "the whole folder" down to whatever was open at settle time.
    folder_wait_deadline = time.time() + float(
        os.environ.get("RAWVIEWER_AUTOTEST_FOLDER_WAIT", "30.0")
    )

    state = {
        "nav_start": 0.0,
        "results": [],  # (name, w, h, elapsed, n_paints)
        "paints_this_nav": [],  # [(w, h, elapsed)] every display_pixmap() call this nav
        "index": 0,
        "waiting": False,
        "count": 0,
        "total_steps": 0,
        "settled_below_sensor": 0,
    }

    heartbeat = str(os.environ.get("RAWVIEWER_AUTOTEST_HEARTBEAT", "0")).strip().lower() in (
        "1",
        "true",
        "yes",
    )
    heartbeat_timer = None
    if heartbeat:
        # Main-thread event-loop liveness probe: fires every 100ms via a
        # QTimer (so it can only tick when the main thread's event loop is
        # actually free to process events). A gap >150ms between consecutive
        # ticks means the main thread itself was blocked/busy for that long
        # -- distinguishes "main thread stalled" from "cross-thread signal
        # delivery delayed while the main thread was idle", two very
        # different root causes for the warm-tail slowdown investigation.
        hb_state = {"last": time.time(), "gaps": []}

        def _heartbeat_tick():
            now = time.time()
            gap = now - hb_state["last"]
            hb_state["last"] = now
            if gap > 0.15:
                hb_state["gaps"].append((now, gap))
                safe_print(
                    f"[HEARTBEAT] main thread event loop GAP: {gap:.3f}s at t={now:.3f}",
                    flush=True,
                    force=True,
                )

        heartbeat_timer = QTimer()
        heartbeat_timer.timeout.connect(_heartbeat_tick)
        heartbeat_timer.start(100)

        # py-spy needs root on macOS and isn't available here. faulthandler
        # gives the same "what is every thread doing right now" answer
        # without privilege: SIGUSR1 dumps all thread stacks to stderr.
        # `kill -USR1 <pid>` from an unprivileged shell during an observed
        # heartbeat gap pinpoints exactly what's blocking the main thread.
        try:
            import faulthandler
            import signal

            faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
            safe_print(
                f"[HEARTBEAT] faulthandler armed on SIGUSR1 for pid={os.getpid()} "
                f"-- 'kill -USR1 {os.getpid()}' dumps all thread stacks to stderr",
                flush=True,
                force=True,
            )
        except Exception as e:
            safe_print(f"[HEARTBEAT] faulthandler registration failed: {e}", flush=True, force=True)

    mem_state = {"proc": None, "baseline_counts": None, "prev_tm_snap": None}
    if memprof:
        try:
            import gc as _gc
            import tracemalloc as _tracemalloc

            import psutil as _psutil

            mem_state["proc"] = _psutil.Process()
            _tracemalloc.start(25)
            mem_state["gc"] = _gc
            mem_state["tracemalloc"] = _tracemalloc
        except Exception as e:
            safe_print(f"[MEMPROF] failed to initialize: {e}", flush=True, force=True)
            memprof = False

    def _mem_snapshot(label: str) -> None:
        """RSS + gc live-object-count histogram + tracemalloc top allocators,
        diffed against the first snapshot (counts) / previous snapshot
        (tracemalloc) so growth -- not just absolute size -- is visible.
        """
        if not memprof:
            return
        try:
            _gc = mem_state["gc"]
            _tracemalloc = mem_state["tracemalloc"]
            proc = mem_state["proc"]
            _gc.collect()
            rss_mb = proc.memory_info().rss / 1e6

            from collections import Counter

            counts = Counter(type(o).__name__ for o in _gc.get_objects())
            baseline = mem_state["baseline_counts"]
            if baseline is None:
                mem_state["baseline_counts"] = counts
                safe_print(
                    f"[MEMPROF] {label}: BASELINE rss={rss_mb:.1f}MB live_objects={sum(counts.values())}",
                    flush=True,
                    force=True,
                )
            else:
                deltas = sorted(
                    (
                        (name, counts.get(name, 0) - baseline.get(name, 0))
                        for name in set(counts) | set(baseline)
                    ),
                    key=lambda kv: -kv[1],
                )
                top = [f"{n}:+{d}" for n, d in deltas[:12] if d > 0]
                safe_print(
                    f"[MEMPROF] {label}: rss={rss_mb:.1f}MB live_objects={sum(counts.values())} "
                    f"top_growth_since_baseline={top}",
                    flush=True,
                    force=True,
                )

            snap = _tracemalloc.take_snapshot()
            prev = mem_state["prev_tm_snap"]
            if prev is not None:
                diffs = snap.compare_to(prev, "lineno")
                lines = [
                    f"{str(d.traceback[-1]).split(os.sep)[-1]} +{d.size_diff/1e6:.2f}MB (+{d.count_diff})"
                    for d in diffs[:8]
                    if d.size_diff > 0
                ]
                if lines:
                    safe_print(
                        f"[MEMPROF] {label}: tracemalloc top growth since last snapshot: {lines}",
                        flush=True,
                        force=True,
                    )
            mem_state["prev_tm_snap"] = snap
        except Exception as e:
            safe_print(f"[MEMPROF] snapshot error: {e}", flush=True, force=True)

    timeout_timer = QTimer()
    timeout_timer.setSingleShot(True)

    settle_timer = QTimer()
    settle_timer.setSingleShot(True)

    def _manager_busy_for(path) -> bool:
        """True while the image load manager still has a live task for path."""
        try:
            mgr = getattr(viewer, "image_manager", None)
            if mgr is None:
                return False
            with mgr._queue_lock:
                for key in mgr._task_keys_by_path.get(path, ()):
                    t = mgr._active_tasks.get(key)
                    if t is not None and not t.is_cancelled():
                        return True
        except Exception:
            pass
        return False

    def _on_settle_idle():
        """No new paint for settle_idle_s: the file may be DONE below sensor
        resolution (embedded-preview-only formats -- some 3FR/RAF never
        produce a sensor-covering buffer). If nothing is still in flight,
        accept the last paint as final rather than burning the hard ceiling;
        if a decode is still running, keep waiting.
        """
        if not state["waiting"]:
            return
        cur_path = getattr(viewer, "current_file_path", None)
        if not state["paints_this_nav"] or not cur_path:
            return
        if _manager_busy_for(cur_path):
            settle_timer.start(int(settle_idle_s * 1000))
            return
        w, h, paint_elapsed = state["paints_this_nav"][-1]
        state["waiting"] = False
        timeout_timer.stop()
        n_paints = len(state["paints_this_nav"])
        state["settled_below_sensor"] += 1
        state["results"].append(
            (os.path.basename(cur_path), w, h, paint_elapsed, n_paints)
        )
        safe_print(
            f"[AUTOTEST] SETTLED below sensor-res: {os.path.basename(cur_path)} "
            f"final paint {w}x{h} at {paint_elapsed:.3f}s ({n_paints} paint(s)); "
            f"pipeline idle, no sensor-covering buffer is coming",
            flush=True,
            force=True,
        )
        QTimer.singleShot(50, advance)

    settle_timer.timeout.connect(_on_settle_idle)

    orig_display_pixmap = viewer.display_pixmap

    def _wrapped_display_pixmap(pixmap):
        orig_display_pixmap(pixmap)
        if not state["waiting"] or pixmap is None or pixmap.isNull():
            return
        try:
            w, h = pixmap.width(), pixmap.height()
            elapsed = time.time() - state["nav_start"]
            state["paints_this_nav"].append((w, h, elapsed))
            cur_path = getattr(viewer, "current_file_path", None)
            if not cur_path:
                return
            exif = None
            try:
                exif = viewer.image_cache.get_exif(cur_path)
            except Exception:
                pass
            if not image_covers_sensor_resolution(w, h, exif):
                # Below sensor res: arm/refresh the settle probe so a file
                # whose pipeline ends here doesn't wait out the hard ceiling.
                settle_timer.start(int(settle_idle_s * 1000))
                return
            state["waiting"] = False
            timeout_timer.stop()
            settle_timer.stop()
            n_paints = len(state["paints_this_nav"])
            state["results"].append((os.path.basename(cur_path), w, h, elapsed, n_paints))
            import threading as _threading

            safe_print(
                f"[AUTOTEST] full-res PAINTED: {os.path.basename(cur_path)} "
                f"{w}x{h} in {elapsed:.3f}s ({n_paints} paint(s) this nav: "
                f"{[(pw, ph) for pw, ph, _ in state['paints_this_nav']]}) "
                f"active_threads={_threading.active_count()}",
                flush=True,
                force=True,
            )
            QTimer.singleShot(50, advance)
        except Exception as e:
            safe_print(f"[AUTOTEST] probe error: {e}", flush=True, force=True)

    viewer.display_pixmap = _wrapped_display_pixmap

    def _on_timeout():
        if state["waiting"]:
            elapsed = time.time() - state["nav_start"]
            n_paints = len(state["paints_this_nav"])
            safe_print(
                f"[AUTOTEST] TIMEOUT waiting for full-res paint after {elapsed:.3f}s "
                f"({n_paints} lower-res paint(s) seen: "
                f"{[(pw, ph) for pw, ph, _ in state['paints_this_nav']]})",
                flush=True,
                force=True,
            )
            state["waiting"] = False
            settle_timer.stop()
            state["results"].append(("<timeout>", 0, 0, elapsed, n_paints))
            advance()

    timeout_timer.timeout.connect(_on_timeout)

    def advance():
        if state["index"] >= state["total_steps"]:
            _finish()
            return
        state["index"] += 1
        if memprof and (state["index"] == 1 or state["index"] % memprof_every == 0):
            _mem_snapshot(f"nav#{state['index']}")
        if state["index"] == state["count"] + 1 and revisit > 0:
            safe_print(
                f"[AUTOTEST] ===== full loop closed ({state['count']} navigations); "
                f"revisiting first {revisit} image(s) to compare warm vs cold =====",
                flush=True,
                force=True,
            )
        state["paints_this_nav"] = []
        state["waiting"] = True
        state["nav_start"] = time.time()
        settle_timer.stop()
        viewer.navigate_to_next_image()
        timeout_timer.start(int(timeout_s * 1000))

    def _wait_for_folder_then_start():
        files = getattr(viewer, "image_files", None) or []
        n = len(files)
        prev_n = getattr(_wait_for_folder_then_start, "_prev_n", None)
        # n > 1 guards against falsely declaring "stable" while image_files is
        # still just the single fast-opened file, before the real background
        # scan lands -- a plain two-poll-match check can lock in at 1 if the
        # scan is slow enough to still be running after two 300ms polls.
        stable = prev_n is not None and prev_n == n and n > 1
        _wait_for_folder_then_start._prev_n = n
        if stable or time.time() >= folder_wait_deadline:
            count = int(count_env) if count_env else n
            state["count"] = count
            state["total_steps"] = count + max(0, revisit)
            safe_print(
                f"[AUTOTEST] folder ready: {n} file(s) found, "
                f"running {count} navigation(s) (full loop back to start)"
                + (f" + {revisit} revisit" if revisit > 0 else ""),
                flush=True,
                force=True,
            )
            if raw_mode and _enter_raw_mode():
                # The toggle reloads the current image; let that settle so it
                # doesn't bleed into the first navigation's measurement.
                QTimer.singleShot(3000, advance)
            else:
                if not raw_mode:
                    safe_print(
                        "[AUTOTEST] workflow: embedded-JPEG (default preview workflow)",
                        flush=True,
                        force=True,
                    )
                advance()
        else:
            QTimer.singleShot(300, _wait_for_folder_then_start)

    def _finish():
        if memprof:
            _mem_snapshot("final")
        safe_print("[AUTOTEST] ===== SUMMARY =====", flush=True, force=True)
        if state["settled_below_sensor"]:
            safe_print(
                f"[AUTOTEST] settled-below-sensor-res files: "
                f"{state['settled_below_sensor']} (final form is the embedded "
                f"preview; timed at their last paint, not the hard ceiling)",
                flush=True,
                force=True,
            )
        count = state["count"]
        cold = state["results"][:count]
        warm = state["results"][count:]
        for label, rows in (("cold (first pass)", cold), ("warm (post-loop revisit)", warm)):
            times = [r[3] for r in rows]
            if not times:
                continue
            ts = sorted(times)
            n = len(ts)
            mean = sum(times) / n
            p95 = ts[min(n - 1, int(round(0.95 * (n - 1))))]
            multi_paint = sum(1 for r in rows if r[4] > 1)
            safe_print(
                f"[AUTOTEST] {label}: n={n} mean={mean:.3f}s median={ts[n // 2]:.3f}s "
                f"p95={p95:.3f}s min={ts[0]:.3f}s max={ts[-1]:.3f}s "
                f"multi-paint={multi_paint}/{n}",
                flush=True,
                force=True,
            )
        if warm:
            # Pair each warm revisit with its cold counterpart (same file, same
            # position in the revisit prefix) to show the reload delta directly.
            safe_print("[AUTOTEST] ----- cold vs warm (same file) -----", flush=True, force=True)
            for i, (w_name, w_w, w_h, w_t, w_np) in enumerate(warm):
                if i < len(cold):
                    c_name, c_w, c_h, c_t, c_np = cold[i]
                    safe_print(
                        f"[AUTOTEST]   {w_name}: cold={c_t:.3f}s({c_np}p) "
                        f"warm={w_t:.3f}s({w_np}p) delta={w_t - c_t:+.3f}s",
                        flush=True,
                        force=True,
                    )
        for name, w, h, t, n_paints in state["results"]:
            safe_print(f"[AUTOTEST]   {name}: {w}x{h} {t:.3f}s paints={n_paints}", flush=True, force=True)
        _restore_workflow_settings()
        QApplication.instance().quit()

    # Keep strong refs alive for the lifetime of the run (avoid GC of closures/timer)
    viewer._autotest_refs = (
        state,
        mem_state,
        timeout_timer,
        settle_timer,
        heartbeat_timer,
        orig_display_pixmap,
        advance,
        _finish,
        _on_timeout,
        _on_settle_idle,
        _wait_for_folder_then_start,
        _mem_snapshot,
        _enter_raw_mode,
        _restore_workflow_settings,
    )
    QTimer.singleShot(int(settle_s * 1000), _wait_for_folder_then_start)
