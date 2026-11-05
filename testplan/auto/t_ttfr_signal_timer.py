#!/usr/bin/env python3
"""Regression guard for the signal + async-timer first-render / upgrade tracking.

Replaces the old polling watchdog (`_schedule_ttfr_watchdog` ->
`_check_ttfr_watchdog` recursion via QTimer.singleShot with a generation
counter). The polled design gave false "load->first-render MISSING" warnings
and never noticed a preview that stayed blurry because the sensor decode was
starved behind neighbor prefetch.

New design (all on the GUI thread):

  paint -> _on_single_view_content_displayed()
        -> emits _single_view_first_render(str, float, int, int) once per load
        -> _on_first_render_ack slot stops _ttfr_timeout_timer and logs TTFR
        -> _note_displayed_resolution() arms/stops the full-resolution upgrade
           timer based on the buffer actually on screen

  _ttfr_timeout_timer.timeout -> _on_first_render_timeout()  (re-arm while busy)
  _full_res_upgrade_timer.timeout -> _on_full_res_upgrade_timeout() (force decode)

This test parses src/main.py with ast so it runs without a Qt install; it
asserts the structure that makes the design work and that the old recursive
watchdog is gone.
"""

import ast
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MAIN_PY = os.path.join(ROOT, "src", "main.py")

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def _viewer_class(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "RAWImageViewer":
            return node
    return None


def _method_source(src, klass, name):
    for node in klass.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(src, node) or ""
    return ""


def _signal_attr_names(klass):
    names = []
    for node in klass.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    value = node.value
                    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                        if value.func.id == "pyqtSignal":
                            names.append(target.id)
    return names


def main() -> int:
    with open(MAIN_PY, "r", encoding="utf-8") as fh:
        src = fh.read()

    try:
        tree = ast.parse(src)
    except SyntaxError as exc:  # pragma: no cover
        print(f"src/main.py does not parse: {exc}")
        return 1

    klass = _viewer_class(tree)
    check("RAWImageViewer class found", klass is not None)
    if klass is None:
        print("\n1 failure(s)")
        return 1

    # --- signals declared -----------------------------------------------------
    signals = _signal_attr_names(klass)
    check(
        "_single_view_first_render signal declared",
        "_single_view_first_render" in signals,
        str(signals),
    )
    check(
        "_single_view_render_timeout signal declared",
        "_single_view_render_timeout" in signals,
    )
    check(
        "_sensor_dims_resolved signal still declared",
        "_sensor_dims_resolved" in signals,
    )

    # --- old recursive watchdog removed --------------------------------------
    method_names = {
        n.name for n in klass.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    check(
        "old _check_ttfr_watchdog method removed",
        "_check_ttfr_watchdog" not in method_names,
    )
    check(
        "_schedule_ttfr_watchdog only survives as a back-compat shim",
        "_schedule_ttfr_watchdog" in method_names,
    )

    # --- new methods present --------------------------------------------------
    for name in (
        "_arm_first_render_timer",
        "_on_first_render_ack",
        "_on_first_render_timeout",
        "_recover_stuck_first_render",
        "_reset_stuck_load_state",
        "_note_displayed_resolution",
        "_arm_full_res_upgrade_timer",
        "_on_full_res_upgrade_timeout",
        "_displayed_buffer_covers_sensor",
    ):
        check(f"method {name} exists", name in method_names)

    # --- arm uses the owned timer, not singleShot recursion -------------------
    arm_src = _method_source(src, klass, "_arm_first_render_timer")
    check(
        "_arm_first_render_timer starts the owned timeout timer",
        "_ttfr_timeout_timer" in arm_src and ".start(" in arm_src,
    )

    ack_src = _method_source(src, klass, "_on_first_render_ack")
    check(
        "_on_first_render_ack stops the timeout timer",
        ".stop()" in ack_src,
    )
    check(
        "_on_first_render_ack sets the logged flag",
        "_single_view_first_render_logged = True" in ack_src,
    )

    timeout_src = _method_source(src, klass, "_on_first_render_timeout")
    check(
        "_on_first_render_timeout re-arms while still loading",
        ".start(" in timeout_src,
    )
    check(
        "_on_first_render_timeout recovers a stuck render",
        "_recover_stuck_first_render" in timeout_src,
    )

    recover_src = _method_source(src, klass, "_recover_stuck_first_render")
    check(
        "recovery evicts phantom in-flight tasks",
        "_evict_stale_active_tasks" in recover_src,
    )
    check(
        "recovery forces a sensor decode",
        "_queue_sensor_full_decode" in recover_src,
    )

    # --- upgrade watchdog -----------------------------------------------------
    note_src = _method_source(src, klass, "_note_displayed_resolution")
    check(
        "_note_displayed_resolution consults the on-screen buffer",
        "_displayed_buffer_covers_sensor" in note_src,
    )
    check(
        "_note_displayed_resolution stops the timer once covered",
        ".stop()" in note_src,
    )

    up_src = _method_source(src, klass, "_on_full_res_upgrade_timeout")
    check(
        "upgrade timeout cancels the file's own stale tasks before forcing",
        "cancel_task(" in up_src,
    )
    check(
        "upgrade timeout preempts non-current heavy RAW work",
        "cancel_non_current_heavy_raw_except" in up_src,
    )
    check(
        "upgrade timeout forces a fresh sensor decode",
        "_queue_sensor_full_decode" in up_src and "force=True" in up_src,
    )
    check(
        "upgrade watchdog is capped by an attempts limit",
        "max_attempts" in up_src,
    )
    check(
        "upgrade watchdog resets stuck state after exhausting attempts",
        "_reset_stuck_load_state" in up_src,
    )
    check(
        "upgrade timeout waits for a genuinely in-flight full decode",
        "has_active_full_stage_for_path" in up_src and "grace_attempts" in up_src,
    )

    # --- wiring ---------------------------------------------------------------
    disp_src = _method_source(src, klass, "_on_single_view_content_displayed")
    check(
        "_on_single_view_content_displayed emits the first-render signal",
        "_single_view_first_render.emit(" in disp_src,
    )
    check(
        "_on_single_view_content_displayed syncs the upgrade watchdog",
        "_note_displayed_resolution" in disp_src,
    )
    check(
        "_note_displayed_resolution is deferred to avoid re-entrancy during paint",
        "singleShot(0" in disp_src,
    )

    load_src = _method_source(src, klass, "load_raw_image")
    check(
        "load_raw_image arms the first-render timer",
        "_arm_first_render_timer(" in load_src,
    )
    check(
        "load_raw_image preempts neighbor heavy RAW for the on-screen file",
        "cancel_non_current_heavy_raw_except" in load_src,
    )
    check(
        "load_raw_image resets the upgrade watchdog state",
        "_full_res_upgrade_path" in load_src,
    )
    check(
        "load_raw_image clears a stale sensor-decode gate from a previous file",
        "_sensor_full_decode_queued_norm" in load_src,
    )

    # Connections must be established (signal -> slot) so the timer is actually
    # stopped on a normal fast load; without it every load would fall through to
    # the timeout path.
    check(
        "_single_view_first_render is connected to its slot",
        "_single_view_first_render.connect(" in src,
    )
    check(
        "_ttfr_timeout_timer is connected to its timeout slot",
        "_ttfr_timeout_timer.timeout.connect(" in src,
    )
    check(
        "_full_res_upgrade_timer is connected to its timeout slot",
        "_full_res_upgrade_timer.timeout.connect(" in src,
    )

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s)")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
