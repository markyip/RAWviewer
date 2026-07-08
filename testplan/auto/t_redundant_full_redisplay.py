#!/usr/bin/env python3
"""Regression guard: don't re-run a ~66MP QPixmap conversion for a full-res
buffer that is already on screen.

Real bug (from a real running log, session restore of a 60MP Sony ARW): the
same sensor-res buffer for the current file was pushed through
display_numpy_image() three times in one restore -- the restored-zoom eager
full decode, the deferred post-restore full decode, and a later cached-full
re-delivery. Each pass drove a full 6656x9984 -> QPixmap conversion; only one
was caught by the pending-converter dedup, so ~2 extra 66MP conversions ran
for a buffer that never changed. display_numpy_image() now short-circuits when
an equal-or-higher-res buffer for the same file is already painted.

The guard must keep the escape hatches that legitimate same-dimension
re-displays rely on:
  * _skip_resolution_downgrade_check -> workflow toggle / RAW recovery repaint
  * _loading_from_gallery            -> gallery blur->sharp upgrade
  * _adjust_preview_painting / _adjust_panel_active -> live edit preview
  * recovery_preview                 -> RAW recovery buffer
...and it must sit AFTER the EDR display paths so EDR upgrades are never
suppressed.

Uses a lightweight fake object (mirrors t_gallery_closes_editor.py) for the
_already_displaying_buffer_for_path predicate, plus source inspection of
display_numpy_image() for the guard wiring (the method itself needs a full Qt
window to run).
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


class FakeArray:
    def __init__(self, h, w):
        self.shape = (h, w, 3)


def main() -> int:
    import main as m

    norm = m._norm_path

    # --- _already_displaying_buffer_for_path predicate ---
    stub = type("V", (), {})()
    stub._already_displaying_buffer_for_path = (
        m.RAWImageViewer._already_displaying_buffer_for_path.__get__(stub)
    )
    stub._single_view_pixels_on_screen = lambda fp: True

    stub._displayed_content_path = "/x/DSC03236.ARW"
    stub._manager_displayed_max_dim = 9984
    check(
        "identical full-res buffer already on screen -> already displaying (skip)",
        stub._already_displaying_buffer_for_path("/x/DSC03236.ARW", FakeArray(6656, 9984))
        is True,
    )
    check(
        "a strictly larger buffer is NOT considered already displayed (real upgrade)",
        stub._already_displaying_buffer_for_path("/x/DSC03236.ARW", FakeArray(8000, 12000))
        is False,
    )
    check(
        "different file on screen -> not already displaying",
        stub._already_displaying_buffer_for_path("/x/OTHER.ARW", FakeArray(6656, 9984))
        is False,
    )

    stub._manager_displayed_max_dim = 512  # only the preview is painted so far
    check(
        "full-res arriving while only the preview is painted -> not skipped",
        stub._already_displaying_buffer_for_path("/x/DSC03236.ARW", FakeArray(6656, 9984))
        is False,
    )

    stub._manager_displayed_max_dim = 9984
    stub._single_view_pixels_on_screen = lambda fp: False
    check(
        "nothing actually on screen -> not already displaying",
        stub._already_displaying_buffer_for_path("/x/DSC03236.ARW", FakeArray(6656, 9984))
        is False,
    )

    # --- display_numpy_image() guard wiring ---
    src = inspect.getsource(m.RAWImageViewer.display_numpy_image)
    check(
        "display_numpy_image consults _already_displaying_buffer_for_path",
        "_already_displaying_buffer_for_path(" in src,
    )
    check(
        "guard skips via an early return (Skipping redundant full-res re-display)",
        "Skipping redundant full-res re-display" in src,
    )
    for hatch in (
        "_skip_resolution_downgrade_check",
        "_loading_from_gallery",
        "_adjust_preview_painting",
        "_adjust_panel_active",
    ):
        check(
            f"guard preserves escape hatch: {hatch}",
            hatch in src,
        )

    # The guard must sit AFTER the EDR display worker dispatch, so EDR upgrades
    # (which legitimately repaint the same file) are never suppressed.
    edr_idx = src.find("_start_display_numpy_pixmap_worker(")
    guard_idx = src.find("Skipping redundant full-res re-display")
    check(
        "redundancy guard runs after the EDR display path (EDR upgrades not suppressed)",
        -1 not in (edr_idx, guard_idx) and edr_idx < guard_idx,
    )

    # --- _manager_displayed_max_dim is the per-file high-water-mark the guard
    # trusts. It MUST reset when the displayed file changes: otherwise a larger
    # previous image's value carries forward and makes a smaller new image's
    # full-res look "already displayed", so its full decode is skipped and it
    # sticks at the blurry preview (real regression: nav 8256px -> 6720px). ---
    disp_src = inspect.getsource(m.RAWImageViewer.display_pixmap)
    commit_idx = disp_src.find("_displayed_content_path = cur_path")
    reset_idx = disp_src.find("self._manager_displayed_max_dim = pm_max")
    maxrun_idx = disp_src.find("self._manager_displayed_max_dim = max(")
    check(
        "_commit_display_metadata resets the high-water-mark on file change (not just max())",
        reset_idx != -1 and maxrun_idx != -1,
    )
    check(
        "the reset is gated on the displayed path actually changing",
        "prev_displayed_path" in disp_src
        and disp_src.find("prev_displayed_path") < maxrun_idx,
    )

    # Behavioral: the predicate correctly reports NOT-already-displaying once the
    # high-water-mark reflects only the smaller current file (post-reset).
    stub2 = type("V2", (), {})()
    stub2._already_displaying_buffer_for_path = (
        m.RAWImageViewer._already_displaying_buffer_for_path.__get__(stub2)
    )
    stub2._single_view_pixels_on_screen = lambda fp: True
    stub2._displayed_content_path = "/x/398A0208.CR3"
    stub2._manager_displayed_max_dim = 512  # only the preview painted after reset
    check(
        "new smaller-image full-res is NOT 'already displayed' when only its preview is on screen",
        stub2._already_displaying_buffer_for_path("/x/398A0208.CR3", FakeArray(4480, 6720))
        is False,
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
