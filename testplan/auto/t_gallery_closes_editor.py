#!/usr/bin/env python3
"""Regression guard: switching to gallery view must close an open editor.

Two real bugs from the same root cause (reported back-to-back from a real
run): (1) after editing an image and returning to the gallery, the Adjust
panel stayed visible, splitting the window between gallery (upper half)
and the still-open editor (lower half), because _show_gallery_view only
ever hid single_view_container -- never single_image_adjust_panel, so the
splitter just gave the panel the space the image view gave up. (2) that
also left _adjust_overlay_visible stuck True, so the *next* gallery ->
single-image click silently behaved as if still editing and failed to
render (a real run's log: load_raw_image completes, but
pixels_on_screen=False at first-render).

_show_gallery_view touches enough unrelated machinery (gallery widgets,
image manager, filmstrip, title bar) that mocking a full call is too
fragile to be a reliable regression guard; this checks the fix is present
via source inspection (same approach as t_dodge_burn_reset.py's wiring
check) plus a direct unit test of the _adjust_panel_active() predicate
the fix relies on.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841
    import inspect
    import re

    import main as mainmod

    # 1. _adjust_panel_active() correctly reflects panel + overlay state.
    class FakePanel:
        def __init__(self, visible):
            self._visible = visible

        def isVisible(self):
            return self._visible

    def make_mock(*, view_mode, overlay_visible, panel_visible):
        m = type("M", (), {})()
        m.view_mode = view_mode
        m._adjust_overlay_visible = overlay_visible
        m.single_image_adjust_panel = FakePanel(panel_visible)
        m._adjust_panel_active = mainmod.RAWImageViewer._adjust_panel_active.__get__(m)
        return m

    m = make_mock(view_mode="single", overlay_visible=True, panel_visible=True)
    check("active when single view + overlay + panel all visible", m._adjust_panel_active() is True)

    m2 = make_mock(view_mode="gallery", overlay_visible=True, panel_visible=True)
    check("inactive in gallery view even if overlay flag is stale True", m2._adjust_panel_active() is False)

    m3 = make_mock(view_mode="single", overlay_visible=False, panel_visible=True)
    check("inactive when overlay flag is False", m3._adjust_panel_active() is False)

    # 2. The fix itself: _show_gallery_view must close the editor before
    # tearing down single-view state, using the same predicate.
    src = inspect.getsource(mainmod.RAWImageViewer._show_gallery_view)
    has_check = re.search(r"if\s+self\._adjust_panel_active\(\)\s*:", src) is not None
    has_close_call = re.search(
        r"self\._set_adjust_panel_visible\(\s*False\s*\)", src
    ) is not None
    check("_show_gallery_view checks _adjust_panel_active()", has_check)
    check("_show_gallery_view closes the panel via _set_adjust_panel_visible(False)", has_close_call)

    # The close call must come before single_view_container is hidden --
    # order matters here (_set_adjust_panel_visible reads current single-view
    # state while deciding what to restore, e.g. the filmstrip).
    hide_idx = src.find("single_view_container.hide()")
    close_idx = src.find("_set_adjust_panel_visible(False)")
    check(
        "editor is closed before single_view_container is hidden",
        -1 not in (hide_idx, close_idx) and close_idx < hide_idx,
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
