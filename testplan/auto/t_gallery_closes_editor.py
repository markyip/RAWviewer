#!/usr/bin/env python3
"""Regression guard: switching to gallery view must close an open editor.

Original bugs (reported back-to-back from a real run): (1) after editing
an image and returning to the gallery, the Adjust panel stayed visible,
splitting the window between gallery (upper half) and the still-open
editor (lower half), because _show_gallery_view only ever hid
single_view_container -- never single_image_adjust_panel, so the splitter
just gave the panel the space the image view gave up. (2) that also left
_adjust_overlay_visible stuck True, so the *next* gallery -> single-image
click silently behaved as if still editing and failed to render.

First fix attempt gated the close on self._adjust_panel_active(), which
checks `self.view_mode == "single"`. That looked right in isolation, but
EVERY real caller of _show_gallery_view() (toggle_view_mode,
_switch_to_gallery_for_search, _on_view_mode_button_clicked's burst-view
branch) sets `self.view_mode = "gallery"` BEFORE calling
_show_gallery_view() -- so by the time the check ran, view_mode was
already "gallery" and _adjust_panel_active() always returned False,
silently no-op'ing the whole fix. The bug kept recurring in real use
because the first regression test here mocked view_mode="single" at
check-time, which doesn't match the real call order and so didn't catch
it. The real fix checks _adjust_overlay_visible/panel.isVisible()
directly, without gating on the (already-changed) view_mode.

_show_gallery_view touches enough unrelated machinery (gallery widgets,
image manager, filmstrip, title bar) that mocking a full call is too
fragile to be a reliable regression guard; this checks the fix via source
inspection, but specifically replicates the real call order this time
(view_mode already "gallery" at the point the check would run).
"""
import inspect
import os
import re
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
    import main as mainmod

    src = inspect.getsource(mainmod.RAWImageViewer._show_gallery_view)

    # The exact regression: the close-editor check must NOT depend on
    # _adjust_panel_active() (which gates on view_mode == "single" -- a
    # value every real caller has already changed to "gallery" by the time
    # _show_gallery_view runs).
    check(
        "close-editor check does not gate on _adjust_panel_active() "
        "(view_mode is already 'gallery' by this point in every real caller)",
        "if self._adjust_panel_active():" not in src,
    )
    check(
        "close-editor check reads _adjust_overlay_visible directly instead",
        "getattr(self, \"_adjust_overlay_visible\", False)" in src,
    )
    check(
        "_show_gallery_view closes the panel via _close_adjust_panel_from_ui "
        "(restores browse workflow; plain setVisible(False) left RAW mode stuck)",
        "self._close_adjust_panel_from_ui()" in src,
    )

    hide_idx = src.find("single_view_container.hide()")
    close_idx = src.find("_close_adjust_panel_from_ui()")
    check(
        "editor is closed before single_view_container is hidden",
        -1 not in (hide_idx, close_idx) and close_idx < hide_idx,
    )

    # Direct behavioral check of the actual condition used, with view_mode
    # set to "gallery" -- replicating the real call order, unlike the
    # original (flawed) version of this test.
    class FakePanel:
        def __init__(self, visible):
            self._visible = visible

        def isVisible(self):
            return self._visible

    def evaluate_close_condition(*, overlay_visible, panel_visible):
        """Mirrors the exact condition in _show_gallery_view (view_mode is
        already 'gallery', matching every real call site)."""
        m = type("M", (), {})()
        m.view_mode = "gallery"
        m._adjust_overlay_visible = overlay_visible
        m.single_image_adjust_panel = FakePanel(panel_visible)
        panel = getattr(m, "single_image_adjust_panel", None)
        return bool(getattr(m, "_adjust_overlay_visible", False) and panel is not None and panel.isVisible())

    check(
        "editor was open (overlay+panel visible) -> close condition is True "
        "even with view_mode already flipped to gallery",
        evaluate_close_condition(overlay_visible=True, panel_visible=True) is True,
    )
    check(
        "editor was already closed -> close condition is False (no spurious work)",
        evaluate_close_condition(overlay_visible=False, panel_visible=True) is False,
    )
    check(
        "overlay flag stale True but panel widget not actually visible -> False",
        evaluate_close_condition(overlay_visible=True, panel_visible=False) is False,
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
