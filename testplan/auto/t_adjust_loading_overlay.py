#!/usr/bin/env python3
"""Regression guard: opening the editor (Adjust panel) must not leave the
generic single-view "Loading Image..." overlay stuck forever.

Root cause: _toggle_adjust_panel() -> toggle_raw_jpeg_workflow() shows the
overlay and dispatches an async reload; immediately after,
_set_adjust_panel_visible(True) -> _begin_adjust_editing_session() cancels
that same image_manager task -- the only thing that would have hidden the
overlay -- before it can complete.
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
    import main as mainmod

    class FakeImageManager:
        def __init__(self):
            self.cancel_calls = []

        def cancel_task(self, path):
            self.cancel_calls.append(path)

    class FakeLoadingOverlay:
        def __init__(self):
            self.hide_calls = 0
            self.visible = False

        def show_loading(self, msg=None):
            self.visible = True

        def hide_loading(self):
            self.hide_calls += 1
            self.visible = False

    def make_mock():
        m = type("M", (), {})()
        m.image_manager = FakeImageManager()
        m.loading_overlay = FakeLoadingOverlay()
        m._reset_adjust_compare_state = lambda: None
        m._request_adjust_edit_base = lambda path: None
        m._begin_adjust_editing_session = mainmod.RAWImageViewer._begin_adjust_editing_session.__get__(m)
        return m

    # Simulate the exact race: the reload path already showed the overlay
    # before the editor-opening path cancels the in-flight task.
    m = make_mock()
    m.loading_overlay.show_loading("Loading Image...")
    check("overlay visible before opening editor (simulated race)", m.loading_overlay.visible is True)

    m._begin_adjust_editing_session("/tmp/a.CR3")

    check("competing task was cancelled", m.image_manager.cancel_calls == ["/tmp/a.CR3"])
    check("overlay is hidden once the editing session begins", m.loading_overlay.visible is False)
    check("hide_loading was actually called", m.loading_overlay.hide_calls >= 1)

    # No-op path: empty file_path must not touch the overlay or crash.
    m2 = make_mock()
    m2.loading_overlay.show_loading("Loading Image...")
    m2._begin_adjust_editing_session("")
    check("empty file_path leaves overlay untouched (early return)", m2.loading_overlay.hide_calls == 0)

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
