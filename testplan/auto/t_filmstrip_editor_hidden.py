#!/usr/bin/env python3
"""Regression guard: the filmstrip must stay disabled/hidden while the
editor (Adjust panel) is open, even when _sync_filmstrip_to_folder() runs
from a background event (e.g. folder-index-ready, which the real app
dispatches while editing is in progress).

Root cause: _sync_filmstrip_to_folder() unconditionally computed
`enabled = len(files) > 1` and called bar.setEnabled(enabled), ignoring
editor state entirely -- undoing the bar.setEnabled(False)/bar.hide()
that _set_adjust_panel_visible() had set up when the editor opened. Once
re-enabled, the filmstrip's own hover-reveal logic (gated only on
isEnabled()) could show it again mid-edit.
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

    class FakeBar:
        def __init__(self):
            self.enabled = True
            self.visible = True
            self.set_files_calls = 0

        def isEnabled(self):
            return self.enabled

        def setEnabled(self, v):
            self.enabled = v

        def isVisible(self):
            return self.visible

        def hide(self):
            self.visible = False

        def show(self):
            self.visible = True

        def set_files(self, *a, **k):
            self.set_files_calls += 1

    def make_mock(*, adjust_active, n_files=3):
        m = type("M", (), {})()
        m.view_mode = "single"
        m._adjust_overlay_visible = adjust_active
        m.image_files = [f"/tmp/{c}.CR3" for c in "abc"][:n_files]
        m.current_file_index = 0
        m.current_file_path = m.image_files[0]
        m._gallery_bulk_metadata = {}
        m._sync_filmstrip_in_progress = False

        class FakePanel:
            def isVisible(self):
                return True

        m.single_image_adjust_panel = FakePanel()
        bar = FakeBar()
        m._bar = bar
        m._filmstrip_bar = lambda: bar
        m._navigation_files = lambda: m.image_files
        m._is_semantic_search_filter_active = lambda: False
        m._hide_filmstrip_chrome = lambda: None
        m._refresh_filmstrip_bookmark_visuals = lambda: None
        m._defer_until_first_paint = lambda *a, **k: None
        m._single_view_first_render_logged = False
        m._sync_filmstrip_index = lambda **k: None
        m._prefetch_filmstrip_thumbnails = lambda: None
        for name in ("_adjust_panel_active", "_sync_filmstrip_to_folder"):
            setattr(m, name, getattr(mainmod.RAWImageViewer, name).__get__(m))
        return m

    # 1. Editor active, bar was previously disabled/hidden by
    # _set_adjust_panel_visible: a background sync must not re-enable it.
    m = make_mock(adjust_active=True)
    m._bar.enabled = False
    m._bar.visible = False
    m._sync_filmstrip_to_folder()
    check("editor active: bar stays disabled after background sync", m._bar.enabled is False)
    check("editor active: bar stays hidden after background sync", m._bar.visible is False)
    check("editor active: set_files was not called (no pointless work)", m._bar.set_files_calls == 0)

    # 2. Editor not active: normal sync re-enables/populates as before.
    m2 = make_mock(adjust_active=False)
    m2._bar.enabled = False
    m2._bar.visible = False
    m2._sync_filmstrip_to_folder()
    check("editor inactive: bar is enabled by a normal sync", m2._bar.enabled is True)
    check("editor inactive: set_files was called", m2._bar.set_files_calls == 1)

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
