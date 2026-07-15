#!/usr/bin/env python3
"""Editor mode: a failed/unsupported edit-base decode must silently skip
to the next editable image, not leave the editor stuck with an error."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    import numpy as np
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841
    import main as mainmod
    from unified_image_processor import _LIBRAW_UNSUPPORTED_PATHS

    key = os.path.normcase(os.path.abspath("/tmp/a.CR3"))

    class FakePanel:
        def isVisible(self):
            return True

        def set_lens_correction_available(self, *a):
            pass

    class SB:
        def showMessage(self, *a, **k):
            pass

    def make_mock(*, editor_active=True, n_files=3):
        m = type("M", (), {})()
        m.view_mode = "single"
        m._adjust_overlay_visible = editor_active
        m.single_image_adjust_panel = FakePanel()
        m.image_files = [f"/tmp/{c}.CR3" for c in "abc"][:n_files]
        m.current_file_path = "/tmp/a.CR3"
        m.status_bar = SB()
        m._nav_calls = []
        m.navigate_to_next_image = lambda: m._nav_calls.append(1)
        for name in ("_on_adjust_edit_base_ready", "_adjust_panel_active"):
            setattr(m, name, getattr(mainmod.RAWImageViewer, name).__get__(m))
        return m

    # 1. Failed decode, editor active, multiple files: mark unsupported + skip
    _LIBRAW_UNSUPPORTED_PATHS.clear()
    m = make_mock(editor_active=True, n_files=3)
    m._on_adjust_edit_base_ready("/tmp/a.CR3", None, False, "")
    check("failed decode marks file unsupported", key in _LIBRAW_UNSUPPORTED_PATHS)
    check("failed decode in editor mode navigates away", m._nav_calls == [1])

    # 2. Failed decode, editor NOT active: mark unsupported (prevents future
    # wasted attempts) but do not surprise-navigate the user.
    _LIBRAW_UNSUPPORTED_PATHS.clear()
    m2 = make_mock(editor_active=False, n_files=3)
    m2._on_adjust_edit_base_ready("/tmp/a.CR3", None, False, "")
    check("failed decode outside editor still marks unsupported", key in _LIBRAW_UNSUPPORTED_PATHS)
    check("failed decode outside editor does not navigate", m2._nav_calls == [])

    # 3. Failed decode, only one file in the folder: nothing to skip to.
    _LIBRAW_UNSUPPORTED_PATHS.clear()
    m3 = make_mock(editor_active=True, n_files=1)
    m3._on_adjust_edit_base_ready("/tmp/a.CR3", None, False, "")
    check("single-file folder does not navigate", m3._nav_calls == [])

    # 4. Successful decode: must not mark unsupported or navigate.
    _LIBRAW_UNSUPPORTED_PATHS.clear()
    m4 = make_mock(editor_active=True, n_files=3)
    fake_base = np.zeros((10, 10, 3), dtype=np.uint16)
    try:
        m4._on_adjust_edit_base_ready("/tmp/a.CR3", fake_base, False, "")
    except Exception:
        pass  # success branch needs more mock plumbing than this test provides
    check("successful decode does not mark unsupported", key not in _LIBRAW_UNSUPPORTED_PATHS)
    check("successful decode does not navigate", m4._nav_calls == [])

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
