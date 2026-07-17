#!/usr/bin/env python3
"""Esc / gallery close after Adjust must restore embedded-JPEG browse mode.

Opening Adjust from non-RAW browse flips use_embedded_jpeg_workflow to False.
_close_adjust_panel_from_ui must flip it back even when view_mode is already
"gallery" (every real gallery transition sets that before the close runs) —
toggle_raw_jpeg_workflow no-ops outside single view, which left RAW stuck.
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


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841
    import main as mainmod

    close_src = inspect.getsource(mainmod.RAWImageViewer._close_adjust_panel_from_ui)
    check(
        "close path restores embedded workflow when not in single view",
        'setValue("use_embedded_jpeg_workflow", True)' in close_src,
    )
    check(
        "close path still uses toggle when still in single view",
        "toggle_raw_jpeg_workflow()" in close_src,
    )
    check(
        "close path invalidates workflow settings memo",
        "invalidate_libraw_consistent_preview_settings" in close_src,
    )

    set_src = inspect.getsource(mainmod.RAWImageViewer._set_adjust_panel_visible)
    check(
        "opening adjust does not gate panel visibility on display image",
        "and self._single_view_has_display_image()" not in set_src,
    )

    block_src = inspect.getsource(mainmod.RAWImageViewer._shortcut_blocked_by_text_input)
    check(
        "read-only text fields do not block editor shortcuts",
        "isReadOnly" in block_src,
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
