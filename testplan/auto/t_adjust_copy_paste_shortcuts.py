#!/usr/bin/env python3
"""Regression guard: Ctrl/Cmd+C and Ctrl/Cmd+V to copy/paste settings in Adjust panel.
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
    from PyQt6.QtCore import Qt, QObject

    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841
    import main as mainmod

    class MockEvent:
        def __init__(self, key, modifiers, auto_repeat=False):
            self._key = key
            self._modifiers = modifiers
            self._auto_repeat = auto_repeat

        def key(self):
            return self._key

        def modifiers(self):
            return self._modifiers

        def isAutoRepeat(self):
            return self._auto_repeat

    class MockPanel:
        def __init__(self):
            self.copy_called = False
            self.paste_called = False

        def _on_copy_settings_clicked(self):
            self.copy_called = True

        def _on_paste_settings_clicked(self):
            self.paste_called = True

    class MockStatusBar:
        def __init__(self):
            self.messages = []

        def showMessage(self, msg, timeout):
            self.messages.append(msg)

    class MockViewer(QObject):
        def __init__(self):
            super().__init__()
            self._focus_widget = None
            self.view_mode = "single"
            self._adjust_overlay_visible = True
            self.single_image_adjust_panel = MockPanel()
            self.status_bar = MockStatusBar()

        def focusWidget(self):
            return self._focus_widget

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    m = MockViewer()

    # Bind the _handle_app_shortcut method
    m._handle_app_shortcut = mainmod.RAWImageViewer._handle_app_shortcut.__get__(m)

    # Test 1: Copy settings via Ctrl+C (Windows/Linux modifier)
    evt_copy_ctrl = MockEvent(Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    res = m._handle_app_shortcut(evt_copy_ctrl)
    check("Ctrl+C handles copy settings", res is True)
    check("copy_called is True after Ctrl+C", m.single_image_adjust_panel.copy_called is True)
    check("status bar message shown for copy", "Edit settings copied" in m.status_bar.messages)

    # Test 2: Paste settings via Ctrl+V
    # We must seed the clipboard so Paste is executed
    from rawviewer_ui.adjust_panel import _EDIT_SETTINGS_CLIPBOARD
    import rawviewer_ui.adjust_panel as ap
    ap._EDIT_SETTINGS_CLIPBOARD = {"Exposure2012": 1.0}

    evt_paste_ctrl = MockEvent(Qt.Key.Key_V, Qt.KeyboardModifier.ControlModifier)
    res2 = m._handle_app_shortcut(evt_paste_ctrl)
    check("Ctrl+V handles paste settings", res2 is True)
    check("paste_called is True after Ctrl+V", m.single_image_adjust_panel.paste_called is True)
    check("status bar message shown for paste", "Edit settings pasted" in m.status_bar.messages)

    # Test 3: Copy settings via Cmd+C (macOS modifier)
    m.single_image_adjust_panel.copy_called = False
    evt_copy_cmd = MockEvent(Qt.Key.Key_C, Qt.KeyboardModifier.MetaModifier)
    res_cmd = m._handle_app_shortcut(evt_copy_cmd)
    check("Cmd+C handles copy settings", res_cmd is True)
    check("copy_called is True after Cmd+C", m.single_image_adjust_panel.copy_called is True)

    # Test 4: Shortcut blocked by text input focus
    from PyQt6.QtWidgets import QLineEdit
    mock_line_edit = QLineEdit()
    m._focus_widget = mock_line_edit
    m.single_image_adjust_panel.copy_called = False
    res_blocked = m._handle_app_shortcut(evt_copy_ctrl)
    check("Blocked when text input is focused", res_blocked is False)
    check("copy_called remains False", m.single_image_adjust_panel.copy_called is False)

    # Test 5: Shortcuts inactive when Adjust panel is closed
    m._focus_widget = None
    m._adjust_overlay_visible = False
    m.single_image_adjust_panel.copy_called = False
    res_inactive = m._handle_app_shortcut(evt_copy_ctrl)
    check("Ctrl+C ignored when Adjust panel is hidden", res_inactive is False or res_inactive is None)
    check("copy_called remains False", m.single_image_adjust_panel.copy_called is False)

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
