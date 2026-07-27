"""Down discards only where discarding is the job.

Down was a culling key that also fired in places culling is not what the
user is doing: in the gallery it threw away the selection whenever one
existed (Down in a grid means "move"), and in single view it discarded the
photograph even with the Adjust editor open, taking the edit with it.

Both are the same failure as the Delete key: a destructive action reachable
by a key pressed meaning something navigational.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


class _Host:
    """Records which action a Down press would take."""

    def __init__(self, view_mode, *, editor_open=False, has_selection=False):
        self.view_mode = view_mode
        self._adjust_overlay_visible = editor_open
        self._has_selection = has_selection
        self.actions = []

    def _shortcut_blocked_by_text_input(self):
        return False

    def _gallery_has_selection(self):
        return self._has_selection

    def discard_gallery_selection(self):
        self.actions.append("discard_selection")

    def _scroll_gallery_vertical(self, n):
        self.actions.append("scroll")

    def move_current_image_to_discard(self):
        self.actions.append("discard_image")

    def _compare_handle_reject_candidate(self):
        self.actions.append("reject_candidate")

    def _compare_handle_reject_select(self):
        self.actions.append("reject_select")


def _press(host, shift=False):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    mods = Qt.KeyboardModifier.ShiftModifier if shift else Qt.KeyboardModifier.NoModifier
    real = QApplication.keyboardModifiers
    QApplication.keyboardModifiers = staticmethod(lambda: mods)
    try:
        import main as mainmod

        mainmod.RAWImageViewer._shortcut_activate_gallery_down(host)
    finally:
        QApplication.keyboardModifiers = real
    return host.actions


def main() -> int:
    # --- single view: the one place plain Down discards ---
    h = _Host("single")
    check("single view: Down discards the image", _press(h) == ["discard_image"], str(h.actions))

    # --- but never with the editor open ---
    h = _Host("single", editor_open=True)
    check(
        "editor open: Down does NOT discard",
        _press(h) == [],
        f"{h.actions} -- discarding mid-edit loses the work as well as the file",
    )

    # --- gallery: Down navigates, always ---
    h = _Host("gallery", has_selection=True)
    check("gallery with a selection: Down scrolls", _press(h) == ["scroll"], str(h.actions))

    h = _Host("gallery", has_selection=False)
    check("gallery with no selection: Down scrolls", _press(h) == ["scroll"], str(h.actions))

    # --- gallery discard needs the modifier ---
    h = _Host("gallery", has_selection=True)
    check(
        "gallery Shift+Down discards the selection",
        _press(h, shift=True) == ["discard_selection"],
        str(h.actions),
    )

    h = _Host("gallery", has_selection=False)
    check(
        "gallery Shift+Down with nothing selected just scrolls",
        _press(h, shift=True) == ["scroll"],
        str(h.actions),
    )

    # --- compare keeps its reject gesture: rejecting IS the job there ---
    h = _Host("compare")
    check("compare: Down rejects the candidate", _press(h) == ["reject_candidate"], str(h.actions))

    h = _Host("compare")
    check("compare: Shift+Down rejects the selected", _press(h, shift=True) == ["reject_select"], str(h.actions))

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
