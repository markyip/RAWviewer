"""Phase 1 item 1: session save debounce on navigation/display paths."""
from __future__ import annotations

import os
import re
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

app = QApplication([])


class _DebounceHarness:
    """Minimal stand-in for RAWImageViewer session debounce."""

    def __init__(self) -> None:
        self.save_count = 0
        self._save_session_debounce_timer = QTimer()
        self._save_session_debounce_timer.setSingleShot(True)
        self._save_session_debounce_timer.setInterval(420)
        self._save_session_debounce_timer.timeout.connect(self.save_session_state)

    def schedule_save_session_state(self) -> None:
        t = getattr(self, "_save_session_debounce_timer", None)
        if t is None:
            self.save_session_state()
            return
        t.start()

    def save_session_state(self) -> None:
        self.save_count += 1


def test_debounce_coalesces_rapid_calls() -> None:
    h = _DebounceHarness()
    for _ in range(20):
        h.schedule_save_session_state()
    assert h.save_count == 0, "save should not run synchronously while debounce pending"
    deadline = time.time() + 2.0
    while h.save_count == 0 and time.time() < deadline:
        app.processEvents()
        time.sleep(0.05)
    assert h.save_count == 1, f"expected 1 save after debounce, got {h.save_count}"


def test_close_flushes_immediately() -> None:
    h = _DebounceHarness()
    h.schedule_save_session_state()
    h._save_session_debounce_timer.stop()
    h.save_session_state()
    assert h.save_count == 1


def test_main_py_call_sites() -> None:
    main_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
    with open(main_path, encoding="utf-8") as f:
        lines = f.readlines()

    def fn_body(name: str) -> str:
        start = next(i for i, ln in enumerate(lines) if ln.startswith(f"    def {name}("))
        end = start + 1
        while end < len(lines) and (lines[end].startswith("    ") or lines[end].strip() == ""):
            if end > start + 1 and lines[end].startswith("    def "):
                break
            end += 1
        return "".join(lines[start:end])

    next_body = fn_body("navigate_to_next_image")
    prev_body = fn_body("navigate_to_previous_image")
    load_body = fn_body("load_raw_image")

    assert "schedule_save_session_state()" in next_body
    assert "save_session_state()" not in next_body.replace("schedule_save_session_state()", "")
    assert "schedule_save_session_state()" in prev_body
    assert "save_session_state()" not in prev_body.replace("schedule_save_session_state()", "")
    assert "schedule_save_session_state()" in load_body

    close_src = fn_body("closeEvent")
    assert "save_session_state()" in close_src
    assert "schedule_save_session_state()" not in close_src

    full = "".join(lines)
    assert re.search(
        r"Background folder load applied[\s\S]{0,400}?self\.save_session_state\(\)",
        full,
    ), "background folder load must call save_session_state() directly"


if __name__ == "__main__":
    test_debounce_coalesces_rapid_calls()
    test_close_flushes_immediately()
    test_main_py_call_sites()
    print("OK: Phase 1 item 1 (session debounce) tests passed")
