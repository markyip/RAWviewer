#!/usr/bin/env python3
"""Regression guard: the adjust panel's Reset button must also erase a
previously painted dodge/burn mask.

Root cause: the mask lives in RAWImageViewer._dodge_burn_mask, not in the
panel's adjustments dict Reset rebuilds -- _dodge_burn_overlay_adj() reads
that instance attribute and re-injects it into every render regardless of
what the "reset" dict contains. reset_requested was emitted by the panel
but never connected to anything in main.py, so Reset silently left the
mask (and its saved sidecar entry) in place.
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
    from raw_dodge_burn import DodgeBurnMask

    class FakePanel:
        def __init__(self):
            self.mask_present_calls = []

        def set_dodge_burn_mask_present(self, present):
            self.mask_present_calls.append(present)

        def dodge_burn_show_mask(self):
            return False

    def make_mock(*, mask):
        m = type("M", (), {})()
        m._dodge_burn_mask = mask
        m.single_image_adjust_panel = FakePanel()
        m.gpu_view = None
        m._on_adjust_panel_reset = mainmod.RAWImageViewer._on_adjust_panel_reset.__get__(m)
        m._sync_dodge_burn_mask_overlay = (
            mainmod.RAWImageViewer._sync_dodge_burn_mask_overlay.__get__(m)
        )
        return m

    # 1. A painted mask must be cleared by reset.
    m = make_mock(mask=DodgeBurnMask.empty(50, 50))
    m._on_adjust_panel_reset()
    check("reset clears a previously painted mask", m._dodge_burn_mask is None)
    check("panel is told the mask is gone", m.single_image_adjust_panel.mask_present_calls == [False])

    # 2. No mask -> no-op, no crash, no spurious UI update.
    m2 = make_mock(mask=None)
    m2._on_adjust_panel_reset()
    check("reset with no mask is a no-op", m2.single_image_adjust_panel.mask_present_calls == [])

    # 3. Wiring: reset_requested must actually be connected in real init_ui
    # (this is exactly what was missing -- the signal existed and fired,
    # nothing was listening).
    import inspect
    import re

    src = inspect.getsource(mainmod.RAWImageViewer.init_ui)
    wired = re.search(
        r"reset_requested\.connect\(\s*self\._on_adjust_panel_reset\s*\)", src
    )
    check(
        "init_ui connects reset_requested to _on_adjust_panel_reset",
        wired is not None,
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
