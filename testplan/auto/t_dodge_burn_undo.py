#!/usr/bin/env python3
"""Regression: Ctrl/Cmd+Z must restore the live dodge/burn mask.

Undo used to call panel.set_adjustments(old_adj) then editing_finished,
which re-serialized the still-current painted mask. The previous mask
lives in old_adj[MASK_KEY] and must be deserialized into
RAWImageViewer._dodge_burn_mask before the XMP write.
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
    from raw_dodge_burn import MASK_KEY, DodgeBurnMask, serialize_mask, stamp_brush

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
        m._dodge_burn_luma_guide = object()  # cleared on restore
        m._dodge_burn_mask_shape_cache_key = ("x",)
        m._dodge_burn_mask_shape_cached = (1, 1)
        m.single_image_adjust_panel = FakePanel()
        m.gpu_view = None
        m._apply_dodge_burn_mask_from_adj = (
            mainmod.RAWImageViewer._apply_dodge_burn_mask_from_adj.__get__(m)
        )
        m._sync_dodge_burn_mask_overlay = (
            mainmod.RAWImageViewer._sync_dodge_burn_mask_overlay.__get__(m)
        )
        return m

    # 1. Undo to a previous painted mask restores that buffer.
    prev = DodgeBurnMask.empty(32, 32)
    stamp_brush(prev, 8.0, 8.0, 4.0, 0.4, dodge=True)
    prev_serial = serialize_mask(prev)
    check("fixture previous mask is non-empty", prev_serial != "")

    current = DodgeBurnMask.empty(32, 32)
    stamp_brush(current, 24.0, 24.0, 4.0, 0.4, dodge=False)

    m = make_mock(mask=current)
    m._apply_dodge_burn_mask_from_adj({MASK_KEY: prev_serial})
    restored = m._dodge_burn_mask
    check("undo restores a non-None mask", restored is not None)
    check(
        "restored mask matches previous paint",
        restored is not None
        and serialize_mask(restored) == prev_serial,
    )
    check("luma guide cache cleared", m._dodge_burn_luma_guide is None)
    check(
        "panel told mask is present",
        m.single_image_adjust_panel.mask_present_calls == [True],
    )

    # 2. Undo to empty (first stroke) clears the live mask.
    m2 = make_mock(mask=current)
    m2._apply_dodge_burn_mask_from_adj({})
    check("undo to empty clears live mask", m2._dodge_burn_mask is None)
    check(
        "panel told mask is gone",
        m2.single_image_adjust_panel.mask_present_calls == [False],
    )

    # 3. Shortcut path must restore mask before editing_finished.
    import inspect

    undo_src = inspect.getsource(mainmod.RAWImageViewer._shortcut_activate_undo)
    check(
        "undo calls _apply_dodge_burn_mask_from_adj before editing_finished",
        "_apply_dodge_burn_mask_from_adj" in undo_src
        and undo_src.find("_apply_dodge_burn_mask_from_adj")
        < undo_src.find("_on_adjust_panel_editing_finished"),
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
