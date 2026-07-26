#!/usr/bin/env python3
"""Adjust panel top-level tabs (Global / Masks) -- headless.

The Masks section moved out of the Global scroll column onto its own tab.
The risk in that move is not the tab bar; it is that a section which changed
parent quietly loses its wiring, or that a page ends up visible when it
should not be. These check the move itself, not the styling.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from PyQt6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from rawviewer_ui.adjust_panel import ImageAdjustPanelWidget, PanelTabBar  # noqa: E402

GLOBAL, MASKS = 0, 1


def _panel():
    p = ImageAdjustPanelWidget()
    p.show()  # visibility assertions are meaningless on an unshown widget
    return p


def _page_of(panel, widget):
    node = widget
    while node is not None:
        if node is panel._tab_page_masks:
            return "masks"
        if node is panel._tab_page_global:
            return "global"
        node = node.parentWidget()
    return "neither"


def test_two_tabs_global_first():
    p = _panel()
    assert [b.text() for b in p._panel_tabs._buttons] == ["GLOBAL", "MASKS"]
    assert p._panel_tabs.current() == GLOBAL
    assert p._tab_page_global.isVisible()
    assert not p._tab_page_masks.isVisible()
    print("  OK   two tabs, Global selected on open")


def test_masks_section_is_on_the_masks_page():
    p = _panel()
    assert _page_of(p, p.sect_masks) == "masks", "Masks section is not on the Masks page"
    # Everything else must have stayed put -- a move that dragged siblings
    # along would still pass a naive "masks tab exists" check.
    for name in ("sect_light", "sect_color", "sect_detail", "sect_local", "sect_lut"):
        sect = getattr(p, name)
        assert _page_of(p, sect) == "global", f"{name} unexpectedly moved off Global"
    print("  OK   only the Masks section moved")


def test_switching_toggles_exactly_one_page():
    p = _panel()
    p._panel_tabs.set_current(MASKS)
    assert p._tab_page_masks.isVisible() and not p._tab_page_global.isVisible()
    p._panel_tabs.set_current(GLOBAL)
    assert p._tab_page_global.isVisible() and not p._tab_page_masks.isVisible()
    print("  OK   exactly one page visible either way")


def test_reclicking_active_tab_does_not_deselect():
    """Checkable buttons untoggle on re-click unless the bar re-asserts state."""
    p = _panel()
    p._panel_tabs._buttons[GLOBAL].click()
    assert p._panel_tabs._buttons[GLOBAL].isChecked()
    assert p._panel_tabs.current() == GLOBAL
    assert p._tab_page_global.isVisible()
    print("  OK   re-clicking the active tab is a no-op")


def test_masks_page_opens_its_section():
    """A section collapsed on Global would leave the Masks page looking empty."""
    p = _panel()
    p.sect_masks.set_expanded(False)
    p._panel_tabs.set_current(MASKS)
    assert p.sect_masks._expanded, "Masks section stayed collapsed on its own page"
    print("  OK   arriving on Masks expands the section")


def test_mask_controls_survived_the_move():
    p = _panel()
    for attr in (
        "_mask_list",
        "_mask_add_btn",
        "_mask_del_btn",
        "_mask_dup_btn",
        "_mask_paint_btn",
        "_mask_erase_btn",
        "_mask_invert_btn",
    ):
        assert getattr(p, attr, None) is not None, f"{attr} lost in the reparent"
    print("  OK   mask control references intact")


def test_show_masks_tab_helper():
    p = _panel()
    p.show_masks_tab()
    assert p._panel_tabs.current() == MASKS
    assert p._tab_page_masks.isVisible()
    print("  OK   show_masks_tab() brings the page forward")


def test_adjustments_round_trip():
    """Reparenting must not disturb the adjustment dictionary."""
    p = _panel()
    before = p.get_adjustments()
    p.set_adjustments(dict(before))
    after = p.get_adjustments()
    assert set(before) == set(after), "adjustment keys changed across the move"
    assert len(after) > 20, f"suspiciously few adjustment keys: {len(after)}"
    print(f"  OK   get/set_adjustments round-trips ({len(after)} keys)")


def test_tab_bar_clamps_out_of_range():
    bar = PanelTabBar(("A", "B"))
    bar.set_current(99)
    assert bar.current() == 1
    bar.set_current(-5)
    assert bar.current() == 0
    print("  OK   tab index clamps to range")


def main() -> int:
    test_two_tabs_global_first()
    test_masks_section_is_on_the_masks_page()
    test_switching_toggles_exactly_one_page()
    test_reclicking_active_tab_does_not_deselect()
    test_masks_page_opens_its_section()
    test_mask_controls_survived_the_move()
    test_show_masks_tab_helper()
    test_adjustments_round_trip()
    test_tab_bar_clamps_out_of_range()
    print("\nPASS t_adjust_panel_tabs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
