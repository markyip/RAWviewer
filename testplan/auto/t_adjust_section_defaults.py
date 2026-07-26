#!/usr/bin/env python3
"""Adjust panel section collapse: per-run defaults, per-run memory.

Two halves that pull against each other:

* Each run of the app starts with the same predictable panel shape --
  Histogram and Light open, everything else collapsed.
* Within a run, however the user arranges the panel sticks, across image
  navigation and closing/reopening the editor.

Previously collapse state lived in QSettings, so it persisted across
restarts: one session spent opening every section to find something left
the panel fully expanded on every launch thereafter.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from PyQt6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from rawviewer_ui.adjust_panel import (  # noqa: E402
    _SECTION_DEFAULT_EXPANDED,
    ImageAdjustPanelWidget,
    reset_section_expanded_session,
)

# Every section that persists state, and the attribute holding it.
SECTIONS = {
    "histogram": "sect_histogram",
    "light": "sect_light",
    "color": "sect_color",
    "curve": "sect_curve",
    "hsl": "sect_hsl",
    "detail": "sect_detail",
    "noise": "sect_noise",
    "effects": "sect_effects",
    "local": "sect_local",
    "lut": "sect_lut",
    "transform": "sect_transform",
}


def _fresh_panel():
    """A panel as it would be built on the first editor open of a run."""
    reset_section_expanded_session()
    return ImageAdjustPanelWidget()


def test_first_open_expands_only_histogram_and_light():
    p = _fresh_panel()
    expanded = {
        key for key, attr in SECTIONS.items() if getattr(p, attr)._expanded
    }
    assert expanded == {"histogram", "light"}, (
        f"expected only Histogram and Light open, got {sorted(expanded)}"
    )
    print(f"  OK   first open expands only {sorted(expanded)}")


def test_defaults_constant_matches_behaviour():
    """The constant is the documented contract; keep it honest."""
    assert set(_SECTION_DEFAULT_EXPANDED) == {"histogram", "light"}
    print("  OK   _SECTION_DEFAULT_EXPANDED matches")


def test_arrangement_survives_a_rebuilt_panel():
    """Standing in for image navigation / reopening the editor tab.

    State has to outlive the widget, since a new panel is a new set of
    CollapsibleSection objects.
    """
    p = _fresh_panel()
    p.sect_light.set_expanded(False)   # close a default-open one
    p.sect_hsl.set_expanded(True)      # open a default-closed one
    p.sect_effects.set_expanded(True)

    reopened = ImageAdjustPanelWidget()  # no reset: same run
    assert reopened.sect_light._expanded is False, "closed section reopened itself"
    assert reopened.sect_hsl._expanded is True, "opened section forgot"
    assert reopened.sect_effects._expanded is True, "opened section forgot"
    assert reopened.sect_histogram._expanded is True, "untouched default changed"
    assert reopened.sect_curve._expanded is False, "untouched default changed"
    print("  OK   arrangement survives a rebuilt panel")


def test_next_run_starts_from_defaults_again():
    """A new run must not inherit the previous run's arrangement."""
    p = _fresh_panel()
    for attr in SECTIONS.values():
        getattr(p, attr).set_expanded(True)  # the "opened everything" session

    nxt = _fresh_panel()  # reset == process restart
    expanded = {
        key for key, attr in SECTIONS.items() if getattr(nxt, attr)._expanded
    }
    assert expanded == {"histogram", "light"}, (
        f"previous run's arrangement leaked into a new run: {sorted(expanded)}"
    )
    print("  OK   a new run starts from defaults again")


def test_collapsed_sections_are_actually_hidden():
    """The flag must match what is on screen, not just bookkeeping."""
    p = _fresh_panel()
    p.show()
    assert p.sect_light.content.isVisible(), "Light should be open and visible"
    assert not p.sect_hsl.content.isVisible(), "HSL should be collapsed and hidden"
    p.sect_hsl.set_expanded(True)
    assert p.sect_hsl.content.isVisible(), "expanding did not reveal the content"
    print("  OK   collapse state matches actual visibility")


def test_masks_section_is_open_on_its_own_tab():
    """It has no header to expand it with, so it must never start collapsed."""
    p = _fresh_panel()
    assert p.sect_masks._expanded is True
    assert p.sect_masks.header.isVisible() is False
    print("  OK   Masks section is open on its headerless tab")


def main() -> int:
    test_first_open_expands_only_histogram_and_light()
    test_defaults_constant_matches_behaviour()
    test_arrangement_survives_a_rebuilt_panel()
    test_next_run_starts_from_defaults_again()
    test_collapsed_sections_are_actually_hidden()
    test_masks_section_is_open_on_its_own_tab()
    print("\nPASS t_adjust_section_defaults")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
