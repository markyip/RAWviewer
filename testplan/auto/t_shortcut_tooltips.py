#!/usr/bin/env python3
"""Controls that have a keyboard shortcut must say so on hover.

A shortcut nobody can discover is a shortcut nobody uses. The cheat sheet
(shortcut_cheatsheet.py) teaches the vocabulary up front; this guards the
other half -- that the control itself names its key at the moment you are
looking at it.

Only controls that ACTUALLY have a shortcut are asserted. Adding a tooltip
naming a key that does nothing would be worse than saying nothing.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])

    import theme
    from image_histogram import ImageHistogramWidget
    from rawviewer_ui.adjust_panel import ImageAdjustPanelWidget

    # --- the helper itself ------------------------------------------------
    mac = sys.platform == "darwin"
    combo = theme.shortcut("Ctrl", "C")
    check(
        "Ctrl renders as the platform's own modifier",
        combo == ("⌘C" if mac else "Ctrl+C"),
        f"got {combo!r}",
    )
    check(
        "a bare key is unchanged",
        theme.shortcut("P") == "P",
        f"got {theme.shortcut('P')!r}",
    )
    check(
        "with_shortcut puts the key at the end, parenthesised",
        theme.with_shortcut("Do a thing", "Ctrl", "C").endswith(f"({combo})"),
    )
    # Naming Ctrl on a Mac points at a key that exists and does something
    # else, so this is a correctness assertion, not a cosmetic one.
    if mac:
        check("macOS tooltips never say 'Ctrl+'", "Ctrl" not in combo)

    p = ImageAdjustPanelWidget(histogram_widget=ImageHistogramWidget())

    # --- controls whose shortcut must be discoverable ---------------------
    # (attribute, substrings the tooltip must contain, what the key does)
    expected = [
        ("_copy_btn", [theme.shortcut("Ctrl", "C")], "copy settings"),
        ("_paste_btn", [theme.shortcut("Ctrl", "V")], "paste settings"),
        ("_mask_paint_btn", ["P"], "arm mask Add"),
        ("_mask_erase_btn", ["X"], "arm mask Erase"),
        ("_dodge_btn", ["D"], "dodge brush"),
        ("_burn_btn", ["B"], "burn brush"),
        ("_erase_btn", ["X"], "dodge/burn eraser"),
        ("_heal_btn", ["H"], "heal brush"),
    ]
    for attr, needles, what in expected:
        btn = getattr(p, attr, None)
        if btn is None:
            check(f"{attr} exists", False, "control not found")
            continue
        tip = btn.toolTip()
        check(f"{attr} has a tooltip at all", bool(tip), what)
        for needle in needles:
            check(
                f"{attr} tooltip names its shortcut {needle!r}",
                needle in tip,
                f"tip={tip[:60]!r}",
            )

    # Reset is the one destructive-looking control in the header and had no
    # tooltip at all; it has no shortcut, so it is checked for explanation
    # rather than for a key.
    for attr in ("_auto_tone_btn",):
        btn = getattr(p, attr, None)
        check(f"{attr} explains itself on hover", btn is not None and bool(btn.toolTip()))

    # --- the cheat sheet must not contradict the tooltips ------------------
    # P latched instead of being held from the tap-to-paint work; a cheat
    # sheet still teaching "hold" would train the wrong gesture.
    from rawviewer_ui import shortcut_cheatsheet as cs

    editor_rows = [row for _title, rows in cs._EDITOR for row in rows]
    p_rows = [desc for keys, desc in editor_rows if keys.strip() == "P"]
    check("the cheat sheet still documents P", len(p_rows) == 1, f"{p_rows}")
    if p_rows:
        check(
            "the cheat sheet does not call P hold-to-paint",
            "hold" not in p_rows[0].lower(),
            f"got {p_rows[0]!r}",
        )
        check(
            "the cheat sheet describes P as a tap latch",
            "tap" in p_rows[0].lower(),
            f"got {p_rows[0]!r}",
        )

    if FAILURES:
        print(f"FAILED: {FAILURES}")
        return 1
    print("PASS t_shortcut_tooltips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
