#!/usr/bin/env python3
"""Pinned histogram + reorderable, persisted Global section order.

Touches the real QSettings the app uses, because that persistence IS the
feature under test -- the user's question was specifically whether the
arrangement survives clearing caches. The original value is saved and put
back at the end, including the case where there wasn't one.
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
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication, QScrollArea

    app = QApplication.instance() or QApplication([])

    from image_histogram import ImageHistogramWidget
    from rawviewer_ui.adjust_panel import (
        _GLOBAL_SECTION_DEFAULT_ORDER,
        _GLOBAL_SECTION_ORDER_SETTING,
        ImageAdjustPanelWidget,
        _sanitize_section_order,
    )

    settings = QSettings("RAWviewer", "RAWviewer")
    had_saved = settings.contains(_GLOBAL_SECTION_ORDER_SETTING)
    saved_before = settings.value(_GLOBAL_SECTION_ORDER_SETTING) if had_saved else None

    def restore() -> None:
        if had_saved:
            settings.setValue(_GLOBAL_SECTION_ORDER_SETTING, saved_before)
        else:
            settings.remove(_GLOBAL_SECTION_ORDER_SETTING)

    def new_panel() -> ImageAdjustPanelWidget:
        return ImageAdjustPanelWidget(histogram_widget=ImageHistogramWidget())

    def order_on_screen(p) -> list:
        """The order the LAYOUT actually holds, not the bookkeeping list.

        Read from the layout on purpose: a bug that updates _section_order but
        fails to move the widgets would pass any assertion made against the
        list itself.
        """
        rev = {v: k for k, v in p._global_sections.items()}
        out = []
        for i in range(p._global_layout.count()):
            w = p._global_layout.itemAt(i).widget()
            if w in rev:
                out.append(rev[w])
        return out

    try:
        # --- sanitizer, before any widgets --------------------------------
        canon = list(_GLOBAL_SECTION_DEFAULT_ORDER)
        check("empty saved order falls back to canonical", _sanitize_section_order([]) == canon)
        check("garbage falls back to canonical", _sanitize_section_order("nonsense") == canon)
        check("None falls back to canonical", _sanitize_section_order(None) == canon)
        check(
            "an unknown key is dropped",
            _sanitize_section_order(["detail", "no_such_section", "light"])[:2]
            == ["detail", "light"],
        )
        # The forward-compat case: a saved order written by an older version
        # that predates a section. The new section must still appear.
        partial = _sanitize_section_order(["detail", "light"])
        check("a section missing from a saved order is appended", sorted(partial) == sorted(canon),
              f"got {partial}")
        check("appending keeps the user's leading choices", partial[:2] == ["detail", "light"])
        check(
            "no duplicates survive sanitizing",
            len(_sanitize_section_order(["light", "light", "detail"])) == len(canon),
        )

        # --- pinned histogram ---------------------------------------------
        settings.remove(_GLOBAL_SECTION_ORDER_SETTING)
        p = new_panel()
        p.resize(360, 700)
        p.show()
        app.processEvents()

        def inside_scroll(w) -> bool:
            node = w
            while node is not None:
                if isinstance(node, QScrollArea):
                    return True
                node = node.parentWidget()
            return False

        check("histogram is NOT in the scroll area", not inside_scroll(p.sect_histogram))
        check("the tab bar is NOT in the scroll area", not inside_scroll(p._panel_tabs))
        check("ordinary sections still scroll", inside_scroll(p.sect_light))
        check("histogram keeps its collapse header", p.sect_histogram.header is not None)
        p.sect_histogram.set_expanded(False)
        check("pinned histogram still collapses", not p.sect_histogram.content.isVisible())
        p.sect_histogram.set_expanded(True)

        check("histogram shows on Global", p.sect_histogram.isVisible())
        p._panel_tabs.set_current(1)
        app.processEvents()
        check("histogram hides on Masks", not p.sect_histogram.isVisible())
        p._panel_tabs.set_current(0)
        app.processEvents()
        check("histogram returns on Global", p.sect_histogram.isVisible())

        # --- ordering ------------------------------------------------------
        check("fresh panel uses canonical order", order_on_screen(p) == canon, f"{order_on_screen(p)}")
        check(
            "the camera profile banner stays first",
            p._global_layout.itemAt(0).widget() is p._camera_profile_row,
        )

        # Reordering is DRAG ONLY. The header's right-click menu offers Reset
        # Order and nothing else, so the drag path below is the whole feature
        # and has to be what the test drives.
        p.resize(380, 1400)
        app.processEvents()

        def drag(key: str, onto: str) -> None:
            """Grab ``key``'s grip and drag it to the top edge of ``onto``."""
            target = p._global_sections[onto]
            y = target.mapToGlobal(target.rect().topLeft()).y() + 2
            p._begin_section_drag(key)
            p._drag_section_to(key, y)
            p._end_section_drag()

        for key, sect in p._global_sections.items():
            if sect.grip is None:
                check(f"{key} has a drag grip", False)
        check(
            "every reorderable section has a grip",
            all(s.grip is not None for s in p._global_sections.values()),
        )
        check("the pinned histogram has no grip", p.sect_histogram.grip is None)
        check("the masks section has no grip", p.sect_masks.grip is None)

        # Export shares global_layout with the sections but is NOT one, and
        # re-adding the sections appends them AFTER whatever is already there.
        # Before this was handled, the first drag teleported a 40px Export
        # button from the bottom of the page to the top and left it there.
        def export_is_last() -> bool:
            widgets = [
                p._global_layout.itemAt(i).widget()
                for i in range(p._global_layout.count())
            ]
            return widgets and widgets[-1] is p._export_btn

        check("Export sits below the sections to begin with", export_is_last())

        drag("detail", "color")
        check("Export stays below the sections after a drag", export_is_last())
        p._reset_section_order()
        check("Export stays below the sections after Reset Order", export_is_last())
        drag("detail", "color")
        moved = order_on_screen(p)
        check("dragging moves the section up", moved.index("detail") < canon.index("detail"),
              f"{moved}")
        check("dragging does not lose a section", sorted(moved) == sorted(canon))
        check("dragging does not duplicate a section", len(moved) == len(set(moved)))

        # A drag that goes nowhere must not scramble anything.
        stable = order_on_screen(p)
        p._begin_section_drag("detail")
        d = p._global_sections["detail"]
        p._drag_section_to("detail", d.mapToGlobal(d.rect().center()).y())
        p._end_section_drag()
        check("a drag onto itself changes nothing", order_on_screen(p) == stable)

        # Movement without a grab must be ignored, or a stray mouse-move
        # after release would keep reordering.
        p._section_drag_key = None
        p._drag_section_to("detail", 0)
        check("moving with no grab in progress is ignored", order_on_screen(p) == stable)

        # --- persistence ---------------------------------------------------
        arranged = order_on_screen(p)
        q = new_panel()
        check("a new panel restores the saved order", order_on_screen(q) == arranged,
              f"{order_on_screen(q)} vs {arranged}")
        check(
            "the order really is in QSettings",
            bool(settings.value(_GLOBAL_SECTION_ORDER_SETTING)),
        )

        q._reset_section_order()
        check("Reset Order restores canonical", order_on_screen(q) == canon)
        r = new_panel()
        check("reset persists too", order_on_screen(r) == canon)
    finally:
        restore()

    if FAILURES:
        print(f"FAILED: {FAILURES}")
        return 1
    print("PASS t_adjust_section_order")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
