#!/usr/bin/env python3
"""Pinned histogram + reorderable, persisted Global section order (3.1).

Touches the real QSettings the app uses, because that persistence IS the
feature under test. The original value is saved and put back at the end.
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
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])

    from image_histogram import ImageHistogramWidget
    from rawviewer_ui.widgets.adjust import (
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
        rev = {v: k for k, v in p._global_sections.items()}
        out = []
        for i in range(p._global_layout.count()):
            w = p._global_layout.itemAt(i).widget()
            if w in rev:
                out.append(rev[w])
        return out

    try:
        canon = list(_GLOBAL_SECTION_DEFAULT_ORDER)
        check("empty saved order falls back to canonical", _sanitize_section_order([]) == canon)
        check("garbage falls back to canonical", _sanitize_section_order("nonsense") == canon)
        check("None falls back to canonical", _sanitize_section_order(None) == canon)
        check(
            "an unknown key is dropped",
            _sanitize_section_order(["detail", "no_such_section", "light"])[:2]
            == ["detail", "light"],
        )
        check("anamorphic is not a 3.1 section", "anamorphic" not in canon)
        partial = _sanitize_section_order(["detail", "light"])
        check(
            "a section missing from a saved order is appended",
            sorted(partial) == sorted(canon),
            f"got {partial}",
        )
        check("appending keeps the user's leading choices", partial[:2] == ["detail", "light"])
        check(
            "no duplicates survive sanitizing",
            len(_sanitize_section_order(["light", "light", "detail"])) == len(canon),
        )

        settings.remove(_GLOBAL_SECTION_ORDER_SETTING)
        p = new_panel()
        p.resize(360, 700)
        p.show()
        app.processEvents()

        check("histogram is not in the reorderable map", "histogram" not in p._global_sections)
        check("histogram keeps its collapse header", p.sect_histogram.header is not None)
        p.sect_histogram.set_expanded(False)
        check("pinned histogram still collapses", not p.sect_histogram.content.isVisible())
        p.sect_histogram.set_expanded(True)

        check("fresh panel uses canonical order", order_on_screen(p) == canon, f"{order_on_screen(p)}")

        def export_is_last_widget() -> bool:
            widgets = [
                p._global_layout.itemAt(i).widget()
                for i in range(p._global_layout.count())
                if p._global_layout.itemAt(i).widget() is not None
            ]
            return bool(widgets) and widgets[-1] is p._export_btn

        check("Export sits below the sections to begin with", export_is_last_widget())

        p._nudge_section("detail", -1)
        check("Export stays below the sections after a move", export_is_last_widget())
        moved = order_on_screen(p)
        check(
            "Move Up changes order",
            moved != canon and sorted(moved) == sorted(canon),
            f"{moved}",
        )
        p._reset_section_order()
        check("Reset Order restores canonical", order_on_screen(p) == canon)
        check("Export stays below the sections after Reset Order", export_is_last_widget())

        p._nudge_section("detail", -1)
        arranged = order_on_screen(p)
        q = new_panel()
        check(
            "a new panel restores the saved order",
            order_on_screen(q) == arranged,
            f"{order_on_screen(q)} vs {arranged}",
        )
        check(
            "the order really is in QSettings",
            bool(settings.value(_GLOBAL_SECTION_ORDER_SETTING)),
        )
        q._reset_section_order()
        check("Reset Order persists as canonical", order_on_screen(new_panel()) == canon)
    finally:
        restore()

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
