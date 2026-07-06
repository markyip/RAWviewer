#!/usr/bin/env python3
"""Regression guard: clicking the search button must actually animate the
search bar open/close, not snap instantly.

Root cause (two-part): (1) _on_search_bottom_clicked -- the only real
trigger for the search bar -- called _set_search_panel_expanded(...,
animate=False) in both branches, even though the full QPropertyAnimation
infrastructure exists and works; the animation code was simply never
wired to the one place a user actually triggers it. (2) even after fixing
that, two secondary calls (_set_gallery_search_input_visible's
animate=False default, and a redundant _ensure_gallery_search_collapsed()
call on the collapse path) immediately re-snapped the container to its
target width in the same call stack, cancelling the just-started
animation before a single frame could render -- which is also what
produced the reported "flash to the right" of everything after the
search bar in the row (an instant width jump in one frame instead of a
smooth slide).
"""
import inspect
import os
import re
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

    src = inspect.getsource(mainmod.RAWImageViewer._on_search_bottom_clicked)

    check(
        "collapse path animates the panel closed",
        "_set_search_panel_expanded(False, animate=True)" in src,
    )
    check(
        "expand path animates the panel open",
        "_set_search_panel_expanded(True, animate=True)" in src,
    )
    check(
        "expand path does not re-snap the width via an unanimated input-visible call",
        "_set_gallery_search_input_visible()" not in src,
    )
    check(
        "collapse path no longer double-calls the unanimated collapse helper",
        not re.search(
            r"_set_search_panel_expanded\(False, animate=True\)\s*\n\s*self\._ensure_gallery_search_collapsed\(\)",
            src,
        ),
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
