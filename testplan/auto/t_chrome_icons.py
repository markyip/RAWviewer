#!/usr/bin/env python3
"""Darkroom chrome icons and gallery rubber-band scroll."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "src", "rawviewer_ui"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def test_icons_render_and_differ():
    from PyQt6.QtWidgets import QApplication
    from rawviewer_ui.chrome_icons import ALIASES, icon, known

    _app = QApplication.instance() or QApplication([])
    folder = icon("fa5s.folder-open", color="#96897a", size=20)
    gallery = icon("fa5s.th", color="#96897a", size=20)
    star = icon("fa5s.star", color="#d9a441", size=13)
    outline = icon("mdi.star-outline", color="#96897a", size=13)
    check("folder icon is not empty", not folder.isNull())
    check("gallery icon is not empty", not gallery.isNull())
    check("star icon is not empty", not star.isNull())
    check("outline star is not empty", not outline.isNull())
    check("folder and gallery are different pixmaps",
          folder.pixmap(20, 20).toImage() != gallery.pixmap(20, 20).toImage())
    check("filled and outline stars differ",
          star.pixmap(13, 13).toImage() != outline.pixmap(13, 13).toImage())
    check("an unknown name is empty, not a crash", icon("fa5s.not-a-real-icon").isNull())
    missing = [name for name in ALIASES if not known(name)]
    check("every alias has a glyph", not missing, str(missing))


def test_rubber_band_eats_overshoot_before_content():
    from rawviewer_ui.gallery_scroll import apply_scroll_delta, bounce_step

    value, os_ = apply_scroll_delta(0, 0, 1000, -40, 0.0, 64)
    check("scrolling past the top leaves value at 0", value == 0, str(value))
    check("and stores positive overshoot", os_ > 0, str(os_))
    value2, os2 = apply_scroll_delta(0, 0, 1000, 20, os_, 64)
    check("scrolling back into content reduces overshoot first",
          value2 == 0 and os2 < os_, f"{os_} -> {os2}, value={value2}")
    value3, os3 = apply_scroll_delta(1000, 0, 1000, 50, 0.0, 64)
    check("scrolling past the bottom stores negative overshoot",
          value3 == 1000 and os3 < 0, f"{value3} {os3}")
    value4, os4 = apply_scroll_delta(200, 0, 1000, 10, 0.0, 64)
    check("in-range motion does not overshoot", value4 == 210 and os4 == 0)
    nxt = bounce_step(40.0)
    check("a bounce tick decays toward rest", 0 < nxt < 40, str(nxt))
    check("a tiny remainder snaps to zero", bounce_step(0.4) == 0.0)


def test_bounce_can_be_disabled():
    os.environ["RAWVIEWER_GALLERY_BOUNCE"] = "0"
    try:
        from importlib import reload
        import rawviewer_ui.gallery_scroll as gs
        reload(gs)
        check("RAWVIEWER_GALLERY_BOUNCE=0 reports no travel", gs.bounce_max_px() == 0)
        v, o = gs.apply_scroll_delta(0, 0, 100, -30, 0.0, 0)
        check("disabled bounce clamps with no overshoot", v == 0 and o == 0.0)
    finally:
        os.environ.pop("RAWVIEWER_GALLERY_BOUNCE", None)
        from importlib import reload
        import rawviewer_ui.gallery_scroll as gs
        reload(gs)


def main() -> int:
    test_icons_render_and_differ()
    test_rubber_band_eats_overshoot_before_content()
    test_bounce_can_be_disabled()
    if FAILURES:
        print(f"\nFAILED: {FAILURES}")
        return 1
    print("\nPASS t_chrome_icons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
