#!/usr/bin/env python3
"""Gallery hover: the tile under the cursor grows and comes forward.

The visual part is easy; what breaks it is everything around it, so that is
what these cover:

  1. Two debounces off one hover event. The decode stays at 220ms (what keeps
     a mouse sweep from queueing a dozen RAW decodes); the expand runs on its
     own shorter timer, because a visual answer at 220ms reads as lag.
  2. Widget pooling -- a recycled tile must not inherit the previous photo's
     expansion, or the next photo shows oversized and pinned above its
     neighbours.
  3. The layout passes reset every visible tile's geometry; the expanded one
     has to survive them, and must collapse rather than slide around when the
     grid scrolls under the cursor.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "src", "rawviewer_ui"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def _gallery():
    """A JustifiedGallery with just enough wired up to exercise hover."""
    from rawviewer_ui.gallery_view import JustifiedGallery

    g = JustifiedGallery.__new__(JustifiedGallery)
    from PyQt6.QtWidgets import QWidget

    QWidget.__init__(g)
    g.resize(800, 600)
    g._last_scroll_event_t = 0.0
    g._hover_expand_widget = None
    g._hover_expand_path = None
    g._hover_expand_base_rect = None
    g._hover_expand_anim = None
    g._hover_collapse_anim = None
    g._hover_expand_pending_path = None
    g._hover_prefetch_pending_path = None
    return g


def _tile(gallery, path, rect):
    from PyQt6.QtCore import QRect
    from rawviewer_ui.widgets import ThumbnailLabel

    w = ThumbnailLabel(gallery)
    w.file_path = path
    w.setGeometry(QRect(*rect))
    w.setFixedSize(QRect(*rect).size())
    return w


def test_hover_grows_the_tile_and_raises_it():
    from PyQt6.QtCore import QRect
    from PyQt6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([])
    g = _gallery()
    w = _tile(g, "/x/a.CR3", (100, 100, 200, 150))
    before = QRect(w.geometry())

    g._hover_expand_pending_path = "/x/a.CR3"
    g._fire_hover_expand("/x/a.CR3", w)
    anim = g._hover_expand_anim
    check("hovering starts a geometry animation", anim is not None)
    target = anim.endValue()
    check("the tile grows", target.width() > before.width() and target.height() > before.height(),
          f"{before.width()}x{before.height()} -> {target.width()}x{target.height()}")
    check("it grows about the centre", abs(target.center().x() - before.center().x()) <= 1
          and abs(target.center().y() - before.center().y()) <= 1)
    check("the grow is a nudge, not a lightbox",
          1.0 < target.width() / before.width() < 1.25,
          f"{target.width() / before.width():.3f}x")
    check("the fixed-size pin is released so the animation can run",
          w.maximumWidth() > before.width())


def test_expansion_is_clamped_to_the_viewport():
    from PyQt6.QtCore import QRect
    from PyQt6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([])
    g = _gallery()
    # A tile hard against the top-left corner.
    grown = g._expanded_rect(QRect(0, 0, 200, 150))
    check("an edge tile does not grow out of view",
          grown.left() >= 0 and grown.top() >= 0, str(grown))
    grown_r = g._expanded_rect(QRect(600, 450, 200, 150))
    check("nor off the bottom-right",
          grown_r.right() <= g.rect().right() and grown_r.bottom() <= g.rect().bottom(),
          f"{grown_r} in {g.rect()}")


def test_scrolling_never_expands():
    from PyQt6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([])
    g = _gallery()
    w = _tile(g, "/x/a.CR3", (10, 10, 200, 150))
    g._last_scroll_event_t = time.time()  # a scroll is in progress

    g._hover_expand_pending_path = "/x/a.CR3"
    g._fire_hover_expand("/x/a.CR3", w)
    check("a tile passing under a still cursor mid-scroll does not grow",
          g._hover_expand_widget is None)


def test_recycled_tile_does_not_inherit_the_expansion():
    from PyQt6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([])
    g = _gallery()
    w = _tile(g, "/x/a.CR3", (10, 10, 200, 150))
    g._hover_expand_pending_path = "/x/a.CR3"
    g._fire_hover_expand("/x/a.CR3", w)
    check("tile is expanded", g._hover_expand_widget is w)

    # The pool hands this widget to a different photo.
    g._clear_widget_thumbnail(w)
    check("recycling clears the expansion", g._hover_expand_widget is None)
    check("and restores the tile's own size",
          w.geometry().width() == 200 and w.geometry().height() == 150,
          str(w.geometry()))
    check("and re-pins it for the layout passes", w.maximumWidth() == 200)


def test_a_stale_tile_is_not_expanded():
    """The debounce fires later; by then the pool may have reassigned the tile."""
    from PyQt6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([])
    g = _gallery()
    w = _tile(g, "/x/a.CR3", (10, 10, 200, 150))
    g._hover_expand_pending_path = "/x/a.CR3"
    w.file_path = "/x/somethingelse.CR3"  # recycled during the debounce window
    g._fire_hover_expand("/x/a.CR3", w)
    check("a recycled tile is not expanded for the photo that was hovered",
          g._hover_expand_widget is None)


def test_layout_pass_keeps_the_tile_expanded():
    from PyQt6.QtCore import QRect
    from PyQt6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([])
    g = _gallery()
    w = _tile(g, "/x/a.CR3", (10, 10, 200, 150))
    g._hover_expand_pending_path = "/x/a.CR3"
    g._fire_hover_expand("/x/a.CR3", w)
    g._hover_expand_anim.stop()  # pretend the animation settled

    layout_rect = QRect(10, 10, 200, 150)
    handled = g._sync_hover_expanded_geometry(w, layout_rect)
    check("the layout pass defers to the expansion", handled is True)
    check("the tile stays grown", w.geometry().width() > layout_rect.width(),
          str(w.geometry()))

    # A tile that is NOT expanded must be left entirely to the caller.
    other = _tile(g, "/x/b.CR3", (300, 10, 200, 150))
    check("other tiles are untouched by the hover logic",
          g._sync_hover_expanded_geometry(other, layout_rect) is False)


def test_layout_pass_collapses_when_scrolling_starts():
    from PyQt6.QtCore import QRect
    from PyQt6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([])
    g = _gallery()
    w = _tile(g, "/x/a.CR3", (10, 10, 200, 150))
    g._hover_expand_pending_path = "/x/a.CR3"
    g._fire_hover_expand("/x/a.CR3", w)

    g._last_scroll_event_t = time.time()
    moved = QRect(10, 60, 200, 150)  # the grid scrolled
    g._sync_hover_expanded_geometry(w, moved)
    check("scrolling collapses the expansion", g._hover_expand_widget is None)
    check("and the tile lands exactly on its layout rect", w.geometry() == moved,
          str(w.geometry()))


def test_leaving_collapses_only_its_own_tile():
    from PyQt6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([])
    g = _gallery()
    w = _tile(g, "/x/a.CR3", (10, 10, 200, 150))
    g._hover_expand_pending_path = "/x/a.CR3"
    g._fire_hover_expand("/x/a.CR3", w)

    # A late leave event from a tile the cursor already left must not cancel
    # the expansion that a newer hover just started.
    g._collapse_hover_expand(path="/x/older.CR3")
    check("a leave for another tile is ignored", g._hover_expand_widget is w)
    g._collapse_hover_expand(path="/x/a.CR3", immediate=True)
    check("a leave for its own tile collapses it", g._hover_expand_widget is None)


def test_hover_schedules_both_timers_with_different_delays():
    from PyQt6.QtWidgets import QApplication
    from rawviewer_ui import gallery_view

    _app = QApplication.instance() or QApplication([])
    g = _gallery()
    w = _tile(g, "/x/a.CR3", (10, 10, 200, 150))

    scheduled = []
    real_single_shot = gallery_view.QTimer.singleShot
    gallery_view.QTimer.singleShot = staticmethod(
        lambda ms, fn: scheduled.append(ms)
    )
    try:
        gallery_view.JustifiedGallery._on_gallery_thumb_hover(g, w, entered=True)
    finally:
        gallery_view.QTimer.singleShot = real_single_shot

    check("one hover schedules both the expand and the decode",
          len(scheduled) == 2, str(scheduled))
    check("the expand answers sooner than the decode",
          len(scheduled) == 2 and scheduled[0] < scheduled[1], str(scheduled))
    check("the decode debounce is unchanged at 220ms",
          220 in scheduled, str(scheduled))
    check("the expand is quick enough to feel immediate",
          len(scheduled) == 2 and scheduled[0] <= 130, str(scheduled))


def test_hover_expand_can_be_switched_off():
    from PyQt6.QtWidgets import QApplication
    from rawviewer_ui import gallery_view

    _app = QApplication.instance() or QApplication([])
    g = _gallery()
    w = _tile(g, "/x/a.CR3", (10, 10, 200, 150))
    os.environ["RAWVIEWER_GALLERY_HOVER_EXPAND"] = "0"
    scheduled = []
    real_single_shot = gallery_view.QTimer.singleShot
    gallery_view.QTimer.singleShot = staticmethod(lambda ms, fn: scheduled.append(ms))
    try:
        gallery_view.JustifiedGallery._on_gallery_thumb_hover(g, w, entered=True)
    finally:
        gallery_view.QTimer.singleShot = real_single_shot
        os.environ.pop("RAWVIEWER_GALLERY_HOVER_EXPAND", None)
    check("with the expand off, only the decode is scheduled",
          scheduled == [220], str(scheduled))


def main() -> int:
    test_hover_grows_the_tile_and_raises_it()
    test_expansion_is_clamped_to_the_viewport()
    test_scrolling_never_expands()
    test_recycled_tile_does_not_inherit_the_expansion()
    test_a_stale_tile_is_not_expanded()
    test_layout_pass_keeps_the_tile_expanded()
    test_layout_pass_collapses_when_scrolling_starts()
    test_leaving_collapses_only_its_own_tile()
    test_hover_schedules_both_timers_with_different_delays()
    test_hover_expand_can_be_switched_off()
    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
