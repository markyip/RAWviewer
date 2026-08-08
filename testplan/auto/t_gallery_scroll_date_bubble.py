#!/usr/bin/env python3
"""Capture-date bubble shown while dragging the gallery scrollbar.

Drives the REAL JustifiedGallery methods, bound to a minimal stub host rather
than to a constructed gallery. Constructing one arms a delayed build that
starts the background thumbnail loader, which then works on the synthetic
paths below and aborts the process at teardown -- so a live gallery here would
test Qt's shutdown ordering rather than this feature. The methods under test
touch only the handful of attributes the stub provides; nothing about them
needs the host to be a QWidget.
"""
from __future__ import annotations

import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))


def main() -> int:
    from PyQt6.QtCore import QRect
    from PyQt6.QtWidgets import QApplication, QScrollArea

    app = QApplication.instance() or QApplication([])

    from rawviewer_ui.gallery_view import JustifiedGallery

    class _StubGallery:
        """Only what the bubble methods actually reach for."""

        _is_scrollbar_dragging = False
        _capture_date_provider = None
        _scroll_date_bubble = None
        _scroll_area = None
        _gallery_layout_items: list = []

        # The real implementations, verbatim.
        set_capture_date_provider = JustifiedGallery.set_capture_date_provider
        _layout_index_at_scroll_y = JustifiedGallery._layout_index_at_scroll_y
        _scroll_date_text = JustifiedGallery._scroll_date_text
        _update_scroll_date_bubble = JustifiedGallery._update_scroll_date_bubble
        _hide_scroll_date_bubble = JustifiedGallery._hide_scroll_date_bubble
        _on_slider_pressed = JustifiedGallery._on_slider_pressed

        def _on_scroll_settled(self):
            pass

        def _on_slider_released(self):
            # Mirrors the real one minus the thumbnail work it also triggers.
            self._is_scrollbar_dragging = False
            self._hide_scroll_date_bubble()

    gallery = _StubGallery()
    scroll = QScrollArea()
    scroll.resize(900, 600)
    scroll.show()
    app.processEvents()
    gallery._scroll_area = scroll

    paths = [f"/nonexistent/IMG_{i:04}.CR3" for i in range(300)]
    gallery._gallery_layout_items = [
        {"rect": QRect(0, i * 200, 300, 180), "file_path": p}
        for i, p in enumerate(paths)
    ]
    dates = {
        p: datetime.datetime(2024, 1, 1) + datetime.timedelta(days=i * 3)
        for i, p in enumerate(paths)
    }

    bar = scroll.verticalScrollBar()
    bar.setRange(0, 60000)

    # --- with no provider the feature is simply absent --------------------
    gallery._is_scrollbar_dragging = True
    gallery._update_scroll_date_bubble()
    check(
        "no date provider means no bubble",
        getattr(gallery, "_scroll_date_bubble", None) is None,
    )
    check("no provider yields empty text", gallery._scroll_date_text() == "")
    gallery._is_scrollbar_dragging = False

    gallery.set_capture_date_provider(lambda p: dates.get(p))

    # --- only while dragging the thumb ------------------------------------
    # Wheel / trackpad scrolling moves a screen at a time; a bubble over the
    # photos there would be noise rather than orientation.
    #
    # Asserted through _update_scroll_date_bubble's own guard rather than by
    # calling _on_scroll: _on_scroll also drives thumbnail prefetch, which on
    # these synthetic paths starts the real load manager and takes the process
    # down at teardown. The guard is the behaviour under test either way --
    # _on_scroll's only bubble-related act is to consult it.
    gallery._is_scrollbar_dragging = False
    gallery._update_scroll_date_bubble()
    existing = getattr(gallery, "_scroll_date_bubble", None)
    check(
        "plain scrolling does not raise the bubble",
        existing is None or not existing.isVisible(),
    )

    gallery._on_slider_pressed()
    bubble = getattr(gallery, "_scroll_date_bubble", None)
    check("dragging the thumb shows the bubble", bubble is not None and bubble.isVisible())
    if bubble is None:
        return _report()

    check(
        "the bubble hovers over the viewport, not the scrolled content",
        bubble.parentWidget() is scroll.viewport(),
        f"parent={bubble.parentWidget()}",
    )
    check(
        "the bubble ignores the mouse",
        bubble.testAttribute(
            __import__("PyQt6.QtCore", fromlist=["Qt"]).Qt.WidgetAttribute.WA_TransparentForMouseEvents
        ),
    )

    # --- it tracks the drag -----------------------------------------------
    seen = []
    for value in (0, 20000, 40000, 59500):
        bar.setValue(value)
        gallery._update_scroll_date_bubble()
        seen.append((bubble.text(), bubble.y()))
    texts = [t for t, _ in seen]
    check("the date changes as you drag", len(set(texts)) > 1, f"{texts}")
    check("every position produced a date", all(texts), f"{texts}")

    ys = [y for _, y in seen]
    check("the bubble follows the thumb down the track", ys == sorted(ys), f"{ys}")
    limit = scroll.viewport().height() - bubble.height()
    check(
        "the bubble stays fully on screen at both ends",
        all(0 <= y <= limit for y in ys),
        f"ys={ys} limit={limit}",
    )

    # --- release ----------------------------------------------------------
    gallery._on_slider_released()
    check("releasing hides the bubble", not bubble.isVisible())
    gallery._update_scroll_date_bubble()
    check("it stays hidden once the drag is over", not bubble.isVisible())

    # --- an unknown path must not block the drag ---------------------------
    gallery.set_capture_date_provider(lambda p: None)
    gallery._is_scrollbar_dragging = True
    gallery._update_scroll_date_bubble()
    check("a photo with no known date hides the bubble", not bubble.isVisible())

    def _boom(_p):
        raise RuntimeError("provider exploded")

    gallery.set_capture_date_provider(_boom)
    try:
        gallery._update_scroll_date_bubble()
        check("a failing provider cannot break scrolling", True)
    except Exception as exc:  # noqa: BLE001
        check("a failing provider cannot break scrolling", False, repr(exc))
    gallery._is_scrollbar_dragging = False

    return _report()


def _report() -> int:
    print()
    failures = [n for n, ok, _ in RESULTS if not ok]
    for name, ok, detail in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if failures:
        print(f"\nFAILED: {failures}")
        return 1
    print("\nPASS t_gallery_scroll_date_bubble")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
