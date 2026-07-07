#!/usr/bin/env python3
"""Gallery hover-prefetch: warms a RAW's full decode when the cursor rests
on a static tile, but never fires because of scroll-induced tile changes."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    import sys as _sys

    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(_sys.argv)  # noqa: F841
    import rawviewer_ui.gallery_view as gv

    class FakeWidget:
        def __init__(self, path):
            self.file_path = path

    class FakeViewer:
        def __init__(self, view_mode="gallery"):
            self.view_mode = view_mode
            self.calls = []

        def _maybe_queue_background_full_decode(self, path):
            self.calls.append(path)

    class Mock:
        pass

    def make_gallery(scrolling=False, view_mode="gallery"):
        m = Mock()
        m._last_scroll_event_t = 0.0 if not scrolling else __import__("time").time()
        m.parent_viewer = FakeViewer(view_mode)
        for name in ("_is_actively_scrolling", "_on_gallery_thumb_hover", "_fire_gallery_hover_prefetch"):
            setattr(m, name, getattr(gv.JustifiedGallery, name).__get__(m))
        return m

    # 1. Hover-enter while NOT scrolling sets the pending path
    m = make_gallery(scrolling=False)
    w = FakeWidget("a.CR3")
    m._on_gallery_thumb_hover(w, entered=True)
    check("hover-enter (no scroll) sets pending path", m._hover_prefetch_pending_path == "a.CR3")

    # 2. Firing the debounce for the still-pending path dispatches the prefetch
    m._fire_gallery_hover_prefetch("a.CR3")
    check("debounce fire dispatches prefetch", m.parent_viewer.calls == ["a.CR3"])

    # 3. Hover-enter WHILE scrolling must not set a pending path at all --
    # this is the core ask: a tile appearing under a stationary cursor
    # because the CONTENT scrolled must not be treated as a real hover.
    m2 = make_gallery(scrolling=True)
    w2 = FakeWidget("b.CR3")
    m2._on_gallery_thumb_hover(w2, entered=True)
    check(
        "hover-enter DURING scroll sets no pending path",
        getattr(m2, "_hover_prefetch_pending_path", None) is None,
    )

    # 4. Hover-enter then leave before the debounce fires: firing the
    # (stale) timer callback must not dispatch anything.
    m3 = make_gallery(scrolling=False)
    w3 = FakeWidget("c.CR3")
    m3._on_gallery_thumb_hover(w3, entered=True)
    m3._on_gallery_thumb_hover(w3, entered=False)
    m3._fire_gallery_hover_prefetch("c.CR3")
    check("leave-before-debounce cancels the pending prefetch", m3.parent_viewer.calls == [])

    # 5. Hover-enter, then scroll starts before the debounce fires: the
    # scroll guard at fire-time must still catch it (covers the case where
    # scrolling starts mid-hover, e.g. a quick flick right after resting).
    m4 = make_gallery(scrolling=False)
    w4 = FakeWidget("d.CR3")
    m4._on_gallery_thumb_hover(w4, entered=True)
    m4._last_scroll_event_t = __import__("time").time()  # scroll just happened
    m4._fire_gallery_hover_prefetch("d.CR3")
    check("scroll starting before debounce fires cancels the prefetch", m4.parent_viewer.calls == [])

    # 6. Hovering a second tile before the first's debounce fires supersedes
    # it -- only the tile actually under the cursor at fire-time should win.
    m5 = make_gallery(scrolling=False)
    w5a, w5b = FakeWidget("e.CR3"), FakeWidget("f.CR3")
    m5._on_gallery_thumb_hover(w5a, entered=True)
    m5._on_gallery_thumb_hover(w5a, entered=False)
    m5._on_gallery_thumb_hover(w5b, entered=True)
    m5._fire_gallery_hover_prefetch("e.CR3")  # stale timer from the first hover
    check("stale timer from a superseded hover does not fire", m5.parent_viewer.calls == [])
    m5._fire_gallery_hover_prefetch("f.CR3")
    check("current hover's own timer fires", m5.parent_viewer.calls == ["f.CR3"])

    # 7. Not in gallery view mode (e.g. single view / burst view): no dispatch.
    m6 = make_gallery(scrolling=False, view_mode="single")
    w6 = FakeWidget("g.CR3")
    m6._on_gallery_thumb_hover(w6, entered=True)
    m6._fire_gallery_hover_prefetch("g.CR3")
    check("no dispatch outside gallery view mode", m6.parent_viewer.calls == [])

    # 8. No file_path on the widget: silently ignored, no crash.
    m7 = make_gallery(scrolling=False)
    w7 = FakeWidget(None)
    try:
        m7._on_gallery_thumb_hover(w7, entered=True)
        ok = True
    except Exception:
        ok = False
    check("widget with no file_path does not raise", ok)

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
