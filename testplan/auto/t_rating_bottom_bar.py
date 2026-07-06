#!/usr/bin/env python3
"""Bottom-bar rating widget: regression guard for two crashes hit when
loading a folder / clicking a rating star --
  1. BottomRatingWidget was never imported at main.py's module scope
     (NameError on every init_ui() call).
  2. _sync_bookmark_indicator called a nonexistent _get_current_image_rating,
     and the rating widget's rating_changed signal was wired to a
     nonexistent set_current_image_rating.
"""
import os
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

    check(
        "BottomRatingWidget is importable at main module scope",
        getattr(mainmod, "BottomRatingWidget", None) is not None,
    )
    check(
        "RAWImageViewer defines _get_current_image_rating",
        hasattr(mainmod.RAWImageViewer, "_get_current_image_rating"),
    )
    check(
        "RAWImageViewer defines rate_current_image (the real setter)",
        hasattr(mainmod.RAWImageViewer, "rate_current_image"),
    )
    check(
        "RAWImageViewer does NOT rely on a nonexistent set_current_image_rating",
        not hasattr(mainmod.RAWImageViewer, "set_current_image_rating"),
    )

    class FakeCache:
        def __init__(self, exif_by_path):
            self._exif_by_path = exif_by_path

        def get_exif(self, path):
            return self._exif_by_path.get(path)

    def make_mock(*, current_file_path, exif_by_path):
        m = type("M", (), {})()
        m.current_file_path = current_file_path
        m.image_cache = FakeCache(exif_by_path)
        m._get_current_image_rating = mainmod.RAWImageViewer._get_current_image_rating.__get__(m)
        return m

    m1 = make_mock(current_file_path=None, exif_by_path={})
    check("no current file -> rating 0", m1._get_current_image_rating() == 0)

    m2 = make_mock(current_file_path="/tmp/a.CR3", exif_by_path={})
    check("no cached exif -> rating 0", m2._get_current_image_rating() == 0)

    m3 = make_mock(
        current_file_path="/tmp/a.CR3",
        exif_by_path={"/tmp/a.CR3": {"rating": 4}},
    )
    check("cached exif with rating -> returns that rating", m3._get_current_image_rating() == 4)

    m4 = make_mock(
        current_file_path="/tmp/a.CR3",
        exif_by_path={"/tmp/a.CR3": {"rating": 0}},
    )
    check("cached exif with rating 0 -> returns 0, not truthy-fallback bug", m4._get_current_image_rating() == 0)

    # _sync_bookmark_indicator must call set_rating with the resolved rating
    # and must not crash (this is the exact call site that hit the NameError).
    class FakeRatingWidget:
        def __init__(self):
            self.calls = []

        def setVisible(self, v):
            pass

        def set_rating(self, r):
            self.calls.append(r)

    m5 = make_mock(
        current_file_path="/tmp/b.CR3",
        exif_by_path={"/tmp/b.CR3": {"rating": 3}},
    )
    m5.image_files = ["/tmp/b.CR3"]
    m5.bottom_rating_widget = FakeRatingWidget()
    m5._sync_bookmark_indicator = mainmod.RAWImageViewer._sync_bookmark_indicator.__get__(m5)
    try:
        m5._sync_bookmark_indicator()
        ok = m5.bottom_rating_widget.calls == [3]
    except Exception as e:
        ok = False
        print(f"  (exception: {e})")
    check("_sync_bookmark_indicator resolves and applies the current rating", ok)

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
