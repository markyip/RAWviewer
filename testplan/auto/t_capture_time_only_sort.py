#!/usr/bin/env python3
"""Regression guard: the session-restore folder-sort refinement must sort by
capture time WITHOUT unpickling a full EXIF blob per file.

Real problem (from a real running log, session restore of a 259-image folder):
"[SORT] Bulk metadata fetch & sort ... took 4.517s" even though only 1 of 259
files needed a fresh probe -- the other 258 were cache hits. The cost was
get_multiple_exif() -> PersistentEXIFCache.get(), which does a pickle.loads of
a ~97-tag EXIF dict per file. That runs on a background QThreadPool worker, so
the pickle.loads burst stole the GIL from the main thread exactly while the
current image was decoding/painting, stalling first-image display on restore.

Fix: sort_files_by_capture_time(capture_time_only=True) reads only the
capture_time column (get_capture_times_for_sort -> get_capture_times_bulk, one
batched query, no per-file unpickle) and returns an EMPTY bulk_metadata so the
gallery lazily fetches full per-file EXIF only when it is actually opened
(_GalleryMetadataFetch). The fast-open refinement worker passes it; explicit
re-sorts keep the full-metadata path.

Stubs the module-level get_image_cache() with a fake recording which accessors
run (mirrors the fake-object convention used by the other t_*.py suites).
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


class FakeCache:
    def __init__(self, capture_times):
        self._ct = dict(capture_times)
        self.calls = []

    def get_capture_times_for_sort(self, paths):
        self.calls.append("get_capture_times_for_sort")
        return {p: self._ct[p] for p in paths if p in self._ct}

    def get_multiple_exif(self, paths, file_stats=None, fast_mode=True):
        self.calls.append("get_multiple_exif")
        # Full EXIF blob (what pickle.loads would have produced) -- includes
        # capture_time plus the heavy extra fields the sort does not need.
        return {
            p: {"capture_time": self._ct.get(p), "original_width": 6000, "orientation": 1}
            for p in paths
            if p in self._ct
        }


def make_viewer_stub(fake_cache):
    import main as m

    v = type("V", (), {})()
    v.sort_files_by_capture_time = m.RAWImageViewer.sort_files_by_capture_time.__get__(v)
    # No fresh probing in the test -- we only exercise cache-hit sorting.
    v._parallel_probe_capture_times = lambda *a, **k: {}
    return v


def main() -> int:
    import image_cache

    # Two files, newest capture time should sort first (newest_first=True).
    ct = {
        "/x/A.NEF": "2024:01:01 08:00:00",  # older
        "/x/B.NEF": "2024:06:01 08:00:00",  # newer
    }
    paths = ["/x/A.NEF", "/x/B.NEF"]

    fake = FakeCache(ct)
    orig_get_cache = image_cache.get_image_cache
    image_cache.get_image_cache = lambda: fake
    try:
        v = make_viewer_stub(fake)

        # --- capture_time_only=True ---
        sorted_files, bulk = v.sort_files_by_capture_time(
            paths, newest_first=True, capture_time_only=True
        )
        check(
            "capture_time_only reads the capture_time column, not the full EXIF blob",
            "get_capture_times_for_sort" in fake.calls
            and "get_multiple_exif" not in fake.calls,
            detail=str(fake.calls),
        )
        check(
            "capture_time_only returns an EMPTY bulk_metadata (gallery fetches lazily)",
            bulk == {},
        )
        check(
            "capture_time_only still sorts newest-first correctly",
            sorted_files == ["/x/B.NEF", "/x/A.NEF"],
            detail=str(sorted_files),
        )

        # --- default (full metadata) path is unchanged ---
        fake2 = FakeCache(ct)
        image_cache.get_image_cache = lambda: fake2
        v2 = make_viewer_stub(fake2)
        sorted_files2, bulk2 = v2.sort_files_by_capture_time(
            paths, newest_first=True
        )
        check(
            "default path uses get_multiple_exif (full per-file EXIF)",
            "get_multiple_exif" in fake2.calls,
            detail=str(fake2.calls),
        )
        check(
            "default path returns non-empty bulk_metadata with the heavy fields",
            bool(bulk2) and bulk2.get("/x/A.NEF", {}).get("original_width") == 6000,
        )
        check(
            "default path sorts newest-first correctly too",
            sorted_files2 == ["/x/B.NEF", "/x/A.NEF"],
            detail=str(sorted_files2),
        )
    finally:
        image_cache.get_image_cache = orig_get_cache

    # --- the fast-open refinement worker actually opts into capture_time_only ---
    import main as m

    src = inspect.getsource(m.RAWImageViewer._schedule_folder_sort_refinement)
    check(
        "fast-open refinement worker passes capture_time_only=True",
        "capture_time_only=True" in src,
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
