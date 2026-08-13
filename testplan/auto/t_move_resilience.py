#!/usr/bin/env python3
"""Moving a folder must not throw away what the app already knows about it.

Both caches are keyed by path, so renaming or moving a folder outside the app
(Finder, Explorer) used to orphan everything in it:

  * the semantic index lost every photo's metadata and embedding, permanently
    -- the search lookup missed by path and fabricated blank rows, while
    _needs_reindex consulted the file signature, correctly said "already
    indexed", and so never repaired anything;
  * every thumbnail, grid tile and preview missed in both tiers and the whole
    folder was decoded again from the RAW.

Both now recover the entry from the identity a move preserves and re-key it.
The third test is the one that keeps the fix affordable: recovery must not add
a query per tile to the gallery scroll path.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))

import cv2  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def _make_folder(root: str, name: str, n: int = 3):
    folder = os.path.join(root, name)
    os.makedirs(folder, exist_ok=True)
    paths = []
    for i in range(n):
        p = os.path.join(folder, f"img{i}.jpg")
        cv2.imwrite(p, (np.random.default_rng(i).random((64, 64, 3)) * 255).astype(np.uint8))
        paths.append(p)
    return folder, paths


def test_semantic_index_survives_a_folder_move():
    from semantic_search import SemanticImageIndex

    with tempfile.TemporaryDirectory() as root:
        folder, paths = _make_folder(root, "before")
        index = SemanticImageIndex(db_path=os.path.join(root, "sem.db"))
        for p in paths:
            index._upsert_metadata(
                index._canonical_path(p), os.stat(p),
                {"camera_model": "TestCam", "city": "Taipei", "iso": 400},
            )
        index._conn.commit()

        before = [str(r["camera_model"]) for r in index._fetch_rows_for_paths(paths)]
        check("metadata is found before the move", before == ["TestCam"] * 3, str(before))

        moved_folder = os.path.join(root, "after")
        shutil.move(folder, moved_folder)
        moved = [os.path.join(moved_folder, os.path.basename(p)) for p in paths]

        rows = index._fetch_rows_for_paths(moved)
        check("metadata survives the move",
              [str(r["camera_model"]) for r in rows] == ["TestCam"] * 3,
              str([str(r["camera_model"]) for r in rows]))
        check("...including the enrichment that costs most to rebuild",
              [str(r["city"]) for r in rows] == ["Taipei"] * 3)

        stored = {os.path.basename(os.path.dirname(str(r[0])))
                  for r in index._conn.execute("SELECT file_path FROM semantic_index")}
        check("rows are re-keyed to where the files now are", stored == {"after"}, str(stored))
        count = index._conn.execute("SELECT COUNT(*) FROM semantic_index").fetchone()[0]
        check("re-keying does not duplicate rows", count == 3, f"{count} rows")
        check("and the set still reads as indexed",
              [index._needs_reindex(p, os.stat(p)) for p in moved] == [False] * 3)


def test_semantic_repoint_handles_a_row_already_at_the_destination():
    """file_path is the PRIMARY KEY, so re-keying onto an occupied path raises.

    Tested against _repoint_row directly: the normal recovery flow only runs
    for paths that MISSED the path lookup, so a row already sitting on the
    destination cannot be reached through it. The guard is there for a row
    written between the SELECT and the UPDATE, and a swallowed IntegrityError
    would leave the recovery silently undone.
    """
    from semantic_search import SemanticImageIndex

    with tempfile.TemporaryDirectory() as root:
        folder, paths = _make_folder(root, "before", n=2)
        index = SemanticImageIndex(db_path=os.path.join(root, "sem.db"))
        old_path, dest_path = (index._canonical_path(p) for p in paths)
        index._upsert_metadata(old_path, os.stat(paths[0]), {"camera_model": "RealCam"})
        index._upsert_metadata(dest_path, os.stat(paths[1]), {"camera_model": "StaleRow"})
        index._conn.commit()

        index._repoint_row(old_path, dest_path)
        index._conn.commit()

        rows = index._conn.execute(
            "SELECT file_path, camera_model FROM semantic_index"
        ).fetchall()
        check("the recovered row takes the destination path despite the collision",
              len(rows) == 1 and str(rows[0][0]) == dest_path
              and str(rows[0][1]) == "RealCam",
              str([(os.path.basename(str(r[0])), str(r[1])) for r in rows]))


def test_semantic_move_leaves_no_duplicate_rows():
    """The ordinary move path must re-key in place, not insert alongside."""
    from semantic_search import SemanticImageIndex

    with tempfile.TemporaryDirectory() as root:
        folder, paths = _make_folder(root, "before", n=3)
        index = SemanticImageIndex(db_path=os.path.join(root, "sem.db"))
        for p in paths:
            index._upsert_metadata(index._canonical_path(p), os.stat(p), {"camera_model": "C"})
        index._conn.commit()

        moved_folder = os.path.join(root, "after")
        shutil.move(folder, moved_folder)
        moved = [os.path.join(moved_folder, os.path.basename(p)) for p in paths]
        index._fetch_rows_for_paths(moved)
        index._fetch_rows_for_paths(moved)  # twice: recovery must be idempotent

        count = index._conn.execute("SELECT COUNT(*) FROM semantic_index").fetchone()[0]
        check("repeated lookups after a move leave one row per file", count == 3,
              f"{count} rows")


def test_thumbnail_cache_survives_a_folder_move():
    from image_cache import PersistentThumbnailCache

    with tempfile.TemporaryDirectory() as root:
        folder, _ = _make_folder(root, "before", n=1)
        src = os.path.join(folder, "img0.jpg")
        cache = PersistentThumbnailCache(cache_dir=os.path.join(root, "cache"))
        cache.put(src, b"CACHED-JPEG-BYTES")
        check("entry is cached before the move", cache.get(src) == b"CACHED-JPEG-BYTES")

        moved_folder = os.path.join(root, "after")
        shutil.move(folder, moved_folder)
        moved = os.path.join(moved_folder, "img0.jpg")

        check("the cached tile is recovered after the move",
              cache.get(moved) == b"CACHED-JPEG-BYTES")
        check("has_valid agrees", cache.has_valid(moved) is True)
        row = cache._get_connection().execute(
            "SELECT file_path FROM thumbnail_cache"
        ).fetchone()
        check("the row is re-keyed, not duplicated",
              os.path.basename(os.path.dirname(str(row[0]))) == "after")


def test_a_copy_does_not_steal_the_originals_entry():
    """Same name, size and mtime, but the original still exists -- that is a
    copy, and re-keying would evict a cache entry still in use."""
    from image_cache import PersistentThumbnailCache

    with tempfile.TemporaryDirectory() as root:
        folder, _ = _make_folder(root, "orig", n=1)
        src = os.path.join(folder, "img0.jpg")
        cache = PersistentThumbnailCache(cache_dir=os.path.join(root, "cache"))
        cache.put(src, b"ORIGINAL")

        copy_folder = os.path.join(root, "copy")
        shutil.copytree(folder, copy_folder)
        copied = os.path.join(copy_folder, "img0.jpg")
        shutil.copystat(src, copied)

        check("the copy does not take the original's entry", cache.get(copied) is None)
        check("and the original still hits", cache.get(src) == b"ORIGINAL")


def test_uncached_file_pays_the_lookup_once():
    """The recovery runs on the gallery scroll path; a cold file must not
    re-query on every pass."""
    from image_cache import PersistentThumbnailCache

    with tempfile.TemporaryDirectory() as root:
        folder, paths = _make_folder(root, "cold", n=1)
        cache = PersistentThumbnailCache(cache_dir=os.path.join(root, "cache"))
        cold = paths[0]

        queries = {"n": 0}
        real_conn = cache._get_connection

        class CountingConn:
            def __init__(self, inner):
                self._inner = inner

            def execute(self, sql, *args, **kwargs):
                if "file_size = ? AND file_mtime = ?" in sql:
                    queries["n"] += 1
                return self._inner.execute(sql, *args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        cache._get_connection = lambda: CountingConn(real_conn())
        try:
            for _ in range(5):
                cache.get(cold)
        finally:
            cache._get_connection = real_conn
        check("five misses run the recovery query once", queries["n"] == 1,
              f"{queries['n']} queries")


def main() -> int:
    test_semantic_index_survives_a_folder_move()
    test_semantic_repoint_handles_a_row_already_at_the_destination()
    test_semantic_move_leaves_no_duplicate_rows()
    test_thumbnail_cache_survives_a_folder_move()
    test_a_copy_does_not_steal_the_originals_entry()
    test_uncached_file_pays_the_lookup_once()
    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
