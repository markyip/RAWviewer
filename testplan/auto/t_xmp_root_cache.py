"""Regression: the shared XMP root cache must never serve stale data.

``load_adjustments_for_file`` calls ~11 parse helpers that each used to
re-parse the whole sidecar; they now share ``_cached_xmp_root``.  The cache is
keyed on (mtime_ns, size), so a save must invalidate it -- including a save
that lands immediately after a read, which is the normal edit loop.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import raw_adjustments as ra


def _sidecar(tmp, exposure):
    path = os.path.join(tmp, "shot.CR3")
    ra.write_xmp_adjustments_for_file(path, {"Exposure2012": exposure})
    return path


def test_write_invalidates_cache():
    with tempfile.TemporaryDirectory() as tmp:
        path = _sidecar(tmp, 0.25)
        first = ra.load_adjustments_for_file(path)
        assert abs(first.get("Exposure2012", 0.0) - 0.25) < 1e-6, first

        # Tight write-then-read loop: the cache must notice every save.
        for value in (0.5, -1.0, 1.75, 0.0):
            ra.write_xmp_adjustments_for_file(path, {"Exposure2012": value})
            got = ra.load_adjustments_for_file(path).get("Exposure2012", 0.0)
            assert abs(got - value) < 1e-6, f"stale read: want {value}, got {got}"
    print("write invalidates cache: OK")


def test_cache_is_bounded():
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i in range(ra._XMP_ROOT_CACHE_MAX + 5):
            p = os.path.join(tmp, f"shot{i}.CR3")
            ra.write_xmp_adjustments_for_file(p, {"Exposure2012": float(i) / 10.0})
            ra.load_adjustments_for_file(p)
            paths.append(p)
        assert len(ra._XMP_ROOT_CACHE) <= ra._XMP_ROOT_CACHE_MAX, len(ra._XMP_ROOT_CACHE)

        # Evicted entries must still read correctly, just via a fresh parse.
        for i, p in enumerate(paths):
            got = ra.load_adjustments_for_file(p).get("Exposure2012", 0.0)
            assert abs(got - float(i) / 10.0) < 1e-6, (p, got)
    print("cache bounded and evictions re-read: OK")


def test_missing_and_malformed_return_none():
    with tempfile.TemporaryDirectory() as tmp:
        assert ra._cached_xmp_root(os.path.join(tmp, "nope.xmp")) is None
        bad = os.path.join(tmp, "bad.xmp")
        with open(bad, "w") as fh:
            fh.write("<not-xml")
        assert ra._cached_xmp_root(bad) is None
        # A malformed file must not poison the cache for a later valid write.
        with open(bad, "w") as fh:
            fh.write('<?xml version="1.0"?><x:xmpmeta xmlns:x="adobe:ns:meta/"/>')
        assert ra._cached_xmp_root(bad) is not None
    print("missing/malformed handled: OK")


if __name__ == "__main__":
    test_write_invalidates_cache()
    test_cache_is_bounded()
    test_missing_and_malformed_return_none()
    print("PASS")
