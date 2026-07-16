#!/usr/bin/env python3
"""Unpack stash is a small LRU so A↔B revisit does not immediately evict."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


class _FakeUnpacked:
    def __init__(self, name):
        self.name = name


def main() -> int:
    os.environ["RAWVIEWER_UNPACK_STASH_SLOTS"] = "2"
    from unified_image_processor import UnifiedImageProcessor

    # Fresh instance; slots must be OrderedDict-backed LRU.
    proc = UnifiedImageProcessor()
    a = os.path.normcase(os.path.abspath("/tmp/a.CR3"))
    b = os.path.normcase(os.path.abspath("/tmp/b.CR3"))
    c = os.path.normcase(os.path.abspath("/tmp/c.CR3"))

    ua, ub, uc = _FakeUnpacked("a"), _FakeUnpacked("b"), _FakeUnpacked("c")
    proc._stash_unpacked_raw(a, ua)
    proc._stash_unpacked_raw(b, ub)
    check("two slots retained", len(proc._unpacked_raw_slots) == 2)

    proc._stash_unpacked_raw(c, uc)
    check("evicts oldest when over max", len(proc._unpacked_raw_slots) == 2)
    check("oldest (a) evicted", a not in proc._unpacked_raw_slots)
    check("b still present", b in proc._unpacked_raw_slots)
    check("c present", c in proc._unpacked_raw_slots)

    taken = proc._take_unpacked_raw(b)
    check("take returns stashed object", taken is ub)
    check("take removes slot", b not in proc._unpacked_raw_slots)
    check("take miss is None", proc._take_unpacked_raw(a) is None)

    # Restash same key updates without growing past max
    proc._stash_unpacked_raw(c, uc)
    proc._stash_unpacked_raw(c, _FakeUnpacked("c2"))
    check("restash same key does not grow", len(proc._unpacked_raw_slots) == 1)

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
