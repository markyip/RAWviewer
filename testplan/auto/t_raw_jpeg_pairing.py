#!/usr/bin/env python3
"""RAW+JPEG same-stem pairing collapses to one cover (the RAW)."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from raw_jpeg_pairing import (
        build_collapsed_display_paths,
        pair_members_for_path,
        pairs_for_folder,
    )

    tmpd = tempfile.mkdtemp()
    raw = os.path.join(tmpd, "DSC_001.CR3")
    jpg = os.path.join(tmpd, "DSC_001.JPG")
    other = os.path.join(tmpd, "DSC_002.CR3")
    for p in (raw, jpg, other):
        open(p, "wb").write(b"x")

    paths = [raw, jpg, other]
    path_to_pair, covers = pairs_for_folder(paths)
    members = pair_members_for_path(raw, path_to_pair)
    check("pair found", members is not None and len(members) == 2)
    check("jpeg in pair", jpg in (members or []))
    collapsed = build_collapsed_display_paths(paths, path_to_pair, covers)
    check("collapsed length 2", len(collapsed) == 2, str(collapsed))
    check("cover is RAW", collapsed[0] == raw)
    check("unpaired RAW kept", other in collapsed)
    check("jpeg hidden", jpg not in collapsed)

    lonely = os.path.join(tmpd, "solo.jpg")
    open(lonely, "wb").write(b"x")
    p2, c2 = pairs_for_folder([lonely, other])
    check("jpeg without RAW is not paired", pair_members_for_path(lonely, p2) is None)

    if FAILURES:
        print(f"\n{len(FAILURES)} failed")
        return 1
    print("\nall passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
