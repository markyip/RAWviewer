#!/usr/bin/env python3
"""UnifiedImageProcessor cache semantics: sidecar memo, stash consumption."""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841
    from unified_image_processor import UnifiedImageProcessor
    from raw_adjustments import (
        AS_SHOT_TEMP_KEY,
        DEFAULT_ADJUSTMENTS,
        write_xmp_adjustments_for_file,
    )

    os.environ["RAWVIEWER_ENABLE_EDITING"] = "1"
    os.environ["RAWVIEWER_SIDECAR_ADJUST"] = "1"

    proc = UnifiedImageProcessor()
    tmpd = tempfile.mkdtemp()
    try:
        img_path = os.path.join(tmpd, "t.CR3")
        open(img_path, "wb").write(b"stub")
        adj = dict(DEFAULT_ADJUSTMENTS)
        adj[AS_SHOT_TEMP_KEY] = 5500.0
        adj["Exposure2012"] = 0.5
        write_xmp_adjustments_for_file(img_path, adj)

        base = (np.random.RandomState(3).rand(256, 384, 3) * 255).astype(np.uint8)

        # memo hit returns the identical object
        a = proc._apply_sidecar_if_needed(img_path, base)
        b = proc._apply_sidecar_if_needed(img_path, base)
        check("sidecar memo hit is identical object", a is b)
        check("sidecar applied (differs from base)", not np.array_equal(a, base))

        # sidecar edit invalidates
        adj["Exposure2012"] = 1.5
        write_xmp_adjustments_for_file(img_path, adj)
        c = proc._apply_sidecar_if_needed(img_path, base)
        check("sidecar edit invalidates memo", c is not b)

        # different base shape (half vs full) never served from memo
        half = base[::2, ::2]
        d = proc._apply_sidecar_if_needed(img_path, half)
        check("shape-keyed memo (half != full)", d.shape == half.shape)

        # unpack stash: consumed exactly once
        class FakeUnpack:
            pass

        proc._stash_unpacked_raw(img_path, FakeUnpack())
        first = proc._take_unpacked_raw(img_path)
        second = proc._take_unpacked_raw(img_path)
        check("unpack stash consumed once", first is not None and second is None)

        # stash keyed by path
        proc._stash_unpacked_raw(img_path, FakeUnpack())
        other = proc._take_unpacked_raw(os.path.join(tmpd, "other.CR3"))
        check("unpack stash path-keyed", other is None)
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
