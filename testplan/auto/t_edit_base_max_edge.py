#!/usr/bin/env python3
"""Working Adjust edit base can be long-edge capped; export bases are not."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    extra = f"  {detail}" if detail else ""
    print(f"{'PASS' if ok else 'FAIL'}  {name}{extra}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from unified_image_processor import cap_working_edit_base

    img = np.zeros((3000, 4500, 3), dtype=np.float32)
    os.environ.pop("RAWVIEWER_EDIT_BASE_MAX_EDGE", None)
    uncapped = cap_working_edit_base(img)
    check(
        "unset env leaves the working base alone",
        uncapped is img or uncapped.shape == img.shape,
        getattr(uncapped, "shape", None),
    )

    os.environ["RAWVIEWER_EDIT_BASE_MAX_EDGE"] = "1600"
    capped = cap_working_edit_base(img)
    check(
        "1600 cap shrinks the long edge",
        capped is not None and max(capped.shape[:2]) == 1600,
        getattr(capped, "shape", None),
    )
    check(
        "aspect is preserved",
        capped is not None
        and abs(capped.shape[1] / capped.shape[0] - 4500 / 3000) < 0.02,
        getattr(capped, "shape", None),
    )
    export = cap_working_edit_base(img, use_full_resolution=True)
    check(
        "full-resolution export bases are not capped",
        export is img or export.shape == img.shape,
        getattr(export, "shape", None),
    )
    os.environ.pop("RAWVIEWER_EDIT_BASE_MAX_EDGE", None)

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
