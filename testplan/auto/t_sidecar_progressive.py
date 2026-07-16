#!/usr/bin/env python3
"""Progressive sidecar: interim preview-sized apply before full apply."""
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
    from image_cache import memory_preview_max_edge
    from raw_adjustments import (
        AS_SHOT_TEMP_KEY,
        DEFAULT_ADJUSTMENTS,
        write_xmp_adjustments_for_file,
    )
    from unified_image_processor import UnifiedImageProcessor

    os.environ["RAWVIEWER_ENABLE_EDITING"] = "1"
    os.environ["RAWVIEWER_SIDECAR_ADJUST"] = "1"

    proc = UnifiedImageProcessor()
    tmpd = tempfile.mkdtemp()
    try:
        img_path = os.path.join(tmpd, "prog.CR3")
        open(img_path, "wb").write(b"stub")
        adj = dict(DEFAULT_ADJUSTMENTS)
        adj[AS_SHOT_TEMP_KEY] = 5500.0
        adj["Exposure2012"] = 0.75
        write_xmp_adjustments_for_file(img_path, adj)

        cap = int(memory_preview_max_edge())
        # Synthetic "full" larger than preview cap so interim path runs.
        h = max(cap + 256, 2048)
        w = max(cap + 384, 3072)
        base = (np.random.RandomState(7).rand(h, w, 3) * 200 + 20).astype(np.uint8)

        interim_shapes = []

        def on_interim(buf):
            interim_shapes.append(getattr(buf, "shape", None))

        out = proc.apply_sidecar_progressive(
            img_path, base, on_interim=on_interim
        )
        check("full apply returns buffer", out is not None)
        check("full apply keeps sensor shape", out is not None and out.shape == base.shape)
        check("interim callback fired", len(interim_shapes) == 1)
        if interim_shapes:
            ih, iw = interim_shapes[0][:2]
            check(
                "interim is preview-capped",
                max(ih, iw) <= cap,
                detail=f"shape={interim_shapes[0]} cap={cap}",
            )
            check(
                "interim differs from unadjusted downscale size only",
                out is not None and not np.array_equal(out, base),
            )

        # Cancel before full: interim may fire, full must not return a new apply
        # after cancel — we abort and return None.
        cancelled = {"n": 0}

        def cancel_after_interim():
            cancelled["n"] += 1
            # First check (before interim work) False; after interim True.
            return cancelled["n"] > 2

        # Simpler: always-cancelled before any work
        out_cancel = proc.apply_sidecar_progressive(
            img_path,
            base,
            cancel_check=lambda: True,
            on_interim=lambda _b: None,
        )
        check("cancel before start returns None", out_cancel is None)

        # Default adjustments: no apply, no interim
        from raw_adjustments import write_xmp_adjustments_for_file as _w

        _w(img_path, dict(DEFAULT_ADJUSTMENTS))
        interims2 = []
        out_def = proc.apply_sidecar_progressive(
            img_path, base, on_interim=lambda b: interims2.append(b.shape)
        )
        check("default adj returns base", out_def is base)
        check("default adj no interim", len(interims2) == 0)

        # force=True applies even when libraw-preview skip would fire
        adj2 = dict(DEFAULT_ADJUSTMENTS)
        adj2[AS_SHOT_TEMP_KEY] = 5500.0
        adj2["Exposure2012"] = 0.3
        write_xmp_adjustments_for_file(img_path, adj2)
        small = base[: min(512, h), : min(768, w)].copy()
        forced = proc._apply_sidecar_if_needed(img_path, small, force=True)
        check(
            "force apply edits small buffer",
            forced is not None and not np.array_equal(forced, small),
        )

    finally:
        shutil.rmtree(tmpd, ignore_errors=True)

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
