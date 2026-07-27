"""Editing a >8-bit raster keeps its depth.

RAW has always been edited at float32 -- decode_raw_edit_base returns
scene-linear float with ~58k distinct levels, beyond any 14-bit sensor. The
non-RAW branch called cv2.imread(path, IMREAD_COLOR), which forces 8 bits
whatever the file holds, so a 16-bit TIFF reached the editor with 256 of its
65536 levels.

That mattered most for this app's own output: HDR merge, panorama and focus
stack all export 16-bit TIFF. Merging to gain latitude and then editing at
8 bits threw the latitude away, and a shadow lift banded where the file
would not have.

8-bit files must stay uint8 -- most photos are JPEGs and float32 would cost
four times the memory for nothing.
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


_app = QApplication.instance() or QApplication([])


def main() -> int:
    import cv2

    from raw_edit_pipeline import PreviewStageCache, render_adjust_preview_uint8
    from unified_image_processor import UnifiedImageProcessor

    proc = UnifiedImageProcessor()

    with tempfile.TemporaryDirectory() as d:
        # A gradient confined to the deep shadows: the case where 8 bits band.
        g16 = np.linspace(0, 4000, 3000).astype(np.uint16)
        deep = np.stack([np.repeat(g16[None, :], 400, 0)] * 3, -1)
        tif = os.path.join(d, "HDR_20260101_000000.tif")
        cv2.imwrite(tif, cv2.cvtColor(deep, cv2.COLOR_RGB2BGR))

        base = proc.decode_raw_edit_base(tif, use_full_resolution=True)
        check("16-bit TIFF opens for editing", base is not None)
        check("and arrives as float32, not uint8", base.dtype == np.float32, str(base.dtype))
        check(
            "carrying the file's levels, not 256",
            len(np.unique(base[..., 0])) > 1000,
            f"{len(np.unique(base[..., 0]))} levels",
        )
        check("normalised to [0,1]", 0.0 <= float(base.min()) and float(base.max()) <= 1.0)

        # The point of all this: a heavy lift must not band.
        adj = {"Exposure2012": 3.0, "Shadows2012": 80.0}
        old8 = (
            cv2.cvtColor(cv2.imread(tif, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(
                np.float32
            )
            / 255.0
        )
        new_out = render_adjust_preview_uint8(base.astype(np.float32), adj, PreviewStageCache())
        old_out = render_adjust_preview_uint8(old8, adj, PreviewStageCache())
        check("the pipeline renders it", new_out is not None and old_out is not None)

        def profile(out):
            row = out[200, :, 0].astype(int)
            uniq = np.unique(row)
            steps = np.diff(uniq)
            return len(uniq), (int(steps.max()) if len(steps) else 0)

        new_levels, new_jump = profile(new_out)
        old_levels, old_jump = profile(old_out)
        check(
            "a heavy shadow lift keeps far more levels than the 8-bit read",
            new_levels > old_levels * 3,
            f"{new_levels} vs {old_levels}",
        )
        check(
            "and no longer jumps between them",
            new_jump < old_jump,
            f"largest step {new_jump} vs {old_jump} (a big step is visible banding)",
        )

        # --- 8-bit sources must not be promoted: memory, for nothing ---
        jpg = os.path.join(d, "photo.jpg")
        cv2.imwrite(jpg, np.full((300, 400, 3), (40, 120, 200), np.uint8))
        jb = proc.decode_raw_edit_base(jpg, use_full_resolution=True)
        check("an 8-bit JPEG stays uint8", jb is not None and jb.dtype == np.uint8, str(jb.dtype))

        # --- shapes IMREAD_UNCHANGED can hand back ---
        gray = os.path.join(d, "gray16.tif")
        cv2.imwrite(gray, np.repeat(g16[None, :], 200, 0))
        gb = proc.decode_raw_edit_base(gray, use_full_resolution=True)
        check(
            "16-bit grayscale becomes 3-channel float",
            gb is not None and gb.ndim == 3 and gb.shape[2] == 3 and gb.dtype == np.float32,
            f"{None if gb is None else (gb.dtype, gb.shape)}",
        )

        rgba = os.path.join(d, "alpha.png")
        cv2.imwrite(
            rgba,
            np.dstack([np.full((200, 300, 3), 90, np.uint8), np.full((200, 300), 255, np.uint8)]),
        )
        ab = proc.decode_raw_edit_base(rgba, use_full_resolution=True)
        check(
            "RGBA loses only its alpha",
            ab is not None and ab.ndim == 3 and ab.shape[2] == 3,
            f"{None if ab is None else ab.shape}",
        )

        # --- RAW was already high precision; that must not have changed ---
        raw = "/Volumes/Development/Development/Canon_Sample/020A1358.CR3"
        if os.path.isfile(raw):
            rb = proc.decode_raw_edit_base(raw, use_full_resolution=False)
            check(
                "RAW still decodes to float32",
                rb is not None and rb.dtype == np.float32,
                str(None if rb is None else rb.dtype),
            )
        else:
            print("SKIP  RAW sample not on this machine")

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
