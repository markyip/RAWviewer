#!/usr/bin/env python3
"""Editing HE/HE* NEF through its embedded JPEG -- headless.

LibRaw cannot demosaic Nikon High Efficiency NEF, so the editor used to
refuse these files outright. They do carry a full-size embedded JPEG, and the
pipeline already edits 8-bit rasters (that is how JPEG/TIFF editing works), so
the edit base falls back to that preview instead of the editor being disabled.

The base is display-referred 8-bit, so white balance and highlight recovery
have far less headroom than a real RAW base -- but tone, colour, masks, crop
and detail behave exactly as they do on the JPEG path.

Needs real HE files; skips cleanly without them, since the point is the
LibRaw failure path and a synthetic file cannot reproduce it.
"""
import glob
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import numpy as np  # noqa: E402

SAMPLE_DIR = os.environ.get(
    "RAWVIEWER_NEF_SAMPLES", "/Volumes/Development/Development/Canon_Sample"
)


def _he_files(limit=2):
    from enhanced_raw_processor import _detect_nef_he_compression

    out = []
    for path in sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.NEF"))):
        if _detect_nef_he_compression(path) is True:
            out.append(path)
        if len(out) >= limit:
            break
    return out


def _non_he_file():
    from enhanced_raw_processor import _detect_nef_he_compression

    for path in sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.NEF"))):
        if _detect_nef_he_compression(path) is False:
            return path
    return None


def test_editor_no_longer_refuses_he_nef():
    """The gate is in main.py; assert it is gone rather than booting the app."""
    src_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "main.py"
    )
    src = open(src_path, encoding="utf-8").read()
    start = src.index("def _editing_supported_for_file")
    end = src.index("\n    def ", start + 1)
    body = src[start:end]
    assert "_nef_he_compressed" not in body, (
        "_editing_supported_for_file still refuses HE/HE* NEF"
    )
    assert "browse-only" not in src[start:end]
    print("  OK   the editor no longer refuses HE/HE* NEF")


def test_he_nef_yields_an_edit_base():
    files = _he_files()
    if not files:
        print("  SKIP no HE/HE* NEF samples available")
        return
    from unified_image_processor import UnifiedImageProcessor

    proc = UnifiedImageProcessor()
    for path in files:
        half = proc.decode_raw_edit_base(path, use_full_resolution=False)
        assert half is not None, f"{os.path.basename(path)}: no half-res edit base"
        assert half.ndim == 3 and half.shape[2] == 3
        assert half.dtype == np.uint8, (
            f"expected an 8-bit embedded base, got {half.dtype}"
        )
        assert max(half.shape[:2]) <= 2048, (
            f"half-res base not downscaled: {half.shape}"
        )

        full = proc.decode_raw_edit_base(path, use_full_resolution=True)
        assert full is not None, f"{os.path.basename(path)}: no full-res edit base"
        assert max(full.shape[:2]) >= max(half.shape[:2]), (
            "full-res base is smaller than the half-res one"
        )
    print(f"  OK   {len(files)} HE file(s) produce 8-bit edit bases at both tiers")


def test_normal_nef_still_gets_a_linear_base():
    """The fallback must not capture files LibRaw can actually demosaic."""
    path = _non_he_file()
    if path is None:
        print("  SKIP no non-HE NEF sample available")
        return
    from unified_image_processor import UnifiedImageProcessor

    base = UnifiedImageProcessor().decode_raw_edit_base(path, use_full_resolution=False)
    assert base is not None
    assert base.dtype != np.uint8, (
        f"{os.path.basename(path)} fell back to the 8-bit embedded base "
        f"({base.dtype}) instead of a scene-linear demosaic"
    )
    print(f"  OK   a normal NEF still demosaics ({base.dtype})")


def test_adjustments_actually_apply_to_the_embedded_base():
    files = _he_files(limit=1)
    if not files:
        print("  SKIP no HE/HE* NEF samples available")
        return
    from raw_adjustments import DEFAULT_ADJUSTMENTS
    from raw_edit_pipeline import process_linear_edit_buffer
    from unified_image_processor import UnifiedImageProcessor

    base = UnifiedImageProcessor().decode_raw_edit_base(files[0], use_full_resolution=False)
    before = float(np.asarray(base, np.float32).mean()) / 255.0

    adj = dict(DEFAULT_ADJUSTMENTS)
    adj["Exposure2012"] = 1.0
    out = process_linear_edit_buffer(base, adj, preview=False)
    after = float(out.mean())

    assert out.shape == base.shape, "pipeline changed the frame size"
    assert after > before * 1.5, (
        f"+1 EV barely moved the image: {before:.3f} -> {after:.3f}"
    )
    print(f"  OK   +1 EV applies to the embedded base ({before:.3f} -> {after:.3f})")


def test_export_runs_on_an_he_file():
    files = _he_files(limit=1)
    if not files:
        print("  SKIP no HE/HE* NEF samples available")
        return
    import tempfile

    from raw_adjustments import DEFAULT_ADJUSTMENTS
    from raw_edit_pipeline import export_adjusted_jpeg
    from unified_image_processor import UnifiedImageProcessor

    base = UnifiedImageProcessor().decode_raw_edit_base(files[0], use_full_resolution=True)
    adj = dict(DEFAULT_ADJUSTMENTS)
    adj["Contrast2012"] = 20.0
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "he.jpg")
        export_adjusted_jpeg(base, adj, out)
        assert os.path.getsize(out) > 20_000, "export produced a suspiciously small file"
    print("  OK   a baked export completes for an HE file")


def main() -> int:
    test_editor_no_longer_refuses_he_nef()
    test_he_nef_yields_an_edit_base()
    test_normal_nef_still_gets_a_linear_base()
    test_adjustments_actually_apply_to_the_embedded_base()
    test_export_runs_on_an_he_file()
    print("\nPASS t_he_nef_editing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
