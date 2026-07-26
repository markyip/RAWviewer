#!/usr/bin/env python3
"""Real-ESRGAN x2 export upscale -- headless.

The model itself is an optional download, so every check that needs weights
skips cleanly when they are absent. What does NOT skip is the wiring: the
export functions must accept and forward use_ai_upscale, and a missing model
must degrade to a native-resolution export rather than raising. Those are the
failure modes that would ship silently.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import onnx_realesrgan as up  # noqa: E402

MODEL_AVAILABLE = up.realesrgan_model_available()
_SKIPPED = []


def _smooth(h, w, seed=0):
    """Band-limited noise. Restoration nets behave erratically on white noise
    (see export_nafnet_onnx.py) and that instability, not the code, would be
    what a strict assertion caught."""
    import cv2

    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.random((h, w, 3), dtype=np.float32), (0, 0), 3.0)


def test_forwards_flag_without_model():
    """A missing model exports at 1x instead of raising."""
    from raw_edit_pipeline import _apply_ai_upscale

    img = _smooth(64, 64)
    if MODEL_AVAILABLE:
        # Force the not-found branch by pointing the lookup at nothing.
        os.environ["RAWVIEWER_EXPORT_REALESRGAN_ONNX"] = "0"
        try:
            out = _apply_ai_upscale(img)
        finally:
            os.environ.pop("RAWVIEWER_EXPORT_REALESRGAN_ONNX", None)
    else:
        out = _apply_ai_upscale(img)
    assert out.shape == img.shape, f"expected passthrough at 1x, got {out.shape}"
    print("  OK   missing/disabled model degrades to native resolution")


def test_export_signature_accepts_upscale():
    """Every export entry point takes use_ai_upscale -- no silent drop."""
    import inspect

    from raw_edit_pipeline import (
        _process_for_export,
        export_adjusted_image,
        export_adjusted_jpeg,
        export_adjusted_tiff16,
        export_adjusted_webp,
    )

    for fn in (
        export_adjusted_jpeg,
        export_adjusted_webp,
        export_adjusted_tiff16,
        export_adjusted_image,
        _process_for_export,
    ):
        params = inspect.signature(fn).parameters
        assert "use_ai_upscale" in params, f"{fn.__name__} drops use_ai_upscale"
    print("  OK   all export entry points accept use_ai_upscale")


def test_doubles_dimensions():
    if not MODEL_AVAILABLE:
        _SKIPPED.append("doubles dimensions")
        return
    img = _smooth(up.TILE_SIZE, up.TILE_SIZE)
    out = up.RealESRGANONNX().process(img)
    assert out.shape == (up.TILE_SIZE * 2, up.TILE_SIZE * 2, 3), out.shape
    assert out.dtype == np.float32, out.dtype
    print("  OK   single tile doubles to 2x")


def test_tiled_result_has_no_seams():
    """Multi-tile stitching must not leave a gradient spike at the boundaries."""
    if not MODEL_AVAILABLE:
        _SKIPPED.append("no seams")
        return
    h, w = up.TILE_SIZE + 60, up.TILE_SIZE * 2 + 100
    out = up.RealESRGANONNX().process(_smooth(h, w, seed=1))
    assert out.shape == (h * 2, w * 2, 3), out.shape

    gray = out.mean(axis=2)
    dx = np.abs(np.diff(gray, axis=1)).mean(axis=0)
    stride = (up.TILE_SIZE - up.TILE_OVERLAP) * up.SCALE
    seams = [stride * i for i in range(1, (w * up.SCALE) // stride)]
    median = float(np.median(dx))
    worst = max(float(dx[max(0, s - 2) : s + 3].max()) for s in seams)
    # A visible seam shows as a multiple of the typical column gradient; the
    # sin^2 partition of unity should keep this near 1.0.
    assert worst < median * 3.0, f"tile seam: {worst / median:.2f}x median column gradient"
    print(f"  OK   no tile seams ({worst / median:.2f}x median gradient)")


def test_preserves_specular_headroom():
    """Values above 1.0 must survive; the model only sees [0, 1]."""
    if not MODEL_AVAILABLE:
        _SKIPPED.append("headroom")
        return
    img = np.full((up.TILE_SIZE, up.TILE_SIZE, 3), 0.5, dtype=np.float32)
    img[10:20, 10:20] = 4.0
    out = up.RealESRGANONNX().process(img)
    assert out.max() > 2.0, f"specular headroom clipped to {out.max():.2f}"
    print(f"  OK   headroom preserved (max {out.max():.2f})")


def test_strength_blends_toward_lanczos():
    """strength=0 must equal a plain resize; 1.0 must differ from it."""
    if not MODEL_AVAILABLE:
        _SKIPPED.append("strength blend")
        return
    import cv2

    img = _smooth(up.TILE_SIZE, up.TILE_SIZE, seed=2)
    eng = up.RealESRGANONNX()
    full = eng.process(img, strength=1.0)
    none = eng.process(img, strength=0.0)

    perceptual = np.power(np.clip(img, 0.0, 1.0), 1.0 / 2.2)
    lanczos = cv2.resize(
        perceptual, (up.TILE_SIZE * 2, up.TILE_SIZE * 2), interpolation=cv2.INTER_LANCZOS4
    )
    expected = np.power(np.clip(lanczos, 0.0, 1.0), 2.2)

    assert np.allclose(none, expected, atol=1e-5), "strength=0 is not plain Lanczos"
    assert not np.allclose(full, none, atol=1e-3), "strength has no effect"
    print("  OK   strength=0 is Lanczos, strength=1 is the model")


def test_cancel_raises_between_tiles():
    if not MODEL_AVAILABLE:
        _SKIPPED.append("cancel")
        return
    from raw_edit_pipeline import ExportCancelled

    img = _smooth(up.TILE_SIZE * 2, up.TILE_SIZE * 2, seed=3)
    try:
        up.RealESRGANONNX().process(img, cancel_check=lambda: True)
    except ExportCancelled:
        print("  OK   cancel is honoured between tiles")
        return
    raise AssertionError("cancel_check was ignored")


def main() -> int:
    print(f"Real-ESRGAN x2 upscale (model {'present' if MODEL_AVAILABLE else 'ABSENT'})")
    test_export_signature_accepts_upscale()
    test_forwards_flag_without_model()
    test_doubles_dimensions()
    test_tiled_result_has_no_seams()
    test_preserves_specular_headroom()
    test_strength_blends_toward_lanczos()
    test_cancel_raises_between_tiles()
    if _SKIPPED:
        print(
            f"\n  {len(_SKIPPED)} check(s) skipped (no model): {', '.join(_SKIPPED)}\n"
            "  Fetch with scripts/models/export_realesrgan_onnx.py to run them."
        )
    print("\nPASS t_realesrgan_upscale")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
