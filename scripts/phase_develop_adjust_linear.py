#!/usr/bin/env python3
"""Smoke tests for linear adjust pipeline."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "src")

from raw_adjustments import (
    DEFAULT_ADJUSTMENTS,
    apply_adjustments_to_linear,
    apply_adjustments_to_rgb,
    is_default_adjustments,
)


def test_uint16_linear_exposure_changes_pixels() -> None:
    gray = np.full((64, 64, 3), 12000, dtype=np.uint16)
    adj = dict(DEFAULT_ADJUSTMENTS)
    adj["Exposure2012"] = 1.0
    out = apply_adjustments_to_linear(gray, adj)
    assert out.dtype == np.uint8
    assert out.shape == gray.shape
    base = apply_adjustments_to_linear(gray, dict(DEFAULT_ADJUSTMENTS))
    assert not np.array_equal(out, base)
    assert int(out.mean()) > int(base.mean())


def test_uint8_legacy_path() -> None:
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    adj = dict(DEFAULT_ADJUSTMENTS)
    adj["Exposure2012"] = 0.5
    out = apply_adjustments_to_rgb(img, adj)
    assert out.dtype == np.uint8
    assert int(out.mean()) > 128


def test_default_noop_uint16() -> None:
    img = np.random.randint(0, 60000, (16, 16, 3), dtype=np.uint16)
    out = apply_adjustments_to_linear(img, dict(DEFAULT_ADJUSTMENTS))
    assert out.dtype == np.uint8
    assert is_default_adjustments(dict(DEFAULT_ADJUSTMENTS))


def test_shadow_lift_mostly_affects_dark_tones() -> None:
    dark = np.full((32, 32, 3), 2000, dtype=np.uint16)
    bright = np.full((32, 32, 3), 50000, dtype=np.uint16)
    combo = np.zeros((32, 64, 3), dtype=np.uint16)
    combo[:, :32] = dark
    combo[:, 32:] = bright
    adj = dict(DEFAULT_ADJUSTMENTS)
    adj["Shadows2012"] = 80.0
    out = apply_adjustments_to_linear(combo, adj)
    dark_delta = int(out[:, :32].mean()) - int(
        apply_adjustments_to_linear(combo, dict(DEFAULT_ADJUSTMENTS))[:, :32].mean()
    )
    bright_delta = int(out[:, 32:].mean()) - int(
        apply_adjustments_to_linear(combo, dict(DEFAULT_ADJUSTMENTS))[:, 32:].mean()
    )
    assert dark_delta > bright_delta * 3


def test_pv2012_process_version_in_xmp() -> None:
    import tempfile
    from raw_adjustments import write_xmp_adjustments, parse_xmp_adjustments

    adj = dict(DEFAULT_ADJUSTMENTS)
    adj["Exposure2012"] = 0.5
    with tempfile.TemporaryDirectory() as tmp:
        xmp = os.path.join(tmp, "test.xmp")
        write_xmp_adjustments(xmp, adj)
        with open(xmp, encoding="utf-8") as f:
            text = f.read()
        assert "ProcessVersion" in text
        assert "11.0" in text
        parsed = parse_xmp_adjustments(xmp)
        assert abs(parsed.get("Exposure2012", 0) - 0.5) < 1e-3


def test_chroma_nlm_preserves_luminance_shape() -> None:
    from raw_chroma_denoise import apply_chroma_nlm

    rng = np.random.default_rng(0)
    rgb = np.clip(rng.random((16, 16, 3), dtype=np.float32) * 0.4 + 0.1, 0, 1)
    out = apply_chroma_nlm(rgb, strength=1.0, preview=True)
    assert out.shape == rgb.shape
    lum_in = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    lum_out = 0.2126 * out[:, :, 0] + 0.7152 * out[:, :, 1] + 0.0722 * out[:, :, 2]
    assert abs(float(lum_out.mean()) - float(lum_in.mean())) < 0.05


def test_shadow_lift_no_extreme_green_shift() -> None:
    dark = np.zeros((32, 32, 3), dtype=np.uint16)
    dark[:, :, 0] = 200
    dark[:, :, 1] = 800
    dark[:, :, 2] = 200
    adj = dict(DEFAULT_ADJUSTMENTS)
    adj["Shadows2012"] = 50.0
    out = apply_adjustments_to_linear(dark, adj)
    r_mean = max(float(out[:, :, 0].mean()), 1.0)
    g_mean = float(out[:, :, 1].mean())
    assert g_mean / r_mean < 3.5


def test_positive_saturation_visible_after_tone_map() -> None:
    """Saturation is applied in display space so +values are clearly visible."""
    img = np.zeros((32, 32, 3), dtype=np.uint16)
    img[:, :, 0] = 28000
    img[:, :, 1] = 8000
    img[:, :, 2] = 8000
    base = apply_adjustments_to_linear(img, dict(DEFAULT_ADJUSTMENTS))
    adj = dict(DEFAULT_ADJUSTMENTS)
    adj["Saturation"] = 50.0
    out = apply_adjustments_to_linear(img, adj)
    base_chroma = float(base[:, :, 0].mean() - base[:, :, 1].mean())
    out_chroma = float(out[:, :, 0].mean() - out[:, :, 1].mean())
    assert out_chroma > base_chroma * 1.15


def test_sharpness_changes_display_output() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(8000, 14000, (48, 48, 3), dtype=np.uint16)
    adj = dict(DEFAULT_ADJUSTMENTS)
    adj["Sharpness"] = 80.0
    base = apply_adjustments_to_linear(img, dict(DEFAULT_ADJUSTMENTS))
    out = apply_adjustments_to_linear(img, adj)
    assert not np.array_equal(out, base)


def test_hsl_red_saturation_changes_output() -> None:
    img = np.zeros((32, 32, 3), dtype=np.uint16)
    img[:, :, 0] = 40000
    img[:, :, 1] = 8000
    img[:, :, 2] = 8000
    adj = dict(DEFAULT_ADJUSTMENTS)
    adj["SaturationAdjustmentRed"] = 60.0
    base = apply_adjustments_to_linear(img, dict(DEFAULT_ADJUSTMENTS))
    out = apply_adjustments_to_linear(img, adj)
    assert not np.array_equal(out, base)


def test_parametric_tone_curve_changes_output() -> None:
    from raw_tone_curve import apply_parametric_tone_curve

    y = np.linspace(0.02, 0.35, 512, dtype=np.float32)
    base = apply_parametric_tone_curve(y, dict(DEFAULT_ADJUSTMENTS))
    adj = dict(DEFAULT_ADJUSTMENTS)
    adj["ParametricShadows"] = 80.0
    out = apply_parametric_tone_curve(y, adj)
    assert not np.allclose(out, base, atol=1e-5)


def test_recovery_baseline_differs_from_default() -> None:
    rng = np.random.default_rng(2)
    img = rng.integers(500, 12000, (40, 40, 3), dtype=np.uint16)
    from raw_adjustments import RECOVERY_BASELINE_KEY, uses_recovery_tone_map

    adj = dict(DEFAULT_ADJUSTMENTS)
    adj[RECOVERY_BASELINE_KEY] = 1.0
    assert uses_recovery_tone_map(adj)
    base = apply_adjustments_to_linear(img, dict(DEFAULT_ADJUSTMENTS))
    out = apply_adjustments_to_linear(img, adj)
    assert not np.array_equal(out, base)


def test_export_jpeg_roundtrip() -> None:
    import tempfile

    from raw_edit_pipeline import export_adjusted_jpeg

    rng = np.random.default_rng(3)
    linear = rng.integers(1000, 20000, (24, 24, 3), dtype=np.uint16)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        out_path = tmp.name
    try:
        export_adjusted_jpeg(linear, dict(DEFAULT_ADJUSTMENTS), out_path)
        assert os.path.isfile(out_path)
        assert os.path.getsize(out_path) > 100
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass


def test_tone_curve_point_helpers() -> None:
    from raw_tone_curve import (
        default_tone_curve_points,
        insert_tone_curve_point,
        is_identity_tone_curve_points,
        move_tone_curve_point,
        normalize_tone_curve_points,
        remove_tone_curve_point,
        serialize_tone_curve_points,
    )

    pts = default_tone_curve_points()
    assert is_identity_tone_curve_points(pts)
    assert serialize_tone_curve_points(pts) == ""
    pts = insert_tone_curve_point(pts, 128.0, 160.0)
    assert len(pts) == 3
    assert not is_identity_tone_curve_points(pts)
    serial = serialize_tone_curve_points(pts)
    assert "128,160" in serial
    pts = move_tone_curve_point(pts, 1, 130.0, 150.0)
    assert abs(pts[1][0] - 130.0) < 1.5
    pts = remove_tone_curve_point(pts, 1)
    assert len(pts) == 2
    assert is_identity_tone_curve_points(pts)
    assert serialize_tone_curve_points(pts) == ""


def main() -> int:
    test_uint16_linear_exposure_changes_pixels()
    test_uint8_legacy_path()
    test_default_noop_uint16()
    test_shadow_lift_mostly_affects_dark_tones()
    test_pv2012_process_version_in_xmp()
    test_chroma_nlm_preserves_luminance_shape()
    test_shadow_lift_no_extreme_green_shift()
    test_positive_saturation_visible_after_tone_map()
    test_sharpness_changes_display_output()
    test_hsl_red_saturation_changes_output()
    test_parametric_tone_curve_changes_output()
    test_recovery_baseline_differs_from_default()
    test_export_jpeg_roundtrip()
    test_tone_curve_point_helpers()
    print("adjust linear pipeline: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
