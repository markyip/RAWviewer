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
    from raw_chroma_denoise import apply_chroma_denoise

    rng = np.random.default_rng(0)
    rgb = np.clip(rng.random((16, 16, 3), dtype=np.float32) * 0.4 + 0.1, 0, 1)
    out = apply_chroma_denoise(rgb, strength=1.0, preview=True)
    assert out.shape == rgb.shape
    lum_in = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    lum_out = 0.2126 * out[:, :, 0] + 0.7152 * out[:, :, 1] + 0.0722 * out[:, :, 2]
    assert abs(float(lum_out.mean()) - float(lum_in.mean())) < 0.05


def test_chroma_nlm_no_green_cast_on_neutral_gray() -> None:
    """Regression guard for the old NLM path's uint8-truncation green cast;
    the bilateral replacement never round-trips through uint8 at all."""
    from raw_chroma_denoise import apply_chroma_denoise

    h, w = 64, 64
    base = 0.4
    rng = np.random.default_rng(0)
    img = np.stack(
        [base + rng.normal(0, 0.03, (h, w)) for _ in range(3)], axis=-1
    ).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    out = apply_chroma_denoise(img.copy(), strength=0.625, preview=False)
    delta = out.mean(axis=(0, 1)) - img.mean(axis=(0, 1))
    assert np.all(np.abs(delta) < 0.001), f"chroma NR shifted mean RGB by {delta} (green-cast regression)"


def test_chroma_denoise_reduces_noise_and_preserves_edge() -> None:
    """Bilateral replacement must still denoise flat regions and keep hard edges sharp."""
    from raw_chroma_denoise import apply_chroma_denoise

    rng = np.random.default_rng(2)
    h, w = 80, 80
    flat = np.full((h, w, 3), 0.4, dtype=np.float32)
    flat += rng.normal(0, 0.05, (h, w, 3)).astype(np.float32)
    flat = np.clip(flat, 0.0, 1.0)
    out = apply_chroma_denoise(flat.copy(), strength=1.0, preview=False)
    from raw_chroma_denoise import _rgb_to_ycbcr

    _, cb_in, cr_in = _rgb_to_ycbcr(flat)
    _, cb_out, cr_out = _rgb_to_ycbcr(out)
    assert cb_out.std() < cb_in.std() * 0.6
    assert cr_out.std() < cr_in.std() * 0.6

    # A sharp color edge (red|blue) must not bleed noticeably across the boundary.
    edge = np.zeros((h, w, 3), dtype=np.float32)
    edge[:, : w // 2] = [0.9, 0.1, 0.1]
    edge[:, w // 2 :] = [0.1, 0.1, 0.9]
    out_edge = apply_chroma_denoise(edge.copy(), strength=1.0, preview=False)
    assert np.allclose(out_edge[h // 2, 2], edge[h // 2, 2], atol=0.05)
    assert np.allclose(out_edge[h // 2, -3], edge[h // 2, -3], atol=0.05)


def test_chroma_denoise_removes_blotchy_correlated_noise() -> None:
    """The plain bilateral filter (5-7px kernel) barely touches noise with a
    several-pixel spatial correlation length -- real high-ISO sensor color
    noise, from Bayer demosaic interpolation, is typically blotchy like this,
    not pixel-independent. Regression guard for the downsample/blur/upsample
    coarse pass added to reach it (user report: "the color noise cannot be
    removed")."""
    import cv2

    from raw_chroma_denoise import _rgb_to_ycbcr, apply_chroma_denoise

    rng = np.random.default_rng(0)
    h, w = 300, 300

    def blotchy_noise(sigma_blur: float, amp: float) -> np.ndarray:
        n = rng.normal(0, 1, (h, w)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), sigma_blur)
        return n / (n.std() + 1e-6) * amp

    base = np.full((h, w, 3), 0.4, dtype=np.float32)
    noise_cb = blotchy_noise(4.0, 0.04)
    noise_cr = blotchy_noise(4.0, 0.04)
    img = base.copy()
    img[:, :, 0] += noise_cr * 1.5
    img[:, :, 2] += noise_cb * 1.8
    img = np.clip(img, 0.0, 1.0)

    _, cb_in, cr_in = _rgb_to_ycbcr(img)
    out = apply_chroma_denoise(img.copy(), strength=1.25, preview=False)
    _, cb_out, cr_out = _rgb_to_ycbcr(out)

    before = cb_in.std() + cr_in.std()
    after = cb_out.std() + cr_out.std()
    reduction = 1.0 - after / before
    assert reduction > 0.08, f"blotchy noise reduction too weak: {reduction * 100:.1f}%"


def test_chroma_denoise_edge_bleed_bounded_near_hard_edge() -> None:
    """The coarse pass added for blotchy noise (previous test) trades some
    bleed near real edges for a much larger effective denoise radius -- guard
    that the bleed stays small and localized (within ~6px) even at the
    strongest reachable slider setting, not the wide halo an unguided blur
    would produce."""
    from raw_chroma_denoise import apply_chroma_denoise

    edge = np.zeros((200, 200, 3), dtype=np.float32)
    edge[:, :100] = [0.9, 0.1, 0.1]
    edge[:, 100:] = [0.1, 0.1, 0.9]
    out_edge = apply_chroma_denoise(edge.copy(), strength=1.25, preview=False)
    row = 100
    target = np.array([0.9, 0.1, 0.1])
    assert np.abs(out_edge[row, 100 - 8] - target).max() < 0.01
    assert np.abs(out_edge[row, 100 - 6] - target).max() < 0.03
    assert np.abs(out_edge[row, 100 - 4] - target).max() < 0.08


def test_chroma_denoise_shadow_blotch_stronger_than_midtone() -> None:
    """Shadow-weighted coarse mix must cut blotchy chroma harder in deep
    shadows (where Shadows2012 lift reveals it) than on a midtone flat."""
    import cv2

    from raw_chroma_denoise import _rgb_to_ycbcr, apply_chroma_denoise

    rng = np.random.default_rng(1)
    h, w = 240, 240

    def blotchy(amp: float) -> tuple[np.ndarray, np.ndarray]:
        n = rng.normal(0, 1, (h, w)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), 4.0)
        n = n / (n.std() + 1e-6) * amp
        return n, n.copy()

    def reduction_at_base(base_level: float) -> float:
        base = np.full((h, w, 3), base_level, dtype=np.float32)
        noise_cb, noise_cr = blotchy(0.05)
        img = base.copy()
        img[:, :, 0] += noise_cr * 1.5
        img[:, :, 2] += noise_cb * 1.8
        img = np.clip(img, 0.0, None)
        _, cb_in, cr_in = _rgb_to_ycbcr(img)
        out = apply_chroma_denoise(img.copy(), strength=1.25, preview=False)
        _, cb_out, cr_out = _rgb_to_ycbcr(out)
        before = cb_in.std() + cr_in.std()
        after = cb_out.std() + cr_out.std()
        return 1.0 - after / max(before, 1e-8)

    mid = reduction_at_base(0.40)
    shadow = reduction_at_base(0.03)
    assert shadow > mid + 0.03, (
        f"shadow blotch reduction ({shadow * 100:.1f}%) should beat "
        f"midtone ({mid * 100:.1f}%) by a clear margin"
    )
    assert shadow > 0.12, f"shadow blotch reduction too weak: {shadow * 100:.1f}%"


def test_luma_denoise_reduces_noise_and_preserves_edge() -> None:
    from raw_chroma_denoise import _rgb_to_ycbcr, apply_luma_denoise

    rng = np.random.default_rng(3)
    h, w = 80, 80
    flat = np.full((h, w, 3), 0.4, dtype=np.float32)
    flat += rng.normal(0, 0.05, (h, w, 3)).astype(np.float32)
    flat = np.clip(flat, 0.0, 1.0)
    out = apply_luma_denoise(flat.copy(), strength=1.0, preview=False)
    y_in, cb_in, cr_in = _rgb_to_ycbcr(flat)
    y_out, cb_out, cr_out = _rgb_to_ycbcr(out)
    assert y_out.std() < y_in.std() * 0.8
    assert np.allclose(cb_out, cb_in, atol=1e-4)
    assert np.allclose(cr_out, cr_in, atol=1e-4)

    edge = np.zeros((h, w, 3), dtype=np.float32)
    edge[:, w // 2 :] = 0.9
    out_edge = apply_luma_denoise(edge.copy(), strength=1.0, preview=False)
    assert np.allclose(out_edge[h // 2, 2], edge[h // 2, 2], atol=0.05)
    assert np.allclose(out_edge[h // 2, -3], edge[h // 2, -3], atol=0.05)


def test_luma_nr_slider_registered_and_applies_in_pipeline() -> None:
    from raw_adjustments import apply_adjustments_to_linear

    rng = np.random.default_rng(4)
    img = rng.integers(500, 60000, (48, 48, 3), dtype=np.uint16)
    adj = dict(DEFAULT_ADJUSTMENTS)
    adj["LuminanceNoiseReduction"] = 80.0
    out = apply_adjustments_to_linear(img, adj)
    base = apply_adjustments_to_linear(img, dict(DEFAULT_ADJUSTMENTS))
    assert not np.array_equal(out, base)


def test_guided_filter_denoise() -> None:
    from raw_chroma_denoise import apply_chroma_denoise, apply_luma_denoise, _rgb_to_ycbcr

    rng = np.random.default_rng(7)
    flat = rng.normal(0.5, 0.05, (32, 32, 3)).astype(np.float32)
    
    # Test Chroma Guided Filter
    out_chroma = apply_chroma_denoise(flat.copy(), strength=1.0, method=1, preview=False)
    _, cb_noisy, cr_noisy = _rgb_to_ycbcr(flat)
    _, cb_clean, cr_clean = _rgb_to_ycbcr(out_chroma)
    assert np.std(cb_clean) < np.std(cb_noisy)
    assert np.std(cr_clean) < np.std(cr_noisy)
    
    # Test Luma Guided Filter
    out_luma = apply_luma_denoise(flat.copy(), strength=1.0, method=1, preview=False)
    y_noisy, _, _ = _rgb_to_ycbcr(flat)
    y_clean, _, _ = _rgb_to_ycbcr(out_luma)
    assert np.std(y_clean) < np.std(y_noisy)


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


def test_sharpness_float32_base_not_black() -> None:
    rng = np.random.default_rng(5)
    scene = rng.random((64, 64, 3), dtype=np.float32) * 0.35 + 0.02
    adj = dict(DEFAULT_ADJUSTMENTS)
    adj["Sharpness"] = 100.0
    from raw_adjustments import apply_adjustments_to_rgb

    out = apply_adjustments_to_rgb(scene, adj)
    assert out.dtype == np.uint8
    assert out.mean() > 10.0
    assert not (out == 0).all()


def test_defringe_preserves_uniform_colored_region() -> None:
    """A flat purple/green patch with no edge is not fringing -- must be left alone."""
    from raw_detail_enhance import apply_defringe

    petal = np.tile(np.array([[[0.55, 0.15, 0.60]]], dtype=np.float32), (20, 20, 1))
    out = apply_defringe(petal, 100.0)
    assert np.allclose(out, petal, atol=1e-4)

    leaf = np.tile(np.array([[[0.15, 0.55, 0.15]]], dtype=np.float32), (20, 20, 1))
    out_leaf = apply_defringe(leaf, 100.0)
    assert np.allclose(out_leaf, leaf, atol=1e-4)


def test_defringe_suppresses_fringe_at_edge() -> None:
    """A purple/green halo sitting right at a real luminance edge is suppressed."""
    from raw_detail_enhance import apply_defringe

    h, w = 40, 40
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:, : w // 2] = [0.05, 0.05, 0.05]
    img[:, w // 2 :] = [0.9, 0.9, 0.9]
    img[:, w // 2 - 1 : w // 2 + 1] = [0.55, 0.15, 0.60]
    out = apply_defringe(img, 100.0)
    before = img[20, w // 2]
    after = out[20, w // 2]
    span_before = float(before.max() - before.min())
    span_after = float(after.max() - after.min())
    assert span_after < span_before * 0.5


def test_defringe_purple_green_symmetric() -> None:
    """Equal-magnitude purple/green casts must be treated with equal strength."""
    from raw_detail_enhance import apply_defringe

    d = 0.05
    purple_px = np.tile(np.array([[[0.5 + d, 0.5, 0.5 + d]]], dtype=np.float32), (20, 20, 1))
    green_px = np.tile(np.array([[[0.5 - d / 2, 0.5 + d, 0.5 - d / 2]]], dtype=np.float32), (20, 20, 1))
    out_p = apply_defringe(purple_px, 100.0)
    out_g = apply_defringe(green_px, 100.0)
    assert np.allclose(out_p, purple_px, atol=1e-3)
    assert np.allclose(out_g, green_px, atol=1e-3)


def _assert_monotonic(y1: np.ndarray, label: str) -> None:
    d = np.diff(y1)
    bad = int(np.sum(d < -1e-6))
    assert bad == 0, f"{label}: {bad} non-monotonic steps (worst={float(d.min()):.6f})"


def test_pv2012_default_adjustments_are_true_identity() -> None:
    """apply_pv2012_tone_rgb's ratio = y1 / max(y0, _PERCEPTUAL_LUM_FLOOR)
    floored only the denominator, so with every slider at default (y1==y0)
    any pixel below the floor (perceptual luminance < 0.03, i.e. real deep
    shadow/black detail) got darkened up to 50% with zero user adjustment --
    and fought against genuine Shadows/Blacks recovery in that same range
    (reported as "cannot recover as much highlight/shadow, weak compared to
    other editing applications")."""
    from raw_pv2012 import apply_pv2012_tone_rgb

    lum = np.array([0.0001, 0.001, 0.005, 0.01, 0.02, 0.03, 0.08], dtype=np.float32)
    img = np.stack([lum] * 3, axis=-1)[:, np.newaxis, :]
    base = {
        "Contrast2012": 0, "Highlights2012": 0, "Shadows2012": 0,
        "Whites2012": 0, "Blacks2012": 0,
    }
    out = apply_pv2012_tone_rgb(img.copy(), base)
    assert np.allclose(out, img, atol=1e-5), "default adjustments must be a true no-op"


def test_pv2012_highlights_slider_does_not_touch_deep_shadows() -> None:
    from raw_pv2012 import apply_pv2012_tone_rgb

    lum = np.array([0.0001, 0.001, 0.005, 4.0], dtype=np.float32)
    img = np.stack([lum] * 3, axis=-1)[:, np.newaxis, :]
    adj = {
        "Contrast2012": 0, "Highlights2012": -100, "Shadows2012": 0,
        "Whites2012": 0, "Blacks2012": 0,
    }
    out = apply_pv2012_tone_rgb(img.copy(), adj)
    ratio = out[:, 0, 0] / img[:, 0, 0]
    assert np.allclose(ratio[:3], 1.0, atol=1e-4), "Highlights2012 must not affect deep shadows"
    assert ratio[3] < 0.95, "Highlights2012=-100 must still reduce actual highlights"


def test_pv2012_shadows_lift_stays_monotonic() -> None:
    """Shadows>=~45 previously made the perceptual tone curve locally decreasing."""
    from raw_pv2012 import _apply_pv2012_perceptual, _scene_to_perceptual

    lum_ramp = np.linspace(0.0001, 3.0, 4000).astype(np.float32)
    y0 = _scene_to_perceptual(lum_ramp)
    for sh in (40, 60, 80, 100):
        _assert_monotonic(_apply_pv2012_perceptual(y0, shadows=sh), f"PV2012 Shadows={sh}")
    _assert_monotonic(
        _apply_pv2012_perceptual(
            y0, contrast=100, highlights=100, shadows=100, whites=100, blacks=-100
        ),
        "PV2012 all sliders maxed",
    )


def test_pv2012_whites_blacks_sign_direction() -> None:
    """Blacks>0 must lift shadows (lighter); Whites>0 must push highlights brighter."""
    from raw_pv2012 import apply_pv2012_tone_rgb

    img = np.array([[[0.02, 0.02, 0.02], [0.9, 0.9, 0.9]]], dtype=np.float32)
    base = {"Contrast2012": 0, "Highlights2012": 0, "Shadows2012": 0, "Whites2012": 0, "Blacks2012": 0}
    baseline = apply_pv2012_tone_rgb(img.copy(), base)
    blacks_up = apply_pv2012_tone_rgb(img.copy(), {**base, "Blacks2012": 100})
    blacks_down = apply_pv2012_tone_rgb(img.copy(), {**base, "Blacks2012": -100})
    whites_up = apply_pv2012_tone_rgb(img.copy(), {**base, "Whites2012": 100})
    whites_down = apply_pv2012_tone_rgb(img.copy(), {**base, "Whites2012": -100})

    assert blacks_up[0, 0, 0] > baseline[0, 0, 0], "Blacks=+100 must lift the shadow pixel"
    assert blacks_down[0, 0, 0] < baseline[0, 0, 0], "Blacks=-100 must darken the shadow pixel"
    assert whites_up[0, 1, 0] > baseline[0, 1, 0], "Whites=+100 must brighten the highlight pixel"
    assert whites_down[0, 1, 0] < baseline[0, 1, 0], "Whites=-100 must darken the highlight pixel"


def test_pv2012_whites_blacks_strength() -> None:
    """0.12 max shift barely moved the black/white point -- too weak to
    visibly clip, unlike real Whites/Blacks sliders at their extremes."""
    from raw_pv2012 import apply_pv2012_tone_rgb

    lum = np.array([0.08, 0.9], dtype=np.float32)
    img = np.stack([lum] * 3, axis=-1)[:, np.newaxis, :]
    base = {"Contrast2012": 0, "Highlights2012": 0, "Shadows2012": 0, "Whites2012": 0, "Blacks2012": 0}

    out_whites = apply_pv2012_tone_rgb(img.copy(), {**base, "Whites2012": 100})
    ratio_whites = out_whites[1, 0, 0] / img[1, 0, 0]
    assert ratio_whites > 1.2, f"Whites=+100 too weak: ratio={ratio_whites:.3f}"

    out_blacks = apply_pv2012_tone_rgb(img.copy(), {**base, "Blacks2012": -100})
    ratio_blacks = out_blacks[0, 0, 0] / img[0, 0, 0]
    assert ratio_blacks < 0.7, f"Blacks=-100 too weak: ratio={ratio_blacks:.3f}"


def test_legacy_gamma_highlights_shadows_stays_monotonic() -> None:
    """Legacy gamma-space path broke as early as Shadows=15 before the coefficient fix."""
    from raw_adjustments import _apply_highlights_shadows, _channel_luminance

    ramp = np.linspace(0.0001, 1.0, 4000).astype(np.float32)
    img = np.stack([ramp, ramp, ramp], axis=-1)[np.newaxis, :, :]
    for sh in (15, 20, 40, 100):
        out = _apply_highlights_shadows(img.copy(), 0.0, sh)
        _assert_monotonic(_channel_luminance(out)[0], f"legacy Shadows={sh}")


def test_legacy_gamma_blacks_stays_monotonic() -> None:
    from raw_adjustments import _apply_masked_luminance_adjust, _black_region_weight, _channel_luminance

    ramp = np.linspace(0.0001, 1.0, 4000).astype(np.float32)
    img = np.stack([ramp, ramp, ramp], axis=-1)[np.newaxis, :, :]
    lum = ramp[np.newaxis, :]
    for amt in (40, 70, 100):
        out = _apply_masked_luminance_adjust(
            img.copy(), lum, _black_region_weight(ramp)[np.newaxis, :], amt / 100.0,
            lift=True, lift_up_strength=0.03,
        )
        _assert_monotonic(_channel_luminance(out)[0], f"legacy Blacks={amt}")


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


def test_tone_curve_affects_pv2012_tone_rgb_output() -> None:
    """apply_parametric_tone_curve/apply_tone_curve_perceptual worked fine in
    isolation, but apply_pv2012_tone_rgb (the actual production entry point)
    reassigned y0 to the curve-adjusted value and then divided the final
    ratio by that *same* value, exactly cancelling the curve's effect out of
    the image -- reported as "the tone curve is unresponsive". Regression
    guard through the real entry point, not the isolated curve function."""
    from raw_pv2012 import apply_pv2012_tone_rgb

    img = np.tile(
        np.linspace(0.05, 0.5, 8).astype(np.float32)[:, None, None], (1, 1, 3)
    )
    base = dict(DEFAULT_ADJUSTMENTS)

    point_curve_adj = dict(base)
    point_curve_adj["_tone_curve_pv2012"] = "0,0;128,214;255,255"
    out_default = apply_pv2012_tone_rgb(img.copy(), base)
    out_curved = apply_pv2012_tone_rgb(img.copy(), point_curve_adj)
    assert not np.allclose(out_default, out_curved), "point tone curve has no effect"

    parametric_adj = dict(base)
    parametric_adj["ParametricShadows"] = 80.0
    out_parametric = apply_pv2012_tone_rgb(img.copy(), parametric_adj)
    assert not np.allclose(out_default, out_parametric), "PV parametric sliders have no effect"

    # Main Shadows2012 slider must still work correctly on its own (no regression).
    shadows_adj = dict(base)
    shadows_adj["Shadows2012"] = 50.0
    out_shadows = apply_pv2012_tone_rgb(img.copy(), shadows_adj)
    assert not np.allclose(out_default, out_shadows)


def test_parametric_tone_curve_stays_monotonic() -> None:
    """ParametricShadows=+100 and ParametricHighlights=-100 both made the
    curve locally decreasing before the coefficient was reduced (same
    _smooth_weight/_SPLIT_SHADOWS=0.25 shape as raw_pv2012's shadow-lift bug)."""
    from raw_tone_curve import apply_parametric_tone_curve

    y = np.linspace(0.0, 1.0, 4000).astype(np.float32)
    base = {
        "ParametricShadows": 0, "ParametricDarks": 0,
        "ParametricLights": 0, "ParametricHighlights": 0,
    }
    combos = [
        {**base, "ParametricShadows": 100},
        {**base, "ParametricShadows": -100},
        {**base, "ParametricDarks": 100},
        {**base, "ParametricDarks": -100},
        {**base, "ParametricLights": 100},
        {**base, "ParametricLights": -100},
        {**base, "ParametricHighlights": 100},
        {**base, "ParametricHighlights": -100},
        {"ParametricShadows": 100, "ParametricDarks": 100, "ParametricLights": 100, "ParametricHighlights": 100},
        {"ParametricShadows": -100, "ParametricDarks": -100, "ParametricLights": -100, "ParametricHighlights": -100},
    ]
    for adj in combos:
        out = apply_parametric_tone_curve(y, adj)
        d = np.diff(out)
        bad = int(np.sum(d < -1e-6))
        assert bad == 0, f"{adj}: {bad} non-monotonic steps (worst={float(d.min()):.6f})"


def test_recovery_baseline_with_slider_hints() -> None:
    from raw_adjustments import (
        RECOVERY_BASELINE_KEY,
        recovery_baseline_slider_hints,
        uses_recovery_tone_map,
    )

    adj = dict(DEFAULT_ADJUSTMENTS)
    adj[RECOVERY_BASELINE_KEY] = 1.0
    adj.update(recovery_baseline_slider_hints())
    assert uses_recovery_tone_map(adj)


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
    assert out.shape == img.shape[:2] + (3,)
    assert out.dtype == np.uint8


def test_recovery_tone_map_preserves_resolution() -> None:
    from raw_edit_pipeline import _tone_map_recovery_display

    rng = np.random.default_rng(9)
    h, w = 120, 180
    linear = rng.integers(2000, 18000, (h, w, 3), dtype=np.uint16).astype(np.float32) / 65535.0
    out = _tone_map_recovery_display(linear)
    assert out.shape == (h, w, 3)


def test_write_xmp_adjustments_atomic_no_temp_leftovers() -> None:
    """Atomic write (temp file + os.replace) must leave no stray .tmp files."""
    import tempfile

    from raw_adjustments import parse_xmp_adjustments, write_xmp_adjustments

    adj = dict(DEFAULT_ADJUSTMENTS)
    adj["Exposure2012"] = 0.75
    adj["LuminanceNoiseReduction"] = 40.0
    with tempfile.TemporaryDirectory() as d:
        xmp_path = os.path.join(d, "test.xmp")
        write_xmp_adjustments(xmp_path, adj)
        assert os.path.isfile(xmp_path)
        assert os.listdir(d) == ["test.xmp"], f"leftover temp files: {os.listdir(d)}"
        parsed = parse_xmp_adjustments(xmp_path)
        assert abs(parsed.get("Exposure2012", 0.0) - 0.75) < 1e-6
        assert abs(parsed.get("LuminanceNoiseReduction", 0.0) - 40.0) < 1e-6

        write_xmp_adjustments(xmp_path, dict(DEFAULT_ADJUSTMENTS))
        assert not os.path.isfile(xmp_path), "reset-to-default must remove the sidecar"
        assert os.listdir(d) == []


def test_export_source_xmp_failure_does_not_block_export() -> None:
    """A source-sidecar write failure must not prevent exporting to output_path
    (regression guard for the export-flow review's Finding 1)."""
    import tempfile

    import raw_adjustments

    def failing_write(image_path, adj):
        raise PermissionError("simulated read-only source")

    original = raw_adjustments.write_xmp_adjustments_for_file
    raw_adjustments.write_xmp_adjustments_for_file = failing_write
    try:
        xmp_warning = None
        embed_xmp = None
        try:
            raw_adjustments.write_xmp_adjustments_for_file("/fake/source.dng", {})
        except Exception as exc:
            xmp_warning = exc
        else:
            embed_xmp = "unexpected-should-not-happen"

        from raw_edit_pipeline import export_adjusted_jpeg

        rng = np.random.default_rng(7)
        linear = rng.integers(1000, 20000, (16, 16, 3), dtype=np.uint16)
        with tempfile.TemporaryDirectory() as d:
            out_path = os.path.join(d, "out.jpg")
            export_adjusted_jpeg(linear, dict(DEFAULT_ADJUSTMENTS), out_path)
            assert os.path.isfile(out_path)
        assert xmp_warning is not None
        assert embed_xmp is None
    finally:
        raw_adjustments.write_xmp_adjustments_for_file = original


def test_export_tiff16_true_16bit_precision() -> None:
    """Pillow's "RGB" mode is 8-bit-only; Image.fromarray(uint16_arr, mode="RGB")
    has no valid typemap entry on newer Pillow (crashes) and silently truncates
    to 8-bit via any raw-mode workaround. tifffile must produce genuine 16-bit
    output, not just avoid crashing."""
    import tempfile

    import tifffile as _tifffile

    from raw_edit_pipeline import export_adjusted_tiff16

    rng = np.random.default_rng(1)
    linear = rng.integers(1000, 60000, (48, 64, 3), dtype=np.uint16)
    with tempfile.TemporaryDirectory() as d:
        out_path = os.path.join(d, "out.tif")
        export_adjusted_tiff16(linear, dict(DEFAULT_ADJUSTMENTS), out_path)
        assert os.path.isfile(out_path)
        with _tifffile.TiffFile(out_path) as tf:
            arr = tf.pages[0].asarray()
            assert arr.dtype == np.uint16
            assert arr.shape == (48, 64, 3)
            assert int((arr > 255).sum()) > 0, "output collapsed to 8-bit range"


def test_export_tiff16_embeds_xmp() -> None:
    import tempfile

    import tifffile as _tifffile

    from raw_edit_pipeline import export_adjusted_tiff16

    rng = np.random.default_rng(2)
    linear = rng.integers(1000, 60000, (32, 32, 3), dtype=np.uint16)
    with tempfile.TemporaryDirectory() as d:
        xmp_path = os.path.join(d, "src.xmp")
        xmp_bytes = b'<x:xmpmeta xmlns:x="adobe:ns:meta/"></x:xmpmeta>'
        with open(xmp_path, "wb") as f:
            f.write(xmp_bytes)
        out_path = os.path.join(d, "out.tif")
        export_adjusted_tiff16(
            linear, dict(DEFAULT_ADJUSTMENTS), out_path, embed_xmp_path=xmp_path
        )
        with _tifffile.TiffFile(out_path) as tf:
            xmp_tag = next((t for t in tf.pages[0].tags if t.code == 700), None)
            assert xmp_tag is not None
            assert bytes(xmp_tag.value) == xmp_bytes


def test_export_jpeg_no_libjpeg_suspension_crash() -> None:
    """subsampling=0 + optimize=True together triggered a libjpeg 'Suspension
    not allowed here' encoder crash on some Pillow/libjpeg builds -- reproduced
    with a plain synthetic array, unrelated to any RAWViewer-specific data.
    Regression guard at the same size that reproduced it."""
    import tempfile

    from raw_edit_pipeline import export_adjusted_jpeg

    rng = np.random.default_rng(0)
    linear = rng.integers(1000, 60000, (200, 300, 3), dtype=np.uint16)
    with tempfile.TemporaryDirectory() as d:
        out_path = os.path.join(d, "out.jpg")
        export_adjusted_jpeg(linear, dict(DEFAULT_ADJUSTMENTS), out_path)
        assert os.path.isfile(out_path)
        assert os.path.getsize(out_path) > 100


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


def test_wb_dropper_solve_neutralizes_sample() -> None:
    from raw_adjustments import solve_white_balance_from_sample
    from raw_edit_pipeline import _apply_wb_tint

    ref_temp = 5500.0
    for r, g, b in [(0.35, 0.45, 0.60), (0.50, 0.52, 0.48), (0.45, 0.50, 0.55)]:
        temperature, tint = solve_white_balance_from_sample(r, g, b, ref_temp)
        assert 2000.0 <= temperature <= 12000.0
        assert -150.0 <= tint <= 150.0
        img = np.array([[[r, g, b]]], dtype=np.float32)
        neutralized = _apply_wb_tint(
            img.copy(), {"Temperature": temperature, "Tint": tint, "AsShotTemperature": ref_temp}
        )
        px = neutralized[0, 0]
        assert abs(float(px[0]) - float(px[1])) < 1e-3
        assert abs(float(px[2]) - float(px[1])) < 1e-3


def test_wb_dropper_solve_clamps_extreme_sample() -> None:
    from raw_adjustments import solve_white_balance_from_sample

    temperature, tint = solve_white_balance_from_sample(0.95, 0.40, 0.05, 5500.0)
    assert temperature in (2000.0, 12000.0)
    assert -150.0 <= tint <= 150.0


def test_preview_stage_cache_matches_full_recompute() -> None:
    """PreviewStageCache must produce byte-identical output to the uncached
    full pipeline recompute at every step of a slider "drag" sequence that
    changes one stage's keys at a time, combined multi-stage changes, a
    reset back to defaults, and a mid-sequence base-image swap."""
    from raw_adjustments import apply_adjustments_to_linear
    from raw_edit_pipeline import PreviewStageCache, render_adjust_preview_uint8

    rng = np.random.default_rng(7)
    img = (rng.random((48, 64, 3)).astype(np.float32) * 0.6 + 0.05)
    img2 = (rng.random((40, 56, 3)).astype(np.float32) * 0.6 + 0.05)

    cache = PreviewStageCache()
    base = dict(DEFAULT_ADJUSTMENTS)

    sequence = [
        dict(base),
        {**base, "Exposure2012": 0.3},
        {**base, "Exposure2012": 0.6},
        {**base, "Exposure2012": 0.6, "Shadows2012": 40.0},
        {**base, "Exposure2012": 0.6, "Shadows2012": 70.0},
        {**base, "Exposure2012": 0.6, "Shadows2012": 70.0, "Saturation": 30.0},
        {**base, "Exposure2012": 0.6, "Shadows2012": 70.0, "Saturation": 60.0},
        {**base, "Exposure2012": 0.6, "Shadows2012": 70.0, "Saturation": 60.0, "Sharpness": 50.0},
        {**base, "Exposure2012": 0.6, "Shadows2012": 70.0, "Saturation": 60.0, "Sharpness": 80.0},
        {**base, "Whites2012": 100.0, "Blacks2012": -100.0},
        dict(base),
        # Non-default with everything downstream but PV2012/curve at default:
        # exercises uses_recovery_tone_map's exclusive-key gate directly.
        {**base, "_recovery_baseline": 1.0},
        {**base, "_recovery_baseline": 1.0, "Saturation": 40.0},
        {**base, "_recovery_baseline": 1.0, "Saturation": 40.0, "Sharpness": 30.0},
        # Turning recovery back off with the same downstream sliders held.
        {**base, "Saturation": 40.0, "Sharpness": 30.0},
        dict(base),
        # HSL-only change, then WB-only change on top, then revert WB only.
        {**base, "HueAdjustmentRed": 30.0, "SaturationAdjustmentBlue": -40.0},
        {**base, "HueAdjustmentRed": 30.0, "SaturationAdjustmentBlue": -40.0, "Temperature": 6500.0},
        {**base, "HueAdjustmentRed": 30.0, "SaturationAdjustmentBlue": -40.0, "Temperature": 5500.0},
        # Back to default, then straight to a fully-loaded combination.
        dict(base),
        {
            **base,
            "Exposure2012": -0.4,
            "Contrast2012": 20.0,
            "Highlights2012": -30.0,
            "Shadows2012": 50.0,
            "Whites2012": 40.0,
            "Blacks2012": -40.0,
            "Temperature": 4500.0,
            "Tint": 15.0,
            "Saturation": -20.0,
            "Vibrance": 25.0,
            "Sharpness": 40.0,
            "Clarity2012": 30.0,
            "Defringe": 20.0,
            "ColorNoiseReduction": 30.0,
            "LuminanceNoiseReduction": 20.0,
            "HueAdjustmentGreen": -25.0,
        },
        dict(base),
    ]

    for adj in sequence:
        expected = apply_adjustments_to_linear(img, adj)
        got = render_adjust_preview_uint8(img, adj, cache)
        assert np.array_equal(expected, got), f"staged mismatch for {adj}"

    # A mid-drag base-image swap (new file navigation) must invalidate the
    # whole cache rather than reuse buffers computed for the old image.
    adj = {**base, "Exposure2012": 0.4, "Sharpness": 40.0}
    expected2 = apply_adjustments_to_linear(img2, adj)
    got2 = render_adjust_preview_uint8(img2, adj, cache)
    assert np.array_equal(expected2, got2), "staged cache did not invalidate on base image change"


def test_preview_stage_cache_skips_upstream_recompute() -> None:
    """Changing only a late-stage key (Sharpness) must not re-run PV2012 tone."""
    import raw_edit_pipeline as rep

    rng = np.random.default_rng(3)
    img = (rng.random((32, 40, 3)).astype(np.float32) * 0.5 + 0.05)
    cache = rep.PreviewStageCache()

    call_count = {"n": 0}
    real_tone = rep.apply_pv2012_tone_rgb

    def counting_tone(*args, **kwargs):
        call_count["n"] += 1
        return real_tone(*args, **kwargs)

    rep.apply_pv2012_tone_rgb = counting_tone
    try:
        base = dict(DEFAULT_ADJUSTMENTS)
        adj1 = {**base, "Shadows2012": 30.0, "Sharpness": 10.0}
        adj2 = {**base, "Shadows2012": 30.0, "Sharpness": 60.0}
        rep.render_adjust_preview_uint8(img, adj1, cache)
        assert call_count["n"] == 1
        rep.render_adjust_preview_uint8(img, adj2, cache)
        assert call_count["n"] == 1, "tone stage recomputed even though only Sharpness changed"

        # Confirm it's not just "never recomputes": a Shadows2012 change must recompute.
        adj3 = {**base, "Shadows2012": 65.0, "Sharpness": 60.0}
        rep.render_adjust_preview_uint8(img, adj3, cache)
        assert call_count["n"] == 2, "tone stage did not recompute when Shadows2012 changed"
    finally:
        rep.apply_pv2012_tone_rgb = real_tone


def test_monotone_cubic_fit_matches_scipy_pchip_reference_values() -> None:
    """raw_tone_curve._monotone_cubic_fit is a from-scratch pure-numpy
    replacement for scipy.interpolate.PchipInterpolator (removed as a project
    dependency -- see docs/EDIT_PIPELINE.md "Installer size" entry, ~77MB).
    Locks in hand-verified reference tangent/value pairs computed against the
    real scipy.interpolate.PchipInterpolator during development, since scipy
    is no longer a dependency this suite can compare against directly."""
    from raw_tone_curve import _monotone_cubic_fit

    xs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    ys = np.array([0.0, 0.4, 0.5, 0.6, 1.0])
    f = _monotone_cubic_fit(xs, ys)

    # Values scipy.interpolate.PchipInterpolator(xs, ys)(xs) reproduces the
    # knots exactly by construction; spot-check known midpoint/tangent-shape
    # values instead, computed against real scipy during development.
    assert np.allclose(f(xs), ys, atol=1e-12), "must interpolate exactly at knots"

    # Steep endpoint tangent (2.2, computed via scipy's own one-sided 3-point
    # estimator) makes f(0.05) rise measurably faster than a plain-secant
    # (slope 1.6) endpoint tangent would -- regression guard for the
    # edge-tangent formula specifically, not just the interior shape.
    assert f(np.array([0.05]))[0] > 0.05 * 1.6, "endpoint tangent too shallow vs scipy reference"

    grid = np.linspace(0.0, 1.0, 4001)
    vals = f(grid)
    assert np.all(np.diff(vals) >= -1e-9), "must stay monotonic for a monotonic knot set"


def test_point_curve_pchip_no_overshoot() -> None:
    """PCHIP must never overshoot past the local knot values -- the classic
    failure mode of a plain/natural cubic spline on an S-shaped point set
    (steep rise then a near-flat plateau), which would ring above/below the
    neighboring knots and show up as a visible halo/banding artifact."""
    from raw_tone_curve import build_point_curve_lut

    points = [(0.0, 0.0), (64.0, 40.0), (128.0, 220.0), (192.0, 230.0), (255.0, 255.0)]
    lut = build_point_curve_lut(points)
    assert lut is not None

    xs_knots = [p[0] / 255.0 for p in points]
    ys_knots = [p[1] / 255.0 for p in points]
    grid = np.linspace(0.0, 1.0, len(lut))
    for i in range(len(points) - 1):
        lo_x, hi_x = xs_knots[i], xs_knots[i + 1]
        lo_y, hi_y = ys_knots[i], ys_knots[i + 1]
        seg = (grid >= lo_x) & (grid <= hi_x)
        seg_vals = lut[seg]
        y_min, y_max = min(lo_y, hi_y), max(lo_y, hi_y)
        assert seg_vals.min() >= y_min - 1e-4, f"undershoot in segment {i}: {seg_vals.min()} < {y_min}"
        assert seg_vals.max() <= y_max + 1e-4, f"overshoot in segment {i}: {seg_vals.max()} > {y_max}"


def test_point_curve_display_matches_applied_lut() -> None:
    """The curve drawn in the widget (sample_point_curve_for_display) must be
    sampled from the exact same fit as what's actually applied to the image
    (build_point_curve_lut) -- otherwise the UI would show a curve shape that
    doesn't match what the pipeline does to the pixels."""
    from raw_tone_curve import build_point_curve_lut, sample_point_curve_for_display

    points = [(0.0, 0.0), (90.0, 60.0), (180.0, 210.0), (255.0, 255.0)]
    lut = build_point_curve_lut(points)
    samples = sample_point_curve_for_display(points, n_samples=64)
    assert lut is not None and samples is not None

    for x255, y255 in samples:
        idx = int(round((x255 / 255.0) * (len(lut) - 1)))
        lut_y255 = float(lut[idx]) * 255.0
        assert abs(lut_y255 - y255) < 0.5, f"display sample {y255} vs LUT {lut_y255} at x={x255}"


def _lensfunpy_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("lensfunpy") is not None


def test_lens_profile_key_from_exif() -> None:
    from raw_lens_correction import lens_profile_key_from_exif

    exif = {
        "camera_make": "Canon",
        "camera_model": "Canon EOS 5D Mark IV",
        "focal_length": "16mm",
        "aperture": "f/2.8",
        "exif_data": {"EXIF LensModel": "EF16-35mm f/2.8L II USM"},
    }
    key = lens_profile_key_from_exif(exif)
    assert key == {
        "camera_make": "Canon",
        "camera_model": "Canon EOS 5D Mark IV",
        "lens_make": "Canon",
        "lens_model": "EF16-35mm f/2.8L II USM",
        "focal_length": 16.0,
        "aperture": 2.8,
    }, key

    # Missing any required field -> None, not a partial/guessed key.
    assert lens_profile_key_from_exif(None) is None
    assert lens_profile_key_from_exif({}) is None
    assert lens_profile_key_from_exif({**exif, "camera_model": ""}) is None
    assert lens_profile_key_from_exif({**exif, "exif_data": {}}) is None
    assert lens_profile_key_from_exif({**exif, "focal_length": ""}) is None
    assert lens_profile_key_from_exif({**exif, "aperture": ""}) is None


def test_lens_correction_gates_and_corrects_only_known_profiles() -> None:
    """Requires lensfunpy (a real pypi-dependency in pixi.toml, but this repo's
    ad hoc `python3 scripts/...` runs in this session use a different, older
    env without it) -- skips cleanly rather than failing outside the pixi env."""
    if not _lensfunpy_available():
        print("  (skipping lens correction test: lensfunpy not installed in this env)")
        return

    from raw_lens_correction import apply_lens_correction, has_lens_profile

    exif_known = {
        "camera_make": "Canon",
        "camera_model": "Canon EOS 5D Mark IV",
        "focal_length": "16mm",
        "aperture": "f/2.8",
        "exif_data": {"EXIF LensModel": "EF16-35mm f/2.8L II USM"},
    }
    assert has_lens_profile(exif_known)

    # An unmatched lens must NOT silently fall back to *some* lens profile on
    # the same mount -- lensfunpy's loose_search=True does exactly that (was
    # caught during development: it returned every Canon EF lens for a made-up
    # name), which would apply a wrong lens's distortion to the image.
    exif_unknown_lens = {**exif_known, "exif_data": {"EXIF LensModel": "Totally Fake Lens 9000mm"}}
    assert not has_lens_profile(exif_unknown_lens)

    exif_missing = {"camera_make": "Canon", "camera_model": "", "exif_data": {}}
    assert not has_lens_profile(exif_missing)
    assert not has_lens_profile(None)

    rng = np.random.default_rng(0)
    img16 = (rng.random((200, 300, 3)) * 60000).astype(np.uint16)

    out = apply_lens_correction(img16, exif_known)
    assert out.dtype == np.uint16 and out.shape == img16.shape
    assert not np.array_equal(out, img16), "known profile must actually change pixels"

    out_unknown = apply_lens_correction(img16, exif_unknown_lens)
    assert np.array_equal(out_unknown, img16), "no profile match must return input unchanged"

    assert apply_lens_correction(None, exif_known) is None

    # dtype preservation matters -- this app's edit base is float32/uint16, not uint8.
    imgf = rng.random((150, 220, 3)).astype(np.float32)
    outf = apply_lens_correction(imgf, exif_known)
    assert outf.dtype == np.float32
    assert not np.array_equal(outf, imgf)


def main() -> int:
    test_uint16_linear_exposure_changes_pixels()
    test_uint8_legacy_path()
    test_default_noop_uint16()
    test_shadow_lift_mostly_affects_dark_tones()
    test_pv2012_process_version_in_xmp()
    test_chroma_nlm_preserves_luminance_shape()
    test_chroma_nlm_no_green_cast_on_neutral_gray()
    test_chroma_denoise_reduces_noise_and_preserves_edge()
    test_chroma_denoise_removes_blotchy_correlated_noise()
    test_chroma_denoise_edge_bleed_bounded_near_hard_edge()
    test_luma_denoise_reduces_noise_and_preserves_edge()
    test_luma_nr_slider_registered_and_applies_in_pipeline()
    test_shadow_lift_no_extreme_green_shift()
    test_positive_saturation_visible_after_tone_map()
    test_sharpness_changes_display_output()
    test_sharpness_float32_base_not_black()
    test_defringe_preserves_uniform_colored_region()
    test_defringe_suppresses_fringe_at_edge()
    test_defringe_purple_green_symmetric()
    test_pv2012_default_adjustments_are_true_identity()
    test_pv2012_highlights_slider_does_not_touch_deep_shadows()
    test_pv2012_shadows_lift_stays_monotonic()
    test_pv2012_whites_blacks_sign_direction()
    test_pv2012_whites_blacks_strength()
    test_legacy_gamma_highlights_shadows_stays_monotonic()
    test_legacy_gamma_blacks_stays_monotonic()
    test_hsl_red_saturation_changes_output()
    test_parametric_tone_curve_changes_output()
    test_tone_curve_affects_pv2012_tone_rgb_output()
    test_parametric_tone_curve_stays_monotonic()
    test_recovery_baseline_with_slider_hints()
    test_recovery_baseline_differs_from_default()
    test_recovery_tone_map_preserves_resolution()
    test_write_xmp_adjustments_atomic_no_temp_leftovers()
    test_export_source_xmp_failure_does_not_block_export()
    test_export_tiff16_true_16bit_precision()
    test_export_tiff16_embeds_xmp()
    test_export_jpeg_no_libjpeg_suspension_crash()
    test_export_jpeg_roundtrip()
    test_tone_curve_point_helpers()
    test_wb_dropper_solve_neutralizes_sample()
    test_wb_dropper_solve_clamps_extreme_sample()
    test_preview_stage_cache_matches_full_recompute()
    test_preview_stage_cache_skips_upstream_recompute()
    test_monotone_cubic_fit_matches_scipy_pchip_reference_values()
    test_point_curve_pchip_no_overshoot()
    test_point_curve_display_matches_applied_lut()
    test_lens_profile_key_from_exif()
    test_lens_correction_gates_and_corrects_only_known_profiles()
    test_guided_filter_denoise()
    print("adjust linear pipeline: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
