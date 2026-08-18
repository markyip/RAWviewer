"""Continuous DOF / mask blur checks."""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
sys.path.insert(0, ROOT)

from raw_depth_blur import (  # noqa: E402
    apply_mask_blur,
    apply_variable_blur,
    sigma_map_from_depth,
    sigma_map_from_weight,
)
from raw_mask_layers import MaskLayer, MaskLayerStack, apply_mask_layers  # noqa: E402


def test_depth_sigma_zero_at_focus():
    near = np.linspace(0.0, 1.0, 50, dtype=np.float32)
    near = np.tile(near, (20, 1))
    sigma = sigma_map_from_depth(near, amount=100, focus=1.0)
    assert float(sigma[:, -1].mean()) < 0.5, "near plane should stay sharp"
    assert float(sigma[:, 0].mean()) > 10.0, "far plane should blur hard"
    print("  OK   depth CoC peaks away from focus")


def test_weight_sigma_scales_with_coverage():
    w = np.zeros((16, 32), dtype=np.float32)
    w[:, 16:] = 1.0
    sigma = sigma_map_from_weight(w, amount=80)
    assert float(sigma[:, 8].mean()) < 0.1
    assert float(sigma[:, 24].mean()) > 5.0
    print("  OK   masked blur follows coverage")


def test_variable_blur_softens_high_sigma():
    rng = np.random.default_rng(0)
    img = rng.random((48, 48, 3), dtype=np.float32)
    # Checker detail
    img[::2, ::2] = 1.0
    sigma = np.full((48, 48), 8.0, dtype=np.float32)
    out = apply_variable_blur(img, sigma, preview=True)
    # Blurred checker loses high-frequency energy.
    assert float(np.std(out)) < float(np.std(img)) * 0.85
    print("  OK   variable blur reduces high-frequency energy")


def test_depth_layer_blur_composites():
    h, w = 40, 60
    depth = np.linspace(0.0, 1.0, w, dtype=np.float32)
    depth = np.tile(depth, (h, 1))
    layer = MaskLayer(
        depth.copy(),
        name="Depth",
        source="depth",
        adjustments={"Blur": 70.0, "BlurFocus": 100.0},
    )
    base = np.zeros((h, w, 3), dtype=np.float32)
    base[:, :, :] = np.linspace(0.1, 0.9, w, dtype=np.float32)[None, :, None]
    # Add vertical stripes so blur is measurable
    base[:, ::3] = 1.0
    out = apply_mask_layers(base, MaskLayerStack([layer]))
    # Far (left) should be softer than near (right).
    left_std = float(np.std(out[:, :10]))
    right_std = float(np.std(out[:, -10:]))
    assert left_std < right_std * 0.95, (
        f"far should be softer than near: left={left_std:.4f} right={right_std:.4f}"
    )
    print("  OK   Depth + Blur continuous DOF through mask stack")


def test_masked_background_blur():
    h, w = 32, 32
    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[:, :16] = 1.0  # left = background to blur
    layer = MaskLayer(
        alpha,
        name="BG",
        source="subject",
        invert=False,
        adjustments={"Blur": 90.0},
    )
    base = np.random.default_rng(1).random((h, w, 3), dtype=np.float32)
    base[:, ::2] = 1.0
    out = apply_mask_layers(base, MaskLayerStack([layer]))
    assert float(np.std(out[:, :16])) < float(np.std(base[:, :16]))
    assert float(np.mean(np.abs(out[:, 16:] - base[:, 16:]))) < 1e-5
    print("  OK   masked blur softens coverage only")


def test_bokeh_layer_blurs_outside_subject():
    h, w = 32, 32
    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[:, 16:] = 1.0  # right = subject kept sharp
    layer = MaskLayer(
        alpha,
        name="Background blur",
        source="bokeh",
        invert=True,
        adjustments={"Blur": 80.0},
    )
    rng = np.random.default_rng(1)
    base = rng.random((h, w, 3), dtype=np.float32)
    base[:, ::2] = 1.0
    out = apply_mask_layers(base, MaskLayerStack([layer]))
    assert float(np.std(out[:, :16])) < float(np.std(base[:, :16]))
    assert float(np.mean(np.abs(out[:, 16:] - base[:, 16:]))) < 1e-5
    print("  OK   bokeh layer blurs the inverted (background) side")


def main() -> int:
    test_depth_sigma_zero_at_focus()
    test_weight_sigma_scales_with_coverage()
    test_variable_blur_softens_high_sigma()
    test_depth_layer_blur_composites()
    test_masked_background_blur()
    test_bokeh_layer_blurs_outside_subject()
    print("All depth-blur checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
