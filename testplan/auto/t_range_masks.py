"""Unit checks for luminance / color range mask alphas."""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
sys.path.insert(0, ROOT)

from raw_range_masks import (  # noqa: E402
    color_range_alpha,
    luminance_range_alpha,
    sample_luminance_at,
    sample_rgb_at,
)


def test_luminance_band_peaks_at_center():
    img = np.zeros((32, 48, 3), dtype=np.float32)
    img[:, :16] = 0.2
    img[:, 16:32] = 0.5
    img[:, 32:] = 0.9
    alpha = luminance_range_alpha(img, center=0.5, width=0.08, feather=0.05)
    assert float(alpha[:, 20].mean()) > 0.9
    assert float(alpha[:, 4].mean()) < 0.2
    assert float(alpha[:, 40].mean()) < 0.2
    print("  OK   luminance range peaks on the sampled band")


def test_color_range_prefers_sample():
    img = np.zeros((24, 24, 3), dtype=np.float32)
    img[:12, :, 0] = 0.8  # red half
    img[12:, :, 1] = 0.8  # green half
    alpha = color_range_alpha(img, (0.8, 0.0, 0.0), tolerance=0.15, feather=0.08)
    assert float(alpha[:12].mean()) > float(alpha[12:].mean()) + 0.4
    print("  OK   color range prefers the sampled hue")


def test_sample_helpers():
    img = np.zeros((10, 10, 3), dtype=np.float32)
    img[4:7, 4:7] = (0.25, 0.5, 0.75)
    rgb = sample_rgb_at(img, 5, 5, radius=1)
    assert rgb is not None
    assert abs(rgb[1] - 0.5) < 0.05
    y = sample_luminance_at(img, 5, 5, radius=1)
    assert y is not None and 0.2 < y < 0.7
    print("  OK   sample helpers average a neighborhood")


def main() -> int:
    test_luminance_band_peaks_at_center()
    test_color_range_prefers_sample()
    test_sample_helpers()
    print("All range-mask checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
