"""AF-prior bias for Smart Object mattes."""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
sys.path.insert(0, ROOT)

from raw_ai_masks import _bias_matte_to_focus  # noqa: E402


def test_bias_keeps_focus_blob():
    alpha = np.zeros((40, 80), dtype=np.float32)
    alpha[5:15, 5:20] = 0.9   # left blob
    alpha[5:15, 55:75] = 0.9  # right blob
    out = _bias_matte_to_focus(alpha, (65.0, 10.0))
    assert float(out[8, 65]) > 0.7
    assert float(out[8, 10]) < 0.25
    print("  OK   AF prior keeps the focused component")


def main() -> int:
    test_bias_keeps_focus_blob()
    print("All AF-prior checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
