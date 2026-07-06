#!/usr/bin/env python3
"""Regression guard: the half-size fast decode path must honor return_linear.

Real bug: _decode_half_from_unpacked_impl accepted a return_linear
parameter but never checked it -- it unconditionally applied the gamma-8
LUT and returned uint8 regardless. Since the Adjust panel's edit base
defaults to half-size decode with return_linear=True (see
UnifiedImageProcessor.decode_raw_edit_base), the entire "scene-linear"
editor pipeline was actually operating on gamma-encoded uint8 data for
every RAW file. This miscalibrated the Shadows/Blacks perceptual-ratio
math (built assuming true scene-linear input), reported as Shadows
producing a blue color cast and less shadow detail than an equivalent
Exposure push -- Exposure's simple multiplicative gain degrades more
gracefully on the wrong colorspace than the perceptual tone engine does.

decode_raw_edit_base's own dtype check (`if rgb_image.dtype ==
np.uint16`) already assumed the correct behavior; it just never actually
ran because the upstream function silently returned uint8 instead.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from fast_raw_decode import UnpackedRaw, decode_half_from_unpacked

    # A tiny synthetic RGGB mosaic: real values (not all-zero) so the demosaic
    # doesn't short-circuit, small enough to run instantly.
    h, w = 8, 8
    rng = np.random.default_rng(0)
    mosaic = rng.integers(1000, 40000, size=(h, w), dtype=np.uint16)
    pattern = np.array([[0, 1], [1, 2]], dtype=np.int32)  # R G / G B
    unpacked = UnpackedRaw(
        mosaic=mosaic,
        pattern=pattern,
        pat_str="RGGB",
        scale_mul={0: 1.5, 1: 1.0, 2: 1.8},
        black={0: 0, 1: 0, 2: 0},
        rgb_cam=np.eye(3, dtype=np.float32),
        file_path="synthetic.CR3",
    )

    out_gamma = decode_half_from_unpacked(unpacked, return_linear=False)
    out_linear = decode_half_from_unpacked(unpacked, return_linear=True)

    check("gamma-mode output is uint8", out_gamma is not None and out_gamma.dtype == np.uint8)
    check(
        "linear-mode output is uint16, NOT uint8 (the actual bug)",
        out_linear is not None and out_linear.dtype == np.uint16,
    )
    if out_gamma is not None and out_linear is not None:
        # The two must differ: linear-mode skips the gamma LUT entirely.
        # Compare on a common footing (both cast to float) since dtypes differ.
        same_shape = out_gamma.shape == out_linear.shape
        check("both modes produce the same shape", same_shape)
        if same_shape:
            g = out_gamma.astype(np.float64) / 255.0
            l = out_linear.astype(np.float64) / 65535.0
            check(
                "linear output is not just a rescaled copy of the gamma output "
                "(i.e. the gamma curve was actually skipped, not reapplied then undone)",
                not np.allclose(g, l, atol=0.01),
            )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
