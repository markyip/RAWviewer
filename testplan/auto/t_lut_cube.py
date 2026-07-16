#!/usr/bin/env python3
"""Creative .cube LUT parse / apply smoke tests."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def _write_identity_cube(path: str, size: int = 3) -> None:
    lines = [f"TITLE \"Identity\"", f"LUT_3D_SIZE {size}"]
    for b in range(size):
        for g in range(size):
            for r in range(size):
                lines.append(
                    f"{r / (size - 1):.6f} {g / (size - 1):.6f} {b / (size - 1):.6f}"
                )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    from raw_lut import apply_cube_lut, apply_creative_lut, parse_cube_file

    with tempfile.TemporaryDirectory() as td:
        cube = os.path.join(td, "id.cube")
        _write_identity_cube(cube, 5)
        lut = parse_cube_file(cube)
        check("parse size", lut.size == 5)
        check("parse data shape", lut.data.shape == (5, 5, 5, 3))

        img = np.zeros((8, 8, 3), np.float32)
        img[..., 0] = 0.4
        img[..., 1] = 0.2
        img[..., 2] = 0.1
        out = apply_cube_lut(img, lut, amount=100.0)
        check(
            "identity LUT ≈ input",
            float(np.max(np.abs(out - img))) < 0.02,
            f"max_delta={float(np.max(np.abs(out - img))):.4f}",
        )
        half = apply_cube_lut(img, lut, amount=0.0)
        check("amount 0 is no-op", np.allclose(half, img))

        # Managed apply path with missing name is no-op.
        check(
            "missing LUT name no-op",
            np.allclose(apply_creative_lut(img, {"CreativeLUTAmount": 100.0}), img),
        )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
