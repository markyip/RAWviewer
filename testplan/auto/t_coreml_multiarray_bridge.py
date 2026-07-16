#!/usr/bin/env python3
"""Unit tests for Core ML MultiArray / CVPixelBuffer bulk-copy helpers."""
from __future__ import annotations

import ctypes
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


class _FakeMultiArray:
    def __init__(self, data: np.ndarray, data_type: int = 65568):
        self._buf = np.ascontiguousarray(data)
        self._dtype = data_type

    def dataPointer(self):
        return self._buf.ctypes.data

    def count(self):
        return int(self._buf.size)

    def dataType(self):
        return self._dtype


def main() -> int:
    from semantic_search import (
        copy_bgra_into_cvpixelbuffer,
        copy_numpy_into_mlmultiarray,
        mlmultiarray_to_numpy_f32,
    )

    # Float32 image-sized tensor round-trip via memmove helpers
    src = np.linspace(0.0, 1.0, 1 * 3 * 256 * 256, dtype=np.float32)
    dst = np.zeros_like(src)
    fake = _FakeMultiArray(dst)
    ok = copy_numpy_into_mlmultiarray(fake, src)
    check("copy_numpy_into_mlmultiarray float32", ok and np.allclose(dst, src))

    out = mlmultiarray_to_numpy_f32(fake)
    check(
        "mlmultiarray_to_numpy_f32 float32",
        out is not None and np.allclose(out, src),
    )

    # Int32 token path (write only; reader refuses non-float)
    tok = np.arange(77, dtype=np.int32)
    tok_dst = np.zeros_like(tok)
    fake_i = _FakeMultiArray(tok_dst, data_type=131104)
    ok_i = copy_numpy_into_mlmultiarray(fake_i, tok)
    check("copy_numpy_into_mlmultiarray int32", ok_i and np.array_equal(tok_dst, tok))
    check(
        "mlmultiarray_to_numpy_f32 refuses int32",
        mlmultiarray_to_numpy_f32(fake_i) is None,
    )

    # Packed BGRA CVPixelBuffer-style destination
    bgra = np.random.randint(0, 255, (256, 256, 4), dtype=np.uint8)
    dest = (ctypes.c_char * (256 * 256 * 4))()
    ok_p = copy_bgra_into_cvpixelbuffer(ctypes.addressof(dest), 256 * 4, bgra)
    got = np.frombuffer(dest, dtype=np.uint8).reshape(256, 256, 4)
    check("copy_bgra packed", ok_p and np.array_equal(got, bgra))

    # Strided rows (extra padding bytes)
    stride = 256 * 4 + 16
    dest2 = (ctypes.c_char * (stride * 256))()
    ok_s = copy_bgra_into_cvpixelbuffer(ctypes.addressof(dest2), stride, bgra)
    rows_ok = True
    for y in range(256):
        row = np.frombuffer(dest2, dtype=np.uint8, count=256 * 4, offset=y * stride)
        if not np.array_equal(row.reshape(256, 4), bgra[y]):
            rows_ok = False
            break
    check("copy_bgra strided", ok_s and rows_ok)

    # Fallback path: bad pointer returns False
    check(
        "copy_numpy rejects None pointer",
        copy_numpy_into_mlmultiarray(
            type("X", (), {"dataPointer": lambda self: None})(), src
        )
        is False,
    )

    if FAILURES:
        print(f"FAILED: {len(FAILURES)} — {FAILURES}")
        return 1
    print("PASS t_coreml_multiarray_bridge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
