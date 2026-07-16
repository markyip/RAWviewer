#!/usr/bin/env python3
"""Regression: edit-base params with corrected user_wb must stay on the fast path.

edit_base_decode_params() sets use_camera_wb=False + user_wb when
get_corrected_camera_wb() has a correction. params_supported() used to reject
that combination, forcing every such file through slow rawpy AHD on both
macOS and Windows. unpack_raw already bakes the same correction into
scale_mul, so accepting these params is color-correct.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from fast_raw_decode import params_supported, params_supported_half

    base_linear = {
        "use_camera_wb": True,
        "use_auto_wb": False,
        "output_bps": 16,
        "gamma": (1, 1),
        "no_auto_bright": True,
        "bright": 1.0,
        "user_flip": 0,
        "highlight_mode": 1,  # Blend
        "demosaic_algorithm": "AHD",
    }

    check(
        "camera WB linear half params supported",
        params_supported_half({**base_linear, "half_size": True}, return_linear=True),
    )

    corrected = {
        **base_linear,
        "use_camera_wb": False,
        "user_wb": [1780.0, 1024.0, 1786.0, 1024.0],
        "half_size": True,
    }
    check(
        "corrected user_wb linear half params supported (edit-base path)",
        params_supported_half(corrected, return_linear=True),
    )

    full_corrected = {k: v for k, v in corrected.items() if k != "half_size"}
    check(
        "corrected user_wb linear full params supported",
        params_supported(full_corrected, return_linear=True),
    )

    bad = {**corrected, "user_wb": [0.0, 1024.0, 1786.0]}
    check(
        "zero-channel user_wb rejected",
        not params_supported_half(bad, return_linear=True),
    )

    no_wb = {
        **base_linear,
        "use_camera_wb": False,
        "half_size": True,
    }
    check(
        "use_camera_wb=False without user_wb rejected",
        not params_supported_half(no_wb, return_linear=True),
    )

    # Browse 8-bit path with user_wb (rawpy fallback shape) should also work
    # if a caller ever injects correction before try_fast_raw_decode.
    browse = {
        "use_camera_wb": False,
        "user_wb": [2000.0, 1024.0, 1500.0, 1024.0],
        "use_auto_wb": False,
        "output_bps": 8,
        "gamma": (2.222, 4.5),
        "no_auto_bright": True,
        "bright": 1.0,
        "user_flip": 0,
    }
    check(
        "corrected user_wb browse 8-bit params supported",
        params_supported(browse, return_linear=False),
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
