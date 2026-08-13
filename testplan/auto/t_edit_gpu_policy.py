#!/usr/bin/env python3
"""Adjust preview stays CPU; full-res edit-base decode may reuse GPU demosaic."""
from __future__ import annotations

import inspect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    extra = f"  {detail}" if detail else ""
    print(f"{'PASS' if ok else 'FAIL'}  {name}{extra}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    import edit_gpu_policy as policy
    from fast_raw_decode import try_fast_raw_decode
    from raw_chroma_denoise import apply_guided_filter
    from raw_edit_pipeline import process_linear_edit_buffer_staged
    from unified_image_processor import UnifiedImageProcessor

    check(
        "live Adjust GPU compute stays off",
        policy.adjust_preview_gpu_compute_enabled() is False,
    )

    src = inspect.getsource(try_fast_raw_decode)
    check(
        "half-size fast decode does not GPU-demosaic",
        "GPU half-size isn't implemented" in src,
    )

    gf_src = inspect.getsource(apply_guided_filter)
    check(
        "guided filter stays on cv2.boxFilter",
        "cv2.boxFilter" in gf_src and "cupy" not in gf_src.lower(),
    )

    staged_src = inspect.getsource(process_linear_edit_buffer_staged)
    check(
        "staged preview does not import torch/kornia/cupy",
        all(tok not in staged_src.lower() for tok in ("torch", "kornia", "cupy")),
    )

    impl_src = inspect.getsource(UnifiedImageProcessor._decode_raw_edit_base_impl)
    check(
        "full-res edit base reuses stash via finish_linear_edit_base_from_unpacked",
        "finish_linear_edit_base_from_unpacked" in impl_src
        and "_peek_unpacked_raw" in impl_src,
    )

    calls = {"n": 0, "prefer_gpu": None, "return_linear": None}

    def _fake_finish(unpacked, cancel_check=None, return_linear=False, *, prefer_gpu=False):
        calls["n"] += 1
        calls["prefer_gpu"] = prefer_gpu
        calls["return_linear"] = return_linear
        return ("linear", unpacked)

    sentinel = object()
    import fast_raw_decode as frd

    real_finish = frd.finish_full_decode
    real_supported = frd.params_supported
    real_prefer = policy.prefer_gpu_linear_edit_decode
    try:
        frd.finish_full_decode = _fake_finish
        frd.params_supported = lambda params, return_linear=False: True
        policy.prefer_gpu_linear_edit_decode = lambda: True
        out = policy.finish_linear_edit_base_from_unpacked(
            sentinel,
            {
                "use_camera_wb": True,
                "use_auto_wb": False,
                "output_bps": 16,
                "gamma": (1, 1),
                "no_auto_bright": True,
                "bright": 1.0,
                "user_flip": 0,
                "highlight_mode": 1,
                "half_size": False,
            },
        )
    finally:
        frd.finish_full_decode = real_finish
        frd.params_supported = real_supported
        policy.prefer_gpu_linear_edit_decode = real_prefer

    check("stash helper calls finish_full_decode once", calls["n"] == 1)
    check("stash helper requests linear output", calls["return_linear"] is True)
    check("stash helper forwards prefer_gpu", calls["prefer_gpu"] is True)
    check("stash helper returns the finish result", out == ("linear", sentinel))
    check("empty unpack is a miss", policy.finish_linear_edit_base_from_unpacked(None) is None)
    check(
        "half_size params are rejected",
        policy.finish_linear_edit_base_from_unpacked(
            sentinel, {"half_size": True, "use_camera_wb": True, "no_auto_bright": True}
        )
        is None,
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
