#!/usr/bin/env python3
"""Plus AI denoise: edition gate, platform model preference, export flag plumbing."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    import numpy as np

    from onnx_scunet import _preferred_model_name, scunet_model_path
    from raw_edit_pipeline import process_linear_edit_buffer
    from rawviewer_profile import ai_export_enabled, is_lite_build

    # Edition gate: Standard (lite) hides AI denoise; Plus / unknown keeps it.
    saved = {k: os.environ.get(k) for k in ("RAWVIEWER_BUILD_PROFILE", "RAWVIEWER_FORCE_AI_EXPORT")}
    try:
        os.environ.pop("RAWVIEWER_FORCE_AI_EXPORT", None)
        os.environ["RAWVIEWER_BUILD_PROFILE"] = "lite"
        # resolved_profile reads env at call time.
        check("Standard (lite) hides AI export", ai_export_enabled() is False)
        os.environ["RAWVIEWER_BUILD_PROFILE"] = "full"
        check("Plus (full) offers AI export", ai_export_enabled() is True)
        os.environ["RAWVIEWER_FORCE_AI_EXPORT"] = "1"
        os.environ["RAWVIEWER_BUILD_PROFILE"] = "lite"
        check("RAWVIEWER_FORCE_AI_EXPORT overrides Standard", ai_export_enabled() is True)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    preferred = _preferred_model_name()
    if sys.platform.startswith("win"):
        check("Windows prefers SCUNet", preferred == "scunet.onnx", preferred)
    else:
        check("non-Windows prefers NAFNet", preferred == "nafnet.onnx", preferred)

    path = scunet_model_path()
    check("scunet_model_path returns a string", isinstance(path, str) and bool(path))

    # Flag plumbing: use_ai_denoise must not NameError, and the env kill-switch
    # must skip ONNX (no model load) while still running the rest of the tail.
    img = (np.random.RandomState(3).rand(32, 32, 3).astype(np.float32)) * 0.4
    prev_env = os.environ.get("RAWVIEWER_EXPORT_SCUNET_ONNX")
    try:
        os.environ["RAWVIEWER_EXPORT_SCUNET_ONNX"] = "0"
        out = process_linear_edit_buffer(
            img, {"Exposure2012": 0.25}, preview=False, use_ai_denoise=True
        )
        check(
            "use_ai_denoise with ONNX disabled still returns an image",
            out is not None and out.shape == img.shape,
        )
        import raw_edit_pipeline as rep

        called = []
        orig = rep._process_linear_edit_tail

        def wrapped(*a, **k):
            called.append(k.get("use_ai_denoise"))
            return orig(*a, **k)

        rep._process_linear_edit_tail = wrapped
        try:
            process_linear_edit_buffer(img, {}, preview=False, use_ai_denoise=True)
        finally:
            rep._process_linear_edit_tail = orig
        check(
            "default adj + use_ai_denoise does not skip the pipeline",
            called == [True],
            f"called={called!r}",
        )
    finally:
        if prev_env is None:
            os.environ.pop("RAWVIEWER_EXPORT_SCUNET_ONNX", None)
        else:
            os.environ["RAWVIEWER_EXPORT_SCUNET_ONNX"] = prev_env

    # Lite build helper still exists (used by the edition gate).
    check("is_lite_build is callable", callable(is_lite_build))

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
