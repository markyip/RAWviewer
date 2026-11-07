#!/usr/bin/env python3
"""AI denoise golden: preview and export share one pipeline and one result.

The SCUNet model is stubbed with a deterministic function, so no ONNX runtime
or model file is needed. The image is tall enough to take the banded executor.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def check_rawrefinery(rep) -> None:
    import rawrefinery_denoise as rr

    os.environ["RAWVIEWER_AI_BACKEND"] = "rawrefinery"
    check("RawRefinery model is present", rr.model_available(), rr.model_path())
    rng = np.random.default_rng(1)
    img = rng.uniform(0.02, 0.5, size=(300, 200, 3)).astype(np.float32)
    adj = {"AIDenoiseAmount": 100.0, "Exposure2012": 0.2}
    prev = rep.run_edit_pipeline(img, adj, target="preview", use_ai_denoise=True)
    exp = rep.run_edit_pipeline(img, adj, target="export", use_ai_denoise=True)
    check("RawRefinery: preview equals export", np.array_equal(prev, exp))
    check("RawRefinery: output is finite", bool(np.isfinite(exp).all()))
    orig_denoise = rr.denoise

    def _must_not_run(*_a, **_k):
        raise AssertionError("model ran at amount 0")

    rr.denoise = _must_not_run
    try:
        zero = rep.run_edit_pipeline(img, {**adj, "AIDenoiseAmount": 0.0}, target="export", use_ai_denoise=True)
        check("RawRefinery: amount 0 skips the model", bool(np.isfinite(zero).all()))
    except AssertionError as exc:
        check("RawRefinery: amount 0 skips the model", False, str(exc))
    finally:
        rr.denoise = orig_denoise
    os.environ["RAWVIEWER_AI_BACKEND"] = "scunet"


def main() -> int:
    os.environ["RAWVIEWER_AI_BACKEND"] = "scunet"
    import onnx_scunet
    import raw_edit_pipeline as rep

    calls = []

    class StubSCUNet:
        def __init__(self, _path):
            pass

        def process(self, img, **kwargs):
            calls.append(kwargs)
            return np.clip(img * 0.5 + 0.1, 0.0, None).astype(np.float32)

    orig_cls, orig_path = onnx_scunet.SCUNetONNX, onnx_scunet.scunet_model_path
    onnx_scunet.SCUNetONNX = StubSCUNet
    onnx_scunet.scunet_model_path = lambda: os.path.abspath(__file__)
    try:
        rng = np.random.default_rng(0)
        img = rng.uniform(0.05, 0.6, size=(2048, 96, 3)).astype(np.float32)
        adj = {"AIDenoiseAmount": 80.0, "Exposure2012": 0.3, "Temperature": 5600.0}

        workers = rep._linear_pipeline_worker_count(img.shape[0])
        check("golden image takes the banded path", workers > 1, f"workers={workers}")

        calls.clear()
        preview = rep.run_edit_pipeline(img, adj, target="preview", use_ai_denoise=True)
        export = rep.run_edit_pipeline(img, adj, target="export", use_ai_denoise=True)
        check(
            "preview equals export (AI on)",
            np.array_equal(preview, export),
            f"max diff={float(np.abs(preview - export).max()):.3e}",
        )
        check(
            "AI runs once per call on the full image",
            len(calls) == 2 and all(c.get("downscale") == 1 for c in calls),
            f"calls={len(calls)}",
        )

        single = rep._process_linear_edit_tail(
            img.copy(),
            {**rep.DEFAULT_ADJUSTMENTS, **adj},
            preview=False,
            chroma_denoise=None,
            use_ai_denoise=True,
        )
        diff = float(np.abs(export - single).max())
        check(
            "banded AI result matches single-pass result",
            diff < 1e-5,
            f"max diff={diff:.3e}",
        )

        calls.clear()
        amount_zero = rep.run_edit_pipeline(
            img, {**adj, "AIDenoiseAmount": 0.0}, target="export", use_ai_denoise=True
        )
        check(
            "amount 0 with AI on skips the model",
            calls == [],
            f"calls={len(calls)}",
        )
        check("amount 0 output is finite", bool(np.isfinite(amount_zero).all()))
    finally:
        onnx_scunet.SCUNetONNX = orig_cls
        onnx_scunet.scunet_model_path = orig_path

    check_rawrefinery(rep)

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
