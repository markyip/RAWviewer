#!/usr/bin/env python3
"""AI denoise amount: strength blend, slider spec, and pipeline plumbing.

No ONNX runtime or model file is needed -- the tiled inference core is stubbed.
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


def main() -> int:
    os.environ["RAWVIEWER_AI_BACKEND"] = "scunet"
    from onnx_scunet import SCUNetONNX
    from raw_adjustments import (
        DEFAULT_ADJUSTMENTS,
        SLIDER_SPECS,
        is_default_adjustments,
        is_default_adjustments_for_sidecar,
    )

    spec = next((s for s in SLIDER_SPECS if s.key == "AIDenoiseAmount"), None)
    check("AIDenoiseAmount slider exists", spec is not None)
    if spec is None:
        print(f"\n{len(FAILURES)} failure(s)")
        return 1
    check(
        "AIDenoiseAmount default is 100",
        DEFAULT_ADJUSTMENTS.get("AIDenoiseAmount") == 100.0,
        f"default={DEFAULT_ADJUSTMENTS.get('AIDenoiseAmount')!r}",
    )
    check(
        "AIDenoiseAmount range is 0-100",
        (spec.minimum, spec.maximum) == (0, 100),
        f"range={spec.minimum}-{spec.maximum}",
    )
    check(
        "AIDenoiseAmount value maps 1:1 to slider",
        spec.slider_to_value(25) == 25.0 and spec.value_to_slider(25.0) == 25,
    )

    # Export-only: invisible to the gallery "edited" badge, but persisted.
    base = dict(DEFAULT_ADJUSTMENTS)
    base["Temperature"] = 5500.0
    base["AsShotTemperature"] = 5500.0
    check("default amount is_default", is_default_adjustments(base))
    amount = dict(base)
    amount["AIDenoiseAmount"] = 25.0
    check("amount change stays is_default (badge stays clean)", is_default_adjustments(amount))
    check(
        "amount change keeps the sidecar",
        not is_default_adjustments_for_sidecar(amount),
    )

    # Blend math against a stubbed core: the model "denoises" to black.
    class FakeDenoiser(SCUNetONNX):
        def __init__(self):
            self.model_path = "<fake>"
            self._session = object()
            self._input_name = "in"
            self._output_name = "out"
            self.tile_calls = 0

        def _process_tiles(self, rgb, *_args, **_kwargs):
            self.tile_calls += 1
            return np.zeros_like(rgb)

    orig = np.full((8, 8, 3), 0.4, dtype=np.float32)
    fake = FakeDenoiser()
    out0 = fake.process(orig, strength=0.0)
    check("strength 0 returns the original", np.array_equal(out0, orig))
    check("strength 0 skips inference", fake.tile_calls == 0)

    out1 = fake.process(orig, strength=1.0)
    check("strength 1 is the full model output", np.allclose(out1, 0.0))

    out5 = fake.process(orig, strength=0.5)
    check(
        "strength 0.5 is halfway between input and model",
        np.allclose(out5, 0.2, atol=1e-6),
        f"max={float(out5.max()):.6f}",
    )

    check("negative strength clamps to original", np.array_equal(fake.process(orig, strength=-2.0), orig))
    check("strength >1 clamps to full output", np.allclose(fake.process(orig, strength=4.0), 0.0))

    try:
        import cv2  # noqa: F401

        big = np.full((32, 32, 3), 0.4, dtype=np.float32)
        half = fake.process(big, strength=0.5, downscale=2)
        check(
            "strength 0.5 applies on the half-res export path too",
            np.allclose(half, 0.2, atol=1e-5),
            f"max={float(half.max()):.6f}",
        )
    except ImportError:
        print("SKIP  cv2 not installed -- half-res path not exercised")

    # Pipeline plumbing: AIDenoiseAmount reaches process() as strength.
    import onnx_scunet
    import raw_edit_pipeline as rep

    calls = []

    class StubSCUNet:
        def __init__(self, _path):
            pass

        def process(self, img, **kwargs):
            calls.append(kwargs)
            return img

    saved_env = os.environ.get("RAWVIEWER_EXPORT_SCUNET_ONNX")
    orig_cls, orig_path = onnx_scunet.SCUNetONNX, onnx_scunet.scunet_model_path
    onnx_scunet.SCUNetONNX = StubSCUNet
    # The pipeline only checks that the model file exists.
    onnx_scunet.scunet_model_path = lambda: os.path.abspath(__file__)
    os.environ["RAWVIEWER_EXPORT_SCUNET_ONNX"] = "1"
    try:
        img = np.full((32, 32, 3), 0.3, dtype=np.float32)
        rep.process_linear_edit_buffer(
            img, {"AIDenoiseAmount": 40.0}, preview=False, use_ai_denoise=True
        )
        check(
            "pipeline passes amount as strength",
            len(calls) == 1 and abs(calls[0].get("strength", -1.0) - 0.4) < 1e-6,
            f"calls={calls!r}",
        )

        calls.clear()
        rep.process_linear_edit_buffer(
            img, {"AIDenoiseAmount": 0.0}, preview=False, use_ai_denoise=True
        )
        check("amount 0 skips the model entirely", calls == [], f"calls={calls!r}")
    finally:
        onnx_scunet.SCUNetONNX = orig_cls
        onnx_scunet.scunet_model_path = orig_path
        if saved_env is None:
            os.environ.pop("RAWVIEWER_EXPORT_SCUNET_ONNX", None)
        else:
            os.environ["RAWVIEWER_EXPORT_SCUNET_ONNX"] = saved_env

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
