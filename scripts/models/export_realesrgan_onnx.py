"""Export models/RealESRGAN_x2plus.pth (xinntao/Real-ESRGAN) to ONNX.

Same tiled-inference contract as export_scunet_onnx.py / export_nafnet_onnx.py:
RealESRGANONNX feeds (1, 3, TILE, TILE) float32 tiles in [0, 1] and gets
(1, 3, 2*TILE, 2*TILE) back. Static shape for the same reason those two use it
-- it matches actual runtime usage, and DirectML has a history of misbehaving
on dynamic axes here.

TILE is 256, not the 512 the denoise models use: output is 4x the pixels of
input, so a 512 tile would allocate a 1024x1024x3 float activation per tile and
roughly quadruple peak memory for no throughput gain.

Weights: RealESRGAN_x2plus.pth from the official v0.2.1 release (BSD-3-Clause).
Download with scripts/models/download_realesrgan_weights.py.

Usage (run with a Python that has torch/onnx/onnxruntime -- torch is a dev-only
dependency, deliberately not in the pixi env):
    python scripts/models/export_realesrgan_onnx.py [--output models/realesrgan_x2.onnx]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network_rrdbnet import RRDBNet

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT = os.path.join(ROOT, "models", "RealESRGAN_x2plus.pth")
TILE_SIZE = 256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(ROOT, "models", "realesrgan_x2.onnx"))
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--tile", type=int, default=TILE_SIZE)
    args = parser.parse_args()

    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    # The release ships both a raw and an EMA copy of the weights. EMA is what
    # the reference inference code uses and what the published samples show.
    if isinstance(state_dict, dict):
        state_dict = state_dict.get("params_ema", state_dict.get("params", state_dict))

    model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=2, num_feat=64, num_block=23, num_grow_ch=32)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()

    # Smoothed noise rather than raw torch.rand, for the reason spelled out in
    # export_nafnet_onnx.py: off-distribution input makes both graphs amplify
    # float32 rounding into a parity mismatch that says nothing about export
    # fidelity. A super-resolution net is if anything more sensitive.
    dummy = torch.rand(1, 3, args.tile, args.tile, dtype=torch.float32)
    dummy = torch.nn.functional.avg_pool2d(dummy, 7, stride=1, padding=3)

    with torch.no_grad():
        torch_out = model(dummy).numpy()
    assert torch_out.shape[2:] == (args.tile * 2, args.tile * 2), torch_out.shape

    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["input"],
        output_names=["output"],
        # 18, not the 17 the denoise exports use: nearest-neighbour upsampling
        # becomes a Resize node whose opset-17 adapter does not exist, so
        # requesting 17 makes the version converter fail and silently fall
        # back to 18 anyway -- with a stack trace in the log. Ask for what we
        # actually get.
        opset_version=18,
        do_constant_folding=True,
    )

    import onnx

    onnx_model = onnx.load(args.output, load_external_data=True)
    data_path = args.output + ".data"
    onnx.save_model(onnx_model, args.output, save_as_external_data=False)
    if os.path.exists(data_path):
        os.remove(data_path)

    print(f"Exported {args.output} ({os.path.getsize(args.output) / 1e6:.1f} MB, single file)")

    import onnxruntime as ort

    sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": dummy.numpy()})[0]

    max_abs_diff = float(np.max(np.abs(torch_out - onnx_out)))
    print(f"Parity vs PyTorch: max abs diff {max_abs_diff:.3e}")
    if max_abs_diff > 1e-3:
        raise SystemExit(f"Export parity check FAILED: {max_abs_diff:.3e} > 1e-3")

    import hashlib

    digest = hashlib.sha256()
    with open(args.output, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    print(f"sha256: {digest.hexdigest()}")


if __name__ == "__main__":
    main()
