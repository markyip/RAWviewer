"""Export models/scunet_color_real_psnr.pth (KAIR SCUNet) to models/restormer.onnx.

The RAWviewer denoise path (src/onnx_restormer.py, src/raw_edit_pipeline.py) loads
whatever ONNX model sits at models/restormer.onnx and feeds it (1,3,H,W) float32
tiles in [0,1]. The engine name is historical (originally targeted Restormer); the
model actually wired in is SCUNet real-image color denoising.

fp16 by default: DirectML runs this graph's window-attention ops far faster
in fp16 (~8-25x on the tile benchmark that motivated this) than fp32 -- SCUNet
is apparently overhead/precision-path bound on DML, not raw-FLOPs bound.
Verified fp16 vs fp32 max abs diff ~0.005 on a real photo tile, no visible
quality change, no NaN/Inf even on a deliberately blown-highlight stress
crop. Pass --no-fp16 to keep the fp32 model instead.

Usage (run with a Python that has torch/einops/timm/onnx/onnxconverter-common
installed):
    python scripts/models/export_scunet_onnx.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network_scunet import SCUNet

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT = os.path.join(ROOT, "models", "scunet_color_real_psnr.pth")
OUTPUT = os.path.join(ROOT, "models", "scunet.onnx")
TILE_SIZE = 512


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16", dest="fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    args = parser.parse_args()

    state_dict = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "params" in state_dict:
        state_dict = state_dict["params"]

    model = SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64, input_resolution=256)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert not missing and not unexpected
    model.eval()

    dummy = torch.rand(1, 3, TILE_SIZE, TILE_SIZE, dtype=torch.float32)

    with torch.no_grad():
        torch_out = model(dummy).numpy()

    # Static 512x512 shape on purpose: RestormerONNX.process() always feeds
    # exactly (tile_size, tile_size) tiles (edge tiles are reflect-padded up
    # to that size), and dynamic H/W axes made the DirectML EP segfault on
    # this graph (window-attention reshape/einsum ops). Static shapes avoid
    # that path entirely and are what's actually used at runtime.
    torch.onnx.export(
        model,
        dummy,
        OUTPUT,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
    )

    # The dynamo exporter defaults to external-data weights (a sidecar
    # restormer.onnx.data file). Inline everything into a single .onnx so
    # there's only one file to ship/package.
    import onnx

    onnx_model = onnx.load(OUTPUT, load_external_data=True)
    data_path = OUTPUT + ".data"
    onnx.save_model(onnx_model, OUTPUT, save_as_external_data=False)
    if os.path.exists(data_path):
        os.remove(data_path)

    print(f"Exported {OUTPUT} ({os.path.getsize(OUTPUT) / 1e6:.1f} MB, single file)")

    import onnxruntime as ort

    sess = ort.InferenceSession(OUTPUT, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": dummy.numpy()})[0]

    max_abs_diff = np.max(np.abs(torch_out - onnx_out))
    print(f"torch vs onnxruntime max abs diff: {max_abs_diff:.6f}")
    assert max_abs_diff < 1e-3, "ONNX export diverges from PyTorch reference"
    print("OK: ONNX output matches PyTorch reference within tolerance.")

    if args.fp16:
        from onnxconverter_common import float16

        # keep_io_types: RestormerONNX feeds/reads plain float32 numpy arrays;
        # only internal compute runs in fp16.
        fp16_model = float16.convert_float_to_float16(
            onnx.load(OUTPUT), keep_io_types=True
        )
        onnx.save(fp16_model, OUTPUT)
        print(f"Converted to fp16: {OUTPUT} ({os.path.getsize(OUTPUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
