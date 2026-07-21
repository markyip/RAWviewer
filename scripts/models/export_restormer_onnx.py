"""Export models/restormer_real_denoising.pth (swz30/Restormer) to ONNX.

Same tiled-inference contract as the other export scripts here. Static shape
on purpose -- see export_scunet_onnx.py for why.

Usage:
    python scripts/models/export_restormer_onnx.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network_restormer import Restormer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT = os.path.join(ROOT, "models", "restormer_real_denoising.pth")
OUTPUT = os.path.join(ROOT, "models", "restormer_true.onnx")
TILE_SIZE = 512


def main():
    state_dict = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "params" in state_dict:
        state_dict = state_dict["params"]

    model = Restormer(inp_channels=3, out_channels=3, dim=48,
                       num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
                       heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                       bias=False, LayerNorm_type="BiasFree", dual_pixel_task=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert not missing and not unexpected
    model.eval()

    dummy = torch.rand(1, 3, TILE_SIZE, TILE_SIZE, dtype=torch.float32)
    dummy = torch.nn.functional.avg_pool2d(dummy, 7, stride=1, padding=3)

    with torch.no_grad():
        torch_out = model(dummy).numpy()

    torch.onnx.export(
        model,
        dummy,
        OUTPUT,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
        do_constant_folding=True,
    )

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


if __name__ == "__main__":
    main()
