"""Export models/NAFNet-SIDD-width32.pth (megvii-research/NAFNet) to ONNX.

Same tiled-inference contract as export_scunet_onnx.py: RestormerONNX feeds
(1,3,tile_size,tile_size) float32 tiles in [0,1]. Static shape on purpose --
see export_scunet_onnx.py for why (DirectML segfaulted on dynamic axes for
the SCUNet graph; NAFNet is pure conv so it likely isn't affected, but static
matches actual runtime usage regardless).

Usage:
    python scripts/models/export_nafnet_onnx.py [--output models/nafnet.onnx]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network_nafnet import NAFNet

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT = os.path.join(ROOT, "models", "NAFNet-SIDD-width32.pth")
TILE_SIZE = 512


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(ROOT, "models", "nafnet.onnx"))
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--width", type=int, default=32)
    args = parser.parse_args()

    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "params" in state_dict:
        state_dict = state_dict["params"]

    model = NAFNet(img_channel=3, width=args.width, middle_blk_num=12,
                   enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert not missing and not unexpected
    model.eval()

    # Smoothed noise, not pure torch.rand: NAFNet (like most restoration nets)
    # is only well-behaved near its natural-image training distribution --
    # raw uniform noise as "the image" makes both the PyTorch and ONNX graphs
    # chaotically amplify float32 rounding differences, producing a spurious
    # multi-unit parity mismatch that has nothing to do with export fidelity.
    dummy = torch.rand(1, 3, TILE_SIZE, TILE_SIZE, dtype=torch.float32)
    dummy = torch.nn.functional.avg_pool2d(dummy, 7, stride=1, padding=3)

    with torch.no_grad():
        torch_out = model(dummy).numpy()

    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
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

    max_abs_diff = np.max(np.abs(torch_out - onnx_out))
    print(f"torch vs onnxruntime max abs diff: {max_abs_diff:.6f}")
    assert max_abs_diff < 1e-3, "ONNX export diverges from PyTorch reference"
    print("OK: ONNX output matches PyTorch reference within tolerance.")


if __name__ == "__main__":
    main()
