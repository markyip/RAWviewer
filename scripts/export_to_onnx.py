import torch
import os
from transformers import ViTForImageClassification

# --- CONFIGURATION ---
CHECKPOINT_DIR = r"D:\Development\RAWviewer\aviation_model_v2" 
ONNX_OUTPUT = r"D:\Development\RAWviewer\src\models\super_specialist.onnx"

def export():
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"ERROR: Checkpoint directory not found at {CHECKPOINT_DIR}")
        return

    print(f"--- Loading Fine-Tuned Model from {CHECKPOINT_DIR} ---")
    model = ViTForImageClassification.from_pretrained(CHECKPOINT_DIR)
    model.eval()
    
    # Create target directory if it doesn't exist
    os.makedirs(os.path.dirname(ONNX_OUTPUT), exist_ok=True)
    
    # Standard ViT input size
    dummy_input = torch.randn(1, 3, 224, 224)
    
    print(f"--- Exporting to ONNX (Opset 18) ---")
    torch.onnx.export(
        model, 
        dummy_input, 
        ONNX_OUTPUT,
        opset_version=18,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "logits": {0: "batch_size"}
        }
    )
    
    print(f"\nSUCCESS! Exported model to: {os.path.abspath(ONNX_OUTPUT)}")
    print("\nNext steps:")
    print(f"1. Copy '{ONNX_OUTPUT}' to your RAWviewer cache folder.")
    print("2. Update 'semantic_search.py' to use the new labels generated in labels.txt.")

if __name__ == "__main__":
    export()
