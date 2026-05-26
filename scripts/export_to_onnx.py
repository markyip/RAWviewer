import torch
import os
from transformers import ViTForImageClassification
from onnxruntime.quantization import quantize_dynamic, QuantType

# --- CONFIGURATION ---
CHECKPOINT_DIR = r"./aviation_model_v3"
ONNX_OUTPUT = r"../src/models/super_specialist.onnx"

def export():
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"ERROR: Checkpoint directory not found at {CHECKPOINT_DIR}")
        return

    print(f"--- Loading Fine-Tuned Model from {CHECKPOINT_DIR} ---")
    model = ViTForImageClassification.from_pretrained(CHECKPOINT_DIR)
    model.eval()
    
    # Create target directory if it doesn't exist
    os.makedirs(os.path.dirname(ONNX_OUTPUT), exist_ok=True)
    
    # High-resolution ViT input size (384x384)
    dummy_input = torch.randn(1, 3, 384, 384)
    
    print(f"--- Exporting to Standard ONNX (FP32) ---")
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
    
    # Quantization Step
    quantized_output = ONNX_OUTPUT.replace(".onnx", "_quantized.onnx")
    print(f"--- Quantizing to INT8 (Reduced Size): {quantized_output} ---")
    
    try:
        quantize_dynamic(
            model_input=ONNX_OUTPUT,
            model_output=quantized_output,
            weight_type=QuantType.QUInt8
        )
        print(f"SUCCESS: Quantized model saved to: {quantized_output}")
        final_model = quantized_output
    except Exception as e:
        print(f"WARNING: Quantization failed: {e}")
        print("Using standard FP32 model instead.")
        final_model = ONNX_OUTPUT

    print(f"\n--- Export Complete ---")
    print(f"Model: {os.path.abspath(final_model)}")
    
    # Also ensure labels.txt is in the same folder as the model for easy access
    labels_src = os.path.join(CHECKPOINT_DIR, "labels.txt")
    labels_dst = os.path.join(os.path.dirname(ONNX_OUTPUT), "labels.txt")
    if os.path.exists(labels_src):
        import shutil
        shutil.copy(labels_src, labels_dst)
        print(f"Labels copied to: {labels_dst}")

    print("\nNext steps:")
    print(f"1. Your optimized model is ready at: {final_model}")
    print("2. Ensure RAWviewer is configured to load this ONNX file in 'semantic_search.py'.")

if __name__ == "__main__":
    export()
