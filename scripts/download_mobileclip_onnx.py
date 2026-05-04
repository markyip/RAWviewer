#!/usr/bin/env python3
"""
Download MobileCLIP2-S0 ONNX models for bundling into RAWviewer.
"""

import os
from pathlib import Path

REPO_ID = "plhery/mobileclip2-onnx"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "mobileclip_onnx"

def main():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[ERROR] Missing 'huggingface-hub'. Run: pip install huggingface-hub")
        return 1

    print(f"[INFO] Downloading MobileCLIP2-S0 ONNX models to {MODELS_DIR}...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    filenames = ["image_encoder.onnx", "text_encoder.onnx"]
    
    for filename in filenames:
        print(f"  - Fetching {filename}...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False
        )

    # Also download the tokenizer if missing (though it's usually downloaded by the backend)
    # But for a "fully baked" experience, we'll keep it in the models folder.
    
    print("[SUCCESS] MobileCLIP2 models ready for bundling.")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
