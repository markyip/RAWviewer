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
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

    # Mapping of remote path in HF repo to local filename expected by RAWviewer
    files_to_download = {
        "onnx/s0/vision_model.onnx": "image_encoder.onnx",
        "onnx/s0/text_model.onnx": "text_encoder.onnx"
    }
    
    for remote_path, local_name in files_to_download.items():
        print(f"  - Fetching {remote_path} -> {local_name}...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=remote_path,
            local_dir=str(MODELS_DIR)
        )
        
        # hf_hub_download with local_dir often creates subdirectories
        # We need to move the file to the root of MODELS_DIR and rename it
        downloaded_path = MODELS_DIR / remote_path
        target_path = MODELS_DIR / local_name
        
        if downloaded_path.exists():
            if target_path.exists():
                target_path.unlink()
            downloaded_path.rename(target_path)
            
    # Clean up empty subdirectories created by hf_hub_download
    if (MODELS_DIR / "onnx").exists():
        import shutil
        shutil.rmtree(MODELS_DIR / "onnx")

    # Also download the tokenizer
    tokenizer_url = "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"
    tokenizer_path = MODELS_DIR / "bpe_simple_vocab_16e6.txt.gz"
    if not tokenizer_path.exists():
        print(f"  - Fetching tokenizer -> {tokenizer_path.name}...")
        import urllib.request
        urllib.request.urlretrieve(tokenizer_url, tokenizer_path)

    print("[SUCCESS] MobileCLIP2 models and tokenizer ready for bundling.")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
