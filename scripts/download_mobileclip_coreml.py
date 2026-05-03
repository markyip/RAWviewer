#!/usr/bin/env python3
"""Download the optional MobileCLIP Core ML assets for RAWviewer.

Requires the Hugging Face `hf` CLI:
    hf download apple/coreml-mobileclip --local-dir ~/.rawviewer_cache/mobileclip_coreml
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request


MODEL_REPO = "apple/coreml-mobileclip"
MODEL_FILES = ("mobileclip_s2_image.mlpackage", "mobileclip_s2_text.mlpackage")
TOKENIZER_URL = "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"
TOKENIZER_FILE = "bpe_simple_vocab_16e6.txt.gz"


def main() -> int:
    target_dir = os.environ.get(
        "RAWVIEWER_MOBILECLIP_MODEL_DIR",
        os.path.expanduser("~/.rawviewer_cache/mobileclip_coreml"),
    )
    os.makedirs(target_dir, exist_ok=True)

    hf = shutil.which("hf")
    if hf is None:
        print("Missing `hf` CLI. Install it first: curl -LsSf https://hf.co/cli/install.sh | bash -s", file=sys.stderr)
        return 1

    for filename in MODEL_FILES:
        print(f"Downloading {filename}...")
        subprocess.check_call(
            [
                hf,
                "download",
                MODEL_REPO,
                filename,
                "--local-dir",
                target_dir,
            ]
        )

    tokenizer_path = os.path.join(target_dir, TOKENIZER_FILE)
    if not os.path.exists(tokenizer_path):
        print(f"Downloading {TOKENIZER_FILE}...")
        urllib.request.urlretrieve(TOKENIZER_URL, tokenizer_path)

    print(f"MobileCLIP Core ML assets are ready in: {target_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
