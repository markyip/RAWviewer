#!/usr/bin/env python3
"""Download MobileCLIP Core ML assets for RAWviewer (macOS semantic search).

Used by:
- ``build_macos.sh`` to populate ``models/mobileclip2_coreml/`` before PyInstaller bundling
- End users / devs who need models in ``~/.rawviewer_cache/mobileclip_coreml``

Example:
    python scripts/download_mobileclip_coreml.py --out-dir models/mobileclip2_coreml
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


MODEL_REPO = "apple/coreml-mobileclip"
MODEL_FILES = ("mobileclip_s2_image.mlpackage", "mobileclip_s2_text.mlpackage")
TOKENIZER_URL = "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"
TOKENIZER_FILE = "bpe_simple_vocab_16e6.txt.gz"


def _mlpackage_complete(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "Manifest.json").is_file()
        and (path / "Data" / "com.apple.CoreML" / "model.mlmodel").is_file()
        and (path / "Data" / "com.apple.CoreML" / "weights" / "weight.bin").is_file()
    )


def models_ready(target_dir: str | Path) -> bool:
    """Return True when image + text Core ML bundles and tokenizer are present."""
    root = Path(target_dir).expanduser().resolve()
    tok = root / TOKENIZER_FILE
    if not tok.is_file():
        return False
    for name in MODEL_FILES:
        if not _mlpackage_complete(root / name):
            return False
    return True


def download_models(
    target_dir: str | Path,
    *,
    progress_callback=None,
) -> Path:
    """Download Apple MobileCLIP S2 Core ML assets into *target_dir*."""
    root = Path(target_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    def _progress(message: str) -> None:
        print(message, flush=True)
        if progress_callback:
            progress_callback(message)

    if models_ready(root):
        _progress(f"MobileCLIP Core ML assets already present in: {root}")
        return root

    _progress(f"Downloading MobileCLIP Core ML models to {root}...")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "MobileCLIP download requires 'huggingface_hub' and 'requests'. "
            "Run: pip install huggingface-hub requests"
        ) from exc

    snapshot_download(
        repo_id=MODEL_REPO,
        allow_patterns=[f"{name}/**" for name in MODEL_FILES],
        local_dir=str(root),
    )

    tokenizer_path = root / TOKENIZER_FILE
    if not tokenizer_path.is_file():
        _progress(f"Downloading {TOKENIZER_FILE}...")
        urllib.request.urlretrieve(TOKENIZER_URL, tokenizer_path)

    if not models_ready(root):
        raise RuntimeError(f"MobileCLIP download incomplete in {root}")
    _progress(f"MobileCLIP Core ML assets are ready in: {root}")
    return root


def _download_via_hf_cli(target_dir: str) -> bool:
    hf = shutil.which("hf")
    if hf is None:
        return False
    os.makedirs(target_dir, exist_ok=True)
    for filename in MODEL_FILES:
        print(f"Downloading {filename} via hf CLI...", flush=True)
        subprocess.check_call(
            [hf, "download", MODEL_REPO, filename, "--local-dir", target_dir]
        )
    tokenizer_path = os.path.join(target_dir, TOKENIZER_FILE)
    if not os.path.exists(tokenizer_path):
        print(f"Downloading {TOKENIZER_FILE}...", flush=True)
        urllib.request.urlretrieve(TOKENIZER_URL, tokenizer_path)
    return models_ready(target_dir)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Download MobileCLIP Core ML assets for RAWviewer")
    parser.add_argument(
        "--out-dir",
        default=os.environ.get(
            "RAWVIEWER_MOBILECLIP_MODEL_DIR",
            str(repo_root / "models" / "mobileclip2_coreml"),
        ),
        help="Destination directory (default: models/mobileclip2_coreml or RAWVIEWER_MOBILECLIP_MODEL_DIR)",
    )
    args = parser.parse_args()
    target_dir = Path(args.out_dir).expanduser().resolve()

    if models_ready(target_dir):
        print(f"MobileCLIP Core ML assets already present in: {target_dir}")
        return 0

    try:
        download_models(target_dir)
        return 0
    except Exception as py_err:
        print(f"[WARN] huggingface_hub download failed: {py_err}", file=sys.stderr)
        if _download_via_hf_cli(str(target_dir)):
            print(f"MobileCLIP Core ML assets are ready in: {target_dir}")
            return 0
        print(
            "[ERROR] Could not download MobileCLIP Core ML assets. "
            "Install huggingface-hub (pip install huggingface-hub requests) or the hf CLI.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
