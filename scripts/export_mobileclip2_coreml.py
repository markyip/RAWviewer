#!/usr/bin/env python3
"""Export MobileCLIP2 image + text encoders to Core ML (.mlpackage) for local testing.

Prerequisites (example):
  - Patch / install Apple ``open_clip`` per ml-mobileclip README (editable ``open_clip`` + ``pip install -e`` ml-mobileclip).
  - ``open_clip/src/open_clip/mobileclip2/__init__.py`` must re-export::
        from .mobileclip2 import fastvit_mci3, fastvit_mci4
    (Without it ``import open_clip`` fails after the inference-only patch.)

Conversion uses ``torch.export`` + ``run_decompositions({})``, then ``coremltools.convert``,
because TorchScript tracing hits unsupported ``int()`` ops inside FastViT.

Example (from repository root, with a venv that has torch + coremltools + editable deps)::

    export HF_HOME=\"$PWD/.hf_cache_mobileclip\"
    python scripts/export_mobileclip2_coreml.py --model MobileCLIP2-S0 --verify
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

_CLIP_BPE_URL = "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"


def _bootstrap_hf_cache(argv: argparse.Namespace, repo_root: Path) -> Path:
    default_hf = repo_root / ".hf_cache_mobileclip"
    if argv.hf_home:
        home = Path(argv.hf_home).expanduser().resolve()
    else:
        home = Path(os.environ.get("HF_HOME", str(default_hf))).expanduser().resolve()
    hub = home / "hub"
    home.mkdir(parents=True, exist_ok=True)
    hub.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(home))
    os.environ.setdefault("HF_HUB_CACHE", str(hub))
    return hub


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    default_out = repo_root / "artifacts" / "mobileclip2_coreml"

    p = argparse.ArgumentParser(description="Export MobileCLIP2 to Core ML mlpackages")
    p.add_argument("--model", default="MobileCLIP2-S0", help="OpenCLIP model name, e.g. MobileCLIP2-S0")
    p.add_argument(
        "--pretrained",
        default="dfndr2b",
        help="Weights tag passed to create_model_and_transforms (default: dfndr2b)",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Where to save *.mlpackage (default: artifacts/mobileclip2_coreml, or models/mobileclip2_coreml with --for-app)",
    )
    p.add_argument(
        "--for-app",
        action="store_true",
        help="Use RAWviewer bundle filenames (mobileclip2_s0_*) under models/mobileclip2_coreml/ and download CLIP tokenizer if missing",
    )
    p.add_argument(
        "--hf-home",
        dest="hf_home",
        default="",
        help="HF cache root (defaults to <repo>/.hf_cache_mobileclip or HF_HOME env)",
    )
    p.add_argument("--verify", action="store_true", help="Run cosine check vs PyTorch on random inputs")
    args = p.parse_args()

    if args.for_app:
        out_dir = (
            Path(args.out_dir).expanduser().resolve()
            if args.out_dir
            else (repo_root / "models" / "mobileclip2_coreml").resolve()
        )
        slug = "mobileclip2_s0"
    else:
        out_dir = (
            Path(args.out_dir).expanduser().resolve()
            if args.out_dir
            else default_out.expanduser().resolve()
        )
        slug = args.model.replace("/", "-")

    hub_cache = _bootstrap_hf_cache(args, repo_root)

    try:
        import numpy as np
        import torch
        import torch.nn as nn
        import coremltools as ct
        import open_clip
        from mobileclip.modules.common.mobileone import reparameterize_model
    except ImportError as e:
        print("Missing dependency:", e, file=sys.stderr)
        print("pip install torch coremltools (and editable open_clip + ml-mobileclip).", file=sys.stderr)
        return 1

    model_name = args.model
    model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}
    if model_name not in {"MobileCLIP2-S0", "MobileCLIP2-S2", "MobileCLIP2-B"}:
        model_kwargs = {}

    model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=args.pretrained,
        cache_dir=str(hub_cache),
        **model_kwargs,
    )
    model.eval()
    model = reparameterize_model(model)

    class ImageEncoder(nn.Module):
        def __init__(self, m: torch.nn.Module) -> None:
            super().__init__()
            self.m = m

        def forward(self, image):
            return self.m.encode_image(image)

    class TextEncoder(nn.Module):
        def __init__(self, m: torch.nn.Module) -> None:
            super().__init__()
            self.m = m

        def forward(self, tokens):
            return self.m.encode_text(tokens.to(torch.int64))

    out_dir.mkdir(parents=True, exist_ok=True)
    path_img = out_dir / f"{slug}_image.mlpackage"
    path_txt = out_dir / f"{slug}_text.mlpackage"

    img_mod = ImageEncoder(model).eval()
    tex_mod = TextEncoder(model).eval()
    example_img = torch.randn(1, 3, 256, 256)
    tok = open_clip.get_tokenizer(model_name)
    example_txt = tok(["export sanity check"]).to(torch.int32)

    img_ep = torch.export.export(img_mod, (example_img,), strict=False).run_decompositions({})
    txt_ep = torch.export.export(tex_mod, (example_txt,), strict=False).run_decompositions({})

    ml_img = ct.convert(
        img_ep,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
    )
    ml_txt = ct.convert(
        txt_ep,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
    )
    ml_img.save(str(path_img))
    ml_txt.save(str(path_txt))
    print(f"Saved image model: {path_img}")
    print(f"Saved text model:  {path_txt}")

    if args.for_app:
        vocab = out_dir / "bpe_simple_vocab_16e6.txt.gz"
        if not vocab.is_file():
            print(f"Downloading tokenizer to {vocab}...")
            urllib.request.urlretrieve(_CLIP_BPE_URL, vocab)

    if args.verify:
        mli = ct.models.MLModel(str(path_img), compute_units=ct.ComputeUnit.CPU_ONLY)
        mlt = ct.models.MLModel(str(path_txt), compute_units=ct.ComputeUnit.CPU_ONLY)
        rng = torch.rand(1, 3, 256, 256)
        tokens = tok(["a kitchen photo"]).numpy().astype(np.int32).reshape(1, -1)
        with torch.no_grad():
            pt_i = img_mod(rng).numpy().ravel()
            pt_t = tex_mod(torch.from_numpy(tokens.copy())).numpy().ravel()
        oi = list(mli.output_description)[0]
        ot = list(mlt.output_description)[0]
        ci = np.asarray(mli.predict({"image": rng.numpy().astype(np.float32)})[oi], dtype=np.float32).ravel()
        ctok = np.asarray(mlt.predict({"tokens": tokens})[ot], dtype=np.float32).ravel()

        def cos(a: np.ndarray, b: np.ndarray) -> float:
            a = a.astype(np.float64)
            b = b.astype(np.float64)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        print(f"Cosine image (PyTorch vs CoreML, rand in [0,1]): {cos(pt_i, ci):.6f}")
        print(f"Cosine text  (PyTorch vs CoreML):                 {cos(pt_t, ctok):.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
