#!/usr/bin/env python3
"""Fetch the AI mask ONNX models into models/ and print their SHA-256.

Run once on a dev machine, then commit the resulting .onnx files. The
Windows installer payload ships models/ directly; macOS/Linux download
them on first use via raw_ai_masks.ensure_model_downloaded, which needs
the hashes printed here pasted into raw_ai_masks._MODELS.

    pixi run python3 scripts/fetch_ai_mask_models.py
    pixi run python3 scripts/fetch_ai_mask_models.py --only subject

Sources (all commercially licensed):
    birefnet.onnx            BiRefNet, MIT
    skyseg.onnx              U^2-Net sky segmentation, MIT
    mobilesam_encoder.onnx   MobileSAM image encoder, Apache 2.0
    mobilesam_decoder.onnx   MobileSAM prompt decoder, Apache 2.0

These URLs point at community ONNX exports, which move around more than
the source repos do. If one 404s, search Hugging Face for the model name
plus "onnx" and drop the direct file URL in below -- raw_ai_masks reads
the input size off the graph, so a differently-sized export still works
without a code change.
"""
import argparse
import hashlib
import os
import sys
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(REPO_ROOT, "models")

SOURCES = {
    "subject": (
        "birefnet.onnx",
        "https://huggingface.co/onnx-community/BiRefNet_lite/resolve/main/onnx/model.onnx",
    ),
    "sky": (
        "skyseg.onnx",
        "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx",
    ),
    "sam_encoder": (
        "mobilesam_encoder.onnx",
        "https://huggingface.co/vietanhdev/segment-anything-onnx-models/resolve/main/mobile_sam.encoder.onnx",
    ),
    "sam_decoder": (
        "mobilesam_decoder.onnx",
        "https://huggingface.co/vietanhdev/segment-anything-onnx-models/resolve/main/mobile_sam.decoder.onnx",
    ),
}


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, dest):
    tmp = dest + ".part"
    print(f"  fetching {url}")
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as fh:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                pct = 100.0 * done / total
                sys.stdout.write(f"\r  {done / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)")
                sys.stdout.flush()
    if total:
        sys.stdout.write("\n")
    os.replace(tmp, dest)


def verify_loads(path):
    """Load the graph so a bad export fails here, not in the app."""
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        ins = [(i.name, i.shape) for i in session.get_inputs()]
        outs = [(o.name, o.shape) for o in session.get_outputs()]
        print(f"  inputs:  {ins}")
        print(f"  outputs: {outs}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] could not load: {exc}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(SOURCES), help="fetch a single model")
    ap.add_argument("--force", action="store_true", help="re-download if present")
    args = ap.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    kinds = [args.only] if args.only else list(SOURCES)

    hashes = {}
    failed = []
    for kind in kinds:
        filename, url = SOURCES[kind]
        dest = os.path.join(MODELS_DIR, filename)
        print(f"\n{kind} -> models/{filename}")
        if os.path.exists(dest) and not args.force:
            print("  already present (use --force to re-download)")
        else:
            try:
                download(url, dest)
            except Exception as exc:  # noqa: BLE001
                print(f"  [ERROR] download failed: {exc}")
                failed.append(kind)
                continue
        verify_loads(dest)
        digest = sha256_of(dest)
        hashes[kind] = digest
        size_mb = os.path.getsize(dest) / 1e6
        print(f"  {size_mb:.1f} MB  sha256={digest}")

    if hashes:
        print("\nPaste into src/raw_ai_masks.py _MODELS:")
        for kind, digest in hashes.items():
            print(f'    "{kind}": ... "sha256": "{digest}",')

    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
