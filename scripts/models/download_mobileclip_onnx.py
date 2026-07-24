#!/usr/bin/env python3
"""
Download MobileCLIP2-B ONNX models and the SCUNet AI denoise ONNX for
bundling into RAWviewer.

Installer progress: @RAWVIEWER_PROGRESS pct=N message=Downloading... N%
Verbose details go to [INFO] log lines only.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO_ID = "plhery/mobileclip2-onnx"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "mobileclip_onnx"
# Real SCUNet (scunet_color_real_psnr, official KAIR weights exported to ONNX),
# stored in the repo via Git LFS. Pinned SHA-256 must match the LFS object.
DENOISE_MODEL_URL = "https://github.com/markyip/RAWviewer/raw/development/models/scunet.onnx"
DENOISE_MODEL_SHA256 = "d426f34a1b71670e43e6219161a3258fe65b03ece946467ae842737a55ef0849"
DOWNLOAD_RETRIES = 3
RETRY_DELAY_SEC = 3

VISION_PCT_END = 57
TEXT_PCT_END = 96
DENOISE_PCT_END = 99


def _human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _describe_error(exc: BaseException) -> str:
    text = str(exc).lower()
    if "timeout" in text or "timed out" in text:
        return "Connection timed out — check network or VPN"
    if "proxy" in text:
        return "Proxy error — verify HTTP/HTTPS proxy settings"
    if "ssl" in text or "certificate" in text:
        return "SSL/TLS error — check HTTPS proxy or antivirus scanning"
    if "no space" in text or "disk" in text:
        return "Disk full or not enough space to save models"
    return str(exc)


def _download_with_retry(label: str, download_fn) -> int:
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            print(f"[INFO] {label} (attempt {attempt}/{DOWNLOAD_RETRIES})", flush=True)
            download_fn()
            return 0
        except Exception as exc:
            print(f"[ERROR] {label} failed: {_describe_error(exc)}", flush=True)
            if attempt < DOWNLOAD_RETRIES:
                print(f"[INFO] Retrying in {RETRY_DELAY_SEC}s...", flush=True)
                time.sleep(RETRY_DELAY_SEC)
            else:
                print(
                    "[ERROR] Tip: check firewall, VPN, or corporate proxy; "
                    "Hugging Face and Azure CDN must be reachable.",
                    flush=True,
                )
                return 1
    return 1


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_url_with_progress(
    url: str,
    dest: Path,
    *,
    stage_start: int,
    stage_end: int,
    emit,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "RAWviewer-Setup/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            chunk_size = 1024 * 256
            with open(tmp, "wb") as out:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if total > 0:
                        frac = min(1.0, done / total)
                        emit(stage_start + int(frac * (stage_end - stage_start)))
                    else:
                        emit(stage_end)
        os.replace(tmp, dest)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
    emit(stage_end)


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from mobileclip_download_progress import (
        emit_installer_progress,
        make_byte_progress_tqdm,
        reset_installer_progress_state,
    )

    try:
        import requests  # noqa: F401
    except ImportError:
        print("[ERROR] Missing 'requests'. Run: pip install requests")
        return 1
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        print(f"[ERROR] Missing 'huggingface-hub': {exc}")
        return 1

    print(f"[INFO] MobileCLIP2-B ONNX -> {MODELS_DIR}", flush=True)
    reset_installer_progress_state()
    emit_installer_progress(0)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    files_to_download = [
        ("onnx/b/vision_model.onnx", "image_encoder.onnx", 0, VISION_PCT_END),
        ("onnx/b/text_model.onnx", "text_encoder.onnx", VISION_PCT_END, TEXT_PCT_END),
    ]

    for remote_path, local_name, stage_start, stage_end in files_to_download:
        def _fetch(
            remote_path=remote_path,
            local_name=local_name,
            stage_start=stage_start,
            stage_end=stage_end,
        ):
            print(f"[INFO] Fetching {local_name} ({stage_start}-{stage_end}%)", flush=True)
            reset_installer_progress_state()
            emit_installer_progress(stage_start)
            tqdm_class = make_byte_progress_tqdm(
                stage_start, stage_end, emit_installer_progress, silent=True
            )
            hf_hub_download(
                repo_id=REPO_ID,
                filename=remote_path,
                local_dir=str(MODELS_DIR),
                tqdm_class=tqdm_class,
            )
            downloaded_path = MODELS_DIR / remote_path
            target_path = MODELS_DIR / local_name
            if downloaded_path.exists():
                if target_path.exists():
                    target_path.unlink()
                downloaded_path.rename(target_path)
            emit_installer_progress(stage_end)

        rc = _download_with_retry(f"{local_name}", _fetch)
        if rc != 0:
            return rc

    if (MODELS_DIR / "onnx").exists():
        import shutil

        shutil.rmtree(MODELS_DIR / "onnx")

    denoise_model_path = MODELS_DIR.parent / "scunet.onnx"
    legacy_denoise_model_path = MODELS_DIR.parent / "restormer.onnx"
    if not denoise_model_path.exists() and not legacy_denoise_model_path.exists():
        def _fetch_denoise_model():
            print("[INFO] Fetching AI denoise model (SCUNet ONNX)", flush=True)
            _download_url_with_progress(
                DENOISE_MODEL_URL,
                denoise_model_path,
                stage_start=TEXT_PCT_END,
                stage_end=DENOISE_PCT_END,
                emit=emit_installer_progress,
            )
            if _sha256_of_file(denoise_model_path).lower() != DENOISE_MODEL_SHA256.lower():
                denoise_model_path.unlink(missing_ok=True)
                raise RuntimeError("denoise model failed SHA-256 verification")

        rc = _download_with_retry("denoise model", _fetch_denoise_model)
        if rc != 0:
            return rc
    else:
        emit_installer_progress(DENOISE_PCT_END)

    tokenizer_path = MODELS_DIR / "bpe_simple_vocab_16e6.txt.gz"
    if not tokenizer_path.exists():
        def _fetch_tokenizer():
            print("[INFO] Fetching tokenizer", flush=True)
            _download_url_with_progress(
                "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz",
                tokenizer_path,
                stage_start=DENOISE_PCT_END,
                stage_end=100,
                emit=emit_installer_progress,
            )

        rc = _download_with_retry("tokenizer", _fetch_tokenizer)
        if rc != 0:
            return rc
    else:
        emit_installer_progress(100)

    emit_installer_progress(100)
    print("[SUCCESS] MobileCLIP2 models, denoise model, and tokenizer ready.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
