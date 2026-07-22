"""Lite packaging: no torch/kornia in Windows Lite pixi; GPU decode default off."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))


def _strip_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def test_windows_lite_pixi_skips_torch_kornia() -> None:
    import build as build_mod

    out = build_mod._prepare_windows_pixi_manifest("cuda", profile="lite")
    # Comments are prose, not dependencies -- pixi.toml's "Torch-free GPU
    # pipeline" note otherwise trips the torch assertion below.
    text = _strip_comments(out.read_text(encoding="utf-8"))
    assert "torch" not in text.lower(), text
    assert "kornia" not in text.lower(), text
    # contrib build: cv2 is core to the Adjust panel, and cv2.mcc backs
    # ColorChecker Auto-Detect. Lite must never strip it.
    assert "opencv-contrib-python-headless" in text
    assert "lensfunpy" in text
    assert "tifffile" in text
    assert "huggingface_hub" not in text
    assert "onnxruntime" not in text


def test_windows_full_pixi_keeps_torch() -> None:
    import build as build_mod

    out = build_mod._prepare_windows_pixi_manifest("cuda", profile="full")
    text = out.read_text(encoding="utf-8")
    assert "torch" in text
    assert "kornia" in text


def test_lite_profile_defaults_gpu_decode_off() -> None:
    # Isolate from ambient env / baked PROFILE.
    for key in (
        "RAWVIEWER_BUILD_PROFILE",
        "RAWVIEWER_PREFER_GPU_DECODE",
        "RAWVIEWER_ENABLE_SEMANTIC_SEARCH",
        "RAWVIEWER_ENABLE_FACE_SCAN",
    ):
        os.environ.pop(key, None)

    os.environ["RAWVIEWER_BUILD_PROFILE"] = "lite"
    from rawviewer_profile import apply_profile_runtime_defaults, is_lite_build

    assert is_lite_build()
    apply_profile_runtime_defaults()
    assert os.environ.get("RAWVIEWER_PREFER_GPU_DECODE") == "0"
    assert os.environ.get("RAWVIEWER_ENABLE_SEMANTIC_SEARCH") == "0"


def main() -> int:
    test_windows_lite_pixi_skips_torch_kornia()
    test_windows_full_pixi_keeps_torch()
    test_lite_profile_defaults_gpu_decode_off()
    print("PASS t_lite_packaging_no_torch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
