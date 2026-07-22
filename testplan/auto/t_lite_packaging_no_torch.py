"""Lite packaging: no torch/kornia in Windows Lite pixi; GPU decode default off."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))


def _windows_scoped_deps(text: str) -> str:
    """Only the sections a Windows install actually resolves.

    These assertions are substring scans, so anything else in the file can
    trip them even though pixi never looks at it on Windows. Two real false
    positives came from exactly that: pixi.toml's "Torch-free GPU pipeline"
    comment, and the [target.osx-*] blocks' own onnxruntime entries -- both
    while the win-64 section was correctly stripped. Keep the shared tables
    and win-64 targets; drop comments and every other platform's targets.
    """
    kept: list[str] = []
    in_scope = True
    for raw in text.splitlines():
        line = raw.split("#", 1)[0]
        stripped = line.strip()
        if stripped.startswith("["):
            section = stripped.strip("[]")
            # Other platforms' target tables are inert on Windows.
            in_scope = not section.startswith("target.") or "win-64" in section
        if in_scope:
            kept.append(line)
    return "\n".join(kept)


def test_windows_lite_pixi_skips_torch_kornia() -> None:
    import build as build_mod

    out = build_mod._prepare_windows_pixi_manifest("cuda", profile="lite")
    text = _windows_scoped_deps(out.read_text(encoding="utf-8"))
    assert "torch" not in text.lower(), text
    assert "kornia" not in text.lower(), text
    # contrib build: cv2 is core to the Adjust panel, and cv2.mcc backs
    # ColorChecker Auto-Detect. Lite must never strip it.
    assert "opencv-contrib-python-headless" in text
    assert "lensfunpy" in text
    assert "tifffile" in text
    assert "huggingface_hub" not in text
    assert "onnxruntime" not in text


def test_windows_full_pixi_keeps_accel_stack() -> None:
    """Full keeps the GPU/AI stack that Lite strips.

    This used to assert torch/kornia were present, which passed only because
    pixi.toml's prose mentions them ("Torch-free GPU pipeline", "vs ~2.4 GB for
    the cu124 torch wheel") -- torch is declared nowhere in the manifest, so the
    assertion was vacuous and claimed the opposite of the project's actual
    torch-free design. Assert what really separates the profiles instead.
    """
    import build as build_mod

    out = build_mod._prepare_windows_pixi_manifest("cuda", profile="full")
    text = _windows_scoped_deps(out.read_text(encoding="utf-8"))
    for pkg in ("cupy-cuda12x", "onnxruntime-directml", "huggingface_hub", "requests"):
        assert pkg in text, f"Full build lost {pkg}"
    # Torch-free in both profiles: CuPy/ONNX replaced it (see pixi.toml).
    assert "torch" not in text.lower(), text


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
    test_windows_full_pixi_keeps_accel_stack()
    test_lite_profile_defaults_gpu_decode_off()
    print("PASS t_lite_packaging_no_torch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
