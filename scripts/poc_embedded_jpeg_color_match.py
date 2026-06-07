#!/usr/bin/env python3
"""
POC: correct GPU-decoded RAW color using a reference from the same RAW file.

Reference modes:
  rawpy     LibRaw/rawpy postprocess (default; raw-to-raw, same geometry)
  embedded  Camera embedded JPEG preview

Default input : D:\\Development\\RAW Image Data
Default output: <input>\\_rawpy_color_match_poc  (rawpy mode)
                <input>\\_color_match_poc       (embedded mode)

Writes per file:
  {stem}_reference.png
  {stem}_gpu_decode.png
  {stem}_gpu_matched.png
  {stem}_compare.png
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

RAW_EXTENSIONS = {
    ".arw", ".cr2", ".cr3", ".nef", ".dng", ".raf", ".rw2", ".orf", ".3fr", ".pef", ".srw",
}


def _save_png(path: Path, rgb: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb[:, :, :3]).save(path, format="PNG", optimize=True)


def _gpu_decode_rgb(file_path: str, *, half_size: bool) -> np.ndarray | None:
    import rawpy
    from gpu_raw_processor import bayer_pattern_from_raw, detect_gpu_backend, try_gpu_raw_decode

    backend = detect_gpu_backend()
    if backend == "cpu_only":
        print("  [skip GPU] no GPU backend (set RAWVIEWER_GPU_BACKEND or install torch/cupy)")
        return None

    with rawpy.imread(file_path) as raw:
        raw_array = raw.raw_image.copy()
        if half_size:
            raw_array = raw_array[::2, ::2]
        pattern = bayer_pattern_from_raw(raw)
        rgb = try_gpu_raw_decode(file_path, raw_array, None, raw_obj=raw)
        if rgb is not None:
            means = rgb[:, :, 0].mean(), rgb[:, :, 1].mean(), rgb[:, :, 2].mean()
            print(
                f"  GPU decode OK ({backend}, bayer={pattern}, {rgb.shape[1]}x{rgb.shape[0]}, "
                f"RGB means={means[0]:.1f}/{means[1]:.1f}/{means[2]:.1f})"
            )
        return rgb


def _channel_means(rgb: np.ndarray) -> tuple[float, float, float]:
    return float(rgb[:, :, 0].mean()), float(rgb[:, :, 1].mean()), float(rgb[:, :, 2].mean())


def _extract_embedded_jpeg(file_path: str, *, max_edge: int) -> np.ndarray | None:
    from enhanced_raw_processor import ThumbnailExtractor

    extractor = ThumbnailExtractor()
    thumb = extractor.extract_thumbnail_from_raw(
        file_path,
        max_size=max_edge,
        allow_scan_fallback=True,
    )
    if thumb is None:
        return None
    if thumb.ndim == 2:
        thumb = np.stack([thumb, thumb, thumb], axis=-1)
    return thumb[:, :, :3].astype(np.uint8)


def _rawpy_reference_rgb(file_path: str, *, half_size: bool) -> np.ndarray:
    """LibRaw/rawpy decode used as the color reference (raw-to-raw POC)."""
    import rawpy

    with rawpy.imread(file_path) as raw:
        return raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            gamma=(2.222, 4.5),
            output_bps=8,
            user_flip=0,
            half_size=half_size,
        )


def process_file(
    file_path: Path,
    output_dir: Path,
    *,
    reference_mode: str,
    max_edge: int,
    method: str,
    half_size: bool,
    chroma_strength: float,
) -> bool:
    from embedded_color_match import (
        align_reference_to_source,
        apply_color_transfer,
        diagonal_gain_match,
        make_comparison_strip,
    )

    stem = file_path.stem
    print(f"\n[{file_path.name}] ref={reference_mode} method={method}")

    gpu_rgb = _gpu_decode_rgb(str(file_path), half_size=half_size)
    if gpu_rgb is None:
        print("  [skip] GPU decode failed")
        return False

    if reference_mode == "rawpy":
        reference = _rawpy_reference_rgb(str(file_path), half_size=half_size)
        ref_label = "rawpy"
    elif reference_mode == "embedded":
        reference = _extract_embedded_jpeg(str(file_path), max_edge=max_edge)
        if reference is None:
            print("  [skip] no embedded JPEG / preview")
            return False
        ref_label = "embedded"
    else:
        raise ValueError(f"Unknown reference mode: {reference_mode}")

    ref_aligned = align_reference_to_source(reference, gpu_rgb)
    ref_means = _channel_means(ref_aligned)
    print(f"  reference RGB means={ref_means[0]:.1f}/{ref_means[1]:.1f}/{ref_means[2]:.1f}")

    gpu_balanced = diagonal_gain_match(gpu_rgb, ref_aligned)
    balanced_means = _channel_means(gpu_balanced)
    print(f"  diagonal gain RGB means={balanced_means[0]:.1f}/{balanced_means[1]:.1f}/{balanced_means[2]:.1f}")

    matched = apply_color_transfer(
        gpu_rgb,
        ref_aligned,
        method,
        chroma_strength=chroma_strength,
    )
    matched_means = _channel_means(matched)
    print(f"  matched RGB means={matched_means[0]:.1f}/{matched_means[1]:.1f}/{matched_means[2]:.1f}")

    _save_png(output_dir / f"{stem}_reference.png", ref_aligned)
    _save_png(output_dir / f"{stem}_gpu_decode.png", gpu_rgb)
    _save_png(output_dir / f"{stem}_gpu_balanced.png", gpu_balanced)
    _save_png(output_dir / f"{stem}_gpu_matched.png", matched)

    compare = make_comparison_strip(
        (
            (ref_label, ref_aligned),
            ("gpu", gpu_rgb),
            ("gain", gpu_balanced),
            ("matched", matched),
        )
    )
    _save_png(output_dir / f"{stem}_compare.png", compare)

    print(f"  wrote {output_dir / (stem + '_gpu_matched.png')}")
    return True


def iter_raw_files(root: Path) -> list[Path]:
    skip_dirs = {"_color_match_poc", "_rawpy_color_match_poc", "_rawpy_color_match_poc_v2"}
    files = [
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in RAW_EXTENSIONS
        and not p.name.startswith(".")
        and not any(part in skip_dirs for part in p.parts)
    ]
    return sorted(files, key=lambda p: p.name.lower())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=r"D:\Development\RAW Image Data",
        help="Folder containing RAW test images",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output folder (default depends on --reference mode)",
    )
    parser.add_argument(
        "--reference",
        choices=("rawpy", "embedded"),
        default="rawpy",
        help="Color reference source (default: rawpy LibRaw decode)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process at most N files (0 = all)")
    parser.add_argument(
        "--max-edge",
        type=int,
        default=2048,
        help="Max long edge when extracting embedded JPEG reference",
    )
    parser.add_argument(
        "--full-size",
        action="store_true",
        help="Decode full-resolution RAW (slow; default is half-size mosaic for POC speed)",
    )
    parser.add_argument(
        "--method",
        choices=("affine-diagonal", "affine", "diagonal", "lab-l", "lab", "rgb"),
        default="affine-diagonal",
        help="Transfer method (default: diagonal gain then affine)",
    )
    parser.add_argument(
        "--chroma-strength",
        type=float,
        default=0.35,
        help="For lab-l: how much to match a/b chroma vs keep GPU chroma (0-1)",
    )
    args = parser.parse_args()
    half_size = not args.full_size

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Input folder not found: {input_dir}", file=sys.stderr)
        return 1

    if args.output:
        output_dir = Path(args.output)
    elif args.reference == "rawpy":
        output_dir = input_dir / "_rawpy_color_match_poc_v2"
    else:
        output_dir = input_dir / "_color_match_poc"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = iter_raw_files(input_dir)
    if args.limit > 0:
        files = files[: args.limit]

    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print(
        f"Files : {len(files)} "
        f"(reference={args.reference}, half_size={half_size}, method={args.method})"
    )

    ok = 0
    fail = 0
    for path in files:
        try:
            if process_file(
                path,
                output_dir,
                reference_mode=args.reference,
                max_edge=args.max_edge,
                method=args.method,
                half_size=half_size,
                chroma_strength=args.chroma_strength,
            ):
                ok += 1
            else:
                fail += 1
        except Exception as exc:
            fail += 1
            print(f"  [error] {exc}")
            traceback.print_exc()

    print(f"\nDone: {ok} succeeded, {fail} skipped/failed -> {output_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
