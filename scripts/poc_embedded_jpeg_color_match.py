#!/usr/bin/env python3
"""
POC: use embedded JPEG as color reference to correct GPU-decoded RAW color.

Default input: D:\\Development\\RAW Image Data
Default output: <input>\\_color_match_poc

Writes per file:
  {stem}_embedded_ref.png
  {stem}_gpu_decode.png
  {stem}_gpu_matched.png
  {stem}_compare.png
"""

from __future__ import annotations

import argparse
import os
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


def _bayer_pattern_from_raw(raw) -> str:
    try:
        desc = bytes(raw.color_desc).decode("ascii", errors="ignore").upper()
        if len(desc) >= 4 and all(c in "RGB" for c in desc[:4]):
            return desc[:4]
    except Exception:
        pass
    return "RGGB"


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


def _gpu_decode_rgb(file_path: str, exif_data: dict | None, *, half_size: bool) -> np.ndarray | None:
    import rawpy
    from gpu_raw_processor import detect_gpu_backend, try_gpu_raw_decode

    backend = detect_gpu_backend()
    if backend == "cpu_only":
        print(f"  [skip GPU] no GPU backend (set RAWVIEWER_GPU_BACKEND or install torch/cupy)")
        return None

    with rawpy.imread(file_path) as raw:
        raw_array = raw.raw_image.copy()
        if half_size:
            raw_array = raw_array[::2, ::2]
        pattern = _bayer_pattern_from_raw(raw)
        rgb = try_gpu_raw_decode(file_path, raw_array, exif_data, raw_obj=raw)
        if rgb is not None:
            print(f"  GPU decode OK ({backend}, bayer={pattern}, {rgb.shape[1]}x{rgb.shape[0]})")
        return rgb


def _libraw_reference_rgb(file_path: str, *, half_size: bool) -> np.ndarray:
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
    max_edge: int,
    method: str,
    half_size: bool,
) -> bool:
    from embedded_color_match import (
        align_reference_to_source,
        make_comparison_strip,
        reinhard_color_transfer,
        reinhard_lab_color_transfer,
    )
    stem = file_path.stem
    print(f"\n[{file_path.name}]")

    embedded = _extract_embedded_jpeg(str(file_path), max_edge=max_edge)
    if embedded is None:
        print("  [skip] no embedded JPEG / preview")
        return False

    gpu_rgb = _gpu_decode_rgb(str(file_path), None, half_size=half_size)
    if gpu_rgb is None:
        print("  [skip] GPU decode failed")
        return False

    ref_aligned = align_reference_to_source(embedded, gpu_rgb)
    if method == "lab":
        matched = reinhard_lab_color_transfer(gpu_rgb, ref_aligned)
    else:
        matched = reinhard_color_transfer(gpu_rgb, ref_aligned)

    _save_png(output_dir / f"{stem}_embedded_ref.png", ref_aligned)
    _save_png(output_dir / f"{stem}_gpu_decode.png", gpu_rgb)
    _save_png(output_dir / f"{stem}_gpu_matched.png", matched)

    try:
        libraw_rgb = _libraw_reference_rgb(str(file_path), half_size=half_size)
        libraw_aligned = align_reference_to_source(libraw_rgb, gpu_rgb)
        compare = make_comparison_strip(
            (
                ("embedded", ref_aligned),
                ("gpu", gpu_rgb),
                ("matched", matched),
                ("libraw", libraw_aligned),
            )
        )
        _save_png(output_dir / f"{stem}_compare.png", compare)
    except Exception as exc:
        print(f"  [warn] compare strip without libraw: {exc}")
        compare = make_comparison_strip(
            (
                ("embedded", ref_aligned),
                ("gpu", gpu_rgb),
                ("matched", matched),
            )
        )
        _save_png(output_dir / f"{stem}_compare.png", compare)

    print(f"  wrote {output_dir / (stem + '_gpu_matched.png')}")
    return True


def iter_raw_files(root: Path) -> list[Path]:
    files = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in RAW_EXTENSIONS and not p.name.startswith(".")
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
        help="Output folder (default: <input>/_color_match_poc)",
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
        choices=("rgb", "lab"),
        default="rgb",
        help="Color transfer method (lab needs scikit-image)",
    )
    args = parser.parse_args()
    half_size = not args.full_size

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Input folder not found: {input_dir}", file=sys.stderr)
        return 1

    output_dir = Path(args.output) if args.output else input_dir / "_color_match_poc"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = iter_raw_files(input_dir)
    if args.limit > 0:
        files = files[: args.limit]

    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Files : {len(files)} (half_size={half_size}, method={args.method})")

    ok = 0
    fail = 0
    for path in files:
        if "_color_match_poc" in path.parts:
            continue
        try:
            if process_file(
                path,
                output_dir,
                max_edge=args.max_edge,
                method=args.method,
                half_size=half_size,
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
