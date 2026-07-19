#!/usr/bin/env python3
"""Rebuild appicon.png / .ico / .icns and favicon from concept masters.

Usage (from repo root, Pixi env recommended):
  python scripts/packaging/make_app_icons.py
"""
from __future__ import annotations

import io
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
ICONS = REPO / "icons"
# Full design (splash / README / app) and simplified favicon masters
MASTER = ICONS / "appicon-master.png"
FAV_MASTER = ICONS / "favicon-master.png"


def _png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def clean_rounded_icon(im: Image.Image) -> Image.Image:
    """Strip light AA fringe that shows as a white border on Windows ICO/taskbar.

    Keeps opaque interior accents (yellow star, cream curve). Only removes
    semi-transparent light halo / near-invisible dust outside the plate.
    """
    a = np.array(im.convert("RGBA"), dtype=np.uint8)
    rgb = a[:, :, :3].astype(np.float32)
    alpha = a[:, :, 3].astype(np.float32)
    lum = rgb.mean(axis=2)
    fringe = (lum >= 100.0) & (alpha > 0) & (alpha < 48)
    dust = (alpha > 0) & (alpha < 10)
    kill = fringe | dust
    a[kill, :] = 0
    a[a[:, :, 3] == 0, :3] = 0
    return Image.fromarray(a, "RGBA")


def _macos_icon_background(im: Image.Image) -> tuple[int, int, int]:
    """Pick the dark plate color used outside the pre-rounded artwork."""
    arr = np.array(im.convert("RGBA"))
    opaque = arr[:, :, 3] > 200
    if not opaque.any():
        return (29, 22, 18)
    rgb = arr[opaque, :3].astype(np.float32)
    lum = rgb.mean(axis=1)
    dark = rgb[lum <= np.percentile(lum, 25)]
    if len(dark) == 0:
        dark = rgb
    return tuple(int(v) for v in dark.mean(axis=0))


def flatten_macos_dock_icon(im: Image.Image) -> Image.Image:
    """Fill a square canvas so macOS Dock does not show a white halo.

    App Store / Dock icons must be full-bleed squares. Pre-rounded artwork with
    transparent corners gets a light fringe once macOS applies its squircle mask.
    """
    cleaned = clean_rounded_icon(im.convert("RGBA"))
    bg = _macos_icon_background(cleaned)
    canvas = Image.new("RGBA", cleaned.size, (*bg, 255))
    canvas.alpha_composite(cleaned)
    return canvas


def write_png_ico(path: Path, src: Image.Image, sizes: list[int]) -> None:
    images = [
        clean_rounded_icon(src.resize((s, s), Image.Resampling.LANCZOS)) for s in sizes
    ]
    pngs = [_png_bytes(im) for im in images]
    n = len(images)
    header = struct.pack("<HHH", 0, 1, n)
    offset = 6 + n * 16
    entries = b""
    data = b""
    for im, png in zip(images, pngs):
        w, h = im.size
        entry_w = 0 if w >= 256 else w
        entry_h = 0 if h >= 256 else h
        entries += struct.pack(
            "<BBBBHHII", entry_w, entry_h, 0, 0, 1, 32, len(png), offset
        )
        data += png
        offset += len(png)
    path.write_bytes(header + entries + data)


def build_icns(master: Image.Image, out_icns: Path) -> None:
    dock_master = flatten_macos_dock_icon(master)
    iconset = REPO / "build" / "appicon.iconset"
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir(parents=True)
    pairs = [
        ("icon_16x16.png", 16),
        ("icon_16x16@2x.png", 32),
        ("icon_32x32.png", 32),
        ("icon_32x32@2x.png", 64),
        ("icon_128x128.png", 128),
        ("icon_128x128@2x.png", 256),
        ("icon_256x256.png", 256),
        ("icon_256x256@2x.png", 512),
        ("icon_512x512.png", 512),
        ("icon_512x512@2x.png", 1024),
    ]
    for name, size in pairs:
        dock_master.resize((size, size), Image.Resampling.LANCZOS).save(
            iconset / name
        )
    subprocess.run(
        ["iconutil", "-c", "icns", str(iconset), "-o", str(out_icns)],
        check=True,
    )


def main() -> int:
    if not MASTER.is_file():
        print(f"[ERROR] missing master: {MASTER}", file=sys.stderr)
        return 1
    if not FAV_MASTER.is_file():
        print(f"[ERROR] missing favicon master: {FAV_MASTER}", file=sys.stderr)
        return 1

    master = clean_rounded_icon(Image.open(MASTER).convert("RGBA"))
    fav = clean_rounded_icon(Image.open(FAV_MASTER).convert("RGBA"))

    # Hi-res PNG for splash + README
    hi = clean_rounded_icon(master.resize((2048, 2048), Image.Resampling.LANCZOS))
    hi.save(ICONS / "appicon.png", optimize=True)
    print(f"[OK] {ICONS / 'appicon.png'} {hi.size}")

    write_png_ico(ICONS / "appicon.ico", master, [16, 24, 32, 48, 64, 128, 256])
    print(f"[OK] {ICONS / 'appicon.ico'}")

    write_png_ico(ICONS / "favicon.ico", fav, [16, 32, 48])
    clean_rounded_icon(fav.resize((64, 64), Image.Resampling.LANCZOS)).save(
        ICONS / "favicon.png", optimize=True
    )
    print(f"[OK] {ICONS / 'favicon.ico'} + favicon.png")

    if sys.platform == "darwin":
        build_icns(master, ICONS / "appicon.icns")
        print(f"[OK] {ICONS / 'appicon.icns'}")
    else:
        print("[SKIP] appicon.icns (iconutil is macOS-only)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
