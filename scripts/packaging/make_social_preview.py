#!/usr/bin/env python3
"""Generate GitHub social preview (1280x640) from app icon + darkroom theme.

Usage (from repo root):
  python scripts/packaging/make_social_preview.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
ICON = REPO / "icons" / "appicon.png"
OUT = REPO / "docs" / "images" / "social-preview.png"

W, H = 1280, 640

# Darkroom palette (keep in sync with src/theme.py)
VOID = (20, 18, 15)
SURFACE = (29, 26, 22)
RAISED = (39, 34, 25)
LINE = (58, 51, 42)
INK = (237, 231, 221)
INK_MUTED = (150, 137, 122)
EMBER = (217, 105, 30)


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend([
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ])
    else:
        candidates.extend([
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ])
    for path in candidates:
        p = Path(path)
        if p.is_file():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def _vertical_gradient(size: tuple[int, int], top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    w, h = size
    base = Image.new("RGB", (w, h), top)
    draw = ImageDraw.Draw(base)
    for y in range(h):
        t = y / max(h - 1, 1)
        rgb = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
        draw.line([(0, y), (w, y)], fill=rgb)
    return base


def compose() -> Image.Image:
    canvas = _vertical_gradient((W, H), VOID, SURFACE).convert("RGBA")
    draw = ImageDraw.Draw(canvas)

    # Subtle raised panel behind content
    pad_x, pad_y = 72, 64
    panel = (pad_x, pad_y, W - pad_x, H - pad_y)
    draw.rounded_rectangle(panel, radius=28, fill=RAISED + (255,), outline=LINE + (255,), width=2)

    if not ICON.is_file():
        raise FileNotFoundError(f"missing icon: {ICON}")

    icon_size = 360
    icon = Image.open(ICON).convert("RGBA")
    icon = icon.resize((icon_size, icon_size), Image.Resampling.LANCZOS)

    ix = panel[0] + 56
    iy = (H - icon_size) // 2
    canvas.alpha_composite(icon, (ix, iy))

    text_x = ix + icon_size + 56
    title_y = H // 2 - 92
    title_font = _load_font(72, bold=True)
    sub_font = _load_font(30, bold=False)

    draw.text((text_x, title_y), "RAWviewer", font=title_font, fill=INK)
    draw.text(
        (text_x, title_y + 92),
        "Fast RAW viewing & develop",
        font=sub_font,
        fill=INK_MUTED,
    )

    accent_y = title_y + 150
    draw.line([(text_x, accent_y), (text_x + 220, accent_y)], fill=EMBER, width=4)

    return canvas.convert("RGB")


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img = compose()
    img.save(OUT, format="PNG", optimize=True)
    if img.size != (W, H):
        print(f"[ERROR] unexpected size {img.size}", file=sys.stderr)
        return 1
    print(f"[OK] {OUT} {img.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
