"""Embedded-JPEG-guided color transfer for GPU/LibRaw decode (POC, MIT-compatible)."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _resize_rgb(image: np.ndarray, width: int, height: int) -> np.ndarray:
    from PIL import Image

    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Expected HxWx3 RGB image")
    pil = Image.fromarray(image[:, :, :3])
    resized = pil.resize((max(1, width), max(1, height)), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.uint8)


def align_reference_to_source(reference: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Resize reference RGB to match source height x width."""
    sh, sw = source.shape[:2]
    rh, rw = reference.shape[:2]
    if (rh, rw) == (sh, sw):
        return reference[:, :, :3].copy()
    return _resize_rgb(reference, sw, sh)


def reinhard_color_transfer(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Reinhard-style per-channel mean/std transfer in RGB.

    ``source`` is the decode to correct; ``reference`` is the embedded JPEG look.
    """
    src = source[:, :, :3].astype(np.float32)
    ref = reference[:, :, :3].astype(np.float32)
    if src.shape[:2] != ref.shape[:2]:
        ref = align_reference_to_source(ref.astype(np.uint8), source).astype(np.float32)

    out = src.copy()
    for channel in range(3):
        src_mean = float(out[:, :, channel].mean())
        src_std = float(out[:, :, channel].std()) + 1e-6
        ref_mean = float(ref[:, :, channel].mean())
        ref_std = float(ref[:, :, channel].std()) + 1e-6
        out[:, :, channel] = (out[:, :, channel] - src_mean) * (ref_std / src_std) + ref_mean
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def reinhard_lab_color_transfer(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Reinhard transfer in CIELAB (closer to the classic paper)."""
    try:
        from skimage import color
    except ImportError:
        return reinhard_color_transfer(source, reference)

    ref = reference
    if source.shape[:2] != reference.shape[:2]:
        ref = align_reference_to_source(reference, source)

    src_lab = color.rgb2lab(source[:, :, :3].astype(np.float32) / 255.0)
    ref_lab = color.rgb2lab(ref[:, :, :3].astype(np.float32) / 255.0)
    out_lab = src_lab.copy()
    for channel in range(3):
        src_mean = float(out_lab[:, :, channel].mean())
        src_std = float(out_lab[:, :, channel].std()) + 1e-6
        ref_mean = float(ref_lab[:, :, channel].mean())
        ref_std = float(ref_lab[:, :, channel].std()) + 1e-6
        out_lab[:, :, channel] = (out_lab[:, :, channel] - src_mean) * (ref_std / src_std) + ref_mean
    rgb = color.lab2rgb(out_lab)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def make_comparison_strip(
    panels: Tuple[Tuple[str, np.ndarray], ...],
    *,
    gap: int = 8,
    label_height: int = 28,
) -> np.ndarray:
    """Stack labeled panels horizontally for quick visual review."""
    from PIL import Image, ImageDraw, ImageFont

    imgs = []
    max_h = 0
    total_w = gap
    for _label, arr in panels:
        h, w = arr.shape[:2]
        max_h = max(max_h, h)
        total_w += w + gap

    canvas = Image.new("RGB", (total_w, max_h + label_height), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    x = gap
    for label, arr in panels:
        h, w = arr.shape[:2]
        pil = Image.fromarray(arr[:, :, :3])
        if h != max_h:
            scale = max_h / h
            pil = pil.resize((int(w * scale), max_h), Image.Resampling.LANCZOS)
            w = pil.width
        canvas.paste(pil, (x, label_height))
        draw.text((x + 6, 6), label, fill=(224, 224, 224))
        x += w + gap
    return np.asarray(canvas, dtype=np.uint8)
