"""Color transfer helpers for GPU/LibRaw decode correction (POC, MIT-compatible)."""

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


def reinhard_lab_luminance_transfer(
    source: np.ndarray,
    reference: np.ndarray,
    *,
    chroma_strength: float = 0.35,
) -> np.ndarray:
    """
    Match lightness (Lab L) to reference; partially match a/b to avoid gray, desaturated output.

    ``chroma_strength`` in [0, 1]: 0 keeps source chroma, 1 fully matches reference a/b stats.
    """
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

    # Always match luminance statistics.
    for channel in (0,):
        src_mean = float(out_lab[:, :, channel].mean())
        src_std = float(out_lab[:, :, channel].std()) + 1e-6
        ref_mean = float(ref_lab[:, :, channel].mean())
        ref_std = float(ref_lab[:, :, channel].std()) + 1e-6
        out_lab[:, :, channel] = (out_lab[:, :, channel] - src_mean) * (ref_std / src_std) + ref_mean

    strength = float(np.clip(chroma_strength, 0.0, 1.0))
    if strength > 0.0:
        for channel in (1, 2):
            src_mean = float(out_lab[:, :, channel].mean())
            src_std = float(out_lab[:, :, channel].std()) + 1e-6
            ref_mean = float(ref_lab[:, :, channel].mean())
            ref_std = float(ref_lab[:, :, channel].std()) + 1e-6
            matched = (out_lab[:, :, channel] - src_mean) * (ref_std / src_std) + ref_mean
            out_lab[:, :, channel] = (1.0 - strength) * out_lab[:, :, channel] + strength * matched

    rgb = color.lab2rgb(out_lab)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def affine_rgb_transfer(
    source: np.ndarray,
    reference: np.ndarray,
    *,
    sample_count: int = 65536,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Fit a per-image 3x3 color matrix + bias mapping source RGB -> reference RGB.

    Best when both images are the same scene/geometry (rawpy ref vs GPU decode).
    """
    ref = reference
    if source.shape[:2] != reference.shape[:2]:
        ref = align_reference_to_source(reference, source)

    src = source[:, :, :3].astype(np.float64)
    ref_f = ref[:, :, :3].astype(np.float64)
    flat_src = src.reshape(-1, 3)
    flat_ref = ref_f.reshape(-1, 3)

    gen = rng or np.random.default_rng(0)
    if flat_src.shape[0] > sample_count:
        idx = gen.choice(flat_src.shape[0], sample_count, replace=False)
        fit_src = flat_src[idx]
        fit_ref = flat_ref[idx]
    else:
        fit_src = flat_src
        fit_ref = flat_ref

    design = np.hstack([fit_src, np.ones((fit_src.shape[0], 1), dtype=np.float64)])
    coeffs, _, _, _ = np.linalg.lstsq(design, fit_ref, rcond=None)
    matrix = coeffs[:3, :].T
    bias = coeffs[3, :]

    out = flat_src @ matrix.T + bias
    out = out.reshape(src.shape)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def diagonal_gain_match(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Scale R/G/B independently so channel means match the reference.

    Removes global color cast (e.g. red hue) before heavier transfer methods.
    """
    ref = reference
    if source.shape[:2] != reference.shape[:2]:
        ref = align_reference_to_source(reference, source)

    src = source[:, :, :3].astype(np.float32)
    ref_f = ref[:, :, :3].astype(np.float32)
    out = src.copy()
    for channel in range(3):
        src_mean = float(out[:, :, channel].mean()) + 1e-6
        ref_mean = float(ref_f[:, :, channel].mean())
        out[:, :, channel] *= ref_mean / src_mean
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def apply_color_transfer(
    source: np.ndarray,
    reference: np.ndarray,
    method: str,
    *,
    chroma_strength: float = 0.35,
    pre_diagonal_gain: bool = False,
) -> np.ndarray:
    """Dispatch to the selected transfer method."""
    src = diagonal_gain_match(source, reference) if pre_diagonal_gain else source
    key = (method or "affine").strip().lower()
    if key in ("diagonal", "gain"):
        return diagonal_gain_match(source, reference)
    if key == "affine":
        return affine_rgb_transfer(src, reference)
    if key in ("affine-diagonal", "affine_diagonal"):
        balanced = diagonal_gain_match(source, reference)
        return affine_rgb_transfer(balanced, reference)
    if key in ("lab-l", "lab_l", "luminance"):
        return reinhard_lab_luminance_transfer(src, reference, chroma_strength=chroma_strength)
    if key == "lab":
        return reinhard_lab_color_transfer(src, reference)
    if key == "rgb":
        return reinhard_color_transfer(src, reference)
    raise ValueError(f"Unknown color transfer method: {method!r}")


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
