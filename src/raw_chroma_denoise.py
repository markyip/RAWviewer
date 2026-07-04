"""Chroma and luminance noise reduction for scene-linear RGB edit buffers.

Both filters operate on a YCbCr split (chroma denoise touches Cb/Cr only;
luma denoise touches Y only) and use an edge-aware bilateral filter directly
on float32 -- no uint8 round-trip, no per-channel patch search. This replaced
an NLM (non-local means) chroma filter that was both slower (see
docs/ADJUST_LINEAR_PIPELINE.md Performance review) and required an intermediate
uint8 quantization step that had its own rounding bug (green-cast regression,
same doc). Benchmarked ~19-22x faster than the old NLM path at preview/export
sizes, with comparable noise reduction and negligible edge bleed (bilateral
filter transition width across a hard edge stays ~0px even at max strength).
"""

from __future__ import annotations

import os

import numpy as np


def _rgb_to_ycbcr(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cb = -0.1146 * r - 0.3854 * g + 0.5000 * b + 0.5
    cr = 0.5000 * r - 0.4542 * g - 0.0458 * b + 0.5
    return y, cb, cr


def _ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    cb0 = cb - 0.5
    cr0 = cr - 0.5
    r = y + 1.5748 * cr0
    g = y - 0.1873 * cb0 - 0.4681 * cr0
    b = y + 1.8556 * cb0
    return np.stack([r, g, b], axis=-1)


def _bilateral_channel(channel: np.ndarray, sigma_color: float, *, preview: bool) -> np.ndarray:
    import cv2

    d = 5 if preview else 7
    return cv2.bilateralFilter(
        np.ascontiguousarray(channel, dtype=np.float32),
        d,
        float(sigma_color),
        float(d),
    )


def _downsample_blur_upsample(channel: np.ndarray, factor: int, blur_sigma: float) -> np.ndarray:
    """
    Chroma-subsampling-style noise reduction: shrink, blur, grow back.

    A bilateral filter's kernel (5-7px diameter here) only ever sees noise
    narrower than itself. Real high-ISO sensor color noise is often blotchy
    -- spatially correlated over several pixels from Bayer demosaic
    interpolation -- which a kernel that size barely touches (measured ~7%
    std reduction on noise with a several-pixel correlation length, vs ~77%
    on pixel-independent noise; see docs/ADJUST_LINEAR_PIPELINE.md). Shrinking
    first makes a much larger effective blur radius nearly free, and is
    perceptually safe for chroma specifically because human vision resolves
    color at far lower spatial resolution than luminance (the same premise
    JPEG/video 4:2:0 chroma subsampling relies on).
    """
    import cv2

    h, w = channel.shape
    sh, sw = max(1, h // factor), max(1, w // factor)
    small = cv2.resize(
        np.ascontiguousarray(channel, dtype=np.float32), (sw, sh), interpolation=cv2.INTER_AREA
    )
    if blur_sigma > 1e-3:
        small = cv2.GaussianBlur(small, (0, 0), blur_sigma)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _luma_edge_weight(y: np.ndarray, *, soft: float = 0.04) -> np.ndarray:
    """0 in flat regions, -> 1 near a real luminance edge (Sobel gradient, soft-knee)."""
    import cv2

    yf = np.ascontiguousarray(y, dtype=np.float32)
    gx = cv2.Sobel(yf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(yf, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    return grad / (grad + soft)


def apply_chroma_denoise(
    rgb_linear: np.ndarray,
    *,
    strength: float = 1.0,
    preview: bool = False,
) -> np.ndarray:
    """
    Denoise chroma (Cb/Cr); luminance is unchanged.

    Two passes, both edge-aware: a small-kernel bilateral filter (fine,
    pixel-scale noise) followed by a downsample/blur/upsample pass (coarse,
    blotchy noise a small kernel can't reach), blended back in proportion to
    how flat the region is (via a luma-gradient edge mask) so real color
    edges stay sharp while flat regions get the much stronger coarse pass.

    ``strength`` is expected in ~[0, 1.5] (matches the existing Chroma NR
    toggle's scaling).
    """
    if rgb_linear is None or strength <= 0.0:
        return rgb_linear
    img = np.clip(rgb_linear.astype(np.float32), 0.0, None)
    y, cb, cr = _rgb_to_ycbcr(img)
    if preview:
        strength *= 0.55
    s = min(strength, 1.5)

    sigma_color = 0.03 + s * 0.08
    cb_fine = _bilateral_channel(cb, sigma_color, preview=preview)
    cr_fine = _bilateral_channel(cr, sigma_color, preview=preview)

    # factor/blur_sigma/coarse_mix were tuned against a synthetic edge test
    # (a hard red|blue boundary) to keep bleed within ~4px of a real edge at
    # even the strongest setting (measured max channel error ~0.055 at 4px,
    # ~0.01 at 6px, ~0 at 8px+) while still meaningfully cutting blotchy,
    # spatially-correlated noise a small bilateral kernel can't reach (~13%
    # extra std reduction at max strength on noise with a several-pixel
    # correlation length, on top of the bilateral pass above).
    factor = 2 if preview else 3
    blur_sigma = 0.3 + s * 0.33
    cb_coarse = _downsample_blur_upsample(cb_fine, factor, blur_sigma)
    cr_coarse = _downsample_blur_upsample(cr_fine, factor, blur_sigma)

    edge_w = _luma_edge_weight(y)
    coarse_mix = min(0.6, 0.15 + s * 0.3)
    blend = coarse_mix * (1.0 - edge_w)
    cb_out = cb_fine * (1.0 - blend) + cb_coarse * blend
    cr_out = cr_fine * (1.0 - blend) + cr_coarse * blend

    return np.clip(_ycbcr_to_rgb(y, cb_out, cr_out), 0.0, None)


def apply_luma_denoise(
    rgb_linear: np.ndarray,
    *,
    strength: float = 1.0,
    preview: bool = False,
) -> np.ndarray:
    """
    Denoise luminance (Y) with a bilateral filter; chroma is unchanged.

    ``strength`` is expected in [0, 1] (a direct 0-100 slider / 100). Capped to
    a gentler sigmaColor range (0.015-0.065, vs chroma's 0.03-0.15) than chroma
    denoise: luminance carries real image detail (unlike chroma, which human
    vision resolves at much lower spatial resolution), so the default here
    favors preserving fine low-contrast texture over maximum noise reduction.
    Bilateral filter still keeps hard edges sharp at any strength.
    """
    if rgb_linear is None or strength <= 0.0:
        return rgb_linear
    img = np.clip(rgb_linear.astype(np.float32), 0.0, None)
    y, cb, cr = _rgb_to_ycbcr(img)
    if preview:
        strength *= 0.55
    sigma_color = 0.015 + min(strength, 1.0) * 0.05
    y_d = _bilateral_channel(y, sigma_color, preview=preview)
    return np.clip(_ycbcr_to_rgb(y_d, cb, cr), 0.0, None)


def chroma_denoise_enabled() -> bool:
    return os.environ.get("RAWVIEWER_EDIT_CHROMA_DENOISE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
