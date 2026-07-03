"""Chroma-only non-local means denoise for scene-linear RGB edit buffers."""

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


def _nlm_channel_u8(
    channel: np.ndarray, strength: float, *, preview: bool
) -> np.ndarray:
    import cv2

    h = max(1.0, strength * (2.5 if preview else 4.0))
    search = 11 if preview else 21
    if preview and max(channel.shape[:2]) > 256:
        scale = 256 / max(channel.shape[:2])
        nw = max(1, int(channel.shape[1] * scale))
        nh = max(1, int(channel.shape[0] * scale))
        small = cv2.resize(channel, (nw, nh), interpolation=cv2.INTER_AREA)
        den = cv2.fastNlMeansDenoising(
            small,
            None,
            h,
            templateWindowSize=5,
            searchWindowSize=search,
        )
        return cv2.resize(den, (channel.shape[1], channel.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.fastNlMeansDenoising(
        channel,
        None,
        h,
        templateWindowSize=7,
        searchWindowSize=search,
    )


def apply_chroma_nlm(
    rgb_linear: np.ndarray,
    *,
    strength: float = 1.0,
    preview: bool = False,
) -> np.ndarray:
    """
    Denoise chroma (Cb/Cr) with NLM; luminance is unchanged.

    ``preview=True`` uses a lighter filter for half-size live editing.
    """
    if rgb_linear is None or strength <= 0.0:
        return rgb_linear
    img = np.clip(rgb_linear.astype(np.float32), 0.0, None)
    y, cb, cr = _rgb_to_ycbcr(img)
    if preview:
        strength *= 0.55

    cb_u8 = np.clip(cb * 255.0, 0, 255).astype(np.uint8)
    cr_u8 = np.clip(cr * 255.0, 0, 255).astype(np.uint8)
    cb_d = _nlm_channel_u8(cb_u8, strength, preview=preview).astype(np.float32) / 255.0
    cr_d = _nlm_channel_u8(cr_u8, strength, preview=preview).astype(np.float32) / 255.0
    return np.clip(_ycbcr_to_rgb(y, cb_d, cr_d), 0.0, None)


def chroma_denoise_enabled() -> bool:
    return os.environ.get("RAWVIEWER_EDIT_CHROMA_DENOISE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
