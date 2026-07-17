"""Neural denoise (SCUNet) for baked export -- NOT a live pipeline stage.

Model: scunet_color_real_psnr from SCUNet / KAIR by Kai Zhang et al.
(Apache-2.0), https://github.com/cszn/SCUNet -- Practical Blind Image Denoising
via Swin-Conv-UNet and Data Synthesis (Machine Intelligence Research, 2023).
The PSNR checkpoint is the faithful real-world blind denoise variant (vs the
GAN checkpoint, which is more aggressive / perceptual).

Export-only by design: inference is too slow for live preview and is applied
on the DISPLAY-REFERRED sRGB buffer (what the model was trained on) immediately
before the file is written -- never during Adjust preview or browsing.

Weights are looked up at (first hit wins):
  1. RAWVIEWER_NN_DENOISE_MODEL (explicit path)
  2. %LOCALAPPDATA%/RAWviewer/models/scunet_color_real_psnr.pth
  3. <repo>/models/scunet_color_real_psnr.pth
Requires the `spandrel` package and CUDA or MPS torch; anything missing makes
nn_denoise_available() False and the export UI hides the option.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_FILENAME = "scunet_color_real_psnr.pth"
_MODEL_URL = (
    "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth"
)
# Approximate download size shown in the download dialog (~68.6 MiB).
_MODEL_SIZE_MB = 69
# SCUNet tiling is discouraged by spandrel (global context); keep a generous
# overlap so export of multi-megapixel photos stays VRAM-safe without seams.
_TILE = 512
_MARGIN = 64
_lock = threading.Lock()
_model = None          # spandrel ImageModelDescriptor (fp16) or None
_model_failed = False
# Optional cancel probe installed by the export dispatcher for the duration of
# one export (see raw_edit_pipeline.export_adjusted_image). Checked between
# tiles, so cancellation lands within one tile of the request.
_cancel_check = None
# Optional progress probe installed alongside _cancel_check for the duration
# of one export. Called (tiles_done, tiles_total) after each tile so the
# export dialog can show real, tile-driven percentage instead of a marquee.
_progress_cb = None


def model_filename() -> str:
    return _MODEL_FILENAME


def model_download_url() -> str:
    return _MODEL_URL


def model_download_size_mb() -> int:
    return _MODEL_SIZE_MB


def set_cancel_check(fn) -> None:
    global _cancel_check
    _cancel_check = fn


def set_progress_cb(fn) -> None:
    global _progress_cb
    _progress_cb = fn


class NNDenoiseCancelled(Exception):
    pass


def _weights_path() -> Optional[str]:
    cands = []
    env = os.environ.get("RAWVIEWER_NN_DENOISE_MODEL", "").strip()
    if env:
        cands.append(env)
    lad = os.environ.get("LOCALAPPDATA", "")
    if lad:
        cands.append(os.path.join(lad, "RAWviewer", "models", _MODEL_FILENAME))
    cands.append(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "models", _MODEL_FILENAME)
    )
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


def nn_denoise_weights_present() -> bool:
    return _weights_path() is not None


def nn_denoise_available() -> bool:
    """CUDA/MPS + spandrel all present (cheap after the first call)."""
    if _model is not None:
        return True
    if _model_failed:
        return False
    try:
        import spandrel  # noqa: F401
        import torch

        if torch.cuda.is_available():
            return True
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return True
        return False
    except Exception:
        return False


def _load_model():
    global _model, _model_failed
    with _lock:
        if _model is not None or _model_failed:
            return _model
        try:
            import torch
            from spandrel import ModelLoader

            path = _weights_path()
            device_name = "cpu"
            if torch.cuda.is_available():
                device_name = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_name = "mps"

            if path is None or device_name == "cpu":
                raise RuntimeError("weights or CUDA/MPS unavailable")
            desc = ModelLoader().load_from_file(path)
            # Keep the ImageModelDescriptor so __call__ applies size padding.
            desc = desc.to(device_name).eval().half()
            _model = desc
            logger.info(
                "[NN_DENOISE] loaded %s (%s) fp16/%s",
                os.path.basename(path), desc.architecture.name, device_name,
            )
        except Exception as e:
            logger.warning("[NN_DENOISE] model load failed: %s", e)
            _model_failed = True
            _model = None
        return _model


def denoise_display_float(rgb01: np.ndarray) -> Optional[np.ndarray]:
    """Denoise a display-referred float32 [0,1] HxWx3 buffer. None on failure.

    Seam-aware margin tiling: each tile carries a context margin and only its
    interior is written back. Luminance-preserving: the model slightly darkens
    the frame (removing noise removes the upward brightness bias noise causes
    after gamma), corrected with a single global gain matching mean luminance
    back to the source (gated to sane ratios).
    """
    if _cancel_check is not None and _cancel_check():
        raise NNDenoiseCancelled()
    model = _load_model()
    if model is None or rgb01 is None or rgb01.ndim != 3:
        return None
    if _cancel_check is not None and _cancel_check():
        raise NNDenoiseCancelled()
    try:
        import torch

        h, w = rgb01.shape[:2]
        out = np.empty((h, w, 3), dtype=np.float32)
        step = _TILE - 2 * _MARGIN
        tiles_total = max(1, len(range(0, h, step)) * len(range(0, w, step)))
        tiles_done = 0
        with torch.no_grad():
            for y0 in range(0, h, step):
                for x0 in range(0, w, step):
                    if _cancel_check is not None and _cancel_check():
                        raise NNDenoiseCancelled()
                    ys, xs = max(0, y0 - _MARGIN), max(0, x0 - _MARGIN)
                    ye, xe = min(h, y0 + step + _MARGIN), min(w, x0 + step + _MARGIN)
                    tile = np.ascontiguousarray(rgb01[ys:ye, xs:xe])
                    t = (
                        torch.from_numpy(tile)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .to(device=model.device, dtype=model.dtype)
                    )
                    # Descriptor pads to size_requirements and strips padding.
                    r = model(t).clamp_(0, 1)
                    r = r.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
                    iy0, ix0 = y0 - ys, x0 - xs
                    iy1 = min(y0 + step, h) - ys
                    ix1 = min(x0 + step, w) - xs
                    out[y0:y0 + (iy1 - iy0), x0:x0 + (ix1 - ix0)] = r[iy0:iy1, ix0:ix1]
                    tiles_done += 1
                    if _progress_cb is not None:
                        try:
                            _progress_cb(tiles_done, tiles_total)
                        except Exception:
                            pass
        # Global luminance re-anchor (see docstring).
        lum_in = float((0.2126 * rgb01[..., 0] + 0.7152 * rgb01[..., 1]
                        + 0.0722 * rgb01[..., 2]).mean())
        lum_out = float((0.2126 * out[..., 0] + 0.7152 * out[..., 1]
                         + 0.0722 * out[..., 2]).mean())
        if lum_out > 1e-6 and abs(lum_in - lum_out) / max(lum_in, 1e-6) < 0.15:
            out = np.clip(out * (lum_in / lum_out), 0.0, 1.0)
        return out
    except NNDenoiseCancelled:
        raise
    except Exception as e:
        logger.warning("[NN_DENOISE] inference failed: %s", e)
        return None


def denoise_display_uint8(rgb: np.ndarray) -> np.ndarray:
    """uint8 in, uint8 out; returns the input unchanged on any failure."""
    out = denoise_display_float(rgb.astype(np.float32) / 255.0)
    if out is None:
        return rgb
    return (out * 255.0 + 0.5).astype(np.uint8)


def denoise_display_uint16(rgb: np.ndarray) -> np.ndarray:
    """uint16 in, uint16 out; returns the input unchanged on any failure."""
    out = denoise_display_float(rgb.astype(np.float32) / 65535.0)
    if out is None:
        return rgb
    return (out * 65535.0 + 0.5).astype(np.uint16)
