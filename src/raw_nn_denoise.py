"""Neural denoise (realPLKSR) for baked export -- NOT a live pipeline stage.

Model: 1xDeNoise_realplksr_otf by Philip Hofmann (CC-BY-4.0),
https://openmodeldb.info/models/1x-DeNoise-realplksr-otf -- a 1x RealPLKSR
restoration net trained with the realesrgan-otf degradation pipeline. On real
high-ISO files it removes chroma speckle while keeping surface texture that
the guided-filter chain blurs away (measured: high-pass energy 9.1 -> 6.5 vs
guided's 2.4 on an ISO 6400 RW2 crop).

Export-only by design: fp16 CUDA inference measures ~830 ms/MP (a 24MP export
adds ~20s -- fp32 was 3.5 s/MP, so fp16 alone is a 4.3x speedup; larger tiles
and channels_last measured no further gain on a 12GB RTX). That is far too
slow for live preview and entirely acceptable for a one-shot export.

The model runs on DISPLAY-REFERRED sRGB (what it was trained on), i.e. after
tone mapping/encode, immediately before the file is written.

Weights are looked up at (first hit wins):
  1. RAWVIEWER_NN_DENOISE_MODEL (explicit path)
  2. %LOCALAPPDATA%/RAWviewer/models/1xDeNoise_realplksr_otf.safetensors
  3. <repo>/models/1xDeNoise_realplksr_otf.safetensors
Requires the `spandrel` package and a CUDA torch build; anything missing makes
nn_denoise_available() False and the export UI hides the option.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_FILENAME = "1xDeNoise_realplksr_otf.safetensors"
_TILE = 512      # processing tile (fastest measured fp16 configuration)
_MARGIN = 32     # per-side context margin; only the inner region is kept,
                 # which makes tiling seam-free without feather blending
_lock = threading.Lock()
_model = None          # loaded torch module (fp16, cuda) or None
_model_failed = False
# Optional cancel probe installed by the export dispatcher for the duration of
# one export (see raw_edit_pipeline.export_adjusted_image). Checked between
# tiles, so cancellation lands within one ~2s tile of the request.
_cancel_check = None


def set_cancel_check(fn) -> None:
    global _cancel_check
    _cancel_check = fn


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


def nn_denoise_available() -> bool:
    """CUDA + spandrel + weights all present (cheap after the first call)."""
    if _model is not None:
        return True
    if _model_failed:
        return False
    if _weights_path() is None:
        return False
    try:
        import spandrel  # noqa: F401
        import torch

        return bool(torch.cuda.is_available())
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
            if path is None or not torch.cuda.is_available():
                raise RuntimeError("weights or CUDA unavailable")
            desc = ModelLoader().load_from_file(path)
            _model = desc.model.cuda().eval().to(torch.float16)
            logger.info(
                "[NN_DENOISE] loaded %s (%s) fp16/cuda",
                os.path.basename(path), desc.architecture.name,
            )
        except Exception as e:
            logger.warning("[NN_DENOISE] model load failed: %s", e)
            _model_failed = True
            _model = None
        return _model


def denoise_display_float(rgb01: np.ndarray) -> Optional[np.ndarray]:
    """Denoise a display-referred float32 [0,1] HxWx3 buffer. None on failure.

    Seam-free margin tiling: each 512px tile carries a 32px context margin
    and only its interior is written back, so no blending pass is needed.
    Luminance-preserving: the model slightly darkens the frame (removing
    noise removes the upward brightness bias noise causes after gamma), which
    reads as a tone shift next to the un-denoised render -- corrected with a
    single global gain matching the output's mean luminance back to the
    source (gated to sane ratios so a pathological output can't be
    over-amplified).
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
                        .to("cuda", torch.float16)
                    )
                    r = model(t).clamp_(0, 1)
                    r = r.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
                    iy0, ix0 = y0 - ys, x0 - xs
                    iy1 = min(y0 + step, h) - ys
                    ix1 = min(x0 + step, w) - xs
                    out[y0:y0 + (iy1 - iy0), x0:x0 + (ix1 - ix0)] = r[iy0:iy1, ix0:ix1]
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
