"""RawRefinery "Tree Net Denoise Heavy" as the AI denoise backend.

Model: ShadowWeightedL1_24_deep_500 (RawRefinery v1.2.1-alpha, MIT), exported
offline from TorchScript to ONNX. Runs on 128 px RGB tiles with 0.25 overlap
and one ISO conditioning value (ISO / 6400).
"""

import logging
import os
import threading

import numpy as np

MODEL_FILENAME = "rawrefinery_deep_128.onnx"
MODEL_SHA256 = "03acd748b5fa25d5120dd09f4bbf2ec09677664c7a4f6d99cd84739ccf43b554"

TILE = 128
STRIDE = 96  # 0.25 overlap
ISO_CONDITIONING = 1.0  # ISO 6400; the pipeline has no per-file ISO yet

_SESSION_LOCK = threading.Lock()
_SESSION = None
_SESSION_PATH = None


def _models_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


def model_path() -> str:
    return os.path.join(_models_dir(), MODEL_FILENAME)


def model_available() -> bool:
    return os.path.exists(model_path())


def _get_session(path: str):
    global _SESSION, _SESSION_PATH
    with _SESSION_LOCK:
        if _SESSION is not None and path == _SESSION_PATH:
            return _SESSION
        import onnxruntime as ort

        available = set(ort.get_available_providers())
        requested = ["DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in requested if p in available] or ["CPUExecutionProvider"]
        _SESSION = ort.InferenceSession(path, providers=providers)
        _SESSION_PATH = path
        logging.getLogger(__name__).info("[DENOISE] RawRefinery session on %s", _SESSION.get_providers()[0])
        return _SESSION


def _tile_starts(n: int) -> list:
    starts = list(range(0, n - TILE + 1, STRIDE))
    if starts[-1] != n - TILE:
        starts.append(n - TILE)
    return starts


def denoise(
    img: np.ndarray,
    *,
    strength: float = 1.0,
    cancel_check=None,
    progress_callback=None,
) -> np.ndarray:
    """Denoise a linear RGB float32 image. ``strength`` blends toward the input."""
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 1e-4:
        return img

    session = _get_session(model_path())
    src = np.asarray(img, dtype=np.float32)
    h, w = src.shape[:2]
    ph, pw = max(h, TILE), max(w, TILE)
    if (ph, pw) != (h, w):
        src = np.pad(src, ((0, ph - h), (0, pw - w), (0, 0)), mode="edge")

    starts = [(y, x) for y in _tile_starts(ph) for x in _tile_starts(pw)]
    window = (
        np.outer(np.hanning(TILE + 2)[1:-1], np.hanning(TILE + 2)[1:-1]).astype(np.float32)[..., None]
        + 1e-3
    )
    acc = np.zeros_like(src)
    wsum = np.zeros((ph, pw, 1), np.float32)
    cond = np.full((1, 1), ISO_CONDITIONING, np.float32)
    total = len(starts)

    for i, (y, x) in enumerate(starts):
        if cancel_check is not None and cancel_check():
            return img
        tile = np.clip(src[y : y + TILE, x : x + TILE], 0.0, 1.0)
        inp = np.ascontiguousarray(tile.transpose(2, 0, 1)[None])
        out = session.run(None, {"inp": inp, "cond": cond})[0][0].transpose(1, 2, 0)
        acc[y : y + TILE, x : x + TILE] += out * window
        wsum[y : y + TILE, x : x + TILE] += window
        if progress_callback is not None:
            progress_callback((i + 1) / total)

    denoised = (acc / wsum)[:h, :w]
    original = np.asarray(img, dtype=np.float32)
    return (strength * denoised + (1.0 - strength) * original).astype(np.float32)
