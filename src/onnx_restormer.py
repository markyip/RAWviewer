"""Restormer ONNX export denoise engine with tiled processing."""

import logging
import os
import sys
from typing import Optional

import numpy as np

DENOISE_MODEL_URL = "https://github.com/markyip/RAWviewer/releases/download/denoise-model-v1/restormer.onnx"


def restormer_model_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "restormer.onnx")


def ensure_restormer_model_downloaded() -> bool:
    """Best-effort fetch of the AI denoise model if it's missing.

    This is the model-installation gap fix: the installer's post-install
    download step (scripts/models/download_mobileclip_onnx.py) fetches this
    model, but the *separate*, narrower in-app "AI models missing -- click
    Download" recovery path (gallery Search -> download_semantic_backend_assets
    -> MobileCLIPONNXBackend.download_assets) only ever fetched the MobileCLIP
    search assets, never this. If the install-time download failed (network
    blip) and the user recovered via that in-app prompt, the denoise model
    would stay silently, permanently missing with no further retry path.
    Called from download_semantic_backend_assets() so both entry points now
    cover it. Non-fatal: returns False on failure, never raises.
    """
    model_path = restormer_model_path()
    if os.path.exists(model_path):
        return True
    try:
        from ssl_certs import urlretrieve

        urlretrieve(DENOISE_MODEL_URL, model_path, timeout=180)
        return os.path.exists(model_path)
    except Exception:
        logging.getLogger(__name__).warning(
            "[DENOISE] Failed to download AI denoise model from %s", DENOISE_MODEL_URL, exc_info=True
        )
        return False


class RestormerONNX:
    """Wraps an ONNX-based image restoration model (like Restormer) with tiled execution."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._session = None

    def _init_session(self):
        import onnxruntime as ort

        providers = [
            "DmlExecutionProvider",
            "CoreMLExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        
        # Suppress warnings if a provider isn't available
        options = ort.SessionOptions()
        options.log_severity_level = 3
        
        self._session = ort.InferenceSession(self.model_path, sess_options=options, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def process(
        self,
        rgb_linear: np.ndarray,
        tile_size: int = 512,
        tile_overlap: int = 64,
        progress_callback=None
    ) -> np.ndarray:
        """
        Process a full-resolution RGB image using overlapping tiles.
        rgb_linear: (H, W, 3) float32 array in [0, 1] (scene linear or similar)
        """
        if self._session is None:
            self._init_session()

        h, w, c = rgb_linear.shape
        stride = tile_size - tile_overlap
        
        # Calculate grid size
        h_idx_list = list(range(0, h - tile_size, stride)) + [max(0, h - tile_size)] if h > tile_size else [0]
        w_idx_list = list(range(0, w - tile_size, stride)) + [max(0, w - tile_size)] if w > tile_size else [0]
        
        # Create output buffer and weight buffer
        output = np.zeros_like(rgb_linear)
        weights = np.zeros((h, w, 1), dtype=np.float32)

        ramp = np.hanning(tile_overlap * 2).astype(np.float32)

        total_tiles = len(h_idx_list) * len(w_idx_list)
        tile_count = 0

        for y in h_idx_list:
            for x in w_idx_list:
                # Extract tile and pad if necessary
                tile = rgb_linear[y:y+tile_size, x:x+tile_size, :]
                th, tw, _ = tile.shape
                
                # Model needs exactly (tile_size, tile_size)
                pad_h = tile_size - th
                pad_w = tile_size - tw
                if pad_h > 0 or pad_w > 0:
                    tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                
                # Convert (H, W, C) to (1, C, H, W)
                input_tensor = np.transpose(tile, (2, 0, 1))
                input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)
                
                # Run inference
                out_tensor = self._session.run([self._output_name], {self._input_name: input_tensor})[0]
                
                # Convert back to (H, W, C)
                out_tile = np.transpose(out_tensor[0], (1, 2, 0))
                
                # Crop padding
                out_tile = out_tile[:th, :tw, :]
                
                win_y = np.ones(th, dtype=np.float32)
                win_x = np.ones(tw, dtype=np.float32)
                
                if y > 0:
                    win_y[:tile_overlap] = ramp[:tile_overlap]
                if y + tile_size < h:
                    if th == tile_size:
                        win_y[-tile_overlap:] = ramp[-tile_overlap:]
                    
                if x > 0:
                    win_x[:tile_overlap] = ramp[:tile_overlap]
                if x + tile_size < w:
                    if tw == tile_size:
                        win_x[-tile_overlap:] = ramp[-tile_overlap:]
                        
                window_2d = np.expand_dims(np.outer(win_y, win_x), axis=-1)
                
                # Blend
                output[y:y+th, x:x+tw, :] += out_tile * window_2d
                weights[y:y+th, x:x+tw, :] += window_2d

                tile_count += 1
                if progress_callback:
                    progress_callback(tile_count / total_tiles)

        # Normalize overlapping regions
        weights = np.clip(weights, 1e-6, None)
        output = output / weights
        return np.clip(output, 0.0, None)
