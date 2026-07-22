"""SCUNet ONNX export denoise engine with tiled processing."""

import logging
import os
import sys
from typing import Optional

import numpy as np

DENOISE_MODEL_URL = "https://github.com/markyip/RAWviewer/releases/download/denoise-model-v1/scunet.onnx"
LEGACY_DENOISE_MODEL_URL = "https://github.com/markyip/RAWviewer/releases/download/denoise-model-v1/restormer.onnx"


def scunet_model_path() -> str:
    """Return the absolute path to the SCUNet ONNX model file.

    Checks for scunet.onnx first, then falls back to restormer.onnx for
    backward compatibility with existing downloads.
    """
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    scunet_path = os.path.join(base_dir, "scunet.onnx")
    if os.path.exists(scunet_path):
        return scunet_path
    restormer_path = os.path.join(base_dir, "restormer.onnx")
    if os.path.exists(restormer_path):
        return restormer_path
    return scunet_path


def ensure_scunet_model_downloaded() -> bool:
    """Best-effort fetch of the SCUNet AI denoise model if it's missing."""
    model_path = scunet_model_path()
    if os.path.exists(model_path):
        return True
    try:
        from ssl_certs import urlretrieve

        # Try scunet.onnx URL first, fallback to restormer.onnx URL if needed
        try:
            urlretrieve(DENOISE_MODEL_URL, model_path, timeout=180)
        except Exception:
            urlretrieve(LEGACY_DENOISE_MODEL_URL, model_path, timeout=180)
        return os.path.exists(model_path)
    except Exception:
        logging.getLogger(__name__).warning(
            "[DENOISE] Failed to download SCUNet AI denoise model from %s", DENOISE_MODEL_URL, exc_info=True
        )
        return False


class SCUNetONNX:
    """Wraps an ONNX-based SCUNet noise reduction model with tiled execution."""

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
        progress_callback=None,
        cancel_check=None,
    ) -> np.ndarray:
        """Process a full-resolution RGB image using overlapping tiles.

        rgb_linear: (H, W, 3) float32 array in [0, 1] (scene linear or similar)
        cancel_check: optional no-arg callable polled once per tile.
        """
        if self._session is None:
            self._init_session()

        h, w, c = rgb_linear.shape
        stride = tile_size - tile_overlap

        # Convert scene-linear RGB to perceptual domain (sRGB gamma ~2.2) for SCUNet,
        # preserving specular highlight headroom > 1.0.
        rgb_clipped = np.clip(rgb_linear, 0.0, None)
        headroom_scale = np.maximum(rgb_clipped, 1.0)
        rgb_perceptual = np.power(np.minimum(rgb_clipped, 1.0), 1.0 / 2.2).astype(np.float32)

        # Calculate grid size
        h_idx_list = list(range(0, h - tile_size, stride)) + [max(0, h - tile_size)] if h > tile_size else [0]
        w_idx_list = list(range(0, w - tile_size, stride)) + [max(0, w - tile_size)] if w > tile_size else [0]

        # Create output buffer and weight buffer (in perceptual domain)
        output_perceptual = np.zeros_like(rgb_linear, dtype=np.float32)
        weights = np.zeros((h, w, 1), dtype=np.float32)

        # Raised cosine (sin^2) ramp: forms an exact partition of unity (sin^2 + cos^2 = 1.0)
        # ensuring 0.000000 weight fluctuation across overlap regions to eliminate banding.
        t = (np.arange(tile_overlap, dtype=np.float32) + 0.5) / float(tile_overlap)
        ramp_in = np.sin(0.5 * np.pi * t) ** 2
        ramp_out = 1.0 - ramp_in

        total_tiles = len(h_idx_list) * len(w_idx_list)
        tile_count = 0

        for y in h_idx_list:
            for x in w_idx_list:
                if cancel_check is not None and cancel_check():
                    from raw_edit_pipeline import ExportCancelled

                    raise ExportCancelled()

                # Extract tile in perceptual domain and pad if necessary
                tile = rgb_perceptual[y : y + tile_size, x : x + tile_size, :]
                th, tw, _ = tile.shape

                # Model needs exactly (tile_size, tile_size)
                pad_h = tile_size - th
                pad_w = tile_size - tw
                if pad_h > 0 or pad_w > 0:
                    tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

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
                    win_y[:tile_overlap] = ramp_in[:tile_overlap]
                if y + tile_size < h:
                    if th == tile_size:
                        win_y[-tile_overlap:] = ramp_out[-tile_overlap:]

                if x > 0:
                    win_x[:tile_overlap] = ramp_in[:tile_overlap]
                if x + tile_size < w:
                    if tw == tile_size:
                        win_x[-tile_overlap:] = ramp_out[-tile_overlap:]

                window_2d = np.expand_dims(np.outer(win_y, win_x), axis=-1)

                # Blend in perceptual domain
                output_perceptual[y : y + th, x : x + tw, :] += out_tile * window_2d
                weights[y : y + th, x : x + tw, :] += window_2d

                tile_count += 1
                if progress_callback:
                    progress_callback(tile_count / total_tiles)

        # Normalize overlapping regions
        weights = np.clip(weights, 1e-6, None)
        output_perceptual = output_perceptual / weights
        output_perceptual = np.clip(output_perceptual, 0.0, 1.0)

        # Convert back from perceptual domain to scene-linear RGB, re-applying highlight headroom
        output_linear = np.power(output_perceptual, 2.2) * headroom_scale
        return np.clip(output_linear, 0.0, None)
