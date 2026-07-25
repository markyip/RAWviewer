"""SCUNet ONNX export denoise engine with tiled processing."""

import logging
import os
import sys
from typing import Optional

import numpy as np

# NAFNet (NAFNet-SIDD-width32, official megvii-research weights exported to
# ONNX, fixed 512x512 input) stored in the repo via Git LFS. Pure-conv model:
# ~5x faster than SCUNet on CPU with equal/better measured denoising, so it is
# the preferred model on platforms without a GPU EP (macOS/Linux CPU).
NAFNET_MODEL_URL = "https://github.com/markyip/RAWviewer/raw/development/models/nafnet.onnx"
NAFNET_MODEL_SHA256 = "9fb510fdce45ff393ca2822886c6c23cd60c09d47df1229bd7d5f0a07c98f3e0"
# SCUNet (scunet_color_real_psnr, official KAIR weights, fp16) via Git LFS.
# Preferred on Windows where DirectML runs its window-attention graph fast.
DENOISE_MODEL_URL = "https://github.com/markyip/RAWviewer/raw/development/models/scunet.onnx"
DENOISE_MODEL_SHA256 = "d426f34a1b71670e43e6219161a3258fe65b03ece946467ae842737a55ef0849"
# Legacy fallback: Restormer weights from the old release asset (what shipped
# as "scunet.onnx" before the real SCUNet export existed).
LEGACY_DENOISE_MODEL_URL = "https://github.com/markyip/RAWviewer/releases/download/denoise-model-v1/restormer.onnx"
LEGACY_DENOISE_MODEL_SHA256 = "fa43b07d631d61f084fb95e5e4942e188a4a0b51e905b651de2f4187f54b4f09"


def _sha256_of_file(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _models_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


def _preferred_model_name() -> str:
    """Which denoise model this platform should prefer.

    Windows: SCUNet (DirectML runs its attention graph fast). Everything else
    (CPU-only in practice): NAFNet -- pure conv, ~5x faster on CPU with equal
    or better measured denoising. Override with RAWVIEWER_DENOISE_MODEL.
    """
    override = os.environ.get("RAWVIEWER_DENOISE_MODEL", "").strip().lower()
    if override in ("nafnet", "nafnet.onnx"):
        return "nafnet.onnx"
    if override in ("scunet", "scunet.onnx"):
        return "scunet.onnx"
    return "scunet.onnx" if sys.platform.startswith("win") else "nafnet.onnx"


def _model_candidates() -> tuple:
    """(filename, url, sha256) download/lookup candidates in preference order."""
    nafnet = ("nafnet.onnx", NAFNET_MODEL_URL, NAFNET_MODEL_SHA256)
    scunet = ("scunet.onnx", DENOISE_MODEL_URL, DENOISE_MODEL_SHA256)
    legacy = ("restormer.onnx", LEGACY_DENOISE_MODEL_URL, LEGACY_DENOISE_MODEL_SHA256)
    primary, secondary = (scunet, nafnet) if _preferred_model_name() == "scunet.onnx" else (nafnet, scunet)
    return (primary, secondary, legacy)


def scunet_model_path() -> str:
    """Absolute path to the best available denoise ONNX model.

    Returns the platform-preferred model if present, else any other known
    model, else the preferred download destination (may not exist yet).
    """
    base_dir = _models_dir()
    first_missing = None
    for name, _url, _sha in _model_candidates():
        p = os.path.join(base_dir, name)
        if os.path.exists(p):
            return p
        if first_missing is None:
            first_missing = p
    return first_missing


def ensure_scunet_model_downloaded() -> bool:
    """Best-effort fetch of the platform-preferred denoise model if missing."""
    logger = logging.getLogger(__name__)
    if os.path.exists(scunet_model_path()):
        return True
    try:
        from ssl_certs import urlretrieve

        # Download to a temp file and verify integrity before moving into place,
        # so a failed/tampered download never leaves a loadable corrupt model.
        for name, url, sha256 in _model_candidates():
            model_path = os.path.join(_models_dir(), name)
            tmp_path = model_path + ".part"
            try:
                urlretrieve(url, tmp_path, timeout=300)
                if _sha256_of_file(tmp_path).lower() != sha256.lower():
                    logger.warning("[DENOISE] Model download from %s failed SHA-256 verification", url)
                    continue
                os.replace(tmp_path, model_path)
                return True
            except Exception:
                logger.warning("[DENOISE] Failed to download denoise model from %s", url, exc_info=True)
            finally:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except OSError:
                    pass
        return False
    except Exception:
        logger.warning(
            "[DENOISE] Failed to download AI denoise model from %s", _model_candidates()[0][1], exc_info=True
        )
        return False


class SCUNetONNX:
    """Wraps an ONNX-based SCUNet noise reduction model with tiled execution."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._session = None

    def _init_session(self):
        import onnxruntime as ort

        # NOTE: CoreMLExecutionProvider intentionally omitted — the SCUNet
        # attention graph partitions into 200+ CoreML segments, making the
        # first-tile compile hang for minutes and run slower than pure CPU.
        requested = [
            "DmlExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        # Keep only providers this onnxruntime build actually has, so
        # unavailable-platform requests (Dml/CUDA on macOS) don't spam
        # UserWarnings into the app log on every session init.
        available = set(ort.get_available_providers())
        providers = [p for p in requested if p in available] or ["CPUExecutionProvider"]

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
        downscale: int = 1,
        detail_gain: float = 0.5,
    ) -> np.ndarray:
        """Process a full-resolution RGB image using overlapping tiles.

        rgb_linear: (H, W, 3) float32 array in [0, 1] (scene linear or similar)
        cancel_check: optional no-arg callable polled once per tile.
        downscale: 1 = full-res. 2 = denoise at half resolution (~4x faster)
        and restore high-frequency luma detail with an edge-weighted blend --
        measured on a noisy ISO-3200 CR3 to match full-res output on both
        residual noise and detail energy at ~1/5 the compute.
        detail_gain: 0.0 = pure denoised surface (maximum smoothness),
        1.0 = full original high-frequency detail at edges (maximum detail,
        noise included). 0.5 measured to match full-res output metrics.
        """
        if self._session is None:
            self._init_session()

        if downscale > 1:
            import cv2

            orig = rgb_linear
            h0, w0 = orig.shape[:2]
            small = cv2.resize(orig, (w0 // downscale, h0 // downscale), interpolation=cv2.INTER_AREA)
            out_small = self._process_tiles(small, tile_size, tile_overlap, progress_callback, cancel_check)
            up = cv2.resize(out_small, (w0, h0), interpolation=cv2.INTER_LINEAR)

            # Edge-weighted high-frequency restore: return original fine detail
            # where real structure dominates (whiskers/edges), keep the denoised
            # surface everywhere flat (where HF would just be noise).
            hf = orig - cv2.GaussianBlur(orig, (0, 0), 2.0)
            y = orig.mean(axis=2).astype(np.float32)
            gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
            grad = np.sqrt(gx * gx + gy * gy)
            w = (grad / (grad + 0.08))[..., None]  # soft edge weight (0 flat -> 1 edge)
            gain = max(0.0, min(1.0, float(detail_gain)))
            return np.clip(up + hf * w * gain, 0.0, None)

        return self._process_tiles(rgb_linear, tile_size, tile_overlap, progress_callback, cancel_check)

    def _process_tiles(
        self,
        rgb_linear: np.ndarray,
        tile_size: int,
        tile_overlap: int,
        progress_callback,
        cancel_check,
    ) -> np.ndarray:
        """Tiled inference core (see process() for the public contract)."""
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
