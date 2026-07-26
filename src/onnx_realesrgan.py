"""Real-ESRGAN x2 ONNX upscale engine with tiled processing.

Export-only, opt-in, and the last stage before encoding: everything upstream
(geometry, masks, denoise, tone) runs at native resolution, so an upscaled
export is the same image the user approved in the Adjust panel, at 2x the
pixels. Doing it any earlier would quadruple the cost of every later stage for
no quality gain.

Delivery mirrors onnx_scunet.py: weights live in ``<repo>/models``, are fetched
on demand rather than bundled, and are SHA-256 verified before being moved into
place. Export the model with scripts/models/export_realesrgan_onnx.py.
"""

import logging
import os
from typing import Optional

import numpy as np

# RealESRGAN_x2plus (xinntao/Real-ESRGAN v0.2.1 official weights, BSD-3-Clause,
# params_ema) exported to ONNX at a static 256x256 -> 512x512 tile shape.
UPSCALE_MODEL_URL = "https://github.com/markyip/RAWviewer/raw/development/models/realesrgan_x2.onnx"
UPSCALE_MODEL_SHA256 = "35fa2e6f2b44bf72afa79c1281b47d6cfc9ec093ad109742ba4c80e1c4fe06d2"
UPSCALE_MODEL_NAME = "realesrgan_x2.onnx"

# The graph's fixed input tile. Kept small deliberately -- output is 4x the
# pixels of input, so peak memory scales with the square of this.
TILE_SIZE = 256
# Enough to cover the receptive field of 23 RRDB blocks well past the point
# where seam energy is measurable; see t_realesrgan_upscale.py, which asserts
# the tiled result matches a single-tile reference.
TILE_OVERLAP = 32

SCALE = 2

logger = logging.getLogger(__name__)


def _sha256_of_file(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _models_dir() -> str:
    """Same directory the denoise models use -- see onnx_scunet._models_dir."""
    try:
        from onnx_scunet import _models_dir as _scunet_models_dir

        return _scunet_models_dir()
    except Exception:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


def realesrgan_model_path() -> str:
    """Absolute path to the upscale model (may not exist yet)."""
    return os.path.join(_models_dir(), UPSCALE_MODEL_NAME)


def realesrgan_model_available() -> bool:
    return os.path.exists(realesrgan_model_path())


def ensure_realesrgan_model_downloaded() -> bool:
    """Best-effort fetch of the upscale model if missing.

    Mirrors onnx_scunet.ensure_scunet_model_downloaded: download to a temp
    file and verify the digest before moving it into place, so a truncated or
    tampered download never leaves a loadable corrupt model behind.
    """
    model_path = realesrgan_model_path()
    if os.path.exists(model_path):
        return True
    tmp_path = model_path + ".part"
    try:
        from ssl_certs import urlretrieve

        urlretrieve(UPSCALE_MODEL_URL, tmp_path, timeout=600)
        if _sha256_of_file(tmp_path).lower() != UPSCALE_MODEL_SHA256.lower():
            logger.warning("[UPSCALE] Model download from %s failed SHA-256 verification", UPSCALE_MODEL_URL)
            return False
        os.replace(tmp_path, model_path)
        return True
    except Exception:
        logger.warning("[UPSCALE] Failed to download upscale model from %s", UPSCALE_MODEL_URL, exc_info=True)
        return False
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


class RealESRGANONNX:
    """Real-ESRGAN x2 super-resolution over overlapping tiles."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or realesrgan_model_path()
        self._session = None
        self._input_name = None
        self._output_name = None

    def _init_session(self):
        import onnxruntime as ort

        # CoreML is requested FIRST here, unlike onnx_scunet, which omits it on
        # purpose. That exclusion is specific to SCUNet's window-attention
        # graph, which partitions into 200+ CoreML segments and ends up slower
        # than CPU. RRDBNet is pure convolution and partitions cleanly:
        # measured 0.06s vs 2.43s per 256px tile on an M-series CPU, a ~40x
        # difference that decides whether this feature is usable at all.
        # Its fp16 execution costs ~5e-3 max deviation from the CPU result
        # (~1/255), which is below the quantization of any format we export.
        requested = [
            "DmlExecutionProvider",
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
        available = set(ort.get_available_providers())
        providers = [p for p in requested if p in available] or ["CPUExecutionProvider"]

        options = ort.SessionOptions()
        options.log_severity_level = 3

        self._session = ort.InferenceSession(self.model_path, sess_options=options, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        logger.info("[UPSCALE] Session ready (providers=%s)", self._session.get_providers()[:1])

    def process(
        self,
        rgb_linear: np.ndarray,
        progress_callback=None,
        cancel_check=None,
        strength: float = 1.0,
    ) -> np.ndarray:
        """Upscale (H, W, 3) scene-linear float32 to (2H, 2W, 3), same domain.

        cancel_check: optional no-arg callable polled once per tile; raises
        ExportCancelled, matching the denoise engine's contract.

        strength: 1.0 = the model's output as-is; 0.0 = plain Lanczos. Values
        between blend the two in the perceptual domain.

        Why the knob exists: RealESRGAN_x2plus is a GAN trained on synthetic
        degradations (blur, noise, JPEG), so it *invents* plausible detail
        rather than reconstructing true detail. Measured against a real
        ground-truth crop from this pipeline it scores 23.8 dB PSNR where
        bicubic scores 32.7, and lands 3x the Laplacian variance of the
        ground truth -- it is visibly over-sharpened on clean RAW output,
        which is exactly the input a photo editor feeds it. Default stays
        1.0 (faithful Real-ESRGAN, what the name promises); dial down for
        material where the crunch shows.
        """
        if self._session is None:
            self._init_session()

        h, w = rgb_linear.shape[:2]

        # Same domain handling as SCUNetONNX._process_tiles: the model was
        # trained on display-referred sRGB, so feed it gamma-encoded data, and
        # carry specular headroom above 1.0 around the model rather than
        # clipping it off. Losing it here would blow out highlights that the
        # tone stage upstream deliberately preserved.
        rgb_clipped = np.clip(rgb_linear, 0.0, None)
        headroom_scale = np.maximum(rgb_clipped, 1.0)
        rgb_perceptual = np.power(np.minimum(rgb_clipped, 1.0), 1.0 / 2.2).astype(np.float32)

        import cv2

        out_perceptual = self._process_tiles(rgb_perceptual, progress_callback, cancel_check)

        strength = max(0.0, min(1.0, float(strength)))
        if strength < 1.0:
            # Blend in the perceptual domain, where the model's output lives --
            # mixing after the 2.2 decode would weight the two toward shadows.
            plain = cv2.resize(
                rgb_perceptual, (w * SCALE, h * SCALE), interpolation=cv2.INTER_LANCZOS4
            )
            out_perceptual = out_perceptual * strength + plain * (1.0 - strength)

        # Headroom was computed at input resolution; bring it up to match.

        headroom_up = cv2.resize(headroom_scale, (w * SCALE, h * SCALE), interpolation=cv2.INTER_LINEAR)
        if headroom_up.ndim == 2:
            headroom_up = headroom_up[..., None]

        out_linear = np.power(np.clip(out_perceptual, 0.0, 1.0), 2.2) * headroom_up
        return np.clip(out_linear, 0.0, None)

    def _process_tiles(self, src: np.ndarray, progress_callback, cancel_check) -> np.ndarray:
        """Tiled inference in whatever domain ``src`` is already in."""
        h, w = src.shape[:2]
        tile = TILE_SIZE
        overlap = TILE_OVERLAP
        stride = tile - overlap

        out_h, out_w = h * SCALE, w * SCALE
        accum = np.zeros((out_h, out_w, 3), dtype=np.float32)
        weights = np.zeros((out_h, out_w, 1), dtype=np.float32)

        # Raised cosine (sin^2) ramp: an exact partition of unity, so overlap
        # regions sum to 1.0 with no residual seam. Same construction as
        # onnx_scunet, but built at output scale since that is where blending
        # happens.
        o2 = overlap * SCALE
        t = (np.arange(o2, dtype=np.float32) + 0.5) / float(o2)
        ramp_in = np.sin(0.5 * np.pi * t) ** 2
        ramp_out = 1.0 - ramp_in

        y_positions = list(range(0, h - tile, stride)) + [max(0, h - tile)] if h > tile else [0]
        x_positions = list(range(0, w - tile, stride)) + [max(0, w - tile)] if w > tile else [0]

        total = len(y_positions) * len(x_positions)
        done = 0

        for y in y_positions:
            for x in x_positions:
                if cancel_check is not None and cancel_check():
                    from raw_edit_pipeline import ExportCancelled

                    raise ExportCancelled()

                patch = src[y : y + tile, x : x + tile, :]
                ph, pw = patch.shape[:2]

                # The graph shape is static, so short edge tiles get reflected
                # out to a full tile and the surplus cropped from the result.
                if ph < tile or pw < tile:
                    patch = np.pad(patch, ((0, tile - ph), (0, tile - pw), (0, 0)), mode="reflect")

                tensor = np.ascontiguousarray(patch.transpose(2, 0, 1)[None, ...].astype(np.float32))
                result = self._session.run([self._output_name], {self._input_name: tensor})[0]
                out_tile = result[0].transpose(1, 2, 0)[: ph * SCALE, : pw * SCALE, :]

                oy, ox = y * SCALE, x * SCALE
                th, tw = ph * SCALE, pw * SCALE

                win_y = np.ones(th, dtype=np.float32)
                win_x = np.ones(tw, dtype=np.float32)
                if y > 0:
                    win_y[:o2] = ramp_in
                if y + tile < h and ph == tile:
                    win_y[-o2:] = ramp_out
                if x > 0:
                    win_x[:o2] = ramp_in
                if x + tile < w and pw == tile:
                    win_x[-o2:] = ramp_out

                window = np.outer(win_y, win_x)[..., None]
                accum[oy : oy + th, ox : ox + tw, :] += out_tile * window
                weights[oy : oy + th, ox : ox + tw, :] += window

                done += 1
                if progress_callback is not None:
                    progress_callback(done / total)

        return accum / np.clip(weights, 1e-6, None)


def upscale_linear(
    rgb_linear: np.ndarray,
    progress_callback=None,
    cancel_check=None,
    strength: float = 1.0,
) -> np.ndarray:
    """Convenience wrapper: one-shot x2 upscale of a scene-linear buffer."""
    return RealESRGANONNX().process(
        rgb_linear,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        strength=strength,
    )
