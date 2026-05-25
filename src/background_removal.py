import os
import urllib.request
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class BackgroundRemover:
    """
    ONNX-based background removal utilizing the U2-Net architecture.
    Isolates the main subject to help reduce noise for downstream classifiers.
    """
    MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
    MODEL_FILENAME = "u2net.onnx"

    def __init__(self):
        self.model_dir = os.path.expanduser("~/.rawviewer_cache/bg_models")
        self.model_path = os.path.join(self.model_dir, self.MODEL_FILENAME)
        self.session = None

    def _ensure_model(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_dir, exist_ok=True)
            logger.info(f"[BG Removal] Downloading U2-Net background removal model...")
            try:
                urllib.request.urlretrieve(self.MODEL_URL, self.model_path)
                logger.info("[BG Removal] Model download complete.")
            except Exception as e:
                logger.error(f"[BG Removal] Failed to download model: {e}")
                raise

    def _ensure_session(self):
        if self.session is None:
            self._ensure_model()
            try:
                import onnxruntime as ort
                providers = ['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
                available_providers = ort.get_available_providers()
                selected = [p for p in providers if p in available_providers]
                self.session = ort.InferenceSession(self.model_path, providers=selected)
                logger.info(f"[BG Removal] ONNX session initialized with providers: {selected}")
            except Exception as e:
                logger.error(f"[BG Removal] Failed to initialize ONNX session: {e}")
                raise

    def remove_background(self, img: Image.Image) -> Image.Image:
        """
        Removes the background from the image.
        Returns a new RGB image with the background replaced by white.
        """
        self._ensure_session()
        
        orig_img = img.convert('RGB')
        orig_size = orig_img.size
        
        # U2-Net expects 320x320 input
        target_size = (320, 320)
        resized_img = orig_img.resize(target_size, Image.Resampling.LANCZOS)
        img_array = np.array(resized_img, dtype=np.float32)
        
        # Normalization for U2-Net
        img_array = img_array / np.max(img_array)
        img_array[:, :, 0] = (img_array[:, :, 0] - 0.485) / 0.229
        img_array[:, :, 1] = (img_array[:, :, 1] - 0.456) / 0.224
        img_array[:, :, 2] = (img_array[:, :, 2] - 0.406) / 0.225
        
        # Transform to NCHW
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)

        # Inference
        input_name = self.session.get_inputs()[0].name
        outs = self.session.run(None, {input_name: img_array})
        
        # First output is the primary mask (D0)
        mask = outs[0][0][0]
        
        # Min-max normalization for the mask
        ma = np.max(mask)
        mi = np.min(mask)
        mask = (mask - mi) / (ma - mi + 1e-8)
        
        # Resize mask back to original image size
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_uint8).resize(orig_size, Image.Resampling.LANCZOS)
        
        # Composite the original image over a white background using the mask
        white_bg = Image.new("RGB", orig_size, (255, 255, 255))
        result = Image.composite(orig_img, white_bg, mask_img)
        
        return result

# Singleton instance
_bg_remover = None

def get_background_remover():
    global _bg_remover
    if _bg_remover is None:
        _bg_remover = BackgroundRemover()
    return _bg_remover
