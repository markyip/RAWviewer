"""
Lightweight local semantic image search (text -> image) with EXIF-aware metadata.

MVP goals:
- 100% local index and search (SQLite + CLIP embeddings)
- Incremental indexing by file mtime/size
- Optional metadata filters for quick narrowing
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
import hashlib
import sys
import gzip
import urllib.request
import tempfile
import threading
import concurrent.futures
from io import BytesIO
from functools import lru_cache
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageOps

import metadata_backend

from raw_file_extensions import RAW_FILE_EXTENSIONS

try:
    import pycountry
except Exception:
    pycountry = None


ProgressCallback = Optional[Callable[[int, int, str], None]]


def _load_index_source_image(file_path: str, max_size: int = 1024) -> Image.Image:
    """Load a small RGB image suitable for indexing/detection, preferring app caches."""
    import threading
    global _THREAD_LOCAL_DETECTORS
    if '_THREAD_LOCAL_DETECTORS' not in globals():
        _THREAD_LOCAL_DETECTORS = threading.local()
    _THREAD_LOCAL_DETECTORS.last_original_sizes = (0, 0)

    try:
        from image_cache import get_image_cache
        cache = get_image_cache()
        for getter_name in ("get_thumbnail", "get_preview"):
            try:
                arr = getattr(cache, getter_name)(file_path)
                if arr is not None:
                    im = Image.fromarray(np.asarray(arr, dtype=np.uint8)).convert("RGB")
                    _THREAD_LOCAL_DETECTORS.last_original_sizes = (im.width, im.height)
                    im.thumbnail((max_size, max_size), Image.Resampling.BICUBIC)
                    return im
            except Exception:
                continue
    except Exception:
        pass

    try:
        with Image.open(file_path) as im:
            _THREAD_LOCAL_DETECTORS.last_original_sizes = (im.width, im.height)
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((max_size, max_size), Image.Resampling.BICUBIC)
            return im.copy()
    except Exception:
        pass

    try:
        import rawpy  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            f"Cannot decode image for semantic index: {os.path.basename(file_path)}"
        ) from exc

    with rawpy.imread(file_path) as raw:
        _THREAD_LOCAL_DETECTORS.last_original_sizes = (raw.sizes.width, raw.sizes.height)
        try:
            thumb = raw.extract_thumb()
            if thumb is not None:
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    im = Image.open(BytesIO(thumb.data))
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    im.thumbnail((max_size, max_size), Image.Resampling.BICUBIC)
                    return im.copy()
                if thumb.format == rawpy.ThumbFormat.BITMAP:
                    im = Image.fromarray(thumb.data, mode="RGB")
                    im.thumbnail((max_size, max_size), Image.Resampling.BICUBIC)
                    return im
        except Exception:
            pass

        rgb = raw.postprocess(
            use_camera_wb=False,
            no_auto_bright=True,
            half_size=True,
            fast_half=True,
            user_flip=0,
            output_bps=8,
        )
    im = Image.fromarray(rgb, mode="RGB")
    im.thumbnail((max_size, max_size), Image.Resampling.BICUBIC)
    return im


@dataclass
class SearchHit:
    file_path: str
    score: float
    file_name: str = ""
    capture_time: str = ""
    camera_model: str = ""
    lens_model: str = ""
    iso: int = 0
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    city: str = ""
    admin1: str = ""
    country: str = ""
    country_code: str = ""
    face_count: int = 0


class MobileCLIPCoreMLBackend:
    """Optional macOS Core ML backend for MobileCLIP.

    Model assets are expected in:
    - RAWVIEWER_MOBILECLIP_MODEL_DIR, or
    - ~/.rawviewer_cache/mobileclip_coreml

    Recognized bundle pairs (first match wins under ``model_dir``):

    - **Apple Hub (S2):** ``mobileclip_s2_image.mlpackage`` / ``mobileclip_s2_text.mlpackage``
    - **App export (``export_mobileclip2_coreml.py --for-app``):**
      ``mobileclip2_s0_image.mlpackage`` / ``mobileclip2_s0_text.mlpackage``

    Note: text encoding also needs a tokenizer compatible with the Core ML text
    encoder. Until a tokenizer asset is present, the backend reports unavailable
    rather than silently falling back to metadata-only results.

    Exported MobileCLIP2 models (see ``scripts/export_mobileclip2_coreml.py``)
    expose the image encoder as FLOAT32 MultiArray ``[1,3,256,256]`` in NCHW
    pixel scale ``[0,1]``. Apple-shipped MobileCLIP S2 bundles use MLFeatureTypeImage;
    ``encode_image`` supports both via model introspection.
    """

    MODEL_ID = "mobileclip-coreml-s2"
    HUB_REPO_ID = "apple/coreml-mobileclip"
    IMAGE_MODEL_FILE = "mobileclip_s2_image.mlpackage"
    TEXT_MODEL_FILE = "mobileclip_s2_text.mlpackage"
    TOKENIZER_URL = "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"
    SUPPORTS_HUB_DOWNLOAD = True

    _COREML_BUNDLE_PAIRS: tuple[tuple[str, str], ...] = (
        ("mobileclip_s2_image.mlpackage", "mobileclip_s2_text.mlpackage"),
        ("mobileclip2_s0_image.mlpackage", "mobileclip2_s0_text.mlpackage"),
    )

    def __init__(self, model_dir: Optional[str] = None):
        if model_dir is None:
            model_dir = self._default_model_dir()
        self.model_dir = model_dir
        pair = self._find_bundle_basenames(model_dir)
        if pair is not None:
            img_f, txt_f = pair
            self.image_model_path = os.path.join(model_dir, img_f)
            self.text_model_path = os.path.join(model_dir, txt_f)
            if img_f.startswith("mobileclip2_s0"):
                self.MODEL_ID = "mobileclip-coreml-2-s0"
            else:
                self.MODEL_ID = MobileCLIPCoreMLBackend.MODEL_ID
        else:
            self.image_model_path = os.path.join(model_dir, self.IMAGE_MODEL_FILE)
            self.text_model_path = os.path.join(model_dir, self.TEXT_MODEL_FILE)
            self.MODEL_ID = MobileCLIPCoreMLBackend.MODEL_ID
        self.tokenizer_path = os.path.join(model_dir, "bpe_simple_vocab_16e6.txt.gz")
        self._image_model = None
        self._text_model = None
        self._tokenizer = None
        self._CoreML = None
        self._Foundation = None
        self._Quartz = None

    @classmethod
    def _find_bundle_basenames(cls, model_dir: str) -> tuple[str, str] | None:
        tok = os.path.join(model_dir, "bpe_simple_vocab_16e6.txt.gz")
        if not os.path.isfile(tok):
            return None
        for img_f, txt_f in cls._COREML_BUNDLE_PAIRS:
            ip = os.path.join(model_dir, img_f)
            tp = os.path.join(model_dir, txt_f)
            if cls._mlpackage_complete(ip) and cls._mlpackage_complete(tp):
                return img_f, txt_f
        return None

    @staticmethod
    def _candidate_model_dirs() -> List[str]:
        dirs: List[str] = []
        env_dir = os.environ.get("RAWVIEWER_MOBILECLIP_MODEL_DIR")
        if env_dir:
            dirs.append(env_dir)
        dirs.append(os.path.expanduser("~/.rawviewer_cache/mobileclip_coreml"))
        if getattr(sys, "frozen", False):
            exe_dir = os.path.dirname(sys.executable)
            meipass = getattr(sys, "_MEIPASS", None)
            dirs.extend(
                [
                    os.path.join(exe_dir, "models", "mobileclip2_coreml"),
                    os.path.join(exe_dir, "mobileclip_coreml"),
                    os.path.join(exe_dir, "..", "Resources", "mobileclip_coreml"),
                    os.path.join(exe_dir, "..", "Resources", "models", "mobileclip2_coreml"),
                    os.path.join(exe_dir, "..", "Resources", "models", "mobileclip_coreml"),
                    os.path.join(exe_dir, "..", "Frameworks", "mobileclip_coreml"),
                    os.path.join(exe_dir, "..", "Frameworks", "models", "mobileclip2_coreml"),
                ]
            )
            if meipass:
                dirs.extend(
                    [
                        os.path.join(meipass, "models", "mobileclip2_coreml"),
                        os.path.join(meipass, "models", "mobileclip_coreml"),
                    ]
                )
        module_dir = os.path.dirname(os.path.abspath(__file__))
        dirs.extend(
            [
                os.path.join(module_dir, "..", "models", "mobileclip2_coreml"),
                os.path.join(module_dir, "..", "models", "mobileclip_coreml"),
                os.path.join(module_dir, "..", "mobileclip_coreml"),
            ]
        )
        out: List[str] = []
        for d in dirs:
            full = os.path.realpath(os.path.abspath(os.path.expanduser(d)))
            if full not in out:
                out.append(full)
        return out

    @classmethod
    def _default_model_dir(cls) -> str:
        for d in cls._candidate_model_dirs():
            if cls._find_bundle_basenames(d) is not None:
                return d
        return cls._candidate_model_dirs()[0]

    def availability_error(self) -> str:
        if sys.platform != "darwin":
            return "MobileCLIP Core ML is only available on macOS"
        if not self._mlpackage_complete(self.image_model_path):
            return f"Missing MobileCLIP image model in {self.model_dir}"
        if not self._mlpackage_complete(self.text_model_path):
            return f"Missing MobileCLIP text model in {self.model_dir}"
        if not os.path.exists(self.tokenizer_path):
            return f"Missing MobileCLIP tokenizer in {self.model_dir}"
        try:
            import CoreML  # noqa: F401
            import Foundation  # noqa: F401
            import Quartz  # noqa: F401
        except Exception as exc:
            return f"Missing native Core ML runtime: {exc}"
        return ""

    def available(self) -> bool:
        return self.availability_error() == ""

    @staticmethod
    def _mlpackage_complete(path: str) -> bool:
        return (
            os.path.isdir(path)
            and os.path.isfile(os.path.join(path, "Manifest.json"))
            and os.path.isfile(os.path.join(path, "Data", "com.apple.CoreML", "model.mlmodel"))
            and os.path.isfile(os.path.join(path, "Data", "com.apple.CoreML", "weights", "weight.bin"))
        )

    def download_assets(self, progress_callback: Optional[Callable[[str], None]] = None) -> str:
        """Download MobileCLIP S2 Core ML assets into the backend model directory."""
        if sys.platform != "darwin":
            raise RuntimeError("MobileCLIP Core ML download is only supported on macOS")

        def _progress(message: str) -> None:
            if progress_callback:
                progress_callback(message)

        os.makedirs(self.model_dir, exist_ok=True)
        _progress("Downloading MobileCLIP Core ML models...")
        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:
            raise RuntimeError(
                "MobileCLIP auto-download requires 'huggingface_hub'. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        snapshot_download(
            repo_id=self.HUB_REPO_ID,
            allow_patterns=[
                f"{self.IMAGE_MODEL_FILE}/**",
                f"{self.TEXT_MODEL_FILE}/**",
            ],
            local_dir=self.model_dir,
        )

        if not os.path.exists(self.tokenizer_path):
            _progress("Downloading MobileCLIP tokenizer...")
            urllib.request.urlretrieve(self.TOKENIZER_URL, self.tokenizer_path)

        err = self.availability_error()
        if err:
            raise RuntimeError(err)
        _progress("MobileCLIP assets ready")
        return self.model_dir

    def _load_models(self):
        if self._image_model is not None and self._text_model is not None:
            return
        import CoreML
        import Foundation
        import Quartz

        self._CoreML = CoreML
        self._Foundation = Foundation
        self._Quartz = Quartz

        def _load_one(path: str):
            url = Foundation.NSURL.fileURLWithPath_(path)
            
            # Persistent cache for compiled models to avoid O(seconds) re-compilation
            cache_dir = os.path.expanduser("~/.rawviewer_cache/compiled_models")
            os.makedirs(cache_dir, exist_ok=True)
            
            try:
                st = os.stat(path)
                mtime = int(st.st_mtime)
            except Exception:
                mtime = 0
            
            h = hashlib.md5(f"{path}_{mtime}".encode()).hexdigest()
            compiled_path = os.path.join(cache_dir, f"{os.path.basename(path)}_{h}.mlmodelc")
            compiled_url = Foundation.NSURL.fileURLWithPath_(compiled_path)
            
            if not os.path.exists(compiled_path):
                tmp_url, compile_error = CoreML.MLModel.compileModelAtURL_error_(url, None)
                if compile_error is not None or tmp_url is None:
                    raise RuntimeError(f"Failed to compile Core ML model: {compile_error}")
                
                mgr = Foundation.NSFileManager.defaultManager()
                if os.path.exists(compiled_path):
                    mgr.removeItemAtURL_error_(compiled_url, None)
                
                success, move_error = mgr.moveItemAtURL_toURL_error_(tmp_url, compiled_url, None)
                if not success:
                    compiled_url = tmp_url
            
            model, load_error = CoreML.MLModel.modelWithContentsOfURL_error_(compiled_url, None)
            if load_error is not None or model is None:
                raise RuntimeError(f"Failed to load Core ML model: {load_error}")
            return model

        self._image_model = _load_one(self.image_model_path)
        self._text_model = _load_one(self.text_model_path)

    @staticmethod
    def _native_feature_name(model, direction: str) -> str:
        desc = model.modelDescription()
        features = (
            desc.inputDescriptionsByName()
            if direction == "input"
            else desc.outputDescriptionsByName()
        )
        names = list(features.keys())
        if not names:
            raise RuntimeError(f"Core ML model has no {direction} features")
        return str(names[0])

    @staticmethod
    def _normalize(vec) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        return arr

    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = _ClipBPETokenizer(self.tokenizer_path)
        return self._tokenizer

    def encode_text(self, text: str) -> np.ndarray:
        self._load_models()
        CoreML = self._CoreML
        tokenizer = self._ensure_tokenizer()
        tokens = np.asarray([tokenizer.encode_for_clip(text)], dtype=np.int32)
        input_name = self._native_feature_name(self._text_model, "input")
        output_name = self._native_feature_name(self._text_model, "output")
        multi_array = self._int32_multi_array(tokens.reshape(-1))
        feature = CoreML.MLFeatureValue.featureValueWithMultiArray_(multi_array)
        provider, err = CoreML.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(
            {input_name: feature}, None
        )
        if err is not None or provider is None:
            raise RuntimeError(f"Failed to create Core ML text input: {err}")
        out, err = self._text_model.predictionFromFeatures_error_(provider, None)
        if err is not None or out is None:
            raise RuntimeError(f"MobileCLIP text prediction failed: {err}")
        return self._normalize(self._multi_array_to_numpy(out.featureValueForName_(output_name).multiArrayValue()))

    def _float32_multi_array_nchw(self, tensor_nchw: np.ndarray):
        CoreML = self._CoreML
        t = np.asarray(tensor_nchw, dtype=np.float32).reshape(1, 3, 256, 256)
        flat = np.ascontiguousarray(t).ravel(order="C")
        arr, err = CoreML.MLMultiArray.alloc().initWithShape_dataType_error_(
            [1, 3, 256, 256], CoreML.MLMultiArrayDataTypeFloat32, None
        )
        if err is not None or arr is None:
            raise RuntimeError(f"Failed to allocate Core ML image tensor: {err}")
        for i, value in enumerate(flat):
            arr.setObject_atIndexedSubscript_(float(value), i)
        return arr

    def encode_image(self, file_path: str) -> np.ndarray:
        self._load_models()
        CoreML = self._CoreML
        # Core ML encoder input is fixed (typically 256×256 NCHW in our exports). Loading a richer
        # source (preview/thumbnail) before bicubic resize to 256 can help vs loading tiny 512-max inputs.
        im = _load_index_source_image(file_path, max_size=1024).resize(
            (256, 256), Image.Resampling.BICUBIC
        )
        desc = self._image_model.modelDescription()
        input_name = self._native_feature_name(self._image_model, "input")
        output_name = self._native_feature_name(self._image_model, "output")
        by_name = desc.inputDescriptionsByName()
        feature_desc = by_name.objectForKey_(input_name) if by_name is not None else None
        if feature_desc is None and by_name is not None:
            for key in by_name:
                if str(key) == str(input_name):
                    feature_desc = by_name.objectForKey_(key)
                    break
        if feature_desc is None:
            raise RuntimeError("Core ML image model has no input description")
        in_type = int(feature_desc.type())

        feature = None
        if in_type == CoreML.MLFeatureTypeMultiArray:
            rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
            nchw = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]
            multi = self._float32_multi_array_nchw(nchw)
            feature = CoreML.MLFeatureValue.featureValueWithMultiArray_(multi)
        elif in_type == CoreML.MLFeatureTypeImage:
            pixel_buffer = self._image_to_pixel_buffer(im)
            feature = CoreML.MLFeatureValue.alloc().initWithValue_type_(
                pixel_buffer, CoreML.MLFeatureTypeImage
            )
        else:
            raise RuntimeError(f"Unsupported Core ML image encoder input type: {in_type}")

        provider, err = CoreML.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(
            {input_name: feature}, None
        )
        if err is not None or provider is None:
            raise RuntimeError(f"Failed to create Core ML image input: {err}")
        out, err = self._image_model.predictionFromFeatures_error_(provider, None)
        if err is not None or out is None:
            raise RuntimeError(f"MobileCLIP image prediction failed: {err}")
        return self._normalize(self._multi_array_to_numpy(out.featureValueForName_(output_name).multiArrayValue()))

    def _int32_multi_array(self, values: np.ndarray):
        CoreML = self._CoreML
        arr, err = CoreML.MLMultiArray.alloc().initWithShape_dataType_error_(
            [1, int(values.size)], CoreML.MLMultiArrayDataTypeInt32, None
        )
        if err is not None or arr is None:
            raise RuntimeError(f"Failed to allocate Core ML token array: {err}")
        flat = np.asarray(values, dtype=np.int32).reshape(-1)
        for i, value in enumerate(flat):
            arr.setObject_atIndexedSubscript_(int(value), i)
        return arr

    @staticmethod
    def _multi_array_to_numpy(multi_array) -> np.ndarray:
        values = multi_array.numberArray()
        return np.asarray([float(values[i]) for i in range(len(values))], dtype=np.float32)

    def _image_to_pixel_buffer(self, im: Image.Image):
        Quartz = self._Quartz
        im = im.convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
        status, pixel_buffer = Quartz.CVPixelBufferCreate(
            None,
            256,
            256,
            Quartz.kCVPixelFormatType_32BGRA,
            {
                Quartz.kCVPixelBufferCGImageCompatibilityKey: True,
                Quartz.kCVPixelBufferCGBitmapContextCompatibilityKey: True,
            },
            None,
        )
        if status != 0 or pixel_buffer is None:
            raise RuntimeError(f"Failed to create CVPixelBuffer: status {status}")
        rgba = np.asarray(im, dtype=np.uint8)
        bgra = np.empty((256, 256, 4), dtype=np.uint8)
        bgra[..., 0] = rgba[..., 2]
        bgra[..., 1] = rgba[..., 1]
        bgra[..., 2] = rgba[..., 0]
        bgra[..., 3] = 255
        Quartz.CVPixelBufferLockBaseAddress(pixel_buffer, 0)
        try:
            bytes_per_row = int(Quartz.CVPixelBufferGetBytesPerRow(pixel_buffer))
            base = Quartz.CVPixelBufferGetBaseAddress(pixel_buffer)
            buf = base.as_buffer(bytes_per_row * 256)
            row_bytes = 256 * 4
            for y in range(256):
                start = y * bytes_per_row
                buf[start:start + row_bytes] = bgra[y].tobytes()
        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(pixel_buffer, 0)
        return pixel_buffer


class MobileCLIPONNXBackend:
    """Windows/Linux ONNX backend for MobileCLIP2-S0.
    
    Requires 'onnxruntime' and 'numpy'.
    """

    MODEL_ID = "mobileclip-onnx-2-s0"
    HUB_REPO_ID = "plhery/mobileclip2-onnx"
    IMAGE_MODEL_FILE = "image_encoder.onnx"
    TEXT_MODEL_FILE = "text_encoder.onnx"
    TOKENIZER_URL = "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"
    SUPPORTS_HUB_DOWNLOAD = True

    def __init__(self, model_dir: Optional[str] = None):
        if model_dir is None:
            model_dir = self._default_model_dir()
        self.model_dir = model_dir
        self.image_model_path = os.path.join(model_dir, self.IMAGE_MODEL_FILE)
        self.text_model_path = os.path.join(model_dir, self.TEXT_MODEL_FILE)
        self.tokenizer_path = os.path.join(model_dir, "bpe_simple_vocab_16e6.txt.gz")
        self._image_session = None
        self._text_session = None
        self._tokenizer = None

    @staticmethod
    def _candidate_model_dirs() -> List[str]:
        dirs: List[str] = []
        env_dir = os.environ.get("RAWVIEWER_MOBILECLIP_MODEL_DIR")
        if env_dir:
            dirs.append(env_dir)
            
        if getattr(sys, "frozen", False):
            # Prioritize the actual executable directory for external (non-bundled) models
            exe_dir = os.path.dirname(sys.executable)
            dirs.append(os.path.join(exe_dir, "models", "mobileclip_onnx"))
            dirs.append(os.path.join(exe_dir, "mobileclip_onnx"))
            
            # Fallback to PyInstaller temporary extract directory (_MEIPASS)
            if hasattr(sys, "_MEIPASS"):
                dirs.append(os.path.join(sys._MEIPASS, "models", "mobileclip_onnx"))
            
        dirs.append(os.path.expanduser("~/.rawviewer_cache/mobileclip_onnx"))
        
        module_dir = os.path.dirname(os.path.abspath(__file__))
        dirs.append(os.path.join(module_dir, "..", "models", "mobileclip_onnx"))
        return dirs

    @classmethod
    def _default_model_dir(cls) -> str:
        for d in cls._candidate_model_dirs():
            if (
                os.path.exists(os.path.join(d, cls.IMAGE_MODEL_FILE))
                and os.path.exists(os.path.join(d, cls.TEXT_MODEL_FILE))
            ):
                return d
        return cls._candidate_model_dirs()[0]

    def availability_error(self) -> str:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return "Missing 'onnxruntime' dependency"
        if not os.path.exists(self.image_model_path):
            return f"Missing image model: {self.IMAGE_MODEL_FILE}"
        if not os.path.exists(self.text_model_path):
            return f"Missing text model: {self.TEXT_MODEL_FILE}"
        return ""

    def available(self) -> bool:
        return self.availability_error() == ""

    def download_assets(self, progress_callback: Optional[Callable[[str], None]] = None) -> str:
        def _progress(message: str) -> None:
            if progress_callback:
                progress_callback(message)

        os.makedirs(self.model_dir, exist_ok=True)
        _progress("Downloading MobileCLIP ONNX models...")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise RuntimeError("MobileCLIP download requires 'huggingface_hub' (pip install huggingface_hub)")

        # Mapping of remote path in HF repo to local filename expected by RAWviewer
        files_to_download = {
            "onnx/s0/vision_model.onnx": self.IMAGE_MODEL_FILE,
            "onnx/s0/text_model.onnx": self.TEXT_MODEL_FILE
        }
        
        for remote_path, local_name in files_to_download.items():
            _progress(f"Fetching {local_name}...")
            hf_hub_download(
                repo_id=self.HUB_REPO_ID,
                filename=remote_path,
                local_dir=self.model_dir,
                local_dir_use_symlinks=False
            )
            
            # Handle nesting created by local_dir
            downloaded_path = os.path.join(self.model_dir, remote_path)
            target_path = os.path.join(self.model_dir, local_name)
            
            if os.path.exists(downloaded_path):
                if os.path.exists(target_path):
                    os.remove(target_path)
                os.rename(downloaded_path, target_path)

        # Clean up empty subdirectories
        onnx_dir = os.path.join(self.model_dir, "onnx")
        if os.path.exists(onnx_dir):
            import shutil
            shutil.rmtree(onnx_dir)
            
        if not os.path.exists(self.tokenizer_path):
            _progress("Downloading CLIP tokenizer...")
            import urllib.request
            urllib.request.urlretrieve(self.TOKENIZER_URL, self.tokenizer_path)
            
        return self.model_dir

    def _ensure_sessions(self):
        if self._image_session is not None:
            return
        import onnxruntime as ort
        
        # Priority-ordered list of high-performance execution providers
        providers = [
            'CUDAExecutionProvider',    # NVIDIA GPU
            'TensorrtExecutionProvider', # NVIDIA TensorRT
            'DmlExecutionProvider',     # Windows DirectML (AMD/Intel/NVIDIA)
            'CPUExecutionProvider'       # Fallback
        ]
        
        available_providers = ort.get_available_providers()
        selected_providers = [p for p in providers if p in available_providers]
        
        print(f"[SemanticSearch] Initializing MobileCLIP ONNX session. Available providers: {available_providers}, using: {selected_providers}", flush=True)
        
        self._image_session = ort.InferenceSession(self.image_model_path, providers=selected_providers)
        self._text_session = ort.InferenceSession(self.text_model_path, providers=selected_providers)

    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = _ClipBPETokenizer(self.tokenizer_path)
        return self._tokenizer

    def encode_text(self, text: str) -> np.ndarray:
        self._ensure_sessions()
        tokenizer = self._ensure_tokenizer()
        # MobileCLIP2 ONNX text encoder expects int64 token IDs (ORT rejects int32).
        tokens = np.asarray([tokenizer.encode_for_clip(text)], dtype=np.int64)
        
        inputs = {self._text_session.get_inputs()[0].name: tokens}
        outputs = self._text_session.run(None, inputs)
        return self._normalize(outputs[0])

    def encode_image(self, file_path: str) -> np.ndarray:
        self._ensure_sessions()
        # MobileCLIP2-S0 typically uses 256x256
        im = _load_index_source_image(file_path, max_size=1024).resize((256, 256), Image.Resampling.BICUBIC)
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        nchw = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]
        
        inputs = {self._image_session.get_inputs()[0].name: nchw}
        outputs = self._image_session.run(None, inputs)
        return self._normalize(outputs[0])

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        return arr


def resolve_mobileclip_backend() -> Any:
    """Prefer platform-native; then ONNX; then Core ML fallback."""
    if sys.platform == "darwin":
        # Prefer Core ML on macOS (single shipped class: MobileCLIPCoreMLBackend / S2)
        for cls in (MobileCLIPCoreMLBackend,):
            inst = cls()
            if inst.available():
                return inst
        # Try ONNX on macOS too if Core ML fails
        inst = MobileCLIPONNXBackend()
        if inst.available():
            return inst
        return MobileCLIPCoreMLBackend()
    else:
        # Windows/Linux prefer ONNX
        inst = MobileCLIPONNXBackend()
        if inst.available():
            return inst
        return inst # Return anyway for download_assets availability


def _bytes_to_unicode() -> Dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs]))


def _get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class _ClipBPETokenizer:
    """Minimal OpenAI CLIP BPE tokenizer for MobileCLIP Core ML text encoder."""

    def __init__(self, bpe_path: str):
        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with gzip.open(bpe_path, "rt", encoding="utf-8") as f:
            merges = f.read().split("\n")
        merges = merges[1:49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges if merge]
        vocab = list(_bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        self.sot = self.encoder["<|startoftext|>"]
        self.eot = self.encoder["<|endoftext|>"]

    @lru_cache(maxsize=8192)
    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        out = " ".join(word)
        self.cache[token] = out
        return out

    def encode(self, text: str) -> List[int]:
        bpe_tokens: List[int] = []
        for token in re.findall(r"<\|startoftext\|>|<\|endoftext\|>|[\w']+|[^\s\w]", text.lower()):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def encode_for_clip(self, text: str, context_length: int = 77) -> List[int]:
        tokens = [self.sot] + self.encode(text)[: context_length - 2] + [self.eot]
        return tokens + [0] * (context_length - len(tokens))


class SemanticImageIndex:
    """SQLite-backed CLIP embedding index for local semantic search."""

    # Camera/LibRaw RAW extensions (shared with is_raw_file); used for format:raw.
    _RAW_FILE_EXTENSIONS = RAW_FILE_EXTENSIONS

    # Tokens that mean "filter by extension" when used as filename:jpg / file raw (not substring search).
    _FORMAT_HINT_FILENAME_TOKENS: frozenset = (
        frozenset(
            {
                "jpeg",
                "jpg",
                "jpe",
                "tif",
                "tiff",
                "png",
                "gif",
                "bmp",
                "webp",
                "heic",
                "heif",
                "avif",
                "raw",
            }
        )
        | _RAW_FILE_EXTENSIONS
    )

    # Gallery search: these map to indexed Vision face_count (>0), not CLIP text similarity.
    _FACE_COUNT_POSITIVE_TOKENS = frozenset(
        {
            "has:face",
            "has:faces",
            "has:people",
            "has:person",
            "face",
            "faces",
            "person",
            "people",
            "human",
            "humans",
            "portrait",
        }
    )
    _FACE_COUNT_NEGATIVE_TOKENS = frozenset(
        {
            "no:face",
            "no:faces",
            "no:people",
            "no:person",
            "no:human",
            "no:humans",
        }
    )

    def __init__(self, db_path: Optional[str] = None, model_name: Optional[str] = None):
        if db_path is None:
            cache_dir = os.path.expanduser("~/.rawviewer_cache")
            os.makedirs(cache_dir, exist_ok=True)
            db_path = os.path.join(cache_dir, "semantic_index.db")
        self.db_path = db_path
        self._mobileclip_backend = None # Lazy load
        self._model_name = model_name
        self._index_conn = None
        self._rg_lock = threading.Lock()
        self._init_db_if_needed()

    @property
    def backend(self):
        if self._mobileclip_backend is None:
            self._mobileclip_backend = resolve_mobileclip_backend()
        return self._mobileclip_backend

    @property
    def model_name(self):
        if self._model_name is None:
            if hasattr(self.backend, "MODEL_ID"):
                self._model_name = self.backend.MODEL_ID
            else:
                self._model_name = "unknown"
        return self._model_name

    def _init_db_if_needed(self):
        self._model = None
        self._reverse_geocoder = None
        self._conn = sqlite3.connect(
            self.db_path, check_same_thread=False, timeout=60.0
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=60000")
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_index (
                file_path TEXT PRIMARY KEY,
                file_name TEXT,
                file_signature TEXT,
                file_size INTEGER NOT NULL,
                file_mtime REAL NOT NULL,
                mtime_ns INTEGER,
                semantic_ready INTEGER DEFAULT 1,
                model_name TEXT NOT NULL,
                dim INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                capture_time TEXT,
                camera_model TEXT,
                lens_model TEXT,
                iso INTEGER,
                width INTEGER,
                height INTEGER,
                gps_lat REAL,
                gps_lon REAL,
                gps_raw TEXT,
                city TEXT,
                admin1 TEXT,
                country TEXT,
                country_code TEXT,
                face_count INTEGER,
                orientation INTEGER,
                updated_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_semantic_capture_time ON semantic_index(capture_time)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_semantic_camera_model ON semantic_index(camera_model)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_semantic_gps_latlon ON semantic_index(gps_lat, gps_lon)"
        )
        # Backward-compatible migration for existing DBs created before GPS fields.
        cols = {
            r[1]
            for r in self._conn.execute("PRAGMA table_info(semantic_index)").fetchall()
        }
        if "file_name" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN file_name TEXT")
        if "file_signature" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN file_signature TEXT")
        if "mtime_ns" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN mtime_ns INTEGER")
        if "semantic_ready" not in cols:
            self._conn.execute(
                "ALTER TABLE semantic_index ADD COLUMN semantic_ready INTEGER DEFAULT 1"
            )
            self._conn.execute(
                "UPDATE semantic_index SET semantic_ready = 1 WHERE semantic_ready IS NULL"
            )
        if "gps_lat" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN gps_lat REAL")
        if "gps_lon" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN gps_lon REAL")
        if "gps_raw" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN gps_raw TEXT")
        if "city" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN city TEXT")
        if "admin1" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN admin1 TEXT")
        if "country" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN country TEXT")
        if "country_code" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN country_code TEXT")
        if "face_count" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN face_count INTEGER")
        if "orientation" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN orientation INTEGER")
        self._conn.commit()
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_semantic_signature_model ON semantic_index(file_signature, model_name)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_semantic_ready_model ON semantic_index(semantic_ready, model_name)"
        )
        self._conn.commit()
        self._backfill_file_signatures()

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Semantic search requires 'sentence-transformers'. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc
        self._model = SentenceTransformer(self.model_name)
        return self._model

    @staticmethod
    def semantic_backend_available() -> bool:
        if resolve_mobileclip_backend().available():
            return True
        try:
            import sentence_transformers  # noqa: F401
            return True
        except Exception:
            return False

    def semantic_backend_error(self) -> str:
        if self.model_name.startswith("mobileclip-"):
            backend = self._mobileclip_backend or resolve_mobileclip_backend()
            return backend.availability_error()
        try:
            self._ensure_model()
            return ""
        except Exception as exc:
            return str(exc)

    def mobileclip_supports_hub_download(self) -> bool:
        return bool(
            self.model_name.startswith("mobileclip-")
            and self._mobileclip_backend is not None
            and getattr(self._mobileclip_backend, "SUPPORTS_HUB_DOWNLOAD", False)
        )

    def download_semantic_backend_assets(
        self, progress_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        if self.model_name.startswith("mobileclip-"):
            backend = self._mobileclip_backend or resolve_mobileclip_backend()
            path = backend.download_assets(progress_callback=progress_callback)
            self._mobileclip_backend = backend
            return path
        raise RuntimeError("This semantic backend has no downloadable assets")

    def _ensure_reverse_geocoder(self):
        if self._reverse_geocoder is not None:
            return self._reverse_geocoder
        with self._rg_lock:
            if self._reverse_geocoder is not None:
                return self._reverse_geocoder
            try:
                import reverse_geocoder as rg  # type: ignore
                # reverse_geocoder lazy-loads its CSV on the first .search() call.
                # Its lazy loader is NOT thread-safe, which causes massive I/O stalls
                # if multiple worker threads call .search() concurrently.
                # We initialize it here once inside the lock.
                # CRITICAL: We MUST use mode=1, otherwise it defaults to mode=2
                # which creates a multiprocessing.Pool. On Windows, this causes a fork bomb
                # that exhausts system memory because it spawns new Python interpreters!
                import logging
                logging.getLogger(__name__).info("[VISION] Initializing reverse geocoder KD-tree...")
                rg.search((0.0, 0.0), mode=1)
            except Exception:
                self._reverse_geocoder = False
                return False
            self._reverse_geocoder = rg
            return rg

    @staticmethod
    def _country_name_from_code(code: str) -> str:
        cc = (code or "").strip().upper()
        if not cc:
            return ""

        # Hardcoded fallbacks for common regions to ensure search reliability
        _FALLBACKS = {
            "JP": "Japan",
            "US": "United States",
            "GB": "United Kingdom",
            "UK": "United Kingdom",
            "TW": "Taiwan",
        }
        if cc in _FALLBACKS:
            return _FALLBACKS[cc]

        if pycountry is not None:
            try:
                c = pycountry.countries.get(alpha_2=cc)
                if c is not None:
                    name = str(getattr(c, "name", "") or "")
                    # Keep UI-friendly labels for search/filter display.
                    if name == "Taiwan, Province of China":
                        return "Taiwan"
                    return name
            except Exception:
                pass
        return cc

    @staticmethod
    def _to_blob(vec: np.ndarray) -> bytes:
        arr = np.asarray(vec, dtype=np.float32)
        return arr.tobytes()

    @staticmethod
    def _from_blob(blob: bytes, dim: int) -> np.ndarray:
        arr = np.frombuffer(blob, dtype=np.float32)
        if dim > 0 and arr.size != dim:
            arr = arr[:dim]
        return arr

    @staticmethod
    def _safe_int(v) -> int:
        try:
            values = getattr(v, "values", None)
            if values:
                return SemanticImageIndex._safe_int(values[0])
            ratio_value = SemanticImageIndex._ratio_to_float(v)
            if ratio_value is not None:
                return int(round(ratio_value))
            text = str(v or "").strip()
            if not text:
                return 0
            # exifread may stringify numeric tags as "[800]" or "800 800".
            m = re.search(r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", text)
            if not m:
                return 0
            token = m.group(0)
            if "/" in token:
                num, den = token.split("/", 1)
                den_f = float(den) if float(den) != 0 else 1.0
                return int(round(float(num) / den_f))
            return int(round(float(token)))
        except Exception:
            return 0

    @staticmethod
    def _tag_text(tags: Dict[str, object], *names: str) -> str:
        for name in names:
            value = tags.get(name)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    @classmethod
    def _tag_int(cls, tags: Dict[str, object], *names: str) -> int:
        for name in names:
            if name not in tags:
                continue
            value = cls._safe_int(tags.get(name))
            if value:
                return value
        return 0

    @staticmethod
    def _pil_dimensions(file_path: str) -> tuple[int, int]:
        # Check thread-local cache first
        global _THREAD_LOCAL_DETECTORS
        if '_THREAD_LOCAL_DETECTORS' in globals() and hasattr(_THREAD_LOCAL_DETECTORS, 'last_original_sizes'):
            w, h = _THREAD_LOCAL_DETECTORS.last_original_sizes
            if w > 0 and h > 0:
                return w, h

        try:
            with Image.open(file_path) as im:
                return int(im.width), int(im.height)
        except Exception:
            pass
        try:
            import rawpy  # type: ignore
            with rawpy.imread(file_path) as raw:
                return int(raw.sizes.width), int(raw.sizes.height)
        except Exception:
            pass
        return 0, 0

    @staticmethod
    def _clean_join(*parts: str) -> str:
        out: List[str] = []
        for part in parts:
            p = (part or "").strip()
            if p and p not in out:
                out.append(p)
        return " ".join(out).strip()

    @staticmethod
    def _legacy_int(v) -> int:
        try:
            return int(v)
        except Exception:
            return 0

    @staticmethod
    def _mtime_matches(stored_mtime: float, st: os.stat_result) -> bool:
        """
        Compare mtime with tolerance to avoid cross-session float precision drift.
        """
        try:
            current = float(st.st_mtime)
            return abs(float(stored_mtime) - current) <= 1e-3
        except Exception:
            return False

    @staticmethod
    @lru_cache(maxsize=16384)
    def _canonical_path(file_path: str) -> str:
        if not file_path:
            return ""
        try:
            # OPTIMIZATION: If already absolute, skip abspath() which can hit disk/slow down on Windows.
            # Pure string normalization is much faster for thousands of paths on the UI thread.
            if os.path.isabs(file_path):
                ret = os.path.normpath(file_path)
            else:
                ret = os.path.normpath(os.path.abspath(file_path))
            if sys.platform.startswith("win"):
                ret = ret.lower()
            return ret
        except Exception:
            if sys.platform.startswith("win"):
                return file_path.lower()
            return file_path

    @staticmethod
    def _path_aliases(file_path: str) -> List[str]:
        # Fast string-based aliases only.
        ap = os.path.abspath(file_path)
        np = os.path.normpath(ap)
        aliases = [file_path]
        if ap != file_path:
            aliases.append(ap)
        if np not in aliases:
            aliases.append(np)
        return aliases

    def _upsert_metadata(self, canonical_fp: str, st: os.stat_result, meta: Dict[str, Any], conn: Optional[sqlite3.Connection] = None) -> None:
        """Helper to insert or update metadata in the index."""
        db = conn if conn is not None else self._conn
        file_name = os.path.basename(canonical_fp)
        file_signature = self._file_signature_from_stat(canonical_fp, st)
        mtime_ns = self._mtime_ns_from_stat(st)
        
        db.execute(
            """
            INSERT INTO semantic_index (
                file_path, file_name, file_signature, file_size, file_mtime, mtime_ns,
                semantic_ready,
                model_name, dim, embedding,
                capture_time, camera_model, lens_model, iso, width, height,
                gps_lat, gps_lon, gps_raw, city, admin1, country, country_code, face_count, orientation, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                file_name=excluded.file_name,
                file_signature=excluded.file_signature,
                file_size=excluded.file_size,
                file_mtime=excluded.file_mtime,
                mtime_ns=excluded.mtime_ns,
                semantic_ready=excluded.semantic_ready,
                model_name=excluded.model_name,
                dim=excluded.dim,
                embedding=excluded.embedding,
                capture_time=excluded.capture_time,
                camera_model=excluded.camera_model,
                lens_model=excluded.lens_model,
                iso=excluded.iso,
                width=excluded.width,
                height=excluded.height,
                gps_lat=excluded.gps_lat,
                gps_lon=excluded.gps_lon,
                gps_raw=excluded.gps_raw,
                city=excluded.city,
                admin1=excluded.admin1,
                country=excluded.country,
                country_code=excluded.country_code,
                face_count=excluded.face_count,
                orientation=excluded.orientation,
                updated_at=excluded.updated_at
            """,
            (
                canonical_fp,
                file_name,
                file_signature,
                int(st.st_size),
                float(st.st_mtime),
                mtime_ns,
                0,  # semantic_ready = 0 until Phase 2
                self.model_name,
                0,
                b"",
                str(meta.get("capture_time") or ""),
                str(meta.get("camera_model") or ""),
                str(meta.get("lens_model") or ""),
                int(meta.get("iso") or 0),
                int(meta.get("width") or 0),
                int(meta.get("height") or 0),
                float(meta["gps_lat"]) if meta.get("gps_lat") is not None else None,
                float(meta["gps_lon"]) if meta.get("gps_lon") is not None else None,
                str(meta.get("gps_raw") or ""),
                str(meta.get("city") or ""),
                str(meta.get("admin1") or ""),
                str(meta.get("country") or ""),
                str(meta.get("country_code") or ""),
                int(meta["face_count"]) if meta.get("face_count") is not None else None,
                int(meta["orientation"]) if meta.get("orientation") is not None else None,
                float(time.time()),
            ),
        )

    @staticmethod
    def _mtime_ns_from_stat(st: os.stat_result) -> int:
        """Extract nanosecond mtime if available, otherwise fallback to float."""
        if hasattr(st, "st_mtime_ns"):
            return int(st.st_mtime_ns)
        return int(st.st_mtime * 1e9)

    @classmethod
    def _file_signature_from_stat(cls, file_path: str, st: os.stat_result) -> str:
        """
        Stable per-file identity that survives path alias changes.
        """
        name = os.path.basename(file_path)
        raw = f"{name}\0{int(st.st_size)}\0{cls._mtime_ns_from_stat(st)}"
        return hashlib.sha1(raw.encode("utf-8", errors="surrogateescape")).hexdigest()

    def _backfill_file_signatures(self) -> None:
        """
        Populate signature columns for existing databases created before stable identity.
        """
        try:
            rows = self._conn.execute(
                """
                SELECT file_path
                FROM semantic_index
                WHERE file_signature IS NULL OR file_signature = '' OR mtime_ns IS NULL OR file_name IS NULL
                """
            ).fetchall()
            for (fp,) in rows:
                try:
                    canonical = self._canonical_path(str(fp))
                    if not os.path.isfile(canonical):
                        continue
                    st = os.stat(canonical)
                    self._conn.execute(
                        """
                        UPDATE semantic_index
                        SET file_name = ?, file_signature = ?, mtime_ns = ?, file_path = ?
                        WHERE file_path = ?
                        """,
                        (
                            os.path.basename(canonical),
                            self._file_signature_from_stat(canonical, st),
                            self._mtime_ns_from_stat(st),
                            canonical,
                            fp,
                        ),
                    )
                except Exception:
                    continue
            self._conn.commit()
        except Exception:
            pass

    @staticmethod
    def _ratio_to_float(v) -> Optional[float]:
        try:
            # exifread.utils.Ratio supports num/den
            if hasattr(v, "num") and hasattr(v, "den"):
                den = float(v.den) if float(v.den) != 0 else 1.0
                return float(v.num) / den
            return float(v)
        except Exception:
            return None

    def _gps_to_decimal(self, gps_vals: Any, ref: str) -> Optional[float]:
        try:
            # If gps_vals is an IfdTag/IfdTagLite, it has a .values attribute
            vals = getattr(gps_vals, "values", gps_vals)
            
            # If it's a string from some backends, try to parse it
            if isinstance(vals, str):
                # Common formats: "25/1 10/1 0/1" or "25, 10, 0"
                import re
                parts = re.split(r"[\s,]+", vals.strip())
                if len(parts) >= 3:
                    parsed = []
                    for p in parts:
                        if "/" in p:
                            num, den = p.split("/", 1)
                            parsed.append(float(num) / (float(den) if float(den) != 0 else 1.0))
                        else:
                            parsed.append(float(p))
                    vals = parsed

            if not vals or not isinstance(vals, (list, tuple)) or len(vals) < 3:
                return None
                
            d = self._ratio_to_float(vals[0])
            m = self._ratio_to_float(vals[1])
            s = self._ratio_to_float(vals[2])
            
            if d is None or m is None or s is None:
                return None
                
            dec = float(d) + float(m) / 60.0 + float(s) / 3600.0
            
            ref_str = str(ref or "").strip().upper()
            if ref_str in ("S", "W"):
                dec = -dec
            return dec
        except Exception:
            return None

    def _extract_exif_brief(self, file_path: str, include_face: bool = False) -> Dict[str, object]:
        result = {
            "capture_time": "",
            "camera_model": "",
            "lens_model": "",
            "iso": 0,
            "width": 0,
            "height": 0,
            "gps_lat": None,
            "gps_lon": None,
            "gps_raw": "",
            "city": "",
            "admin1": "",
            "country": "",
            "country_code": "",
            "face_count": None,
            "orientation": 1,
        }
        try:
            im = None
            if include_face:
                try:
                    im = _load_index_source_image(file_path, max_size=1280)
                except Exception:
                    pass

            tags = metadata_backend.process_file_from_path(
                file_path, details=False
            )
            result["capture_time"] = self._tag_text(
                tags,
                "EXIF DateTimeOriginal",
                "EXIF DateTimeDigitized",
                "Image DateTime",
                "EXIF DateTime",
            )
            make = self._tag_text(tags, "Image Make")
            model = self._tag_text(tags, "Image Model", "EXIF BodySerialNumber")
            result["camera_model"] = self._clean_join(make, model)
            result["lens_model"] = self._tag_text(
                tags,
                "EXIF LensModel",
                "EXIF LensMake",
                "MakerNote LensType",
                "MakerNote Lens",
                "Image LensModel",
                "Composite LensID",
            )
            result["iso"] = self._tag_int(
                tags,
                "EXIF ISOSpeedRatings",
                "EXIF PhotographicSensitivity",
                "EXIF RecommendedExposureIndex",
                "EXIF ISO",
                "MakerNote ISO",
            )
            result["width"] = self._tag_int(
                tags,
                "EXIF ExifImageWidth",
                "Image ImageWidth",
                "Image Width",
                "EXIF PixelXDimension",
            )
            result["height"] = self._tag_int(
                tags,
                "EXIF ExifImageLength",
                "Image ImageLength",
                "Image Height",
                "Image Length",
                "EXIF PixelYDimension",
            )
            if not result["width"] or not result["height"]:
                w, h = self._pil_dimensions(file_path)
                result["width"] = int(result["width"] or w)
                result["height"] = int(result["height"] or h)
                
            result["orientation"] = self._tag_int(
                tags,
                "EXIF Orientation",
                "Image Orientation"
            ) or 1
            
            # GPS Extraction
            lat = tags.get("GPS GPSLatitude") or tags.get("EXIF GPSLatitude") or tags.get("GPSLatitude")
            lon = tags.get("GPS GPSLongitude") or tags.get("EXIF GPSLongitude") or tags.get("GPSLongitude")
            lat_ref = self._tag_text(tags, "GPS GPSLatitudeRef", "EXIF GPSLatitudeRef", "GPSLatitudeRef")
            lon_ref = self._tag_text(tags, "GPS GPSLongitudeRef", "EXIF GPSLongitudeRef", "GPSLongitudeRef")
            
            result["gps_lat"] = self._gps_to_decimal(lat, lat_ref) if lat else None
            result["gps_lon"] = self._gps_to_decimal(lon, lon_ref) if lon else None
            
            if lat and lon:
                result["gps_raw"] = f"{lat_ref} {lat} | {lon_ref} {lon}"
            
            # Reverse Geocoding
            if result["gps_lat"] is not None and result["gps_lon"] is not None:
                # Skip (0,0) as it's often a placeholder for no-fix
                if abs(result["gps_lat"]) > 0.001 or abs(result["gps_lon"]) > 0.001:
                    geo = self._ensure_reverse_geocoder()
                    if geo:
                        try:
                            # Use a small timeout or limit to avoid blocking too long if multiple calls
                            recs = geo.search(
                                [(float(result["gps_lat"]), float(result["gps_lon"]))],
                                mode=1,
                            )
                            if recs:
                                rec = recs[0] or {}
                                result["city"] = str(rec.get("name", "") or "")
                                result["admin1"] = str(rec.get("admin1", "") or "")
                                cc = str(rec.get("cc", "") or "").upper()
                                result["country_code"] = cc
                                result["country"] = self._country_name_from_code(cc)
                        except Exception:
                            pass
            if include_face:
                result["face_count"] = self._detect_face_count(file_path, preloaded_im=im)
            
            # Automatically backfill and populate the main gallery's PersistentEXIFCache/ImageCache!
            try:
                orientation = 1
                for key in ("Exif.Image.Orientation", "Exif.Photo.Orientation", "Image Orientation", "EXIF Orientation"):
                    val = tags.get(key)
                    if val is not None:
                        try:
                            s = str(val).strip()
                            first = s.split()[0]
                            o = int(first)
                            if 1 <= o <= 8:
                                orientation = o
                                break
                        except Exception:
                            pass

                from image_cache import get_image_cache
                cache = get_image_cache()
                cache_dict = {
                    "orientation": int(orientation),
                    "camera_make": str(tags.get("Image Make") or ""),
                    "camera_model": str(result["camera_model"]),
                    "capture_time": str(result["capture_time"]),
                    "original_width": int(result["width"] or 0),
                    "original_height": int(result["height"] or 0),
                    "exif_data": {
                        "original_width": int(result["width"] or 0),
                        "original_height": int(result["height"] or 0),
                        "orientation": int(orientation),
                        "capture_time": str(result["capture_time"]),
                        "camera_make": str(tags.get("Image Make") or ""),
                        "camera_model": str(result["camera_model"]),
                        "lens_model": str(result["lens_model"]),
                        "iso": int(result["iso"] or 0),
                    }
                }
                cache.put_exif(file_path, cache_dict)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"[INDEX] Could not populate ImageCache for {file_path}: {e}")
        except Exception:
            pass
        return result

    @staticmethod
    def _detect_face_count(file_path: str, preloaded_im: Optional[Image.Image] = None) -> int:
        if sys.platform == "darwin":
            try:
                import Foundation
                import Vision
            except Exception:
                pass

        def _run_vision(path: str) -> Optional[int]:
            try:
                url = Foundation.NSURL.fileURLWithPath_(path)
                request = Vision.VNDetectFaceRectanglesRequest.alloc().init()
                handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, {})
                ok, err = handler.performRequests_error_([request], None)
                if not ok or err is not None:
                    return None
                return len(request.results() or [])
            except Exception:
                return None

        tmp_path = ""
        try:
            if preloaded_im is not None:
                im = preloaded_im
            else:
                im = _load_index_source_image(file_path, max_size=1280)
            
            if sys.platform != "darwin":
                # 1. Try OpenCV YuNet (Ultra-Lightweight, extremely fast, highly accurate)
                try:
                    import cv2
                    import numpy as np
                    import os
                    import urllib.request
                    import threading
                    
                    global _THREAD_LOCAL_DETECTORS
                    if '_THREAD_LOCAL_DETECTORS' not in globals():
                        _THREAD_LOCAL_DETECTORS = threading.local()
                    
                    # Ensure the model file is present
                    cache_dir = os.path.expanduser("~/.rawviewer_cache/models")
                    os.makedirs(cache_dir, exist_ok=True)
                    model_path = os.path.join(cache_dir, "face_detection_yunet_2023mar.onnx")
                    
                    if not os.path.exists(model_path):
                        import logging
                        logging.getLogger(__name__).info("[VISION] Downloading YuNet ONNX face detection model (353 KB)...")
                        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
                        urllib.request.urlretrieve(url, model_path)
                        logging.getLogger(__name__).info("[VISION] YuNet ONNX model downloaded successfully.")
                    
                    # YuNet requires BGR input
                    img_bgr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
                    h, w = img_bgr.shape[:2]
                    
                    if not hasattr(_THREAD_LOCAL_DETECTORS, 'yunet'):
                        import logging
                        logging.getLogger(__name__).info("[VISION] Initializing OpenCV YuNet thread-local instance...")
                        _THREAD_LOCAL_DETECTORS.yunet = cv2.FaceDetectorYN.create(
                            model=model_path,
                            config="",
                            input_size=(w, h),
                            score_threshold=0.6,
                            nms_threshold=0.3,
                            top_k=5000,
                            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                            target_id=cv2.dnn.DNN_TARGET_CPU
                        )
                        logging.getLogger(__name__).info("[VISION] OpenCV YuNet thread-local instance initialized successfully.")
                    
                    detector = _THREAD_LOCAL_DETECTORS.yunet
                    # Crucial: input size must match the actual image size for YuNet
                    detector.setInputSize((w, h))
                    retval, faces = detector.detect(img_bgr)
                    
                    return len(faces) if faces is not None else 0
                except Exception as yn_err:
                    import logging
                    logging.getLogger(__name__).debug(f"[VISION] OpenCV YuNet failed on {file_path}, trying OpenCV DNN: {yn_err}")

                # 2. Windows / Linux fallback using OpenCV DNN Face Detector
                try:
                    import cv2
                    import numpy as np
                    import os
                    import urllib.request
                    import threading
                    
                    global _FACE_DETECTOR_NET
                    global _FACE_DETECTOR_LOCK
                    if '_FACE_DETECTOR_NET' not in globals():
                        _FACE_DETECTOR_NET = None
                        _FACE_DETECTOR_LOCK = threading.Lock()

                    if _FACE_DETECTOR_NET is None:
                        with _FACE_DETECTOR_LOCK:
                            if _FACE_DETECTOR_NET is None:
                                models_dir = os.path.expanduser("~/.rawviewer_cache/models")
                                os.makedirs(models_dir, exist_ok=True)
                                
                                prototxt_path = os.path.join(models_dir, "deploy.prototxt")
                                caffemodel_path = os.path.join(models_dir, "res10_300x300_ssd_iter_140000.caffemodel")
                                
                                if not os.path.exists(prototxt_path):
                                    import logging
                                    logging.getLogger(__name__).info("[VISION] Downloading DNN face detector prototxt...")
                                    urllib.request.urlretrieve(
                                        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                                        prototxt_path
                                    )
                                if not os.path.exists(caffemodel_path):
                                    import logging
                                    logging.getLogger(__name__).info("[VISION] Downloading DNN face detector weights...")
                                    urllib.request.urlretrieve(
                                        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                                        caffemodel_path
                                    )
                                _FACE_DETECTOR_NET = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
                    
                    # Convert PIL image to BGR for OpenCV DNN
                    img_bgr = np.array(im.convert('RGB'))[:, :, ::-1].copy()
                    
                    # Prepare input blob and perform forward pass
                    blob = cv2.dnn.blobFromImage(cv2.resize(img_bgr, (600, 600)), 1.0, (600, 600), (104.0, 177.0, 123.0))
                    
                    with _FACE_DETECTOR_LOCK:
                        _FACE_DETECTOR_NET.setInput(blob)
                        detections = _FACE_DETECTOR_NET.forward()
                    
                    face_count = 0
                    for i in range(detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        if confidence > 0.6: # 60% confidence threshold
                            face_count += 1
                            
                    return face_count
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"[VISION] OpenCV DNN face detection fallback error on {file_path}: {e}")
                    return 0

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
            im.save(tmp_path, "JPEG", quality=90)
            fallback = _run_vision(tmp_path)
            return int(fallback or 0)
        except Exception:
            return 0
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def _lookup_index_rows(self, file_path: str, st: Optional[os.stat_result] = None) -> List[sqlite3.Row]:
        canonical = self._canonical_path(file_path)
        if st is None:
            st = os.stat(canonical)
        signature = self._file_signature_from_stat(canonical, st)
        aliases = self._path_aliases(canonical)
        placeholders = ",".join(["?"] * len(aliases))
        self._conn.row_factory = sqlite3.Row
        return self._conn.execute(
            f"""
            SELECT file_path, file_name, file_signature, file_size, file_mtime, mtime_ns, model_name, face_count
            FROM semantic_index
            WHERE (file_signature = ? AND model_name = ?)
               OR file_path IN ({placeholders})
            """,
            [signature, self.model_name, *aliases],
        ).fetchall()

    def _row_matches_file(self, row: sqlite3.Row, st: os.stat_result) -> bool:
        if str(row["model_name"]) != self.model_name:
            return False
        if int(row["file_size"]) != int(st.st_size):
            return False
        row_mtime_ns = row["mtime_ns"] if "mtime_ns" in row.keys() else None
        if row_mtime_ns is not None:
            try:
                return int(row_mtime_ns) == self._mtime_ns_from_stat(st)
            except Exception:
                pass
        return self._mtime_matches(float(row["file_mtime"]), st)

    def _row_semantic_ready(self, row: sqlite3.Row) -> bool:
        try:
            return int(row["semantic_ready"] or 0) == 1
        except Exception:
            # Backward compatibility for rows loaded before migration.
            return True

    def _needs_reindex(self, file_path: str, st: os.stat_result) -> bool:
        rows = self._lookup_index_rows(file_path, st)
        if not rows:
            return True
        for row in rows:
            if self._row_matches_file(row, st) and self._row_semantic_ready(row):
                return False
        return True

    def _encode_image(self, file_path: str) -> np.ndarray:
        if self.model_name.startswith("mobileclip-"):
            backend = self._mobileclip_backend or resolve_mobileclip_backend()
            err = backend.availability_error()
            if err:
                raise RuntimeError(f"MobileCLIP backend unavailable: {err}")
            return backend.encode_image(file_path)
        model = self._ensure_model()
        try:
            im = _load_index_source_image(file_path, max_size=1024)
            emb = model.encode(im, normalize_embeddings=True)
            return np.asarray(emb, dtype=np.float32)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot decode image for semantic index: {os.path.basename(file_path)}"
            ) from exc

    def _encode_text(self, text: str) -> np.ndarray:
        if self.model_name.startswith("mobileclip-"):
            backend = self._mobileclip_backend or resolve_mobileclip_backend()
            err = backend.availability_error()
            if err:
                raise RuntimeError(f"MobileCLIP backend unavailable: {err}")
            return backend.encode_text(text)
        model = self._ensure_model()
        return np.asarray(model.encode(text, normalize_embeddings=True), dtype=np.float32)

    def build_index(self, file_paths: Sequence[str], progress_callback: ProgressCallback = None) -> Dict[str, int]:
        import logging
        import sys
        import sqlite3
        logger = logging.getLogger(__name__)
        t_start = time.time()
        logger.info(f"[INDEX] Starting indexing of {len(file_paths)} file paths.")
        if sys.platform != "darwin":
            logger.info("[VISION] Using OpenCV offline face scanner for Windows.")
            
        # Identify raw files that have a companion non-raw image to avoid duplicate indexing and slash indexing times
        filtered_paths = []
        skipped_companions = 0
        
        # Build a set of non-raw companion base keys in each directory
        def get_companion_key(fp: str) -> str:
            base = os.path.basename(fp)
            parts = base.split('.')
            if len(parts) > 1:
                for idx, part in enumerate(parts):
                    if part.startswith("RAW-"):
                        return ".".join(parts[:idx]).lower()
                return ".".join(parts[:-1]).lower()
            return base.lower()

        # Find all non-raw files in the list
        non_raw_keys = set()
        for fp in file_paths:
            if not fp: continue
            ext = os.path.splitext(fp)[1].lower().lstrip(".")
            if ext and ext not in RAW_FILE_EXTENSIONS:
                dirname = os.path.dirname(fp)
                non_raw_keys.add((dirname, get_companion_key(fp)))

        for fp in file_paths:
            if not fp: continue
            ext = os.path.splitext(fp)[1].lower().lstrip(".")
            if ext in RAW_FILE_EXTENSIONS:
                dirname = os.path.dirname(fp)
                comp_key = get_companion_key(fp)
                if (dirname, comp_key) in non_raw_keys:
                    skipped_companions += 1
                    logger.info(f"[INDEX] Skipping RAW companion file to avoid duplicate results: {os.path.basename(fp)}")
                    continue
            filtered_paths.append(fp)
            
        logger.info(f"[INDEX] Filtered out {skipped_companions} RAW companion files. Actual files to evaluate: {len(filtered_paths)}")
            
        total = len(file_paths)
        indexed = 0
        skipped = skipped_companions
        failed = 0
        pending_for_semantic: List[tuple[str, os.stat_result]] = []
        
        # Create a thread-local, dedicated connection for this background worker thread!
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=60000")
        
        try:
            # 1.1 Pre-fetch existing metadata in bulk to avoid thousands of small SQL queries
            existing_meta = {} # {canonical_path: (mtime, size, semantic_ready)}
            canonical_map = {self._canonical_path(fp): fp for fp in filtered_paths if fp}
            unique_canonical = list(canonical_map.keys())
            
            logger.info(f"[INDEX] Pre-fetching existing metadata database entries for {len(unique_canonical)} canonical paths...")
            t0 = time.time()
            chunk_size = 900
            for i in range(0, len(unique_canonical), chunk_size):
                chunk = unique_canonical[i:i+chunk_size]
                qs = ",".join(["?"] * len(chunk))
                cursor = conn.execute(
                    f"SELECT file_path, file_mtime, file_size, semantic_ready, gps_lat, city FROM semantic_index WHERE file_path IN ({qs})",
                    chunk
                )
                for row in cursor.fetchall():
                    existing_meta[row[0]] = {
                        "mtime": row[1],
                        "size": row[2],
                        "semantic_ready": row[3],
                        "gps_lat": row[4],
                        "city": row[5]
                    }
            logger.info(f"[INDEX] Pre-fetch completed in {time.time() - t0:.4f}s. Found {len(existing_meta)} matches in database.")

            def needs_reindex_local(cp, st):
                if cp not in existing_meta:
                    return True
                row = existing_meta[cp]
                if not self._mtime_matches(row["mtime"], st):
                    return True
                if int(row["size"]) != int(st.st_size):
                    return True
                
                return False

            # 1.2 Identify files that actually need metadata extraction
            to_extract = []
            for fp in filtered_paths:
                if not fp: continue
                canonical_fp = self._canonical_path(fp)
                try:
                    st = os.stat(canonical_fp)
                    if needs_reindex_local(canonical_fp, st):
                        to_extract.append((canonical_fp, st))
                    else:
                        row = existing_meta[canonical_fp]
                        if not row["semantic_ready"]:
                            pending_for_semantic.append((canonical_fp, st))
                        else:
                            skipped += 1
                except OSError:
                    failed += 1
                    continue

            logger.info(f"[INDEX] Needs metadata extraction: {len(to_extract)} files. Already indexed & skipped: {skipped} files.")

            # 1.3 Parallel extraction of metadata
            total_extract = len(to_extract)
            batch_writes = 0
            commit_every = 40
            
            if total_extract > 0:
                logger.info(f"[INDEX] Starting parallel metadata extraction for {total_extract} files using ThreadPoolExecutor...")
                t_meta_start = time.time()
                max_workers = min(3, os.cpu_count() or 2)
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    def extract_task(item):
                        cp, st = item
                        try:
                            t_single_start = time.time()
                            meta = self._extract_exif_brief(cp, include_face=False)
                            t_single_dur = time.time() - t_single_start
                            if t_single_dur > 0.1:
                                logger.warning(f"[INDEX] Slow metadata extraction (>100ms) for {os.path.basename(cp)}: {t_single_dur:.4f}s")
                            return cp, st, meta
                        except Exception as e:
                            logger.error(f"[INDEX] Failed to extract EXIF for {os.path.basename(cp)}: {e}")
                            return cp, st, None

                    futures = [executor.submit(extract_task, item) for item in to_extract]
                    
                    for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                        cp, st, meta = future.result()
                        if meta:
                            try:
                                self._upsert_metadata(cp, st, meta, conn=conn)
                                pending_for_semantic.append((cp, st))
                                batch_writes += 1
                                # Do not increment 'indexed' here; it will be incremented in Phase 2
                                # when the semantic embedding is actually ready.
                                
                                if batch_writes >= commit_every:
                                    conn.commit()
                                    batch_writes = 0
                            except Exception as e:
                                logger.error(f"[INDEX] Database upsert failed for {os.path.basename(cp)}: {e}")
                                failed += 1
                        else:
                            failed += 1
                        
                        if progress_callback and (i <= 2 or i >= total_extract or i % 10 == 0):
                            progress_callback(i, total_extract, "Scanning metadata...")

                if batch_writes > 0:
                    conn.commit()
                    batch_writes = 0
                logger.info(f"[INDEX] Completed metadata extraction for {total_extract} files in {time.time() - t_meta_start:.4f}s.")

            # Phase 1.5: Background Face Scanning (silent, progressive backfill)
            face_pending = []
            for i in range(0, len(unique_canonical), chunk_size):
                chunk = unique_canonical[i:i+chunk_size]
                qs = ",".join(["?"] * len(chunk))
                cursor = conn.execute(
                    f"SELECT file_path FROM semantic_index WHERE file_path IN ({qs}) AND face_count IS NULL",
                    chunk
                )
                for row in cursor.fetchall():
                    face_pending.append(row[0])

            total_face = len(face_pending)
            if total_face > 0:
                logger.info(f"[INDEX] Starting progressive background face scanning for {total_face} files...")
                t_face_start = time.time()
                for idx, cp in enumerate(face_pending, start=1):
                    time.sleep(0.08)  # 80ms breathing room to prevent OS disk/CPU freeze
                    if progress_callback and (idx <= 2 or idx >= total_face or idx % 10 == 0):
                        progress_callback(idx, total_face, "Scanning faces...")
                    try:
                        face_count = self._detect_face_count(cp)
                        self._store_face_count(cp, int(face_count or 0), conn=conn)
                        batch_writes += 1
                        if batch_writes >= commit_every:
                            conn.commit()
                            batch_writes = 0
                    except Exception as e:
                        logger.error(f"[INDEX] Face scanning failed for {os.path.basename(cp)}: {e}")
                if batch_writes > 0:
                    conn.commit()
                    batch_writes = 0
                logger.info(f"[INDEX] Completed background face scanning in {time.time() - t_face_start:.4f}s.")

            # Phase 2: semantic embeddings (slow). Update existing rows in-place.
            total_sem = len(pending_for_semantic)
            if total_sem > 0:
                logger.info(f"[INDEX] Starting AI features neural pass (MobileCLIP) for {total_sem} files...")
                t_sem_start = time.time()
                for i, (canonical_fp, st) in enumerate(pending_for_semantic, start=1):
                    time.sleep(0.15)  # 150ms breathing room for MobileCLIP heavy neural models
                    if progress_callback:
                        # Keep callback signature unchanged while reporting semantic pass progress.
                        if i <= 2 or i >= total_sem or (i % 12 == 0):
                            progress_callback(i, total_sem, "Processing AI features...")
                    try:
                        t_single_neural = time.time()
                        vec = self._encode_image(canonical_fp)
                        t_neural_dur = time.time() - t_single_neural
                        if t_neural_dur > 0.5:
                            logger.info(f"[INDEX] MobileCLIP encoding for {os.path.basename(canonical_fp)} took {t_neural_dur:.4f}s")
                        conn.execute(
                            """
                            UPDATE semantic_index
                            SET dim = ?, embedding = ?, semantic_ready = 1, updated_at = ?
                            WHERE file_path = ? AND model_name = ? AND file_size = ? AND mtime_ns = ?
                            """,
                            (
                                int(vec.size),
                                self._to_blob(vec),
                                float(time.time()),
                                canonical_fp,
                                self.model_name,
                                int(st.st_size),
                                self._mtime_ns_from_stat(st),
                            ),
                        )
                        indexed += 1
                        batch_writes += 1
                        if batch_writes >= commit_every:
                            conn.commit()
                            batch_writes = 0
                    except Exception as e:
                        logger.error(f"[INDEX] AI encoding failed for {os.path.basename(canonical_fp)}: {e}")
                        failed += 1
                if batch_writes:
                    conn.commit()
                logger.info(f"[INDEX] Completed AI neural pass in {time.time() - t_sem_start:.4f}s.")
        finally:
            conn.close()
            
        duration = time.time() - t_start
        logger.info(f"[INDEX] Finished indexing process in {duration:.4f}s. Results -> indexed: {indexed}, skipped: {skipped}, failed: {failed}, total: {total}")
        return {"indexed": indexed, "skipped": skipped, "failed": failed, "total": total}

    def get_index_coverage(self, file_paths: Sequence[str]) -> Dict[str, int]:
        """
        Return index coverage for the given file set.
        Metadata-lazy version: only checks database presence, does not touch disk.
        """
        if not file_paths:
            return {"total": 0, "indexed": 0, "missing": 0, "ready": 1}

        total = len(file_paths)
        canonical_paths = [self._canonical_path(p) for p in file_paths if p]
        
        # Bulk lookup in database
        placeholders = ",".join(["?"] * len(canonical_paths))
        # SQLite has a limit on parameters (usually 999), so we batch if needed.
        batch_size = 900
        indexed_count = 0
        
        try:
            for i in range(0, len(canonical_paths), batch_size):
                batch = canonical_paths[i : i + batch_size]
                qs = ",".join(["?"] * len(batch))
                cursor = self._conn.execute(
                    f"SELECT COUNT(*) FROM semantic_index WHERE file_path IN ({qs}) AND semantic_ready = 1 AND model_name = ?",
                    [*batch, self.model_name]
                )
                indexed_count += cursor.fetchone()[0]
        except Exception:
            # Fallback if table doesn't exist or other error
            indexed_count = 0

        missing = max(0, total - indexed_count)
        return {
            "total": total,
            "indexed": indexed_count,
            "missing": missing,
            "ready": 1 if missing == 0 else 0,
        }

    def get_pending_paths(self, file_paths: Sequence[str]) -> List[str]:
        """
        Return only files that are missing or stale and need reindexing.
        Uses bulk database check to identify missing files quickly.
        """
        if not file_paths:
            return []

        canonical_map = {self._canonical_path(p): p for p in file_paths if p}
        canonical_paths = list(canonical_map.keys())
        
        # 1. Identify which paths are already indexed and UP-TO-DATE in metadata
        # We check mtime/size only for files we find in the DB.
        # Files NOT in the DB are automatically pending.
        indexed_up_to_date = set()
        batch_size = 900
        for i in range(0, len(canonical_paths), batch_size):
            batch = canonical_paths[i : i + batch_size]
            qs = ",".join(["?"] * len(batch))
            cursor = self._conn.execute(
                f"SELECT file_path FROM semantic_index WHERE file_path IN ({qs}) AND semantic_ready = 1 AND model_name = ?",
                [*batch, self.model_name]
            )
            for (fp,) in cursor.fetchall():
                indexed_up_to_date.add(fp)

        pending = []
        for cp in canonical_paths:
            if cp not in indexed_up_to_date:
                # If not indexed, it's definitely pending.
                # We skip os.stat here for speed; the indexer will check it later.
                pending.append(cp)
        
        return pending

    def _fetch_rows_for_paths(self, paths: Sequence[str]) -> List[sqlite3.Row]:
        """Bulk fetch rows for a list of paths using optimized batch queries."""
        if not paths:
            return []
        
        # 1. Deduplicate and canonicalize
        canonical_to_original = {}
        unique_canonical = []
        for p in paths:
            if not p:
                continue
            cp = self._canonical_path(p)
            if cp not in canonical_to_original:
                canonical_to_original[cp] = p
                unique_canonical.append(cp)
        
        if not unique_canonical:
            return []

        # 2. Bulk fetch by file_path
        self._conn.row_factory = sqlite3.Row
        
        found_map = {}
        # SQLite parameter limit is usually 999; use 500 to be safe
        chunk_size = 500
        for i in range(0, len(unique_canonical), chunk_size):
            chunk = unique_canonical[i:i+chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            # IMPORTANT: Do NOT filter by model_name here.
            # Metadata fields (face_count, gps_lat/lon, camera_model, city, iso,
            # capture_time, etc.) are model-independent and must be accessible
            # regardless of which AI/embedding model is currently active.
            # Only `embedding` and `semantic_ready` are model-specific.
            # If multiple rows exist for the same file_path (different model_names),
            # prefer the current model's row so that semantic_ready/embedding are
            # correct; otherwise fall back to any row for the path.
            query = f"""
                SELECT file_path, file_name, file_signature, dim, embedding, capture_time, 
                       camera_model, lens_model, iso, gps_lat, gps_lon, width, height, 
                       city, admin1, country, country_code, face_count, semantic_ready,
                       file_size, file_mtime, mtime_ns, model_name
                FROM semantic_index
                WHERE file_path IN ({placeholders})
                ORDER BY
                    CASE WHEN model_name = ? THEN 0 ELSE 1 END,
                    rowid DESC
            """
            rows = self._conn.execute(query, [*chunk, self.model_name]).fetchall()
            for r in rows:
                fp = r['file_path']
                if fp not in found_map:
                    # First match wins: preferred model first (ORDER BY above), then any row
                    found_map[fp] = r
        
        # 3. Assemble results in original order, injecting mock rows for missing files
        # This ensures that newly added files appear in search results (as unindexed) 
        # instead of being filtered out.
        results = []
        for cp in unique_canonical:
            if cp in found_map:
                results.append(found_map[cp])
            else:
                # Create a mock row for files not yet indexed
                full_path = canonical_to_original[cp]
                file_name = os.path.basename(full_path)
                
                # Use a dictionary as a proxy for sqlite3.Row
                # RAWviewer's UI components are hardened to handle missing metadata
                mock_row = {
                    'file_path': cp,
                    'file_name': file_name,
                    'file_signature': "",
                    'dim': 0,
                    'embedding': None,
                    'capture_time': "",
                    'camera_model': "",
                    'lens_model': "",
                    'iso': 0,
                    'gps_lat': None,
                    'gps_lon': None,
                    'width': 0,
                    'height': 0,
                    'city': "",
                    'admin1': "",
                    'country': "",
                    'country_code': "",
                    'face_count': 0,
                    'semantic_ready': 0,
                    'file_size': 0,
                    'file_mtime': 0,
                    'mtime_ns': 0,
                    'model_name': self.model_name
                }
                results.append(mock_row)
            
        return results

    def get_layout_metadata_for_paths(self, paths: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fast synchronous fetch of layout metadata (width, height, orientation, capture_time) 
        for pre-seeding the gallery layout to avoid flashing on first render.
        """
        if not paths:
            return {}
            
        canonical_to_original = {}
        unique_canonical = []
        for p in paths:
            if not p:
                continue
            cp = self._canonical_path(p)
            if cp not in canonical_to_original:
                canonical_to_original[cp] = p
                unique_canonical.append(cp)
                
        if not unique_canonical:
            return {}
            
        # Bulk fetch by file_path
        self._conn.row_factory = sqlite3.Row
        
        result_map = {}
        chunk_size = 500
        for i in range(0, len(unique_canonical), chunk_size):
            chunk = unique_canonical[i:i+chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            query = f"""
                SELECT file_path, width, height, orientation, capture_time
                FROM semantic_index
                WHERE file_path IN ({placeholders})
                ORDER BY
                    CASE WHEN model_name = ? THEN 0 ELSE 1 END,
                    rowid DESC
            """
            rows = self._conn.execute(query, [*chunk, self.model_name]).fetchall()
            for r in rows:
                fp = r['file_path']
                if fp not in result_map:
                    orig_path = canonical_to_original[fp]
                    result_map[orig_path] = {
                        "width": r["width"] or 0,
                        "height": r["height"] or 0,
                        "orientation": r["orientation"] or 1,
                        "capture_time": r["capture_time"] or "",
                    }
                    
        return result_map

    @staticmethod
    def _parse_capture_year(capture_time: str) -> int:
        # EXIF style: "YYYY:MM:DD HH:MM:SS"
        try:
            return int((capture_time or "")[:4])
        except Exception:
            return 0

    @staticmethod
    def _parse_capture_month(capture_time: str) -> int:
        try:
            return int((capture_time or "")[5:7])
        except Exception:
            return 0

    @staticmethod
    def _row_value(row, key: str, default=""):
        try:
            return row[key]
        except Exception:
            try:
                return row.get(key, default)
            except Exception:
                return default

    @staticmethod
    def _normalize_date_value(value: str) -> str:
        return (value or "").strip().replace("-", ":").replace("/", ":")

    @staticmethod
    def _row_file_extension(row) -> str:
        fp = str(SemanticImageIndex._row_value(row, "file_path") or "")
        fn = str(SemanticImageIndex._row_value(row, "file_name") or "")
        # Prefer path basename (primary key truth); fallback to indexed file_name.
        for base in (os.path.basename(fp), fn):
            ext = os.path.splitext(base)[1].lower().lstrip(".")
            if ext:
                return ext
        return ""

    def _needle_is_solitary_format_token(self, needle: str) -> bool:
        n = (needle or "").strip().lower().replace("_", "")
        if len(n) < 2 or len(n) > 12:
            return False
        if not re.fullmatch(r"[a-z0-9]+", n):
            return False
        return n in self._FORMAT_HINT_FILENAME_TOKENS
    @classmethod
    def _format_specs_to_accept_set(cls, spec: str) -> frozenset:
        """Map one user-facing format keyword to a set of lowercase extensions."""
        s = (spec or "").strip().lower().lstrip(".")
        if not s:
            return frozenset()
        if s in ("jpeg", "jpg", "jpe"):
            return frozenset({"jpg", "jpeg", "jpe"})
        if s in ("tif", "tiff"):
            return frozenset({"tif", "tiff"})
        if s in ("heic", "heif"):
            return frozenset({"heic", "heif"})
        if s == "raw":
            return cls._RAW_FILE_EXTENSIONS
        return frozenset({s})

    @classmethod
    def _row_matches_format_specs(cls, row, specs: Sequence[str]) -> bool:
        fe = cls._row_file_extension(row)
        for spec in specs:
            accepted = cls._format_specs_to_accept_set(spec)
            if fe in accepted:
                return True
        return False

    @classmethod
    def _capture_time_matches_date(cls, capture_time: str, value: str) -> bool:
        normalized = cls._normalize_date_value(value)
        if not normalized:
            return False
        capture = str(capture_time or "")
        return capture.startswith(normalized)

    @staticmethod
    def _contains_loose(haystack: str, needle: str) -> bool:
        h = str(haystack or "").lower()
        n = str(needle or "").strip().lower()
        if not n:
            return False
        variants = {n, n.replace("_", " "), n.replace(" ", "_")}
        return any(v and v in h for v in variants)

    def _apply_filters(
        self, rows: Sequence[sqlite3.Row], query_text: str
    ) -> tuple[List[sqlite3.Row], str]:
        """
        Parse a mixed query and filter rows.

        Supported filter tokens:
        - camera:<text>
        - lens:<text>
        - city:<text>
        - admin:<text>
        - country:<text>          (country name or code)
        - date:<prefix>          (e.g. date:2026:05, date:2026-05, date:2026-05-01)
        - filename:<text> / file:<text> / name:<text>
        - format:<ext> / type:<ext> / ext:<ext>
          Comma-separated OR: format:jpg,png. Leading dot optional. Synonyms:
          jpg/jpeg/jpe, tif/tiff, heif/heic, raw=<common RAW extensions>.
        - iso<800 / iso<=800 / iso=400 / iso>=200 / iso>100
        - year=2026 / year>=2024 / year<2020
        - month=5 / month>=6 / month<3
        - width>=3000 / height<2000
        - has:gps / no:gps
        - has:face / no:face — also shorthand face, faces, people, person, human(s),
          has:people (uses indexed Vision face counts, not CLIP similarity)
        """
        raw = (query_text or "").strip()
        if not raw:
            return list(rows), ""

        # Normalize numeric filters so both styles work:
        # - iso<800
        # - iso < 800
        raw = re.sub(
            r"\b(iso|year|month|width|height)\s*(<=|>=|=|<|>)\s*(\d+)\b",
            lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}",
            raw,
            flags=re.I,
        )
        raw = re.sub(
            r"\b(iso|year|month|width|height)\s*:\s*(\d+)\b",
            lambda m: f"{m.group(1)}={m.group(2)}",
            raw,
            flags=re.I,
        )
        raw = self._normalize_loose_metadata_query(raw)

        parts = [p for p in raw.split() if p.strip()]
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[SEARCH] Raw query: '{query_text}' -> Normalized: '{raw}'")
        
        semantic_terms: List[str] = []
        filtered = list(rows)

        num_pat = re.compile(r"^(iso|year|month|width|height)\s*(<=|>=|=|<|>)\s*(\d+)$", re.I)
        date_like_pat = re.compile(r"^\d{4}(?:[:-]\d{1,2}){0,2}$")

        for token in parts:
            t = token.strip()
            low = t.lower()
            matched = False

            if low.startswith("camera:"):
                matched = True
                needle = t.split(":", 1)[1].strip().lower()
                if needle:
                    filtered = [r for r in filtered if self._contains_loose(str(r["camera_model"] or ""), needle)]
                continue

            if low.startswith("lens:"):
                matched = True
                needle = t.split(":", 1)[1].strip().lower()
                if needle:
                    filtered = [r for r in filtered if self._contains_loose(str(r["lens_model"] or ""), needle)]
                continue

            if low.startswith("date:"):
                matched = True
                pref = t.split(":", 1)[1].strip()
                if pref:
                    filtered = [
                        r
                        for r in filtered
                        if self._capture_time_matches_date(str(self._row_value(r, "capture_time") or ""), pref)
                    ]
                continue

            if low.startswith(("filename:", "file:", "name:")):
                matched = True
                needle = t.split(":", 1)[1].strip().lower()
                if needle:
                    if self._needle_is_solitary_format_token(needle):
                        specs = [needle]
                        filtered = [r for r in filtered if self._row_matches_format_specs(r, specs)]
                    else:
                        filtered = [
                            r
                            for r in filtered
                            if self._contains_loose(
                                str(self._row_value(r, "file_name") or os.path.basename(str(self._row_value(r, "file_path") or ""))),
                                needle,
                            )
                        ]
                continue

            if low.startswith(("format:", "type:", "ext:")):
                matched = True
                rest = t.split(":", 1)[1].strip()
                if rest:
                    specs = [s.strip().lower().lstrip(".") for s in re.split(r"[,;/|]", rest) if s.strip()]
                    if specs:
                        filtered = [r for r in filtered if self._row_matches_format_specs(r, specs)]
                continue

            if low.startswith("city:"):
                matched = True
                needle = t.split(":", 1)[1].strip().lower()
                if needle:
                    filtered = [r for r in filtered if self._contains_loose(str(r["city"] or ""), needle)]
                continue

            if low.startswith("admin:"):
                matched = True
                needle = t.split(":", 1)[1].strip().lower()
                if needle:
                    filtered = [r for r in filtered if self._contains_loose(str(r["admin1"] or ""), needle)]
                continue

            if low.startswith("country:"):
                matched = True
                needle = t.split(":", 1)[1].strip().lower()
                if needle:
                    filtered = [
                        r
                        for r in filtered
                        if (
                            self._contains_loose(str(r["country"] or ""), needle)
                            or self._contains_loose(str(r["country_code"] or ""), needle)
                        )
                    ]
                continue

            if low == "has:gps":
                matched = True
                filtered = [
                    r
                    for r in filtered
                    if r["gps_lat"] is not None and r["gps_lon"] is not None
                ]
                continue

            if low == "no:gps":
                matched = True
                filtered = [
                    r
                    for r in filtered
                    if r["gps_lat"] is None or r["gps_lon"] is None
                ]
                continue

            if low in self._FACE_COUNT_POSITIVE_TOKENS:
                matched = True
                filtered = [r for r in filtered if int(r["face_count"] or 0) > 0]
                continue

            if low in self._FACE_COUNT_NEGATIVE_TOKENS:
                matched = True
                filtered = [r for r in filtered if int(r["face_count"] or 0) <= 0]
                continue

            if low == "has:gps":
                matched = True
                filtered = [r for r in filtered if r["gps_lat"] is not None]
                continue

            if low == "no:gps":
                matched = True
                filtered = [r for r in filtered if r["gps_lat"] is None]
                continue

            if low.startswith("gps:"):
                matched = True
                try:
                    parts = low[4:].split(",")
                    if len(parts) == 2:
                        target_lat = float(parts[0])
                        target_lon = float(parts[1])
                        # Filter by images within ~50km radius (approx 0.5 degrees)
                        filtered = [
                            r for r in filtered 
                            if r["gps_lat"] is not None and 
                            abs(float(r["gps_lat"]) - target_lat) < 0.5 and 
                            abs(float(r["gps_lon"]) - target_lon) < 0.5
                        ]
                except Exception:
                    pass
                continue

            m = num_pat.match(low)
            if m:
                matched = True
                key, op, val_str = m.group(1), m.group(2), m.group(3)
                val = int(val_str)

                def _ok(x: int) -> bool:
                    if op == "<":
                        return x < val
                    if op == "<=":
                        return x <= val
                    if op == "=":
                        return x == val
                    if op == ">=":
                        return x >= val
                    return x > val

                if key == "iso":
                    filtered = [r for r in filtered if _ok(int(self._row_value(r, "iso", 0) or 0))]
                elif key == "year":
                    filtered = [r for r in filtered if _ok(self._parse_capture_year(str(self._row_value(r, "capture_time") or "")))]
                elif key == "month":
                    filtered = [r for r in filtered if _ok(self._parse_capture_month(str(self._row_value(r, "capture_time") or "")))]
                elif key == "width":
                    filtered = [r for r in filtered if _ok(int(self._row_value(r, "width", 0) or 0))]
                elif key == "height":
                    filtered = [r for r in filtered if _ok(int(self._row_value(r, "height", 0) or 0))]
                continue

            if date_like_pat.match(low):
                matched = True
                filtered = [
                    r
                    for r in filtered
                    if self._capture_time_matches_date(str(self._row_value(r, "capture_time") or ""), low)
                ]
                continue

            if not matched:
                bare = low.strip().lstrip(".")
                if (
                    re.fullmatch(r"[a-z0-9]{2,15}", bare or "")
                    and bare in self._FORMAT_HINT_FILENAME_TOKENS
                ):
                    matched = True
                    filtered = [r for r in filtered if self._row_matches_format_specs(r, [bare])]
                    continue
                semantic_terms.append(t)

        filtered, semantic_terms = self._auto_match_metadata_keywords(filtered, semantic_terms)
        metadata_stopwords = {
            "a",
            "an",
            "the",
            "of",
            "with",
            "for",
            "photo",
            "photos",
            "image",
            "images",
            "picture",
            "pictures",
            "shot",
            "shots",
            "taken",
            "exif",
            "metadata",
        }
        semantic_terms = [
            t for t in semantic_terms if (t or "").strip().lower() not in metadata_stopwords
        ]
        return filtered, " ".join(semantic_terms).strip()

    def _normalize_loose_metadata_query(self, raw: str) -> str:
        """
        Accept light natural-language metadata filters without an LLM.
        Examples:
        - iso under 800 -> iso<800
        - in tokyo -> city:tokyo
        - from japan -> country:japan
        """
        text = f" {raw.strip()} "

        text = re.sub(
            r"\b(iso|year|month|width|height)\s+(\d+)\b",
            r"\1=\2",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\b(camera|lens|city|admin|country|date|filename|file|name|format|type|ext)\s*=\s*([^\s]+)\b",
            r"\1:\2",
            text,
            flags=re.I,
        )

        numeric_phrases = [
            (r"\b(iso|year|month|width|height)\s+(?:under|below|less\s+than|smaller\s+than)\s+(\d+)\b", r"\1<\2"),
            (r"\b(iso|year|month|width|height)\s+(?:at\s+most|up\s+to|no\s+more\s+than|less\s+than\s+or\s+equal\s+to)\s+(\d+)\b", r"\1<=\2"),
            (r"\b(iso|year|month|width|height)\s+(?:over|above|more\s+than|greater\s+than|larger\s+than)\s+(\d+)\b", r"\1>\2"),
            (r"\b(iso|year|month|width|height)\s+(?:at\s+least|no\s+less\s+than|greater\s+than\s+or\s+equal\s+to)\s+(\d+)\b", r"\1>=\2"),
            (r"\b(iso|year|month|width|height)\s+(?:is|equals?|equal\s+to)\s+(\d+)\b", r"\1=\2"),
        ]
        for pattern, replacement in numeric_phrases:
            text = re.sub(pattern, replacement, text, flags=re.I)

        # File-format phrases MUST run before loose "in <place>" / "from <place>" patterns,
        # otherwise "in jpeg", "near raw", "from raw" become city:/country: filters.
        text = re.sub(
            r"\b(format|type|ext)\s+(jpeg|jpe|jpg|tif|tiff|png|gif|bmp|webp|heic|heif|avif|raw"
            r"|cr2|cr3|arw|nef|dng|orf|raf|rw2|pef|srw|rwl|erf|x3f|3fr)\b",
            lambda m: f"{m.group(1).lower()}:{m.group(2).lower()}",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\b(?:file|file\s+name|name)\s+(jpeg|jpe|jpg|tif|tiff|png|gif|bmp|webp|heic|heif|avif|raw)\b",
            lambda m: f"format:{m.group(1).lower()}",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\b(?:file|file\s+name|name)\s+(cr2|cr3|arw|nef|dng|orf|raf|rw2|pef|srw|rwl|erf|x3f|3fr)\b",
            lambda m: f"format:{m.group(1).lower()}",
            text,
            flags=re.I,
        )
        
        # Multi-word location detection pass
        # This joins known multi-word places into single tokens to prevent splitting
        _MULTI_WORD_LOCATIONS = [
            "hong kong", "new york", "san francisco", "los angeles", "united states", 
            "united kingdom", "south korea", "north korea", "saudi arabia", "south africa",
            "kuala Lumpur", "ho chi minh", "san diego", "las vegas", "new zealand",
            "buenos aires", "rio de janeiro", "mexico city", "cape town", "saint petersburg",
            "san jose", "nha trang", "da nang", "koh samui", "koh phangan", "bora bora",
            "puerto rico", "costa rica", "el salvador", "gran canaria", "san sebastian"
        ]
        for loc in _MULTI_WORD_LOCATIONS:
            # Match word boundaries to avoid partial matches (e.g., "Hong Kong" vs "Hong Konger")
            pattern = rf"\b{re.escape(loc)}\b"
            if re.search(pattern, text, re.I):
                # Convert to underscore-joined city:token
                token = "_".join(loc.split())
                text = re.sub(pattern, f"city:{token}", text, flags=re.I)

        fmt_hints = self._FORMAT_HINT_FILENAME_TOKENS

        def _loose_place_to_city(m):
            place = m.group(1).strip()
            if not re.search(r"\s", place):
                token = place.lower().lstrip(".")
                if token in fmt_hints:
                    return f"format:{token}"
            return f"city:{'_'.join(place.split())}"

        def _loose_place_to_country(m):
            place = m.group(1).strip()
            if not re.search(r"\s", place):
                token = place.lower().lstrip(".")
                if token in fmt_hints:
                    return f"format:{token}"
            return f"country:{'_'.join(place.split())}"

        text = re.sub(
            r"\b(?:in|at|near)\s+([A-Za-z][\w\-]*(?:\s+[A-Za-z][\w\-]*){0,2})\b",
            _loose_place_to_city,
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\b(?:from|country)\s+([A-Za-z][\w\-]*(?:\s+[A-Za-z][\w\-]*){0,2})\b",
            _loose_place_to_country,
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\bcamera\s+([A-Za-z0-9][\w\-.]*(?:\s+[A-Za-z0-9][\w\-.]*){0,2})\b",
            lambda m: f"camera:{'_'.join(m.group(1).split())}",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\blens\s+([A-Za-z0-9][\w\-.]*(?:\s+[A-Za-z0-9][\w\-.]*){0,3})\b",
            lambda m: f"lens:{'_'.join(m.group(1).split())}",
            text,
            flags=re.I,
        )

        text = re.sub(
            r"\b(?:filename|file\s+name|file|name)\s+([A-Za-z0-9][\w\-.]*(?:\s+[A-Za-z0-9][\w\-.]*){0,2})\b",
            lambda m: f"filename:{'_'.join(m.group(1).split())}",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\bdate\s+(\d{4}(?:[-:/]\d{1,2}){0,2})\b",
            r"date:\1",
            text,
            flags=re.I,
        )

        # GPS / Location shortcuts
        text = re.sub(r"\b(?:with|has|include|containing|containing\s+exif)\s+gps\b", "has:gps", text, flags=re.I)
        text = re.sub(r"\b(?:no|without|missing|missing\s+exif)\s+gps\b", "no:gps", text, flags=re.I)
        text = re.sub(r"\bcoords?:\s*(-?\d+(?:\.\d+)?)\s*[,/ ]\s*(-?\d+(?:\.\d+)?)\b", r"gps:\1,\2", text, flags=re.I)
        
        return text.strip()

    @lru_cache(maxsize=128)
    def _is_strictly_location_name(self, term: str) -> bool:
        """
        Check if a term is strictly a known country or major city name.
        This is used to prevent semantic search 'guessing' when metadata contradicts.
        """
        needle = term.strip().lower()
        if len(needle) < 3:
            return False
            
        # Hardcoded set of common countries, cities, and travel destinations for quick check
        _LOCATIONS = {
            # Countries
            "japan", "korea", "china", "taiwan", "usa", "uk", "canada", "france", 
            "germany", "italy", "spain", "russia", "australia", "india", "brazil", 
            "mexico", "thailand", "vietnam", "singapore", "malaysia", "indonesia",
            "philippines", "switzerland", "austria", "netherlands", "greece", "turkey",
            "egypt", "south africa", "new zealand", "united states", "united kingdom",
            "south korea", "north korea",
            
            # Major Cities & Capitals
            "tokyo", "seoul", "london", "paris", "berlin", "new york", "los angeles",
            "san francisco", "hong kong", "beijing", "shanghai", "bangkok", "singapore",
            "sydney", "melbourne", "toronto", "vancouver", "rome", "milan", "madrid",
            "barcelona", "amsterdam", "vienna", "zurich", "geneva", "mumbai", "delhi",
            "bangalore", "jakarta", "manila", "ho chi minh", "hanoi", "taipei", "macao",
            "moscow", "istanbul", "dubai", "abu dhabi", "riyadh", "cairo", "nairobi",
            "chicago", "boston", "seattle", "miami", "munich", "frankfurt", "lyon",
            "prague", "budapest", "warsaw", "stockholm", "oslo", "copenhagen", "helsinki",
            "lisbon", "athens", "dublin", "tel aviv", "mexico city", "buenos aires",
            "sao paulo", "rio de janeiro", "santiago", "lima", "bogota",
            
            # Popular Travel Destinations (FB/IG Hotspots)
            "santorini", "mykonos", "bali", "phuket", "kyoto", "osaka", "nara", "hokkaido",
            "jeju", "busan", "boracay", "cebu", "nha trang", "da nang", "koh samui",
            "krabi", "chiang mai", "luang prabang", "angkor wat", "siem reap", "halong bay",
            "maldives", "fiji", "bora bora", "tahiti", "maui", "honolulu", "oahu", "kauai",
            "ibiza", "mallorca", "tenerife", "capri", "amalfi", "positano", "venice",
            "florence", "tuscany", "provence", "cannes", "nice", "monaco", "st moritz",
            "zermatt", "interlaken", "hallstatt", "salzburg", "innsbruck", "banff",
            "whistler", "yellowstone", "yosemite", "grand canyon", "sedona", "reykjavik",
            "blue lagoon", "cappadocia", "petra", "machu picchu", "cusco", "uyuni",
            "patagonia", "queenstown", "rotorua", "milford sound"
        }
        if needle in _LOCATIONS:
            return True
            
        if pycountry is not None:
            try:
                # Check for country names
                for c in pycountry.countries:
                    if c.name.lower() == needle:
                        return True
                    if hasattr(c, 'common_name') and c.common_name.lower() == needle:
                        return True
                    if hasattr(c, 'official_name') and c.official_name.lower() == needle:
                        return True
            except Exception:
                pass
        return False

    def _auto_match_metadata_keywords(
        self, rows: Sequence[sqlite3.Row], semantic_terms: Sequence[str]
    ) -> tuple[List[sqlite3.Row], List[str]]:
        """
        If a free-text token appears in metadata fields, use it as a metadata filter
        and remove it from the semantic text query.
        """
        filtered = list(rows)
        remaining: List[str] = []
        metadata_fields = ("city", "admin1", "country", "country_code", "camera_model", "lens_model", "file_name")

        import logging
        logger = logging.getLogger(__name__)

        for term in semantic_terms:
            needle = (term or "").strip().lower()
            if len(needle) < 3:
                remaining.append(term)
                continue
            if re.fullmatch(r"\d{4}(?:[:-]\d{1,2}){0,2}", needle):
                matched_rows = [
                    r
                    for r in filtered
                    if self._capture_time_matches_date(str(self._row_value(r, "capture_time") or ""), needle)
                ]
                if matched_rows:
                    filtered = matched_rows
                    continue
            matched_rows = [
                r
                for r in filtered
                if any(self._contains_loose(str(self._row_value(r, field) or ""), needle) for field in metadata_fields)
            ]
            if matched_rows:
                logger.info(f"[SEARCH] Metadata match for '{needle}' found in {len(matched_rows)} image(s)")
                filtered = matched_rows
            else:
                # CONTRADICTION FILTER: If this is a location name (e.g. "Korea") but 
                # doesn't match any image in the folder, check if any images HAVE 
                # verified location metadata. If an image says "Japan", and we search 
                # "Korea", we should exclude it rather than letting AI "guess".
                if self._is_strictly_location_name(needle):
                    new_filtered = []
                    contradiction_count = 0
                    for r in filtered:
                        city = str(self._row_value(r, "city") or "").lower()
                        country = str(self._row_value(r, "country") or "").lower()
                        # If the image HAS location metadata, it must match the term
                        if city or country:
                            if self._contains_loose(city, needle) or self._contains_loose(country, needle):
                                new_filtered.append(r)
                            else:
                                # Contradiction! (Has location, but it's different)
                                contradiction_count += 1
                                pass
                        else:
                            # No location metadata - keep it for semantic guessing
                            new_filtered.append(r)
                    
                    if contradiction_count > 0:
                        logger.warning(f"[SEARCH] Contradiction Filter: Excluded {contradiction_count} image(s) from '{needle}' due to conflicting GPS metadata")
                    filtered = new_filtered
                else:
                    logger.debug(f"[SEARCH] No metadata match for '{needle}', passing to semantic AI")
                    remaining.append(term)

        return filtered, remaining

    def search_text(
        self, query: str, candidate_paths: Sequence[str], top_k: int = 200, min_score: float = 0.0
    ) -> List[SearchHit]:
        raw_query = (query or "").strip()
        if not raw_query:
            return []
        canonical_to_original = {}
        signature_to_original = {}
        for p in candidate_paths:
            if not p or not os.path.isfile(p):
                continue
            try:
                canonical = self._canonical_path(p)
                st = os.stat(canonical)
                canonical_to_original[canonical] = p
                signature_to_original[self._file_signature_from_stat(canonical, st)] = p
            except Exception:
                canonical_to_original[self._canonical_path(p)] = p
        rows = self._fetch_rows_for_paths(candidate_paths)
        if not rows:
            return []
        rows, semantic_query = self._apply_filters(rows, raw_query)
        if not rows:
            return []

        # Filter-only query: return deterministic list without semantic ranking.
        if not semantic_query:
            hits = [
                SearchHit(
                    file_path=str(
                        signature_to_original.get(str(r["file_signature"] or ""))
                        or canonical_to_original.get(self._canonical_path(str(r["file_path"])))
                        or str(r["file_path"])
                    ),
                    score=1.0,
                    file_name=str(r["file_name"] or os.path.basename(str(r["file_path"]))),
                    capture_time=str(r["capture_time"] or ""),
                    camera_model=str(r["camera_model"] or ""),
                    lens_model=str(r["lens_model"] or ""),
                    iso=int(r["iso"] or 0),
                    gps_lat=float(r["gps_lat"]) if r["gps_lat"] is not None else None,
                    gps_lon=float(r["gps_lon"]) if r["gps_lon"] is not None else None,
                    city=str(r["city"] or ""),
                    admin1=str(r["admin1"] or ""),
                    country=str(r["country"] or ""),
                    country_code=str(r["country_code"] or ""),
                    face_count=int(r["face_count"] or 0),
                )
                for r in rows
            ]
            hits.sort(key=lambda h: h.capture_time, reverse=True)
            return hits[: max(1, int(top_k))]

        query_vec = self._encode_text(semantic_query)

        scores: List[SearchHit] = []
        for r in rows:
            if not self._row_semantic_ready(r):
                continue
            vec = self._from_blob(r["embedding"], int(r["dim"]))
            if vec.size == 0:
                continue
            score = float(np.dot(query_vec, vec))
            # Semantic query should filter out clearly non-matching images.
            # Keep strict filtering (not just ranking) by removing sub-threshold scores.
            if score <= float(min_score):
                continue
            scores.append(
                SearchHit(
                    file_path=str(
                        signature_to_original.get(str(r["file_signature"] or ""))
                        or canonical_to_original.get(self._canonical_path(str(r["file_path"])))
                        or str(r["file_path"])
                    ),
                    score=score,
                    file_name=str(r["file_name"] or os.path.basename(str(r["file_path"]))),
                    capture_time=str(r["capture_time"] or ""),
                    camera_model=str(r["camera_model"] or ""),
                    lens_model=str(r["lens_model"] or ""),
                    iso=int(r["iso"] or 0),
                    gps_lat=float(r["gps_lat"]) if r["gps_lat"] is not None else None,
                    gps_lon=float(r["gps_lon"]) if r["gps_lon"] is not None else None,
                    city=str(r["city"] or ""),
                    admin1=str(r["admin1"] or ""),
                    country=str(r["country"] or ""),
                    country_code=str(r["country_code"] or ""),
                    face_count=int(r["face_count"] or 0),
                )
            )
        scores.sort(key=lambda x: x.score, reverse=True)
        return scores[: max(1, int(top_k))]

    def search_metadata_text(
        self, query: str, candidate_paths: Sequence[str], top_k: int = 500, sort_newest: bool = True
    ) -> tuple[List[SearchHit], str]:
        """
        Metadata/EXIF-only search that works without any embedding backend.

        Returns (hits, remaining_semantic_query). If remaining_semantic_query is empty,
        the query was fully satisfied by metadata filters/auto-matches.
        """
        raw_query = (query or "").strip()
        if not raw_query:
            return [], ""

        needs_face = self._query_needs_face_detection(raw_query)
        rows = self._metadata_rows_for_search(candidate_paths, needs_face=needs_face)

        filtered, semantic_query = self._apply_filters(rows, raw_query)
        hits = [
            SearchHit(
                file_path=str(r["file_path"]),
                score=1.0,
                file_name=str(r.get("file_name") or os.path.basename(str(r["file_path"]))),
                capture_time=str(r["capture_time"] or ""),
                camera_model=str(r["camera_model"] or ""),
                lens_model=str(r["lens_model"] or ""),
                iso=int(r["iso"] or 0),
                gps_lat=float(r["gps_lat"]) if r["gps_lat"] is not None else None,
                gps_lon=float(r["gps_lon"]) if r["gps_lon"] is not None else None,
                city=str(r["city"] or ""),
                admin1=str(r["admin1"] or ""),
                country=str(r["country"] or ""),
                country_code=str(r["country_code"] or ""),
                face_count=int(r["face_count"] or 0),
            )
            for r in filtered
        ]
        hits.sort(key=lambda h: h.capture_time, reverse=sort_newest)
        return hits[: max(1, int(top_k))], semantic_query

    def _metadata_rows_for_search(
        self, candidate_paths: Sequence[str], needs_face: bool = False
    ) -> List[Dict[str, object]]:
        """Metadata rows for search, with DB-first lookup and fallback EXIF extraction."""
        if not candidate_paths:
            return []

        # Build a canonical→original map so DB rows can be resolved back to the
        # original-case path that self.image_files uses. On Windows the DB stores
        # lowercase paths (_canonical_path lowercases), but the UI file list keeps
        # the original case from os.listdir(). Returning lowercase paths from search
        # causes a mismatch where the gallery cannot locate the files.
        canonical_to_original: Dict[str, str] = {}
        for p in candidate_paths:
            if p:
                canonical_to_original[self._canonical_path(p)] = p

        # Bulk fetch all rows that exist in the index
        rows: List[Dict[str, object]] = []
        db_rows = self._fetch_rows_for_paths(candidate_paths)
        found_paths: set[str] = set()

        # Rows present in DB (fast path).
        for row in db_rows:
            db_path = str(row["file_path"])
            # Resolve back to original-case path from candidate_paths; fall back to
            # the DB path (already canonical/lowercase) if not found.
            canonical = self._canonical_path(db_path)
            original = canonical_to_original.get(canonical, db_path)
            found_paths.add(self._canonical_path(original))
            
            face_count = row["face_count"] if "face_count" in row.keys() else None
            # Only detect faces during search if explicitly needed and missing, 
            # but ideally this should be done during background indexing.
            if needs_face and face_count is None:
                if os.path.isfile(original):
                    face_count = self._detect_face_count(original)
                    self._store_face_count(original, int(face_count or 0))
            
            # Lazy metadata repair: If we have GPS but no location names (city/country),
            # it might have been indexed when the geocoder was unavailable. 
            # Try to fix it on the fly.
            city = str(row["city"] or "")
            admin1 = str(row["admin1"] or "")
            country = str(row["country"] or "")
            gps_lat = row["gps_lat"]
            gps_lon = row["gps_lon"]
            
            if not city and gps_lat is not None and gps_lon is not None:
                geo = self._ensure_reverse_geocoder()
                if geo:
                    try:
                        recs = geo.search([(float(gps_lat), float(gps_lon))], mode=1)
                        if recs:
                            rec = recs[0] or {}
                            city = str(rec.get("name", "") or "")
                            admin1 = str(rec.get("admin1", "") or "")
                            cc = str(rec.get("cc", "") or "").upper()
                            country = self._country_name_from_code(cc)
                            # Update DB so we don't have to geocode this file again
                            self._conn.execute(
                                "UPDATE semantic_index SET city=?, admin1=?, country=?, country_code=? WHERE file_path=?",
                                (city, admin1, country, cc, original)
                            )
                            self._conn.commit()
                    except Exception:
                        pass

            rows.append(
                {
                    "file_path": original,
                    "file_name": str(row["file_name"] or os.path.basename(original)),
                    "capture_time": str(row["capture_time"] or ""),
                    "camera_model": str(row["camera_model"] or ""),
                    "lens_model": str(row["lens_model"] or ""),
                    "iso": int(row["iso"] or 0),
                    "width": int(row["width"] or 0),
                    "height": int(row["height"] or 0),
                    "gps_lat": gps_lat,
                    "gps_lon": gps_lon,
                    "city": city,
                    "admin1": admin1,
                    "country": country,
                    "country_code": str(row["country_code"] or ""),
                    "face_count": int(face_count or 0),
                }
            )
        # Fallback for files not yet indexed: extract EXIF so metadata search can still
        # cover the whole album while semantic indexing continues in background.
        for p in candidate_paths:
            if not p or not os.path.isfile(p):
                continue
            canonical = self._canonical_path(p)
            if canonical in found_paths:
                continue
            meta = self._extract_exif_brief(canonical, include_face=needs_face)
            rows.append(
                {
                    "file_path": canonical,
                    "file_name": os.path.basename(canonical),
                    "capture_time": str(meta.get("capture_time") or ""),
                    "camera_model": str(meta.get("camera_model") or ""),
                    "lens_model": str(meta.get("lens_model") or ""),
                    "iso": int(meta.get("iso") or 0),
                    "width": int(meta.get("width") or 0),
                    "height": int(meta.get("height") or 0),
                    "gps_lat": meta.get("gps_lat"),
                    "gps_lon": meta.get("gps_lon"),
                    "city": str(meta.get("city") or ""),
                    "admin1": str(meta.get("admin1") or ""),
                    "country": str(meta.get("country") or ""),
                    "country_code": str(meta.get("country_code") or ""),
                    "face_count": int(meta.get("face_count") or 0),
                }
            )
        return rows

    def _store_face_count(self, file_path: str, face_count: int, conn: Optional[sqlite3.Connection] = None) -> None:
        try:
            db = conn if conn is not None else self._conn
            canonical = self._canonical_path(file_path)
            st = os.stat(canonical)
            signature = self._file_signature_from_stat(canonical, st)
            aliases = self._path_aliases(canonical)
            placeholders = ",".join(["?"] * len(aliases))
            db.execute(
                f"""
                UPDATE semantic_index
                SET face_count = ?
                WHERE (file_signature = ? AND model_name = ?)
                   OR file_path IN ({placeholders})
                """,
                [int(face_count), signature, self.model_name, *aliases],
            )
            db.commit()
        except Exception:
            pass

    @staticmethod
    def _query_needs_face_detection(query: str) -> bool:
        return bool(
            re.search(
                r"\b(?:has:faces?|has:people|has:person|has:humans?"
                r"|no:faces?|no:people|no:person|no:humans?"
                r"|faces?|people|person|humans?)\b",
                query or "",
                flags=re.I,
            )
        )

