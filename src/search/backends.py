"""MobileCLIP backends (CoreML + ONNX) and tokenizer."""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import os
import re
import shutil
import sys
import threading
from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
from PIL import Image
from PIL.Image import Image as PilImage

from search.indexer import (
    _coreml_compute_units,
    _coreml_compute_units_setting,
    _get_shared_index_executor,
    _load_index_source_image,
    _materialize_cached_file,
    _onnxruntime_availability_error,
    _ort_get_available_providers,
    _prep_mobileclip_image_chw_resized,
    resolve_onnx_execution_providers,
    semantic_coreml_use_native_batch,
    semantic_encode_prep_workers,
)
from search.indexer._config import _COREML_PREDICTION_LOCK
from search.indexer._rest import _log_coreml_compute_units_profile_once


def _objc_pointer_addr(ptr: Any) -> int | None:
    """Best-effort address from a PyObjC / ctypes pointer-like value."""
    if ptr is None:
        return None
    try:
        addr = int(ptr)
        return addr if addr else None
    except (TypeError, ValueError):
        pass
    try:
        import ctypes

        return ctypes.cast(ptr, ctypes.c_void_p).value
    except Exception:
        return None

def copy_numpy_into_mlmultiarray(multi_array: Any, flat: np.ndarray) -> bool:
    """Bulk-fill an MLMultiArray from a contiguous 1-D numpy buffer.

    Uses ``dataPointer`` + ``memmove`` instead of per-element
    ``setObject_atIndexedSubscript_`` (hundreds of thousands of ObjC calls for
    a 1×3×256×256 float image tensor). Returns False so callers can fall back.
    """
    try:
        import ctypes

        src = np.ascontiguousarray(flat).reshape(-1)
        nbytes = int(src.nbytes)
        if nbytes <= 0:
            return False
        addr = _objc_pointer_addr(multi_array.dataPointer())
        if not addr:
            return False
        ctypes.memmove(addr, src.ctypes.data, nbytes)
        return True
    except Exception:
        return False

def mlmultiarray_to_numpy_f32(multi_array: Any) -> np.ndarray | None:
    """Bulk-read MLMultiArray contents via ``dataPointer`` into float32.

    Supports Float32 / Float16 / Double outputs used by MobileCLIP. Returns None
    when the pointer path is unavailable so callers can use ``numberArray()``.
    """
    try:
        import ctypes

        count = int(multi_array.count())
        if count <= 0:
            return np.zeros((0,), dtype=np.float32)
        dtype = int(multi_array.dataType())
        # Apple MLMultiArrayDataType: Float16=16, Float32=65568, Double=65552
        if dtype == 65568:
            np_dtype: Any = np.float32
            itemsize = 4
        elif dtype == 65552:
            np_dtype = np.float64
            itemsize = 8
        elif dtype == 16:
            np_dtype = np.float16
            itemsize = 2
        else:
            return None
        addr = _objc_pointer_addr(multi_array.dataPointer())
        if not addr:
            return None
        buf_type = ctypes.c_char * (count * itemsize)
        raw = buf_type.from_address(addr)
        arr = np.frombuffer(raw, dtype=np_dtype, count=count).astype(
            np.float32, copy=True
        )
        return np.ascontiguousarray(arr)
    except Exception:
        return None

def copy_bgra_into_cvpixelbuffer(
    base_addr: Any,
    bytes_per_row: int,
    bgra: np.ndarray,
    height: int = 256,
    width: int = 256,
) -> bool:
    """Bulk-write a contiguous BGRA HxWx4 array into a locked CVPixelBuffer.

    When ``bytes_per_row`` matches tightly packed rows, one ``memmove`` replaces
    a Python per-row loop. Returns False to fall back to row copies.
    """
    try:
        import ctypes

        packed = int(width) * 4
        src = np.ascontiguousarray(bgra, dtype=np.uint8)
        if src.shape != (height, width, 4):
            return False
        addr = _objc_pointer_addr(base_addr)
        if not addr:
            return False
        if int(bytes_per_row) == packed:
            ctypes.memmove(addr, src.ctypes.data, height * packed)
            return True
        # Strided destination: still avoid tobytes() per row.
        row_src = src.reshape(height, packed)
        for y in range(height):
            ctypes.memmove(addr + y * int(bytes_per_row), row_src[y].ctypes.data, packed)
        return True
    except Exception:
        return False

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

    Exported MobileCLIP2 models (see ``scripts/models/export_mobileclip2_coreml.py``)
    expose the image encoder as FLOAT32 MultiArray ``[1,3,256,256]`` in NCHW
    pixel scale ``[0,1]``. Apple-shipped MobileCLIP S2 bundles use MLFeatureTypeImage;
    ``encode_image`` supports both via model introspection.
    """

    MODEL_ID = "mobileclip-coreml-s2"
    HUB_REPO_ID = "apple/coreml-mobileclip"
    HUB_REVISION = "3e0a7bfb9fe83da8a3efaa3fd8f7df24214bb947"
    IMAGE_MODEL_FILE = "mobileclip_s2_image.mlpackage"
    TEXT_MODEL_FILE = "mobileclip_s2_text.mlpackage"
    TOKENIZER_URL = "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"
    SUPPORTS_HUB_DOWNLOAD = True

    _COREML_BUNDLE_PAIRS: tuple[tuple[str, str], ...] = (
        ("mobileclip_s2_image.mlpackage", "mobileclip_s2_text.mlpackage"),
        ("mobileclip2_s0_image.mlpackage", "mobileclip2_s0_text.mlpackage"),
    )

    def __init__(self, model_dir: str | None = None):
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
        self._objc = None

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
    def _candidate_model_dirs() -> list[str]:
        dirs: list[str] = []
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
        out: list[str] = []
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
            pass
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

    @staticmethod
    def _mlpackage_hub_files(bundle_name: str) -> list[str]:
        return [
            f"{bundle_name}/Manifest.json",
            f"{bundle_name}/Data/com.apple.CoreML/model.mlmodel",
            f"{bundle_name}/Data/com.apple.CoreML/weights/weight.bin",
        ]

    def download_assets(
        self, progress_callback: Callable[..., None] | None = None
    ) -> str:
        """Download MobileCLIP S2 Core ML assets into the backend model directory."""
        if sys.platform != "darwin":
            raise RuntimeError("MobileCLIP Core ML download is only supported on macOS")

        def _report(pct: int) -> None:
            from core.download_progress import report_progress

            report_progress(progress_callback, pct, installer=False)

        os.makedirs(self.model_dir, exist_ok=True)
        _report(0)
        try:
            from core.ssl_certs import configure_ssl_certificates

            configure_ssl_certificates()
        except Exception:
            pass
        try:
            from huggingface_hub import hf_hub_download as _hf_hub_download
        except Exception as exc:
            raise RuntimeError(
                "MobileCLIP auto-download requires 'huggingface_hub'. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc
        hf_hub_download = cast(Any, _hf_hub_download)

        from core.download_progress import make_byte_progress_tqdm

        weight_files = [
            (
                f"{self.IMAGE_MODEL_FILE}/Data/com.apple.CoreML/weights/weight.bin",
                0,
                45,
            ),
            (
                f"{self.TEXT_MODEL_FILE}/Data/com.apple.CoreML/weights/weight.bin",
                45,
                85,
            ),
        ]
        small_files = [
            path
            for bundle in (self.IMAGE_MODEL_FILE, self.TEXT_MODEL_FILE)
            for path in self._mlpackage_hub_files(bundle)
            if not path.endswith("weight.bin")
        ]

        for remote_path, stage_start, stage_end in weight_files:
            tqdm_class = make_byte_progress_tqdm(stage_start, stage_end, _report)
            cached_path = hf_hub_download(
                repo_id=self.HUB_REPO_ID,
                filename=remote_path,
                revision=self.HUB_REVISION,
                tqdm_class=tqdm_class,
            )
            _materialize_cached_file(
                str(cached_path), os.path.join(self.model_dir, remote_path)
            )
            _report(stage_end)

        for idx, remote_path in enumerate(small_files):
            cached_path = hf_hub_download(
                repo_id=self.HUB_REPO_ID,
                filename=remote_path,
                revision=self.HUB_REVISION,
            )
            _materialize_cached_file(
                str(cached_path), os.path.join(self.model_dir, remote_path)
            )
            pct = 85 + int((idx + 1) * 8 / max(1, len(small_files)))
            _report(min(93, pct))

        if not os.path.exists(self.tokenizer_path):
            from core.ssl_certs import urlretrieve

            urlretrieve(self.TOKENIZER_URL, self.tokenizer_path)
        _report(100)

        err = self.availability_error()
        if err:
            raise RuntimeError(err)
        return self.model_dir

    def _load_models(self):
        if self._image_model is not None and self._text_model is not None:
            return
        import CoreML
        import Foundation
        import objc
        import Quartz

        self._CoreML = CoreML
        self._Foundation = Foundation
        self._Quartz = Quartz
        self._objc = objc
        _log_coreml_compute_units_profile_once()

        with _COREML_PREDICTION_LOCK:
            if self._image_model is not None and self._text_model is not None:
                return

            def _load_one(path: str, role: str):
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
                cu = _coreml_compute_units_setting(role)
                compiled_path = os.path.join(
                    cache_dir, f"{os.path.basename(path)}_{h}_{cu}.mlmodelc"
                )
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

                config = CoreML.MLModelConfiguration.alloc().init()
                with contextlib.suppress(Exception):
                    config.setComputeUnits_(_coreml_compute_units(role))
                model, load_error = CoreML.MLModel.modelWithContentsOfURL_configuration_error_(
                    compiled_url, config, None
                )
                if load_error is not None or model is None:
                    model, load_error = CoreML.MLModel.modelWithContentsOfURL_error_(compiled_url, None)
                if load_error is not None or model is None:
                    raise RuntimeError(f"Failed to load Core ML model: {load_error}")
                return model

            if self._image_model is None:
                self._image_model = _load_one(self.image_model_path, "index")
            if self._text_model is None:
                self._text_model = _load_one(self.text_model_path, "search")

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
        CoreML, Foundation, Quartz, objc, image_model, text_model = self._require_coreml_runtime()
        with objc.autorelease_pool():
            tokenizer = self._ensure_tokenizer()
            tokens = np.asarray([tokenizer.encode_for_clip(text)], dtype=np.int32)
            input_name = self._native_feature_name(text_model, "input")
            output_name = self._native_feature_name(text_model, "output")
            multi_array = self._int32_multi_array(tokens.reshape(-1), CoreML)
            feature = CoreML.MLFeatureValue.featureValueWithMultiArray_(multi_array)
            provider, err = CoreML.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(
                {input_name: feature}, None
            )
            if err is not None or provider is None:
                raise RuntimeError(f"Failed to create Core ML text input: {err}")
            with _COREML_PREDICTION_LOCK:
                out, err = text_model.predictionFromFeatures_error_(provider, None)
            if err is not None or out is None:
                raise RuntimeError(f"MobileCLIP text prediction failed: {err}")
            numpy_arr = self._multi_array_to_numpy(out.featureValueForName_(output_name).multiArrayValue())
        return self._normalize(numpy_arr)

    def _float32_multi_array_nchw(self, tensor_nchw: np.ndarray, CoreML):
        t = np.asarray(tensor_nchw, dtype=np.float32).reshape(1, 3, 256, 256)
        flat = np.ascontiguousarray(t).ravel(order="C")
        arr, err = CoreML.MLMultiArray.alloc().initWithShape_dataType_error_(
            [1, 3, 256, 256], CoreML.MLMultiArrayDataTypeFloat32, None
        )
        if err is not None or arr is None:
            raise RuntimeError(f"Failed to allocate Core ML image tensor: {err}")
        if not copy_numpy_into_mlmultiarray(arr, flat):
            for i, value in enumerate(flat):
                arr.setObject_atIndexedSubscript_(float(value), i)
        return arr

    def encode_image(self, file_path: str) -> np.ndarray:
        CoreML, Foundation, Quartz, objc, image_model, text_model = self._require_coreml_runtime()
        with objc.autorelease_pool():
            im = _load_index_source_image(file_path, max_size=1024, qt_decode=False).resize(
                (256, 256), Image.Resampling.BICUBIC
            )
            return self._encode_pil_image(im, CoreML, objc, image_model)

    def encode_images(self, file_paths: Sequence[str]) -> list[np.ndarray]:
        """Parallel CPU prep; Core ML predict serial by default (batch opt-in via env)."""
        CoreML, Foundation, Quartz, objc, image_model, text_model = self._require_coreml_runtime()
        paths = [p for p in (file_paths or []) if p]
        if not paths:
            return []
        if len(paths) == 1:
            return [self.encode_image(paths[0])]

        sample_path = paths[0]
        workers = semantic_encode_prep_workers(sample_path)

        def _prep(path: str) -> PilImage:
            return _load_index_source_image(path, max_size=1024, qt_decode=False).resize(
                (256, 256), Image.Resampling.BICUBIC
            )

        images = [_prep(p) for p in paths] if workers <= 1 else list(_get_shared_index_executor().map(_prep, paths))

        if not semantic_coreml_use_native_batch():
            with objc.autorelease_pool():
                return [self._encode_pil_image(im, CoreML, objc, image_model) for im in images]

        with objc.autorelease_pool():
            desc = image_model.modelDescription()
            input_name = self._native_feature_name(image_model, "input")
            output_name = self._native_feature_name(image_model, "output")
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

            providers = []
            for im in images:
                feature = None
                if in_type == CoreML.MLFeatureTypeMultiArray:
                    rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
                    nchw = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]
                    multi = self._float32_multi_array_nchw(nchw, CoreML)
                    feature = CoreML.MLFeatureValue.featureValueWithMultiArray_(multi)
                elif in_type == CoreML.MLFeatureTypeImage:
                    pixel_buffer = self._image_to_pixel_buffer(im, Quartz)
                    feature = CoreML.MLFeatureValue.alloc().initWithValue_type_(
                        pixel_buffer, CoreML.MLFeatureTypeImage
                    )
                else:
                    raise RuntimeError(f"Unsupported Core ML image encoder input type: {in_type}")

                provider, err = CoreML.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(
                    {input_name: feature}, None
                )
                if err is not None or provider is None:
                    raise RuntimeError(f"Failed to create Core ML feature provider: {err}")
                providers.append(provider)

            return self._coreml_predict_batch(
                providers, output_name, paths, CoreML, image_model
            )

    def _coreml_predict_batch(
        self,
        providers: list,
        output_name: str,
        paths: Sequence[str],
        CoreML,
        image_model,
    ) -> list[np.ndarray]:
        """Run Core ML batch inference; halve batch size on memory/ANE failures."""
        if not providers:
            return []
        if len(providers) == 1:
            return [self.encode_image(paths[0])]

        batch_provider = CoreML.MLArrayBatchProvider.alloc().initWithFeatureProviderArray_(
            providers
        )
        try:
            with _COREML_PREDICTION_LOCK:
                predictions, err = image_model.predictionsFromBatch_error_(
                    batch_provider, None
                )
            if err is not None or predictions is None:
                raise RuntimeError(f"MobileCLIP Core ML batch prediction failed: {err}")
        except Exception:
            mid = len(providers) // 2
            if mid <= 0:
                return [self.encode_image(p) for p in paths]
            left = self._coreml_predict_batch(
                providers[:mid], output_name, paths[:mid], CoreML, image_model
            )
            right = self._coreml_predict_batch(
                providers[mid:], output_name, paths[mid:], CoreML, image_model
            )
            return left + right

        embeddings: list[np.ndarray] = []
        count = int(predictions.count())
        for i in range(count):
            out_provider = predictions.featuresAtIndex_(i)
            numpy_arr = self._multi_array_to_numpy(
                out_provider.featureValueForName_(output_name).multiArrayValue()
            )
            embeddings.append(self._normalize(numpy_arr))
        return embeddings

    def _encode_pil_image(self, im: PilImage, CoreML, objc, image_model) -> np.ndarray:
        with objc.autorelease_pool():
            desc = image_model.modelDescription()
            input_name = self._native_feature_name(image_model, "input")
            output_name = self._native_feature_name(image_model, "output")
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
                multi = self._float32_multi_array_nchw(nchw, CoreML)
                feature = CoreML.MLFeatureValue.featureValueWithMultiArray_(multi)
            elif in_type == CoreML.MLFeatureTypeImage:
                _, _, Quartz, _, _, _ = self._require_coreml_runtime()
                pixel_buffer = self._image_to_pixel_buffer(im, Quartz)
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
            with _COREML_PREDICTION_LOCK:
                out, err = image_model.predictionFromFeatures_error_(provider, None)
            if err is not None or out is None:
                raise RuntimeError(f"MobileCLIP image prediction failed: {err}")
            numpy_arr = self._multi_array_to_numpy(out.featureValueForName_(output_name).multiArrayValue())
        return self._normalize(numpy_arr)

    def _int32_multi_array(self, values: np.ndarray, CoreML):
        arr, err = CoreML.MLMultiArray.alloc().initWithShape_dataType_error_(
            [1, int(values.size)], CoreML.MLMultiArrayDataTypeInt32, None
        )
        if err is not None or arr is None:
            raise RuntimeError(f"Failed to allocate Core ML token array: {err}")
        flat = np.asarray(values, dtype=np.int32).reshape(-1)
        if not copy_numpy_into_mlmultiarray(arr, flat):
            for i, value in enumerate(flat):
                arr.setObject_atIndexedSubscript_(int(value), i)
        return arr

    @staticmethod
    def _multi_array_to_numpy(multi_array) -> np.ndarray:
        fast = mlmultiarray_to_numpy_f32(multi_array)
        if fast is not None:
            return fast
        values = multi_array.numberArray()
        return np.asarray([float(values[i]) for i in range(len(values))], dtype=np.float32)

    def _image_to_pixel_buffer(self, im: PilImage, Quartz):
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
            if not copy_bgra_into_cvpixelbuffer(base, bytes_per_row, bgra):
                buf = base.as_buffer(bytes_per_row * 256)
                row_bytes = 256 * 4
                for y in range(256):
                    start = y * bytes_per_row
                    buf[start:start + row_bytes] = bgra[y].tobytes()
        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(pixel_buffer, 0)
        return pixel_buffer

    def _require_coreml_runtime(self):
        """Return loaded Core ML modules; raises if unavailable."""
        self._load_models()
        coreml = self._CoreML
        foundation = self._Foundation
        quartz = self._Quartz
        objc = self._objc
        image_model = self._image_model
        text_model = self._text_model
        if (
            coreml is None
            or foundation is None
            or quartz is None
            or objc is None
            or image_model is None
            or text_model is None
        ):
            raise RuntimeError("Core ML models not loaded")
        return coreml, foundation, quartz, objc, image_model, text_model

    def release_gpu_sessions(self) -> None:
        """Drop Core ML models and sessions to free VRAM/system memory."""
        self._image_model = None
        self._text_model = None
        self._CoreML = None
        self._Foundation = None
        self._Quartz = None
        self._objc = None
        import gc
        gc.collect()

class MobileCLIPONNXBackend:
    """Windows ONNX backend for MobileCLIP2 (supports S0, S2, B, L14 variants).

    Requires 'onnxruntime' and 'numpy'.
    """

    MODEL_ID = "mobileclip-onnx-2-s0"
    HUB_REPO_ID = "plhery/mobileclip2-onnx"
    HUB_REVISION = "ba95759a5bdbaca53e9111e2550a76ec09c8fd9e"
    IMAGE_MODEL_FILE = "image_encoder.onnx"
    TEXT_MODEL_FILE = "text_encoder.onnx"
    ONNX_SHA256 = {
        "s0": (
            "13d20ebfa8a8f63890eb2727fe4dc63009ff970f43e0f7d9d2ed999659f70c8a",
            "df590d47744f2ee9f3ccb67c4414d17419568c05bca0c4d166f2faeedf8b92f3",
        ),
        "s2": (
            "a841f72c5a5085748bbe271a1d5718aba877822a15cba865bdbd0d37036b849e",
            "622f10372bca71b5017f2efc5f8c2886610a2592b636de8984d717f03213f031",
        ),
        "b": (
            "2f3873e400fd1adc5298b72d29e81bb69337e58601cf2853c43bc1211631eecd",
            "f5cbb08e1eb6a5da690554a43cd37304835c0412c547ddb4ae86f0f8759686e0",
        ),
        "l14": (
            "d48d4392f5805f59896009ea5460c7fa29f59722edd0d7d26c054b0980020998",
            "9dac2284d0cbbf767ef25363d8536ac183f6c1d02df3a00c477843f7be0e9d17",
        ),
    }
    TOKENIZER_URL = "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"
    SUPPORTS_HUB_DOWNLOAD = True

    def __init__(self, model_dir: str | None = None):
        self.variant = os.environ.get("RAWVIEWER_MOBILECLIP_VARIANT", "b").strip().lower()
        if self.variant not in ("s0", "s2", "b", "l14"):
            self.variant = "b"
        self.MODEL_ID = f"mobileclip-onnx-2-{self.variant}"

        if model_dir is None:
            model_dir = self._default_model_dir(self.variant)
        self.model_dir = model_dir
        self.image_model_path = os.path.join(model_dir, self.IMAGE_MODEL_FILE)
        self.text_model_path = os.path.join(model_dir, self.TEXT_MODEL_FILE)
        self.tokenizer_path = os.path.join(model_dir, "bpe_simple_vocab_16e6.txt.gz")
        self._image_session = None
        self._text_session = None
        self._tokenizer = None

    @staticmethod
    def _candidate_model_dirs(variant: str = "b") -> list[str]:
        dirs: list[str] = []
        env_dir = os.environ.get("RAWVIEWER_MOBILECLIP_MODEL_DIR")
        if env_dir:
            dirs.append(env_dir)

        suffix = f"_{variant}" if variant != "b" else ""
        if getattr(sys, "frozen", False):
            # Prioritize the actual executable directory for external (non-bundled) models
            exe_dir = os.path.dirname(sys.executable)
            dirs.append(os.path.join(exe_dir, "models", f"mobileclip_onnx{suffix}"))
            dirs.append(os.path.join(exe_dir, f"mobileclip_onnx{suffix}"))

            # Fallback to PyInstaller temporary extract directory (_MEIPASS)
            meipass = getattr(sys, "_MEIPASS", None)
            if meipass is not None:
                dirs.append(os.path.join(meipass, "models", f"mobileclip_onnx{suffix}"))

        dirs.append(os.path.expanduser(f"~/.rawviewer_cache/mobileclip_onnx{suffix}"))

        module_dir = os.path.dirname(os.path.abspath(__file__))
        dirs.append(os.path.join(module_dir, "..", "models", f"mobileclip_onnx{suffix}"))
        return dirs

    @classmethod
    def _default_model_dir(cls, variant: str = "b") -> str:
        for d in cls._candidate_model_dirs(variant):
            if (
                os.path.exists(os.path.join(d, cls.IMAGE_MODEL_FILE))
                and os.path.exists(os.path.join(d, cls.TEXT_MODEL_FILE))
            ):
                return d
        return cls._candidate_model_dirs(variant)[0]

    def availability_error(self) -> str:
        err = _onnxruntime_availability_error()
        if err:
            return err
        if not os.path.exists(self.image_model_path):
            return f"Missing MobileCLIP image model: {self.IMAGE_MODEL_FILE}"
        if not os.path.exists(self.text_model_path):
            return f"Missing MobileCLIP text model: {self.TEXT_MODEL_FILE}"
        return ""

    def available(self) -> bool:
        return self.availability_error() == ""

    def download_assets(
        self, progress_callback: Callable[..., None] | None = None
    ) -> str:
        from core.download_progress import make_byte_progress_tqdm, report_progress

        def _report(pct: int) -> None:
            report_progress(progress_callback, pct, installer=False)

        os.makedirs(self.model_dir, exist_ok=True)
        _report(0)

        try:
            from huggingface_hub import hf_hub_download as _hf_hub_download
        except ImportError:
            raise RuntimeError("MobileCLIP download requires 'huggingface_hub' (pip install huggingface_hub)") from None
        hf_hub_download = cast(Any, _hf_hub_download)

        files_to_download = [
            (f"onnx/{self.variant}/vision_model.onnx", self.IMAGE_MODEL_FILE, 0, 58),
            (f"onnx/{self.variant}/text_model.onnx", self.TEXT_MODEL_FILE, 58, 98),
        ]

        for remote_path, local_name, stage_start, stage_end in files_to_download:
            tqdm_class = make_byte_progress_tqdm(stage_start, stage_end, _report)
            cached_path = hf_hub_download(
                repo_id=self.HUB_REPO_ID,
                filename=remote_path,
                revision=self.HUB_REVISION,
                tqdm_class=tqdm_class,
            )

            target_path = os.path.join(self.model_dir, local_name)
            _materialize_cached_file(str(cached_path), target_path)
            if not os.path.exists(target_path):
                raise RuntimeError(f"MobileCLIP ONNX download missing: {local_name}")
            expected = (self.ONNX_SHA256.get(self.variant) or (None, None))[
                0 if local_name == self.IMAGE_MODEL_FILE else 1
            ]
            if expected:
                digest = hashlib.sha256()
                with open(target_path, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest().lower() != expected.lower():
                    with contextlib.suppress(OSError):
                        os.remove(target_path)
                    raise RuntimeError(
                        f"MobileCLIP ONNX {local_name} failed SHA-256 verification"
                    )
            _report(stage_end)

        # Remove the layout left by older versions that downloaded with
        # ``local_dir`` (files are now served from the shared HF cache).
        onnx_dir = os.path.join(self.model_dir, "onnx")
        if os.path.exists(onnx_dir):
            shutil.rmtree(onnx_dir)

        if not os.path.exists(self.tokenizer_path):
            from core.ssl_certs import urlretrieve

            urlretrieve(self.TOKENIZER_URL, self.tokenizer_path)
        _report(100)

        return self.model_dir

    def _ensure_sessions(self):
        if self._image_session is not None:
            return
        err = _onnxruntime_availability_error()
        if err:
            raise RuntimeError(err)
        import onnxruntime as ort

        available = _ort_get_available_providers(ort)
        selected_providers = resolve_onnx_execution_providers(available)

        import logging

        logging.getLogger(__name__).info(
            "[SemanticSearch] MobileCLIP ONNX providers available=%s using=%s",
            available,
            selected_providers,
        )

        self._image_session = ort.InferenceSession(
            self.image_model_path, providers=selected_providers
        )
        self._text_session = ort.InferenceSession(
            self.text_model_path, providers=selected_providers
        )

    def _require_sessions(self):
        self._ensure_sessions()
        image_session = self._image_session
        text_session = self._text_session
        if image_session is None or text_session is None:
            raise RuntimeError("ONNX sessions not loaded")
        return image_session, text_session

    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = _ClipBPETokenizer(self.tokenizer_path)
        return self._tokenizer

    def _get_input_size(self) -> tuple[int, int]:
        try:
            image_session, _ = self._require_sessions()
            shape = image_session.get_inputs()[0].shape
            h = shape[2] if isinstance(shape[2], int) else 256
            w = shape[3] if isinstance(shape[3], int) else 256
            return (w, h)
        except Exception:
            return (256, 256)

    def encode_text(self, text: str) -> np.ndarray:
        _, text_session = self._require_sessions()
        tokenizer = self._ensure_tokenizer()
        # MobileCLIP2 ONNX text encoder expects int64 token IDs (ORT rejects int32).
        tokens = np.asarray([tokenizer.encode_for_clip(text)], dtype=np.int64)

        inputs = {text_session.get_inputs()[0].name: tokens}
        outputs = text_session.run(None, inputs)
        return self._normalize(np.asarray(outputs[0], dtype=np.float32))

    def encode_image(self, file_path: str) -> np.ndarray:
        image_session, _ = self._require_sessions()
        size = self._get_input_size()
        nchw = _prep_mobileclip_image_chw_resized(file_path, size)[np.newaxis, ...]
        inputs = {image_session.get_inputs()[0].name: nchw}
        outputs = image_session.run(None, inputs)
        return self._normalize(np.asarray(outputs[0], dtype=np.float32))

    def encode_images(self, file_paths: Sequence[str]) -> list[np.ndarray]:
        """Best-effort batched image encoding for ONNX backend."""
        image_session, _ = self._require_sessions()
        paths = [p for p in (file_paths or []) if p]
        if not paths:
            return []
        size = self._get_input_size()
        sample_path = paths[0] if paths else None
        workers = semantic_encode_prep_workers(sample_path)
        if len(paths) == 1 or workers <= 1:
            tensors = [_prep_mobileclip_image_chw_resized(p, size) for p in paths]
        else:
            tensors = list(_get_shared_index_executor().map(lambda p: _prep_mobileclip_image_chw_resized(p, size), paths))
        nchw = np.stack(tensors, axis=0)
        inputs = {image_session.get_inputs()[0].name: nchw}
        outputs = image_session.run(None, inputs)
        out = np.asarray(outputs[0], dtype=np.float32)
        return [self._normalize(out[i]) for i in range(out.shape[0])]

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        return arr

    def release_gpu_sessions(self) -> None:
        """Drop ONNX sessions so a follow-up GPU pass (e.g. YuNet) can use VRAM."""
        self._image_session = None
        self._text_session = None
        import gc

        gc.collect()

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
        # Windows prefer ONNX (official release); macOS uses Core ML above.
        inst = MobileCLIPONNXBackend()
        if inst.available():
            return inst
        return inst # Return anyway for download_assets availability

def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs], strict=False))

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
        self.encoder = dict(zip(vocab, range(len(vocab)), strict=False))
        self.bpe_ranks = dict(zip(merges, range(len(merges)), strict=False))
        import collections
        self.bpe_cache = collections.OrderedDict()
        self.bpe_cache["<|startoftext|>"] = "<|startoftext|>"
        self.bpe_cache["<|endoftext|>"] = "<|endoftext|>"
        self.sot = self.encoder["<|startoftext|>"]
        self.eot = self.encoder["<|endoftext|>"]

    def bpe(self, token: str) -> str:
        # Prevent memory leak: previously used global @lru_cache kept `self` alive.
        # Now using instance-level cache `self.bpe_cache`.
        if not hasattr(self, "bpe_cache"):
            import collections
            self.bpe_cache = collections.OrderedDict()

        if token in self.bpe_cache:
            # Move to end to mark as recently used
            self.bpe_cache.move_to_end(token)
            return self.bpe_cache[token]

        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)
        if not pairs:
            res = token + "</w>"
            self.bpe_cache[token] = res
            if len(self.bpe_cache) > 8192:
                self.bpe_cache.popitem(last=False)
            return res

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
        self.bpe_cache[token] = out
        if len(self.bpe_cache) > 8192:
            self.bpe_cache.popitem(last=False)
        return out
    def encode(self, text: str) -> list[int]:
        bpe_tokens: list[int] = []
        for token in re.findall(r"<\|startoftext\|>|<\|endoftext\|>|[\w']+|[^\s\w]", text.lower()):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def encode_for_clip(self, text: str, context_length: int = 77) -> list[int]:
        tokens = [self.sot] + self.encode(text)[: context_length - 2] + [self.eot]
        return tokens + [0] * (context_length - len(tokens))

_semantic_sort_read_local = threading.local()
