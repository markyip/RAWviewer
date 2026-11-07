"""Thumbnail prep, ONNX/DNN/CoreML, and source image loading helpers."""
from __future__ import annotations

import contextlib
import json
import os
import platform
import sys
from collections.abc import Sequence
from typing import cast

import numpy as np
from PIL import Image, ImageOps
from PIL.Image import Image as PilImage

from raw_adjustments.file_extensions import RAW_FILE_EXTENSIONS

from ._config import (
    _INDEX_THREAD_LOCAL,
    _YUNET_INFER_LOCK,
    _YUNET_SINGLETON,
    _YUNET_SINGLETON_MODEL_PATH,
    _dnn_get_backend_name,
)


def _thumbnail_jpeg_bytes(arr: np.ndarray, quality: int = 85) -> bytes:
    from common_image_loader import encode_tile_bytes

    pil = Image.fromarray(np.asarray(arr, dtype=np.uint8)).convert("RGB")
    return encode_tile_bytes(pil, quality=quality)

def _maybe_persist_index_thumbnail(file_path: str, im: PilImage) -> None:
    """Disk-back index thumbnails so deferred face scan can reuse semantic loads."""
    try:
        from image_cache import get_image_cache

        cache = get_image_cache()
        if _index_source_image_cached(cache, file_path):
            return
        rgb = im.convert("RGB")
        if max(rgb.width, rgb.height) > 1024:
            rgb.thumbnail((1024, 1024), Image.Resampling.BICUBIC)
        arr = np.asarray(rgb, dtype=np.uint8)
        cache.put_thumbnail(file_path, arr, _thumbnail_jpeg_bytes(arr))
    except Exception:
        pass

def _prep_mobileclip_image_chw_resized(file_path: str, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Load, resize to size (width, height), and return NCHW float tensor slice for one image."""
    im = _load_index_source_image(file_path, max_size=1024, qt_decode=False)
    _maybe_persist_index_thumbnail(file_path, im)
    im = im.resize(size, Image.Resampling.BICUBIC)
    rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))

def _prep_mobileclip_image_chw(file_path: str) -> np.ndarray:
    """Load, resize to 256, and return NCHW float tensor slice for one image."""
    return _prep_mobileclip_image_chw_resized(file_path, (256, 256))

def _semantic_batch_cache_path() -> str:
    return os.path.join(
        os.path.expanduser("~"), ".rawviewer_cache", "semantic_batch_tuning.json"
    )

def _load_semantic_batch_cache() -> dict[str, int]:
    try:
        p = _semantic_batch_cache_path()
        if not os.path.exists(p):
            return {}
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): int(v) for k, v in data.items()}
    except Exception:
        pass
    return {}

def _save_semantic_batch_cache(cache: dict[str, int]) -> None:
    try:
        p = _semantic_batch_cache_path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, sort_keys=True)
    except Exception:
        pass

def _import_onnxruntime():
    """Return onnxruntime module, or None if missing or broken (e.g. partial install)."""
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    if not callable(getattr(ort, "InferenceSession", None)):
        return None
    return ort

def _onnxruntime_availability_error() -> str:
    ort = _import_onnxruntime()
    if ort is None:
        return (
            "Broken or missing 'onnxruntime' dependency "
            "(run: pixi run pip install --force-reinstall onnxruntime-gpu)"
        )
    if not callable(getattr(ort, "get_available_providers", None)):
        return (
            "Broken 'onnxruntime' install (missing get_available_providers; "
            "run: pixi run pip install --force-reinstall onnxruntime-gpu)"
        )
    return ""

def _ort_get_available_providers(ort=None) -> list[str]:
    if ort is None:
        ort = _import_onnxruntime()
    if ort is None:
        return ["CPUExecutionProvider"]
    getter = getattr(ort, "get_available_providers", None)
    if callable(getter):
        try:
            return list(cast(Sequence[str], getter()))
        except Exception:
            pass
    return ["CPUExecutionProvider"]

def resolve_onnx_execution_providers(
    available: Sequence[str] | None = None,
) -> list[str]:
    """Pick ONNX Runtime EPs for the current OS and GPU vendor.

    Windows: CUDA / TensorRT first when installed (NVIDIA); DirectML next for
    AMD / Intel or when CUDA is unavailable (default pixi build uses DirectML only).
    macOS: Core ML EP when using ONNX fallback; otherwise app uses native Core ML.
    Other Unix (no official Linux release): CUDA / ROCm / OpenVINO when present in a
    custom onnxruntime build, else CPU — source-only, not tested as a product target.
    Override with RAWVIEWER_ORT_PROVIDERS (comma-separated EP names).
    """
    if available is None:
        available = _ort_get_available_providers()

    override = os.environ.get("RAWVIEWER_ORT_PROVIDERS", "").strip()
    if override:
        names = [p.strip() for p in override.split(",") if p.strip()]
        selected = [p for p in names if p in available]
        return selected or ["CPUExecutionProvider"]

    if sys.platform == "win32":
        preferred = [
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
            "DmlExecutionProvider",
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        ]
    elif sys.platform == "darwin":
        preferred = [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
    else:
        preferred = [
            "CUDAExecutionProvider",
            "ROCMExecutionProvider",
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        ]

    selected = [p for p in preferred if p in available]
    return selected or ["CPUExecutionProvider"]

def resolve_opencv_dnn_backend_target() -> tuple:
    """OpenCV DNN backend/target for YuNet / SSD face models (Windows).

    GPU-first when CUDA is available. Face scan runs after semantic indexing
    (never in parallel with MobileCLIP ONNX), so both can use the GPU sequentially.
    OpenCL is opt-in via RAWVIEWER_FACE_DNN_TARGET=opencl (some Windows stacks crash).
    Override with RAWVIEWER_FACE_DNN_TARGET=cpu|opencl|cuda|openvino.
    """
    import cv2

    override = os.environ.get("RAWVIEWER_FACE_DNN_TARGET", "").strip().lower()
    backend_opencv = cv2.dnn.DNN_BACKEND_OPENCV
    target_cpu = cv2.dnn.DNN_TARGET_CPU

    if override == "cpu":
        return backend_opencv, target_cpu
    if override == "opencl":
        _log_opencl_runtime_details_once(cv2)
        return backend_opencv, cv2.dnn.DNN_TARGET_OPENCL
    if override == "cuda":
        return cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA
    if override == "openvino":
        return cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE, target_cpu

    # Auto: CUDA when available, else CPU. OpenCL stays disabled unless requested.
    with contextlib.suppress(Exception):
        cv2.ocl.setUseOpenCL(False)
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            return cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA
    except Exception:
        pass
    return backend_opencv, target_cpu

def _apply_opencv_dnn_acceleration(net) -> None:
    """Set preferable backend/target on an OpenCV dnn.Net (best-effort)."""
    import cv2

    backend_id, target_id = resolve_opencv_dnn_backend_target()
    _log_face_backend_once(backend_id, target_id, cv2)
    try:
        net.setPreferableBackend(backend_id)
        net.setPreferableTarget(target_id)
    except Exception:
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except Exception:
            pass

def _yunet_model_path() -> str:
    cache_dir = os.path.expanduser("~/.rawviewer_cache/models")
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, "face_detection_yunet_2023mar.onnx")
    if not os.path.exists(model_path):
        import logging

        logging.getLogger(__name__).info(
            "[VISION] Downloading YuNet ONNX face detection model (353 KB)..."
        )
        url = (
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx"
        )
        from core.ssl_certs import urlretrieve

        urlretrieve(url, model_path)
        logging.getLogger(__name__).info("[VISION] YuNet ONNX model downloaded successfully.")
    return model_path

def _yunet_warmup() -> None:
    """Initialize YuNet once on the calling thread before any worker pool starts."""
    import logging

    import numpy as np

    try:
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        count = _yunet_detect_faces_bgr(dummy)
        logging.getLogger(__name__).info("[VISION] YuNet warmup ok (faces=%d)", count)
    except Exception as exc:
        logging.getLogger(__name__).warning("[VISION] YuNet warmup failed: %s", exc)

def _yunet_detect_faces_bgr(img_bgr) -> int:
    """Run YuNet on a BGR image using one shared detector (thread-safe, GPU-friendly)."""
    global _YUNET_SINGLETON, _YUNET_SINGLETON_MODEL_PATH
    import logging

    import cv2

    model_path = _yunet_model_path()
    h, w = img_bgr.shape[:2]
    with _YUNET_INFER_LOCK:
        if _YUNET_SINGLETON is None or model_path != _YUNET_SINGLETON_MODEL_PATH:
            backend_id, target_id = resolve_opencv_dnn_backend_target()
            logger = logging.getLogger(__name__)
            logger.info("[VISION] Initializing OpenCV YuNet (singleton)...")
            try:
                _YUNET_SINGLETON = cv2.FaceDetectorYN.create(
                    model=model_path,
                    config="",
                    input_size=(w, h),
                    score_threshold=0.85,
                    nms_threshold=0.3,
                    top_k=5000,
                    backend_id=backend_id,
                    target_id=target_id,
                )
            except Exception:
                _YUNET_SINGLETON = cv2.FaceDetectorYN.create(
                    model=model_path,
                    config="",
                    input_size=(w, h),
                    score_threshold=0.85,
                    nms_threshold=0.3,
                    top_k=5000,
                    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                    target_id=cv2.dnn.DNN_TARGET_CPU,
                )
                backend_id = cv2.dnn.DNN_BACKEND_OPENCV
                target_id = cv2.dnn.DNN_TARGET_CPU
            _YUNET_SINGLETON_MODEL_PATH = model_path
            logger.info(
                "[VISION] OpenCV YuNet ready (backend=%s target=%s).",
                backend_id,
                target_id,
            )
        _YUNET_SINGLETON.setInputSize((w, h))
        _retval, faces = _YUNET_SINGLETON.detect(img_bgr)
    return len(faces) if faces is not None else 0

_OPENCL_RUNTIME_LOGGED = False
_FACE_BACKEND_LOGGED = False

def _log_opencl_runtime_details_once(cv2_module=None) -> None:
    """Best-effort one-time OpenCL runtime/device details for diagnostics."""
    global _OPENCL_RUNTIME_LOGGED
    if _OPENCL_RUNTIME_LOGGED:
        return
    _OPENCL_RUNTIME_LOGGED = True

    import logging

    logger = logging.getLogger(__name__)
    try:
        cv2 = cv2_module
        if cv2 is None:
            import cv2 as _cv2
            cv2 = _cv2

        have = bool(cv2.ocl.haveOpenCL())
        enabled = bool(cv2.ocl.useOpenCL())
        if not have:
            logger.info("[ACCEL] OpenCL runtime: unavailable")
            return

        try:
            dev = cv2.ocl.Device.getDefault()
        except Exception:
            dev = None

        if dev is None:
            logger.info("[ACCEL] OpenCL runtime: available, enabled=%s (device details unavailable)", enabled)
            return

        def _safe(callable_or_attr, default="unknown"):
            try:
                v = callable_or_attr() if callable(callable_or_attr) else callable_or_attr
                return str(v) if v not in (None, "") else default
            except Exception:
                return default

        name = _safe(getattr(dev, "name", None))
        vendor = _safe(getattr(dev, "vendorName", None))
        version = _safe(getattr(dev, "version", None))
        driver = _safe(getattr(dev, "driverVersion", None))
        ocl_c = _safe(getattr(dev, "OpenCL_C_Version", None))
        logger.info(
            "[ACCEL] OpenCL runtime: enabled=%s; vendor=%s; device=%s; version=%s; OpenCL_C=%s; driver=%s",
            enabled,
            vendor,
            name,
            version,
            ocl_c,
            driver,
        )
    except Exception as e:
        logger.info("[ACCEL] OpenCL runtime details unavailable: %s", e)

def _log_face_backend_once(backend_id: int, target_id: int, cv2_module=None) -> None:
    """One-time log of the actual selected OpenCV face backend/target."""
    global _FACE_BACKEND_LOGGED
    if _FACE_BACKEND_LOGGED:
        return
    _FACE_BACKEND_LOGGED = True

    import logging

    logger = logging.getLogger(__name__)
    try:
        cv2 = cv2_module
        if cv2 is None:
            import cv2 as _cv2
            cv2 = _cv2

        backend_name = _dnn_get_backend_name(cv2, backend_id)

        target_name = "CPU"
        if target_id == cv2.dnn.DNN_TARGET_OPENCL:
            target_name = "OpenCL"
        elif target_id == cv2.dnn.DNN_TARGET_CUDA:
            target_name = "CUDA"
        elif target_id == getattr(cv2.dnn, "DNN_TARGET_OPENCL_FP16", -1):
            target_name = "OpenCL FP16"

        logger.info(
            "[ACCEL] Face DNN backend selected: backend=%s target=%s (id=%s)",
            backend_name,
            target_name,
            target_id,
        )
    except Exception as e:
        logger.info("[ACCEL] Face DNN backend selected: backend_id=%s target_id=%s (%s)", backend_id, target_id, e)

_COREML_COMPUTE_UNITS_WARNED = False

def _macos_version_major() -> int:
    if sys.platform != "darwin":
        return 0
    try:
        ver = platform.mac_ver()[0]
        if not ver:
            return 0
        return int(str(ver).split(".")[0])
    except Exception:
        return 0

def _coreml_compute_units_setting_for_indexing() -> str:
    """Compute units for the image encoder (indexing)."""
    env_index = os.environ.get("RAWVIEWER_COREML_COMPUTE_UNITS_INDEX")
    if env_index is not None and env_index.strip():
        return env_index.strip().lower()
    env_val = os.environ.get("RAWVIEWER_COREML_COMPUTE_UNITS")
    if env_val is not None and env_val.strip():
        return env_val.strip().lower()
    return "all"

def _coreml_compute_units_setting_for_search() -> str:
    """Compute units for the text encoder (semantic search queries)."""
    env_search = os.environ.get("RAWVIEWER_COREML_COMPUTE_UNITS_SEARCH")
    if env_search is not None and env_search.strip():
        return env_search.strip().lower()
    # macOS 26+ beta: text encoder hits MPSGraph SDPA crash; image indexing is stable on GPU.
    if _macos_version_major() >= 26:
        return "cpu"
    env_val = os.environ.get("RAWVIEWER_COREML_COMPUTE_UNITS")
    if env_val is not None and env_val.strip():
        return env_val.strip().lower()
    return "all"

def _coreml_compute_units_setting(role: str = "index") -> str:
    if role == "search":
        return _coreml_compute_units_setting_for_search()
    return _coreml_compute_units_setting_for_indexing()

def _log_coreml_compute_units_profile_once() -> None:
    global _COREML_COMPUTE_UNITS_WARNED
    if _COREML_COMPUTE_UNITS_WARNED:
        return
    _COREML_COMPUTE_UNITS_WARNED = True

    import logging

    logger = logging.getLogger(__name__)
    macos_major = _macos_version_major()
    index_cu = _coreml_compute_units_setting_for_indexing()
    search_cu = _coreml_compute_units_setting_for_search()

    if macos_major >= 26 and index_cu != search_cu:
        logger.warning(
            "[ACCEL] macOS %d+: Core ML split mode index=%s search=%s "
            "(text encoder MPSGraph instability on beta). "
            "Override with RAWVIEWER_COREML_COMPUTE_UNITS_INDEX / _SEARCH.",
            macos_major,
            index_cu,
            search_cu,
        )
    elif macos_major >= 26 and search_cu in ("all", "gpu", "ane"):
        logger.warning(
            "[ACCEL] RAWVIEWER_COREML_COMPUTE_UNITS_SEARCH=%s on macOS %d+ beta may crash "
            "during semantic search.",
            search_cu,
            macos_major,
        )

def _coreml_compute_units(role: str = "index"):
    """Core ML compute-unit preference for MobileCLIP on macOS."""
    import CoreML

    raw = _coreml_compute_units_setting(role)
    mapping = {
        "cpu": CoreML.MLComputeUnitsCPUOnly,
        "cpuonly": CoreML.MLComputeUnitsCPUOnly,
        "gpu": CoreML.MLComputeUnitsCPUAndGPU,
        "ane": CoreML.MLComputeUnitsCPUAndNeuralEngine,
        "all": CoreML.MLComputeUnitsAll,
    }
    return mapping.get(raw, CoreML.MLComputeUnitsAll)

_ACCEL_PROFILE_LOGGED = False

def log_inference_acceleration_profile(force: bool = False) -> None:
    """Log once which inference accelerators are active (semantic + face)."""
    global _ACCEL_PROFILE_LOGGED
    if _ACCEL_PROFILE_LOGGED and not force:
        return
    _ACCEL_PROFILE_LOGGED = True

    import logging

    logger = logging.getLogger(__name__)
    parts: list[str] = []

    if sys.platform == "darwin":
        index_cu = _coreml_compute_units_setting_for_indexing()
        search_cu = _coreml_compute_units_setting_for_search()
        if index_cu == search_cu:
            parts.append(f"semantic=Core ML compute_units={index_cu}")
        else:
            parts.append(f"semantic=Core ML index={index_cu} search={search_cu}")
    else:
        try:
            available = _ort_get_available_providers()
            selected = resolve_onnx_execution_providers(available)
            parts.append(f"semantic=ONNX [{', '.join(selected)}]")
        except Exception:
            parts.append("semantic=ONNX [CPUExecutionProvider]")

    if sys.platform == "darwin":
        parts.append("face=Apple Vision")
    else:
        try:
            import cv2

            b, t = resolve_opencv_dnn_backend_target()
            bname = _dnn_get_backend_name(cv2, b)
            parts.append(f"face=OpenCV YuNet backend={bname} target={t}")
        except Exception:
            parts.append("face=OpenCV YuNet (CPU)")

    gpu_view = os.environ.get("RAWVIEWER_GPU_VIEW", "1").strip().lower()
    if gpu_view not in ("0", "false", "no", "off"):
        no_gl = os.environ.get("RAWVIEWER_GPU_VIEW_NO_GL", "").strip().lower()
        if no_gl in ("1", "true", "yes", "on"):
            parts.append("display=Qt raster viewport")
        else:
            parts.append("display=Qt OpenGL viewport (vendor-agnostic)")

    logger.info("[ACCEL] Inference profile: %s", "; ".join(parts))

def _cached_index_source_array(cache, file_path: str) -> np.ndarray | None:
    """RGB array from ImageCache tiers usable for semantic/face indexing (gallery may warm preview/grid)."""
    for getter_name in ("get_preview", "get_grid", "get_thumbnail"):
        try:
            arr = getattr(cache, getter_name)(file_path)
            if arr is not None:
                return np.asarray(arr, dtype=np.uint8)
        except Exception:
            continue
    return None

def _index_source_cache_tier(cache, file_path: str) -> str:
    """Which ImageCache tier satisfied the lookup, or empty string if miss."""
    for tier, getter_name in (
        ("preview", "get_preview"),
        ("grid", "get_grid"),
        ("thumbnail", "get_thumbnail"),
    ):
        try:
            arr = getattr(cache, getter_name)(file_path)
            if arr is not None:
                return tier
        except Exception:
            continue
    return ""

def _index_source_image_cached(cache, file_path: str) -> bool:
    return _cached_index_source_array(cache, file_path) is not None

def _index_source_thumbnail_present(cache, file_path: str) -> bool:
    """Fast presence check for index/face thumbnails (no JPEG decode)."""
    key = cache._path_key(file_path)
    if cache.thumbnail_cache.get(key) is not None:
        return True
    if cache.grid_cache.get(key) is not None:
        return True
    if cache.preview_cache.get(key) is not None:
        return True
    for disk_cache in (
        cache.disk_thumbnail_cache,
        cache.disk_grid_cache,
        cache.disk_preview_cache,
    ):
        try:
            if hasattr(disk_cache, "has_valid") and disk_cache.has_valid(file_path):
                return True
        except Exception:
            continue
    return False

def _index_thumbnail_needs_warm(cache, file_path: str) -> bool:
    """True when index tiers are missing."""
    return not _index_source_thumbnail_present(cache, file_path)

def _load_face_scan_image(file_path: str, max_size: int) -> PilImage | None:
    """Face detection input from ImageCache only (semantic / gallery warm-up).

    YuNet only needs a downscaled RGB image; reusing warmed thumbnails avoids
    per-file RAW decode and keeps Qt off background worker threads.
    """
    from image_cache import get_image_cache

    _INDEX_THREAD_LOCAL.last_original_sizes = (0, 0)
    cache = get_image_cache()
    arr = _cached_index_source_array(cache, file_path)
    if arr is None:
        return None
    im = Image.fromarray(arr).convert("RGB")
    _INDEX_THREAD_LOCAL.last_original_sizes = (im.width, im.height)
    if max(im.width, im.height) > max_size:
        im.thumbnail((max_size, max_size), Image.Resampling.BICUBIC)
    return im

def _apply_exif_rotation_to_pil(file_path: str, im: PilImage, cache) -> PilImage:
    try:
        from PIL import ImageOps

        from common_image_loader import exif_display_dimensions

        # verify=True: orientation here transposes indexed pixels (worker thread).
        exif_data = cache.get_exif(file_path, verify=True)
        if not exif_data:
            from enhanced_raw_processor import EXIFExtractor
            exif_data = EXIFExtractor().extract_exif_data(file_path)

        if exif_data:
            orientation = int(exif_data.get("orientation", 1) or 1)
            if orientation in (2, 3, 4, 5, 6, 7, 8):
                ow = int(exif_data.get("original_width") or 0)
                oh = int(exif_data.get("original_height") or 0)
                if ow > 0 and oh > 0:
                    dw, dh = exif_display_dimensions(ow, oh, orientation)
                    if (dh > dw) != (im.height > im.width):
                        exif = im.getexif()
                        exif[0x0112] = orientation
                        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    return im

def _load_index_source_image(
    file_path: str, max_size: int = 1024, *, qt_decode: bool = True
) -> PilImage:
    """Load a small RGB image suitable for indexing/detection, preferring app caches.

    When ``qt_decode`` is False, skip QImageReader / Qt-based decoders so face scan
    workers never touch Qt from a background thread pool (can crash the process).
    """
    _INDEX_THREAD_LOCAL.last_original_sizes = (0, 0)
    from image_cache import get_image_cache
    cache = get_image_cache()

    try:
        arr = _cached_index_source_array(cache, file_path)
        if arr is not None:
            from common_image_loader import (
                finalize_index_thumbnail_array,
                index_thumbnail_needs_orient_fix,
            )

            if index_thumbnail_needs_orient_fix(file_path, arr, cache=cache):
                fixed = finalize_index_thumbnail_array(file_path, arr, cache=cache)
                if fixed is not None:
                    arr = fixed
                    with contextlib.suppress(Exception):
                        cache.put_thumbnail(
                            file_path, arr, _thumbnail_jpeg_bytes(arr)
                        )
            im = Image.fromarray(arr).convert("RGB")
            _INDEX_THREAD_LOCAL.last_original_sizes = (im.width, im.height)
            im.thumbnail((max_size, max_size), Image.Resampling.BICUBIC)
            tier = _index_source_cache_tier(cache, file_path)
            if tier:
                hits = getattr(_INDEX_THREAD_LOCAL, "index_source_cache_hits", None)
                if hits is None:
                    hits = {"preview": 0, "grid": 0, "thumbnail": 0}
                    _INDEX_THREAD_LOCAL.index_source_cache_hits = hits
                hits[tier] = int(hits.get(tier, 0)) + 1
            return im
    except Exception:
        pass

    try:
        with Image.open(file_path) as im:
            _INDEX_THREAD_LOCAL.last_original_sizes = (im.width, im.height)
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((max_size, max_size), Image.Resampling.BICUBIC)
            return im.copy()
    except Exception:
        pass

    ext = os.path.splitext(file_path)[1].lower().lstrip(".")
    if ext in RAW_FILE_EXTENSIONS:
        if not qt_decode:
            try:
                from enhanced_raw_processor import (
                    ThumbnailExtractor,
                    extract_embedded_jpeg_by_scan,
                )

                arr = extract_embedded_jpeg_by_scan(file_path, max_size)
                if arr is None:
                    thumb = ThumbnailExtractor().extract_thumbnail_from_raw(
                        file_path,
                        max_size=max_size,
                        allow_scan_fallback=False,
                    )
                    if thumb is not None:
                        arr = np.asarray(thumb, dtype=np.uint8)
                if arr is not None:
                    from common_image_loader import (
                        finalize_index_thumbnail_array,
                        publish_mipmap_tiers,
                    )

                    arr = finalize_index_thumbnail_array(file_path, arr, cache=cache)
                    # Semantic/face indexing sometimes decodes a RAW before gallery or
                    # single-view ever touches it (background indexing racing ahead).
                    # Publish here too so that first decode isn't wasted -- gallery/
                    # film strip can reuse these tiers instead of re-extracting.
                    publish_mipmap_tiers(file_path, arr, cache=cache, source="semantic")
                    im = Image.fromarray(arr).convert("RGB")
                    _INDEX_THREAD_LOCAL.last_original_sizes = (im.width, im.height)
                    im.thumbnail((max_size, max_size), Image.Resampling.BICUBIC)
                    return im
            except Exception:
                pass
        try:
            from PyQt6.QtGui import QImage

            from enhanced_raw_processor import ThumbnailExtractor, _thumbnail_via_qimage_reader

            thumb = ThumbnailExtractor().extract_thumbnail_from_raw(
                file_path, max_size=max_size, allow_scan_fallback=True
            )
            if thumb is None and qt_decode:
                arr = _thumbnail_via_qimage_reader(file_path, max_size)
                if arr is not None:
                    thumb = arr
            if thumb is not None:
                if isinstance(thumb, QImage):
                    if not qt_decode:
                        raise RuntimeError("Qt image in face-scan path")
                    from enhanced_raw_processor import _qimage_to_rgb_array

                    arr = _qimage_to_rgb_array(thumb)
                    if arr is None:
                        raise ValueError("QImage conversion failed")
                else:
                    arr = np.asarray(thumb, dtype=np.uint8)
                from common_image_loader import (
                    finalize_index_thumbnail_array,
                    publish_mipmap_tiers,
                )

                arr = finalize_index_thumbnail_array(file_path, arr, cache=cache)
                publish_mipmap_tiers(file_path, arr, cache=cache, source="semantic")
                im = Image.fromarray(arr).convert("RGB")
                _INDEX_THREAD_LOCAL.last_original_sizes = (im.width, im.height)
                im.thumbnail((max_size, max_size), Image.Resampling.BICUBIC)
                return im
        except Exception:
            pass
        if qt_decode:
            try:
                from processing.unified import UnifiedImageProcessor

                thumb = UnifiedImageProcessor().process_thumbnail(
                    file_path, allow_heavy_fallback=True
                )
                if thumb is not None:
                    if isinstance(thumb, np.ndarray):
                        im = Image.fromarray(np.asarray(thumb, dtype=np.uint8)).convert("RGB")
                    else:
                        from PyQt6.QtGui import QImage

                        from enhanced_raw_processor import _qimage_to_rgb_array

                        if isinstance(thumb, QImage):
                            arr = _qimage_to_rgb_array(thumb)
                            im = Image.fromarray(arr).convert("RGB") if arr is not None else None
                        else:
                            im = None
                    if im is not None:
                        _INDEX_THREAD_LOCAL.last_original_sizes = (im.width, im.height)
                        im.thumbnail((max_size, max_size), Image.Resampling.BICUBIC)
                        return im
            except Exception:
                pass

    raise RuntimeError(
        f"Cannot decode image for semantic index: {os.path.basename(file_path)}"
    )
