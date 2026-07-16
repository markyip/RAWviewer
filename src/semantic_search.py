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
import json
import sys
import gzip
import tempfile
import threading
import concurrent.futures
from io import BytesIO
from functools import lru_cache
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, TypeAlias, cast

_SHARED_INDEX_EXECUTOR = None
_SHARED_EXECUTOR_LOCK = threading.Lock()

def _get_shared_index_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _SHARED_INDEX_EXECUTOR
    if _SHARED_INDEX_EXECUTOR is None:
        with _SHARED_EXECUTOR_LOCK:
            if _SHARED_INDEX_EXECUTOR is None:
                max_w = os.cpu_count() or 4
                # Limit max workers to avoid overwhelming the system
                max_w = min(16, max_w)
                _SHARED_INDEX_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_w, thread_name_prefix="shared-index"
                )
    return _SHARED_INDEX_EXECUTOR

import numpy as np
from PIL import Image, ImageOps
from PIL.Image import Image as PilImage


def _bundled_resource_path(relative_path: str) -> str:
    """Absolute path to bundled data (dev tree or PyInstaller _MEIPASS)."""
    base = getattr(sys, "_MEIPASS", None)
    if not base:
        base = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    return os.path.join(base, relative_path)

import metadata_backend

from raw_file_extensions import RAW_FILE_EXTENSIONS

try:
    import pycountry
except Exception:
    pycountry = None


class IndexAborted(Exception):
    """Raised when background indexing is cancelled (e.g. folder scope changed)."""


ProgressCallback = Optional[Callable[[int, int, str], None]]

MetadataRow: TypeAlias = sqlite3.Row | Dict[str, object]


def _coerce_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _dnn_get_backend_name(cv2_module, backend_id: int) -> str:
    getter = getattr(cv2_module.dnn, "getBackendName", None)
    if callable(getter):
        try:
            return str(getter(backend_id))
        except Exception:
            pass
    return str(backend_id)

_YUNET_INFER_LOCK = threading.Lock()
_COREML_PREDICTION_LOCK = threading.Lock()
_INDEX_THREAD_LOCAL = threading.local()
_YUNET_SINGLETON = None
_YUNET_SINGLETON_MODEL_PATH = ""


def format_index_progress(stage: str, done: int, total: int) -> str:
    """Concise progress for the search field, e.g. 'Semantic: 12/48'."""
    try:
        total_i = max(0, int(total))
        done_i = max(0, int(done))
    except (TypeError, ValueError):
        return stage
    if total_i <= 0:
        return stage
    done_i = min(done_i, total_i)
    return f"{stage}: {done_i}/{total_i}"


def semantic_embeddings_enabled() -> bool:
    """Check if semantic search embeddings are enabled via environment."""
    flag = os.environ.get("RAWVIEWER_ENABLE_SEMANTIC_SEARCH", "0").strip().lower()
    return flag in ("1", "true", "yes", "on")


def face_scan_enabled() -> bool:
    """Check if face scan/indexing is enabled via environment."""
    flag = os.environ.get("RAWVIEWER_ENABLE_FACE_SCAN", "1").strip().lower()
    return flag in ("1", "true", "yes", "on")


def metadata_auto_index_enabled() -> bool:
    """Check if auto metadata indexing is enabled via environment."""
    flag = os.environ.get("RAWVIEWER_AUTO_METADATA_INDEX", "1").strip().lower()
    return flag in ("1", "true", "yes", "on")


def semantic_gpu_throttle_seconds() -> float:
    """Optional per-image pacing for semantic pass.

    Set RAWVIEWER_SEMANTIC_GPU_THROTTLE_MS to a positive value when you want
    to reduce indexing pressure (e.g., keep UI extra responsive).
    Default is 0ms so indexing is not artificially slowed down.
    """
    raw = os.environ.get("RAWVIEWER_SEMANTIC_GPU_THROTTLE_MS", "0").strip()
    try:
        ms = float(raw)
    except Exception:
        ms = 0.0
    ms = max(0.0, min(ms, 2000.0))
    return ms / 1000.0


def semantic_batch_max() -> int:
    """Upper cap for semantic ONNX batch size (auto-tune and forced)."""
    raw = os.environ.get("RAWVIEWER_SEMANTIC_BATCH_MAX", "128").strip()
    try:
        v = int(raw)
    except Exception:
        v = 128
    return max(1, min(v, 256))


def semantic_batch_size() -> int:
    """Semantic ONNX batch size when auto-tune is off (default 12, tier may override)."""
    raw = os.environ.get("RAWVIEWER_SEMANTIC_BATCH_SIZE", "").strip()
    if not raw:
        return min(semantic_batch_max(), _default_semantic_batch_size())
    try:
        v = int(raw)
    except Exception:
        v = _default_semantic_batch_size()
    return max(1, min(v, semantic_batch_max()))


def _default_semantic_batch_size() -> int:
    """Platform/tier default batch (8–16) when env is unset and perf v2 is active."""
    try:
        from upgrade_compat import perf_v2_enabled

        if not perf_v2_enabled():
            return 1
    except Exception:
        return 1
    try:
        from rawviewer_profile import classify_memory_tier, system_total_ram_gb

        tier = classify_memory_tier(system_total_ram_gb())
        return {
            "low": 8,
            "medium": 12,
            "balanced": 8,
            "high": 16,
            "ultra": 16,
        }.get(tier, 12)
    except Exception:
        return 12


def semantic_batch_size_forced() -> Optional[int]:
    """Return forced batch size when explicitly set by env, else None."""
    raw = os.environ.get("RAWVIEWER_SEMANTIC_BATCH_SIZE", "").strip()
    if not raw:
        return None
    try:
        v = int(raw)
    except Exception:
        return 1
    return max(1, min(v, semantic_batch_max()))


def semantic_batch_auto_enabled() -> bool:
    """Auto tune semantic batch size when no forced batch is provided."""
    raw = os.environ.get("RAWVIEWER_SEMANTIC_BATCH_AUTO", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def semantic_batch_tune_sample_count() -> int:
    raw = os.environ.get("RAWVIEWER_SEMANTIC_BATCH_TUNE_SAMPLES", "32").strip()
    try:
        v = int(raw)
    except Exception:
        v = 32
    return max(8, min(v, 128))


def semantic_coreml_tune_sample_count() -> int:
    """Core ML encodes serially — keep auto-tune short so indexing does not look stuck."""
    raw = os.environ.get("RAWVIEWER_SEMANTIC_COREML_TUNE_SAMPLES", "").strip()
    if not raw:
        return 24
    try:
        v = int(raw)
    except Exception:
        v = 24
    return max(8, min(v, 48))


def semantic_coreml_batch_candidates() -> List[int]:
    """Chunk sizes to benchmark for Core ML (parallel prep only; inference is per-image)."""
    raw = os.environ.get("RAWVIEWER_SEMANTIC_COREML_CHUNK_CANDIDATES", "").strip()
    cap = min(semantic_batch_max(), 32)
    if raw:
        out: List[int] = []
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                out.append(max(1, min(cap, int(item))))
            except Exception:
                continue
        if out:
            seen: set[int] = set()
            uniq: List[int] = []
            for v in sorted(out):
                if v not in seen:
                    seen.add(v)
                    uniq.append(v)
            return uniq
    return [c for c in (4, 8, 16, 32) if c <= cap]


def semantic_coreml_tune_early_stop_ratio() -> float:
    """Stop trying larger chunks when throughput drops below best * ratio."""
    raw = os.environ.get("RAWVIEWER_SEMANTIC_COREML_TUNE_EARLY_STOP", "0.85").strip()
    try:
        r = float(raw)
    except Exception:
        r = 0.85
    return max(0.5, min(r, 1.0))


def semantic_batch_tie_ratio() -> float:
    """On equal throughput (within this ratio of the best), prefer a larger batch."""
    raw = os.environ.get("RAWVIEWER_SEMANTIC_BATCH_TIE_RATIO", "0.92").strip()
    try:
        r = float(raw)
    except Exception:
        r = 0.92
    return max(0.5, min(r, 1.0))


def semantic_encode_prep_workers(sample_path: Optional[str] = None) -> int:
    """Parallel CPU workers for resize/load before ONNX image batch."""
    raw = os.environ.get("RAWVIEWER_SEMANTIC_PREP_WORKERS", "").strip()
    if raw:
        try:
            return max(1, min(int(raw), 48))
        except Exception:
            pass


    try:
        from rawviewer_profile import classify_memory_tier, system_total_ram_gb

        tier = classify_memory_tier(system_total_ram_gb())
        tier_default = {
            "low": 2,
            "medium": 3,
            "balanced": 3,
            "high": 6,
            "ultra": 8,
        }.get(tier)
        if tier_default is not None:
            return tier_default
    except Exception:
        pass
    cpu = os.cpu_count() or 4
    return max(2, min(8, cpu // 2))


def semantic_coreml_use_native_batch() -> bool:
    """Use MLArrayBatchProvider batch predict (off by default on macOS — often slower than serial)."""
    raw = os.environ.get("RAWVIEWER_SEMANTIC_COREML_BATCH", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return False


def semantic_coreml_prep_chunk_size() -> int:
    """Parallel load/resize group for Core ML indexing (predict stays serial by default)."""
    forced = semantic_coreml_chunk_forced()
    if forced is not None:
        return forced
    return max(1, min(8, semantic_encode_prep_workers()))


def semantic_coreml_chunk_size() -> int:
    """Core ML MLArrayBatchProvider chunk when native batch is enabled."""
    forced = semantic_coreml_chunk_forced()
    if forced is not None:
        return forced
    try:
        from upgrade_compat import perf_v2_enabled

        if not perf_v2_enabled():
            return 8
    except Exception:
        return 8
    try:
        from rawviewer_profile import classify_memory_tier, system_total_ram_gb

        tier = classify_memory_tier(system_total_ram_gb())
        default = {
            "low": 8,
            "medium": 12,
            "balanced": 8,
            "high": 16,
            "ultra": 16,
        }.get(tier, 12)
    except Exception:
        default = 12
    return min(semantic_batch_max(), default)


def semantic_coreml_chunk_forced() -> Optional[int]:
    """Explicit RAWVIEWER_SEMANTIC_COREML_CHUNK only (unset = not forced)."""
    if "RAWVIEWER_SEMANTIC_COREML_CHUNK" not in os.environ:
        return None
    raw = os.environ.get("RAWVIEWER_SEMANTIC_COREML_CHUNK", "").strip()
    if not raw:
        return None
    try:
        v = int(raw)
    except Exception:
        v = 8
    return max(1, min(v, semantic_batch_max()))


def semantic_warm_thumbs_before_index() -> bool:
    """Pre-extract embedded thumbnails before MobileCLIP (avoids per-file rawpy in encode)."""
    raw = os.environ.get("RAWVIEWER_SEMANTIC_WARM_THUMBS", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def semantic_batch_candidates() -> List[int]:
    raw = os.environ.get(
        "RAWVIEWER_SEMANTIC_BATCH_CANDIDATES", "8,16,32,64,128"
    ).strip()
    cap = semantic_batch_max()
    out: List[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            out.append(max(1, min(cap, int(item))))
        except Exception:
            continue
    if not out:
        out = [8, 16, 32, 64, 128]
    # stable dedupe
    seen = set()
    uniq: List[int] = []
    for v in out:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


# Bump when auto-tune defaults/logic change so ~/.rawviewer_cache picks are refreshed.
SEMANTIC_BATCH_TUNE_CACHE_VERSION = "v8"


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


def _load_semantic_batch_cache() -> Dict[str, int]:
    try:
        p = _semantic_batch_cache_path()
        if not os.path.exists(p):
            return {}
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): int(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def _save_semantic_batch_cache(cache: Dict[str, int]) -> None:
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


def _ort_get_available_providers(ort=None) -> List[str]:
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
    available: Optional[Sequence[str]] = None,
) -> List[str]:
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
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
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
        from ssl_certs import urlretrieve

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
    import cv2
    import logging

    model_path = _yunet_model_path()
    h, w = img_bgr.shape[:2]
    with _YUNET_INFER_LOCK:
        if _YUNET_SINGLETON is None or _YUNET_SINGLETON_MODEL_PATH != model_path:
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


def _coreml_compute_units():
    """Core ML compute-unit preference for MobileCLIP on macOS."""
    import CoreML  # type: ignore[import-not-found]

    raw = os.environ.get("RAWVIEWER_COREML_COMPUTE_UNITS", "all").strip().lower()
    mapping = {
        "cpu": CoreML.MLComputeUnitsCPUOnly,
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
    parts: List[str] = []

    if sys.platform == "darwin":
        parts.append("semantic=Core ML (Apple GPU/ANE when available)")
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


def _cached_index_source_array(cache, file_path: str) -> Optional[np.ndarray]:
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


def _load_face_scan_image(file_path: str, max_size: int) -> Optional[PilImage]:
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
        from common_image_loader import exif_display_dimensions
        from PIL import ImageOps
        
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
                    if not ((dh > dw) == (im.height > im.width)):
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
                    try:
                        cache.put_thumbnail(
                            file_path, arr, _thumbnail_jpeg_bytes(arr)
                        )
                    except Exception:
                        pass
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
            from enhanced_raw_processor import ThumbnailExtractor, _thumbnail_via_qimage_reader
            from PyQt6.QtGui import QImage

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
                from unified_image_processor import UnifiedImageProcessor

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
    landmark: str = ""
    admin1: str = ""
    country: str = ""
    country_code: str = ""
    face_count: int = 0


def _objc_pointer_addr(ptr: Any) -> Optional[int]:
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


def mlmultiarray_to_numpy_f32(multi_array: Any) -> Optional[np.ndarray]:
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
            import CoreML  # type: ignore[import-not-found]
            import Foundation  # type: ignore[import-not-found]
            import Quartz  # type: ignore[import-not-found]
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
        self, progress_callback: Optional[Callable[..., None]] = None
    ) -> str:
        """Download MobileCLIP S2 Core ML assets into the backend model directory."""
        if sys.platform != "darwin":
            raise RuntimeError("MobileCLIP Core ML download is only supported on macOS")

        def _report(pct: int) -> None:
            from mobileclip_download_progress import report_progress

            report_progress(progress_callback, pct, installer=False)

        os.makedirs(self.model_dir, exist_ok=True)
        _report(0)
        try:
            from ssl_certs import configure_ssl_certificates

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

        from mobileclip_download_progress import make_byte_progress_tqdm

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
            hf_hub_download(
                repo_id=self.HUB_REPO_ID,
                filename=remote_path,
                local_dir=self.model_dir,
                local_dir_use_symlinks=False,
                tqdm_class=tqdm_class,
            )
            _report(stage_end)

        for idx, remote_path in enumerate(small_files):
            hf_hub_download(
                repo_id=self.HUB_REPO_ID,
                filename=remote_path,
                local_dir=self.model_dir,
                local_dir_use_symlinks=False,
            )
            pct = 85 + int((idx + 1) * 8 / max(1, len(small_files)))
            _report(min(93, pct))

        if not os.path.exists(self.tokenizer_path):
            from ssl_certs import urlretrieve

            urlretrieve(self.TOKENIZER_URL, self.tokenizer_path)
        _report(100)

        err = self.availability_error()
        if err:
            raise RuntimeError(err)
        return self.model_dir

    def _load_models(self):
        if self._image_model is not None and self._text_model is not None:
            return
        import CoreML  # type: ignore[import-not-found]
        import Foundation  # type: ignore[import-not-found]
        import Quartz  # type: ignore[import-not-found]
        import objc  # type: ignore[import-not-found]

        self._CoreML = CoreML
        self._Foundation = Foundation
        self._Quartz = Quartz
        self._objc = objc

        with _COREML_PREDICTION_LOCK:
            if self._image_model is not None and self._text_model is not None:
                return

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

                config = CoreML.MLModelConfiguration.alloc().init()
                try:
                    config.setComputeUnits_(_coreml_compute_units())
                except Exception:
                    pass
                model, load_error = CoreML.MLModel.modelWithContentsOfURL_configuration_error_(
                    compiled_url, config, None
                )
                if load_error is not None or model is None:
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

    def encode_images(self, file_paths: Sequence[str]) -> List[np.ndarray]:
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

        if workers <= 1:
            images = [_prep(p) for p in paths]
        else:
            images = list(_get_shared_index_executor().map(_prep, paths))

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
    ) -> List[np.ndarray]:
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

        embeddings: List[np.ndarray] = []
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
    IMAGE_MODEL_FILE = "image_encoder.onnx"
    TEXT_MODEL_FILE = "text_encoder.onnx"
    TOKENIZER_URL = "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"
    SUPPORTS_HUB_DOWNLOAD = True

    def __init__(self, model_dir: Optional[str] = None):
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
    def _candidate_model_dirs(variant: str = "b") -> List[str]:
        dirs: List[str] = []
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
        self, progress_callback: Optional[Callable[..., None]] = None
    ) -> str:
        from mobileclip_download_progress import make_byte_progress_tqdm, report_progress

        def _report(pct: int) -> None:
            report_progress(progress_callback, pct, installer=False)

        os.makedirs(self.model_dir, exist_ok=True)
        _report(0)

        try:
            from huggingface_hub import hf_hub_download as _hf_hub_download
        except ImportError:
            raise RuntimeError("MobileCLIP download requires 'huggingface_hub' (pip install huggingface_hub)")
        hf_hub_download = cast(Any, _hf_hub_download)

        files_to_download = [
            (f"onnx/{self.variant}/vision_model.onnx", self.IMAGE_MODEL_FILE, 0, 58),
            (f"onnx/{self.variant}/text_model.onnx", self.TEXT_MODEL_FILE, 58, 98),
        ]

        for remote_path, local_name, stage_start, stage_end in files_to_download:
            tqdm_class = make_byte_progress_tqdm(stage_start, stage_end, _report)
            hf_hub_download(
                repo_id=self.HUB_REPO_ID,
                filename=remote_path,
                local_dir=self.model_dir,
                local_dir_use_symlinks=False,
                tqdm_class=tqdm_class,
            )

            downloaded_path = os.path.join(self.model_dir, remote_path)
            target_path = os.path.join(self.model_dir, local_name)

            if os.path.exists(downloaded_path):
                if os.path.exists(target_path):
                    os.remove(target_path)
                os.rename(downloaded_path, target_path)
            _report(stage_end)

        onnx_dir = os.path.join(self.model_dir, "onnx")
        if os.path.exists(onnx_dir):
            import shutil

            shutil.rmtree(onnx_dir)

        if not os.path.exists(self.tokenizer_path):
            from ssl_certs import urlretrieve

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

    def encode_images(self, file_paths: Sequence[str]) -> List[np.ndarray]:
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
    def encode(self, text: str) -> List[int]:
        bpe_tokens: List[int] = []
        for token in re.findall(r"<\|startoftext\|>|<\|endoftext\|>|[\w']+|[^\s\w]", text.lower()):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def encode_for_clip(self, text: str, context_length: int = 77) -> List[int]:
        tokens = [self.sot] + self.encode(text)[: context_length - 2] + [self.eot]
        return tokens + [0] * (context_length - len(tokens))


_semantic_sort_read_local = threading.local()


def _semantic_sort_read_conn(db_path: str) -> sqlite3.Connection:
    """Per-thread read connection for capture_time lookups during background folder sort."""
    conn = getattr(_semantic_sort_read_local, "conn", None)
    if conn is None or getattr(_semantic_sort_read_local, "db_path", None) != db_path:
        conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=60000")
        try:
            conn.execute("PRAGMA query_only=ON")
        except sqlite3.OperationalError:
            pass
        _semantic_sort_read_local.conn = conn
        _semantic_sort_read_local.db_path = db_path
    return conn


def get_semantic_capture_times_for_paths(
    paths: Sequence[str],
    db_path: Optional[str] = None,
) -> Dict[str, str]:
    """
    Thread-safe bulk read of capture_time from semantic_index.db.

    Used by folder-load / refinement workers (not only the UI thread) so navigation
    order matches the semantic index instead of degrading to mtime when EXIF cache is cold.
    """
    if not paths:
        return {}
    if db_path is None:
        cache_dir = os.path.expanduser("~/.rawviewer_cache")
        db_path = os.path.join(cache_dir, "semantic_index.db")
    if not os.path.isfile(db_path):
        return {}

    canonical_to_original: Dict[str, str] = {}
    unique_canonical: List[str] = []
    for p in paths:
        if not p:
            continue
        cp = SemanticImageIndex._canonical_path(p)
        if cp not in canonical_to_original:
            canonical_to_original[cp] = p
            unique_canonical.append(cp)
    if not unique_canonical:
        return {}

    conn = _semantic_sort_read_conn(db_path)
    result_map: Dict[str, str] = {}
    chunk_size = 500
    for i in range(0, len(unique_canonical), chunk_size):
        chunk = unique_canonical[i : i + chunk_size]
        placeholders = ",".join(["?"] * len(chunk))
        rows = conn.execute(
            f"""
            SELECT file_path, capture_time
            FROM semantic_index
            WHERE lower(file_path) IN ({placeholders})
              AND capture_time IS NOT NULL AND capture_time != ''
            ORDER BY rowid DESC
            """,
            chunk,
        ).fetchall()
        for fp, ct in rows:
            if fp in result_map:
                continue
            orig = canonical_to_original.get(fp)
            if orig and ct:
                result_map[orig] = str(ct)
    return result_map


def get_semantic_capture_times_for_folder(
    folder_path: str,
    db_path: Optional[str] = None,
) -> Dict[str, str]:
    """Bulk read capture_time for all indexed files under folder_path (any thread)."""
    if not folder_path:
        return {}
    if db_path is None:
        cache_dir = os.path.expanduser("~/.rawviewer_cache")
        db_path = os.path.join(cache_dir, "semantic_index.db")
    if not os.path.isfile(db_path):
        return {}
    try:
        root = SemanticImageIndex._canonical_path(
            os.path.abspath(folder_path)
        )
    except OSError:
        return {}
    if not root:
        return {}
    root_lower = root.rstrip("\\/").lower()
    if os.sep == "\\":
        prefix = root_lower + "\\%"
    else:
        prefix = root_lower.replace("\\", "/") + "/%"
    path_expr = (
        "lower(replace(file_path, '/', '\\\\'))"
        if os.sep == "\\"
        else "lower(replace(file_path, '\\\\', '/'))"
    )
    conn = _semantic_sort_read_conn(db_path)
    result_map: Dict[str, str] = {}
    rows = conn.execute(
        f"""
        SELECT file_path, capture_time
        FROM semantic_index
        WHERE {path_expr} LIKE ?
          AND capture_time IS NOT NULL AND capture_time != ''
        ORDER BY rowid DESC
        """,
        (prefix,),
    ).fetchall()
    for fp, ct in rows:
        if fp in result_map:
            continue
        if ct:
            result_map[fp] = str(ct)
    return result_map


def _normalize_location_term(term: str) -> str:
    return " ".join((term or "").strip().lower().split())


def _semantic_text_variants(text: str) -> List[str]:
    """CLIP query variants for a semantic term (singular/plural + short prompt)."""
    raw = (text or "").strip()
    if not raw:
        return []
    seen: set[str] = set()
    out: List[str] = []

    def add(s: str) -> None:
        s = (s or "").strip()
        if not s:
            return
        key = s.casefold()
        if key in seen:
            return
        seen.add(key)
        out.append(s)

    add(raw)
    tokens = raw.split()
    if len(tokens) != 1:
        return out
    word = tokens[0]
    low = word.lower()
    if len(low) < 3 or not low.isalpha():
        return out
    if low.endswith("s") and len(low) > 3 and not low.endswith("ss"):
        add(word[:-1])
    elif not low.endswith("s"):
        add(word + "s")
    add(f"a photo of {word}")
    if low.endswith("s") and len(low) > 3 and not low.endswith("ss"):
        add(f"a photo of {word[:-1]}")
    elif not low.endswith("s"):
        add(f"a photo of {word}s")
    return out


# Quick O(1) checks before any geocoder CSV load (contradiction filter).
KNOWN_LOCATION_NAMES: frozenset[str] = frozenset({
    # Countries
    "japan", "korea", "china", "taiwan", "usa", "uk", "canada", "france",
    "germany", "italy", "spain", "russia", "australia", "india", "brazil",
    "mexico", "thailand", "vietnam", "singapore", "malaysia", "indonesia",
    "philippines", "switzerland", "austria", "netherlands", "greece", "turkey",
    "egypt", "south africa", "new zealand", "united states", "united kingdom",
    "south korea", "north korea",
    # Major cities & capitals
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
    "manchester", "birmingham", "liverpool", "edinburgh", "glasgow", "leeds",
    "bristol", "cardiff", "belfast", "nottingham", "sheffield", "newcastle",
    # Popular travel destinations
    "santorini", "mykonos", "bali", "phuket", "kyoto", "osaka", "nara", "hokkaido",
    "jeju", "busan", "boracay", "cebu", "nha trang", "da nang", "koh samui",
    "krabi", "chiang mai", "luang prabang", "angkor wat", "siem reap", "halong bay",
    "maldives", "fiji", "bora bora", "tahiti", "maui", "honolulu", "oahu", "kauai",
    "ibiza", "mallorca", "tenerife", "capri", "amalfi", "positano", "venice",
    "florence", "tuscany", "provence", "cannes", "nice", "monaco", "st moritz",
    "zermatt", "interlaken", "hallstatt", "salzburg", "innsbruck", "banff",
    "whistler", "yellowstone", "yosemite", "grand canyon", "sedona", "reykjavik",
    "blue lagoon", "cappadocia", "petra", "machu picchu", "cusco", "uyuni",
    "patagonia", "queenstown", "rotorua", "milford sound",
})

_PYCOUNTRY_NAME_INDEX: Optional[frozenset[str]] = None


def _pycountry_name_index() -> frozenset[str]:
    global _PYCOUNTRY_NAME_INDEX
    if _PYCOUNTRY_NAME_INDEX is not None:
        return _PYCOUNTRY_NAME_INDEX
    names: set[str] = set()
    if pycountry is not None:
        try:
            for c in pycountry.countries:
                for attr in ("name", "common_name", "official_name"):
                    raw = getattr(c, attr, None)
                    if raw:
                        tok = _normalize_location_term(str(raw))
                        if len(tok) >= 3:
                            names.add(tok)
        except Exception:
            pass
    _PYCOUNTRY_NAME_INDEX = frozenset(names)
    return _PYCOUNTRY_NAME_INDEX


class CustomReverseGeocoder:
    """GeoNames + Wikidata reverse geocoder with O(log n) nearest-neighbor lookup."""

    _LANDMARK_DIST_SQ = 0.0182  # ~15 km squared degree distance

    def __init__(self, cities: list, landmarks: Optional[list] = None):
        self.cities = cities
        self.landmarks = landmarks or []
        self._city_coords = (
            np.asarray([(float(c[0]), float(c[1])) for c in self.cities], dtype=np.float64)
            if self.cities
            else None
        )
        self._landmark_coords = (
            np.asarray([(float(lm[0]), float(lm[1])) for lm in self.landmarks], dtype=np.float64)
            if self.landmarks
            else None
        )
        self._place_names = self._build_place_name_index()

    @staticmethod
    def _build_place_name_index_from_rows(
        cities: list,
        landmarks: list,
    ) -> frozenset[str]:
        names: set[str] = set()
        for city in cities:
            primary = _normalize_location_term(str(city[2]))
            if len(primary) >= 3:
                names.add(primary)
            alternates = city[5] if len(city) > 5 else ""
            if alternates:
                for alt in str(alternates).split(","):
                    tok = _normalize_location_term(alt)
                    if len(tok) >= 3:
                        names.add(tok)
        for lm in landmarks:
            tok = _normalize_location_term(str(lm[2]))
            if len(tok) >= 3:
                names.add(tok)
        return frozenset(names)

    def _build_place_name_index(self) -> frozenset[str]:
        return self._build_place_name_index_from_rows(self.cities, self.landmarks)

    def has_place_name(self, term: str) -> bool:
        """O(1) exact match against indexed city / alternate / landmark names."""
        needle = _normalize_location_term(term)
        return len(needle) >= 3 and needle in self._place_names

    @staticmethod
    def _nearest_index(
        coords: np.ndarray,
        target_lat: float,
        target_lon: float,
    ) -> tuple[int, float]:
        """Nearest-neighbor by squared degree distance, vectorized over numpy
        rather than a scipy.spatial.cKDTree -- avoids a ~77MB scipy dependency
        for a lookup against a few hundred thousand rows at most (cities500 +
        landmarks), called once per GPS-tagged photo, not in a hot loop."""
        if coords.size == 0:
            return 0, float("inf")
        dist_sq = (coords[:, 0] - target_lat) ** 2 + (coords[:, 1] - target_lon) ** 2
        idx = int(np.argmin(dist_sq))
        return idx, float(dist_sq[idx])

    def search(self, coords: list[tuple[float, float]], mode: int = 1) -> list[dict]:
        if not coords or not self.cities:
            return []
        target_lat, target_lon = float(coords[0][0]), float(coords[0][1])

        city_idx, _ = self._nearest_index(
            self._city_coords if self._city_coords is not None else np.empty((0, 2)),
            target_lat,
            target_lon,
        )
        lat, lon, name, admin1, cc = self.cities[city_idx][:5]
        alternates = self.cities[city_idx][5] if len(self.cities[city_idx]) > 5 else ""
        best_city = {"name": name, "admin1": admin1, "cc": cc, "alternates": alternates}

        landmark_name = ""
        if self.landmarks and self._landmark_coords is not None:
            lm_idx, lm_dist_sq = self._nearest_index(
                self._landmark_coords, target_lat, target_lon
            )
            if lm_dist_sq < self._LANDMARK_DIST_SQ:
                landmark_name = str(self.landmarks[lm_idx][2])

        return [{
            "name": best_city["name"],
            "admin1": best_city["admin1"],
            "cc": best_city["cc"],
            "alternates": best_city["alternates"],
            "landmark": landmark_name,
        }]


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
        self._paused = False
        self._abort_requested = False
        self._pause_lock = threading.Lock()
        self._metadata_backfill_lock = threading.Lock()
        self._metadata_backfill_paths: set[str] = set()
        self._metadata_backfill_geocode: set[str] = set()
        self._metadata_backfill_running = False
        # ImageLoadManager throttle hooks. Acquired only around heavy work (neural
        # pass / thumbnail warm / face scan) and released while paused, so that a
        # background index sitting idle in the gallery never starves thumbnail decode.
        self._load_throttle_enter = None
        self._load_throttle_exit = None
        self._load_throttle_held = False
        self._load_throttle_lock = threading.Lock()
        self._init_db_if_needed()

    def set_load_throttle_hooks(self, enter, exit_) -> None:
        """Wire callables that raise/lower ImageLoadManager concurrency.

        Called by the UI before a background index run. ``enter``/``exit_`` are the
        ImageLoadManager throttle toggles; they are thread-safe and depth-counted.
        """
        with self._load_throttle_lock:
            self._load_throttle_enter = enter
            self._load_throttle_exit = exit_

    def _acquire_load_throttle(self) -> None:
        """Engage the ILM indexing throttle for the duration of heavy work."""
        with self._load_throttle_lock:
            if self._load_throttle_held:
                return
            fn = self._load_throttle_enter
            self._load_throttle_held = True
        if fn is not None:
            try:
                fn()
            except Exception:
                pass

    def _release_load_throttle(self) -> None:
        """Release the ILM indexing throttle (idempotent)."""
        with self._load_throttle_lock:
            if not self._load_throttle_held:
                return
            fn = self._load_throttle_exit
            self._load_throttle_held = False
        if fn is not None:
            try:
                fn()
            except Exception:
                pass

    def pause_indexing(self, pause: bool) -> None:
        """Pause or resume the background indexing loops."""
        with self._pause_lock:
            self._paused = pause
            if not pause:
                self._abort_requested = False

    def request_abort_indexing(self) -> None:
        """Stop in-flight indexing from a previous folder scope."""
        with self._pause_lock:
            self._abort_requested = True
            self._paused = True

    def _wait_if_paused(self) -> None:
        """Block while paused; raise IndexAborted when folder scope is invalidated.

        While blocked we drop any ImageLoadManager throttle we are holding so a
        paused index never serializes gallery thumbnail decode, then re-acquire it
        on resume only if it was held when the pause began.
        """
        flag = os.environ.get("RAWVIEWER_INDEX_PAUSE_IN_GALLERY", "1").strip().lower()
        throttle_dropped = False
        while True:
            with self._pause_lock:
                if self._abort_requested:
                    # Tearing down this run; leave the throttle released.
                    raise IndexAborted()
                if flag not in ("1", "true", "yes", "on") or not self._paused:
                    if throttle_dropped:
                        self._acquire_load_throttle()
                    return
            if self._load_throttle_held and not throttle_dropped:
                self._release_load_throttle()
                throttle_dropped = True
            time.sleep(0.5)

    def _mark_metadata_ready_without_embedding(
        self,
        canonical_fp: str,
        st: os.stat_result,
        conn: sqlite3.Connection,
    ) -> None:
        """Mark file as metadata ready but without semantic embeddings (dim = 0)."""
        conn.execute(
            """
            UPDATE semantic_index
            SET semantic_ready = 1, dim = 0, embedding = ?, updated_at = ?
            WHERE file_path = ? AND model_name = ? AND file_size = ? AND mtime_ns = ?
            """,
            (
                b"",
                float(time.time()),
                canonical_fp,
                self.model_name,
                int(st.st_size),
                self._mtime_ns_from_stat(st),
            ),
        )


    @property
    def backend(self):
        if self._mobileclip_backend is None:
            self._mobileclip_backend = resolve_mobileclip_backend()
        return self._mobileclip_backend

    def release_mobileclip_gpu_memory(self) -> None:
        """Free ONNX GPU memory after semantic indexing, before GPU face detection."""
        import logging

        backend = self._mobileclip_backend
        if backend is None:
            return
        if hasattr(backend, "release_gpu_sessions"):
            try:
                backend.release_gpu_sessions()
                logging.getLogger(__name__).info(
                    "[INDEX][FACE] Released MobileCLIP ONNX sessions before face scan"
                )
            except Exception as exc:
                logging.getLogger(__name__).debug(
                    "[INDEX][FACE] MobileCLIP GPU release skipped: %s", exc
                )

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
        self._reverse_geocoder: CustomReverseGeocoder | Literal[False] | None = None
        self._conn = sqlite3.connect(
            self.db_path, check_same_thread=False, timeout=60.0
        )
        self._conn.isolation_level = None
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
                landmark TEXT,
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
        if "landmark" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN landmark TEXT")
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
        # Must run after the migration block above adds gps_lat/gps_lon -- creating
        # this index earlier crashes _init_db() on any pre-GPS-feature (pre-2.5.0)
        # database with "no such column: gps_lat" (found while verifying upgrade
        # compatibility from older versions).
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_semantic_gps_latlon ON semantic_index(gps_lat, gps_lon)"
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
        if not semantic_embeddings_enabled():
            return False
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
        if not self.model_name.startswith("mobileclip-"):
            return False
        backend = self.backend
        return bool(getattr(backend, "SUPPORTS_HUB_DOWNLOAD", False))

    def download_semantic_backend_assets(
        self, progress_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        if self.model_name.startswith("mobileclip-"):
            backend = self._mobileclip_backend or resolve_mobileclip_backend()
            path = backend.download_assets(progress_callback=progress_callback)
            self._mobileclip_backend = backend
            return path
        raise RuntimeError("This semantic backend has no downloadable assets")

    def _ensure_reverse_geocoder(self) -> CustomReverseGeocoder | Literal[False]:
        if self._reverse_geocoder is not None:
            return self._reverse_geocoder
        with self._rg_lock:
            if self._reverse_geocoder is not None:
                return self._reverse_geocoder
            try:
                import gzip
                import csv
                import logging
                
                db_path = _bundled_resource_path(os.path.join("icons", "cities500.csv.gz"))
                if not os.path.isfile(db_path):
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    db_path = os.path.join(base_dir, "icons", "cities500.csv.gz")
                if not os.path.isfile(db_path):
                    db_path = os.path.join("src", "icons", "cities500.csv.gz")
                
                logging.getLogger(__name__).info("[VISION] Loading custom reverse geocoder database from %s...", db_path)
                cities = []
                with gzip.open(db_path, "rt", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        cities.append((
                            float(row[0]),
                            float(row[1]),
                            row[2],
                            row[3],
                            row[4],
                            row[5] if len(row) > 5 else ""
                        ))
                
                # Load landmarks
                landmarks_path = _bundled_resource_path(os.path.join("icons", "landmarks.csv.gz"))
                if not os.path.isfile(landmarks_path):
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    landmarks_path = os.path.join(base_dir, "icons", "landmarks.csv.gz")
                if not os.path.isfile(landmarks_path):
                    landmarks_path = os.path.join("src", "icons", "landmarks.csv.gz")
                
                landmarks = []
                if os.path.isfile(landmarks_path):
                    logging.getLogger(__name__).info("[VISION] Loading custom landmarks database from %s...", landmarks_path)
                    try:
                        with gzip.open(landmarks_path, "rt", encoding="utf-8") as f:
                            reader = csv.reader(f)
                            for row in reader:
                                landmarks.append((
                                    float(row[0]),
                                    float(row[1]),
                                    row[2],
                                    row[3]
                                ))
                    except Exception as e:
                        logging.getLogger(__name__).warning("[VISION] Failed to parse landmarks file: %s", e)
                
                self._reverse_geocoder = CustomReverseGeocoder(cities, landmarks)
                logging.getLogger(__name__).info(
                    "[VISION] Place name index ready: %d exact tokens",
                    len(self._reverse_geocoder._place_names),
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("[VISION] Failed to load custom reverse geocoder: %s", e)
                self._reverse_geocoder = False
            return self._reverse_geocoder

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
        """Fallback dimensions when EXIF tags omit width/height.

        For RAW, prefer rawpy header read only (PIL often fails on ARW). On network
        drives this can still take seconds per file under parallel indexing.
        """
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        if ext in RAW_FILE_EXTENSIONS:
            try:
                import rawpy  # type: ignore

                with rawpy.imread(file_path) as raw:
                    w = int(raw.sizes.width)
                    h = int(raw.sizes.height)
                    if w > 0 and h > 0:
                        return w, h
            except Exception:
                pass
            return 0, 0

        try:
            with Image.open(file_path) as im:
                return int(im.width), int(im.height)
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
    @lru_cache(maxsize=1024)
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
                gps_lat, gps_lon, gps_raw, city, landmark, admin1, country, country_code, face_count, orientation, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                landmark=excluded.landmark,
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
                _coerce_float(meta.get("gps_lat")),
                _coerce_float(meta.get("gps_lon")),
                str(meta.get("gps_raw") or ""),
                str(meta.get("city") or ""),
                str(meta.get("landmark") or ""),
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

    def _extract_exif_brief(
        self,
        file_path: str,
        include_face: bool = False,
        file_stat: Optional[os.stat_result] = None,
        *,
        persist_exif: bool = True,
    ) -> Dict[str, object]:
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
            "landmark": "",
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
                    im = _load_index_source_image(file_path, max_size=1280, qt_decode=False)
                except Exception:
                    pass

            t_exif = time.time()
            # Stop after GPS block when possible; RAW uses exifread (header-only) in auto mode.
            tags = metadata_backend.process_file_from_path(
                file_path,
                details=False,
                stop_tag="GPS GPSLongitudeRef",
            )
            dur_exif = time.time() - t_exif
            dur_dims = 0.0
            dur_geo = 0.0
            dur_cache = 0.0
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
                "EXIF ImageWidth",
                "Image ImageWidth",
                "Image Width",
                "EXIF PixelXDimension",
                "MakerNote ImageWidth",
            )
            result["height"] = self._tag_int(
                tags,
                "EXIF ExifImageLength",
                "EXIF ImageLength",
                "Image ImageLength",
                "Image Height",
                "Image Length",
                "EXIF PixelYDimension",
                "MakerNote ImageHeight",
            )
            if not result["width"] or not result["height"]:
                t_dims = time.time()
                w, h = self._pil_dimensions(file_path)
                dur_dims = time.time() - t_dims
                result["width"] = int(result["width"] or w)
                result["height"] = int(result["height"] or h)
                
            # Robust orientation extraction logic
            orientation = 1
            for key in ("Exif.Image.Orientation", "Exif.Photo.Orientation", "Image Orientation", "EXIF Orientation"):
                val = tags.get(key)
                if val is not None:
                    try:
                        vals = getattr(val, "values", val)
                        if isinstance(vals, (list, tuple)) and vals:
                            o = vals[0]
                        else:
                            o = vals
                        if isinstance(o, int) and 1 <= o <= 8:
                            orientation = o
                            break
                    except Exception:
                        pass
                    
                    orientation_str = str(val).strip()
                    orientation_map = {
                        'Horizontal (normal)': 1,
                        'Mirrored horizontal': 2,
                        'Rotated 180': 3,
                        'Mirrored vertical': 4,
                        'Mirrored horizontal then rotated 90 CCW': 5,
                        'Rotated 90 CW': 6,
                        'Mirrored horizontal then rotated 90 CW': 7,
                        'Rotated 90 CCW': 8
                    }
                    if orientation_str in orientation_map:
                        orientation = orientation_map[orientation_str]
                        break
                    
                    try:
                        first = orientation_str.split()[0]
                        o = int(first)
                        if 1 <= o <= 8:
                            orientation = o
                            break
                    except Exception:
                        pass
            result["orientation"] = orientation
            
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
                    if isinstance(geo, CustomReverseGeocoder):
                        try:
                            t_geo = time.time()
                            # reverse_geocoder is not thread-safe; serialize lookups.
                            with self._rg_lock:
                                recs = geo.search(
                                    [(
                                        _coerce_float(result["gps_lat"]) or 0.0,
                                        _coerce_float(result["gps_lon"]) or 0.0,
                                    )],
                                    mode=1,
                                )
                            dur_geo = time.time() - t_geo
                            if recs:
                                rec = recs[0] or {}
                                city_name = str(rec.get("name", "") or "")
                                alternates = str(rec.get("alternates", "") or "")
                                result["city"] = f"{city_name}, {alternates}" if alternates else city_name
                                result["landmark"] = str(rec.get("landmark", "") or "")
                                result["admin1"] = str(rec.get("admin1", "") or "")
                                cc = str(rec.get("cc", "") or "").upper()
                                result["country_code"] = cc
                                result["country"] = self._country_name_from_code(cc)
                        except Exception:
                            pass
            if include_face:
                result["face_count"] = self._detect_face_count(file_path, preloaded_im=im)
            
            # Automatically backfill and populate the main gallery's PersistentEXIFCache/ImageCache!
            if persist_exif:
                try:
                    orientation = result.get("orientation") or 1

                    from image_cache import get_image_cache
                    from enhanced_raw_processor import RAW_EXIF_SENSOR_META_VER

                    t_cache = time.time()
                    cache = get_image_cache()
                    cache_dict = {
                        "orientation": int(orientation),
                        "camera_make": str(tags.get("Image Make") or ""),
                        "camera_model": str(result["camera_model"]),
                        "capture_time": str(result["capture_time"]),
                        "original_width": int(result["width"] or 0),
                        "original_height": int(result["height"] or 0),
                        "raw_exif_sensor_meta_ver": RAW_EXIF_SENSOR_META_VER,
                        "exif_data": {
                            "original_width": int(result["width"] or 0),
                            "original_height": int(result["height"] or 0),
                            "orientation": int(orientation),
                            "capture_time": str(result["capture_time"]),
                            "camera_make": str(tags.get("Image Make") or ""),
                            "camera_model": str(result["camera_model"]),
                            "lens_model": str(result["lens_model"]),
                            "iso": int(result["iso"] or 0),
                            "raw_exif_sensor_meta_ver": RAW_EXIF_SENSOR_META_VER,
                        }
                    }
                    # Skip the write when the shared cache already holds an
                    # equivalent, version-fresh row (e.g. populated moments earlier
                    # by gallery's bulk EXIF fetch or a prior sort/prefetch pass).
                    # A redundant INSERT OR REPLACE here holds the EXIF DB's single
                    # writer connection just as pointlessly as the per-item SQLite
                    # reads that turned out to be the real cause of gallery's
                    # multi-second metadata-rebuild stalls (see gallery_view.py's
                    # _metadata_exif_for_path) -- this is the write-side twin of
                    # that same contention, worth avoiding for the same reason.
                    existing = None
                    try:
                        existing = cache.get_exif(file_path)
                    except Exception:
                        existing = None
                    already_fresh = (
                        existing is not None
                        and not existing.get("minimal_preview_exif")
                        and existing.get("orientation") == cache_dict["orientation"]
                        and existing.get("camera_model") == cache_dict["camera_model"]
                        and existing.get("capture_time") == cache_dict["capture_time"]
                        and int(existing.get("original_width") or 0) == cache_dict["original_width"]
                        and int(existing.get("original_height") or 0) == cache_dict["original_height"]
                    )
                    if not already_fresh:
                        if file_stat is not None:
                            cache.put_exif(
                                file_path,
                                cache_dict,
                                file_size=int(file_stat.st_size),
                                file_mtime=float(file_stat.st_mtime),
                            )
                        else:
                            cache.put_exif(file_path, cache_dict)
                    dur_cache = time.time() - t_cache
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"[INDEX] Could not populate ImageCache for {file_path}: {e}")
                    dur_cache = 0.0
            else:
                dur_cache = 0.0
            total_dur = dur_exif + dur_dims + dur_geo + dur_cache
            if total_dur > 2.0:
                import logging
                logging.getLogger(__name__).warning(
                    "[INDEX] Slow metadata phases for %s: total=%.3fs exif=%.3fs dims=%.3fs geo=%.3fs cache_put=%.3fs",
                    os.path.basename(file_path),
                    total_dur,
                    dur_exif,
                    dur_dims,
                    dur_geo,
                    dur_cache,
                )
        except Exception:
            pass
        return result

    @staticmethod
    def _detect_face_count(file_path: str, preloaded_im: Optional[PilImage] = None) -> int:
        if sys.platform == "darwin":
            try:
                import Foundation  # type: ignore[import-not-found]
                import Vision  # type: ignore[import-not-found]
            except Exception:
                pass

        def _run_vision(path: str) -> Optional[int]:
            try:
                url = Foundation.NSURL.fileURLWithPath_(path)
                request = Vision.VNDetectFaceRectanglesRequest.alloc().init()
                handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, None)
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
                im = _load_index_source_image(
                    file_path, max_size=SemanticImageIndex.face_detection_max_edge(), qt_decode=False
                )
            
            if sys.platform != "darwin":
                # 1. Try OpenCV YuNet (Ultra-Lightweight, extremely fast, highly accurate)
                try:
                    import cv2
                    import numpy as np

                    img_bgr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
                    return _yunet_detect_faces_bgr(img_bgr)
                except Exception as yn_err:
                    import logging
                    logging.getLogger(__name__).debug(f"[VISION] OpenCV YuNet failed on {file_path}, trying OpenCV DNN: {yn_err}")

                # 2. Windows fallback using OpenCV DNN Face Detector
                try:
                    import cv2
                    import numpy as np
                    import os
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
                                    from ssl_certs import urlretrieve

                                    urlretrieve(
                                        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                                        prototxt_path,
                                    )
                                if not os.path.exists(caffemodel_path):
                                    import logging
                                    logging.getLogger(__name__).info("[VISION] Downloading DNN face detector weights...")
                                    from ssl_certs import urlretrieve

                                    urlretrieve(
                                        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                                        caffemodel_path,
                                    )
                                _FACE_DETECTOR_NET = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
                                _apply_opencv_dnn_acceleration(_FACE_DETECTOR_NET)
                    
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
                        if confidence > 0.85:  # 85% confidence threshold
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
        return self._mtime_matches(_coerce_float(row["file_mtime"]) or 0.0, st)

    def _row_semantic_ready(self, row: MetadataRow) -> bool:
        try:
            return _coerce_int(row["semantic_ready"]) == 1
        except Exception:
            # Backward compatibility for rows loaded before migration.
            return True

    @staticmethod
    def _row_semantic_skipped(row: sqlite3.Row) -> bool:
        """Permanent skip for this file revision (decode failed); do not block index completion."""
        try:
            return _coerce_int(row["semantic_ready"]) == -1
        except Exception:
            return False

    def _needs_reindex(self, file_path: str, st: os.stat_result) -> bool:
        rows = self._lookup_index_rows(file_path, st)
        if not rows:
            return True
        for row in rows:
            if self._row_matches_file(row, st) and (
                self._row_semantic_ready(row) or self._row_semantic_skipped(row)
            ):
                return False
        return True

    def _mark_semantic_skipped(
        self,
        canonical_fp: str,
        st: os.stat_result,
        conn: sqlite3.Connection,
    ) -> None:
        """Mark file as non-indexable for this model revision (won't retry until file changes)."""
        conn.execute(
            """
            UPDATE semantic_index
            SET semantic_ready = -1, dim = 0, embedding = ?, updated_at = ?
            WHERE file_path = ? AND model_name = ? AND file_size = ? AND mtime_ns = ?
            """,
            (
                b"",
                float(time.time()),
                canonical_fp,
                self.model_name,
                int(st.st_size),
                self._mtime_ns_from_stat(st),
            ),
        )

    def _encode_image(self, file_path: str) -> np.ndarray:
        if self.model_name.startswith("mobileclip-"):
            backend = self._mobileclip_backend or resolve_mobileclip_backend()
            err = backend.availability_error()
            if err:
                raise RuntimeError(f"MobileCLIP backend unavailable: {err}")
            return backend.encode_image(file_path)
        model = self._ensure_model()
        try:
            im = _load_index_source_image(file_path, max_size=1024, qt_decode=False)
            emb = model.encode(im, normalize_embeddings=True)
            return np.asarray(emb, dtype=np.float32)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot decode image for semantic index: {os.path.basename(file_path)}"
            ) from exc

    def _encode_images_best_effort(self, file_paths: Sequence[str]) -> List[np.ndarray]:
        """
        Encode multiple images. Falls back to per-image encoding when batch path
        is unavailable or unsupported by the model graph.
        """
        paths = [p for p in (file_paths or []) if p]
        if not paths:
            return []
        if self.model_name.startswith("mobileclip-"):
            backend = self._mobileclip_backend or resolve_mobileclip_backend()
            err = backend.availability_error()
            if err:
                raise RuntimeError(f"MobileCLIP backend unavailable: {err}")
            if hasattr(backend, "encode_images"):
                try:
                    return list(backend.encode_images(paths))
                except Exception:
                    pass
        return [self._encode_image(p) for p in paths]

    def _semantic_batch_cache_key(self, accel_detail: str = "") -> str:
        ep = ""
        try:
            if self.model_name.startswith("mobileclip-"):
                backend = self._mobileclip_backend or resolve_mobileclip_backend()
                self._mobileclip_backend = backend
                if isinstance(backend, MobileCLIPONNXBackend):
                    selected = resolve_onnx_execution_providers(
                        _ort_get_available_providers()
                    )
                    ep = ",".join(selected)
                else:
                    ep = backend.__class__.__name__
        except Exception:
            ep = accel_detail or "unknown"
        return (
            f"{sys.platform}|{self.model_name}|{ep}|{accel_detail or ''}"
            f"|{SEMANTIC_BATCH_TUNE_CACHE_VERSION}"
        )

    def _auto_select_semantic_batch_size(
        self,
        pending_for_semantic: Sequence[tuple[str, os.stat_result]],
        accel_detail: str = "",
        progress_callback: ProgressCallback = None,
        *,
        progress_album_total: int = 0,
        progress_indexed_base: int = 0,
    ) -> int:
        import logging

        logger = logging.getLogger(__name__)
        total = len(pending_for_semantic or [])
        if total <= 1:
            return 1

        is_coreml = False
        if self.model_name.startswith("mobileclip-"):
            backend = self._mobileclip_backend or resolve_mobileclip_backend()
            self._mobileclip_backend = backend
            is_coreml = isinstance(backend, MobileCLIPCoreMLBackend)
            if is_coreml:
                coreml_forced = semantic_coreml_chunk_forced()
                if coreml_forced is not None:
                    logger.info(
                        "[INDEX][SPEED] Core ML encode chunk (forced): %d",
                        coreml_forced,
                    )
                    return min(total, coreml_forced)
                if os.environ.get("RAWVIEWER_SEMANTIC_COREML_TUNE", "").strip().lower() not in ("1", "true", "yes", "on"):
                    if semantic_coreml_use_native_batch():
                        chunk = semantic_coreml_chunk_size()
                        logger.info(
                            "[INDEX][SPEED] Core ML encode chunk size: %d (native batch)",
                            chunk,
                        )
                    else:
                        chunk = semantic_coreml_prep_chunk_size()
                        logger.info(
                            "[INDEX][SPEED] Core ML prep chunk: %d (serial predict)",
                            chunk,
                        )
                    return min(total, chunk)
            elif not isinstance(backend, MobileCLIPONNXBackend):
                return 1

        forced = semantic_batch_size_forced()
        if forced is not None:
            return forced

        if not semantic_batch_auto_enabled():
            if is_coreml:
                if semantic_coreml_use_native_batch():
                    chunk = semantic_coreml_chunk_size()
                    logger.info(
                        "[INDEX][SPEED] Core ML encode chunk size: %d (auto-tune off, native batch)",
                        chunk,
                    )
                else:
                    chunk = semantic_coreml_prep_chunk_size()
                    logger.info(
                        "[INDEX][SPEED] Core ML prep chunk: %d (auto-tune off, serial predict)",
                        chunk,
                    )
                return min(total, chunk)
            return semantic_batch_size()

        candidates = (
            semantic_coreml_batch_candidates()
            if is_coreml
            else semantic_batch_candidates()
        )
        tune_label = "Core ML encode chunk" if is_coreml else "Semantic batch"

        cache_key = self._semantic_batch_cache_key(accel_detail)
        cache = _load_semantic_batch_cache()
        cached = cache.get(cache_key)
        if cached not in candidates and cache:
            # Upgrades: reuse v6 auto-tune picks when v7 key is not cached yet.
            legacy_key = cache_key.replace(
                f"|{SEMANTIC_BATCH_TUNE_CACHE_VERSION}",
                "|v6",
            )
            if legacy_key != cache_key:
                cached = cache.get(legacy_key)
        if cached in candidates:
            logger.info(
                "[INDEX][SPEED] %s auto cached: %d",
                tune_label,
                int(cached),
            )
            return int(cached)

        if is_coreml:
            sample_n = min(total, semantic_coreml_tune_sample_count())
        else:
            max_cand = max(candidates)
            sample_n = min(
                total,
                max(semantic_batch_tune_sample_count(), max_cand),
            )
        sample_paths = [cp for cp, _ in list(pending_for_semantic)[:sample_n]]
        best_batch = candidates[0] if candidates else 1
        best_tput = 0.0
        tie_ratio = semantic_batch_tie_ratio()
        early_stop_ratio = semantic_coreml_tune_early_stop_ratio() if is_coreml else 0.0
        logger.info(
            "[INDEX][SPEED] Auto-tuning %s on %d samples; candidates=%s (tie_ratio=%.2f)",
            tune_label.lower(),
            sample_n,
            candidates,
            tie_ratio,
        )
        if progress_callback:
            progress_callback(
                progress_indexed_base,
                progress_album_total or total,
                format_index_progress("Semantic", progress_indexed_base, progress_album_total or total)
                if progress_album_total > 0
                else f"Semantic: tuning 0/{len(candidates)}",
            )

        for ci, cand in enumerate(candidates):
            logger.info(
                "[INDEX][SPEED] Testing %s candidate %d/%d (size=%d)...",
                tune_label.lower(),
                ci + 1,
                len(candidates),
                cand,
            )
            if progress_callback:
                tune_msg = f"Semantic: tuning {ci + 1}/{len(candidates)}"
                if progress_album_total > 0:
                    progress_callback(
                        progress_indexed_base,
                        progress_album_total,
                        tune_msg,
                    )
                else:
                    progress_callback(ci + 1, len(candidates), tune_msg)
            t0 = time.time()
            ok = True
            try:
                for i in range(0, sample_n, cand):
                    chunk = sample_paths[i : i + cand]
                    self._encode_images_best_effort(chunk)
            except Exception as e:
                ok = False
                logger.info(
                    "[INDEX][SPEED] Batch candidate %d failed during auto-tune: %s",
                    cand,
                    e,
                )
            elapsed = max(1e-9, time.time() - t0)
            tput = (sample_n / elapsed) if ok else 0.0
            logger.info(
                "[INDEX][SPEED] Batch candidate %d -> %.2f img/s (%s, %.1fs)",
                cand,
                tput,
                "ok" if ok else "failed",
                elapsed,
            )
            if ok:
                if tput > best_tput:
                    best_tput = tput
                    best_batch = cand
                elif (
                    best_tput > 0
                    and tput >= best_tput
                    and tput >= best_tput * tie_ratio
                    and cand > best_batch
                ):
                    # True tie only: never prefer a larger batch when it is slower.
                    best_batch = cand
                elif (
                    is_coreml
                    and early_stop_ratio > 0
                    and best_tput > 0
                    and tput < best_tput * early_stop_ratio
                    and cand > best_batch
                ):
                    logger.info(
                        "[INDEX][SPEED] Core ML tune early stop at size=%d "
                        "(%.2f img/s vs best %d at %.2f img/s)",
                        cand,
                        tput,
                        best_batch,
                        best_tput,
                    )
                    break

        cache[cache_key] = int(best_batch)
        _save_semantic_batch_cache(cache)
        logger.info(
            "[INDEX][SPEED] Auto-selected %s size: %d (%.2f img/s on %d samples)",
            tune_label.lower(),
            best_batch,
            best_tput,
            sample_n,
        )
        return int(best_batch)

    def _encode_text(self, text: str) -> np.ndarray:
        if self.model_name.startswith("mobileclip-"):
            backend = self._mobileclip_backend or resolve_mobileclip_backend()
            err = backend.availability_error()
            if err:
                raise RuntimeError(f"MobileCLIP backend unavailable: {err}")
            return backend.encode_text(text)
        model = self._ensure_model()
        return np.asarray(model.encode(text, normalize_embeddings=True), dtype=np.float32)

    def _encode_semantic_query(self, text: str) -> List[np.ndarray]:
        """Encode semantic query with singular/plural and CLIP-style prompt variants."""
        variants = _semantic_text_variants(text)
        if not variants:
            return []
        return [self._encode_text(v) for v in variants]

    @staticmethod
    def _semantic_similarity(query_vecs: Sequence[np.ndarray], image_vec: np.ndarray) -> float:
        if not query_vecs:
            return 0.0
        return max(float(np.dot(qv, image_vec)) for qv in query_vecs)

    @staticmethod
    def _face_scan_worker_count(sample_path: Optional[str] = None) -> int:
        """Parallel image-load workers for face scan. YuNet inference is serialized."""
        raw = os.environ.get("RAWVIEWER_FACE_SCAN_WORKERS", "").strip()
        if raw:
            try:
                return max(1, min(16, int(raw)))
            except ValueError:
                pass

        from common_image_loader import is_external_or_network_volume
        if is_external_or_network_volume(sample_path):
            return 2
        try:
            from rawviewer_profile import classify_memory_tier, system_total_ram_gb

            tier = classify_memory_tier(system_total_ram_gb())
            if tier in ("low", "medium", "balanced"):
                return 2
        except Exception:
            pass
        cpu = os.cpu_count() or 4
        return min(4, max(2, cpu // 2))

    @staticmethod
    def _face_scan_inflight_chunk() -> int:
        raw = os.environ.get("RAWVIEWER_FACE_SCAN_CHUNK", "64").strip()
        try:
            return max(4, min(512, int(raw)))
        except ValueError:
            return 64

    @staticmethod
    def _face_scan_parallel_enabled() -> bool:
        v = os.environ.get("RAWVIEWER_FACE_SCAN_PARALLEL", "").strip().lower()
        if not v:
            # Nested thread pools + Qt/raw decode from worker threads crash on Windows.
            return sys.platform != "win32"
        return v not in ("0", "false", "no", "off")

    @staticmethod
    def _defer_face_scan_during_build() -> bool:
        """
        When true, build_index finishes after metadata + semantic embeddings.
        Face backfill runs in a second background pass so search becomes usable sooner.
        """
        v = os.environ.get("RAWVIEWER_INDEX_DEFER_FACE_SCAN", "1").strip().lower()
        return v not in ("0", "false", "no", "off")

    @staticmethod
    def _thumbnail_warm_before_face_scan() -> bool:
        # Default off: warming 6k+ RAWs before face scan blocks indexing for hours.
        v = os.environ.get("RAWVIEWER_FACE_SCAN_WARM_THUMBS", "0").strip().lower()
        return v not in ("0", "false", "no", "off")

    @staticmethod
    def _face_scan_warm_max_files() -> int:
        """
        Cap warm-up work so face indexing cannot appear frozen for giant albums.
        0 disables warm-up entirely; negative means unlimited.
        """
        raw = os.environ.get("RAWVIEWER_FACE_SCAN_WARM_MAX_FILES", "256").strip()
        if raw:
            try:
                return int(raw)
            except ValueError:
                pass
        return 256

    @staticmethod
    def _face_scan_warm_max_seconds() -> float:
        """Best-effort warm-up deadline; exceeded budget continues with face scan directly."""
        raw = os.environ.get("RAWVIEWER_FACE_SCAN_WARM_MAX_SECONDS", "25").strip()
        if raw:
            try:
                return max(0.0, float(raw))
            except ValueError:
                pass
        return 25.0

    @staticmethod
    def face_detection_max_edge() -> int:
        """Longest edge passed to YuNet/Vision (see _load_index_source_image in _detect_face_count)."""
        raw = os.environ.get("RAWVIEWER_FACE_DETECT_MAX_EDGE", "").strip()
        if raw:
            try:
                return max(320, min(2048, int(raw)))
            except ValueError:
                pass
        return 1280

    def _warm_thumbnail_cache_for_semantic_index(
        self,
        paths: List[str],
        progress_callback: ProgressCallback = None,
        *,
        progress_album_total: int = 0,
        progress_indexed_base: int = 0,
        stage_label: str = "Semantic",
    ) -> int:
        """Pre-extract thumbnails so MobileCLIP encode hits ImageCache instead of rawpy per file."""
        if not paths or not semantic_warm_thumbs_before_index():
            return 0
        return self._ensure_index_thumbnails_cached(
            paths,
            progress_callback,
            progress_album_total=progress_album_total,
            progress_indexed_base=progress_indexed_base,
            purpose="semantic",
        )

    def _ensure_index_thumbnails_cached(
        self,
        paths: List[str],
        progress_callback: ProgressCallback = None,
        *,
        progress_album_total: int = 0,
        progress_indexed_base: int = 0,
        purpose: str = "index",
        max_workers: Optional[int] = None,
    ) -> int:
        """
        Fill ImageCache for paths missing preview/grid/thumbnail tiers.
        Shared by semantic encoding and deferred face scan (same warmed pixels).
        """
        import logging
        from image_cache import get_image_cache

        logger = logging.getLogger(__name__)
        if not paths:
            return 0

        cache = get_image_cache()
        pending = [p for p in paths if _index_thumbnail_needs_warm(cache, p)]
        already_cached = len(paths) - len(pending)
        if not pending:
            logger.info(
                "[INDEX] %s thumbnail cache: all %d paths already in ImageCache",
                purpose,
                len(paths),
            )
            return 0
        if already_cached:
            logger.info(
                "[INDEX] %s thumbnail cache: %d/%d already in ImageCache; %d still need extract",
                purpose,
                already_cached,
                len(paths),
                len(pending),
            )

        if max_workers is None:
            sample_path = pending[0] if pending else None
            max_workers = semantic_encode_prep_workers(sample_path)
        workers = max(1, int(max_workers))
        t0 = time.time()
        warmed = 0

        def _warm_one(path: str) -> bool:
            try:
                self._wait_if_paused()
                from enhanced_raw_processor import (
                    ThumbnailExtractor,
                    extract_embedded_jpeg_by_scan,
                )
                from common_image_loader import (
                    finalize_index_thumbnail_array,
                    index_thumbnail_needs_orient_fix,
                    is_raw_file,
                )

                arr = _cached_index_source_array(cache, path)
                if arr is not None and not index_thumbnail_needs_orient_fix(
                    path, arr, cache=cache
                ):
                    return True

                if arr is not None and index_thumbnail_needs_orient_fix(
                    path, arr, cache=cache
                ):
                    arr = finalize_index_thumbnail_array(path, arr, cache=cache)
                else:
                    if is_raw_file(path):
                        arr = extract_embedded_jpeg_by_scan(path, 1024)
                        if arr is None:
                            arr = ThumbnailExtractor().extract_thumbnail_from_raw(
                                path, max_size=1024, allow_scan_fallback=False
                            )
                            if arr is not None:
                                arr = np.asarray(arr, dtype=np.uint8)
                        if arr is not None:
                            arr = finalize_index_thumbnail_array(
                                path, arr, cache=cache
                            )
                    else:
                        with Image.open(path) as im:
                            im = ImageOps.exif_transpose(im).convert("RGB")
                            im.thumbnail((1024, 1024), Image.Resampling.BICUBIC)
                            arr = np.asarray(im)

                if arr is not None:
                    jpeg = None
                    try:
                        jpeg = _thumbnail_jpeg_bytes(arr)
                    except Exception:
                        pass
                    cache.put_thumbnail(path, arr, jpeg if jpeg is not None else b"")
                    return True
            except Exception:
                pass
            return False

        logger.info(
            "[INDEX] Warming thumbnail cache for %s: %d/%d files (%d workers)",
            purpose,
            len(pending),
            len(paths),
            workers,
        )
        total = len(pending)
        warmed_base = progress_indexed_base + (len(paths) - len(pending)) * 0.1

        def _thumb_progress_ui_count(done: int) -> int:
            """User-visible thumbnail count (not album-weighted 10% slice)."""
            return min(total, max(0, int(done)))

        def _thumb_progress_message(done: int) -> str:
            return format_index_progress(
                "Preparing thumbnails", _thumb_progress_ui_count(done), total
            )

        def _emit_thumb_progress(done: int) -> None:
            if not progress_callback:
                return
            ui_done = _thumb_progress_ui_count(done)
            if progress_album_total > 0:
                progress_callback(
                    int(warmed_base + ui_done * 0.1),
                    progress_album_total,
                    _thumb_progress_message(done),
                )
            else:
                progress_callback(ui_done, total, _thumb_progress_message(done))

        def _should_emit_thumb_progress(done: int) -> bool:
            if done <= 3 or done >= total:
                return True
            # ~1% steps, at least every 10 files (backend can finish >>10/s).
            step = max(10, total // 100)
            return done % step == 0

        if workers <= 1 or total < 4:
            for i, path in enumerate(pending, start=1):
                self._wait_if_paused()
                if _warm_one(path):
                    warmed += 1
                if _should_emit_thumb_progress(i):
                    _emit_thumb_progress(i)
        else:
            completed = 0
            executor = _get_shared_index_executor()
            futures = {executor.submit(_warm_one, p): p for p in pending}
            try:
                for future in concurrent.futures.as_completed(futures):
                    self._wait_if_paused()
                    completed += 1
                    if future.result():
                        warmed += 1
                    if _should_emit_thumb_progress(completed):
                        _emit_thumb_progress(completed)
            except IndexAborted:
                for pending_future in futures:
                    pending_future.cancel()
                raise
            # Removed executor.shutdown() since it's shared
        logger.info(
            "[INDEX][SPEED] %s thumbnail warm-up: warmed=%d pending=%d already_cached=%d "
            "elapsed=%.3fs throughput=%.2f img/s",
            purpose,
            warmed,
            len(pending),
            already_cached,
            time.time() - t0,
            float(len(pending)) / max(1e-9, time.time() - t0),
        )
        return warmed

    def _face_pending_paths(
        self, conn: sqlite3.Connection, canonical_paths: Sequence[str]
    ) -> List[str]:
        """Paths in this batch that still need face_count in the DB."""
        if not canonical_paths:
            return []
        pending: List[str] = []
        chunk_size = 900
        unique = list(dict.fromkeys(canonical_paths))
        for i in range(0, len(unique), chunk_size):
            chunk = unique[i : i + chunk_size]
            qs = ",".join(["?"] * len(chunk))
            cursor = conn.execute(
                f"SELECT file_path FROM semantic_index WHERE file_path IN ({qs}) AND face_count IS NULL AND model_name = ?",
                [*chunk, self.model_name],
            )
            pending.extend(row[0] for row in cursor.fetchall())
        return pending

    def get_face_pending_count(self, file_paths: Sequence[str]) -> int:
        if not face_scan_enabled():
            return 0
        if not file_paths:
            return 0
        indexable_paths, _ = self._filter_duplicate_raw_companions(file_paths)
        canonical = [self._canonical_path(p) for p in indexable_paths if p]
        if not canonical:
            return 0
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            try:
                pending = set(self._face_pending_paths(conn, canonical))
                # Rows not in the DB yet also need a face scan pass.
                chunk_size = 900
                indexed_paths: set[str] = set()
                for i in range(0, len(canonical), chunk_size):
                    chunk = canonical[i : i + chunk_size]
                    qs = ",".join(["?"] * len(chunk))
                    cursor = conn.execute(
                        f"SELECT file_path FROM semantic_index WHERE file_path IN ({qs})",
                        chunk,
                    )
                    indexed_paths.update(row[0] for row in cursor.fetchall())
                for cp in canonical:
                    if cp not in indexed_paths:
                        pending.add(cp)
                return len(pending)
            finally:
                conn.close()
        except Exception:
            return 0

    def _enqueue_metadata_backfill(
        self,
        file_paths: Sequence[str],
        *,
        geocode_paths: Optional[Sequence[str]] = None,
    ) -> None:
        """Queue missing metadata / geocode repair for a background worker."""
        if not metadata_auto_index_enabled():
            return
        with self._metadata_backfill_lock:
            for p in file_paths or []:
                if p:
                    self._metadata_backfill_paths.add(p)
            for p in geocode_paths or []:
                if p:
                    self._metadata_backfill_geocode.add(p)
            if self._metadata_backfill_running:
                return
            if not self._metadata_backfill_paths and not self._metadata_backfill_geocode:
                return
            self._metadata_backfill_running = True
            thread = threading.Thread(
                target=self._run_metadata_backfill_worker,
                name="rawviewer-metadata-backfill",
                daemon=True,
            )
            thread.start()

    def _run_metadata_backfill_worker(self) -> None:
        import logging

        logger = logging.getLogger(__name__)
        try:
            while True:
                with self._metadata_backfill_lock:
                    extract_paths = list(self._metadata_backfill_paths)
                    geocode_paths = list(self._metadata_backfill_geocode)
                    self._metadata_backfill_paths.clear()
                    self._metadata_backfill_geocode.clear()
                    if not extract_paths and not geocode_paths:
                        self._metadata_backfill_running = False
                        return

                conn = sqlite3.connect(self.db_path, timeout=60.0)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=60000")
                exif_batch: List[tuple] = []
                commit_every = 40
                batch_writes = 0
                try:
                    if geocode_paths:
                        geo = self._ensure_reverse_geocoder()
                        if isinstance(geo, CustomReverseGeocoder):
                            for fp in geocode_paths:
                                try:
                                    canonical = self._canonical_path(fp)
                                    row = conn.execute(
                                        "SELECT gps_lat, gps_lon, city FROM semantic_index WHERE file_path = ?",
                                        (canonical,),
                                    ).fetchone()
                                    if not row or row[2]:
                                        continue
                                    lat, lon = row[0], row[1]
                                    if lat is None or lon is None:
                                        continue
                                    with self._rg_lock:
                                        recs = geo.search(
                                            [(_coerce_float(lat) or 0.0, _coerce_float(lon) or 0.0)],
                                            mode=1,
                                        )
                                    if not recs:
                                        continue
                                    rec = recs[0] or {}
                                    city_name = str(rec.get("name", "") or "")
                                    alternates = str(rec.get("alternates", "") or "")
                                    city = f"{city_name}, {alternates}" if alternates else city_name
                                    landmark = str(rec.get("landmark", "") or "")
                                    admin1 = str(rec.get("admin1", "") or "")
                                    cc = str(rec.get("cc", "") or "").upper()
                                    country = self._country_name_from_code(cc)
                                    conn.execute(
                                        "UPDATE semantic_index SET city=?, landmark=?, admin1=?, country=?, country_code=? WHERE file_path=?",
                                        (city, landmark, admin1, country, cc, canonical),
                                    )
                                    batch_writes += 1
                                    if batch_writes >= commit_every:
                                        conn.commit()
                                        batch_writes = 0
                                except Exception:
                                    continue

                    for fp in extract_paths:
                        try:
                            canonical = self._canonical_path(fp)
                            if not os.path.isfile(canonical):
                                continue
                            st = os.stat(canonical)
                            meta = self._extract_exif_brief(
                                canonical, include_face=False, file_stat=st, persist_exif=False
                            )
                            if meta:
                                self._upsert_metadata(canonical, st, meta, conn=conn)
                                batch_writes += 1
                                exif_batch.append((canonical, meta, st))
                            if batch_writes >= commit_every:
                                conn.commit()
                                batch_writes = 0
                                self._flush_exif_backfill_batch(exif_batch)
                                exif_batch.clear()
                        except Exception:
                            continue

                    if batch_writes > 0:
                        conn.commit()
                    self._flush_exif_backfill_batch(exif_batch)
                finally:
                    conn.close()
        except Exception as exc:
            logger.warning("[INDEX] Metadata backfill worker failed: %s", exc)
            with self._metadata_backfill_lock:
                self._metadata_backfill_running = False

    @staticmethod
    def _flush_exif_backfill_batch(exif_batch: List[tuple]) -> None:
        if not exif_batch:
            return
        try:
            from enhanced_raw_processor import RAW_EXIF_SENSOR_META_VER
            from image_cache import get_image_cache

            cache = get_image_cache()
            items = []
            for canonical, meta, st in exif_batch:
                items.append(
                    (
                        canonical,
                        {
                            "orientation": int(meta.get("orientation") or 1),
                            "camera_make": "",
                            "camera_model": str(meta.get("camera_model") or ""),
                            "capture_time": str(meta.get("capture_time") or ""),
                            "original_width": int(meta.get("width") or 0),
                            "original_height": int(meta.get("height") or 0),
                            "raw_exif_sensor_meta_ver": RAW_EXIF_SENSOR_META_VER,
                            "exif_data": {
                                "capture_time": str(meta.get("capture_time") or ""),
                                "camera_model": str(meta.get("camera_model") or ""),
                                "lens_model": str(meta.get("lens_model") or ""),
                                "iso": int(meta.get("iso") or 0),
                                "raw_exif_sensor_meta_ver": RAW_EXIF_SENSOR_META_VER,
                            },
                        },
                        int(st.st_size),
                        float(st.st_mtime),
                    )
                )
            put_many = getattr(cache.exif_cache, "put_many", None)
            if callable(put_many):
                put_many(items, commit=True)
        except Exception:
            pass

    def backfill_search_metadata(
        self,
        file_paths: Sequence[str],
        progress_callback: ProgressCallback = None,
    ) -> int:
        """Background metadata extraction + geocode repair (search/index gap fill)."""
        if not file_paths:
            return 0
        self._enqueue_metadata_backfill(list(file_paths))
        return len(file_paths)

    def backfill_face_counts(
        self,
        file_paths: Sequence[str],
        progress_callback: ProgressCallback = None,
        *,
        album_total: int = 0,
        album_indexed_base: int = 0,
    ) -> int:
        """Deferred face-only pass (after semantic index is ready for search)."""
        import sqlite3

        if not face_scan_enabled():
            return 0
        if not file_paths:
            return 0
        self.release_mobileclip_gpu_memory()
        canonical_map = {self._canonical_path(p): p for p in file_paths if p}
        unique_canonical = list(canonical_map.keys())
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=60000")
        try:
            face_pending = self._face_pending_paths(conn, unique_canonical)
            if face_pending:
                if progress_callback:
                    progress_callback(
                        album_indexed_base,
                        album_total or len(file_paths),
                        format_index_progress("Face", 0, len(face_pending)),
                    )
                self._run_parallel_face_scan(
                    face_pending,
                    conn,
                    progress_callback,
                    commit_every=40,
                    progress_album_total=album_total or len(file_paths),
                    progress_indexed_base=album_indexed_base,
                )
            return len(face_pending)
        finally:
            conn.close()

    def _run_parallel_face_scan(
        self,
        face_pending: List[str],
        conn: sqlite3.Connection,
        progress_callback: ProgressCallback,
        *,
        conn_lock: Optional[threading.Lock] = None,
        commit_every: int = 40,
        progress_album_total: int = 0,
        progress_indexed_base: int = 0,
    ) -> None:
        import logging

        logger = logging.getLogger(__name__)
        total_face = len(face_pending)
        if total_face <= 0:
            return

        max_edge = self.face_detection_max_edge()
        cache_misses: List[str] = []
        try:
            from image_cache import get_image_cache

            cache = get_image_cache()
            cached_n = 0
            for p in face_pending:
                if _index_source_thumbnail_present(cache, p):
                    cached_n += 1
                else:
                    cache_misses.append(p)
            logger.info(
                "[INDEX] Face scan ImageCache ready: %d/%d (preview/grid/thumbnail or disk)",
                cached_n,
                total_face,
            )
        except Exception:
            cache_misses = list(face_pending)

        if cache_misses:
            logger.info(
                "[INDEX] Face scan filling %d missing thumbnails (semantic pass should have cached most)",
                len(cache_misses),
            )
            sample_path = cache_misses[0] if cache_misses else None
            face_warm_workers = (
                1
                if sys.platform == "win32"
                else min(semantic_encode_prep_workers(sample_path), 4)
            )
            self._ensure_index_thumbnails_cached(
                cache_misses,
                progress_callback,
                progress_album_total=progress_album_total,
                progress_indexed_base=progress_indexed_base,
                purpose="face",
                max_workers=face_warm_workers,
            )
        else:
            logger.info(
                "[INDEX] Face scan reusing semantic thumbnails for all %d files",
                total_face,
            )

        _yunet_warmup()

        sample_path = face_pending[0] if face_pending else None
        workers = self._face_scan_worker_count(sample_path)
        chunk_size = self._face_scan_inflight_chunk()
        t_face_start = time.time()
        batch_writes = 0
        lock = conn_lock or threading.Lock()
        parallel = self._face_scan_parallel_enabled() and total_face >= 4
        try:
            import cv2

            backend_id, target_id = resolve_opencv_dnn_backend_target()
            backend_name = _dnn_get_backend_name(cv2, backend_id)
            logger.info(
                "[INDEX] Face scan config: backend=%s target=%s workers=%d chunk=%d parallel=%s",
                backend_name,
                target_id,
                workers,
                chunk_size,
                parallel,
            )
        except Exception:
            pass

        def _scan_one(cp: str) -> tuple:
            try:
                im = _load_face_scan_image(cp, max_size=max_edge)
                if im is None:
                    logger.warning(
                        "[INDEX] Face scan cache miss (no warmed thumbnail): %s",
                        os.path.basename(cp),
                    )
                    return cp, 0
                return cp, int(self._detect_face_count(cp, preloaded_im=im) or 0)
            except Exception as e:
                logger.error(
                    "[INDEX] Face scanning failed for %s: %s",
                    os.path.basename(cp),
                    e,
                )
                return cp, 0

        if not parallel:
            for idx, cp in enumerate(face_pending, start=1):
                if progress_callback and (
                    idx <= 2 or idx >= total_face or idx % 10 == 0
                ):
                    progress_callback(
                        progress_indexed_base,
                        progress_album_total,
                        format_index_progress("Face", idx, total_face),
                    )
                cp, face_count = _scan_one(cp)
                with lock:
                    self._store_face_count(cp, face_count, conn=conn, commit=False)
                    batch_writes += 1
                    if batch_writes >= commit_every:
                        conn.commit()
                        batch_writes = 0
            if batch_writes > 0:
                with lock:
                    conn.commit()
            dur = time.time() - t_face_start
            logger.info(
                "[INDEX] Face scan (sequential): %d files in %.2fs (%.0f ms/img)",
                total_face,
                dur,
                (dur / total_face) * 1000.0 if total_face else 0,
            )
            return

        logger.info(
            "[INDEX] Parallel face scan: %d files, %d workers",
            total_face,
            workers,
        )
        completed = 0
        executor = _get_shared_index_executor()
        for chunk_start in range(0, total_face, chunk_size):
            chunk = face_pending[chunk_start : chunk_start + chunk_size]
            futures = [executor.submit(_scan_one, cp) for cp in chunk]
            for future in concurrent.futures.as_completed(futures):
                cp, face_count = future.result()
                with lock:
                    self._store_face_count(cp, face_count, conn=conn, commit=False)
                    batch_writes += 1
                    if batch_writes >= commit_every:
                        conn.commit()
                        batch_writes = 0
                completed += 1
                if progress_callback and (
                    completed <= 2
                    or completed >= total_face
                    or completed % 10 == 0
                ):
                    progress_callback(
                        progress_indexed_base,
                        progress_album_total,
                        format_index_progress("Face", completed, total_face),
                    )

        if batch_writes > 0:
            with lock:
                conn.commit()

        dur = time.time() - t_face_start
        logger.info(
            "[INDEX] Face scan (parallel, %d workers): %d files in %.2fs (%.0f ms/img)",
            workers,
            total_face,
            dur,
            (dur / total_face) * 1000.0 if total_face else 0,
        )

    @staticmethod
    def _raw_companion_key(fp: str) -> str:
        base = os.path.basename(fp)
        parts = base.split(".")
        if len(parts) > 1:
            for idx, part in enumerate(parts):
                if part.startswith("RAW-"):
                    return ".".join(parts[:idx]).lower()
            return ".".join(parts[:-1]).lower()
        return base.lower()

    @classmethod
    def _filter_duplicate_raw_companions(
        cls, file_paths: Sequence[str]
    ) -> tuple[List[str], List[str]]:
        """Return (indexable paths, skipped RAW paths with a non-raw companion)."""
        filtered_paths: List[str] = []
        skipped_raw: List[str] = []
        non_raw_keys: set = set()
        for fp in file_paths:
            if not fp:
                continue
            ext = os.path.splitext(fp)[1].lower().lstrip(".")
            if ext and ext not in RAW_FILE_EXTENSIONS:
                non_raw_keys.add((os.path.dirname(fp), cls._raw_companion_key(fp)))
        for fp in file_paths:
            if not fp:
                continue
            ext = os.path.splitext(fp)[1].lower().lstrip(".")
            if ext in RAW_FILE_EXTENSIONS:
                if (os.path.dirname(fp), cls._raw_companion_key(fp)) in non_raw_keys:
                    skipped_raw.append(fp)
                    continue
            filtered_paths.append(fp)
        return filtered_paths, skipped_raw

    def build_index(
        self,
        file_paths: Sequence[str],
        progress_callback: ProgressCallback = None,
        *,
        album_total: int = 0,
        album_indexed_base: int = 0,
        run_face_scan: Optional[bool] = None,
        run_semantic_embeddings: Optional[bool] = None,
    ) -> Dict[str, int]:
        import logging
        import sys
        import sqlite3
        logger = logging.getLogger(__name__)
        t_start = time.time()
        if run_semantic_embeddings is None:
            run_semantic_embeddings = semantic_embeddings_enabled()
        logger.info(f"[INDEX] Starting indexing of {len(file_paths)} file paths.")
        if run_semantic_embeddings:
            log_inference_acceleration_profile()
        if sys.platform != "darwin":
            logger.info("[VISION] Using OpenCV offline face scanner on Windows.")

        filtered_paths, skipped_raw_paths = self._filter_duplicate_raw_companions(file_paths)
        skipped_companions = len(skipped_raw_paths)
        for fp in skipped_raw_paths:
            logger.info(
                "[INDEX] Skipping RAW companion file to avoid duplicate results: %s",
                os.path.basename(fp),
            )
        logger.info(
            "[INDEX] Filtered out %d RAW companion files. Actual files to evaluate: %d",
            skipped_companions,
            len(filtered_paths),
        )
            
        total = len(file_paths)
        indexed = 0
        skipped = skipped_companions
        failed = 0
        pending_for_semantic: List[tuple[str, os.stat_result]] = []
        progress_album_total = album_total if album_total > 0 else total
        progress_indexed_base = max(0, album_indexed_base)
        
        # Create a thread-local, dedicated connection for this background worker thread!
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=60000")

        if skipped_raw_paths:
            for fp in skipped_raw_paths:
                try:
                    cp = self._canonical_path(fp)
                    st = os.stat(cp)
                    self._mark_semantic_skipped(cp, st, conn)
                except OSError:
                    pass
            conn.commit()

        total_face = 0
        run_face_inline = False
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
                    f"SELECT file_path, file_mtime, file_size, semantic_ready, dim, gps_lat, city FROM semantic_index WHERE file_path IN ({qs})",
                    chunk
                )
                for row in cursor.fetchall():
                    existing_meta[row[0]] = {
                        "mtime": row[1],
                        "size": row[2],
                        "semantic_ready": row[3],
                        "dim": row[4],
                        "gps_lat": row[5],
                        "city": row[6]
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
                        sr = int(row["semantic_ready"] or 0)
                        dim = _coerce_int(self._row_value(row, "dim"))
                        needs_semantic = (sr == 0) or (
                            bool(run_semantic_embeddings) and sr == 1 and dim == 0
                        )
                        if needs_semantic:
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
                self._acquire_load_throttle()
                if progress_callback:
                    progress_callback(
                        progress_indexed_base,
                        progress_album_total,
                        format_index_progress("Metadata", progress_indexed_base, progress_album_total),
                    )
                t_meta_start = time.time()
                from common_image_loader import index_metadata_worker_count

                sample_fp = to_extract[0][0] if to_extract else None
                max_workers = index_metadata_worker_count(total_extract, sample_fp)
                logger.info(
                    "[INDEX] Metadata workers=%d for %d files",
                    max_workers,
                    total_extract,
                )
                executor = _get_shared_index_executor()
                try:
                    def extract_task(item):
                        self._wait_if_paused()
                        cp, st = item
                        try:
                            t_single_start = time.time()
                            meta = self._extract_exif_brief(
                                cp, include_face=False, file_stat=st
                            )
                            t_single_dur = time.time() - t_single_start
                            if t_single_dur > 2.0:
                                logger.warning(
                                    "[INDEX] Slow metadata extraction for %s: %.3fs "
                                    "(often EXIF cache lock vs folder sort refinement)",
                                    os.path.basename(cp),
                                    t_single_dur,
                                )
                            return cp, st, meta
                        except Exception as e:
                            logger.error(f"[INDEX] Failed to extract EXIF for {os.path.basename(cp)}: {e}")
                            return cp, st, None

                    futures = [executor.submit(extract_task, item) for item in to_extract]
                    try:
                        for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                            self._wait_if_paused()
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

                            if i % 100 == 0:
                                import gc
                                gc.collect()

                            if progress_callback and (i <= 2 or i >= total_extract or i % 10 == 0):
                                progress_callback(
                                    progress_indexed_base,
                                    progress_album_total,
                                    format_index_progress(
                                        "Metadata", progress_indexed_base + i, progress_album_total
                                    ),
                                )
                    except IndexAborted:
                        for pending_future in futures:
                            pending_future.cancel()
                        raise
                except IndexAborted:
                    raise

                if batch_writes > 0:
                    conn.commit()
                    batch_writes = 0
                logger.info(f"[INDEX] Completed metadata extraction for {total_extract} files in {time.time() - t_meta_start:.4f}s.")
                try:
                    from image_cache import get_image_cache
                    get_image_cache().flush_exif_disk_cache()
                except Exception:
                    pass

            face_pending = self._face_pending_paths(conn, unique_canonical) if face_scan_enabled() else []
            total_face = len(face_pending)
            total_sem = len(pending_for_semantic)
            if run_face_scan is None:
                run_face_scan = face_scan_enabled() and not self._defer_face_scan_during_build()
            run_face_inline = total_face > 0 and run_face_scan

            if total_face > 0 and not run_face_inline:
                logger.info(
                    "[INDEX] Deferring face scan for %d files until after semantic pass",
                    total_face,
                )

            # Phase 2: semantic embeddings (search-critical). Run before face scan.
            if total_sem > 0:
                self._wait_if_paused()
                if run_semantic_embeddings:
                    # Heavy GPU/RAM + thumbnail decode work: throttle ILM now (not
                    # at index start) so light/paused phases never starve the gallery.
                    self._acquire_load_throttle()
                    logger.info(f"[INDEX] Starting AI features neural pass (MobileCLIP) for {total_sem} files...")
                    if progress_callback:
                        progress_callback(
                            progress_indexed_base,
                            progress_album_total,
                            format_index_progress("Semantic", progress_indexed_base, progress_album_total),
                        )
                    sem_gpu_accel = False
                    sem_accel_detail = "unknown"
                    try:
                        if self.model_name.startswith("mobileclip-"):
                            backend = self._mobileclip_backend or resolve_mobileclip_backend()
                            self._mobileclip_backend = backend
                            if isinstance(backend, MobileCLIPONNXBackend):
                                try:
                                    selected = resolve_onnx_execution_providers(
                                        _ort_get_available_providers()
                                    )
                                except Exception:
                                    selected = ["CPUExecutionProvider"]
                                sem_gpu_accel = any(
                                    str(p) != "CPUExecutionProvider" for p in selected
                                )
                                sem_accel_detail = f"ONNX providers=[{', '.join(selected)}]"
                            elif isinstance(backend, MobileCLIPCoreMLBackend):
                                cu = os.environ.get(
                                    "RAWVIEWER_COREML_COMPUTE_UNITS", "all"
                                ).strip().lower()
                                sem_gpu_accel = cu not in ("cpu", "cpuonly")
                                sem_accel_detail = f"CoreML compute_units={cu}"
                            else:
                                sem_accel_detail = backend.__class__.__name__
                        else:
                            sem_accel_detail = f"model={self.model_name}"
                    except Exception as accel_exc:
                        sem_accel_detail = f"unavailable ({accel_exc})"
                    logger.info(
                        "[INDEX][ACCEL] Semantic acceleration: gpu=%s (%s)",
                        "ON" if sem_gpu_accel else "OFF",
                        sem_accel_detail,
                    )
                    try:
                        if self.model_name.startswith("mobileclip-"):
                            backend = self._mobileclip_backend or resolve_mobileclip_backend()
                            self._mobileclip_backend = backend
                            if isinstance(backend, MobileCLIPONNXBackend):
                                img_session, txt_session = backend._require_sessions()
                                img_providers = img_session.get_providers()
                                txt_providers = txt_session.get_providers()
                                logger.info(
                                    "[INDEX][ACCEL] ONNX active session providers: image=%s text=%s",
                                    img_providers,
                                    txt_providers,
                                )
                    except Exception as provider_exc:
                        logger.debug(
                            "[INDEX][ACCEL] Could not read active ONNX session providers: %s",
                            provider_exc,
                        )
                    warm_paths = [cp for cp, _ in pending_for_semantic]
                    t_warm_start = time.time()
                    warm_count = self._warm_thumbnail_cache_for_semantic_index(
                        warm_paths,
                        progress_callback,
                        progress_album_total=progress_album_total,
                        progress_indexed_base=progress_indexed_base,
                    )
                    t_warm_elapsed = max(0.0, time.time() - t_warm_start)
                    throttle_sec = semantic_gpu_throttle_seconds()
                    batch_size = self._auto_select_semantic_batch_size(
                        pending_for_semantic,
                        sem_accel_detail,
                        progress_callback,
                        progress_album_total=progress_album_total,
                        progress_indexed_base=progress_indexed_base,
                    )
                    logger.info(
                        "[INDEX][SPEED] Semantic batch size: %d",
                        batch_size,
                    )
                    if throttle_sec > 0:
                        logger.info(
                            "[INDEX][SPEED] Semantic throttle enabled: %.0f ms/image",
                            throttle_sec * 1000.0,
                        )
                    t_sem_start = time.time()
                    sem_success = 0
                    sem_fail = 0
                    i = 0
                    t_encode_cpu = 0.0
                    for batch_start in range(0, total_sem, batch_size):
                        self._wait_if_paused()
                        batch_items = pending_for_semantic[batch_start : batch_start + batch_size]
                        batch_paths = [cp for cp, _ in batch_items]
                        try:
                            t_batch_neural = time.time()
                            vecs = self._encode_images_best_effort(batch_paths)
                            t_batch_dur = time.time() - t_batch_neural
                            t_encode_cpu += t_batch_dur
                            if t_batch_dur > 0.5:
                                logger.info(
                                    "[INDEX] MobileCLIP encoding batch=%d took %.4fs",
                                    len(batch_items),
                                    t_batch_dur,
                                )
                            for (canonical_fp, st), vec in zip(batch_items, vecs):
                                i += 1
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
                                sem_success += 1
                                batch_writes += 1
                                if batch_writes >= commit_every:
                                    conn.commit()
                                    batch_writes = 0
                                if throttle_sec > 0:
                                    time.sleep(throttle_sec)

                            if progress_callback:
                                progress_callback(
                                    int(progress_indexed_base + total_sem * 0.1 + i * 0.9),
                                    progress_album_total,
                                    format_index_progress("Semantic", int(progress_indexed_base + total_sem * 0.1 + i * 0.9), progress_album_total),
                                )
                        except Exception as batch_exc:
                            logger.warning(
                                "[INDEX] Batch semantic encode failed (size=%d): %s. Falling back to per-image.",
                                len(batch_items),
                                batch_exc,
                            )
                            for canonical_fp, st in batch_items:
                                i += 1
                                try:
                                    t_single_neural = time.time()
                                    vec = self._encode_image(canonical_fp)
                                    t_neural_dur = time.time() - t_single_neural
                                    t_encode_cpu += t_neural_dur
                                    if t_neural_dur > 0.5:
                                        logger.info(
                                            f"[INDEX] MobileCLIP encoding for {os.path.basename(canonical_fp)} took {t_neural_dur:.4f}s"
                                        )
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
                                    sem_success += 1
                                    batch_writes += 1
                                    if batch_writes >= commit_every:
                                        conn.commit()
                                        batch_writes = 0
                                except Exception as e:
                                    logger.warning(
                                        "[INDEX] Skipping semantic embedding for %s: %s",
                                        os.path.basename(canonical_fp),
                                        e,
                                    )
                                    failed += 1
                                    sem_fail += 1
                                    try:
                                        self._mark_semantic_skipped(canonical_fp, st, conn)
                                        self._store_face_count(
                                            canonical_fp, 0, conn=conn, commit=False
                                        )
                                        batch_writes += 1
                                        if batch_writes >= commit_every:
                                            conn.commit()
                                            batch_writes = 0
                                    except Exception as mark_exc:
                                        logger.debug(
                                            "[INDEX] Could not mark skipped for %s: %s",
                                            os.path.basename(canonical_fp),
                                            mark_exc,
                                        )
                                if progress_callback and (
                                    i <= 2 or i >= total_sem or (i % 5 == 0)
                                ):
                                    progress_callback(
                                        int(progress_indexed_base + total_sem * 0.1 + i * 0.9),
                                        progress_album_total,
                                        format_index_progress("Semantic", int(progress_indexed_base + total_sem * 0.1 + i * 0.9), progress_album_total),
                                    )
                                if throttle_sec > 0:
                                    time.sleep(throttle_sec)

                        if i % 25 == 0 or i == total_sem:
                            elapsed_so_far = max(1e-9, time.time() - t_sem_start)
                            throughput_so_far = float(i) / elapsed_so_far
                            logger.info(
                                "[INDEX][SPEED] Semantic progress: %d/%d in %.3fs (%.2f img/s)",
                                i,
                                total_sem,
                                elapsed_so_far,
                                throughput_so_far,
                            )
                        import gc
                        gc.collect()
                    if batch_writes:
                        conn.commit()
                        batch_writes = 0
                    sem_elapsed = max(1e-9, time.time() - t_sem_start)
                    sem_avg = sem_elapsed / max(1, total_sem)
                    sem_throughput = float(total_sem) / sem_elapsed
                    encode_elapsed = max(1e-9, t_encode_cpu)
                    warm_throughput = (
                        float(len(warm_paths)) / max(1e-9, t_warm_elapsed)
                        if warm_paths
                        else 0.0
                    )
                    encode_throughput = float(total_sem) / encode_elapsed
                    wall_phases = max(1e-9, t_warm_elapsed + sem_elapsed)
                    logger.info(f"[INDEX] Completed AI neural pass in {sem_elapsed:.4f}s.")
                    logger.info(
                        "[INDEX][SPEED] Semantic pass stats: total=%d success=%d failed=%d elapsed=%.3fs avg=%.4fs/img throughput=%.2f img/s",
                        total_sem,
                        sem_success,
                        sem_fail,
                        sem_elapsed,
                        sem_avg,
                        sem_throughput,
                    )
                    logger.info(
                        "[INDEX][SPEED] Phase split: warm=%.3fs (%.2f img/s, warmed=%d) "
                        "encode=%.3fs (%.2f img/s) encode_wall=%.3fs warm_share=%.0f%%",
                        t_warm_elapsed,
                        warm_throughput,
                        int(warm_count),
                        encode_elapsed,
                        encode_throughput,
                        sem_elapsed,
                        100.0 * t_warm_elapsed / wall_phases,
                    )
                else:
                    logger.info(f"[INDEX] Skipping AI features neural pass (MobileCLIP) for {total_sem} files (metadata-only index).")
                    
                    # Thumbnail warm-up runs once in the user semantic pass (before MobileCLIP).
                    # Skip here when semantic search is enabled to avoid duplicate work and a
                    # second misleading "Semantic" progress phase in the search field.
                    if semantic_embeddings_enabled():
                        pass
                    else:
                        # Skip background thumbnail warming for metadata-only index to avoid
                        # disk/CPU contention and locking up the gallery image loader.
                        pass

                    for canonical_fp, st in pending_for_semantic:
                        self._wait_if_paused()
                        self._mark_metadata_ready_without_embedding(canonical_fp, st, conn)
                        indexed += 1
                    conn.commit()

            # Phase 3: face scan (optional; skipped when deferred to a follow-up worker).
            if run_face_inline:
                # Face detection decodes images at scale — heavy, so throttle here.
                self._acquire_load_throttle()
                self.release_mobileclip_gpu_memory()
                logger.info(
                    "[INDEX] Face scanning %d files (semantic pending was %d)",
                    total_face,
                    total_sem,
                )
                if progress_callback:
                    progress_callback(
                        progress_indexed_base,
                        progress_album_total,
                        format_index_progress("Face", 0, total_face),
                    )
                self._run_parallel_face_scan(
                    face_pending,
                    conn,
                    progress_callback,
                    conn_lock=threading.Lock(),
                    commit_every=commit_every,
                    progress_album_total=progress_album_total,
                    progress_indexed_base=progress_indexed_base,
                )
        except IndexAborted:
            logger.info("[INDEX] Indexing aborted (folder scope changed)")
        finally:
            # Always drop the ILM throttle so concurrency is restored for the gallery.
            self._release_load_throttle()
            conn.close()
            
        duration = time.time() - t_start
        logger.info(f"[INDEX] Finished indexing process in {duration:.4f}s. Results -> indexed: {indexed}, skipped: {skipped}, failed: {failed}, total: {total}")
        faces_deferred = total_face > 0 and not run_face_inline
        return {
            "indexed": indexed,
            "skipped": skipped,
            "failed": failed,
            "total": total,
            "faces_deferred": int(faces_deferred),
            "faces_pending": total_face if faces_deferred else 0,
        }

    def get_index_coverage(self, file_paths: Sequence[str]) -> Dict[str, int]:
        """
        Return index coverage for the given file set.
        Metadata-lazy version: only checks database presence, does not touch disk.
        """
        if not file_paths:
            return {"total": 0, "indexed": 0, "missing": 0, "ready": 1}

        indexable_paths, _ = self._filter_duplicate_raw_companions(file_paths)
        total = len(indexable_paths)
        canonical_paths = [self._canonical_path(p) for p in indexable_paths if p]
        
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
                    f"SELECT COUNT(*) FROM semantic_index WHERE file_path IN ({qs}) AND semantic_ready IN (1, -1) AND model_name = ?",
                    [*batch, self.model_name],
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

    def get_semantic_embedding_coverage(self, file_paths: Sequence[str]) -> Dict[str, int]:
        """Count corpus rows with real CLIP embeddings (semantic_ready=1, dim>0)."""
        indexable_paths, _ = self._filter_duplicate_raw_companions(file_paths)
        total = len(indexable_paths)
        if not total:
            return {"total": 0, "embeddable": 0, "skipped": 0, "pending": 0}

        rows = self._fetch_rows_for_paths(indexable_paths)
        row_map = {self._canonical_path(str(r["file_path"])): r for r in rows}
        embeddable = skipped = pending = 0
        for p in indexable_paths:
            row = row_map.get(self._canonical_path(p))
            if not row:
                pending += 1
                continue
            sr = _coerce_int(self._row_value(row, "semantic_ready"))
            dim = _coerce_int(self._row_value(row, "dim"))
            if sr == 1 and dim > 0:
                embeddable += 1
            elif sr == -1:
                skipped += 1
            else:
                pending += 1
        return {
            "total": total,
            "embeddable": embeddable,
            "skipped": skipped,
            "pending": pending,
        }

    def get_pending_paths(self, file_paths: Sequence[str]) -> List[str]:
        """
        Return paths that need indexing (missing row, semantic_ready=0, or file changed on disk).
        Completed work is persisted in semantic_index.db and survives app restarts.
        """
        if not file_paths:
            return []

        import time
        t0 = time.time()

        indexable_paths, _ = self._filter_duplicate_raw_companions(file_paths)
        if not indexable_paths:
            return []

        # Bulk fetch rows in one (or a few) database queries!
        rows = self._fetch_rows_for_paths(indexable_paths)
        
        # Build map of canonical path to row
        row_map = {}
        for r in rows:
            fp = r['file_path']
            row_map[self._canonical_path(fp)] = r

        pending: List[str] = []
        
        canonical_map = {self._canonical_path(p): p for p in indexable_paths if p}
        for cp, original in canonical_map.items():
            try:
                st = os.stat(cp)
            except OSError:
                pending.append(original)
                continue

            row = row_map.get(cp)
            sr = _coerce_int(self._row_value(row, 'semantic_ready')) if row else 0
            dim = _coerce_int(self._row_value(row, 'dim')) if row else 0
            if (not row) or sr == 0 or (sr == 1 and dim == 0):
                pending.append(original)
                continue

            # Check if file changed on disk (mtime or size mismatch)
            row_size = self._row_value(row, 'file_size', None)
            if row_size is None or _coerce_int(row_size) != int(st.st_size):
                pending.append(original)
                continue

            row_mtime_ns = self._row_value(row, 'mtime_ns', None)
            if row_mtime_ns is not None:
                try:
                    if _coerce_int(row_mtime_ns) != self._mtime_ns_from_stat(st):
                        pending.append(original)
                        continue
                except Exception:
                    pass
            else:
                row_mtime = self._row_value(row, 'file_mtime', None)
                mtime = _coerce_float(row_mtime)
                if mtime is None or not self._mtime_matches(mtime, st):
                    pending.append(original)
                    continue

        import logging
        logger = logging.getLogger(__name__)
        logger.debug(
            "[INDEX] Pending scan: %d pending of %d paths in %.4fs (bulk)",
            len(pending),
            len(file_paths),
            time.time() - t0
        )
        return pending

    def get_metadata_extraction_pending_paths(self, file_paths: Sequence[str]) -> List[str]:
        """Paths that need on-disk EXIF/metadata extraction (no row or file changed).

        Narrower than :meth:`get_pending_paths`, which also lists files awaiting
        semantic embeddings. Use this to skip ``build_index`` when the silent
        metadata pass has nothing to read from disk.
        """
        if not file_paths:
            return []

        indexable_paths, _ = self._filter_duplicate_raw_companions(file_paths)
        if not indexable_paths:
            return []

        rows = self._fetch_rows_for_paths(indexable_paths)
        row_map = {}
        for r in rows:
            row_map[self._canonical_path(r["file_path"])] = r

        pending: List[str] = []
        canonical_map = {self._canonical_path(p): p for p in indexable_paths if p}
        for cp, original in canonical_map.items():
            try:
                st = os.stat(cp)
            except OSError:
                pending.append(original)
                continue

            row = row_map.get(cp)
            if not row:
                pending.append(original)
                continue

            row_size = self._row_value(row, "file_size", None)
            if row_size is None or _coerce_int(row_size) != int(st.st_size):
                pending.append(original)
                continue

            row_mtime_ns = self._row_value(row, "mtime_ns", None)
            if row_mtime_ns is not None:
                try:
                    if _coerce_int(row_mtime_ns) != self._mtime_ns_from_stat(st):
                        pending.append(original)
                except Exception:
                    pending.append(original)
                continue

            row_mtime = self._row_value(row, "file_mtime", None)
            if row_mtime is None or not self._mtime_matches(_coerce_float(row_mtime) or 0.0, st):
                pending.append(original)

        return pending

    def get_pending_embedding_paths(self, file_paths: Sequence[str]) -> List[str]:
        """
        Return paths that have metadata in database but are missing semantic embeddings (dim = 0).
        """
        if not file_paths:
            return []
        
        indexable_paths, _ = self._filter_duplicate_raw_companions(file_paths)
        if not indexable_paths:
            return []

        rows = self._fetch_rows_for_paths(indexable_paths)
        row_map = {}
        for r in rows:
            fp = r['file_path']
            row_map[self._canonical_path(fp)] = r

        pending: List[str] = []
        canonical_map = {self._canonical_path(p): p for p in indexable_paths if p}
        for cp, original in canonical_map.items():
            row = row_map.get(cp)
            sr = _coerce_int(self._row_value(row, "semantic_ready")) if row else 0
            dim = _coerce_int(self._row_value(row, "dim")) if row else 0
            # Match get_pending_paths: missing row, metadata-only (sr=0), or no embedding yet.
            if (not row) or sr == 0 or (sr == 1 and dim == 0):
                pending.append(original)

        return pending


    def _fetch_rows_for_paths(self, paths: Sequence[str]) -> List[MetadataRow]:
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
                       city, landmark, admin1, country, country_code, face_count, semantic_ready,
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
                    'landmark': "",
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
                    w = _coerce_int(r["width"])
                    h = _coerce_int(r["height"])
                    o = _coerce_int(r["orientation"], 1)
                    result_map[orig_path] = {
                        "original_width": w,
                        "original_height": h,
                        "orientation": o,
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
    def _row_value(row: MetadataRow | None, key: str, default: object = "") -> object:
        if row is None:
            return default
        try:
            return row[key]
        except Exception:
            try:
                if isinstance(row, dict):
                    return row.get(key, default)
            except Exception:
                pass
            return default

    @staticmethod
    def _to_int(value: object, default: int = 0) -> int:
        return _coerce_int(value, default)

    @staticmethod
    def _to_float(value: object) -> Optional[float]:
        return _coerce_float(value)

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
        h = str(haystack or "").strip().lower()
        n = str(needle or "").strip().lower()
        if not n or not h:
            return False
        variants = {n, n.replace("_", " "), n.replace(" ", "_")}
        if any(v and v in h for v in variants):
            return True
        if len(h) >= 4:
            for v in variants:
                if h in v:
                    return True
        return False

    @staticmethod
    def _contains_location_place(haystack: str, needle: str) -> bool:
        """Location field match — forward substring with word boundaries only."""
        h = str(haystack or "").strip().lower()
        n = str(needle or "").strip().lower()
        if not n or not h:
            return False
        variants = {n, n.replace("_", " "), n.replace(" ", "_")}
        for v in variants:
            if not v:
                continue
            if h == v:
                return True
            start = 0
            while True:
                idx = h.find(v, start)
                if idx < 0:
                    break
                before_ok = idx == 0 or not h[idx - 1].isalnum()
                end = idx + len(v)
                after_ok = end == len(h) or not h[end].isalnum()
                if before_ok and after_ok:
                    return True
                start = idx + 1
        return False

    def _apply_filters(
        self, rows: Sequence[MetadataRow], query_text: str
    ) -> tuple[List[MetadataRow], str]:
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

            if low.startswith("city:") or low.startswith("landmark:"):
                matched = True
                needle = t.split(":", 1)[1].strip().lower()
                if needle:
                    filtered = [
                        r for r in filtered 
                        if self._contains_location_place(str(r["city"] or ""), needle)
                        or self._contains_location_place(str(self._row_value(r, "landmark") or ""), needle)
                    ]
                continue

            if low.startswith("admin:"):
                matched = True
                needle = t.split(":", 1)[1].strip().lower()
                if needle:
                    filtered = [
                        r for r in filtered
                        if self._contains_location_place(str(r["admin1"] or ""), needle)
                    ]
                continue

            if low.startswith("country:"):
                matched = True
                needle = t.split(":", 1)[1].strip().lower()
                if needle:
                    filtered = [
                        r
                        for r in filtered
                        if (
                            self._contains_location_place(str(r["country"] or ""), needle)
                            or self._contains_location_place(str(r["country_code"] or ""), needle)
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
                filtered = [r for r in filtered if _coerce_int(r["face_count"]) > 0]
                continue

            if low in self._FACE_COUNT_NEGATIVE_TOKENS:
                matched = True
                filtered = [r for r in filtered if _coerce_int(r["face_count"]) <= 0]
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
                            abs((_coerce_float(r["gps_lat"]) or 0.0) - target_lat) < 0.5 and 
                            abs((_coerce_float(r["gps_lon"]) or 0.0) - target_lon) < 0.5
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
                    if x <= 0:
                        if op == "=":
                            return val == 0
                        if op == "<=":
                            return val == 0
                        return False
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
                    filtered = [r for r in filtered if _ok(_coerce_int(self._row_value(r, "iso")))]
                elif key == "year":
                    filtered = [r for r in filtered if _ok(self._parse_capture_year(str(self._row_value(r, "capture_time") or "")))]
                elif key == "month":
                    filtered = [r for r in filtered if _ok(self._parse_capture_month(str(self._row_value(r, "capture_time") or "")))]
                elif key == "width":
                    filtered = [r for r in filtered if _ok(_coerce_int(self._row_value(r, "width")))]
                elif key == "height":
                    filtered = [r for r in filtered if _ok(_coerce_int(self._row_value(r, "height")))]
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
                bare_place = bare.replace("_", " ")
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

    def _is_strictly_location_name(self, term: str) -> bool:
        """
        Check if a term is a known country or city/place name.
        Uses static lists plus the bundled GeoNames place index (O(1) lookup).
        """
        needle = _normalize_location_term(term)
        if len(needle) < 3:
            return False

        if needle in KNOWN_LOCATION_NAMES:
            return True

        if needle in _pycountry_name_index():
            return True

        geo = self._reverse_geocoder
        if not isinstance(geo, CustomReverseGeocoder):
            ensured = self._ensure_reverse_geocoder()
            if isinstance(ensured, CustomReverseGeocoder):
                geo = ensured
        if isinstance(geo, CustomReverseGeocoder):
            return geo.has_place_name(needle)

        return False

    def _row_matches_location_place(self, row: MetadataRow, needle: str) -> bool:
        place = (needle or "").strip().lower()
        if not place:
            return False
        return (
            self._contains_location_place(str(row["city"] or ""), place)
            or self._contains_location_place(str(self._row_value(row, "landmark") or ""), place)
            or self._contains_location_place(str(row["admin1"] or ""), place)
            or self._contains_location_place(str(row["country"] or ""), place)
            or self._contains_location_place(str(row["country_code"] or ""), place)
        )

    def _row_has_resolved_location(self, row: MetadataRow) -> bool:
        return bool(
            str(row["city"] or "").strip()
            or str(self._row_value(row, "landmark") or "").strip()
            or str(row["country"] or "").strip()
        )

    def _auto_match_metadata_keywords(
        self, rows: Sequence[MetadataRow], semantic_terms: Sequence[str]
    ) -> tuple[List[MetadataRow], List[str]]:
        """
        If a free-text token appears in metadata fields, use it as a metadata filter
        and remove it from the semantic text query.
        """
        filtered = list(rows)
        remaining: List[str] = []
        metadata_fields = ("city", "landmark", "admin1", "country", "country_code", "camera_model", "lens_model", "file_name")

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
            is_location = self._is_strictly_location_name(needle)
            if is_location:
                matched_rows = [
                    r for r in filtered if self._row_matches_location_place(r, needle)
                ]
            else:
                matched_rows = [
                    r
                    for r in filtered
                    if any(
                        self._contains_loose(str(self._row_value(r, field) or ""), needle)
                        for field in metadata_fields
                    )
                ]
            if matched_rows:
                logger.info(f"[SEARCH] Metadata match for '{needle}' found in {len(matched_rows)} image(s)")
                filtered = matched_rows
            else:
                # CONTRADICTION FILTER: If this is a location name (e.g. "Korea") but 
                # doesn't match any image in the folder, check if any images HAVE 
                # verified location metadata. If an image says "Japan", and we search 
                # "Korea", we should exclude it rather than letting AI "guess".
                if is_location and not semantic_embeddings_enabled():
                    new_filtered = []
                    contradiction_count = 0
                    for r in filtered:
                        if self._row_has_resolved_location(r):
                            if self._row_matches_location_place(r, needle):
                                new_filtered.append(r)
                            else:
                                # Contradiction! (Has location, but it's different)
                                contradiction_count += 1
                    if contradiction_count > 0:
                        logger.warning(f"[SEARCH] Contradiction Filter: Excluded {contradiction_count} image(s) from '{needle}' due to conflicting GPS metadata")
                    filtered = new_filtered
                elif is_location and semantic_embeddings_enabled():
                    # Ambiguous bare token that is also a GeoNames place (e.g. "train"
                    # the town in Germany vs trains the vehicle). With no GPS hits in this
                    # folder, prefer semantic CLIP rather than returning zero results.
                    logger.debug(
                        "[SEARCH] No GPS match for place name '%s'; passing to semantic AI",
                        needle,
                    )
                    remaining.append(term)
                else:
                    if semantic_embeddings_enabled():
                        logger.debug(f"[SEARCH] No metadata match for '{needle}', passing to semantic AI")
                        remaining.append(term)
                    else:
                        # Metadata-only build: unresolved free-text is not a match.
                        filtered = []

        return filtered, remaining

    def search_text(
        self, query: str, candidate_paths: Sequence[str], top_k: int = 200, min_score: float = 0.0
    ) -> List[SearchHit]:
        if not semantic_embeddings_enabled():
            return []
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
                    iso=_coerce_int(r["iso"]),
                    gps_lat=_coerce_float(r["gps_lat"]),
                    gps_lon=_coerce_float(r["gps_lon"]),
                    city=str(r["city"] or ""),
                    landmark=str(r["landmark"] or ""),
                    admin1=str(r["admin1"] or ""),
                    country=str(r["country"] or ""),
                    country_code=str(r["country_code"] or ""),
                    face_count=_coerce_int(r["face_count"]),
                )
                for r in rows
            ]
            hits.sort(key=lambda h: h.capture_time, reverse=True)
            return hits[: max(1, int(top_k))]

        query_vecs = self._encode_semantic_query(semantic_query)
        if not query_vecs:
            return []

        scores: List[SearchHit] = []
        for r in rows:
            if not self._row_semantic_ready(r):
                continue
            vec = self._from_blob(cast(bytes, r["embedding"]), _coerce_int(r["dim"]))
            if vec.size == 0:
                continue
            score = self._semantic_similarity(query_vecs, vec)
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
                    iso=_coerce_int(r["iso"]),
                    gps_lat=_coerce_float(r["gps_lat"]),
                    gps_lon=_coerce_float(r["gps_lon"]),
                    city=str(r["city"] or ""),
                    landmark=str(r["landmark"] or ""),
                    admin1=str(r["admin1"] or ""),
                    country=str(r["country"] or ""),
                    country_code=str(r["country_code"] or ""),
                    face_count=_coerce_int(r["face_count"]),
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
                file_name=str(self._row_value(r, "file_name") or os.path.basename(str(r["file_path"]))),
                capture_time=str(r["capture_time"] or ""),
                camera_model=str(r["camera_model"] or ""),
                lens_model=str(r["lens_model"] or ""),
                iso=_coerce_int(r["iso"]),
                gps_lat=_coerce_float(r["gps_lat"]),
                gps_lon=_coerce_float(r["gps_lon"]),
                city=str(r["city"] or ""),
                landmark=str(r["landmark"] or ""),
                admin1=str(r["admin1"] or ""),
                country=str(r["country"] or ""),
                country_code=str(r["country_code"] or ""),
                face_count=_coerce_int(r["face_count"]),
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
            # Never run YuNet/OpenCV on the UI thread during gallery search — use
            # indexed face_count only (NULL → 0). Live detection belongs in indexing.
            if face_count is None:
                face_count = 0

            city = str(row["city"] or "")
            landmark = str(row["landmark"] or "")
            admin1 = str(row["admin1"] or "")
            country = str(row["country"] or "")
            gps_lat = row["gps_lat"]
            gps_lon = row["gps_lon"]

            try:
                from upgrade_compat import legacy_search_metadata_sync

                sync_search = legacy_search_metadata_sync()
            except Exception:
                sync_search = True

            if sync_search and not city and gps_lat is not None and gps_lon is not None:
                geo = self._ensure_reverse_geocoder()
                if isinstance(geo, CustomReverseGeocoder):
                    try:
                        with self._rg_lock:
                            recs = geo.search(
                                [
                                    (
                                        _coerce_float(gps_lat) or 0.0,
                                        _coerce_float(gps_lon) or 0.0,
                                    )
                                ],
                                mode=1,
                            )
                        if recs:
                            rec = recs[0] or {}
                            city_name = str(rec.get("name", "") or "")
                            alternates = str(rec.get("alternates", "") or "")
                            city = f"{city_name}, {alternates}" if alternates else city_name
                            landmark = str(rec.get("landmark", "") or "")
                            admin1 = str(rec.get("admin1", "") or "")
                            cc = str(rec.get("cc", "") or "").upper()
                            country = self._country_name_from_code(cc)
                            self._conn.execute(
                                "UPDATE semantic_index SET city=?, landmark=?, admin1=?, country=?, country_code=? WHERE file_path=?",
                                (city, landmark, admin1, country, cc, original),
                            )
                            self._conn.commit()
                    except Exception:
                        pass
            elif not sync_search and not city and gps_lat is not None and gps_lon is not None:
                self._enqueue_metadata_backfill([], geocode_paths=[original])

            rows.append(
                {
                    "file_path": original,
                    "file_name": str(row["file_name"] or os.path.basename(original)),
                    "capture_time": str(row["capture_time"] or ""),
                    "camera_model": str(row["camera_model"] or ""),
                    "lens_model": str(row["lens_model"] or ""),
                    "iso": _coerce_int(row["iso"]),
                    "width": _coerce_int(row["width"]),
                    "height": _coerce_int(row["height"]),
                    "gps_lat": gps_lat,
                    "gps_lon": gps_lon,
                    "city": city,
                    "landmark": landmark,
                    "admin1": admin1,
                    "country": country,
                    "country_code": str(row["country_code"] or ""),
                    "face_count": _coerce_int(face_count),
                }
            )
        try:
            from upgrade_compat import legacy_search_metadata_sync

            sync_search = legacy_search_metadata_sync()
        except Exception:
            sync_search = True

        missing_paths = [
            p
            for p in candidate_paths
            if p
            and os.path.isfile(p)
            and self._canonical_path(p) not in found_paths
        ]

        if sync_search:
            for p in missing_paths:
                meta = self._extract_exif_brief(p, include_face=False)
                rows.append(
                    {
                        "file_path": p,
                        "file_name": os.path.basename(p),
                        "capture_time": str(meta.get("capture_time") or ""),
                        "camera_model": str(meta.get("camera_model") or ""),
                        "lens_model": str(meta.get("lens_model") or ""),
                        "iso": self._to_int(meta.get("iso")),
                        "width": self._to_int(meta.get("width")),
                        "height": self._to_int(meta.get("height")),
                        "gps_lat": meta.get("gps_lat"),
                        "gps_lon": meta.get("gps_lon"),
                        "city": str(meta.get("city") or ""),
                        "landmark": str(meta.get("landmark") or ""),
                        "admin1": str(meta.get("admin1") or ""),
                        "country": str(meta.get("country") or ""),
                        "country_code": str(meta.get("country_code") or ""),
                        "face_count": self._to_int(meta.get("face_count")),
                    }
                )
        else:
            if missing_paths:
                cached_rows = self._metadata_rows_from_exif_cache(missing_paths)
                rows.extend(cached_rows)
                cached_canonical = {
                    self._canonical_path(str(r["file_path"])) for r in cached_rows
                }
                still_missing = [
                    p
                    for p in missing_paths
                    if self._canonical_path(p) not in cached_canonical
                ]
                if still_missing:
                    self._enqueue_metadata_backfill(still_missing)
        return rows

    def _metadata_rows_from_exif_cache(
        self, file_paths: Sequence[str]
    ) -> List[Dict[str, object]]:
        """Read-only EXIF cache rows for search (no disk extract on UI thread)."""
        out: List[Dict[str, object]] = []
        if not file_paths:
            return out
        try:
            from image_cache import get_image_cache

            cache = get_image_cache()
            cached = cache.get_multiple_exif(list(file_paths), fast_mode=True)
        except Exception:
            return out
        for p in file_paths:
            exif = cached.get(p)
            if not isinstance(exif, dict):
                continue
            raw_exif_data = exif.get("exif_data")
            exif_data: Dict[str, object] = (
                raw_exif_data if isinstance(raw_exif_data, dict) else {}
            )
            out.append(
                {
                    "file_path": p,
                    "file_name": os.path.basename(p),
                    "capture_time": str(exif.get("capture_time") or exif_data.get("capture_time") or ""),
                    "camera_model": str(exif.get("camera_model") or exif_data.get("camera_model") or ""),
                    "lens_model": str(exif_data.get("lens_model") or ""),
                    "iso": self._to_int(exif_data.get("iso")),
                    "width": self._to_int(exif.get("original_width") or exif_data.get("original_width")),
                    "height": self._to_int(exif.get("original_height") or exif_data.get("original_height")),
                    "gps_lat": exif_data.get("gps_lat"),
                    "gps_lon": exif_data.get("gps_lon"),
                    "city": str(exif_data.get("city") or ""),
                    "landmark": str(exif_data.get("landmark") or ""),
                    "admin1": str(exif_data.get("admin1") or ""),
                    "country": str(exif_data.get("country") or ""),
                    "country_code": str(exif_data.get("country_code") or ""),
                    "face_count": 0,
                }
            )
        return out

    def _store_face_count(
        self,
        file_path: str,
        face_count: int,
        conn: Optional[sqlite3.Connection] = None,
        *,
        commit: bool = True,
    ) -> None:
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
            if commit:
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

