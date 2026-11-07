"""Configuration and feature toggle functions for the semantic indexer."""
from __future__ import annotations

import os
import shutil
import sqlite3
import threading
from collections.abc import Callable
from typing import TypeAlias

try:
    import pycountry
except Exception:
    pycountry = None

try:
    import pycountry
except Exception:
    pycountry = None

from core.errors import IndexAborted, RAWViewerError  # noqa: F401

ProgressCallback = Callable[[int, int, str], None] | None

MetadataRow: TypeAlias = sqlite3.Row | dict[str, object]

def _materialize_cached_file(src: str, dst: str) -> None:
    """Expose a file downloaded into the shared Hugging Face cache at *dst*.

    ``hf_hub_download`` is called without ``local_dir`` so files are stored once
    in ``HF_HUB_CACHE`` (``~/.cache/huggingface/hub`` by default). Repeated runs
    are then served from that cache instead of being downloaded again. Hard-link
    when possible to avoid duplicating multi-hundred-MB weights; fall back to a
    copy when the cache and destination live on different volumes.
    """
    parent = os.path.dirname(dst)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.exists(dst):
        if os.path.samefile(src, dst):
            return
        os.remove(dst)
    try:
        os.link(src, dst)
        return
    except OSError:
        pass
    tmp = f"{dst}.tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)

def _coerce_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def _coerce_float(value: object) -> float | None:
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
        from core.upgrade_compat import perf_v2_enabled

        if not perf_v2_enabled():
            return 1
    except Exception:
        return 1
    try:
        from core.profile import classify_memory_tier, system_total_ram_gb

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

def semantic_batch_size_forced() -> int | None:
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

def semantic_coreml_batch_candidates() -> list[int]:
    """Chunk sizes to benchmark for Core ML (parallel prep only; inference is per-image)."""
    raw = os.environ.get("RAWVIEWER_SEMANTIC_COREML_CHUNK_CANDIDATES", "").strip()
    cap = min(semantic_batch_max(), 32)
    if raw:
        out: list[int] = []
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
            uniq: list[int] = []
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

def semantic_encode_prep_workers(sample_path: str | None = None) -> int:
    """Parallel CPU workers for resize/load before ONNX image batch."""
    raw = os.environ.get("RAWVIEWER_SEMANTIC_PREP_WORKERS", "").strip()
    if raw:
        try:
            return max(1, min(int(raw), 48))
        except Exception:
            pass

    try:
        from core.profile import classify_memory_tier, system_total_ram_gb

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
        from core.upgrade_compat import perf_v2_enabled

        if not perf_v2_enabled():
            return 8
    except Exception:
        return 8
    try:
        from core.profile import classify_memory_tier, system_total_ram_gb

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

def semantic_coreml_chunk_forced() -> int | None:
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

def semantic_batch_candidates() -> list[int]:
    raw = os.environ.get(
        "RAWVIEWER_SEMANTIC_BATCH_CANDIDATES", "8,16,32,64,128"
    ).strip()
    cap = semantic_batch_max()
    out: list[int] = []
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
    uniq: list[int] = []
    for v in out:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq

# Bump when auto-tune defaults/logic change so ~/.rawviewer_cache picks are refreshed.
SEMANTIC_BATCH_TUNE_CACHE_VERSION = "v8"
