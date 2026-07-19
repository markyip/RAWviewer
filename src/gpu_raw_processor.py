import logging
import os
import time
import threading
from typing import Any, Callable, Optional
import numpy as np

logger = logging.getLogger(__name__)

# GPU decode concurrency control.
#
# PyTorch backends: torch ops are thread-safe and CUDA kernels from different
# host threads overlap via per-thread streams (see gpu_demosaic_pytorch_
# unpacked), so decodes are gated by a SEMAPHORE (default 2, tune with
# RAWVIEWER_GPU_CONCURRENCY) instead of fully serialized -- a prefetch decode
# can upload while the current decode computes. Bounded because each in-
# flight decode holds ~130MB of float32 tensors per 33MP frame in VRAM.
#
# CuPy keeps a hard LOCK: the historical "multi-threaded CUDA access
# violation" came from the custom raw ElementwiseKernel path, which is not
# audited for concurrent launches with shared kernel objects.
_GPU_LOCK = threading.Lock()
_GPU_CONCURRENCY_CACHED: Optional[int] = None
_GPU_SEMAPHORE: Optional[threading.Semaphore] = None
# Debounce state for the idle-triggered allocator flush (see
# maybe_release_gpu_memory_after_decode): a pending threading.Timer that
# fires release_cached_gpu_memory() only once no new decode has completed
# for RAWVIEWER_GPU_EMPTY_CACHE_S seconds, re-armed (cancel + reschedule) on
# every decode instead of firing on a flat calendar-time cadence.
_IDLE_FLUSH_TIMER: Optional[threading.Timer] = None
_IDLE_FLUSH_LOCK = threading.Lock()


def _default_gpu_concurrency() -> int:
    """Adaptive default: MPS=1 (unified RAM); CUDA scales with VRAM; else 2."""
    backend = detect_gpu_backend()
    if backend == "pytorch_mps" or backend == "cupy":
        # CuPy demosaic is fully serialized by _GPU_LOCK; keep permits=1 for
        # empty_cache drain logic. MPS thrashing two full floats hurts RSS.
        base = 1
    elif backend == "pytorch_cuda" and _HAS_TORCH:
        try:
            idx = _cuda_device_index()
            props = torch.cuda.get_device_properties(idx)
            # ~700-900MB peak float workspace per ~45MP decode → budget conservatively.
            vram_gb = float(props.total_memory) / (1024.0 ** 3)
            if vram_gb >= 16:
                base = 3
            elif vram_gb >= 10:
                base = 2
            elif vram_gb >= 6:
                base = 2
            else:
                base = 1
        except Exception:
            base = 2
    else:
        base = 2
    return max(1, min(4, base))


def _gpu_concurrency() -> int:
    global _GPU_CONCURRENCY_CACHED
    if _GPU_CONCURRENCY_CACHED is not None:
        return _GPU_CONCURRENCY_CACHED
    raw = os.environ.get("RAWVIEWER_GPU_CONCURRENCY", "").strip()
    if raw:
        try:
            _GPU_CONCURRENCY_CACHED = max(1, min(4, int(raw)))
            return _GPU_CONCURRENCY_CACHED
        except ValueError:
            pass
    _GPU_CONCURRENCY_CACHED = _default_gpu_concurrency()
    return _GPU_CONCURRENCY_CACHED


def gpu_decode_concurrency() -> int:
    """Public: max in-flight GPU demosaics (for aligning RAW load slots)."""
    return _gpu_concurrency()


def _gpu_semaphore() -> threading.Semaphore:
    global _GPU_SEMAPHORE
    if _GPU_SEMAPHORE is None:
        _GPU_SEMAPHORE = threading.Semaphore(_gpu_concurrency())
    return _GPU_SEMAPHORE


def _cuda_device_index() -> int:
    raw = os.environ.get("RAWVIEWER_CUDA_DEVICE", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _cuda_device_str() -> str:
    return f"cuda:{_cuda_device_index()}"


# Cache detection results to avoid repeated import attempts
_GPU_BACKEND: Optional[str] = None

# Pre-import heavy ML libraries on the main thread.
# Importing torch for the first time inside a background thread (e.g., QRunnable)
# causes OpenMP (libomp.dylib) to fatally abort on macOS.
try:
    import torch
    import kornia
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import cupy
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False

def detect_gpu_backend() -> str:
    """
    Detect if the system has PyTorch (with CUDA or MPS) or CuPy installed.
    Allows overriding via the RAWVIEWER_GPU_BACKEND environment variable.
    Returns: 'pytorch_cuda', 'pytorch_mps', 'cupy', or 'cpu_only'
    """
    global _GPU_BACKEND
    if _GPU_BACKEND is not None:
        return _GPU_BACKEND

    # Check for environment override
    override = os.environ.get("RAWVIEWER_GPU_BACKEND", "").strip().lower()
    if override in ("pytorch_cuda", "pytorch_mps", "cupy", "cpu_only"):
        _GPU_BACKEND = override
        logger.info(f"GPU Processor: Using backend override '{_GPU_BACKEND}' from RAWVIEWER_GPU_BACKEND.")
        return _GPU_BACKEND

    # 1. Check for PyTorch (CUDA / Apple Silicon MPS)
    if _HAS_TORCH:
        if torch.cuda.is_available():
            _GPU_BACKEND = "pytorch_cuda"
            logger.info("GPU Processor: PyTorch CUDA backend detected.")
            return _GPU_BACKEND
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _GPU_BACKEND = "pytorch_mps"
            logger.info("GPU Processor: PyTorch MPS (Apple Silicon) backend detected.")
            return _GPU_BACKEND

    # 2. Check for CuPy (NVIDIA CUDA)
    if _HAS_CUPY:
        try:
            # Try creating a device context to ensure CUDA driver is working
            if cupy.cuda.Device().id >= 0:
                _GPU_BACKEND = "cupy"
                logger.info("GPU Processor: CuPy CUDA backend detected.")
                return _GPU_BACKEND
        except Exception:
            pass

    _GPU_BACKEND = "cpu_only"
    return _GPU_BACKEND


def release_cached_gpu_memory() -> None:
    """Return the GPU/unified-memory backend's caching allocator pool to the OS.

    PyTorch's CUDA/MPS allocators cache freed device buffers for reuse rather
    than releasing them (a normal, deliberate speed-over-memory tradeoff) --
    on Apple Silicon this pool is UNIFIED memory shared with system RAM, so it
    shows up directly as process RSS growth that's invisible to Python's own
    gc/tracemalloc (it's not Python-heap memory). Never called before this,
    the cache had no way back to the OS for the life of the process.

    CUDA's caching allocator is documented safe to empty_cache() concurrently
    with in-flight work on other threads (it only frees currently-unused
    blocks, never memory backing a live tensor). MPS (Apple Silicon) is a
    newer backend without the same track record for this exact concurrency
    pattern, and this is reachable from ANY thread that touches the image
    cache (get_full_image/get_thumbnail/etc. all funnel through
    _check_memory_pressure), including background prefetch/decode workers --
    so a call here could otherwise land while another thread is mid-decode
    inside the _GPU_SEMAPHORE-gated region below. Drain the semaphore
    (non-blocking) first: if a decode is in flight, skip this cycle rather
    than block the caller waiting for it -- _check_memory_pressure's own
    cooldown means the next opportunity is only ~20-60s away.
    """
    backend = detect_gpu_backend()
    gate = _GPU_LOCK if backend == "cupy" else _gpu_semaphore()
    concurrency = 1 if backend == "cupy" else _gpu_concurrency()
    acquired = 0
    try:
        for _ in range(concurrency):
            if not gate.acquire(blocking=False):
                break
            acquired += 1
        if acquired < concurrency:
            logger.debug(
                "release_cached_gpu_memory: decode in flight, skipping this cycle "
                "(acquired %d/%d permits)",
                acquired,
                concurrency,
            )
            return
        if backend == "pytorch_mps" and _HAS_TORCH:
            torch.mps.empty_cache()
        elif backend == "pytorch_cuda" and _HAS_TORCH:
            try:
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
            except Exception:
                torch.cuda.empty_cache()
        elif backend == "cupy" and _HAS_CUPY:
            cupy.get_default_memory_pool().free_all_blocks()
    except Exception:
        logger.debug("release_cached_gpu_memory failed", exc_info=True)
    finally:
        for _ in range(acquired):
            gate.release()


def maybe_release_gpu_memory_after_decode() -> None:
    """Schedule an allocator flush after the GPU has been genuinely idle.

    The previous implementation flushed whenever >=N seconds had elapsed
    since the last flush -- a flat calendar-time cadence with no notion of
    whether the user was still actively navigating. During continuous rapid
    browsing that fires roughly every N seconds regardless, so whichever
    decode happened to finish right after the flush paid a full realloc
    (cudaMalloc) instead of reusing the caching allocator's pooled blocks --
    observed in practice as a same-sized 45MP CR3 decode jumping from
    ~215ms to ~1250ms for no content-related reason, purely because it
    landed at the unlucky moment.

    Debouncing instead (cancel + reschedule on every decode) means the
    flush only ever fires once no new decode has completed for
    RAWVIEWER_GPU_EMPTY_CACHE_S seconds -- i.e. actual idle time, never
    interrupting an active browsing streak. The independent memory-pressure
    path in image_cache._release_gpu_memory() is unaffected and still backs
    this up under genuine RAM/VRAM pressure even mid-session.
    """
    global _IDLE_FLUSH_TIMER
    min_interval = float(os.environ.get("RAWVIEWER_GPU_EMPTY_CACHE_S", "8") or 8)
    if min_interval <= 0:
        return
    with _IDLE_FLUSH_LOCK:
        if _IDLE_FLUSH_TIMER is not None:
            _IDLE_FLUSH_TIMER.cancel()
        timer = threading.Timer(min_interval, _run_idle_gpu_flush)
        timer.daemon = True
        _IDLE_FLUSH_TIMER = timer
        timer.start()


def _run_idle_gpu_flush() -> None:
    global _IDLE_FLUSH_TIMER
    with _IDLE_FLUSH_LOCK:
        _IDLE_FLUSH_TIMER = None
    release_cached_gpu_memory()


def gpu_demosaic_pytorch_unpacked(
    unpacked,
    device_str: str = "cuda",
    cancel_check: Optional[Callable[[], bool]] = None,
    return_linear: bool = False,
    *,
    retain_device: bool = False,
):
    """
    GPU-accelerated Demosaicing using PyTorch and Kornia.
    Consumes UnpackedRaw from fast_raw_decode.py to guarantee color math parity.

    When ``retain_device`` is True (CUDA + Phase 2c), returns a ``DeviceRgb``
    instead of downloading to numpy.
    """
    if device_str in ("cuda", "cuda:0") and os.environ.get("RAWVIEWER_CUDA_DEVICE", "").strip():
        device_str = _cuda_device_str()
    elif device_str == "cuda":
        device_str = _cuda_device_str()
    device = torch.device(device_str)

    # Reuse one stream per device so consecutive develops don't allocate a
    # new CUDA stream each call (fresh Stream() diluted allocator locality
    # under rapid re-decode). Final .cpu() still synchronizes.
    stream_ctx = (
        torch.cuda.stream(_cuda_stream_for(device))
        if device.type == "cuda"
        else _NullCtx()
    )
    with stream_ctx:
        return _gpu_demosaic_pytorch_body(
            unpacked,
            device,
            cancel_check,
            return_linear,
            retain_device=retain_device,
        )


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# Per-device cache of the dcraw BT.709 gamma curve as a GPU-resident LUT:
# fast_raw_decode.gamma_curve16() already computes and caches this curve
# (a 48-iteration bisection) once at module scope for the CPU path: reuse
# that instead of re-deriving the identical formula in Python on every GPU
# decode call. Keyed by device string ("cuda"/"mps") since the underlying
# tensor must live on the same device as the data it indexes.
_GAMMA_LUT_GPU: dict = {}
_CUDA_STREAMS: dict = {}
_GPU_WORKSPACES: dict = {}
_RGB_CAM_GPU: dict = {}


def _cuda_stream_for(device) -> Any:
    key = str(device)
    stream = _CUDA_STREAMS.get(key)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _CUDA_STREAMS[key] = stream
    return stream


class _GpuIspWorkspace:
    """Growable device + pinned-host buffers shared across consecutive develops."""

    __slots__ = (
        "device",
        "h",
        "w",
        "raw_f",
        "raw_norm",
        "pinned_u16",
        "pinned_rgb_u8",
        "pinned_rgb_u16",
    )

    def __init__(self, device):
        self.device = device
        self.h = 0
        self.w = 0
        self.raw_f = None
        self.raw_norm = None
        self.pinned_u16 = None
        self.pinned_rgb_u8 = None
        self.pinned_rgb_u16 = None

    def ensure(self, h: int, w: int) -> None:
        if self.raw_f is not None and self.h == h and self.w == w:
            return
        self.h, self.w = h, w
        self.raw_f = torch.empty((h, w), device=self.device, dtype=torch.float32)
        self.raw_norm = torch.empty((h, w), device=self.device, dtype=torch.float32)
        if self.device.type == "cuda":
            try:
                self.pinned_u16 = torch.empty((h, w), dtype=torch.uint16, pin_memory=True)
            except Exception:
                self.pinned_u16 = None
            try:
                self.pinned_rgb_u8 = torch.empty(
                    (h, w, 3), dtype=torch.uint8, pin_memory=True
                )
            except Exception:
                self.pinned_rgb_u8 = None
            try:
                self.pinned_rgb_u16 = torch.empty(
                    (h, w, 3), dtype=torch.uint16, pin_memory=True
                )
            except Exception:
                self.pinned_rgb_u16 = None
        else:
            self.pinned_u16 = None
            self.pinned_rgb_u8 = None
            self.pinned_rgb_u16 = None


def _workspace_for(device) -> _GpuIspWorkspace:
    key = str(device)
    ws = _GPU_WORKSPACES.get(key)
    if ws is None:
        ws = _GpuIspWorkspace(device)
        _GPU_WORKSPACES[key] = ws
    return ws


def _rgb_cam_on_device(unpacked, device) -> Any:
    cam = unpacked.rgb_cam
    key = (str(device), cam.tobytes() if hasattr(cam, "tobytes") else id(cam))
    t = _RGB_CAM_GPU.get(key)
    if t is None or t.device != device:
        t = torch.as_tensor(cam, device=device, dtype=torch.float32)
        _RGB_CAM_GPU[key] = t
        if len(_RGB_CAM_GPU) > 32:
            _RGB_CAM_GPU.pop(next(iter(_RGB_CAM_GPU)))
    return t


def gpu_vram_snapshot(device_str: Optional[str] = None) -> dict[str, float]:
    """Return CUDA memory stats in MiB (empty dict if CUDA unavailable)."""
    if not _HAS_TORCH:
        return {}
    try:
        if not torch.cuda.is_available():
            return {}
        if device_str:
            idx = int(torch.device(device_str).index or 0)
        else:
            idx = _cuda_device_index()
        free_b, total_b = torch.cuda.mem_get_info(idx)
        return {
            "allocated_mib": torch.cuda.memory_allocated(idx) / (1024.0 ** 2),
            "reserved_mib": torch.cuda.memory_reserved(idx) / (1024.0 ** 2),
            "free_mib": free_b / (1024.0 ** 2),
            "total_mib": total_b / (1024.0 ** 2),
        }
    except Exception:
        return {}


def _download_device_rgb(
    rgb_device: Any,
    device,
    ws: _GpuIspWorkspace,
    *,
    as_uint16: bool = False,
) -> np.ndarray:
    """D2H into a reused pinned host buffer, then return an owned numpy copy.

    Avoids ``tensor.cpu().numpy()``'s pageable staging alloc on the hot path.
    Full CUDA↔GL interop (skip host) still needs a custom GL viewport that
    does not go through ``QGraphicsPixmapItem`` — see ``gpu_gl_bridge.py``.
    """
    if device.type != "cuda":
        arr = rgb_device.detach().cpu().numpy()
        if as_uint16 and arr.dtype != np.uint16:
            return arr.astype(np.uint16, copy=False)
        return arr

    rgb_device = rgb_device.contiguous()
    if as_uint16:
        pinned = ws.pinned_rgb_u16
        if pinned is None or pinned.shape != rgb_device.shape:
            try:
                pinned = torch.empty(
                    rgb_device.shape, dtype=torch.uint16, pin_memory=True
                )
                ws.pinned_rgb_u16 = pinned
            except Exception:
                return rgb_device.detach().cpu().numpy().astype(np.uint16)
        if rgb_device.dtype != torch.uint16:
            # int32 linear path → write into uint16 pin via clamp
            tmp = rgb_device.clamp(0, 65535).to(torch.uint16)
            pinned.copy_(tmp, non_blocking=True)
            del tmp
        else:
            pinned.copy_(rgb_device, non_blocking=True)
    else:
        pinned = ws.pinned_rgb_u8
        if pinned is None or pinned.shape != rgb_device.shape:
            try:
                pinned = torch.empty(
                    rgb_device.shape, dtype=torch.uint8, pin_memory=True
                )
                ws.pinned_rgb_u8 = pinned
            except Exception:
                return rgb_device.detach().cpu().numpy()
        pinned.copy_(rgb_device, non_blocking=True)

    torch.cuda.current_stream(device).synchronize()
    # Owned copy: next develop reuses the pin and would otherwise overwrite
    # arrays still held by PixmapConverter / image cache.
    return pinned.numpy().copy()


def _gpu_gamma_lut(device, torch_mod) -> Any:
    key = str(device)
    lut = _GAMMA_LUT_GPU.get(key)
    if lut is None:
        from fast_raw_decode import gamma_curve16

        # 16-bit curve, indexed by a 16-bit input -> keep as int64 index
        # source; convert once to a device-resident float LUT for gather.
        curve16 = gamma_curve16()  # np.uint16, 65536 entries
        lut = torch_mod.from_numpy(curve16.astype(np.float32)).to(device)
        _GAMMA_LUT_GPU[key] = lut
    return lut


# Last fused-ISP stage timings (ms), for benches / callers. Cleared/set only
# when RAWVIEWER_GPU_ISP_TIMING is on.
LAST_GPU_ISP_STAGES: dict[str, float] = {}


def _gpu_isp_timing_enabled() -> bool:
    """Stage timing for fused GPU develop (upload/scale/demosaic/...)."""
    return str(os.environ.get("RAWVIEWER_GPU_ISP_TIMING", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _gpu_isp_timing_sync_enabled() -> bool:
    """When stage timing is on, sync device between stages (default on).

    Set RAWVIEWER_GPU_ISP_TIMING_SYNC=0 for host timestamps without
    cudaSynchronize — wall time closer to production, stage borders blurrier.
    """
    return str(os.environ.get("RAWVIEWER_GPU_ISP_TIMING_SYNC", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _sync_if_timing(device, enabled: bool) -> None:
    if not enabled or not _gpu_isp_timing_sync_enabled():
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def _record_gpu_isp_stages(stage_ms: dict[str, float], file_path: str = "") -> None:
    """Publish stage timings for the bench script even when logging is quiet."""
    global LAST_GPU_ISP_STAGES
    LAST_GPU_ISP_STAGES = dict(stage_ms)
    total = sum(stage_ms.values())
    try:
        from perf_metrics import perf_mark

        perf_mark(
            "gpu_isp_stages",
            total,
            file_path or None,
            **{k: round(v, 1) for k, v in stage_ms.items()},
        )
    except Exception:
        pass
    sync = "sync" if _gpu_isp_timing_sync_enabled() else "nosync"
    line = (
        f"[GPU_ISP] stages({sync}) ms "
        + " ".join(f"{k}={v:.1f}" for k, v in stage_ms.items())
        + f" total={total:.1f}"
    )
    if file_path:
        line += f" file={os.path.basename(file_path)}"
    print(line, flush=True)
    logger.info(line)


def _gpu_demosaic_pytorch_body(
    unpacked,
    device,
    cancel_check=None,
    return_linear: bool = False,
    *,
    retain_device: bool = False,
):
    """Fused GPU develop: scale → demosaic → matrix → gamma; single D2H at end.

    Uploads the uint16 Bayer mosaic and performs black/WB scale on device so
    the host does not allocate a full-frame float32 copy before H2D.
    """
    def _abort_if_cancelled() -> None:
        if cancel_check is not None and cancel_check():
            from fast_raw_decode import DecodeCancelled

            raise DecodeCancelled(unpacked.file_path)

    timing = _gpu_isp_timing_enabled()
    stage_ms: dict[str, float] = {}
    t_prev = time.perf_counter()

    def _mark(stage: str) -> None:
        nonlocal t_prev
        if not timing:
            return
        _sync_if_timing(device, True)
        now = time.perf_counter()
        stage_ms[stage] = (now - t_prev) * 1000.0
        t_prev = now

    # 1. Upload uint16 mosaic via reused pinned buffer / device float workspace.
    mosaic = np.ascontiguousarray(unpacked.mosaic)
    if mosaic.dtype != np.uint16:
        mosaic = mosaic.astype(np.uint16, copy=False)
    h, w = mosaic.shape
    ws = _workspace_for(device)
    ws.ensure(h, w)

    if device.type == "cuda" and ws.pinned_u16 is not None:
        try:
            ws.pinned_u16.copy_(torch.from_numpy(mosaic))
            raw_u16 = ws.pinned_u16.to(device, non_blocking=True)
        except Exception:
            raw_u16 = torch.from_numpy(mosaic).to(device)
    elif device.type == "cuda":
        try:
            host = torch.from_numpy(mosaic)
            if not host.is_pinned():
                host = host.pin_memory()
            raw_u16 = host.to(device, non_blocking=True)
        except Exception:
            raw_u16 = torch.from_numpy(mosaic).to(device)
    else:
        raw_u16 = torch.from_numpy(mosaic).to(device)
    _mark("upload")

    raw_tensor = ws.raw_f
    raw_tensor.copy_(raw_u16.to(dtype=torch.float32))
    del raw_u16
    _mark("to_float")

    raw_norm = ws.raw_norm

    # dcraw scale_colors: (raw - black) * (scale_mul / 65535.0), clip to [0, 1]
    for (dy, dx), ci in np.ndenumerate(unpacked.pattern):
        slc = (slice(dy, None, 2), slice(dx, None, 2))
        black_val = float(unpacked.black[ci])
        scale_val = float(unpacked.scale_mul[ci] / 65535.0)
        raw_norm[slc] = torch.clamp((raw_tensor[slc] - black_val) * scale_val, 0.0, 1.0)
    _mark("scale")

    _abort_if_cancelled()

    # Kornia CFA naming expects the 2x2 block at the origin.
    # Slot for Debayer5x5 / other demosaics later: keep (1,1,H,W) float in [0,1].
    cfa_map = {
        "RGGB": kornia.color.CFA.BG,
        "BGGR": kornia.color.CFA.RG,
        "GRBG": kornia.color.CFA.GB,
        "GBRG": kornia.color.CFA.GR,
    }
    cfa = cfa_map.get(unpacked.pat_str, kornia.color.CFA.BG)
    raw_input = raw_norm.view(1, 1, h, w)
    rgb_tensor = kornia.color.raw_to_rgb(raw_input, cfa)
    rgb_tensor = rgb_tensor.squeeze(0).permute(1, 2, 0)
    _mark("demosaic")

    _abort_if_cancelled()

    m_color_tensor = _rgb_cam_on_device(unpacked, device)
    rgb_srgb = torch.matmul(rgb_tensor, m_color_tensor.t()).clamp(0.0, 1.0)
    del rgb_tensor
    _mark("matrix")

    _abort_if_cancelled()

    if return_linear:
        rgb_final16 = (rgb_srgb * 65535.0 + 0.5).to(torch.int32)
        _mark("gamma")
        out = _download_device_rgb(rgb_final16, device, ws, as_uint16=True)
        _mark("download")
    else:
        gamma_lut = _gpu_gamma_lut(device, torch)
        # int32 indices: ~half the temporary VRAM vs int64 (45MP RGB ≈ 0.5GB).
        idx = torch.clamp(rgb_srgb * 65535.0 + 0.5, 0, 65535).to(torch.int32)
        # CUDA indexing accepts int32; MPS gather wants long.
        rgb_gamma16 = gamma_lut[idx.to(torch.int64) if device.type == "mps" else idx]
        rgb_final = torch.clamp(rgb_gamma16 / 256.0, 0, 255).to(torch.uint8)
        del rgb_srgb, idx, rgb_gamma16
        _mark("gamma")
        keep_device = (
            bool(retain_device)
            and device.type == "cuda"
            and not return_linear
        )
        if keep_device:
            # Synchronize producer stream so GUI-thread CUDA-GL map sees data.
            try:
                torch.cuda.current_stream(device).synchronize()
            except Exception:
                pass
            from gpu_gl_bridge import DeviceRgb

            out = DeviceRgb(
                tensor=rgb_final.contiguous(),
                file_path=str(getattr(unpacked, "file_path", "") or ""),
            )
            if timing:
                stage_ms["download"] = 0.0
                t_prev = time.perf_counter()
        else:
            out = _download_device_rgb(rgb_final, device, ws, as_uint16=False)
            _mark("download")

    if timing:
        _record_gpu_isp_stages(
            stage_ms, getattr(unpacked, "file_path", "") or ""
        )

    return out


_GAMMA_LUT_CUPY: Optional[Any] = None


def _cupy_gamma_lut() -> Any:
    """CuPy-resident copy of the shared 16-bit gamma curve (see _gpu_gamma_lut)."""
    global _GAMMA_LUT_CUPY
    if _GAMMA_LUT_CUPY is None:
        from fast_raw_decode import gamma_curve16

        _GAMMA_LUT_CUPY = cupy.array(gamma_curve16().astype(np.float32))
    return _GAMMA_LUT_CUPY


# R-sample position (row, col) inside the 2x2 Bayer tile per pattern string;
# B is always at the opposite corner. Non-Bayer layouts (X-Trans) stay on CPU.
CUPY_BAYER_PHASE: dict = {
    "RGGB": (0, 0),
    "GRBG": (0, 1),
    "GBRG": (1, 0),
    "BGGR": (1, 1),
}


def gpu_demosaic_cupy_unpacked(unpacked, cancel_check: Optional[Callable[[], bool]] = None, return_linear: bool = False) -> np.ndarray:
    """
    GPU-accelerated demosaicing implementation using CuPy.
    Uses CuPy elementwise/raw CUDA kernels for maximum speed.
    """
    def _abort_if_cancelled() -> None:
        if cancel_check is not None and cancel_check():
            from fast_raw_decode import DecodeCancelled

            raise DecodeCancelled(unpacked.file_path)

    # Upload to GPU
    raw_gpu = cupy.array(unpacked.mosaic, dtype=cupy.float32)
    h, w = raw_gpu.shape

    # Pre-allocate normalized array
    raw_norm = cupy.empty_like(raw_gpu)

    for (dy, dx), ci in np.ndenumerate(unpacked.pattern):
        slc = (slice(dy, None, 2), slice(dx, None, 2))
        black_val = float(unpacked.black[ci])
        scale_val = float(unpacked.scale_mul[ci] / 65535.0)

        # Clamp to [0, 1]
        raw_norm[slc] = cupy.clip((raw_gpu[slc] - black_val) * scale_val, 0.0, 1.0)

    _abort_if_cancelled()

    phase = CUPY_BAYER_PHASE.get(unpacked.pat_str)
    if phase is None:
        raise ValueError(
            f"CuPy demosaic supports Bayer patterns {sorted(CUPY_BAYER_PHASE)}, "
            f"got {unpacked.pat_str}"
        )
    r_y, r_x = phase

    # Pre-allocate output RGB image on GPU
    rgb_gpu = cupy.zeros((h, w, 3), dtype=cupy.float32)

    # Bilinear Bayer demosaic on the pre-normalized, WB-scaled array.
    # (ry, rx) is the position of the R sample in the 2x2 tile; B sits at the
    # opposite corner and the two G samples fill the remaining diagonal, which
    # covers all four patterns (RGGB/BGGR/GRBG/GBRG) with one kernel.
    demosaic_kernel = cupy.ElementwiseKernel(
        'raw float32 raw_data, int32 h, int32 w, int32 ry, int32 rx',
        'raw float32 rgb_out',
        '''
        int y = i / w;
        int x = i % w;
        if (y < 1 || y >= h - 1 || x < 1 || x >= w - 1) return;

        int py = y & 1;
        int px = x & 1;
        float r = 0, g = 0, b = 0;

        if (py == ry && px == rx) {
            // R pixel
            r = raw_data[i];
            g = (raw_data[i-1] + raw_data[i+1] + raw_data[i-w] + raw_data[i+w]) * 0.25f;
            b = (raw_data[i-w-1] + raw_data[i-w+1] + raw_data[i+w-1] + raw_data[i+w+1]) * 0.25f;
        } else if (py == 1 - ry && px == 1 - rx) {
            // B pixel
            r = (raw_data[i-w-1] + raw_data[i-w+1] + raw_data[i+w-1] + raw_data[i+w+1]) * 0.25f;
            g = (raw_data[i-1] + raw_data[i+1] + raw_data[i-w] + raw_data[i+w]) * 0.25f;
            b = raw_data[i];
        } else {
            // G pixel: R neighbors are horizontal when this is an R row
            g = raw_data[i];
            if (py == ry) {
                r = (raw_data[i-1] + raw_data[i+1]) * 0.5f;
                b = (raw_data[i-w] + raw_data[i+w]) * 0.5f;
            } else {
                r = (raw_data[i-w] + raw_data[i+w]) * 0.5f;
                b = (raw_data[i-1] + raw_data[i+1]) * 0.5f;
            }
        }

        rgb_out[i * 3 + 0] = r;
        rgb_out[i * 3 + 1] = g;
        rgb_out[i * 3 + 2] = b;
        ''',
        'demosaic_kernel'
    )
    
    # Both raw_data and rgb_out are declared `raw` (manual indexing with
    # negative/row-stride offsets), so CuPy has no non-raw array argument to
    # infer the loop size from -- without an explicit `size`, the launch
    # collapses to a single element (i=0), which hits the border guard above
    # and returns immediately: the kernel silently demosaiced nothing but a
    # black frame. Every pixel must be visited, so size=h*w.
    demosaic_kernel(raw_norm, h, w, r_y, r_x, rgb_gpu, size=h * w)

    _abort_if_cancelled()

    # Apply Color Matrix Multiplication
    m_color_gpu = cupy.array(unpacked.rgb_cam, dtype=cupy.float32)
    rgb_srgb = cupy.dot(rgb_gpu, m_color_gpu.T)

    # Clamp to [0, 1]
    rgb_srgb = cupy.clip(rgb_srgb, 0.0, 1.0)

    _abort_if_cancelled()

    if return_linear:
        rgb_final16 = cupy.clip(rgb_srgb * 65535.0 + 0.5, 0, 65535).astype(cupy.uint16)
        return cupy.asnumpy(rgb_final16)

    # BT.709 gamma via the shared 16-bit LUT (same curve as
    # fast_raw_decode._gamma_lut8 / gamma_curve16; see _cupy_gamma_lut).
    gamma_lut = _cupy_gamma_lut()
    idx = cupy.clip(rgb_srgb * 65535.0 + 0.5, 0, 65535).astype(cupy.int64)
    rgb_gamma16 = gamma_lut[idx]

    # Scale and convert to uint8 (curve16 >> 8 semantics)
    rgb_final = cupy.clip(rgb_gamma16 / 256.0, 0, 255).astype(cupy.uint8)

    _abort_if_cancelled()

    # Download result to CPU
    return cupy.asnumpy(rgb_final)


def try_gpu_decode_from_unpacked(
    unpacked,
    cancel_check: Optional[Callable[[], bool]] = None,
    return_linear: bool = False,
    *,
    retain_device: Optional[bool] = None,
):
    """
    Attempt to decode the UnpackedRaw using the detected GPU backend.
    Falls back to None if GPU is not available or if processing errors occur.

    When Phase 2c (``RAWVIEWER_GPU_CUDA_GL=1``) is on and ``retain_device`` is
    not explicitly False, CUDA develops may return ``DeviceRgb`` instead of
    numpy (skipping D2H for the display path).
    """
    backend = detect_gpu_backend()
    if backend == "cpu_only":
        return None

    # Odd-sized mosaics: Kornia's raw_to_rgb hard-rejects odd H/W (raises
    # "Input H&W must be evenly disible by 2"), so e.g. the EOS R6 Mark III
    # (4639x6959 visible area) would throw and fall back on EVERY decode --
    # a wasted GPU upload attempt plus a logged error each time. Skip
    # straight to the CPU path (cv2's demosaic handles odd dims fine).
    h, w = unpacked.mosaic.shape[:2]
    if (h % 2) or (w % 2):
        logger.debug(
            "GPU Processor: odd-sized mosaic %dx%d for %s; using CPU path.",
            w, h, unpacked.file_path,
        )
        return None

    if retain_device is None:
        try:
            from gpu_gl_bridge import cuda_gl_enabled

            retain_device = bool(cuda_gl_enabled()) and not return_linear
        except Exception:
            retain_device = False

    if backend == "cupy" and unpacked.pat_str not in CUPY_BAYER_PHASE:
        # Non-Bayer layout (e.g. X-Trans): torch can't demosaic it either
        # (kornia is Bayer-only) — go straight to the CPU path.
        logger.debug(
            "GPU Processor: CuPy demosaic has no kernel for %s; using CPU path.",
            unpacked.pat_str,
        )
        return None

    # PyTorch backends run under the bounded semaphore (parallel decodes,
    # streams overlap upload/compute); CuPy keeps the hard lock -- see the
    # concurrency-control comment at the top of this module.
    gate = _GPU_LOCK if backend == "cupy" else _gpu_semaphore()
    res = None
    elapsed_ms = 0.0
    with gate:
        if cancel_check is not None and cancel_check():
            from fast_raw_decode import DecodeCancelled
            raise DecodeCancelled(unpacked.file_path)

        t_start = time.time()
        try:
            if backend == "pytorch_cuda":
                res = gpu_demosaic_pytorch_unpacked(
                    unpacked,
                    device_str=_cuda_device_str(),
                    cancel_check=cancel_check,
                    return_linear=return_linear,
                    retain_device=bool(retain_device),
                )
            elif backend == "pytorch_mps":
                res = gpu_demosaic_pytorch_unpacked(
                    unpacked,
                    device_str="mps",
                    cancel_check=cancel_check,
                    return_linear=return_linear,
                    retain_device=False,
                )
            elif backend == "cupy":
                res = gpu_demosaic_cupy_unpacked(unpacked, cancel_check=cancel_check, return_linear=return_linear)
            elapsed_ms = (time.time() - t_start) * 1000.0
        except Exception as e:
            # DecodeCancelled must be caught BEFORE the generic Exception
            # branch and propagated, not swallowed as a decode failure --
            # it is IS an Exception subclass, so a single broad `except`
            # below would catch it too. The previous string-matching check
            # (`"DecodeCancelled" in str(e)`) never actually matched:
            # DecodeCancelled's message is just the file path (inherited
            # str() from Exception(file_path)), which never contains the
            # literal text "DecodeCancelled" -- so every real cancellation
            # was silently logged as a GPU failure and returned None,
            # defeating the cancel_check calls added to the demosaic
            # bodies (the caller never learns the decode was cancelled and
            # falls through to a rawpy decode nobody wants anymore).
            from fast_raw_decode import DecodeCancelled

            if isinstance(e, DecodeCancelled):
                raise
            logger.warning(f"GPU Processor: Failed to decode RAW on GPU ({backend}): {e}", exc_info=True)
            res = None
    # Outside `with gate:` (lock/semaphore already released): calling this
    # while still holding our own permit made release_cached_gpu_memory()'s
    # "acquire all permits non-blocking, else skip" check always see this
    # thread's own held permit as "a decode in flight" and skip every single
    # time -- the post-decode flush was silently a no-op. Releasing first
    # lets it actually drain the allocator when no other decode is running.
    if res is not None:
        logger.info(
            "GPU Processor: Decoded RAW via %s in %.1fms (bayer=%s)",
            backend,
            elapsed_ms,
            unpacked.pat_str,
        )
        try:
            from perf_metrics import perf_mark

            perf_mark(
                "decode_full_gpu",
                elapsed_ms,
                unpacked.file_path,
                backend=backend,
                mp=res.shape[0] * res.shape[1] / 1e6,
            )
        except Exception:
            pass
        try:
            maybe_release_gpu_memory_after_decode()
        except Exception:
            pass
        return res

    return None
