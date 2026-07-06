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


def _gpu_concurrency() -> int:
    try:
        return max(1, min(4, int(os.environ.get("RAWVIEWER_GPU_CONCURRENCY", "2"))))
    except ValueError:
        return 2


_GPU_SEMAPHORE = threading.Semaphore(_gpu_concurrency())

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


def gpu_demosaic_pytorch_unpacked(unpacked, device_str: str = "cuda", cancel_check: Optional[Callable[[], bool]] = None) -> np.ndarray:
    """
    GPU-accelerated Demosaicing using PyTorch and Kornia.
    Consumes UnpackedRaw from fast_raw_decode.py to guarantee color math parity.
    """
    device = torch.device(device_str)

    # CUDA: run this decode on its own stream so decodes issued from
    # different worker threads (semaphore allows up to _gpu_concurrency())
    # overlap upload/compute instead of queueing on the default stream.
    # The final .cpu() copy synchronizes the stream before returning.
    stream_ctx = (
        torch.cuda.stream(torch.cuda.Stream())
        if device.type == "cuda"
        else _NullCtx()
    )
    with stream_ctx:
        return _gpu_demosaic_pytorch_body(unpacked, device, cancel_check)


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


def _gpu_demosaic_pytorch_body(unpacked, device, cancel_check=None) -> np.ndarray:
    def _abort_if_cancelled() -> None:
        if cancel_check is not None and cancel_check():
            from fast_raw_decode import DecodeCancelled

            raise DecodeCancelled(unpacked.file_path)

    # 1. Upload raw array to GPU as float32
    raw_tensor = torch.from_numpy(unpacked.mosaic.astype(np.float32)).to(device)
    h, w = raw_tensor.shape

    # Pre-allocate normalized array
    raw_norm = torch.empty_like(raw_tensor)

    # Apply dcraw scale_colors exactly: (raw - black) * (scale_mul / 65535.0), clipped to [0, 1]
    # We iterate over the 2x2 CFA pattern offsets to apply per-channel scaling
    for (dy, dx), ci in np.ndenumerate(unpacked.pattern):
        slc = (slice(dy, None, 2), slice(dx, None, 2))
        black_val = float(unpacked.black[ci])
        scale_val = float(unpacked.scale_mul[ci] / 65535.0)

        # Clamp after scale guarantees [0.0, 1.0] and handles highlights properly
        raw_norm[slc] = torch.clamp((raw_tensor[slc] - black_val) * scale_val, 0.0, 1.0)

    _abort_if_cancelled()

    # Map the CFA string to Kornia's enum.
    # Note: Kornia's naming expects the 2x2 block starting at the origin.
    cfa_map = {
        "RGGB": kornia.color.CFA.BG,
        "BGGR": kornia.color.CFA.RG,
        "GRBG": kornia.color.CFA.GB,
        "GBRG": kornia.color.CFA.GR
    }
    cfa = cfa_map.get(unpacked.pat_str, kornia.color.CFA.BG)

    # 2. Reshape for Kornia: (batch_size=1, channels=1, height, width)
    raw_input = raw_norm.view(1, 1, h, w)

    # 3. Kornia Demosaicing
    rgb_tensor = kornia.color.raw_to_rgb(raw_input, cfa)

    # Squeeze and permute to (H, W, 3)
    rgb_tensor = rgb_tensor.squeeze(0).permute(1, 2, 0)

    _abort_if_cancelled()

    # 4. Apply Color Matrix Multiplication (maps sensor RGB directly to sRGB space)
    m_color_tensor = torch.from_numpy(unpacked.rgb_cam).to(device)
    rgb_srgb = torch.matmul(rgb_tensor, m_color_tensor.t())

    # Clamp to [0, 1]
    rgb_srgb = rgb_srgb.clamp(0.0, 1.0)

    _abort_if_cancelled()

    # 5. BT.709 gamma via the shared 16-bit LUT (same curve as
    # fast_raw_decode._gamma_lut8 / gamma_curve16, computed once and cached
    # per-device instead of re-derived by bisection on every call).
    gamma_lut = _gpu_gamma_lut(device, torch)
    idx = torch.clamp(rgb_srgb * 65535.0 + 0.5, 0, 65535).to(torch.int64)
    rgb_gamma16 = gamma_lut[idx]  # gather: 16-bit-equivalent curve value

    # Convert to uint8 (curve16 >> 8 semantics, matching the CPU LUT path)
    rgb_final = torch.clamp(rgb_gamma16 / 256.0, 0, 255).to(torch.uint8)

    _abort_if_cancelled()

    # 6. Download result back to CPU (synchronizes the stream)
    return rgb_final.cpu().numpy()


_GAMMA_LUT_CUPY: Optional[Any] = None


def _cupy_gamma_lut() -> Any:
    """CuPy-resident copy of the shared 16-bit gamma curve (see _gpu_gamma_lut)."""
    global _GAMMA_LUT_CUPY
    if _GAMMA_LUT_CUPY is None:
        from fast_raw_decode import gamma_curve16

        _GAMMA_LUT_CUPY = cupy.array(gamma_curve16().astype(np.float32))
    return _GAMMA_LUT_CUPY


def gpu_demosaic_cupy_unpacked(unpacked, cancel_check: Optional[Callable[[], bool]] = None) -> np.ndarray:
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

    if unpacked.pat_str != "RGGB":
        raise ValueError(f"CuPy demosaic currently only supports RGGB, got {unpacked.pat_str}")

    # Pre-allocate output RGB image on GPU
    rgb_gpu = cupy.zeros((h, w, 3), dtype=cupy.float32)
    
    # Simple CUDA kernel for bilinear demosaicing on pre-normalized and WB-scaled array
    demosaic_kernel = cupy.ElementwiseKernel(
        'raw float32 raw_data, int32 h, int32 w',
        'raw float32 rgb_out',
        '''
        int y = i / w;
        int x = i % w;
        if (y < 1 || y >= h - 1 || x < 1 || x >= w - 1) return;
        
        float r = 0, g = 0, b = 0;
        
        // Simple RGGB bayer interpolation
        if (y % 2 == 0) {
            if (x % 2 == 0) {
                // R pixel
                r = raw_data[i];
                g = (raw_data[i-1] + raw_data[i+1] + raw_data[i-w] + raw_data[i+w]) * 0.25f;
                b = (raw_data[i-w-1] + raw_data[i-w+1] + raw_data[i+w-1] + raw_data[i+w+1]) * 0.25f;
            } else {
                // G pixel (R row)
                r = (raw_data[i-1] + raw_data[i+1]) * 0.5f;
                g = raw_data[i];
                b = (raw_data[i-w] + raw_data[i+w]) * 0.5f;
            }
        } else {
            if (x % 2 == 0) {
                // G pixel (B row)
                r = (raw_data[i-w] + raw_data[i+w]) * 0.5f;
                g = raw_data[i];
                b = (raw_data[i-1] + raw_data[i+1]) * 0.5f;
            } else {
                // B pixel
                r = (raw_data[i-w-1] + raw_data[i-w+1] + raw_data[i+w-1] + raw_data[i+w+1]) * 0.25f;
                g = (raw_data[i-1] + raw_data[i+1] + raw_data[i-w] + raw_data[i+w]) * 0.25f;
                b = raw_data[i];
            }
        }
        
        rgb_out[i * 3 + 0] = r;
        rgb_out[i * 3 + 1] = g;
        rgb_out[i * 3 + 2] = b;
        ''',
        'demosaic_kernel'
    )
    
    demosaic_kernel(raw_norm, h, w, rgb_gpu)

    _abort_if_cancelled()

    # Apply Color Matrix Multiplication
    m_color_gpu = cupy.array(unpacked.rgb_cam, dtype=cupy.float32)
    rgb_srgb = cupy.dot(rgb_gpu, m_color_gpu.T)

    # Clamp to [0, 1]
    rgb_srgb = cupy.clip(rgb_srgb, 0.0, 1.0)

    _abort_if_cancelled()

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
    cancel_check: Optional[Callable[[], bool]] = None
) -> Optional[np.ndarray]:
    """
    Attempt to decode the UnpackedRaw using the detected GPU backend.
    Falls back to None if GPU is not available or if processing errors occur.
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

    if backend == "cupy" and unpacked.pat_str != "RGGB":
        logger.warning(
            "GPU Processor: CuPy demosaic only supports RGGB; trying PyTorch for %s.",
            unpacked.pat_str,
        )
        if _HAS_TORCH:
            if torch.cuda.is_available():
                backend = "pytorch_cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                backend = "pytorch_mps"
            else:
                return None
        else:
            return None

    # PyTorch backends run under the bounded semaphore (parallel decodes,
    # streams overlap upload/compute); CuPy keeps the hard lock -- see the
    # concurrency-control comment at the top of this module.
    gate = _GPU_LOCK if backend == "cupy" else _GPU_SEMAPHORE
    with gate:
        if cancel_check is not None and cancel_check():
            from fast_raw_decode import DecodeCancelled
            raise DecodeCancelled(unpacked.file_path)

        t_start = time.time()
        try:
            res = None
            if backend == "pytorch_cuda":
                res = gpu_demosaic_pytorch_unpacked(
                    unpacked, device_str="cuda", cancel_check=cancel_check
                )
            elif backend == "pytorch_mps":
                res = gpu_demosaic_pytorch_unpacked(
                    unpacked, device_str="mps", cancel_check=cancel_check
                )
            elif backend == "cupy":
                res = gpu_demosaic_cupy_unpacked(unpacked, cancel_check=cancel_check)
            if res is not None:
                elapsed_ms = (time.time() - t_start) * 1000.0
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
                return res
                
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

        return None
