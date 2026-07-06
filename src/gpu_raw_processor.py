import logging
import os
import time
import threading
from typing import Any, Callable, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Global lock to serialize GPU decoding and prevent concurrent CUDA context/VRAM allocation conflicts
_GPU_LOCK = threading.Lock()

# Cache detection results to avoid repeated import attempts
_GPU_BACKEND: Optional[str] = None

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
    try:
        import torch
        if torch.cuda.is_available():
            _GPU_BACKEND = "pytorch_cuda"
            logger.info("GPU Processor: PyTorch CUDA backend detected.")
            return _GPU_BACKEND
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _GPU_BACKEND = "pytorch_mps"
            logger.info("GPU Processor: PyTorch MPS (Apple Silicon) backend detected.")
            return _GPU_BACKEND
    except ImportError:
        pass

    # 2. Check for CuPy (NVIDIA CUDA)
    try:
        import cupy
        # Try creating a device context to ensure CUDA driver is working
        if cupy.cuda.Device().id >= 0:
            _GPU_BACKEND = "cupy"
            logger.info("GPU Processor: CuPy CUDA backend detected.")
            return _GPU_BACKEND
    except (ImportError, Exception):
        pass

    _GPU_BACKEND = "cpu_only"
    return _GPU_BACKEND


def gpu_demosaic_pytorch_unpacked(unpacked, device_str: str = "cuda", cancel_check: Optional[Callable[[], bool]] = None) -> np.ndarray:
    """
    GPU-accelerated Demosaicing using PyTorch and Kornia.
    Consumes UnpackedRaw from fast_raw_decode.py to guarantee color math parity.
    """
    import torch
    import kornia

    device = torch.device(device_str)
    
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
    
    # 4. Apply Color Matrix Multiplication (maps sensor RGB directly to sRGB space)
    m_color_tensor = torch.from_numpy(unpacked.rgb_cam).to(device)
    rgb_srgb = torch.matmul(rgb_tensor, m_color_tensor.t())
    
    # Clamp to [0, 1]
    rgb_srgb = rgb_srgb.clamp(0.0, 1.0)
    
    # 5. Apply BT.709 Gamma Curve (matching fast_raw_decode _gamma_lut8)
    pwr = 1.0 / 2.222
    ts = 4.5
    bnd = [0.0, 1.0]
    g2 = 0.0
    for _ in range(48):
        g2 = (bnd[0] + bnd[1]) / 2
        if ((g2 / ts) ** -pwr - 1) / pwr - 1 / g2 > -1:
            bnd[1] = g2
        else:
            bnd[0] = g2
    g3 = g2 / ts
    g4 = g2 * (1 / pwr - 1)
    
    mask = rgb_srgb < g3
    rgb_gamma = torch.where(
        mask,
        rgb_srgb * ts,
        (1.0 + g4) * torch.pow(torch.clamp(rgb_srgb, min=1e-12), pwr) - g4
    )
    
    # Convert to uint8 [0, 255] (matching curve16 >> 8 semantics)
    rgb_final = torch.clamp(rgb_gamma * 255.0 + 0.5, 0, 255).to(torch.uint8)
    
    # 6. Download result back to CPU
    return rgb_final.cpu().numpy()


def gpu_demosaic_cupy_unpacked(unpacked, cancel_check: Optional[Callable[[], bool]] = None) -> np.ndarray:
    """
    GPU-accelerated demosaicing implementation using CuPy.
    Uses CuPy elementwise/raw CUDA kernels for maximum speed.
    """
    import cupy as cp
    
    # Upload to GPU
    raw_gpu = cp.array(unpacked.mosaic, dtype=cp.float32)
    h, w = raw_gpu.shape
    
    # Pre-allocate normalized array
    raw_norm = cp.empty_like(raw_gpu)
    
    for (dy, dx), ci in np.ndenumerate(unpacked.pattern):
        slc = (slice(dy, None, 2), slice(dx, None, 2))
        black_val = float(unpacked.black[ci])
        scale_val = float(unpacked.scale_mul[ci] / 65535.0)
        
        # Clamp to [0, 1]
        raw_norm[slc] = cp.clip((raw_gpu[slc] - black_val) * scale_val, 0.0, 1.0)

    if unpacked.pat_str != "RGGB":
        raise ValueError(f"CuPy demosaic currently only supports RGGB, got {unpacked.pat_str}")

    # Pre-allocate output RGB image on GPU
    rgb_gpu = cp.zeros((h, w, 3), dtype=cp.float32)
    
    # Simple CUDA kernel for bilinear demosaicing on pre-normalized and WB-scaled array
    demosaic_kernel = cp.ElementwiseKernel(
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
    
    # Apply Color Matrix Multiplication
    m_color_gpu = cp.array(unpacked.rgb_cam, dtype=cp.float32)
    rgb_srgb = cp.dot(rgb_gpu, m_color_gpu.T)
    
    # Clamp to [0, 1]
    rgb_srgb = cp.clip(rgb_srgb, 0.0, 1.0)
    
    # BT.709 Gamma Curve
    pwr = 1.0 / 2.222
    ts = 4.5
    bnd = [0.0, 1.0]
    g2 = 0.0
    for _ in range(48):
        g2 = (bnd[0] + bnd[1]) / 2
        if ((g2 / ts) ** -pwr - 1) / pwr - 1 / g2 > -1:
            bnd[1] = g2
        else:
            bnd[0] = g2
    g3 = g2 / ts
    g4 = g2 * (1 / pwr - 1)
    
    mask = rgb_srgb < g3
    rgb_gamma = cp.where(
        mask,
        rgb_srgb * ts,
        (1.0 + g4) * cp.power(cp.maximum(rgb_srgb, 1e-12), pwr) - g4
    )
    
    # Scale and convert to uint8
    rgb_final = cp.clip(rgb_gamma * 255.0 + 0.5, 0, 255).astype(cp.uint8)
    
    # Download result to CPU
    return cp.asnumpy(rgb_final)


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
        
    global _GPU_LOCK
    with _GPU_LOCK:
        if cancel_check is not None and cancel_check():
            from fast_raw_decode import DecodeCancelled
            raise DecodeCancelled(unpacked.file_path)
            
        t_start = time.time()
        try:
            if backend == "cupy" and unpacked.pat_str != "RGGB":
                logger.warning(
                    "GPU Processor: CuPy demosaic only supports RGGB; trying PyTorch for %s.",
                    unpacked.pat_str,
                )
                try:
                    import torch
                    if torch.cuda.is_available():
                        backend = "pytorch_cuda"
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        backend = "pytorch_mps"
                    else:
                        return None
                except ImportError:
                    return None

            if backend == "pytorch_cuda":
                res = gpu_demosaic_pytorch_unpacked(
                    unpacked, device_str="cuda", cancel_check=cancel_check
                )
                logger.info(
                    "GPU Processor: Decoded RAW via PyTorch CUDA in %.1fms (bayer=%s)",
                    (time.time() - t_start) * 1000.0,
                    unpacked.pat_str,
                )
                return res
            elif backend == "pytorch_mps":
                res = gpu_demosaic_pytorch_unpacked(
                    unpacked, device_str="mps", cancel_check=cancel_check
                )
                logger.info(
                    "GPU Processor: Decoded RAW via PyTorch MPS in %.1fms (bayer=%s)",
                    (time.time() - t_start) * 1000.0,
                    unpacked.pat_str,
                )
                return res
            elif backend == "cupy":
                res = gpu_demosaic_cupy_unpacked(
                    unpacked, cancel_check=cancel_check
                )
                logger.info(
                    "GPU Processor: Decoded RAW via CuPy in %.1fms (bayer=%s)",
                    (time.time() - t_start) * 1000.0,
                    unpacked.pat_str,
                )
                return res
                
        except Exception as e:
            if "DecodeCancelled" in str(e):
                from fast_raw_decode import DecodeCancelled
                raise DecodeCancelled(unpacked.file_path)
            logger.warning(f"GPU Processor: Failed to decode RAW on GPU ({backend}): {e}", exc_info=True)
            
        return None
