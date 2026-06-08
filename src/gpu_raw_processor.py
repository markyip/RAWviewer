import logging
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Cache detection results to avoid repeated import attempts
_GPU_BACKEND: Optional[str] = None

def detect_gpu_backend() -> str:
    """
    Detect if the system has PyTorch (with CUDA or MPS) or CuPy installed.
    Returns: 'pytorch_cuda', 'pytorch_mps', 'cupy', or 'cpu_only'
    """
    global _GPU_BACKEND
    if _GPU_BACKEND is not None:
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


def gpu_demosaic_pytorch(
    raw_array: np.ndarray, 
    bayer_pattern: str = "RGGB", 
    device_str: str = "cuda"
) -> np.ndarray:
    """
    A proof-of-concept GPU-accelerated Bilinear Demosaicing implementation using PyTorch.
    Uses torch.nn.functional.conv2d with fixed convolution filters to interpolate Bayer channels.
    """
    import torch
    import torch.nn.functional as F

    device = torch.device(device_str)
    
    # 1. Upload raw array to GPU as float32
    raw_tensor = torch.from_numpy(raw_array.astype(np.float32)).to(device)
    h, w = raw_tensor.shape
    
    # Create channel masks based on the Bayer pattern
    # Assuming standard 2x2 repeating block:
    # "RGGB":
    #   Row 0: R G
    #   Row 1: G B
    r_mask = torch.zeros((h, w), dtype=torch.float32, device=device)
    g_mask = torch.zeros((h, w), dtype=torch.float32, device=device)
    b_mask = torch.zeros((h, w), dtype=torch.float32, device=device)
    
    if bayer_pattern == "RGGB":
        # R at (even, even)
        r_mask[0::2, 0::2] = 1.0
        # G at (even, odd) and (odd, even)
        g_mask[0::2, 1::2] = 1.0
        g_mask[1::2, 0::2] = 1.0
        # B at (odd, odd)
        b_mask[1::2, 1::2] = 1.0
    elif bayer_pattern == "BGGR":
        b_mask[0::2, 0::2] = 1.0
        g_mask[0::2, 1::2] = 1.0
        g_mask[1::2, 0::2] = 1.0
        r_mask[1::2, 1::2] = 1.0
    elif bayer_pattern == "GRBG":
        g_mask[0::2, 0::2] = 1.0
        r_mask[0::2, 1::2] = 1.0
        b_mask[1::2, 0::2] = 1.0
        g_mask[1::2, 1::2] = 1.0
    elif bayer_pattern == "GBRG":
        g_mask[0::2, 0::2] = 1.0
        b_mask[0::2, 1::2] = 1.0
        r_mask[1::2, 0::2] = 1.0
        g_mask[1::2, 1::2] = 1.0

    # Extract raw channels
    r_raw = raw_tensor * r_mask
    g_raw = raw_tensor * g_mask
    b_raw = raw_tensor * b_mask
    
    # 2. Setup interpolation filters for bilinear demosaicing
    # G-channel filter: cross shape for horizontal/vertical neighbors
    g_filter = torch.tensor([[0.0, 0.25, 0.0],
                             [0.25, 1.0, 0.25],
                             [0.0, 0.25, 0.0]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
                             
    # R/B-channel filter: box shape for diagonal/cardinal neighbors
    rb_filter = torch.tensor([[0.25, 0.5, 0.25],
                              [0.5,  1.0, 0.5],
                              [0.25, 0.5, 0.25]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

    # 3. Reshape for conv2d: (batch_size, channels, height, width)
    r_raw = r_raw.view(1, 1, h, w)
    g_raw = g_raw.view(1, 1, h, w)
    b_raw = b_raw.view(1, 1, h, w)
    
    # 4. Perform GPU Convolutions
    r_interp = F.conv2d(r_raw, rb_filter, padding=1)
    g_interp = F.conv2d(g_raw, g_filter, padding=1)
    b_interp = F.conv2d(b_raw, rb_filter, padding=1)
    
    # Stack channels to RGB (1, 3, h, w) -> squeeze and permute to (h, w, 3)
    rgb_tensor = torch.cat([r_interp, g_interp, b_interp], dim=1).squeeze(0).permute(1, 2, 0)
    
    # Normalize to uint8 (clipping extreme values)
    # Note: RAW arrays are often 12-bit/14-bit (max 4095/16383).
    # Real pipeline uses white balance multipliers and camera black levels.
    max_val = float(raw_array.max() or 1.0)
    rgb_tensor = (rgb_tensor / max_val * 255.0).clamp(0, 255).to(torch.uint8)
    
    # 5. Download result back to CPU
    return rgb_tensor.cpu().numpy()


def gpu_demosaic_cupy(raw_array: np.ndarray, bayer_pattern: str = "RGGB") -> np.ndarray:
    """
    A proof-of-concept GPU-accelerated demosaicing implementation using CuPy.
    Uses CuPy elementwise/raw CUDA kernels for maximum speed.
    """
    import cupy as cp
    
    # Upload to GPU
    raw_gpu = cp.array(raw_array, dtype=cp.float32)
    h, w = raw_gpu.shape
    
    # Pre-allocate output RGB image on GPU
    rgb_gpu = cp.zeros((h, w, 3), dtype=cp.uint8)
    
    # Simple CUDA kernel for bilinear demosaicing
    # For proof of concept, we implement a fast elementwise kernel in CUDA C++
    demosaic_kernel = cp.ElementwiseKernel(
        'raw float32 raw_data, int32 h, int32 w',
        'raw uint8 rgb_out',
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
        
        // Simple normalization (clamping output)
        rgb_out[i * 3 + 0] = (uint8_t)fmaxf(0.0f, fminf(255.0f, r / 64.0f));
        rgb_out[i * 3 + 1] = (uint8_t)fmaxf(0.0f, fminf(255.0f, g / 64.0f));
        rgb_out[i * 3 + 2] = (uint8_t)fmaxf(0.0f, fminf(255.0f, b / 64.0f));
        ''',
        'demosaic_kernel'
    )
    
    demosaic_kernel(raw_gpu, h, w, rgb_gpu)
    
    # Download result to CPU
    return cp.asnumpy(rgb_gpu)


def try_gpu_raw_decode(
    file_path: str, 
    raw_array: np.ndarray, 
    exif_data: Optional[Dict[str, Any]] = None
) -> Optional[np.ndarray]:
    """
    Attempt to decode the RAW image using the detected GPU backend.
    Falls back to None if GPU is not available or if processing errors occur.
    """
    backend = detect_gpu_backend()
    if backend == "cpu_only":
        return None
        
    t_start = time.time()
    try:
        # Determine bayer pattern (default to RGGB)
        bayer_pattern = "RGGB"
        if exif_data and exif_data.get("exif_data"):
            # Some raw formats store sensor layout/bayer filter pattern in EXIF
            pattern = exif_data["exif_data"].get("CFAPattern")
            if pattern:
                bayer_pattern = str(pattern)

        if backend == "pytorch_cuda":
            res = gpu_demosaic_pytorch(raw_array, bayer_pattern, device_str="cuda")
            logger.info(f"GPU Processor: Decoded RAW via PyTorch CUDA in {(time.time() - t_start)*1000:.1f}ms")
            return res
        elif backend == "pytorch_mps":
            res = gpu_demosaic_pytorch(raw_array, bayer_pattern, device_str="mps")
            logger.info(f"GPU Processor: Decoded RAW via PyTorch MPS in {(time.time() - t_start)*1000:.1f}ms")
            return res
        elif backend == "cupy":
            res = gpu_demosaic_cupy(raw_array, bayer_pattern)
            logger.info(f"GPU Processor: Decoded RAW via CuPy in {(time.time() - t_start)*1000:.1f}ms")
            return res
            
    except Exception as e:
        logger.warning(f"GPU Processor: Failed to decode RAW on GPU ({backend}): {e}", exc_info=True)
        
    return None
