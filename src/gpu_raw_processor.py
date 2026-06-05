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
    device_str: str = "cuda",
    wb_coeffs: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    m_color: np.ndarray = np.eye(3, dtype=np.float32)
) -> np.ndarray:
    """
    A GPU-accelerated Bilinear Demosaicing implementation using PyTorch.
    Uses torch.nn.functional.conv2d with fixed convolution filters to interpolate Bayer channels.
    """
    import torch
    import torch.nn.functional as F

    device = torch.device(device_str)
    
    # 1. Upload raw array to GPU as float32 and normalize to [0, 1]
    max_val = float(raw_array.max() or 1.0)
    raw_tensor = torch.from_numpy(raw_array.astype(np.float32)).to(device) / max_val
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

    wb_r, wb_g, wb_b, wb_g2 = wb_coeffs

    # Extract raw channels and apply white balance multipliers
    r_raw = raw_tensor * r_mask * wb_r
    
    g_raw = torch.zeros_like(raw_tensor)
    if bayer_pattern in ("RGGB", "BGGR"):
        g_raw[0::2, 1::2] = raw_tensor[0::2, 1::2] * wb_g
        g_raw[1::2, 0::2] = raw_tensor[1::2, 0::2] * wb_g2
    else:
        g_raw[0::2, 0::2] = raw_tensor[0::2, 0::2] * wb_g
        g_raw[1::2, 1::2] = raw_tensor[1::2, 1::2] * wb_g2
        
    b_raw = raw_tensor * b_mask * wb_b
    
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
    
    # Apply Color Matrix Multiplication (maps sensor RGB to sRGB space)
    m_color_tensor = torch.from_numpy(m_color).to(device)
    rgb_srgb = torch.matmul(rgb_tensor, m_color_tensor.t())
    
    # Clamp to [0, 1]
    rgb_srgb = rgb_srgb.clamp(0.0, 1.0)
    
    # Apply Gamma Correction (gamma compression V_out = V_in^(1 / 2.2))
    rgb_gamma = rgb_srgb ** (1.0 / 2.2)
    
    # Convert to uint8 [0, 255]
    rgb_final = (rgb_gamma * 255.0).clamp(0, 255).to(torch.uint8)
    
    # 5. Download result back to CPU
    return rgb_final.cpu().numpy()


def gpu_demosaic_cupy(
    raw_array: np.ndarray, 
    bayer_pattern: str = "RGGB",
    wb_coeffs: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    m_color: np.ndarray = np.eye(3, dtype=np.float32)
) -> np.ndarray:
    """
    A GPU-accelerated demosaicing implementation using CuPy.
    Uses CuPy elementwise/raw CUDA kernels for maximum speed.
    """
    import cupy as cp
    
    # Upload to GPU and normalize to [0, 1]
    max_val = float(raw_array.max() or 1.0)
    raw_gpu = cp.array(raw_array, dtype=cp.float32) / max_val
    h, w = raw_gpu.shape
    
    # Pre-allocate output RGB image on GPU (use float32 temporarily for matrix/gamma processing)
    rgb_gpu = cp.zeros((h, w, 3), dtype=cp.float32)
    
    wb_r, wb_g, wb_b, wb_g2 = wb_coeffs
    
    # Simple CUDA kernel for bilinear demosaicing with white balance
    demosaic_kernel = cp.ElementwiseKernel(
        'raw float32 raw_data, int32 h, int32 w, float32 wb_r, float32 wb_g, float32 wb_b, float32 wb_g2',
        'raw float32 rgb_out',
        '''
        int y = i / w;
        int x = i % w;
        if (y < 1 || y >= h - 1 || x < 1 || x >= w - 1) return;
        
        float r = 0, g = 0, b = 0;
        
        // Simple RGGB bayer interpolation with white balance
        if (y % 2 == 0) {
            if (x % 2 == 0) {
                // R pixel
                r = raw_data[i] * wb_r;
                g = (raw_data[i-1]*wb_g + raw_data[i+1]*wb_g + raw_data[i-w]*wb_g2 + raw_data[i+w]*wb_g2) * 0.25f;
                b = (raw_data[i-w-1] + raw_data[i-w+1] + raw_data[i+w-1] + raw_data[i+w+1]) * 0.25f * wb_b;
            } else {
                // G pixel (R row)
                r = (raw_data[i-1] + raw_data[i+1]) * 0.5f * wb_r;
                g = raw_data[i] * wb_g;
                b = (raw_data[i-w] + raw_data[i+w]) * 0.5f * wb_b;
            }
        } else {
            if (x % 2 == 0) {
                // G pixel (B row)
                r = (raw_data[i-w] + raw_data[i+w]) * 0.5f * wb_r;
                g = raw_data[i] * wb_g2;
                b = (raw_data[i-1] + raw_data[i+1]) * 0.5f * wb_b;
            } else {
                // B pixel
                r = (raw_data[i-w-1] + raw_data[i-w+1] + raw_data[i+w-1] + raw_data[i+w+1]) * 0.25f * wb_r;
                g = (raw_data[i-1]*wb_g + raw_data[i+1]*wb_g + raw_data[i-w]*wb_g2 + raw_data[i+w]*wb_g2) * 0.25f;
                b = raw_data[i] * wb_b;
            }
        }
        
        rgb_out[i * 3 + 0] = r;
        rgb_out[i * 3 + 1] = g;
        rgb_out[i * 3 + 2] = b;
        ''',
        'demosaic_kernel'
    )
    
    demosaic_kernel(raw_gpu, h, w, rgb_gpu)
    
    # Apply Color Matrix Multiplication (maps sensor RGB to sRGB space)
    m_color_gpu = cp.array(m_color, dtype=cp.float32)
    rgb_srgb = cp.dot(rgb_gpu, m_color_gpu.T)
    
    # Clamp to [0, 1]
    rgb_srgb = cp.clip(rgb_srgb, 0.0, 1.0)
    
    # Apply Gamma Correction (gamma compression V_out = V_in^(1 / 2.2))
    rgb_gamma = cp.power(rgb_srgb, 1.0 / 2.2)
    
    # Scale and convert to uint8
    rgb_final = (rgb_gamma * 255.0).clip(0, 255).astype(cp.uint8)
    
    # Download result to CPU
    return cp.asnumpy(rgb_final)


def try_gpu_raw_decode(
    file_path: str, 
    raw_array: np.ndarray, 
    exif_data: Optional[Dict[str, Any]] = None,
    raw_obj: Optional[Any] = None
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

        # Retrieve White Balance Coefficients
        wb_coeffs = (1.0, 1.0, 1.0, 1.0)
        
        # Standard conversion matrix from XYZ to sRGB (D65 white point)
        XYZ_TO_SRGB = np.array([
            [ 3.2406, -1.5372, -0.4986],
            [-0.9689,  1.8758,  0.0415],
            [ 0.0557, -0.2040,  1.0570]
        ], dtype=np.float32)
        
        cam_xyz = None
        
        if raw_obj is not None:
            if hasattr(raw_obj, "camera_whitebalance") and raw_obj.camera_whitebalance:
                wb = list(raw_obj.camera_whitebalance)
                if wb[1] != 0:
                    wb_coeffs = (wb[0]/wb[1], 1.0, wb[2]/wb[1], wb[3]/wb[1] if len(wb) > 3 else 1.0)
            if hasattr(raw_obj, "rgb_xyz_matrix") and raw_obj.rgb_xyz_matrix is not None:
                cam_xyz = raw_obj.rgb_xyz_matrix[:3, :3]
        else:
            import rawpy
            try:
                with rawpy.imread(file_path) as raw:
                    if hasattr(raw, "camera_whitebalance") and raw.camera_whitebalance:
                        wb = list(raw.camera_whitebalance)
                        if wb[1] != 0:
                            wb_coeffs = (wb[0]/wb[1], 1.0, wb[2]/wb[1], wb[3]/wb[1] if len(wb) > 3 else 1.0)
                    if hasattr(raw, "rgb_xyz_matrix") and raw.rgb_xyz_matrix is not None:
                        cam_xyz = raw.rgb_xyz_matrix[:3, :3]
            except Exception:
                pass

        if cam_xyz is None or np.allclose(cam_xyz, 0):
            m_color = np.eye(3, dtype=np.float32)
        else:
            m_color = np.dot(XYZ_TO_SRGB, cam_xyz)

        if backend == "pytorch_cuda":
            res = gpu_demosaic_pytorch(raw_array, bayer_pattern, device_str="cuda", wb_coeffs=wb_coeffs, m_color=m_color)
            logger.info(f"GPU Processor: Decoded RAW via PyTorch CUDA in {(time.time() - t_start)*1000:.1f}ms")
            return res
        elif backend == "pytorch_mps":
            res = gpu_demosaic_pytorch(raw_array, bayer_pattern, device_str="mps", wb_coeffs=wb_coeffs, m_color=m_color)
            logger.info(f"GPU Processor: Decoded RAW via PyTorch MPS in {(time.time() - t_start)*1000:.1f}ms")
            return res
        elif backend == "cupy":
            res = gpu_demosaic_cupy(raw_array, bayer_pattern, wb_coeffs=wb_coeffs, m_color=m_color)
            logger.info(f"GPU Processor: Decoded RAW via CuPy in {(time.time() - t_start)*1000:.1f}ms")
            return res
            
    except Exception as e:
        logger.warning(f"GPU Processor: Failed to decode RAW on GPU ({backend}): {e}", exc_info=True)
        
    return None
