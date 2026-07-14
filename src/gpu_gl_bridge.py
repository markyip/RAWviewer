"""CUDA ↔ OpenGL display bridge (Phase 2b scaffold).

Current single-view Route B still paints via ``QGraphicsView`` +
``QGraphicsPixmapItem.setPixmap(QPixmap)``. That path **requires** a host
buffer before Qt uploads a GL texture, so true zero-copy CUDA→screen is not
possible without replacing the pixmap item with a custom GL viewport.

What RAWviewer does today for the develop hot path
-------------------------------------------------
* Fused GPU ISP stays on device until gamma produces uint8 RGB.
* Download uses a **reused pinned-host** buffer (``_download_device_rgb`` in
  ``gpu_raw_processor.py``) instead of pageable ``tensor.cpu().numpy()``.
* Display continues: numpy → ``PixmapConverter`` / ``QPixmap`` →
  ``GpuImageView.set_pixmap``.

What would be needed for real CUDA↔GL
-------------------------------------
1. Own a ``QOpenGLWidget`` (or share the graphics-view viewport context on
   the GUI thread) and create an RGB8 texture of sensor size.
2. On the GUI thread (holding that context):
   ``cudaGraphicsGLRegisterImage`` → map → ``cudaMemcpy2DToArray`` / 3D copy
   from the develop stream's uint8 device buffer → unmap / bind / draw.
3. Decode workers must **not** touch the GL context: pass a CUDA IPC handle
   or wait and enqueue the copy from the GUI thread after ``image_ready``.
4. Fallback to pinned D2H + ``set_pixmap`` when GL context is unavailable
   (``RAWVIEWER_GPU_VIEW_NO_GL``, macOS MPS, Intel iGPU-only, etc.).

Until that viewport rewrite lands, call :func:`download_for_pixmap_path` from
debug scripts if you need the same pinned staging outside the ISP module.
"""

from __future__ import annotations

from typing import Any, Optional


def cuda_gl_interop_available() -> bool:
    """True when a future custom GL viewport could attempt interop (CUDA only)."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def download_for_pixmap_path(rgb_device: Any, *, as_uint16: bool = False) -> Optional[Any]:
    """Pinned D2H helper mirroring the ISP download path (numpy owned copy).

    ``rgb_device`` must be a contiguous CUDA (or CPU) torch tensor shaped HxWx3.
    Returns ``numpy.ndarray`` or None on failure.
    """
    try:
        import torch
        from gpu_raw_processor import _download_device_rgb, _workspace_for

        if not isinstance(rgb_device, torch.Tensor):
            return None
        device = rgb_device.device
        h, w = int(rgb_device.shape[0]), int(rgb_device.shape[1])
        ws = _workspace_for(device)
        ws.ensure(h, w)
        return _download_device_rgb(rgb_device, device, ws, as_uint16=as_uint16)
    except Exception:
        return None
