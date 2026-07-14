"""CUDA ↔ OpenGL display bridge (Phase 2b).

Route today
-----------
* Fused GPU ISP downloads uint8 RGB into a **reused pinned** host buffer
  (``gpu_raw_processor._download_device_rgb``).
* With ``RAWVIEWER_GPU_GL_TEX=1``, ``GpuImageView.set_rgb_numpy`` paints that
  buffer via ``QPainter.drawImage`` on the OpenGL viewport — **no QPixmap /
  PixmapConverter**. Qt uploads a GL texture from the QImage.
* Default path (flag off) remains: numpy → ``QPixmap`` → ``set_pixmap``.

True zero-copy CUDA↔GL (still future)
-------------------------------------
Requires owning a texture id on the GUI thread and
``cudaGraphicsGLRegisterImage`` from a worker-visible device buffer (IPC or
deferred GUI copy). See earlier notes in this module's history; surface APIs
below are the stable extension points.
"""

from __future__ import annotations

from typing import Any, Optional


def cuda_gl_interop_available() -> bool:
    """True when CUDA is present (necessary but not sufficient for interop)."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def gl_tex_display_enabled() -> bool:
    """Host-side OpenGL RGB paint without QPixmap (``RAWVIEWER_GPU_GL_TEX=1``)."""
    import os

    return str(os.environ.get("RAWVIEWER_GPU_GL_TEX", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def download_for_pixmap_path(rgb_device: Any, *, as_uint16: bool = False) -> Optional[Any]:
    """Pinned D2H helper mirroring the ISP download path (numpy owned copy)."""
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
