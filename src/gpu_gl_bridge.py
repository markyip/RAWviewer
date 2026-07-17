"""CUDA ↔ OpenGL display bridge (Phase 2b / 2c).

Routes
------
* **Phase 2b** (``RAWVIEWER_GPU_GL_TEX=1``): pinned D2H → ``GpuImageView.set_rgb_numpy``
  (``QPainter.drawImage``; Qt uploads a GL texture; no ``QPixmap``).
* **Phase 2c** (``RAWVIEWER_GPU_CUDA_GL=1``): keep uint8 RGB on CUDA as
  ``DeviceRgb``, register a GUI-thread GL texture with
  ``cudaGraphicsGLRegisterImage``, ``cudaMemcpy2DToArray`` into that texture,
  and paint via native OpenGL (no host download for display). Falls back to
  Phase 2b / pixmap on any failure.

True ISP→GL surface write without an intermediate device RGB buffer is left for
a later phase. CTA: GUI thread only for GL / ``cudaGraphics*``.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# cudaGraphicsRegisterFlagsWriteDiscard
_CUDA_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 0x02
_CUDA_SUCCESS = 0


def _env_true(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def gl_tex_display_enabled() -> bool:
    """Host-side OpenGL RGB paint without QPixmap (``RAWVIEWER_GPU_GL_TEX=1``)."""
    return _env_true("RAWVIEWER_GPU_GL_TEX")


def cuda_gl_enabled() -> bool:
    """Opt-in CUDA↔GL zero-copy display (``RAWVIEWER_GPU_CUDA_GL=1``)."""
    return _env_true("RAWVIEWER_GPU_CUDA_GL")


def cuda_gl_interop_available() -> bool:
    """True when CUDA + cudart load succeed (not sufficient alone for interop)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        return _load_cudart() is not None
    except Exception:
        return False


@dataclass
class DeviceRgb:
    """CUDA-resident HxWx3 uint8 sRGB (sensor / post-orientation).

    ``shape`` / ``dtype`` mirror numpy so existing display bookkeeping that
    only reads dimensions keeps working. Call ``to_numpy()`` when a host copy
    is required (cache, export, histogram, fallback).
    """

    tensor: Any  # torch.Tensor on CUDA
    file_path: str = ""
    generation: int = 0
    _host_cache: Any = field(default=None, repr=False, compare=False)

    @property
    def shape(self) -> tuple[int, ...]:
        t = self.tensor
        return (int(t.shape[0]), int(t.shape[1]), int(t.shape[2]))

    @property
    def dtype(self):
        import numpy as np

        return np.uint8

    @property
    def height(self) -> int:
        return int(self.tensor.shape[0])

    @property
    def width(self) -> int:
        return int(self.tensor.shape[1])

    def is_cuda(self) -> bool:
        try:
            return bool(self.tensor.is_cuda)
        except Exception:
            return False

    def synchronize(self) -> None:
        try:
            import torch

            if self.tensor.is_cuda:
                torch.cuda.current_stream(self.tensor.device).synchronize()
        except Exception:
            pass

    def to_numpy(self):
        """Pinned D2H owned copy (or cached host buffer)."""
        if self._host_cache is not None:
            return self._host_cache
        try:
            from gpu_raw_processor import _download_device_rgb, _workspace_for

            device = self.tensor.device
            h, w = self.height, self.width
            ws = _workspace_for(device)
            ws.ensure(h, w)
            arr = _download_device_rgb(self.tensor, device, ws, as_uint16=False)
            self._host_cache = arr
            return arr
        except Exception:
            arr = self.tensor.detach().contiguous().cpu().numpy()
            self._host_cache = arr
            return arr


def apply_orientation_device_rgb(dev: DeviceRgb, orientation: int) -> DeviceRgb:
    """EXIF orientation on a CUDA HxWx3 tensor (1 = identity)."""
    o = int(orientation or 1)
    if o == 1 or not dev.is_cuda():
        return dev
    t = dev.tensor
    # Matches common EXIF orientation matrix used elsewhere in the app.
    if o == 2:  # mirror horizontal
        t = t.flip(1)
    elif o == 3:  # rotate 180
        t = t.flip(0).flip(1)
    elif o == 4:  # mirror vertical
        t = t.flip(0)
    elif o == 5:  # mirror horizontal + rotate 270 CW
        t = t.flip(1).transpose(0, 1)
    elif o == 6:  # rotate 90 CW
        t = t.transpose(0, 1).flip(1)
    elif o == 7:  # mirror horizontal + rotate 90 CW
        t = t.flip(1).transpose(0, 1).flip(1)
    elif o == 8:  # rotate 270 CW
        t = t.transpose(0, 1).flip(0)
    else:
        return dev
    return DeviceRgb(
        tensor=t.contiguous(),
        file_path=dev.file_path,
        generation=dev.generation,
    )


# ---------------------------------------------------------------------------
# cudart (ctypes)
# ---------------------------------------------------------------------------

_CUDART = None
_CUDART_FAIL = False


def _cudart_path() -> Optional[str]:
    try:
        import torch

        root = os.path.dirname(torch.__file__)
        candidates = [
            os.path.join(root, "lib", "cudart64_12.dll"),
            os.path.join(root, "lib", "cudart64_11.dll"),
            os.path.join(root, "bin", "cudart64_12.dll"),
            os.path.join(root, "lib", "libcudart.so.12"),
            os.path.join(root, "lib", "libcudart.so.11"),
            os.path.join(root, "lib", "libcudart.so"),
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
    except Exception:
        pass
    return None


def _load_cudart():
    global _CUDART, _CUDART_FAIL
    if _CUDART is not None:
        return _CUDART
    if _CUDART_FAIL:
        return None
    path = _cudart_path()
    if not path:
        _CUDART_FAIL = True
        return None
    try:
        lib = ctypes.CDLL(path)
        # cudaError_t cudaGraphicsGLRegisterImage(cudaGraphicsResource_t*, GLuint, GLenum, unsigned int)
        lib.cudaGraphicsGLRegisterImage.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
        ]
        lib.cudaGraphicsGLRegisterImage.restype = ctypes.c_int
        lib.cudaGraphicsUnregisterResource.argtypes = [ctypes.c_void_p]
        lib.cudaGraphicsUnregisterResource.restype = ctypes.c_int
        lib.cudaGraphicsMapResources.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        lib.cudaGraphicsMapResources.restype = ctypes.c_int
        lib.cudaGraphicsUnmapResources.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        lib.cudaGraphicsUnmapResources.restype = ctypes.c_int
        lib.cudaGraphicsSubResourceGetMappedArray.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
        ]
        lib.cudaGraphicsSubResourceGetMappedArray.restype = ctypes.c_int
        lib.cudaMemcpy2DToArray.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        lib.cudaMemcpy2DToArray.restype = ctypes.c_int
        # cudaMemcpyDeviceToDevice = 3
        _CUDART = lib
        return lib
    except Exception as e:
        logger.debug("cudart load failed: %s", e)
        _CUDART_FAIL = True
        return None


def _cuda_check(err: int, what: str) -> None:
    if err != _CUDA_SUCCESS:
        raise RuntimeError(f"{what} failed: cudaError={err}")


@dataclass
class CudaGlTextureSlot:
    """GUI-thread-owned GL texture registered with CUDA."""

    tex_id: int = 0
    resource: int = 0  # cudaGraphicsResource_t as int
    width: int = 0
    height: int = 0
    rgba: bool = True

    def release(self, gl_delete_fn=None) -> None:
        lib = _load_cudart()
        if self.resource and lib is not None:
            try:
                lib.cudaGraphicsUnregisterResource(ctypes.c_void_p(self.resource))
            except Exception:
                pass
            self.resource = 0
        if self.tex_id and gl_delete_fn is not None:
            try:
                gl_delete_fn(self.tex_id)
            except Exception:
                pass
        self.tex_id = 0
        self.width = self.height = 0


def ensure_rgba_u8(dev: DeviceRgb) -> Any:
    """Return contiguous CUDA HxWx4 uint8 (alpha=255), allocating only if needed."""
    import torch

    t = dev.tensor
    if t.ndim != 3 or t.shape[2] < 3:
        raise ValueError(f"DeviceRgb expected HxWx3+, got {tuple(t.shape)}")
    t = t[..., :3].contiguous()
    if t.dtype != torch.uint8:
        t = t.clamp(0, 255).to(torch.uint8)
    alpha = torch.full(
        (t.shape[0], t.shape[1], 1), 255, dtype=torch.uint8, device=t.device
    )
    return torch.cat([t, alpha], dim=2).contiguous()


def register_gl_texture(
    tex_id: int,
    *,
    target: int = 0x0DE1,  # GL_TEXTURE_2D
) -> int:
    """Register an existing GL texture; returns cudaGraphicsResource handle."""
    lib = _load_cudart()
    if lib is None:
        raise RuntimeError("cudart unavailable")
    res = ctypes.c_void_p()
    err = lib.cudaGraphicsGLRegisterImage(
        ctypes.byref(res),
        ctypes.c_uint(int(tex_id)),
        ctypes.c_uint(int(target)),
        ctypes.c_uint(_CUDA_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD),
    )
    _cuda_check(err, "cudaGraphicsGLRegisterImage")
    return int(res.value or 0)


def upload_device_rgba_to_gl(
    slot: CudaGlTextureSlot,
    rgba_device: Any,
    *,
    stream_ptr: int = 0,
) -> None:
    """Map registered texture and ``cudaMemcpy2DToArray`` from device RGBA uint8."""
    lib = _load_cudart()
    if lib is None or not slot.resource:
        raise RuntimeError("CUDA-GL resource missing")
    import torch

    if not isinstance(rgba_device, torch.Tensor) or not rgba_device.is_cuda:
        raise TypeError("rgba_device must be a CUDA torch.Tensor")
    rgba = rgba_device.contiguous()
    h, w, c = int(rgba.shape[0]), int(rgba.shape[1]), int(rgba.shape[2])
    if c < 4:
        raise ValueError("expected HxWx4 RGBA")
    if (w, h) != (slot.width, slot.height):
        raise ValueError(
            f"size mismatch texture {slot.width}x{slot.height} vs tensor {w}x{h}"
        )

    rgba.device  # noqa: B018 — keep tensor live
    # Sync only the producer stream (not torch.cuda.synchronize() device-wide)
    # so CUDA-GL map sees finished kernels without stalling unrelated work.
    torch.cuda.current_stream(rgba.device).synchronize()

    res_arr = (ctypes.c_void_p * 1)(ctypes.c_void_p(slot.resource))
    cu_stream = ctypes.c_void_p(stream_ptr) if stream_ptr else ctypes.c_void_p(None)
    _cuda_check(
        lib.cudaGraphicsMapResources(1, res_arr, cu_stream),
        "cudaGraphicsMapResources",
    )
    try:
        array = ctypes.c_void_p()
        _cuda_check(
            lib.cudaGraphicsSubResourceGetMappedArray(
                ctypes.byref(array),
                ctypes.c_void_p(slot.resource),
                0,
                0,
            ),
            "cudaGraphicsSubResourceGetMappedArray",
        )
        pitch = w * 4
        # cudaMemcpyDeviceToDevice = 3
        _cuda_check(
            lib.cudaMemcpy2DToArray(
                array,
                0,
                0,
                ctypes.c_void_p(int(rgba.data_ptr())),
                pitch,
                pitch,
                h,
                3,
            ),
            "cudaMemcpy2DToArray",
        )
    finally:
        _cuda_check(
            lib.cudaGraphicsUnmapResources(1, res_arr, cu_stream),
            "cudaGraphicsUnmapResources",
        )


def gl_tex_image_2d_rgba_empty(width: int, height: int) -> None:
    """Allocate an empty GL_RGBA8 texture (NULL pixels) while context is current.

    PyQt's ``glTexImage2D`` rejects a null ``sip.voidptr``. On Windows,
    ``opengl32.dll`` exports are only reliable for the legacy 1.1 ICD path;
    resolve ``glTexImage2D`` via the *current* context's ``getProcAddress``.
    """
    GL_TEXTURE_2D = 0x0DE1
    GL_RGBA = 0x1908
    GL_UNSIGNED_BYTE = 0x1401
    from PyQt6.QtGui import QOpenGLContext

    ctx = QOpenGLContext.currentContext()
    if ctx is None:
        raise RuntimeError("gl_tex_image_2d_rgba_empty: no current GL context")
    addr = ctx.getProcAddress(b"glTexImage2D")
    addr_i = 0
    try:
        addr_i = int(addr) if addr else 0
    except Exception:
        addr_i = 0
    if not addr_i:
        # Rare: fall back to library export (compatibility contexts).
        if sys.platform == "win32":
            lib = ctypes.WinDLL("opengl32")
        else:
            lib = ctypes.CDLL("libGL.so.1")
        addr_i = int(ctypes.cast(lib.glTexImage2D, ctypes.c_void_p).value or 0)
    if not addr_i:
        raise RuntimeError("glTexImage2D entry point not found")
    proto = ctypes.CFUNCTYPE(
        None,
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
    )
    glTexImage2D = proto(addr_i)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        int(width),
        int(height),
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        None,
    )


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
