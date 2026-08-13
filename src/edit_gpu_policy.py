"""GPU vs CPU policy for the Adjust / export edit pipeline.

Display zoom/pan already uses ``GpuImageView`` (OpenGL) on a CPU-rendered
bitmap. Plus CUDA already GPU-demosaics **full-resolution** RAW (CuPy, or
optional PyTorch CUDA) via ``fast_raw_decode`` / ``gpu_raw_processor``.
This module records why live Adjust ticks stay on host OpenCV, and the one
decode path that is allowed to reuse that existing GPU demosaic.

Do not import torch here — Standard / Lite must stay torch-free, and the
Windows Plus CUDA product path is CuPy, not a CUDA OpenCV build.
"""

from __future__ import annotations

from typing import Any, Optional


def prefer_gpu_linear_edit_decode() -> bool:
    """True when a full-res linear edit-base demosaic may run on GPU.

    Same gate as browse: CUDA/CuPy by default, MPS off, Lite forced off,
    ``RAWVIEWER_PREFER_GPU_DECODE`` always wins. Half-size working bases
    never use this — GPU demosaic is a full-mosaic kernel; half-size is
    2x2 binning and is unpack-bound on CPU.
    """
    try:
        from fast_raw_decode import prefer_gpu_decode_enabled

        return bool(prefer_gpu_decode_enabled())
    except Exception:
        return False


def adjust_preview_gpu_compute_enabled() -> bool:
    """Live Adjust / settle composite stays on CPU.

    Kept as an explicit gate so a later CuPy/OpenCL experiment has one
    place to flip — and so tests can lock the current choice.

    Reasons this stays False:

    * Preview settle must not drift from export. PV2012 y0 smoothing,
      bilateral/guided NR, mask composite, depth blur and geometry remap
      are cv2/numpy. A CuPy or OpenCL rewrite of ``boxFilter`` /
      ``bilateralFilter`` is not bit-identical at edges.
    * Standard omits torch/kornia. pip OpenCV wheels have no ``cv2.cuda``.
      Face-DNN already disables OpenCL on some Windows stacks that crash.
    * Live-drag is already ~640 px + ``preview_lite`` (skips guided
      filters). Settle is RAM-tier-capped on ≤16 GB. Transfer to a GPU
      for those sizes is overhead, not a win, on machines without a
      discrete CUDA device.
    """
    return False


def finish_linear_edit_base_from_unpacked(
    unpacked: Any,
    edit_params: Optional[dict] = None,
) -> Optional[Any]:
    """Full-res scene-linear uint16 from an existing unpack (peek, not take).

    Reuses browse's stashed mosaic so export / ``use_full_resolution``
    Adjust bases skip a second LibRaw unpack, then runs the same
    ``finish_full_decode(prefer_gpu=...)`` path as sensor-res browse.
    Returns None to mean "caller should unpack or fall back".
    """
    if unpacked is None:
        return None
    try:
        from fast_raw_decode import finish_full_decode, params_supported
    except Exception:
        return None
    params = dict(edit_params or {})
    # Full mosaic only. Half-size working bases stay on CPU 2x2 binning.
    if params.get("half_size", False):
        return None
    if not params_supported(params, return_linear=True):
        return None
    try:
        return finish_full_decode(
            unpacked,
            return_linear=True,
            prefer_gpu=prefer_gpu_linear_edit_decode(),
        )
    except Exception:
        return None
