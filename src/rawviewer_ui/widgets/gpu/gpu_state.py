"""GPU OpenGL environment toggles for the single-image view."""

from __future__ import annotations

import os


def _env_true(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}

def gpu_gl_tex_enabled() -> bool:
    """Opt-in QImage→OpenGL paint path (skips QPixmap); see set_rgb_numpy."""
    return _env_true("RAWVIEWER_GPU_GL_TEX")

def gpu_cuda_gl_enabled() -> bool:
    """Opt-in CUDA↔GL zero-copy display; see set_device_rgb."""
    return _env_true("RAWVIEWER_GPU_CUDA_GL")
