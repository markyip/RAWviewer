"""Deferred, thread-safe import of the torch/kornia-based GPU decode backend.

PyTorch's OpenMP runtime must be initialized on the main/GUI thread: doing it
for the first time on a background QRunnable thread causes __kmp_abort_process
on macOS -- a hard process abort, not a catchable exception. The old fix was
to eagerly `import torch` as the very first line of main.py, guaranteeing the
main thread got there first -- correct, but it paid torch+kornia's ~0.9s
import cost before the window could even be constructed, on every launch,
whether or not GPU decode is ever used for that image.

This module makes the import lazy (triggered once, right after the window is
shown) while preserving the safety guarantee: background decode threads never
import the GPU backend themselves. They call wait_for_gpu_backend_ready() and
either wait for the main thread to finish importing it, or fall back to CPU
decode if that takes unexpectedly long.
"""
import threading

_ready = threading.Event()
_import_started = threading.Event()


def import_gpu_backend_on_main_thread() -> None:
    """Call once, from the main/GUI thread, as soon as possible after the
    window is shown. Safe to call more than once -- a no-op after the first.
    """
    if _import_started.is_set():
        return
    _import_started.set()
    try:
        import gpu_raw_processor  # noqa: F401 -- imports torch/kornia/cupy as a side effect
        try:
            # Align load manager with GPU reality: process-pool vs CUDA/MPS slots.
            from image_load_manager import apply_gpu_decode_profile_to_manager

            apply_gpu_decode_profile_to_manager()
        except Exception:
            pass
    except Exception:
        pass
    finally:
        _ready.set()


def wait_for_gpu_backend_ready(timeout: float = 10.0) -> bool:
    """Background decode threads call this instead of importing the GPU
    backend themselves. Returns False on timeout, meaning the caller should
    fall back to CPU decode rather than risk a background-thread torch
    import.

    If the CALLER is itself the main thread (e.g. a standalone script or test
    that calls the decode path directly, with no GUI event loop ever driving
    import_gpu_backend_on_main_thread()), it's always safe to import right
    here rather than wait for an event nobody else will ever set.
    """
    if threading.current_thread() is threading.main_thread():
        import_gpu_backend_on_main_thread()
        return True
    return _ready.wait(timeout=timeout)
