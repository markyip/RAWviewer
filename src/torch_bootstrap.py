"""Deferred, thread-safe import of the GPU decode backend (CuPy and/or torch).

PyTorch's OpenMP runtime must be initialized on the main/GUI thread: doing it
for the first time on a background QRunnable thread causes __kmp_abort_process
on macOS -- a hard process abort, not a catchable exception. The old fix was
to eagerly `import torch` as the very first line of main.py, guaranteeing the
main thread got there first -- correct, but it paid torch+kornia's ~0.9s
import cost before the window could even be constructed, on every launch,
whether or not GPU decode is ever used for that image.

Product Windows CUDA installs are torch-free (CuPy). Optional torch/kornia may
still be present in a developer environment or on macOS (MPS). This module
never downloads a multi-GB torch wheel as a "repair" — missing GPU libs simply
leave demosaic on CPU Fast RAW.

The import itself still runs on the main thread (OpenMP constraint), but
callers can pass a ``pump`` callback (typically QApplication.processEvents)
between heavy stages so paints/timers can flush mid-import.
"""
import threading
from typing import Callable, Optional

_ready = threading.Event()
_import_started = threading.Event()


def import_gpu_backend_on_main_thread(
    pump: Optional[Callable[[], None]] = None,
) -> None:
    """Call once, from the main/GUI thread, as soon as possible after the
    window is shown. Safe to call more than once -- a no-op after the first.

    ``pump`` is invoked between heavy import stages so the GUI can drain
    pending paints/events during a cold import (often several seconds).
    """
    if _import_started.is_set():
        return
    _import_started.set()
    import logging
    import time

    logger = logging.getLogger(__name__)
    t0 = time.perf_counter()
    logger.info("[GPU] Importing GPU demosaic backend on main thread...")

    def _pump() -> None:
        if pump is None:
            return
        try:
            pump()
        except Exception:
            pass

    try:
        # Prefer CuPy (torch-free Windows CUDA product path). Optional torch/
        # kornia is for macOS MPS or a local dev env that still has them.
        cupy_ok = False
        try:
            import cupy  # noqa: F401

            if cupy.cuda.runtime.getDeviceCount() > 0:
                cupy_ok = True
                logger.info("[GPU] CuPy CUDA demosaic backend available.")
        except Exception:
            pass

        _pump()

        torch_ok = False
        try:
            import torch  # noqa: F401

            if torch.cuda.is_available() or (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ):
                torch_ok = True
                logger.info("[GPU] Optional torch GPU backend available.")
        except Exception:
            pass

        if torch_ok:
            _pump()
            try:
                import kornia  # noqa: F401
            except Exception:
                logger.debug("[GPU] kornia not importable; torch path limited", exc_info=True)

        if not cupy_ok and not torch_ok:
            logger.info(
                "[GPU] No CuPy/torch CUDA/MPS backend; demosaic stays on CPU Fast RAW."
            )

        _pump()
        import gpu_raw_processor  # noqa: F401 -- wires decode + backend probe

        try:
            from image_load_manager import apply_gpu_decode_profile_to_manager

            apply_gpu_decode_profile_to_manager()
        except Exception:
            pass
    except Exception:
        logger.exception("[GPU] Backend import failed; decode will stay on CPU")
    finally:
        _ready.set()
        logger.info(
            "[GPU] Backend import finished in %.3fs",
            time.perf_counter() - t0,
        )


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


def gpu_backend_import_started() -> bool:
    return _import_started.is_set()


def gpu_backend_ready() -> bool:
    return _ready.is_set()
