"""Runtime helpers: process pools, GPU limits, volume speed probes, EDR toggles."""

from __future__ import annotations

import contextlib
import os
import threading
import time

from .helpers import io_pressure_active

_volume_speed_cache: dict[str, float] = {}
_volume_speed_lock = threading.Lock()
_VOLUME_SPEED_CACHE_CEILING_MBPS = 3000.0
def use_progressive_raw_loading() -> bool:
    """
    When True, show embedded preview/thumbnail first and upgrade to LibRaw half-res in
    the background (may cause a visible color shift). Default: off; use
    RAWVIEWER_LIBRAW_CONSISTENT_PREVIEW=1 for a single LibRaw pipeline instead.
    """
    v = os.environ.get("RAWVIEWER_PROGRESSIVE_RAW_LOAD") or os.environ.get("SKYSPOTTER_PROGRESSIVE_RAW_LOAD", "")
    v = v.strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return v in ("1", "true", "yes", "on")
def pil_downscale_resample():
    """CPU downscale filter for thumbnails ΓÇö HAMMING is faster than LANCZOS for shrink."""
    from PIL import Image

    return Image.Resampling.HAMMING
def metadata_index_idle_delay_ms() -> int:
    """Idle delay after first single-view paint before background metadata indexing starts."""
    raw = os.environ.get("RAWVIEWER_AUTO_METADATA_INDEX_IDLE_MS", "5000").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 5000
def use_raw_process_pool() -> bool:
    """
    Offload LibRaw postprocess to a process pool (multi-core). Opt-out with
    SkySpotter_USE_PROCESS_POOL=0. Default: on when CPU count >= 4 (Windows/Linux).
    macOS default off: spawn re-executes the .app (extra PIDs / splash). Opt in via
    RAWVIEWER_USE_PROCESS_POOL=1. For frozen macOS builds, also set
    RAWVIEWER_PROCESS_POOL_PYTHON to a plain interpreter (see libraw_pool_worker.py)
    so workers do not re-launch the GUI .app.

    When an in-process GPU demosaic backend is already loaded (cuda/mps/cupy),
    default off so process-pool LibRaw does not compete with GPU VRAM/host slots.
    Explicit RAWVIEWER_USE_PROCESS_POOL=1 still forces the pool on.
    """
    import os as _os
    import sys as _sys

    if _sys.platform == "darwin":
        raw = os.environ.get("RAWVIEWER_USE_PROCESS_POOL", "").strip().lower()
        if raw in ("0", "false", "no", "off"):
            return False
        if raw in ("1", "true", "yes", "on"):
            # Frozen .app without a plain-python helper would spawn the GUI
            # binary again ΓÇö refuse unless the helper is configured.
            return not (getattr(_sys, "frozen", False) and not process_pool_python_executable())
        return False

    raw = os.environ.get("RAWVIEWER_USE_PROCESS_POOL") or os.environ.get(
        "SKYSPOTTER_USE_PROCESS_POOL", ""
    )
    raw = raw.strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    # Prefer GPU/in-process fast_raw when that stack is available AND actually
    # in use (late probe: gpu_raw_processor only after main-thread bootstrap;
    # until then pool may start, then ImageLoadManager.apply_gpu_decode_profile()
    # tears it down). GPU decode now defaults ON for CUDA and OFF for MPS/CPU
    # (see fast_raw_decode.prefer_gpu_decode_enabled), so a merely-present GPU
    # backend must not disable the process pool: on MPS -- where the default is
    # still off -- that would throw away LibRaw's out-of-process parallelism to
    # serve a GPU decode path nobody is exercising. Gate on in_use, not
    # available.
    if _gpu_demosaic_backend_in_use():
        return False
    return (_os.cpu_count() or 0) >= 4
def process_pool_python_executable() -> str:
    """Optional plain-Python interpreter for ProcessPool workers (macOS .app safe)."""
    return (os.environ.get("RAWVIEWER_PROCESS_POOL_PYTHON") or "").strip()
def process_pool_mp_context():
    """Spawn context; retarget executable when RAWVIEWER_PROCESS_POOL_PYTHON is set."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    helper = process_pool_python_executable()
    if helper and os.path.isfile(helper) and os.access(helper, os.X_OK):
        with contextlib.suppress(Exception):
            ctx.set_executable(helper)
    return ctx
def _gpu_demosaic_backend_available() -> bool:
    """True when gpu_raw_processor is loaded and reports a non-CPU backend."""
    try:
        import sys as _sys

        mod = _sys.modules.get("gpu_raw_processor")
        if mod is None:
            return False
        backend = mod.detect_gpu_backend()
        return backend in ("pytorch_cuda", "pytorch_mps", "cupy")
    except Exception:
        return False
def _gpu_demosaic_backend_in_use() -> bool:
    """True when a GPU backend is available AND GPU decode is actually enabled.

    ``_gpu_demosaic_backend_available()`` alone answers "is a GPU present",
    not "will decodes use it" -- the default is per-backend (ON for CUDA, OFF
    for MPS; see fast_raw_decode.prefer_gpu_decode_enabled), and either can be
    forced by RAWVIEWER_PREFER_GPU_DECODE. On a machine where GPU decode is NOT
    in use (Apple MPS at its default, or an explicit =0), callers that gated on
    availability alone were tearing down the LibRaw process pool and capping RAW
    concurrency to GPU semaphore slots (1-3) for a GPU path that never actually
    decodes anything -- a pure throughput regression vs CPU-only hardware.

    On CUDA, where GPU decode is now on by default, that pool teardown is the
    INTENDED trade: decodes run on the GPU (measured 2.35x on full sensor-res
    demosaic), so LibRaw's CPU-side parallelism is worth less than the GPU path
    it would compete with for the same files.
    """
    if not _gpu_demosaic_backend_available():
        return False
    try:
        from processing.raw_decode.fast import prefer_gpu_decode_enabled

        return prefer_gpu_decode_enabled()
    except Exception:
        return False
def raw_load_limit_aligned_to_gpu(base_limit: int) -> int:
    """Cap concurrent heavy RAW work to GPU demosaic concurrency when active."""
    limit = max(1, int(base_limit or 1))
    try:
        import sys as _sys

        mod = _sys.modules.get("gpu_raw_processor")
        if mod is None:
            return limit
        if not _gpu_demosaic_backend_in_use():
            return limit
        gpu_n = int(mod.gpu_decode_concurrency())
        return max(1, min(limit, gpu_n))
    except Exception:
        return limit
def _env_int_bounded(name: str, default: int, *, minimum: int = 1, maximum: int = 64) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, min(maximum, int(raw)))
    except ValueError:
        return default
def _windows_drive_type(path: str) -> int | None:
    """Windows GetDriveTypeW for the volume containing *path* (None if unavailable)."""
    if os.name != "nt":
        return None
    norm = os.path.normpath(path or "")
    if len(norm) < 2 or norm[1] != ":":
        return None
    try:
        import ctypes

        return int(ctypes.windll.kernel32.GetDriveTypeW(norm[:2] + "\\"))
    except Exception:
        return None
def is_slow_storage_path(path: str) -> bool:
    """
    Heuristic for paths where aggressive parallel reads hurt more than help
    (UNC shares, user-configured prefixes). Safe on Windows and macOS.
    """
    if not path:
        return False
    norm = os.path.normpath(path)
    if norm.startswith("\\\\"):
        return True
    prefixes = os.environ.get("RAWVIEWER_SLOW_STORAGE_PREFIXES", "").strip()
    if prefixes:
        for prefix in prefixes.split(","):
            p = prefix.strip()
            if p and norm.lower().startswith(os.path.normpath(p).lower()):
                return True
    if os.name == "nt" and len(norm) >= 2 and norm[1] == ":":
        drive_type = _windows_drive_type(norm)
        # DRIVE_REMOTE=4, DRIVE_CDROM=5 ΓÇö prefer scan-first embedded JPEG extraction
        if drive_type in (4, 5):
            return True
    return False
def _windows_system_drive() -> str:
    """Boot/system volume letter (e.g. ``C:``) for probe heuristics."""
    drive = os.environ.get("SYSTEMDRIVE", "C:").strip()
    if len(drive) >= 2 and drive[1] == ":":
        return drive[:2].upper()
    return "C:"
def invalidate_volume_speed_cache(path: str | None) -> None:
    """Drop cached probe result for the volume containing *path* (e.g. before retry)."""
    root = _volume_root_for(path)
    if not root:
        return
    with _volume_speed_lock:
        _volume_speed_cache.pop(root, None)
def record_volume_speed(path: str | None, mbps: float) -> None:
    """Store a probed throughput for the volume containing *path*."""
    root = _volume_root_for(path)
    if not root or mbps <= 0:
        return
    with _volume_speed_lock:
        _volume_speed_cache[root] = float(mbps)
def _windows_drive_is_fixed_local(path: str | None) -> bool:
    """True for non-system fixed volumes (secondary SSD/NVMe) on Windows."""
    if not path or os.name != "nt":
        return False
    norm = os.path.normpath(path)
    if len(norm) < 2 or norm[1] != ":":
        return False
    letter = norm[:2].upper()
    if letter == _windows_system_drive():
        return False
    return _windows_drive_type(norm) == 3
def is_external_or_network_volume(path: str | None = None) -> bool:
    """True when *path* is on an external or network volume (not the boot fixed disk).

    macOS: paths under ``/Volumes/`` (the boot volume is ``/``, not under Volumes).
    Windows: UNC shares; removable USB/SD; mapped network drives; and any
    **non-system** fixed volume (``D:\\``, ``I:\\``, ΓÇª). USB HDD enclosures often
    report as DRIVE_FIXED, so we probe every letter except ``%SystemDrive%`` once
    per session ΓÇö fast secondary SSDs stay unthrottled after the probe.
    """
    import sys

    if path:
        norm = os.path.normpath(path)
    else:
        try:
            norm = os.path.normpath(os.getcwd())
        except Exception:
            return False

    if sys.platform == "darwin":
        return norm.startswith("/Volumes/")

    if norm.startswith("\\\\"):
        return True

    if os.name == "nt" and len(norm) >= 2 and norm[1] == ":":
        letter = norm[:2].upper()
        drive_type = _windows_drive_type(norm)
        if drive_type in (2, 4, 5):
            return True
        if drive_type == 3 and letter != _windows_system_drive():
            return True
    return False
def sort_probe_worker_count(
    sample_path: str | None = None,
    *,
    conservative: bool = False,
) -> int:
    """
    Thread pool size for cold EXIF header probes during folder sort.

    Override: RAWVIEWER_SORT_PROBE_WORKERS (1ΓÇô32).
    Conservative mode (fast-open window): RAWVIEWER_SORT_PROBE_WORKERS_CONSERVATIVE or
    half the machine's cores (was a flat 3 regardless of core count -- on an
    8-core machine that's less than half of even the *conservative* budget a
    proportional formula gives, and the flat cap doesn't back off further on
    a weak machine either. Caller triggers conservative mode for any folder
    over 400 uncached files, not just briefly at fast-open -- on a 6880-file
    folder this flat cap measured a 23s sort at 3 workers).
    Slow storage (UNC / RAWVIEWER_SLOW_STORAGE_PREFIXES): capped at 3.
    """
    override = os.environ.get("RAWVIEWER_SORT_PROBE_WORKERS", "").strip()
    if override:
        return _env_int_bounded("RAWVIEWER_SORT_PROBE_WORKERS", 4, minimum=1, maximum=32)

    if io_pressure_active():
        return 2

    cpu = os.cpu_count() or 4

    if conservative:
        cons = os.environ.get("RAWVIEWER_SORT_PROBE_WORKERS_CONSERVATIVE", "").strip()
        if cons:
            return _env_int_bounded(
                "RAWVIEWER_SORT_PROBE_WORKERS_CONSERVATIVE", 3, minimum=1, maximum=8
            )
        return min(6, max(2, cpu // 2))

    if sample_path and is_slow_storage_path(sample_path):
        # Network/UNC paths have high latency. We need *more* workers to hide the I/O wait.
        return max(8, cpu * 2)

    # Local SSD/NVMe: I/O-bound header reads; scale past 3 workers (old hard cap under-used CPU).
    return min(12, max(4, cpu - 1))
def index_metadata_worker_count(total_files: int, sample_path: str | None = None) -> int:
    """
    Parallel metadata extraction during semantic index build.

    Override: RAWVIEWER_INDEX_METADATA_WORKERS.
    Large folders (>2000) use a lower default to reduce SQLite EXIF cache lock contention.
    External/USB/network volumes ΓÇö and any confirmed-slow volume ΓÇö use a smaller
    pool so background indexing does not saturate the disk that gallery thumbnail
    decodes are also reading from.
    """
    override = os.environ.get("RAWVIEWER_INDEX_METADATA_WORKERS", "").strip()
    if override:
        return _env_int_bounded("RAWVIEWER_INDEX_METADATA_WORKERS", 2, minimum=1, maximum=16)

    cpu = os.cpu_count() or 4

    count = min(3, max(2, cpu // 2)) if total_files > 2000 else min(6, max(2, cpu - 1))

    if sample_path is not None:
        try:
            if volume_speed_tier(sample_path) == "slow":
                count = min(count, _env_int_bounded(
                    "RAWVIEWER_SLOW_VOLUME_INDEX_WORKERS", 2, minimum=1, maximum=8
                ))
            elif is_external_or_network_volume(sample_path):
                count = min(count, _env_int_bounded(
                    "RAWVIEWER_EXTERNAL_VOLUME_INDEX_WORKERS", 4, minimum=1, maximum=8
                ))
        except Exception:
            pass
    return count
def _volume_root_for(path: str | None) -> str | None:
    """Stable cache key for the volume/mount that contains *path*."""
    if not path:
        return None
    try:
        norm = os.path.abspath(path)
    except Exception:
        return None
    import sys

    if sys.platform == "darwin" and norm.startswith("/Volumes/"):
        parts = norm.split(os.sep)
        if len(parts) >= 3:
            return os.sep.join(parts[:3])  # /Volumes/<name>
        return norm
    drive, _ = os.path.splitdrive(norm)
    if drive:
        return drive + os.sep
    return os.sep
def _measure_read_mbps(
    path: str | None,
    sample_bytes: int,
    timeout_s: float,
) -> float | None:
    """One cold-ish read-throughput measurement (MB/s) for *path*; no caching.

    Reads ``sample_bytes`` starting from a pseudo-random offset rather than the
    file head. Thumbnail/preview extraction (and a prior fast-open) caches the
    head/embedded-JPEG region, so reading the head would report RAM speed; a
    random mid-file offset is far more likely to hit the disk.
    """
    try:
        if not (path and os.path.isfile(path)):
            return None
        size = os.path.getsize(path)
        to_read = min(sample_bytes, size)
        if to_read <= 0:
            return None
        max_offset = max(0, size - to_read)
        offset = 0
        if max_offset > 0:
            try:
                offset = (os.getpid() * 2654435761 + time.monotonic_ns()) % (max_offset + 1)
            except Exception:
                offset = max_offset // 2
        read_total = 0
        start = time.monotonic()
        with open(path, "rb", buffering=0) as fh:
            if offset:
                fh.seek(offset)
            while read_total < to_read:
                chunk = fh.read(min(1024 * 1024, to_read - read_total))
                if not chunk:
                    break
                read_total += len(chunk)
                if time.monotonic() - start > timeout_s:
                    break
        elapsed = max(1e-4, time.monotonic() - start)
        # Require a meaningful sample (>=512 KB) before trusting the number.
        if read_total >= 512 * 1024:
            return (read_total / (1024 * 1024)) / elapsed
    except Exception:
        return None
    return None
def probe_volume_speed(
    sample_path: str | None,
    *,
    sample_bytes: int = 8 * 1024 * 1024,
    timeout_s: float = 3.0,
) -> float | None:
    """Very lightweight one-shot read-speed probe for the volume holding *sample_path*.

    Reads up to ``sample_bytes`` (default 4 MB) from a single file and estimates
    throughput in MB/s. The result is cached per volume root and reused for the
    whole session, so this measures each mount at most once. Designed to run off
    the main thread (see ImageLoadManager.prime_volume_speed_async).

    Page-cache hits (implausibly high readings, e.g. probing a just-opened file)
    are rejected and NOT cached, so the caller can retry with a colder file.

    Returns the measured MB/s, or ``None`` if it could not be measured/trusted.
    """
    root = _volume_root_for(sample_path)
    if not root:
        return None
    with _volume_speed_lock:
        cached = _volume_speed_cache.get(root)
    if cached is not None:
        return cached if cached >= 0 else None

    mbps = _measure_read_mbps(sample_path, sample_bytes, timeout_s)

    if mbps is not None and mbps > _VOLUME_SPEED_CACHE_CEILING_MBPS:
        # RAM/page-cache read, not the disk. Do not trust or cache it ΓÇö leave the
        # volume unprobed so prime_volume_speed_async can try a colder file.
        return None

    with _volume_speed_lock:
        _volume_speed_cache[root] = mbps if mbps is not None else -1.0
    return mbps
def volume_speed_tier(sample_path: str | None) -> str:
    """Classify a probed volume as ``"fast"``, ``"slow"`` or ``"unknown"``.

    A volume is only treated as ``"slow"`` once it has actually been probed and
    measured below RAWVIEWER_VOLUME_SLOW_MBPS (default 120 MB/s) ΓÇö fast external
    interfaces (Thunderbolt 3/4, USB4, NVMe enclosures) stay ``"fast"`` and are
    never throttled. Unprobed/local volumes return ``"unknown"`` (no throttle).
    """
    root = _volume_root_for(sample_path)
    if not root:
        return "unknown"
    with _volume_speed_lock:
        cached = _volume_speed_cache.get(root)
    if cached is None or cached < 0:
        return "unknown"
    try:
        slow_threshold = float(
            os.environ.get("RAWVIEWER_VOLUME_SLOW_MBPS", "").strip() or 120.0
        )
    except ValueError:
        slow_threshold = 120.0
    return "slow" if cached < slow_threshold else "fast"
def moderate_external_cap_enabled() -> bool:
    """Whether fast (not just confirmed-slow) external volumes get a moderate cap.

    Windows-only by default: the native gallery crash (high concurrent LibRaw +
    GDI/handle pressure) only reproduces on Windows. macOS handled fast external
    (Thunderbolt/USB4) drives at full concurrency without crashing, so we keep its
    original behaviour ΓÇö only confirmed-slow volumes are throttled there.
    Override with RAWVIEWER_MODERATE_EXTERNAL_CAP=1/0.
    """
    import sys

    raw = os.environ.get("RAWVIEWER_MODERATE_EXTERNAL_CAP", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return sys.platform == "win32"
def raw_concurrent_load_limit(sample_path: str | None = None) -> int:
    """Max concurrent LibRaw full/preview decodes in ImageLoadManager.

    An explicit RAWVIEWER_RAW_LOAD_LIMIT always wins. Otherwise concurrency is
    reduced for confirmed-slow external/network volumes (all platforms). On
    Windows, every external volume also gets a moderate cap (see
    moderate_external_cap_enabled) to avoid the native gallery crash; macOS keeps
    full concurrency on fast external drives.
    """
    cpu = os.cpu_count() or 4
    default = max(4, cpu)
    limit = _env_int_bounded(
        "RAWVIEWER_RAW_LOAD_LIMIT", default, minimum=1, maximum=32
    )

    if not os.environ.get("RAWVIEWER_RAW_LOAD_LIMIT", "").strip() and is_external_or_network_volume(sample_path):
        if volume_speed_tier(sample_path) == "slow":
            limit = min(
                limit,
                _env_int_bounded(
                    "RAWVIEWER_SLOW_VOLUME_RAW_LIMIT", 3, minimum=1, maximum=8
                ),
            )
        elif moderate_external_cap_enabled():
            limit = min(
                limit,
                _env_int_bounded(
                    "RAWVIEWER_EXTERNAL_VOLUME_RAW_LIMIT", 8, minimum=1, maximum=32
                ),
            )
    return raw_load_limit_aligned_to_gpu(limit)
def process_pool_worker_count() -> int:
    """LibRaw postprocess process-pool size when RAWVIEWER_USE_PROCESS_POOL is on."""
    cpu = os.cpu_count() or 4
    default = max(2, cpu - 1)
    return _env_int_bounded("RAWVIEWER_PROCESS_POOL_WORKERS", default, minimum=1, maximum=32)

