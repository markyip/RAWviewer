"""Session-sticky OS memory-pressure backoff.

Installed-RAM tiering (``classify_memory_tier`` / ``apply_memory_tier_defaults``)
stays based on total physical RAM. This module is a separate path: when the OS
is under *sustained* pressure (or available RAM stays low), shrink caches,
idle preload, and unpack overlap for the rest of the session.

Next launch starts from the installed-RAM tier again. Mid-session we never
auto-upgrade back to the full budgets (no oscillation).

Kill switch: ``RAWVIEWER_MEMORY_PRESSURE_BACKOFF=0``.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from dataclasses import dataclass
from typing import Literal, Optional

logger = logging.getLogger(__name__)

PressureLevel = Literal["none", "warn", "critical"]

_LEVEL_RANK = {"none": 0, "warn": 1, "critical": 2}

_EDIT_BASE_MAX_EDGE_KEY = "RAWVIEWER_EDIT_BASE_MAX_EDGE"

_state_lock = threading.Lock()
_consecutive = {"warn": 0, "critical": 0}
_latched: PressureLevel = "none"
_applied: PressureLevel = "none"
_thumb_scale = 1.0
_full_image_scale = 1.0
_preload_cap: Optional[int] = None
_env_originals: dict[str, Optional[str]] = {}


@dataclass(frozen=True)
class MemorySample:
    """One OS memory snapshot. Any field may be None if that probe failed."""

    total_bytes: Optional[int] = None
    available_bytes: Optional[int] = None
    percent_used: Optional[float] = None
    windows_memory_load: Optional[int] = None
    macos_available_bytes: Optional[int] = None


def backoff_enabled() -> bool:
    raw = os.environ.get("RAWVIEWER_MEMORY_PRESSURE_BACKOFF", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def debounce_samples_required() -> int:
    raw = os.environ.get("RAWVIEWER_MEMORY_PRESSURE_DEBOUNCE_SAMPLES", "3").strip()
    try:
        return max(2, min(int(raw), 8))
    except ValueError:
        return 3


def latched_level() -> PressureLevel:
    return _latched


def pressure_level() -> PressureLevel:
    """Public alias: current session-latched level (not the installed-RAM tier)."""
    return _latched


def thumb_budget_scale() -> float:
    return _thumb_scale


def full_image_budget_scale() -> float:
    return _full_image_scale


def adjacent_preload_cap() -> Optional[int]:
    """None = no extra cap (use the installed-tier / class default)."""
    return _preload_cap


def scale_thumb_budget_bytes(
    base_bytes: int, *, floor_bytes: int = 64 * 1024 * 1024
) -> int:
    return max(int(floor_bytes), int(base_bytes * thumb_budget_scale()))


def reset_for_tests() -> None:
    """Clear session latch / scales. Tests only."""
    global _latched, _applied, _thumb_scale, _full_image_scale, _preload_cap
    with _state_lock:
        _consecutive["warn"] = 0
        _consecutive["critical"] = 0
        _latched = "none"
        _applied = "none"
        _thumb_scale = 1.0
        _full_image_scale = 1.0
        _preload_cap = None
        _env_originals.clear()


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, min(float(raw), maximum))
    except ValueError:
        return default


def _windows_memory_status() -> Optional[tuple[int, int, int]]:
    """Return (total_phys, avail_phys, dwMemoryLoad) via GlobalMemoryStatusEx."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return None
        return int(stat.ullTotalPhys), int(stat.ullAvailPhys), int(stat.dwMemoryLoad)
    except Exception:
        return None


def _macos_os_proc_available_memory() -> Optional[int]:
    """Bytes the process may still allocate before jetsam; 0 / missing → unknown."""
    if sys.platform != "darwin":
        return None
    try:
        import ctypes

        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        lib.os_proc_available_memory.restype = ctypes.c_uint64
        lib.os_proc_available_memory.argtypes = []
        n = int(lib.os_proc_available_memory())
        return n if n > 0 else None
    except Exception:
        return None


def _fallback_virtual_memory() -> Optional[tuple[int, int, float]]:
    try:
        from image_cache import _safe_virtual_memory

        vm = _safe_virtual_memory()
        return int(vm.total), int(vm.available), float(vm.percent)
    except Exception:
        pass
    try:
        import psutil

        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available), float(vm.percent)
    except Exception:
        return None


def sample_memory() -> MemorySample:
    """Collect a cheap OS memory snapshot (no subprocess ``memory_pressure``)."""
    total = available = None
    percent = None
    win_load = None
    mac_avail = _macos_os_proc_available_memory()

    win = _windows_memory_status()
    if win is not None:
        total, available, win_load = win
        if total > 0:
            percent = max(0.0, min(100.0, (1.0 - (available / float(total))) * 100.0))

    if total is None or available is None:
        fb = _fallback_virtual_memory()
        if fb is not None:
            total, available, percent = fb

    return MemorySample(
        total_bytes=total,
        available_bytes=available,
        percent_used=percent,
        windows_memory_load=win_load,
        macos_available_bytes=mac_avail,
    )


def classify_sample(sample: MemorySample) -> PressureLevel:
    """Map one snapshot to none / warn / critical. Missing data → none."""
    crit_pct = _env_float(
        "RAWVIEWER_MEMORY_PRESSURE_PERCENT_CRITICAL", 90.0, minimum=50.0, maximum=99.0
    )
    warn_pct = _env_float(
        "RAWVIEWER_MEMORY_PRESSURE_PERCENT_WARN", 85.0, minimum=50.0, maximum=98.0
    )
    crit_avail_gb = _env_float(
        "RAWVIEWER_MEMORY_PRESSURE_AVAILABLE_GB_CRITICAL", 2.0, minimum=0.25, maximum=32.0
    )
    warn_avail_gb = _env_float(
        "RAWVIEWER_MEMORY_PRESSURE_AVAILABLE_GB_WARN", 4.0, minimum=0.5, maximum=64.0
    )
    mac_crit = _env_float(
        "RAWVIEWER_MEMORY_PRESSURE_MACOS_AVAILABLE_MB_CRITICAL",
        512.0,
        minimum=64.0,
        maximum=8192.0,
    )
    mac_warn = _env_float(
        "RAWVIEWER_MEMORY_PRESSURE_MACOS_AVAILABLE_MB_WARN",
        1536.0,
        minimum=128.0,
        maximum=16384.0,
    )

    available_gb = None
    if sample.available_bytes is not None:
        available_gb = sample.available_bytes / (1024**3)

    percent = sample.percent_used
    load = sample.windows_memory_load
    mac_mb = None
    if sample.macos_available_bytes is not None:
        mac_mb = sample.macos_available_bytes / (1024 * 1024)

    def _is_critical() -> bool:
        if percent is not None and percent >= crit_pct:
            return True
        if load is not None and load >= int(crit_pct):
            return True
        if available_gb is not None and available_gb <= crit_avail_gb:
            return True
        if mac_mb is not None and mac_mb <= mac_crit:
            return True
        return False

    def _is_warn() -> bool:
        if percent is not None and percent >= warn_pct:
            return True
        if load is not None and load >= int(warn_pct):
            return True
        if available_gb is not None and available_gb <= warn_avail_gb:
            return True
        if mac_mb is not None and mac_mb <= mac_warn:
            return True
        return False

    if _is_critical():
        return "critical"
    if _is_warn():
        return "warn"
    return "none"


def observe(sample: Optional[MemorySample] = None) -> PressureLevel:
    """Debounced, session-sticky latch. Never auto-upgrades mid-session."""
    global _latched
    if not backoff_enabled():
        return "none"
    if sample is None:
        sample = sample_memory()
    instant = classify_sample(sample)
    need = debounce_samples_required()
    with _state_lock:
        if instant == "critical":
            _consecutive["critical"] += 1
            _consecutive["warn"] += 1
        elif instant == "warn":
            _consecutive["warn"] += 1
            _consecutive["critical"] = 0
        else:
            _consecutive["warn"] = 0
            _consecutive["critical"] = 0

        candidate: PressureLevel = "none"
        if _consecutive["critical"] >= need:
            candidate = "critical"
        elif _consecutive["warn"] >= need:
            candidate = "warn"

        if _LEVEL_RANK[candidate] > _LEVEL_RANK[_latched]:
            _latched = candidate
            logger.info(
                "[memory-pressure] latched %s (instant=%s, warn_streak=%d, critical_streak=%d)",
                _latched,
                instant,
                _consecutive["warn"],
                _consecutive["critical"],
            )
        return _latched


def _snapshot_env(key: str) -> None:
    if key not in _env_originals:
        _env_originals[key] = os.environ.get(key)


def _cap_env_int(key: str, cap: int, *, minimum: int = 1) -> None:
    _snapshot_env(key)
    cap = max(minimum, int(cap))
    raw = os.environ.get(key, "").strip()
    current = None
    if raw:
        try:
            current = int(raw)
        except ValueError:
            current = None
    new = cap if current is None else min(current, cap)
    os.environ[key] = str(max(minimum, new))


def _set_env(key: str, value: str) -> None:
    _snapshot_env(key)
    os.environ[key] = value


def _apply_env_caps(level: PressureLevel) -> None:
    if level == "warn":
        _cap_env_int("RAWVIEWER_GALLERY_IDLE_PRELOAD_BATCH", 16, minimum=4)
        _cap_env_int("RAWVIEWER_PRELOAD_THREADS", 4, minimum=1)
        _cap_env_int("RAWVIEWER_ADJACENT_PRELOAD_NEXT", 2, minimum=1)
        _cap_env_int("RAWVIEWER_ADJACENT_PRELOAD_PREV", 1, minimum=1)
        _cap_env_int("RAWVIEWER_UNPACK_CONCURRENCY", 1, minimum=1)
        _cap_env_int("RAWVIEWER_IDLE_DISPLAY_PREFETCH_BATCH", 2, minimum=1)
        _cap_env_int("RAWVIEWER_NAV_PRELOAD_NEAR", 1, minimum=1)
        _cap_env_int("RAWVIEWER_EDIT_BASE_LRU_MAX", 1, minimum=1)
        _cap_env_int("RAWVIEWER_FILMSTRIP_PREFETCH_RADIUS", 8, minimum=2)
        _set_env("RAWVIEWER_GALLERY_HOVER_PREFETCH", "0")
    elif level == "critical":
        _cap_env_int("RAWVIEWER_GALLERY_IDLE_PRELOAD_BATCH", 8, minimum=4)
        _cap_env_int("RAWVIEWER_PRELOAD_THREADS", 2, minimum=1)
        _cap_env_int("RAWVIEWER_ADJACENT_PRELOAD_NEXT", 1, minimum=1)
        _cap_env_int("RAWVIEWER_ADJACENT_PRELOAD_PREV", 1, minimum=1)
        _cap_env_int("RAWVIEWER_UNPACK_CONCURRENCY", 1, minimum=1)
        _cap_env_int("RAWVIEWER_IDLE_DISPLAY_PREFETCH_BATCH", 1, minimum=1)
        _cap_env_int("RAWVIEWER_NAV_PRELOAD_NEAR", 1, minimum=1)
        _cap_env_int("RAWVIEWER_EDIT_BASE_LRU_MAX", 1, minimum=1)
        _cap_env_int("RAWVIEWER_FILMSTRIP_PREFETCH_RADIUS", 4, minimum=2)
        _set_env("RAWVIEWER_GALLERY_HOVER_PREFETCH", "0")
        _set_env("RAWVIEWER_IDLE_DISPLAY_PREFETCH", "0")


def _apply_scales(level: PressureLevel) -> None:
    global _thumb_scale, _full_image_scale, _preload_cap
    if level == "warn":
        _thumb_scale = min(_thumb_scale, 0.5)
        _full_image_scale = min(_full_image_scale, 0.5)
        cap = 2
    elif level == "critical":
        _thumb_scale = min(_thumb_scale, 0.25)
        _full_image_scale = min(_full_image_scale, 0.25)
        cap = 1
    else:
        return
    if _preload_cap is None:
        _preload_cap = cap
    else:
        _preload_cap = min(_preload_cap, cap)


def _trim_image_cache() -> None:
    try:
        import image_cache as ic

        cache = getattr(ic, "_global_cache", None)
        if cache is None:
            return
    except Exception:
        return
    apply = getattr(cache, "apply_pressure_budget_scale", None)
    if callable(apply):
        try:
            apply(full_image_budget_scale())
        except Exception:
            logger.debug("image-cache pressure scale failed", exc_info=True)


def _cap_unpack() -> None:
    try:
        from enhanced_raw_processor import set_unpack_concurrency_cap

        set_unpack_concurrency_cap(1)
    except Exception:
        logger.debug("unpack concurrency cap failed", exc_info=True)


def _trim_edit_base_lru() -> None:
    try:
        from unified_image_processor import trim_edit_base_lru

        trim_edit_base_lru(1)
    except Exception:
        logger.debug("edit-base LRU trim failed", exc_info=True)


def _pause_viewer_preload(viewer: object) -> None:
    cap = adjacent_preload_cap()
    if cap is not None:
        if hasattr(viewer, "_PRELOAD_NEXT_COUNT"):
            try:
                setattr(
                    viewer,
                    "_PRELOAD_NEXT_COUNT",
                    min(int(getattr(viewer, "_PRELOAD_NEXT_COUNT")), cap),
                )
            except Exception:
                pass
        prev_cap = 1 if cap <= 1 else max(1, cap - 1)
        if hasattr(viewer, "_PRELOAD_PREV_COUNT"):
            try:
                setattr(
                    viewer,
                    "_PRELOAD_PREV_COUNT",
                    min(int(getattr(viewer, "_PRELOAD_PREV_COUNT")), prev_cap),
                )
            except Exception:
                pass

    timer = getattr(viewer, "_idle_display_prefetch_timer", None)
    if timer is not None and hasattr(timer, "stop"):
        try:
            timer.stop()
        except Exception:
            pass

    gv = getattr(viewer, "gallery_view", None)
    if gv is None:
        return
    pause = getattr(gv, "pause_idle_preload", None)
    if callable(pause):
        try:
            pause()
        except Exception:
            logger.debug("pause idle preload failed", exc_info=True)
    enforce = getattr(gv, "_enforce_thumbnail_cache_budget", None)
    if not callable(enforce):
        return
    try:
        enforce(force=True)
    except TypeError:
        try:
            setattr(gv, "_thumb_cache_puts_since_check", 10**6)
            enforce()
        except Exception:
            logger.debug("thumb budget enforce failed", exc_info=True)
    except Exception:
        logger.debug("thumb budget enforce failed", exc_info=True)


def apply_backoff(level: PressureLevel, *, viewer: object | None = None) -> PressureLevel:
    """Shrink budgets / trim caches for ``level``. Session-sticky; never raises."""
    global _applied
    if level not in ("warn", "critical") or not backoff_enabled():
        return _applied
    if _LEVEL_RANK[level] <= _LEVEL_RANK[_applied]:
        return _applied

    edge_before = os.environ.get(_EDIT_BASE_MAX_EDGE_KEY)
    _apply_scales(level)
    _apply_env_caps(level)
    _trim_image_cache()
    _cap_unpack()
    _trim_edit_base_lru()
    if viewer is not None:
        _pause_viewer_preload(viewer)

    # Never rebuild the working Adjust buffer by changing the long-edge cap.
    if edge_before is None:
        os.environ.pop(_EDIT_BASE_MAX_EDGE_KEY, None)
    else:
        os.environ[_EDIT_BASE_MAX_EDGE_KEY] = edge_before

    _applied = level
    logger.info(
        "[memory-pressure] applied %s backoff (thumb_scale=%.2f, full_scale=%.2f, preload_cap=%s)",
        level,
        _thumb_scale,
        _full_image_scale,
        _preload_cap,
    )
    return _applied


def poll_and_apply(
    *, viewer: object | None = None, sample: Optional[MemorySample] = None
) -> PressureLevel:
    """Idle-timer entry: observe one sample, apply backoff if the latch rose."""
    if not backoff_enabled():
        return "none"
    level = observe(sample)
    if level != "none":
        apply_backoff(level, viewer=viewer)
    return level
