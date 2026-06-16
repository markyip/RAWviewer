"""Build profile (full vs lite) and lite performance defaults."""

from __future__ import annotations

import os

LITE_PREFETCH_DEFAULTS: dict[str, str] = {
    "RAWVIEWER_ENABLE_SEMANTIC_SEARCH": "0",
    "RAWVIEWER_ENABLE_FACE_SCAN": "0",
    "RAWVIEWER_AUTO_METADATA_INDEX": "1",
    "RAWVIEWER_NAV_PRELOAD_ADAPTIVE": "1",
    "RAWVIEWER_NAV_PRELOAD_NEAR": "4",
    "RAWVIEWER_NAV_PRELOAD_DISPLAY": "1",
    "RAWVIEWER_IDLE_DISPLAY_PREFETCH": "1",
    "RAWVIEWER_IDLE_DISPLAY_PREFETCH_BATCH": "4",
    "RAWVIEWER_IDLE_DISPLAY_PREFETCH_QUEUE_CAP": "16",
    "RAWVIEWER_IDLE_DISPLAY_PREFETCH_DELAY_MS": "400",
    "RAWVIEWER_IDLE_DISPLAY_PREFETCH_INTERVAL_MS": "300",
    "RAWVIEWER_FILMSTRIP_PREFETCH_RADIUS": "36",
    "RAWVIEWER_PRELOAD_THREADS": "12",
    "RAWVIEWER_ADJACENT_PRELOAD_NEXT": "8",
    "RAWVIEWER_ADJACENT_PRELOAD_PREV": "6",
    "RAWVIEWER_GALLERY_IDLE_PRELOAD_BATCH": "120",
    "RAWVIEWER_GALLERY_IDLE_PRELOAD_MS": "180",
    "RAWVIEWER_GALLERY_IDLE_PRELOAD_PRIORITY": "preload_next",
    "RAWVIEWER_PREVIEW_CACHE_ADAPTIVE": "1",
    "RAWVIEWER_MEMORY_PREVIEW_MAX": "2304",
    "RAWVIEWER_ENABLE_LOCATION_MAP": "0",
}


def resolved_profile() -> str:
    env = os.environ.get("RAWVIEWER_BUILD_PROFILE", "").strip().lower()
    if env in ("lite", "full"):
        return env
    try:
        from build_profile import PROFILE as baked

        profile = str(baked).strip().lower()
        if profile in ("lite", "full"):
            return profile
    except Exception:
        pass
    return "full"


def is_lite_build() -> bool:
    return resolved_profile() == "lite"


def location_map_enabled() -> bool:
    """Tile map overlay (OSM). Off by default; opt in via RAWVIEWER_ENABLE_LOCATION_MAP=1."""
    raw = os.environ.get("RAWVIEWER_ENABLE_LOCATION_MAP")
    if raw is None:
        return False
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def apply_profile_runtime_defaults() -> None:
    if not is_lite_build():
        return
    for key, value in LITE_PREFETCH_DEFAULTS.items():
        os.environ.setdefault(key, value)


def _read_int(name: str, default: int, *, minimum: int = 1, maximum: int = 64) -> int:
    raw = os.environ.get(name, "").strip()
    if raw:
        try:
            return max(minimum, min(int(raw), maximum))
        except ValueError:
            pass
    return max(minimum, min(default, maximum))


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _available_ram_gb() -> float | None:
    try:
        from image_cache import _safe_virtual_memory

        return float(_safe_virtual_memory().available) / (1024**3)
    except Exception:
        return None


def adaptive_nav_preload_radius(*, default_full: int = 6, default_lite: int = 10) -> int:
    """Navigation prefetch radius; scales with free RAM unless explicitly overridden."""
    explicit = os.environ.get("RAWVIEWER_NAV_PRELOAD_RADIUS", "").strip()
    if explicit:
        return _read_int("RAWVIEWER_NAV_PRELOAD_RADIUS", default_full, minimum=1, maximum=32)

    if not _env_flag("RAWVIEWER_NAV_PRELOAD_ADAPTIVE", default=True):
        base = default_lite if is_lite_build() else default_full
        return base

    avail_gb = _available_ram_gb()
    if avail_gb is None:
        return default_lite if is_lite_build() else default_full

    lite = is_lite_build()
    if avail_gb >= 24:
        radius = 16 if lite else 10
    elif avail_gb >= 16:
        radius = 14 if lite else 8
    elif avail_gb >= 10:
        radius = 12 if lite else 7
    elif avail_gb >= 6:
        radius = 10 if lite else 6
    elif avail_gb >= 3:
        radius = 7 if lite else 5
    else:
        radius = 5 if lite else 4

    min_cap = _read_int("RAWVIEWER_NAV_PRELOAD_RADIUS_MIN", 4, minimum=2, maximum=32)
    max_cap = _read_int(
        "RAWVIEWER_NAV_PRELOAD_RADIUS_MAX",
        20 if lite else 12,
        minimum=4,
        maximum=32,
    )
    return max(min_cap, min(radius, max_cap))


def adaptive_idle_display_prefetch_radius(*, default_full: int = 6) -> int:
    explicit = os.environ.get("RAWVIEWER_IDLE_DISPLAY_PREFETCH_RADIUS", "").strip()
    if explicit:
        return _read_int(
            "RAWVIEWER_IDLE_DISPLAY_PREFETCH_RADIUS",
            default_full,
            minimum=2,
            maximum=32,
        )
    nav = adaptive_nav_preload_radius()
    if is_lite_build():
        return max(6, nav)
    return max(default_full, min(nav, 12))


def preload_thread_count(*, default_full: int = 8) -> int:
    default = 12 if is_lite_build() else default_full
    return _read_int("RAWVIEWER_PRELOAD_THREADS", default, minimum=1, maximum=32)


def adjacent_preload_next(*, default_full: int = 6) -> int:
    default = 8 if is_lite_build() else default_full
    return _read_int("RAWVIEWER_ADJACENT_PRELOAD_NEXT", default, minimum=1, maximum=32)


def adjacent_preload_prev(*, default_full: int = 4) -> int:
    default = 6 if is_lite_build() else default_full
    return _read_int("RAWVIEWER_ADJACENT_PRELOAD_PREV", default, minimum=1, maximum=32)


def gallery_idle_preload_batch(*, default_full: int = 72) -> int:
    default = 120 if is_lite_build() else default_full
    return _read_int(
        "RAWVIEWER_GALLERY_IDLE_PRELOAD_BATCH",
        default,
        minimum=4,
        maximum=256,
    )


def gallery_idle_preload_ms(*, default_full: int = 250) -> int:
    default = 180 if is_lite_build() else default_full
    return _read_int(
        "RAWVIEWER_GALLERY_IDLE_PRELOAD_MS",
        default,
        minimum=50,
        maximum=2000,
    )


def indexing_loads_compete() -> bool:
    """True when semantic or face indexing may flood the load queue at BACKGROUND priority."""
    if is_lite_build():
        return False
    sem = os.environ.get("RAWVIEWER_ENABLE_SEMANTIC_SEARCH", "0").strip().lower()
    semantic_on = sem in ("1", "true", "yes", "on")
    face = os.environ.get("RAWVIEWER_ENABLE_FACE_SCAN", "1").strip().lower()
    face_on = face in ("1", "true", "yes", "on")
    return semantic_on or face_on


def gallery_idle_load_priority():
    """Queue priority for off-screen gallery thumbnail idle preload."""
    from image_load_manager import Priority

    raw = os.environ.get("RAWVIEWER_GALLERY_IDLE_PRELOAD_PRIORITY", "").strip().lower()
    if raw in ("background", "low"):
        return Priority.BACKGROUND
    if raw in ("preload_prev", "prev"):
        return Priority.PRELOAD_PREV
    if raw in ("preload_next", "preload", "high", "next"):
        return Priority.PRELOAD_NEXT
    if not indexing_loads_compete():
        return Priority.PRELOAD_NEXT
    return Priority.BACKGROUND
