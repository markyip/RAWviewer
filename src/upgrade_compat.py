"""
Upgrade compatibility for existing ~/.rawviewer_cache installs.

Fresh installs get perf-v2 defaults (async search metadata, batch 8–16).
Existing installs keep legacy behavior until RAWVIEWER_PERF_V2=1 or the user
clears cache / opts in via env.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, Optional

_CACHE_DIR = os.path.expanduser("~/.rawviewer_cache")
_STATE_PATH = os.path.join(_CACHE_DIR, "upgrade_state.json")
_LEGACY_MARKERS = (
    "semantic_index.db",
    "exif_cache.db",
    "semantic_batch_tuning.json",
)

# Bump when perf defaults or migration rules change.
UPGRADE_STATE_VERSION = 1


def _load_state() -> Dict[str, Any]:
    try:
        if os.path.isfile(_STATE_PATH):
            with open(_STATE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _save_state(state: Dict[str, Any]) -> None:
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        state.setdefault("version", UPGRADE_STATE_VERSION)
        state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)
    except Exception:
        pass


def is_existing_install() -> bool:
    """True when the user already has a pre-v2 cache / index on disk."""
    if not os.path.isdir(_CACHE_DIR):
        return False
    for name in _LEGACY_MARKERS:
        if os.path.isfile(os.path.join(_CACHE_DIR, name)):
            return True
    return False


def perf_v2_explicitly_enabled() -> bool:
    raw = os.environ.get("RAWVIEWER_PERF_V2", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def perf_v2_explicitly_disabled() -> bool:
    raw = os.environ.get("RAWVIEWER_PERF_V2", "").strip().lower()
    return raw in ("0", "false", "no", "off")


def perf_v2_enabled() -> bool:
    """New perf defaults (batch 8–16, async search metadata)."""
    if perf_v2_explicitly_enabled():
        return True
    if perf_v2_explicitly_disabled():
        return False
    # Re-evaluate each launch (clear_cache.sh may remove markers between runs).
    return not is_existing_install()


def legacy_search_metadata_sync() -> bool:
    """Sync EXIF/geocode on the search UI thread (pre-v2 behavior)."""
    raw = os.environ.get("RAWVIEWER_SEARCH_SYNC_METADATA", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return not perf_v2_enabled()


def _tier_semantic_batch_default() -> int:
    try:
        from rawviewer_profile import classify_memory_tier, system_total_ram_gb

        tier = classify_memory_tier(system_total_ram_gb())
        return {
            "low": 8,
            "medium": 12,
            "balanced": 8,
            "high": 16,
            "ultra": 16,
        }.get(tier, 12)
    except Exception:
        return 12


def _tier_coreml_chunk_default() -> int:
    try:
        from rawviewer_profile import classify_memory_tier, system_total_ram_gb

        tier = classify_memory_tier(system_total_ram_gb())
        return {
            "low": 8,
            "medium": 12,
            "balanced": 8,
            "high": 16,
            "ultra": 16,
        }.get(tier, 12)
    except Exception:
        return 12


def apply_upgrade_compat_defaults() -> str:
    """
    Apply install-aware defaults after RAM-tier profile defaults.

    Returns a short label for startup logging: legacy | perf_v2 | explicit.
    """
    existing = is_existing_install()
    v2 = perf_v2_enabled()
    state = _load_state()
    state["existing_install"] = existing
    state["perf_v2"] = v2

    if v2:
        # Perf v2: ONNX batch on Windows; macOS Core ML uses serial predict (see semantic_coreml_use_native_batch).
        if "RAWVIEWER_SEMANTIC_BATCH_SIZE" not in os.environ:
            os.environ.setdefault(
                "RAWVIEWER_SEMANTIC_BATCH_SIZE", str(_tier_semantic_batch_default())
            )
        if sys.platform != "darwin" and "RAWVIEWER_SEMANTIC_COREML_CHUNK" not in os.environ:
            os.environ.setdefault(
                "RAWVIEWER_SEMANTIC_COREML_CHUNK", str(_tier_coreml_chunk_default())
            )
        os.environ.setdefault("RAWVIEWER_SEARCH_SYNC_METADATA", "0")
        label = "perf_v2"
    else:
        # Legacy upgrade path: preserve pre-v2 semantic + search behavior.
        if sys.platform != "darwin":
            os.environ.setdefault("RAWVIEWER_SEMANTIC_BATCH_SIZE", "1")
            os.environ.setdefault("RAWVIEWER_SEMANTIC_COREML_CHUNK", "8")
        os.environ.setdefault("RAWVIEWER_SEARCH_SYNC_METADATA", "1")
        label = "legacy"

    if perf_v2_explicitly_enabled():
        label = "perf_v2 (explicit)"
    elif perf_v2_explicitly_disabled():
        label = "legacy (explicit)"

    state["mode"] = label
    _save_state(state)
    return label

