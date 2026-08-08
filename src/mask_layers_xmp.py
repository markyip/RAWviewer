"""
XMP persistence for raw_mask_layers.MaskLayerStack -- own schema (decided
against attempting a Lightroom MaskGroupBasedCorrections round-trip for v1;
see docs/FEATURE_FEASIBILITY.md and the plan's XMP decision).

Shape: a single ``crs:RVMaskLayers`` child element (RAWviewer-private,
additive to the existing crs: attributes, following the exact same
child-element pattern as crs:DodgeBurnMask / crs:SpotHealMask) whose text
is a JSON list, one entry per layer:

    {"alpha": "<base64 8-bit PNG>", "adjustments": {...non-zero only...},
     "name": str, "enabled": bool, "invert": bool, "blend": "add"|"subtract"}

Alpha uses the same base64-PNG encoding as DodgeBurnMask (0-255 -> 0-1),
simpler than DodgeBurnMask's signed [-1.5, 1.5] range since MaskLayer alpha
is already unsigned [0, 1]. JSON (not a second binary blob) for the
metadata keeps this human-diffable and avoids inventing another binary
sub-format for a handful of small values.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
from typing import Optional

import numpy as np

from raw_mask_layers import (
    SUPPORTED_ADJUSTMENT_KEYS,
    MaskLayer,
    MaskLayerStack,
    _hsl_keys,
)

logger = logging.getLogger(__name__)

# --- Layer count / memory cap ----------------------------------------------
#
# Each mask keeps float32 alpha plus working RGB for its adjustments (~16–24
# B/px at the half-res edit base). Unbounded N blows both RAM and the live-
# preview budget. Cap shape matches docs/FEATURE_FEASIBILITY.md and focus
# stacking's _MAX_FRAMES: byte budget from available RAM first, hard count
# as a backstop. Env overrides:
#   RAWVIEWER_MASK_LAYERS_MAX       hard count (default 24)
#   RAWVIEWER_MASK_LAYERS_RAM_FRAC  fraction of available RAM (default 0.12)
#   RAWVIEWER_MASK_LAYERS_BUDGET_MB fixed budget when RAM is unknown (default 1536)

_HARD_MAX_LAYERS = 24
_BYTES_PER_MASK_PX = 20  # mid of the 16–24 B/px estimate
_DEFAULT_WORKING_PX = 6_000_000  # ~24 MP half-res
_SERIAL_MAX_CHARS = 48 * 1024 * 1024  # refuse before json.loads
_ALPHA_MAX_PIXELS = 40_000_000  # refuse a single absurd PNG


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _available_ram_bytes() -> Optional[int]:
    try:
        from image_cache import _safe_virtual_memory

        return int(_safe_virtual_memory().available)
    except Exception:
        pass
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def max_mask_layers(
    *,
    available_bytes: Optional[int] = None,
    frame_pixels: Optional[int] = None,
) -> int:
    """How many mask layers this machine can reasonably hold right now.

    Byte-budget primary (fraction of available RAM / per-mask footprint at
    the working resolution), hard count as backstop. Callers that know the
    current edit-base size should pass ``frame_pixels`` so a 61 MP file gets
    a tighter count than a 24 MP one.
    """
    hard = max(1, _env_int("RAWVIEWER_MASK_LAYERS_MAX", _HARD_MAX_LAYERS))
    px = (
        int(frame_pixels)
        if frame_pixels is not None and int(frame_pixels) > 0
        else _DEFAULT_WORKING_PX
    )
    per_mask = max(1, int(px) * _BYTES_PER_MASK_PX)

    if available_bytes is None:
        available_bytes = _available_ram_bytes()
    if available_bytes is not None and available_bytes > 0:
        frac = _env_float("RAWVIEWER_MASK_LAYERS_RAM_FRAC", 0.12)
        frac = max(0.02, min(0.50, frac))
        budget = int(available_bytes * frac)
    else:
        budget = max(1, _env_int("RAWVIEWER_MASK_LAYERS_BUDGET_MB", 1536)) * 1024 * 1024

    by_bytes = max(1, budget // per_mask)
    return max(1, min(hard, by_bytes))


def _encode_alpha(alpha: np.ndarray) -> str:
    import cv2

    u8 = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", u8)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


# Brush alpha is stored at half linear resolution. A mask buffer is far
# lower-frequency than the image it masks, and the compositor already resizes
# every layer to the render resolution (raw_mask_layers.resize_alpha_to), so
# the stored size is just another input to a resize that happens regardless.
# Nothing records the resolution -- it is implicit in the PNG -- so sidecars
# written by earlier versions at full resolution keep loading unchanged.
#
# Measured on a 4111x2744 two-layer stack: sidecar 1399 KB -> 312 KB and
# decode 119 ms -> 28 ms, for a worst-case alpha error of 0.12 (AI cutout) to
# 0.20 (soft brush). Half is the floor that is safe for *both* mask kinds --
# soft-brush error plateaus below this, but hard-edged AI cutouts degrade
# steadily (max error 0.53 at 1/8), which reads as haloing along the subject.
#
# This does not compound across save/load: main.py resizes a layer back to the
# working resolution before painting into it, so a re-save always re-encodes
# from full-resolution pixels.
_ALPHA_STORE_DIV = 2

# Below this, halving saves bytes that do not matter and risks mangling a
# small mask, so such buffers are stored verbatim.
_ALPHA_STORE_MIN_DIM = 64


def _downscale_for_store(alpha: np.ndarray) -> np.ndarray:
    """Alpha at its storage resolution (see _ALPHA_STORE_DIV)."""
    if _ALPHA_STORE_DIV <= 1 or alpha is None or getattr(alpha, "ndim", 0) != 2:
        return alpha
    h, w = alpha.shape[:2]
    if min(h, w) < _ALPHA_STORE_MIN_DIM * _ALPHA_STORE_DIV:
        return alpha
    import cv2

    # INTER_AREA, not INTER_LINEAR: averaging over the source footprint keeps
    # a soft brush's falloff intact instead of point-sampling through it.
    return cv2.resize(
        alpha,
        (w // _ALPHA_STORE_DIV, h // _ALPHA_STORE_DIV),
        interpolation=cv2.INTER_AREA,
    )


def _decode_alpha(serial: str) -> Optional[np.ndarray]:
    if not serial:
        return None
    try:
        import cv2

        # Reject before allocating: a base64 string longer than 4/3 of the
        # pixel budget cannot decode into a legal alpha anyway.
        if len(serial) > (_ALPHA_MAX_PIXELS * 4) // 3 + 1024:
            logger.warning("mask alpha serial exceeds size budget; skipped")
            return None
        raw = base64.b64decode(serial.encode("ascii"), validate=False)
        if len(raw) > _ALPHA_MAX_PIXELS * 2:
            logger.warning("mask alpha PNG exceeds size budget; skipped")
            return None
        u8 = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if u8 is None:
            return None
        if int(u8.shape[0]) * int(u8.shape[1]) > _ALPHA_MAX_PIXELS:
            logger.warning(
                "mask alpha %sx%s exceeds pixel budget; skipped",
                u8.shape[1],
                u8.shape[0],
            )
            return None
        return (u8.astype(np.float32) / 255.0)
    except Exception:
        return None


def _all_keys() -> tuple:
    return SUPPORTED_ADJUSTMENT_KEYS + _hsl_keys()


def _entry_slot_count(entry: dict) -> int:
    """How many alpha buffers an entry costs toward the layer budget.

    A group stores each component separately (plus a baked alpha we ignore
    for counting — components dominate). A flat layer costs one.
    """
    raw_components = entry.get("components")
    if isinstance(raw_components, list) and raw_components:
        return max(1, sum(1 for c in raw_components if isinstance(c, dict)))
    return 1


class MaskStackTruncated(Exception):
    """Raised rather than serializing a stack that was truncated on load.

    Returning "" here would be catastrophic: the save path reads "" as "no
    masks" and DELETES the sidecar key, which is precisely the data the
    truncation already put at risk. Callers must leave the stored value alone.
    """


def _worth_storing(key: str, value) -> bool:
    """Is this adjustment a departure from its default, and so worth writing?

    "Non-zero" is NOT the same question, and treating them as one lost data.
    BlurFocus defaults to 100 (raw_mask_layers._ADJUSTMENT_DEFAULTS), so a
    user who pulled Focus all the way to the background (0) had that value
    dropped as if it were untouched -- and on reload _adj_value handed back
    the 100 default, flipping the sharp plane from far to near. The depth of
    field inverted itself between sessions.
    """
    from raw_mask_layers import _ADJUSTMENT_DEFAULTS

    default = float(_ADJUSTMENT_DEFAULTS.get(key, 0.0))
    return abs(float(value) - default) > 1e-4


def _storable_adjustments(adjustments: dict) -> dict:
    return {
        k: round(float(v), 4)
        for k, v in adjustments.items()
        if k in _all_keys() and _worth_storing(k, v)
    }


def serialize_stack(stack: Optional[MaskLayerStack]) -> str:
    """Encode as a JSON list; "" for None/empty (matches serialize_mask's contract)."""
    # Not stack.is_empty(): that reports empty when every layer is merely
    # HIDDEN, which threw the whole stack away on save. The per-layer loop
    # below decides what is worth recording, and the `if not entries` at the
    # end still returns "" when nothing is.
    if stack is None or not stack.layers:
        return ""
    if getattr(stack, "truncated", False):
        raise MaskStackTruncated(
            "refusing to write a mask stack that was truncated on load"
        )
    entries = []
    for layer in stack.layers:
        # Keep empty layers. Skipping them meant Create→Brush never reached
        # XMP, so the first paint's undo snapshot was "no masks" and Ctrl+Z
        # deleted the whole mask instead of just the stroke.
        if getattr(layer, "is_group", False):
            entry = _serialize_group(layer)
            if entry is not None:
                entries.append(entry)
            continue

        alpha_serial = _encode_alpha(_downscale_for_store(layer.alpha))
        if not alpha_serial:
            continue
        adjustments = _storable_adjustments(layer.adjustments)
        entries.append(
            {
                "alpha": alpha_serial,
                "adjustments": adjustments,
                "name": layer.name,
                "enabled": bool(layer.enabled),
                "invert": bool(layer.invert),
                "blend": layer.blend,
                "source": layer.source,
            }
        )
    if not entries:
        return ""
    return json.dumps(entries, separators=(",", ":"))


def _component_entry(comp: MaskLayer) -> Optional[dict]:
    """One component of a group. No adjustments -- the group owns those."""
    entry = {
        "name": comp.name,
        "blend": "subtract" if comp.blend == "subtract" else "add",
        "enabled": bool(comp.enabled),
        "invert": bool(comp.invert),
        "source": comp.source,
        "kind": comp.kind,
    }
    alpha = _encode_alpha(_downscale_for_store(comp.alpha))
    if not alpha:
        return None
    entry["alpha"] = alpha
    return entry


def _serialize_group(layer: MaskLayer) -> Optional[dict]:
    """A group, written so an older build still renders it correctly.

    Two representations of the same coverage share one entry:

      ``alpha``       the components already combined, at storage resolution
      ``components``  the pieces, so this build can still ungroup and edit

    A build predating grouping reads only ``alpha`` and ``adjustments`` and
    draws exactly the right thing -- it just cannot take the group apart.
    Nesting the components instead would leave such a build looking at an
    entry with no ``alpha``, silently dropping the mask. The duplication
    costs bytes; losing a user's mask costs more.
    """
    components = [c for c in layer.components if c is not None]
    if not components:
        return None
    h, w = layer.frame_shape()
    baked = _encode_alpha(_downscale_for_store(layer.alpha_at(h, w)))
    if not baked:
        return None

    comp_entries = [e for e in (_component_entry(c) for c in components) if e]
    if not comp_entries:
        return None

    return {
        "alpha": baked,
        "adjustments": _storable_adjustments(layer.adjustments),
        "name": layer.name,
        "enabled": bool(layer.enabled),
        "invert": bool(layer.invert),
        "blend": layer.blend,
        "source": layer.source,
        "components": comp_entries,
    }


def _deserialize_group(entry: dict, common: dict, raw_components: list):
    """Rebuild a group from ``components``; the baked alpha is ignored here."""
    components = []
    for raw in raw_components:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("kind", "") or "brush")
        shared = {
            "name": str(raw.get("name", "") or ""),
            "blend": "subtract" if raw.get("blend") == "subtract" else "add",
            "enabled": bool(raw.get("enabled", True)),
            "invert": bool(raw.get("invert", False)),
            "source": str(raw.get("source", "") or ""),
            "kind": str(raw.get("kind", "") or "brush"),
        }
        alpha = _decode_alpha(str(raw.get("alpha", "") or ""))
        if alpha is None:
            continue
        components.append(MaskLayer(alpha, **shared))

    if not components:
        return None

    # The group's own alpha is a placeholder -- coverage comes from the
    # components -- but it is sized like one so frame_shape() is sane.
    h, w = components[0].alpha.shape[:2]
    return MaskLayer(
        np.zeros((h, w), dtype=np.float32), components=components, **common
    )


def deserialize_stack(serial: str) -> Optional[MaskLayerStack]:
    if not serial:
        return None
    if len(serial) > _SERIAL_MAX_CHARS:
        logger.warning(
            "RVMaskLayers serial is %s chars (cap %s); refusing to load",
            len(serial),
            _SERIAL_MAX_CHARS,
        )
        return None
    try:
        entries = json.loads(serial)
    except Exception:
        return None
    if not isinstance(entries, list):
        return None

    # Cap before decoding PNGs so a huge list cannot force work proportional
    # to its length.
    #
    # DELIBERATELY the deterministic count cap, NOT max_mask_layers(): that one
    # is a fraction of *currently available RAM*, so it made loading a sidecar
    # nondeterministic. A 12-layer file opened on a busy machine kept the first
    # few layers and dropped the rest; the next slider release then serialized
    # the truncated stack straight back over the sidecar and the masks were
    # gone for good. The same file on the same machine an hour later loaded
    # fine. A memory budget is a sensible limit on masks you are CREATING and a
    # data-loss bug on masks that already exist.
    #
    # This also removes the per-layer psutil query the old code did (the cap
    # was recomputed inside the loop on every entry).
    cap = max(1, _env_int("RAWVIEWER_MASK_LAYERS_MAX", _HARD_MAX_LAYERS))
    layers = []
    slots_used = 0
    truncated = False
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        slots = _entry_slot_count(entry)
        if slots_used + slots > cap:
            truncated = True
            break
        adjustments = entry.get("adjustments") or {}
        if not isinstance(adjustments, dict):
            adjustments = {}
        common = {
            "adjustments": {k: float(v) for k, v in adjustments.items()},
            "name": str(entry.get("name", "") or ""),
            "enabled": bool(entry.get("enabled", True)),
            "invert": bool(entry.get("invert", False)),
            "blend": str(entry.get("blend", "add") or "add"),
            "source": str(entry.get("source", "") or ""),
        }

        raw_components = entry.get("components")
        if isinstance(raw_components, list) and raw_components:
            # Trim components to the remaining budget before building.
            room = cap - slots_used
            if len(raw_components) > room:
                raw_components = [
                    c for c in raw_components if isinstance(c, dict)
                ][:room]
                truncated = True
            group = _deserialize_group(entry, common, raw_components)
            if group is not None:
                layers.append(group)
                slots_used += max(1, len(group.components or []))
                continue
            # Fall through on a malformed component list: the baked "alpha"
            # is still a faithful render of the group, so the mask survives
            # as a flat layer rather than vanishing.

        alpha = _decode_alpha(str(entry.get("alpha", "")))
        if alpha is None:
            continue
        layers.append(
            MaskLayer(alpha, kind=str(entry.get("kind", "") or "brush"), **common)
        )
        slots_used += 1

    if truncated:
        # An error, not a warning: this is the only path that can lose a user's
        # saved masks, and the stack is flagged so it can never be written back
        # over the file it came from (see serialize_stack).
        logger.error(
            "RVMaskLayers has more than %s layer slots; loaded %s and marked "
            "the stack read-only so it cannot overwrite the sidecar",
            cap,
            slots_used,
        )
    if not layers:
        return None
    return MaskLayerStack(layers=layers, truncated=truncated)


# Bounded memo for the render pipeline's read-only lookups, same shape and
# rationale as raw_dodge_burn._deserialize_mask_cached: every render tick
# that has a mask-layers serial (but no live object yet) would otherwise
# re-decode every layer's PNG from scratch and defeat each MaskLayer's own
# bbox/empty caches (a fresh object always starts at version 0).
_DESERIALIZE_CACHE: dict = {}
_DESERIALIZE_CACHE_ORDER: list = []
_DESERIALIZE_CACHE_MAX = 4
_DESERIALIZE_CACHE_LOCK = threading.Lock()


def deserialize_stack_cached(serial: str) -> Optional[MaskLayerStack]:
    """Read-only variant of deserialize_stack() for the render pipeline.

    CALLERS MUST NEVER MUTATE THE RETURNED STACK OR ITS LAYERS -- the same
    instance is shared across calls. Use plain deserialize_stack() for any
    call site that intends to paint on the result.
    """
    if not serial:
        return None
    with _DESERIALIZE_CACHE_LOCK:
        cached = _DESERIALIZE_CACHE.get(serial)
        if cached is not None:
            return cached
    stack = deserialize_stack(serial)
    if stack is None:
        return None
    with _DESERIALIZE_CACHE_LOCK:
        _DESERIALIZE_CACHE[serial] = stack
        _DESERIALIZE_CACHE_ORDER.append(serial)
        if len(_DESERIALIZE_CACHE_ORDER) > _DESERIALIZE_CACHE_MAX:
            oldest = _DESERIALIZE_CACHE_ORDER.pop(0)
            _DESERIALIZE_CACHE.pop(oldest, None)
    return stack
