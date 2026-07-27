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
import threading
from typing import Optional

import numpy as np

from raw_mask_layers import (
    SUPPORTED_ADJUSTMENT_KEYS,
    MaskLayer,
    MaskLayerStack,
    _hsl_keys,
)
from raw_mask_shapes import PARAMETRIC_KINDS

# Reference resolution for a deserialized parametric layer's placeholder
# alpha; see the comment at its construction below.
_SHAPE_REF_DIM = 128


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

        raw = base64.b64decode(serial.encode("ascii"))
        u8 = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if u8 is None:
            return None
        return (u8.astype(np.float32) / 255.0)
    except Exception:
        return None


def _all_keys() -> tuple:
    return SUPPORTED_ADJUSTMENT_KEYS + _hsl_keys()


def serialize_stack(stack: Optional[MaskLayerStack]) -> str:
    """Encode as a JSON list; "" for None/empty (matches serialize_mask's contract)."""
    if stack is None or stack.is_empty():
        return ""
    entries = []
    for layer in stack.layers:
        if layer.is_empty and not layer.adjustments:
            continue
        # A parametric mask stores its geometry, not its pixels: a few dozen
        # bytes instead of a frame-sized PNG, exact at any resolution, and
        # still re-draggable after a reload. Encoding its generated alpha
        # would throw all three away.
        if getattr(layer, "is_parametric", False):
            entries.append(
                {
                    "kind": layer.kind,
                    "params": {
                        k: round(float(v), 5) for k, v in (layer.params or {}).items()
                    },
                    "adjustments": {
                        k: round(float(v), 4)
                        for k, v in layer.adjustments.items()
                        if k in _all_keys() and abs(float(v)) > 1e-4
                    },
                    "name": layer.name,
                    "enabled": bool(layer.enabled),
                    "invert": bool(layer.invert),
                    "blend": layer.blend,
                    "source": layer.source,
                }
            )
            continue
        if getattr(layer, "is_group", False):
            entry = _serialize_group(layer)
            if entry is not None:
                entries.append(entry)
            continue

        alpha_serial = _encode_alpha(_downscale_for_store(layer.alpha))
        if not alpha_serial:
            continue
        adjustments = {
            k: round(float(v), 4)
            for k, v in layer.adjustments.items()
            if k in _all_keys() and abs(float(v)) > 1e-4
        }
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
    if getattr(comp, "is_parametric", False):
        entry["params"] = {
            k: round(float(v), 5) for k, v in (comp.params or {}).items()
        }
        # Its alpha buffer is never read, but the SHAPE is -- alpha_at
        # regenerates at whatever size is asked for.
        entry["shape"] = [int(comp.alpha.shape[0]), int(comp.alpha.shape[1])]
        return entry
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
        "adjustments": {
            k: round(float(v), 4)
            for k, v in layer.adjustments.items()
            if k in _all_keys() and abs(float(v)) > 1e-4
        },
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
            "kind": kind,
        }
        if kind in PARAMETRIC_KINDS:
            params = raw.get("params")
            if not isinstance(params, dict):
                continue
            shape = raw.get("shape") or [_SHAPE_REF_DIM, _SHAPE_REF_DIM]
            try:
                hh, ww = int(shape[0]), int(shape[1])
            except Exception:
                hh = ww = _SHAPE_REF_DIM
            components.append(
                MaskLayer(
                    np.zeros((max(1, hh), max(1, ww)), dtype=np.float32),
                    params={k: float(v) for k, v in params.items()},
                    **shared,
                )
            )
            continue
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
    try:
        entries = json.loads(serial)
    except Exception:
        return None
    if not isinstance(entries, list):
        return None
    layers = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
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
            group = _deserialize_group(entry, common, raw_components)
            if group is not None:
                layers.append(group)
                continue
            # Fall through on a malformed component list: the baked "alpha"
            # is still a faithful render of the group, so the mask survives
            # as a flat layer rather than vanishing.

        kind = str(entry.get("kind", "") or "brush")
        if kind in PARAMETRIC_KINDS:
            params = entry.get("params") or {}
            if not isinstance(params, dict):
                continue
            # A parametric layer's alpha buffer is never read -- alpha_at
            # generates from params at the caller's resolution. But its SHAPE is
            # the coordinate space bbox() reports in, which the compositor then
            # scales to the frame, so it cannot be 1x1: a radial's analytic
            # bbox would quantise to the whole frame and give up the
            # bbox-limited compute that keeps a mask tick inside the preview
            # budget. _SHAPE_REF_DIM is small enough to be free (64 KB) and
            # fine enough that the scaled bbox lands within a couple of pixels.
            layers.append(
                MaskLayer(
                    np.zeros((_SHAPE_REF_DIM, _SHAPE_REF_DIM), dtype=np.float32),
                    kind=kind,
                    params={k: float(v) for k, v in params.items()},
                    **common,
                )
            )
            continue

        alpha = _decode_alpha(str(entry.get("alpha", "")))
        if alpha is None:
            continue
        layers.append(MaskLayer(alpha, **common))
    if not layers:
        return None
    return MaskLayerStack(layers=layers)


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
