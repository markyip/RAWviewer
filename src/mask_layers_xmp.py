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
                }
            )
            continue
        alpha_serial = _encode_alpha(layer.alpha)
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
            }
        )
    if not entries:
        return ""
    return json.dumps(entries, separators=(",", ":"))


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
        }

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
