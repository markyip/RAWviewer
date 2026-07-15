"""
Dodge & burn: local exposure brush with edge-aware mask snapping.

Design:
    - The mask is a single-channel float32 array at the edit base's working
      resolution (half-res live-preview base or full-res export base),
      range roughly [-1.5, +1.5]: positive = dodge (brighten), negative =
      burn (darken), accumulated additively per stroke and clipped.
    - Painting stamps a soft (gaussian-falloff) circular brush at the
      cursor position, scaled by trackpad/tablet pressure, directly into
      the mask -- cheap, O(brush area) per point, not O(image).
    - On stroke release the touched region is edge-snapped: a guided
      filter (existing raw_chroma_denoise.apply_guided_filter, reused
      as-is) using the base image's luminance as the guide pulls the
      mask's soft edges onto real image edges (a hand-painted circle over
      a sky/foreground boundary ends up following the boundary, not the
      brush's geometric edge). This is the "edge-assisted selection".
    - Applied as a per-pixel exposure multiplier: 2**(mask * stops), where
      stops (typically 1.5-2.0) is the panel's Dodge/Burn Strength slider
      -- consistent with how Exposure2012 is applied elsewhere in the
      scene-linear pipeline (raw_edit_pipeline.process_linear_edit_buffer).
    - Persisted in the XMP sidecar as a base64 PNG blob (see
      serialize_mask/deserialize_mask) -- there is no Lightroom-compatible
      local-adjustment schema to target, so this is a RAWviewer-private
      custom element, additive to the existing crs: attributes.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

MASK_KEY = "_dodge_burn_mask_v1"
STRENGTH_KEY = "DodgeBurnStrength"
DEFAULT_STRENGTH = 1.75  # stops at mask value +/-1.0
MASK_CLIP = 1.5


@dataclass
class DodgeBurnMask:
    """A dodge/burn mask at a fixed working resolution.

    ``version`` bumps on every in-place mutation (stamp_brush,
    edge_snap_region) and gates the caches below: apply_dodge_burn() runs
    on EVERY editor render tick (every slider drag, not just brush
    strokes), but the mask itself only changes during active painting --
    without these caches, an unchanged mask still paid a full is_empty
    scan + resize + gain recompute on every unrelated slider tick,
    measured at ~56ms on a 2200x3300 half-res base (over half the 80ms
    preview throttle window). Cache fields are excluded from dataclass
    __eq__/repr (they're derived, not identity).
    """

    data: np.ndarray  # float32, (H, W), range [-MASK_CLIP, MASK_CLIP]
    version: int = field(default=0, compare=False, repr=False)
    _empty_cache: Optional[tuple] = field(default=None, compare=False, repr=False)
    _gain_cache: Optional[tuple] = field(default=None, compare=False, repr=False)

    @classmethod
    def empty(cls, height: int, width: int) -> "DodgeBurnMask":
        return cls(np.zeros((height, width), dtype=np.float32))

    def touch(self) -> None:
        """Call after any in-place mutation of ``data`` to invalidate caches."""
        self.version += 1

    @property
    def is_empty(self) -> bool:
        cached = self._empty_cache
        if cached is not None and cached[0] == self.version:
            return cached[1]
        result = not np.any(np.abs(self.data) > 1e-4)
        self._empty_cache = (self.version, result)
        return result


def stamp_brush(
    mask: DodgeBurnMask,
    cx: float,
    cy: float,
    radius: float,
    strength: float,
    dodge: bool,
) -> tuple[int, int, int, int]:
    """Add one soft circular stamp to ``mask`` in place.

    ``strength`` is the per-point delta at the brush center, already scaled
    by pressure and the UI strength slider by the caller (0..~0.35 per
    point is a reasonable single-point delta at 60fps-ish sampling; repeated
    overlapping stamps during a slow drag accumulate to the full effect,
    exactly like a real brush).

    Returns the touched (x0, y0, x1, y1) bounding box in mask pixel
    coordinates, clamped to the mask bounds, for incremental/edge-snap work.
    """
    h, w = mask.data.shape
    r = max(1.0, float(radius))
    x0 = max(0, int(cx - r - 1))
    x1 = min(w, int(cx + r + 2))
    y0 = max(0, int(cy - r - 1))
    y1 = min(h, int(cy + r + 2))
    if x1 <= x0 or y1 <= y0:
        return (x0, y0, x1, y1)

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    # Soft falloff: solid core to ~60% radius, cosine falloff to the edge.
    falloff = np.clip(1.0 - (dist - r * 0.6) / max(r * 0.4, 1e-3), 0.0, 1.0)
    falloff = falloff * falloff * (3.0 - 2.0 * falloff)  # smoothstep

    delta = falloff * float(strength) * (1.0 if dodge else -1.0)
    region = mask.data[y0:y1, x0:x1]
    np.clip(region + delta, -MASK_CLIP, MASK_CLIP, out=region)
    mask.touch()
    return (x0, y0, x1, y1)


def edge_snap_region(
    mask: DodgeBurnMask,
    luminance: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    radius: int = 12,
    eps: float = 1e-3,
    pad: int = 16,
) -> None:
    """Edge-snap ``mask`` within (and a bit around) ``bbox``, in place.

    Uses the guided filter already shipped for chroma denoise
    (raw_chroma_denoise.apply_guided_filter) with the base image's
    luminance as the guide: the mask's soft brush edges relax onto real
    luminance edges within the filter radius, which is what makes a rough
    stroke "stick" to a subject's outline instead of bleeding across it.
    ``pad`` extends the filtered region beyond the touched bbox so the
    guided filter has context outside the stroke (avoids a visible seam at
    the exact stroke boundary).
    """
    from raw_chroma_denoise import apply_guided_filter

    h, w = mask.data.shape
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return
    guide = luminance[y0:y1, x0:x1]
    if guide.shape != (y1 - y0, x1 - x0):
        return
    src = mask.data[y0:y1, x0:x1]
    snapped = apply_guided_filter(guide, src, radius, eps)
    np.clip(snapped, -MASK_CLIP, MASK_CLIP, out=snapped)
    mask.data[y0:y1, x0:x1] = snapped
    mask.touch()


def resize_mask_to(mask: DodgeBurnMask, height: int, width: int) -> np.ndarray:
    """Nearest-scale the mask to a target (H, W) via cv2 (bilinear)."""
    if mask.data.shape == (height, width):
        return mask.data
    import cv2

    return cv2.resize(mask.data, (width, height), interpolation=cv2.INTER_LINEAR)


def apply_dodge_burn(
    img: np.ndarray, mask: Optional[DodgeBurnMask], stops: float
) -> np.ndarray:
    """Scene-linear per-pixel exposure multiplier: img * 2**(mask * stops).

    ``img`` is (H, W, 3) scene-linear float. No-op (returns img unchanged)
    when the mask is None/empty/all-zero -- the common case, so callers can
    unconditionally call this without an extra branch.

    The resize+exp2 gain map is cached on the mask, keyed by
    (mask.version, target shape, stops): this function runs on every
    editor render tick (every slider drag, not just brush strokes), and
    without the cache, an UNCHANGED mask still paid a full resize + exp2
    over the whole frame every tick -- measured ~56ms on a 2200x3300 half-
    res base, more than half the 80ms live-preview throttle window.
    """
    if mask is None or mask.is_empty:
        return img
    h, w = img.shape[:2]
    cache_key = (mask.version, h, w, round(float(stops), 6))
    cached = mask._gain_cache
    if cached is not None and cached[0] == cache_key:
        gain = cached[1]
    else:
        m = resize_mask_to(mask, h, w)
        gain = np.exp2(m * float(stops)).astype(np.float32)
        mask._gain_cache = (cache_key, gain)
    return img * gain[..., np.newaxis]


def serialize_mask(mask: Optional[DodgeBurnMask]) -> str:
    """Encode as base64 PNG (8-bit, mask value = (px/255)*2*CLIP - CLIP)."""
    if mask is None or mask.is_empty:
        return ""
    import cv2

    u8 = np.clip(
        (mask.data + MASK_CLIP) / (2.0 * MASK_CLIP) * 255.0 + 0.5, 0, 255
    ).astype(np.uint8)
    ok, buf = cv2.imencode(".png", u8)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def deserialize_mask(serial: str) -> Optional[DodgeBurnMask]:
    if not serial:
        return None
    try:
        import cv2

        raw = base64.b64decode(serial.encode("ascii"))
        u8 = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if u8 is None:
            return None
        data = (u8.astype(np.float32) / 255.0) * (2.0 * MASK_CLIP) - MASK_CLIP
        return DodgeBurnMask(data)
    except Exception:
        return None


# Bounded memo for the render pipeline's read-only mask lookups (see
# _deserialize_mask_cached). Small (few entries): the working set is
# realistically 1-2 distinct mask strings at a time (current file, maybe a
# stale in-flight render generation).
_DESERIALIZE_CACHE: dict = {}
_DESERIALIZE_CACHE_ORDER: list = []
_DESERIALIZE_CACHE_MAX = 4


def _deserialize_mask_cached(serial: str) -> Optional[DodgeBurnMask]:
    """Read-only variant of deserialize_mask() for the render pipeline.

    process_linear_edit_buffer(_staged) and _apply_adjustments_to_srgb call
    this on EVERY render tick whenever a mask is present, including ticks
    where an unrelated pre-tone key (Exposure/Temperature/Tint/NR) changed
    but the mask itself did not -- deserialize_mask() re-decodes the PNG
    and builds a brand-new DodgeBurnMask from scratch every call, which
    also defeats apply_dodge_burn's own per-instance gain cache (a fresh
    object always starts at version 0). Caching the decoded object here
    (keyed by the serial string) lets repeated same-string calls reuse
    both the decode AND the gain cache.

    CALLERS MUST NEVER MUTATE THE RETURNED MASK (stamp_brush /
    edge_snap_region) -- the same instance is shared across calls. Use
    plain deserialize_mask() for any call site that intends to paint on
    the result (e.g. loading a file's saved mask into main.py's editable
    ``self._dodge_burn_mask``).
    """
    if not serial:
        return None
    cached = _DESERIALIZE_CACHE.get(serial)
    if cached is not None:
        return cached
    mask = deserialize_mask(serial)
    if mask is None:
        return None
    _DESERIALIZE_CACHE[serial] = mask
    _DESERIALIZE_CACHE_ORDER.append(serial)
    if len(_DESERIALIZE_CACHE_ORDER) > _DESERIALIZE_CACHE_MAX:
        oldest = _DESERIALIZE_CACHE_ORDER.pop(0)
        _DESERIALIZE_CACHE.pop(oldest, None)
    return mask
