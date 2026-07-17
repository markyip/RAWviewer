"""
Spot heal: paint a soft coverage mask, remove smudges via OpenCV inpaint.

Design:
    - HealMask is float32 in [0, 1] at the edit-base working resolution
      (same framing as dodge/burn). Positive coverage = region to remove.
    - Painting stamps a soft circular brush (reuses dodge/burn falloff).
    - Applied in the scene-linear pipeline after dodge/burn: encode to
      8-bit gamma, ``cv2.inpaint`` (Telea), decode, soft-composite with
      the coverage mask so feathered brush edges blend cleanly.
    - Inpaint result is cached on the mask instance keyed by
      (version, H, W, radius) so slider ticks do not re-run OpenCV.
    - Persisted in XMP as a private base64 PNG (``SpotHealMask``), same
      pattern as DodgeBurnMask — not a Lightroom-compatible schema.
"""

from __future__ import annotations

import base64
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

MASK_KEY = "_spot_heal_mask_v1"
MASK_OBJ_KEY = "_spot_heal_mask_obj"  # live HealMask; never write to XMP
INPAINT_RADIUS = 5  # OpenCV inpaint neighborhood radius (pixels at buffer res)


def mask_stage_fingerprint(mask: "HealMask") -> str:
    """Cheap stage-cache key: shape + mutation version (no PNG encode)."""
    h, w = mask.data.shape[:2]
    return f"mem:{int(h)}x{int(w)}:v{int(mask.version)}"


def resolve_mask_from_adj(adj: dict | None) -> Optional["HealMask"]:
    """Prefer live mask object; fall back to XMP/base64 serial."""
    if not adj:
        return None
    obj = adj.get(MASK_OBJ_KEY)
    if isinstance(obj, HealMask):
        return obj
    serial = str(adj.get(MASK_KEY, "") or "")
    if not serial or serial.startswith("mem:"):
        return None
    return _deserialize_mask_cached(serial)


@dataclass
class HealMask:
    """Soft coverage mask (0..1) for regions to inpaint."""

    data: np.ndarray  # float32 (H, W), 0..1
    version: int = 0
    _inpaint_cache: object = field(default=None, repr=False, compare=False)
    _empty_cache: object = field(default=None, repr=False, compare=False)

    @classmethod
    def empty(cls, h: int, w: int) -> "HealMask":
        return cls(np.zeros((int(h), int(w)), dtype=np.float32))

    @property
    def is_empty(self) -> bool:
        cached = self._empty_cache
        if cached is not None and cached[0] == self.version:
            return bool(cached[1])
        empty = not bool(np.any(self.data > 1e-4))
        self._empty_cache = (self.version, empty)
        return empty

    def touch(self) -> None:
        self.version = int(self.version) + 1
        self._inpaint_cache = None
        self._empty_cache = None


def stamp_heal_brush(
    mask: HealMask,
    cx: float,
    cy: float,
    radius: float,
    strength: float,
) -> tuple[int, int, int, int]:
    """Accumulate soft coverage into ``mask`` (max-blend). Returns bbox."""
    from raw_dodge_burn import circular_brush_falloff

    h, w = mask.data.shape
    r = max(1.0, float(radius))
    x0 = max(0, int(cx - r - 1))
    x1 = min(w, int(cx + r + 2))
    y0 = max(0, int(cy - r - 1))
    y1 = min(h, int(cy + r + 2))
    if x1 <= x0 or y1 <= y0:
        return (x0, y0, x1, y1)

    falloff = circular_brush_falloff(y0, y1, x0, x1, cx, cy, r)
    # Heal stamps need a strong single-click presence; max-blend with a
    # boosted center so a short dab covers the defect for Telea.
    amt = falloff * min(1.0, float(strength) * 1.35)
    region = mask.data[y0:y1, x0:x1]
    np.maximum(region, amt, out=region)
    np.clip(region, 0.0, 1.0, out=region)
    mask.touch()
    return (x0, y0, x1, y1)


def erase_heal_brush(
    mask: HealMask,
    cx: float,
    cy: float,
    radius: float,
    strength: float,
) -> tuple[int, int, int, int]:
    """Pull heal coverage toward zero under a soft brush."""
    from raw_dodge_burn import circular_brush_falloff

    h, w = mask.data.shape
    r = max(1.0, float(radius))
    x0 = max(0, int(cx - r - 1))
    x1 = min(w, int(cx + r + 2))
    y0 = max(0, int(cy - r - 1))
    y1 = min(h, int(cy + r + 2))
    if x1 <= x0 or y1 <= y0:
        return (x0, y0, x1, y1)

    falloff = circular_brush_falloff(y0, y1, x0, x1, cx, cy, r)
    region = mask.data[y0:y1, x0:x1]
    region *= np.clip(1.0 - falloff * float(strength), 0.0, 1.0)
    mask.touch()
    return (x0, y0, x1, y1)


def resize_mask_to(mask: HealMask, h: int, w: int) -> np.ndarray:
    """Resize coverage to (h, w); returns a float32 view/copy."""
    import cv2

    if mask.data.shape == (h, w):
        return mask.data
    return cv2.resize(mask.data, (int(w), int(h)), interpolation=cv2.INTER_LINEAR).astype(
        np.float32
    )


def apply_spot_heal(
    img: np.ndarray,
    mask: Optional[HealMask],
    *,
    inpaint_radius: int = INPAINT_RADIUS,
) -> np.ndarray:
    """Inpaint covered pixels; soft-composite back onto ``img``.

    ``img`` is (H, W, 3) scene-linear float. No-op when mask is empty.
    OpenCV inpaint needs 8-bit, so we gamma-encode → inpaint → decode,
    then lerp with the soft coverage so brush edges stay feathered.

    Work is limited to the painted bounding box (+ pad) so a small heal
    stroke does not Telea-process a full sensor frame (which looked like
    a hang / no-op on large RAW bases).
    """
    if mask is None or mask.is_empty or img is None:
        return img
    h, w = img.shape[:2]
    radius = max(1, int(inpaint_radius))
    # Must include the source buffer identity — caching only by mask
    # version returned a stale healed frame after Exposure/WB changes.
    cache_key = (mask.version, h, w, radius, id(img))
    cached = mask._inpaint_cache
    if cached is not None and cached[0] == cache_key:
        return cached[1]

    import cv2

    coverage = resize_mask_to(mask, h, w)
    bin_mask = ((coverage > 0.08).astype(np.uint8)) * 255
    if not np.any(bin_mask):
        return img

    ys, xs = np.where(bin_mask > 0)
    pad = max(radius * 3, 12)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)
    if y1 <= y0 or x1 <= x0:
        return img

    roi = img[y0:y1, x0:x1]
    cov_roi = coverage[y0:y1, x0:x1]
    bin_roi = bin_mask[y0:y1, x0:x1]
    # Dilate inside the ROI so Telea has a clean border of source pixels.
    bin_roi = cv2.dilate(bin_roi, np.ones((3, 3), np.uint8), iterations=1)

    peak = float(np.percentile(img, 99.8))
    peak = max(peak, 1e-4)
    norm = np.clip(roi / peak, 0.0, 1.0)
    u8 = np.clip(np.power(norm, 1.0 / 2.2) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    healed_u8 = cv2.inpaint(u8, bin_roi, float(radius), cv2.INPAINT_TELEA)
    healed = np.power(healed_u8.astype(np.float32) / 255.0, 2.2) * peak

    alpha = np.clip(cov_roi, 0.0, 1.0).astype(np.float32)
    # Heal always runs at full inpaint strength inside the painted region.
    # Soft coverage only feathers the rim (Flow grows how much is covered).
    alpha = np.where(alpha > 0.08, np.clip(0.35 + 0.65 * alpha, 0.0, 1.0), alpha)
    alpha = alpha[..., np.newaxis]
    patched = roi * (1.0 - alpha) + healed * alpha
    out = np.array(img, dtype=np.float32, copy=True)
    out[y0:y1, x0:x1] = patched
    mask._inpaint_cache = (cache_key, out)
    return out


def serialize_mask(mask: Optional[HealMask]) -> str:
    """Encode as base64 PNG (8-bit coverage)."""
    if mask is None or mask.is_empty:
        return ""
    import cv2

    u8 = np.clip(mask.data * 255.0 + 0.5, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", u8)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def deserialize_mask(serial: str) -> Optional[HealMask]:
    if not serial:
        return None
    try:
        import cv2

        raw = base64.b64decode(serial.encode("ascii"))
        u8 = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if u8 is None:
            return None
        data = (u8.astype(np.float32) / 255.0).clip(0.0, 1.0)
        return HealMask(data)
    except Exception:
        return None


_DESERIALIZE_CACHE: dict = {}
_DESERIALIZE_CACHE_ORDER: list = []
_DESERIALIZE_CACHE_MAX = 4
_DESERIALIZE_CACHE_LOCK = threading.Lock()


def _deserialize_mask_cached(serial: str) -> Optional[HealMask]:
    """Read-only deserialize for the render pipeline (do not mutate result)."""
    if not serial:
        return None
    with _DESERIALIZE_CACHE_LOCK:
        cached = _DESERIALIZE_CACHE.get(serial)
        if cached is not None:
            return cached
    mask = deserialize_mask(serial)
    if mask is None:
        return None
    with _DESERIALIZE_CACHE_LOCK:
        _DESERIALIZE_CACHE[serial] = mask
        _DESERIALIZE_CACHE_ORDER.append(serial)
        if len(_DESERIALIZE_CACHE_ORDER) > _DESERIALIZE_CACHE_MAX:
            oldest = _DESERIALIZE_CACHE_ORDER.pop(0)
            _DESERIALIZE_CACHE.pop(oldest, None)
    return mask
