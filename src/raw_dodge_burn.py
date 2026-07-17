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
    - Edge-assisted painting (default on): each stamp is gated by a seed
      flood-fill within a luma tolerance (connected component), so a hard
      subject boundary blocks paint from bleeding onto a neighbor mid-drag
      even when both sides share a similar tone. Soft luma similarity and a
      small gaussian soften the region mask; release-time guided-filter
      edge-snap still tidies the stroke onto real luminance edges.
    - On stroke release the touched region is edge-snapped: a guided
      filter (existing raw_chroma_denoise.apply_guided_filter, reused
      as-is) using the base image's luminance as the guide pulls the
      mask's soft edges onto real image edges (a hand-painted circle over
      a sky/foreground boundary ends up following the boundary, not the
      brush's geometric edge). This is the final tidy-up pass.
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
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

MASK_KEY = "_dodge_burn_mask_v1"
MASK_OBJ_KEY = "_dodge_burn_mask_obj"  # live DodgeBurnMask; never write to XMP
STRENGTH_KEY = "DodgeBurnStrength"
DEFAULT_STRENGTH = 1.75  # stops at mask value +/-1.0
MASK_CLIP = 1.5


def mask_stage_fingerprint(mask: "DodgeBurnMask") -> str:
    """Cheap stage-cache key: shape + mutation version (no PNG encode)."""
    h, w = mask.data.shape[:2]
    return f"mem:{int(h)}x{int(w)}:v{int(mask.version)}"


def resolve_mask_from_adj(adj: dict | None) -> Optional["DodgeBurnMask"]:
    """Prefer live mask object; fall back to XMP/base64 serial (not mem: fingerprints)."""
    if not adj:
        return None
    obj = adj.get(MASK_OBJ_KEY)
    if isinstance(obj, DodgeBurnMask):
        return obj
    serial = str(adj.get(MASK_KEY, "") or "")
    if not serial or serial.startswith("mem:"):
        return None
    return _deserialize_mask_cached(serial)


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


def _sample_luma(luminance: np.ndarray, cx: float, cy: float) -> float:
    """Nearest-pixel luminance at (cx, cy), clamped to bounds."""
    h, w = luminance.shape[:2]
    x = int(np.clip(round(cx), 0, w - 1))
    y = int(np.clip(round(cy), 0, h - 1))
    return float(luminance[y, x])


def _edge_assist_gate(
    luminance: np.ndarray,
    y0: int,
    x0: int,
    y1: int,
    x1: int,
    cx: float,
    cy: float,
    *,
    luma_tol: float = 0.10,
) -> np.ndarray:
    """Per-pixel [0,1] weight: keep paint on the seed's connected subject.

    Previous approach multiplied by *local* Sobel magnitude — that dims paint
    sitting *on* an edge, but does not stop bleed onto a neighbor across the
    edge (the classic failure mode). Correct behavior is a seed flood-fill
    within a luma tolerance so a hard subject boundary breaks connectivity
    even when the far side happens to share a similar tone.

    Steps:
      1. Soft luma similarity to the brush-center seed (smooth falloff)
      2. Binary connected component of similar pixels, grown from the seed
      3. Soften the region mask so stamp edges aren't a hard cut
    """
    patch = luminance[y0:y1, x0:x1].astype(np.float32, copy=False)
    h, w = patch.shape
    seed = _sample_luma(luminance, cx, cy)
    # Similarity gate (display-linear or scene-linear luma both work; tol is
    # relative to the local dynamic range).
    local_span = float(max(0.08, np.percentile(patch, 90) - np.percentile(patch, 10)))
    tol = max(float(luma_tol), 0.45 * local_span)
    diff = np.abs(patch - seed)
    sim = np.clip(1.0 - diff / tol, 0.0, 1.0)
    sim = sim * sim * (3.0 - 2.0 * sim)  # smoothstep

    similar = diff <= tol
    sx = int(np.clip(round(cx), x0, max(x0, x1 - 1))) - x0
    sy = int(np.clip(round(cy), y0, max(y0, y1 - 1))) - y0
    sx = int(np.clip(sx, 0, w - 1))
    sy = int(np.clip(sy, 0, h - 1))

    connected = np.zeros((h, w), dtype=np.float32)
    if similar[sy, sx]:
        try:
            import cv2

            # FLOODFILL_MASK_ONLY: non-zero mask cells block fill; filled
            # cells are written as 255 in the padded mask.
            ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            ff_mask[1 : h + 1, 1 : w + 1][~similar] = 1
            fill_img = np.zeros((h, w), dtype=np.uint8)
            flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY
            cv2.floodFill(fill_img, ff_mask, (sx, sy), 0, 0, 0, flags)
            connected = (ff_mask[1 : h + 1, 1 : w + 1] == 255).astype(np.float32)
        except Exception:
            # BFS fallback (4-connected) when OpenCV is unavailable.
            from collections import deque

            seen = np.zeros((h, w), dtype=bool)
            q: deque = deque()
            q.append((sy, sx))
            seen[sy, sx] = True
            while q:
                y, x = q.popleft()
                connected[y, x] = 1.0
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and not seen[ny, nx]
                        and similar[ny, nx]
                    ):
                        seen[ny, nx] = True
                        q.append((ny, nx))
    else:
        # Seed pixel itself failed the hard threshold — still allow a tiny core.
        connected[sy, sx] = 1.0

    # Soften the hard flood region so stamps don't look cut with scissors.
    try:
        import cv2

        soft_sigma = max(0.8, 0.06 * max(h, w))
        region = cv2.GaussianBlur(connected, (0, 0), soft_sigma)
    except Exception:
        region = connected

    gate = region * sim
    # Preserve a small core around the seed so the brush never fully dies
    # under aggressive tolerances.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2)
    core_r = max(2.0, 0.12 * max(h, w))
    core = np.clip(1.0 - dist / core_r, 0.0, 1.0)
    gate = np.maximum(gate, core * sim)
    return np.clip(gate, 0.0, 1.0).astype(np.float32)


def circular_brush_falloff(
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    cx: float,
    cy: float,
    radius: float,
) -> np.ndarray:
    """Soft circular brush kernel on ``[y0:y1, x0:x1]`` (peak 1 at center).

    Raised-cosine (Hann) radial profile: full strength at the center, smooth
    gradient to zero at ``radius``. Hard-clips outside ``radius`` so the
    axis-aligned stamp bbox corners stay empty (avoids rectangular live blits).
    """
    r = max(1.0, float(radius))
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dist = np.sqrt((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2)
    t = np.clip(dist / r, 0.0, 1.0)
    # Hann window: 1 at center, 0 at edge, continuous first derivative.
    falloff = (0.5 * (1.0 + np.cos(np.pi * t))).astype(np.float32)
    falloff[dist > r] = 0.0
    return falloff


def stamp_brush(
    mask: DodgeBurnMask,
    cx: float,
    cy: float,
    radius: float,
    strength: float,
    dodge: bool,
    *,
    luminance: Optional[np.ndarray] = None,
    edge_assist: bool = True,
    luma_tol: float = 0.10,
) -> tuple[int, int, int, int]:
    """Add one soft circular stamp to ``mask`` in place.

    ``strength`` is the per-point delta at the brush center, already scaled
    by pressure and the UI strength slider by the caller (0..~0.35 per
    point is a reasonable single-point delta at 60fps-ish sampling; repeated
    overlapping stamps during a slow drag accumulate to the full effect,
    exactly like a real brush).

    When ``luminance`` is provided and ``edge_assist`` is True, the stamp is
    gated to the seed's connected luma region (flood-fill + soft similarity)
    so paint stays on the surface under the cursor instead of spilling onto a
    neighboring subject mid-stroke. Release-time ``edge_snap_region`` still
    runs for a final guided-filter tidy-up.

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

    falloff = circular_brush_falloff(y0, y1, x0, x1, cx, cy, r)

    if (
        edge_assist
        and luminance is not None
        and luminance.shape[:2] == (h, w)
    ):
        falloff = falloff * _edge_assist_gate(
            luminance, y0, x0, y1, x1, cx, cy, luma_tol=luma_tol
        )

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
    # Preserve peak energy: guided filter can flatten soft stamps so the
    # post-settle render looks like "no effect" vs the live uint8 patch.
    src_peak = float(np.max(np.abs(src)))
    snapped = apply_guided_filter(guide, src, radius, eps)
    np.clip(snapped, -MASK_CLIP, MASK_CLIP, out=snapped)
    if src_peak > 1e-4:
        sn_peak = float(np.max(np.abs(snapped)))
        if sn_peak > 1e-6 and sn_peak < src_peak * 0.85:
            snapped *= src_peak / sn_peak
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
        try:
            from perf_metrics import perf_mark
            import time as _time

            t0 = _time.perf_counter()
            m = resize_mask_to(mask, h, w)
            gain = np.exp2(m * float(stops)).astype(np.float32)
            mask._gain_cache = (cache_key, gain)
            perf_mark(
                "db_apply",
                (_time.perf_counter() - t0) * 1000.0,
                h=h,
                w=w,
                cache="miss",
            )
        except Exception:
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
_DESERIALIZE_CACHE_LOCK = threading.Lock()


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
