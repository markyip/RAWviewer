"""
Mask Editing / Local Adjustments.

Design (mirrors raw_dodge_burn.DodgeBurnMask):
    - Each MaskLayer carries a float32 [0, 1] alpha buffer at the edit
      base's working resolution plus a per-layer adjustment dict (a
      subset of the global crs: keys -- see SUPPORTED_ADJUSTMENT_KEYS).
      Tone Curve is deliberately NOT supported per mask yet: it is the
      single most expensive item in the global pipeline (a 65,536-entry
      LUT build per evaluation, per docs/FEATURE_FEASIBILITY.md), and
      N per-mask LUT builds per tick needs its own perf pass before it's
      safe to add -- out of scope here.
    - Layers composite back-to-front: out = out*(1-a) + adjusted(out)*a,
      each layer's adjustment computed against the upstream image (not
      the original base), matching how a real stacked local-adjustment
      tool behaves when regions overlap.
    - **bbox-limited compute** (Phase 0 finding): a naive full-frame
      adjustment+blend per layer measured 120-140ms/layer at a 2200x3300
      half-res base -- one layer alone blows the entire 80ms preview
      budget. Real masks (brush/gradient) cover a fraction of the frame,
      so each layer's adjustment and blend run only inside the alpha
      channel's non-zero bounding box (cached alongside is_empty, gated
      on version) -- cost scales with mask *area*, not frame area, same
      idea as raw_spot_heal.py's ROI-limited inpaint. Spatially-aware
      adjustments (Dehaze, Sharpness/Clarity/Defringe) additionally pad
      that bbox before filtering and crop back to the tight bbox after
      (same pad-before-filter-crop-after pattern raw_spot_heal.py uses,
      cited explicitly in the feasibility doc as the fix for edge
      fringing when a neighborhood op is cropped tightly to a mask).
    - **stack-level composite cache** (Phase 0 finding): caching each
      layer's adjusted render individually breaks down for N>1 layers,
      because every layer's blend output is a *new* array each call -- so
      layer 2's upstream-image identity never matches between ticks even
      when nothing changed, and the cache never hits past layer 0.
      Instead the whole stack's composite output is cached once, keyed by
      (stack fingerprint, shape, id(base image)): an unrelated LATER
      slider tick (mask layers + everything upstream of them unchanged)
      hits this cache directly, matching how apply_dodge_burn caches its
      single gain map on the mask instance.
    - **Drag-time throttling** (Phase 1, addressing the Phase 0 verdict
      that a mask's OWN edit tick still costs real time): apply_mask_layers
      accepts an optional ``active_index``. While set (a live brush stroke
      or slider drag on that one layer), everything BEFORE it in the stack
      is served from a separate prefix cache keyed the same way as the
      full composite, so a drag on layer i no longer repays layers
      0..i-1 on every tick -- only the active layer (and any layers
      stacked after it, the rare case) recompute. The caller passes
      active_index=None to get the fully-cached, fully up-to-date settle
      render once the drag ends, mirroring the app's existing
      preview_lite (drag) vs. settle split for the main pipeline.
    - This module is the data model + compositing engine only -- it is
      not yet wired into the XMP sidecar (see mask_layers_xmp.py for that)
      or the Adjust panel UI. See docs/FEATURE_FEASIBILITY.md "Mask
      Editing / Local Adjustments" for the full design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

MASK_LAYERS_KEY = "_mask_layers_v1"  # XMP serial key, see mask_layers_xmp.py
MASK_LAYERS_OBJ_KEY = "_mask_layers_obj"  # live MaskLayerStack; never write to XMP

# WB/Light keys are absolute targets (e.g. Temperature default 5500K = a
# no-op), not deltas, matching the global crs: semantics -- a per-mask
# "Temperature=6500" means "push this region toward 6500K", independent of
# the global WB. Tone Curve is intentionally excluded (see module docstring).
SUPPORTED_ADJUSTMENT_KEYS = (
    "Exposure2012",
    "Contrast2012",
    "Temperature",
    "Tint",
    "Saturation",
    "Vibrance",
    "Dehaze",
    "Sharpness",
    "Clarity2012",
    "Defringe",
)


def _hsl_keys() -> tuple:
    from raw_hsl import HSL_COLOR_NAMES

    return tuple(
        f"{prefix}Adjustment{color}"
        for color in HSL_COLOR_NAMES
        for prefix in ("Hue", "Saturation", "Luminance")
    )


def _all_fingerprint_keys() -> tuple:
    return SUPPORTED_ADJUSTMENT_KEYS + _hsl_keys()


# Neighborhood ops (blur/guided-filter based) need context outside the
# tight alpha bbox or their result fringes at the crop boundary -- see
# raw_spot_heal.py:191's "clean border of source pixels" comment, the
# precedent this pads-then-crops pattern follows.
_SPATIAL_ADJUSTMENT_KEYS = ("Dehaze", "Sharpness", "Clarity2012", "Defringe")
_SPATIAL_FILTER_PAD = 40

# Bbox padding (px, at the mask's own resolution) for the tight alpha
# extent itself, independent of any neighborhood-op padding above.
_BBOX_PAD = 2


def _needs_spatial_pad(adjustments: dict) -> bool:
    return any(abs(float(adjustments.get(k, 0.0))) > 1e-4 for k in _SPATIAL_ADJUSTMENT_KEYS)


def _apply_layer_adjustments(img: np.ndarray, adjustments: dict) -> np.ndarray:
    """Apply one layer's adjustment set, reusing the real global adjustment
    math (WB/tint, saturation-vibrance, dehaze, HSL, detail) wherever it
    already operates as a pure per-region function -- only Exposure and
    Contrast use a local approximation here (Contrast2012 is normally
    folded into the global PV2012 tone LUT, out of scope per the module
    docstring). Order mirrors the global pipeline: WB -> exposure ->
    contrast -> saturation/vibrance -> dehaze -> HSL -> detail.
    """
    from raw_edit_pipeline import _apply_saturation_vibrance, _apply_wb_tint
    from raw_effects import apply_dehaze
    from raw_detail_enhance import apply_detail_enhancements
    from raw_hsl import apply_hsl_adjustments

    out = img
    temperature = float(adjustments.get("Temperature", 0.0))
    tint = float(adjustments.get("Tint", 0.0))
    if abs(temperature) > 1e-4 or abs(tint) > 1e-4:
        # _apply_wb_tint mutates in place and reads the ref temp (5500,
        # neutral) off an empty dict when Temperature isn't itself passed
        # through -- pass 5500 explicitly when the layer hasn't set an
        # absolute Temperature so Tint-only layers don't accidentally hit
        # the <=1000K "relative scale" branch.
        wb_adj = {
            "Temperature": temperature if abs(temperature) > 1e-4 else 5500.0,
            "Tint": tint,
        }
        out = _apply_wb_tint(out.copy(), wb_adj)

    exposure = float(adjustments.get("Exposure2012", 0.0))
    if abs(exposure) > 1e-4:
        out = out * (2.0 ** exposure)

    contrast = float(adjustments.get("Contrast2012", 0.0))
    if abs(contrast) > 1e-4:
        factor = 1.0 + (contrast / 100.0)
        out = (out - 0.5) * factor + 0.5

    saturation = float(adjustments.get("Saturation", 0.0))
    vibrance = float(adjustments.get("Vibrance", 0.0))
    if abs(saturation) > 1e-4 or abs(vibrance) > 1e-4:
        out = _apply_saturation_vibrance(out, {"Saturation": saturation, "Vibrance": vibrance})

    dehaze = float(adjustments.get("Dehaze", 0.0))
    if abs(dehaze) > 1e-4:
        out = apply_dehaze(out, dehaze, preview=True)

    if any(abs(float(adjustments.get(k, 0.0))) > 1e-4 for k in _hsl_keys()):
        out = apply_hsl_adjustments(out, adjustments)

    if any(abs(float(adjustments.get(k, 0.0))) > 1e-4 for k in ("Sharpness", "Clarity2012", "Defringe")):
        out = apply_detail_enhancements(out, adjustments)

    return out


def _nonzero_bbox(alpha: np.ndarray, pad: int = _BBOX_PAD) -> Optional[tuple]:
    """(y0, y1, x0, x1) tight bbox around alpha > 0, padded and clamped. None if empty."""
    rows = np.any(alpha > 1e-4, axis=1)
    if not rows.any():
        return None
    cols = np.any(alpha > 1e-4, axis=0)
    h, w = alpha.shape[:2]
    y_idx = np.flatnonzero(rows)
    x_idx = np.flatnonzero(cols)
    y0 = max(0, int(y_idx[0]) - pad)
    y1 = min(h, int(y_idx[-1]) + 1 + pad)
    x0 = max(0, int(x_idx[0]) - pad)
    x1 = min(w, int(x_idx[-1]) + 1 + pad)
    return (y0, y1, x0, x1)


@dataclass
class MaskLayer:
    """A single masked local-adjustment layer at a fixed working resolution."""

    alpha: np.ndarray  # float32, (H, W), range [0, 1]
    adjustments: dict = field(default_factory=dict)
    name: str = ""
    enabled: bool = True
    invert: bool = False
    blend: str = "add"  # "add" | "subtract" -- reserved for Phase 2 stack composability
    version: int = field(default=0, compare=False, repr=False)
    _empty_cache: Optional[tuple] = field(default=None, compare=False, repr=False)
    _bbox_cache: Optional[tuple] = field(default=None, compare=False, repr=False)

    @classmethod
    def empty(cls, height: int, width: int, **kwargs) -> "MaskLayer":
        return cls(np.zeros((height, width), dtype=np.float32), **kwargs)

    def touch(self) -> None:
        """Call after any in-place mutation of ``alpha`` to invalidate caches."""
        self.version += 1
        self._empty_cache = None
        self._bbox_cache = None

    def effective_alpha(self) -> np.ndarray:
        return (1.0 - self.alpha) if self.invert else self.alpha

    @property
    def is_empty(self) -> bool:
        if not self.enabled:
            return True
        cached = self._empty_cache
        if cached is not None and cached[0] == self.version:
            return cached[1]
        result = not bool(np.any(self.alpha > 1e-4))
        self._empty_cache = (self.version, result)
        return result

    def bbox(self) -> Optional[tuple]:
        """Non-zero alpha bbox in this layer's own (alpha-resolution) coordinates.

        Computed on ``alpha`` regardless of ``invert`` -- an inverted layer's
        *effective* coverage is everywhere alpha is low, but its editable
        (non-trivial) region is still where the painted alpha itself is
        non-zero; parametric masks (gradient/radial, Phase 2) will instead
        derive a bbox analytically from their params.
        """
        cached = self._bbox_cache
        if cached is not None and cached[0] == self.version:
            return cached[1]
        result = _nonzero_bbox(self.alpha)
        self._bbox_cache = (self.version, result)
        return result

    def fingerprint(self) -> str:
        """Cheap stage-cache key: shape + mutation version + adjustment values."""
        h, w = self.alpha.shape[:2]
        adj_sig = ",".join(
            f"{k}={self.adjustments.get(k, 0.0):.4f}" for k in _all_fingerprint_keys()
        )
        enabled_sig = int(self.enabled)
        return (
            f"mem:{int(h)}x{int(w)}:v{int(self.version)}:inv{int(self.invert)}"
            f":en{enabled_sig}:bl{self.blend}:{adj_sig}"
        )


@dataclass
class MaskLayerStack:
    """Ordered list of MaskLayer, composited back-to-front."""

    layers: list = field(default_factory=list)
    _composite_cache: Optional[tuple] = field(default=None, compare=False, repr=False)
    _prefix_cache: Optional[tuple] = field(default=None, compare=False, repr=False)

    def is_empty(self) -> bool:
        return not any(not layer.is_empty for layer in self.layers)

    def fingerprint(self) -> str:
        if not self.layers:
            return "empty"
        return "|".join(layer.fingerprint() for layer in self.layers)


def resolve_stack_from_adj(adj: Optional[dict]) -> Optional[MaskLayerStack]:
    """Prefer the live stack object; fall back to the XMP serial form."""
    if not adj:
        return None
    obj = adj.get(MASK_LAYERS_OBJ_KEY)
    if isinstance(obj, MaskLayerStack):
        return obj
    serial = adj.get(MASK_LAYERS_KEY)
    if not serial:
        return None
    from mask_layers_xmp import deserialize_stack_cached

    return deserialize_stack_cached(serial)


def resize_alpha_to(alpha: np.ndarray, height: int, width: int) -> np.ndarray:
    if alpha.shape == (height, width):
        return alpha
    import cv2

    return cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR)


def _scaled_bbox(bbox: tuple, src_h: int, src_w: int, dst_h: int, dst_w: int) -> tuple:
    """Map a bbox from the mask's own resolution to a differently-sized target."""
    if (src_h, src_w) == (dst_h, dst_w):
        return bbox
    y0, y1, x0, x1 = bbox
    sy = dst_h / float(src_h)
    sx = dst_w / float(src_w)
    ty0 = max(0, int(np.floor(y0 * sy)))
    ty1 = min(dst_h, int(np.ceil(y1 * sy)))
    tx0 = max(0, int(np.floor(x0 * sx)))
    tx1 = min(dst_w, int(np.ceil(x1 * sx)))
    return (ty0, ty1, tx0, tx1)


def _composite_one_layer(img: np.ndarray, layer: "MaskLayer") -> np.ndarray:
    """Blend one layer's adjustment into ``img``, limited to its alpha bbox.

    Spatially-aware adjustments (Dehaze/Sharpness/Clarity/Defringe) render
    over a padded extraction region so their neighborhood ops see clean
    context, then only the tight bbox is cropped back out of that padded
    result before blending -- the pad-before-filter-crop-after pattern.
    """
    bbox = layer.bbox()
    if bbox is None:
        return img
    mh, mw = layer.alpha.shape[:2]
    h, w = img.shape[:2]
    y0, y1, x0, x1 = _scaled_bbox(bbox, mh, mw, h, w)
    if y1 <= y0 or x1 <= x0:
        return img

    pad = _SPATIAL_FILTER_PAD if _needs_spatial_pad(layer.adjustments) else 0
    ey0, ey1 = max(0, y0 - pad), min(h, y1 + pad)
    ex0, ex1 = max(0, x0 - pad), min(w, x1 + pad)

    extraction = img[ey0:ey1, ex0:ex1]
    adjusted_padded = _apply_layer_adjustments(extraction, layer.adjustments)
    ry0, ry1, rx0, rx1 = y0 - ey0, y1 - ey0, x0 - ex0, x1 - ex0
    adjusted_region = adjusted_padded[ry0:ry1, rx0:rx1]

    alpha_full = resize_alpha_to(layer.effective_alpha(), h, w)
    alpha_region = alpha_full[y0:y1, x0:x1]
    tight_region = img[y0:y1, x0:x1]
    blended = tight_region * (1.0 - alpha_region[..., np.newaxis]) + adjusted_region * alpha_region[..., np.newaxis]

    if (y0, y1, x0, x1) == (0, h, 0, w):
        return blended
    out = img.copy()
    out[y0:y1, x0:x1] = blended
    return out


def stamp_mask_layer_brush(
    layer: MaskLayer,
    cx: float,
    cy: float,
    radius: float,
    strength: float,
    *,
    luminance: Optional[np.ndarray] = None,
    chroma: Optional[np.ndarray] = None,
    edge_assist: bool = True,
    luma_tol: float = 0.10,
) -> tuple[int, int, int, int]:
    """Accumulate soft brush coverage into ``layer.alpha`` (max-blend, 0..1).

    Max-blend (not additive) matches raw_spot_heal.stamp_heal_brush's
    coverage semantics -- a mask layer's alpha means "how much of this
    pixel belongs to this region", not a signed accumulating strength like
    dodge/burn's exposure delta, so repeated overlapping stamps saturate
    toward full coverage rather than overshooting past it.

    Edge-assist (reused from raw_dodge_burn, same flood-fill-within-
    tolerance gate that keeps dodge/burn strokes from bleeding across a
    subject boundary) is optional since a mask's brush may deliberately
    want to paint across an edge (e.g. hand-erasing part of an AI mask).
    """
    from raw_dodge_burn import circular_brush_falloff

    h, w = layer.alpha.shape
    r = max(1.0, float(radius))
    x0 = max(0, int(cx - r - 1))
    x1 = min(w, int(cx + r + 2))
    y0 = max(0, int(cy - r - 1))
    y1 = min(h, int(cy + r + 2))
    if x1 <= x0 or y1 <= y0:
        return (x0, y0, x1, y1)

    falloff = circular_brush_falloff(y0, y1, x0, x1, cx, cy, r)

    if edge_assist and luminance is not None and luminance.shape[:2] == (h, w):
        from raw_dodge_burn import _edge_assist_gate

        falloff = falloff * _edge_assist_gate(
            luminance, y0, x0, y1, x1, cx, cy, luma_tol=luma_tol, chroma=chroma
        )

    amount = np.clip(falloff * float(strength), 0.0, 1.0)
    region = layer.alpha[y0:y1, x0:x1]
    np.maximum(region, amount, out=region)
    np.clip(region, 0.0, 1.0, out=region)
    layer.touch()
    return (x0, y0, x1, y1)


def erase_mask_layer_brush(
    layer: MaskLayer,
    cx: float,
    cy: float,
    radius: float,
    strength: float,
    *,
    luminance: Optional[np.ndarray] = None,
    chroma: Optional[np.ndarray] = None,
    edge_assist: bool = True,
    luma_tol: float = 0.10,
) -> tuple[int, int, int, int]:
    """Pull ``layer.alpha`` toward zero under a soft circular brush."""
    from raw_dodge_burn import circular_brush_falloff

    h, w = layer.alpha.shape
    r = max(1.0, float(radius))
    x0 = max(0, int(cx - r - 1))
    x1 = min(w, int(cx + r + 2))
    y0 = max(0, int(cy - r - 1))
    y1 = min(h, int(cy + r + 2))
    if x1 <= x0 or y1 <= y0:
        return (x0, y0, x1, y1)

    falloff = circular_brush_falloff(y0, y1, x0, x1, cx, cy, r)

    if edge_assist and luminance is not None and luminance.shape[:2] == (h, w):
        from raw_dodge_burn import _edge_assist_gate

        falloff = falloff * _edge_assist_gate(
            luminance, y0, x0, y1, x1, cx, cy, luma_tol=luma_tol, chroma=chroma
        )

    region = layer.alpha[y0:y1, x0:x1]
    region *= np.clip(1.0 - falloff * float(strength), 0.0, 1.0)
    layer.touch()
    return (x0, y0, x1, y1)


def _composite_layers(img: np.ndarray, layers: list) -> np.ndarray:
    out = img
    for layer in layers:
        if layer.is_empty:
            continue
        out = _composite_one_layer(out, layer)
    return out


def apply_mask_layers(
    img: np.ndarray, stack: Optional[MaskLayerStack], *, active_index: Optional[int] = None
) -> np.ndarray:
    """Composite every enabled, non-empty layer's adjustment onto ``img``.

    No-op when the stack is None/empty -- callers can call this
    unconditionally, matching apply_dodge_burn's contract.

    ``active_index=None`` (settle / export / any tick not actively editing
    a mask): the whole composite is cached on the stack instance, keyed by
    (stack fingerprint, target shape, id(base image)) -- see the module
    docstring for why per-layer caching doesn't work here. An unrelated
    LATER slider tick (mask layers and everything upstream of them
    unchanged, so ``img`` is the same object as last call) is then a
    single cache lookup.

    ``active_index=i`` (a live brush stroke or slider drag on layer i):
    layers before ``i`` are served from a separate prefix cache (same
    id(img)-keyed shape), so the drag doesn't repay their cost every tick;
    only layer ``i`` and any layers stacked after it (the rare case)
    recompute. This result is intentionally NOT written to the full
    composite cache -- it's a mid-drag approximation the caller should
    replace with an active_index=None settle call once the drag ends.
    """
    if stack is None or stack.is_empty():
        return img
    h, w = img.shape[:2]

    if active_index is None:
        cache_key = (stack.fingerprint(), h, w, id(img))
        cached = stack._composite_cache
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        out = _composite_layers(img, stack.layers)
        stack._composite_cache = (cache_key, out)
        return out

    n = len(stack.layers)
    if n == 0:
        return img
    active_index = max(0, min(active_index, n - 1))
    prefix_layers = stack.layers[:active_index]
    prefix_fp = "|".join(l.fingerprint() for l in prefix_layers) if prefix_layers else "empty"
    prefix_key = (prefix_fp, h, w, id(img))
    cached_prefix = stack._prefix_cache
    if cached_prefix is not None and cached_prefix[0] == prefix_key:
        prefix_out = cached_prefix[1]
    else:
        prefix_out = _composite_layers(img, prefix_layers)
        stack._prefix_cache = (prefix_key, prefix_out)

    return _composite_layers(prefix_out, stack.layers[active_index:])
