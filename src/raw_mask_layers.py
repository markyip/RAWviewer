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

from raw_dodge_burn import DEFAULT_BRUSH_FEATHER

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
    # VIEW STATE ONLY: hides this layer's coloured overlay while leaving its
    # adjustment fully applied. Distinct from ``enabled``, which turns the
    # adjustment off.
    #
    # Deliberately outside fingerprint(), serialize_stack() and every
    # is_empty/bbox path: showing or hiding a tint changes nothing about the
    # rendered photo, so folding it into the cache key would discard a correct
    # composite and re-render the frame to produce identical pixels.
    overlay_hidden: bool = False
    blend: str = "add"  # "add" | "subtract" -- reserved for Phase 2 stack composability
    # Parametric shapes (see raw_mask_shapes). "brush" keeps ``alpha`` as the
    # source of truth; a gradient keeps ``params`` as the source of truth and
    # regenerates alpha at whatever resolution is asked for, which is what
    # makes it re-draggable after reload and resolution-independent between
    # the preview and export bases.
    kind: str = "brush"
    params: dict = field(default_factory=dict)
    # What produced this layer: "" (hand-painted), "subject", "sky", "sam".
    # Distinct from ``kind``, which is how the alpha is *stored*. Used to stop
    # a one-shot AI tool being offered again when its mask already exists --
    # matching on the layer NAME would break the moment a user renames a row.
    source: str = ""
    # Grouped masks (Lightroom's model). When non-empty this layer is a GROUP:
    # its coverage is its components combined by each component's ``blend``,
    # not its own ``alpha``. Components are MaskLayer instances too, but only
    # their alpha/kind/params/blend/invert/enabled are consulted -- the
    # adjustments live on the group, exactly one set per mask.
    #
    # That is precisely why dropping one mask onto another discards the
    # dragged mask's adjustments: a component cannot own any.
    #
    # One pass, not N: a group applies its adjustment once through the
    # combined alpha, so grouping three masks costs less than three separate
    # layers rather than more.
    components: list = field(default_factory=list)
    version: int = field(default=0, compare=False, repr=False)
    _empty_cache: Optional[tuple] = field(default=None, compare=False, repr=False)
    _bbox_cache: Optional[tuple] = field(default=None, compare=False, repr=False)
    _shape_alpha_cache: Optional[tuple] = field(default=None, compare=False, repr=False)
    _group_alpha_cache: Optional[tuple] = field(default=None, compare=False, repr=False)

    @classmethod
    def empty(cls, height: int, width: int, **kwargs) -> "MaskLayer":
        return cls(np.zeros((height, width), dtype=np.float32), **kwargs)

    def touch(self) -> None:
        """Call after any in-place mutation of ``alpha`` or ``params``."""
        self.version += 1
        self._empty_cache = None
        self._bbox_cache = None
        self._shape_alpha_cache = None
        self._group_alpha_cache = None

    @property
    def is_parametric(self) -> bool:
        from raw_mask_shapes import PARAMETRIC_KINDS

        return self.kind in PARAMETRIC_KINDS

    @property
    def is_group(self) -> bool:
        return bool(self.components)

    def _combined_alpha_at(self, height: int, width: int) -> np.ndarray:
        """Components combined by their ``blend``, at the given resolution.

        Add is a union (max), not a sum: two overlapping brush strokes should
        cover the overlap once, not twice as hard. Subtract removes coverage
        proportionally, so a soft eraser gradient thins the mask rather than
        punching a hard hole.

        The first component is always added regardless of its blend -- a mask
        whose first component subtracts would otherwise be empty forever, and
        silently so.
        """
        cache = self._group_alpha_cache
        key = (self._components_version(), height, width)
        if cache is not None and cache[0] == key:
            return cache[1]

        out = None
        for comp in self.components:
            if not comp.enabled:
                continue
            a = comp.effective_alpha_at(height, width)
            if out is None:
                out = a.astype(np.float32, copy=True)
                continue
            if comp.blend == "subtract":
                out *= 1.0 - a
            else:
                np.maximum(out, a, out=out)
        if out is None:
            out = np.zeros((height, width), dtype=np.float32)
        np.clip(out, 0.0, 1.0, out=out)
        self._group_alpha_cache = (key, out)
        return out

    def _components_version(self) -> tuple:
        """Version tuple that changes whenever any component does."""
        return tuple(
            (c.version, c.blend, c.enabled, c.invert) for c in self.components
        )

    def alpha_at(self, height: int, width: int) -> np.ndarray:
        """This layer's alpha at a given resolution.

        A brush layer resizes its buffer; a parametric layer regenerates from
        params, which is exact at any size rather than an interpolation of a
        buffer authored at some other one. A group combines its components.
        Cached per (version, shape) because the compositor asks for the same
        size every tick.
        """
        if self.is_group:
            return self._combined_alpha_at(height, width)
        if not self.is_parametric:
            return resize_alpha_to(self.alpha, height, width)
        cached = self._shape_alpha_cache
        if cached is not None and cached[0] == (self.version, height, width):
            return cached[1]
        from raw_mask_shapes import generate_alpha

        generated = generate_alpha(self.kind, self.params, height, width)
        self._shape_alpha_cache = ((self.version, height, width), generated)
        return generated

    def effective_alpha(self) -> np.ndarray:
        return (1.0 - self.alpha) if self.invert else self.alpha

    def effective_alpha_at(self, height: int, width: int) -> np.ndarray:
        """``alpha_at`` with ``invert`` applied -- what the compositor blends."""
        a = self.alpha_at(height, width)
        return (1.0 - a) if self.invert else a

    @property
    def is_empty(self) -> bool:
        if not self.enabled:
            return True
        if self.is_group:
            # Not cached on self.version: a component can be edited without
            # the group's own version moving.
            return all(c.is_empty for c in self.components)
        cached = self._empty_cache
        if cached is not None and cached[0] == self.version:
            return cached[1]
        if self.is_parametric:
            # Its coverage comes from params, not the (unused) alpha buffer --
            # testing alpha would call every gradient empty and skip it.
            result = False
        else:
            result = not bool(np.any(self.alpha > 1e-4))
        self._empty_cache = (self.version, result)
        return result

    def effective_bbox(self) -> Optional[tuple]:
        """Where this layer actually *applies* -- what the compositor needs.

        An inverted layer covers everything OUTSIDE its painted region, so its
        applying region is the whole frame. Compositing an inverted layer over
        ``bbox()`` confined the adjustment to the one area where effective
        alpha is near zero, so Invert appeared to do almost nothing beyond a
        small patch while the mask overlay showed it correctly.
        """
        if self.invert:
            if not self.enabled:
                return None
            h, w = self.frame_shape()
            return (0, h, 0, w)
        return self.bbox()

    def frame_shape(self) -> tuple:
        """(h, w) of this mask's coordinate space.

        A group's own ``alpha`` is unused, so its frame comes from a
        component instead -- reading self.alpha.shape there would give the
        placeholder size and confine an inverted group to a corner.
        """
        if self.is_group:
            for c in self.components:
                h, w = c.frame_shape()
                if h > 1 and w > 1:
                    return (h, w)
        return tuple(self.alpha.shape[:2])

    def bbox(self) -> Optional[tuple]:
        """Non-zero alpha bbox in this layer's own (alpha-resolution) coordinates.

        Computed on ``alpha`` regardless of ``invert``: this is the *editable*
        region -- what a brush has touched -- which is what the UI and the mask
        overlay want. The compositor wants ``effective_bbox()``. Parametric
        masks (gradient/radial) will derive a bbox analytically from params.
        """
        if self.is_group:
            # Union of the ADDING components. A subtracting one can only
            # remove coverage, so it never widens where the group applies.
            boxes = [
                c.bbox()
                for c in self.components
                if c.enabled and c.blend != "subtract" and c.bbox() is not None
            ]
            if not boxes:
                return None
            return (
                min(b[0] for b in boxes), max(b[1] for b in boxes),
                min(b[2] for b in boxes), max(b[3] for b in boxes),
            )
        cached = self._bbox_cache
        if cached is not None and cached[0] == self.version:
            return cached[1]
        if self.is_parametric:
            from raw_mask_shapes import alpha_bbox

            h, w = self.alpha.shape[:2]
            y0, y1, x0, x1 = alpha_bbox(self.kind, self.params, h, w)
            result = (y0, y1, x0, x1)
        else:
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
        shape_sig = ""
        if self.is_parametric:
            shape_sig = ":" + self.kind + ":" + ",".join(
                f"{k}={float(self.params.get(k, 0.0)):.5f}"
                for k in sorted(self.params)
            )
        group_sig = ""
        if self.is_group:
            group_sig = ":grp[" + "|".join(c.fingerprint() for c in self.components) + "]"
        return (
            f"mem:{int(h)}x{int(w)}:v{int(self.version)}:inv{int(self.invert)}"
            f":en{enabled_sig}:bl{self.blend}{shape_sig}{group_sig}:{adj_sig}"
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
    # effective_bbox, not bbox: an inverted layer applies everywhere outside
    # its painted region, so confining it to the painted bbox made Invert a
    # near no-op. See MaskLayer.effective_bbox.
    bbox = layer.effective_bbox()
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

    # effective_alpha_at, not resize(effective_alpha()): a parametric layer
    # generates exactly at the target resolution, so the same gradient lands
    # identically on the half-res preview and the full-res export instead of
    # being interpolated up from whatever size it was authored at.
    alpha_full = layer.effective_alpha_at(h, w)
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
    feather: float = DEFAULT_BRUSH_FEATHER,
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

    falloff = circular_brush_falloff(y0, y1, x0, x1, cx, cy, r, feather)

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
    feather: float = DEFAULT_BRUSH_FEATHER,
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

    falloff = circular_brush_falloff(y0, y1, x0, x1, cx, cy, r, feather)

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

    import weakref as _weakref

    if active_index is None:
        # id(img) identifies the upstream buffer, but numpy reuses the same
        # address for the next same-shaped allocation almost every time, so a
        # different base with an identical fingerprint and shape would hit
        # this entry and render the previous frame. The weakref makes the
        # match exact: a dead or different referent cannot satisfy it.
        cache_key = (stack.fingerprint(), h, w, id(img))
        cached = stack._composite_cache
        if cached is not None and cached[0] == cache_key:
            ref = cached[2] if len(cached) > 2 else None
            if ref is not None and ref() is img:
                return cached[1]
        out = _composite_layers(img, stack.layers)
        try:
            stack._composite_cache = (cache_key, out, _weakref.ref(img))
        except TypeError:
            stack._composite_cache = None
        return out

    n = len(stack.layers)
    if n == 0:
        return img
    active_index = max(0, min(active_index, n - 1))
    prefix_layers = stack.layers[:active_index]
    prefix_fp = "|".join(l.fingerprint() for l in prefix_layers) if prefix_layers else "empty"
    prefix_key = (prefix_fp, h, w, id(img))
    cached_prefix = stack._prefix_cache
    prefix_out = None
    if cached_prefix is not None and cached_prefix[0] == prefix_key:
        ref = cached_prefix[2] if len(cached_prefix) > 2 else None
        if ref is not None and ref() is img:
            prefix_out = cached_prefix[1]
    if prefix_out is None:
        prefix_out = _composite_layers(img, prefix_layers)
        try:
            stack._prefix_cache = (prefix_key, prefix_out, _weakref.ref(img))
        except TypeError:
            stack._prefix_cache = None

    return _composite_layers(prefix_out, stack.layers[active_index:])
