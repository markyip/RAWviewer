"""Parametric mask shapes: linear and radial gradients.

A brush mask is inherently a pixel buffer -- there is nothing to describe it
with but its pixels. A gradient is the opposite: two points and a feather
describe it completely, and keeping it that way is what makes it re-draggable
after a reload, resolution-independent, and a few dozen bytes in the sidecar
instead of a PNG the size of the frame.

So ``MaskLayer`` carries ``kind`` + ``params`` for these, and the alpha is
generated on demand at whatever resolution the caller needs (see
``generate_alpha``). Params are stored NORMALISED to the frame (0..1 across
width/height) rather than in pixels, so the same mask lands identically on the
half-res preview base and the full-res export base -- the alternative is a
mask that shifts when you export, which is the classic bug in this area.

Geometry, matching how Lightroom/darktable present these:

  linear  p0 -> p1 defines the axis. Alpha is 0 behind p0, 1 beyond p1, and
          ramps between. The perpendicular extent is infinite -- a linear
          gradient covers the whole frame, graded along one direction.
  radial  centre + radii + rotation. Alpha is 1 inside the inner ellipse and
          falls to 0 at the outer edge, feather controlling how much of the
          radius that fall takes.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

KIND_BRUSH = "brush"
KIND_LINEAR = "linear"
KIND_RADIAL = "radial"

PARAMETRIC_KINDS = (KIND_LINEAR, KIND_RADIAL)

# Minimum axis length / radius, normalised. A zero-length drag would otherwise
# divide by zero; this makes a stray click produce a small mask rather than a
# NaN buffer.
_MIN_EXTENT = 1e-3


def default_params(kind: str, *, aspect: float = 1.0) -> dict:
    """Sensible starting geometry, used when a mask is created without a drag.

    ``aspect`` is width/height, so a radial default comes out visually round
    rather than stretched on a non-square frame.
    """
    if kind == KIND_LINEAR:
        # Top-to-bottom, the overwhelmingly common case (skies).
        return {"x0": 0.5, "y0": 0.0, "x1": 0.5, "y1": 0.45, "feather": 1.0}
    if kind == KIND_RADIAL:
        ry = 0.30
        rx = ry / float(aspect) if aspect > 1e-6 else ry
        return {
            "cx": 0.5,
            "cy": 0.5,
            "rx": min(0.45, rx),
            "ry": ry,
            "rotation": 0.0,
            "feather": 0.5,
        }
    return {}


def params_from_drag(
    kind: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    feather: Optional[float] = None,
) -> dict:
    """Geometry from one drag, in normalised coordinates.

    Linear: the drag IS the axis, start to end -- drag down for a mask that
    darkens the sky, exactly like Lightroom's graduated filter.

    Radial: the drag is a corner-to-corner box, so the ellipse is inscribed in
    what the user swept out. Dragging from the centre outwards would double the
    apparent size of every gesture, which reads as overshoot.
    """
    if kind == KIND_LINEAR:
        p = {"x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1)}
        p["feather"] = 1.0 if feather is None else float(feather)
        return p
    if kind == KIND_RADIAL:
        cx, cy = (float(x0) + float(x1)) * 0.5, (float(y0) + float(y1)) * 0.5
        return {
            "cx": cx,
            "cy": cy,
            "rx": max(_MIN_EXTENT, abs(float(x1) - float(x0)) * 0.5),
            "ry": max(_MIN_EXTENT, abs(float(y1) - float(y0)) * 0.5),
            "rotation": 0.0,
            "feather": 0.5 if feather is None else float(feather),
        }
    return {}


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Cubic ease. A linear ramp leaves a visible mach band where it meets the
    flat region; this is what every gradient tool uses instead."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _linear_alpha(params: dict, height: int, width: int) -> np.ndarray:
    x0 = float(params.get("x0", 0.5)) * (width - 1)
    y0 = float(params.get("y0", 0.0)) * (height - 1)
    x1 = float(params.get("x1", 0.5)) * (width - 1)
    y1 = float(params.get("y1", 0.45)) * (height - 1)

    dx, dy = x1 - x0, y1 - y0
    length_sq = dx * dx + dy * dy
    if length_sq < (_MIN_EXTENT * max(width, height)) ** 2:
        # Degenerate axis (a click, not a drag): everything past the point.
        return np.ones((height, width), dtype=np.float32)

    ys, xs = np.indices((height, width), dtype=np.float32)
    # Projection of each pixel onto the axis, 0 at p0 and 1 at p1.
    t = ((xs - x0) * dx + (ys - y0) * dy) / length_sq

    feather = max(0.0, min(1.0, float(params.get("feather", 1.0))))
    if feather < 1e-3:
        # Hard edge at the midpoint of the axis.
        return (t >= 0.5).astype(np.float32)
    # feather < 1 tightens the ramp around the axis midpoint, so the same drag
    # can mean "graded across this span" or "edge here, softened a little".
    mid = 0.5
    half = 0.5 * feather
    return _smoothstep((t - (mid - half)) / (2.0 * half)).astype(np.float32)


def _radial_alpha(params: dict, height: int, width: int) -> np.ndarray:
    cx = float(params.get("cx", 0.5)) * (width - 1)
    cy = float(params.get("cy", 0.5)) * (height - 1)
    rx = max(_MIN_EXTENT, float(params.get("rx", 0.3))) * (width - 1)
    ry = max(_MIN_EXTENT, float(params.get("ry", 0.3))) * (height - 1)
    rotation = math.radians(float(params.get("rotation", 0.0)))

    ys, xs = np.indices((height, width), dtype=np.float32)
    dx = xs - cx
    dy = ys - cy
    if abs(rotation) > 1e-6:
        cos_r, sin_r = math.cos(-rotation), math.sin(-rotation)
        dx, dy = dx * cos_r - dy * sin_r, dx * sin_r + dy * cos_r

    # Elliptical radius: 1.0 exactly on the ellipse.
    r = np.sqrt((dx / rx) ** 2 + (dy / ry) ** 2)

    feather = max(0.0, min(1.0, float(params.get("feather", 0.5))))
    if feather < 1e-3:
        return (r <= 1.0).astype(np.float32)
    inner = 1.0 - feather
    return _smoothstep((1.0 - r) / max(1e-6, 1.0 - inner)).astype(np.float32)


def generate_alpha(kind: str, params: dict, height: int, width: int) -> np.ndarray:
    """(H, W) float32 alpha in [0, 1] for a parametric mask kind."""
    if height < 1 or width < 1:
        return np.zeros((max(1, height), max(1, width)), dtype=np.float32)
    if kind == KIND_LINEAR:
        return _linear_alpha(params or {}, height, width)
    if kind == KIND_RADIAL:
        return _radial_alpha(params or {}, height, width)
    return np.zeros((height, width), dtype=np.float32)


def alpha_bbox(kind: str, params: dict, height: int, width: int) -> tuple:
    """Where a parametric mask can be non-zero, without generating it.

    Analytic so the compositor's bbox-limited path (which is what keeps a mask
    tick inside the preview budget) does not have to build a full-frame alpha
    just to find its extent.
    """
    if kind == KIND_LINEAR:
        # A linear gradient grades across the whole frame; there is no
        # perpendicular limit to exploit.
        return (0, height, 0, width)
    if kind == KIND_RADIAL:
        p = params or {}
        cx = float(p.get("cx", 0.5)) * (width - 1)
        cy = float(p.get("cy", 0.5)) * (height - 1)
        rx = max(_MIN_EXTENT, float(p.get("rx", 0.3))) * (width - 1)
        ry = max(_MIN_EXTENT, float(p.get("ry", 0.3))) * (height - 1)
        rot = abs(math.radians(float(p.get("rotation", 0.0))))
        # Rotated ellipse's axis-aligned half-extents.
        ex = math.hypot(rx * math.cos(rot), ry * math.sin(rot))
        ey = math.hypot(rx * math.sin(rot), ry * math.cos(rot))
        y0 = max(0, int(math.floor(cy - ey)) - 1)
        y1 = min(height, int(math.ceil(cy + ey)) + 2)
        x0 = max(0, int(math.floor(cx - ex)) - 1)
        x1 = min(width, int(math.ceil(cx + ex)) + 2)
        if y1 <= y0 or x1 <= x0:
            return (0, height, 0, width)
        return (y0, y1, x0, x1)
    return (0, height, 0, width)
