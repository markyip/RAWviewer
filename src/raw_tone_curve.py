"""PV2012 parametric tone curve + Lightroom ToneCurvePV2012 point LUT."""

from __future__ import annotations

import numpy as np

TONE_CURVE_SERIAL_KEY = "_tone_curve_pv2012"

_SPLIT_SHADOWS = 0.25
_SPLIT_DARKS = 0.50
_SPLIT_LIGHTS = 0.75
_LUT_SIZE = 65536


def serialize_tone_curve_points(points: list[tuple[float, float]]) -> str:
    if len(points) < 2:
        return ""
    if is_identity_tone_curve_points(points):
        return ""
    return ";".join(f"{int(round(x))},{int(round(y))}" for x, y in points)


def default_tone_curve_points() -> list[tuple[float, float]]:
    return [(0.0, 0.0), (255.0, 255.0)]


def is_identity_tone_curve_points(points: list[tuple[float, float]]) -> bool:
    """Linear identity: only endpoints at (0,0) and (255,255)."""
    if not points:
        return True
    if len(points) == 2:
        x0, y0 = points[0]
        x1, y1 = points[1]
        return (
            abs(x0) < 1e-3
            and abs(y0) < 1e-3
            and abs(x1 - 255.0) < 1e-3
            and abs(y1 - 255.0) < 1e-3
        )
    return False


def normalize_tone_curve_points(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Sort by input, clamp to 0–255, enforce monotonic x with endpoints."""
    if not points:
        return default_tone_curve_points()
    cleaned: list[tuple[float, float]] = []
    for x, y in points:
        cleaned.append(
            (float(max(0.0, min(255.0, x))), float(max(0.0, min(255.0, y))))
        )
    cleaned.sort(key=lambda p: p[0])
    if cleaned[0][0] > 0.0:
        cleaned.insert(0, (0.0, cleaned[0][1]))
    else:
        cleaned[0] = (0.0, cleaned[0][1])
    if cleaned[-1][0] < 255.0:
        cleaned.append((255.0, cleaned[-1][1]))
    else:
        cleaned[-1] = (255.0, cleaned[-1][1])
    out: list[tuple[float, float]] = [cleaned[0]]
    for x, y in cleaned[1:]:
        px, _ = out[-1]
        if x <= px + 0.5:
            continue
        out.append((x, y))
    if len(out) < 2:
        return default_tone_curve_points()
    out[0] = (0.0, out[0][1])
    out[-1] = (255.0, out[-1][1])
    return out


def insert_tone_curve_point(
    points: list[tuple[float, float]], x: float, y: float, *, max_points: int = 14
) -> list[tuple[float, float]]:
    pts = normalize_tone_curve_points(list(points))
    if len(pts) >= max_points:
        return pts
    x = float(max(1.0, min(254.0, x)))
    y = float(max(0.0, min(255.0, y)))
    for px, _ in pts:
        if abs(px - x) < 3.0:
            return pts
    pts.append((x, y))
    return normalize_tone_curve_points(pts)


def remove_tone_curve_point(points: list[tuple[float, float]], index: int) -> list[tuple[float, float]]:
    pts = normalize_tone_curve_points(list(points))
    if index <= 0 or index >= len(pts) - 1:
        return pts
    del pts[index]
    return normalize_tone_curve_points(pts)


def move_tone_curve_point(
    points: list[tuple[float, float]], index: int, x: float, y: float
) -> list[tuple[float, float]]:
    pts = normalize_tone_curve_points(list(points))
    if index < 0 or index >= len(pts):
        return pts
    y = float(max(0.0, min(255.0, y)))
    if index == 0:
        pts[0] = (0.0, y)
    elif index == len(pts) - 1:
        pts[-1] = (255.0, y)
    else:
        x_min = pts[index - 1][0] + 1.0
        x_max = pts[index + 1][0] - 1.0
        x = float(max(x_min, min(x_max, x)))
        pts[index] = (x, y)
    return normalize_tone_curve_points(pts)


def deserialize_tone_curve_points(serialized: str) -> list[tuple[float, float]]:
    if not serialized or not str(serialized).strip():
        return []
    out: list[tuple[float, float]] = []
    for part in str(serialized).split(";"):
        part = part.strip()
        if not part:
            continue
        xy = part.split(",")
        if len(xy) < 2:
            continue
        try:
            out.append((float(xy[0]), float(xy[1])))
        except ValueError:
            continue
    out.sort(key=lambda p: p[0])
    return out


def _lookup_lut(y: np.ndarray, lut: np.ndarray) -> np.ndarray:
    yi = np.clip(y.astype(np.float32), 0.0, 1.0) * (_LUT_SIZE - 1)
    lo = np.floor(yi).astype(np.int32)
    hi = np.minimum(lo + 1, _LUT_SIZE - 1)
    frac = yi - lo
    return lut[lo] * (1.0 - frac) + lut[hi] * frac


def build_point_curve_lut(points: list[tuple[float, float]]) -> np.ndarray | None:
    if len(points) < 2:
        return None
    xs = np.array([max(0.0, min(255.0, p[0])) / 255.0 for p in points], dtype=np.float32)
    ys = np.array([max(0.0, min(255.0, p[1])) / 255.0 for p in points], dtype=np.float32)
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    xs = np.clip(xs, 0.0, 1.0)
    ys = np.clip(ys, 0.0, 1.0)
    if xs[0] > 0.0:
        xs = np.concatenate(([0.0], xs))
        ys = np.concatenate(([ys[0]], ys))
    if xs[-1] < 1.0:
        xs = np.concatenate((xs, [1.0]))
        ys = np.concatenate((ys, [ys[-1]]))
    grid = np.linspace(0.0, 1.0, _LUT_SIZE, dtype=np.float32)
    return np.interp(grid, xs, ys).astype(np.float32)


def _smooth_weight(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _region_masks(y: np.ndarray) -> dict[str, np.ndarray]:
    shadows = _smooth_weight(np.clip((_SPLIT_SHADOWS - y) / max(_SPLIT_SHADOWS, 1e-6), 0.0, 1.0))
    darks_lo = _smooth_weight(np.clip((y - _SPLIT_SHADOWS) / max(_SPLIT_DARKS - _SPLIT_SHADOWS, 1e-6), 0.0, 1.0))
    darks_hi = _smooth_weight(np.clip((_SPLIT_DARKS - y) / max(_SPLIT_DARKS - _SPLIT_SHADOWS, 1e-6), 0.0, 1.0))
    darks = darks_lo * darks_hi
    lights_lo = _smooth_weight(np.clip((y - _SPLIT_DARKS) / max(_SPLIT_LIGHTS - _SPLIT_DARKS, 1e-6), 0.0, 1.0))
    lights_hi = _smooth_weight(np.clip((_SPLIT_LIGHTS - y) / max(_SPLIT_LIGHTS - _SPLIT_DARKS, 1e-6), 0.0, 1.0))
    lights = lights_lo * lights_hi
    highlights = _smooth_weight(np.clip((y - _SPLIT_LIGHTS) / max(1.0 - _SPLIT_LIGHTS, 1e-6), 0.0, 1.0))
    return {
        "shadows": shadows,
        "darks": darks,
        "lights": lights,
        "highlights": highlights,
    }


def apply_parametric_tone_curve(y: np.ndarray, adj: dict[str, float]) -> np.ndarray:
    """Lightroom parametric tone regions on perceptual luminance."""
    ps = float(adj.get("ParametricShadows", 0.0))
    pd = float(adj.get("ParametricDarks", 0.0))
    pl = float(adj.get("ParametricLights", 0.0))
    ph = float(adj.get("ParametricHighlights", 0.0))
    if all(abs(v) < 1e-4 for v in (ps, pd, pl, ph)):
        return y
    out = y.astype(np.float32).copy()
    masks = _region_masks(out)
    for key, val in (
        ("shadows", ps),
        ("darks", pd),
        ("lights", pl),
        ("highlights", ph),
    ):
        if abs(val) < 1e-4:
            continue
        w = masks[key]
        amt = val / 100.0
        if amt > 0:
            out = out + w * amt * (1.0 - out) * 0.35
        else:
            out = out + w * amt * out * 0.35
    return np.clip(out, 0.0, 1.0)


def apply_tone_curve_perceptual(y: np.ndarray, adj: dict[str, float]) -> np.ndarray:
    """Point curve (from LR XMP) then parametric regions."""
    out = y.astype(np.float32)
    serial = str(adj.get(TONE_CURVE_SERIAL_KEY, "") or "")
    points = deserialize_tone_curve_points(serial)
    lut = build_point_curve_lut(points)
    if lut is not None:
        out = _lookup_lut(out, lut)
    out = apply_parametric_tone_curve(out, adj)
    return out
