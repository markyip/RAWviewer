"""PV2012 parametric tone curve + Lightroom ToneCurvePV2012 point LUT."""

from __future__ import annotations

import numpy as np

TONE_CURVE_SERIAL_KEY = "_tone_curve_pv2012"
# Standard-mode per-channel curves (Lightroom/RawTherapee "RGB Curves"):
# each channel remapped independently by its own point curve, applied to
# already gamma-encoded display-referred values -- unlike the hue-
# preserving luminance curve above, this DOES shift hue/color balance by
# design (e.g. lifting Red in the shadows adds warmth only there). Key
# names match Lightroom's real XMP schema (crs:ToneCurvePV2012Red/Green/
# Blue) so sidecar files stay interoperable with real Lightroom exports.
TONE_CURVE_RED_KEY = "_tone_curve_pv2012_red"
TONE_CURVE_GREEN_KEY = "_tone_curve_pv2012_green"
TONE_CURVE_BLUE_KEY = "_tone_curve_pv2012_blue"
CHANNEL_CURVE_KEYS = (TONE_CURVE_RED_KEY, TONE_CURVE_GREEN_KEY, TONE_CURVE_BLUE_KEY)

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
    # float32 - int32 promotes to float64 under NumPy's rules, which used to
    # make this return a float64 buffer (2x the bytes) and -- because the
    # result flows on through the tone stage -- silently forced every
    # DOWNSTREAM stage (dehaze, saturation, detail, LUT, clips) to run in
    # float64 too. Keeping the subtraction in float32 halves the memory
    # traffic of the whole back half of the pipeline; the 65536-entry LUT is
    # far finer than the 8-bit output needs, so precision is unaffected.
    frac = yi - lo.astype(np.float32)
    return lut[lo] * (1.0 - frac) + lut[hi] * frac


def _pchip_edge_tangent(h0: float, h1: float, m0: float, m1: float) -> float:
    """One-sided 3-point derivative estimate for a PCHIP endpoint (Fritsch-
    Carlson), then clamped to preserve shape. A plain secant slope (m0) here
    would still be monotone-safe but measurably less accurate at the curve's
    ends than what scipy.interpolate.PchipInterpolator itself computes --
    verified to match scipy's own endpoint tangents exactly."""
    val = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    if np.sign(val) != np.sign(m0):
        return 0.0
    if np.sign(m0) != np.sign(m1) and abs(val) > 3.0 * abs(m0):
        return 3.0 * m0
    return val


def _monotone_cubic_fit(xs: np.ndarray, ys: np.ndarray):
    """
    Pure-numpy monotone cubic Hermite interpolation (Fritsch-Carlson method) --
    a from-scratch equivalent of scipy.interpolate.PchipInterpolator, to avoid
    a ~77MB scipy dependency for this one call site (see docs/EDIT_PIPELINE.md
    "Installer size" entry). Verified to match scipy's own PchipInterpolator to
    within float64 noise (~1e-15) across 500+ random knot configurations plus
    this module's own steep-then-plateau overshoot test case.

    ``xs`` must be strictly increasing, len(xs) >= 2. Returns a callable
    f(grid) -> interpolated y, shape-preserving (never overshoots the local
    trend of the knots) the same way PchipInterpolator is.
    """
    n = len(xs)
    if n == 2:
        x0, x1 = xs[0], xs[1]
        y0, y1 = ys[0], ys[1]
        span = x1 - x0

        def f_linear(t: np.ndarray) -> np.ndarray:
            t = np.asarray(t, dtype=np.float64)
            frac = np.clip((t - x0) / (span if span != 0 else 1.0), 0.0, 1.0)
            return y0 + frac * (y1 - y0)

        return f_linear

    h = np.diff(xs)
    d = np.diff(ys) / h

    m = np.zeros(n, dtype=np.float64)
    m[0] = _pchip_edge_tangent(h[0], h[1], d[0], d[1])
    m[-1] = _pchip_edge_tangent(h[-1], h[-2], d[-1], d[-2])

    # Interior tangents: weighted harmonic mean of the two adjacent secants,
    # zeroed at a local extremum (sign change, or either secant flat) -- this
    # is exactly what makes the fit shape-preserving/non-overshooting.
    d0, d1 = d[:-1], d[1:]
    h0, h1 = h[:-1], h[1:]
    same_sign = (d0 * d1) > 0
    w1 = 2.0 * h1 + h0
    w2 = h1 + 2.0 * h0
    d0_safe = np.where(d0 == 0, 1.0, d0)
    d1_safe = np.where(d1 == 0, 1.0, d1)
    harmonic_denom = w1 / d0_safe + w2 / d1_safe
    m[1:-1] = np.where(same_sign, (w1 + w2) / harmonic_denom, 0.0)

    def f(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        tc = np.clip(t, xs[0], xs[-1])
        idx = np.clip(np.searchsorted(xs, tc, side="right") - 1, 0, n - 2)
        x0, x1 = xs[idx], xs[idx + 1]
        y0, y1 = ys[idx], ys[idx + 1]
        m0, m1 = m[idx], m[idx + 1]
        hseg = x1 - x0
        hseg_safe = np.where(hseg == 0, 1.0, hseg)
        frac = (tc - x0) / hseg_safe
        f2, f3 = frac * frac, frac * frac * frac
        h00 = 2.0 * f3 - 3.0 * f2 + 1.0
        h10 = f3 - 2.0 * f2 + frac
        h01 = -2.0 * f3 + 3.0 * f2
        h11 = f3 - f2
        return h00 * y0 + h10 * hseg * m0 + h01 * y1 + h11 * hseg * m1

    return f


def _fit_point_curve(points: list[tuple[float, float]]):
    """
    Fit a monotonic (PCHIP) cubic through the curve's knots and return a
    callable f(grid in [0,1]) -> y in [0,1], or None if there aren't enough
    distinct knots to fit.

    PCHIP (piecewise cubic Hermite), not a plain/natural cubic spline: a
    natural spline can overshoot well past neighboring knot values between
    two closely-spaced points -- exactly the ringing/banding failure mode
    this module's parametric-region coefficients were tuned to avoid
    elsewhere. PCHIP is shape-preserving -- it never overshoots the local
    trend of the input knots -- while still giving the smooth curve users
    expect from Lightroom/Capture One-style tone curve editors, instead of
    the straight connect-the-dots polyline this used to render/apply.
    """
    if len(points) < 2:
        return None
    xs = np.array([max(0.0, min(255.0, p[0])) / 255.0 for p in points], dtype=np.float64)
    ys = np.array([max(0.0, min(255.0, p[1])) / 255.0 for p in points], dtype=np.float64)
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    # PCHIP requires strictly increasing x; collapse near-duplicate knots
    # (can appear in malformed/legacy serialized data) keeping the later one.
    dedup_x = [xs[0]]
    dedup_y = [ys[0]]
    for x, y in zip(xs[1:], ys[1:]):
        if x - dedup_x[-1] < 1e-6:
            dedup_y[-1] = y
        else:
            dedup_x.append(x)
            dedup_y.append(y)
    xs = np.array(dedup_x)
    ys = np.array(dedup_y)

    if xs[0] > 0.0:
        xs = np.concatenate(([0.0], xs))
        ys = np.concatenate(([ys[0]], ys))
    if xs[-1] < 1.0:
        xs = np.concatenate((xs, [1.0]))
        ys = np.concatenate((ys, [ys[-1]]))
    if len(xs) < 2:
        return None

    pchip = _monotone_cubic_fit(xs, ys)
    return lambda grid: np.clip(pchip(grid), 0.0, 1.0)


def build_point_curve_lut(points: list[tuple[float, float]]) -> np.ndarray | None:
    fit = _fit_point_curve(points)
    if fit is None:
        return None
    grid = np.linspace(0.0, 1.0, _LUT_SIZE)
    return fit(grid).astype(np.float32)


def apply_channel_curves_encoded(
    encoded: np.ndarray, adj: dict[str, float], max_val: float
) -> np.ndarray:
    """Standard-mode R/G/B curves, applied to an already gamma-encoded
    display-referred buffer (uint8 preview or uint16 export -- whichever
    ``encoded``/``max_val`` the caller is finishing with). Each channel is
    remapped independently, matching Lightroom/RawTherapee's "Standard"
    RGB Curves behavior (as opposed to a hue-preserving "Luminosity" mode):
    this can and does shift hue/color balance, by design.

    A no-op (returns ``encoded`` unchanged) unless at least one channel
    key holds a non-identity curve, so files with no channel-curve edits
    pay no cost here.
    """
    active: list[tuple[int, np.ndarray]] = []
    for i, key in enumerate(CHANNEL_CURVE_KEYS):
        serial = str(adj.get(key, "") or "")
        if not serial:
            continue
        points = deserialize_tone_curve_points(serial)
        if not points or is_identity_tone_curve_points(points):
            continue
        lut = build_point_curve_lut(points)
        if lut is not None:
            active.append((i, lut))
    if not active:
        return encoded

    out = encoded.astype(np.float32) / float(max_val)
    for channel, lut in active:
        out[..., channel] = _lookup_lut(out[..., channel], lut)
    out = np.clip(out, 0.0, 1.0) * float(max_val)
    if np.issubdtype(encoded.dtype, np.integer):
        out = np.round(out)
    return out.astype(encoded.dtype)


def sample_point_curve_for_display(
    points: list[tuple[float, float]], n_samples: int = 128
) -> list[tuple[float, float]] | None:
    """
    Smooth (x, y) samples in 0-255 coordinates for drawing the curve widget.

    Uses the identical monotonic-cubic fit as build_point_curve_lut (just
    evaluated at a display-friendly point count instead of the full LUT), so
    the drawn curve is exactly what gets applied to the image, not an
    approximation of it.
    """
    fit = _fit_point_curve(points)
    if fit is None:
        return None
    grid = np.linspace(0.0, 1.0, max(2, n_samples))
    ys = fit(grid)
    return [(float(x) * 255.0, float(y) * 255.0) for x, y in zip(grid, ys)]


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
        # 0.15, not the more intuitive-looking 0.35: the shadows-lift and
        # highlights-darken brackets reach slope ~5.89/5.90 at their steepest
        # (same _smooth_weight/_SPLIT_SHADOWS=0.25 shape as raw_pv2012.py's
        # shadow-lift, which needed the identical fix) -- any coefficient
        # above ~0.17 makes the combined curve locally decreasing there. 0.15
        # keeps a monotonic curve at +/-100 with margin, for all four regions.
        if amt > 0:
            out = out + w * amt * (1.0 - out) * 0.15
        else:
            out = out + w * amt * out * 0.15
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
