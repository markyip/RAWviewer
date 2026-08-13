"""One-shot automatic adjustments: Auto white balance, tone and straighten.

All three are ESTIMATORS feeding the existing manual controls -- Auto WB
produces a neutral-gray sample for raw_adjustments.solve_white_balance_from_sample
(the same solver the WB dropper uses), Auto tone produces the six PV2012 Light
slider values, Auto straighten produces a CropAngle for the Transform slider.
The user sees exactly which sliders moved and can refine or undo with the normal
controls; nothing here writes pixels or sidecars itself.

That is the whole reason this module exists rather than reusing
raw_tone_recovery: recovery renders display pixels for the P-key preview, so
there are no slider values behind it to record, refine, or write to a sidecar.
An "auto" that produced pixels would be the one destructive step in a
non-destructive pipeline.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def estimate_neutral_rgb(linear_rgb: np.ndarray) -> Optional[tuple]:
    """Robust gray-world neutral estimate from a scene-linear RGB buffer.

    Plain gray-world (mean of everything) is famously wrecked by dominant
    subject colors. Two standard robustifiers:
      - midtone gate: only pixels between the 25th and 90th luminance
        percentile vote (shadows carry mostly noise, highlights clip);
      - saturation gate: only the least-saturated half of those vote --
        near-neutral surfaces are the pixels that SHOULD be gray, strongly
        colored ones say nothing about the illuminant.
    Returns (r, g, b) means of the voting pixels, or None if too few.
    """
    try:
        a = linear_rgb
        if a is None or a.ndim != 3 or a.shape[2] < 3:
            return None
        # Subsample for speed: the estimate is a global mean, ~100k pixels
        # is plenty.
        h, w = a.shape[:2]
        step = max(1, int((h * w / 150_000) ** 0.5))
        s = a[::step, ::step, :3].astype(np.float32)
        lum = 0.2126 * s[..., 0] + 0.7152 * s[..., 1] + 0.0722 * s[..., 2]
        lo, hi = np.percentile(lum, (25.0, 90.0))
        mid = (lum > max(lo, 1e-5)) & (lum < hi)
        if int(mid.sum()) < 500:
            return None
        px = s[mid]
        mx = px.max(axis=1)
        mn = px.min(axis=1)
        sat = (mx - mn) / np.maximum(mx, 1e-6)
        keep = sat <= np.percentile(sat, 50.0)
        px = px[keep]
        if px.shape[0] < 300:
            return None
        r, g, b = (float(px[:, i].mean()) for i in range(3))
        if min(r, g, b) <= 1e-6:
            return None
        return (r, g, b)
    except Exception as e:
        logger.debug("[AUTO_WB] estimate failed: %s", e)
        return None


# --- Auto tone -------------------------------------------------------------
#
# Histogram targets (median, clip/crush fractions, IQR, endpoints) are stated
# in PV2012 PERCEPTUAL space (raw_pv2012._scene_to_perceptual). That matches
# Contrast / Whites / Blacks, which still run on the perceptual ratio path.
# Shadows2012 / Highlights2012 on the live RGB path are scene-linear regional
# gain (apply_regional_linear_exposure: ±1.55 / ±1.75 stop at slider ±100),
# not the dead perceptual HS branch. Clip/crush *fractions* stay perceptual
# (same space as Contrast/Whites/Blacks). The slider magnitude for HS is a
# dedicated cap so CLIP_FULL / CRUSH_FULL map to the scene-linear stop the
# old Auto intended on the perceptual HS path — not the stale ±60 → ±0.93 /
# ±1.05 stop that fell out when HS went linear while the cap stayed 60.
# Measuring clip in scene-linear or display sRGB would put those fractions in
# a different space from Contrast/Whites/Blacks and from _PERCEPTUAL_WHITE.
#
# The single most important constant here is where white sits. In perceptual
# space a clipped white lands at _PERCEPTUAL_WHITE (~0.813), NOT at 1.0 --
# raw_pv2012 documents at length how measuring highlight math against 1.0
# silently attenuates it into a no-op. Every highlight threshold below is a
# fraction of _PERCEPTUAL_WHITE for exactly that reason.
_AUTO_TONE_MEDIAN_TARGET = 0.38  # perceptual; ~0.16 linear, just under mid-grey
_AUTO_TONE_MAX_EV = 2.0
# Unused for live HS (see the dedicated caps below). Kept as the historical
# Auto magnitude from when all six sliders were treated as perceptual.
_AUTO_TONE_MAX_SLIDER = 60.0
#
# Perceptual _apply_highlights_shadows on clipped white (_PERCEPTUAL_WHITE,
# scene-linear 1.0) at Highlights=-60 (old Auto CLIP_FULL) lands at
# perceptual 0.595 → linear 0.359 = −1.479 stop. The linear engine is
# slider/100 * 1.55 stop, so matching that stop would be slider −95.4 —
# effectively pinning Highlights. Cap at 90 (−1.395 stop, 94% of the old
# intent) so CLIP_FULL stays bounded away from −100 and does not out-punch
# Contrast/Whites/Blacks (those still top out at _AUTO_TONE_MAX_ENDPOINT).
# Shadows=+60 on crush y=0.05 was +1.317 stop; 75/100*1.75 = +1.3125.
_AUTO_TONE_HIGHLIGHTS_MAX_SLIDER = 90.0
_AUTO_TONE_SHADOWS_MAX_SLIDER = 75.0

# Dead zones. Below these an image is "already balanced" and AUTO refuses
# rather than twitching six sliders by amounts nobody can see.
_AUTO_TONE_EV_DEADZONE = 0.15
_AUTO_TONE_SLIDER_DEADZONE = 5.0

# A pixel at/above this fraction of perceptual white counts as near-clipped;
# at/below the low one it counts as crushed.
_AUTO_TONE_CLIP_FRAC = 0.98
_AUTO_TONE_CRUSH_Y = 0.05  # perceptual; ~0.02 linear
#
# Every (start, full) pair below was measured, not guessed: the estimator's
# own statistics were run over 20 real Canon CR3 edit bases (see the header
# note on why the space matters) and each ramp starts above what a normal
# frame shows and reaches its cap near the worst frame in that set. Tuning
# these on synthetic histograms produced wildly wrong numbers -- a lognormal
# test image has no true blacks and no specular highlights, so Whites and
# Blacks both pinned near their caps on frames that needed neither.
#
# Near-clipped fraction. Measured median 0.8%: a little clipping is NORMAL
# (specular highlights, a bright sky), so recovery must not engage there.
# The worst frame in the set clipped 13.8%. Edit-base Reconstruct moves
# recovered CFA off the clip floor, so this fraction can drop versus the
# old Blend bases — remaining counts are true leftover clips; that is
# correct, not a reason to lower these thresholds.
_AUTO_TONE_CLIP_START = 0.01
_AUTO_TONE_CLIP_FULL = 0.10

# Crushed fraction. Measured median 8.6%, upper quartile 24.5%, worst 58%.
# Scene-linear RAW is legitimately dark in the shadows -- this is the ramp
# that a fraction-of-frame threshold gets most wrong if it is set from
# intuition, which is why it starts so much higher than the clipping one.
_AUTO_TONE_CRUSH_START = 0.05
_AUTO_TONE_CRUSH_FULL = 0.40

# Unused top-end headroom, as a fraction of perceptual white. Measured median
# deficit 0 (most frames already reach white); the flattest frame in the set
# sat 30% short.
_AUTO_TONE_WHITES_START = 0.08
_AUTO_TONE_WHITES_FULL = 0.30

# Black point. Measured median 0.011, worst 0.090 -- an unhazed RAW already
# has real blacks, so this fires only for haze / flare / a lifted profile.
_AUTO_TONE_BLACKS_START = 0.04
_AUTO_TONE_BLACKS_FULL = 0.12

# Perceptual interquartile spread of a scene with normal contrast (measured
# median 0.31, lower quartile 0.22). Below this the image is flat (haze,
# overcast, a flat picture profile) and gains contrast; the estimator never
# REMOVES contrast, because "this scene is genuinely contrasty" and "this
# scene is badly exposed" look identical from a histogram and only one of
# them wants flattening.
_AUTO_TONE_NORMAL_IQR = 0.20
_AUTO_TONE_MAX_ENDPOINT = 40.0


def _ramp(value: float, start: float, full: float) -> float:
    """0 below ``start``, 1 at/above ``full``, linear between."""
    if full <= start:
        return 1.0 if value >= full else 0.0
    return float(np.clip((value - start) / (full - start), 0.0, 1.0))

_AUTO_TONE_KEYS = (
    "Exposure2012",
    "Contrast2012",
    "Highlights2012",
    "Shadows2012",
    "Whites2012",
    "Blacks2012",
)


def _solve_ev_for_median(median_lum: float, target_y: float) -> float:
    """EV that puts ``median_lum`` at ``target_y`` in perceptual space.

    Bisection rather than an analytic inverse: _scene_to_perceptual is a
    Reinhard-style compression composed with an interpolated base curve, so
    inverting it in closed form means inverting that curve too. It is monotone,
    which is all bisection needs, and this runs on ONE scalar -- the median --
    not on the image, because a monotone map commutes with the median.
    """
    from raw_pv2012 import _scene_to_perceptual

    def y_at(ev: float) -> float:
        return float(
            _scene_to_perceptual(np.array([median_lum * (2.0**ev)], np.float32))[0]
        )

    lo, hi = -_AUTO_TONE_MAX_EV, _AUTO_TONE_MAX_EV
    if y_at(lo) >= target_y:
        return lo
    if y_at(hi) <= target_y:
        return hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if y_at(mid) < target_y:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def estimate_auto_tone(linear_rgb: np.ndarray) -> Optional[dict]:
    """Auto tone: the six PV2012 Light sliders from a scene-linear buffer.

    Takes the SAME input as Auto WB -- the un-adjusted decoded edit base
    (``_adjust_preview_base_rgb``) -- and returns ABSOLUTE slider values that
    overwrite whatever is on the sliders now, exactly as the WB dropper
    overwrites Temperature/Tint. Two consequences worth stating:

      * pressing AUTO twice is a no-op by construction. It does not compound,
        because the second press measures the same untouched base as the first.
      * all six keys are always returned, zeros included, so applying is a
        clean overwrite rather than a merge that could leave a stale Blacks
        from a previous press fighting the new estimate.

    Returns None when the image is already balanced (every correction inside
    its dead zone) or when the buffer is unusable, mirroring
    ``estimate_straighten_angle``'s refuse-rather-than-guess contract.
    """
    try:
        from raw_pv2012 import _PERCEPTUAL_WHITE, _scene_to_perceptual

        a = linear_rgb
        if a is None or getattr(a, "ndim", 0) != 3 or a.shape[2] < 3:
            return None
        h, w = a.shape[:2]
        step = max(1, int((h * w / 150_000) ** 0.5))
        s = a[::step, ::step, :3].astype(np.float32)
        if s.size < 3 * 300:
            return None
        lum = 0.2126 * s[..., 0] + 0.7152 * s[..., 1] + 0.0722 * s[..., 2]
        lum = np.clip(lum, 0.0, None)
        median = float(np.median(lum))
        if not np.isfinite(median) or median <= 1e-6:
            # An all-black (or unrendered) buffer. Exposure would want +inf and
            # every other measure is meaningless.
            return None

        # Exposure: anchor the median to a midtone, but never past the point
        # where the top of the histogram starts clipping. Median-anchoring
        # alone turns a night scene into +2 EV of mud; the highlight limiter is
        # what makes negative Highlights (below) the recovery mechanism instead
        # of having Exposure and Highlights fight each other.
        #
        # Perceptual white corresponds to linear 1.0 by construction
        # (t = 1/(1+0.25) = 0.8 -> base curve -> _PERCEPTUAL_WHITE), so the
        # headroom solve is a plain log2 in linear -- no bisection needed.
        # The 99.5th percentile, not the max: a specular highlight or a light
        # source is allowed to clip, and on a typical frame it is well under
        # half a percent of the pixels.
        ev_median = _solve_ev_for_median(median, _AUTO_TONE_MEDIAN_TARGET)
        p995 = float(np.percentile(lum, 99.5))
        ev_headroom = (
            float(np.log2(1.0 / p995)) if p995 > 1e-6 else _AUTO_TONE_MAX_EV
        )
        ev = float(np.clip(min(ev_median, ev_headroom), -_AUTO_TONE_MAX_EV, _AUTO_TONE_MAX_EV))

        # Everything below is measured on the frame as Exposure will leave it,
        # so the sliders do not re-correct what Exposure already fixed.
        y = _scene_to_perceptual(lum * (2.0**ev))
        clip_y = _AUTO_TONE_CLIP_FRAC * _PERCEPTUAL_WHITE

        frac_clipped = float(np.mean(y >= clip_y))
        frac_crushed = float(np.mean(y <= _AUTO_TONE_CRUSH_Y))

        highlights = -_AUTO_TONE_HIGHLIGHTS_MAX_SLIDER * _ramp(
            frac_clipped, _AUTO_TONE_CLIP_START, _AUTO_TONE_CLIP_FULL
        )
        shadows = _AUTO_TONE_SHADOWS_MAX_SLIDER * _ramp(
            frac_crushed, _AUTO_TONE_CRUSH_START, _AUTO_TONE_CRUSH_FULL
        )

        # Endpoints. Whites fills unused headroom at the top; Blacks restores a
        # black point the scene never reached (haze, flare, a lifted profile).
        p_hi = float(np.percentile(y, 99.5))
        p_lo = float(np.percentile(y, 0.5))
        headroom_deficit = max(_PERCEPTUAL_WHITE - p_hi, 0.0) / _PERCEPTUAL_WHITE
        whites = _AUTO_TONE_MAX_ENDPOINT * _ramp(
            headroom_deficit, _AUTO_TONE_WHITES_START, _AUTO_TONE_WHITES_FULL
        )
        # Never both: raising Whites on a frame that is already clipping would
        # undo the Highlights recovery two lines up.
        if frac_clipped > _AUTO_TONE_CLIP_START:
            whites = 0.0
        blacks = -_AUTO_TONE_MAX_ENDPOINT * _ramp(
            max(p_lo, 0.0), _AUTO_TONE_BLACKS_START, _AUTO_TONE_BLACKS_FULL
        )

        # Contrast from the perceptual interquartile spread. Positive only --
        # see _AUTO_TONE_NORMAL_IQR.
        q25, q75 = (float(v) for v in np.percentile(y, (25.0, 75.0)))
        iqr = max(q75 - q25, 0.0)
        contrast = _AUTO_TONE_MAX_ENDPOINT * _ramp(
            _AUTO_TONE_NORMAL_IQR - iqr, 0.0, _AUTO_TONE_NORMAL_IQR
        )

        out = {
            "Exposure2012": round(ev, 2),
            "Contrast2012": round(contrast),
            "Highlights2012": round(highlights),
            "Shadows2012": round(shadows),
            "Whites2012": round(whites),
            "Blacks2012": round(blacks),
        }
        if abs(out["Exposure2012"]) < _AUTO_TONE_EV_DEADZONE and all(
            abs(out[k]) < _AUTO_TONE_SLIDER_DEADZONE
            for k in _AUTO_TONE_KEYS
            if k != "Exposure2012"
        ):
            return None
        return {k: float(out[k]) for k in _AUTO_TONE_KEYS}
    except Exception as e:
        logger.debug("[AUTO_TONE] estimate failed: %s", e)
        return None


def _to_uint8_gray(rgb: np.ndarray) -> Optional[np.ndarray]:
    """Accept display uint8, float[0,1], or scene-linear float → uint8 gray."""
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] < 3:
        return None
    a = rgb[..., :3]
    if a.dtype == np.uint8:
        g = a.astype(np.float32)
    else:
        a = a.astype(np.float32)
        # Scene-linear buffers are often >>1 in highlights; tone-map lightly.
        if float(np.percentile(a, 99)) > 1.5:
            a = a / (1.0 + a)
        g = np.clip(a, 0.0, 1.0) * 255.0
    gray = (
        0.2126 * g[..., 0] + 0.7152 * g[..., 1] + 0.0722 * g[..., 2]
    ).astype(np.uint8)
    return gray


def estimate_straighten_angle(display_rgb: np.ndarray) -> Optional[float]:
    """Dominant small tilt (degrees, for CropAngle) from near-axis lines.

    Pipeline:
      1. Canny + probabilistic Hough for long near-H/V segments
      2. Fallback: gradient-orientation histogram peak near the axes

    Each Hough segment within ±12° of horizontal or vertical votes
    (length-weighted) for the correction that would bring it true.
    Returns None when there is nothing lawful to align to.

    ``CropAngle`` is +CCW image rotation (see raw_transform._warped_quad).
    A line whose image-space angle is +dev (CCW from +X) needs CropAngle
    = -dev so the line becomes axis-aligned after the warp.
    """
    try:
        import cv2

        gray = _to_uint8_gray(display_rgb)
        if gray is None:
            return None
        h, w = gray.shape[:2]
        # Work at a bounded size for stable thresholds across megapixel RAWs.
        max_edge = 1400.0
        scale = min(1.0, max_edge / float(max(h, w)))
        if scale < 0.999:
            gray = cv2.resize(
                gray,
                (max(32, int(w * scale)), max(32, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        # Mild blur keeps Canny from latching onto sensor noise / fine foliage.
        gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2)

        edges = cv2.Canny(gray, 40, 140, apertureSize=3)
        min_side = min(gray.shape[:2])
        # Was 0.15 — too strict on many interiors / horizons; 0.06 catches
        # typical horizon + architectural lines without drowning in noise.
        min_len = max(24, int(0.06 * min_side))
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 360.0,
            threshold=max(30, min_len // 3),
            minLineLength=min_len,
            maxLineGap=max(6, min_len // 8),
        )

        votes: list[float] = []
        weights: list[float] = []
        tol = 12.0
        if lines is not None:
            # OpenCV may return (N,1,4) or (N,4) depending on build/version.
            for seg in lines:
                x1, y1, x2, y2 = (float(v) for v in np.asarray(seg).reshape(-1)[:4])
                length = float(np.hypot(x2 - x1, y2 - y1))
                if length < 1.0:
                    continue
                # Image coords: +Y down. arctan2(dy, dx) in degrees.
                ang = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                # Deviation from nearest axis, signed, in (-45, 45].
                folded = ((ang + 45.0) % 90.0) - 45.0
                if abs(folded) <= tol:
                    votes.append(folded)
                    weights.append(length)

        med: Optional[float] = None
        if len(votes) >= 2:
            v = np.asarray(votes, dtype=np.float64)
            wts = np.asarray(weights, dtype=np.float64)
            order = np.argsort(v)
            cw = np.cumsum(wts[order])
            med = float(v[order][np.searchsorted(cw, cw[-1] / 2.0)])
            spread = float(np.sqrt(np.average((v - med) ** 2, weights=wts)))
            # Was 3.0 — slightly more tolerant of mixed architecture + horizon.
            if spread > 4.5:
                med = None

        if med is None:
            med = _gradient_orientation_peak(gray, tol=tol)
        if med is None:
            return None
        if abs(med) < 0.08:
            return None  # already straight enough
        # CropAngle +CCW; line at +med needs opposite rotation.
        return float(np.clip(-med, -45.0, 45.0))
    except Exception as e:
        logger.debug("[AUTO_STRAIGHTEN] estimate failed: %s", e)
        return None


def _gradient_orientation_peak(gray_u8: np.ndarray, *, tol: float = 12.0) -> Optional[float]:
    """Fallback: Sobel orientation histogram peak near H/V axes."""
    try:
        import cv2

        gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        # Edge direction is perpendicular to gradient; line angle = atan2(gy,gx)+90
        # but we want the line's deviation the same way as Hough: use gradient
        # angle folded to nearest axis of the *edge* (gradient ±90°).
        ang = np.degrees(np.arctan2(gy, gx))  # gradient angle
        # Edge orientation = gradient + 90°, then fold to (-45, 45]
        edge = ang + 90.0
        folded = ((edge + 45.0) % 90.0) - 45.0
        strong = mag > max(12.0, float(np.percentile(mag, 90)))
        if int(strong.sum()) < 500:
            return None
        sel = folded[strong]
        # Keep near-axis only
        near = sel[np.abs(sel) <= tol]
        if near.size < 300:
            return None
        # Weighted by magnitude
        w = mag[strong][np.abs(sel) <= tol]
        order = np.argsort(near)
        cw = np.cumsum(w[order])
        med = float(near[order][np.searchsorted(cw, cw[-1] / 2.0)])
        if abs(med) < 0.08:
            return None
        return med
    except Exception:
        return None
