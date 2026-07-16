"""
Adobe Lightroom / Camera Raw Process Version 2012 tone engine.

PV2012 (XMP ProcessVersion 11.0) applies a built-in medium-contrast base curve,
then parametric Exposure / Contrast / Highlights / Shadows / Whites / Blacks in
perceptual space. Tone work is luminance-only to preserve hue.
"""

from __future__ import annotations

import numpy as np

from raw_tone_curve import apply_tone_curve_perceptual

PROCESS_VERSION = "11.0"
TONE_CURVE_NAME_2012 = "Linear"

# Medium-contrast base baked into PV2012 “Linear” (0–255 coordinates).
_BASE_CURVE_XY = np.array(
    [
        [0.0, 0.0],
        [32.0 / 255.0, 22.0 / 255.0],
        [64.0 / 255.0, 56.0 / 255.0],
        [128.0 / 255.0, 128.0 / 255.0],
        [192.0 / 255.0, 196.0 / 255.0],
        [1.0, 1.0],
    ],
    dtype=np.float32,
)

# Default parametric region splits (Shadows | Darks | Lights | Highlights).
_SPLIT_SHADOWS = 0.25
_SPLIT_DARKS = 0.50
_SPLIT_LIGHTS = 0.75

_LUT_SIZE = 65536
_BASE_LUT = np.interp(
    np.linspace(0.0, 1.0, _LUT_SIZE, dtype=np.float32),
    _BASE_CURVE_XY[:, 0],
    _BASE_CURVE_XY[:, 1],
).astype(np.float32)

# Ratio-stability epsilon: added to BOTH numerator and denominator so the
# no-op case (y1 == y0) is exactly ratio 1.0 at every luminance, while a
# genuine near-zero denominator can't blow up unbounded. Replaces the old
# hard floor of 0.03, which zeroed slider response for every pixel whose
# adjusted luminance stayed under the floor -- exactly the deep-shadow range
# Shadows/Blacks exist to recover.
_RATIO_EPS = 0.005
# 8x = 3 stops of shadow lift, 0.25x = 2 stops of crush. Modern sensors
# hold recoverable detail well past 3 stops under (Exposure, applied
# scene-linear upstream, is uncapped and proves it); the old 3.0 cap
# (~1.6 stops) made Shadows/Blacks top out far below what the file
# contains.
#
# A first pass raised this to 16x (4 stops) and relied on the chroma damp
# below to control noise. In practice, real per-pixel sensor chroma noise
# (the blue channel especially -- lowest QE/highest relative read noise
# on most Bayer sensors) gets amplified by the SAME ratio as genuine
# shadow detail: a real photo test measured a synthetic 0.0004 scene-
# linear blue-channel noise std turning into a display-space blue-vs-green
# channel deviation of std ~3.0 (about 4x the luminance grain) at 16x,
# visible as colored speckle in dark clothing/hair -- reported as "blue
# color artifact" with less perceived detail (the speckle masks it). 8x
# roughly halves the worst-case amplification while still recovering far
# more than the original 3.0 cap; the strengthened chroma damp below
# (raised from 0.35 to 0.85 max) does most of the remaining noise control.
#
# NOTE (post return_linear fix): fixing decode_half_from_unpacked's
# ignored return_linear flag (fast_raw_decode.py) means y0 here is now
# genuinely scene-linear instead of gamma-encoded data mislabeled as
# linear -- true deep-shadow y0 on a real CR3 runs as low as ~0.0006 in
# the darkest 5%, which saturates this 8x cap across nearly the entire
# Shadows 50-100 slider range (measured: Shadows=50 and Shadows=100
# produce identical output there) and likely contributes to a recurring
# "grey casting" report. Tried raising back to 16x post-fix: t_tone_engine
# regressed (chroma speckle test failed, b-g std 0.87 > luma std 0.43),
# so the cap is not the right knob to turn -- a naive increase reproduces
# the exact speckle regression 8x was chosen to fix, just via the
# now-corrected data scale instead of the old buggy one. Left at 8x;
# the anchor band and lift_frac saturation point (both below) were also
# tuned against the same buggy data and need their own recalibration
# pass against real files before this is revisited.
_MAX_TONE_RATIO = 8.0
_MIN_TONE_RATIO = 0.25


# Where scene-linear white (camera clip, lum=1.0) lands in the perceptual
# working space: t = 1/(1+0.25) = 0.8, then the base curve -> ~0.813. The
# parametric splits above assume display-referred space where white is 1.0,
# but _scene_to_perceptual never reaches 1.0 for finite luminance -- clipped
# whites sit at ~0.813, only a quarter of the way into the [0.75, 1.0]
# highlight band. Any highlight math that measures its region weight or knee
# against 1.0 is therefore triple-attenuated into a no-op on the very pixels
# Highlights exists for (measured: Highlights=-100 changed a clipped white by
# 0.7%). Highlight-region math must normalize by this constant.
_PERCEPTUAL_WHITE = float(
    np.interp(1.0 / 1.25, _BASE_CURVE_XY[:, 0], _BASE_CURVE_XY[:, 1])
)  # ~0.813


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    )


def _region_weight_highlights(y: np.ndarray) -> np.ndarray:
    span = max(1.0 - _SPLIT_LIGHTS, 1e-6)
    t = np.clip((y - _SPLIT_LIGHTS) / span, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _region_weight_shadows(y: np.ndarray) -> np.ndarray:
    t = np.clip((_SPLIT_SHADOWS - y) / max(_SPLIT_SHADOWS, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _apply_base_curve(y: np.ndarray) -> np.ndarray:
    yi = np.clip(y, 0.0, 1.0) * (_LUT_SIZE - 1)
    lo = np.floor(yi).astype(np.int32)
    hi = np.minimum(lo + 1, _LUT_SIZE - 1)
    frac = yi - lo
    return _BASE_LUT[lo] * (1.0 - frac) + _BASE_LUT[hi] * frac


def _scene_to_perceptual(lum: np.ndarray) -> np.ndarray:
    """Scene-linear luminance → PV2012 perceptual working space."""
    t = np.clip(lum.astype(np.float32), 0.0, None)
    t = t / (t + 0.25)
    return _apply_base_curve(t)


def _apply_pv2012_contrast(y: np.ndarray, contrast_val: float) -> np.ndarray:
    c = contrast_val / 100.0
    factor = (259.0 * (c * 100.0 + 255.0)) / (255.0 * (259.0 - c * 100.0))
    return (y - 0.5) * factor + 0.5


def _apply_whites_blacks(y: np.ndarray, whites: float, blacks: float) -> np.ndarray:
    """
    Whites raises the input needed to hit output 1.0 (more headroom = less
    clipping = darker at the top when whites > 0 is wrong -- whites > 0 must
    brighten/clip highlights, matching Lightroom and this module's own legacy
    gamma-space path). Signs are therefore inverted from the naive "raise the
    point" reading: whites > 0 lowers white_pt (less input needed to clip),
    blacks > 0 lowers black_pt (extends below 0, lifting shadows off the floor).
    """
    w = whites / 100.0
    b = blacks / 100.0
    # 0.25, not the original 0.12: at +-12% max shift, Whites/Blacks barely
    # moved the black/white point at all -- nowhere near strong enough to
    # visibly clip highlights/crush shadows the way real Whites/Blacks
    # sliders do at their extremes. 0.25 gives genuine clipping at +-100
    # (worst-case combined span, both maxed toward each other, is 0.5 --
    # still comfortably away from a degenerate/near-zero span).
    white_pt = 1.0 - w * 0.25
    black_pt = -b * 0.25
    span = max(white_pt - black_pt, 1e-6)
    return np.clip((y - black_pt) / span, 0.0, 1.0)


def _apply_highlights_shadows(y: np.ndarray, highlights: float, shadows: float) -> np.ndarray:
    hi = highlights / 100.0
    sh = shadows / 100.0
    if abs(hi) < 1e-6 and abs(sh) < 1e-6:
        return y
    out = y.copy()
    if abs(sh) > 1e-6:
        sw = _region_weight_shadows(out)
        if sh > 0:
            # Coefficient capped at 0.15 (not the more intuitive-looking 0.42): the
            # shadow-lift weight's slope reaches ~5.85 at its steepest (y~0.11), so
            # any coefficient above ~0.17 makes d(out)/dy go negative there --
            # a real, verified tone-curve inversion (banding) that showed up at
            # Shadows>=~45 with the old 0.42. 0.15 keeps a full-range monotonic
            # curve at Shadows=100 with margin to spare.
            out = out + sw * sh * (1.0 - out) * 0.15
        else:
            out = out + sw * sh * out * 0.38
    if abs(hi) > 1e-6:
        if hi < 0:
            # Highlight recovery as a blend between identity and a fixed
            # rational shoulder, in WHITE-NORMALIZED space (see
            # _PERCEPTUAL_WHITE: clipped whites live at ~0.813, so the old
            # weight*knee*0.55 form -- all measured against 1.0 -- moved a
            # clipped white by 0.7% at Highlights=-100, reported as
            # "highlights recovery does not reduce the highlight intensity").
            #
            # C(yn) = 0.5 + 0.5*x/(1+8.4*x), x=(yn-0.5)/0.5: identity-joined
            # (C'(0.5)=1) at normalized mid, compressing to C(1)=0.553 -- a
            # clipped white at -100 ends ~0.85 stop down in linear, near-
            # clipped gradations (yn~0.9) pull ~0.7 stop, and scene mid-grey
            # (yn~0.50) is untouched. Monotone by construction: a fixed-weight
            # blend of two monotone curves needs no coefficient/slope safety
            # analysis, unlike the subtractive form it replaces.
            yn = out / _PERCEPTUAL_WHITE
            x = np.maximum((yn - 0.5) / 0.5, 0.0)
            target = (0.5 + 0.5 * x / (1.0 + 8.4 * x)) * _PERCEPTUAL_WHITE
            target = np.minimum(target, out)  # never brightens
            out = out + (-hi) * (target - out)
        else:
            # Brighten-highlights keeps the legacy form, but with the region
            # weight measured in white-normalized space so it actually engages
            # on real highlight data (same 0.813 issue as above), and the
            # headroom term measured to normalized white rather than 1.0.
            hw = _region_weight_highlights(np.minimum(out / _PERCEPTUAL_WHITE, 1.0))
            out = out + hw * hi * np.maximum(_PERCEPTUAL_WHITE - out, 0.0) * 0.35
    return np.clip(out, 0.0, 1.0)


def _apply_pv2012_perceptual(
    y: np.ndarray,
    *,
    contrast: float = 0.0,
    highlights: float = 0.0,
    shadows: float = 0.0,
    whites: float = 0.0,
    blacks: float = 0.0,
) -> np.ndarray:
    out = y.copy()
    if abs(contrast) > 1e-4:
        out = _apply_pv2012_contrast(out, contrast)
    if abs(highlights) > 1e-4 or abs(shadows) > 1e-4:
        out = _apply_highlights_shadows(out, highlights, shadows)
    if abs(whites) > 1e-4 or abs(blacks) > 1e-4:
        out = _apply_whites_blacks(out, whites, blacks)
    return np.clip(out, 0.0, 1.0)


def apply_pv2012_tone_rgb(
    img: np.ndarray,
    adj: dict[str, float],
    *,
    preview_lite: bool = False,
) -> np.ndarray:
    """Hue-preserving PV2012 tone; ratio capped in perceptual space.

    ``preview_lite=True`` (Adjust live-drag only): skip the y0 guided filter
    and shadow chroma-damp guided filters. Those dominate tick cost on the
    fast base; settle / export always use the full path. Live-drag uses a
    separate PreviewStageCache, so lite outputs never poison full-quality
    stages.
    """
    lum = _luminance(img)
    y0_raw = _scene_to_perceptual(lum)
    # Smooth y0 before it drives anything downstream (tone curve, PV2012
    # curve, ratio) -- darktable's tone-equalizer approach (see research
    # notes) to a problem our per-pixel ratio has always had: ratio = y1/y0
    # computed independently per pixel inherits whatever noise is in that
    # pixel's own y0, so neighbouring pixels with the same *true* local
    # brightness get visibly different ratios purely from sensor noise
    # jitter -- amplifying that jitter into color speckle on top of
    # whatever the sensor noise already was. A self-guided filter on y0
    # (small radius: this is noise-scale smoothing, not the coarse
    # region-scale masking the chroma damp below does) averages out that
    # per-pixel jitter while a real edge (self-guided, so it's guided by
    # its own structure) still survives. Verified on two real files
    # (Canon_Sample/6J8A0376.CR3, a Sony ARW): at r=2, both moved *closer*
    # to Exposure's color richness at matched brightness (Canon 58% -> 71%
    # of Exposure's chroma, Sony 90% -> 97%), not just the noisier one --
    # larger radii (8+) helped the noisy file further but started
    # over-smoothing the cleaner file's real structure, so this is a light
    # touch specifically for noise-scale jitter, not a substitute for the
    # region-scale chroma damp below.
    if preview_lite:
        y0 = np.clip(y0_raw.astype(np.float32), 0.0, None)
    else:
        from raw_chroma_denoise import apply_guided_filter

        y0 = np.clip(
            apply_guided_filter(
                y0_raw.astype(np.float32), y0_raw.astype(np.float32), 2, 0.0005
            ),
            0.0,
            None,
        )
    # y0 must stay the true pre-curve, pre-PV2012 baseline: it's the ratio's
    # denominator below. Reassigning it to the curve-adjusted value here (as
    # this used to do) makes the point tone curve's effect appear in *both*
    # the numerator (y1, computed from the curved value) and the denominator,
    # so it exactly cancels out of the ratio -- the point curve UI was fully
    # wired end-to-end (widget -> XMP -> pipeline) but had zero visible
    # effect on the image, reported as "the tone curve is unresponsive".
    y_curve = apply_tone_curve_perceptual(y0, adj)
    y1 = _apply_pv2012_perceptual(
        y_curve,
        contrast=float(adj.get("Contrast2012", 0.0)),
        highlights=float(adj.get("Highlights2012", 0.0)),
        shadows=float(adj.get("Shadows2012", 0.0)),
        whites=float(adj.get("Whites2012", 0.0)),
        blacks=float(adj.get("Blacks2012", 0.0)),
    )

    # Shared epsilon (not a shared hard floor): identity (y1 == y0) maps to
    # exactly ratio 1.0 at every luminance -- the old max(y, 0.03) floor on
    # both sides met that requirement too, but at the cost of clamping the
    # *numerator*: any pixel whose adjusted luminance stayed under the floor
    # got ratio 1.0 = zero slider response, silencing Shadows/Blacks in
    # precisely the deep-shadow range they're for. The epsilon keeps the
    # blowup protection (bounded by _MAX_TONE_RATIO) without dead zones.
    ratio = (y1 + _RATIO_EPS) / (y0 + _RATIO_EPS)
    ratio = np.clip(ratio, _MIN_TONE_RATIO, None)
    # Soft knee instead of a hard clip at _MAX_TONE_RATIO. The hard clip
    # saturated for deep-shadow pixels well before the slider maxed out
    # (raw ratio at scene-linear ~0.002 is ~8 at Shadows=50 and ~15 at
    # Shadows=100 -- both clipped to 8), so Shadows=50 and Shadows=100
    # produced IDENTICAL output across most of the deep-shadow range: the
    # "shadow recovery is limited" report, and part of the grey-veil one
    # (every deep tone collapsing onto the same cap flattens local
    # contrast). tanh keeps the same asymptotic ceiling and identity below
    # the knee, but stays strictly increasing in the raw ratio, so slider
    # positions above ~50 remain distinguishable (measured 7.06 vs 7.97 at
    # the levels above).
    _knee = 4.0
    over = ratio > _knee
    ratio = np.where(
        over,
        _knee
        + (_MAX_TONE_RATIO - _knee)
        * np.tanh((ratio - _knee) / (_MAX_TONE_RATIO - _knee)),
        ratio,
    )

    # Black-point anchor: taper the lift back toward 1.0 as y0 approaches
    # black. First tuning ramped over y0 in [0, 0.004] (noise floor only) --
    # in practice near-black CONTENT just above that band (~7-9 stops under
    # middle grey: black fabric, deep interior shadow) still took the full
    # 16x cap, flattening every dark region into the same mid-grey veil,
    # reported twice as "grey shade/casting". Ramping over [0.001, 0.012]
    # instead gives a film-toe-like response: absolute black pinned at 0,
    # ~6.5-stops-under content lifts moderately (still several times more
    # than the old 3.0-cap engine), and the 5-7-stops-under detail band
    # (y0 >= ~0.012, scene-linear >= ~0.004) keeps the full recovery range
    # the Shadows/Blacks rework was built for.
    t = np.clip((y0 - 0.001) / (0.012 - 0.001), 0.0, 1.0)
    anchor = t * t * (3.0 - 2.0 * t)
    # Anchor LIFT only (ratio > 1). Tapering the darkening side too made
    # near-black pixels darken LESS than slightly-brighter neighbours at
    # Blacks < 0 -- a real 2-LSB tone inversion at the band edge (caught by
    # testplan/auto/t_tone_engine.py monotonicity). Darkening needs no
    # anchor: img * ratio at img ~ 0 stays ~ 0, so there is no grey-veil or
    # blowup risk in that direction.
    ratio = np.where(ratio > 1.0, 1.0 + (ratio - 1.0) * anchor, ratio)
    out = img * ratio[..., np.newaxis]

    # Shadow lift amplifies demosaic/sensor chroma noise — pull chroma
    # toward luma in proportion to how much a pixel was actually lifted.
    #
    # Gated on the RATIO itself (any(ratio > 1)), not on the Shadows slider
    # value: Blacks alone also raises `ratio` in the shadow region (see
    # _apply_whites_blacks), and gating on `sh` left a Blacks-only push
    # completely undamped -- the exact same colored-speckle failure mode,
    # just reachable without touching Shadows at all.
    if np.any(ratio > 1.0 + 1e-6) and not preview_lite:
        sw = _region_weight_shadows(y_curve)
        # Anchor on the POST-lift luminance: with the pre-lift ``lum`` here,
        # a lifted neutral pixel's uniform (out - lum) offset -- which is the
        # lift itself, not chroma -- was damped too, cancelling most of the
        # recovery and making Shadows+Blacks lift *less* than Blacks alone
        # (measured 38 vs 48 at scene-linear 0.01). Post-lift luminance
        # makes the damp act only on true color deviation.
        #
        # lift_frac saturates by ratio=4 (2 stops), not just at the ratio
        # cap: noise amplification is already a problem well before max
        # lift, so damp strength should ramp up early, not only at the tail.
        lift_frac = np.clip((ratio - 1.0) / 3.0, 0.0, 1.0)
        luma_2d = lum * ratio

        # Edge-aware damp strength (max strength history: 0.35 -> 0.85 ->
        # 0.6 flat -> edge-aware here). A flat global strength can't tell a
        # real fold line or fabric weave (genuine local luminance
        # structure) apart from per-pixel sensor noise -- both produce
        # similar-looking local chroma deviation, so damping hard enough to
        # control the noise also flattens real color texture into grey
        # (measured: at matched brightness, a real fabric crop carried
        # 2.4x more chroma under an equivalent Exposure push than under
        # Shadows+100 -- reported as "duller than Exposure"). A plain
        # Sobel edge weight on the raw lifted luminance doesn't fix this:
        # per-pixel noise produces gradients just as sharp as a real edge
        # to a 3x3 operator, so nearly the whole shadow region reads as
        # "edge" regardless. Self-guided-filtering the luminance first
        # (eps large enough to actually smooth, ~10x the local noise
        # variance) suppresses the pixel-independent noise while leaving
        # multi-pixel real structure (fold lines, weave) intact, so edge
        # detection on *that* actually separates the two -- verified
        # visually (scripts/shadow_tuning_*.py): fold lines read as bright
        # ridges in the edge map, flat/noisy fabric reads dark.
        #
        # Edge multiplier history: 0.5 -> 1.0 (full damp removal at max
        # edge confidence). 0.5 (halved, not zeroed, at a real edge) was
        # the cautious first value; a real-file sweep (Canon fabric crop +
        # a Sony ARW) showed 1.0 recovers meaningfully more color on both
        # (sat_proxy +19% and +5.5% respectively over the 0.5 value) with
        # the flat/noisy region completely unaffected either way (a
        # synthetic hard-edge-vs-flat-noise test held flat_chroma constant
        # at 2.32-2.33x its no-lift baseline across 0.5-1.0 -- the edge
        # weight is ~0 there regardless, so edge_mult only ever acts where
        # real structure was actually detected).
        from raw_chroma_denoise import _luma_edge_weight, apply_guided_filter

        smooth_luma = apply_guided_filter(luma_2d, luma_2d, 10, 0.003)
        edge_w = _luma_edge_weight(smooth_luma.astype(np.float32), soft=0.008)
        damp_strength = 0.6 * (1.0 - edge_w)

        damp = 1.0 - (sw * damp_strength * lift_frac)[..., np.newaxis]
        luma = luma_2d[..., np.newaxis]
        chroma = out - luma
        # Damp only the HIGH-FREQUENCY chroma component. Sensor/demosaic
        # chroma noise -- the thing this damp exists for -- is per-pixel
        # (high-frequency) by nature, while a genuinely colored surface
        # (fabric, skin, painted wall) carries its color as a low-frequency
        # chroma field. The previous full-chroma damp could not tell them
        # apart, so a flat colored surface under a strong lift lost ~60% of
        # its saturation (measured sat retention 0.44 at Shadows>=50): the
        # "significant grey artifacts" report. The edge-aware strength above
        # only protects luminance STRUCTURE (folds, weave); a uniformly
        # colored flat region has no luma edges and was damped to grey.
        # Splitting chroma with a small self-guided filter keeps the speckle
        # suppression bit-for-bit on the noise (per-pixel deviations have
        # ~zero low-frequency component) while passing surface color through
        # untouched.
        # The LF field is low-frequency by definition, so extract it at
        # quarter resolution: full-res self-guided filtering here measured
        # 462ms at 3004x2004 (tripling this stage), the 4x-downsampled
        # equivalent ~35ms with visually identical output (INTER_AREA down /
        # INTER_LINEAR up cannot introduce content above the very cutoff the
        # filter is discarding anyway).
        import cv2

        h, w = chroma.shape[:2]
        ds = cv2.resize(chroma, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
        for c in range(3):
            ch = np.ascontiguousarray(ds[..., c])
            ds[..., c] = apply_guided_filter(ch, ch, 2, 0.0005)
        chroma_lf = cv2.resize(ds, (w, h), interpolation=cv2.INTER_LINEAR)
        out = luma + chroma_lf + (chroma - chroma_lf) * damp

    return np.clip(out, 0.0, None)
