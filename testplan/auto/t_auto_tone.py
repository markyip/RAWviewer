#!/usr/bin/env python3
"""Auto-tone estimator: synthetic scene-linear buffers → expected slider moves.

Estimator only. Nothing here renders through the pipeline: the contract under
test is "what does it ask the sliders to do", and rendering would test the tone
engine instead.

READ THIS BEFORE RETUNING raw_auto_adjust's constants against these numbers.
The thresholds there were calibrated on 20 real Canon CR3 edit bases, NOT on
the fixtures below, and the two disagree in a specific and knowable way: a
lognormal fixture has no true blacks and no specular highlights, so Whites and
Blacks read far higher here than on any real frame (real median p_lo was 0.011
against a 0.04 threshold -- i.e. zero). These tests therefore assert
DIRECTION and BOUNDS, not magnitudes. Asserting a magnitude here would lock
the estimator to a distribution no camera produces.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def _scene(
    median_lin: float,
    *,
    spread: float = 0.5,
    size: int = 400,
    seed: int = 0,
    clipped: float = 0.0,
    lift: float = 0.0,
) -> np.ndarray:
    """Scene-linear RGB with a lognormal luminance distribution.

    ``clipped`` forces that fraction of pixels above sensor white; ``lift``
    adds a flare/haze floor so nothing reaches black.
    """
    rng = np.random.RandomState(seed)
    lum = np.exp(
        rng.normal(np.log(median_lin), spread, (size, size))
    ).astype(np.float32)
    if clipped > 0:
        n = int(clipped * lum.size)
        lum.flat[rng.choice(lum.size, n, replace=False)] = 2.0
    lum = lum + lift
    return np.repeat(lum[:, :, None], 3, axis=2)


def main() -> int:
    from raw_adjustments import DEFAULT_ADJUSTMENTS, SLIDER_SPECS
    from raw_auto_adjust import (
        _AUTO_TONE_CLIP_FULL,
        _AUTO_TONE_CLIP_START,
        _AUTO_TONE_CRUSH_FULL,
        _AUTO_TONE_CRUSH_START,
        _AUTO_TONE_HIGHLIGHTS_MAX_SLIDER,
        _AUTO_TONE_KEYS,
        _AUTO_TONE_MAX_EV,
        _AUTO_TONE_SHADOWS_MAX_SLIDER,
        _ramp,
        estimate_auto_tone,
    )
    from raw_pv2012 import _HIGHLIGHT_LINEAR_MAX_STOPS, _SHADOW_LINEAR_MAX_STOPS

    # --- refusals ---------------------------------------------------------
    check("all-black buffer refuses", estimate_auto_tone(np.zeros((200, 200, 3), np.float32)) is None)
    check("tiny buffer refuses", estimate_auto_tone(np.full((3, 3, 3), 0.2, np.float32)) is None)
    check("None refuses", estimate_auto_tone(None) is None)
    check("2-D buffer refuses", estimate_auto_tone(np.zeros((50, 50), np.float32)) is None)

    # --- direction --------------------------------------------------------
    dark = estimate_auto_tone(_scene(0.02))
    check("underexposed returns an estimate", dark is not None)
    if dark:
        check("underexposed brightens", dark["Exposure2012"] > 0.5, f"EV {dark['Exposure2012']}")

    bright = estimate_auto_tone(_scene(0.60))
    check("overexposed returns an estimate", bright is not None)
    if bright:
        check("overexposed darkens", bright["Exposure2012"] < -0.3, f"EV {bright['Exposure2012']}")

    # A frame that is already clipping must not be brightened into more
    # clipping, however dark its median is -- this is the highlight limiter,
    # and without it a night scene with a lamp gets +2 EV of mud.
    night = estimate_auto_tone(_scene(0.02, clipped=0.02, seed=3))
    check("clipping caps the brightening", night is not None and night["Exposure2012"] <= 0.0,
          f"EV {night['Exposure2012'] if night else None}")

    blown = estimate_auto_tone(_scene(0.25, clipped=0.12, seed=1))
    check("heavily clipped recovers highlights", blown is not None and blown["Highlights2012"] < -20,
          f"Hi {blown['Highlights2012'] if blown else None}")
    if blown:
        check("recovery does not also raise Whites", blown["Whites2012"] == 0.0,
              f"Wh {blown['Whites2012']}")
        check(
            "CLIP_FULL Highlights stay bounded (not -100)",
            blown["Highlights2012"] > -100.0,
            f"Hi {blown['Highlights2012']}",
        )

    # Stop budget: Auto at CLIP_FULL / CRUSH_FULL is the dedicated HS cap,
    # not the stale perceptual-era ±60. Linear engine: slider/100 * max_stops.
    hi_full_stops = _AUTO_TONE_HIGHLIGHTS_MAX_SLIDER / 100.0 * _HIGHLIGHT_LINEAR_MAX_STOPS
    sh_full_stops = _AUTO_TONE_SHADOWS_MAX_SLIDER / 100.0 * _SHADOW_LINEAR_MAX_STOPS
    check(
        "CLIP_FULL Highlights cap is 90 (not 60, not 100)",
        _AUTO_TONE_HIGHLIGHTS_MAX_SLIDER == 90.0,
        f"cap {_AUTO_TONE_HIGHLIGHTS_MAX_SLIDER}",
    )
    check(
        "CLIP_FULL stop budget is ~1.40 (old perceptual -60 was 1.48)",
        1.35 <= hi_full_stops <= 1.45,
        f"stops {hi_full_stops:.3f}",
    )
    check(
        "CRUSH_FULL Shadows cap is 75 (not 60, not 100)",
        _AUTO_TONE_SHADOWS_MAX_SLIDER == 75.0,
        f"cap {_AUTO_TONE_SHADOWS_MAX_SLIDER}",
    )
    check(
        "CRUSH_FULL stop budget is ~1.31 (old perceptual +60 was 1.32)",
        1.28 <= sh_full_stops <= 1.35,
        f"stops {sh_full_stops:.3f}",
    )
    check(
        "ramp at CLIP_FULL is 1",
        _ramp(_AUTO_TONE_CLIP_FULL, _AUTO_TONE_CLIP_START, _AUTO_TONE_CLIP_FULL) == 1.0,
    )
    check(
        "ramp at CRUSH_FULL is 1",
        _ramp(_AUTO_TONE_CRUSH_FULL, _AUTO_TONE_CRUSH_START, _AUTO_TONE_CRUSH_FULL) == 1.0,
    )

    # Mild specular (~2%) must not pin Highlights at the old −60 or the new −90.
    mild = estimate_auto_tone(_scene(0.18, clipped=0.02, seed=4))
    if mild:
        check(
            "mild clip does not pin Highlights at the Auto cap",
            mild["Highlights2012"] > -_AUTO_TONE_HIGHLIGHTS_MAX_SLIDER + 1.0,
            f"Hi {mild['Highlights2012']}",
        )

    flat = estimate_auto_tone(_scene(0.16, spread=0.10, lift=0.06, seed=2))
    check("flat scene gains contrast", flat is not None and flat["Contrast2012"] > 10,
          f"Con {flat['Contrast2012'] if flat else None}")
    if flat:
        check("hazy scene restores a black point", flat["Blacks2012"] < -5, f"Bl {flat['Blacks2012']}")

    # --- ranges and shape -------------------------------------------------
    ranges = {}
    for spec in SLIDER_SPECS:
        if spec.key in _AUTO_TONE_KEYS:
            ranges[spec.key] = (
                spec.slider_to_value(spec.minimum),
                spec.slider_to_value(spec.maximum),
            )

    for name, est in (("underexposed", dark), ("overexposed", bright), ("flat", flat), ("blown", blown)):
        if est is None:
            continue
        check(f"{name}: all six keys present", set(est) == set(_AUTO_TONE_KEYS), f"got {sorted(est)}")
        check(
            f"{name}: every key is a real adjustment",
            all(k in DEFAULT_ADJUSTMENTS for k in est),
        )
        in_range = all(ranges[k][0] <= v <= ranges[k][1] for k, v in est.items())
        check(f"{name}: values inside slider ranges", in_range, f"{est}")
        check(
            f"{name}: EV inside the estimator's own clamp",
            abs(est["Exposure2012"]) <= _AUTO_TONE_MAX_EV + 1e-6,
        )

    # --- idempotence ------------------------------------------------------
    # The estimator reads the UNEDITED base, so the same buffer must give the
    # same answer every time. This is what makes pressing AUTO twice a no-op
    # rather than a compounding edit.
    buf = _scene(0.05, seed=7)
    a, b = estimate_auto_tone(buf), estimate_auto_tone(buf)
    check("same buffer gives the same estimate", a == b, f"{a} vs {b}")

    # --- already balanced -------------------------------------------------
    # "Balanced" means the frame already covers the tonal range evenly: both
    # endpoints reached, nothing piled up at either one. Built by inverting the
    # perceptual map so the PERCEPTUAL histogram is uniform -- a scene-linear
    # ramp will not do, because a power law puts a fifth of its pixels in the
    # crushed band and trips Shadows every time.
    #
    # This is the one fixture that proves the refuse path is reachable at all.
    # Without it, the dead-zone branch is dead code that no test would notice.
    from raw_pv2012 import _scene_to_perceptual

    grid = np.linspace(0.0, 4.0, 20000, dtype=np.float32)
    grid_y = _scene_to_perceptual(grid)
    flat_y = np.linspace(0.0, 0.79, 400 * 400, dtype=np.float32)
    balanced_lum = np.interp(flat_y, grid_y, grid).astype(np.float32).reshape(400, 400)
    balanced = np.repeat(balanced_lum[:, :, None], 3, axis=2)
    check(
        "an already-balanced scene refuses",
        estimate_auto_tone(balanced) is None,
        f"got {estimate_auto_tone(balanced)}",
    )

    if FAILURES:
        print(f"FAILED: {FAILURES}")
        return 1
    print("PASS t_auto_tone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
