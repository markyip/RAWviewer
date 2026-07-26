#!/usr/bin/env python3
"""Tone-engine invariants: identity, monotonicity, black anchor, recovery floor."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from raw_edit_pipeline import linear_to_display_uint8, process_linear_edit_buffer

    def render(img, adj):
        return linear_to_display_uint8(process_linear_edit_buffer(img, adj, preview=True), adj)

    # The display encode adds +/-1 LSB of TPDF dither so denoised gradients
    # do not posterize on 8-bit export. That is correct and deliberate, but
    # two independent TPDF samples differ by up to 2 LSB, so a dithered
    # ramp legitimately steps backwards by 1-2 codes and cannot be used to
    # test the CURVE's monotonicity. render_undithered isolates the curve;
    # the dither's own contract (never more than 1 LSB from the undithered
    # value) is asserted separately below.
    import contextlib

    import raw_edit_pipeline as _rep

    @contextlib.contextmanager
    def dither_disabled():
        saved_fn, saved_cache = _rep._dither_tile, _rep._dither_tile_cache
        _rep._dither_tile = lambda: np.zeros((512, 512), dtype=np.float32)
        _rep._dither_tile_cache = None
        try:
            yield
        finally:
            _rep._dither_tile, _rep._dither_tile_cache = saved_fn, saved_cache

    def render_undithered(img, adj):
        with dither_disabled():
            return render(img, adj)

    # 1. Identity: defaults / zeroed sliders are an exact no-op
    img = (np.random.RandomState(7).rand(64, 64, 3).astype(np.float32)) ** 2
    a0 = render(img, {})
    for adj in ({"Exposure2012": 0.0}, {"Shadows2012": 0.0}, {"Blacks2012": 0.0}):
        d = np.abs(a0.astype(int) - render(img, adj).astype(int)).max()
        check(f"identity {list(adj)[0]}=0", d == 0, f"maxdiff={d}")

    # 2. Monotonicity at slider extremes (no tone-curve inversion/banding).
    # Measured on the undithered encode -- see render_undithered above.
    xs = np.linspace(1e-5, 1.2, 800, dtype=np.float32)
    ramp = np.repeat(xs[None, :, None], 3, axis=2).reshape(1, -1, 3)
    extremes = (
        {"Shadows2012": 100.0}, {"Shadows2012": -100.0},
        {"Blacks2012": 100.0}, {"Blacks2012": -100.0},
        {"Whites2012": 100.0}, {"Highlights2012": -100.0},
        {"Shadows2012": 100.0, "Blacks2012": 100.0},
    )
    for adj in extremes:
        out = render_undithered(ramp, adj)[0, :, 0].astype(int)
        worst = int(np.diff(out).min())
        check(f"monotonic {adj}", worst >= -1, f"worst step={worst}")

    # 2b. Dither contract: it must perturb by at most 1 LSB, at EVERY
    # brightness. This is the guard for a real defect -- the dither was
    # added to the LINEAR buffer before the BT.709 encode, whose toe has a
    # 4.5x slope, so near black "1 LSB" arrived as 4-5 output codes: noisy
    # deep shadows, a black frame quantizing to 1, and a ramp that lost
    # monotonicity. Dithering after the encode makes 1 LSB mean 1 LSB
    # everywhere. A regression to linear-space dither fails here loudly.
    for adj in extremes:
        dithered = render(ramp, adj)[0, :, 0].astype(int)
        clean = render_undithered(ramp, adj)[0, :, 0].astype(int)
        worst = int(np.abs(dithered - clean).max())
        check(f"dither <= 1 LSB {adj}", worst <= 1, f"worst deviation={worst}")

    # 3. Black anchor: absolute black stays 0 under max lift
    black = np.zeros((4, 4, 3), dtype=np.float32)
    v = int(render(black, {"Shadows2012": 100.0, "Blacks2012": 100.0})[0, 0, 0])
    check("absolute black pinned", v == 0, f"out={v}")

    # 4. Toe shape: noise floor barely lifts, detail band lifts strongly
    def lift(scene, adj):
        img2 = np.full((4, 4, 3), scene, dtype=np.float32)
        return int(render(img2, adj)[0, 0, 0])

    sb = {"Shadows2012": 100.0, "Blacks2012": 100.0}
    check("noise floor (8.5 stops under) stays dark", lift(0.0005, sb) <= 6,
          f"out={lift(0.0005, sb)}")
    # Relative, not a fixed pixel value: _MAX_TONE_RATIO was deliberately
    # lowered from 16x to 8x (real-photo chroma-noise speckle regression,
    # see raw_pv2012.py's comment on _MAX_TONE_RATIO) which intentionally
    # reduces the raw recovery magnitude at this exact scene level. Check
    # recovery is still strong RELATIVE to the unlifted base rather than
    # pinning to the old (buggier) engine's absolute output.
    base_at_5_2_stops = lift(0.005, {})
    recovered = lift(0.005, sb)
    check(
        "detail band (5.2 stops under) recovers",
        recovered >= base_at_5_2_stops * 6,
        f"base={base_at_5_2_stops} recovered={recovered}",
    )

    # 5. Combined >= individual (regression: chroma damp once cancelled lift)
    s_only = lift(0.01, {"Shadows2012": 100.0})
    b_only = lift(0.01, {"Blacks2012": 100.0})
    both = lift(0.01, sb)
    check("shadows+blacks >= max(individual)", both >= max(s_only, b_only) - 1,
          f"s={s_only} b={b_only} both={both}")

    # 6. Recovery floor: >= 2x the old 3.0-ratio-cap engine's reach
    base = lift(0.01, {})
    check("recovery strength floor", both >= base * 3, f"base={base} both={both}")

    # 6b. Scene-linear Shadows: deep detail recovers like a partial Exposure
    # push, while midtones stay nearly put (the whole point vs Exposure).
    s100 = lift(0.01, {"Shadows2012": 100.0})
    exp_half = lift(0.01, {"Exposure2012": 0.9})
    check(
        "Shadows=100 deep lift rivals ~+0.9 EV locally",
        s100 >= exp_half - 2,
        f"shadows={s100} exp0.9={exp_half}",
    )
    mid0 = lift(0.25, {})
    mid_s = lift(0.25, {"Shadows2012": 100.0})
    check(
        "Shadows=100 leaves midtones nearly unchanged",
        abs(mid_s - mid0) <= 8,
        f"mid0={mid0} mid_s={mid_s}",
    )
    hi0 = lift(0.85, {})
    hi_rec = lift(0.85, {"Highlights2012": -100.0})
    check(
        "Highlights=-100 darkens near-white",
        hi_rec < hi0 - 5,
        f"hi0={hi0} hi_rec={hi_rec}",
    )
    mid_hi = lift(0.25, {"Highlights2012": -100.0})
    check(
        "Highlights=-100 leaves midtones nearly unchanged",
        abs(mid_hi - mid0) <= 8,
        f"mid0={mid0} mid_hi={mid_hi}",
    )

    # 7. Chroma-speckle regression: real per-pixel sensor noise must not be
    # amplified into a visible color cast by strong Shadows/Blacks lift.
    # Reproduces the reported bug (blue speckle in dark clothing/hair,
    # ISO 1100 NEF) with synthetic noise standing in for sensor chroma
    # noise; asserts the fix's channel-deviation ceiling rather than an
    # exact value, since some residual chroma is expected and fine.
    rng = np.random.RandomState(5)
    h, w = 40, 40
    noisy = np.full((h, w, 3), 0.01, dtype=np.float32)
    noisy += rng.normal(0, 0.0006, (h, w, 3)).astype(np.float32)
    noisy[:, :, 2] += rng.normal(0, 0.0004, (h, w)).astype(np.float32)
    noisy = np.clip(noisy, 0, None)
    out_noisy = render(noisy, {"Shadows2012": 94.0, "Blacks2012": 94.0})
    b_minus_g_std = float((out_noisy[..., 2].astype(np.float32) - out_noisy[..., 1].astype(np.float32)).std())
    luma_std = float(out_noisy.mean(axis=-1).std())
    check(
        "chroma speckle contained (blue-vs-green deviation <= luma grain)",
        b_minus_g_std <= luma_std * 1.5,
        f"b-g std={b_minus_g_std:.2f} luma std={luma_std:.2f}",
    )

    # 8. Blacks-only push must be damped too (bug: damp used to gate on the
    # Shadows slider value specifically, so a Blacks-only push skipped it).
    out_blacks_only = render(noisy, {"Blacks2012": 94.0})
    bg_blacks_only = float((out_blacks_only[..., 2].astype(np.float32) - out_blacks_only[..., 1].astype(np.float32)).std())
    check(
        "Blacks-only push also damped",
        bg_blacks_only <= luma_std * 1.5,
        f"b-g std={bg_blacks_only:.2f}",
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
