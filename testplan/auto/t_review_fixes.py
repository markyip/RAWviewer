#!/usr/bin/env python3
"""Regressions for the code-review findings. Each test names the failure.

These are all "silently wrong" bugs -- none of them raised, and several
destroyed or misrepresented the user's work while looking like they had
succeeded. That is why they are pinned here rather than left to be noticed.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def _layer(i: int = 0, **kw):
    from raw_mask_layers import MaskLayer

    layer = MaskLayer.empty(64, 64, name=f"M{i}", **kw)
    layer.alpha[10:30, 10:30] = 1.0
    layer.touch()
    return layer


def test_mask_load_is_deterministic_and_never_overwrites_after_truncation():
    """A RAM-derived cap on LOAD silently destroyed saved masks.

    max_mask_layers() is a fraction of currently-available RAM. Using it to
    decide how much of a sidecar to load meant a 12-layer file opened on a
    busy machine kept the first few layers and dropped the rest -- and the
    next slider release serialized the truncated stack back over the sidecar.
    Same file, same machine, an hour later: loads fine. Nondeterministic,
    permanent data loss.
    """
    import mask_layers_xmp as MX
    from mask_layers_xmp import (
        MaskStackTruncated,
        deserialize_stack,
        serialize_stack,
    )
    from raw_mask_layers import MaskLayerStack

    big = MaskLayerStack([_layer(i) for i in range(MX._HARD_MAX_LAYERS + 6)])
    serial = serialize_stack(big)

    original = MX._available_ram_bytes
    try:
        MX._available_ram_bytes = lambda: 64 * 1024 * 1024  # starved
        low = deserialize_stack(serial)
        MX._available_ram_bytes = lambda: 64 * 1024 * 1024 * 1024  # plenty
        high = deserialize_stack(serial)
    finally:
        MX._available_ram_bytes = original

    check(
        "available RAM does not change how many layers load",
        len(low.layers) == len(high.layers),
        f"{len(low.layers)} starved vs {len(high.layers)} plentiful",
    )
    check("a truncated stack is flagged", low.truncated)
    check(
        "truncated stacks report how many loaded",
        int(getattr(low, "truncated_loaded", 0)) == MX._HARD_MAX_LAYERS
        and int(getattr(low, "truncated_total", 0)) == MX._HARD_MAX_LAYERS + 6,
        f"loaded={getattr(low, 'truncated_loaded', None)} "
        f"total={getattr(low, 'truncated_total', None)}",
    )
    try:
        serialize_stack(low)
        check("a truncated stack refuses to be written back", False)
    except MaskStackTruncated:
        check("a truncated stack refuses to be written back", True)

    # Unlock path the UI offers: accept the incomplete stack and save it.
    low.truncated = False
    low.truncated_loaded = 0
    low.truncated_total = 0
    low.truncated_cap = 0
    check(
        "clearing truncated unlocks serialize",
        bool(serialize_stack(low)),
    )

    # And the ordinary case is untouched.
    ok = deserialize_stack(serialize_stack(MaskLayerStack([_layer(0), _layer(1)])))
    check(
        "a normal stack still round-trips and saves",
        len(ok.layers) == 2 and not ok.truncated and bool(serialize_stack(ok)),
    )


def test_blur_focus_zero_survives_a_save():
    """BlurFocus defaults to 100, so "non-zero" was the wrong save test.

    Pulling Focus fully to the background (0) was dropped from the sidecar as
    if untouched, and reloaded as the 100 default -- the sharp plane flipped
    from far to near and the depth of field inverted between sessions.
    """
    from mask_layers_xmp import deserialize_stack, serialize_stack
    from raw_mask_layers import MaskLayerStack, _adj_value

    for focus in (0.0, 25.0, 100.0):
        layer = _layer(0, adjustments={"Blur": 30.0, "BlurFocus": focus})
        layer.source = "depth"
        stack = deserialize_stack(serialize_stack(MaskLayerStack([layer])))
        got = _adj_value(stack.layers[0].adjustments, "BlurFocus")
        check(f"BlurFocus {focus:.0f} survives a save/load", abs(got - focus) < 1e-3,
              f"reloaded as {got}")


def test_depth_blur_keeps_the_other_adjustments_masked():
    """Turning on Blur must not change what every other slider does.

    The depth branch wrote the blur result for the whole bbox without the
    alpha lerp, so a Depth mask at +1 EV rendered as a correct gradient until
    Blur was touched -- at which point it became a flat +1 EV everywhere.
    """
    from raw_mask_layers import MaskLayer, MaskLayerStack, apply_mask_layers

    h = w = 128
    img = np.full((h, w, 3), 0.25, np.float32)
    # Depth alpha is NEARNESS: 1 at the top row, fading to 0 at the bottom.
    alpha = np.tile(np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None], (1, w))

    def render(blur: float):
        layer = MaskLayer(
            alpha.copy(),
            kind="depth",
            source="depth",
            adjustments={"Exposure2012": 1.0, "Blur": blur},
        )
        layer.touch()
        return apply_mask_layers(img.copy(), MaskLayerStack([layer]))

    plain, blurred = render(0.0), render(40.0)
    far_plain = float(plain[-3, w // 2, 0])
    far_blurred = float(blurred[-3, w // 2, 0])
    grad_plain = float(plain[2, w // 2, 0]) - far_plain
    grad_blurred = float(blurred[2, w // 2, 0]) - far_blurred

    check(
        "the far end is not lifted to full strength once Blur is on",
        abs(far_blurred - 0.25) < 0.05,
        f"far={far_blurred:.3f}, unadjusted=0.25",
    )
    check(
        "the depth gradient survives turning Blur on",
        abs(grad_blurred - grad_plain) < 0.12,
        f"{grad_plain:.3f} -> {grad_blurred:.3f}",
    )


def test_heal_cache_hit_skips_the_full_frame_prelude():
    """The cache existed to make an unrelated slider tick free; it wasn't.

    A full-frame resize, threshold and np.where ran BEFORE the cache check,
    so every hit still paid them -- the exact cost the cache was added to
    remove. Counted rather than timed: a timing threshold would be flaky.
    """
    import raw_spot_heal as SH
    from raw_spot_heal import HealMask, apply_spot_heal, stamp_heal_brush

    h, w = 600, 900
    img = np.random.RandomState(0).rand(h, w, 3).astype(np.float32) * 0.5
    mask = HealMask.empty(h, w)
    stamp_heal_brush(mask, w // 2, h // 2, 20, 1.0, 0.5)
    mask.touch()

    calls = [0]
    original = SH.resize_mask_to

    def counted(*a, **k):
        calls[0] += 1
        return original(*a, **k)

    SH.resize_mask_to = counted
    try:
        first = apply_spot_heal(img, mask)
        after_cold = calls[0]
        for _ in range(10):
            again = apply_spot_heal(img, mask)
    finally:
        SH.resize_mask_to = original

    check("the cold call does the full-frame work once", after_cold >= 1)
    check(
        "ten cache hits add no full-frame work",
        calls[0] == after_cold,
        f"{calls[0] - after_cold} extra resize(s)",
    )
    check("a cache hit returns the cached buffer", again is first)

    # Painting more must still invalidate.
    stamp_heal_brush(mask, w // 2 + 40, h // 2, 20, 1.0, 0.5)
    mask.touch()
    check("touching the mask invalidates the geometry memo", mask._geom_cache is None)


def test_range_masks_measure_in_display_space():
    """Display-space thresholds on a scene-linear buffer select everything.

    The edit base is scene-linear, where most of a frame sits below 0.2, but
    every threshold is a fraction of a perceptual [0, 1]. Measured on real
    CR3 frames, one centre click covered 99.2% of the photo.
    """
    from raw_range_masks import (
        color_range_alpha,
        luminance_range_alpha,
        sample_luminance_at,
        to_display_rgb,
    )

    # A synthetic scene-linear frame: a realistic spread, mostly dark.
    rng = np.random.RandomState(3)
    lum = np.clip(rng.lognormal(np.log(0.06), 0.8, (256, 256)), 0.0, 1.0)
    linear = np.repeat(lum[:, :, None], 3, axis=2).astype(np.float32)

    display = to_display_rgb(linear)
    check(
        "conversion lifts a dark linear frame into display range",
        float(np.median(display)) > float(np.median(linear)),
        f"median {float(np.median(linear)):.3f} -> {float(np.median(display)):.3f}",
    )
    check("a uint8 buffer is passed through, not double-encoded",
          float(to_display_rgb(np.full((4, 4, 3), 128, np.uint8)).max()) <= 1.0)

    centre = sample_luminance_at(display, 128, 128)
    on_display = float(np.mean(luminance_range_alpha(display, center=centre) > 0.5))
    centre_lin = sample_luminance_at(linear, 128, 128)
    on_linear = float(np.mean(luminance_range_alpha(linear, center=centre_lin) > 0.5))
    check(
        "measuring in display space selects far less than in linear",
        on_display < on_linear,
        f"display {on_display * 100:.0f}% vs linear {on_linear * 100:.0f}%",
    )
    check(
        "a luminance click is a selection, not the whole frame",
        on_display < 0.75,
        f"{on_display * 100:.0f}% of the frame",
    )

    # The "nothing near that sample" refusal must be reachable at all.
    absent = color_range_alpha(display, (1.0, 0.0, 1.0))
    coverage = float(np.count_nonzero(absent > 0.10)) / absent.size
    check(
        "sampling a colour that is not present is refused",
        coverage < 0.002,
        f"coverage {coverage * 100:.3f}%",
    )


def test_mask_dehaze_respects_preview_flag():
    """Export used to force preview=True, so a mask Dehaze looked softer than global."""
    import raw_effects
    from raw_mask_layers import _apply_layer_adjustments

    seen = []
    original = raw_effects.apply_dehaze

    def spy(img, amount, *, preview=False):
        seen.append(preview)
        return original(img, amount, preview=preview)

    raw_effects.apply_dehaze = spy
    try:
        img = np.full((8, 8, 3), 0.4, np.float32)
        _apply_layer_adjustments(img, {"Dehaze": 20.0}, preview=False)
        _apply_layer_adjustments(img, {"Dehaze": 20.0}, preview=True)
    finally:
        raw_effects.apply_dehaze = original
    check("export path asks for full dehaze", seen == [False, True], str(seen))


def main() -> int:
    test_mask_load_is_deterministic_and_never_overwrites_after_truncation()
    test_blur_focus_zero_survives_a_save()
    test_depth_blur_keeps_the_other_adjustments_masked()
    test_heal_cache_hit_skips_the_full_frame_prelude()
    test_range_masks_measure_in_display_space()
    test_mask_dehaze_respects_preview_flag()
    if FAILURES:
        print(f"\nFAILED: {FAILURES}")
        return 1
    print("\nPASS t_review_fixes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
