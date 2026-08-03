#!/usr/bin/env python3
"""Mask Editing / Local Adjustments -- Phase 0 perf spike gate.

docs/FEATURE_FEASIBILITY.md: "Phases 0-1 (data model + pipeline perf) are
the risk and the gate... build Phase 0 as a throwaway-friendly perf spike
-- one brush-alpha mask carrying 3 sliders (exposure/contrast/temperature)
through raw_edit_pipeline, benchmarked against the 80ms throttle on a real
RAW. That single question decides whether this is a ~6-week or a ~4-month
project."

This suite checks:
  1. raw_mask_layers data model correctness (empty/version/fingerprint,
     invert, alpha compositing math).
  2. apply_mask_layers is a no-op for an empty/None stack.
  3. The id(img)-keyed per-layer render cache: an unchanged upstream image
     skips recompute; a changed upstream image (new array, same mask
     version) does NOT reuse a stale render.
  4. The perf budget itself: N masked layers on a half-res edit-base-sized
     buffer, at counts spanning the realistic range up to the Phase 2
     hard-count cap (24), must fit inside the 80ms live-preview throttle
     with headroom for the rest of the pipeline (WB/exposure/denoise/tone
     also run in the same tick).
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np  # noqa: E402

FAILURES = []

# These are regression guards on MEASURED behavior (~20% headroom over this
# machine's numbers), not the 80ms live-preview target -- see the Phase 0
# finding in the module-level comment above main(): bbox-limiting brings
# cost down from "one layer alone blows the budget" (full-frame, no bbox:
# ~120-140ms/layer) to proportional-to-mask-area, but medium/large masks
# (this suite's synthetic masks span ~15-35% of the frame's short side,
# representative of a real sky/subject selection or gradient, not a small
# brush dab) still cost tens of ms EACH on first render or after an edit.
# Multiple large concurrent masks can approach or exceed 80ms on their own
# settle tick even after bbox-limiting -- Phase 1 cannot ship live per-tick
# recompute of every mask; it needs drag-time throttling (recompute only
# the actively-edited layer while dragging, matching the existing
# preview_lite/settle split) plus the stack-level cache validated below,
# which DOES make the dominant "some OTHER slider ticked, masks unchanged"
# case free regardless of layer count or size.
MASK_LAYER_BUDGET_MS = {1: 60.0, 4: 150.0, 8: 300.0, 16: 600.0, 24: 950.0}


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def _make_layer(MaskLayer, h, w, seed, adjustments=None):
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    r = min(h, w) * rng.uniform(0.15, 0.35)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    alpha = np.clip(1.0 - dist / r, 0.0, 1.0).astype(np.float32)
    return MaskLayer(
        alpha,
        adjustments=adjustments
        or {"Exposure2012": rng.uniform(-1.0, 1.0), "Contrast2012": rng.uniform(-30, 30), "Temperature": rng.uniform(-2000, 2000)},
    )


def main() -> int:
    from raw_mask_layers import (
        MaskLayer,
        MaskLayerStack,
        apply_mask_layers,
        erase_mask_layer_brush,
        stamp_mask_layer_brush,
    )

    # --- Brush stamping (mirrors raw_spot_heal.stamp_heal_brush's coverage
    # semantics: max-blend, not additive, and edge-assist is optional) ---
    brush_layer = MaskLayer.empty(100, 100)
    bbox1 = stamp_mask_layer_brush(brush_layer, 50, 50, radius=15, strength=0.6, edge_assist=False)
    check("stamp makes an empty layer non-empty", not brush_layer.is_empty)
    check("stamp bumps version", brush_layer.version == 1)
    check("stamped bbox is a well-formed rect", bbox1[2] > bbox1[0] and bbox1[3] > bbox1[1])
    check("center receives full brush strength", abs(float(brush_layer.alpha[50, 50]) - 0.6) < 1e-3)
    check("far corner untouched", float(brush_layer.alpha[2, 2]) == 0.0)

    v_before = float(brush_layer.alpha[50, 50])
    stamp_mask_layer_brush(brush_layer, 50, 50, radius=15, strength=0.6, edge_assist=False)
    v_after = float(brush_layer.alpha[50, 50])
    check(
        "max-blend: a second overlapping stamp does NOT exceed the brush strength (unlike additive accumulation)",
        abs(v_after - v_before) < 1e-4 and v_after <= 0.6 + 1e-4,
        f"before={v_before:.3f} after={v_after:.3f}",
    )

    stamp_mask_layer_brush(brush_layer, 50, 50, radius=15, strength=1.0, edge_assist=False)
    check("a stronger overlapping stamp raises coverage (max-blend picks the larger value)", float(brush_layer.alpha[50, 50]) > v_after)
    check("alpha stays within [0, 1] after repeated stamping", brush_layer.alpha.min() >= 0.0 and brush_layer.alpha.max() <= 1.0)

    erase_layer = MaskLayer(np.full((100, 100), 0.8, dtype=np.float32))
    erase_mask_layer_brush(erase_layer, 50, 50, radius=15, strength=1.0, edge_assist=False)
    check("erase pulls center coverage toward zero", float(erase_layer.alpha[50, 50]) < 0.1)
    check("erase leaves far corner unchanged", abs(float(erase_layer.alpha[2, 2]) - 0.8) < 1e-4)
    check("erase bumps version", erase_layer.version == 1)

    # Mask erase must ignore edge assist (same contract as dodge/burn erase):
    # otherwise paint that bled across an edge cannot be trimmed away.
    split = np.zeros((100, 200), dtype=np.float32)
    split[:, 100:] = 0.85
    split[:, :100] = 0.15
    painted = MaskLayer.empty(100, 200)
    painted.alpha[:, :] = 0.9
    gated = MaskLayer(painted.alpha.copy())
    plain = MaskLayer(painted.alpha.copy())
    # Centre just on the dark side; radius reaches well into the bright half.
    for _ in range(8):
        erase_mask_layer_brush(
            gated, 90, 50, radius=50, strength=1.0, luminance=split, edge_assist=True
        )
        erase_mask_layer_brush(
            plain, 90, 50, radius=50, strength=1.0, luminance=split, edge_assist=False
        )
    check(
        "mask erase ignores edge assist (flag is inert)",
        np.allclose(gated.alpha, plain.alpha),
        f"maxdiff={float(np.abs(gated.alpha - plain.alpha).max()):.4f}",
    )
    check(
        "mask erase reaches the far side of a hard edge",
        float(gated.alpha[50, 120]) < 0.2,
        f"far={float(gated.alpha[50, 120]):.4f}",
    )

    # Edge-assist gating: a hard luminance boundary should block the brush
    # from bleeding across it, same contract as raw_dodge_burn's own test.
    luminance = np.zeros((100, 100), dtype=np.float32)
    luminance[:, 50:] = 1.0  # hard vertical edge at x=50
    gated_layer = MaskLayer.empty(100, 100)
    plain_layer = MaskLayer.empty(100, 100)
    stamp_mask_layer_brush(gated_layer, 50, 50, radius=25, strength=1.0, luminance=luminance, edge_assist=True)
    stamp_mask_layer_brush(plain_layer, 50, 50, radius=25, strength=1.0, edge_assist=False)
    check(
        "edge-assist keeps paint on the seed's side of a hard luminance boundary",
        float(gated_layer.alpha[50, 30]) < float(plain_layer.alpha[50, 30]),
        f"gated={float(gated_layer.alpha[50, 30]):.4f} plain={float(plain_layer.alpha[50, 30]):.4f}",
    )

    # A painted brush layer must flow through the real compositing path.
    painted_stack = MaskLayerStack(layers=[brush_layer])
    brush_layer.adjustments["Exposure2012"] = 1.0
    brush_layer.touch()
    stamp_mask_layer_brush(brush_layer, 50, 50, radius=15, strength=1.0, edge_assist=False)
    base_img = np.full((100, 100, 3), 0.2, dtype=np.float32)
    composited = apply_mask_layers(base_img, painted_stack)
    check(
        "a brush-painted layer's region is brightened by its Exposure adjustment",
        float(composited[50, 50, 0]) > float(base_img[50, 50, 0]),
    )
    check(
        "a brush-painted layer leaves the untouched far corner alone",
        abs(float(composited[2, 2, 0]) - float(base_img[2, 2, 0])) < 1e-4,
    )

    # --- Data model correctness ---
    empty = MaskLayer.empty(64, 64)
    check("empty layer starts empty", empty.is_empty)
    check("empty layer starts at version 0", empty.version == 0)

    empty.alpha[10:20, 10:20] = 0.5
    empty.touch()
    check("touch() bumps version", empty.version == 1)
    check("painted layer is no longer empty", not empty.is_empty)

    fp_before = empty.fingerprint()
    empty.adjustments["Exposure2012"] = 1.0
    fp_after = empty.fingerprint()
    check("fingerprint changes when adjustments change", fp_before != fp_after)

    layer = MaskLayer(np.full((8, 8), 0.7, dtype=np.float32))
    check("effective_alpha() returns alpha as-is when not inverted", np.allclose(layer.effective_alpha(), 0.7))
    layer.invert = True
    check("effective_alpha() returns 1-alpha when inverted", np.allclose(layer.effective_alpha(), 0.3))

    stack = MaskLayerStack(layers=[MaskLayer.empty(8, 8), MaskLayer.empty(8, 8)])
    check("stack of only-empty layers is empty", stack.is_empty())

    # --- No-op contract ---
    img = np.random.RandomState(0).uniform(0.0, 1.0, (32, 32, 3)).astype(np.float32)
    check("apply_mask_layers(None) is a no-op", apply_mask_layers(img, None) is img)
    check("apply_mask_layers(empty stack) is a no-op", apply_mask_layers(img, stack) is img)

    # --- Compositing math: a full-strength constant layer over its full
    # extent should reproduce the plain (non-masked) adjustment exactly.
    h, w = 16, 16
    base = np.full((h, w, 3), 0.2, dtype=np.float32)
    full_alpha = np.ones((h, w), dtype=np.float32)
    exp_layer = MaskLayer(full_alpha, adjustments={"Exposure2012": 1.0})
    one_layer_stack = MaskLayerStack(layers=[exp_layer])
    out = apply_mask_layers(base.copy(), one_layer_stack)
    expected = base * 2.0
    check(
        "full-coverage exposure layer matches a plain 2x exposure gain",
        np.allclose(out, expected, atol=1e-5),
        f"max diff {float(np.max(np.abs(out - expected))):.6f}",
    )

    half_alpha = np.full((h, w), 0.5, dtype=np.float32)
    half_layer = MaskLayer(half_alpha, adjustments={"Exposure2012": 1.0})
    half_stack = MaskLayerStack(layers=[half_layer])
    out_half = apply_mask_layers(base.copy(), half_stack)
    expected_half = base * 0.5 + (base * 2.0) * 0.5
    check(
        "50% alpha blends linearly between base and adjusted",
        np.allclose(out_half, expected_half, atol=1e-5),
    )

    # --- stack-level id(img)-keyed cache correctness ---
    cache_layer = MaskLayer(np.full((h, w), 1.0, dtype=np.float32), adjustments={"Exposure2012": 0.5})
    cache_stack = MaskLayerStack(layers=[cache_layer])
    img_a = np.full((h, w, 3), 0.1, dtype=np.float32)
    out_a1 = apply_mask_layers(img_a, cache_stack)
    cache_tuple_before = cache_stack._composite_cache
    out_a2 = apply_mask_layers(img_a, cache_stack)  # same object -> cache hit
    check(
        "unchanged upstream array reuses the cached composite (same cache tuple)",
        cache_stack._composite_cache is cache_tuple_before,
    )
    check("cached-path output matches first-call output", np.allclose(out_a1, out_a2))

    img_b = np.full((h, w, 3), 0.9, dtype=np.float32)  # different upstream content, same mask
    out_b = apply_mask_layers(img_b, cache_stack)
    expected_b = img_b * (2.0 ** 0.5)
    check(
        "changed upstream array (new id) does NOT reuse the stale cached composite",
        np.allclose(out_b, expected_b, atol=1e-5),
        f"max diff {float(np.max(np.abs(out_b - expected_b))):.6f}",
    )

    # A repaint (touch()) on the same upstream array must also invalidate,
    # since the stack fingerprint changes even though id(img) is unchanged.
    cache_layer.adjustments["Exposure2012"] = 1.5
    out_a3 = apply_mask_layers(img_a, cache_stack)
    expected_a3 = img_a * (2.0 ** 1.5)
    check(
        "changing a layer's adjustments invalidates the composite cache even with the same img identity",
        np.allclose(out_a3, expected_a3, atol=1e-5),
    )

    # --- PreviewStageCache invalidation through the REAL staged pipeline ---
    # Regression (code review): _stage_key only fingerprinted DodgeBurnMask/
    # HealMask-shaped objects (version+data attrs); a MaskLayerStack fell
    # through to float() -> constant 0.0, so painting a layer or moving its
    # sliders never invalidated the cached pre_tone stage (stale preview).
    # Also covers the serial key (_mask_layers_v1), which was missing from
    # _PRE_TONE_KEYS entirely.
    from mask_layers_xmp import serialize_stack
    from raw_edit_pipeline import PreviewStageCache, process_linear_edit_buffer_staged
    from raw_mask_layers import MASK_LAYERS_KEY, MASK_LAYERS_OBJ_KEY

    stage_base = np.random.RandomState(3).uniform(0.05, 0.6, (80, 120, 3)).astype(np.float32)
    stage_layer = MaskLayer(np.zeros((80, 120), dtype=np.float32), adjustments={"Exposure2012": 1.5})
    stage_layer.alpha[20:60, 30:90] = 1.0
    stage_layer.touch()
    stage_stack = MaskLayerStack(layers=[stage_layer])

    cache = PreviewStageCache()
    adj_live = {"Exposure2012": 0.3, MASK_LAYERS_OBJ_KEY: stage_stack}
    out1 = process_linear_edit_buffer_staged(stage_base, adj_live, cache)
    out2 = process_linear_edit_buffer_staged(stage_base, adj_live, cache)
    check("staged pipeline: unchanged live stack reuses cached output", np.allclose(out1, out2))

    stage_layer.adjustments["Exposure2012"] = -1.5
    stage_layer.touch()
    out3 = process_linear_edit_buffer_staged(stage_base, adj_live, cache)
    check(
        "staged pipeline: editing a live layer's sliders invalidates the cached stage (was stale-served pre-fix)",
        not np.allclose(out2, out3, atol=1e-4),
    )

    stage_layer.alpha[:, :] = 0.0
    stage_layer.alpha[0:10, 0:10] = 1.0
    stage_layer.touch()
    out4 = process_linear_edit_buffer_staged(stage_base, adj_live, cache)
    check(
        "staged pipeline: repainting a live layer's alpha invalidates the cached stage",
        not np.allclose(out3, out4, atol=1e-4),
    )

    # Serial-only form (stack loaded from XMP, no live object yet).
    cache2 = PreviewStageCache()
    serial_a = serialize_stack(MaskLayerStack(layers=[MaskLayer(np.ones((80, 120), dtype=np.float32), adjustments={"Exposure2012": 1.0})]))
    serial_b = serialize_stack(MaskLayerStack(layers=[MaskLayer(np.ones((80, 120), dtype=np.float32), adjustments={"Exposure2012": -1.0})]))
    out_a = process_linear_edit_buffer_staged(stage_base, {"Exposure2012": 0.3, MASK_LAYERS_KEY: serial_a}, cache2)
    out_b = process_linear_edit_buffer_staged(stage_base, {"Exposure2012": 0.3, MASK_LAYERS_KEY: serial_b}, cache2)
    check(
        "staged pipeline: a changed XMP serial (no live object) invalidates the cached stage (key was unregistered pre-fix)",
        not np.allclose(out_a, out_b, atol=1e-4),
    )

    # --- Perf budget: the actual Phase 0 spike question ---
    # Half-res edit-base-sized buffer (matches the resolution masks/dodge-
    # burn actually operate at -- see _dodge_burn_mask_shape's docstring).
    perf_h, perf_w = 2200, 3300
    rng = np.random.RandomState(7)
    perf_img = rng.uniform(0.0, 1.0, (perf_h, perf_w, 3)).astype(np.float32)

    # IMPORTANT: apply_mask_layers caches by id(img). A bare `perf_img.copy()`
    # passed straight into a call with nothing else referencing it gets
    # garbage-collected the moment the call returns -- CPython then commonly
    # reuses that exact freed address for the *next* .copy(), so two
    # logically-independent "cold" calls end up with the SAME id() and
    # silently hit the cache, making the benchmark measure nothing. Keep
    # every copy alive in a list for the module's lifetime so ids are
    # genuinely distinct, and warm on a THIRD distinct array so the warm-up
    # can't accidentally donate its address to the timed call either.
    _keepalive = []

    def _fresh_copy():
        arr = perf_img.copy()
        _keepalive.append(arr)
        return arr

    for n_layers, budget_ms in MASK_LAYER_BUDGET_MS.items():
        layers = [_make_layer(MaskLayer, perf_h, perf_w, seed=i) for i in range(n_layers)]
        perf_stack = MaskLayerStack(layers=layers)

        # Warm-up call (JIT-ish caches, first-touch page faults) excluded
        # from the measured tick, matching how a real render-tick benchmark
        # only cares about the *steady-state* recompute cost -- on a genuinely
        # distinct array so this stack's cache is cold for the timed call.
        apply_mask_layers(_fresh_copy(), perf_stack)

        timed_img = _fresh_copy()
        t0 = time.perf_counter()
        apply_mask_layers(timed_img, perf_stack)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        check(
            f"{n_layers} mask layer(s) at {perf_h}x{perf_w} fit the per-tick budget",
            elapsed_ms < budget_ms,
            f"{elapsed_ms:.1f}ms (budget {budget_ms:.0f}ms)",
        )

    # Unchanged-stack repeat tick (the common case: user drags an unrelated
    # LATER slider, e.g. Sharpness, while mask layers and everything
    # upstream of them are untouched, so `img` is the same array identity
    # as last tick) should be near-free: a single stack-level cache hit
    # instead of recomputing every layer's bbox-limited adjustment+blend.
    layers_16 = [_make_layer(MaskLayer, perf_h, perf_w, seed=i) for i in range(16)]
    repeat_stack = MaskLayerStack(layers=layers_16)
    apply_mask_layers(perf_img, repeat_stack)  # warm
    t0 = time.perf_counter()
    apply_mask_layers(perf_img, repeat_stack)  # same array identity -> cache hit
    repeat_ms = (time.perf_counter() - t0) * 1000.0
    check(
        "unchanged-stack repeat tick (16 layers, cache-hit path) is near-free",
        repeat_ms < 2.0,
        f"{repeat_ms:.1f}ms",
    )

    # Worst case: full-frame-coverage masks (e.g. a linear gradient spanning
    # the whole image, or Dehaze applied to the whole sky) get no benefit
    # from bbox-limiting -- this is the honest upper bound the favorable
    # local-brush numbers above don't show.
    full_budget_ms = {1: 160.0, 4: 600.0, 8: 1150.0}
    for n_layers, budget_ms in full_budget_ms.items():
        full_layers = [
            MaskLayer(np.ones((perf_h, perf_w), dtype=np.float32), adjustments={"Exposure2012": 0.3, "Contrast2012": 10.0, "Temperature": 500.0})
            for _ in range(n_layers)
        ]
        full_stack = MaskLayerStack(layers=full_layers)
        apply_mask_layers(_fresh_copy(), full_stack)  # warm, distinct id
        timed_img = _fresh_copy()
        t0 = time.perf_counter()
        apply_mask_layers(timed_img, full_stack)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        check(
            f"worst case: {n_layers} FULL-FRAME layer(s) at {perf_h}x{perf_w} fit the per-tick budget",
            elapsed_ms < budget_ms,
            f"{elapsed_ms:.1f}ms (budget {budget_ms:.0f}ms)",
        )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
