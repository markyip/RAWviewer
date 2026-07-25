#!/usr/bin/env python3
"""Mask Editing / Local Adjustments -- drag-time throttling (Phase 1).

The Phase 0 spike (t_mask_layers.py) found that bbox-limiting + a
stack-level composite cache make the DOMINANT case (some unrelated later
slider ticking, masks untouched) free, but a mask's OWN edit tick still
pays real work proportional to (layer count x mask area) -- multiple
large masks can approach or exceed the 80ms budget on that tick alone.

apply_mask_layers(img, stack, active_index=i) is the fix: while dragging
layer i, everything before it is served from a separate prefix cache, so
the drag doesn't repay layers 0..i-1 every tick. This suite checks:
  1. Correctness: active_index output matches the full (active_index=None)
     composite when nothing is actually mid-drag (same layers/img).
  2. The prefix cache is actually being used (a second drag tick on the
     same img reuses it rather than recomputing).
  3. Perf: dragging the LAST layer of a large stack costs close to ONE
     layer's worth of work, not N layers' worth.
  4. Editing a layer at index i and re-querying active_index=i produces a
     fresh (non-stale) result reflecting the edit.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def _make_layer(MaskLayer, h, w, seed):
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    r = min(h, w) * rng.uniform(0.15, 0.35)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    alpha = np.clip(1.0 - dist / r, 0.0, 1.0).astype(np.float32)
    return MaskLayer(alpha, adjustments={"Exposure2012": rng.uniform(-1.0, 1.0)})


def main() -> int:
    from raw_mask_layers import MaskLayer, MaskLayerStack, apply_mask_layers

    h, w = 400, 600
    layers = [_make_layer(MaskLayer, h, w, seed=i) for i in range(5)]
    stack = MaskLayerStack(layers=layers)
    img = np.random.RandomState(0).uniform(0.0, 1.0, (h, w, 3)).astype(np.float32)

    full = apply_mask_layers(img, stack)
    dragged_last = apply_mask_layers(img, stack, active_index=4)
    check(
        "active_index on the last layer matches the full composite when nothing changed",
        np.allclose(full, dragged_last, atol=1e-5),
    )

    dragged_mid = apply_mask_layers(img, stack, active_index=2)
    check(
        "active_index on a middle layer still produces the full downstream composite",
        np.allclose(full, dragged_mid, atol=1e-5),
    )

    # --- prefix cache reuse ---
    apply_mask_layers(img, stack, active_index=4)  # warm the prefix cache
    prefix_cache_after_first = stack._prefix_cache
    apply_mask_layers(img, stack, active_index=4)
    check(
        "a second drag tick on the same img reuses the prefix cache (same tuple)",
        stack._prefix_cache is prefix_cache_after_first,
    )

    # --- perf: dragging the last of many layers should cost ~1 layer, not N ---
    perf_h, perf_w = 2200, 3300
    many_layers = [_make_layer(MaskLayer, perf_h, perf_w, seed=100 + i) for i in range(16)]
    many_stack = MaskLayerStack(layers=many_layers)
    perf_img = np.random.RandomState(1).uniform(0.0, 1.0, (perf_h, perf_w, 3)).astype(np.float32)

    apply_mask_layers(perf_img, many_stack, active_index=15)  # warm prefix
    t0 = time.perf_counter()
    apply_mask_layers(perf_img, many_stack, active_index=15)  # prefix cache hit + 1 layer recompute
    drag_ms = (time.perf_counter() - t0) * 1000.0

    single_stack = MaskLayerStack(layers=[many_layers[-1]])
    single_img = perf_img.copy()  # distinct id, force a real (uncached) single-layer compute
    t0 = time.perf_counter()
    apply_mask_layers(single_img, single_stack)
    single_layer_ms = (time.perf_counter() - t0) * 1000.0

    check(
        "dragging layer 15 of 16 costs roughly one layer's worth of work, not sixteen",
        drag_ms < single_layer_ms * 3.0 + 5.0,
        f"drag={drag_ms:.1f}ms  one-layer-cold={single_layer_ms:.1f}ms",
    )

    # --- editing the active layer produces a fresh, non-stale result ---
    edit_layers = [_make_layer(MaskLayer, h, w, seed=200 + i) for i in range(3)]
    edit_stack = MaskLayerStack(layers=edit_layers)
    edit_img = np.random.RandomState(2).uniform(0.0, 1.0, (h, w, 3)).astype(np.float32)

    out_before = apply_mask_layers(edit_img, edit_stack, active_index=2)
    edit_layers[2].adjustments["Exposure2012"] = 2.0
    edit_layers[2].touch()
    out_after = apply_mask_layers(edit_img, edit_stack, active_index=2)
    check(
        "editing the actively-dragged layer changes its output on the next tick",
        not np.allclose(out_before, out_after, atol=1e-4),
    )

    # An edit to a layer BEFORE the active index must also invalidate the
    # prefix cache -- otherwise a drag on layer 2 while layer 0 is (say)
    # simultaneously toggled off would show a stale prefix.
    out_before2 = apply_mask_layers(edit_img, edit_stack, active_index=2)
    edit_layers[0].enabled = False
    edit_layers[0].touch()
    out_after2 = apply_mask_layers(edit_img, edit_stack, active_index=2)
    check(
        "disabling a layer before the active index invalidates the prefix cache",
        not np.allclose(out_before2, out_after2, atol=1e-4),
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
