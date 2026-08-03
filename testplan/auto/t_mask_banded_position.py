#!/usr/bin/env python3
"""Banded edit pipeline must not squash mask alpha into each row strip.

When process_linear_edit_buffer splits a tall frame into horizontal bands,
apply_mask_layers used to call effective_alpha_at(band_h, w) — bilinearly
crushing the whole mask into every strip. Live-drag (often h < 512, no
banding) looked correct; settle/export (banded) jumped the effect vertically.
Anamorphic desqueeze made the lite/settle height split especially common.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def _centroid_y(diff: "np.ndarray") -> float:
    import numpy as np

    w = np.clip(diff, 0, None)
    total = float(w.sum())
    if total < 1e-9:
        return 0.0
    ys = np.arange(w.shape[0], dtype=np.float64)[:, None]
    return float((w * ys).sum() / total) / float(max(1, w.shape[0] - 1))


def main() -> int:
    import numpy as np
    from raw_adjustments import DEFAULT_ADJUSTMENTS, banded_worker_count
    from raw_edit_pipeline import process_linear_edit_buffer
    from raw_mask_layers import MASK_LAYERS_OBJ_KEY, MaskLayer, MaskLayerStack

    # Tall enough to force banding (>= 512 → multiple workers).
    h, w = 640, 960
    assert banded_worker_count(h) > 1, "fixture must engage the banded path"

    base = np.full((h, w, 3), 0.25, dtype=np.float32)
    alpha = np.zeros((h, w), dtype=np.float32)
    # Paint a blob in the UPPER third — squash-into-band would move it.
    alpha[80:160, 400:560] = 1.0
    layer = MaskLayer(name="M1", alpha=alpha)
    layer.adjustments["Exposure2012"] = 2.0
    layer.touch()
    stack = MaskLayerStack([layer])

    adj = dict(DEFAULT_ADJUSTMENTS)
    adj[MASK_LAYERS_OBJ_KEY] = stack
    # Force banding regardless of machine worker heuristics.
    out_banded = process_linear_edit_buffer(base.copy(), adj, preview=True)
    # Single-pass reference: temporarily monkey via full_height identity path
    # by using a short frame that does not band, then upscaling conceptually —
    # simpler: call apply_mask_layers on the post-WB image ourselves.
    from raw_mask_layers import apply_mask_layers
    from raw_edit_pipeline import _apply_wb_tint

    ref = _apply_wb_tint(base.copy(), adj)
    ref = apply_mask_layers(ref, stack)

    diff_b = np.abs(out_banded[..., 0] - base[..., 0])
    diff_r = np.abs(ref[..., 0] - base[..., 0])
    cy_b = _centroid_y(diff_b)
    cy_r = _centroid_y(diff_r)
    check(
        "banded mask effect stays near the painted upper third",
        cy_b < 0.35,
        f"centroid_y={cy_b:.3f}",
    )
    check(
        "banded mask centroid matches full-frame composite",
        abs(cy_b - cy_r) < 0.05,
        f"banded={cy_b:.3f} full={cy_r:.3f}",
    )
    # Pre-fix squash put the peak near y≈0 (top of every band).
    check(
        "effect is not crushed to the top of the frame",
        cy_b > 0.08,
        f"centroid_y={cy_b:.3f}",
    )

    # Direct strip API: squash would light the whole strip; slice keeps a hole.
    strip_h = 80
    row0 = 200  # below the painted blob
    strip = np.full((strip_h, w, 3), 0.25, dtype=np.float32)
    out_strip = apply_mask_layers(
        strip.copy(), stack, row_offset=row0, full_height=h
    )
    check(
        "band below the blob is unchanged when row_offset is correct",
        float(np.max(np.abs(out_strip - strip))) < 1e-5,
        f"maxdiff={float(np.max(np.abs(out_strip - strip))):.6f}",
    )
    out_squash = apply_mask_layers(strip.copy(), stack)  # no offset = old bug
    check(
        "without row_offset a tall mask WOULD light an off-blob strip (documents the bug)",
        float(np.max(np.abs(out_squash - strip))) > 0.01,
        "expected squash to still be wrong without offset",
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
