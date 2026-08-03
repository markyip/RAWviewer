#!/usr/bin/env python3
"""Anamorphic desqueeze must keep local masks in the post-geometry frame.

Covers the P0 gaps from the anamorphic+mask audit:
  - uncropped mask shape still follows AnamorphicRatio
  - preview_lite applies D&B after geometry (not pre-warp)
  - banded spot heal uses row_offset / full_height
  - instant-transform source is not cached with anamorphic baked in
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def test_uncropped_mask_shape_follows_anamorphic():
    """Crop-overlay uncropped + anamorphic must not fall back to base (H,W)."""
    import numpy as np
    from PyQt6.QtWidgets import QApplication

    _ = QApplication.instance() or QApplication(sys.argv)

    class _Fake:
        def __init__(self):
            self._adjust_preview_base_rgb = np.zeros((100, 100, 3), np.float32)
            self._crop_preview_uncropped = True
            self.single_image_adjust_panel = None
            self._dodge_burn_mask_shape_cache_key = None
            self._dodge_burn_mask_shape_cached = None

        def _wb_linear_sample_buffer(self, adj):
            # Desqueezed 1.5x width with crop insets zeroed.
            return np.zeros((100, 150, 3), np.float32)

    import main as mainmod

    fake = _Fake()
    shape_fn = mainmod.RAWImageViewer._dodge_burn_mask_shape.__get__(fake)

    class _Panel:
        def get_adjustments(self):
            return {"AnamorphicRatio": 1.5}

    fake.single_image_adjust_panel = _Panel()
    shape = shape_fn()
    check(
        "uncropped+anamorphic mask shape uses post-geo sample",
        shape == (100, 150),
        f"got {shape}",
    )


def test_preview_lite_db_after_geometry():
    """Live-drag D&B must land on the desqueezed frame, not pre-warp."""
    import numpy as np
    from raw_adjustments import DEFAULT_ADJUSTMENTS
    from raw_dodge_burn import DEFAULT_STRENGTH, DodgeBurnMask, MASK_OBJ_KEY, STRENGTH_KEY
    from raw_edit_pipeline import PreviewStageCache, process_linear_edit_buffer_staged

    h, w = 80, 80
    base = np.full((h, w, 3), 0.25, dtype=np.float32)
    # Paint a vertical strip on the LEFT of the post-geo (desqueezed) frame.
    post_w = int(round(w * 1.5))
    mask = DodgeBurnMask.empty(h, post_w)
    mask.data[:, : post_w // 3] = 1.0
    mask.touch()

    adj = dict(DEFAULT_ADJUSTMENTS)
    adj["AnamorphicRatio"] = 1.5
    adj[MASK_OBJ_KEY] = mask
    adj[STRENGTH_KEY] = DEFAULT_STRENGTH

    cache = PreviewStageCache()
    out = process_linear_edit_buffer_staged(
        base, adj, cache, preview=True, preview_lite=True
    )
    check(
        "lite+anamorphic+D&B returns an image",
        out is not None and getattr(out, "size", 0) > 0,
    )
    if out is None or getattr(out, "size", 0) == 0:
        return
    oh, ow = out.shape[:2]
    check(
        "lite output is wider after anamorphic",
        ow > w,
        f"{ow}x{oh} vs base {w}x{h}",
    )
    bright = out.astype(np.float32).mean(axis=2)
    left = float(bright[:, : max(1, ow // 3)].mean())
    right = float(bright[:, 2 * ow // 3 :].mean())
    # Float scene-linear or display uint8 — require a clear left > right gap.
    gap = left - right
    check(
        "D&B lift stays on the left of the desqueezed frame",
        gap > 0.05,
        f"left={left:.2f} right={right:.2f} gap={gap:.2f}",
    )


def test_heal_banded_row_offset():
    import numpy as np
    from raw_spot_heal import HealMask, apply_spot_heal

    h, w = 640, 320
    img = np.full((h, w, 3), 0.3, dtype=np.float32)
    # Distinct defect in the upper third.
    img[100:140, 140:180] = 0.95
    mask = HealMask.empty(h, w)
    mask.data[100:140, 140:180] = 1.0
    mask.touch()

    y0, band_h = 80, 100
    band = img[y0 : y0 + band_h].copy()
    healed = apply_spot_heal(band, mask, row_offset=y0, full_height=h)
    check(
        "banded heal modifies the defect strip",
        float(np.max(np.abs(healed - band))) > 0.01,
        f"maxdiff={float(np.max(np.abs(healed - band))):.4f}",
    )
    below = img[400:480].copy()
    below_out = apply_spot_heal(below, mask, row_offset=400, full_height=h)
    check(
        "banded heal below the defect is unchanged",
        float(np.max(np.abs(below_out - below))) < 1e-5,
        f"maxdiff={float(np.max(np.abs(below_out - below))):.6f}",
    )
    # Direct API: coverage slice for a band that contains the defect must
    # be non-empty; a band below must be empty (documents row_offset).
    from raw_spot_heal import resize_mask_to

    full = resize_mask_to(mask, h, w)
    check(
        "row_offset slice for the defect band has coverage",
        float(full[y0 : y0 + band_h].max()) > 0.5,
    )
    check(
        "row_offset slice for a below band has no coverage",
        float(full[400:480].max()) < 1e-5,
    )


def test_instant_transform_source_rejects_anamorphic():
    main_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "main.py"
    )
    src = open(main_path, encoding="utf-8").read()
    check(
        "transform source cache checks AnamorphicRatio",
        'adj_used.get("AnamorphicRatio"' in src,
    )
    check(
        "instant warp strips AnamorphicRatio to 1.0",
        'geo_adj["AnamorphicRatio"] = 1.0' in src,
    )
    check(
        "instant warp strips Distortion to 0.0",
        'geo_adj["Distortion"] = 0.0' in src,
    )


def test_ai_mask_source_uses_post_geometry():
    main_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "main.py"
    )
    src = open(main_path, encoding="utf-8").read()
    start = src.find("def _ai_mask_source_rgb")
    check("_ai_mask_source_rgb exists", start > 0)
    if start > 0:
        chunk = src[start : start + 800]
        check(
            "AI mask source prefers _wb_linear_sample_buffer",
            "_wb_linear_sample_buffer" in chunk,
        )


def test_preview_lite_source_order():
    pipe_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "raw_edit_pipeline.py"
    )
    src = open(pipe_path, encoding="utf-8").read()
    lite = src.find("if preview_lite:")
    check("preview_lite block exists", lite > 0)
    if lite > 0:
        block = src[lite : lite + 2500]
        geo = block.find("apply_geometry(")
        db = block.find("apply_dodge_burn(")
        heal = block.find("apply_spot_heal(")
        check(
            "lite apply_dodge_burn is after apply_geometry",
            geo >= 0 and db > geo,
            f"geo={geo} db={db}",
        )
        check(
            "lite apply_spot_heal is after apply_geometry",
            geo >= 0 and heal > geo,
            f"geo={geo} heal={heal}",
        )
        pre_assign = block.find('cache.stage_out["pre_geom"]')
        check(
            "pre_geom cache has no D&B before it",
            pre_assign > 0 and db > pre_assign,
            f"pre={pre_assign} db={db}",
        )


def main() -> int:
    test_uncropped_mask_shape_follows_anamorphic()
    test_preview_lite_db_after_geometry()
    test_heal_banded_row_offset()
    test_instant_transform_source_rejects_anamorphic()
    test_ai_mask_source_uses_post_geometry()
    test_preview_lite_source_order()
    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
