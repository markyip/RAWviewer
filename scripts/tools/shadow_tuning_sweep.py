#!/usr/bin/env python3
"""Parameter-sweep variant of shadow_tuning_render.py: re-renders one crop
under several candidate raw_pv2012 constant overrides, for direct visual
A/B comparison. Does not modify the source file -- patches the module's
globals in-process only.

Usage: pixi run python3 scripts/tools/shadow_tuning_sweep.py <file.CR3> <out_dir>
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

CROP_SIZE = 420


def darkest_crop_box(lum: np.ndarray, size: int) -> tuple[int, int]:
    h, w = lum.shape
    size = min(size, h, w)
    block = max(16, size // 4)
    best_score = None
    best_yx = (0, 0)
    for y in range(0, h - size + 1, block):
        for x in range(0, w - size + 1, block):
            score = float(lum[y : y + size, x : x + size].mean())
            if best_score is None or score < best_score:
                best_score = score
                best_yx = (y, x)
    return best_yx


# (label, {module_attr: value}) -- applied to raw_pv2012 before each render.
VARIANTS = [
    ("baseline_cap8_damp85_div3", {}),  # current committed constants
    ("damp60_div3", {"_DAMP_MAX_OVERRIDE": 0.60}),
    ("damp40_div3", {"_DAMP_MAX_OVERRIDE": 0.40}),
    ("damp60_div5", {"_DAMP_MAX_OVERRIDE": 0.60, "_LIFT_FRAC_DIV_OVERRIDE": 5.0}),
    ("cap8_anchor_wide", {"_ANCHOR_HI_OVERRIDE": 0.03}),
]


def main() -> None:
    path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/shadow_sweep"
    os.makedirs(out_dir, exist_ok=True)

    from unified_image_processor import UnifiedImageProcessor
    from raw_adjustments import DEFAULT_ADJUSTMENTS
    import raw_pv2012 as pv

    proc = UnifiedImageProcessor()
    stem = os.path.splitext(os.path.basename(path))[0]
    base = proc.decode_raw_edit_base(path, use_full_resolution=False)
    lum = pv._luminance(base)
    y0, x0 = darkest_crop_box(lum, CROP_SIZE)
    size = min(CROP_SIZE, base.shape[0], base.shape[1])
    crop = base[y0 : y0 + size, x0 : x0 + size]

    orig_apply = pv.apply_pv2012_tone_rgb

    def patched_apply(img, adj, *, damp_max=0.85, lift_div=3.0, anchor_hi=0.012):
        # Re-implements apply_pv2012_tone_rgb's body with the damp/anchor
        # knobs parameterized, so we can sweep without editing the module.
        lum_ = pv._luminance(img)
        y0_ = pv._scene_to_perceptual(lum_)
        from raw_tone_curve import apply_tone_curve_perceptual

        y_curve = apply_tone_curve_perceptual(y0_, adj)
        y1 = pv._apply_pv2012_perceptual(
            y_curve,
            contrast=float(adj.get("Contrast2012", 0.0)),
            highlights=float(adj.get("Highlights2012", 0.0)),
            shadows=float(adj.get("Shadows2012", 0.0)),
            whites=float(adj.get("Whites2012", 0.0)),
            blacks=float(adj.get("Blacks2012", 0.0)),
        )
        ratio = (y1 + pv._RATIO_EPS) / (y0_ + pv._RATIO_EPS)
        ratio = np.clip(ratio, pv._MIN_TONE_RATIO, pv._MAX_TONE_RATIO)
        t = np.clip((y0_ - 0.001) / (anchor_hi - 0.001), 0.0, 1.0)
        anchor = t * t * (3.0 - 2.0 * t)
        ratio = np.where(ratio > 1.0, 1.0 + (ratio - 1.0) * anchor, ratio)
        out = img * ratio[..., np.newaxis]
        if np.any(ratio > 1.0 + 1e-6):
            sw = pv._region_weight_shadows(y_curve)
            lift_frac = np.clip((ratio - 1.0) / lift_div, 0.0, 1.0)
            damp = 1.0 - (sw * damp_max * lift_frac)[..., np.newaxis]
            luma = (lum_ * ratio)[..., np.newaxis]
            chroma = out - luma
            out = luma + chroma * damp
        return np.clip(out, 0.0, None)

    import raw_edit_pipeline as rep
    from raw_edit_pipeline import process_linear_edit_buffer, linear_to_display_uint8

    for label, overrides in VARIANTS:
        damp_max = overrides.get("_DAMP_MAX_OVERRIDE", 0.85)
        lift_div = overrides.get("_LIFT_FRAC_DIV_OVERRIDE", 3.0)
        anchor_hi = overrides.get("_ANCHOR_HI_OVERRIDE", 0.012)

        # raw_edit_pipeline did `from raw_pv2012 import apply_pv2012_tone_rgb`,
        # binding its own name at import time -- patching pv.apply_pv2012_tone_rgb
        # would not affect it, so patch raw_edit_pipeline's own reference instead.
        rep.apply_pv2012_tone_rgb = lambda img, adj, dm=damp_max, ld=lift_div, ah=anchor_hi: patched_apply(
            img, adj, damp_max=dm, lift_div=ld, anchor_hi=ah
        )
        try:
            merged = dict(DEFAULT_ADJUSTMENTS)
            merged["Shadows2012"] = 100.0
            processed = process_linear_edit_buffer(crop, merged, preview=True)
            out = linear_to_display_uint8(processed, merged)
            out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            out_path = os.path.join(out_dir, f"{stem}__{label}.png")
            cv2.imwrite(out_path, out_bgr)
            print("wrote", out_path)
        finally:
            rep.apply_pv2012_tone_rgb = orig_apply


if __name__ == "__main__":
    main()
