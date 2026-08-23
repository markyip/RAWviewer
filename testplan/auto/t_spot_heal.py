#!/usr/bin/env python3
"""Spot heal (cv2.inpaint) invariants + undo restore wiring."""
import inspect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from raw_spot_heal import (
        HealMask,
        apply_spot_heal,
        deserialize_mask,
        serialize_mask,
        stamp_heal_brush,
    )

    # 1. Empty mask is a no-op
    img = np.full((64, 80, 3), 0.25, dtype=np.float32)
    img[20:28, 30:38] = 0.9  # bright smudge
    empty = HealMask.empty(64, 80)
    check("empty heal is identity", apply_spot_heal(img, empty) is img)
    check("None heal is identity", apply_spot_heal(img, None) is img)

    # 2. Painting + inpaint reduces the bright blob toward neighbors
    mask = HealMask.empty(64, 80)
    stamp_heal_brush(mask, 34.0, 24.0, 10.0, 1.0)
    check("stamp makes heal mask non-empty", not mask.is_empty)
    out = apply_spot_heal(img, mask)
    check("inpaint returns a new buffer", out is not img)
    before = float(img[24, 34, 0])
    after = float(out[24, 34, 0])
    check(
        "heal pulls smudge toward neighbors",
        after < before - 0.05,
        f"before={before:.3f} after={after:.3f}",
    )
    # Far corner unchanged
    check(
        "heal leaves far corner unchanged",
        np.allclose(out[2, 2], img[2, 2], atol=1e-4),
    )

    # 2b. Cache must not reuse a healed buffer for a different source image
    img2 = (img * 1.5).astype(np.float32)
    out2 = apply_spot_heal(img2, mask)
    check("heal cache is per-source-image", out2 is not out)
    check(
        "heal still works on a brighter source",
        float(out2[24, 34, 0]) < float(img2[24, 34, 0]) - 0.05,
    )
    serial = serialize_mask(mask)
    check("serialize non-empty", bool(serial))
    back = deserialize_mask(serial)
    check("deserialize restores shape", back is not None and back.data.shape == mask.data.shape)
    check(
        "deserialize restores coverage",
        back is not None and np.allclose(back.data, mask.data, atol=1.0 / 255.0),
    )

    # 2c. Cache hit skips the full-frame prelude (resize / threshold / where)
    import raw_spot_heal as SH

    h, w = 120, 160
    img_c = np.random.RandomState(0).rand(h, w, 3).astype(np.float32) * 0.5
    mask_c = HealMask.empty(h, w)
    stamp_heal_brush(mask_c, w // 2, h // 2, 16, 1.0, 0.5)
    calls = [0]
    original_resize = SH.resize_mask_to

    def counted(*a, **k):
        calls[0] += 1
        return original_resize(*a, **k)

    SH.resize_mask_to = counted
    try:
        first = apply_spot_heal(img_c, mask_c)
        after_cold = calls[0]
        again = first
        for _ in range(8):
            again = apply_spot_heal(img_c, mask_c)
    finally:
        SH.resize_mask_to = original_resize
    check("cold heal does the full-frame resize once", after_cold >= 1)
    check(
        "cache hits add no full-frame resize",
        calls[0] == after_cold,
        f"{calls[0] - after_cold} extra resize(s)",
    )
    check("a cache hit returns the cached buffer", again is first)
    check(
        "inpaint cache stores a weakref to the source",
        mask_c._inpaint_cache is not None and len(mask_c._inpaint_cache) >= 3,
    )
    stamp_heal_brush(mask_c, w // 2 + 20, h // 2, 12, 1.0, 0.5)
    check("touching the mask invalidates the geometry memo", mask_c._geom_cache is None)

    # 2d. Small dust stays Telea; a large hole records ns+telea
    small = HealMask.empty(64, 80)
    stamp_heal_brush(small, 34.0, 24.0, 8.0, 1.0)
    apply_spot_heal(img, small)
    check(
        "small dust uses telea",
        small._geom_cache is not None and small._geom_cache[3] == "telea",
    )
    large = HealMask.empty(80, 80)
    large.data[20:60, 20:60] = 1.0
    large.touch()
    img_l = np.full((80, 80, 3), 0.25, dtype=np.float32)
    img_l[20:60, 20:60] = 0.9
    apply_spot_heal(img_l, large)
    check(
        "large hole uses ns+telea",
        large._geom_cache is not None and large._geom_cache[3] == "ns+telea",
        "" if large._geom_cache is None else str(large._geom_cache[3]),
    )

    # 2e. Banded row_offset keeps heal aligned with the painted rows
    H, W = 200, 80
    frame = np.full((H, W, 3), 0.2, dtype=np.float32)
    frame[30:50, 20:40] = 0.9
    band_mask = HealMask.empty(H, W)
    band_mask.data[30:50, 20:40] = 1.0
    band_mask.touch()
    y0, y1 = 20, 70
    band = frame[y0:y1].copy()
    healed_band = apply_spot_heal(band, band_mask, row_offset=y0, full_height=H)
    check(
        "banded heal with row_offset pulls the defect",
        float(healed_band[20, 30, 0]) < 0.9 - 0.05,
        f"got {float(healed_band[20, 30, 0]):.3f}",
    )
    below = frame[80:120].copy()
    below_out = apply_spot_heal(below, band_mask, row_offset=80, full_height=H)
    check(
        "banded heal below the stamp is a no-op",
        below_out is below or np.allclose(below_out, below),
    )

    # 2f. Peak is taken on the ROI, not the whole frame
    img_a = np.full((64, 80, 3), 0.25, dtype=np.float32)
    img_a[20:28, 30:38] = 0.9
    img_b = img_a.copy()
    img_b[2, 2] = 8.0
    roi_mask = HealMask.empty(64, 80)
    stamp_heal_brush(roi_mask, 34.0, 24.0, 10.0, 1.0)
    out_a = apply_spot_heal(img_a, roi_mask)
    out_b = apply_spot_heal(img_b, roi_mask)
    check(
        "heal peak is ROI-local (distant specular does not retarget)",
        abs(float(out_a[24, 34, 0]) - float(out_b[24, 34, 0])) < 0.08,
        f"{float(out_a[24, 34, 0]):.3f} vs {float(out_b[24, 34, 0]):.3f}",
    )

    sig = inspect.signature(stamp_heal_brush)
    check("stamp_heal_brush accepts feather", "feather" in sig.parameters)
    from raw_edit_pipeline import _process_linear_edit_buffer_banded, _process_linear_edit_tail

    check(
        "pipeline tail passes row_offset to heal",
        "row_offset=row_offset" in inspect.getsource(_process_linear_edit_tail),
    )
    check(
        "banded path passes row_offset to tail",
        "row_offset=" in inspect.getsource(_process_linear_edit_buffer_banded),
    )

    # 4. Undo restore helper wiring
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841
    import main as mainmod

    class FakePanel:
        def __init__(self):
            self.heal_calls = []

        def set_spot_heal_mask_present(self, present):
            self.heal_calls.append(present)

        def dodge_burn_show_mask(self):
            return False

        def dodge_burn_mode(self):
            return "heal"

    m = type("M", (), {})()
    m._spot_heal_mask = HealMask.empty(16, 16)
    stamp_heal_brush(m._spot_heal_mask, 8.0, 8.0, 4.0, 1.0)
    m.single_image_adjust_panel = FakePanel()
    m.gpu_view = None
    m._apply_spot_heal_mask_from_adj = (
        mainmod.RAWImageViewer._apply_spot_heal_mask_from_adj.__get__(m)
    )
    m._sync_dodge_burn_mask_overlay = (
        mainmod.RAWImageViewer._sync_dodge_burn_mask_overlay.__get__(m)
    )
    m._active_local_mask_for_overlay = (
        mainmod.RAWImageViewer._active_local_mask_for_overlay.__get__(m)
    )
    m._apply_spot_heal_mask_from_adj({})
    check("undo to empty clears heal mask", m._spot_heal_mask is None)
    check("panel told heal gone", m.single_image_adjust_panel.heal_calls == [False])

    undo_src = inspect.getsource(mainmod.RAWImageViewer._shortcut_activate_undo)
    check(
        "undo restores spot heal before editing_finished",
        "_apply_spot_heal_mask_from_adj" in undo_src
        and undo_src.find("_apply_spot_heal_mask_from_adj")
        < undo_src.find("_on_adjust_panel_editing_finished"),
    )

    # 5. Pipeline stage key includes heal mask
    from raw_edit_pipeline import _PRE_TONE_KEYS
    from raw_spot_heal import MASK_KEY as HEAL_KEY

    check("pre_tone keys include spot heal", HEAL_KEY in _PRE_TONE_KEYS)

    # 6. Reset must strip SpotHealMask from XMP (else gallery stays "edited")
    from raw_adjustments import _OUR_CRS_CHILD_LOCALS, write_xmp_adjustments
    from raw_spot_heal import MASK_KEY as HEAL_MASK_KEY
    import tempfile

    check(
        "SpotHealMask is in our CRS strip set",
        "SpotHealMask" in _OUR_CRS_CHILD_LOCALS,
    )
    with tempfile.TemporaryDirectory() as td:
        xmp = os.path.join(td, "t.xmp")
        write_xmp_adjustments(xmp, {HEAL_MASK_KEY: serial})
        from raw_adjustments import parse_spot_heal_mask_from_xmp, is_default_adjustments
        from raw_adjustments import load_adjustments_from_xmp

        check(
            "heal serial round-trips into XMP",
            bool(parse_spot_heal_mask_from_xmp(xmp)),
        )
        write_xmp_adjustments(xmp, {})  # reset / defaults
        check(
            "reset strips SpotHealMask from XMP",
            not parse_spot_heal_mask_from_xmp(xmp),
        )
        check(
            "reset XMP loads as default (gallery not edited)",
            is_default_adjustments(load_adjustments_from_xmp(xmp)),
        )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
