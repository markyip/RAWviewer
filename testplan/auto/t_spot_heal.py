#!/usr/bin/env python3
"""Spot heal (cv2.inpaint) invariants + undo restore wiring."""
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

    # 4. Undo restore helper wiring
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841
    import inspect
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
