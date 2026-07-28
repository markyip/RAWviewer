"""Painting a mask does not re-render the photo on every stamp.

While the brush is down the user is aiming coverage, and the coloured
overlay is what shows coverage. The edit underneath it was being re-rendered
every 120ms anyway -- and masks composite inside the pre_tone cache stage
with denoise after them, so each stamp invalidated WB, exposure, dodge/burn,
heal and denoise together: ~1360ms at a 2200x3300 base, 454ms of it chroma
denoise, requested eight times a second. The renders could not keep up with
the brush.

Now the stroke updates the overlay only, and the photo renders once on
release. The overlay is also capped mid-stroke, because once it is the only
per-stamp work left its own 186ms is the thing that cannot keep up. Full
resolution returns on release, so a capped buffer is never what you are left
looking at -- it would show when zoomed in.
"""

import inspect
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


_app = QApplication.instance() or QApplication([])


def main() -> int:
    import main as mainmod
    from raw_mask_layers import MaskLayer
    from rawviewer_ui.gpu_image_view import GpuImageView

    # --- the stroke path must not ask for a pipeline render mid-stroke ---
    src = inspect.getsource(mainmod.RAWImageViewer._on_mask_layer_stroke)
    body = src[src.index("if is_end:"):]
    check(
        "the photo renders only when the stroke ends",
        "_apply_adjust_panel_preview" not in src,
        "a per-stamp pipeline render cannot keep up with a brush",
    )
    check(
        "and it does render on release",
        "_on_adjust_panel_editing_finished" in body,
        "the edit still has to appear when you let go",
    )

    # --- the overlay is capped while painting, full at the end ---
    sync = inspect.getsource(mainmod.RAWImageViewer._sync_mask_layer_overlay)
    check("the overlay sync accepts a cap", "max_dim" in sync)
    check(
        "the stroke passes one, and clears it on release",
        "max_dim=None if is_end" in src,
        "a capped buffer must not be what you are left looking at",
    )

    cap = mainmod.RAWImageViewer._MASK_OVERLAY_STROKE_MAX_DIM
    check("the cap is around display resolution", 800 <= cap <= 2048, str(cap))

    # --- capping is honoured, and still marks the right region ---
    H, W = 1100, 1650
    m = MaskLayer(np.zeros((H, W), np.float32))
    m.alpha[200:600, 300:900] = 1.0
    m.touch()

    view = GpuImageView()
    view.update_mask_layer_overlay([m], 0)
    full_shape = view._mask_overlay_shape
    check("uncapped builds at mask resolution", full_shape == (H, W), str(full_shape))

    view.update_mask_layer_overlay([m], 0, max_dim=400)
    small = view._mask_overlay_shape
    check(
        "a cap shrinks the buffer",
        small is not None and max(small) <= 400,
        str(small),
    )
    check(
        "and keeps the aspect ratio",
        small is not None and abs((small[1] / small[0]) - (W / H)) < 0.02,
        f"{small} vs {(H, W)} -- a squashed overlay would sit off the mask",
    )

    # The overlay must still cover the painted region, not drift off it.
    view.update_mask_layer_overlay([m], 0, max_dim=400)
    px = view._mask_item.pixmap().toImage()
    sh, sw = small
    inside = px.pixelColor(int(0.36 * sw), int(0.36 * sh)).alpha()   # in the paint
    outside = px.pixelColor(int(0.95 * sw), int(0.95 * sh)).alpha()  # well outside
    check("the capped overlay still tints the mask", inside > 0, str(inside))
    check("and leaves the rest clear", outside == 0, str(outside))

    # --- a cap larger than the buffer changes nothing ---
    view.update_mask_layer_overlay([m], 0, max_dim=99_999)
    check(
        "a cap above the mask size is a no-op",
        view._mask_overlay_shape == (H, W),
        str(view._mask_overlay_shape),
    )

    # --- and it is actually faster, which is the whole point ---
    big = MaskLayer(np.zeros((2200, 3300), np.float32))
    big.alpha[300:1300, 300:2000] = 1.0
    big.touch()

    def timed(fn, n=3):
        fn()
        best = 1e9
        for _ in range(n):
            s = time.perf_counter()
            fn()
            best = min(best, (time.perf_counter() - s) * 1000)
        return best

    full_ms = timed(lambda: view.update_mask_layer_overlay([big], 0))
    cap_ms = timed(lambda: view.update_mask_layer_overlay([big], 0, max_dim=cap))
    check(
        "capping is materially cheaper at a real edit-base size",
        cap_ms < full_ms * 0.5,
        f"{cap_ms:.0f} ms capped vs {full_ms:.0f} ms full",
    )
    check(
        "and lands inside a frame budget",
        cap_ms < 50.0,
        f"{cap_ms:.0f} ms -- the brush polls faster than this if it cannot",
    )

    # --- a stroke must never paint into nothing ---
    from raw_mask_layers import MaskLayerStack
    from rawviewer_ui.adjust_panel import ImageAdjustPanelWidget

    panel = ImageAdjustPanelWidget()
    panel.show()
    panel._panel_tabs.set_current(1)
    layer = MaskLayer(np.zeros((40, 60), np.float32))
    layer.alpha[5:15, 5:20] = 1.0
    layer.touch()
    panel.set_mask_layer_stack(MaskLayerStack(layers=[layer]))

    panel._mask_paint_btn.setChecked(True)
    check("arming Add shows the overlay", panel.dodge_burn_show_mask())

    panel.toggle_dodge_burn_show_mask()
    check("M can still switch it off", not panel.dodge_burn_show_mask())
    check("and that is recorded as a user choice", panel._mask_user_hidden is True)

    panel._mask_overlay_for_editing_coverage()
    check(
        "but a stroke turns it back on",
        panel.dodge_burn_show_mask(),
        "the overlay is the only feedback a stroke has now, so painting with "
        "it off would show nothing at all until release",
    )
    check(
        "without overwriting what the user chose",
        panel._mask_user_hidden is True,
        "M is still their setting; the stroke is a temporary override",
    )
    check(
        "and the stroke path actually calls it",
        "_mask_overlay_for_editing_coverage" in src,
        "rule 2: editing coverage shows the mask being edited",
    )

    # --- one mask's overlay at a time, expressed in overlay_hidden ---
    # This replaced a separate "solo" flag. Exclusivity belongs in the same
    # state the eye and the selection already write, or the two disagree --
    # which is exactly what left a latched solo blocking every eye click.
    from raw_mask_layers import MaskLayerStack
    from rawviewer_ui.adjust_panel import ImageAdjustPanelWidget

    panel2 = ImageAdjustPanelWidget()
    panel2.show()
    panel2._panel_tabs.set_current(1)

    def _lyr(y):
        lyr = MaskLayer(np.zeros((200, 300), np.float32))
        lyr.alpha[y:y + 40, 20:120] = 1.0
        lyr.touch()
        return lyr

    three = MaskLayerStack(layers=[_lyr(10), _lyr(60), _lyr(120)])
    panel2.set_mask_layer_stack(three)

    def shown():
        return [not l.overlay_hidden for l in three.layers]

    check(
        "a freshly loaded stack shows exactly one",
        shown().count(True) == 1,
        f"{shown()} -- overlay_hidden defaults to False, so without this "
        "every mask starts marked visible",
    )

    panel2.show_only_mask_overlay(2)
    check("showing one hides the rest", shown() == [False, False, True], str(shown()))

    check(
        "the overlay builder no longer takes a solo",
        "solo_index" not in inspect.getsource(GpuImageView.update_mask_layer_overlay),
        "a second source of truth for the same thing",
    )
    check(
        "and neither does the host",
        "_mask_overlay_forced_by_stroke" not in inspect.getsource(mainmod),
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
