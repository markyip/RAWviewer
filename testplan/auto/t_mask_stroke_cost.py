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

    panel.force_mask_overlay_visible()
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
        "force_mask_overlay_visible" in src,
    )

    # --- a forced-on overlay shows only the mask being edited ---
    # Turning the overlay on to give a stroke something to show must not
    # also bring back every mask the user had put away.
    def L(y):
        lyr = MaskLayer(np.zeros((200, 300), np.float32))
        lyr.alpha[y:y + 40, 20:120] = 1.0
        lyr.touch()
        return lyr

    la, lb = L(10), L(100)
    v2 = GpuImageView()

    def tints(**kw):
        v2.update_mask_layer_overlay([la, lb], 0, **kw)
        img = v2._mask_item.pixmap().toImage()
        hh, ww = v2._mask_overlay_shape
        return (
            img.pixelColor(int(60 * ww / 300), int(25 * hh / 200)).alpha(),
            img.pixelColor(int(60 * ww / 300), int(115 * hh / 200)).alpha(),
        )

    both = tints()
    check("normally every mask is tinted", both[0] > 0 and both[1] > 0, str(both))

    solo0 = tints(solo_index=0)
    check(
        "solo draws only the mask being edited",
        solo0[0] > 0 and solo0[1] == 0,
        f"{solo0} -- the other mask must not reappear because you painted",
    )
    solo1 = tints(solo_index=1)
    check("and follows which mask that is", solo1[0] == 0 and solo1[1] > 0, str(solo1))

    lb.overlay_hidden = True
    eye = tints()
    check(
        "the per-mask eye is still independent of solo",
        eye[0] > 0 and eye[1] == 0,
        str(eye),
    )
    lb.overlay_hidden = False

    check(
        "the stroke marks the overlay as forced",
        "_mask_overlay_forced_by_stroke = True" in src,
    )
    toggled = inspect.getsource(mainmod.RAWImageViewer._on_dodge_burn_mask_toggled)
    check(
        "and a deliberate M press ends the solo",
        "_mask_overlay_forced_by_stroke = False" in toggled,
        "once the user says anything about the overlay, soloing no longer applies",
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
