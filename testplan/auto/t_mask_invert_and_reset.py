"""Invert repaints the overlay, and a mask slider resets like a global one.

Two ways the Masks tab drifted from the rest of the panel.

Invert selects everything outside the mask, and the render honoured it --
the overlay builder already reads effective_alpha_at. But nothing rebuilt
the overlay when the checkbox moved. In-place mask edits ride
preview_changed / editing_finished, and that flow re-renders the photo
only; the overlay is refreshed from _sync_mask_layer_overlay, which fires
on selection, mode, gradient and stroke. So the photo showed the inverted
selection while the tint still sat on the old region.

Reset is the same shape of gap. Every mask row builds an AdjustValueLabel
-- the clickable readout Light and Colour reset through -- so it carried
the pointing-hand cursor and looked live, but its clicked signal was never
connected. Clicking it did nothing at all.
"""

import os
import sys

import numpy as np
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from raw_mask_layers import MaskLayer, MaskLayerStack  # noqa: E402
from rawviewer_ui.adjust_panel import AdjustValueLabel, ImageAdjustPanelWidget  # noqa: E402

H, W = 40, 60
FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


_app = QApplication.instance() or QApplication([])


def _layer(name="Mask 1"):
    m = MaskLayer(np.zeros((H, W), np.float32), name=name)
    m.alpha[5:15, 5:20] = 1.0
    m.touch()
    return m


def _panel(stack):
    p = ImageAdjustPanelWidget()
    p.show()
    p._panel_tabs.set_current(1)
    p.set_mask_layer_stack(stack)
    return p


def main() -> int:
    # --- the model was never the problem: invert flips coverage ---
    m = _layer()
    inside, outside = (10, 10), (30, 40)
    a = m.effective_alpha_at(H, W)
    check("painted region is covered", a[inside] > 0.5, f"{a[inside]:.2f}")
    check("and the rest is not", a[outside] < 0.5, f"{a[outside]:.2f}")

    m.invert = True
    a = m.effective_alpha_at(H, W)
    check("invert drops the painted region", a[inside] < 0.5, f"{a[inside]:.2f}")
    check("and picks up everything else", a[outside] > 0.5, f"{a[outside]:.2f}")

    # --- the overlay is built from the inverted coverage, not the raw alpha ---
    import inspect

    from rawviewer_ui import gpu_image_view

    src = inspect.getsource(gpu_image_view.GpuImageView.update_mask_layer_overlay)
    check(
        "the overlay reads effective_alpha_at",
        "effective_alpha_at" in src and "layer.alpha_at(" not in src,
        "drawing raw alpha would tint the pre-invert region",
    )

    # --- toggling invert has to announce that coverage moved ---
    stack = MaskLayerStack(layers=[_layer()])
    p = _panel(stack)
    p._mask_list.setCurrentItem(p._mask_list.topLevelItem(0))

    seen = []
    p.mask_coverage_changed.connect(lambda: seen.append(1))
    p._on_mask_invert_toggled(True)

    check("the layer is inverted", stack.layers[0].invert is True)
    check(
        "and the overlay is told to rebuild",
        len(seen) == 1,
        f"{len(seen)} signal(s) -- without this the tint stays on the old region",
    )

    p._on_mask_invert_toggled(False)
    check("un-inverting announces it too", len(seen) == 2, f"{len(seen)}")

    # --- while the panel is repopulating, nothing should fire ---
    p._mask_block = True
    p._on_mask_invert_toggled(True)
    p._mask_block = False
    check(
        "a blocked toggle stays silent",
        len(seen) == 2,
        "repopulating the panel must not look like a user edit",
    )

    # --- the host actually listens ---
    import main as mainmod

    lines = open(mainmod.__file__).read().splitlines()
    hit = next(
        (
            i
            for i, ln in enumerate(lines)
            if "mask_coverage_changed.connect" in ln
            and "_on_mask_coverage_changed" in "".join(lines[i : i + 3])
        ),
        None,
    )
    check(
        "main connects it to a handler",
        hit is not None,
        f"main.py:{hit + 1}" if hit is not None else "the signal is emitted but nobody listens",
    )
    handler = inspect.getsource(mainmod.RAWImageViewer._on_mask_coverage_changed)
    check(
        "and that handler redraws the overlay",
        "_sync_mask_layer_overlay" in handler,
    )
    check(
        "after ending any latched solo",
        "_mask_overlay_forced_by_stroke = False" in handler,
        "a stroke's solo would otherwise stop the eye showing any other mask",
    )

    # --- reset: the readout is wired, and it zeroes the LAYER, not just the UI ---
    stack = MaskLayerStack(layers=[_layer()])
    p = _panel(stack)
    p._mask_list.setCurrentItem(p._mask_list.topLevelItem(0))

    key = "Exposure2012"
    label = p._mask_value_labels[key]
    check("the readout is the clickable kind", isinstance(label, AdjustValueLabel))

    p._mask_sliders[key].setValue(150)  # +1.50 EV
    check(
        "the slider writes through to the layer",
        abs(stack.layers[0].adjustments.get(key, 0.0) - 1.5) < 1e-6,
        str(stack.layers[0].adjustments.get(key)),
    )

    before = stack.layers[0].version
    label.clicked.emit()

    check("clicking the readout returns the slider to 0", p._mask_sliders[key].value() == 0)
    check(
        "and clears it on the mask itself",
        abs(stack.layers[0].adjustments.get(key, 0.0)) < 1e-9,
        f"{stack.layers[0].adjustments.get(key)} -- a UI-only reset would leave the edit applied",
    )
    check("the readout follows", label.text().strip() in ("0", "0.00", "+0.00", "0.0"), label.text())
    check(
        "the layer is re-versioned so caches drop",
        stack.layers[0].version != before,
        "a stale composite would keep rendering the old value",
    )

    # --- resetting an already-zero slider is a no-op, not a spurious edit ---
    v = stack.layers[0].version
    label.clicked.emit()
    check("resetting a zeroed slider changes nothing", stack.layers[0].version == v)

    # --- every mask slider is wired, not just the first ---
    unwired = []
    for k, *_ in p._MASK_SLIDER_SPECS:
        sl = p._mask_sliders[k]
        sl.setValue(sl.maximum())
        p._mask_value_labels[k].clicked.emit()
        if sl.value() != 0:
            unwired.append(k)
    check("every mask slider resets", not unwired, str(unwired))

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
