#!/usr/bin/env python3
"""Masks tab: invert scope + UI contract -- headless.

The bug: an inverted layer covers everything OUTSIDE its painted region, but
the compositor limited work to the painted alpha's bbox -- the one place
effective alpha is near zero. Invert therefore did almost nothing beyond a
small patch while the mask overlay showed it correctly.

The UI items are asserted because each one is a deliberate removal, and a
removal is exactly what a later "tidy-up" re-adds by accident.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import numpy as np  # noqa: E402
from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from raw_mask_layers import MaskLayer, MaskLayerStack, apply_mask_layers  # noqa: E402
from rawviewer_ui.adjust_panel import (  # noqa: E402
    ImageAdjustPanelWidget,
    reset_section_expanded_session,
)

H = W = 200


def _painted_layer(**kw):
    layer = MaskLayer.empty(H, W, adjustments={"Exposure2012": 1.5}, **kw)
    layer.alpha[90:110, 90:110] = 1.0  # small central square
    layer.touch()
    return layer


def test_invert_applies_outside_the_painted_region():
    img = np.full((H, W, 3), 0.30, np.float32)
    layer = _painted_layer()

    layer.invert = False
    layer.touch()
    normal = apply_mask_layers(img.copy(), MaskLayerStack([layer]))
    assert normal[100, 100, 0] > 0.6, "un-inverted mask did not lift the painted square"
    assert abs(float(normal[10, 10, 0]) - 0.30) < 1e-3, "un-inverted mask leaked outside"

    layer.invert = True
    layer.touch()
    inverted = apply_mask_layers(img.copy(), MaskLayerStack([layer]))
    assert inverted[10, 10, 0] > 0.6, (
        f"inverted mask did not apply at the far corner (got {inverted[10, 10, 0]:.3f})"
    )
    assert abs(float(inverted[100, 100, 0]) - 0.30) < 1e-3, (
        "inverted mask still applied inside the painted region"
    )
    print("  OK   invert applies outside the painted region, across the frame")


def test_effective_bbox_is_full_frame_when_inverted():
    layer = _painted_layer()
    layer.invert = False
    layer.touch()
    assert layer.effective_bbox() == layer.bbox(), "un-inverted should use the tight bbox"

    layer.invert = True
    layer.touch()
    assert layer.effective_bbox() == (0, H, 0, W), (
        f"inverted effective_bbox should be the whole frame, got {layer.effective_bbox()}"
    )
    # bbox() itself stays the editable region -- the overlay and UI want that.
    assert layer.bbox() != (0, H, 0, W), "bbox() should still report the painted extent"
    print("  OK   effective_bbox spans the frame when inverted; bbox stays tight")


def test_disabled_inverted_layer_applies_nothing():
    """Invert must not resurrect a layer the user switched off."""
    layer = _painted_layer()
    layer.invert = True
    layer.enabled = False
    layer.touch()
    assert layer.effective_bbox() is None
    img = np.full((H, W, 3), 0.30, np.float32)
    out = apply_mask_layers(img.copy(), MaskLayerStack([layer]))
    assert np.allclose(out, img), "a disabled inverted layer still applied"
    print("  OK   a disabled inverted layer applies nothing")


# ------------------------------------------------------------------ panel UI


def _panel():
    reset_section_expanded_session()
    p = ImageAdjustPanelWidget()
    p.show()
    p._panel_tabs.set_current(1)  # Masks
    return p


def test_parameters_hidden_until_a_mask_exists():
    p = _panel()
    assert not p._mask_params_wrap.isVisible(), "parameter sliders shown with no mask"
    assert not p._mask_list.isVisible(), "empty mask list box shown"
    assert p._mask_empty_hint.isVisible(), "no guidance shown in the empty state"

    stack = MaskLayerStack([_painted_layer(name="Mask 1")])
    p.set_mask_layer_stack(stack)
    assert p._mask_params_wrap.isVisible(), "parameters stayed hidden with a mask present"
    assert p._mask_list.isVisible()
    assert not p._mask_empty_hint.isVisible()
    print("  OK   parameters appear only once a mask exists")


def test_export_button_is_not_on_the_masks_tab():
    p = _panel()
    assert not p._export_btn.isVisible(), "Export is visible while masking"
    p._panel_tabs.set_current(0)
    assert p._export_btn.isVisible(), "Export vanished from the Global tab"
    print("  OK   Export lives on Global only")


def test_duplicate_button_is_gone():
    p = _panel()
    assert getattr(p, "_mask_dup_btn", None) is None, "Duplicate button came back"
    assert getattr(p, "_mask_del_btn", None) is not None, "Delete button missing"
    print("  OK   Duplicate removed, Delete kept")


def test_rows_are_selectable_not_checkable():
    p = _panel()
    p.set_mask_layer_stack(MaskLayerStack([
        _painted_layer(name="Mask 1"), _painted_layer(name="Mask 2"),
    ]))
    for row in range(p._mask_list.count()):
        item = p._mask_list.item(row)
        assert not (item.flags() & Qt.ItemFlag.ItemIsUserCheckable), (
            f"row {row} is still checkable"
        )
        assert item.flags() & Qt.ItemFlag.ItemIsSelectable, f"row {row} is not selectable"
    p._mask_list.setCurrentRow(1)
    assert p.active_mask_index() == 1, "clicking a row did not select it"
    print("  OK   rows select on click, with no check box")


def test_select_button_is_named_for_what_it_does():
    p = _panel()
    assert p._mask_ai_click_btn.text() == "Select", (
        f'AI point-prompt button is labelled "{p._mask_ai_click_btn.text()}"'
    )
    print("  OK   the point-prompt tool is called Select, not Click")


def test_empty_ai_mask_is_rejected():
    """A skyless photo yields a near-empty alpha, not a failure."""
    src = (
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "main.py")
    )
    text = open(src, encoding="utf-8").read()
    assert "_AI_MASK_MIN_COVERAGE" in text, "no coverage floor for one-shot AI masks"
    assert "no mask added" in text, "empty AI result is still added as a layer"
    print("  OK   an empty AI result is refused with a message")


def main() -> int:
    test_invert_applies_outside_the_painted_region()
    test_effective_bbox_is_full_frame_when_inverted()
    test_disabled_inverted_layer_applies_nothing()
    test_parameters_hidden_until_a_mask_exists()
    test_export_button_is_not_on_the_masks_tab()
    test_duplicate_button_is_gone()
    test_rows_are_selectable_not_checkable()
    test_select_button_is_named_for_what_it_does()
    test_empty_ai_mask_is_rejected()
    print("\nPASS t_masks_ui_cleanup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
