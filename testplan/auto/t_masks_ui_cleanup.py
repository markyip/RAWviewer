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
    label = p._mask_ai_click_btn.text()
    # "Click" named the input; "Select" read as a synonym for Subject. The
    # label has to say that YOU choose the thing.
    assert label not in ("Click", "Select"), f"point-prompt button reverted to {label!r}"
    assert "point" in label.lower(), f"point-prompt button is labelled {label!r}"
    print(f"  OK   the point-prompt tool is called {label!r}, distinct from Subject")


def test_empty_ai_mask_is_rejected():
    """A skyless photo yields a near-empty alpha, not a failure."""
    src = (
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "main.py")
    )
    text = open(src, encoding="utf-8").read()
    assert "_AI_MASK_MIN_COVERAGE" in text, "no coverage floor for one-shot AI masks"
    assert "no mask added" in text, "empty AI result is still added as a layer"
    print("  OK   an empty AI result is refused with a message")


def test_paint_creates_its_own_mask_like_every_other_tool():
    """Paint used to be the one coverage tool that needed Add Mask first."""
    p = _panel()
    assert p.active_mask_index() is None

    created = []
    p.mask_add_requested.connect(lambda: created.append(True))
    p._mask_paint_btn.setChecked(True)
    assert created, "arming Paint with no mask did not ask for one"

    # With no host connected nothing is actually created, so Paint must not
    # leave itself armed over a mask that does not exist.
    assert not p._mask_paint_btn.isChecked(), (
        "Paint stayed armed with no mask to paint into"
    )
    print("  OK   Paint asks for a mask instead of silently doing nothing")


def test_new_mask_button_is_hidden_in_the_empty_state():
    """With no masks, every tool creates one -- a New Mask button is a step."""
    p = _panel()
    assert not p._mask_add_btn.isVisible(), "New Mask shown before any mask exists"
    p.set_mask_layer_stack(MaskLayerStack([_painted_layer(name="Mask 1")]))
    assert p._mask_add_btn.isVisible(), "New Mask hidden when a second one is possible"
    assert p._mask_add_btn.text() == "New Mask", p._mask_add_btn.text()
    print("  OK   New Mask appears only once it means 'another one'")


# --------------------------------------------------------------- mask overlay


def _view():
    from rawviewer_ui.gpu_image_view import GpuImageView

    v = GpuImageView()
    v._mask_overlay_wanted = True
    v._img_w = v._img_h = W
    return v


def _near(got, want, tol=2):
    return all(abs(int(a) - int(b)) <= tol for a, b in zip(got, want))


def _overlay_rgba(view):
    pm = view._mask_item.pixmap()
    assert not pm.isNull(), "no overlay pixmap was produced"
    img = pm.toImage()
    img = img.convertToFormat(img.Format.Format_RGBA8888)
    ptr = img.bits()
    ptr.setsize(img.sizeInBytes())
    arr = np.frombuffer(ptr, np.uint8).reshape(
        img.height(), img.bytesPerLine() // 4, 4
    )[:, : img.width()]
    return arr


def test_overlay_covers_every_kind_of_mask():
    """Brush, linear, radial and model-produced alphas alike."""
    import raw_mask_shapes as shapes

    def brush():
        layer = MaskLayer.empty(W, W, name="brush")
        layer.alpha[60:140, 60:140] = 1.0
        layer.touch()
        return layer

    def gradient(kind, drag):
        return MaskLayer(
            np.zeros((W, W), np.float32),
            kind=kind,
            params=shapes.params_from_drag(kind, *drag),
            name=kind,
        )

    cases = {
        "brush": brush(),
        "linear": gradient("linear", (0.5, 0.05, 0.5, 0.6)),
        "radial": gradient("radial", (0.25, 0.25, 0.75, 0.75)),
        # Subject / Sky / Point-select all arrive as a plain alpha buffer.
        "ai": brush(),
    }
    for name, layer in cases.items():
        v = _view()
        v.update_mask_layer_overlay([layer], 0)
        rgba = _overlay_rgba(v)
        assert int(rgba[..., 3].max()) > 0, f"{name} mask produced no visible overlay"
    print("  OK   every mask kind produces a visible overlay")


def test_gradients_render_from_params_not_the_placeholder():
    """A gradient's alpha buffer is a placeholder; the overlay must generate."""
    import raw_mask_shapes as shapes

    layer = MaskLayer(
        np.zeros((128, 128), np.float32),
        kind="radial",
        params=shapes.params_from_drag("radial", 0.3, 0.3, 0.7, 0.7),
    )
    assert float(layer.alpha.max()) == 0.0, "fixture is not a placeholder buffer"
    v = _view()
    v.update_mask_layer_overlay([layer], 0)
    rgba = _overlay_rgba(v)
    assert int(rgba[..., 3].max()) > 0, (
        "gradient overlay read the empty placeholder instead of generating"
    )
    print("  OK   a gradient overlay is generated from its params")


def test_selected_mask_is_ember_others_recede():
    from rawviewer_ui.gpu_image_view import _MASK_TINT_ACTIVE, _MASK_TINT_OTHER

    left = MaskLayer.empty(W, W, name="left")
    left.alpha[:, : W // 2 - 10] = 1.0
    left.touch()
    right = MaskLayer.empty(W, W, name="right")
    right.alpha[:, W // 2 + 10 :] = 1.0
    right.touch()

    v = _view()
    v.update_mask_layer_overlay([left, right], 1)  # right selected
    rgba = _overlay_rgba(v)
    got_right = tuple(int(c) for c in rgba[W // 2, W - 5, :3])
    got_left = tuple(int(c) for c in rgba[W // 2, 5, :3])
    # +/-2 per channel: QPixmap round-trips through premultiplied alpha, which
    # costs a least-significant bit. The point is which colour, not the bit.
    assert _near(got_right, _MASK_TINT_ACTIVE), f"selected mask is {got_right}, not EMBER"
    assert _near(got_left, _MASK_TINT_OTHER), f"unselected mask is {got_left}, not BURN"
    print("  OK   selected mask reads EMBER, the rest recede in BURN")


def test_selected_mask_wins_where_masks_overlap():
    a = MaskLayer.empty(W, W, name="a")
    a.alpha[:, :] = 1.0
    a.touch()
    b = MaskLayer.empty(W, W, name="b")
    b.alpha[50:150, 50:150] = 1.0
    b.touch()
    from rawviewer_ui.gpu_image_view import _MASK_TINT_ACTIVE

    v = _view()
    v.update_mask_layer_overlay([a, b], 1)  # b selected, drawn inside a
    rgba = _overlay_rgba(v)
    assert _near(tuple(int(c) for c in rgba[100, 100, :3]), _MASK_TINT_ACTIVE), (
        "an unselected mask painted over the selected one"
    )
    print("  OK   the selected mask wins where masks overlap")


def test_disabled_masks_are_not_shown():
    layer = MaskLayer.empty(W, W, name="off")
    layer.alpha[:, :] = 1.0
    layer.enabled = False
    layer.touch()
    v = _view()
    v.update_mask_layer_overlay([layer], 0)
    pm = v._mask_item.pixmap()
    assert pm.isNull() or not v._mask_item.isVisible(), "a disabled mask was drawn"
    print("  OK   a disabled mask is not drawn")


def test_overlay_is_not_gated_on_an_armed_brush():
    """The bug: a subject/sky/gradient mask was invisible unless Paint was armed."""
    src = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "main.py"
    )
    text = open(src, encoding="utf-8").read()
    start = text.index("def _sync_mask_layer_overlay")
    end = text.index("\n    def ", start + 1)
    body = text[start:end]
    assert "mask_layer_mode" not in body, (
        "the mask overlay is gated on a brush tool being armed again"
    )
    assert "update_mask_layer_overlay" in body
    print("  OK   the overlay does not require an armed brush")


def test_masks_tab_shows_how_many_masks():
    p = _panel()
    assert p._panel_tabs._buttons[1].text() == "MASKS", "empty state should carry no count"
    p.set_mask_layer_stack(MaskLayerStack([
        _painted_layer(name="a"), _painted_layer(name="b"),
    ]))
    assert p._panel_tabs._buttons[1].text().endswith("2"), (
        f"Masks tab does not show the count: {p._panel_tabs._buttons[1].text()!r}"
    )
    print("  OK   the Masks tab shows how many masks the photo has")


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
    test_paint_creates_its_own_mask_like_every_other_tool()
    test_new_mask_button_is_hidden_in_the_empty_state()
    test_overlay_covers_every_kind_of_mask()
    test_gradients_render_from_params_not_the_placeholder()
    test_selected_mask_is_ember_others_recede()
    test_selected_mask_wins_where_masks_overlap()
    test_disabled_masks_are_not_shown()
    test_overlay_is_not_gated_on_an_armed_brush()
    test_masks_tab_shows_how_many_masks()
    print("\nPASS t_masks_ui_cleanup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
