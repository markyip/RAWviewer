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
    for row in range(p._mask_list.topLevelItemCount()):
        item = p._mask_list.topLevelItem(row)
        assert not (item.flags() & Qt.ItemFlag.ItemIsUserCheckable), (
            f"row {row} is still checkable"
        )
        assert item.flags() & Qt.ItemFlag.ItemIsSelectable, f"row {row} is not selectable"
    p._mask_list.setCurrentItem(p._mask_list.topLevelItem(1))
    assert p.active_mask_index() == 1, "clicking a row did not select it"
    print("  OK   rows select on click, with no check box")


def test_select_button_is_named_for_what_it_does():
    p = _panel()
    label = p._mask_ai_click_btn.text()
    # "Click" named the input rather than the action, so it meant nothing.
    assert label == "AI Selection", f"point-prompt button is labelled {label!r}"
    subject = p._mask_ai_subject_btn.text()
    assert subject == "Smart Object", f"one-press button is labelled {subject!r}"

    # Every mask tool needs a tooltip: the names alone cannot carry the
    # difference between "the app chooses" and "you choose".
    for btn in (
        p._mask_ai_subject_btn, p._mask_ai_sky_btn, p._mask_ai_click_btn,
        p._mask_paint_btn, p._mask_erase_btn, p._mask_invert_btn,
        p._mask_linear_btn, p._mask_radial_btn,
    ):
        tip = btn.toolTip()
        assert len(tip) > 40, f"{btn.text()!r} has no real tooltip: {tip!r}"
        assert tip.startswith(btn.text().split(" (")[0]), (
            f"{btn.text()!r} tooltip does not lead with the tool name: {tip[:40]!r}"
        )

    # Smart Object returns ONE mask even when it finds several objects --
    # measured, not assumed: on the DPReview studio scene it comes back with
    # 11 separate regions in a single mask. The tooltip has to say so, or the
    # user reaches for it expecting to isolate one thing.
    assert "ONE mask" in p._mask_ai_subject_btn.toolTip(), (
        "Smart Object tooltip does not say it returns a single mask"
    )
    assert "AI Selection" in p._mask_ai_subject_btn.toolTip(), (
        "Smart Object tooltip does not point at the tool that isolates one thing"
    )
    print("  OK   tools are named and explained, Smart Object states its limit")


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


def test_create_menu_is_the_one_way_to_start_a_mask():
    """One control starts a mask, and it names the tool up front.

    It replaced "New empty mask" sitting above six tool buttons. That button
    was load-bearing for the brush alone -- the gradients and the AI tools
    each create their own mask, while the brush paints into the selected one
    -- so it was inert next to five of the six things beside it.

    Unlike the button it replaced, Create is available with no masks yet: it
    is the only way to get the first one.
    """
    p = _panel()
    assert p._mask_create_btn.text() == "Create new mask", p._mask_create_btn.text()

    menu = p._mask_create_btn.menu()
    assert menu is not None, "Create new mask has no menu"
    items = [a.text() for a in menu.actions() if not a.isSeparator()]
    assert items == ["Brush", "Linear Gradient", "Radial Gradient",
                     "Smart Object", "Sky", "AI Selection"], items

    # Brush is the only entry that needs a layer made for it up front.
    fired = []
    p.mask_add_requested.connect(lambda: fired.append(1))
    p._on_mask_create("brush")
    assert fired == [1], "Brush did not ask the host for a mask"

    for kind in ("linear", "radial", "ai_click"):
        fired.clear()
        p._on_mask_create(kind)
        assert fired == [], (
            f"{kind} pre-created a mask -- it makes its own, so arming and "
            "then changing your mind would leave an empty one behind"
        )
    print("  OK   one control starts a mask, and it names the tool")


def test_armed_creation_tool_says_what_it_wants():
    """A menu item cannot stay lit the way an armed button did."""
    p = _panel()
    assert not p._mask_armed_hint.isVisible(), "hint shown with nothing armed"

    p._on_mask_create("linear")
    assert p._mask_armed_hint.isVisible(), "no hint after arming Linear"
    assert "Drag" in p._mask_armed_hint.text(), p._mask_armed_hint.text()

    p._on_mask_create("radial")
    assert "box" in p._mask_armed_hint.text(), (
        "the hint did not follow the newly armed tool"
    )

    p.disarm_gradient_tools()
    p._sync_mask_armed_hint()
    assert not p._mask_armed_hint.isVisible(), "hint outlived the armed tool"
    print("  OK   the armed tool says what it is waiting for")


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
    tabs = p._panel_tabs
    assert tabs.badge_text(1) == "", "empty state should carry no count"
    # The label itself must stay intact -- the chip is a separate widget, and
    # an earlier version reserved its space with trailing spaces, which Qt
    # trims, so the chip printed over the last letters ("MAS 3").
    assert tabs._buttons[1].text() == "MASKS", tabs._buttons[1].text()

    p.set_mask_layer_stack(MaskLayerStack([
        _painted_layer(name="a"), _painted_layer(name="b"),
    ]))
    assert tabs.badge_text(1) == "2", f"count chip reads {tabs.badge_text(1)!r}"
    assert tabs._buttons[1].text() == "MASKS", (
        f"the chip corrupted the tab label: {tabs._buttons[1].text()!r}"
    )
    # And the button must be wide enough to hold both.
    metrics = tabs._buttons[1].fontMetrics()
    assert tabs._buttons[1].minimumWidth() > metrics.horizontalAdvance("MASKS"), (
        "no room reserved for the count chip"
    )
    print("  OK   the Masks tab shows how many masks, without eating its label")


def test_repeating_smart_object_does_not_stack_a_duplicate():
    """It repeats itself, so a second press must not add a second layer.

    Smart Object is a saliency segmenter: no instances, no memory, so the
    same photo returns a byte-identical matte every time (verified: three
    presses, max abs difference 0.0). Pressing it again therefore cannot
    find "the next" object -- and silently stacking the identical mask would
    compound its adjustments on the same pixels.
    """
    src = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "main.py"
    )
    text = open(src, encoding="utf-8").read()
    assert "_find_equivalent_mask_layer" in text, "no duplicate guard for AI masks"
    start = text.index("def _on_ai_mask_finished")
    end = text.index("\n    def ", start + 1)
    body = text[start:end]
    assert "_find_equivalent_mask_layer" in body, (
        "the duplicate guard is not consulted before adding an AI mask"
    )
    assert "already masked" in body, "no message explaining why nothing was added"
    assert "AI Selection" in body, (
        "the message does not point at the tool that CAN pick another object"
    )
    print("  OK   a repeated AI mask selects the existing layer instead of stacking")


def test_a_brushed_mask_is_not_treated_as_the_same_mask():
    """Tolerance must not swallow a real edit."""
    src = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "main.py"
    )
    text = open(src, encoding="utf-8").read()
    i = text.index("_AI_MASK_SAME_TOLERANCE = ")
    value = float(text[i:].split("=")[1].split("\n")[0].strip())
    # Above the 8-bit sidecar round trip (1/255) so a reloaded mask still
    # matches, well below what any deliberate brush stroke would move it.
    assert 1.0 / 255.0 < value < 0.05, f"tolerance {value} is not in a sane band"
    print(f"  OK   same-mask tolerance {value} clears the 8-bit round trip only")


def test_one_shot_ai_tools_disable_once_used():
    """Better than letting the press fail and explaining afterwards."""
    p = _panel()
    assert p._mask_ai_subject_btn.isEnabled()
    base_tip = p._mask_ai_subject_btn.toolTip()

    p.set_ai_tool_used("subject", True)
    assert not p._mask_ai_subject_btn.isEnabled(), "Smart Object stayed live"
    assert p._mask_ai_sky_btn.isEnabled(), "Sky was disabled by a subject mask"
    tip = p._mask_ai_subject_btn.toolTip()
    assert "already masked" in tip, f"disabled button does not say why: {tip!r}"
    assert "AI Selection" in tip, "disabled button offers no alternative"

    # Deleting the mask has to give the tool back, tooltip included.
    p.set_ai_tool_used("subject", False)
    assert p._mask_ai_subject_btn.isEnabled()
    assert p._mask_ai_subject_btn.toolTip() == base_tip, "original tooltip not restored"
    print("  OK   a used one-shot tool disables, explains, and comes back")


def test_ai_selection_is_never_disabled():
    """Every click adds something, so it is never 'already done'."""
    p = _panel()
    for kind in ("subject", "sky"):
        p.set_ai_tool_used(kind, True)
    assert p._mask_ai_click_btn.isEnabled(), "AI Selection was disabled"
    print("  OK   AI Selection stays available")


def test_layer_remembers_what_made_it():
    """Availability keys off source, not the row name, which users rename."""
    from mask_layers_xmp import deserialize_stack, serialize_stack

    layer = MaskLayer.empty(32, 32, name="Smart Object", source="subject",
                            adjustments={"Exposure2012": 1.0})
    layer.alpha[5:20, 5:20] = 1.0
    layer.touch()
    assert layer.source == "subject"

    back = deserialize_stack(serialize_stack(MaskLayerStack([layer])))
    assert back.layers[0].source == "subject", "source lost through the sidecar"

    # Renaming must not lose the association.
    back.layers[0].name = "my sky thing"
    assert back.layers[0].source == "subject"

    # A hand-painted layer carries no source and blocks nothing.
    assert MaskLayer.empty(8, 8).source == ""
    print("  OK   layers remember their source across rename and reload")


def test_overlay_draws_without_any_dodge_burn_mask():
    """The bug behind "the overlay is still missing".

    update_mask_layer_overlay used to early-return unless
    _mask_overlay_wanted was set -- a flag owned by the dodge/burn overlay
    path and only ever set when THAT path ran. So a subject, sky or gradient
    mask on a photo with no dodge/burn strokes could never draw.
    """
    v = _view()
    v._mask_overlay_wanted = False  # exactly the state that used to block it
    layer = _painted_layer(name="Smart Object")
    v.update_mask_layer_overlay([layer], 0)
    assert not v._mask_item.pixmap().isNull(), "overlay refused to draw"
    assert v._mask_item.isVisible()
    print("  OK   the overlay draws with no dodge/burn mask in play")


def test_hiding_the_overlay_clears_it():
    v = _view()
    v.update_mask_layer_overlay([_painted_layer()], 0)
    v.hide_mask_overlay()
    assert v._mask_item.pixmap().isNull(), "overlay survived being hidden"
    assert v._mask_overlay_wanted is False
    print("  OK   hiding the overlay clears it")


def test_mask_toggle_routes_to_the_layer_overlay():
    src = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "main.py"
    )
    text = open(src, encoding="utf-8").read()
    start = text.index("def _on_dodge_burn_mask_toggled")
    end = text.index("\n    def ", start + 1)
    body = text[start:end]
    assert "_sync_mask_layer_overlay" in body, (
        "the Mask toggle still routes only to the dodge/burn overlay"
    )
    assert "hide_mask_overlay" in body, "turning the toggle off does not hide it"
    print("  OK   the Mask toggle reaches the mask-layer overlay")


def test_brush_controls_are_reachable_while_masking():
    """Local lives on the Global tab; masking needed its own copy."""
    p = _panel()
    assert set(p._mask_brush_sliders) == {"Brush Size", "Brush Flow", "Brush Feather"}, (
        f"missing brush controls on the Masks tab: {sorted(p._mask_brush_sliders)}"
    )

    mirror = p._mask_brush_sliders["Brush Size"]
    local = p._db_size_slider
    mirror.setValue(mirror.value() + 17)
    assert local.value() == mirror.value(), "Masks-tab slider did not drive the brush"
    local.setValue(local.value() - 9)
    assert mirror.value() == local.value(), "Local slider did not update the mirror"

    # And the mirrored value must reach the code that actually stamps.
    p._mask_brush_sliders["Brush Feather"].setValue(15)
    assert abs(p.dodge_burn_brush_feather() - 0.15) < 1e-6, (
        "feather set while masking never reached the brush"
    )
    print("  OK   brush controls mirror both ways and reach the brush")


def test_mask_rows_show_the_mask_shape():
    """A row called "Mask 3" says nothing about what it covers."""
    import raw_mask_shapes as shapes

    p = _panel()
    brush = _painted_layer(name="Brush 1")
    gradient = MaskLayer(
        np.zeros((64, 64), np.float32),
        kind="radial",
        params=shapes.params_from_drag("radial", 0.3, 0.3, 0.7, 0.7),
        name="Radial Gradient",
    )
    p.set_mask_layer_stack(MaskLayerStack([brush, gradient]))

    for row in range(p._mask_list.topLevelItemCount()):
        icon = p._mask_list.topLevelItem(row).icon(0)
        assert not icon.isNull(), f"row {row} has no thumbnail"
        sizes = icon.availableSizes()
        assert sizes and max(sizes[0].width(), sizes[0].height()) > 8, (
            f"row {row} thumbnail is too small to read: {sizes}"
        )

    # A gradient's thumbnail must be generated, not read off its placeholder.
    icon = p._mask_row_icon(gradient)
    assert icon is not None and not icon.isNull(), "gradient row has no thumbnail"
    print("  OK   every row shows the mask's own shape")


def test_masks_tab_is_grouped_by_what_a_control_acts_on():
    """Four sections in working order, not one flat grid of buttons.

    A control's meaning should come from where it sits: the stack, a new
    mask, the selected mask, the brush, the effect. Asserted because the
    grouping is the whole point of the layout and a later tidy-up that
    flattens it back would look like simplification.
    """
    p = _panel()
    p.set_mask_layer_stack(MaskLayerStack([_painted_layer(name="M1")]))

    for attr in ("_mask_stack_head", "_mask_this_head", "_mask_brush_wrap",
                 "_mask_params_wrap"):
        assert getattr(p, attr, None) is not None, f"{attr} missing"

    # The six creation tools keep their names: they are now menu entries
    # driven by these buttons rather than buttons of their own.
    labels = {b.text() for b in (
        p._mask_paint_btn, p._mask_linear_btn, p._mask_radial_btn,
        p._mask_ai_subject_btn, p._mask_ai_sky_btn, p._mask_ai_click_btn,
    )}
    assert labels == {"Paint (P)", "Linear", "Radial", "Smart Object", "Sky",
                      "AI Selection"}, labels
    print("  OK   the tab is grouped by what each control acts on")


def test_brush_section_appears_only_with_a_brush_in_hand():
    p = _panel()
    p.set_mask_layer_stack(MaskLayerStack([_painted_layer(name="M1")]))
    assert not p._mask_brush_wrap.isVisible(), "Brush shown with no tool armed"

    p._mask_paint_btn.setChecked(True)
    assert p._mask_brush_wrap.isVisible(), "Brush hidden while painting"
    p._mask_erase_btn.setChecked(True)
    assert p._mask_brush_wrap.isVisible(), "Brush hidden while erasing"
    p._mask_erase_btn.setChecked(False)
    assert not p._mask_brush_wrap.isVisible(), "Brush stayed after disarming"
    print("  OK   Brush is tool state: it comes and goes with the tool")


def test_delete_does_not_look_like_its_neighbours():
    """Weight follows consequence -- Erase and Invert undo, Delete does not."""
    p = _panel()
    p.set_mask_layer_stack(MaskLayerStack([_painted_layer(name="M1")]))
    assert p._mask_del_btn.text() == "✕", (
        f"Delete is back to a full-width label: {p._mask_del_btn.text()!r}"
    )
    assert p._mask_del_btn.width() <= 40, "Delete is as wide as a normal button"
    sheet = p._mask_del_btn.styleSheet()
    assert "e5484d" in sheet.lower(), "Delete has no destructive colour on hover"
    # ...and only on hover: resting state must not shout.
    resting = sheet.split(":hover")[0]
    assert "e5484d" not in resting.lower(), "Delete is red before you reach for it"
    print("  OK   Delete is quiet until reached for, then the one red")


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
    test_create_menu_is_the_one_way_to_start_a_mask()
    test_armed_creation_tool_says_what_it_wants()
    test_masks_tab_is_grouped_by_what_a_control_acts_on()
    test_brush_section_appears_only_with_a_brush_in_hand()
    test_delete_does_not_look_like_its_neighbours()
    test_overlay_covers_every_kind_of_mask()
    test_gradients_render_from_params_not_the_placeholder()
    test_selected_mask_is_ember_others_recede()
    test_selected_mask_wins_where_masks_overlap()
    test_disabled_masks_are_not_shown()
    test_overlay_is_not_gated_on_an_armed_brush()
    test_masks_tab_shows_how_many_masks()
    test_repeating_smart_object_does_not_stack_a_duplicate()
    test_a_brushed_mask_is_not_treated_as_the_same_mask()
    test_one_shot_ai_tools_disable_once_used()
    test_ai_selection_is_never_disabled()
    test_layer_remembers_what_made_it()
    test_overlay_draws_without_any_dodge_burn_mask()
    test_hiding_the_overlay_clears_it()
    test_mask_toggle_routes_to_the_layer_overlay()
    test_brush_controls_are_reachable_while_masking()
    test_mask_rows_show_the_mask_shape()
    print("\nPASS t_masks_ui_cleanup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
