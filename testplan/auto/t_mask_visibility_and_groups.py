"""The per-mask eye hides the tint only, and the grouped adjustment list.

The eye is a VIEW control. Hiding a mask's overlay must leave its adjustment
fully applied -- you turn the tint off precisely so you can see the result of
the mask you just aimed. That makes it distinct from MaskLayer.enabled, which
turns the adjustment itself off, and the two must not be confused.

Because it changes no pixels, overlay_hidden stays out of fingerprint() and
out of the sidecar: keying a render cache on it would discard a correct
composite and re-render the frame to produce identical output, and writing it
would put view state in the user's XMP.

It cannot be a check box on the row. That is how enable/disable used to work,
and it was removed because the check box and the row selection shared one
click -- "did I select this mask or turn it off?". The eye gets its own
column, sized to clear the icon box Qt hands every column.

The adjustment list is also grouped Light / Color / Detail, matching the
Global tab's names and order, and Defringe is exposed. Defringe needed no
backend work: it was already in SUPPORTED_ADJUSTMENT_KEYS, already applied by
_apply_layer_adjustments, and already padded as a spatial op.
"""

import os
import sys

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from raw_mask_layers import (  # noqa: E402
    SUPPORTED_ADJUSTMENT_KEYS,
    MaskLayer,
    MaskLayerStack,
    apply_mask_layers,
)
from rawviewer_ui.adjust_panel import ImageAdjustPanelWidget  # noqa: E402

H, W = 120, 160
FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


_app = QApplication.instance() or QApplication([])


def _layer(name="Mask 1", adj=None):
    m = MaskLayer(np.zeros((H, W), np.float32), name=name)
    m.alpha[20:60, 20:80] = 1.0
    m.touch()
    m.adjustments = dict(adj or {})
    return m


def _panel(stack):
    p = ImageAdjustPanelWidget()
    p.show()
    p._panel_tabs.set_current(1)
    p.set_mask_layer_stack(stack)
    return p


def main() -> int:
    rng = np.random.default_rng(0)
    img = (rng.random((H, W, 3)).astype(np.float32) * 0.6) + 0.2

    # --- the eye hides the TINT, not the edit ---
    m = _layer(adj={"Exposure2012": 2.0})
    lit = apply_mask_layers(img.copy(), MaskLayerStack(layers=[m]))
    check("a mask changes the photo", not np.array_equal(lit, img))

    m.overlay_hidden = True
    still = apply_mask_layers(img.copy(), MaskLayerStack(layers=[m]))
    check(
        "hiding the overlay leaves the adjustment applied",
        np.array_equal(still, lit),
        "the eye is a view control -- it must not undo the edit",
    )

    # --- and it must not invalidate anything, or the frame re-renders ---
    a = _layer(adj={"Exposure2012": 2.0})
    before = a.fingerprint()
    a.overlay_hidden = True
    check(
        "overlay state is not in the cache key",
        a.fingerprint() == before,
        "a tint is not a pixel -- keying on it re-renders to identical output",
    )

    # --- nor reach the sidecar: it is not an edit ---
    import mask_layers_xmp

    st = MaskLayerStack(layers=[_layer("Shown"), _layer("Tint off")])
    st.layers[1].overlay_hidden = True
    blob = mask_layers_xmp.serialize_stack(st)
    check("overlay state is not written to XMP", "overlay_hidden" not in blob)
    back = mask_layers_xmp.deserialize_stack(blob)
    check(
        "both masks survive the round trip",
        back is not None and len(back.layers) == 2,
        f"{0 if back is None else len(back.layers)} of 2",
    )
    check(
        "and come back with their overlay shown",
        back is not None and not any(l.overlay_hidden for l in back.layers),
        "view state does not persist across a reload",
    )

    # --- enabled is a different thing, and still works ---
    m2 = _layer(adj={"Exposure2012": 2.0})
    m2.enabled = False
    off = apply_mask_layers(img.copy(), MaskLayerStack(layers=[m2]))
    check(
        "enabled=False still turns the adjustment off",
        np.array_equal(off, img),
        "the two flags must stay distinct",
    )

    # --- the eye is its own column, so a click cannot mean two things ---
    stack = MaskLayerStack(layers=[_layer()])
    p = _panel(stack)
    tree = p._mask_list
    item = tree.topLevelItem(0)

    check("the tree has a second column", tree.columnCount() == 2)
    check("the eye is drawn in it", not item.icon(1).isNull())
    check(
        "and the row is not checkable",
        not (item.flags() & Qt.ItemFlag.ItemIsUserCheckable),
        "a check box would make one click mean select AND toggle again",
    )

    seen = []
    p.mask_coverage_changed.connect(lambda: seen.append(1))

    p._on_mask_row_clicked(item, 1)
    check("clicking the eye hides the tint", stack.layers[0].overlay_hidden is True)
    check(
        "and leaves the adjustment on",
        stack.layers[0].enabled is True,
        "the eye must never disable the mask",
    )
    check(
        "and the overlay is told to rebuild",
        len(seen) == 1,
        "a hidden mask must lose its tint as well as its effect",
    )
    check("the icon follows the state", not item.icon(1).isNull())

    p._on_mask_row_clicked(item, 1)
    check("clicking again shows it", stack.layers[0].overlay_hidden is False)

    p._on_mask_row_clicked(item, 0)
    check(
        "clicking the name does not toggle",
        stack.layers[0].overlay_hidden is False,
        "column 0 selects and renames; it must never change visibility",
    )

    # --- the eye has to fit the column it is drawn in ---
    from rawviewer_ui.adjust_panel import _MASK_EYE_COL_W, _MASK_EYE_PX

    canvas = item.icon(1).availableSizes()[0].width()
    check(
        "the eye fits its column",
        canvas + 11 <= _MASK_EYE_COL_W,  # 11 = the tree's item padding
        f"a {canvas}px canvas in a {_MASK_EYE_COL_W}px column clips at the "
        "panel edge -- setIconSize is per widget, so column 1 gets the same "
        "box the mask thumbnails use",
    )
    check(
        "and is drawn small inside it",
        _MASK_EYE_PX < canvas,
        f"{_MASK_EYE_PX}px glyph on a {canvas}px canvas -- a full-box eye "
        "would carry as much weight as the mask thumbnail",
    )
    check(
        "the name column still gets the room",
        tree.columnWidth(0) > _MASK_EYE_COL_W * 3,
        f"{tree.columnWidth(0)} vs {_MASK_EYE_COL_W}",
    )

    # --- the eye sits in the middle of its cell, on any screen ---
    from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap

    import rawviewer_ui.adjust_panel as _ap

    def _centres(dpr):
        """Where the drawn glyph lands inside the returned canvas."""
        def fake(_name, color):
            side = int(15 * dpr)
            pm = QPixmap(side, side)
            pm.setDevicePixelRatio(float(dpr))
            pm.fill(Qt.GlobalColor.transparent)
            pnt = QPainter(pm)
            pnt.fillRect(0, 0, side, side, QColor("white"))
            pnt.end()
            return QIcon(pm)

        real = _ap._qta_icon_safe
        _ap._qta_icon_safe = fake
        try:
            img = _ap._mask_eye_icon(True, 34).pixmap(34, 34).toImage()
        finally:
            _ap._qta_icon_safe = real
        pts = [
            (x, y)
            for y in range(img.height())
            for x in range(img.width())
            if img.pixelColor(x, y).alpha() > 0
        ]
        xs = [q[0] for q in pts]
        ys = [q[1] for q in pts]
        return (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, img.width()

    for dpr in (1, 2):
        cx, cy, side = _centres(dpr)
        mid = side / 2.0
        check(
            f"the eye is centred at dpr {dpr}",
            abs(cx - mid) <= 1.0 and abs(cy - mid) <= 1.0,
            f"glyph centre ({cx:.1f}, {cy:.1f}) vs box centre {mid:.1f} -- at "
            "dpr 2 a retina icon reports twice its logical size, so centring "
            "on the raw width pushes it toward the top-left corner",
        )

    # --- an inverted mask's thumbnail shows what it actually covers ---
    inv = _layer(adj={})
    plain_icon = p._mask_row_icon(inv)
    inv.invert = True
    inv.touch()
    inverted_icon = p._mask_row_icon(inv)

    def _shape(icon):
        im = icon.pixmap(34, 34).toImage()
        return [
            im.pixelColor(x, y).value()
            for y in range(0, im.height(), 4)
            for x in range(0, im.width(), 4)
        ]

    check(
        "inverting redraws the row thumbnail",
        _shape(plain_icon) != _shape(inverted_icon),
        "drawing the raw alpha showed the painted region -- exactly the part "
        "an inverted mask does NOT cover -- so a mask and its inverted copy "
        "were indistinguishable in the list",
    )
    lit_plain = sum(1 for v in _shape(plain_icon) if v > 127)
    lit_inv = sum(1 for v in _shape(inverted_icon) if v > 127)
    check(
        "and shows the complement of it",
        lit_plain + lit_inv > 0 and abs((lit_plain + lit_inv) - len(_shape(plain_icon))) <= 2,
        f"{lit_plain} lit normally, {lit_inv} inverted, of "
        f"{len(_shape(plain_icon))} sampled",
    )

    # --- the overlay follows the tool, and does not follow you off the page ---
    p.set_mask_layer_stack(MaskLayerStack(layers=[_layer()]))
    p._set_mask_overlay_visible(False)
    p._mask_paint_btn.setChecked(True)
    check(
        "arming Paint shows what is masked",
        p.dodge_burn_show_mask(),
        "painting coverage you cannot see is guesswork",
    )

    p._panel_tabs.set_current(0)
    check(
        "leaving the Masks page puts the overlay away",
        not p.dodge_burn_show_mask(),
        "the tint would sit over the photo you switched to Global to judge",
    )
    p._panel_tabs.set_current(1)
    check("and coming back does not bring it straight back", not p.dodge_burn_show_mask())

    p._mask_erase_btn.setChecked(True)
    check("arming Erase shows it too", p.dodge_burn_show_mask())

    # M must still win: this follows the tool, it does not overrule the user.
    p.toggle_dodge_burn_show_mask()
    check("M still turns it off with a brush armed", not p.dodge_burn_show_mask())

    # --- clicking an eye has to produce something to look at ---
    import main as _mainmod

    two = MaskLayerStack(layers=[_layer("A"), _layer("B")])
    p5 = _panel(two)
    drawn = []

    class _GV:
        def update_mask_layer_overlay(self, *a, **k):
            drawn.append(k.get("solo_index"))

        def update_dodge_burn_mask(self, *a, **k):
            pass

        def hide_mask_overlay(self):
            pass

    class _H:
        def __init__(self):
            self.single_image_adjust_panel = p5
            self._mask_layer_stack = two
            self.gpu_view = _GV()
            self._mask_overlay_forced_by_stroke = False

        def _sync_dodge_burn_mask_overlay(self):
            pass

        def _sync_mask_layer_overlay(self, **k):
            _mainmod.RAWImageViewer._sync_mask_layer_overlay(self, **k)

        def _on_mask_coverage_changed(self):
            _mainmod.RAWImageViewer._on_mask_coverage_changed(self)

    host = _H()
    p5.mask_coverage_changed.connect(host._on_mask_coverage_changed)
    p5._set_mask_overlay_visible(False)
    row_a = p5._mask_list.topLevelItem(0)
    row_b = p5._mask_list.topLevelItem(1)

    def shown():
        return [not l.overlay_hidden for l in two.layers]

    check(
        "a fresh stack starts with exactly one shown",
        shown().count(True) == 1,
        str(shown()),
    )

    drawn.clear()
    p5._on_mask_row_clicked(row_b, 1)  # show B
    check(
        "showing a mask switches the overlay on",
        p5.dodge_burn_show_mask(),
        "with the overlay off, clicking an eye changed a flag nothing was "
        "drawing and the control looked dead",
    )
    check("and shows only that one", shown() == [False, True], str(shown()))
    check("and actually redraws", len(drawn) >= 1, str(len(drawn)))

    p5._on_mask_row_clicked(row_a, 1)  # show A instead
    check("showing another swaps them", shown() == [True, False], str(shown()))

    p5._on_mask_row_clicked(row_a, 1)  # hide it again
    check(
        "and a mask can be hidden with nothing taking its place",
        shown() == [False, False],
        str(shown()),
    )

    # --- a component inside a group hides on its own ---
    from raw_mask_layers import MaskLayer as ML

    grp = _layer("Group")
    grp.components = [ML(np.zeros((H, W), np.float32), name="A"),
                      ML(np.zeros((H, W), np.float32), name="B")]
    for c in grp.components:
        c.alpha[20:40, 20:40] = 1.0
        c.touch()
    stack = MaskLayerStack(layers=[grp])
    p2 = _panel(stack)
    parent = p2._mask_list.topLevelItem(0)
    check("the group has child rows", parent.childCount() == 2, str(parent.childCount()))

    before = [c.overlay_hidden for c in stack.layers[0].components]
    p2._on_mask_row_clicked(parent.child(0), 1)
    check(
        "a component row carries no eye",
        [c.overlay_hidden for c in stack.layers[0].components] == before,
        "the overlay is built from top-level layers and _combined_alpha_at "
        "folds a group's parts without consulting overlay_hidden, so a "
        "per-component eye could never change anything on screen",
    )
    check(
        "and only top-level rows draw one",
        parent.child(0).icon(1).isNull() and not parent.icon(1).isNull(),
    )

    # --- renaming still works, and does not disturb visibility ---
    p3 = _panel(MaskLayerStack(layers=[_layer("Before")]))
    st3 = p3._mask_stack
    it3 = p3._mask_list.topLevelItem(0)
    it3.setText(0, "After")
    p3._on_mask_item_changed(it3, 0)
    check("column 0 still renames", st3.layers[0].name == "After", st3.layers[0].name)
    check("and visibility is unchanged", st3.layers[0].overlay_hidden is False)

    # --- grouped adjustments, named as Global names them ---
    p4 = _panel(MaskLayerStack(layers=[_layer()]))
    titles = [t for t, _ in p4._MASK_SLIDER_GROUPS]
    check("adjustments are grouped", titles == ["Light", "Color", "Detail"], str(titles))
    # --- the groups fold, the way the Global page's sections do ---
    from rawviewer_ui.adjust_panel import CollapsibleSection

    secs = p4._mask_group_sections
    check("there is a section per group", len(secs) == 3, str(len(secs)))
    check(
        "and they are the Global page's own section widget",
        all(isinstance(s, CollapsibleSection) for s in secs),
        "a styled label cannot fold; these carry the same controls as Global "
        "and should behave the same",
    )
    check(
        "they start open",
        all(s._expanded for s in secs),
        "collapsed by default would hide every slider behind a header on the "
        "one page whose purpose is adjusting the selected mask",
    )

    secs[1].set_expanded(False)
    check("a group collapses", not secs[1]._expanded)
    check("and takes its sliders with it", not secs[1].content.isVisible())
    check("leaving the others alone", secs[0]._expanded and secs[2]._expanded)
    secs[1].set_expanded(True)
    check("and re-opens", secs[1]._expanded)

    check(
        "the mask groups do not collide with Global's sections",
        p4.sect_light._expanded is True,
        "'mask_light' and 'light' must be separate settings keys",
    )

    # --- Copy/Paste are global-only, so they leave the Masks page ---
    p4._panel_tabs.set_current(0)
    check(
        "Copy and Paste show on Global",
        p4._copy_btn.isVisible() and p4._paste_btn.isVisible(),
    )
    p4._panel_tabs.set_current(1)
    check(
        "and are hidden on Masks",
        not p4._copy_btn.isVisible() and not p4._paste_btn.isVisible(),
        "they copy get_adjustments(), which never contains a mask -- on this "
        "page they look like they would copy the mask and silently would not",
    )
    p4._panel_tabs.set_current(0)
    check("and come back on Global", p4._copy_btn.isVisible())
    p4._panel_tabs.set_current(1)

    import inspect as _inspect

    copy_src = _inspect.getsource(p4._on_copy_settings_clicked)
    check(
        "Copy really is global-only",
        "get_adjustments()" in copy_src and "mask" not in copy_src.lower().split('"""')[-1],
        "if it ever learns about masks, it should come back to this page",
    )

    # Exposure/Contrast under Light, WB and saturation under Color: the same
    # split the Global tab uses.
    by_group = {t: [k for k, *_ in specs] for t, specs in p4._MASK_SLIDER_GROUPS}
    check("Light holds the tone controls", by_group["Light"] == ["Exposure2012", "Contrast2012"])
    check(
        "Color holds WB and saturation",
        by_group["Color"] == ["Temperature", "Tint", "Saturation", "Vibrance"],
        str(by_group["Color"]),
    )
    check("Defringe is exposed", "Defringe" in by_group["Detail"], str(by_group["Detail"]))

    # --- the panel offers exactly what the compositor can apply ---
    shown = {k for k, *_ in p4._MASK_SLIDER_SPECS}
    check(
        "every slider is a key the compositor supports",
        shown <= set(SUPPORTED_ADJUSTMENT_KEYS),
        f"unsupported: {sorted(shown - set(SUPPORTED_ADJUSTMENT_KEYS))}",
    )
    missing = set(SUPPORTED_ADJUSTMENT_KEYS) - shown
    check(
        "and nothing supported is left without one",
        not missing,
        f"unreachable from the UI: {sorted(missing)}",
    )

    # --- Defringe is not decorative: it reaches the pixels ---
    out = apply_mask_layers(
        img.copy(), MaskLayerStack(layers=[_layer(adj={"Defringe": 80.0})])
    )
    check(
        "Defringe actually changes the masked region",
        float(np.abs(out - img).max()) > 1e-4,
        f"max delta {float(np.abs(out - img).max()):.4f}",
    )
    check(
        "and only inside the mask",
        float(np.abs(out - img)[100:, 120:].max()) < 1e-6,
        "a spatial op must stay inside its bbox",
    )

    # --- every slider still resets, including the new one ---
    unwired = []
    for k, *_ in p4._MASK_SLIDER_SPECS:
        sl = p4._mask_sliders[k]
        sl.setValue(sl.maximum())
        p4._mask_value_labels[k].clicked.emit()
        if sl.value() != 0:
            unwired.append(k)
    check("all grouped sliders reset", not unwired, str(unwired))

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
