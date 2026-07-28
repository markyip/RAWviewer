"""Per-mask visibility, and the adjustments grouped the way Global groups them.

MaskLayer.enabled was fully implemented and had no switch: the compositor
skipped a disabled layer, the overlay skipped it, the fingerprint folded it
in, and mask_layers_xmp persisted it. Only the UI was missing.

It cannot come back as a check box on the row. That is how it used to work,
and it was removed because the check box and the row selection shared one
click -- "did I select this mask or turn it off?". The eye gets its own
column so the two actions live in two places.

The adjustment list is also now grouped Light / Color / Detail, matching the
Global tab's names and order, and Defringe is exposed. Defringe needed no
backend work at all: it was already in SUPPORTED_ADJUSTMENT_KEYS, already
applied by _apply_layer_adjustments, and already padded as a spatial op.
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

    # --- hiding a mask is an exact no-op, not merely a faint one ---
    m = _layer(adj={"Exposure2012": 2.0})
    lit = apply_mask_layers(img.copy(), MaskLayerStack(layers=[m]))
    check("an enabled mask changes the photo", not np.array_equal(lit, img))

    m.enabled = False
    hidden = apply_mask_layers(img.copy(), MaskLayerStack(layers=[m]))
    check(
        "a hidden mask leaves it untouched",
        np.array_equal(hidden, img),
        "a partial effect would make the toggle useless for judging the edit",
    )

    # --- and the fingerprint moves, so nothing serves a cached composite ---
    a = _layer(adj={"Exposure2012": 2.0})
    before = a.fingerprint()
    a.enabled = False
    check("visibility is in the cache key", a.fingerprint() != before)

    # --- it survives a save/load round trip ---
    import mask_layers_xmp

    st = MaskLayerStack(layers=[_layer("Visible"), _layer("Hidden")])
    st.layers[1].enabled = False
    blob = mask_layers_xmp.serialize_stack(st)
    back = mask_layers_xmp.deserialize_stack(blob)
    check(
        "hidden survives a reload",
        back is not None
        and len(back.layers) == 2
        and back.layers[0].enabled is True
        and back.layers[1].enabled is False,
        str([l.enabled for l in (back.layers if back else [])]),
    )

    # A hidden mask reports is_empty, and the serializer skipped empty
    # layers -- so hiding a mask and saving used to delete it outright,
    # taking the coverage the user painted with it. Unreachable until the
    # eye existed, which is exactly why it has to be covered now.
    painted = _layer("Hidden, no adjustments")
    painted.enabled = False
    back = mask_layers_xmp.deserialize_stack(
        mask_layers_xmp.serialize_stack(MaskLayerStack(layers=[painted]))
    )
    check(
        "a hidden mask with no adjustments is still saved",
        back is not None and len(back.layers) == 1,
        "hiding a mask must never destroy it",
    )
    check(
        "and its coverage comes back",
        back is not None
        and back.layers
        and float(back.layers[0].alpha.max()) > 0.5,
        "the painted region is the part that cannot be recreated",
    )

    all_hidden = MaskLayerStack(layers=[_layer("A"), _layer("B")])
    for lyr in all_hidden.layers:
        lyr.enabled = False
    back = mask_layers_xmp.deserialize_stack(
        mask_layers_xmp.serialize_stack(all_hidden)
    )
    check(
        "hiding every mask does not wipe the stack",
        back is not None and len(back.layers) == 2,
        f"{0 if back is None else len(back.layers)} of 2 survived",
    )

    # A stack that is genuinely empty must still serialize to "", or every
    # untouched photo would start writing a mask blob into its sidecar.
    blank = MaskLayer(np.zeros((H, W), np.float32), name="Untouched")
    check(
        "a genuinely empty stack still writes nothing",
        mask_layers_xmp.serialize_stack(MaskLayerStack(layers=[blank])) == "",
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
    check("clicking the eye hides the mask", stack.layers[0].enabled is False)
    check(
        "and the overlay is told to rebuild",
        len(seen) == 1,
        "a hidden mask must lose its tint as well as its effect",
    )
    check("the icon follows the state", not item.icon(1).isNull())

    p._on_mask_row_clicked(item, 1)
    check("clicking again shows it", stack.layers[0].enabled is True)

    p._on_mask_row_clicked(item, 0)
    check(
        "clicking the name does not toggle",
        stack.layers[0].enabled is True,
        "column 0 selects and renames; it must never change visibility",
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

    p2._on_mask_row_clicked(parent.child(0), 1)
    check(
        "hiding a component leaves the group visible",
        stack.layers[0].components[0].enabled is False
        and stack.layers[0].enabled is True,
        "the eye must act on the row it is on, not the whole group",
    )
    check(
        "the other component is untouched",
        stack.layers[0].components[1].enabled is True,
    )

    # --- renaming still works, and does not disturb visibility ---
    p3 = _panel(MaskLayerStack(layers=[_layer("Before")]))
    st3 = p3._mask_stack
    it3 = p3._mask_list.topLevelItem(0)
    it3.setText(0, "After")
    p3._on_mask_item_changed(it3, 0)
    check("column 0 still renames", st3.layers[0].name == "After", st3.layers[0].name)
    check("and visibility is unchanged", st3.layers[0].enabled is True)

    # --- grouped adjustments, named as Global names them ---
    p4 = _panel(MaskLayerStack(layers=[_layer()]))
    titles = [t for t, _ in p4._MASK_SLIDER_GROUPS]
    check("adjustments are grouped", titles == ["Light", "Color", "Detail"], str(titles))
    check(
        "the headings are rendered",
        [l.text() for l in p4._mask_group_labels] == ["LIGHT", "COLOR", "DETAIL"],
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
