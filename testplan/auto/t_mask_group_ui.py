"""Grouping from the panel: the tree, the drop, and what the drop costs.

Drag one mask onto another and they become one mask. The target keeps its
adjustments; the source's are discarded, because a component holds coverage
only -- one adjustment set per mask, which is the whole point of the model.
That loss is the reason the action announces itself.
"""

import os
import sys

import numpy as np
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from raw_mask_layers import MaskLayer, MaskLayerStack  # noqa: E402
from rawviewer_ui.adjust_panel import ImageAdjustPanelWidget  # noqa: E402

H, W = 40, 60
FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


_app = QApplication.instance() or QApplication([])


def _layer(name, y, adj=None):
    m = MaskLayer(np.zeros((H, W), np.float32), name=name)
    m.alpha[y : y + 10, 5:20] = 1.0
    m.touch()
    m.adjustments = dict(adj or {})
    return m


class _Host:
    """Just enough host to drive the grouping handlers."""

    def __init__(self, panel, stack):
        self.single_image_adjust_panel = panel
        self._mask_layer_stack = stack
        self.status = []

    def _show_status(self, text, ms=0):
        self.status.append(text)

    def _on_adjust_panel_editing_finished(self, adj):
        pass


def _panel(stack):
    p = ImageAdjustPanelWidget()
    p.show()
    p._panel_tabs.set_current(1)
    p.set_mask_layer_stack(stack)
    return p


def main() -> int:
    import main as mainmod

    group_fn = mainmod.RAWImageViewer._on_mask_group_requested
    reorder_fn = mainmod.RAWImageViewer._on_mask_reorder_requested

    # --- plain onto plain: target is promoted to a group ---
    a = _layer("Keeper", 5, {"Exposure2012": 0.7})
    b = _layer("Donor", 20, {"Saturation": 30.0})
    stack = MaskLayerStack(layers=[a, b])
    p = _panel(stack)
    host = _Host(p, stack)

    group_fn(host, 1, 0)  # drag Donor onto Keeper

    check("stack collapses to one mask", len(stack.layers) == 1, f"{len(stack.layers)}")
    g = stack.layers[0]
    check("target became a group", g.is_group)
    check("group has both components", len(g.components) == 2, f"{len(g.components)}")
    check("target's adjustments survive", abs(g.adjustments.get("Exposure2012", 0) - 0.7) < 1e-6)
    check(
        "dragged mask's adjustments are gone",
        "Saturation" not in g.adjustments,
        str(g.adjustments),
    )
    check(
        "components carry no adjustments of their own",
        all(not c.adjustments for c in g.components),
    )
    check("the loss is announced", any("discarded" in s for s in host.status), str(host.status))

    # Coverage of both must survive even though the adjustments did not.
    al = g.alpha_at(H, W)
    check("target coverage kept", float(al[8, 10]) > 0.9, f"{al[8,10]:.2f}")
    check("dragged coverage kept", float(al[24, 10]) > 0.9, f"{al[24,10]:.2f}")

    # --- the tree shows it ---
    t = p._mask_list
    check("one top-level row", t.topLevelItemCount() == 1)
    check("with two children", t.topLevelItem(0).childCount() == 2)
    labels = [t.topLevelItem(0).child(i).text(0) for i in range(2)]
    check("components labelled by blend", all(l.startswith("+") for l in labels), str(labels))

    # --- dropping onto an existing group appends, does not nest ---
    c = _layer("Third", 30)
    stack.layers.append(c)
    p.set_mask_layer_stack(stack)
    group_fn(host, 1, 0)
    check("dropping onto a group appends", len(stack.layers[0].components) == 3)
    check("nesting stays two levels deep", all(not x.is_group for x in stack.layers[0].components))

    # --- merging two groups flattens ---
    g1 = MaskLayer(np.zeros((H, W), np.float32), name="G1",
                   components=[_layer("c1", 5), _layer("c2", 10)])
    g2 = MaskLayer(np.zeros((H, W), np.float32), name="G2",
                   components=[_layer("c3", 15)])
    st2 = MaskLayerStack(layers=[g1, g2])
    p2 = _panel(st2)
    h2 = _Host(p2, st2)
    group_fn(h2, 1, 0)
    check("merged group is flat", len(st2.layers) == 1 and len(st2.layers[0].components) == 3)
    check("no group inside a group", all(not x.is_group for x in st2.layers[0].components))

    # --- guards ---
    before = len(stack.layers)
    group_fn(host, 0, 0)
    check("dropping a mask on itself is a no-op", len(stack.layers) == before)
    group_fn(host, 99, 0)
    check("out-of-range source is a no-op", len(stack.layers) == before)

    # --- reorder is a different gesture ---
    x = _layer("X", 5)
    y = _layer("Y", 20)
    z = _layer("Z", 30)
    st3 = MaskLayerStack(layers=[x, y, z])
    p3 = _panel(st3)
    h3 = _Host(p3, st3)
    reorder_fn(h3, 0, 3)  # move X to the end
    check("reorder moves without grouping", [l.name for l in st3.layers] == ["Y", "Z", "X"],
          str([l.name for l in st3.layers]))
    check("reorder groups nothing", all(not l.is_group for l in st3.layers))

    # --- drag a component out of its group ---
    ungroup_fn = mainmod.RAWImageViewer._on_mask_ungroup_requested

    k = _layer("Keeper", 5, {"Exposure2012": 0.5})
    d = _layer("Donor", 20)
    st4 = MaskLayerStack(layers=[k, d])
    p4 = _panel(st4)
    h4 = _Host(p4, st4)
    group_fn(h4, 1, 0)
    grp = st4.layers[0]
    check("set-up: one group of two", grp.is_group and len(grp.components) == 2)

    ungroup_fn(h4, 0, 1, 1)  # pull the second component out, place it after
    check("component left the group", len(st4.layers[0].components) == 1)
    check("it became its own mask", len(st4.layers) == 2, str([l.name for l in st4.layers]))
    check("promoted mask keeps its coverage", float(st4.layers[1].alpha_at(H, W)[24, 10]) > 0.9)
    check(
        "promoted mask has no adjustments",
        not st4.layers[1].adjustments,
        "a component had none to give back -- undo is what restores those",
    )
    check("group keeps its own adjustments", abs(grp.adjustments.get("Exposure2012", 0) - 0.5) < 1e-6)
    check("the promotion is announced", any("own mask" in s for s in h4.status), str(h4.status[-1:]))

    # --- emptying a group removes it ---
    ungroup_fn(h4, 0, 0, 0)
    check("group with no components left is removed", all(not l.is_group for l in st4.layers))
    check("both masks survive as plain masks", len(st4.layers) == 2, str([l.name for l in st4.layers]))

    # --- guards ---
    n_before = len(st4.layers)
    ungroup_fn(h4, 0, 0, 0)  # layer 0 is not a group any more
    check("ungrouping a non-group is a no-op", len(st4.layers) == n_before)
    ungroup_fn(h4, 99, 0, 0)
    check("out-of-range group is a no-op", len(st4.layers) == n_before)

    # --- the tree lets a component be dragged, but never dropped on ---
    g5 = MaskLayer(np.zeros((H, W), np.float32), name="G", components=[_layer("c1", 5)])
    st5 = MaskLayerStack(layers=[g5])
    p5 = _panel(st5)
    child = p5._mask_list.topLevelItem(0).child(0)
    from PyQt6.QtCore import Qt as _Qt

    check("component row is draggable", bool(child.flags() & _Qt.ItemFlag.ItemIsDragEnabled))
    check(
        "component row is not a drop target",
        not (child.flags() & _Qt.ItemFlag.ItemIsDropEnabled),
        "two levels only",
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
