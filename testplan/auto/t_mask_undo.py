#!/usr/bin/env python3
"""Undo must not wipe mask layers when reverting a paint stroke.

Empty masks used to be omitted from XMP serialization, and Create→Brush did
not persist. The first stroke's undo snapshot was therefore "no masks", so
Ctrl+Z deleted the whole mask instead of restoring the empty layer.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def test_empty_mask_round_trips_through_xmp_serial():
    from mask_layers_xmp import deserialize_stack, serialize_stack
    from raw_mask_layers import MaskLayer, MaskLayerStack

    empty = MaskLayer.empty(64, 80, name="Mask 1")
    serial = serialize_stack(MaskLayerStack([empty]))
    check("empty mask serializes to non-empty JSON", bool(serial))
    restored = deserialize_stack(serial)
    check("empty mask deserializes", restored is not None and len(restored.layers) == 1)
    if restored is not None:
        check("restored layer keeps its name", restored.layers[0].name == "Mask 1")
        check(
            "restored layer is still empty coverage",
            bool(restored.layers[0].is_empty),
        )


def test_undo_after_paint_keeps_the_mask_row():
    """Simulate: create empty mask → persist → paint → undo → empty mask remains."""
    from mask_layers_xmp import deserialize_stack, serialize_stack
    from raw_mask_layers import MASK_LAYERS_KEY, MaskLayer, MaskLayerStack

    h, w = 48, 64
    empty = MaskLayer.empty(h, w, name="Mask 1")
    before_paint = {MASK_LAYERS_KEY: serialize_stack(MaskLayerStack([empty]))}

    painted = MaskLayer(empty.alpha.copy(), name="Mask 1")
    painted.alpha[10:30, 10:40] = 1.0
    painted.touch()
    after_paint = {MASK_LAYERS_KEY: serialize_stack(MaskLayerStack([painted]))}

    # Undo restores the pre-paint snapshot (what editing_finished pushed).
    undone = deserialize_stack(before_paint[MASK_LAYERS_KEY])
    check("undo snapshot still has one mask", undone is not None and len(undone.layers) == 1)
    check("undo restores empty coverage, not zero masks", bool(undone.layers[0].is_empty))

    # And the painted state really differs.
    check(
        "paint serial differs from empty serial",
        before_paint[MASK_LAYERS_KEY] != after_paint[MASK_LAYERS_KEY],
    )


def test_apply_mask_layers_from_adj_restores_empty_mask():
    import main as mainmod
    from mask_layers_xmp import serialize_stack
    from raw_mask_layers import MASK_LAYERS_KEY, MaskLayer, MaskLayerStack

    class _Panel:
        def __init__(self):
            self.stack = None

        def set_mask_layer_stack(self, stack):
            self.stack = stack

    host = type("H", (), {})()
    host._mask_layer_stack = MaskLayerStack(
        [MaskLayer.empty(32, 32, name="gone")]
    )
    host.single_image_adjust_panel = _Panel()
    apply = mainmod.RAWImageViewer._apply_mask_layers_from_adj.__get__(host)

    serial = serialize_stack(MaskLayerStack([MaskLayer.empty(32, 32, name="kept")]))
    apply({MASK_LAYERS_KEY: serial})
    check("host stack restored", host._mask_layer_stack is not None)
    check("panel stack restored", host.single_image_adjust_panel.stack is not None)
    check(
        "restored name",
        host._mask_layer_stack.layers[0].name == "kept",
    )


def test_mask_add_persists_via_editing_finished():
    import inspect
    import main as mainmod

    src = inspect.getsource(mainmod.RAWImageViewer._on_mask_layer_add)
    check(
        "mask add calls editing_finished so undo can snapshot",
        "_on_adjust_panel_editing_finished" in src,
    )


def main() -> int:
    test_empty_mask_round_trips_through_xmp_serial()
    test_undo_after_paint_keeps_the_mask_row()
    test_apply_mask_layers_from_adj_restores_empty_mask()
    test_mask_add_persists_via_editing_finished()
    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
