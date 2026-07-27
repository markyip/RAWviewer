"""Undo restores what grouping discarded.

Grouping throws away the dragged mask's adjustments -- a component holds
coverage only. Undo is the thing that gives them back, and it does so
through the existing per-image _undo_stack, which snapshots the whole
adjustment dict before each save. No grouping-specific undo code exists,
which is exactly why this needs a test: the behaviour is inherited, and
inherited behaviour is what breaks unnoticed.

The round trip runs through the real XMP serial the undo stack stores, not
through live objects, so a group that failed to serialise would fail here.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from raw_mask_layers import MaskLayer, MaskLayerStack  # noqa: E402
import mask_layers_xmp as mx  # noqa: E402

H, W = 40, 60
FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def _layer(name, y, adj=None):
    m = MaskLayer(np.zeros((H, W), np.float32), name=name)
    m.alpha[y : y + 10, 5:20] = 1.0
    m.touch()
    m.adjustments = dict(adj or {})
    return m


def main() -> int:
    import main as mainmod

    group_fn = mainmod.RAWImageViewer._on_mask_group_requested

    class _Panel:
        def __init__(self, stack):
            self._stack = stack
            self._selected = 0

        def set_mask_layer_stack(self, s):
            self._stack = s

        def select_mask_index(self, i):
            self._selected = i

        def get_adjustments(self):
            return {}

    class _Host:
        def __init__(self, stack):
            self.single_image_adjust_panel = _Panel(stack)
            self._mask_layer_stack = stack
            self.status = []

        def _show_status(self, t, ms=0):
            self.status.append(t)

        def _on_adjust_panel_editing_finished(self, adj):
            pass

    a = _layer("Keeper", 5, {"Exposure2012": 0.7})
    b = _layer("Donor", 20, {"Saturation": 30.0})
    stack = MaskLayerStack(layers=[a, b])

    # What the undo stack would snapshot immediately before the group action.
    snapshot = mx.serialize_stack(stack)

    host = _Host(stack)
    group_fn(host, 1, 0)
    check("grouped", len(stack.layers) == 1 and stack.layers[0].is_group)
    check(
        "the dragged mask's adjustments are gone from the live stack",
        "Saturation" not in stack.layers[0].adjustments,
    )

    # Ctrl+Z: _apply_mask_layers_from_adj deserialises the snapshot.
    restored = mx.deserialize_stack(snapshot)
    check("undo brings both masks back", len(restored.layers) == 2, str(len(restored.layers)))
    check("undo un-groups", all(not l.is_group for l in restored.layers))
    check("names restored", [l.name for l in restored.layers] == ["Keeper", "Donor"])
    check(
        "the DISCARDED adjustments come back",
        abs(restored.layers[1].adjustments.get("Saturation", 0.0) - 30.0) < 1e-4,
        str(restored.layers[1].adjustments),
    )
    check(
        "the kept adjustments are unharmed",
        abs(restored.layers[0].adjustments.get("Exposure2012", 0.0) - 0.7) < 1e-4,
    )
    cov = restored.layers[1].alpha_at(H, W)
    check("coverage restored too", float(cov[24, 10]) > 0.9, f"{cov[24,10]:.2f}")

    # The restore path must hand back MUTABLE objects -- the panel edits them
    # in place, so a shared cached stack would be corrupted by the next brush
    # stroke. This is why _apply_mask_layers_from_adj avoids the read-only memo.
    again = mx.deserialize_stack(snapshot)
    check("each restore is a fresh object", again is not restored)
    again.layers[0].alpha[0, 0] = 0.5
    check(
        "mutating one restore does not touch another",
        float(restored.layers[0].alpha[0, 0]) == 0.0,
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
