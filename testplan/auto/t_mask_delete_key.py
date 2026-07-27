"""Delete removes the selected MASK, not the photograph.

In single-image view Delete deleted the current image unconditionally, even
with the Masks page forward and a mask selected. That is the most
destructive action in the app fired by a key pressed meaning something
small and undoable, so the targeting is pinned from both sides: it must
retarget when the user is looking at a mask, and it must NOT retarget in
any other state.
"""

import os
import sys

import numpy as np
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from raw_mask_layers import MaskLayer, MaskLayerStack  # noqa: E402
from rawviewer_ui.adjust_panel import ImageAdjustPanelWidget  # noqa: E402

GLOBAL, MASKS = 0, 1
FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


_app = QApplication.instance() or QApplication([])


def _panel_with_masks():
    p = ImageAdjustPanelWidget()
    p.show()
    a = MaskLayer(np.zeros((40, 60), np.float32), name="Sky")
    a.alpha[5:15, 5:15] = 1.0
    a.touch()
    b = MaskLayer(np.zeros((40, 60), np.float32), name="Foreground")
    b.alpha[20:30, 20:30] = 1.0
    b.touch()
    p.set_mask_layer_stack(MaskLayerStack(layers=[a, b]))
    return p


def main() -> int:
    # --- the panel-side predicate ---
    p = _panel_with_masks()

    p._panel_tabs.set_current(GLOBAL)
    check("Global page: Delete is not a mask delete", not p.masks_page_has_selection())

    p._panel_tabs.set_current(MASKS)
    p.select_mask_index(0)
    check("Masks page with a selection: Delete targets the mask", p.masks_page_has_selection())

    p._panel_tabs.set_current(GLOBAL)
    check(
        "selection alone is not enough -- the page must be forward",
        not p.masks_page_has_selection(),
    )

    # No stack bound at all: nothing to delete.
    empty = ImageAdjustPanelWidget()
    empty.show()
    empty._panel_tabs.set_current(MASKS)
    check(
        "Masks page with no stack: Delete is not a mask delete",
        not empty.masks_page_has_selection(),
    )

    # --- the host-side gate, without constructing the whole app ---
    import main as mainmod

    class _Host:
        _adjust_overlay_visible = True

        def __init__(self, panel):
            self.single_image_adjust_panel = panel

    gate = mainmod.RAWImageViewer._masks_page_delete_target

    p._panel_tabs.set_current(MASKS)
    p.select_mask_index(1)
    host = _Host(p)
    check("host gate opens when the panel says so", gate(host) is True)

    host._adjust_overlay_visible = False
    check("editor closed: Delete keeps its normal meaning", gate(host) is False)

    host._adjust_overlay_visible = True
    host.single_image_adjust_panel = None
    check("no panel at all: Delete keeps its normal meaning", gate(host) is False)

    # A panel missing the predicate (older widget) must not open the gate.
    class _Bare:
        def isVisible(self):
            return True

    host.single_image_adjust_panel = _Bare()
    check("panel without the predicate: gate stays shut", gate(host) is False)

    # A predicate that raises must fail closed -- deleting a photo because a
    # helper threw would be the worst possible outcome.
    class _Angry:
        def isVisible(self):
            return True

        def masks_page_has_selection(self):
            raise RuntimeError("boom")

    host.single_image_adjust_panel = _Angry()
    check("a raising predicate fails closed", gate(host) is False)

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
