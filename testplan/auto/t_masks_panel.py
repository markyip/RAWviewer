#!/usr/bin/env python3
"""Masks panel (Adjust panel "Masks" section) — headless behavior checks.

Runs the real ImageAdjustPanelWidget offscreen and exercises the Masks
section's contract with main.py: stack binding, per-mask slider edits
mutating the live MaskLayer (with touch/version bumps), tool arming with
dodge/burn mutual exclusion, enable/invert plumbing, and the structure-op
request signals. The interactive feel (brush strokes on the GPU view)
needs a real display and is covered by the manual checklist instead.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["RAWVIEWER_ENABLE_EDITING"] = "1"

import numpy as np  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841

    from rawviewer_ui.adjust_panel import ImageAdjustPanelWidget
    from raw_mask_layers import MaskLayer, MaskLayerStack

    panel = ImageAdjustPanelWidget()

    # --- section exists and starts empty/disabled ---
    check("Masks section exists", hasattr(panel, "sect_masks"))
    check("no stack bound at startup", panel.mask_layer_stack() is None)
    check("no active index without a stack", panel.active_mask_index() is None)
    check("mask tools start disarmed", panel.mask_layer_mode() is None)
    check("sliders disabled without a layer", not panel._mask_sliders["Exposure2012"].isEnabled())

    # --- structure-op signals ---
    fired = {"add": 0, "delete": [], "dup": [], "mode": []}
    panel.mask_add_requested.connect(lambda: fired.__setitem__("add", fired["add"] + 1))
    panel.mask_delete_requested.connect(lambda i: fired["delete"].append(i))
    panel.mask_duplicate_requested.connect(lambda i: fired["dup"].append(i))
    panel.mask_layer_mode_changed.connect(lambda m: fired["mode"].append(m))

    panel._mask_add_btn.click()
    check("Add Mask emits mask_add_requested", fired["add"] == 1)

    # --- bind a stack (host side of the contract) ---
    layer_a = MaskLayer(np.zeros((40, 60), dtype=np.float32), name="Sky")
    layer_a.alpha[5:15, 5:15] = 1.0
    layer_a.touch()
    layer_b = MaskLayer(np.zeros((40, 60), dtype=np.float32), name="Subject")
    stack = MaskLayerStack(layers=[layer_a, layer_b])
    panel.set_mask_layer_stack(stack)

    check("stack bound", panel.mask_layer_stack() is stack)
    check("list shows one row per layer", panel._mask_list.topLevelItemCount() == 2)
    check("first layer active by default", panel.active_mask_index() == 0)
    check("layer names shown", panel._mask_list.topLevelItem(0).text(0) == "Sky")
    check("sliders enabled with a layer", panel._mask_sliders["Exposure2012"].isEnabled())

    # --- slider edits mutate the ACTIVE layer in place ---
    v_before = layer_a.version
    panel._mask_sliders["Exposure2012"].setValue(150)  # /100 -> 1.5
    check(
        "exposure slider writes into the active layer's adjustments",
        abs(layer_a.adjustments.get("Exposure2012", 0.0) - 1.5) < 1e-6,
    )
    check("slider edit bumps the layer version (cache invalidation)", layer_a.version > v_before)
    check("inactive layer untouched", "Exposure2012" not in layer_b.adjustments)

    # --- selection switches the slider target ---
    panel.select_mask_index(1)
    check("selection moves active index", panel.active_mask_index() == 1)
    check(
        "sliders resync to the newly selected layer (neutral)",
        panel._mask_sliders["Exposure2012"].value() == 0,
    )
    panel._mask_sliders["Saturation"].setValue(40)
    check(
        "edit after switching targets the new layer",
        abs(layer_b.adjustments.get("Saturation", 0.0) - 40.0) < 1e-6
        and "Saturation" not in layer_a.adjustments,
    )

    # --- delete/duplicate request the HOST with the active index ---
    panel._mask_del_btn.click()
    check("Delete emits with active index", fired["delete"] == [1])

    # The Masks-tab redesign removed the Duplicate button, but the handler,
    # the signal and main.py's connection all survive -- so the operation is
    # intact and simply has no way to be invoked. Drive the handler directly:
    # that keeps the host contract covered, and this file stops at an
    # AttributeError (skipping every check below) if it clicks a button that
    # no longer exists.
    panel._on_mask_duplicate_clicked()
    check("Duplicate emits with active index", fired["dup"] == [1])
    check(
        "Duplicate has no UI affordance (redesign left the handler orphaned)",
        not hasattr(panel, "_mask_dup_btn"),
    )

    # --- tool arming + dodge/burn mutual exclusion ---
    panel._mask_paint_btn.setChecked(True)
    check("Paint arms mask mode", panel.mask_layer_mode() == "paint")
    check("mode change signal fired with 'paint'", fired["mode"][-1] == "paint")

    panel._mask_erase_btn.setChecked(True)
    check("Erase disarms Paint (mutually exclusive)", panel.mask_layer_mode() == "erase")
    check("Paint button unchecked after Erase armed", not panel._mask_paint_btn.isChecked())

    panel.set_dodge_burn_mode("dodge")
    check("arming Dodge disarms the mask brush", panel.mask_layer_mode() is None)
    check("dodge armed", panel.dodge_burn_mode() == "dodge")

    panel._mask_paint_btn.setChecked(True)
    check("arming mask Paint disarms Dodge", panel.dodge_burn_mode() is None)
    check("mask paint armed", panel.mask_layer_mode() == "paint")

    panel.arm_mask_paint()
    check("arm_mask_paint is idempotent when already armed", panel.mask_layer_mode() == "paint")

    # --- invert + enable plumbing ---
    saves = []
    panel.editing_finished.connect(lambda adj: saves.append(True))
    panel.select_mask_index(0)
    v_before = layer_a.version
    panel._mask_invert_btn.setChecked(True)
    check("Invert writes layer.invert", layer_a.invert is True)
    check("Invert bumps version", layer_a.version > v_before)
    check("Invert triggers editing_finished (persist)", len(saves) >= 1)

    from PyQt6.QtCore import Qt

    # The redesign dropped the per-row check box: it doubled as an enable
    # toggle, which made a single click ambiguous ("did I select this row or
    # turn it off?"). Rows are now select-and-rename only.
    item0 = panel._mask_list.topLevelItem(0)
    check(
        "list rows are not user-checkable (no ambiguous select/disable click)",
        not (item0.flags() & Qt.ItemFlag.ItemIsUserCheckable),
    )
    item0.setText(0, "Renamed Sky")
    check("editing a row renames its layer", layer_a.name == "Renamed Sky")

    # layer.enabled is still honoured by the compositor and still persisted to
    # XMP, so the model contract is tested directly -- there is currently no
    # UI that can set it. See the note in the summary about the orphaned
    # enable toggle.
    layer_a.enabled = False
    layer_a.touch()
    check("disabled layer reads as empty (skipped by the pipeline)", layer_a.is_empty)
    layer_a.enabled = True
    layer_a.touch()
    check("re-enabled layer is non-empty again", not layer_a.is_empty)

    # --- AI mask buttons (raw_ai_masks) ---
    ai_requests = []
    panel.mask_ai_requested.connect(ai_requests.append)
    mode_before = panel.mask_layer_mode()
    panel._mask_ai_subject_btn.click()
    check("Subject emits mask_ai_requested('subject')", ai_requests == ["subject"], f"got {ai_requests}")
    panel._mask_ai_sky_btn.click()
    check("Sky emits mask_ai_requested('sky')", ai_requests == ["subject", "sky"], f"got {ai_requests}")

    # Subject/Sky are one-shot actions, not modes -- firing them must leave
    # whatever tool was armed exactly as it was.
    check(
        "Subject/Sky leave the armed tool unchanged",
        panel.mask_layer_mode() == mode_before,
        f"{mode_before!r} -> {panel.mask_layer_mode()!r}",
    )

    # Click IS a mode and shares mutual exclusion with Paint/Erase.
    panel._mask_ai_click_btn.setChecked(True)
    check("Click arms ai_click mode", panel.mask_layer_mode() == "ai_click")
    panel._mask_paint_btn.setChecked(True)
    check("Paint disarms Click", panel._mask_ai_click_btn.isChecked() is False)
    check("paint mode after switch", panel.mask_layer_mode() == "paint")
    panel._mask_ai_click_btn.setChecked(True)
    check("Click disarms Paint", panel._mask_paint_btn.isChecked() is False)
    check("ai_click mode after switch back", panel.mask_layer_mode() == "ai_click")

    # --- unbinding (file change / reset) ---
    panel.set_mask_layer_stack(None)
    check("unbinding clears the list", panel._mask_list.topLevelItemCount() == 0)
    check("unbinding disables sliders", not panel._mask_sliders["Exposure2012"].isEnabled())
    check("unbinding disarms tools", panel.mask_layer_mode() is None)

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
