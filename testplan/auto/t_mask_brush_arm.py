#!/usr/bin/env python3
"""Mask brush arm/disarm + Adjust header hint clarity."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from PyQt6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv)

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def test_header_hints_name_the_tools():
    from rawviewer_ui.adjust_panel import ImageAdjustPanelWidget

    global_hint, masks_hint = ImageAdjustPanelWidget._HEADER_HINTS
    check(
        "Global hint names Dodge/Burn/Erase/Heal",
        "Dodge" in global_hint and "Burn" in global_hint and "Heal" in global_hint,
        global_hint,
    )
    check(
        "Global hint separates keys with slashes",
        "D / B / X / H" in global_hint,
        global_hint,
    )
    check(
        "Masks hint is tap-to-toggle, not hold-to-paint",
        "hold P" not in masks_hint.lower() and "P = Add" in masks_hint,
        masks_hint,
    )
    check(
        "Masks hint mentions putting the brush away",
        "put away" in masks_hint.lower() or "again" in masks_hint.lower(),
        masks_hint,
    )


def test_toggle_mask_layer_mode():
    from raw_mask_layers import MaskLayer, MaskLayerStack
    from rawviewer_ui.adjust_panel import ImageAdjustPanelWidget

    p = ImageAdjustPanelWidget()
    p.show()
    p.show_masks_tab()
    stack = MaskLayerStack([MaskLayer.empty(32, 32, name="M1")])
    p.set_mask_layer_stack(stack)
    p.select_mask_index(0)

    check("starts disarmed", p.mask_layer_mode() is None)
    check("P arms paint", p.toggle_mask_layer_mode("paint") is True)
    check("mode is paint", p.mask_layer_mode() == "paint")
    check("P again disarms", p.toggle_mask_layer_mode("paint") is False)
    check("mode cleared", p.mask_layer_mode() is None)

    p.toggle_mask_layer_mode("paint")
    check("X switches to erase", p.toggle_mask_layer_mode("erase") is True)
    check("mode is erase", p.mask_layer_mode() == "erase")
    check("X again disarms", p.toggle_mask_layer_mode("erase") is False)


def test_brush_create_does_not_arm_paint():
    """Create→Brush makes a mask; P (or Add) is what arms painting."""
    from raw_mask_layers import MaskLayer, MaskLayerStack
    from rawviewer_ui.adjust_panel import ImageAdjustPanelWidget

    p = ImageAdjustPanelWidget()
    p.show()
    p.show_masks_tab()

    created = []

    def _on_add():
        stack = getattr(p, "_mask_stack", None) or MaskLayerStack()
        stack.layers.append(MaskLayer.empty(32, 32, name=f"M{len(stack.layers)+1}"))
        p.set_mask_layer_stack(stack)
        p.select_mask_index(len(stack.layers) - 1)
        created.append(1)

    p.mask_add_requested.connect(_on_add)
    p._on_mask_create_clicked("brush")
    check("Brush creates a mask", len(created) == 1)
    check("Brush leaves paint disarmed", p.mask_layer_mode() is None)
    check("a mask row is selected", p.active_mask_index() == 0)

    check("P then arms paint", p.toggle_mask_layer_mode("paint") is True)
    check("mode is paint after P", p.mask_layer_mode() == "paint")


def test_host_add_does_not_arm_paint():
    import main as mainmod
    from rawviewer_ui.adjust_panel import ImageAdjustPanelWidget

    panel = ImageAdjustPanelWidget()
    panel.show()
    panel.show_masks_tab()

    host = type("H", (), {})()
    host.single_image_adjust_panel = panel
    host._mask_layer_stack = None
    host._dodge_burn_mask_shape = lambda: (32, 32)
    host.status_bar = None
    host._on_mask_layer_add = (
        mainmod.RAWImageViewer._on_mask_layer_add.__get__(host)
    )
    panel.mask_add_requested.connect(host._on_mask_layer_add)

    panel._on_mask_create_clicked("brush")
    check("host add created a layer", host._mask_layer_stack is not None)
    check(
        "host add did not arm paint",
        panel.mask_layer_mode() is None,
    )


def test_mask_key_toggle_via_host():
    import main as mainmod
    from raw_mask_layers import MaskLayer, MaskLayerStack
    from rawviewer_ui.adjust_panel import ImageAdjustPanelWidget

    class _View:
        def __init__(self):
            self.latched = 0
            self.ended = 0

        def set_dodge_burn_mode(self, *_a, **_k):
            pass

        def set_click_tool_mode(self, *_a, **_k):
            pass

        def begin_latched_paint(self):
            self.latched += 1

        def begin_key_paint(self):
            self.latched += 1

        def end_key_paint(self):
            self.ended += 1

        def set_dodge_burn_brush_radius(self, *_a, **_k):
            pass

        def set_dodge_burn_brush_flow(self, *_a, **_k):
            pass

        def set_dodge_burn_brush_feather(self, *_a, **_k):
            pass

        def update_dodge_burn_mask(self, *_a, **_k):
            pass

        def update_mask_layer_overlay(self, *_a, **_k):
            pass

    panel = ImageAdjustPanelWidget()
    panel.show()
    panel.show_masks_tab()
    stack = MaskLayerStack([MaskLayer.empty(32, 32, name="M1")])
    panel.set_mask_layer_stack(stack)
    panel.select_mask_index(0)

    host = type("H", (), {})()
    host.single_image_adjust_panel = panel
    host.gpu_view = _View()
    host._brush_key_held = None
    host._dodge_burn_luma_guide = None
    host._dodge_burn_chroma_guide = None
    host._mask_layer_stack = stack
    host._show_status = lambda *_a, **_k: None
    host.ensure_mask_layer_for_painting = (
        mainmod.RAWImageViewer.ensure_mask_layer_for_painting.__get__(host)
    )
    host._handle_mask_brush_key_down = (
        mainmod.RAWImageViewer._handle_mask_brush_key_down.__get__(host)
    )
    host._handle_brush_key_up = (
        mainmod.RAWImageViewer._handle_brush_key_up.__get__(host)
    )
    host._disarm_adjust_brushes_if_any = (
        mainmod.RAWImageViewer._disarm_adjust_brushes_if_any.__get__(host)
    )
    host._on_mask_layer_mode_changed = (
        mainmod.RAWImageViewer._on_mask_layer_mode_changed.__get__(host)
    )
    host._any_brush_tool_armed = (
        mainmod.RAWImageViewer._any_brush_tool_armed.__get__(host)
    )
    host._sync_dodge_burn_brush_cursor = (
        mainmod.RAWImageViewer._sync_dodge_burn_brush_cursor.__get__(host)
    )
    host._sync_mask_layer_overlay = lambda **_k: None
    panel.mask_layer_mode_changed.connect(host._on_mask_layer_mode_changed)

    check(
        "P arms mask paint",
        host._handle_mask_brush_key_down(panel, "mask_paint") is True,
    )
    check("panel armed after P", panel.mask_layer_mode() == "paint")
    check(
        "P release does not disarm mask brush",
        host._handle_brush_key_up("mask_paint") is False,
    )
    check("still armed after P release", panel.mask_layer_mode() == "paint")
    check(
        "P again disarms",
        host._handle_mask_brush_key_down(panel, "mask_paint") is True,
    )
    check("disarmed after second P", panel.mask_layer_mode() is None)

    panel.toggle_mask_layer_mode("paint")
    check("Esc helper reports armed", host._disarm_adjust_brushes_if_any() is True)
    check("Esc helper cleared the brush", panel.mask_layer_mode() is None)
    check("Esc helper is idempotent", host._disarm_adjust_brushes_if_any() is False)


def test_p_shortcut_while_adjust_arms_mask_paint():
    """ApplicationShortcut P must not swallow Adjust's mask-Add binding."""
    import main as mainmod

    src = open(
        os.path.join(os.path.dirname(__file__), "..", "..", "src", "main.py"),
        encoding="utf-8",
    ).read()
    check(
        "raw-recovery shortcut disabled while Adjust is open",
        'sc.setEnabled(not bool(self._adjust_overlay_visible))' in src
        or 'setEnabled(not bool(self._adjust_overlay_visible))' in src,
    )
    check(
        "raw-recovery activate routes to mask paint while Adjust is open",
        '_handle_brush_key_down("mask_paint")' in src
        and "_shortcut_activate_raw_recovery" in src,
    )


def main() -> int:
    test_header_hints_name_the_tools()
    test_toggle_mask_layer_mode()
    test_brush_create_does_not_arm_paint()
    test_host_add_does_not_arm_paint()
    test_mask_key_toggle_via_host()
    test_p_shortcut_while_adjust_arms_mask_paint()
    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
