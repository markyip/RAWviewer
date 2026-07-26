#!/usr/bin/env python3
"""Linear / radial gradient masks -- headless.

Gradients are stored as PARAMETERS, not pixels: two points and a feather
describe one completely, which is what makes it re-draggable after a reload,
exact at any resolution, and ~180 bytes in the sidecar instead of a
frame-sized PNG. The tests below pin all three, because the tempting
shortcut -- bake an alpha at creation time -- silently loses every one of
them and still looks correct on the screen it was authored on.

The drag itself is driven through the view's own mouse handlers rather than by
calling the emit helpers, so the press/move/release routing is covered too.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "rawviewer_ui")
)

import numpy as np  # noqa: E402
from PyQt6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QMouseEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

import raw_mask_shapes as shapes  # noqa: E402
from mask_layers_xmp import deserialize_stack, serialize_stack  # noqa: E402
from raw_mask_layers import MaskLayer, MaskLayerStack, apply_mask_layers  # noqa: E402

H = W = 200


def _layer(kind, drag, **kw):
    return MaskLayer(
        np.zeros((128, 128), dtype=np.float32),
        kind=kind,
        params=shapes.params_from_drag(kind, *drag),
        **kw,
    )


# --------------------------------------------------------------- generation


def test_linear_grades_along_the_drag():
    layer = _layer(shapes.KIND_LINEAR, (0.5, 0.0, 0.5, 0.6))
    a = layer.alpha_at(H, W)
    assert a.shape == (H, W) and a.dtype == np.float32
    top = float(a[2, W // 2])
    mid = float(a[int(H * 0.30), W // 2])
    bottom = float(a[H - 3, W // 2])
    assert top < 0.05, f"top of a downward gradient should be clear, got {top:.3f}"
    assert bottom > 0.95, f"past the axis end should be full, got {bottom:.3f}"
    assert 0.2 < mid < 0.8, f"midpoint should be part-way, got {mid:.3f}"
    # Perpendicular direction must be uniform -- it is a *linear* gradient.
    row = a[int(H * 0.30)]
    assert float(row.max() - row.min()) < 0.02, "linear gradient varies across its axis"
    print(f"  OK   linear grades along the drag ({top:.2f} -> {mid:.2f} -> {bottom:.2f})")


def test_linear_direction_follows_the_drag():
    down = _layer(shapes.KIND_LINEAR, (0.5, 0.0, 0.5, 0.6)).alpha_at(H, W)
    up = _layer(shapes.KIND_LINEAR, (0.5, 1.0, 0.5, 0.4)).alpha_at(H, W)
    assert down[2, W // 2] < down[H - 3, W // 2], "downward drag did not grade downward"
    assert up[2, W // 2] > up[H - 3, W // 2], "upward drag did not grade upward"

    across = _layer(shapes.KIND_LINEAR, (0.0, 0.5, 0.6, 0.5)).alpha_at(H, W)
    assert across[H // 2, 2] < across[H // 2, W - 3], "horizontal drag did not grade sideways"
    print("  OK   direction follows the drag (down / up / across)")


def test_radial_is_centred_on_the_dragged_box():
    layer = _layer(shapes.KIND_RADIAL, (0.2, 0.2, 0.8, 0.8))
    a = layer.alpha_at(H, W)
    assert float(a[H // 2, W // 2]) > 0.95, "radial centre should be full strength"
    for corner in ((3, 3), (3, W - 4), (H - 4, 3), (H - 4, W - 4)):
        assert float(a[corner]) < 0.05, f"radial leaked to corner {corner}"

    # Off-centre drag must move the mask, not just resize it.
    off = _layer(shapes.KIND_RADIAL, (0.0, 0.0, 0.4, 0.4)).alpha_at(H, W)
    assert float(off[H // 5, W // 5]) > float(off[H // 2, W // 2]), (
        "radial did not follow the drag box's centre"
    )
    print("  OK   radial is inscribed in the dragged box")


def test_radial_bbox_is_tight_and_analytic():
    """The compositor's bbox-limited path is what keeps a tick in budget."""
    layer = _layer(shapes.KIND_RADIAL, (0.3, 0.3, 0.7, 0.7))
    y0, y1, x0, x1 = layer.bbox()
    assert (y1 - y0) < 128 and (x1 - x0) < 128, f"radial bbox is the whole frame: {layer.bbox()}"
    # Nothing outside the reported bbox may be non-zero.
    a = shapes.generate_alpha(layer.kind, layer.params, 128, 128)
    outside = a.copy()
    outside[y0:y1, x0:x1] = 0.0
    assert float(outside.max()) < 1e-6, "alpha is non-zero outside the analytic bbox"

    linear = _layer(shapes.KIND_LINEAR, (0.5, 0.0, 0.5, 0.5))
    assert linear.bbox() == (0, 128, 0, 128), "a linear gradient covers the frame"
    print("  OK   radial bbox is tight and encloses all coverage")


def test_resolution_independence():
    """The same mask must land identically on preview and export bases."""
    layer = _layer(shapes.KIND_RADIAL, (0.25, 0.25, 0.75, 0.75))
    small = layer.alpha_at(100, 100)
    large = layer.alpha_at(400, 400)
    import cv2

    upscaled = cv2.resize(small, (400, 400), interpolation=cv2.INTER_LINEAR)
    # Generated-at-size, not interpolated: agreement should be close but the
    # point is that the geometry matches, not the sampling.
    assert abs(float(large.mean()) - float(small.mean())) < 0.02, (
        f"coverage changed with resolution: {small.mean():.3f} vs {large.mean():.3f}"
    )
    assert float(np.abs(large - upscaled).mean()) < 0.05
    print("  OK   coverage is the same at 100px and 400px")


def test_parametric_layer_is_never_empty():
    """is_empty tests the alpha buffer, which a gradient does not use."""
    layer = _layer(shapes.KIND_LINEAR, (0.5, 0.0, 0.5, 0.5))
    assert not layer.is_empty, "a gradient was reported empty and would be skipped"
    print("  OK   a gradient is not mistaken for an empty layer")


def test_gradient_applies_its_adjustment():
    img = np.full((H, W, 3), 0.30, np.float32)
    layer = _layer(shapes.KIND_LINEAR, (0.5, 0.0, 0.5, 0.6),
                   adjustments={"Exposure2012": 1.5})
    out = apply_mask_layers(img.copy(), MaskLayerStack([layer]))
    assert float(out[2, W // 2, 0]) < 0.32, "gradient applied where it should be clear"
    assert float(out[H - 3, W // 2, 0]) > 0.6, "gradient did not apply at full coverage"
    print("  OK   the adjustment follows the gradient")


def test_fingerprint_changes_when_dragged():
    """Otherwise a drag would not invalidate the composite cache."""
    layer = _layer(shapes.KIND_RADIAL, (0.2, 0.2, 0.8, 0.8))
    before = layer.fingerprint()
    layer.params = shapes.params_from_drag(shapes.KIND_RADIAL, 0.1, 0.1, 0.5, 0.5)
    layer.touch()
    assert layer.fingerprint() != before, "fingerprint ignored a geometry change"
    print("  OK   moving a gradient invalidates its cache key")


# ------------------------------------------------------------ serialization


def test_round_trip_is_exact_and_small():
    for kind, drag in (
        (shapes.KIND_LINEAR, (0.5, 0.1, 0.5, 0.7)),
        (shapes.KIND_RADIAL, (0.2, 0.3, 0.7, 0.8)),
    ):
        layer = _layer(kind, drag, adjustments={"Exposure2012": 1.0}, name=kind)
        serial = serialize_stack(MaskLayerStack([layer]))
        assert len(serial) < 600, f"{kind} serial is {len(serial)} bytes -- pixels, not params?"

        back = deserialize_stack(serial)
        assert back is not None and len(back.layers) == 1
        restored = back.layers[0]
        assert restored.kind == kind, f"kind lost: {restored.kind}"
        assert restored.is_parametric
        assert np.allclose(restored.alpha_at(H, W), layer.alpha_at(H, W), atol=1e-6), (
            f"{kind} did not survive the round trip"
        )
        # Re-draggable: the params themselves came back, not just the pixels.
        for key, value in layer.params.items():
            assert abs(float(restored.params[key]) - float(value)) < 1e-4, (
                f"{kind} param {key} lost"
            )
    print("  OK   params round-trip exactly, in under 600 bytes")


def test_brush_layers_still_serialize_as_pixels():
    """The new branch must not swallow ordinary painted masks."""
    brush = MaskLayer.empty(64, 64, adjustments={"Exposure2012": 1.0})
    brush.alpha[20:40, 20:40] = 1.0
    brush.touch()
    back = deserialize_stack(serialize_stack(MaskLayerStack([brush])))
    assert back is not None
    restored = back.layers[0]
    assert restored.kind == "brush" and not restored.is_parametric
    assert float(restored.alpha.max()) > 0.9, "painted alpha lost"
    print("  OK   brush masks still serialize their pixels")


# ------------------------------------------------------------------- the drag


def _view():
    from rawviewer_ui.gpu_image_view import GpuImageView

    v = GpuImageView()
    v._has_pixmap = True
    v._img_w, v._img_h = 1000, 500
    v._view_pos_on_image = lambda pos=None: True
    v.mapToScene = lambda pt: QPointF(float(pt.x()), float(pt.y()))
    seen = []
    v.gradientDragged.connect(
        lambda kind, x0, y0, x1, y1, end: seen.append((kind, x0, y0, x1, y1, end))
    )
    return v, seen


def _press(v, x, y):
    v.mousePressEvent(
        QMouseEvent(
            QMouseEvent.Type.MouseButtonPress, QPointF(x, y), Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
        )
    )


def _move(v, x, y):
    v.mouseMoveEvent(
        QMouseEvent(
            QMouseEvent.Type.MouseMove, QPointF(x, y), Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
        )
    )


def _release(v, x, y):
    v.mouseReleaseEvent(
        QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease, QPointF(x, y), Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        )
    )


def test_drag_emits_normalised_updates_then_an_end():
    v, seen = _view()
    v.set_gradient_drag_kind("linear")
    assert v.gradient_drag_kind() == "linear"

    _press(v, 100, 50)
    assert seen == [], "press alone should not define a gradient"
    _move(v, 500, 250)
    _move(v, 900, 400)
    _release(v, 900, 400)

    assert len(seen) >= 3, f"expected updates then an end, got {seen}"
    assert [e[-1] for e in seen[:-1]] == [False] * (len(seen) - 1)
    assert seen[-1][-1] is True, "no terminating event -- the edit never settles"

    kind, x0, y0, x1, y1, _end = seen[-1]
    assert kind == "linear"
    # Normalised against _img_w/_img_h (1000x500), not raw view pixels.
    assert abs(x0 - 100 / 999.0) < 1e-3, f"x0 not normalised: {x0}"
    assert abs(y0 - 50 / 499.0) < 1e-3, f"y0 not normalised: {y0}"
    assert 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0
    print(f"  OK   drag emits {len(seen) - 1} live update(s) then one end event")


def test_disarming_stops_the_drag_being_claimed():
    v, seen = _view()
    v.set_gradient_drag_kind(None)
    assert v.gradient_drag_kind() is None
    _press(v, 100, 50)
    _move(v, 400, 200)
    assert seen == [], f"disarmed view still emitted a gradient: {seen}"
    print("  OK   a disarmed view claims nothing")


def test_panel_gradient_tools_are_exclusive():
    from rawviewer_ui.adjust_panel import ImageAdjustPanelWidget

    p = ImageAdjustPanelWidget()
    p.show()
    emitted = []
    p.mask_gradient_tool_changed.connect(emitted.append)

    p._mask_linear_btn.setChecked(True)
    assert p.gradient_tool() == "linear"
    p._mask_radial_btn.setChecked(True)
    assert p.gradient_tool() == "radial", "two gradient tools armed at once"
    assert not p._mask_linear_btn.isChecked()

    # Arming a brush must put the gradient away -- they share the press.
    p._mask_paint_btn.setChecked(True)
    assert p.gradient_tool() is None, "gradient stayed armed alongside the brush"

    p._mask_radial_btn.setChecked(True)
    p.disarm_gradient_tools()
    assert p.gradient_tool() is None
    assert emitted and emitted[-1] == ""
    print("  OK   gradient tools are exclusive with each other and the brush")


def main() -> int:
    test_linear_grades_along_the_drag()
    test_linear_direction_follows_the_drag()
    test_radial_is_centred_on_the_dragged_box()
    test_radial_bbox_is_tight_and_analytic()
    test_resolution_independence()
    test_parametric_layer_is_never_empty()
    test_gradient_applies_its_adjustment()
    test_fingerprint_changes_when_dragged()
    test_round_trip_is_exact_and_small()
    test_brush_layers_still_serialize_as_pixels()
    test_drag_emits_normalised_updates_then_an_end()
    test_disarming_stops_the_drag_being_claimed()
    test_panel_gradient_tools_are_exclusive()
    print("\nPASS t_mask_gradients")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
