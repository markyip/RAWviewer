"""Grouped masks: components combine into one alpha, one adjustment set.

Lightroom's model. A group owns the adjustments; its components own only
coverage, combined by each one's blend. That is why dropping one mask onto
another discards the dragged mask's adjustments -- a component cannot hold
any.

The format property worth guarding hardest is that a build predating groups
still renders a grouped mask correctly: the entry carries both the baked
combined alpha (which such a build reads) and the components (which it
ignores). Nesting instead would have left it with no alpha and it would have
dropped the mask silently.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from raw_mask_layers import MaskLayer, MaskLayerStack, apply_mask_layers  # noqa: E402
import mask_layers_xmp as mx  # noqa: E402

H, W = 200, 300
FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def blob(cy, cx, r):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return np.clip(1 - np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / r, 0, 1).astype(
        np.float32
    )


def cover(a):
    return float((a > 0.01).mean())


def _group(components, **kw):
    g = MaskLayer(np.zeros((H, W), np.float32), components=components, **kw)
    return g


def test_add_is_union_not_sum():
    a = MaskLayer(blob(100, 140, 70), name="A")
    b = MaskLayer(blob(100, 160, 70), name="B")  # deliberate overlap
    g = _group([a, b], name="G")
    al = g.alpha_at(H, W)

    check("group reports itself as a group", g.is_group)
    check(
        "add widens coverage",
        cover(al) > cover(a.alpha) + 0.01,
        f"{cover(al):.3f} vs {cover(a.alpha):.3f}",
    )
    # Union, not sum: overlapping strokes must not exceed full coverage.
    check("alpha stays within [0,1]", float(al.max()) <= 1.0 + 1e-6, f"max={al.max():.3f}")
    overlap = np.minimum(a.alpha, b.alpha) > 0.5
    check(
        "overlap is covered once, not twice",
        bool(np.allclose(al[overlap], np.maximum(a.alpha, b.alpha)[overlap], atol=1e-5)),
    )


def test_subtract_removes_coverage():
    a = MaskLayer(blob(100, 150, 80), name="A")
    b = MaskLayer(blob(100, 150, 40), name="B", blend="subtract")
    g = _group([a, b], name="G")
    al = g.alpha_at(H, W)
    # Mean alpha, not thresholded coverage: a soft eraser thins the mask
    # long before it drops any pixel under the coverage threshold, so
    # coverage barely moves and would pass on a no-op implementation.
    check(
        "subtract removes weight from the mask",
        float(al.mean()) < float(a.alpha.mean()) * 0.9,
        f"mean {al.mean():.4f} vs {a.alpha.mean():.4f}",
    )
    check("centre is fully removed", float(al[100, 150]) < 0.01, f"{al[100,150]:.3f}")
    check("outside the eraser survives", float(al[100, 220]) > 0.1)


def test_first_component_always_adds():
    """A mask whose first component subtracts would be empty forever."""
    a = MaskLayer(blob(100, 150, 60), name="A", blend="subtract")
    g = _group([a], name="G")
    check("leading subtract still contributes", cover(g.alpha_at(H, W)) > 0.01)


def test_disabled_component_is_skipped():
    a = MaskLayer(blob(100, 90, 50), name="A")
    b = MaskLayer(blob(100, 210, 50), name="B")
    g = _group([a, b], name="G")
    both = cover(g.alpha_at(H, W))
    b.enabled = False
    g._group_alpha_cache = None
    check("disabling a component shrinks the group", cover(g.alpha_at(H, W)) < both)


def test_empty_and_bbox():
    a = MaskLayer(np.zeros((H, W), np.float32), name="A")
    g = _group([a], name="G")
    check("group of empty components is empty", g.is_empty)
    a.alpha[:] = blob(100, 150, 40)
    a.touch()
    check("group is non-empty once a component is painted", not g.is_empty)

    b = MaskLayer(blob(100, 260, 30), name="B")
    g2 = _group([a, b], name="G2")
    bb = g2.bbox()
    check("group bbox spans both components", bb is not None and bb[3] > 240, str(bb))

    # A subtracting component cannot widen where the group applies.
    c = MaskLayer(blob(20, 20, 15), name="C", blend="subtract")
    g3 = _group([a, c], name="G3")
    check(
        "subtracting component does not widen the bbox",
        g3.bbox()[2] > 30,
        str(g3.bbox()),
    )


def test_fingerprint_tracks_components():
    a = MaskLayer(blob(100, 150, 50), name="A")
    g = _group([a], name="G")
    before = g.fingerprint()
    a.touch()
    check("editing a component moves the group fingerprint", g.fingerprint() != before)
    before = g.fingerprint()
    a.blend = "subtract"
    check("changing a blend moves the fingerprint", g.fingerprint() != before)


def test_inverted_group_covers_the_frame():
    """frame_shape() must come from a component, not the placeholder alpha."""
    a = MaskLayer(blob(100, 150, 40), name="A")
    g = _group([a], name="G", invert=True)
    check("inverted group frame is the component's", g.frame_shape() == (H, W), str(g.frame_shape()))
    eb = g.effective_bbox()
    check("inverted group applies over the whole frame", eb == (0, H, 0, W), str(eb))


def test_compositor_applies_a_group_once():
    a = MaskLayer(blob(100, 90, 50), name="A")
    b = MaskLayer(blob(100, 210, 50), name="B")
    g = _group([a, b], name="G")
    g.adjustments = {"Exposure2012": 1.0}
    img = np.full((H, W, 3), 0.25, np.float32)
    out = apply_mask_layers(img, MaskLayerStack(layers=[g]))
    check("compositor renders a group", not np.allclose(img, out))
    check("brightened where covered", float(out[100, 90, 0]) > 0.3, f"{out[100,90,0]:.3f}")
    check("untouched where not covered", abs(float(out[10, 150, 0]) - 0.25) < 0.01)


def test_xmp_roundtrip():
    a = MaskLayer(blob(100, 90, 60), name="Brush 1")
    b = MaskLayer(blob(100, 210, 45), name="Brush 2", blend="subtract")
    g = _group([a, b], name="Sky group")
    g.adjustments = {"Exposure2012": 0.8, "Saturation": 20.0}

    serial = mx.serialize_stack(MaskLayerStack(layers=[g]))
    back = mx.deserialize_stack(serial).layers[0]

    check("round-trips as a group", back.is_group)
    check("component count preserved", len(back.components) == 2)
    check("component kinds preserved", [c.kind for c in back.components] == ["brush", "brush"])
    check("component blends preserved", [c.blend for c in back.components] == ["add", "subtract"])
    check("group adjustments preserved", abs(back.adjustments.get("Exposure2012", 0) - 0.8) < 1e-4)
    check(
        "components carry no adjustments",
        all(not c.adjustments for c in back.components),
    )
    err = float(np.abs(g.alpha_at(H, W) - back.alpha_at(H, W)).max())
    check("coverage survives the round trip", err < 0.05, f"max err {err:.4f}")


def test_old_build_still_renders_a_group():
    """The whole reason the baked alpha is stored alongside the components."""
    a = MaskLayer(blob(100, 90, 60), name="A")
    b = MaskLayer(blob(100, 210, 60), name="B")
    g = _group([a, b], name="G")
    g.adjustments = {"Exposure2012": 0.8}
    serial = mx.serialize_stack(MaskLayerStack(layers=[g]))

    entries = json.loads(serial)
    check("entry carries a baked alpha for old readers", bool(entries[0].get("alpha")))
    for e in entries:
        e.pop("components", None)  # what a pre-grouping build effectively sees
    old = mx.deserialize_stack(json.dumps(entries)).layers[0]

    check("old build sees a flat layer", not old.is_group)
    check("old build keeps the adjustments", abs(old.adjustments.get("Exposure2012", 0) - 0.8) < 1e-4)
    err = float(np.abs(g.alpha_at(H, W) - old.alpha_at(H, W)).max())
    check("old build renders the same coverage", err < 0.05, f"max err {err:.4f}")


def test_malformed_components_fall_back_not_vanish():
    a = MaskLayer(blob(100, 150, 50), name="A")
    g = _group([a], name="G")
    serial = mx.serialize_stack(MaskLayerStack(layers=[g]))
    entries = json.loads(serial)
    entries[0]["components"] = ["not a dict", 42]
    stack = mx.deserialize_stack(json.dumps(entries))
    check("a malformed component list still yields a mask", stack is not None and len(stack.layers) == 1)
    if stack and stack.layers:
        check("it falls back to the baked alpha", cover(stack.layers[0].alpha_at(H, W)) > 0.01)


def main() -> int:
    for fn in (
        test_add_is_union_not_sum,
        test_subtract_removes_coverage,
        test_first_component_always_adds,
        test_disabled_component_is_skipped,
        test_empty_and_bbox,
        test_fingerprint_tracks_components,
        test_inverted_group_covers_the_frame,
        test_compositor_applies_a_group_once,
        test_xmp_roundtrip,
        test_old_build_still_renders_a_group,
        test_malformed_components_fall_back_not_vanish,
    ):
        fn()
    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
