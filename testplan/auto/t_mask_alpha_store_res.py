"""Brush alpha is stored at half linear resolution (mask_layers_xmp).

Guards the three things that make that safe: old full-resolution sidecars keep
loading, repeated save/load does not compound the downscale, and the alpha a
layer actually composites with is still correct at the render resolution.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import mask_layers_xmp as mx
from raw_mask_layers import MaskLayer, MaskLayerStack, resize_alpha_to


def _soft_blob(h, w, cy, cx, radius):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / float(radius)
    return np.clip(1.0 - d, 0.0, 1.0).astype(np.float32)


def _stack(alpha):
    layer = MaskLayer(alpha=alpha, name="Mask 1")
    layer.adjustments = {"Exposure2012": 0.5}
    return MaskLayerStack(layers=[layer])


def test_stored_at_half_resolution():
    h, w = 800, 1200
    alpha = _soft_blob(h, w, 400, 600, 300)
    layer = mx.deserialize_stack(mx.serialize_stack(_stack(alpha))).layers[0]
    assert layer.alpha.shape == (h // 2, w // 2), layer.alpha.shape
    print("stored at half resolution: OK")


def test_full_res_sidecar_still_loads():
    """A 3.0.x sidecar stored alpha at full resolution and said so nowhere."""
    h, w = 400, 600
    alpha = _soft_blob(h, w, 200, 300, 150)
    serial = mx.serialize_stack(_stack(alpha))
    # Rebuild the entry the old encoder would have written: no downscale.
    import json

    entries = json.loads(serial)
    entries[0]["alpha"] = mx._encode_alpha(alpha)
    layer = mx.deserialize_stack(json.dumps(entries)).layers[0]
    assert layer.alpha.shape == (h, w), layer.alpha.shape
    assert abs(layer.adjustments.get("Exposure2012", 0.0) - 0.5) < 1e-4
    err = np.abs(layer.alpha - alpha).max()
    assert err < 0.01, err
    print("full-resolution sidecar loads unchanged: OK")


def test_reencode_does_not_compound():
    """Save/load/save must not halve twice.

    main.py restores a layer to the working resolution before painting into
    it; this asserts the encoder is stable once that has happened.
    """
    h, w = 800, 1200
    alpha = _soft_blob(h, w, 400, 600, 300)
    layer = mx.deserialize_stack(mx.serialize_stack(_stack(alpha))).layers[0]
    assert layer.alpha.shape == (h // 2, w // 2)

    for _ in range(4):
        # What the paint path does: bring the layer back to working res first.
        layer.alpha = resize_alpha_to(layer.alpha, h, w).copy()
        layer = mx.deserialize_stack(mx.serialize_stack(_stack(layer.alpha))).layers[0]
        assert layer.alpha.shape == (h // 2, w // 2), layer.alpha.shape

    # After four round trips the mask must still be the same blob, not mush.
    restored = resize_alpha_to(layer.alpha, h, w)
    err = np.abs(restored - alpha)
    assert err.max() < 0.10, err.max()
    assert err.mean() < 0.005, err.mean()
    print(f"4 round trips: max err {err.max():.4f}, mean {err.mean():.5f}: OK")


def test_small_masks_stored_verbatim():
    small = _soft_blob(48, 48, 24, 24, 20)
    layer = mx.deserialize_stack(mx.serialize_stack(_stack(small))).layers[0]
    assert layer.alpha.shape == (48, 48), layer.alpha.shape
    print("small mask stored verbatim: OK")


def test_composites_at_render_resolution():
    h, w = 800, 1200
    alpha = _soft_blob(h, w, 400, 600, 300)
    layer = mx.deserialize_stack(mx.serialize_stack(_stack(alpha))).layers[0]
    at_full = layer.alpha_at(h, w)
    assert at_full.shape == (h, w), at_full.shape
    err = np.abs(at_full - alpha)
    assert err.max() < 0.05, err.max()
    # Half-resolution preview, the other size the compositor asks for.
    assert layer.alpha_at(h // 2, w // 2).shape == (h // 2, w // 2)
    print(f"composites at render res, max err {err.max():.4f}: OK")


if __name__ == "__main__":
    test_stored_at_half_resolution()
    test_full_res_sidecar_still_loads()
    test_reencode_does_not_compound()
    test_small_masks_stored_verbatim()
    test_composites_at_render_resolution()
    print("PASS")
