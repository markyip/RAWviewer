#!/usr/bin/env python3
"""Mask Editing / Local Adjustments -- XMP persistence (own schema).

Checks mask_layers_xmp.py's serialize/deserialize round-trip and its
wiring into raw_adjustments.py's write_xmp_adjustments / load_adjustments
path (same shape as t_dodge_burn.py / t_spot_heal.py's XMP checks, own
crs:RVMaskLayers child element rather than reusing DodgeBurnMask/SpotHealMask).
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    os.environ["RAWVIEWER_ENABLE_EDITING"] = "1"
    from mask_layers_xmp import deserialize_stack, serialize_stack
    from raw_mask_layers import MASK_LAYERS_KEY, MASK_LAYERS_OBJ_KEY, MaskLayer, MaskLayerStack

    # --- serialize/deserialize round-trip ---
    check("serialize(None) returns empty string", serialize_stack(None) == "")
    check("serialize(empty stack) returns empty string", serialize_stack(MaskLayerStack()) == "")
    check("deserialize('') returns None", deserialize_stack("") is None)
    check("deserialize(garbage) returns None", deserialize_stack("{not json") is None)

    h, w = 48, 64
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    alpha1 = np.clip(1.0 - np.sqrt((xx - 20) ** 2 + (yy - 20) ** 2) / 15.0, 0.0, 1.0).astype(np.float32)
    alpha2 = np.clip(1.0 - np.sqrt((xx - 45) ** 2 + (yy - 30) ** 2) / 10.0, 0.0, 1.0).astype(np.float32)
    layer1 = MaskLayer(
        alpha1,
        adjustments={"Exposure2012": 0.85, "Contrast2012": -12.0, "Temperature": 6200.0},
        name="Sky",
        invert=False,
    )
    layer2 = MaskLayer(
        alpha2,
        adjustments={"Saturation": 30.0, "Dehaze": -20.0},
        name="Subject",
        invert=True,
        enabled=False,
    )
    stack = MaskLayerStack(layers=[layer1, layer2])

    serial = serialize_stack(stack)
    check("non-empty stack serializes to a non-empty string", bool(serial))

    restored = deserialize_stack(serial)
    check("deserialize returns a stack", restored is not None)
    check("layer count round-trips", restored is not None and len(restored.layers) == 2)

    if restored is not None and len(restored.layers) == 2:
        r1, r2 = restored.layers
        check("layer name round-trips", r1.name == "Sky" and r2.name == "Subject")
        check("layer invert round-trips", r1.invert is False and r2.invert is True)
        check("layer enabled round-trips", r1.enabled is True and r2.enabled is False)
        check(
            "layer adjustments round-trip within rounding",
            abs(r1.adjustments.get("Exposure2012", 0.0) - 0.85) < 1e-3
            and abs(r1.adjustments.get("Temperature", 0.0) - 6200.0) < 1e-3
            and abs(r2.adjustments.get("Dehaze", 0.0) - (-20.0)) < 1e-3,
        )
        # Alpha is quantized to 8-bit -- allow one quantization step of slack.
        alpha_diff = float(np.max(np.abs(r1.alpha - alpha1)))
        check("layer alpha round-trips within 8-bit quantization", alpha_diff <= 1.0 / 255.0 + 1e-4, f"maxdiff={alpha_diff:.5f}")

    # A zero-valued adjustment key must not round-trip back as an explicit
    # near-zero float that then reads as "active" by a naive >0 check.
    only_default = MaskLayer(alpha1.copy(), adjustments={"Exposure2012": 0.0})
    only_default_serial = serialize_stack(MaskLayerStack(layers=[only_default]))
    restored_default = deserialize_stack(only_default_serial)
    check(
        "a layer with only-zero adjustments still round-trips (alpha alone makes it non-empty)",
        restored_default is not None and len(restored_default.layers) == 1,
    )

    # --- raw_adjustments.py wiring: write -> read back through the real XMP path ---
    from raw_adjustments import (
        is_default_adjustments,
        load_adjustments_from_xmp,
        write_xmp_adjustments,
    )

    with tempfile.TemporaryDirectory() as tmp:
        xmp_path = os.path.join(tmp, "test.xmp")
        adj = {MASK_LAYERS_KEY: serial}
        write_xmp_adjustments(xmp_path, adj)
        check("sidecar written for a mask-layers-only edit", os.path.isfile(xmp_path))

        loaded = load_adjustments_from_xmp(xmp_path)
        loaded_serial = loaded.get(MASK_LAYERS_KEY, "")
        check("mask layers serial round-trips via XMP", bool(loaded_serial))

        loaded_stack = deserialize_stack(loaded_serial) if loaded_serial else None
        check(
            "mask layers deserialize correctly after an XMP round-trip",
            loaded_stack is not None and len(loaded_stack.layers) == 2,
        )

        check("is_default_adjustments is False when a mask-layers serial is present", not is_default_adjustments({MASK_LAYERS_KEY: serial}))
        check("is_default_adjustments is False when a live mask-layers object is present", not is_default_adjustments({MASK_LAYERS_OBJ_KEY: stack}))
        check("is_default_adjustments is True with neither present", is_default_adjustments({}))

    # Foreign-content preservation: writing mask layers must not disturb an
    # unrelated crs: attribute already in the file (same guarantee dodge/
    # burn and spot heal already rely on via _strip_our_crs_fields).
    with tempfile.TemporaryDirectory() as tmp:
        xmp_path = os.path.join(tmp, "test2.xmp")
        write_xmp_adjustments(xmp_path, {"Exposure2012": 1.5})
        write_xmp_adjustments(xmp_path, {"Exposure2012": 1.5, MASK_LAYERS_KEY: serial})
        loaded2 = load_adjustments_from_xmp(xmp_path)
        check(
            "writing mask layers preserves a pre-existing unrelated adjustment",
            abs(float(loaded2.get("Exposure2012", 0.0)) - 1.5) < 1e-3,
        )
        check("mask layers serial present alongside the unrelated adjustment", bool(loaded2.get(MASK_LAYERS_KEY, "")))

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
