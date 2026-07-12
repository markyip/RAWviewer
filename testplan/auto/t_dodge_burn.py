#!/usr/bin/env python3
"""Dodge & burn engine invariants: stamping, apply, edge-snap, persistence."""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ["RAWVIEWER_ENABLE_EDITING"] = "1"

import numpy as np  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from raw_dodge_burn import (
        DodgeBurnMask,
        MASK_KEY,
        STRENGTH_KEY,
        apply_dodge_burn,
        deserialize_mask,
        edge_snap_region,
        serialize_mask,
        stamp_brush,
    )

    # 1. Empty mask is a true no-op
    mask = DodgeBurnMask.empty(100, 150)
    check("empty mask reports empty", mask.is_empty)
    img = np.full((100, 150, 3), 0.2, dtype=np.float32)
    out = apply_dodge_burn(img, mask, 1.75)
    check("empty mask apply is identity", out is img)
    check("None mask apply is identity", apply_dodge_burn(img, None, 1.75) is img)

    # 2. Dodge brightens, burn darkens, effect localized to brush area
    stamp_brush(mask, 75, 50, 20, 0.4, dodge=True)
    check("stamp makes mask non-empty", not mask.is_empty)
    dodged = apply_dodge_burn(img, mask, 1.75)
    check("dodge brightens center", dodged[50, 75, 0] > img[50, 75, 0])
    check("dodge leaves far corner unchanged", np.allclose(dodged[2, 2], img[2, 2], atol=1e-4))

    mask_b = DodgeBurnMask.empty(100, 150)
    stamp_brush(mask_b, 75, 50, 20, 0.4, dodge=False)
    burned = apply_dodge_burn(img, mask_b, 1.75)
    check("burn darkens center", burned[50, 75, 0] < img[50, 75, 0])

    # 3. Repeated same-sign strokes accumulate (brush builds up like a real tool)
    mask_acc = DodgeBurnMask.empty(100, 150)
    stamp_brush(mask_acc, 75, 50, 20, 0.1, dodge=True)
    v1 = float(mask_acc.data[50, 75])
    stamp_brush(mask_acc, 75, 50, 20, 0.1, dodge=True)
    v2 = float(mask_acc.data[50, 75])
    check("repeated stamps accumulate", v2 > v1, f"v1={v1:.3f} v2={v2:.3f}")

    # 4. Mask value clipped, never explodes with many strokes
    mask_clip = DodgeBurnMask.empty(20, 20)
    for _ in range(50):
        stamp_brush(mask_clip, 10, 10, 8, 0.5, dodge=True)
    check("mask clipped to bound", mask_clip.data.max() <= 1.5 + 1e-6)

    # 5. Edge snap runs without error and stays finite
    luminance = np.random.RandomState(2).rand(100, 150).astype(np.float32)
    bbox = (55, 30, 95, 70)
    edge_snap_region(mask, luminance, bbox)
    check("edge-snap produces finite values", np.all(np.isfinite(mask.data)))
    check("edge-snap keeps mask non-empty", not mask.is_empty)

    # 6. Serialize round-trip (8-bit quantization tolerance)
    serial = serialize_mask(mask)
    check("serialize non-empty mask produces data", len(serial) > 0)
    back = deserialize_mask(serial)
    check("deserialize succeeds", back is not None)
    if back is not None:
        diff = np.abs(back.data - mask.data).max()
        check("round-trip within 8-bit quantization", diff < 0.02, f"maxdiff={diff:.4f}")
    check("serialize empty mask returns empty string", serialize_mask(DodgeBurnMask.empty(10, 10)) == "")
    check("deserialize empty string returns None", deserialize_mask("") is None)

    # 7. is_default_adjustments treats a present mask as non-default
    from raw_adjustments import DEFAULT_ADJUSTMENTS, AS_SHOT_TEMP_KEY, is_default_adjustments

    adj = dict(DEFAULT_ADJUSTMENTS)
    adj[AS_SHOT_TEMP_KEY] = 5500.0
    check("no mask -> is_default True", is_default_adjustments(adj))
    adj[MASK_KEY] = serial
    check("mask present -> is_default False", not is_default_adjustments(adj))

    # 8. Full XMP round-trip
    tmpd = tempfile.mkdtemp()
    try:
        from raw_adjustments import (
            load_adjustments_for_file,
            resolve_xmp_path,
            write_xmp_adjustments_for_file,
        )

        img_path = os.path.join(tmpd, "t.CR3")
        open(img_path, "wb").write(b"stub")
        adj[STRENGTH_KEY] = 2.1
        write_xmp_adjustments_for_file(img_path, adj)
        check("sidecar written for mask-only edit", os.path.isfile(resolve_xmp_path(img_path)))
        loaded = load_adjustments_for_file(img_path)
        check("mask string round-trips via XMP", loaded.get(MASK_KEY, "") == serial)
        check("strength round-trips via XMP", abs(float(loaded.get(STRENGTH_KEY, 0)) - 2.1) < 1e-6)
        loaded_mask = deserialize_mask(loaded.get(MASK_KEY, ""))
        check("mask deserializes after XMP round-trip", loaded_mask is not None)
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)

    # 9. Pipeline application (both edit-pipeline entry points)
    from raw_edit_pipeline import linear_to_display_uint8, process_linear_edit_buffer

    pipe_adj = {MASK_KEY: serial, STRENGTH_KEY: 1.75}
    flat = np.full((100, 150, 3), 0.05, dtype=np.float32)
    disp = linear_to_display_uint8(process_linear_edit_buffer(flat, pipe_adj, preview=True), pipe_adj)
    check(
        "process_linear_edit_buffer applies mask (touched region differs from untouched)",
        int(disp[50, 75, 0]) != int(disp[5, 5, 0]),
    )

    from raw_adjustments import apply_adjustments_to_rgb

    full_adj = dict(DEFAULT_ADJUSTMENTS)
    full_adj[MASK_KEY] = serial
    img8 = np.full((100, 150, 3), 60, dtype=np.uint8)
    out8 = apply_adjustments_to_rgb(img8, full_adj)
    check(
        "legacy sRGB path applies mask",
        int(out8[50, 75, 0]) != int(out8[5, 5, 0]),
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
