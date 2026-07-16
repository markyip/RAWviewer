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
    # Soft circular kernel: bbox corners stay near empty (not a filled square).
    x0, y0, x1, y1 = 75 - 20 - 1, 50 - 20 - 1, 75 + 20 + 2, 50 + 20 + 2
    corner = float(abs(mask.data[y0, x0]))
    center = float(abs(mask.data[50, 75]))
    check(
        "stamp is circular (bbox corner ≪ center)",
        center > 0 and corner < center * 0.08,
        f"corner={corner:.4f} center={center:.4f}",
    )

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

    # 5b. Edge-assist stamp: paint on dark subject next to bright neighbor
    # should not bleed strongly onto the bright side.
    split = np.zeros((120, 200), dtype=np.float32)
    split[:, 100:] = 0.85  # bright right half
    split[:, :100] = 0.15  # dark left half
    mask_e = DodgeBurnMask.empty(120, 200)
    # Brush centered on dark side, large enough to geometrically cover the bright side
    stamp_brush(
        mask_e, 90, 60, 40, 0.5, dodge=True, luminance=split, edge_assist=True
    )
    left = float(np.abs(mask_e.data[60, 70]).mean()) if False else float(np.abs(mask_e.data[55:65, 60:80]).mean())
    right = float(np.abs(mask_e.data[55:65, 120:140]).mean())
    check(
        "edge-assist attenuates across subject boundary",
        left > right * 2.5,
        f"left={left:.4f} right={right:.4f}",
    )
    mask_plain = DodgeBurnMask.empty(120, 200)
    stamp_brush(
        mask_plain, 90, 60, 40, 0.5, dodge=True, luminance=split, edge_assist=False
    )
    right_plain = float(np.abs(mask_plain.data[55:65, 120:140]).mean())
    check(
        "without edge-assist bright side receives more paint",
        right_plain > right * 1.5,
        f"assist={right:.4f} plain={right_plain:.4f}",
    )

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

    # 10. Regression: the mask must survive a render/save/export request
    # that only carries the panel's plain get_adjustments() dict (which has
    # no knowledge of the mask -- it lives in main.py state). Simulates
    # main.py._dodge_burn_overlay_adj without importing main.py (heavy Qt
    # app dependency): a "bare" adj dict from the panel must NOT apply the
    # mask on its own; only after folding in MASK_KEY/STRENGTH_KEY does it.
    def _render(img, adj):
        return linear_to_display_uint8(process_linear_edit_buffer(img, adj, preview=True), adj)

    bare_adj = dict(DEFAULT_ADJUSTMENTS)
    bare_adj[AS_SHOT_TEMP_KEY] = 5500.0
    out_bare = _render(flat, bare_adj)
    check(
        "bare panel adj (no mask folded in) is flat -- proves overlay is required",
        int(out_bare[50, 75, 0]) == int(out_bare[5, 5, 0]),
    )
    overlaid_adj = dict(bare_adj)
    overlaid_adj[MASK_KEY] = serial
    overlaid_adj[STRENGTH_KEY] = 1.75
    out_overlaid = _render(flat, overlaid_adj)
    check(
        "overlaid adj (mask folded in) actually differs -- overlay must be applied somewhere",
        int(out_overlaid[50, 75, 0]) != int(out_overlaid[5, 5, 0]),
    )

    # 11. Deserialize cache: repeated calls with the SAME serial string
    # must reuse the same decoded mask object (this is what makes the
    # per-instance gain cache actually pay off across render ticks -- see
    # _deserialize_mask_cached's docstring). Regression for the ~56ms/tick
    # cost measured before caching (resize+exp2 recomputed from scratch on
    # every unrelated slider tick once a mask existed).
    from raw_dodge_burn import _DESERIALIZE_CACHE, _deserialize_mask_cached

    _DESERIALIZE_CACHE.clear()
    m1 = _deserialize_mask_cached(serial)
    m2 = _deserialize_mask_cached(serial)
    check("deserialize cache returns the same object for the same string", m1 is m2)
    check("deserialize cache populated exactly once", len(_DESERIALIZE_CACHE) == 1)

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
