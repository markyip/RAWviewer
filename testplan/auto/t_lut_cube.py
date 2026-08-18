#!/usr/bin/env python3
"""Creative .cube LUT parse / apply smoke tests."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def _write_identity_cube(path: str, size: int = 3) -> None:
    lines = [f"TITLE \"Identity\"", f"LUT_3D_SIZE {size}"]
    for b in range(size):
        for g in range(size):
            for r in range(size):
                lines.append(
                    f"{r / (size - 1):.6f} {g / (size - 1):.6f} {b / (size - 1):.6f}"
                )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_scale_cube(path: str, scale, size: int = 5) -> None:
    sr, sg, sb = scale
    lines = [f'TITLE "Scale"', f"LUT_3D_SIZE {size}"]
    for b in range(size):
        for g in range(size):
            for r in range(size):
                lines.append(
                    f"{min(1.0, sr * r / (size - 1)):.6f} "
                    f"{min(1.0, sg * g / (size - 1)):.6f} "
                    f"{min(1.0, sb * b / (size - 1)):.6f}"
                )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    from raw_lut import apply_cube_lut, apply_creative_lut, parse_cube_file

    with tempfile.TemporaryDirectory() as td:
        cube = os.path.join(td, "id.cube")
        _write_identity_cube(cube, 5)
        lut = parse_cube_file(cube)
        check("parse size", lut.size == 5)
        check("parse data shape", lut.data.shape == (5, 5, 5, 3))

        img = np.zeros((8, 8, 3), np.float32)
        img[..., 0] = 0.4
        img[..., 1] = 0.2
        img[..., 2] = 0.1
        out = apply_cube_lut(img, lut, amount=100.0)
        check(
            "identity LUT ≈ input",
            float(np.max(np.abs(out - img))) < 0.02,
            f"max_delta={float(np.max(np.abs(out - img))):.4f}",
        )
        half = apply_cube_lut(img, lut, amount=0.0)
        check("amount 0 is no-op", np.allclose(half, img))

        # Managed apply path with missing name is no-op.
        check(
            "missing LUT name no-op",
            np.allclose(apply_creative_lut(img, {"CreativeLUTAmount": 100.0}), img),
        )

        from raw_lut import (
            LUT_SPACE_LINEAR,
            LUT_SPACE_REC709,
            apply_lut_to_preview_uint8,
            apply_pipeline_lut_encoded,
            apply_pipeline_lut_linear,
            downscale_rgb_for_lut_preview,
            lut_preview_cache_key,
            lut_working_space,
            render_lut_chooser_previews,
        )

        check("default space Rec709", lut_working_space({}) == LUT_SPACE_REC709)
        check(
            "linear space parse",
            lut_working_space({"CreativeLUTWorkingSpace": "Linear"}) == LUT_SPACE_LINEAR,
        )
        # Without a managed name, pipeline helpers are no-ops in both spaces.
        enc = (img * 255).astype(np.uint8)
        check(
            "pipeline encoded no-op without name",
            np.array_equal(
                apply_pipeline_lut_encoded(enc, {"CreativeLUTAmount": 100}, 255.0),
                enc,
            ),
        )
        check(
            "pipeline linear no-op without name",
            np.allclose(
                apply_pipeline_lut_linear(img, {"CreativeLUTWorkingSpace": "Linear"}),
                img,
            ),
        )

        big = np.zeros((200, 120, 3), np.uint8)
        big[..., 0] = 180
        big[..., 1] = 90
        big[..., 2] = 40
        tiny = downscale_rgb_for_lut_preview(big, long_edge=80)
        check(
            "preview long edge ≤ 80",
            max(int(tiny.shape[0]), int(tiny.shape[1])) <= 80,
            f"shape={tiny.shape}",
        )
        key_a = lut_preview_cache_key("/tmp/a.nef#3", "warm.cube", 100)
        key_b = lut_preview_cache_key("/tmp/a.nef#3", "cool.cube", 100)
        check("cache key includes lut_id", key_a != key_b)
        check("cache key includes amount", key_a[2] == 100)

        red_cube = os.path.join(td, "red.cube")
        blue_cube = os.path.join(td, "blue.cube")
        _write_scale_cube(red_cube, (1.6, 0.4, 0.4))
        _write_scale_cube(blue_cube, (0.4, 0.4, 1.6))
        import raw_lut as _rl

        prev_dir = _rl.luts_dir
        _rl.luts_dir = lambda: td
        try:
            previews = render_lut_chooser_previews(
                tiny, ["red.cube", "blue.cube"], amount=100.0, space=LUT_SPACE_REC709
            )
            r = previews["red.cube"]
            b = previews["blue.cube"]
            ident = previews[""]
            check("identity thumb present", ident.shape == tiny.shape)
            check(
                "two LUT thumbs differ",
                float(np.mean(np.abs(r.astype(np.float32) - b.astype(np.float32)))) > 5.0,
                f"mae={float(np.mean(np.abs(r.astype(np.float32) - b.astype(np.float32)))):.2f}",
            )
            direct_r = apply_lut_to_preview_uint8(tiny, "red.cube", 100.0, LUT_SPACE_REC709)
            check("named preview matches helper", np.array_equal(r, direct_r))
            check("thumbs render at 100%", _rl.LUT_PREVIEW_AMOUNT == 100.0)
        finally:
            _rl.luts_dir = prev_dir

        from raw_adjustments import AS_SHOT_TEMP_KEY, is_default_adjustments
        from raw_lut import lut_chooser_as_shot_display_u8, lut_chooser_identity_adj

        graded = {
            "Exposure2012": 1.75,
            "Contrast2012": 40.0,
            "Temperature": 8000.0,
            AS_SHOT_TEMP_KEY: 5200.0,
            "CreativeLUTName": "red.cube",
            "CreativeLUTAmount": 40.0,
        }
        ident = lut_chooser_identity_adj(graded)
        check(
            "identity adj ignores Exposure",
            abs(float(ident.get("Exposure2012", 0.0))) < 1e-9,
            f"got {ident}",
        )
        check(
            "identity adj ignores live LUT amount",
            abs(float(ident.get("CreativeLUTAmount", 0.0))) < 1e-9,
        )
        check(
            "identity adj is as-shot default",
            is_default_adjustments(ident),
        )

        lin = np.full((48, 48, 3), 0.22, dtype=np.float32)
        lin[..., 1] = 0.18
        lin[..., 2] = 0.12
        u8_graded = lut_chooser_as_shot_display_u8(lin, graded)
        u8_plain = lut_chooser_as_shot_display_u8(lin, {})
        check(
            "as-shot thumb helper ignores Exposure in adj",
            u8_graded.shape == u8_plain.shape
            and float(np.max(np.abs(u8_graded.astype(np.int16) - u8_plain.astype(np.int16))))
            == 0.0,
        )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
