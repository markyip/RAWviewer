#!/usr/bin/env python3
"""Edit-base decode: Reconstruct + browse-brightness gain, not P-key, not Blend."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def _mode_int(mode) -> int:
    return int(getattr(mode, "value", mode))


def main() -> int:
    from fast_raw_decode import params_supported, params_supported_half
    from raw_tone_recovery import (
        edit_base_decode_params,
        edit_base_highlight_brightness_gain,
        linearize_reconstruct_edit_base,
        recovery_decode_params,
    )

    edit = edit_base_decode_params(half_size=True, demosaic="AHD")
    recov = recovery_decode_params(half_size=True)

    check("edit-base uses Reconstruct (not Blend=2)", _mode_int(edit["highlight_mode"]) >= 3)
    check("P-key also Reconstructs", _mode_int(recov["highlight_mode"]) >= 3)
    check("edit-base has no exp_shift", "exp_shift" not in edit)
    check("edit-base has no exp_preserve_highlights", "exp_preserve_highlights" not in edit)
    check("P-key keeps exp_shift=1.0", recov.get("exp_shift") == 1.0)
    check(
        "P-key keeps exp_preserve_highlights=0.45",
        recov.get("exp_preserve_highlights") == 0.45,
    )
    check("edit-base stays 16-bit linear", edit.get("output_bps") == 16 and edit.get("gamma") == (1, 1))
    check("edit-base no_auto_bright", edit.get("no_auto_bright") is True)

    # Fast path must not pretend to Reconstruct (it only implements 0–2).
    check(
        "fast path rejects Reconstruct edit-base params",
        not params_supported_half(edit, return_linear=True),
    )
    # Explicit replica contract: mode 5 is unsupported even without demosaic.
    core = {
        "use_camera_wb": True,
        "use_auto_wb": False,
        "output_bps": 16,
        "gamma": (1, 1),
        "no_auto_bright": True,
        "bright": 1.0,
        "user_flip": 0,
        "highlight_mode": 5,
    }
    check("params_supported rejects highlight_mode=5", not params_supported(core, return_linear=True))
    core["highlight_mode"] = 2
    check("params_supported accepts highlight_mode=2", params_supported(core, return_linear=True))

    # Documented WB restore: cam_mul from the unified_image_processor comment.
    cam_mul = [1780.0, 1024.0, 1786.0]
    gain = edit_base_highlight_brightness_gain(cam_mul)
    check("compensation is not a no-op on typical cam_mul", gain > 1.5, f"gain={gain:.4f}")
    check(
        "compensation is max/min (measured 1.74x)",
        abs(gain - 1786.0 / 1024.0) < 1e-9,
        f"gain={gain:.6f}",
    )
    check("missing mul is identity", edit_base_highlight_brightness_gain(None) == 1.0)
    check("1:1:1 mul is identity", edit_base_highlight_brightness_gain([1.0, 1.0, 1.0]) == 1.0)

    # Synthetic Blend vs max-normalized Reconstruct: same unclipped mid-grey
    # after compensation. Reconstruct "darkening" here is the WB scale only.
    mid_lin = 0.18
    blend = np.full((32, 32, 3), mid_lin, np.float32)
    recon_u16 = np.full((32, 32, 3), mid_lin * 65535.0 / gain, np.float32).astype(np.uint16)
    restored = linearize_reconstruct_edit_base(recon_u16, cam_mul)
    med_b = float(np.median(blend))
    med_r = float(np.median(restored))
    check(
        "Reconstruct+compensate median matches Blend mid-grey",
        abs(med_r - med_b) / mid_lin < 0.02,
        f"blend={med_b:.5f} restored={med_r:.5f}",
    )
    # Headroom: a 16-bit ceiling pixel must land > 1.0 after gain so HS sees it.
    ceiling = np.full((4, 4, 3), 65535, np.uint16)
    hi = linearize_reconstruct_edit_base(ceiling, cam_mul)
    check(
        "reconstructed ceiling stays above clip after gain",
        float(hi.max()) > 1.5,
        f"max={float(hi.max()):.3f}",
    )

    if FAILURES:
        print(f"FAILED: {FAILURES}")
        return 1
    print("PASS t_edit_base_decode_params")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
