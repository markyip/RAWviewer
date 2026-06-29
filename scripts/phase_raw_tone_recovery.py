#!/usr/bin/env python3
"""Phase tests for RAW-only local shadow/highlight recovery (T key)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from raw_tone_recovery import (
    _encode_srgb8,
    apply_local_shadow_highlight_recovery,
    decode_and_recover_raw,
    encode_edr_rgbx64,
    linear_tone_map_to_display,
    linear_tone_map_to_edr,
    process_linear_edr_rgb,
    process_linear_recovery_rgb,
    recovery_decode_params,
)


def _ref_srgb8(level: float) -> int:
    px = _encode_srgb8(np.full((1, 1, 3), level, dtype=np.float32))
    return int(px[0, 0, 0])


def test_recovery_params():
    p = recovery_decode_params()
    assert p.output_bps == 16
    assert p.half_size is True
    assert p.no_auto_bright is True
    assert p.gamm == (1.0, 1)
    assert p.highlight is not None
    assert p.exp_preser == 0.45
    print("OK recovery_decode_params")


def test_linear_tone_map_extreme_contrast():
    h, w = 256, 256
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[: h // 2, :, :] = 0.002
    rgb[h // 2 :, :, :] = 12.0
    out = linear_tone_map_to_display(rgb)
    dark = float(out[h // 4, w // 2, 0])
    bright = float(out[3 * h // 4, w // 2, 0])
    assert dark < 0.12, f"shadow should not be globally lifted: {dark}"
    assert bright < 0.96, f"highlight should compress: {bright}"
    assert bright > dark, "scene should retain contrast"
    print("OK linear_tone_map_extreme_contrast")


def test_full_pipeline_backlit_local_lift():
    """Backlit scene: sky stays put, shadows lift only via local recovery."""
    h, w = 256, 256
    rgb16 = np.zeros((h, w, 3), dtype=np.uint16)
    rgb16[: h // 3, :, :] = 300
    rgb16[h // 3 :, :, :] = 48000
    tm = linear_tone_map_to_display(rgb16.astype(np.float32) / 65535.0)
    sky_tm = float(tm[3 * h // 4, w // 2, 0])
    dark_tm = float(tm[h // 8, w // 2, 0])
    assert dark_tm < sky_tm * 0.5, "tone map must not globally lift shadows"
    out = process_linear_recovery_rgb(rgb16, max_edge=256)
    dark_out = int(out[h // 8, w // 2, 0])
    sky_ref = _ref_srgb8(float(sky_tm))
    sky_out = int(out[3 * h // 4, w // 2, 0])
    assert dark_out > 15, "local recovery should lift shadow"
    assert abs(sky_out - sky_ref) <= 18, f"sky should stay near tone-map level: {sky_out} vs {sky_ref}"
    print("OK full_pipeline_backlit_local_lift")


def test_local_recovery_synthetic():
    h, w = 256, 256
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[: h // 2, :, :] = 0.04
    rgb[h // 2 :, :, :] = 0.92
    out = apply_local_shadow_highlight_recovery(rgb)
    assert out.shape == (h, w, 3)
    assert out.dtype == np.uint8
    dark_ref = _ref_srgb8(0.04)
    dark_out = int(out[h // 4, w // 2, 0])
    bright_ref = _ref_srgb8(0.92)
    bright_out = int(out[3 * h // 4, w // 2, 0])
    assert dark_out > dark_ref, "shadow region should be lifted"
    assert bright_out < bright_ref - 4, "highlight region should be rolled off"
    print("OK apply_local_shadow_highlight_recovery synthetic")


def test_backlit_shadow_not_crushed():
    """Dark foreground beside bright sky must lift, not darken or solarize."""
    h, w = 256, 256
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[:, : w // 3, :] = 0.05
    rgb[:, w // 3 :, :] = 0.94
    out = apply_local_shadow_highlight_recovery(rgb)
    y, x = h // 2, w // 8
    dark = out[y, x]
    assert int(dark[0]) > _ref_srgb8(0.05), "backlit shadow should lift"
    assert dark[0] > 0 and dark[1] > 0 and dark[2] > 0, "shadow should stay neutral"
    spread = int(max(dark)) - int(min(dark))
    assert spread < 40, f"shadow color should stay neutral, spread={spread}"
    print("OK backlit shadow not crushed")


def test_highlight_recovery_strong():
    """Near-clipped highlights should compress noticeably."""
    rgb = np.full((128, 128, 3), 0.97, dtype=np.float32)
    out = apply_local_shadow_highlight_recovery(rgb)
    bright_ref = _ref_srgb8(0.97)
    bright_out = int(out[64, 64, 0])
    assert bright_out < bright_ref - 8, f"highlight should roll off: {bright_out} vs {bright_ref}"
    print("OK highlight recovery strong")


def test_hard_clip_no_gray_disk():
    """Sensor-saturated whites must not turn into flat gray."""
    rgb = np.ones((64, 64, 3), dtype=np.float32)
    out = apply_local_shadow_highlight_recovery(rgb)
    mid = int(out[32, 32, 0])
    assert mid >= 250, f"hard clip should stay near white, got {mid}"
    print("OK hard_clip_no_gray_disk")


def test_deep_shadow_no_solarization():
    """Very dark values must not blow up into single-channel clipping."""
    rgb = np.full((64, 64, 3), 0.008, dtype=np.float32)
    out = apply_local_shadow_highlight_recovery(rgb)
    mid = out[32, 32]
    spread = int(max(mid)) - int(min(mid))
    assert spread < 25, f"deep shadow solarized: {mid.tolist()}"
    print("OK deep shadow no solarization")


def test_process_linear_recovery_rgb():
    rgb16 = np.zeros((128, 128, 3), dtype=np.uint16)
    rgb16[:64, :, :] = 200
    rgb16[64:, :, :] = 52000
    out = process_linear_recovery_rgb(rgb16, max_edge=128)
    assert out.dtype == np.uint8
    assert int(out[96, 64, 0]) > int(out[32, 64, 0]), "tone map should preserve ordering"
    print("OK process_linear_recovery_rgb")


def test_linear_tone_map_to_edr_headroom():
    h, w = 128, 128
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[: h // 2, :, :] = 0.002
    rgb[h // 2 :, :, :] = 8.0
    rgb[h // 2 : h // 2 + 4, :8, :] = 24.0  # specular above anchor percentile
    sdr = linear_tone_map_to_display(rgb)
    edr = linear_tone_map_to_edr(rgb, peak_display=4.0)
    bright_sdr = float(sdr[h // 2 + 2, 4, 0])
    bright_edr = float(edr[h // 2 + 2, 4, 0])
    assert bright_edr > bright_sdr + 0.05, (
        f"EDR should preserve highlight headroom: edr={bright_edr} sdr={bright_sdr}"
    )
    assert bright_edr <= 4.0
    print("OK linear_tone_map_to_edr_headroom")


def test_encode_edr_rgbx64():
    rgb = np.zeros((8, 8, 3), dtype=np.float32)
    rgb[:, :, :] = (0.2, 1.0, 3.5)
    packed = encode_edr_rgbx64(rgb, peak_display=4.0)
    assert packed.dtype == np.uint16
    assert packed.shape == (8, 8, 4)
    assert int(packed[0, 0, 3]) == 65535
    assert int(packed[0, 0, 2]) > int(packed[0, 0, 0])
    print("OK encode_edr_rgbx64")


def test_process_linear_edr_rgb():
    rgb16 = np.zeros((64, 64, 3), dtype=np.uint16)
    rgb16[:32, :, :] = 500
    rgb16[32:, :, :] = 60000
    out = process_linear_edr_rgb(rgb16, max_edge=64, peak_display=4.0)
    assert out.dtype == np.float32
    assert float(out[48, 32, 0]) > float(out[16, 32, 0])
    print("OK process_linear_edr_rgb")


def test_decode_and_recover_optional(raw_path: str | None):
    if not raw_path or not Path(raw_path).is_file():
        print("SKIP decode_and_recover_raw (no RAW path)")
        return
    rgb = decode_and_recover_raw(raw_path, max_edge=1024)
    assert rgb.ndim == 3 and rgb.shape[2] == 3
    assert rgb.dtype == np.uint8
    print(f"OK decode_and_recover_raw {Path(raw_path).name} shape={rgb.shape}")


def main():
    test_recovery_params()
    test_linear_tone_map_extreme_contrast()
    test_full_pipeline_backlit_local_lift()
    test_local_recovery_synthetic()
    test_backlit_shadow_not_crushed()
    test_highlight_recovery_strong()
    test_hard_clip_no_gray_disk()
    test_deep_shadow_no_solarization()
    test_process_linear_recovery_rgb()
    test_linear_tone_map_to_edr_headroom()
    test_encode_edr_rgbx64()
    test_process_linear_edr_rgb()
    raw = sys.argv[1] if len(sys.argv) > 1 else None
    test_decode_and_recover_optional(raw)
    print("All phase_raw_tone_recovery tests passed.")


if __name__ == "__main__":
    main()
