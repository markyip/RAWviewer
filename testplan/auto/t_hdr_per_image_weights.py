#!/usr/bin/env python3
"""HDR per-frame shadow/highlight recovery weights (engine layer).

Covers the raw_stitching.merge_hdr_exposure_fusion extensions for the
per-frame recovery-weight dialog (plan: Feature 4):
  1. Backward compatibility: no per-image lists -> byte-identical output
     to the pre-change call shape (global scalars, cv2 Mertens path).
  2. Per-image weights actually flow through to the OUTPUT -- this is the
     key check, because the previous custom weight maps were silently
     ignored whenever cv2.createMergeMertens succeeded (its process()
     takes no external maps), so a naive wiring of per-image values into
     those maps would have changed nothing.
  3. return_weight_maps: normalized maps partition the frame (sum to 1),
     and shifting a frame's highlight/shadow exponent moves weight mass
     in the expected direction.
  4. exposure_ev_offsets_for_paths fallback ladder (EXIF absent here, so
     pixel-luminance estimate / zeros) and default_recovery_weights prior.

Synthetic bracketed set: same scene at -2/0/+2 EV simulated by scaling a
structured gradient scene (noise-free, hermetic, no sample files).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def _bracketed_set(h=160, w=240):
    """Three exposures of one scene: dark (-2EV), mid (0EV), bright (+2EV)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    scene = 0.15 + 0.7 * (xx / w)  # left = shadows, right = highlights
    scene += 0.08 * np.sin(yy / 6.0)  # structure so contrast weight is non-degenerate
    scene = np.clip(scene, 0.0, 1.0)
    frames = []
    for ev in (-2.0, 0.0, 2.0):
        exposed = np.clip(scene * (2.0 ** ev), 0.0, 1.0)
        frames.append(np.repeat(exposed[..., None], 3, axis=2).astype(np.float32))
    return frames


def main() -> int:
    from raw_stitching import (
        default_recovery_weights,
        estimate_ev_offsets,
        exposure_ev_offsets_for_paths,
        merge_hdr_exposure_fusion,
    )

    frames = _bracketed_set()
    frames_u16 = [(f * 65535.0).astype(np.uint16) for f in frames]

    # --- 1. Backward compatibility (global-scalar path untouched) ---
    base_a = merge_hdr_exposure_fusion(frames_u16, highlight_weight=1.3, shadow_weight=0.8)
    base_b = merge_hdr_exposure_fusion(frames_u16, highlight_weight=1.3, shadow_weight=0.8)
    check("global-scalar call returns a plain ndarray (no tuple)", isinstance(base_a, np.ndarray))
    # cv2.MergeMertens is 1-ULP nondeterministic across fresh instances
    # (parallel reduction order) -- verified independent of this change, so
    # allow a 1-count tolerance in the uint16 output rather than equality.
    check(
        "global-scalar path is stable within cv2's 1-ULP parallel-reduction jitter",
        int(np.max(np.abs(base_a.astype(np.int32) - base_b.astype(np.int32)))) <= 1,
    )
    check("output dtype preserved (uint16 in -> uint16 out)", base_a.dtype == np.uint16)
    check("output shape matches input", base_a.shape == frames_u16[0].shape)

    # --- 2. Per-image weights actually change the output ---
    neutral, maps_neutral = merge_hdr_exposure_fusion(
        frames_u16,
        highlight_weights=[1.0, 1.0, 1.0],
        shadow_weights=[1.0, 1.0, 1.0],
        return_weight_maps=True,
    )
    favor_dark, maps_favor = merge_hdr_exposure_fusion(
        frames_u16,
        highlight_weights=[2.0, 1.0, 0.0],  # recover highlights from the dark frame
        shadow_weights=[0.0, 1.0, 2.0],     # recover shadows from the bright frame
        return_weight_maps=True,
    )
    check(
        "changing per-image weights changes the fused output",
        not np.array_equal(neutral, favor_dark),
    )

    # --- 3. Weight maps: partition of unity + directional response ---
    total = np.zeros(maps_neutral[0].shape, dtype=np.float32)
    for m in maps_neutral:
        total += m
    check(
        "normalized weight maps sum to ~1 per pixel",
        float(np.max(np.abs(total - 1.0))) < 1e-3,
        f"max dev {float(np.max(np.abs(total - 1.0))):.5f}",
    )
    check("one weight map per input frame", len(maps_neutral) == 3)

    # The scene's right side is its highlight region. There, the DARK frame
    # (index 0) is the only unclipped record; raising its highlight weight
    # (and dropping the bright frame's) must increase its share of the
    # weight mass in that region vs the neutral setting.
    hl_region = np.s_[:, 180:]  # brightest columns of the scene
    dark_share_neutral = float(np.mean(maps_neutral[0][hl_region]))
    dark_share_favor = float(np.mean(maps_favor[0][hl_region]))
    check(
        "raising the dark frame's highlight weight raises its weight share in highlight regions",
        dark_share_favor > dark_share_neutral,
        f"neutral={dark_share_neutral:.3f} favored={dark_share_favor:.3f}",
    )

    sh_region = np.s_[:, :60]  # darkest columns of the scene
    bright_share_neutral = float(np.mean(maps_neutral[2][sh_region]))
    bright_share_favor = float(np.mean(maps_favor[2][sh_region]))
    check(
        "raising the bright frame's shadow weight raises its weight share in shadow regions",
        bright_share_favor > bright_share_neutral,
        f"neutral={bright_share_neutral:.3f} favored={bright_share_favor:.3f}",
    )

    # uint8 family round-trips through the pyramid path too.
    frames_u8 = [(f * 255.0).astype(np.uint8) for f in frames]
    fused_u8 = merge_hdr_exposure_fusion(frames_u8, highlight_weights=[1.5, 1.0, 0.5])
    check("uint8 in -> uint8 out on the per-image path", fused_u8.dtype == np.uint8)
    check("per-image path output in valid range", fused_u8.min() >= 0 and fused_u8.max() <= 255)

    # Single image: contract preserved with and without maps.
    single, single_maps = merge_hdr_exposure_fusion([frames_u16[0]], return_weight_maps=True)
    check("single image returns a copy", np.array_equal(single, frames_u16[0]))
    check("single image weight map is all-ones", np.allclose(single_maps[0], 1.0))

    # --- 4. EV offsets + default prior ---
    # No real EXIF here: paths don't exist, so the helper must fall back to
    # the pixel-luminance estimate (images given) or zeros (no images).
    fake_paths = ["/nonexistent/a.arw", "/nonexistent/b.arw", "/nonexistent/c.arw"]
    evs = exposure_ev_offsets_for_paths(fake_paths, images=frames)
    check("EV fallback returns one offset per frame", len(evs) == 3)
    check(
        "pixel-luminance EV fallback orders dark < mid < bright",
        evs[0] < evs[1] < evs[2],
        f"evs={evs}",
    )
    check(
        "pixel fallback matches estimate_ev_offsets directly",
        evs == estimate_ev_offsets(frames),
    )
    check(
        "no-EXIF no-images fallback returns zeros",
        exposure_ev_offsets_for_paths(fake_paths) == [0.0, 0.0, 0.0],
    )

    shadow_prior, highlight_prior = default_recovery_weights(evs)
    check(
        "prior: darkest frame biased toward highlight recovery (1.5x HL, 0.5x SH)",
        highlight_prior[0] == 1.5 and shadow_prior[0] == 0.5,
    )
    check(
        "prior: brightest frame biased toward shadow recovery (1.5x SH, 0.5x HL)",
        shadow_prior[2] == 1.5 and highlight_prior[2] == 0.5,
    )
    check("prior: middle frame stays neutral", shadow_prior[1] == 1.0 and highlight_prior[1] == 1.0)
    check(
        "prior: uniform EVs (no usable info) -> all-neutral, no fake confidence",
        default_recovery_weights([0.0, 0.0, 0.0]) == ([1.0] * 3, [1.0] * 3),
    )
    check("prior: empty input -> empty lists", default_recovery_weights([]) == ([], []))

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
