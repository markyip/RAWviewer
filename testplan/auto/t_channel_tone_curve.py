#!/usr/bin/env python3
"""Standard-mode R/G/B tone curve: per-channel remap applied post-gamma.

Mirrors Lightroom/RawTherapee "Standard" mode: each channel is remapped
independently (unlike the hue-preserving luminance point curve), so this
IS expected to shift hue -- that is the feature. Checks:
  - a curve on one channel only changes that channel's output.
  - identity/absent curves are a true no-op (matches is_default_adjustments
    treating an all-empty CHANNEL_CURVE_KEYS as "no edit").
  - reset (all keys absent) equals doing nothing.
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


def main() -> int:
    from raw_tone_curve import (
        TONE_CURVE_BLUE_KEY,
        TONE_CURVE_GREEN_KEY,
        TONE_CURVE_RED_KEY,
        apply_channel_curves_encoded,
    )

    # Flat mid-grey 8-bit image.
    img = np.full((16, 16, 3), 128, dtype=np.uint8)

    # No channel-curve keys at all -> untouched.
    out_none = apply_channel_curves_encoded(img, {}, 255.0)
    check("no keys is a no-op", np.array_equal(out_none, img))

    # Red curve only: push red channel up, leave green/blue alone.
    adj_red = {TONE_CURVE_RED_KEY: "0,0;128,200;255,255"}
    out_red = apply_channel_curves_encoded(img, adj_red, 255.0)
    check(
        "red-only curve changes red channel",
        int(out_red[0, 0, 0]) != 128,
        f"red={out_red[0, 0, 0]}",
    )
    check(
        "red-only curve leaves green/blue untouched",
        int(out_red[0, 0, 1]) == 128 and int(out_red[0, 0, 2]) == 128,
        f"g={out_red[0, 0, 1]} b={out_red[0, 0, 2]}",
    )

    # Identity curve (straight line, no real points) is a no-op even if the
    # key is present with a non-empty serial.
    adj_identity = {TONE_CURVE_GREEN_KEY: "0,0;255,255"}
    out_identity = apply_channel_curves_encoded(img, adj_identity, 255.0)
    check("identity curve is a no-op", np.array_equal(out_identity, img))

    # All three channels active simultaneously, each independent.
    adj_all = {
        TONE_CURVE_RED_KEY: "0,0;128,180;255,255",
        TONE_CURVE_GREEN_KEY: "0,0;128,90;255,255",
        TONE_CURVE_BLUE_KEY: "0,20;128,128;255,235",
    }
    out_all = apply_channel_curves_encoded(img, adj_all, 255.0)
    r, g, b = int(out_all[0, 0, 0]), int(out_all[0, 0, 1]), int(out_all[0, 0, 2])
    check(
        "all three channels independently remapped (introduces hue shift by design)",
        r > 128 and g < 128 and b != r and b != g,
        f"r={r} g={g} b={b}",
    )

    # dtype/shape preserved.
    check("output dtype preserved", out_all.dtype == img.dtype)
    check("output shape preserved", out_all.shape == img.shape)

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
