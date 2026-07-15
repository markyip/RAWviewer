#!/usr/bin/env python3
"""WB embedded-JPEG sanity: model verdict cache + misparse correction (golden files)."""
import glob
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

def _sample_dirs():
    return [
        os.environ.get("RAWVIEWER_WB_SAMPLE_DIR", "").strip(),
        r"I:\RAW_Sample",
        "/Volumes/Development/Development/Canon_Sample",
    ]


def _pick_samples():
    for d in _sample_dirs():
        if not d or not os.path.isdir(d):
            continue
        good_a = os.path.join(d, "025A2279.CR3")
        good_b = os.path.join(d, "025A2705.CR3")
        bad = sorted(glob.glob(os.path.join(d, "683A*.CR3")))
        if os.path.isfile(good_a) and os.path.isfile(good_b) and bad:
            return good_a, good_b, bad
    return None, None, []


GOOD_A, GOOD_B, BAD = _pick_samples()

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    if not (os.path.isfile(GOOD_A) and os.path.isfile(GOOD_B) and BAD):
        print("SKIP  golden RAW files not available on this machine")
        return 0

    os.environ["RAWVIEWER_WB_SANITY"] = "1"
    import fast_raw_decode as frd

    frd._WB_CORRECTION_CACHE.clear()
    frd._WB_MODEL_VERDICT.clear()

    # Clean model: first file measures, verdict recorded, no correction
    frd.unpack_raw(GOOD_A)
    check("clean file uncorrected", frd.get_corrected_camera_wb(GOOD_A) is None)
    check("clean model verdict recorded", False in frd._WB_MODEL_VERDICT.values())

    # Second file of the clean model: skipped (near-zero overhead)
    t0 = time.perf_counter()
    frd.unpack_raw(GOOD_B)
    with_verdict = time.perf_counter() - t0
    check("second clean file uncorrected", frd.get_corrected_camera_wb(GOOD_B) is None)
    print(f"INFO  second-file unpack with verdict skip: {with_verdict*1000:.0f}ms")

    # Body newer than the bundled LibRaw (EOS R6 III): LibRaw reports an
    # incomplete black level (deltas over a zero floor, e.g. [0, 33, 100, 67]),
    # which _resolve_black repairs from the sensor's masked optical-black
    # columns. The as-shot WB on these files was never wrong -- the pedestal was
    # -- so once black is right they must pass the embedded-JPEG check untouched.
    # A correction firing here means the black repair regressed: the WB solver
    # would then be bending the multipliers to hide a pedestal (which is what
    # produced the visible color shift on these very files).
    for p in BAD:
        u = frd.unpack_raw(p)
        name = os.path.basename(p)
        check(f"black repaired {name}", u is not None and float(min(u.black)) > 0.0,
              detail=f"black={[round(float(x)) for x in u.black]}" if u is not None else "")
        check(f"flat black floor {name}",
              u is not None and len(set(round(float(x)) for x in u.black)) == 1)
        check(f"WB left alone {name}", frd.get_corrected_camera_wb(p) is None)

    # Clean bodies keep LibRaw's own black untouched (the repair is gated on
    # cblack.min() == 0): a Sony ARW's left margin reads well above its true
    # black, so measuring unconditionally would corrupt a body that works.
    check("no model flagged as misparsed", True not in frd._WB_MODEL_VERDICT.values())

    # Model-key disambiguation regression: LibRaw aliases the newer body onto an
    # older body's color matrix, so rgb_cam alone must never key the verdict
    # cache -- one body's "clean" verdict would suppress checks on the other.
    import rawpy
    import numpy as np

    def _model_key(path):
        with rawpy.imread(path) as r:
            return frd._wb_model_key(
                frd._rgb_cam_from_cam_xyz(r.rgb_xyz_matrix),
                r.raw_image_visible,
                np.asarray(r.black_level_per_channel, dtype=np.float64),
                float(r.white_level),
            )

    check("aliased-matrix models disambiguated", _model_key(GOOD_A) != _model_key(BAD[0]))

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
