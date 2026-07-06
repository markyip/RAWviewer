"""
Golden-file color-parity gate for src/fast_raw_decode.py.

For every sample RAW found (or passed as arguments), compares the module's
demosaic-free half-size reference decode against
``rawpy.postprocess(half_size=True, use_camera_wb=True, no_auto_bright=True,
output_bps=8, user_flip=0)``. Both sides skip demosaic (2x2 binning), so the
comparison isolates the color math — black level, white balance scaling,
highlight clipping, color matrix, gamma. The gate requires every pixel of
every file to match within +/-1 8-bit LSB.

Also smoke-tests the full-resolution fast path on the first file: output
shape/dtype, and color agreement with the rawpy LINEAR reference after
downsampling (loose threshold — demosaic algorithms differ by design; only
gross color errors like the old GPU PoC's shifted highlights would trip it).

Usage:
    PYTHONPATH=src pixi run python3 scripts/fast_raw_decode_parity_gate.py [files...]

Exits nonzero on any failure. Run this when adding support for a new camera
brand or touching fast_raw_decode.py.
"""

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DEFAULT_SAMPLE_GLOBS = [
    "/Volumes/Development/Manchester/DSC01089.ARW",
    "/Volumes/Development/Development/Canon_Sample/*.[cC][rR]3",
]


def collect_files(args):
    if args:
        return args
    files = []
    for g in DEFAULT_SAMPLE_GLOBS:
        files.extend(sorted(glob.glob(g)))
    return files


def main() -> int:
    import rawpy

    from fast_raw_decode import (
        decode_half_from_unpacked,
        finish_full_decode,
        halfsize_reference_decode,
        try_fast_raw_decode,
        unpack_raw,
    )

    files = collect_files(sys.argv[1:])
    if not files:
        print("No sample RAW files found; pass paths as arguments.")
        return 1

    failures = 0
    tested = 0
    for path in files:
        name = os.path.basename(path)
        mine = halfsize_reference_decode(path)
        if mine is None:
            print(f"SKIP  {name} (unsupported sensor — fast path would fall back)")
            continue
        # Production half tier (fit-view paint) must match rawpy just as
        # tightly as the float reference does.
        unpacked = unpack_raw(path)
        prod_half = decode_half_from_unpacked(unpacked) if unpacked else None
        if prod_half is None:
            print(f"FAIL  {name}: decode_half_from_unpacked returned None")
            failures += 1
            continue
        with rawpy.imread(path) as raw:
            ref = raw.postprocess(
                half_size=True,
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=8,
                user_flip=0,
            )
        if mine.shape != ref.shape:
            print(f"FAIL  {name}: shape {mine.shape} vs rawpy {ref.shape}")
            failures += 1
            continue
        d = np.abs(mine.astype(np.int16) - ref.astype(np.int16))
        n_over = int((d > 1).sum())
        hl = ref.max(axis=-1) >= 250
        hl_max = int(d[hl].max()) if hl.any() else 0
        status = "PASS" if n_over == 0 else "FAIL"
        if n_over:
            failures += 1
        tested += 1
        print(
            f"{status}  {name}: maxdiff={int(d.max())} mean={d.mean():.4f} "
            f"pixels>1LSB={n_over} highlight_maxdiff={hl_max}"
        )
        if prod_half.shape != ref.shape:
            print(f"FAIL  {name}: prod half shape {prod_half.shape} vs rawpy {ref.shape}")
            failures += 1
        else:
            dp = np.abs(prod_half.astype(np.int16) - ref.astype(np.int16))
            n_over_p = int((dp > 1).sum())
            if n_over_p:
                failures += 1
            print(
                f"{'PASS' if n_over_p == 0 else 'FAIL'}  {name} [prod half]: "
                f"maxdiff={int(dp.max())} mean={dp.mean():.4f} pixels>1LSB={n_over_p}"
            )

    if tested == 0:
        print("No supported files were tested — gate inconclusive.")
        return 1

    # Full-res smoke test on the first supported file.
    smoke = next((f for f in files if halfsize_reference_decode(f) is not None), None)
    if smoke is not None:
        import cv2

        params = dict(
            use_camera_wb=True,
            use_auto_wb=False,
            no_auto_bright=True,
            output_bps=8,
            gamma=(2.222, 4.5),
            bright=1.0,
            user_flip=0,
            half_size=False,
        )
        out = try_fast_raw_decode(smoke, params)
        name = os.path.basename(smoke)
        if out is None:
            print(f"FAIL  full-res fast path returned None for {name}")
            failures += 1
        else:
            # Split path (stashed-unpack reuse) must be byte-identical to
            # the one-shot path.
            u = unpack_raw(smoke)
            split = finish_full_decode(u) if u else None
            if split is None or not np.array_equal(split, out):
                print(f"FAIL  split unpack+finish differs from one-shot for {name}")
                failures += 1
            else:
                print(f"PASS  split unpack+finish byte-identical for {name}")
            with rawpy.imread(smoke) as raw:
                ref = raw.postprocess(
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR, **{
                        k: v for k, v in params.items() if k != "half_size"
                    }
                )
            ok = out.dtype == np.uint8 and out.shape == ref.shape
            small_o = cv2.resize(out, (ref.shape[1] // 4, ref.shape[0] // 4), interpolation=cv2.INTER_AREA)
            small_r = cv2.resize(ref, (ref.shape[1] // 4, ref.shape[0] // 4), interpolation=cv2.INTER_AREA)
            d = np.abs(small_o.astype(np.int16) - small_r.astype(np.int16))
            # Demosaic algorithms differ (EA vs LINEAR): allow small residue,
            # catch gross color errors (old PoC measured mean ~30-70 here).
            ok = ok and d.mean() < 2.0 and np.percentile(d, 99) <= 6
            print(
                f"{'PASS' if ok else 'FAIL'}  full-res smoke {name}: shape={out.shape} "
                f"vs LINEAR downsampled mean={d.mean():.3f} p99={np.percentile(d, 99):.0f}"
            )
            if not ok:
                failures += 1

    print(f"\n{tested} parity file(s) tested, {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
