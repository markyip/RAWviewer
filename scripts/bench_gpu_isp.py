#!/usr/bin/env python3
"""Compare CPU vs GPU finish_full_decode on a cached UnpackedRaw.

Spike diagnosis modes:
  # Wall time without per-stage cudaSynchronize (closer to production):
  pixi run python scripts/bench_gpu_isp.py FILE --mode wall --repeats 7

  # Stage breakdown with sync (attribution):
  pixi run python scripts/bench_gpu_isp.py FILE --mode stages --repeats 7

  # Both: wall first, then stages (default):
  pixi run python scripts/bench_gpu_isp.py FILE --mode both --repeats 7

Exits non-zero only on hard failure, not when GPU is slower.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

_STAGE_KEYS = (
    "upload",
    "to_float",
    "scale",
    "demosaic",
    "matrix",
    "gamma",
    "download",
)


def _median_ms(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return float("nan")
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _stats_line(label: str, times: list[float]) -> str:
    return (
        f"{label}  min={min(times):.1f}  median={_median_ms(times):.1f}  "
        f"max={max(times):.1f}  ms  n={len(times)}"
    )


def _vram_line(prefix: str = "VRAM") -> str:
    from gpu_raw_processor import gpu_vram_snapshot

    snap = gpu_vram_snapshot()
    if not snap:
        return f"{prefix}: (CUDA n/a)"
    return (
        f"{prefix}: alloc={snap['allocated_mib']:.0f}MiB  "
        f"reserved={snap['reserved_mib']:.0f}MiB  "
        f"free={snap['free_mib']:.0f}/{snap['total_mib']:.0f}MiB"
    )


def _bench_gpu(
    fn,
    repeats: int,
    warmup: int,
    *,
    capture_stages: bool,
) -> tuple[list[float], object, list[dict[str, float]], list[dict[str, float]]]:
    import gpu_raw_processor as grp

    out = None
    for _ in range(max(0, warmup)):
        out = fn()
    times: list[float] = []
    stage_rows: list[dict[str, float]] = []
    vram_rows: list[dict[str, float]] = []
    for i in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        ms = (time.perf_counter() - t0) * 1000.0
        times.append(ms)
        snap = grp.gpu_vram_snapshot()
        vram_rows.append(snap)
        stages = dict(getattr(grp, "LAST_GPU_ISP_STAGES", {}) or {}) if capture_stages else {}
        if stages:
            stage_rows.append(stages)
            stage_bits = " ".join(
                f"{k}={stages.get(k, 0.0):.1f}" for k in _STAGE_KEYS if k in stages
            )
            vbits = (
                f"alloc={snap.get('allocated_mib', 0):.0f} "
                f"res={snap.get('reserved_mib', 0):.0f}"
                if snap
                else ""
            )
            print(f"  gpu[{i}] wall={ms:.1f} ms  {vbits}  stages: {stage_bits}", flush=True)
        else:
            vbits = (
                f"alloc={snap.get('allocated_mib', 0):.0f}MiB "
                f"reserved={snap.get('reserved_mib', 0):.0f}MiB"
                if snap
                else ""
            )
            print(f"  gpu[{i}] {ms:.1f} ms  {vbits}", flush=True)
    return times, out, stage_rows, vram_rows


def _bench_cpu(fn, repeats: int, warmup: int) -> tuple[list[float], object]:
    out = None
    for _ in range(max(0, warmup)):
        out = fn()
    times: list[float] = []
    for i in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        ms = (time.perf_counter() - t0) * 1000.0
        times.append(ms)
        print(f"  cpu[{i}] {ms:.1f} ms", flush=True)
    return times, out


def _print_stage_summary(stage_rows: list[dict[str, float]]) -> None:
    if not stage_rows:
        return
    print("\n--- GPU stage medians (ms) ---")
    keys = [k for k in _STAGE_KEYS if any(k in r for r in stage_rows)]
    for k in keys:
        vals = [r[k] for r in stage_rows if k in r]
        print(
            f"  {k:10s}  min={min(vals):5.1f}  med={_median_ms(vals):5.1f}  "
            f"max={max(vals):5.1f}"
        )
    med = {k: _median_ms([r[k] for r in stage_rows if k in r]) for k in keys}
    print("\n--- Per-run stage spike flags (>1.5× stage median) ---")
    for i, row in enumerate(stage_rows):
        flags = [
            f"{k}={row[k]:.1f}(med {med[k]:.1f})"
            for k in keys
            if k in row and med[k] > 0 and row[k] > 1.5 * med[k]
        ]
        print(f"  gpu[{i}]  " + (", ".join(flags) if flags else "(none)"))


def _print_vram_summary(vram_rows: list[dict[str, float]]) -> None:
    if not vram_rows or not any(vram_rows):
        return
    alloc = [r["allocated_mib"] for r in vram_rows if r]
    reserved = [r["reserved_mib"] for r in vram_rows if r]
    free = [r["free_mib"] for r in vram_rows if r]
    print("\n--- VRAM across GPU timed runs (MiB) ---")
    print(
        f"  allocated  min={min(alloc):.0f}  med={_median_ms(alloc):.0f}  max={max(alloc):.0f}"
    )
    print(
        f"  reserved   min={min(reserved):.0f}  med={_median_ms(reserved):.0f}  "
        f"max={max(reserved):.0f}"
    )
    print(f"  free       min={min(free):.0f}  med={_median_ms(free):.0f}  max={max(free):.0f}")
    print("  per-run: " + ", ".join(
        f"[{i}]a={r.get('allocated_mib', 0):.0f}/r={r.get('reserved_mib', 0):.0f}"
        for i, r in enumerate(vram_rows)
        if r
    ))


def _run_gpu_series(unpacked, repeats: int, warmup: int, *, stages: bool, sync: bool):
    from fast_raw_decode import finish_full_decode

    os.environ["RAWVIEWER_GPU_ISP_TIMING"] = "1" if stages else "0"
    os.environ["RAWVIEWER_GPU_ISP_TIMING_SYNC"] = "1" if sync else "0"
    label = "stages+sync" if stages and sync else ("stages+nosync" if stages else "wall/no-timing")
    print(f"\n=== GPU series ({label}) ===")
    print(_vram_line("VRAM before"))
    times, out, stage_rows, vram_rows = _bench_gpu(
        lambda: finish_full_decode(unpacked, prefer_gpu=True),
        repeats=repeats,
        warmup=warmup,
        capture_stages=stages,
    )
    print(_vram_line("VRAM after"))
    _print_vram_summary(vram_rows)
    if stages:
        _print_stage_summary(stage_rows)
    return times, out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_path", help="Path to a Bayer RAW file")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--mode",
        choices=("wall", "stages", "both"),
        default="both",
        help="wall=no stage sync/timing; stages=sync attribution; both=wall then stages",
    )
    args = parser.parse_args()

    path = os.path.abspath(args.raw_path)
    if not os.path.isfile(path):
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 2

    os.environ["RAWVIEWER_PERF"] = "1"
    os.environ["RAWVIEWER_GPU_EMPTY_CACHE_S"] = "0"
    os.environ.setdefault("RAWVIEWER_WB_SANITY", "0")

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)

    import torch_bootstrap

    if not torch_bootstrap.wait_for_gpu_backend_ready(timeout=30.0):
        print("WARNING: GPU backend not ready")

    from fast_raw_decode import finish_full_decode, unpack_raw
    from gpu_raw_processor import detect_gpu_backend

    print(f"file={path}")
    print(f"gpu_backend={detect_gpu_backend()}  mode={args.mode}")
    print("RAWVIEWER_GPU_EMPTY_CACHE_S=0  (idle empty_cache disabled for bench)")

    try:
        import torch

        if torch.cuda.is_available():
            x = torch.zeros((64, 64), device="cuda")
            _ = (x + 1).sum().item()
            print("CUDA context warmed")
            del x
    except Exception as e:
        print(f"CUDA warm skipped: {e}")

    t0 = time.perf_counter()
    unpacked = unpack_raw(path)
    unpack_ms = (time.perf_counter() - t0) * 1000.0
    if unpacked is None:
        print("ERROR: unpack_raw returned None", file=sys.stderr)
        return 1
    h, w = unpacked.mosaic.shape[:2]
    print(
        f"unpacked pattern={unpacked.pat_str} mosaic={w}x{h} "
        f"({w * h / 1e6:.1f} MP) in {unpack_ms:.1f} ms"
    )
    print(_vram_line("VRAM after unpack"))

    print("\n=== CPU finish_full_decode ===")
    cpu_times, cpu_out = _bench_cpu(
        lambda: finish_full_decode(unpacked, prefer_gpu=False),
        repeats=args.repeats,
        warmup=args.warmup,
    )
    if cpu_out is None:
        print("ERROR: CPU path returned None", file=sys.stderr)
        return 1

    wall_times = None
    stage_times = None
    gpu_out = None

    if args.mode in ("wall", "both"):
        wall_times, gpu_out = _run_gpu_series(
            unpacked, args.repeats, args.warmup, stages=False, sync=False
        )
    if args.mode in ("stages", "both"):
        stage_times, gpu_out2 = _run_gpu_series(
            unpacked, args.repeats, args.warmup, stages=True, sync=True
        )
        if gpu_out is None:
            gpu_out = gpu_out2

    if gpu_out is None:
        print("ERROR: GPU path returned None", file=sys.stderr)
        return 1

    cpu_med = _median_ms(cpu_times)
    print("\n=== Summary ===")
    print(_stats_line("CPU", cpu_times))
    if wall_times is not None:
        print(_stats_line("GPU wall (no stage timing)", wall_times))
        r = _median_ms(wall_times) / cpu_med if cpu_med else float("inf")
        print(
            f"  GPU_wall/CPU median={r:.3f}x  "
            f"min/CPU={min(wall_times) / cpu_med:.3f}x"
        )
    if stage_times is not None:
        print(_stats_line("GPU stages+sync wall", stage_times))
        r = _median_ms(stage_times) / cpu_med if cpu_med else float("inf")
        print(
            f"  GPU_sync/CPU median={r:.3f}x  "
            f"min/CPU={min(stage_times) / cpu_med:.3f}x"
        )
        if wall_times is not None:
            inflate = _median_ms(stage_times) / max(_median_ms(wall_times), 1e-6)
            print(f"  sync timing inflates wall median by {inflate:.2f}x vs no-timing")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
