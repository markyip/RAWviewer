#!/usr/bin/env python3
"""
Aggregate / compare RAWviewer [PERF] metric lines (see src/perf_metrics.py).

Usage:
    # Single-run summary (count, mean, median, p95, min/max per metric)
    python scripts/bench/perf_report.py /tmp/run.log

    # Before/after comparison (mean + p95 deltas per metric)
    python scripts/bench/perf_report.py --compare baseline.log new.log

    # Filter to one metric
    python scripts/bench/perf_report.py /tmp/run.log --metric decode_full_cpu

Capture a run with:
    RAWVIEWER_PERF=1 pixi run python src/main.py 2>&1 | tee /tmp/run.log
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict

_LINE = re.compile(r"\[PERF\] metric=(\S+) ms=([0-9.]+)")


def parse(path: str) -> dict[str, list[float]]:
    metrics: dict[str, list[float]] = defaultdict(list)
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = _LINE.search(line)
            if m:
                metrics[m.group(1)].append(float(m.group(2)))
    return metrics


def _p(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def summarize(metrics: dict[str, list[float]], only: str | None = None) -> None:
    rows = []
    for name in sorted(metrics):
        if only and name != only:
            continue
        v = metrics[name]
        rows.append(
            (
                name,
                len(v),
                statistics.fmean(v),
                statistics.median(v),
                _p(v, 0.95),
                min(v),
                max(v),
            )
        )
    if not rows:
        print("No [PERF] lines found.")
        return
    hdr = f"{'metric':<20} {'n':>5} {'mean':>9} {'median':>9} {'p95':>9} {'min':>8} {'max':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name, n, mean, med, p95, lo, hi in rows:
        print(
            f"{name:<20} {n:>5} {mean:>8.1f}ms {med:>8.1f}ms {p95:>8.1f}ms "
            f"{lo:>7.1f}ms {hi:>8.1f}ms"
        )


def compare(base_path: str, new_path: str, only: str | None = None) -> None:
    base = parse(base_path)
    new = parse(new_path)
    names = sorted(set(base) | set(new))
    hdr = (
        f"{'metric':<20} {'n(base/new)':>12} {'mean base':>10} {'mean new':>10} "
        f"{'delta':>8} {'p95 base':>9} {'p95 new':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name in names:
        if only and name != only:
            continue
        b, n = base.get(name, []), new.get(name, [])
        mb = statistics.fmean(b) if b else float("nan")
        mn = statistics.fmean(n) if n else float("nan")
        delta = ""
        if b and n and mb > 0:
            delta = f"{(mn - mb) / mb * 100.0:+.0f}%"
        print(
            f"{name:<20} {len(b):>5}/{len(n):<6} {mb:>9.1f}ms {mn:>9.1f}ms "
            f"{delta:>8} {_p(b, 0.95):>8.1f}ms {_p(n, 0.95):>8.1f}ms"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+", help="log file (1) or baseline+new (2 with --compare)")
    ap.add_argument("--compare", action="store_true", help="compare two logs")
    ap.add_argument("--metric", help="only this metric")
    args = ap.parse_args()
    if args.compare:
        if len(args.logs) != 2:
            ap.error("--compare needs exactly two log files")
        compare(args.logs[0], args.logs[1], args.metric)
    else:
        if len(args.logs) != 1:
            ap.error("summary mode takes exactly one log file")
        summarize(parse(args.logs[0]), args.metric)
    return 0


if __name__ == "__main__":
    sys.exit(main())
