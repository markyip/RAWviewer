#!/usr/bin/env python3
"""
POC: compare capture-time extraction (App vs Windows Shell), sort by capture time only.

Workflow:
  1. Scan folder (top-level, same as RAWviewer).
  2. Extract capture timestamps for all images (timed separately):
       - App (default): resolve_folder_sort_timestamp with NO cache — metadata_backend
         probe only (cold-start / first-open code path).
       - App (--use-cache): bulk EXIF cache + resolve (warm production path).
       - Windows Shell: bulk_shell_date_taken_timestamps (System.Photo.DateTaken)
  3. Optionally focus on the N chronologically oldest files by **app capture time**
     (--limit; default 1000).
  4. Sort by capture time only (newest-first, like RAWviewer) and compare order/values.

Usage (repo root):
  python scripts/compare_shell_capture_times.py "K:\\Photos\\Japan Trip" --limit 1000
  python scripts/compare_shell_capture_times.py "K:\\Photos\\Japan Trip" --use-cache
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from datetime import datetime
from typing import Dict, List, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

SUPPORTED_EXTENSIONS = {
    ".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".pef",
    ".srw", ".x3f", ".raf", ".3fr", ".fff", ".iiq", ".cap", ".erf",
    ".mef", ".mos", ".nrw", ".rwl", ".srf",
    ".jpeg", ".jpg", ".png", ".webp", ".heif", ".heic", ".tif", ".tiff",
}

StatRow = Tuple[int, float]  # size, mtime


def scan_all_images(folder: str) -> Tuple[List[str], Dict[str, StatRow]]:
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise SystemExit(f"Folder not found: {folder}")

    paths: List[str] = []
    stats: Dict[str, StatRow] = {}
    with os.scandir(folder) as it:
        for entry in it:
            if entry.name.startswith("."):
                continue
            if not entry.is_file(follow_symlinks=False):
                continue
            ext = os.path.splitext(entry.name)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            try:
                st = entry.stat()
                if st.st_size <= 0:
                    continue
                ap = os.path.abspath(entry.path)
                paths.append(ap)
                stats[ap] = (st.st_size, st.st_mtime)
            except OSError:
                continue
    return paths, stats


def fmt_ts(ts: float) -> str:
    if ts <= 0:
        return "(none)"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def sort_by_capture(
    paths: List[str],
    timestamps: Dict[str, float],
    *,
    newest_first: bool = True,
) -> List[str]:
    def key(p: str) -> tuple:
        ts = timestamps.get(p, 0.0)
        primary = -ts if newest_first else ts
        base = os.path.basename(p).lower()
        stem, ext = os.path.splitext(base)
        return (primary, stem, ext, base)

    return sorted(paths, key=key)


def oldest_by_capture_time(
    paths: List[str],
    timestamps: Dict[str, float],
    limit: int,
) -> List[str]:
    """N files with earliest capture timestamp (missing capture time excluded first)."""
    with_ts = [p for p in paths if timestamps.get(p, 0.0) > 0]
    ranked = sorted(with_ts, key=lambda p: (timestamps[p], os.path.basename(p).lower()))
    if limit > 0:
        return ranked[:limit]
    return ranked


def order_mismatches(order_a: List[str], order_b: List[str]) -> int:
    if len(order_a) != len(order_b):
        return max(len(order_a), len(order_b))
    return sum(1 for a, b in zip(order_a, order_b) if a != b)


def extract_app_probe(
    paths: List[str],
    file_stats: Dict[str, StatRow],
) -> Tuple[Dict[str, Tuple[bool, float, str]], float]:
    """Cold-start app path: no EXIF cache read; metadata_backend probe per file."""
    from common_image_loader import resolve_folder_sort_timestamp

    t0 = time.perf_counter()
    results: Dict[str, Tuple[bool, float, str]] = {}
    for fp in paths:
        mtime = file_stats[fp][1]
        results[fp] = resolve_folder_sort_timestamp(
            fp,
            None,
            mtime,
            probe_file=True,
            shell_capture_timestamp=0.0,
        )
    return results, time.perf_counter() - t0


def extract_app_cached(
    paths: List[str],
    file_stats: Dict[str, StatRow],
) -> Tuple[Dict[str, Tuple[bool, float, str]], Dict[str, float]]:
    from common_image_loader import resolve_folder_sort_timestamp
    from image_cache import get_image_cache

    timings: Dict[str, float] = {}
    t0 = time.perf_counter()
    cache = get_image_cache()
    bulk_meta = cache.get_multiple_exif(
        paths,
        {p: file_stats[p] for p in paths},
    )
    timings["app_cache_bulk_sec"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    results: Dict[str, Tuple[bool, float, str]] = {}
    for fp in paths:
        mtime = file_stats[fp][1]
        meta = bulk_meta.get(fp)
        results[fp] = resolve_folder_sort_timestamp(
            fp,
            meta,
            mtime,
            probe_file=True,
            shell_capture_timestamp=0.0,
        )
    timings["app_resolve_loop_sec"] = time.perf_counter() - t1
    timings["app_total_sec"] = timings["app_cache_bulk_sec"] + timings["app_resolve_loop_sec"]
    return results, timings


def extract_shell(paths: List[str]) -> Tuple[Dict[str, float], float]:
    from windows_shell_meta import bulk_shell_date_taken_timestamps, clear_shell_date_cache

    clear_shell_date_cache()
    t0 = time.perf_counter()
    raw = bulk_shell_date_taken_timestamps(paths)
    elapsed = time.perf_counter() - t0
    out = {fp: raw.get(os.path.normcase(fp), 0.0) for fp in paths}
    return out, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="Folder to scan (top-level only)")
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Compare the N oldest files by app capture time (0 = entire folder)",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use warm EXIF cache (bulk get_multiple_exif) instead of cold probe",
    )
    parser.add_argument("--csv", default="", help="Optional CSV output path")
    args = parser.parse_args()

    os.environ.setdefault("RAWVIEWER_USE_SHELL_SORT_DATES", "1")

    app_mode = "cached EXIF" if args.use_cache else "probe only (no cache read)"
    print(f"Folder: {args.folder}")
    print(f"App capture path: {app_mode}")
    if args.limit > 0:
        print(f"Report set: {args.limit} chronologically oldest by app capture time\n")
    else:
        print("Report set: entire folder\n")

    t_scan = time.perf_counter()
    all_paths, all_stats = scan_all_images(args.folder)
    scan_sec = time.perf_counter() - t_scan
    print(f"Scan: {len(all_paths)} images in {scan_sec:.2f}s")
    if not all_paths:
        return 1

    print("\n--- Timing: capture extraction (full folder) ---")
    if args.use_cache:
        app_results_all, app_times = extract_app_cached(all_paths, all_stats)
        print(
            f"App (cached)      cache bulk: {app_times['app_cache_bulk_sec']:.3f}s | "
            f"resolve loop: {app_times['app_resolve_loop_sec']:.3f}s | "
            f"total: {app_times['app_total_sec']:.3f}s  ({len(all_paths)} files)"
        )
    else:
        app_results_all, app_sec = extract_app_probe(all_paths, all_stats)
        print(
            f"App (probe, no cache read): {app_sec:.3f}s  ({len(all_paths)} files)"
        )

    shell_ts_all, shell_sec = extract_shell(all_paths)
    shell_count = sum(1 for v in shell_ts_all.values() if v > 0)
    print(f"Shell (DateTaken) total:     {shell_sec:.3f}s  ({shell_count} with date > 0)")

    app_ts_all: Dict[str, float] = {}
    for fp, (has_cap, ts, _src) in app_results_all.items():
        app_ts_all[fp] = ts if has_cap and ts > 0 else 0.0

    if args.limit > 0:
        paths = oldest_by_capture_time(all_paths, app_ts_all, args.limit)
        if len(paths) < args.limit:
            print(f"\nWarning: only {len(paths)} files have app capture time (< {args.limit})")
    else:
        paths = list(all_paths)

    app_results = {p: app_results_all[p] for p in paths}
    app_ts = {p: app_ts_all[p] for p in paths}
    shell_ts = {p: shell_ts_all[p] for p in paths}

    cap_min = min((app_ts[p] for p in paths if app_ts[p] > 0), default=0)
    cap_max = max((app_ts[p] for p in paths if app_ts[p] > 0), default=0)
    print(f"\nReport set: {len(paths)} files")
    print(f"App capture range: {fmt_ts(cap_min)}  ->  {fmt_ts(cap_max)}")
    oldest_first = sort_by_capture(paths, app_ts, newest_first=False)
    if oldest_first:
        print(f"  oldest: {os.path.basename(oldest_first[0])}")
        print(f"  newest: {os.path.basename(oldest_first[-1])}")

    app_src = Counter(src for _, _, src in app_results.values())

    identical_1s = identical_strict = 0
    differ: List[Tuple[float, str, str, str, str]] = []
    only_app = only_shell = 0
    for fp in paths:
        has_a, a_ts, src = app_results[fp]
        s_ts = shell_ts[fp]
        a_cap = has_a and a_ts > 0
        s_cap = s_ts > 0
        if a_cap and s_cap:
            d = abs(a_ts - s_ts)
            if d < 1.0:
                identical_1s += 1
            if d < 0.001:
                identical_strict += 1
            if d >= 1.0:
                differ.append((d, os.path.basename(fp), src, fmt_ts(a_ts), fmt_ts(s_ts)))
        elif a_cap:
            only_app += 1
        elif s_cap:
            only_shell += 1
    differ.sort(key=lambda x: -x[0])

    order_app = sort_by_capture(paths, app_ts, newest_first=True)
    order_shell = sort_by_capture(paths, shell_ts, newest_first=True)
    mismatches = order_mismatches(order_app, order_shell)

    app_ts_sec = {p: round(app_ts[p]) for p in paths}
    shell_ts_sec = {p: round(shell_ts[p]) for p in paths}
    mismatches_sec = order_mismatches(
        sort_by_capture(paths, app_ts_sec, newest_first=True),
        sort_by_capture(paths, shell_ts_sec, newest_first=True),
    )

    print("\n--- Capture values (same timestamp?) ---")
    print(f"Both have capture time:      {identical_1s + len(differ)}")
    print(f"  within 1s (display match):  {identical_1s}")
    print(f"  exact (<1ms):               {identical_strict}")
    print(f"  differ (>=1s):              {len(differ)}")
    subsec = identical_1s - identical_strict
    if subsec > 0:
        print(f"  sub-second only:            {subsec}  (can reorder burst shots)")
    print(f"App only:                    {only_app}")
    print(f"Shell only:                  {only_shell}")
    print(f"App sort sources:            {dict(app_src)}")

    print("\n--- Sort by capture time only (newest first) ---")
    print(f"Order identical (full precision):   {mismatches == 0} ({mismatches} mismatches)")
    print(f"Order identical (rounded to second): {mismatches_sec == 0} ({mismatches_sec} mismatches)")

    if differ:
        print("\nLargest timestamp deltas (>=1s):")
        for row in differ[:15]:
            d, base, src, a, s = row
            print(f"  {d/86400:7.2f} d  {base[:36]:36}  {src:6}  app={a}  shell={s}")

    print("\n--- First 5 newest in app capture-time sort ---")
    for fp in order_app[:5]:
        print(f"  {os.path.basename(fp):40}  app={fmt_ts(app_ts[fp])}  shell={fmt_ts(shell_ts[fp])}")

    print("\n--- First 5 oldest in app capture-time sort ---")
    for fp in oldest_first[:5]:
        print(f"  {os.path.basename(fp):40}  app={fmt_ts(app_ts[fp])}  shell={fmt_ts(shell_ts[fp])}")

    if args.csv:
        import csv

        out_dir = os.path.dirname(os.path.abspath(args.csv)) or "."
        os.makedirs(out_dir, exist_ok=True)
        rank_app = {p: i + 1 for i, p in enumerate(order_app)}
        rank_shell = {p: i + 1 for i, p in enumerate(order_shell)}
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "path",
                    "basename",
                    "app_has_capture",
                    "app_source",
                    "app_ts",
                    "app_iso",
                    "shell_ts",
                    "shell_iso",
                    "delta_sec",
                    "rank_app_newest_first",
                    "rank_shell_newest_first",
                    "rank_delta",
                ]
            )
            for fp in paths:
                has_cap, a_ts, src = app_results[fp]
                s_ts = shell_ts[fp]
                delta = abs(a_ts - s_ts) if has_cap and s_ts > 0 else ""
                w.writerow(
                    [
                        fp,
                        os.path.basename(fp),
                        int(has_cap),
                        src,
                        a_ts,
                        fmt_ts(a_ts),
                        s_ts,
                        fmt_ts(s_ts),
                        delta,
                        rank_app[fp],
                        rank_shell[fp],
                        abs(rank_app[fp] - rank_shell[fp]),
                    ]
                )
        print(f"\nWrote CSV: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
