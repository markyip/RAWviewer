#!/usr/bin/env python3
"""Banded pixel pipelines must not spawn one worker per laptop core."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    extra = f"  {detail}" if detail else ""
    print(f"{'PASS' if ok else 'FAIL'}  {name}{extra}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from raw_adjustments import banded_worker_count

    check("short frames stay single-threaded", banded_worker_count(400) == 1)

    os.environ["RAWVIEWER_BANDED_WORKERS"] = "2"
    os.environ["OMP_NUM_THREADS"] = "8"
    n = banded_worker_count(2000)
    cores = os.cpu_count() or 1
    check(
        "explicit BANDED_WORKERS wins over OMP",
        n == min(2, cores),
        n,
    )
    os.environ.pop("RAWVIEWER_BANDED_WORKERS", None)

    os.environ["OMP_NUM_THREADS"] = "8"
    n = banded_worker_count(2000)
    check("default banding never exceeds 4", n <= 4, n)

    os.environ["OMP_NUM_THREADS"] = "2"
    n = banded_worker_count(640)
    check("640-row fixture still bands when OMP and cores allow", n == min(2, cores), n)
    os.environ.pop("OMP_NUM_THREADS", None)

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
