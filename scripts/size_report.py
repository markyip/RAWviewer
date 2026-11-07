#!/usr/bin/env python3
"""Print Python files under ``src/`` sorted by line count.

Companion to ``scripts/check_file_sizes.py``; used to pick the next refactor
target and to snapshot the baseline.

Usage:
    python scripts/size_report.py [--limit 40] [--root src] [--baseline PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from check_file_sizes import DEFAULT_MAX_LINES, count_lines, iter_python_files


def main(argv: list[str] | None = None) -> int:
    """Print the size report and optionally write a baseline snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=40,
                        help="number of rows to print")
    parser.add_argument("--root", type=Path, default=Path("src"))
    parser.add_argument("--baseline", type=Path, default=None,
                        help="write the full listing to this path")
    args = parser.parse_args(argv)

    rows = sorted(((count_lines(p), p) for p in iter_python_files(args.root)),
                  reverse=True)

    for lines, path in rows[: args.limit]:
        marker = "OVER" if lines > DEFAULT_MAX_LINES else "    "
        print(f"{marker} {lines:>6}  {path.as_posix()}")

    if args.baseline is not None:
        args.baseline.write_text(
            "\n".join(f"{lines}\t{path.as_posix()}" for lines, path in rows) + "\n",
            encoding="utf-8",
        )
        print(f"wrote baseline with {len(rows)} rows to {args.baseline}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
