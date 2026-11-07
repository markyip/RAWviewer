#!/usr/bin/env python3
"""Fail when any Python file under ``src/`` exceeds the size limit.

The repository enforces a hard cap of 1,000 lines per module so that files stay
reviewable and classes stay extractable. Files in the warning band are reported
but do not fail the run.

Usage:
    python scripts/check_file_sizes.py [--max 1000] [--warn 750] [--root src]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_MAX_LINES = 1000
DEFAULT_WARN_LINES = 750


def count_lines(path: Path) -> int:
    """Return the line count using ``splitlines`` for a stable cross-platform count."""
    return len(path.read_text(encoding="utf-8").splitlines())


def iter_python_files(root: Path):
    """Yield every Python file under ``root``, skipping virtualenv/cache dirs."""
    skipped = {".venv", "venv", "__pycache__", "build", "dist", ".git", ".mypy_cache"}
    for path in sorted(root.rglob("*.py")):
        if any(part in skipped for part in path.parts):
            continue
        yield path


def main(argv: list[str] | None = None) -> int:
    """Run the size gate and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max", type=int, default=DEFAULT_MAX_LINES,
                        help="hard maximum number of lines per file")
    parser.add_argument("--warn", type=int, default=DEFAULT_WARN_LINES,
                        help="warn when a file exceeds this many lines")
    parser.add_argument("--root", type=Path, default=Path("src"),
                        help="directory to scan")
    args = parser.parse_args(argv)

    violations: list[tuple[int, Path]] = []
    warnings: list[tuple[int, Path]] = []
    total = 0

    for path in iter_python_files(args.root):
        total += 1
        lines = count_lines(path)
        if lines > args.max:
            violations.append((lines, path))
        elif lines > args.warn:
            warnings.append((lines, path))

    for lines, path in warnings:
        print(f"warning: {path} has {lines} lines (target <= {args.warn})",
              file=sys.stderr)

    for lines, path in sorted(violations, reverse=True):
        print(f"error: {path} has {lines} lines (max {args.max})", file=sys.stderr)

    print(f"checked {total} files under {args.root}: "
          f"{len(violations)} over limit, {len(warnings)} in warning band")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
