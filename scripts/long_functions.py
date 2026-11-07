#!/usr/bin/env python3
"""Report functions and methods longer than a threshold.

Used by Phase 3 of the size-reduction plan: no callable should exceed 200 lines.
Exits non-zero when a callable is over ``--max``.

Usage:
    python scripts/long_functions.py [--max 200] [--root src]
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

from check_file_sizes import iter_python_files

DEFAULT_MAX_LINES = 200


class _Collector(ast.NodeVisitor):
    """Collect qualified callable names and their source line spans."""

    def __init__(self, module: str) -> None:
        self.module = module
        self.stack: list[str] = []
        self.found: list[tuple[int, str, int]] = []

    def _visit_callable(self, node: ast.AST) -> None:
        name = getattr(node, "name", "<lambda>")
        self.stack.append(name)
        end = getattr(node, "end_lineno", None) or node.lineno
        self.found.append((end - node.lineno + 1,
                           ".".join([self.module, *self.stack]), node.lineno))
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_callable(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_callable(node)


def main(argv: list[str] | None = None) -> int:
    """Run the long-function gate and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max", type=int, default=DEFAULT_MAX_LINES)
    parser.add_argument("--root", type=Path, default=Path("src"))
    args = parser.parse_args(argv)

    over: list[tuple[int, str, int]] = []
    for path in iter_python_files(args.root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            print(f"error: cannot parse {path}: {exc}", file=sys.stderr)
            continue
        collector = _Collector(path.as_posix())
        collector.visit(tree)
        over.extend(row for row in collector.found if row[0] > args.max)

    for lines, name, lineno in sorted(over, reverse=True):
        print(f"error: {name} at line {lineno} has {lines} lines (max {args.max})",
              file=sys.stderr)

    print(f"found {len(over)} callables over {args.max} lines")
    return 1 if over else 0


if __name__ == "__main__":
    raise SystemExit(main())
