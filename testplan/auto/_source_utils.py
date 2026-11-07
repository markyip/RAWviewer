"""Helpers for tests that pin behaviour by inspecting source text."""

from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "src"))
_WINDOW = os.path.join(_SRC, "rawviewer_app", "window")


def window_source() -> str:
    """Return the concatenated source of the main-window package."""
    chunks: list[str] = []
    for root, _dirs, files in os.walk(_WINDOW):
        for name in sorted(files):
            if name.endswith(".py"):
                with open(os.path.join(root, name), encoding="utf-8") as handle:
                    chunks.append(handle.read())
    return "".join(chunks)


def main_source() -> str:
    """Return the entry facade plus the main-window package source."""
    with open(os.path.join(_SRC, "main.py"), encoding="utf-8") as handle:
        return handle.read() + window_source()
