"""Lightroom-compatible color labels and keywords on XMP sidecars.

Ratings already live in ``raw_adjustments`` (``xmp:Rating``). Labels use
``xmp:Label`` (Red / Yellow / Green / Blue / Purple). Keywords use
``dc:subject`` as an RDF Bag, the same shape Lightroom and Photo Mechanic write.
"""

from __future__ import annotations

from typing import Iterable, Sequence

# Lightroom Classic color-label strings (xmp:Label). Empty means unlabeled.
COLOR_LABELS: tuple[str, ...] = ("Red", "Yellow", "Green", "Blue", "Purple")

# Gallery / single-view dots. Not theme tokens: they must match the LR names.
COLOR_LABEL_RGB: dict[str, tuple[int, int, int]] = {
    "Red": (220, 64, 64),
    "Yellow": (220, 180, 40),
    "Green": (72, 168, 88),
    "Blue": (64, 128, 220),
    "Purple": (148, 88, 196),
}

# Number keys 6–9 (and Shift+6 for Purple), matching Lightroom Classic.
LABEL_KEY_MAP: dict[int, str] = {
    6: "Red",
    7: "Yellow",
    8: "Green",
    9: "Blue",
}


def normalize_color_label(raw: object) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    lower = text.lower()
    for name in COLOR_LABELS:
        if lower == name.lower():
            return name
    return text


def normalize_keywords(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace(";", ",").split(",")]
        return [p for p in parts if p]
    if isinstance(raw, (list, tuple, set)):
        out: list[str] = []
        seen: set[str] = set()
        for item in raw:
            text = str(item or "").strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
        return out
    return []


def keywords_to_display(keywords: Sequence[str]) -> str:
    return ", ".join(normalize_keywords(list(keywords)))


def parse_keywords_text(text: str) -> list[str]:
    return normalize_keywords(text)


def label_shortcut_help() -> str:
    return "6–9 — Color label (Red / Yellow / Green / Blue; Shift+6 Purple; same key clears)"
