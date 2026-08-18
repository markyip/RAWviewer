"""Collapse a RAW and its same-stem JPEG into one gallery tile.

Burst grouping requires 3+ frames so it never swallows these pairs. This module
handles the two-file case: one RAW + one JPEG (or several JPEGs) sharing a
basename.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from common_image_loader import is_raw_file

PAIRING_SETTINGS_KEY = "gallery/raw_jpeg_pairing_enabled"
JPEG_EXTENSIONS = frozenset({".jpg", ".jpeg"})


def pairing_enabled_from_settings(settings) -> bool:
    """Default on: most camera cards are RAW+JPEG and the JPEG is a duplicate."""
    try:
        raw = settings.value(PAIRING_SETTINGS_KEY, True)
        if isinstance(raw, str):
            return raw.strip().lower() not in ("0", "false", "no", "off")
        if raw is None:
            return True
        return bool(raw)
    except Exception:
        return True


def _norm(path: str) -> str:
    if not path:
        return ""
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def stem_key(path: str) -> str:
    base = os.path.basename(path or "")
    stem, _ext = os.path.splitext(base)
    return os.path.normcase(stem)


def is_jpeg_file(path: str) -> bool:
    ext = os.path.splitext(path or "")[1].lower()
    return ext in JPEG_EXTENSIONS


def _is_pairable_raw(path: str) -> bool:
    return is_raw_file(path)


def pairs_for_folder(
    paths: Sequence[str],
    *,
    rejected: Optional[Set[str]] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Group same-stem RAW + JPEG files.

    Returns:
        path_to_pair: normpath -> member list (RAW first, then JPEGs, folder order)
        pair_covers: group_id (cover normpath) -> cover path (the RAW)
    """
    rejected = rejected or set()
    visible = [p for p in paths if p and _norm(p) not in rejected]
    buckets: Dict[str, List[str]] = {}
    for path in visible:
        buckets.setdefault(stem_key(path), []).append(path)

    path_to_pair: Dict[str, List[str]] = {}
    pair_covers: Dict[str, str] = {}
    for members in buckets.values():
        raws = [p for p in members if _is_pairable_raw(p)]
        jpegs = [p for p in members if is_jpeg_file(p)]
        if not raws or not jpegs:
            continue
        ordered = list(raws) + list(jpegs)
        cover = raws[0]
        cover_id = _norm(cover)
        pair_covers[cover_id] = cover
        for member in ordered:
            path_to_pair[_norm(member)] = ordered
    return path_to_pair, pair_covers


def build_collapsed_display_paths(
    paths: Sequence[str],
    path_to_pair: Dict[str, List[str]],
    covers: Dict[str, str],
    rejected: Optional[Set[str]] = None,
) -> List[str]:
    rejected = rejected or set()
    seen_groups: Set[str] = set()
    out: List[str] = []
    for path in paths:
        if not path or _norm(path) in rejected:
            continue
        members = path_to_pair.get(_norm(path))
        if not members:
            out.append(path)
            continue
        cover = _cover_for_members(members, covers)
        cover_id = _norm(cover)
        if cover_id in seen_groups:
            continue
        seen_groups.add(cover_id)
        out.append(cover)
    return out


def _cover_for_members(members: Sequence[str], covers: Dict[str, str]) -> str:
    for member in members:
        key = _norm(member)
        if key in covers:
            return covers[key]
        cover = covers.get(key)
        if cover:
            return cover
    raws = [p for p in members if _is_pairable_raw(p)]
    return raws[0] if raws else members[0]


def pair_members_for_path(
    file_path: str,
    path_to_pair: Dict[str, List[str]],
    *,
    rejected: Optional[Set[str]] = None,
) -> Optional[List[str]]:
    if not file_path:
        return None
    members = path_to_pair.get(_norm(file_path))
    if not members:
        return None
    rejected = rejected or set()
    visible = [p for p in members if _norm(p) not in rejected]
    raws = [p for p in visible if _is_pairable_raw(p)]
    jpegs = [p for p in visible if is_jpeg_file(p)]
    if not raws or not jpegs:
        return None
    return visible


def is_pair_cover(file_path: str, path_to_pair: Dict[str, List[str]]) -> bool:
    members = pair_members_for_path(file_path, path_to_pair)
    if not members:
        return False
    raws = [p for p in members if _is_pairable_raw(p)]
    return bool(raws) and _norm(raws[0]) == _norm(file_path)
