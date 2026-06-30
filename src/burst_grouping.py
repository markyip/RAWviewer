"""Burst image grouping by EXIF capture time (H1)."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Set, Tuple

from common_image_loader import parse_capture_time_to_timestamp

BURST_MIN_COUNT = 3  # Avoid stacking RAW+JPEG pairs; real bursts are usually 3+ frames/s
DEFAULT_BURST_GAP_SECONDS = 1.0
BURST_GROUPING_SETTINGS_KEY = "gallery/burst_grouping_enabled"


def burst_grouping_enabled_from_settings(settings) -> bool:
    """Whether burst stacks are shown in the gallery (default off)."""
    try:
        raw = settings.value(BURST_GROUPING_SETTINGS_KEY, False)
        if isinstance(raw, str):
            return raw.strip().lower() not in ("0", "false", "no", "off")
        return bool(raw)
    except Exception:
        return False


def burst_gap_seconds_from_settings(settings) -> float:
    try:
        value = float(
            settings.value("gallery/burst_gap_seconds", DEFAULT_BURST_GAP_SECONDS)
        )
        return max(0.05, value)
    except (TypeError, ValueError):
        return DEFAULT_BURST_GAP_SECONDS


def _norm(path: str) -> str:
    if not path:
        return ""
    # normcase keeps these keys consistent with image_cache._exif_cache_path_key
    # and main._norm_path so capture-time lookups, the rejected set, and
    # path_to_group all match on Windows (case-insensitive paths). Without it,
    # capture_times keys (lowercased) never matched and no bursts grouped.
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def _capture_times_lookup(
    path: str, capture_times: Dict[str, str]
) -> Optional[str]:
    if path in capture_times:
        return capture_times[path]
    key = _norm(path)
    if key in capture_times:
        return capture_times[key]
    for raw, value in capture_times.items():
        if _norm(raw) == key:
            return value
    return None


def timestamp_for_path(path: str, capture_times: Dict[str, str]) -> float:
    ct = _capture_times_lookup(path, capture_times)
    if ct:
        ts = parse_capture_time_to_timestamp(ct)
        if ts > 0:
            return ts
    return 0.0


def sort_paths_by_capture_time(
    paths: List[str],
    capture_times: Dict[str, str],
) -> List[str]:
    indexed = list(enumerate(paths))
    indexed.sort(
        key=lambda item: (
            timestamp_for_path(item[1], capture_times),
            item[0],
        )
    )
    return [path for _, path in indexed]


def burst_groups_for_folder(
    paths: List[str],
    capture_times: Dict[str, str],
    *,
    gap_sec: float = DEFAULT_BURST_GAP_SECONDS,
    min_count: int = BURST_MIN_COUNT,
    rejected: Optional[Set[str]] = None,
) -> Tuple[List[List[str]], Dict[str, List[str]], Dict[str, str]]:
    """Group adjacent captures by time gap.

    Returns:
        groups: member lists for stacks with at least ``min_count`` visible members
        path_to_group: normpath -> full member list (capture-time order)
        group_covers: group_id (earliest member normpath) -> cover path
    """
    rejected = rejected or set()
    visible = [p for p in paths if p and _norm(p) not in rejected]
    if not visible:
        return [], {}, {}

    by_time = sort_paths_by_capture_time(visible, capture_times)
    groups: List[List[str]] = []
    current: List[str] = [by_time[0]]

    for path in by_time[1:]:
        prev_ts = timestamp_for_path(current[-1], capture_times)
        cur_ts = timestamp_for_path(path, capture_times)
        gap = abs(cur_ts - prev_ts) if prev_ts > 0 and cur_ts > 0 else gap_sec + 1.0
        if prev_ts <= 0 or cur_ts <= 0:
            gap = gap_sec + 1.0
        if gap <= gap_sec:
            current.append(path)
        else:
            groups.append(current)
            current = [path]
    groups.append(current)

    path_to_group: Dict[str, List[str]] = {}
    group_covers: Dict[str, str] = {}
    stack_groups: List[List[str]] = []

    for members in groups:
        visible_members = [m for m in members if _norm(m) not in rejected]
        if len(visible_members) < min_count:
            continue
        group_id = _norm(visible_members[0])
        stack_groups.append(visible_members)
        group_covers[group_id] = visible_members[0]
        for member in visible_members:
            path_to_group[_norm(member)] = visible_members

    return stack_groups, path_to_group, group_covers


def build_collapsed_display_paths(
    ordered_paths: List[str],
    path_to_group: Dict[str, List[str]],
    group_covers: Dict[str, str],
    rejected: Set[str],
    *,
    min_count: int = BURST_MIN_COUNT,
) -> List[str]:
    """One tile per burst stack (cover) plus standalone images."""
    emitted: Set[str] = set()
    collapsed: List[str] = []
    for path in ordered_paths:
        if not path or _norm(path) in rejected:
            continue
        members = path_to_group.get(_norm(path))
        if members:
            visible = [m for m in members if _norm(m) not in rejected]
            if len(visible) >= min_count:
                group_id = _norm(visible[0])
                if group_id in emitted:
                    continue
                emitted.add(group_id)
                cover = group_covers.get(group_id) or visible[0]
                if _norm(cover) in rejected:
                    cover = visible[0]
                collapsed.append(cover)
                continue
        collapsed.append(path)
    return collapsed


def stack_count_for_cover(
    cover_path: str,
    path_to_group: Dict[str, List[str]],
    rejected: Set[str],
    *,
    min_count: int = BURST_MIN_COUNT,
) -> int:
    members = path_to_group.get(_norm(cover_path))
    if not members:
        return 0
    visible = [m for m in members if _norm(m) not in rejected]
    return len(visible) if len(visible) >= min_count else 0


def visible_burst_members(
    path: str,
    path_to_group: Dict[str, List[str]],
    rejected: Set[str],
    *,
    min_count: int = BURST_MIN_COUNT,
) -> Optional[List[str]]:
    members = path_to_group.get(_norm(path))
    if not members:
        return None
    visible = [m for m in members if _norm(m) not in rejected]
    return visible if len(visible) >= min_count else None


def burst_group_id(
    path: str,
    path_to_group: Dict[str, List[str]],
    rejected: Set[str],
) -> Optional[str]:
    visible = visible_burst_members(path, path_to_group, rejected)
    if not visible:
        return None
    return _norm(visible[0])


def selection_covers_single_burst_group(
    selected_norm_keys: Set[str],
    path_to_group: Dict[str, List[str]],
    rejected: Set[str],
) -> Optional[List[str]]:
    """Return member paths when ``selected_norm_keys`` is exactly one burst group."""
    if len(selected_norm_keys) < BURST_MIN_COUNT:
        return None
    group_ids: Set[str] = set()
    for key in selected_norm_keys:
        gid = burst_group_id(key, path_to_group, rejected)
        if gid is None:
            return None
        group_ids.add(gid)
    if len(group_ids) != 1:
        return None
    gid = next(iter(group_ids))
    visible = visible_burst_members(gid, path_to_group, rejected)
    if not visible:
        return None
    visible_keys = {_norm(m) for m in visible}
    if selected_norm_keys != visible_keys:
        return None
    return visible


def burst_span_seconds(
    members: List[str],
    capture_times: Dict[str, str],
) -> float:
    """Elapsed seconds between earliest and latest capture in ``members``."""
    if not members:
        return 0.0
    stamps = [
        timestamp_for_path(p, capture_times)
        for p in members
    ]
    stamps = [t for t in stamps if t > 0]
    if len(stamps) < 2:
        return 0.0
    return max(0.0, max(stamps) - min(stamps))


def format_burst_group_status(
    members: List[str],
    capture_times: Dict[str, str],
) -> str:
    """Top metadata for burst group view, e.g. ``12 images / 2.34s`` or ``1 image``."""
    count = len(members)
    if count <= 0:
        return ""
    noun = "image" if count == 1 else "images"
    span = burst_span_seconds(members, capture_times)
    if span <= 0:
        return f"{count} {noun}"
    if span >= 10:
        span_text = f"{span:.1f}"
    elif span >= 1:
        span_text = f"{span:.2f}".rstrip("0").rstrip(".")
    else:
        span_text = f"{span:.2f}".rstrip("0").rstrip(".") or "0"
    return f"{count} {noun} / {span_text}s"
