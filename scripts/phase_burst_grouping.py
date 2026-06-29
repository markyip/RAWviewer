#!/usr/bin/env python3
"""Smoke tests for burst_grouping helpers."""

import sys

sys.path.insert(0, "src")

from burst_grouping import (
    burst_grouping_enabled_from_settings,
    burst_groups_for_folder,
    format_burst_group_status,
    burst_span_seconds,
    selection_covers_single_burst_group,
)


class _SettingsStub:
    def __init__(self, value):
        self._value = value

    def value(self, key, default=None):
        return self._value if self._value is not None else default


def test_burst_grouping_enabled_from_settings():
    assert burst_grouping_enabled_from_settings(_SettingsStub(True)) is True
    assert burst_grouping_enabled_from_settings(_SettingsStub(False)) is False
    assert burst_grouping_enabled_from_settings(_SettingsStub("0")) is False
    assert burst_grouping_enabled_from_settings(_SettingsStub("off")) is False
    assert burst_grouping_enabled_from_settings(_SettingsStub(None)) is True


def test_format_burst_group_status():
    times = {
        "/a.ARW": "2024:01:01 12:00:00",
        "/b.ARW": "2024:01:01 12:00:02",
        "/c.ARW": "2024:01:01 12:00:02.500000",
    }
    members = ["/a.ARW", "/b.ARW", "/c.ARW"]
    label = format_burst_group_status(members, times)
    assert label.startswith("3 images /"), label
    assert label.endswith("s"), label
    assert format_burst_group_status(["/a.ARW"], times) == "1 image"


def test_burst_groups_for_folder():
    from burst_grouping import _norm

    paths = ["p1.ARW", "p2.ARW", "p3.ARW", "solo.ARW"]
    times = {
        "p1.ARW": "2024:01:01 12:00:00",
        "p2.ARW": "2024:01:01 12:00:00.2",
        "p3.ARW": "2024:01:01 12:00:00.5",
        "solo.ARW": "2024:01:01 12:05:00",
    }
    groups, path_to_group, covers = burst_groups_for_folder(
        paths, times, gap_sec=1.0
    )
    assert len(groups) == 1
    assert len(groups[0]) == 3
    assert _norm("solo.ARW") not in path_to_group

    # RAW+JPEG pair (2 frames within 1s) must not stack
    pair_paths = ["raw.ARW", "raw.jpg", "solo2.ARW"]
    pair_times = {
        "raw.ARW": "2024:01:01 12:00:00",
        "raw.jpg": "2024:01:01 12:00:00.1",
        "solo2.ARW": "2024:01:01 12:05:00",
    }
    pair_groups, pair_map, _ = burst_groups_for_folder(
        pair_paths, pair_times, gap_sec=1.0
    )
    assert len(pair_groups) == 0
    assert _norm("raw.ARW") not in pair_map


def test_selection_covers_single_burst_group():
    from burst_grouping import _norm

    paths = ["p1.ARW", "p2.ARW", "p3.ARW", "solo.ARW"]
    times = {
        "p1.ARW": "2024:01:01 12:00:00",
        "p2.ARW": "2024:01:01 12:00:00.2",
        "p3.ARW": "2024:01:01 12:00:00.5",
        "solo.ARW": "2024:01:01 12:05:00",
    }
    _, path_to_group, _ = burst_groups_for_folder(
        paths, times, gap_sec=1.0
    )
    group_keys = {_norm(m) for m in next(iter(path_to_group.values()))}
    assert selection_covers_single_burst_group(group_keys, path_to_group, set()) is not None
    mixed = group_keys | {_norm("solo.ARW")}
    assert selection_covers_single_burst_group(mixed, path_to_group, set()) is None


if __name__ == "__main__":
    test_burst_grouping_enabled_from_settings()
    test_format_burst_group_status()
    test_burst_groups_for_folder()
    test_selection_covers_single_burst_group()
    print("phase_burst_grouping: OK")
