#!/usr/bin/env python3
"""Smoke tests for burst_grouping helpers."""

import sys

sys.path.insert(0, "src")

from burst_grouping import (
    burst_groups_for_folder,
    format_burst_group_status,
    burst_span_seconds,
)


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
    paths = ["p1.ARW", "p2.ARW", "p3.ARW", "solo.ARW"]
    times = {
        "p1.ARW": "2024:01:01 12:00:00",
        "p2.ARW": "2024:01:01 12:00:01",
        "p3.ARW": "2024:01:01 12:00:02",
        "solo.ARW": "2024:01:01 12:05:00",
    }
    groups, path_to_group, covers = burst_groups_for_folder(
        paths, times, gap_sec=1.0, min_count=2
    )
    assert len(groups) == 1
    assert len(groups[0]) == 3
    assert "/solo.ARW" not in path_to_group


if __name__ == "__main__":
    test_format_burst_group_status()
    test_burst_groups_for_folder()
    print("phase_burst_grouping: OK")
