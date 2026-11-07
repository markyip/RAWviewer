"""Shared types for semantic search."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

from core.errors import IndexAborted  # re-export

ProgressCallback: TypeAlias = Callable[[int, int, str], None] | None

Row: TypeAlias = sqlite3.Row

MetadataRow: TypeAlias = sqlite3.Row | dict[str, object]


@dataclass
class SearchHit:
    file_path: str
    score: float
    file_name: str = ""
    capture_time: str = ""
    camera_model: str = ""
    lens_model: str = ""
    iso: int = 0
    gps_lat: float | None = None
    gps_lon: float | None = None
    city: str = ""
    landmark: str = ""
    admin1: str = ""
    country: str = ""
    country_code: str = ""
    face_count: int = 0

# Re-export for convenience
__all__ = [
    "IndexAborted",
    "MetadataRow",
    "ProgressCallback",
    "SearchHit",
]
