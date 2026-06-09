"""
GPS clustering and nearest-cluster selection for the location map overlay (G).

Groups folder images within a radius (default 5 m), shows the current file's
cluster centroid (green) plus up to 5 nearest other clusters (red).
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import metadata_backend

DEFAULT_CLUSTER_RADIUS_M = 5.0
DEFAULT_NEIGHBOR_CLUSTER_LIMIT = 5

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
    ".dng",
    ".raw",
    ".cr2",
    ".nef",
    ".arw",
    ".orf",
    ".rw2",
}


@dataclass(frozen=True)
class GpsPoint:
    path: str
    lat: float
    lon: float
    capture_time: str = ""


@dataclass(frozen=True)
class GpsCluster:
    cluster_id: int
    centroid_lat: float
    centroid_lon: float
    members: tuple[GpsPoint, ...]

    @property
    def count(self) -> int:
        return len(self.members)

    def sorted_members(self) -> list[GpsPoint]:
        return list(self.members)


@dataclass(frozen=True)
class GpsMapView:
    """Green = current cluster centroid; red = up to 5 nearest other clusters."""

    current_file: GpsPoint
    current_cluster: GpsCluster
    neighbor_clusters: tuple[GpsCluster, ...]
    radius_m: float


@dataclass(frozen=True)
class MapPin:
    lat: float
    lon: float
    color: str
    cluster: GpsCluster
    badge: str


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def format_cluster_gallery_title(lat: float, lon: float, count: int) -> str:
    """Gallery header: GPS cluster · Long/Lat · X images."""
    return f"GPS cluster · {lon:.5f}, {lat:.5f} · {count} images"


def _tag_text(tags: dict, *keys: str) -> str:
    for key in keys:
        tag = tags.get(key)
        if tag is None:
            continue
        val = getattr(tag, "values", tag)
        if isinstance(val, (list, tuple)) and val:
            val = val[0]
        s = str(val).strip()
        if s:
            return s
    return ""


def _ratio_to_float(v: Any) -> Optional[float]:
    try:
        if hasattr(v, "num") and hasattr(v, "den"):
            den = float(v.den) if float(v.den) != 0 else 1.0
            return float(v.num) / den
        if isinstance(v, str) and "/" in v:
            num, den = v.split("/", 1)
            d = float(den) if float(den) != 0 else 1.0
            return float(num) / d
        return float(v)
    except Exception:
        return None


def gps_to_decimal(gps_vals: Any, ref: str) -> Optional[float]:
    try:
        vals = getattr(gps_vals, "values", gps_vals)
        if isinstance(vals, str):
            import re

            parts = re.split(r"[\s,]+", vals.strip())
            if len(parts) >= 3:
                parsed = []
                for p in parts:
                    if "/" in p:
                        num, den = p.split("/", 1)
                        parsed.append(float(num) / (float(den) if float(den) != 0 else 1.0))
                    else:
                        parsed.append(float(p))
                vals = parsed
        if not vals or not isinstance(vals, (list, tuple)) or len(vals) < 3:
            return None
        d = _ratio_to_float(vals[0])
        m = _ratio_to_float(vals[1])
        s = _ratio_to_float(vals[2])
        if d is None or m is None or s is None:
            return None
        dec = float(d) + float(m) / 60.0 + float(s) / 3600.0
        ref_str = str(ref or "").strip().upper()
        if ref_str in ("S", "W"):
            dec = -dec
        return dec
    except Exception:
        return None


def extract_gps_point(file_path: str) -> Optional[GpsPoint]:
    try:
        tags = metadata_backend.process_file_from_path(
            file_path,
            details=False,
            stop_tag="GPS GPSLongitudeRef",
        )
    except Exception:
        return None
    lat = tags.get("GPS GPSLatitude") or tags.get("EXIF GPSLatitude") or tags.get("GPSLatitude")
    lon = tags.get("GPS GPSLongitude") or tags.get("EXIF GPSLongitude") or tags.get("GPSLongitude")
    if not lat or not lon:
        return None
    lat_ref = _tag_text(tags, "GPS GPSLatitudeRef", "EXIF GPSLatitudeRef", "GPSLatitudeRef")
    lon_ref = _tag_text(tags, "GPS GPSLongitudeRef", "EXIF GPSLongitudeRef", "GPSLongitudeRef")
    lat_dec = gps_to_decimal(lat, lat_ref)
    lon_dec = gps_to_decimal(lon, lon_ref)
    if lat_dec is None or lon_dec is None:
        return None
    if abs(lat_dec) <= 0.001 and abs(lon_dec) <= 0.001:
        return None
    capture_time = _tag_text(
        tags,
        "EXIF DateTimeOriginal",
        "Image DateTime",
        "EXIF DateTime",
    )
    return GpsPoint(path=file_path, lat=lat_dec, lon=lon_dec, capture_time=capture_time)


def scan_folder_gps(folder: Path) -> list[GpsPoint]:
    points: list[GpsPoint] = []
    for entry in sorted(folder.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        pt = extract_gps_point(str(entry))
        if pt:
            points.append(pt)
    return points


def _sort_key_capture_time(pt: GpsPoint) -> tuple[str, str]:
    return (pt.capture_time or "", pt.path)


def cluster_points(
    points: Iterable[GpsPoint],
    radius_m: float = DEFAULT_CLUSTER_RADIUS_M,
) -> list[GpsCluster]:
    """Union-find clustering: any pair within radius_m shares a cluster."""
    pts = list(points)
    n = len(pts)
    if not pts:
        return []

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if haversine_m(pts[i].lat, pts[i].lon, pts[j].lat, pts[j].lon) <= radius_m:
                union(i, j)

    groups: dict[int, list[GpsPoint]] = defaultdict(list)
    for i, pt in enumerate(pts):
        groups[find(i)].append(pt)

    clusters: list[GpsCluster] = []
    for cluster_id, members in enumerate(groups.values()):
        members_sorted = tuple(sorted(members, key=_sort_key_capture_time))
        lat = sum(m.lat for m in members_sorted) / len(members_sorted)
        lon = sum(m.lon for m in members_sorted) / len(members_sorted)
        clusters.append(
            GpsCluster(
                cluster_id=cluster_id,
                centroid_lat=lat,
                centroid_lon=lon,
                members=members_sorted,
            )
        )
    return clusters


def cluster_for_point(point: GpsPoint, clusters: list[GpsCluster]) -> Optional[GpsCluster]:
    path = os.path.normpath(point.path)
    for cluster in clusters:
        for m in cluster.members:
            if os.path.normpath(m.path) == path:
                return cluster
    return None


def nearest_neighbor_clusters(
    current_cluster: GpsCluster,
    all_clusters: list[GpsCluster],
    limit: int = DEFAULT_NEIGHBOR_CLUSTER_LIMIT,
) -> list[GpsCluster]:
    others = [c for c in all_clusters if c.cluster_id != current_cluster.cluster_id]
    ranked = sorted(
        others,
        key=lambda c: (
            haversine_m(
                current_cluster.centroid_lat,
                current_cluster.centroid_lon,
                c.centroid_lat,
                c.centroid_lon,
            ),
            c.cluster_id,
        ),
    )
    return ranked[:limit]


def build_map_view(
    current: GpsPoint,
    corpus: list[GpsPoint],
    *,
    radius_m: float = DEFAULT_CLUSTER_RADIUS_M,
    neighbor_limit: int = DEFAULT_NEIGHBOR_CLUSTER_LIMIT,
) -> Optional[GpsMapView]:
    clusters = cluster_points(corpus, radius_m)
    current_cluster = cluster_for_point(current, clusters)
    if current_cluster is None:
        return None
    neighbors = nearest_neighbor_clusters(current_cluster, clusters, neighbor_limit)
    return GpsMapView(
        current_file=current,
        current_cluster=current_cluster,
        neighbor_clusters=tuple(neighbors),
        radius_m=radius_m,
    )


def map_pins_for_view(view: GpsMapView) -> list[MapPin]:
    pins: list[MapPin] = [
        MapPin(
            lat=view.current_cluster.centroid_lat,
            lon=view.current_cluster.centroid_lon,
            color="green",
            cluster=view.current_cluster,
            badge=str(view.current_cluster.count),
        )
    ]
    for cluster in view.neighbor_clusters:
        pins.append(
            MapPin(
                lat=cluster.centroid_lat,
                lon=cluster.centroid_lon,
                color="red",
                cluster=cluster,
                badge=str(cluster.count),
            )
        )
    return pins


def corpus_gps_points(file_paths: Iterable[str]) -> list[GpsPoint]:
    """GPS points for the viewer corpus (e.g. current folder file list)."""
    points: list[GpsPoint] = []
    for file_path in file_paths:
        pt = extract_gps_point(file_path)
        if pt:
            points.append(pt)
    return points


def current_point_for_path(
    points: list[GpsPoint],
    current_path: Optional[str],
) -> Optional[GpsPoint]:
    if not current_path or not points:
        return None
    target = os.path.normpath(current_path)
    for pt in points:
        if os.path.normpath(pt.path) == target:
            return pt
    name = Path(current_path).name.lower()
    for pt in points:
        if Path(pt.path).name.lower() == name:
            return pt
    return None


def resolve_current_point(
    points: list[GpsPoint],
    current_index: int = 0,
    current_file: Optional[str] = None,
) -> tuple[int, GpsPoint]:
    if current_file:
        needle = current_file.lower()
        for i, pt in enumerate(points):
            name = Path(pt.path).name.lower()
            if name == needle or needle in name:
                return i, pt
        raise SystemExit(f"No GPS image matching --current-file {current_file!r}")
    idx = min(max(0, current_index), len(points) - 1)
    return idx, points[idx]
