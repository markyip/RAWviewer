#!/usr/bin/env python3
"""
GPS mini-map POC — OSM tiles + PyQt6 (no WebEngine).

Clusters folder images within 5 m, shows current cluster centroid (green) plus
5 nearest other clusters (red). Click red cluster → gallery list or single open.

Usage:
  cd /path/to/RAWviewer
  export PYTHONPATH="$PWD/src:$PYTHONPATH"
  python scripts/poc/location_map_poc.py \\
      --folder "/Volumes/Development/Wales Aug 2025/Mark's phone"

  # Headless: GPS scan + bounds only (no GUI)
  python scripts/poc/location_map_poc.py --folder "..." --probe-only

  # Interactive GUI (scroll zoom; click red cluster → gallery / single open)
  python scripts/poc/location_map_poc.py --folder "..." \\
      --current-file PXL_20250816_150629009.RAW-01.MP.COVER.jpg --gui

  # Pick main image by filename and save map PNG
  python scripts/poc/location_map_poc.py --folder "..." \\
      --current-file PXL_20250816_150629009.RAW-01.MP.COVER.jpg \\
      --save /tmp/map.png
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

# Repo src on path when run from project root (see launch_dev.sh).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from gps_neighbors import (  # noqa: E402
    DEFAULT_CLUSTER_RADIUS_M,
    GpsCluster,
    GpsMapView,
    GpsPoint,
    MapPin,
    build_map_view,
    format_cluster_gallery_title,
    haversine_m,
    map_pins_for_view,
    resolve_current_point as resolve_current,
    scan_folder_gps,
)

OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
ESRI_SATELLITE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
OPENTOPOMAP_URL = "https://tile.opentopomap.org/{z}/{x}/{y}.png"
MAX_MAP_ZOOM = 20  # absolute ceiling; clusters stay one pin (no per-file fan-out)
MAX_ZOOM_IN_ABOVE_FIT = 2  # headroom when Z_fit is already high (tight cluster spread)
MIN_DETAIL_ZOOM = 18  # floor when Z_fit is low (wide spread — +2 alone is not enough)
TILE_NATIVE_MAX_ZOOM = 19
MIN_GEO_HALF_DEG = 0.00012  # ~13 m; keeps a sensible minimum view
PIN_MARGIN_PX = 14
VIEW_PADDING = 0.15
LEGACY_NEIGHBOR_FILE_LIMIT = 15
FIT_BOUNDS_PADDING = 0.04  # fractional pad on cluster pin bbox (Leaflet .pad() style)
FIT_BOUNDS_PIXEL_PAD = 4  # Leaflet fitBounds screen padding (px)
# OpenTopoMap z18+ returns ~4343 byte fully transparent PNGs; real tiles are usually larger.
BLANK_TILE_MAX_BYTES = 4500


@dataclass(frozen=True)
class TileStyleConfig:
    url: str
    attribution: str
    max_zoom: int


TILE_STYLES: dict[str, TileStyleConfig] = {
    "map": TileStyleConfig(OSM_TILE_URL, "© OpenStreetMap contributors", 19),
    "terrain": TileStyleConfig(OPENTOPOMAP_URL, "© OpenTopoMap · © OpenStreetMap", 17),
    "satellite": TileStyleConfig(ESRI_SATELLITE_URL, "© Esri · © OpenStreetMap", 19),
}
DEFAULT_TILE_STYLE = "map"
OSM_USER_AGENT = "RAWviewer/POC location_map_poc (contact: dev@rawviewer.local)"
TILE_SIZE = 256
DEFAULT_CACHE_DIR = Path("/tmp/rawviewer_map_poc_tiles")
WIDGET_W = 352
WIDGET_H = WIDGET_W * 3 // 4  # 4:3 (352×264)
WIDGET_ASPECT = WIDGET_W / WIDGET_H


def lat_lon_to_tile_xy(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    n = 2**zoom
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def nearest_file_neighbors(
    current: GpsPoint,
    corpus: list[GpsPoint],
    limit: int = LEGACY_NEIGHBOR_FILE_LIMIT,
) -> list[GpsPoint]:
    """Legacy G spec: Haversine-nearest files excluding current."""
    cur = os.path.normpath(current.path)
    ranked = sorted(
        (p for p in corpus if os.path.normpath(p.path) != cur),
        key=lambda p: (haversine_m(current.lat, current.lon, p.lat, p.lon), p.path),
    )
    return ranked[:limit]


def cluster_member_points(view: GpsMapView) -> list[GpsPoint]:
    """All GPS files in the green cluster + 5 red neighbor clusters."""
    pts: list[GpsPoint] = list(view.current_cluster.members)
    for cluster in view.neighbor_clusters:
        pts.extend(cluster.members)
    return pts


def centered_square_bounds(
    center: GpsPoint,
    points: Iterable[GpsPoint],
    *,
    padding: float = VIEW_PADDING,
) -> tuple[float, float, float, float]:
    """Legacy centered-square viewport: green at center, square side fits all pins."""
    half_lat = MIN_GEO_HALF_DEG
    half_lon = MIN_GEO_HALF_DEG
    for pt in points:
        half_lat = max(half_lat, abs(pt.lat - center.lat))
        half_lon = max(half_lon, abs(pt.lon - center.lon))
    half_lat *= 1 + padding
    half_lon *= 1 + padding
    half = max(half_lat, half_lon)
    return (
        center.lat - half,
        center.lat + half,
        center.lon - half,
        center.lon + half,
    )


def bounds_span_m(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
) -> tuple[float, float]:
    """Approximate geographic width (lon) and height (lat) in meters."""
    mid_lat = (min_lat + max_lat) / 2
    h = haversine_m(min_lat, min_lon, max_lat, min_lon)
    w = haversine_m(mid_lat, min_lon, mid_lat, max_lon)
    return w, h


def zoom_for_bounds(
    bounds: tuple[float, float, float, float],
    provider_max_zoom: int,
) -> int:
    return fit_base_zoom(*bounds, WIDGET_W, WIDGET_H, provider_max_zoom)


def max_display_zoom(z_fit: int, provider_max_zoom: int) -> int:
    """Upper zoom: max(Z_fit + headroom, MIN_DETAIL_ZOOM), capped by provider/absolute max."""
    target = max(z_fit + MAX_ZOOM_IN_ABOVE_FIT, MIN_DETAIL_ZOOM)
    return min(provider_max_zoom, MAX_MAP_ZOOM, target)


def padded_bounds(
    points: Iterable[GpsPoint],
    padding: float = FIT_BOUNDS_PADDING,
) -> tuple[float, float, float, float]:
    """Leaflet-style fractional pad; tiny-span guard only (~13 m), not a 100 m floor."""
    lats = [p.lat for p in points]
    lons = [p.lon for p in points]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    lat_span = max(max_lat - min_lat, MIN_GEO_HALF_DEG * 2)
    lon_span = max(max_lon - min_lon, MIN_GEO_HALF_DEG * 2)
    lat_pad = lat_span * padding
    lon_pad = lon_span * padding
    return (
        min_lat - lat_pad,
        max_lat + lat_pad,
        min_lon - lon_pad,
        max_lon + lon_pad,
    )


def bounds_pixel_size(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    zoom: int,
) -> tuple[float, float]:
    """Geographic bounds span in pixels at zoom (fractional tile coords)."""
    xnw, ynw = lat_lon_to_tile_xy(max_lat, min_lon, zoom)
    xse, yse = lat_lon_to_tile_xy(min_lat, max_lon, zoom)
    return (xse - xnw) * TILE_SIZE, (yse - ynw) * TILE_SIZE


def fit_zoom(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    width: int,
    height: int,
) -> int:
    for z in range(18, 0, -1):
        px_w, px_h = bounds_pixel_size(min_lat, max_lat, min_lon, max_lon, z)
        if px_w <= width and px_h <= height:
            return z
    return 1


def tile_range_for_bounds(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    zoom: int,
) -> tuple[int, int, int, int, float, float]:
    xnw, ynw = lat_lon_to_tile_xy(max_lat, min_lon, zoom)
    xse, yse = lat_lon_to_tile_xy(min_lat, max_lon, zoom)
    x0 = int(math.floor(xnw))
    y0 = int(math.floor(ynw))
    x1 = int(math.floor(xse))
    y1 = int(math.floor(yse))
    return x0, y0, x1, y1, xnw, ynw


def lat_lon_to_world_pixel(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    tx, ty = lat_lon_to_tile_xy(lat, lon, zoom)
    return tx * TILE_SIZE, ty * TILE_SIZE


def effective_bounds(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    base_zoom: int,
    zoom: int,
) -> tuple[float, float, float, float]:
    """Shrink geographic bbox toward its center on zoom-in (Leaflet-style)."""
    if zoom <= base_zoom:
        return min_lat, max_lat, min_lon, max_lon
    factor = 1 / (2 ** (zoom - base_zoom))
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    half_lat = max((max_lat - min_lat) / 2 * factor, MIN_GEO_HALF_DEG)
    half_lon = max((max_lon - min_lon) / 2 * factor, MIN_GEO_HALF_DEG)
    return (
        center_lat - half_lat,
        center_lat + half_lat,
        center_lon - half_lon,
        center_lon + half_lon,
    )


def viewport_for_bounds(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    zoom: int,
    base_zoom: int | None = None,
) -> tuple[float, float, float, float]:
    """Viewport from pin bounding box (fitBounds); green pin need not be centered."""
    if base_zoom is not None:
        min_lat, max_lat, min_lon, max_lon = effective_bounds(
            min_lat, max_lat, min_lon, max_lon, base_zoom, zoom
        )
    xnw, ynw = lat_lon_to_world_pixel(max_lat, min_lon, zoom)
    xse, yse = lat_lon_to_world_pixel(min_lat, max_lon, zoom)
    world_left = min(xnw, xse)
    world_top = min(ynw, yse)
    view_w = abs(xse - xnw)
    view_h = abs(yse - ynw)
    return max(view_w, 1.0), max(view_h, 1.0), world_left, world_top


def expand_world_viewport_to_aspect(
    world_left: float,
    world_top: float,
    view_w: float,
    view_h: float,
    aspect: float,
) -> tuple[float, float, float, float]:
    """Pad world-pixel viewport to target width/height aspect ratio (centered)."""
    view_h = max(view_h, 1.0)
    view_w = max(view_w, 1.0)
    current = view_w / view_h
    if current > aspect:
        new_w = view_w
        new_h = view_w / aspect
    else:
        new_h = view_h
        new_w = view_h * aspect
    cx = world_left + view_w / 2
    cy = world_top + view_h / 2
    return cx - new_w / 2, cy - new_h / 2, new_w, new_h


def viewport_for_widget(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    zoom: int,
    widget_w: int,
    widget_h: int,
    base_zoom: int | None = None,
) -> tuple[float, float, float, float]:
    """Geographic viewport cropped/padded to widget aspect (4:3)."""
    view_w, view_h, world_left, world_top = viewport_for_bounds(
        min_lat, max_lat, min_lon, max_lon, zoom, base_zoom
    )
    world_left, world_top, view_w, view_h = expand_world_viewport_to_aspect(
        world_left, world_top, view_w, view_h, widget_w / widget_h
    )
    return view_w, view_h, world_left, world_top


def fit_base_zoom(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    width: int,
    height: int,
    provider_max_zoom: int,
) -> int:
    cap = min(provider_max_zoom, MAX_MAP_ZOOM - 1)
    for z in range(cap, 0, -1):
        view_w, view_h, _, _ = viewport_for_widget(
            min_lat, max_lat, min_lon, max_lon, z, width, height
        )
        if view_w <= width and view_h <= height:
            return z
    return 1


def tile_range_for_world_rect(
    world_left: float,
    world_top: float,
    view_w: float,
    view_h: float,
) -> tuple[int, int, int, int]:
    x0 = int(math.floor(world_left / TILE_SIZE))
    y0 = int(math.floor(world_top / TILE_SIZE))
    x1 = int(math.floor((world_left + view_w - 1) / TILE_SIZE))
    y1 = int(math.floor((world_top + view_h - 1) / TILE_SIZE))
    return x0, y0, x1, y1


def world_rect_at_tile_zoom(
    world_left: float,
    world_top: float,
    view_w: float,
    view_h: float,
    display_zoom: int,
    tile_zoom: int,
) -> tuple[float, float, float, float]:
    """Convert a display-zoom world rect to tile-fetch coordinates."""
    scale = 2 ** (tile_zoom - display_zoom)
    return (
        world_left * scale,
        world_top * scale,
        view_w * scale,
        view_h * scale,
    )


def render_world_viewport(
    stitched: "QPixmap",
    world_left: float,
    world_top: float,
    view_w: float,
    view_h: float,
    x0: int,
    y0: int,
    out_w: int,
    out_h: int,
) -> "QPixmap":
    """Map world rect to widget pixels (same transform as pin positions)."""
    from PyQt6.QtCore import QRectF
    from PyQt6.QtGui import QPainter, QPixmap

    crop_x = world_left - x0 * TILE_SIZE
    crop_y = world_top - y0 * TILE_SIZE
    out = QPixmap(out_w, out_h)
    out.fill()
    painter = QPainter(out)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    painter.drawPixmap(
        QRectF(0, 0, out_w, out_h),
        stitched,
        QRectF(crop_x, crop_y, view_w, view_h),
    )
    painter.end()
    return out


def pin_positions_in_viewport(
    pins: list[MapPin],
    zoom: int,
    world_left: float,
    world_top: float,
) -> list[tuple[MapPin, float, float, str]]:
    placed: list[tuple[MapPin, float, float, str]] = []
    for pin in pins:
        px, py = lat_lon_to_world_pixel(pin.lat, pin.lon, zoom)
        placed.append((pin, px - world_left, py - world_top, pin.color))
    return placed


class TileCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, z: int, x: int, y: int) -> Path:
        return self.cache_dir / str(z) / str(x) / f"{y}.png"

    def get(self, z: int, x: int, y: int) -> Optional[Path]:
        p = self.path_for(z, x, y)
        if p.is_file() and p.stat().st_size > 0:
            if p.stat().st_size <= BLANK_TILE_MAX_BYTES:
                p.unlink(missing_ok=True)
                return None
            return p
        return None

    def put(self, z: int, x: int, y: int, data: bytes) -> Path:
        p = self.path_for(z, x, y)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return p


def fetch_tiles_sync(
    zoom: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    cache: TileCache,
    tile_url: str = OSM_TILE_URL,
) -> dict[tuple[int, int], Path]:
    results: dict[tuple[int, int], Path] = {}
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            cached = cache.get(zoom, x, y)
            if cached:
                results[(x, y)] = cached
                continue
            url = tile_url.format(z=zoom, x=x, y=y)
            req = urllib.request.Request(url, headers={"User-Agent": OSM_USER_AGENT})
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = resp.read()
            except (urllib.error.URLError, TimeoutError) as exc:
                print(f"[tile] error ({x},{y}): {exc}", file=sys.stderr)
                continue
            if len(data) <= BLANK_TILE_MAX_BYTES:
                print(
                    f"[tile] blank or missing at z={zoom} ({x},{y}) — "
                    f"try lower zoom or another --tiles layer",
                    file=sys.stderr,
                )
                continue
            if data:
                results[(x, y)] = cache.put(zoom, x, y, data)
    return results


def stitch_tiles(
    zoom: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    tiles: dict[tuple[int, int], Path],
) -> "QPixmap":
    from PyQt6.QtGui import QPixmap, QPainter

    cols = x1 - x0 + 1
    rows = y1 - y0 + 1
    w, h = cols * TILE_SIZE, rows * TILE_SIZE
    canvas = QPixmap(w, h)
    canvas.fill()
    painter = QPainter(canvas)
    for tx in range(x0, x1 + 1):
        for ty in range(y0, y1 + 1):
            src = tiles.get((tx, ty))
            if not src:
                continue
            tile = QPixmap(str(src))
            dx = (tx - x0) * TILE_SIZE
            dy = (ty - y0) * TILE_SIZE
            painter.drawPixmap(dx, dy, tile)
    painter.end()
    return canvas


def crop_background_to_bounds(
    background: "QPixmap",
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    zoom: int,
    x0: int,
    y0: int,
    origin_tx: float,
    origin_ty: float,
) -> "QPixmap":
    px_w, px_h = bounds_pixel_size(min_lat, max_lat, min_lon, max_lon, zoom)
    crop_x = int((origin_tx - x0) * TILE_SIZE)
    crop_y = int((origin_ty - y0) * TILE_SIZE)
    cw = max(1, int(math.ceil(px_w)))
    ch = max(1, int(math.ceil(px_h)))
    return background.copy(crop_x, crop_y, cw, ch)


def _draw_pin(
    painter: "QPainter",
    x: float,
    y: float,
    *,
    is_current: bool,
    badge: str = "",
) -> None:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QBrush, QColor, QFont, QPen

    radius = 8 if is_current else 5
    fill = QColor(40, 220, 90) if is_current else QColor(220, 60, 60)
    if is_current:
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        painter.setPen(QPen(QColor(255, 255, 255), 2.5))
        painter.drawEllipse(int(x - radius - 2), int(y - radius - 2), (radius + 2) * 2, (radius + 2) * 2)
    painter.setBrush(QBrush(fill))
    painter.setPen(QPen(QColor(255, 255, 255), 1.5))
    painter.drawEllipse(int(x - radius), int(y - radius), radius * 2, radius * 2)
    if badge:
        from PyQt6.QtCore import QRectF

        font_px = (8 if is_current else 7) - (1 if len(badge) > 1 else 0)
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.setFont(QFont("", font_px, QFont.Weight.Bold))
        rect = QRectF(x - radius, y - radius, radius * 2, radius * 2)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, badge)


RED_HIT_RADIUS = 8.0


def viewport_pin_to_widget(
    px: float,
    py: float,
    view_w: float,
    view_h: float,
    widget_w: int,
    widget_h: int,
) -> tuple[float, float]:
    """Map pin coords in viewport space to widget pixels (locked to map transform)."""
    return px / view_w * widget_w, py / view_h * widget_h


def layout_pins_for_display(
    pin_pixels: list[tuple[MapPin, float, float, str]],
    view_w: float,
    view_h: float,
    widget_w: int,
    widget_h: int,
) -> list[tuple[MapPin, float, float, str]]:
    """Screen positions for cluster pins (true GPS centroids)."""
    layout: list[tuple[MapPin, float, float, str]] = []
    for pin, px, py, color in pin_pixels:
        x, y = viewport_pin_to_widget(px, py, view_w, view_h, widget_w, widget_h)
        layout.append((pin, x, y, color))
    return layout


def hit_test_pin_at(
    layout: list[tuple[MapPin, float, float, str]],
    widget_x: float,
    widget_y: float,
) -> Optional[MapPin]:
    """Return clicked red cluster pin; green / miss → None."""
    best: Optional[MapPin] = None
    best_dist = float("inf")
    for pin, x, y, color in layout:
        if color == "green":
            continue
        dist = math.hypot(widget_x - x, widget_y - y)
        if dist <= RED_HIT_RADIUS and dist < best_dist:
            best = pin
            best_dist = dist
    return best


def paint_map_content(
    painter: "QPainter",
    map_frame: "QPixmap",
    pin_pixels: list[tuple[MapPin, float, float, str]],
    view_w: float,
    view_h: float,
    widget_w: int,
    widget_h: int,
    attribution: str,
) -> None:
    from PyQt6.QtGui import QColor, QFont, QPen

    painter.drawPixmap(0, 0, map_frame)

    for pin, x, y, color in layout_pins_for_display(
        pin_pixels, view_w, view_h, widget_w, widget_h
    ):
        _draw_pin(
            painter,
            x,
            y,
            is_current=(color == "green"),
            badge=pin.badge,
        )

    painter.setPen(QPen(QColor(255, 255, 255, 200)))
    painter.setFont(QFont("", 9))
    painter.drawText(6, widget_h - 8, attribution)


class LocationMapWidget:
    """Map model + optional interactive QWidget factory."""

    def __init__(
        self,
        map_view: GpsMapView,
        cache_dir: Path,
        tile_style: str = DEFAULT_TILE_STYLE,
    ):
        if tile_style not in TILE_STYLES:
            raise ValueError(f"Unknown tile style: {tile_style}")
        style = TILE_STYLES[tile_style]
        self.tile_url = style.url
        self.attribution = style.attribution
        self.max_zoom = style.max_zoom
        self.map_view = map_view
        self.pins = map_pins_for_view(map_view)
        self.cache_dir = cache_dir
        self.tile_style = tile_style
        self.bounds = padded_bounds(self.pins)
        # Initial zoom = fitBounds from cluster pin spread (not a fixed level).
        self.base_zoom = fit_base_zoom(
            *self.bounds, WIDGET_W, WIDGET_H, self.max_zoom
        )
        self.max_zoom_limit = max_display_zoom(self.base_zoom, self.max_zoom)
        self.zoom = self.base_zoom
        self.cache = TileCache(cache_dir / tile_style)
        self.cropped: "QPixmap"
        self.pin_pixels: list[tuple[MapPin, float, float, str]]
        self.view_w = 0.0
        self.view_h = 0.0
        self._rebuild_at_zoom(self.zoom, log=True)

    def can_zoom_in(self) -> bool:
        return self.zoom < self.max_zoom_limit

    def can_zoom_out(self) -> bool:
        return self.zoom > self.base_zoom

    def zoom_in(self) -> bool:
        if not self.can_zoom_in():
            return False
        prev = self.zoom
        self._rebuild_at_zoom(self.zoom + 1)
        return self.zoom > prev

    def zoom_out(self) -> bool:
        if not self.can_zoom_out():
            return False
        prev = self.zoom
        self._rebuild_at_zoom(self.zoom - 1)
        return self.zoom < prev

    def _rebuild_at_zoom(self, zoom: int, log: bool = False) -> None:
        self.zoom = min(zoom, self.max_zoom_limit)
        self.view_w, self.view_h, self.world_left, self.world_top = viewport_for_widget(
            *self.bounds,
            self.zoom,
            WIDGET_W,
            WIDGET_H,
            self.base_zoom,
        )

        tile_zoom = self.zoom
        tile_paths: dict[tuple[int, int], Path] = {}
        tw_left = tw_top = tw_w = tw_h = 0.0
        x0 = y0 = x1 = y1 = 0
        expected = 0
        native_cap = min(self.max_zoom, TILE_NATIVE_MAX_ZOOM)

        while tile_zoom >= 1:
            tw_left, tw_top, tw_w, tw_h = world_rect_at_tile_zoom(
                self.world_left,
                self.world_top,
                self.view_w,
                self.view_h,
                self.zoom,
                tile_zoom,
            )
            x0, y0, x1, y1 = tile_range_for_world_rect(tw_left, tw_top, tw_w, tw_h)
            expected = (x1 - x0 + 1) * (y1 - y0 + 1)
            t0 = time.time()
            tile_paths = fetch_tiles_sync(
                tile_zoom,
                x0,
                y0,
                x1,
                y1,
                self.cache,
                tile_url=self.tile_url,
            )
            if len(tile_paths) >= expected:
                if log:
                    extra = f" tile_z={tile_zoom}" if tile_zoom != self.zoom else ""
                    print(
                        f"[map] style={self.tile_style} zoom={self.zoom} (base={self.base_zoom})"
                        f"{extra} tiles x={x0}..{x1} y={y0}..{y1} ({expected} tiles)"
                    )
                    print(f"[map] fetched {len(tile_paths)} tiles in {time.time() - t0:.1f}s")
                break
            if tile_zoom == 1:
                if log:
                    print(
                        f"[map] warning: only {len(tile_paths)}/{expected} tiles at z=1",
                        file=sys.stderr,
                    )
                break
            if log:
                print(
                    f"[map] incomplete tiles at z={tile_zoom} ({len(tile_paths)}/{expected}) "
                    f"— upsample from z={tile_zoom - 1}",
                    file=sys.stderr,
                )
            tile_zoom -= 1

        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        stitched = stitch_tiles(tile_zoom, x0, y0, x1, y1, tile_paths)
        self.cropped = render_world_viewport(
            stitched,
            tw_left,
            tw_top,
            tw_w,
            tw_h,
            x0,
            y0,
            WIDGET_W,
            WIDGET_H,
        )
        self.pin_pixels = pin_positions_in_viewport(
            self.pins,
            self.zoom,
            self.world_left,
            self.world_top,
        )

    def hit_test_pin(self, widget_x: float, widget_y: float) -> Optional[MapPin]:
        layout = layout_pins_for_display(
            self.pin_pixels, self.view_w, self.view_h, WIDGET_W, WIDGET_H
        )
        return hit_test_pin_at(layout, widget_x, widget_y)

    @staticmethod
    def handle_cluster_click(cluster: GpsCluster) -> None:
        title = format_cluster_gallery_title(
            cluster.centroid_lat, cluster.centroid_lon, cluster.count
        )
        if cluster.count == 1:
            print(f"[map] open single: {Path(cluster.members[0].path).name}")
            return
        print(f"[map] gallery: {title}")
        for member in cluster.sorted_members():
            ts = f"  {member.capture_time}" if member.capture_time else ""
            print(f"    {Path(member.path).name}{ts}")

    def render_frame(self) -> "QPixmap":
        from PyQt6.QtGui import QColor, QPainter, QPixmap

        frame = QPixmap(WIDGET_W, WIDGET_H)
        frame.fill(QColor(20, 22, 26))
        p = QPainter(frame)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        paint_map_content(
            p,
            self.cropped,
            self.pin_pixels,
            self.view_w,
            self.view_h,
            WIDGET_W,
            WIDGET_H,
            self.attribution,
        )
        p.end()
        return frame

    def build_widget(self, on_zoom_changed=None, on_cluster_clicked=None):
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QMouseEvent, QPainter, QWheelEvent
        from PyQt6.QtWidgets import QSizePolicy, QWidget

        model = self

        class _MapWidget(QWidget):
            def __init__(inner_self, parent=None):
                super().__init__(parent)
                inner_self.setFixedSize(WIDGET_W, WIDGET_H)
                inner_self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
                inner_self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
                inner_self.setMouseTracking(False)

            def paintEvent(inner_self, event):
                del event
                p = QPainter(inner_self)
                p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                paint_map_content(
                    p,
                    model.cropped,
                    model.pin_pixels,
                    model.view_w,
                    model.view_h,
                    WIDGET_W,
                    WIDGET_H,
                    model.attribution,
                )
                p.end()

            def wheelEvent(inner_self, event: QWheelEvent):
                delta = event.angleDelta().y()
                if delta == 0:
                    return
                changed = model.zoom_in() if delta > 0 else model.zoom_out()
                if changed:
                    inner_self.update()
                    if on_zoom_changed:
                        on_zoom_changed(model.zoom)
                event.accept()

            def mousePressEvent(inner_self, event: QMouseEvent):
                if event.button() != Qt.MouseButton.LeftButton:
                    return
                pos = event.position()
                picked = model.hit_test_pin(pos.x(), pos.y())
                if picked is None:
                    return
                model.handle_cluster_click(picked.cluster)
                if on_cluster_clicked:
                    on_cluster_clicked(picked.cluster)
                event.accept()

        return _MapWidget()


def run_probe(
    folder: Path,
    current_index: int = 0,
    current_file: Optional[str] = None,
) -> int:
    points = scan_folder_gps(folder)
    print(f"Folder: {folder}")
    print(f"Images with GPS: {len(points)}")
    if not points:
        print("No GPS data found.")
        return 1
    _, current = resolve_current(points, current_index, current_file)
    view = build_map_view(current, points)
    if view is None:
        print("Could not build cluster view.")
        return 1
    cc = view.current_cluster
    print(f"\nCurrent file: {Path(current.path).name}")
    print(
        f"Green cluster centroid: ({cc.centroid_lat:.6f}, {cc.centroid_lon:.6f}) "
        f"· {cc.count} images · radius {view.radius_m:.0f}m"
    )
    print(f"\nNearest {len(view.neighbor_clusters)} clusters (red):")
    for i, cluster in enumerate(view.neighbor_clusters, 1):
        d = haversine_m(
            cc.centroid_lat, cc.centroid_lon, cluster.centroid_lat, cluster.centroid_lon
        )
        print(
            f"  {i:2d}. {cluster.count} images @ ({cluster.centroid_lat:.6f}, "
            f"{cluster.centroid_lon:.6f})  {d:.0f}m"
        )
    style = TILE_STYLES[DEFAULT_TILE_STYLE]
    pins = map_pins_for_view(view)
    members = cluster_member_points(view)
    legacy_neighbors = nearest_file_neighbors(current, points)
    legacy_pins = [current, *legacy_neighbors]

    strategies: list[tuple[str, tuple[float, float, float, float]]] = [
        ("fitBounds · cluster centroids (current)", padded_bounds(pins)),
        ("fitBounds · cluster member files", padded_bounds(members)),
        ("fitBounds · nearest 15 files (legacy)", padded_bounds(legacy_pins)),
        ("centered-square · cluster centroids", centered_square_bounds(current, pins)),
        ("centered-square · nearest 15 files (legacy)", centered_square_bounds(current, legacy_pins)),
    ]

    print(f"\nZoom comparison for {WIDGET_W}x{WIDGET_H} (higher Z = closer):")
    best_name = ""
    best_z = -1
    rows: list[tuple[str, int, float, float]] = []
    for name, raw_bounds in strategies:
        z = zoom_for_bounds(raw_bounds, style.max_zoom)
        w_m, h_m = bounds_span_m(*raw_bounds)
        rows.append((name, z, w_m, h_m))
        if z > best_z:
            best_z = z
            best_name = name

    for name, z, w_m, h_m in rows:
        marker = " ← current" if name.startswith("fitBounds · cluster centroids") else ""
        winner = " · closest" if z == best_z else ""
        print(f"  {name:42s}  z{z:2d}  {w_m:5.0f}m × {h_m:5.0f}m{marker}{winner}")

    current_z = next(z for n, z, _, _ in rows if n.startswith("fitBounds · cluster centroids"))
    legacy_z = next(z for n, z, _, _ in rows if n.startswith("fitBounds · nearest 15"))
    delta = current_z - legacy_z
    if delta > 0:
        print(f"\nCluster fitBounds is {delta} zoom level(s) closer than legacy 15-file fitBounds.")
    elif delta < 0:
        print(f"\nLegacy 15-file fitBounds is {-delta} zoom level(s) closer than cluster fitBounds.")
    else:
        print("\nCluster and legacy 15-file fitBounds yield the same Z_fit.")

    bounds = padded_bounds(pins)
    vw, vh, _, _ = viewport_for_widget(*bounds, current_z, WIDGET_W, WIDGET_H)
    print(f"\nActive bounds (cluster fitBounds): lat [{bounds[0]:.6f}, {bounds[1]:.6f}] "
          f"lon [{bounds[2]:.6f}, {bounds[3]:.6f}]")
    z_max = max_display_zoom(current_z, style.max_zoom)
    print(
        f"Z_fit (initial + min zoom): {current_z} "
        f"(view {vw:.0f}x{vh:.0f}px · {DEFAULT_CLUSTER_RADIUS_M:.0f}m clusters)"
    )
    z_by_fit = current_z + MAX_ZOOM_IN_ABOVE_FIT
    limiting = "fit+headroom" if z_by_fit >= MIN_DETAIL_ZOOM else "min detail floor"
    print(
        f"Z_max (zoom-in limit): {z_max} (max(fit+{MAX_ZOOM_IN_ABOVE_FIT}={z_by_fit}, "
        f"detail>={MIN_DETAIL_ZOOM}) · {limiting} · clusters stay grouped)"
    )
    return 0


def run_gui(
    folder: Path,
    current_index: int,
    cache_dir: Path,
    current_file: Optional[str] = None,
    tile_style: str = DEFAULT_TILE_STYLE,
) -> int:
    from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

    points = scan_folder_gps(folder)
    if not points:
        print("No GPS data found.", file=sys.stderr)
        return 1
    _, current = resolve_current(points, current_index, current_file)
    view = build_map_view(current, points)
    if view is None:
        print("Could not build cluster view.", file=sys.stderr)
        return 1

    app = QApplication([sys.argv[0]])
    app.setApplicationName("RAWviewer Map POC")

    model = LocationMapWidget(view, cache_dir, tile_style=tile_style)

    def _status_text() -> str:
        cc = model.map_view.current_cluster
        return (
            f"Green = current cluster ({cc.count} img) · "
            f"Red = {len(model.map_view.neighbor_clusters)} clusters · "
            f"zoom {model.zoom} ({model.base_zoom}–{model.max_zoom_limit}) · "
            f"click red cluster · scroll to zoom · {model.attribution}"
        )

    status = QLabel(_status_text())
    window = QMainWindow()

    def _refresh_chrome() -> None:
        status.setText(_status_text())
        window.setWindowTitle(
            f"RAWviewer GPS Map POC — {Path(model.map_view.current_file.path).name}"
        )

    map_widget = model.build_widget(
        on_zoom_changed=lambda _z: _refresh_chrome(),
        on_cluster_clicked=lambda _c: None,
    )

    window.setWindowTitle(f"RAWviewer GPS Map POC — {Path(current.path).name}")
    central = QWidget()
    layout = QVBoxLayout(central)
    layout.addWidget(status)
    layout.addWidget(map_widget)
    window.setCentralWidget(central)
    window.resize(WIDGET_W + 40, WIDGET_H + 80)
    window.show()

    print("[gui] Cluster map — click red cluster for gallery / single open.")
    return app.exec()


def run_save(
    folder: Path,
    save_path: Path,
    current_index: int,
    cache_dir: Path,
    current_file: Optional[str] = None,
    tile_style: str = DEFAULT_TILE_STYLE,
) -> int:
    from PyQt6.QtWidgets import QApplication

    points = scan_folder_gps(folder)
    if not points:
        print("No GPS data found.", file=sys.stderr)
        return 1
    _, current = resolve_current(points, current_index, current_file)
    view = build_map_view(current, points)
    if view is None:
        print("Could not build cluster view.", file=sys.stderr)
        return 1

    app = QApplication([sys.argv[0]])
    app.setApplicationName("RAWviewer Map POC")

    model = LocationMapWidget(view, cache_dir, tile_style=tile_style)
    frame = model.render_frame()
    save_path = save_path.expanduser()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not frame.save(str(save_path)):
        print(f"Failed to save: {save_path}", file=sys.stderr)
        return 1
    cc = view.current_cluster
    print(f"Saved map → {save_path}")
    print(f"  Green cluster: {cc.count} images ({cc.centroid_lat:.6f}, {cc.centroid_lon:.6f})")
    print(f"  Red clusters: {len(view.neighbor_clusters)} · zoom {model.zoom}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="RAWviewer GPS mini-map POC (OSM + Qt)")
    parser.add_argument(
        "--folder",
        type=Path,
        default=Path("/Volumes/Development/Wales Aug 2025/Mark's phone"),
        help="Folder of images to scan",
    )
    parser.add_argument(
        "--current-index",
        type=int,
        default=0,
        help="Index into GPS-sorted file list for green pin (default: 0)",
    )
    parser.add_argument(
        "--current-file",
        type=str,
        default=None,
        help="Filename (or substring) for green pin; overrides --current-index",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        metavar="PATH",
        help="Render map to PNG and exit (no interactive window)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Tile disk cache (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--tiles",
        dest="tile_style",
        choices=sorted(TILE_STYLES),
        default=DEFAULT_TILE_STYLE,
        help="Tile layer: terrain (OpenTopoMap), map (OSM), or satellite (Esri aerial)",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Scan GPS + print neighbors/bounds; no GUI or tile download",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open interactive map window (default when --save / --probe-only omitted)",
    )
    args = parser.parse_args()
    folder = args.folder.expanduser()
    if not folder.is_dir():
        print(f"Not a directory: {folder}", file=sys.stderr)
        return 1
    if args.probe_only:
        return run_probe(folder, args.current_index, args.current_file)
    if args.save:
        return run_save(
            folder,
            args.save,
            args.current_index,
            args.cache_dir,
            args.current_file,
            args.tile_style,
        )
    return run_gui(folder, args.current_index, args.cache_dir, args.current_file, args.tile_style)


if __name__ == "__main__":
    raise SystemExit(main())
