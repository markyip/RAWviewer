"""Simplified OSM tile map engine for displaying single-image GPS coordinates."""

from __future__ import annotations

import logging
import math
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
OSM_USER_AGENT = "RAWviewer/3.0 (contact: dev@rawviewer.local)"
TILE_SIZE = 256
WIDGET_W = 352
WIDGET_H = 264
DEFAULT_ZOOM = 15
BLANK_TILE_MAX_BYTES = 4500


def lat_lon_to_tile_xy(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    """Convert geographic coordinates to fractional tile index (Web Mercator)."""
    n = 2**zoom
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def lat_lon_to_world_pixel(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    tx, ty = lat_lon_to_tile_xy(lat, lon, zoom)
    return tx * TILE_SIZE, ty * TILE_SIZE


class TileCache:
    """Caching of map tiles locally on disk."""

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
    """Fetch tiles in range x0..x1 and y0..y1, returning paths to the cached image files."""
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
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = resp.read()
            except (urllib.error.URLError, TimeoutError) as exc:
                logger.warning("[tile] error fetching tile (%d,%d) at z=%d: %s", x, y, zoom, exc)
                continue
            if len(data) <= BLANK_TILE_MAX_BYTES:
                logger.warning("[tile] blank or missing tile at z=%d (%d,%d)", zoom, x, y)
                continue
            if data:
                results[(x, y)] = cache.put(zoom, x, y, data)
    return results


def stitch_tiles(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    tiles: dict[tuple[int, int], Path],
) -> "QPixmap":
    """Stitch a grid of tiles into a single QPixmap."""
    from PyQt6.QtGui import QPainter, QPixmap

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
    """Crop the stitched tiles QPixmap to match the viewport bounding box."""
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


def _draw_pin(
    painter: "QPainter",
    x: float,
    y: float,
    *,
    is_current: bool,
) -> None:
    """Draw a reverse-teardrop pin matching the location-card reference.

    Solid ember fill, sharp tip on the GPS point, soft radial glow behind the
    head (current pin only). Neighbors are smaller/muted without glow.
    """
    from PyQt6.QtCore import Qt, QPointF, QRectF
    from PyQt6.QtGui import QBrush, QColor, QPainterPath, QRadialGradient

    # theme.EMBER = #d9691e → (217, 105, 30)
    if is_current:
        head_r = 8.0
        tip_h = 13.0
        fill = QColor(217, 105, 30)
        glow_alpha = 120
        glow_r = 24.0
    else:
        head_r = 5.5
        tip_h = 9.0
        fill = QColor(217, 105, 30, 175)
        glow_alpha = 0
        glow_r = 0.0

    head_cx = float(x)
    # Tip on GPS; circular head sits above (reverse teardrop / map marker).
    head_cy = float(y) - tip_h
    tip = QPointF(float(x), float(y))

    # Continuous silhouette: tip → left flank → full circular head → right flank → tip.
    path = QPainterPath()
    path.moveTo(tip)
    path.lineTo(QPointF(head_cx - head_r * 0.72, head_cy + head_r * 0.55))
    path.arcTo(
        QRectF(head_cx - head_r, head_cy - head_r, head_r * 2.0, head_r * 2.0),
        200.0,
        -220.0,
    )
    path.lineTo(tip)
    path.closeSubpath()

    painter.setRenderHint(painter.RenderHint.Antialiasing, True)

    if glow_alpha > 0:
        glow = QRadialGradient(QPointF(head_cx, head_cy + head_r * 0.15), glow_r)
        glow.setColorAt(0.0, QColor(217, 105, 30, glow_alpha))
        glow.setColorAt(0.4, QColor(217, 105, 30, int(glow_alpha * 0.5)))
        glow.setColorAt(1.0, QColor(217, 105, 30, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(glow))
        painter.drawEllipse(
            QPointF(head_cx, head_cy + head_r * 0.15),
            glow_r,
            glow_r,
        )

    painter.setBrush(QBrush(fill))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawPath(path)




class LocationMapModel:
    """Model that handles coordinates for a single image, tile fetching, and stitching."""

    def __init__(
        self,
        lat: float,
        lon: float,
        cache_dir: Path,
        other_points: Optional[list[tuple[float, float]]] = None,
    ):
        self.lat = lat
        self.lon = lon
        self.zoom = DEFAULT_ZOOM
        self.cache_dir = cache_dir
        self.cache = TileCache(cache_dir / "map")
        
        # Center of map is the active image's location
        cx, cy = lat_lon_to_tile_xy(self.lat, self.lon, self.zoom)
        center_px = cx * TILE_SIZE
        center_py = cy * TILE_SIZE
        
        # Determine top-left corner of the widget in world coordinates
        self.world_left = center_px - (WIDGET_W / 2)
        self.world_top = center_py - (WIDGET_H / 2)
        self.view_w = WIDGET_W
        self.view_h = WIDGET_H
        
        # Determine bounds of tiles to fetch
        x0 = int(math.floor(self.world_left / TILE_SIZE))
        y0 = int(math.floor(self.world_top / TILE_SIZE))
        x1 = int(math.floor((self.world_left + self.view_w - 1) / TILE_SIZE))
        y1 = int(math.floor((self.world_top + self.view_h - 1) / TILE_SIZE))
        
        # Fetch, stitch, and crop tiles to fit widget viewport
        tile_paths = fetch_tiles_sync(self.zoom, x0, y0, x1, y1, self.cache)
        stitched = stitch_tiles(x0, y0, x1, y1, tile_paths)
        self.cropped = render_world_viewport(
            stitched,
            self.world_left,
            self.world_top,
            self.view_w,
            self.view_h,
            x0,
            y0,
            WIDGET_W,
            WIDGET_H,
        )
        
        # Layout pins
        self.pins: list[tuple[float, float, bool]] = []  # x, y, is_current
        # Add current pin (centered)
        self.pins.append((WIDGET_W / 2, WIDGET_H / 2, True))

    def paint_pins(self, painter: "QPainter") -> None:
        """Draw all pins in viewport coordinates."""
        for x, y, is_current in self.pins:
            _draw_pin(painter, x, y, is_current=is_current)


def probe_map_tiles_online(timeout: float = 4.0) -> bool:
    """Quick check for OSM availability."""
    url = OSM_TILE_URL.format(z=0, x=0, y=0)
    req = urllib.request.Request(url, headers={"User-Agent": OSM_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= getattr(resp, "status", 200) < 400
    except Exception:
        return False


def default_map_cache_dir() -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches" / "RAWviewer" / "map_tiles"
    elif os.name == "nt":
        app_data = os.environ.get("LOCALAPPDATA") or str(Path.home())
        base = Path(app_data) / "RAWviewer" / "map_tiles"
    else:
        base = Path.home() / ".cache" / "rawviewer" / "map_tiles"
    base.mkdir(parents=True, exist_ok=True)
    return base
