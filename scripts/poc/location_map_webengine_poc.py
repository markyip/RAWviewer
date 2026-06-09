#!/usr/bin/env python3
"""
GPS mini-map POC — QWebEngineView + Leaflet + OSM (Google Maps–class slippy workflow).

Requires: pip install PyQt6-WebEngine

Usage:
  cd /path/to/RAWviewer
  source rawviewer_env/bin/activate
  export PYTHONPATH="$PWD/src:$PYTHONPATH"
  pip install PyQt6-WebEngine

  python scripts/poc/location_map_webengine_poc.py \\
      --folder "/Volumes/Development/Wales Aug 2025/Mark's phone" \\
      --current-file PXL_20250816_150629009.RAW-01.MP.COVER.jpg \\
      --gui

  python scripts/poc/location_map_webengine_poc.py --folder "..." \\
      --current-file PXL_....jpg \\
      --save docs/local/poc/map_webengine.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# Software rendering improves headless / grab() reliability on macOS.
os.environ.setdefault(
    "QTWEBENGINE_CHROMIUM_FLAGS",
    "--disable-gpu --disable-gpu-compositing --no-sandbox",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from location_map_poc import (  # noqa: E402
    FIT_BOUNDS_PADDING,
    FIT_BOUNDS_PIXEL_PAD,
    WIDGET_H,
    WIDGET_W,
    LocationMapWidget,
    MAX_MAP_ZOOM,
    MAX_ZOOM_IN_ABOVE_FIT,
    MIN_DETAIL_ZOOM,
    TILE_NATIVE_MAX_ZOOM,
)
from gps_neighbors import (  # noqa: E402
    DEFAULT_CLUSTER_RADIUS_M,
    GpsCluster,
    GpsMapView,
    build_map_view,
    format_cluster_gallery_title,
    resolve_current_point as resolve_current,
    scan_folder_gps,
)

try:
    from PyQt6.QtCore import QTimer, QUrl, QObject, pyqtSlot
    from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWebChannel import QWebChannel
except ImportError as exc:
    _venv_python = _REPO_ROOT / "rawviewer_env" / "bin" / "python"
    _hint = (
        f"Use the project venv (not conda base):\n"
        f"  source rawviewer_env/bin/activate\n"
        f"  export PYTHONPATH=\"$PWD/src:$PYTHONPATH\"\n"
        f"  python scripts/poc/location_map_webengine_poc.py ...\n"
        f"Or: scripts/Launch/shell/launch_map_webengine_poc.sh --gui ...\n"
    )
    if "libglib" in str(exc) or "Library not loaded" in str(exc):
        _hint += (
            "\nConda base PyQt6 is missing libglib for QtWebEngine. "
            f"rawviewer_env works ({_venv_python}).\n"
        )
    raise SystemExit(
        "PyQt6-WebEngine is required: pip install PyQt6-WebEngine\n"
        f"{_hint}"
        f"Original error: {exc}"
    ) from exc


VENDOR_LEAFLET = Path(__file__).resolve().parent / "vendor" / "leaflet"


def _cluster_payload(view: GpsMapView) -> dict:
    def _one(cluster: GpsCluster) -> dict:
        return {
            "id": cluster.cluster_id,
            "lat": cluster.centroid_lat,
            "lon": cluster.centroid_lon,
            "count": cluster.count,
            "title": format_cluster_gallery_title(
                cluster.centroid_lat, cluster.centroid_lon, cluster.count
            ),
            "members": [
                {
                    "name": Path(m.path).name,
                    "capture_time": m.capture_time,
                }
                for m in cluster.sorted_members()
            ],
        }

    return {
        "current_cluster": _one(view.current_cluster),
        "neighbor_clusters": [_one(c) for c in view.neighbor_clusters],
    }


def _write_map_html(
    pins: dict,
    *,
    interactive: bool,
) -> Path:
    pins_json = json.dumps(pins)
    qwebchannel_script = (
        '<script src="qrc:///qtwebchannel/qwebchannel.js"></script>' if interactive else ""
    )
    bridge_block = ""
    if interactive:
        bridge_block = """
    window.initWebChannel = function() {
      if (window.__rawviewerBridgeReady) return;
      new QWebChannel(qt.webChannelTransport, function(channel) {
        window.bridge = channel.objects.bridge;
        window.__rawviewerBridgeReady = true;
      });
    };
"""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <link rel="stylesheet" href="leaflet.css" />
  <script src="leaflet.js"></script>
  {qwebchannel_script}
  <style>
    html, body, #map {{ margin: 0; padding: 0; width: 100%; height: 100%; background: #1a1a1a; }}
  </style>
</head>
<body>
  <div id="map"></div>
  <script>
    function clusterIcon(color, badge) {{
      const bg = color === "green" ? "#28dc5a" : "#dc3c3c";
      const size = color === "green" ? 16 : 10;
      const label = badge || "";
      const fontPx = (color === "green" ? 9 : 8) - (label.length > 1 ? 1 : 0);
      return L.divIcon({{
        className: "",
        html: `<div style="width:${{size}}px;height:${{size}}px;border-radius:50%;background:${{bg}};border:2px solid #fff;box-shadow:0 0 4px rgba(0,0,0,.4);display:flex;align-items:center;justify-content:center;color:#fff;font:bold ${{fontPx}}px sans-serif;line-height:1;text-shadow:0 0 2px rgba(0,0,0,.6);">${{label}}</div>`,
        iconSize: [size, size],
        iconAnchor: [size / 2, size / 2],
      }});
    }}

    const MAP_MAX_ZOOM = {MAX_MAP_ZOOM};
    const MAX_ZOOM_IN_ABOVE_FIT = {MAX_ZOOM_IN_ABOVE_FIT};
    const MIN_DETAIL_ZOOM = {MIN_DETAIL_ZOOM};
    const TILE_NATIVE_MAX = {TILE_NATIVE_MAX_ZOOM};
    const FIT_BOUNDS_PAD = {FIT_BOUNDS_PADDING};
    const FIT_BOUNDS_PIXEL_PAD = {FIT_BOUNDS_PIXEL_PAD};

    const map = L.map("map", {{
      zoomControl: true,
      attributionControl: true,
      scrollWheelZoom: true,
      maxZoom: MAP_MAX_ZOOM,
    }});

    L.tileLayer("https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
      maxNativeZoom: TILE_NATIVE_MAX,
      maxZoom: MAP_MAX_ZOOM,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    }}).addTo(map);

    let markerLayers = [];
    let minZoomLimit = null;
    let maxZoomLimit = MAP_MAX_ZOOM;

    function rebuildMarkers(data) {{
      markerLayers.forEach((m) => map.removeLayer(m));
      markerLayers = [];

      const latlngs = [];
      const cur = data.current_cluster;
      const greenLatLng = L.latLng(cur.lat, cur.lon);
      const green = L.marker(greenLatLng, {{
        icon: clusterIcon("green", String(cur.count)),
        zIndexOffset: 1000,
      }}).addTo(map).bindPopup(cur.title);
      markerLayers.push(green);
      latlngs.push(greenLatLng);

      data.neighbor_clusters.forEach((cluster) => {{
        const ll = L.latLng(cluster.lat, cluster.lon);
        const red = L.marker(ll, {{
          icon: clusterIcon("red", String(cluster.count)),
        }}).addTo(map).bindPopup(cluster.title);
        red.on("click", () => {{
          if (window.bridge) window.bridge.selectCluster(cluster.id);
        }});
        markerLayers.push(red);
        latlngs.push(ll);
      }});

      return L.latLngBounds(latlngs).pad(FIT_BOUNDS_PAD);
    }}

    function fitAllPins(bounds, maxFitZoom, updateMinZoom) {{
      map.invalidateSize();
      map.setMaxBounds(null);
      window.__rawviewerLastBounds = bounds;
      if (updateMinZoom) {{
        map.setMinZoom(0);
      }}
      map.fitBounds(bounds, {{
        animate: false,
        padding: [FIT_BOUNDS_PIXEL_PAD, FIT_BOUNDS_PIXEL_PAD],
        maxZoom: maxFitZoom,
      }});
      if (updateMinZoom) {{
        minZoomLimit = map.getZoom();
        maxZoomLimit = Math.min(
          MAP_MAX_ZOOM,
          Math.max(minZoomLimit + MAX_ZOOM_IN_ABOVE_FIT, MIN_DETAIL_ZOOM)
        );
        map.setMinZoom(minZoomLimit);
        map.setMaxZoom(maxZoomLimit);
      }}
    }}

    function applyMapData(data, opts) {{
      opts = opts || {{}};
      const preserveZoom = opts.preserveZoom === true;
      const maxFitZoom = preserveZoom ? map.getZoom() : MAP_MAX_ZOOM;
      const bounds = rebuildMarkers(data);
      fitAllPins(bounds, maxFitZoom, !preserveZoom);
    }}

    map.on("zoomend", () => {{
      if (minZoomLimit !== null && map.getZoom() < minZoomLimit) {{
        map.setZoom(minZoomLimit);
      }}
      if (map.getZoom() > maxZoomLimit) {{
        map.setZoom(maxZoomLimit);
      }}
    }});

    window.rawviewerUpdateMap = applyMapData;
    const INITIAL_DATA = {pins_json};

    window.__rawviewerMapReady = false;
    window.__rawviewerMapError = null;
    window.bootMap = function bootMap() {{
      try {{
        map.invalidateSize();
        applyMapData(INITIAL_DATA, {{ preserveZoom: false }});
        if (typeof window.initWebChannel === "function") {{
          window.initWebChannel();
        }}
        window.__rawviewerMapReady = true;
        setTimeout(() => {{
          map.invalidateSize();
          if (window.__rawviewerLastBounds) {{
            fitAllPins(window.__rawviewerLastBounds, map.getZoom(), false);
          }}
        }}, 150);
      }} catch (err) {{
        window.__rawviewerMapError = String(err);
        console.error("bootMap failed:", err);
      }}
    }};

    map.whenReady(() => {{
      /* bootMap() is invoked from Qt after the widget has a real size */
    }});
{bridge_block}
  </script>
</body>
</html>
"""
    out = VENDOR_LEAFLET / "_poc_map.html"
    out.write_text(html, encoding="utf-8")
    return out


@dataclass
class MapSession:
    map_view: GpsMapView

    def cluster_by_id(self, cluster_id: int) -> Optional[GpsCluster]:
        if self.map_view.current_cluster.cluster_id == cluster_id:
            return self.map_view.current_cluster
        for cluster in self.map_view.neighbor_clusters:
            if cluster.cluster_id == cluster_id:
                return cluster
        return None


class MapBridge(QObject):
    """JS → Python: click red cluster → gallery filter or open single image."""

    def __init__(
        self,
        view: QWebEngineView,
        session: MapSession,
        on_cluster_clicked: Optional[Callable[[GpsCluster], None]] = None,
    ):
        super().__init__()
        self._view = view
        self._session = session
        self._on_cluster_clicked = on_cluster_clicked

    @pyqtSlot(int)
    def selectCluster(self, cluster_id: int) -> None:
        cluster = self._session.cluster_by_id(cluster_id)
        if cluster is None:
            return
        if cluster.cluster_id == self._session.map_view.current_cluster.cluster_id:
            return
        LocationMapWidget.handle_cluster_click(cluster)
        if self._on_cluster_clicked:
            self._on_cluster_clicked(cluster)


class MapCaptureView(QWebEngineView):
    def __init__(self, html_path: Path, parent=None):
        super().__init__(parent)
        self._html_path = html_path
        self._ready = False
        from PyQt6.QtWebEngineCore import QWebEngineSettings

        settings = self.page().settings()
        settings.setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True
        )
        settings.setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True
        )
        self.loadFinished.connect(self._on_load_finished)
        try:
            self.page().javaScriptConsoleMessage.connect(self._on_js_console)
        except Exception:
            pass

    @staticmethod
    def _on_js_console(level: int, message: str, line: int, source: str) -> None:
        del level, line
        if message:
            print(f"[webengine js] {source}: {message}", file=sys.stderr)

    def _on_load_finished(self, ok: bool) -> None:
        if not ok:
            print("[webengine] page load failed", file=sys.stderr)
            return
        QTimer.singleShot(50, self._boot_map)

    def _boot_map(self) -> None:
        self.page().runJavaScript(
            "window.bootMap && window.bootMap()",
            lambda _result: None,
        )

    def attach_bridge(self, bridge: MapBridge) -> None:
        channel = QWebChannel(self.page())
        channel.registerObject("bridge", bridge)
        self.page().setWebChannel(channel)

    def start(self) -> None:
        self.load(QUrl.fromLocalFile(str(self._html_path.resolve())))

    def wait_until_ready(self, timeout_ms: int = 15000) -> bool:
        from PyQt6.QtCore import QEventLoop

        loop = QEventLoop()
        self._ready = False

        def poll_ready() -> None:
            self.page().runJavaScript(
                "Boolean(window.__rawviewerMapReady)",
                lambda ok: _check(bool(ok)),
            )

        def _check(ready: bool) -> None:
            if ready and not self._ready:
                self._ready = True
                loop.quit()

        timer = QTimer()
        timer.timeout.connect(poll_ready)
        timer.start(250)
        poll_ready()

        killer = QTimer()
        killer.setSingleShot(True)
        killer.timeout.connect(loop.quit)
        killer.start(timeout_ms)

        loop.exec()
        timer.stop()
        return self._ready


def _load_session(folder: Path, current_file: str | None) -> MapSession:
    points = scan_folder_gps(folder)
    if not points:
        raise SystemExit("No GPS data found.")
    _, current = resolve_current(points, 0, current_file)
    view = build_map_view(current, points)
    if view is None:
        raise SystemExit("Could not build cluster view.")
    return MapSession(map_view=view)


def run_save(folder: Path, save_path: Path, current_file: str | None) -> int:
    from PyQt6.QtCore import QEventLoop

    session = _load_session(folder, current_file)
    html_path = _write_map_html(
        _cluster_payload(session.map_view),
        interactive=False,
    )

    app = QApplication([sys.argv[0]])
    view = MapCaptureView(html_path)
    view.setFixedSize(WIDGET_W, WIDGET_H)
    view.show()
    view.start()

    if not view.wait_until_ready():
        print("[webengine] warning: map ready timeout; capturing anyway", file=sys.stderr)

    settle = QTimer()
    settle.setSingleShot(True)
    loop = QEventLoop()
    settle.timeout.connect(loop.quit)
    settle.start(2500)
    loop.exec()

    QApplication.processEvents()
    screen = QApplication.primaryScreen()
    if screen is not None and view.winId():
        pixmap = screen.grabWindow(int(view.winId()))
    else:
        pixmap = view.grab()
    save_path = save_path.expanduser()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if pixmap.isNull() or not pixmap.save(str(save_path)):
        print(f"Failed to save {save_path}", file=sys.stderr)
        return 1

    cc = session.map_view.current_cluster
    print(f"Saved WebEngine map → {save_path}")
    print(f"  Green cluster: {cc.count} images ({cc.centroid_lat:.6f}, {cc.centroid_lon:.6f})")
    print(f"  Red clusters: {len(session.map_view.neighbor_clusters)} · Leaflet/OSM")
    return 0


def run_gui(folder: Path, current_file: str | None) -> int:
    session = _load_session(folder, current_file)
    html_path = _write_map_html(
        _cluster_payload(session.map_view),
        interactive=True,
    )

    app = QApplication([sys.argv[0]])
    app.setApplicationName("RAWviewer WebEngine Map POC")

    window = QMainWindow()
    mv = session.map_view

    def _status_text() -> str:
        cc = mv.current_cluster
        return (
            f"Green = current cluster ({cc.count} img) · "
            f"Red = {len(mv.neighbor_clusters)} clusters · "
            f"fitBounds · max z{MAX_MAP_ZOOM} · click red cluster"
        )

    status = QLabel(_status_text())

    view = MapCaptureView(html_path)
    view.setFixedSize(WIDGET_W, WIDGET_H)

    bridge = MapBridge(view, session)
    view.attach_bridge(bridge)

    window.setWindowTitle(
        f"RAWviewer WebEngine Map POC — {Path(mv.current_file.path).name}"
    )
    central = QWidget()
    layout = QVBoxLayout(central)
    layout.addWidget(status)
    layout.addWidget(view)
    window.setCentralWidget(central)
    window.resize(WIDGET_W + 40, WIDGET_H + 80)
    window.show()

    view.start()

    print(f"[gui] Leaflet clusters — {DEFAULT_CLUSTER_RADIUS_M:.0f}m radius · max z{MAX_MAP_ZOOM}.")
    return app.exec()


def main() -> int:
    parser = argparse.ArgumentParser(description="RAWviewer GPS map POC (QWebEngine + Leaflet)")
    parser.add_argument(
        "--folder",
        type=Path,
        default=Path("/Volumes/Development/Wales Aug 2025/Mark's phone"),
    )
    parser.add_argument("--current-file", type=str, default=None)
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save rendered frame to PNG and exit",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Show interactive Leaflet window (default if --save omitted)",
    )
    args = parser.parse_args()
    folder = args.folder.expanduser()
    if not folder.is_dir():
        print(f"Not a directory: {folder}", file=sys.stderr)
        return 1
    if args.save:
        return run_save(folder, args.save, args.current_file)
    return run_gui(folder, args.current_file)


if __name__ == "__main__":
    raise SystemExit(main())
