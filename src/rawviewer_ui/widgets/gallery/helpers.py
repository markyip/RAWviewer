"""Module-level helpers for the JustifiedGallery package (Phase 4 split)."""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap, QTransform
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from core import theme
from image_load_manager import Priority

logger = logging.getLogger(__name__)


class _ScrollDateBubble(QWidget):
    """Floating date + optional keyframe, shown while dragging the scrollbar.

    Dragging the thumb through a few thousand photos otherwise gives no idea
    where you have landed until you let go and the thumbnails decode. The
    capture date is already known from EXIF long before any pixels are, so it
    can answer immediately. A tiny in-memory thumbnail (cache-hit only) is
    shown beside the date when one is already decoded. A miss stays date-only
    for that path until the scrub target changes; a later cache fill must not
    flash the preview in.

    Only while DRAGGING THE THUMB -- not on wheel or trackpad scrolling, which
    move a screen at a time and where a bubble over the photos would be noise.

    Parented to the scroll area's viewport rather than the gallery widget: the
    gallery is the scrolled content and is taller than the window, so a child
    of it would scroll away with the photos instead of hovering over them.
    """

    _MARGIN_RIGHT = 14
    _PREVIEW_PX = 80

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setObjectName("scroll_date_bubble")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setStyleSheet(
            f"""
            QWidget#scroll_date_bubble {{
                background-color: {theme.rgba(theme.VOID_RGB, 235)};
                border: 1px solid {theme.LINE};
                border-radius: 6px;
            }}
            QWidget#scroll_date_bubble QLabel {{
                background: transparent;
                border: none;
                color: {theme.INK};
                font-size: 13px;
                font-weight: 600;
            }}
            QWidget#scroll_date_bubble QLabel#scroll_date_bubble_preview {{
                background-color: {theme.VOID};
                border: 1px solid {theme.LINE};
                border-radius: 3px;
            }}
            """
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 12, 6)
        layout.setSpacing(8)
        self._preview = QLabel(self)
        self._preview.setObjectName("scroll_date_bubble_preview")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._preview.hide()
        self._date = QLabel(self)
        self._date.setObjectName("scroll_date_bubble_date")
        self._date.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._date.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(self._preview)
        layout.addWidget(self._date)
        self._shown_path = None
        self.hide()

    def text(self) -> str:
        return self._date.text()

    def preview_is_visible(self) -> bool:
        pm = self._preview.pixmap()
        return self._preview.isVisible() and pm is not None and not pm.isNull()

    def show_for(
        self,
        text: str,
        *,
        center_y: int,
        pixmap: QPixmap | None = None,
        file_path: str = "",
    ) -> None:
        if not text:
            self._shown_path = None
            self._preview.clear()
            self._preview.hide()
            self.hide()
            return
        self._date.setText(text)
        # Path identity: never rescale / swap the keyframe until the scrub
        # target changes. Re-setting the pixmap on every thumb tick flashes.
        if file_path != self._shown_path:
            self._shown_path = file_path
            if pixmap is not None and not pixmap.isNull():
                side = self._PREVIEW_PX
                scaled = pixmap.scaled(
                    side,
                    side,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation,
                )
                self._preview.setPixmap(scaled)
                self._preview.setFixedSize(side, side)
                self._preview.show()
            else:
                self._preview.clear()
                self._preview.hide()
            self.adjustSize()
        parent = self.parentWidget()
        if parent is None:
            return
        x = max(0, parent.width() - self.width() - self._MARGIN_RIGHT)
        # Clamped so the bubble stays fully on screen at the very top and
        # bottom of the range, where the thumb's centre is within half a
        # bubble of the edge.
        y = max(0, min(parent.height() - self.height(), center_y - self.height() // 2))
        self.move(x, y)
        self.raise_()
        self.show()


def _focus_gallery_switch_logs() -> bool:
    return os.environ.get("RAWVIEWER_FOCUS_GALLERY_SWITCH", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default)).strip()))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.environ.get(name, str(default)).strip()))
    except (TypeError, ValueError):
        return default


def _env_true(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name, "1" if default else "0").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _gallery_prefetch_screens(fast: bool) -> int:
    """Viewport heights to prefetch above/below scroll center (embedded JPEG makes this cheap)."""
    if fast:
        return _env_int("RAWVIEWER_GALLERY_PREFETCH_SCREENS_FAST", 5, minimum=1)
    return _env_int("RAWVIEWER_GALLERY_PREFETCH_SCREENS", 6, minimum=1)


def _gallery_viewport_buffer_screens() -> float:
    """Extra viewport heights kept for visible tile widgets above/below the scroll window."""
    raw = os.environ.get("RAWVIEWER_GALLERY_VIEWPORT_BUFFER_SCREENS", "1.25").strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 1.25


def _gallery_scroll_buffer_screens() -> float:
    """Wider widget pool while scrolling to reduce tile churn when thumbnails are warm."""
    base = _gallery_viewport_buffer_screens()
    raw = os.environ.get("RAWVIEWER_GALLERY_SCROLL_BUFFER_SCREENS", "2.5").strip()
    try:
        return max(base, float(raw))
    except ValueError:
        return max(base, 2.5)


def _gallery_current_active_reserve(active_cap: int) -> int:
    """Slots reserved for visible (CURRENT) gallery tiles; prefetch cannot use them."""
    default = max(4, active_cap // 3)
    reserve = _env_int("RAWVIEWER_GALLERY_CURRENT_ACTIVE_RESERVE", default, minimum=2)
    return min(reserve, max(2, active_cap - 4))


def _gallery_scheduling_budgets(fast: bool) -> tuple[int, int, int]:
    """Return (max_widgets, max_tasks, active_cap) for the current scroll mode."""
    if fast:
        return (
            _env_int("RAWVIEWER_GALLERY_MAX_WIDGETS_FAST", 16, minimum=1),
            _env_int("RAWVIEWER_GALLERY_MAX_TASKS_FAST", 24, minimum=1),
            _env_int("RAWVIEWER_GALLERY_ACTIVE_CAP_FAST", 32, minimum=4),
        )
    return (
        _env_int("RAWVIEWER_GALLERY_MAX_WIDGETS", 48, minimum=1),
        _env_int("RAWVIEWER_GALLERY_MAX_TASKS", 64, minimum=1),
        _env_int("RAWVIEWER_GALLERY_ACTIVE_CAP", 64, minimum=4),
    )


def _apply_external_gallery_caps(
    max_widgets: int,
    max_tasks: int,
    active_cap: int,
    folder_path: str | None,
) -> tuple[int, int, int]:
    """Lower gallery scheduling on external/USB/network volumes (post-warmup too).

    Applies when the volume is confirmed-slow (all platforms) or, on Windows,
    for any external volume (moderate_external_cap_enabled). macOS fast external
    drives keep the full gallery budget — they never showed the native crash.
    """
    if not folder_path:
        return max_widgets, max_tasks, active_cap
    try:
        from common_image_loader import (
            is_external_or_network_volume,
            moderate_external_cap_enabled,
            volume_speed_tier,
        )

        if not is_external_or_network_volume(folder_path):
            return max_widgets, max_tasks, active_cap
        is_slow = volume_speed_tier(folder_path) == "slow"
        if is_slow:
            cap = _env_int("RAWVIEWER_SLOW_GALLERY_ACTIVE_CAP", 32, minimum=2)
            tasks = _env_int("RAWVIEWER_SLOW_GALLERY_MAX_TASKS", 24, minimum=2)
            widgets = _env_int("RAWVIEWER_SLOW_GALLERY_MAX_WIDGETS", 40, minimum=4)
            return (
                min(max_widgets, widgets),
                min(max_tasks, tasks),
                min(active_cap, cap),
            )
        if not moderate_external_cap_enabled() and sys.platform != "win32":
            # Stability without throttling: cap fast external on Windows only (GL crash
            # was mode-switch teardown, not this concurrency). macOS keeps full budget.
            return max_widgets, max_tasks, active_cap
    except Exception:
        return max_widgets, max_tasks, active_cap
    # Fast external (e.g. USB3/TB probing >120 MB/s).
    cap = _env_int("RAWVIEWER_EXTERNAL_GALLERY_ACTIVE_CAP", 32, minimum=2)
    tasks = _env_int("RAWVIEWER_EXTERNAL_GALLERY_MAX_TASKS", 24, minimum=2)
    widgets = _env_int("RAWVIEWER_EXTERNAL_GALLERY_MAX_WIDGETS", 40, minimum=4)
    return (
        min(max_widgets, widgets),
        min(max_tasks, tasks),
        min(active_cap, cap),
    )


def _gallery_warmup_scheduling_budgets(file_count: int) -> tuple[int, int, int]:
    """Brief soft cap during gallery entry (GL teardown is the real crash fix)."""
    if file_count >= 2500:
        return (12, 8, 16)
    if file_count >= 1200:
        return (20, 14, 28)
    if file_count >= 500:
        return (24, 18, 32)
    return (28, 20, 36)


def _batch_tile_apply_enabled() -> bool:
    """Default ON. Set RAWVIEWER_GALLERY_BATCH_TILE_APPLY=0 to apply tile pixmaps inline."""
    return os.environ.get(
        "RAWVIEWER_GALLERY_BATCH_TILE_APPLY", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}


def _tile_apply_batch_size() -> int:
    """Max tile pixmaps applied to the UI per event-loop tick when batching is on."""
    return _env_int("RAWVIEWER_GALLERY_TILE_APPLY_BATCH", 8, minimum=1)


def _orient_debug_enabled() -> bool:
    """Diagnostic only: RAWVIEWER_ORIENT_DEBUG=1 logs per-tile orientation decisions."""
    return os.environ.get("RAWVIEWER_ORIENT_DEBUG", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _gallery_idle_preload_batch() -> int:
    return _env_int("RAWVIEWER_GALLERY_IDLE_PRELOAD_BATCH", 72, minimum=4)


def _gallery_idle_preload_ms() -> int:
    return _env_int("RAWVIEWER_GALLERY_IDLE_PRELOAD_MS", 250, minimum=50)


def _min_layout_pixmap_dim() -> int:
    """Nav/filmstrip previews below this size must not drive justified layout aspect."""
    return _env_int("RAWVIEWER_GALLERY_MIN_LAYOUT_PIXMAP_DIM", 384, minimum=128)


def _gallery_idle_load_priority() -> Priority:
    """Queue priority for off-screen gallery thumbnail idle preload.

    Delegates to rawviewer_profile so Plus only demotes while the ILM indexing
    throttle is actually engaged (not merely because AI features are enabled).
    """
    try:
        from core.profile import gallery_idle_load_priority

        return gallery_idle_load_priority()
    except Exception:
        raw = os.environ.get("RAWVIEWER_GALLERY_IDLE_PRELOAD_PRIORITY", "").strip().lower()
        if raw in ("background", "low"):
            return Priority.BACKGROUND
        return Priority.PRELOAD_NEXT


def _thumbnail_data_to_base_pixmap(thumbnail_data) -> QPixmap | None:
    """Convert manager/cache thumbnail payload to a gallery base QPixmap."""
    if thumbnail_data is None:
        return None
    if isinstance(thumbnail_data, QPixmap):
        return thumbnail_data if not thumbnail_data.isNull() else None
    if isinstance(thumbnail_data, np.ndarray):
        arr = np.ascontiguousarray(thumbnail_data)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        if arr.ndim == 3:
            h, w, c = arr.shape[:3]
            if h <= 0 or w <= 0:
                return None
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
                arr = np.ascontiguousarray(arr)
            if c == 3:
                bytes_per_line = arr.strides[0]
                qimg = QImage(
                    arr.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
                ).copy()
            elif c == 4:
                bytes_per_line = arr.strides[0]
                qimg = QImage(
                    arr.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888
                ).copy()
            else:
                return None
        else:
            return None
        if qimg.isNull():
            return None
        return QPixmap.fromImage(qimg)
    if isinstance(thumbnail_data, QImage):
        if thumbnail_data.isNull():
            return None
        rgb = thumbnail_data.convertToFormat(QImage.Format.Format_RGB888)
        return QPixmap.fromImage(rgb) if not rgb.isNull() else None
    return None


def _rotate_pixmap_cw(pixmap: QPixmap, degrees: int) -> QPixmap:
    """Rotate pixmap clockwise for on-screen display (matches single-image view)."""
    degrees = int(degrees) % 360
    if pixmap is None or pixmap.isNull() or degrees == 0:
        return pixmap
    transform = QTransform()
    transform.rotate(degrees)
    return pixmap.transformed(transform, Qt.TransformationMode.SmoothTransformation)
