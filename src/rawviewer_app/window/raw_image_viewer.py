"""Main application window, assembled from semantic mixins."""
from __future__ import annotations

from typing import Any

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QLabel, QMainWindow, QScrollArea, QStatusBar

from rawviewer_app.viewer.session_mixin import SessionMixin

from .mixins import (
    AdjustPart1,
    AdjustPart2,
    AdjustPart3,
    AdjustPart4,
    AdjustPart5,
    BatchPart1,
    DialogsPart1,
    DisplayPart1,
    DisplayPart2,
    DisplayPart3,
    DisplayPart4,
    DisplayPart5,
    DisplayPart6,
    ExternalPart1,
    ExternalPart2,
    GalleryPart1,
    GalleryPart2,
    GalleryPart3,
    GalleryPart4,
    GalleryPart5,
    ImageLoadPart1,
    ImageLoadPart2,
    ImageLoadPart3,
    LifecyclePart1,
    LifecyclePart2,
    MetadataPart1,
    MetadataPart2,
    MiscPart1,
    MiscPart2,
    MiscPart3,
    MiscPart4,
    NavigationPart1,
    NavigationPart2,
    SearchPart1,
    SearchPart2,
    SearchPart3,
    SearchPart4,
    SettingsPart1,
    ShortcutsPart1,
    StatusPart1,
    StatusPart2,
    UiBuildPart1,
    UiBuildPart2,
    UiBuildPart3,
)


class RAWImageViewer(
    LifecyclePart1,
    LifecyclePart2,
    SettingsPart1,
    StatusPart1,
    StatusPart2,
    UiBuildPart1,
    UiBuildPart2,
    UiBuildPart3,
    DisplayPart1,
    DisplayPart2,
    DisplayPart3,
    DisplayPart4,
    DisplayPart5,
    DisplayPart6,
    ImageLoadPart1,
    ImageLoadPart2,
    ImageLoadPart3,
    GalleryPart1,
    GalleryPart2,
    GalleryPart3,
    GalleryPart4,
    GalleryPart5,
    AdjustPart1,
    AdjustPart2,
    AdjustPart3,
    AdjustPart4,
    AdjustPart5,
    MetadataPart1,
    MetadataPart2,
    SearchPart1,
    SearchPart2,
    SearchPart3,
    SearchPart4,
    NavigationPart1,
    NavigationPart2,
    ShortcutsPart1,
    ExternalPart1,
    ExternalPart2,
    BatchPart1,
    DialogsPart1,
    MiscPart1,
    MiscPart2,
    MiscPart3,
    MiscPart4,
    SessionMixin,
    QMainWindow,
):
    """Application main window composed from feature mixins."""

    _sensor_dims_resolved = pyqtSignal(str)

    status_bar: QStatusBar

    image_label: QLabel

    scroll_area: QScrollArea

    image_manager: Any

    image_cache: Any

    preload_manager: Any

    gpu_view: Any

    gallery_justified: Any

    gallery_view: Any

    gallery_content_layout: Any

    current_file_path: str | None

    thread_pool: Any

    _macos_active_service_delegate: Any

    _GALLERY_UI_SORT_MAX_FILES = 400

    _GALLERY_LAYOUT_META_SEED_MAX = 400

    _GALLERY_BUILD_EXIF_PREFETCH_MAX = 128

    _PRELOAD_NEXT_COUNT = 6

    _PRELOAD_PREV_COUNT = 4

    _MAX_PINCH_ZOOM_LEVEL = 4.0  # 400% -- pinch / Ctrl+wheel only; Space / double-click stay at 100%

    _INSTANT_GAIN_KEYS = frozenset({"Exposure2012", "Temperature", "Tint"})

    _INSTANT_TRANSFORM_KEYS = frozenset(
        {
            "CropAngle",
            "PerspectiveVertical",
            "PerspectiveHorizontal",
            "CropLeft",
            "CropRight",
            "CropTop",
            "CropBottom",
        }
    )

    _FRAME_RESIZE_BORDER = 10

    _GALLERY_SCROLLBAR_HANDLE_MARGIN = 6

    _GALLERY_FRAME_RIGHT_GUTTER = _GALLERY_SCROLLBAR_HANDLE_MARGIN
