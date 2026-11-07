"""GPU-accelerated single-image view (Route B).

A self-contained ``QGraphicsView`` that renders one image and performs fit / zoom /
pan as GPU matrix transforms instead of re-scaling the pixmap on the CPU every frame.
This gives QuickLook-style smoothness for wheel/pinch zoom and drag pan.

The optional ``QOpenGLWidget`` viewport is vendor-agnostic (NVIDIA / AMD / Intel on
Windows; Qt typically uses a Metal-backed GL stack on macOS).

Enabled by default in release builds; set ``RAWVIEWER_GPU_VIEW=0`` for the legacy
``QScrollArea`` + ``QLabel`` path. The widget is intentionally
decoupled from the main window: it exposes a small API plus a few signals so the
host can keep navigation / status / histogram behaviour identical.

Environment toggles:
- ``RAWVIEWER_GPU_VIEW=0``       Disable; use legacy scroll-area single-image view.
- ``RAWVIEWER_GPU_VIEW_NO_GL=1`` Use the raster viewport (debug / fallback).
- ``RAWVIEWER_GPU_GL_TEX=1``     Paint RGB via ``QImage`` on the OpenGL viewport
  (skip ``QPixmap``). Opt-in Phase 2b step toward CUDA↔GL.
- ``RAWVIEWER_GPU_CUDA_GL=1``    Keep decode on CUDA and upload via
  ``cudaGraphicsGLRegisterImage`` (Phase 2c; falls back to GL_TEX / pixmap).
"""

from typing import Any

from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
)

from rawviewer_ui.overlays.composition_grid import CompositionGridGraphicsItem
from rawviewer_ui.widgets.gpu.gpu_item import CompareHandleItem, RgbGlImageItem
from rawviewer_ui.widgets.gpu.gpu_view_part1 import GpuImageViewPart1
from rawviewer_ui.widgets.gpu.gpu_view_part2 import GpuImageViewPart2


class GpuImageView(
    GpuImageViewPart1,
    GpuImageViewPart2,
    QGraphicsView,
):
    """Hardware-accelerated single image viewport.

    Coordinate model: scene units == image pixels. ``transform().m11()`` is therefore
    "view pixels per image pixel"; a scale of 1.0 means 100% (actual pixels).
    """

    # bool: True when showing fit-to-window, False when zoomed
    fitModeChanged = pyqtSignal(bool)
    # float: current scale (view px per image px); 1.0 == 100%
    zoomChanged = pyqtSignal(float)
    # int: +1 request next image, -1 request previous image (plain wheel in fit mode)
    wheelNavigate = pyqtSignal(int)
    # Image pixel coordinates (scene space) where the user double-clicked.
    doubleClickedAt = pyqtSignal(QPointF)
    # Image pixel coordinates (scene space) where the user clicked while color-pick
    # mode was armed (see set_color_pick_mode). One-shot: mode disarms after firing.
    colorPickRequested = pyqtSignal(QPointF)
    # Dodge & burn brush: (scene_x, scene_y, pressure 0..1, is_stroke_end).
    # Emitted continuously while painting (set_dodge_burn_mode(True)) so the
    # host can stamp+repaint per point; is_stroke_end=True on release, the
    # host's cue to edge-snap the touched region and trigger the exact
    # (worker-thread) re-render.
    dodgeBurnStroke = pyqtSignal(QPointF, float, bool)
    # Mouse left the photo (or the view) while a brush tool was armed — host
    # should disarm Dodge/Burn/Eraser/Heal so the tool does not stay sticky.
    brushToolLeftImage = pyqtSignal()
    # Vertical wheel/trackpad delta while D&B is armed — host nudges Brush Size
    # instead of navigating images.
    dodgeBurnBrushSizeWheel = pyqtSignal(int)
    # Crop overlay: insets changed / drag finished (CropLeft/Right/Top/Bottom).
    cropInsetsChanged = pyqtSignal(float, float, float, float)
    cropEditingFinished = pyqtSignal()

    # Absolute floor for set_scale; wheel zoom cannot go below fit_scale() (fit-to-window).
    MIN_SCALE = 0.01
    # Pinch / wheel cap (400%). Space and double-click use zoom_to_actual() at 100%.
    MAX_SCALE = 4.0
    _FIT_SCALE_EPS = 1.002  # treat within ~0.2% of fit as fit-to-window
    _shortcut_handler: Any

    def __init__(self, parent=None, background="#14120F"):
        # Attributes read from event()/viewportEvent() must exist before super().__init__()
        # because Qt may deliver events during construction.
        self._fit_mode = True
        self._wheel_navigate_in_fit = True
        self._zoom_intent_100 = False
        # (fx, fy) image-fraction under the pointer and (dx, dy) its viewport offset
        # from the viewport centre, recorded by the last pointer-anchored zoom.
        self._zoom_anchor = None
        self._zoom_locked = False
        self._has_pixmap = False
        self._img_w = 0
        self._img_h = 0
        self._overlay_item = None
        self._clipping_item = None
        self._grid_mode = "off"
        self._shortcut_handler = None
        self._edr_initialized = False
        self.file_path = None
        self._dodge_burn_mode = False
        # True once the pointer has landed on the image at least once since
        # the tool was armed. Guards _maybe_emit_brush_tool_left_image so
        # arming (D/B/X, or clicking the tool button) while the cursor is
        # still over the Adjust panel / letterbox margin can't immediately
        # disarm itself on the very next mouse-move -- only a genuine
        # "was on the image, now isn't" transition should disarm.
        self._dodge_burn_confirmed_on_image = False
        self._crop_mode = False
        self._export_drag_enabled = True
        self._drag_start_pos = None
        self._drag_started = False
        self._color_pick_mode = False
        self._compare_active = False
        self._compare_dragging_divider = False

        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._item = QGraphicsPixmapItem()
        self._item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.addItem(self._item)

        # Sibling of the image (not a child of _item) so Show Mask stays
        # visible on both the QPixmap and RGB-GL display paths.
        self._mask_item = QGraphicsPixmapItem()
        self._mask_item.setOpacity(0.45)
        self._mask_item.setZValue(5)
        self._mask_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._mask_item.hide()
        self._mask_overlay_wanted = False
        self._mask_overlay_mask = None
        self._scene.addItem(self._mask_item)

        # Dodge/burn brush preview: soft radial falloff (matches stamp kernel).
        self._brush_cursor_item = QGraphicsPixmapItem()
        self._brush_cursor_item.setZValue(25)
        self._brush_cursor_item.setTransformationMode(
            Qt.TransformationMode.SmoothTransformation
        )
        self._brush_cursor_item.hide()
        self._dodge_burn_brush_radius = 40.0
        self._dodge_burn_brush_flow = 0.55
        self._brush_cursor_pixmap_r = -1.0
        self._brush_cursor_pixmap_flow = -1.0
        self._scene.addItem(self._brush_cursor_item)

        # Opt-in RGB→OpenGL paint path (RAWVIEWER_GPU_GL_TEX=1): skips QPixmap.
        self._rgb_item = RgbGlImageItem()
        self._rgb_item.setZValue(0)
        self._rgb_item.hide()
        self._scene.addItem(self._rgb_item)
        self._use_rgb_gl = False

        # Dashed focus / subject overlay rectangle (scene coords == image pixels). Uses a
        # cosmetic pen so the on-screen line width stays constant at any zoom level.
        self._grid_item = CompositionGridGraphicsItem()
        self._scene.addItem(self._grid_item)

        self._clipping_item = QGraphicsPixmapItem()
        self._clipping_item.setZValue(15)
        self._clipping_item.setTransformationMode(Qt.TransformationMode.FastTransformation)
        self._clipping_item.hide()
        self._scene.addItem(self._clipping_item)

        # Compare-with-original split view (Adjust panel only). The original
        # (unedited) render sits in an overlay item above the always-current
        # edited pixmap in self._item, cropped to only the region left of a
        # draggable divider -- so the main item/zoom/pan/fit machinery needs
        # no changes at all; this is purely an additional overlay.
        self._compare_split_frac = 0.5
        self._compare_original_pixmap = QPixmap()
        self._compare_overlay_item = QGraphicsPixmapItem()
        self._compare_overlay_item.setZValue(12)
        self._compare_overlay_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._compare_overlay_item.hide()
        self._scene.addItem(self._compare_overlay_item)

        from rawviewer_ui.overlays.crop_overlay import CropOverlayItem

        self._crop_item = CropOverlayItem()
        self._crop_item.hide()
        self._crop_item.insetsChanged.connect(self.cropInsetsChanged.emit)
        self._crop_item.editingFinished.connect(self.cropEditingFinished.emit)
        self._scene.addItem(self._crop_item)

        self._compare_divider_line = QGraphicsLineItem()
        self._compare_divider_line.setZValue(13)
        pen = QPen(QColor(255, 255, 255, 220))
        pen.setCosmetic(True)
        pen.setWidth(2)
        self._compare_divider_line.setPen(pen)
        self._compare_divider_line.hide()
        self._scene.addItem(self._compare_divider_line)

        self._compare_divider_handle = CompareHandleItem()
        self._compare_divider_handle.setZValue(14)
        self._compare_divider_handle.setPen(QPen(QColor(255, 255, 255, 220), 1.5))
        self._compare_divider_handle.setBrush(QColor(0, 0, 0, 140))
        self._compare_divider_handle.hide()
        self._scene.addItem(self._compare_divider_handle)

        self.setRenderHints(
            QPainter.RenderHint.SmoothPixmapTransform | QPainter.RenderHint.Antialiasing
        )
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setBackgroundBrush(QColor(background))
        self.setStyleSheet(f"border: none; background-color: {background};")
        self.setOptimizationFlag(
            QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True
        )
        self.viewport().setMouseTracking(True)
        self.viewport().setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
        self.viewport().setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._maybe_enable_opengl()
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(500, self._enable_macos_edr)

        # Centered placeholder text shown when no image is loaded (parity with the
        # legacy QLabel instruction screen, which the GPU view would otherwise cover).
        self._placeholder = QLabel(self.viewport())
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._placeholder.setStyleSheet(
            "QLabel { color: #666; font-size: 14px; background-color: transparent; }"
        )
        self._placeholder.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._placeholder.hide()
