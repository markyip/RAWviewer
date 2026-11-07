"""ui_build mixin, part 2 of 3 (11 methods, split init_ui into helpers)."""
from __future__ import annotations

import contextlib
import logging
import platform
import sys
from typing import Any, cast

from PyQt6.QtCore import QSize, Qt, QTimer
from PyQt6.QtGui import QCursor, QKeySequence, QRegion, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedLayout,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from core import theme
from core.config import _env_true, safe_print
from rawviewer_app.core.widgets import (
    CustomTitleBar,
    LoadingOverlay,
    ResizeGripIndicator,
    SingleImageViewOverlay,
    TopMetadataBar,
)

from .. import lazy
from ..adjust_workers import (
    _AdjustCompareOriginalSignals,
    _AdjustEditBaseSignals,
    _AdjustExportSignals,
    _AdjustMaskLoadSignals,
    _AdjustPreviewSignals,
)
from ..lazy import ImageAdjustPanelWidget, ImageHistogramWidget, ImageLocationMapWidget
from ..macos import _macos_install_line_edit_completion_guard
from ..platform_win import _apply_windows_native_window_icon
from ..qt_helpers import _qta_icon
from ..splash import _load_branded_qicon

lazy.register(sys.modules[__name__])


class UiBuildPart2:
    def _init_window_chrome(self):
        if platform.system() == 'Windows':
            self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle('RAWviewer')

        # Set simple background style (no rounded corners - simplifies window resizing)
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {theme.VOID};
            }}
            QToolTip {{
                color: {theme.INK};
                background-color: {theme.SURFACE};
                border: 1px solid {theme.LINE};
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 12px;
            }}
        """)

        # Set icon based on platform and available files
        _win_icon = _load_branded_qicon()
        if _win_icon is not None:
            self.setWindowIcon(_win_icon)
            _app = QApplication.instance()
            if _app is not None:
                _app.setWindowIcon(_win_icon)
        if platform.system() == "Windows":
            # Frameless windows often need a native WM_SETICON for the taskbar.
            _apply_windows_native_window_icon(self)

        # Calculate minimum width for 5 images at 4:3 aspect ratio
        # Each image: height=200px, width=200*(4/3)=267px
        # 5 images: 5*267 = 1335px
        # Spacing: 4 gaps * 4px = 16px
        # Margins: 8px * 2 = 16px
        # Total: 1335 + 16 + 16 = 1367px, round up to 1400px for comfortable display
        # Restore window geometry from session or center on screen
        settings = self.get_settings()
        if settings.contains("window_geometry"):
            self.restoreGeometry(settings.value("window_geometry"))
            if settings.contains("window_state"):
                self.restoreState(settings.value("window_state"))
        else:
            # Default size and center
            width, height = 1400, 800
            screen = QApplication.primaryScreen().availableGeometry()
            x = (screen.width() - width) // 2
            y = (screen.height() - height) // 2
            self.setGeometry(x, y, width, height)

        self.setMinimumSize(800, 600)
        self.setAcceptDrops(True)

        # Initialize resize tracking for frameless window edge resizing
        self._resize_edge_active = None
        self._resize_start_pos = None
        self._resize_start_geometry = None

        # Enable native window resizing for Windows (allows edge dragging)
        if platform.system() == 'Windows':
            self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
            # Ensure no mask is set (allows mouse events at edges)
            self.setMask(QRegion())
            # Don't use translucent background (simplifies event handling)
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

        # Create custom title bar only on Windows
        if platform.system() == 'Windows':
            self.title_bar = CustomTitleBar(self)
        else:
            self.title_bar = None

        # Initialize loading overlay for single view
        self.loading_overlay = LoadingOverlay(self)
        self.loading_overlay.hide()
        # Separate instance for "applying a saved edit". It deliberately does
        # NOT share self.loading_overlay: that one is hidden from ~30 call
        # sites across the image-load pipeline (display_pixmap,
        # on_manager_*_ready, _begin_adjust_editing_session, ...), several of
        # which fire while entering the editor -- sharing it meant this
        # overlay was torn down within milliseconds of being shown, long
        # before the edit render it announces had finished.
        self.edit_loading_overlay = LoadingOverlay(self)
        self.edit_loading_overlay.hide()


    def _init_content(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)
        main_layout = self.main_layout
        # Reduce padding for image view
        main_layout.setContentsMargins(0, 0, 0, 0)  # Set padding to 0
        main_layout.setSpacing(0)  # Remove spacing

        # Add title bar first if it exists
        if hasattr(self, 'title_bar') and self.title_bar is not None:
            main_layout.addWidget(self.title_bar)
            self.title_bar.setMouseTracking(True)
            self.title_bar.installEventFilter(self)
            for _title_ctrl in (
                self.title_bar.min_btn,
                self.title_bar.max_btn,
                self.title_bar.close_btn,
            ):
                _title_ctrl.setMouseTracking(True)
                _title_ctrl.installEventFilter(self)
            self.title_bar._sync_maximize_state()
            self.status_metadata_label = self.title_bar.metadata_label
        else:
            self.top_metadata_bar = TopMetadataBar(self)
            main_layout.addWidget(self.top_metadata_bar)
            self.status_metadata_label = self.top_metadata_bar.metadata_label
        # Native menu bar (File, View, keyboard shortcuts)
        self.create_menu_bar()
        # Windows uses a frameless window: the system menu bar is often invisible; embed it in-window.
        if platform.system() == "Windows":
            with contextlib.suppress(Exception):
                self.menuBar().setNativeMenuBar(False)
            with contextlib.suppress(Exception):
                self.menuBar().setVisible(False)
        self.scroll_area = QScrollArea()
        # Key: allow scrolling when image is larger
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Disable scrollbars completely - user can pan by dragging
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Apply Material Design 3 scrollbar styling
        self.scroll_area.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: {theme.VOID};
            }}
            QScrollArea > QWidget > QWidget {{
                border: none;
                background-color: {theme.VOID};
            }}
            /* Material Design 3 Scrollbar Styling */
            QScrollBar:vertical {{
                background: transparent;
                width: 12px;
                margin: 0px;
                border: none;
            }}
            QScrollBar::handle:vertical {{
                background: rgba(255, 255, 255, 0.2);
                min-height: 30px;
                border-radius: 6px;
                margin: 2px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: rgba(255, 255, 255, 0.3);
            }}
            QScrollBar::handle:vertical:pressed {{
                background: rgba(255, 255, 255, 0.4);
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{
                height: 0px;
                width: 0px;
            }}
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {{
                background: transparent;
            }}
            QScrollBar:horizontal {{
                background: transparent;
                height: 12px;
                margin: 0px;
                border: none;
            }}
            QScrollBar::handle:horizontal {{
                background: rgba(255, 255, 255, 0.2);
                min-width: 30px;
                border-radius: 6px;
                margin: 2px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background: rgba(255, 255, 255, 0.3);
            }}
            QScrollBar::handle:horizontal:pressed {{
                background: rgba(255, 255, 255, 0.4);
            }}
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {{
                height: 0px;
                width: 0px;
            }}
            QScrollBar::add-page:horizontal,
            QScrollBar::sub-page:horizontal {{
                background: transparent;
            }}
        """)
        self.image_label = QLabel()
        # This label displays both the loaded photo and (when nothing is
        # loaded) empty-state/error help text -- always centered, so a
        # photo narrower than the viewport doesn't end up pinned to the
        # left edge with the empty space pushed entirely to the right.
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Create instruction text with modern folder icon
        # Use modern folder icon (📁) instead of old style (🗁)
        self.image_label.setText(self._no_image_loaded_help_text())
        self.image_label.setStyleSheet(
            "QLabel { color: #666; font-size: 14px; background-color: transparent; }")
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setMouseTracking(True)
        cast(Any, self.image_label).mousePressEvent = self.image_mouse_press_event
        cast(Any, self.image_label).mouseMoveEvent = self.image_mouse_move_event
        cast(Any, self.image_label).mouseReleaseEvent = self.image_mouse_release_event
        cast(Any, self.image_label).mouseDoubleClickEvent = self.image_double_click_event
        self.scroll_area.setWidget(self.image_label)


    def _init_harness_and_overlays(self):
        self.scroll_area.viewport().setMouseTracking(True)
        self.scroll_area.viewport().installEventFilter(self)

        self.single_image_histogram = ImageHistogramWidget()
        self.single_image_histogram.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._histogram_overlay_visible = False
        self.single_image_histogram.hide()

        self.single_image_location_map = ImageLocationMapWidget()
        self.single_image_location_map.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.single_image_location_map.hide()

        from raw_adjustments import editing_features_enabled

        self.single_image_adjust_panel = None
        self.single_image_adjust_histogram = None
        self._adjust_overlay_visible = False
        self._adjust_user_hidden = True
        self._pending_adjust_preview = None
        self._adjust_preview_stage_cache = None
        self._adjust_preview_stage_cache_fast = None
        if editing_features_enabled() and ImageAdjustPanelWidget is not None:
            self.single_image_adjust_histogram = ImageHistogramWidget()
            self.single_image_adjust_panel = ImageAdjustPanelWidget(
                histogram_widget=self.single_image_adjust_histogram
            )
            self.single_image_adjust_panel.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.single_image_adjust_panel.hide()
            self.single_image_adjust_panel.editing_finished.connect(
                self._on_adjust_panel_editing_finished
            )
            self.single_image_adjust_panel.preview_changed.connect(
                self._on_adjust_panel_preview_changed
            )
            self.single_image_adjust_panel.transform_interaction.connect(
                self._on_transform_interaction
            )
            self.single_image_adjust_panel.auto_wb_requested.connect(
                self._on_auto_wb_requested
            )
            self.single_image_adjust_panel.auto_straighten_requested.connect(
                self._on_auto_straighten_requested
            )
            if hasattr(self.single_image_adjust_panel, "auto_tone_requested"):
                self.single_image_adjust_panel.auto_tone_requested.connect(
                    self._on_auto_tone_requested
                )
            if hasattr(self.single_image_adjust_panel, "dock_side_requested"):
                self.single_image_adjust_panel.dock_side_requested.connect(
                    self._on_adjust_panel_dock_requested
                )
            self.single_image_adjust_panel.export_requested.connect(
                self._on_adjust_panel_export_requested
            )
            self.single_image_adjust_panel.recovery_baseline_requested.connect(
                self._on_adjust_recovery_baseline_requested
            )
            self.single_image_adjust_panel.wb_picker_toggled.connect(
                self._on_adjust_wb_picker_toggled
            )
            self.single_image_adjust_panel.compare_toggled.connect(
                self._on_adjust_compare_toggled
            )
            self.single_image_adjust_panel.lens_correction_toggled.connect(
                self._on_adjust_lens_correction_toggled
            )
            self.single_image_adjust_panel.dodge_burn_mode_changed.connect(
                self._on_dodge_burn_mode_changed
            )
            self.single_image_adjust_panel.dodge_burn_clear_requested.connect(
                self._on_dodge_burn_clear_requested
            )
            self.single_image_adjust_panel.spot_heal_clear_requested.connect(
                self._on_spot_heal_clear_requested
            )
            self.single_image_adjust_panel.dodgeBurnMaskToggled.connect(
                self._on_dodge_burn_mask_toggled
            )
            self.single_image_adjust_panel.dodge_burn_brush_changed.connect(
                self._sync_dodge_burn_brush_cursor
            )
            self.single_image_adjust_panel.xmp_preset_applied.connect(
                self._on_xmp_preset_applied
            )
            self.single_image_adjust_panel.looks_drop_rejected.connect(
                self._on_looks_drop_rejected
            )
            self.single_image_adjust_panel.ai_denoise_toggled.connect(
                self._on_adjust_ai_denoise_toggled
            )
            self.single_image_adjust_panel.ai_preview_requested.connect(
                self._on_adjust_ai_preview_requested
            )
            self.single_image_adjust_panel.set_ai_denoise_enabled(
                bool(self.get_settings().value("ai_denoise_enabled", False) in (True, "1", "true"))
            )
            self.single_image_adjust_panel.crop_mode_changed.connect(
                self._on_crop_mode_changed
            )
            self.single_image_adjust_panel.crop_aspect_changed.connect(
                self._on_crop_aspect_changed
            )
            self.single_image_adjust_panel.crop_apply_requested.connect(
                self._on_crop_apply_requested
            )
            self.single_image_adjust_panel.crop_cancel_requested.connect(
                self._on_crop_cancel_requested
            )
            self.single_image_adjust_panel.crop_reset_requested.connect(
                self._on_crop_reset_requested
            )
            self.single_image_adjust_panel.reset_requested.connect(
                self._on_adjust_panel_reset
            )
            self.single_image_adjust_panel.calibrate_requested.connect(
                self._on_calibrate_camera_requested
            )
            self.single_image_adjust_panel.camera_profile_removed.connect(
                self._on_camera_profile_removed
            )
            self._adjust_edit_base_signals = _AdjustEditBaseSignals()
            self._adjust_edit_base_signals.finished.connect(self._on_adjust_edit_base_ready)
            self._adjust_mask_load_signals = _AdjustMaskLoadSignals()
            self._adjust_mask_load_signals.finished.connect(self._on_adjust_mask_load_ready)
            self._adjust_mask_load_gen = 0
            self._adjust_preview_gen = 0
            self._adjust_preview_busy = False
            self._adjust_preview_dirty = False
            self._adjust_preview_dirty_full_quality = False
            self._adjust_preview_signals = _AdjustPreviewSignals()
            self._adjust_preview_signals.finished.connect(self._on_adjust_preview_ready)
            self._adjust_preview_signals.ai_progress.connect(self._on_adjust_ai_progress)
            from raw_adjustments.pipeline import PreviewStageCache

            self._adjust_preview_stage_cache = PreviewStageCache()
            self._adjust_preview_base_rgb_fast: Any = None
            self._adjust_preview_stage_cache_fast = PreviewStageCache()
            self._adjust_compare_original_pixmap: Any = None
            self._adjust_compare_original_path: str | None = None
            self._adjust_compare_original_signals = _AdjustCompareOriginalSignals()
            self._adjust_compare_original_signals.finished.connect(
                self._on_adjust_compare_original_ready
            )
            self._adjust_edit_base_path: str | None = None
            self._adjust_edit_base_loading_norm: str | None = None
            self._adjust_export_in_progress = False
            self._adjust_export_signals = _AdjustExportSignals()
            self._adjust_export_signals.finished.connect(self._on_adjust_export_finished)
            self._adjust_export_signals.progress.connect(self._on_adjust_export_progress)
            self._adjust_preview_painting = False
            self._adjust_hist_sync_timer = QTimer(self)
            self._adjust_hist_sync_timer.setSingleShot(True)
            self._adjust_hist_sync_timer.timeout.connect(self._sync_single_image_histogram)
        self._dodge_burn_mask = None
        self._spot_heal_mask = None
        self._dodge_burn_stroke_active = False
        self._dodge_burn_stroke_baseline = None
        self._dodge_burn_mask_at_stroke_start = None
        self._dodge_burn_stroke_dirty = None
        self._crop_preview_uncropped = False
        self._crop_working_insets = (0.0, 0.0, 0.0, 0.0)
        self._geometry_overlay_preview = False
        self._transform_interacting = False

        # Route B: GPU-accelerated single-image view (QGraphicsView + OpenGL).
        # On by default; set RAWVIEWER_GPU_VIEW=0 for the legacy QScrollArea/QLabel path.
        self.gpu_view = None
        if _env_true("RAWVIEWER_GPU_VIEW", default=True):
            try:
                from rawviewer_ui.widgets.gpu import GpuImageView
                self.gpu_view = GpuImageView()
                # Reuse the legacy instruction screen text as the empty-state placeholder.
                self.gpu_view.set_placeholder_text(self.image_label.text())
                self.gpu_view.wheelNavigate.connect(self._on_gpu_wheel_navigate)
                self.gpu_view.fitModeChanged.connect(self._on_gpu_fit_mode_changed)
                self.gpu_view.zoomChanged.connect(self._on_gpu_zoom_changed)
                self.gpu_view.doubleClickedAt.connect(self._on_gpu_double_clicked)
                if editing_features_enabled():
                    self.gpu_view.colorPickRequested.connect(self._on_wb_color_picked)
                    self.gpu_view.dodgeBurnStroke.connect(self._on_dodge_burn_stroke)
                    self.gpu_view.dodgeBurnBrushSizeWheel.connect(
                        self._on_dodge_burn_brush_size_wheel
                    )
                    self.gpu_view.brushToolLeftImage.connect(
                        self._on_brush_tool_left_image
                    )
                    self.gpu_view.cropInsetsChanged.connect(self._on_crop_insets_live)
                    self.gpu_view.cropEditingFinished.connect(
                        lambda: None
                    )  # live only; commit via Apply
                self.gpu_view._shortcut_handler = self._handle_app_shortcut
                self._sync_composition_grid_display()
            except Exception as _gpu_exc:
                logging.getLogger(__name__).warning(
                    "GPU view requested but failed to initialize: %s", _gpu_exc
                )
                self.gpu_view = None


    def _init_splitter_and_status_bar(self):
        self.single_view_container = SingleImageViewOverlay(
            self.scroll_area, self.single_image_histogram, viewer=self,
            gpu_view=self.gpu_view, map_widget=self.single_image_location_map)
        self._wire_filmstrip()

        self.single_view_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.single_view_splitter.addWidget(self.single_view_container)
        if self.single_image_adjust_panel is not None:
            # None when editing is disabled (RAWVIEWER_ENABLE_EDITING=0) --
            # QSplitter.addWidget(None) segfaults at the sip
            # binding layer rather than raising a catchable Python error.
            self.single_view_splitter.addWidget(self.single_image_adjust_panel)
        # Give most space to the image viewer, but allow the panel to expand/shrink within its limits.
        self.single_view_splitter.setStretchFactor(0, 1)
        self.single_view_splitter.setStretchFactor(1, 0)
        if self.single_image_adjust_panel is not None:
            self._apply_adjust_panel_dock(self._adjust_panel_dock_side())

        self.main_layout.addWidget(self.single_view_splitter)
        self._init_compare_view(self.main_layout)
        QTimer.singleShot(800, self._show_edr_startup_status)
        # --- Status bar with darkroom styling (see theme.py) ---
        # - Surface: theme.VOID (dark background)
        # - On Surface: theme.INK (primary text)
        # - Surface Variant: theme.SURFACE (elevated surface)
        # - Outline: theme.RAISED (borders)
        # - Secondary: theme.INK_MUTED (secondary text)
        # Create status bar
        _status = self.statusBar()
        if _status is None:
            _status = QStatusBar(self)
            self.setStatusBar(_status)
        self.status_bar = _status
        self.status_bar.showMessage("")  # Empty message when no image loaded
        # Edge resize is handled on the frameless window; grip only adds dead space on the right.
        self.status_bar.setSizeGripEnabled(False)
        # Symmetric horizontal padding -- was 20px right-only effective inset vs ~12px left.
        self.status_bar.setStyleSheet(f"""
            QStatusBar {{
                background-color: {theme.VOID} !important;
                color: {theme.INK};
                border-top: 1px solid {theme.RAISED};
                padding: 10px 12px;
                font-size: 13px;
                font-weight: 400;
            }}
            QStatusBar::item {{
                border: none;
            }}
        """)


    def _init_status_bar(self):
        status_widget = QWidget()
        status_widget.setObjectName("status_widget")  # Set object name for finding it later
        status_layout = QHBoxLayout(status_widget)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(12)

        # Left side buttons container -- fixed footprint; search field overlays beside it.
        left_buttons_widget = QWidget()
        left_buttons_widget.setFixedHeight(28)
        left_buttons_widget.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed
        )
        left_buttons_layout = QHBoxLayout(left_buttons_widget)
        left_buttons_layout.setContentsMargins(8, 0, 0, 0)
        # Add 8px spacing between buttons
        left_buttons_layout.setSpacing(8)

        import qtawesome as qta
        bottom_icon_only_btn_style = f"""
            QPushButton,
            QPushButton:flat {{
                color: {theme.INK_MUTED};
                padding: 0px;
                margin: 0px;
                border: none;
                background: transparent;
                min-width: 28px;
                max-width: 28px;
                min-height: 28px;
                max-height: 28px;
                border-radius: 4px;
            }}
            QPushButton:hover,
            QPushButton:flat:hover,
            QPushButton:hover:enabled,
            QPushButton:flat:hover:enabled {{
                color: {theme.INK};
                padding: 0px;
                margin: 0px;
                background-color: rgba(255, 255, 255, 0.05);
            }}
            QPushButton:pressed,
            QPushButton:flat:pressed {{
                padding: 0px;
                margin: 0px;
                background-color: rgba(255, 255, 255, 0.1);
            }}
        """
        # Store shared resources as instance attributes so subsequent
        # init methods can access them (scope boundary after method split).
        # The parent widget must also be stored here — if it is GC'd before later
        # methods run, PyQt6 deletes its child layout C++ objects even though the
        # Python layout wrapper is still alive (→ RuntimeError: deleted).
        self.status_bar_widget = status_widget
        self.status_layout = status_layout
        self.left_buttons_widget = left_buttons_widget
        self.left_buttons_layout = left_buttons_layout
        self.bottom_icon_only_btn_style = bottom_icon_only_btn_style
        self.qta = qta


    def _style_bottom_icon_button(self, btn: QPushButton) -> None:
        btn.setFlat(True)
        btn.setAutoDefault(False)
        btn.setDefault(False)
        btn.setFixedSize(28, 28)
        btn.setIconSize(QSize(20, 20))
        btn.setStyleSheet(self.bottom_icon_only_btn_style)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)


    def _init_bottom_actions(self):

        # Open button (left side) - Using qtawesome for reliability
        self.open_button = QPushButton()
        self.open_button.setIcon(_qta_icon('fa5s.folder-open', color=theme.INK_MUTED))
        self.open_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.open_button.setToolTip("Open Image File")
        self.open_button.clicked.connect(self.open_file)
        self._style_bottom_icon_button(self.open_button)
        self.left_buttons_layout.addWidget(self.open_button, 0, alignment=Qt.AlignmentFlag.AlignVCenter)

        # Gallery view toggle button
        self.view_mode_button = QPushButton()
        # Use qtawesome icon if available, otherwise fallback to text
        if self.qta is not None:
            try:
                gallery_icon = _qta_icon('fa5s.th', color=theme.INK_MUTED)
                self.view_mode_button.setIcon(gallery_icon)
                self.view_mode_button.setIconSize(QSize(20, 20))
            except Exception as e:
                safe_print(f"[WARNING] Failed to set qtawesome icon: {e}, using text fallback", flush=True)
                self.view_mode_button.setText("Gallery")
        else:
            self.view_mode_button.setText("Gallery")
        self.view_mode_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.view_mode_button.clicked.connect(self._on_view_mode_button_clicked)
        self._style_bottom_icon_button(self.view_mode_button)
        self.left_buttons_layout.addWidget(self.view_mode_button, 0, alignment=Qt.AlignmentFlag.AlignVCenter)
        self.view_mode_button.hide()  # Hidden by default until images are loaded

        self.share_bottom_button = QPushButton()
        self.share_bottom_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        try:
            self.share_bottom_button.setIcon(_qta_icon("fa5s.share-alt", color=theme.INK_MUTED))
        except Exception as e:
            safe_print(f"[WARNING] Share icon unavailable: {e}, using text fallback", flush=True)
            self.share_bottom_button.setText("Open")
        self._style_bottom_icon_button(self.share_bottom_button)
        self.share_bottom_button.setToolTip(
            "Open in another app" if sys.platform == "win32" else "Share"
        )
        # pressed (mouse-down), not clicked (mouse-up): NSSharingServicePicker must not
        # open on mouseUp or the macOS share sheet spins empty (AppKit console warning).
        self.share_bottom_button.pressed.connect(self._on_share_bottom_button_clicked)
        self.share_bottom_button.hide()

        self.batch_mark_indicator = QPushButton()
        self.batch_mark_indicator.setObjectName("batchMarkIndicator")
        self.batch_mark_indicator.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._bookmark_filter_star_icon = _qta_icon("fa5s.star", color=theme.DODGE)
        try:
            self._bookmark_star_outline_icon = _qta_icon("mdi.star-outline", color=theme.INK_MUTED)
        except Exception:
            # fa5r/far regular fonts are not bundled in all qtawesome builds.
            self._bookmark_star_outline_icon = _qta_icon("fa5s.star", color=theme.INK_FAINT)
        self._style_bottom_icon_button(self.batch_mark_indicator)
        self.batch_mark_indicator.clicked.connect(self._on_bookmark_slot_clicked)
        self.batch_mark_indicator.setToolTip(
            "Bookmarks retired -- use 1–5 ratings and the gallery rating-filter stars."
        )
        self.batch_mark_indicator.hide()

        # Single-view rating control: 5 clickable stars (replaces the old
        # bookmark star). ↑ is unused in single/gallery for bookmarks;
        # Compare mode still uses ↑ to promote the candidate.
        self.single_view_rating_stars = QWidget()
        self.single_view_rating_stars.setObjectName("singleViewRatingStars")
        _rating_stars_layout = QHBoxLayout(self.single_view_rating_stars)
        _rating_stars_layout.setContentsMargins(0, 0, 0, 0)
        _rating_stars_layout.setSpacing(0)
        self._rating_star_filled_icon = _qta_icon("fa5s.star", color=theme.DODGE)
        self._rating_star_empty_icon = self._bookmark_star_outline_icon
        self._rating_star_buttons = []
        for _star_n in range(1, 6):
            _star_btn = QPushButton()
            _star_btn.setObjectName("ratingStarButton")
            _star_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            _star_btn.setFlat(True)
            _star_btn.setFixedSize(20, 28)
            _star_btn.setIconSize(QSize(13, 13))
            _star_btn.setIcon(self._rating_star_empty_icon)
            _star_btn.setStyleSheet(
                "QPushButton { background: transparent; border: none; }"
                f" QPushButton:hover {{ background: {theme.rgba(theme.INK_RGB, 18)}; border-radius: 3px; }}"
            )
            _star_btn.setToolTip(f"Rate {_star_n} star{'s' if _star_n != 1 else ''} ({_star_n})")
            _star_btn.clicked.connect(
                lambda checked=False, n=_star_n: self._on_rating_star_clicked(n)
            )
            self._rating_star_buttons.append(_star_btn)
            _rating_stars_layout.addWidget(_star_btn, 0, alignment=Qt.AlignmentFlag.AlignVCenter)
        self.single_view_rating_stars.hide()

        # Gallery-view rating filter: 5 clickable stars, click star N to show
        # only images rated N and above (click the active threshold again to
        # clear). Independent of the single-view rating control above and of
        # the bookmark filter -- both can be active at once (intersection).
        self.gallery_rating_filter_stars = QWidget()
        self.gallery_rating_filter_stars.setObjectName("galleryRatingFilterStars")
        _gallery_filter_stars_layout = QHBoxLayout(self.gallery_rating_filter_stars)
        _gallery_filter_stars_layout.setContentsMargins(0, 0, 0, 0)
        _gallery_filter_stars_layout.setSpacing(0)
        self._gallery_rating_filter_buttons = []
        for _star_n in range(1, 6):
            _star_btn = QPushButton()
            _star_btn.setObjectName("galleryRatingFilterStarButton")
            _star_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            _star_btn.setFlat(True)
            _star_btn.setFixedSize(20, 28)
            _star_btn.setIconSize(QSize(13, 13))
            _star_btn.setIcon(self._rating_star_empty_icon)
            _star_btn.setStyleSheet(
                "QPushButton { background: transparent; border: none; }"
                f" QPushButton:hover {{ background: {theme.rgba(theme.INK_RGB, 18)}; border-radius: 3px; }}"
            )
            _star_btn.setToolTip(f"Show {_star_n}★ and above")
            _star_btn.clicked.connect(
                lambda checked=False, n=_star_n: self._on_gallery_rating_star_clicked(n)
            )
            self._gallery_rating_filter_buttons.append(_star_btn)
            _gallery_filter_stars_layout.addWidget(
                _star_btn, 0, alignment=Qt.AlignmentFlag.AlignVCenter
            )
        self.gallery_rating_filter_stars.hide()

        # Explicit "show all images" escape hatch for when the rating (or
        # bookmark) filter narrows the gallery -- clicking the active
        # threshold star again does the same thing, but that's not obvious,
        # so surface a dedicated button whenever a filter is actually active.
        self.gallery_clear_filter_button = QPushButton()
        self.gallery_clear_filter_button.setObjectName("galleryClearFilterButton")
        self.gallery_clear_filter_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        try:
            self.gallery_clear_filter_button.setIcon(_qta_icon("fa5s.th", color=theme.INK_MUTED))
        except Exception:
            self.gallery_clear_filter_button.setText("Gallery")
        self._style_bottom_icon_button(self.gallery_clear_filter_button)
        self.gallery_clear_filter_button.setToolTip("Show all images (clear filter)")
        self.gallery_clear_filter_button.clicked.connect(self._on_gallery_clear_filter_clicked)
        self.gallery_clear_filter_button.hide()
        # Sits with the other view-scope controls (left side) rather than beside the
        # rating-filter stars on the right -- it toggles gallery scope, not rating.
        self.left_buttons_layout.addWidget(
            self.gallery_clear_filter_button, 0, alignment=Qt.AlignmentFlag.AlignVCenter
        )

        self.slideshow_bottom_button = QPushButton()
        self.slideshow_bottom_button.setObjectName("slideshowBottomButton")
        self.slideshow_bottom_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.slideshow_bottom_button.setIcon(_qta_icon("fa5s.play", color=theme.INK_MUTED))
        self._style_bottom_icon_button(self.slideshow_bottom_button)
        self.slideshow_bottom_button.clicked.connect(self._on_slideshow_bottom_clicked)
        self.slideshow_bottom_button.hide()
        self.left_buttons_layout.addWidget(self.slideshow_bottom_button, 0, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.rotate_bottom_button = QPushButton()
        self.rotate_bottom_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.rotate_bottom_button.setIcon(_qta_icon("fa5s.redo", color=theme.INK_MUTED))
        self._style_bottom_icon_button(self.rotate_bottom_button)
        self.rotate_bottom_button.clicked.connect(self._rotate_current_image_clockwise_persist)
        self.rotate_bottom_button.hide()
        self.left_buttons_layout.addWidget(self.rotate_bottom_button, 0, alignment=Qt.AlignmentFlag.AlignVCenter)


    def _init_more_buttons(self):
        self.adjust_bottom_button = QPushButton()
        self.adjust_bottom_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.adjust_bottom_button.setIcon(_qta_icon("fa5s.sliders-h", color=theme.INK_MUTED))
        self.adjust_bottom_button.setToolTip("Adjust -- develop this photo (E)")
        self._style_bottom_icon_button(self.adjust_bottom_button)
        self.adjust_bottom_button.clicked.connect(self._toggle_adjust_panel)
        self.adjust_bottom_button.hide()
        self.left_buttons_layout.addWidget(self.adjust_bottom_button, 0, alignment=Qt.AlignmentFlag.AlignVCenter)

        # RAW/JPEG Toggle button (before search so expand field stays adjacent to search icon)
        self.raw_toggle_button = QPushButton()
        self.raw_toggle_button.setFlat(True)
        self.raw_toggle_button.setAutoDefault(False)
        self.raw_toggle_button.setDefault(False)
        self.raw_toggle_button.setFixedSize(44, 28)
        self.raw_toggle_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.raw_toggle_button.clicked.connect(self.toggle_raw_jpeg_workflow)
        self.raw_toggle_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.raw_toggle_button.hide()
        self.left_buttons_layout.addWidget(self.raw_toggle_button, 0, alignment=Qt.AlignmentFlag.AlignVCenter)

        # EDR display toggle (macOS only; shown beside RAW when the RAW
        # workflow is active). EDR re-decodes each RAW with LibRaw highlight
        # reconstruction (~12x a plain decode), so users can turn it off here
        # to browse RAW much faster at standard dynamic range.
        from common_image_loader import is_macos_edr_enabled as _edr_available

        if _edr_available():
            self.edr_toggle_button = QPushButton()
            self.edr_toggle_button.setFlat(True)
            self.edr_toggle_button.setAutoDefault(False)
            self.edr_toggle_button.setDefault(False)
            self.edr_toggle_button.setFixedSize(44, 28)
            self.edr_toggle_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            self.edr_toggle_button.clicked.connect(self.toggle_raw_edr_display)
            self.edr_toggle_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.edr_toggle_button.hide()
            self.left_buttons_layout.addWidget(
                self.edr_toggle_button, 0, alignment=Qt.AlignmentFlag.AlignVCenter
            )

        # Sort toggle (gallery only) -- before search icon so Search sits beside the expand field.
        self.sort_toggle_button = QPushButton()
        self.sort_toggle_button.setFlat(True)
        self.sort_toggle_button.setAutoDefault(False)
        self.sort_toggle_button.setDefault(False)
        self.sort_toggle_button.setFixedHeight(28)
        self.sort_toggle_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._update_sort_button_text()
        self.sort_toggle_button.clicked.connect(self.toggle_sort_method)
        self.sort_toggle_button.setStyleSheet(f"""
            QPushButton {{
                color: {theme.INK_MUTED};
                font-size: 13px;
                font-weight: 500;
                padding: 0px 10px;
                margin: 0px;
                border: none;
                background: transparent;
                text-align: center;
                letter-spacing: 0.25px;
                min-height: 28px;
                max-height: 28px;
            }}
            QPushButton:hover {{
                color: {theme.INK};
                background-color: rgba(255, 255, 255, 0.05);
                border-radius: 4px;
            }}
            QPushButton:pressed {{
                background-color: rgba(255, 255, 255, 0.1);
            }}
        """)
        self.sort_toggle_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.left_buttons_layout.addWidget(
            self.sort_toggle_button, 0, alignment=Qt.AlignmentFlag.AlignVCenter
        )
        self.sort_toggle_button.hide()  # Hidden by default (single view)

        # Size slider (gallery zoom)
        from rawviewer_ui.widgets import GalleryZoomSlider
        self.size_slider = GalleryZoomSlider(Qt.Orientation.Horizontal, self)

        # Read saved row height from settings
        settings = self.get_settings()
        saved_row_height = int(settings.value("gallery_row_height", 220))
        saved_row_height = max(220, min(380, saved_row_height))

        self.size_slider.setRange(220, 380)
        self.size_slider.setValue(saved_row_height)
        self.size_slider.setToolTip("Adjust thumbnail size")
        self.size_slider.hide()  # Hidden by default (single view)

        # Add to layout
        self.left_buttons_layout.addWidget(
            self.size_slider, 0, alignment=Qt.AlignmentFlag.AlignVCenter
        )

        # Debounce timer for smooth dragging without layout lag
        self._zoom_debounce_timer = QTimer(self)
        self._zoom_debounce_timer.setSingleShot(True)
        self._zoom_debounce_timer.timeout.connect(self._rebuild_gallery_zoom)

        self.size_slider.sliderPressed.connect(self._on_zoom_slider_pressed)
        self.size_slider.valueChanged.connect(self._on_zoom_slider_changed)
        self.size_slider.sliderReleased.connect(self._on_zoom_slider_released)

        self.search_bottom_button = QPushButton()
        if self.qta is not None:
            try:
                self.search_bottom_button.setIcon(_qta_icon("fa5s.search", color=theme.INK_MUTED))
            except Exception:
                self.search_bottom_button.setText("Search")
        else:
            self.search_bottom_button.setText("Search")
        self.search_bottom_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._style_bottom_icon_button(self.search_bottom_button)
        self.search_bottom_button.clicked.connect(self._on_search_bottom_clicked)
        self.search_bottom_button.hide()
        self.left_buttons_layout.addWidget(self.search_bottom_button, 0, alignment=Qt.AlignmentFlag.AlignVCenter)


    def _init_search_ui(self):
        self.search_expand_container = QWidget()
        self.search_expand_container.setFixedHeight(28)
        self.search_expand_container.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        self.search_expand_container.setMinimumWidth(0)
        self.search_expand_container.setMaximumWidth(0)
        self.search_expand_layout = QStackedLayout(self.search_expand_container)
        self.search_expand_layout.setContentsMargins(0, 0, 0, 0)
        # Width animation clips a fixed-size field; layout must not fight setFixedWidth.
        with contextlib.suppress(Exception):
            self.search_expand_layout.setSizeConstraint(
                QLayout.SizeConstraint.SetNoConstraint
            )

        self.gallery_search_panel = QWidget()
        self.gallery_search_panel.setFixedHeight(28)
        _gsp = QHBoxLayout(self.gallery_search_panel)
        _gsp.setContentsMargins(0, 0, 10, 0)
        _gsp.setSpacing(8)

        self.gallery_search_status_label = QLabel("")
        self.gallery_search_status_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.gallery_search_status_label.setWordWrap(False)
        self.gallery_search_status_label.setMinimumWidth(0)
        self.gallery_search_status_label.setMaximumWidth(220)
        self.gallery_search_status_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        self.gallery_search_status_label.setStyleSheet(f"""
            QLabel {{
                color: {theme.INK_MUTED};
                font-size: 12px;
                font-weight: 500;
                padding: 0px 0px 0px 2px;
            }}
        """)
        self.gallery_search_status_label.hide()

        self.gallery_search_input = QLineEdit()
        self.gallery_search_input.setPlaceholderText("Search gallery")
        self.gallery_search_input.setClearButtonEnabled(True)
        self.gallery_search_input.setMinimumWidth(140)
        self.gallery_search_input.setMaximumWidth(300)
        self.gallery_search_input.setFixedHeight(28)
        self.gallery_search_input.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        self.gallery_search_style_input = f"""
            QLineEdit {{
                color: {theme.INK};
                background-color: {theme.SURFACE};
                border: 1px solid {theme.LINE};
                border-radius: 6px;
                padding: 0px 28px 0px 10px;
                font-size: 13px;
                min-height: 0px;
                max-height: 28px;
                outline: none;
            }}
            QLineEdit:focus {{
                border: 1px solid {theme.EMBER};
                outline: none;
            }}
        """
        self.gallery_search_input.setStyleSheet(self.gallery_search_style_input)
        # macOS draws a bright blue native focus ring unless disabled (dialogs already do this).
        self.gallery_search_input.setAttribute(
            Qt.WidgetAttribute.WA_MacShowFocusRect, False
        )
        _macos_install_line_edit_completion_guard(self.gallery_search_input)
        self.gallery_search_input.returnPressed.connect(self._semantic_search_from_bar)
        self.gallery_search_input.textChanged.connect(self._on_gallery_search_text_changed)

        _gsp.addWidget(self.gallery_search_input, 0)
        _gsp.addWidget(self.gallery_search_status_label, 0)
        _gsp.addStretch(1)
        self.search_expand_layout.addWidget(self.gallery_search_panel)
        self.search_expand_layout.setCurrentWidget(self.gallery_search_panel)
        self._gallery_search_status_full = ""


    def _init_semantic_and_status_flags(self):
        self.search_expand_container.hide()
        # Index progress in the search field only after the user opens search (not folder load).
        self._gallery_search_show_index_progress = False

        self._load_burst_grouping_preference()

        self.burst_grouping_button = QPushButton()
        self.burst_grouping_button.setCheckable(True)
        self.burst_grouping_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.burst_grouping_button.setToolTip(
            "Group burst sequences -- stack rapid captures into one tile"
        )
        self._style_bottom_icon_button(self.burst_grouping_button)
        self.burst_grouping_button.setStyleSheet(
            self.bottom_icon_only_btn_style
            + f"""
            QPushButton:checked,
            QPushButton:checked:flat {{
                color: {theme.INK};
                background: transparent;
            }}
        """
        )
        self.burst_grouping_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.burst_grouping_button.toggled.connect(self._on_burst_grouping_toggled)
        self._sync_burst_grouping_button_icon()
        self.burst_grouping_button.hide()

        self.compare_bottom_button = QPushButton()
        self.compare_bottom_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.compare_bottom_button.setToolTip(
            "Compare side by side (C) -- one burst group or multi-selection"
        )
        if self.qta is not None:
            try:
                self.compare_bottom_button.setIcon(
                    _qta_icon("fa5s.columns", color=theme.INK_MUTED)
                )
            except Exception:
                self.compare_bottom_button.setText("Compare")
        else:
            self.compare_bottom_button.setText("Compare")
        self._style_bottom_icon_button(self.compare_bottom_button)
        self.compare_bottom_button.clicked.connect(self._enter_compare_mode)
        self.compare_bottom_button.hide()
        self.compare_bottom_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # User opened search: run/resume semantic indexing, then allow gallery queries.
        self._gallery_search_user_wants_semantic = False
        # Latch user intent when search is clicked before metadata pass completes.
        self._defer_semantic_after_metadata = False
        self._semantic_index_pass_kind = None  # silent_metadata | user_semantic

        self.shortcuts_hint_button = QPushButton("i")
        self.shortcuts_hint_button.setFlat(True)
        self.shortcuts_hint_button.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        self.shortcuts_hint_button.setFixedSize(28, 28)
        self.shortcuts_hint_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.shortcuts_hint_button.setToolTip(self._keyboard_shortcuts_tooltip())
        self.shortcuts_hint_button.setStyleSheet(f"""
            QPushButton {{
                color: {theme.INK_FAINT};
                font-size: 12px;
                font-weight: 600;
                padding: 0px;
                border: none;
                background: transparent;
                border-radius: 14px;
                min-width: 28px;
                max-width: 28px;
                min-height: 28px;
                max-height: 28px;
            }}
            QPushButton:hover {{
                color: {theme.INK};
                background-color: rgba(255, 255, 255, 0.08);
            }}
            QPushButton:pressed {{
                background-color: rgba(255, 255, 255, 0.12);
            }}
        """)

        # Trailing status: [share][i] in right_status_actions, then counter (v2.1 two-level layout).
        _counter_right_pad = 6 if platform.system() == "Windows" else 4
        _counter_label_style = f"""
            QLabel {{
                color: {theme.INK_MUTED};
                font-size: 13px;
                font-weight: 400;
                padding: 0px {_counter_right_pad}px 0px 0px;
                letter-spacing: 0.25px;
            }}
        """
        self.status_counter_label = QLabel("")
        self.status_counter_label.setFixedHeight(28)
        self.status_counter_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.status_counter_label.setStyleSheet(_counter_label_style)

        self.right_status_actions = QWidget()
        self.right_status_actions.setFixedHeight(28)
        self.right_status_actions.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed
        )
        right_status_actions_layout = QHBoxLayout(self.right_status_actions)
        right_status_actions_layout.setContentsMargins(0, 0, 0, 0)
        right_status_actions_layout.setSpacing(8)
        right_status_actions_layout.addWidget(
            self.burst_grouping_button, 0, alignment=Qt.AlignmentFlag.AlignVCenter)
        right_status_actions_layout.addWidget(
            self.compare_bottom_button, 0, alignment=Qt.AlignmentFlag.AlignVCenter)
        right_status_actions_layout.addWidget(
            self.batch_mark_indicator, 0, alignment=Qt.AlignmentFlag.AlignVCenter)
        right_status_actions_layout.addWidget(
            self.gallery_rating_filter_stars, 0, alignment=Qt.AlignmentFlag.AlignVCenter)
        right_status_actions_layout.addWidget(
            self.single_view_rating_stars, 0, alignment=Qt.AlignmentFlag.AlignVCenter)
        right_status_actions_layout.addWidget(
            self.share_bottom_button, 0, alignment=Qt.AlignmentFlag.AlignVCenter)
        right_status_actions_layout.addWidget(
            self.shortcuts_hint_button, 0, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.right_status_trailing = QWidget()
        self.right_status_trailing.setFixedHeight(28)
        self.right_status_trailing.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed
        )
        _rtl = QHBoxLayout(self.right_status_trailing)
        _rtl.setContentsMargins(0, 0, 0, 0)
        _rtl.setSpacing(12)
        _rtl.addWidget(self.right_status_actions, 0, alignment=Qt.AlignmentFlag.AlignVCenter)
        _rtl.addWidget(self.status_counter_label, 0, alignment=Qt.AlignmentFlag.AlignVCenter)

        # Add left buttons to main layout; EXIF/metadata lives in the top bar (center).
        self.status_layout.addWidget(self.left_buttons_widget)
        self.status_layout.addStretch(1)
        self.status_layout.addWidget(self.right_status_trailing, 0, alignment=Qt.AlignmentFlag.AlignVCenter)
        self._set_shortcuts_hint_hovered(True)

        # Search field overlays the status bar to the right of the search
        # button so left/right chrome keep a fixed footprint (no icon flicker).
        if getattr(self, "search_expand_container", None) is not None:
            self.search_expand_container.setParent(self.status_bar_widget)
            self.search_expand_container.raise_()
            self.search_expand_container.hide()

        # Add custom widget to status bar
        self.status_bar.addPermanentWidget(self.status_bar_widget, 1)

        # Bottom-right resize grip (frameless Windows) -- raised above status bar chrome.
        self._resize_grip = ResizeGripIndicator(self)
        self._position_resize_grip()
        self.status_bar.setMouseTracking(True)
        self.status_bar.installEventFilter(self)
        self.status_bar_widget.setMouseTracking(True)
        self.status_bar_widget.installEventFilter(self)

        # Track mouse over client area so edge cursors / grip appear without clicking first.
        self.setMouseTracking(True)


    def _init_shortcuts_and_wiring(self):
        cw = self.centralWidget()
        if cw is not None:
            cw.setMouseTracking(True)
            # No event filter on centralWidget -- POC/v2.1 parity; filter here breaks macOS share sheet.

        # Status bar created - no rounded corners to update

        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # v2.1-style filters: image scroll + optional GPU view only (no status bar / container).
        self.scroll_area.installEventFilter(self)
        self.image_label.installEventFilter(self)
        self.single_view_container.installEventFilter(self)
        if self.gpu_view is not None:
            self.gpu_view.setMouseTracking(True)
            self.gpu_view.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
            self.gpu_view.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
            self.gpu_view.viewport().setMouseTracking(True)
            self.gpu_view.viewport().setAttribute(Qt.WidgetAttribute.WA_Hover, True)
            self.gpu_view.viewport().setAttribute(
                Qt.WidgetAttribute.WA_AcceptTouchEvents, True
            )
            self.gpu_view.installEventFilter(self)
            self.gpu_view.viewport().installEventFilter(self)
        self.single_image_histogram.installEventFilter(self)
        if self.single_image_location_map is not None:
            self.single_image_location_map.installEventFilter(self)
        if self.single_image_adjust_panel is not None:
            self.single_image_adjust_panel.installEventFilter(self)

        # F must work even when no child widget accepts keyboard focus (QLabel default
        # is NoFocus). WindowShortcut fires while this window is active.
        self._shortcut_toggle_focus_subject_outline = QShortcut(
            QKeySequence(Qt.Key.Key_F), self
        )
        self._shortcut_toggle_focus_subject_outline.setContext(
            Qt.ShortcutContext.ApplicationShortcut
        )
        self._shortcut_toggle_focus_subject_outline.setAutoRepeat(False)
        self._shortcut_toggle_focus_subject_outline.activated.connect(
            self._toggle_focus_subject_outline
        )
        self._shortcut_toggle_composition_grid = QShortcut(
            QKeySequence(Qt.Key.Key_G), self
        )
        self._shortcut_toggle_composition_grid.setContext(
            Qt.ShortcutContext.ApplicationShortcut
        )
        self._shortcut_toggle_composition_grid.setAutoRepeat(False)
        self._shortcut_toggle_composition_grid.activated.connect(
            self._toggle_composition_grid
        )
        self._shortcut_toggle_clipping_overlay = QShortcut(
            QKeySequence(Qt.Key.Key_J), self
        )
        self._shortcut_toggle_clipping_overlay.setContext(
            Qt.ShortcutContext.ApplicationShortcut
        )
        self._shortcut_toggle_clipping_overlay.setAutoRepeat(False)
        self._shortcut_toggle_clipping_overlay.activated.connect(
            self._toggle_clipping_overlay
        )

        self._shortcut_toggle_adjust_panel = None
        from raw_adjustments import editing_features_enabled as _editing_on

        if _editing_on():
            self._shortcut_toggle_adjust_panel = QShortcut(QKeySequence(Qt.Key.Key_E), self)
            self._shortcut_toggle_adjust_panel.setContext(
                Qt.ShortcutContext.ApplicationShortcut
            )
            self._shortcut_toggle_adjust_panel.setAutoRepeat(False)
            self._shortcut_toggle_adjust_panel.activated.connect(
                self._shortcut_activate_adjust_panel
            )

        # H/M/P also use ApplicationShortcut; _handle_app_shortcut is the fallback when
        # focus is on overlay chrome or the OpenGL viewport swallows key delivery.
        self._shortcut_toggle_histogram = QShortcut(QKeySequence(Qt.Key.Key_H), self)
        self._shortcut_toggle_histogram.setContext(
            Qt.ShortcutContext.ApplicationShortcut
        )
        self._shortcut_toggle_histogram.setAutoRepeat(False)
        self._shortcut_toggle_histogram.activated.connect(
            self._shortcut_activate_histogram
        )

        self._shortcut_toggle_location_map = QShortcut(QKeySequence(Qt.Key.Key_M), self)
        self._shortcut_toggle_location_map.setContext(
            Qt.ShortcutContext.ApplicationShortcut
        )
        self._shortcut_toggle_location_map.setAutoRepeat(False)
        self._shortcut_toggle_location_map.activated.connect(
            self._shortcut_activate_location_map
        )

        self._shortcut_toggle_raw_recovery = QShortcut(QKeySequence(Qt.Key.Key_P), self)
        self._shortcut_toggle_raw_recovery.setContext(
            Qt.ShortcutContext.ApplicationShortcut
        )
        self._shortcut_toggle_raw_recovery.setAutoRepeat(False)
        self._shortcut_toggle_raw_recovery.activated.connect(
            self._shortcut_activate_raw_recovery
        )

        # Arrow keys must work when the GPU OpenGL viewport (or other NoFocus
        # children) holds focus; ApplicationShortcut matches the F-key pattern.
        self._shortcut_nav_prev = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        self._shortcut_nav_prev.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._shortcut_nav_prev.activated.connect(self._shortcut_activate_nav_prev)
        self._shortcut_nav_next = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        self._shortcut_nav_next.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._shortcut_nav_next.activated.connect(self._shortcut_activate_nav_next)

        self._shortcut_escape = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self._shortcut_escape.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._shortcut_escape.setAutoRepeat(False)
        self._shortcut_escape.activated.connect(self._shortcut_activate_escape)

        self._shortcut_compare = QShortcut(QKeySequence(Qt.Key.Key_C), self)
        self._shortcut_compare.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._shortcut_compare.setAutoRepeat(False)
        self._shortcut_compare.activated.connect(self._shortcut_activate_compare)

        self._shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        self._shortcut_undo.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._shortcut_undo.setAutoRepeat(False)
        self._shortcut_undo.activated.connect(self._shortcut_activate_undo)

        self._shortcut_gallery_up = QShortcut(QKeySequence(Qt.Key.Key_Up), self)
        self._shortcut_gallery_up.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._shortcut_gallery_up.activated.connect(self._shortcut_activate_gallery_up)

        self._shortcut_gallery_down = QShortcut(QKeySequence(Qt.Key.Key_Down), self)
        self._shortcut_gallery_down.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._shortcut_gallery_down.activated.connect(self._shortcut_activate_gallery_down)

        self._shortcut_gallery_down_shift = QShortcut(QKeySequence("Shift+Down"), self)
        self._shortcut_gallery_down_shift.setContext(
            Qt.ShortcutContext.ApplicationShortcut
        )
        self._shortcut_gallery_down_shift.activated.connect(self._shortcut_activate_gallery_down)

        app = QApplication.instance()
        if app is not None and not getattr(self, "_app_key_filter_installed", False):
            app.installEventFilter(self)
            self._app_key_filter_installed = True

        self._sync_single_image_histogram()

        self._setup_file_drop_targets()

        if platform.system() == "Darwin":
            QTimer.singleShot(0, self._apply_macos_titlebar_appearance)

    def init_ui(self):
        """Initialize the user interface"""
        # qtawesome doesn't require initialization - can be used directly
        self._init_window_chrome()
        self._init_content()
        self._init_harness_and_overlays()
        self._init_splitter_and_status_bar()
        self._init_status_bar()
        self._init_bottom_actions()
        self._init_more_buttons()
        self._init_search_ui()
        self._init_semantic_and_status_flags()
        self._init_shortcuts_and_wiring()
