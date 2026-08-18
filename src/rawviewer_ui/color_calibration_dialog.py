"""Interactive ColorChecker target alignment & camera calibration dialog for RAWviewer."""

from __future__ import annotations

import logging
import os
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QPointF, QRectF, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QDesktopServices, QFont, QImage, QPainter, QPen, QBrush, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    import rawviewer_ui.theme as theme
except ModuleNotFoundError:
    import theme as theme
from color_calibration import (
    camera_identity_for_file,
    validate_and_detect_color_checker,
    calibrate_camera_curves_and_hsl,
    save_camera_profile,
)
from dpreview_samples import dpreview_open_url

logger = logging.getLogger(__name__)


def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Normalize an edit-base buffer to 8-bit RGB.

    The adjust-panel base is 16-bit (0-65535) and other callers may pass
    float 0-1; sampling and the sRGB ColorChecker references both assume
    0-255, so rescale rather than clip (clipping blows 16-bit data to white).
    """
    if img is None or img.size == 0 or img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return (img.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
    if img.dtype in (np.float32, np.float64):
        scale = 255.0 if float(img.max()) <= 1.0 else 1.0
        return (img * scale).clip(0, 255).astype(np.uint8)
    return img.clip(0, 255).astype(np.uint8)


_SRGB_ENCODE_LUT = np.clip(
    np.where(
        (np.arange(256) / 255.0) <= 0.0031308,
        (np.arange(256) / 255.0) * 12.92,
        1.055 * np.power(np.arange(256) / 255.0, 1.0 / 2.4) - 0.055,
    )
    * 255.0,
    0,
    255,
).astype(np.uint8)


def _srgb_encode_uint8(img: np.ndarray) -> np.ndarray:
    """Scene-linear uint8 -> display sRGB uint8, via LUT (display only)."""
    return _SRGB_ENCODE_LUT[img]


class ColorCheckerPreviewCanvas(QWidget):
    """Canvas widget showing the image with 4 draggable corner handles over the color chart."""

    corners_changed = pyqtSignal()

    def __init__(self, image: np.ndarray, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.image = image
        self._pixmap = self._numpy_to_pixmap(image)
        
        # 4 corners in normalized [0, 1] coordinates
        self.corners: List[QPointF] = [
            QPointF(0.2, 0.3),
            QPointF(0.8, 0.3),
            QPointF(0.8, 0.7),
            QPointF(0.2, 0.7),
        ]
        self._drag_idx = -1
        self._hover_idx = -1
        self._handle_radius = 9.0
        self.setMouseTracking(True)
        self.setMinimumSize(480, 360)

    def set_image(self, image: Optional[np.ndarray]) -> None:
        """Replace the calibration reference without touching the viewer's current photo."""
        self._drag_idx = -1
        self._hover_idx = -1
        self.corners = [
            QPointF(0.2, 0.3),
            QPointF(0.8, 0.3),
            QPointF(0.8, 0.7),
            QPointF(0.2, 0.7),
        ]
        if image is None or getattr(image, "size", 0) == 0:
            self.image = None
            self.display_image = None
            self._pixmap = QPixmap()
        else:
            self._pixmap = self._numpy_to_pixmap(image)
        self.update()
        self.corners_changed.emit()

    def _numpy_to_pixmap(self, img: np.ndarray) -> QPixmap:
        if img is None or getattr(img, "size", 0) == 0:
            self.image = img
            self.display_image = img
            return QPixmap()

        # self.image stays scene-linear (calibrate_camera_curves_and_hsl does
        # its own exposure-normalise + sRGB encode and expects linear input),
        # but the *displayed* pixmap must be gamma-encoded or the chart is far
        # too dark to align corners against -- on the Canon test frame the
        # white patch sits at 81/255 linear, so the whole scene reads near
        # black without this.
        img = _to_uint8_rgb(img)
        self.image = np.ascontiguousarray(img)
        self.display_image = np.ascontiguousarray(_srgb_encode_uint8(img))
        img = self.display_image

        h, w = img.shape[:2]
        if img.ndim == 3:
            bytes_per_line = 3 * w
            qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            bytes_per_line = w
            qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(qimg)

    def auto_detect(self) -> bool:
        """Run OpenCV MCC auto-detection to find 24-patch ColorChecker grid corners."""
        src = getattr(self, "display_image", self.image)
        if src is None or getattr(src, "size", 0) == 0:
            return False
        try:
            import cv2.mcc as mcc
            detector = mcc.CCheckerDetector.create()
            # Detect on the display-encoded copy: MCC's patch models assume a
            # normally-rendered image, not a scene-linear one.
            img_bgr = cv2.cvtColor(src, cv2.COLOR_RGB2BGR) if src.ndim == 3 else src
            # process(image[, nc]) -- nc is the NUMBER OF CHARTS to look for,
            # not a chart type. Passing mcc.MCC24 here asked for nc=0 (the
            # enum's value), so detection could never succeed. Ask for a few
            # candidates; only the first is used.
            success = detector.process(img_bgr, 3)
            if success:
                checkers = detector.getListColorChecker()
                if checkers and len(checkers) > 0:
                    box = checkers[0].getBox()
                    h, w = self.image.shape[:2]
                    self.corners = [QPointF(float(p[0]) / w, float(p[1]) / h) for p in box]
                    self.update()
                    self.corners_changed.emit()
                    return True
        except Exception as exc:
            logger.debug("Auto detect MCC failed: %s", exc)
        return False

    def get_pixel_corners(self) -> List[Tuple[float, float]]:
        if self.image is None or getattr(self.image, "size", 0) == 0:
            return []
        h, w = self.image.shape[:2]
        return [(p.x() * w, p.y() * h) for p in self.corners]

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            rect = self._get_image_rect()
            pos = event.position()
            for idx, p in enumerate(self.corners):
                px = rect.left() + p.x() * rect.width()
                py = rect.top() + p.y() * rect.height()
                dist = np.hypot(pos.x() - px, pos.y() - py)
                if dist <= self._handle_radius * 1.8:
                    self._drag_idx = idx
                    self.update()
                    event.accept()
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        rect = self._get_image_rect()
        pos = event.position()
        
        if self._drag_idx != -1:
            nx = max(0.0, min(1.0, (pos.x() - rect.left()) / rect.width()))
            ny = max(0.0, min(1.0, (pos.y() - rect.top()) / rect.height()))
            self.corners[self._drag_idx] = QPointF(nx, ny)
            self.corners_changed.emit()
            self.update()
            event.accept()
            return

        old_hover = self._hover_idx
        self._hover_idx = -1
        for idx, p in enumerate(self.corners):
            px = rect.left() + p.x() * rect.width()
            py = rect.top() + p.y() * rect.height()
            dist = np.hypot(pos.x() - px, pos.y() - py)
            if dist <= self._handle_radius * 1.8:
                self._hover_idx = idx
                break

        if old_hover != self._hover_idx:
            self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor if self._hover_idx != -1 else Qt.CursorShape.ArrowCursor))
            self.update()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._drag_idx != -1:
            self._drag_idx = -1
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _get_image_rect(self) -> QRectF:
        if self._pixmap.isNull():
            return QRectF(self.rect())
        pw, ph = self._pixmap.width(), self._pixmap.height()
        cw, ch = self.width(), self.height()
        scale = min(cw / pw, ch / ph)
        rw, rh = pw * scale, ph * scale
        rx = (cw - rw) / 2.0
        ry = (ch - rh) / 2.0
        return QRectF(rx, ry, rw, rh)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self._get_image_rect()
        if self._pixmap.isNull():
            painter.setPen(QColor(theme.INK_FAINT))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Choose a ColorChecker / studio-scene file,\nor find a DPReview sample.",
            )
            painter.end()
            return

        painter.drawPixmap(rect.toRect(), self._pixmap)

        poly_pts = [
            QPointF(rect.left() + p.x() * rect.width(), rect.top() + p.y() * rect.height())
            for p in self.corners
        ]
        polygon = QPolygonF(poly_pts)

        # Draw semi-transparent target grid overlay
        painter.setPen(QPen(QColor(theme.EMBER), 2, Qt.PenStyle.SolidLine))
        painter.setBrush(QBrush(QColor(245, 140, 66, 40)))
        painter.drawPolygon(polygon)

        # Draw 4 corner handle circles
        for idx, pt in enumerate(poly_pts):
            is_active = (idx == self._drag_idx or idx == self._hover_idx)
            r = self._handle_radius * (1.3 if is_active else 1.0)
            painter.setPen(QPen(QColor("#FFFFFF"), 2))
            painter.setBrush(QBrush(QColor(theme.EMBER if is_active else theme.EMBER_DIM)))
            painter.drawEllipse(pt, r, r)

        painter.end()


class ColorCalibrationDialog(QDialog):
    """Interactive camera calibration dialog for adjusting ColorChecker target pins."""

    def __init__(
        self,
        image: Optional[np.ndarray],
        make: str,
        model: str,
        iso: Optional[int] = None,
        parent: Optional[QWidget] = None,
        has_factory_profile: Optional[bool] = None,
        source_file: Optional[str] = None,
    ):
        super().__init__(parent)
        # True  -> LibRaw ships a colour matrix for this body; calibrating
        #          overrides colour science that already works.
        # False -> generic colour (new/unprofiled body); calibration helps most.
        # None  -> unknown; say nothing rather than guess.
        self.has_factory_profile = has_factory_profile
        # Sample from the same 8-bit buffer the canvas displays, so the corner
        # pins the user places map 1:1 onto the pixels that get averaged.
        self.image = _to_uint8_rgb(image) if image is not None else None
        self._exif_make = (make or "").strip()
        self._exif_model = (model or "").strip()
        self.make = self._exif_make or "Camera"
        self.model = self._exif_model or "Model"
        self.iso = iso
        # The frame the chart was shot on, so the saved profile can record
        # whether the WB guardrail was active for it -- see
        # color_calibration.current_decode_signature.
        self.source_file = source_file
        self.calibrated_profile: Optional[Dict[str, Any]] = None

        iso_str = f" (ISO {iso})" if iso else ""
        self.setWindowTitle(f"Camera Color Calibration — {self.make} {self.model}{iso_str}")
        self.setMinimumSize(820, 600)
        self.setStyleSheet(
            f"""
            QDialog {{
                background-color: {theme.SURFACE};
                color: {theme.INK};
            }}
            QLabel {{
                color: {theme.INK};
                font-size: 13px;
            }}
            QGroupBox {{
                font-weight: 600;
                font-size: 13px;
                color: {theme.EMBER};
                border: 1px solid {theme.LINE};
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 12px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }}
            QComboBox {{
                background-color: {theme.RAISED};
                color: {theme.INK};
                border: 1px solid {theme.LINE};
                border-radius: 6px;
                padding: 4px 8px;
                min-height: 22px;
            }}
            QComboBox:hover {{
                border-color: {theme.rgba(theme.INK_RGB, 70)};
            }}
            QComboBox QAbstractItemView {{
                background-color: {theme.SURFACE};
                border: 1px solid {theme.LINE};
                color: {theme.INK};
                selection-background-color: {theme.rgba(theme.EMBER_RGB, 140)};
                selection-color: {theme.INK};
                outline: none;
            }}
            """
        )

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Header Title
        iso_str = f" (ISO {self.iso})" if self.iso else ""
        header_lbl = QLabel(f"🎯 Color Target Calibration — {self.make} {self.model}{iso_str}")
        # Not QFont("Segoe UI", ...): naming one family gives Qt no list to
        # fall through, so on macOS it silently substitutes rather than
        # reaching the platform UI font.
        header_lbl.setFont(theme.ui_font(19, QFont.Weight.Bold))
        header_lbl.setStyleSheet(f"color: {theme.INK};")
        layout.addWidget(header_lbl)

        sub_lbl = QLabel(
            "Drag the 4 corner handles over your 24-patch ColorChecker chart, or click Auto-Detect.\n"
            "The chart must be evenly lit and square-on — a shadow or falloff across the patches is "
            "measured as camera colour error and baked into the profile."
        )
        sub_lbl.setStyleSheet(f"color: {theme.INK_FAINT}; font-size: 12px;")
        layout.addWidget(sub_lbl)

        # Advisory, not a gate: a factory-profiled body can still need
        # calibration (mixed lighting, an aging chart), but the user should
        # know they are overriding working colour science rather than filling
        # a gap. Unprofiled bodies are where this feature genuinely helps.
        if self.has_factory_profile is not None:
            if self.has_factory_profile:
                note = (
                    "⚠️ LibRaw has a factory colour profile for this camera. "
                    "Calibrating overrides it for every photo from this body "
                    "that has no XMP sidecar — you can remove it later from the "
                    "Adjust panel."
                )
                colour = theme.EMBER
            else:
                note = (
                    "This camera has no factory colour profile in LibRaw, so it "
                    "currently decodes with generic colour. Calibration should "
                    "noticeably improve it."
                )
                colour = theme.INK_FAINT
            note_lbl = QLabel(note)
            note_lbl.setWordWrap(True)
            note_lbl.setStyleSheet(f"color: {colour}; font-size: 12px;")
            layout.addWidget(note_lbl)

        layout.addWidget(self._build_source_bar())

        # Canvas Widget
        self.canvas = ColorCheckerPreviewCanvas(self.image, self)
        layout.addWidget(self.canvas, 1)

        # Controls Row
        ctrl_box = QHBoxLayout()
        
        auto_btn = QPushButton("🎯 Auto-Detect Chart", self)
        auto_btn.setMinimumHeight(34)
        auto_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        auto_btn.setStyleSheet(self._secondary_button_style())
        auto_btn.clicked.connect(self._on_auto_detect)
        ctrl_box.addWidget(auto_btn)

        ctrl_box.addSpacing(16)
        ctrl_box.addWidget(QLabel("White Balance Reference:"))
        self.wb_combo = QComboBox(self)
        self.wb_combo.addItem("Auto (6-Patch Neutral Scale)", "auto")
        self.wb_combo.addItem("Patch 19 (White .05 D)", "white_19")
        self.wb_combo.addItem("Patch 22 (Neutral 5 / 18% Gray)", "neutral_22")
        ctrl_box.addWidget(self.wb_combo)

        ctrl_box.addStretch()

        cancel_btn = QPushButton("Cancel", self)
        cancel_btn.setMinimumHeight(34)
        cancel_btn.setMinimumWidth(80)
        cancel_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        cancel_btn.setStyleSheet(theme.form_secondary_button_stylesheet())
        cancel_btn.clicked.connect(self.reject)
        ctrl_box.addWidget(cancel_btn)

        # "&&" -- a single & is a Qt mnemonic and renders as "Calibrate _Save".
        apply_btn = QPushButton("Calibrate && Save Camera Profile", self)
        apply_btn.setMinimumHeight(34)
        apply_btn.setMinimumWidth(230)
        apply_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        apply_btn.setStyleSheet(theme.form_primary_button_stylesheet())
        apply_btn.clicked.connect(self._on_apply_calibration)
        ctrl_box.addWidget(apply_btn)

        layout.addLayout(ctrl_box)

        # Attempt initial auto-detection
        if self.image is not None:
            self.canvas.auto_detect()

    def _camera_caption(self) -> str:
        name = f"{self._exif_make} {self._exif_model}".strip()
        iso_str = f"  ·  ISO {self.iso}" if self.iso else ""
        if name:
            return f"Camera (EXIF): {name}{iso_str}"
        return "Camera (EXIF): unknown — DPReview will open a generic studio/cameras page."

    def _secondary_button_style(self) -> str:
        return theme.form_secondary_button_stylesheet()

    def _alert(self, title: str, body: str) -> None:
        try:
            from rawviewer_app.widgets import CustomWarningDialog
        except ModuleNotFoundError:
            from widgets import CustomWarningDialog
        CustomWarningDialog(
            self,
            title=title,
            message=title,
            informative_text=body,
        ).exec()

    def _build_source_bar(self) -> QWidget:
        box = QGroupBox("Calibration image")
        col = QVBoxLayout(box)
        col.setContentsMargins(10, 14, 10, 10)
        col.setSpacing(8)

        self._camera_lbl = QLabel(self._camera_caption())
        self._camera_lbl.setWordWrap(True)
        self._camera_lbl.setStyleSheet(f"color: {theme.INK}; font-size: 12px; font-weight: 500;")
        col.addWidget(self._camera_lbl)

        advise = QLabel(
            "Prefer a DPReview studio or sample image when you can: those scenes are "
            "standardized, and RAWviewer's calibration was developed against them as "
            "reference. Search on DPReview, download the file yourself, then choose it "
            "here. This does not replace the photo you were editing."
        )
        advise.setWordWrap(True)
        advise.setStyleSheet(f"color: {theme.INK_FAINT}; font-size: 11px;")
        col.addWidget(advise)

        row = QHBoxLayout()
        row.setSpacing(8)

        own_btn = QPushButton("Use my own file…", self)
        own_btn.setObjectName("calibrate_own_file_btn")
        own_btn.setMinimumHeight(32)
        own_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        own_btn.setStyleSheet(self._secondary_button_style())
        own_btn.clicked.connect(self._on_choose_own_file)
        row.addWidget(own_btn)

        dpr_btn = QPushButton("Find a DPReview sample", self)
        dpr_btn.setObjectName("calibrate_dpreview_btn")
        dpr_btn.setMinimumHeight(32)
        dpr_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        dpr_btn.setStyleSheet(self._secondary_button_style())
        dpr_btn.clicked.connect(self._on_open_dpreview)
        row.addWidget(dpr_btn)
        row.addStretch()
        col.addLayout(row)

        self._source_status = QLabel("")
        self._source_status.setWordWrap(True)
        self._source_status.setStyleSheet(f"color: {theme.INK_FAINT}; font-size: 11px;")
        col.addWidget(self._source_status)
        return box

    def _on_open_dpreview(self) -> None:
        url = dpreview_open_url(self._exif_make, self._exif_model)
        QDesktopServices.openUrl(QUrl(url))
        terms = f"{self._exif_make} {self._exif_model}".strip()
        if terms:
            self._source_status.setText(
                f"Opened DPReview search for {terms}. Download a studio/sample file, "
                "then choose it with Use my own file."
            )
        else:
            self._source_status.setText(
                "Opened DPReview's studio comparison. Pick a camera, download a "
                "sample, then choose it with Use my own file."
            )

    def _on_choose_own_file(self) -> None:
        try:
            from raw_file_extensions import get_supported_extensions

            exts = " ".join(f"*{e}" for e in get_supported_extensions())
        except Exception:
            exts = "*.nef *.cr2 *.cr3 *.arw *.dng *.raf *.rw2 *.orf *.jpg *.jpeg *.tif *.tiff"
        start = ""
        if self.source_file:
            start = os.path.dirname(self.source_file)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose a ColorChecker / studio-scene file",
            start,
            f"Images ({exts});;All Files (*)",
        )
        if not path:
            return
        self._load_calibration_file(path)

    def _load_calibration_file(self, path: str) -> None:
        try:
            from unified_image_processor import UnifiedImageProcessor

            img = UnifiedImageProcessor().decode_raw_edit_base(
                path, use_full_resolution=False
            )
        except Exception as exc:
            self._alert(
                "Could not open file",
                f"Failed to decode {os.path.basename(path)}:\n{exc}",
            )
            return
        if img is None or not hasattr(img, "shape"):
            self._alert(
                "Could not open file",
                f"Could not decode {os.path.basename(path)} as a calibration image.",
            )
            return

        make, model, iso_val = camera_identity_for_file(path)
        self.image = _to_uint8_rgb(img)
        self.source_file = path
        if make or model:
            self._exif_make = (make or "").strip()
            self._exif_model = (model or "").strip()
            self.make = self._exif_make or "Camera"
            self.model = self._exif_model or "Model"
            self.iso = iso_val
            iso_str = f" (ISO {self.iso})" if self.iso else ""
            self.setWindowTitle(
                f"Camera Color Calibration — {self.make} {self.model}{iso_str}"
            )
        self._camera_lbl.setText(self._camera_caption())
        self.canvas.set_image(self.image)
        self.canvas.auto_detect()
        self._source_status.setText(
            f"Using {os.path.basename(path)} as the calibration reference "
            "(the photo you were editing is unchanged)."
        )

    def _on_auto_detect(self) -> None:
        if not self.canvas.auto_detect():
            self._alert(
                "Auto-Detect Chart",
                "OpenCV could not auto-locate the ColorChecker chart. Please manually drag the "
                "4 corner handles over the chart.\n\n"
                "Auto-detection is most reliable on an evenly lit, square-on chart. "
                "Use “Find a DPReview sample” to search DPReview, download a studio/sample "
                "file yourself, then choose it with “Use my own file”.",
            )

    def _on_apply_calibration(self) -> None:
        img = self.canvas.image if getattr(self.canvas, "image", None) is not None else self.image
        if img is None or getattr(img, "size", 0) == 0:
            self._alert(
                "No calibration image",
                "Choose a ColorChecker / studio-scene file, or find a DPReview sample "
                "and upload it here.",
            )
            return
        corners = self.canvas.get_pixel_corners()
        valid, err_msg, sampled = validate_and_detect_color_checker(img, corners)
        
        if not valid or not sampled:
            self._alert(
                "Color Checker Target Not Detected",
                (err_msg or "No valid 24-patch ColorChecker chart was detected within the 4 corners.")
                + "\n\nA usable calibration frame needs the chart evenly lit, square-on to the "
                "camera, and free of glare — uneven lighting is indistinguishable from camera "
                "colour error and would be baked into the profile.\n\n"
                "Prefer a DPReview studio/sample image (standardized; this calibrator was "
                "developed against them). Search with “Find a DPReview sample”, download "
                "the file yourself, then choose it with “Use my own file”.",
            )
            return

        wb_mode = self.wb_combo.currentData() or "auto"
        profile = calibrate_camera_curves_and_hsl(sampled, wb_mode=wb_mode)
        save_camera_profile(
            self.make, self.model, profile, iso=self.iso,
            source_file=self.source_file,
        )
        self.calibrated_profile = profile

        self._alert(
            "Camera Calibration Complete",
            f"Successfully calibrated and saved color profile for {self.make} {self.model}.\n\n"
            "All future photos from this camera model will automatically receive this calibrated "
            "baseline color science without needing manual XMP or 3D LUT exports.",
        )
        self.accept()
