"""Interactive ColorChecker target alignment & camera calibration dialog for RAWviewer."""

from __future__ import annotations

import logging
import os
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QFont, QImage, QPainter, QPen, QBrush, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    import rawviewer_ui.theme as theme
except ModuleNotFoundError:
    import theme as theme
from color_calibration import (
    extract_patch_colors,
    validate_and_detect_color_checker,
    calibrate_camera_curves_and_hsl,
    save_camera_profile,
)

logger = logging.getLogger(__name__)


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

    def _numpy_to_pixmap(self, img: np.ndarray) -> QPixmap:
        if img is None or img.size == 0:
            return QPixmap()
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
        try:
            import cv2.mcc as mcc
            detector = mcc.CCheckerDetector.create()
            img_bgr = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR) if self.image.ndim == 3 else self.image
            success = detector.process(img_bgr, mcc.MCC24)
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
        if not self._pixmap.isNull():
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
        image: np.ndarray,
        make: str,
        model: str,
        iso: Optional[int] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.image = image
        self.make = make or "Camera"
        self.model = model or "Model"
        self.iso = iso
        self.calibrated_profile: Optional[Dict[str, Any]] = None

        iso_str = f" (ISO {iso})" if iso else ""
        self.setWindowTitle(f"Camera Color Calibration — {self.make} {self.model}{iso_str}")
        self.setMinimumSize(820, 600)
        self.setStyleSheet(
            f"""
            QDialog {{
                background-color: {theme.SURFACE};
                color: {theme.INK};
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }}
            QLabel {{
                color: {theme.INK};
                font-size: 12px;
            }}
            QGroupBox {{
                font-weight: 600;
                font-size: 12px;
                color: {theme.EMBER};
                border: 1px solid {theme.LINE};
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 12px;
            }}
            QComboBox {{
                background-color: {theme.SURFACE};
                color: {theme.INK};
                border: 1px solid {theme.LINE};
                border-radius: 4px;
                padding: 4px 8px;
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
        header_lbl.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        header_lbl.setStyleSheet(f"color: {theme.INK};")
        layout.addWidget(header_lbl)

        sub_lbl = QLabel(
            "Drag the 4 corner handles over your 24-patch ColorChecker chart, or click Auto-Detect."
        )
        sub_lbl.setStyleSheet(f"color: {theme.INK_FAINT}; font-size: 11px;")
        layout.addWidget(sub_lbl)

        # Canvas Widget
        self.canvas = ColorCheckerPreviewCanvas(self.image, self)
        layout.addWidget(self.canvas, 1)

        # Controls Row
        ctrl_box = QHBoxLayout()
        
        auto_btn = QPushButton("🎯 Auto-Detect Chart", self)
        auto_btn.setMinimumHeight(34)
        auto_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        auto_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {theme.SURFACE};
                border: 1px solid {theme.LINE};
                border-radius: 6px;
                color: {theme.INK};
                padding: 4px 12px;
                font-weight: 500;
            }}
            """
        )
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
        cancel_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {theme.SURFACE};
                border: 1px solid {theme.LINE};
                border-radius: 6px;
                color: {theme.INK};
            }}
            """
        )
        cancel_btn.clicked.connect(self.reject)
        ctrl_box.addWidget(cancel_btn)

        apply_btn = QPushButton("Calibrate & Save Camera Profile", self)
        apply_btn.setMinimumHeight(34)
        apply_btn.setMinimumWidth(180)
        apply_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        apply_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {theme.EMBER};
                border: none;
                border-radius: 6px;
                color: #FFFFFF;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {theme.EMBER_DIM};
            }}
            """
        )
        apply_btn.clicked.connect(self._on_apply_calibration)
        ctrl_box.addWidget(apply_btn)

        layout.addLayout(ctrl_box)

        # Attempt initial auto-detection
        self.canvas.auto_detect()

    def _on_auto_detect(self) -> None:
        if not self.canvas.auto_detect():
            QMessageBox.information(
                self,
                "Auto-Detect Chart",
                "OpenCV could not auto-locate the ColorChecker chart. Please manually drag the 4 corner handles over the chart.",
            )

    def _on_apply_calibration(self) -> None:
        corners = self.canvas.get_pixel_corners()
        valid, err_msg, sampled = validate_and_detect_color_checker(self.image, corners)
        
        if not valid or not sampled:
            QMessageBox.warning(
                self,
                "Color Checker Target Not Detected",
                err_msg or "No valid 24-patch ColorChecker chart was detected within the 4 corners.",
            )
            return

        wb_mode = self.wb_combo.currentData() or "auto"
        profile = calibrate_camera_curves_and_hsl(sampled, wb_mode=wb_mode)
        save_camera_profile(self.make, self.model, profile, iso=self.iso)
        self.calibrated_profile = profile

        QMessageBox.information(
            self,
            "Camera Calibration Complete",
            f"Successfully calibrated and saved color profile for {self.make} {self.model}!\n\n"
            f"All future photos from this camera model will automatically receive this calibrated baseline color science without needing manual XMP or 3D LUT exports.",
        )
        self.accept()
