"""Interactive HDR and Panorama tuning dialog for RAWviewer."""

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

from PyQt6.QtCore import Qt, Signal
from PyQt6.QtGui import QColor, QCursor, QFont, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

import rawviewer_ui.theme as theme


class HDRPanoramaDialog(QDialog):
    """Dialog for configuring HDR weights, Panorama auto-crop, and image selection."""

    def __init__(
        self,
        image_paths: List[str],
        mode: str = "hdr",  # "panorama", "hdr", "hdr_panorama"
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.image_paths = list(image_paths)
        self.mode = mode.lower()
        self._block_signals = False

        title_map = {
            "panorama": "Standard Panorama Stitching",
            "hdr": "HDR Exposure Fusion & Weight Tuning",
            "hdr_panorama": "Panorama HDR Merge",
        }
        self.setWindowTitle(title_map.get(self.mode, "Stitch & Merge"))
        self.setMinimumSize(540, 480)
        self.setStyleSheet(
            f"""
            QDialog {{
                background-color: {theme.DARK_BG};
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
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }}
            QSlider::groove:horizontal {{
                height: 4px;
                background: {theme.SURFACE};
                border-radius: 2px;
            }}
            QSlider::sub-page:horizontal {{
                background: {theme.EMBER};
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {theme.INK};
                border: 1px solid {theme.LINE};
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }}
            """
        )

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Header Title
        header_lbl = QLabel(self.windowTitle())
        header_lbl.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        header_lbl.setStyleSheet(f"color: {theme.INK};")
        layout.addWidget(header_lbl)

        n = len(self.image_paths)

        # 1. HDR Weight Tuning Section (for HDR and HDR Panorama)
        if self.mode in ("hdr", "hdr_panorama"):
            hdr_box = QGroupBox("HDR Exposure Weights", self)
            hdr_layout = QVBoxLayout(hdr_box)
            hdr_layout.setSpacing(8)

            # Highlight Weight
            hl_row = QHBoxLayout()
            hl_row.addWidget(QLabel("Highlight Weight (Underexposed shots):"))
            self.hl_slider = QSlider(Qt.Orientation.Horizontal, self)
            self.hl_slider.setRange(0, 20)
            self.hl_slider.setValue(10)
            self.hl_val_lbl = QLabel("1.0x")
            self.hl_slider.valueChanged.connect(
                lambda v: self.hl_val_lbl.setText(f"{v/10.0:.1f}x")
            )
            hl_row.addWidget(self.hl_slider, 1)
            hl_row.addWidget(self.hl_val_lbl)
            hdr_layout.addLayout(hl_row)

            # Shadow Weight
            sh_row = QHBoxLayout()
            sh_row.addWidget(QLabel("Shadow Weight (Overexposed shots):"))
            self.sh_slider = QSlider(Qt.Orientation.Horizontal, self)
            self.sh_slider.setRange(0, 20)
            self.sh_slider.setValue(10)
            self.sh_val_lbl = QLabel("1.0x")
            self.sh_slider.valueChanged.connect(
                lambda v: self.sh_val_lbl.setText(f"{v/10.0:.1f}x")
            )
            sh_row.addWidget(self.sh_slider, 1)
            sh_row.addWidget(self.sh_val_lbl)
            hdr_layout.addLayout(sh_row)

            # Midtone Weight
            mid_row = QHBoxLayout()
            mid_row.addWidget(QLabel("Midtone Weight:"))
            self.mid_slider = QSlider(Qt.Orientation.Horizontal, self)
            self.mid_slider.setRange(0, 20)
            self.mid_slider.setValue(10)
            self.mid_val_lbl = QLabel("1.0x")
            self.mid_slider.valueChanged.connect(
                lambda v: self.mid_val_lbl.setText(f"{v/10.0:.1f}x")
            )
            mid_row.addWidget(self.mid_slider, 1)
            mid_row.addWidget(self.mid_val_lbl)
            hdr_layout.addLayout(mid_row)

            layout.addWidget(hdr_box)

        # 2. Selected Images List
        img_box = QGroupBox(f"Selected Photos ({n})", self)
        img_layout = QVBoxLayout(img_box)

        self.img_list = QListWidget(self)
        self.img_list.setStyleSheet(
            f"""
            QListWidget {{
                background-color: {theme.SURFACE};
                border: 1px solid {theme.LINE};
                border-radius: 4px;
            }}
            QListWidget::item {{
                padding: 4px;
                border-bottom: 1px solid {theme.LINE};
            }}
            """
        )
        for path in self.image_paths:
            fname = os.path.basename(path)
            item = QListWidgetItem(fname, self.img_list)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)

        img_layout.addWidget(self.img_list)
        layout.addWidget(img_box, 1)

        # 3. Stitching Options
        opt_box = QVBoxLayout()
        self.crop_cb = QCheckBox("Auto-crop warped black borders (Maintain rectangular frame)", self)
        self.crop_cb.setChecked(True)
        self.crop_cb.setStyleSheet(f"color: {theme.INK}; font-size: 12px;")
        opt_box.addWidget(self.crop_cb)

        self.align_cb = QCheckBox("Auto-align handheld exposures (MTB)", self)
        self.align_cb.setChecked(True)
        self.align_cb.setStyleSheet(f"color: {theme.INK}; font-size: 12px;")
        opt_box.addWidget(self.align_cb)

        layout.addLayout(opt_box)

        # 4. Dialog Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel", self)
        cancel_btn.setMinimumHeight(36)
        cancel_btn.setMinimumWidth(90)
        cancel_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        cancel_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {theme.SURFACE};
                color: {theme.INK};
                border: 1px solid {theme.LINE};
                border-radius: 6px;
                font-weight: 500;
            }}
            """
        )
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        self.merge_btn = QPushButton("Merge & Stitch", self)
        self.merge_btn.setMinimumHeight(36)
        self.merge_btn.setMinimumWidth(120)
        self.merge_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.merge_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {theme.EMBER};
                color: #FFFFFF;
                border: none;
                border-radius: 6px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {theme.EMBER_BRIGHT};
            }}
            """
        )
        self.merge_btn.clicked.connect(self.accept)
        btn_row.addWidget(self.merge_btn)

        layout.addLayout(btn_row)

    def should_auto_crop(self) -> bool:
        """Returns True if auto-cropping warped borders is enabled."""
        return self.crop_cb.isChecked()

    def get_hdr_weights(self) -> Dict[str, float]:
        """Returns dict of highlight, shadow, midtone weight multipliers."""
        if self.mode not in ("hdr", "hdr_panorama"):
            return {"highlight": 1.0, "shadow": 1.0, "midtone": 1.0}
        return {
            "highlight": float(self.hl_slider.value()) / 10.0,
            "shadow": float(self.sh_slider.value()) / 10.0,
            "midtone": float(self.mid_slider.value()) / 10.0,
        }

    def get_active_image_paths(self) -> List[str]:
        """Returns list of checked image paths."""
        active = []
        for i in range(self.img_list.count()):
            item = self.img_list.item(i)
            if item and item.checkState() == Qt.CheckState.Checked:
                active.append(self.image_paths[i])
        return active
