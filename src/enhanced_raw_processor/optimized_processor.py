"""
Optimized RAW processor with format-specific optimizations.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import rawpy
from PyQt6.QtCore import QObject

from image_load_manager import yield_if_current_task_active
from processing.raw_decode.gil import rawpy_imread_warm

from .extraction import (
    _heavy_fallback_semaphore,
    _rawpy_global_lock,
)


class OptimizedRAWProcessor(QObject):
    """Optimized RAW processor with format-specific optimizations."""

    def __init__(self):
        super().__init__()

    def is_canon_camera(self, file_path: str, exif_data: dict[str, Any] = None) -> bool:
        """Check if this is a Canon camera."""
        # Check file extension first (faster)
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in ['.cr2', '.cr3']:
            return True

        # Check EXIF if available
        if exif_data and exif_data.get('camera_make'):
            return 'CANON' in exif_data['camera_make'].upper()

        return False

    def is_fujifilm_camera(self, file_path: str, exif_data: dict[str, Any] = None) -> bool:
        """Check if this is a Fujifilm camera."""
        # Check file extension first (faster)
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in ['.raf']:
            return True

        # Check EXIF if available
        if exif_data and exif_data.get('camera_make'):
            make = exif_data['camera_make'].upper()
            return 'FUJIFILM' in make or 'FUJI' in make

        return False

    def is_sony_camera(self, file_path: str, exif_data: dict[str, Any] = None) -> bool:
        """Check if this is a Sony camera."""
        # Check file extension first (faster)
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in ['.arw']:
            return True

        # Check EXIF if available
        if exif_data and exif_data.get('camera_make'):
            return 'SONY' in exif_data['camera_make'].upper()

        return False

    def get_optimized_processing_params(self, file_path: str, exif_data: dict[str, Any] = None) -> dict[str, Any]:
        """Get optimized processing parameters based on camera and file."""
        # Default fast processing parameters
        params = {
            'use_camera_wb': True,
            'use_auto_wb': False,
            'output_bps': 8,  # 8-bit for speed
            'no_auto_bright': True,
            'gamma': (2.222, 4.5),  # Standard sRGB gamma
            'bright': 1.0,
            # 'highlight': 0,  # Removed for rawpy 0.25.0 compatibility
            # 'shadow': 0,  # Removed for rawpy 0.25.0 compatibility
            'user_flip': 0,  # Force rawpy to ignore EXIF orientation
            'demosaic_algorithm': rawpy.DemosaicAlgorithm.LINEAR, # MUCH faster than default for fast previews
        }

        # Get file size for processing decisions
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        except Exception:
            file_size_mb = 50  # Default assumption

        # Adjust for large files (use faster processing)
        if file_size_mb > 80:
            params.update({
                'half_size': True,  # Process at half resolution for speed
                'use_camera_wb': True,
                'no_auto_bright': True
            })

        # Camera-specific optimizations
        if self.is_canon_camera(file_path, exif_data):
            # Canon cameras benefit from camera white balance
            params.update({
                'use_camera_wb': True,
                'use_auto_wb': False
            })
        elif self.is_fujifilm_camera(file_path, exif_data):
            # Fujifilm X-Trans sensors
            # DCB is high quality but slower - disable for fast processing
            params.update({
                'use_camera_wb': True,
                'dcb_enhance': not file_size_mb > 20
            })
        elif self.is_sony_camera(file_path, exif_data):
            # Sony cameras
            params.update({
                'use_camera_wb': True,
                'use_auto_wb': False
            })

        return params

    def process_raw_fast(self, file_path: str, exif_data: dict[str, Any] = None) -> np.ndarray | None:
        """Process RAW file with optimized parameters for speed."""
        try:
            yield_if_current_task_active()
            with _rawpy_global_lock:
                raw_ctx = rawpy_imread_warm(file_path)
            with raw_ctx as raw:
                params = self.get_optimized_processing_params(file_path, exif_data)
                with _heavy_fallback_semaphore:
                    rgb_image = raw.postprocess(**params)

                    return rgb_image

        except Exception:
            # Log the actual error for debugging
            # print(f"RAW processing error in process_raw_fast: {str(e)}")
            pass
            return None

    def process_raw_quality(self, file_path: str, exif_data: dict[str, Any] = None) -> np.ndarray | None:
        """Process RAW file with quality parameters (slower)."""
        try:
            yield_if_current_task_active()
            with _rawpy_global_lock:
                raw_ctx = rawpy_imread_warm(file_path)
            with raw_ctx as raw:
                # Quality processing parameters
                params = {
                    'use_camera_wb': True,
                    'output_bps': 16,  # 16-bit for quality
                    'gamma': (1, 1),   # Linear gamma for better quality
                    'no_auto_bright': True,
                    'bright': 1.0
                }

                # Camera-specific quality adjustments
                if self.is_canon_camera(file_path, exif_data):
                    params.update({
                        'use_camera_wb': True,
                        # 'highlight': 1,  # Removed for rawpy 0.25.0 compatibility
                    })
                elif self.is_fujifilm_camera(file_path, exif_data):
                    params.update({
                        'dcb_enhance': True,
                        'fbdd_noise_reduction': rawpy.FBDDNoiseReductionMode.Full
                    })

                with _heavy_fallback_semaphore:
                    rgb_image = raw.postprocess(**params)

                # Convert 16-bit to 8-bit for display
                if rgb_image is not None and hasattr(rgb_image, 'dtype') and rgb_image.dtype == np.uint16:
                    rgb_image = (rgb_image / 256).astype(np.uint8)

                return rgb_image

        except Exception:
            # Log the actual error for debugging
            # print(f"RAW processing error in process_raw_quality: {str(e)}")
            pass
            return None
