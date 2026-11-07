"""
Fast EXIF data extractor with caching.
"""
from __future__ import annotations

import contextlib
import os
import sys
from typing import Any

import rawpy
from PyQt6.QtCore import QObject
from PyQt6.QtGui import QImageReader

from common_image_loader import (
    normalize_capture_time_string,
)
from image_cache import get_image_cache
from image_load_manager import yield_if_current_task_active
from processing.raw_decode.gil import rawpy_imread_warm

from .extraction import (
    RAW_EXIF_SENSOR_META_VER,
    _detect_nef_he_compression,
    _orientation_from_embedded_preview,
    _rawpy_global_lock,
    cached_raw_exif_orientation_trustworthy,
)


class EXIFExtractor(QObject):
    """Fast EXIF data extractor with caching."""

    def __init__(self):
        super().__init__()
        self.cache = get_image_cache()

    def _persist_exif_result(self, file_path: str, result: dict[str, Any] | None) -> None:
        """Write freshly extracted metadata to memory + persistent EXIF cache."""
        if not file_path or not isinstance(result, dict) or not result:
            return
        with contextlib.suppress(Exception):
            self.cache.put_exif(file_path, result)

    def build_minimal_raw_exif(
        self, file_path: str, raw_object: rawpy.RawPy
    ) -> dict[str, Any]:
        """Fast sensor/orientation metadata from an open rawpy handle (no exiftool)."""
        orientation = 1
        original_width = 0
        original_height = 0
        try:
            sizes = raw_object.sizes
            original_width = int(sizes.width)
            original_height = int(sizes.height)
            if sizes.flip != 0:
                flip_map = {0: 1, 3: 3, 5: 8, 6: 6}
                orientation = flip_map.get(sizes.flip, sizes.flip)
        except Exception:
            pass
        if orientation <= 1:
            embedded_o = _orientation_from_embedded_preview(file_path, raw_object)
            if embedded_o not in (1, orientation):
                orientation = embedded_o
        return {
            "orientation": orientation,
            "camera_make": "",
            "camera_model": "",
            "exif_data": {},
            "original_width": original_width,
            "original_height": original_height,
            "capture_time": None,
            "focal_length": None,
            "aperture": None,
            "iso": None,
            "verified_orientation": True,
            "raw_exif_sensor_meta_ver": RAW_EXIF_SENSOR_META_VER,
            "minimal_preview_exif": True,
        }

    def extract_exif_data(self, file_path: str, raw_object: rawpy.RawPy | None = None) -> dict[str, Any] | None:
        """Extract EXIF data from image file with RAW-specific orientation fallbacks."""
        # Check SQLite cache first (verify=True: this is the canonical
        # orientation source for callers that rotate pixels).
        cached = self.cache.get_exif(file_path, verify=True)
        if cached and not cached.get("minimal_preview_exif"):
            cached_ver = cached.get('raw_exif_sensor_meta_ver', 0)
            if cached_ver < RAW_EXIF_SENSOR_META_VER:
                if os.environ.get("RAWVIEWER_VERBOSE_ORIENTATION_LOGS") == "1":
                    # print(f"[ORIENTATION] EXIFExtractor: Stale cache version ({cached_ver} < {RAW_EXIF_SENSOR_META_VER}) for {os.path.basename(file_path)}, forcing re-extraction...")
                    pass
            elif cached_raw_exif_orientation_trustworthy(file_path, cached):
                if os.environ.get("RAWVIEWER_VERBOSE_ORIENTATION_LOGS") == "1":
                    # print(f"[ORIENTATION] EXIFExtractor: Found valid cached orientation={cached.get('orientation')} for {os.path.basename(file_path)}")
                    pass
                return cached
            else:
                try:
                    self.cache.exif_cache.remove(file_path)
                    self.cache.exif_memory_cache.remove(file_path)
                except Exception:
                    pass

        try:
            from metadata import backend as metadata_backend

            tags = metadata_backend.process_file_from_path(file_path, details=False)
        except Exception as e:
            import logging

            from common_image_loader import is_emfile_error, note_emfile_pressure

            logging.getLogger(__name__).warning(f"metadata_backend failed on {file_path}: {e}")
            if is_emfile_error(e):
                note_emfile_pressure()
            tags = {}

        try:
            # Standard orientation tags
            orientation = 1
            orientation_tag_found = None

            # DEBUG: Log all potential orientation tags to help troubleshoot
            if os.environ.get("RAWVIEWER_VERBOSE_ORIENTATION_LOGS") == "1":
                orient_tags = {k: v for k, v in tags.items() if 'orient' in k.lower()}
                if orient_tags:
                    # print(f"[ORIENTATION] EXIFExtractor debug for {os.path.basename(file_path)}: Found orientation-like tags: {orient_tags}")
                    pass

            for tag_name in ('Image Orientation', 'EXIF Orientation', 'Orientation', 'MakerNote Orientation', 'EXIF SceneType', 'Sony Orientation', 'Sony Orientation 2'):
                tag = tags.get(tag_name)
                if tag:
                    # Try to get the numeric value directly
                    try:
                        if hasattr(tag, 'values') and tag.values:
                            val = tag.values[0]
                            if isinstance(val, int):
                                orientation = val
                                orientation_tag_found = tag_name
                                break
                    except Exception:
                        pass

                    # Fallback to string mapping (same as main.py)
                    orientation_str = str(tag).strip()
                    orientation_map = {
                        'Horizontal (normal)': 1,
                        'Mirrored horizontal': 2,
                        'Rotated 180': 3,
                        'Mirrored vertical': 4,
                        'Mirrored horizontal then rotated 90 CCW': 5,
                        'Rotated 90 CW': 6,
                        'Mirrored horizontal then rotated 90 CW': 7,
                        'Rotated 90 CCW': 8
                    }
                    if orientation_str in orientation_map:
                        orientation = orientation_map[orientation_str]
                        orientation_tag_found = tag_name
                        break

            if os.environ.get("RAWVIEWER_VERBOSE_ORIENTATION_LOGS") == "1" and orientation != 1:
                # print(f"[ORIENTATION] EXIFExtractor: Found orientation={orientation} via tag '{orientation_tag_found}' for {os.path.basename(file_path)}")
                pass

            camera_make = str(tags.get('Image Make', '')).strip()
            camera_model = str(tags.get('Image Model', '')).strip()

            exif_dict = {k: str(v) for k, v in tags.items()}

            # Dimensions
            original_width = 0
            original_height = 0

            # 1. Try DNG DefaultCropSize (often found in Apple ProRAW and Android DNGs)
            crop_size = tags.get('Image DefaultCropSize')
            if crop_size and hasattr(crop_size, 'values') and len(crop_size.values) >= 2:
                try:
                    original_width = int(crop_size.values[0])
                    original_height = int(crop_size.values[1])
                except Exception:
                    pass

            # 2. Try DNG ActiveArea (Top, Left, Bottom, Right)
            if original_width <= 0 or original_height <= 0:
                active_area = tags.get('Image ActiveArea')
                if active_area and hasattr(active_area, 'values') and len(active_area.values) >= 4:
                    try:
                        # Values: Top, Left, Bottom, Right
                        top = int(active_area.values[0])
                        left = int(active_area.values[1])
                        bottom = int(active_area.values[2])
                        right = int(active_area.values[3])
                        original_height = bottom - top
                        original_width = right - left
                    except Exception:
                        pass

            # 3. Fallback to standard EXIF width/height
            if original_width <= 0:
                for tag in ('EXIF ExifImageWidth', 'Image ImageWidth'):
                    if tag in tags:
                        try:
                            original_width = int(tags[tag].values[0])
                            break
                        except Exception:
                            pass

            if original_height <= 0:
                for tag in ('EXIF ExifImageLength', 'Image ImageLength'):
                    if tag in tags:
                        try:
                            original_height = int(tags[tag].values[0])
                            break
                        except Exception:
                            pass

            # Second pass: If it's a RAW file, use rawpy to verify dimensions and
            # orientation (flip), and -- while the file is already open -- also
            # resolve the embedded-preview orientation fallback in the SAME
            # rawpy.imread() call. These used to be two separate opens (one here,
            # one in the block below), and each rawpy.imread() on a RAW file is a
            # full LibRaw parse (tens of ms even for embedded-preview-only use) --
            # doubling that cost on every cold gallery thumbnail for no reason.
            import common_image_loader
            if common_image_loader.is_raw_file(file_path):
                try:
                    if raw_object is not None:
                        sizes = raw_object.sizes
                        original_width = sizes.width
                        original_height = sizes.height
                        if orientation == 1 and sizes.flip != 0:
                            # Map LibRaw flip codes to EXIF Orientation
                            # LibRaw flip: 0=0, 3=180, 5=90CCW, 6=90CW
                            # EXIF Orient: 1=0, 3=180, 8=90CCW, 6=90CW
                            flip_map = {0: 1, 3: 3, 5: 8, 6: 6}
                            orientation = flip_map.get(sizes.flip, sizes.flip)
                        if orientation <= 1:
                            embedded_o = _orientation_from_embedded_preview(file_path, raw_object)
                            if embedded_o not in (1, orientation):
                                orientation = embedded_o
                    else:
                        yield_if_current_task_active()
                        with _rawpy_global_lock, rawpy_imread_warm(file_path) as raw:
                            sizes = raw.sizes
                            original_width = sizes.width
                            original_height = sizes.height
                            if orientation == 1 and sizes.flip != 0:
                                flip_map = {0: 1, 3: 3, 5: 8, 6: 6}
                                orientation = flip_map.get(sizes.flip, sizes.flip)
                            if orientation <= 1:
                                embedded_o = _orientation_from_embedded_preview(file_path, raw)
                                if embedded_o not in (1, orientation):
                                    orientation = embedded_o
                except Exception:
                    pass

            # Third pass: If dimensions are still 0 (e.g. non-RAW missing tags), use QImageReader (fast header read)
            if original_width <= 0 or original_height <= 0:
                try:
                    reader = QImageReader(file_path)
                    size = reader.size()
                    if size.isValid():
                        # QImageReader.size() takes autoTransform into account if set.
                        # Since we want to provide dimensions that will be combined with
                        # 'orientation' later, we should be careful.
                        # But standard JPEGs with missing tags usually have orientation=1.
                        original_width = size.width()
                        original_height = size.height()
                except Exception:
                    pass

            # Fourth pass: If on macOS and missing dimensions, fallback to native sips
            if (original_width <= 0 or original_height <= 0) and sys.platform == 'darwin':
                try:
                    import subprocess
                    result = subprocess.run(['sips', '-g', 'pixelWidth', '-g', 'pixelHeight', '-g', 'orientation', file_path],
                                          capture_output=True, text=True)
                    for line in result.stdout.splitlines():
                        if 'pixelWidth:' in line and original_width <= 0:
                            original_width = int(line.split(':')[1].strip())
                        elif 'pixelHeight:' in line and original_height <= 0:
                            original_height = int(line.split(':')[1].strip())
                        elif 'orientation:' in line and orientation == 1:
                            val = line.split(':')[1].strip()
                            # sips returns e.g. "1 (Normal)", "6 (Rotated 90 CW)", "8 (Rotated 90 CCW)"
                            import re
                            m = re.match(r'^(\d+)', val)
                            if m:
                                orientation = int(m.group(1))
                    if os.environ.get("RAWVIEWER_VERBOSE_ORIENTATION_LOGS") == "1":
                        print(f"[EXIF] Extracted via sips: {original_width}x{original_height}, orientation={orientation} for {os.path.basename(file_path)}")
                except Exception:
                    pass

            # Extract technical metadata for top-level cache columns
            focal_length = None
            aperture = None
            iso = None

            # Focal Length
            for tag in ('EXIF FocalLength', 'EXIF FocalLengthIn35mmFilm'):
                if tag in exif_dict:
                    try:
                        val = exif_dict[tag]
                        if '/' in val:
                            num, den = val.split('/')
                            focal_length = f"{round(float(num) / float(den))}mm"
                        else:
                            focal_length = f"{round(float(val))}mm"
                        break
                    except Exception:
                        pass

            # Aperture
            for tag in ('EXIF FNumber', 'EXIF ApertureValue'):
                if tag in exif_dict:
                    try:
                        val = exif_dict[tag]
                        if '/' in val:
                            num, den = val.split('/')
                            aperture = f"f/{float(num) / float(den):.1f}"
                        else:
                            aperture = f"f/{float(val):.1f}"
                        break
                    except Exception:
                        pass

            # ISO
            for tag in ('EXIF ISOSpeedRatings', 'EXIF ISO', 'EXIF PhotographicSensitivity'):
                if tag in exif_dict:
                    try:
                        val = exif_dict[tag]
                        if '/' in val:
                            num, den = val.split('/')
                            iso = f"ISO {int(float(num) / float(den))}"
                        else:
                            iso = f"ISO {val}"
                        break
                    except Exception:
                        pass

            rating = 0
            for tag in ('XMP-xmp:Rating', 'XMP Rating', 'Rating', 'EXIF Rating'):
                if tag in tags:
                    try:
                        from raw_adjustments import _parse_rating_value

                        rating = _parse_rating_value(tags[tag])
                        if rating:
                            break
                    except Exception:
                        pass
                if tag in exif_dict:
                    try:
                        from raw_adjustments import _parse_rating_value

                        rating = _parse_rating_value(exif_dict[tag])
                        if rating:
                            break
                    except Exception:
                        pass
            if not rating:
                try:
                    from raw_adjustments import load_rating_for_file

                    rating = load_rating_for_file(file_path)
                except Exception:
                    pass

            # Build return dict
            result = {
                'orientation': orientation,
                'camera_make': camera_make,
                'camera_model': camera_model,
                'exif_data': exif_dict,
                'original_width': original_width,
                'original_height': original_height,
                'capture_time': (
                    normalize_capture_time_string(
                        exif_dict.get('EXIF DateTimeOriginal') or exif_dict.get('Image DateTime')
                    )
                    or exif_dict.get('EXIF DateTimeOriginal')
                    or exif_dict.get('Image DateTime')
                ),
                'focal_length': focal_length,
                'aperture': aperture,
                'iso': iso,
                'rating': rating,
                'verified_orientation': True,
                'raw_exif_sensor_meta_ver': RAW_EXIF_SENSOR_META_VER,
            }

            if file_path.lower().endswith(".nef"):
                # Proactive Nikon HE/HE* detection (see _detect_nef_he_compression):
                # lets callers route straight to the embedded-JPEG fallback instead
                # of paying for one failed LibRaw decode attempt per file.
                result['nef_he_compressed'] = _detect_nef_he_compression(file_path)

            if os.environ.get("RAWVIEWER_VERBOSE_ORIENTATION_LOGS") == "1":
                # print(f"[ORIENTATION] EXIFExtractor: Successfully returning metadata with orientation={orientation} for {os.path.basename(file_path)}")
                pass

            self._persist_exif_result(file_path, result)
            return result

        except Exception:
            if os.environ.get("RAWVIEWER_VERBOSE_ORIENTATION_LOGS") == "1":
                # print(f"[ORIENTATION] EXIFExtractor error for {os.path.basename(file_path)}: {e}")
                pass
            pass

        return {
            'orientation': orientation if 'orientation' in locals() else 1,
            'camera_make': camera_make if 'camera_make' in locals() else '',
            'camera_model': camera_model if 'camera_model' in locals() else '',
            'exif_data': {},
            'verified_orientation': True
        }

