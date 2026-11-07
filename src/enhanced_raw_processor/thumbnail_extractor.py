"""
Fast thumbnail extractor for immediate display.
"""
from __future__ import annotations

import contextlib
import os
import sys

import numpy as np
import rawpy
from PyQt6.QtCore import QObject, QSize, Qt
from PyQt6.QtGui import QImage, QImageReader

from common_image_loader import (
    decode_embedded_jpeg_bytes,
    is_slow_storage_path,
)
from image_load_manager import yield_if_current_task_active
from processing.raw_decode.gil import rawpy_imread_warm

from .extraction import (
    _BYTESCAN_FIRST_EXTS,
    _bytescan_first_enabled,
    _rawpy_global_lock,
    _thumbnail_via_qimage_reader,
    extract_embedded_jpeg_by_scan,
)


class ThumbnailExtractor(QObject):
    """Fast thumbnail extractor for immediate display."""

    def __init__(self):
        super().__init__()

    def extract_thumbnail_from_raw(self, file_path: str, max_size: int = 1024,
                                   allow_scan_fallback: bool = True,
                                   raw_object: rawpy.RawPy | None = None) -> np.ndarray | None:
        """Extract embedded thumbnail from RAW file and resize to max_size."""
        thumb = None

        # Gallery / display-tier extraction for verified formats: try the
        # lock-free byte-scan (TIFF-parse) FIRST so we skip _rawpy_global_lock.
        # That lock serializes ALL RAW thumbnail decodes app-wide, so under
        # gallery concurrency it caps throughput regardless of drive —
        # measured ~7.8 vs ~29 tiles/sec on a slow external drive, and it also
        # lets one slow-format file stall the whole queue.
        # Cap matches memory_preview_max_edge so single-view ~1920–2304 upgrades
        # also skip LibRaw (previously only gallery <=1024 used bytescan-first).
        bytescan_cap = 1024
        try:
            from image_cache import memory_preview_max_edge

            bytescan_cap = max(1024, int(memory_preview_max_edge()))
        except Exception:
            bytescan_cap = 2304
        if (
            raw_object is None
            and 0 < max_size <= bytescan_cap
            and _bytescan_first_enabled()
            and os.path.splitext(file_path)[1].lower() in _BYTESCAN_FIRST_EXTS
        ):
            try:
                scanned = extract_embedded_jpeg_by_scan(file_path, max_size)
            except Exception:
                scanned = None
            if scanned is not None:
                return scanned

        # Prefer LibRaw's embedded JPEG segment (keeps preview EXIF orientation). Byte-scan
        # previews are faster but often lack orientation tags (Canon CR2/CR3).
        try:
            if raw_object is not None:
                thumb = self._extract_from_raw_obj(raw_object, file_path, max_size)
            else:
                yield_if_current_task_active()
                with _rawpy_global_lock:  # noqa: SIM117  # Intentional: nested with to avoid deadlock (see comment above)
                    # NEVER call yield_if_current_task_active() while holding
                    # _rawpy_global_lock: it sleep-loops until no CURRENT task is
                    # active, so a background (non-CURRENT) worker would sleep here
                    # forever holding the global lock, blocking every CURRENT gallery
                    # tile / single-view decode -> total decode-pipeline deadlock.
                    # The outer yield above (before the lock) is the safe place to
                    # defer background work to the foreground.
                    with rawpy_imread_warm(file_path) as raw:
                        thumb = self._extract_from_raw_obj(raw, file_path, max_size)
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(
                "raw.extract_thumb failed for %s: %s",
                os.path.basename(file_path),
                e,
            )

        if thumb is not None:
            import logging
            logging.getLogger(__name__).debug(
                "Embedded thumbnail via rawpy.extract_thumb, shape=%s",
                getattr(thumb, "shape", thumb),
            )
            return thumb

        # If rawpy fails entirely (e.g. unsupported DNG) or returns None, use non-LibRaw fallbacks.
        is_slow = is_slow_storage_path(file_path)
        if thumb is None and allow_scan_fallback and not is_slow:
            yield_if_current_task_active()
            scan_max = max_size if max_size > 0 else 8192
            thumb = extract_embedded_jpeg_by_scan(file_path, scan_max)
            if thumb is not None:
                return thumb
        if thumb is None and allow_scan_fallback and not is_slow:
            yield_if_current_task_active()
            # Fallback 2: OS decode via Qt (handles DNGs without LibRaw JPEG previews, or weird TIFFs)
            thumb = _thumbnail_via_qimage_reader(file_path, max_size, auto_transform=False)
        if thumb is None and allow_scan_fallback and sys.platform == "darwin":
                try:
                    import subprocess
                    import tempfile

                    from PIL import Image
                    tmp_name = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                            tmp_name = tmp.name

                        result = subprocess.run(['sips', '-Z', str(max_size), '-s', 'format', 'jpeg', file_path, '--out', tmp_name],
                                     capture_output=True)

                        if os.path.exists(tmp_name) and os.path.getsize(tmp_name) > 0:
                            jpeg_image = Image.open(tmp_name)
                            if jpeg_image.mode != 'RGB':
                                jpeg_image = jpeg_image.convert('RGB')
                            thumb = np.array(jpeg_image)
                            jpeg_image.close()
                        else:
                            import logging
                            logging.getLogger(__name__).debug(f"[SIPS] Failed to generate thumbnail for {file_path}. Exit code: {result.returncode}, Error: {result.stderr.decode('utf-8', 'ignore')}")

                            # Fallback to QuickLook (qlmanage)
                            try:
                                ql_dir = tempfile.mkdtemp()
                                subprocess.run(['qlmanage', '-t', '-s', str(max_size), '-o', ql_dir, file_path], capture_output=True)

                                ql_out = None
                                for f in os.listdir(ql_dir):
                                    if f.endswith('.png'):
                                        ql_out = os.path.join(ql_dir, f)
                                        break

                                if ql_out and os.path.exists(ql_out) and os.path.getsize(ql_out) > 0:
                                    ql_image = Image.open(ql_out)
                                    if ql_image.mode != 'RGB':
                                        ql_image = ql_image.convert('RGB')
                                    thumb = np.array(ql_image)
                                    ql_image.close()
                            except Exception as ql_e:
                                logging.getLogger(__name__).warning(f"[QLMANAGE] Fallback failed: {ql_e}")
                            finally:
                                if 'ql_dir' in locals() and os.path.exists(ql_dir):
                                    import shutil
                                    shutil.rmtree(ql_dir, ignore_errors=True)
                    finally:
                        if tmp_name and os.path.exists(tmp_name):
                            with contextlib.suppress(Exception):
                                os.remove(tmp_name)
                except Exception:
                    pass
        return thumb

    def _orient_thumb_array_by_flip(self, arr, raw):
        """Orient a decoded thumb to display orientation using LibRaw's own flip.

        Single orientation step for BOTH JPEG and BITMAP thumbs. raw.sizes.flip is the
        authoritative, always-fresh orientation (no stale-cache dependency), and the
        make-independent guard in apply_container_orientation_to_array makes this a no-op
        when the embedded preview is already display-oriented (so pre-rotated cameras are
        never double-rotated). This is what keeps the LibRaw path consistent with the
        byte-scan path, which orients via finalize_index_thumbnail_array.
        """
        if arr is None or not hasattr(arr, "shape"):
            return arr
        flip = int(getattr(raw.sizes, "flip", 0) or 0)
        if flip == 0:
            return arr
        flip_map = {0: 1, 3: 3, 5: 8, 6: 6}
        orientation = int(flip_map.get(flip, flip))
        from common_image_loader import apply_container_orientation_to_array

        oriented = apply_container_orientation_to_array(arr, orientation)
        return oriented if oriented is not None else arr

    def _extract_from_raw_obj(self, raw, file_path, max_size):
        """Internal helper to extract thumb from an open rawpy object."""
        try:
            thumb = raw.extract_thumb()

            if thumb.format == rawpy.ThumbFormat.JPEG:
                arr = decode_embedded_jpeg_bytes(thumb.data, max_size)
                return self._orient_thumb_array_by_flip(arr, raw)

            elif thumb.format == rawpy.ThumbFormat.BITMAP:
                thumb_array = thumb.data.copy()
                if thumb_array is None or not hasattr(thumb_array, 'shape'):
                    return None

                thumb_array = self._orient_thumb_array_by_flip(thumb_array, raw)

                h, w = thumb_array.shape[:2]
                if max_size > 0 and (w > max_size or h > max_size):
                     from PIL import Image
                     pil_thumb = Image.fromarray(thumb_array)
                     pil_thumb.thumbnail((max_size, max_size), Image.Resampling.HAMMING)
                     return np.array(pil_thumb)
                return thumb_array

            return None
        except Exception:
            return None

    def extract_preview_from_raw(self, file_path: str, max_size: int = 2048,
                                 allow_scan_fallback: bool = True) -> np.ndarray | None:
        """Extract high-quality preview from RAW file (embedded JPEG)."""
        # Reuse thumbnail logic but with larger size
        return self.extract_thumbnail_from_raw(
            file_path,
            max_size=max_size,
            allow_scan_fallback=allow_scan_fallback,
        )

    def extract_embedded_native_preview(
        self, file_path: str, allow_scan_fallback: bool = True
    ) -> np.ndarray | None:
        """Extract embedded JPEG at native resolution (max_size=0 disables downscaling)."""
        return self.extract_thumbnail_from_raw(
            file_path,
            max_size=0,
            allow_scan_fallback=allow_scan_fallback,
        )

    def extract_thumbnail_from_image(self, file_path: str, max_size: int = 1024,
                                     target_size: QSize | None = None) -> np.ndarray | QImage | None:
        """Extract thumbnail from regular image file. Returns QImage (preferred) or np.ndarray."""
        try:
            reader = QImageReader(file_path)
            reader.setAutoTransform(True)
            size = reader.size()
            if size.isValid() and size.width() > 0 and size.height() > 0:
                w, h = size.width(), size.height()

                # OPTIMIZATION: If we have a target_size, scale directly to it during decode.
                # Use KeepAspectRatioByExpanding so we have enough pixels to crop later
                # without stretching (which happens if we just set target_size directly).
                if target_size is not None and isinstance(target_size, QSize) and target_size.isValid():
                    # HIGH FIDELITY: Load at 2x the target size (oversampling) for sharper downscaling
                    load_w, load_h = target_size.width() * 2, target_size.height() * 2
                    if load_w < w and load_h < h:
                        reader.setScaledSize(size.scaled(QSize(load_w, load_h), Qt.AspectRatioMode.KeepAspectRatioByExpanding))
                    else:
                        reader.setScaledSize(size.scaled(target_size, Qt.AspectRatioMode.KeepAspectRatioByExpanding))
                elif w > max_size or h > max_size:
                    scale = min(max_size / w, max_size / h)
                    # For standard gallery thumbnails, load at 1.5x for better quality/speed balance
                    load_max = int(max_size * 1.5)
                    if w > load_max or h > load_max:
                        new_w = max(1, int(w * scale * 1.5))
                        new_h = max(1, int(h * scale * 1.5))
                        reader.setScaledSize(QSize(new_w, new_h))
                    else:
                        reader.setScaledSize(QSize(max(1, int(w * scale)), max(1, int(h * scale))))

            image = reader.read()
            if not image.isNull():
                if image.format() != QImage.Format.Format_RGB888:
                    image = image.convertToFormat(QImage.Format.Format_RGB888)
                # Perform high-quality final scaling if the loaded image is still larger than needed
                if target_size is not None and isinstance(target_size, QSize) and target_size.isValid():
                    if image.size().width() > target_size.width() or image.size().height() > target_size.height():
                        image = image.scaled(target_size, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
                elif image.width() > max_size or image.height() > max_size:
                    image = image.scaled(max_size, max_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

                return image
        except Exception:
            pass

        try:
            from PIL import Image, ImageOps
            with Image.open(file_path) as img:
                img = ImageOps.exif_transpose(img)
                if target_size is not None and isinstance(target_size, QSize):
                    img.thumbnail((target_size.width(), target_size.height()), Image.Resampling.HAMMING)
                else:
                    img.thumbnail((max_size, max_size), Image.Resampling.HAMMING)

                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return np.array(img)
        except Exception:
            pass

        return None
