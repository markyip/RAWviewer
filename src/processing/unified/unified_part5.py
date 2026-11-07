"""Unified mixin part 5 of 7 for UnifiedImageProcessor."""
from __future__ import annotations

import os

import numpy as np
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QImage

from .helpers import (
    _LIBRAW_UNSUPPORTED_PATHS,
    _mark_libraw_unsupported,
    array_matches_exif_display,
    disk_preview_max_edge,
    memory_preview_max_edge,
    rawpy_imread_warm,
)


class UnifiedPart5:
    def process_thumbnail(self, file_path: str, allow_heavy_fallback: bool = True,
                          target_size: QSize | None = None) -> np.ndarray | None:
        """處理縮圖（統一接口）"""
        if self.cache.is_libraw_preview(file_path):
            prev = self.cache.get_preview(file_path)
            if prev is not None:
                from raw_adjustments import mark_array_as_already_adjusted
                # Downscale to target_size or grid size
                single_view_preview = self._single_view_display_thumbnail(
                    target_size, allow_heavy_fallback
                )
                if target_size is not None:
                    h, w = prev.shape[:2]
                    scale = min(target_size.width() / w, target_size.height() / h)
                    if scale < 1.0:
                        import cv2
                        prev = cv2.resize(
                            prev,
                            (max(1, int(w * scale)), max(1, int(h * scale))),
                            interpolation=cv2.INTER_AREA,
                        )
                elif not single_view_preview:
                    h, w = prev.shape[:2]
                    cap = 512
                    if max(h, w) > cap:
                        scale = cap / max(h, w)
                        import cv2
                        prev = cv2.resize(
                            prev,
                            (max(1, int(w * scale)), max(1, int(h * scale))),
                            interpolation=cv2.INTER_AREA,
                        )
                mark_array_as_already_adjusted(prev)
                return prev

        MAX_THUMB_DIM = 1024
        single_view_preview = self._single_view_display_thumbnail(
            target_size, allow_heavy_fallback
        )

        if not single_view_preview:
            grid = self.cache.get_grid(file_path)
            if grid is not None:
                return grid

        # 檢查快取
        cached = self.cache.get_thumbnail(file_path)
        if cached is not None:
            # Hard safety: never return an oversized "thumbnail" (e.g. RAW BITMAP thumb)
            try:
                if hasattr(cached, 'shape'):
                    h0, w0 = cached.shape[:2]
                    if max(h0, w0) > MAX_THUMB_DIM:
                        cached = None
            except Exception:
                cached = None

            if cached is not None and hasattr(cached, "shape"):
                # verify=True: feeds an orientation-match decision (worker thread).
                exif_data = self.cache.get_exif(file_path, verify=True)
                if exif_data is None:
                    exif_data = self.exif_extractor.extract_exif_data(file_path)
                if exif_data and not array_matches_exif_display(
                    cached.shape[1], cached.shape[0], exif_data
                ):
                    import logging

                    logging.getLogger(__name__).debug(
                        "Thumbnail orientation mismatch; re-extracting %s",
                        os.path.basename(file_path),
                    )
                    self.cache.thumbnail_cache.remove(file_path)
                    self.cache.disk_thumbnail_cache.remove(file_path)
                    cached = None

            if cached is not None:
                if single_view_preview and hasattr(cached, "shape"):
                    try:
                        if max(int(cached.shape[0]), int(cached.shape[1])) < int(
                            memory_preview_max_edge() * 0.85
                        ):
                            cached = None
                    except Exception:
                        pass
                elif hasattr(cached, "shape"):
                    try:
                        if max(int(cached.shape[0]), int(cached.shape[1])) < int(
                            disk_preview_max_edge() * 0.85
                        ):
                            cached = None
                    except Exception:
                        pass
                if cached is not None:
                    return cached

        # Gallery grid: disk tier (~512px). Single-view CURRENT: memory preview tier (~1920px).
        is_raw = self._is_raw_file(file_path)
        skip_key = os.path.normcase(os.path.abspath(file_path))
        if is_raw and skip_key in _LIBRAW_UNSUPPORTED_PATHS:
            return None
        if is_raw:
            max_size_val = self._raw_thumbnail_extract_max_size(
                target_size, allow_heavy_fallback=allow_heavy_fallback
            )
            thumbnail = self.thumbnail_extractor.extract_thumbnail_from_raw(
                file_path,
                max_size=max_size_val,
                allow_scan_fallback=allow_heavy_fallback,
            )
        else:
            thumbnail = self.thumbnail_extractor.extract_thumbnail_from_image(
                file_path,
                max_size=MAX_THUMB_DIM,
                target_size=target_size
            )

        if thumbnail is None:
            if allow_heavy_fallback and is_raw:
                # Fallback: If no embedded thumbnail found, or it was rejected as too small,
                # do a fast half-size RAW decode to get a high-quality thumbnail.
                try:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.debug(f"No usable thumbnail for {file_path}, falling back to fast RAW decode")

                    # Use OptimizedRAWProcessor for a fast decode
                    params = self.raw_processor.get_optimized_processing_params(file_path, None)
                    params['half_size'] = True
                    params['user_flip'] = 0 # Handle orientation manually


                    from enhanced_raw_processor import _heavy_fallback_semaphore, _rawpy_global_lock
                    with _rawpy_global_lock:
                        raw_ctx = rawpy_imread_warm(file_path)
                    with raw_ctx as raw, _heavy_fallback_semaphore:
                        thumbnail = raw.postprocess(**params)
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    _mark_libraw_unsupported(skip_key)
                    logger.warning(f"Heavy thumbnail fallback failed for {file_path}: {e}")
                    return None
            else:
                return None

        # Non-RAW reader path can return QImage directly.
        from_qimage_autotransform = False
        if isinstance(thumbnail, QImage):
            from enhanced_raw_processor import _qimage_to_rgb_array

            from_qimage_autotransform = True
            rgb = _qimage_to_rgb_array(thumbnail)
            if rgb is None:
                return None
            thumbnail = rgb

        # For RAW or fallback paths that return np.ndarray:
        # 獲取 EXIF 數據以獲取 orientation (verify=True: rotates pixels; worker thread)
        exif_data = self.cache.get_exif(file_path, verify=True)
        if exif_data is None or exif_data.get("minimal_preview_exif"):
            if single_view_preview and self._is_raw_file(file_path):
                try:

                    from enhanced_raw_processor import _rawpy_global_lock
                    with _rawpy_global_lock, rawpy_imread_warm(file_path) as raw:
                        exif_data = self.exif_extractor.build_minimal_raw_exif(
                            file_path, raw
                        )
                except Exception:
                    exif_data = None
            else:
                exif_data = None
        if exif_data is None:
            exif_data = self.exif_extractor.extract_exif_data(file_path)
        decoder_baked_exif = from_qimage_autotransform or (
            not is_raw and thumbnail is not None
        )
        if decoder_baked_exif and exif_data:
            from common_image_loader import mark_exif_pixels_display_oriented

            exif_data = mark_exif_pixels_display_oriented(exif_data)
            self.cache.put_exif(file_path, exif_data, persist_disk=False)
        orientation = exif_data.get('orientation', 1) if exif_data else 1
        if orientation != 1 and not decoder_baked_exif:
            thumbnail = self._apply_orientation_correction(thumbnail, orientation, exif_data)
        if exif_data and exif_data.get("minimal_preview_exif"):
            self.cache.put_exif(file_path, exif_data, persist_disk=False)

        # Apply XMP adjustments if they exist and are not default
        try:
            from raw_adjustments import (
                apply_adjustments_to_rgb,
                is_default_adjustments,
                load_adjustments_for_file,
                sidecar_adjustments_enabled,
            )
            if sidecar_adjustments_enabled() and thumbnail is not None:
                adj = load_adjustments_for_file(file_path)
                if adj and not is_default_adjustments(adj):
                    if not isinstance(thumbnail, np.ndarray) and isinstance(thumbnail, QImage):
                        from_qimage_autotransform = True
                        from enhanced_raw_processor import _qimage_to_rgb_array
                        rgb = _qimage_to_rgb_array(thumbnail)
                        if rgb is not None:
                            thumbnail = rgb
                    if isinstance(thumbnail, np.ndarray):
                        thumbnail = apply_adjustments_to_rgb(thumbnail, adj)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Failed to apply sidecar adjustments to thumbnail for {file_path}: {e}"
            )

        try:

            from PIL import Image

            if not isinstance(thumbnail, np.ndarray):
                return thumbnail

            if thumbnail.dtype != np.uint8:
                thumbnail = thumbnail.astype(np.uint8)

            if len(thumbnail.shape) == 2:
                pil_img = Image.fromarray(thumbnail, 'L').convert('RGB')
            elif len(thumbnail.shape) == 3:
                pil_img = Image.fromarray(thumbnail, 'RGB')
            else:
                return thumbnail

            # --- Structured Mipmap (Multi-Scale) Caching Pipeline ---
            w, h = pil_img.size

            from common_image_loader import encode_tile_bytes

            preview_pil = None
            grid_pil = None
            thumb_pil = None

            if single_view_preview:
                pv_max = memory_preview_max_edge()
                if w <= pv_max and h <= pv_max:
                    preview_pil = pil_img
                else:
                    scale_1 = min(pv_max / w, pv_max / h)
                    new_w1 = max(1, int(w * scale_1))
                    new_h1 = max(1, int(h * scale_1))
                    preview_pil = pil_img.resize(
                        (new_w1, new_h1), Image.Resampling.HAMMING
                    )
                self.cache.put_preview(file_path, np.array(preview_pil))

            grid_max = disk_preview_max_edge()
            if w <= grid_max and h <= grid_max:
                grid_pil = pil_img
            else:
                scale_2 = min(grid_max / w, grid_max / h)
                new_w2 = max(1, int(w * scale_2))
                new_h2 = max(1, int(h * scale_2))
                grid_pil = pil_img.resize((new_w2, new_h2), Image.Resampling.HAMMING)

            self.cache.put_grid(
                file_path, np.array(grid_pil), encode_tile_bytes(grid_pil)
            )

            if w <= 256 and h <= 256:
                thumb_pil = pil_img
            else:
                scale_3 = min(256 / w, 256 / h)
                new_w3 = max(1, int(w * scale_3))
                new_h3 = max(1, int(h * scale_3))
                thumb_pil = pil_img.resize((new_w3, new_h3), Image.Resampling.HAMMING)

            thumb_np = np.array(thumb_pil)
            self.cache.put_thumbnail(
                file_path, thumb_np, encode_tile_bytes(thumb_pil)
            )

            # Single-view: return screen-preview tier; gallery/preload: return grid tier.
            if single_view_preview:
                return np.array(preview_pil)
            return np.array(grid_pil)

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error processing structured mipmaps with PIL: {e}")
            self.cache.put_thumbnail(file_path, thumbnail)
            return thumbnail
