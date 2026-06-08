"""
統一圖像處理器 - 處理所有圖像類型（RAW、JPEG、PNG等）

這個模組實現了重構提案中的 UnifiedImageProcessor，提供統一的接口
來處理所有圖像類型，內置快取檢查和錯誤處理。
"""

import os
import numpy as np
from typing import Optional, Dict, Any, Union, Tuple
from PyQt6.QtCore import QSize, QLoggingCategory
from PyQt6.QtGui import QPixmap, QImage
# Silence qt.imageformats warnings (e.g. missing TIFF tag warnings on RAW files)
QLoggingCategory.setFilterRules("qt.imageformats*=false")

import logging
logging.getLogger("exifread").setLevel(logging.ERROR)
# PIL Image, rawpy, and exifread will be imported lazily to avoid import delays

from image_cache import (
    disk_preview_max_edge,
    get_image_cache,
    memory_preview_max_edge,
)
from enhanced_raw_processor import (
    ThumbnailExtractor, 
    EXIFExtractor, 
    OptimizedRAWProcessor
)
from common_image_loader import (
    array_matches_exif_display,
    exif_display_dimensions,
    image_covers_sensor_resolution,
    is_raw_file,
    is_tiff_file,
    load_pixmap_safe,
    use_full_embedded_raw_preview,
    use_libraw_consistent_preview_first,
)


def _verbose_orientation_logs() -> bool:
    return os.environ.get("RAWVIEWER_VERBOSE_ORIENTATION_LOGS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


# LibRaw cannot open some composite HDR DNGs; avoid repeated preview/decode attempts.
_LIBRAW_UNSUPPORTED_PATHS: set[str] = set()


def decode_raw_file(file_path: str, params: Dict[str, Any]) -> np.ndarray:
    """
    Helper function for process pool to decode RAW files.
    Must be at top level for pickling.
    """
    import rawpy
    with rawpy.imread(file_path) as raw:
        return raw.postprocess(**params)


class UnifiedImageProcessor:
    """統一的圖像處理器，處理所有圖像類型"""
    
    def __init__(self):
        self.cache = get_image_cache()
        self.thumbnail_extractor = ThumbnailExtractor()
        self.exif_extractor = EXIFExtractor()
        self.raw_processor = OptimizedRAWProcessor()
    
    def _is_raw_file(self, file_path: str) -> bool:
        """檢查是否為 RAW 文件"""
        return is_raw_file(file_path)

    def _raw_thumbnail_extract_max_size(
        self,
        target_size: Optional[QSize],
        *,
        allow_heavy_fallback: bool,
    ) -> int:
        """Gallery grid uses disk tier (~512px); single-view CURRENT uses memory preview tier (~1920px)."""
        if target_size is not None:
            tw = int(target_size.width()) if target_size.width() > 0 else 512
            th = int(target_size.height()) if target_size.height() > 0 else 512
            return max(256, min(1024, tw, th))
        if allow_heavy_fallback:
            return memory_preview_max_edge()
        return disk_preview_max_edge()

    def _single_view_display_thumbnail(self, target_size: Optional[QSize], allow_heavy_fallback: bool) -> bool:
        return target_size is None and bool(allow_heavy_fallback)

    def _cached_preview_meets_display_tier(self, cached_thumb) -> bool:
        if cached_thumb is None:
            return False
        try:
            min_px = int(memory_preview_max_edge() * 0.85)
            if hasattr(cached_thumb, "shape"):
                return max(int(cached_thumb.shape[0]), int(cached_thumb.shape[1])) >= min_px
            if hasattr(cached_thumb, "width"):
                return max(int(cached_thumb.height()), int(cached_thumb.width())) >= min_px
        except Exception:
            pass
        return False

    def _cache_display_tier_result(
        self,
        file_path: str,
        thumbnail: np.ndarray,
        exif_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Memory-cache display-tier preview for preview-first warm opens only.

        Single ``put_preview`` copy (no duplicate thumbnail_cache write) so
        thumb+exif / full_pipeline cold paths are not penalized.
        """
        if thumbnail is None:
            return
        try:
            if self._cached_preview_meets_display_tier(
                self.cache.get_preview(file_path)
            ):
                return
            self.cache.put_preview(file_path, thumbnail)
        except Exception:
            pass
        if exif_data:
            try:
                cur = self.cache.get_exif(file_path)
                if cur is not None and not cur.get("minimal_preview_exif"):
                    return
                self.cache.put_exif(
                    file_path,
                    exif_data,
                    persist_disk=not bool(exif_data.get("minimal_preview_exif")),
                )
            except Exception:
                pass

    def cache_index_source_mipmap_tiers(
        self,
        file_path: str,
        thumbnail,
        *,
        exif_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write preview/grid/thumbnail tiers so semantic indexing reuses gallery extractions."""
        if thumbnail is None:
            return
        try:
            if (
                self.cache.get_preview(file_path) is not None
                or self.cache.get_grid(file_path) is not None
                or self.cache.get_thumbnail(file_path) is not None
            ):
                return
        except Exception:
            pass

        from PyQt6.QtGui import QImage

        if isinstance(thumbnail, QImage):
            from enhanced_raw_processor import _qimage_to_rgb_array

            thumbnail = _qimage_to_rgb_array(thumbnail)
        if thumbnail is None or not isinstance(thumbnail, np.ndarray):
            return

        try:
            from PIL import Image
            import io

            arr = thumbnail.astype(np.uint8) if thumbnail.dtype != np.uint8 else thumbnail
            if len(arr.shape) == 2:
                pil_img = Image.fromarray(arr, "L").convert("RGB")
            elif len(arr.shape) == 3:
                pil_img = Image.fromarray(arr, "RGB")
            else:
                return

            w, h = pil_img.size
            pv_max = disk_preview_max_edge()
            if w <= pv_max and h <= pv_max:
                preview_pil = pil_img
            else:
                scale_1 = min(pv_max / w, pv_max / h)
                preview_pil = pil_img.resize(
                    (max(1, int(w * scale_1)), max(1, int(h * scale_1))),
                    Image.Resampling.HAMMING,
                )
            self.cache.put_preview(file_path, np.array(preview_pil))

            grid_max = disk_preview_max_edge()
            if w <= grid_max and h <= grid_max:
                grid_pil = pil_img
            else:
                scale_2 = min(grid_max / w, grid_max / h)
                grid_pil = pil_img.resize(
                    (max(1, int(w * scale_2)), max(1, int(h * scale_2))),
                    Image.Resampling.HAMMING,
                )
            buffer_g = io.BytesIO()
            grid_pil.save(buffer_g, format="JPEG", quality=85)
            self.cache.put_grid(file_path, np.array(grid_pil), buffer_g.getvalue())

            if w <= 256 and h <= 256:
                thumb_pil = pil_img
            else:
                scale_3 = min(256 / w, 256 / h)
                thumb_pil = pil_img.resize(
                    (max(1, int(w * scale_3)), max(1, int(h * scale_3))),
                    Image.Resampling.HAMMING,
                )
            buffer_t = io.BytesIO()
            thumb_pil.save(buffer_t, format="JPEG", quality=85)
            self.cache.put_thumbnail(file_path, np.array(thumb_pil), buffer_t.getvalue())
        except Exception:
            pass

        if exif_data:
            try:
                cur = self.cache.get_exif(file_path)
                if cur is not None and not cur.get("minimal_preview_exif"):
                    return
                self.cache.put_exif(
                    file_path,
                    exif_data,
                    persist_disk=not bool(exif_data.get("minimal_preview_exif")),
                )
            except Exception:
                pass

    def _extract_raw_preview_before_full_exiftool(
        self,
        file_path: str,
        cached_exif: Optional[Dict[str, Any]],
        cached_thumb,
    ) -> Optional[Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]]:
        """Single rawpy open: embedded preview first, defer exiftool to a follow-up task."""
        if cached_exif is not None and self._cached_preview_meets_display_tier(cached_thumb):
            return cached_exif, cached_thumb
        import rawpy

        try:
            with rawpy.imread(file_path) as raw:
                max_size_val = self._raw_thumbnail_extract_max_size(
                    None, allow_heavy_fallback=True
                )
                thumb = cached_thumb
                if not self._cached_preview_meets_display_tier(thumb):
                    thumb = self.thumbnail_extractor.extract_thumbnail_from_raw(
                        file_path,
                        max_size=max_size_val,
                        allow_scan_fallback=True,
                        raw_object=raw,
                    )
                if thumb is None:
                    return None
                exif = cached_exif
                if exif is None or exif.get("minimal_preview_exif"):
                    exif = self.exif_extractor.build_minimal_raw_exif(file_path, raw)
                if exif and exif.get("orientation", 1) != 1:
                    thumb = self._apply_orientation_correction(
                        thumb, exif["orientation"], exif
                    )
                return exif, thumb
        except Exception:
            return None

    def ensure_display_tier_preview(
        self, file_path: str, thumbnail: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Upgrade sub-1400px worker thumbs to ~1920 embedded preview when possible."""
        if thumbnail is None or not self._is_raw_file(file_path):
            return thumbnail
        from common_image_loader import dng_prefers_embedded_preview_first
        is_pano_dng = dng_prefers_embedded_preview_first(file_path)
        try:
            min_px = 1024 if is_pano_dng else int(memory_preview_max_edge() * 0.85)
            if hasattr(thumbnail, "shape"):
                md = max(int(thumbnail.shape[0]), int(thumbnail.shape[1]))
            elif hasattr(thumbnail, "width"):
                md = max(int(thumbnail.width()), int(thumbnail.height()))
            else:
                return thumbnail
            if md >= min_px:
                return thumbnail
        except Exception:
            return thumbnail
        import logging

        logger = logging.getLogger(__name__)
        target = memory_preview_max_edge()
        try:
            from enhanced_raw_processor import extract_embedded_jpeg_by_scan

            scanned = extract_embedded_jpeg_by_scan(file_path, target)
            if scanned is not None:
                smd = max(int(scanned.shape[0]), int(scanned.shape[1]))
                if smd >= min_px:
                    logger.info(
                        "[PREVIEW] Scan fallback raised preview to %dpx for %s",
                        smd,
                        os.path.basename(file_path),
                    )
                    return scanned
        except Exception:
            pass
        pair = self._extract_raw_preview_before_full_exiftool(file_path, None, None)
        if pair is not None and pair[1] is not None:
            thumb = pair[1]
            if hasattr(thumb, "shape"):
                smd = max(int(thumb.shape[0]), int(thumb.shape[1]))
                if smd >= min_px:
                    logger.info(
                        "[PREVIEW] Rawpy fallback raised preview to %dpx for %s",
                        smd,
                        os.path.basename(file_path),
                    )
                    return thumb
        half = self.try_half_size_display_preview(file_path)
        if half is not None:
            try:
                smd = max(int(half.shape[0]), int(half.shape[1]))
                if smd >= min_px:
                    logger.info(
                        "[PREVIEW] Half-size fallback raised preview to %dpx for %s",
                        smd,
                        os.path.basename(file_path),
                    )
                    return half
            except Exception:
                pass
        logger.warning(
            "[PREVIEW] Could not reach display tier (%dpx) for %s (stuck at %dpx)",
            min_px,
            os.path.basename(file_path),
            self._preview_buffer_max_dim(thumbnail),
        )
        return thumbnail

    def is_libraw_unsupported(self, file_path: str) -> bool:
        from common_image_loader import dng_prefers_embedded_preview_first
        if dng_prefers_embedded_preview_first(file_path):
            return True
        skip_key = os.path.normcase(os.path.abspath(file_path))
        return skip_key in _LIBRAW_UNSUPPORTED_PATHS

    @staticmethod
    def _preview_buffer_max_dim(buf) -> int:
        try:
            if hasattr(buf, "shape"):
                return max(int(buf.shape[0]), int(buf.shape[1]))
            if hasattr(buf, "width") and hasattr(buf, "height"):
                return max(int(buf.width()), int(buf.height()))
        except Exception:
            pass
        return 0

    def try_half_size_display_preview(self, file_path: str) -> Optional[np.ndarray]:
        """Fast LibRaw half-size decode when embedded JPEG tiers are unusable."""
        if not self._is_raw_file(file_path):
            return None
        import logging

        logger = logging.getLogger(__name__)
        target = memory_preview_max_edge()
        try:
            import rawpy

            with rawpy.imread(file_path) as raw:
                rgb = raw.postprocess(
                    half_size=True,
                    use_camera_wb=True,
                    no_auto_bright=False,
                    output_bps=8,
                )
            if rgb is None or rgb.size == 0:
                return None
            from PIL import Image

            pil = Image.fromarray(rgb.astype("uint8"), "RGB")
            w, h = pil.size
            scale = min(target / max(w, h, 1), 1.0)
            if scale < 1.0:
                pil = pil.resize(
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    Image.Resampling.BILINEAR,
                )
            return np.array(pil)
        except Exception as exc:
            logger.debug(
                "Half-size display preview failed for %s: %s",
                os.path.basename(file_path),
                exc,
            )
            return None
    
    def process_thumbnail(self, file_path: str, allow_heavy_fallback: bool = True,
                          target_size: Optional[QSize] = None) -> Optional[np.ndarray]:
        """處理縮圖（統一接口）"""
        MAX_THUMB_DIM = 1024
        single_view_preview = self._single_view_display_thumbnail(
            target_size, allow_heavy_fallback
        )

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
                exif_data = self.cache.get_exif(file_path)
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
                    
                    import rawpy
                    with rawpy.imread(file_path) as raw:
                        thumbnail = raw.postprocess(**params)
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    _LIBRAW_UNSUPPORTED_PATHS.add(skip_key)
                    logger.warning(f"Heavy thumbnail fallback failed for {file_path}: {e}")
                    return None
            else:
                return None

        # Non-RAW reader path can return QImage directly.
        if isinstance(thumbnail, QImage):
            # If we decoded to target_size, it's already perfect.
            # Don't cache custom-sized QImages in the global persistent cache
            # (which expects 512px JPEGs), but return it for immediate UI display.
            return thumbnail

        # For RAW or fallback paths that return np.ndarray:
        # 獲取 EXIF 數據以獲取 orientation
        exif_data = self.cache.get_exif(file_path)
        if exif_data is None or exif_data.get("minimal_preview_exif"):
            if single_view_preview and self._is_raw_file(file_path):
                try:
                    import rawpy

                    with rawpy.imread(file_path) as raw:
                        exif_data = self.exif_extractor.build_minimal_raw_exif(
                            file_path, raw
                        )
                except Exception:
                    exif_data = None
            else:
                exif_data = None
        if exif_data is None:
            exif_data = self.exif_extractor.extract_exif_data(file_path)
        orientation = exif_data.get('orientation', 1) if exif_data else 1
        if orientation != 1:
            thumbnail = self._apply_orientation_correction(thumbnail, orientation, exif_data)
        if exif_data and exif_data.get("minimal_preview_exif"):
            self.cache.put_exif(file_path, exif_data, persist_disk=False)
        
        try:
            from PIL import Image
            import io
            
            if not isinstance(thumbnail, np.ndarray):
                return thumbnail
                
            if thumbnail.dtype != np.uint8:
                thumbnail = thumbnail.astype(np.uint8)
            
            if len(thumbnail.shape) == 2:
                pil_img = Image.fromarray(thumbnail, 'L')
            elif len(thumbnail.shape) == 3:
                pil_img = Image.fromarray(thumbnail, 'RGB')
            else:
                return thumbnail

            # --- Structured Mipmap (Multi-Scale) Caching Pipeline ---
            w, h = pil_img.size

            pv_max = (
                memory_preview_max_edge()
                if single_view_preview
                else disk_preview_max_edge()
            )
            if w <= pv_max and h <= pv_max:
                preview_pil = pil_img
            else:
                scale_1 = min(pv_max / w, pv_max / h)
                new_w1 = max(1, int(w * scale_1))
                new_h1 = max(1, int(h * scale_1))
                preview_pil = pil_img.resize((new_w1, new_h1), Image.Resampling.HAMMING)
            self.cache.put_preview(file_path, np.array(preview_pil))

            grid_max = disk_preview_max_edge()
            if w <= grid_max and h <= grid_max:
                grid_pil = pil_img
            else:
                scale_2 = min(grid_max / w, grid_max / h)
                new_w2 = max(1, int(w * scale_2))
                new_h2 = max(1, int(h * scale_2))
                grid_pil = pil_img.resize((new_w2, new_h2), Image.Resampling.HAMMING)

            buffer_g = io.BytesIO()
            grid_pil.save(buffer_g, format='JPEG', quality=85)
            self.cache.put_grid(file_path, np.array(grid_pil), buffer_g.getvalue())

            if w <= 256 and h <= 256:
                thumb_pil = pil_img
            else:
                scale_3 = min(256 / w, 256 / h)
                new_w3 = max(1, int(w * scale_3))
                new_h3 = max(1, int(h * scale_3))
                thumb_pil = pil_img.resize((new_w3, new_h3), Image.Resampling.HAMMING)

            buffer_t = io.BytesIO()
            thumb_pil.save(buffer_t, format='JPEG', quality=85)
            thumb_np = np.array(thumb_pil)
            self.cache.put_thumbnail(file_path, thumb_np, buffer_t.getvalue())

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

    
    def process_full_image(self, file_path: str, 
                          use_full_resolution: bool = False,
                          executor: Optional[Any] = None) -> Optional[Union[np.ndarray, QPixmap]]:
        """處理完整圖像（統一接口）"""
        # 檢查快取
        # CRITICAL: For RAW files, only check full_image cache, not pixmap cache
        # This ensures RAW files always go through proper processing with orientation correction
        is_raw = self._is_raw_file(file_path)
        
        if use_full_resolution:
            cached_image = self.cache.get_full_image(file_path)
            if cached_image is not None:
                # Orientation verification for cache hit
                exif_data = self.exif_extractor.extract_exif_data(file_path)
                orientation = exif_data.get('orientation', 1) if exif_data else 1
                
                if hasattr(cached_image, 'shape'):
                    h, w = cached_image.shape[:2]
                    
                    # Check for orientation mismatch (Portrait metadata but Landscape cached image)
                    if orientation in (6, 8) and w > h:
                        if _verbose_orientation_logs():
                            # print(f"[ORIENTATION] UnifiedImageProcessor: Cached full_image for {os.path.basename(file_path)} is UNROTATED (stale). Re-processing.")
                            pass
                        cached_image = None
                    
                    if cached_image is not None:
                        if _verbose_orientation_logs():
                            # print(f"[ORIENTATION] Using VALID cached full_image for {os.path.basename(file_path)}. Shape: {w}x{h}")
                            pass
                        return cached_image
                else:
                    return cached_image

        
        # Only check pixmap cache for non-RAW files
        if not is_raw:
            cached_pixmap = self.cache.get_pixmap(file_path)
            if cached_pixmap is not None:
                if _verbose_orientation_logs():
                    # print(f"[ORIENTATION] Using cached pixmap for {os.path.basename(file_path)} (non-RAW file)")
                    pass
                return cached_pixmap
        else:
            # For RAW files, don't use cached pixmap - always process fresh
            if _verbose_orientation_logs():
                # print(f"[ORIENTATION] RAW file {os.path.basename(file_path)} - skipping pixmap cache, will process as RAW")
                pass
        
        # 處理圖像
        if is_raw:
            return self._process_raw_image(file_path, use_full_resolution, executor=executor)
        else:
            return self._process_regular_image(
                file_path, use_full_resolution=use_full_resolution
            )
    
    def _try_full_embedded_raw_preview(
        self, file_path: str, exif_data: Optional[Dict[str, Any]]
    ) -> Optional[np.ndarray]:
        """
        Return oriented embedded JPEG when it covers sensor resolution; cache as preview + full.
        """
        if not use_full_embedded_raw_preview():
            return None

        for cached in (
            self.cache.get_full_image(file_path),
            self.cache.get_preview(file_path),
        ):
            if cached is not None:
                h, w = cached.shape[:2]
                if image_covers_sensor_resolution(w, h, exif_data):
                    return cached

        embedded = self.thumbnail_extractor.extract_embedded_native_preview(file_path)
        if embedded is None:
            return None

        h, w = embedded.shape[:2]
        if not image_covers_sensor_resolution(w, h, exif_data):
            return None

        orientation = exif_data.get("orientation", 1) if exif_data else 1
        if orientation != 1:
            embedded = self._apply_orientation_correction(embedded, orientation, exif_data)

        self.cache.put_preview(file_path, embedded)
        self.cache.put_full_image(file_path, embedded)
        import logging

        logging.getLogger(__name__).info(
            "[PREVIEW] Using full-resolution embedded JPEG for %s (%dx%d)",
            os.path.basename(file_path),
            w,
            h,
        )
        return embedded

    def _process_raw_image(self, file_path: str, 
                          use_full_resolution: bool = False,
                          executor: Optional[Any] = None) -> Optional[np.ndarray]:
        """處理 RAW 圖像"""
        libraw_first = use_libraw_consistent_preview_first()
        skip_key = os.path.normcase(os.path.abspath(file_path))
        from common_image_loader import dng_prefers_embedded_preview_first
        if dng_prefers_embedded_preview_first(file_path):
            _LIBRAW_UNSUPPORTED_PATHS.add(skip_key)
        if skip_key in _LIBRAW_UNSUPPORTED_PATHS:
            cached = self.cache.get_preview(file_path)
            if cached is None:
                cached = self.cache.get_full_image(file_path)
            if cached is not None:
                cached_dim = max(cached.shape[0], cached.shape[1]) if hasattr(cached, "shape") else 0
                if cached_dim >= 1024:
                    return cached
            if use_full_resolution:
                exif_data = self.exif_extractor.extract_exif_data(file_path)
                
                # First try decoding the full image via PIL TIFF reader (useful for linear/composite DNG panoramas)
                try:
                    from PIL import Image
                    with Image.open(file_path) as im:
                        best_w = best_h = 0
                        best_idx = -1
                        n_frames = getattr(im, "n_frames", 1)
                        for idx in range(n_frames):
                            im.seek(idx)
                            w, h = im.size
                            if w * h > best_w * best_h:
                                best_w, best_h = w, h
                                best_idx = idx
                        if best_idx >= 0 and best_w >= 1024:
                            im.seek(best_idx)
                            rgb_im = im.convert("RGB")
                            full_pixels = np.array(rgb_im)
                            orientation = exif_data.get("orientation", 1) if exif_data else 1
                            if orientation != 1:
                                full_pixels = self._apply_orientation_correction(
                                    full_pixels, orientation, exif_data
                                )
                            self.cache.put_full_image(file_path, full_pixels)
                            self.cache.put_preview(file_path, full_pixels)
                            import logging
                            logging.getLogger(__name__).info(
                                "[FULL_RES] Successfully loaded full resolution DNG panorama via PIL TIFF reader: %dx%d",
                                best_w, best_h
                            )
                            return full_pixels
                except Exception as pil_e:
                    import logging
                    logging.getLogger(__name__).warning(
                        "[FULL_RES] PIL TIFF decode fallback failed/unsupported for %s: %s",
                        os.path.basename(file_path), pil_e
                    )

                full_embedded = self._try_full_embedded_raw_preview(file_path, exif_data)
                if full_embedded is not None:
                    return full_embedded
                
                # If that didn't work (due to resolution coverage reject or workflow toggle),
                # extract the largest native preview since LibRaw can't demosaic this file anyway.
                native_preview = self.thumbnail_extractor.extract_embedded_native_preview(file_path)
                if native_preview is not None:
                    orientation = exif_data.get("orientation", 1) if exif_data else 1
                    if orientation != 1:
                        native_preview = self._apply_orientation_correction(
                            native_preview, orientation, exif_data
                        )
                    self.cache.put_preview(file_path, native_preview)
                    self.cache.put_full_image(file_path, native_preview)
                    return native_preview
                
                mem_max = memory_preview_max_edge()
                preview = self.thumbnail_extractor.extract_preview_from_raw(
                    file_path, max_size=mem_max
                )
                if preview is not None:
                    orientation = (
                        exif_data.get("orientation", 1) if exif_data else 1
                    )
                    if orientation != 1:
                        preview = self._apply_orientation_correction(
                            preview, orientation, exif_data
                        )
                    self.cache.put_preview(file_path, preview)
                    return preview
                try:
                    from enhanced_raw_processor import extract_embedded_jpeg_by_scan

                    scanned = extract_embedded_jpeg_by_scan(file_path, mem_max)
                    if scanned is not None:
                        orientation = (
                            exif_data.get("orientation", 1) if exif_data else 1
                        )
                        if orientation != 1:
                            scanned = self._apply_orientation_correction(
                                scanned, orientation, exif_data
                            )
                        self.cache.put_preview(file_path, scanned)
                        return scanned
                except Exception:
                    pass
                return None
            return None
        try:
            # 獲取 EXIF 數據（用於處理參數）
            exif_data = self.exif_extractor.extract_exif_data(file_path)
            
            full_embedded = self._try_full_embedded_raw_preview(file_path, exif_data)
            if full_embedded is not None:
                return full_embedded
            
            # 檢查文件大小以決定處理策略
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            use_fast_processing = file_size_mb > 20 and not use_full_resolution
            
            # 處理 RAW 文件
            # Check if we should use Preview (embedded JPEG) instead of processing RAW
            if not use_full_resolution and not libraw_first:
               if skip_key in _LIBRAW_UNSUPPORTED_PATHS:
                   return self.cache.get_preview(file_path)
               # OPTIMIZATION: Try embedded / cached preview instead of full RAW demosaic.
               
               # Check Preview Cache
               cached_preview = self.cache.get_preview(file_path)
               if cached_preview is not None:
                   if _verbose_orientation_logs():
                       print(f"[PREVIEW] Using cached preview for {os.path.basename(file_path)}")
                   return cached_preview
               
               # Extract Preview
               if _verbose_orientation_logs():
                   print(f"[PREVIEW] Extracting preview from RAW for {os.path.basename(file_path)}")
               mem_max = memory_preview_max_edge()
               preview = self.thumbnail_extractor.extract_preview_from_raw(
                   file_path, max_size=mem_max
               )
               
               if preview is not None:
                   # Check if preview is large enough for a good "Fit" view
                   # Many phone DNGs have tiny previews (e.g. 1024px) that look bad
                   h, w = preview.shape[:2]
                   fit_min = max(1024, int(mem_max * 0.75))
                   if max(h, w) >= fit_min:
                       # Apply Orientation to Preview
                       orientation = exif_data.get('orientation', 1) if exif_data else 1
                       if orientation != 1:
                           preview = self._apply_orientation_correction(preview, orientation, exif_data)
                       
                       # Cache Preview
                       self.cache.put_preview(file_path, preview)
                       return preview
                   else:
                       if _verbose_orientation_logs():
                           print(f"[PREVIEW] Embedded preview is too small ({w}x{h}), falling back to RAW processing for better quality")
               
               # Fallback to RAW processing if preview extraction fails
               if _verbose_orientation_logs():
                   print(f"[PREVIEW] Preview extraction failed, falling back to RAW processing")

            # 獲取處理器參數
            params = self.raw_processor.get_optimized_processing_params(
                file_path, exif_data
            )
            
            # 根據需求調整參數
            if use_fast_processing:
                params['half_size'] = True

            # LibRaw-first: first fit view uses same pipeline as zoom (half-res decode is faster than full demosaic)
            if libraw_first and not use_full_resolution:
                params['half_size'] = True
            
            if use_full_resolution:
                params['half_size'] = False
                params['output_bps'] = 8  # 8-bit 足夠顯示
            
            # CRITICAL: Force rawpy to ignore EXIF orientation
            params['user_flip'] = 0
            
            # 處理 RAW - 使用 Process Pool 繞過 GIL
            rgb_image = None
            if executor:
                try:
                    future = executor.submit(decode_raw_file, file_path, params)
                    rgb_image = future.result()
                except Exception as pool_err:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"[PROCESS_POOL] Process pool decode failed for {os.path.basename(file_path)}: {pool_err}. "
                        "Falling back to in-process RAW decode."
                    )

            # Proof of Concept: Optional GPU-Accelerated demosaicing (development branch)
            if rgb_image is None:
                try:
                    from gpu_raw_processor import try_gpu_raw_decode, detect_gpu_backend

                    if detect_gpu_backend() != "cpu_only":
                        import rawpy

                        with rawpy.imread(file_path) as raw:
                            raw_array = raw.raw_image.copy()
                        gpu_rgb = try_gpu_raw_decode(file_path, raw_array, exif_data)
                        if gpu_rgb is not None:
                            rgb_image = gpu_rgb
                except Exception as gpu_err:
                    import logging

                    logging.getLogger(__name__).warning(
                        "[GPU] PoC GPU decode failed: %s", gpu_err
                    )

            if rgb_image is None:
                import rawpy
                with rawpy.imread(file_path) as raw:
                    rgb_image = raw.postprocess(**params)
            
            # 應用方向校正 - 確保使用最新的 EXIFExtractor 邏輯
            if not exif_data or exif_data.get('orientation', 1) == 1:
                # Last-minute check: if we think it's 1, try a deep extraction just in case
                exif_data = self.exif_extractor.extract_exif_data(file_path)
            
            orientation = exif_data.get('orientation', 1) if exif_data else 1
            if os.environ.get("RAWVIEWER_VERBOSE_ORIENTATION_LOGS") == "1":
                # print(f"[ORIENTATION] UnifiedImageProcessor (RAW): {os.path.basename(file_path)} final_orientation={orientation}")
                pass
                
            if orientation != 1 and rgb_image is not None:
                rgb_image = self._apply_orientation_correction(rgb_image, orientation, exif_data)
            
            # 快取完整圖像
            if rgb_image is not None:
                self.cache.put_full_image(file_path, rgb_image)
            
            return rgb_image
                
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            
            is_unsupported = "unsupported" in str(e).lower() or "not recognized" in str(e).lower()
            if is_unsupported:
                _LIBRAW_UNSUPPORTED_PATHS.add(skip_key)
                logger.warning(
                    f"RAW file unsupported by LibRaw (e.g., composite DNG): {os.path.basename(file_path)}. Error: {e}"
                )
            else:
                logger.error(f"Error processing RAW image {file_path}: {e}", exc_info=True)

            # LibRaw failed entirely; try byte-scan for embedded JPEG (already used in thumbnail
            # path, but preview may have been skipped or a different limit may help).
            try:
                from enhanced_raw_processor import extract_embedded_jpeg_by_scan
                lim = 8192 if use_full_resolution else memory_preview_max_edge()
                scanned = extract_embedded_jpeg_by_scan(file_path, lim)
                if scanned is not None:
                    exif_data = self.exif_extractor.extract_exif_data(file_path)
                    orientation = exif_data.get("orientation", 1) if exif_data else 1
                    if orientation != 1:
                        scanned = self._apply_orientation_correction(
                            scanned, orientation, exif_data
                        )
                    self.cache.put_preview(file_path, scanned)
                    if use_full_resolution:
                        self.cache.put_full_image(file_path, scanned)
                    return scanned
            except Exception:
                pass
            return None
    
    def _process_regular_image(
        self, file_path: str, use_full_resolution: bool = False
    ) -> Optional[QPixmap]:
        """處理常規圖像（JPEG/PNG/WebP）"""
        try:
            from image_cache import memory_preview_max_edge

            max_edge = 0 if use_full_resolution else memory_preview_max_edge()
            pixmap = load_pixmap_safe(file_path, max_edge=max_edge)
            if not pixmap.isNull():
                self.cache.put_pixmap(file_path, pixmap)
                return pixmap
            return None
                
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error processing regular image {file_path}: {e}", exc_info=True)
            return None
    
    def process_exif(self, file_path: str) -> Optional[Dict[str, Any]]:
        """處理 EXIF 數據（統一接口）"""
        return self.exif_extractor.extract_exif_data(file_path)

    def process_metadata_and_thumbnail(self, file_path: str, allow_heavy_fallback: bool = True,
                                      target_size: Optional[QSize] = None) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]:
        """
        COMBINED OPTIMIZATION: Extract both metadata and thumbnail in a single pass.
        This is crucial for RAW files to avoid opening the large file twice.
        """
        is_raw = self._is_raw_file(file_path)
        
        if not is_raw:
            # For non-RAW, standard sequential calls are fine as they use different backends
            exif = self.process_exif(file_path)
            thumb = self.process_thumbnail(file_path, allow_heavy_fallback, target_size)
            return exif, thumb

        # Gallery tiles: try cache + embedded JPEG without parsing the full RAW container.
        cached_exif = self.cache.get_exif(file_path)
        cached_thumb = self.cache.get_thumbnail(file_path)
        if cached_thumb is None and target_size is None:
            cached_thumb = self.cache.get_preview(file_path)

        if cached_thumb is not None and target_size is None and allow_heavy_fallback:
            try:
                if hasattr(cached_thumb, "shape"):
                    if max(int(cached_thumb.shape[0]), int(cached_thumb.shape[1])) < int(
                        memory_preview_max_edge() * 0.85
                    ):
                        cached_thumb = None
                elif hasattr(cached_thumb, "width"):
                    if max(int(cached_thumb.height()), int(cached_thumb.width())) < int(
                        memory_preview_max_edge() * 0.85
                    ):
                        cached_thumb = None
            except Exception:
                pass

        if cached_thumb is not None and target_size is None:
            try:
                min_px = int(memory_preview_max_edge() * 0.85)
                if hasattr(cached_thumb, "shape"):
                    if max(int(cached_thumb.shape[0]), int(cached_thumb.shape[1])) < min_px:
                        cached_thumb = None
                elif hasattr(cached_thumb, "width"):
                    if max(int(cached_thumb.height()), int(cached_thumb.width())) < min_px:
                        cached_thumb = None
            except Exception:
                pass

        if cached_exif is not None and cached_thumb is not None:
            if not cached_exif.get("minimal_preview_exif") and self._cached_preview_meets_display_tier(
                cached_thumb
            ):
                return cached_exif, cached_thumb

        if allow_heavy_fallback and target_size is None:
            fast_pair = self._extract_raw_preview_before_full_exiftool(
                file_path, cached_exif, cached_thumb
            )
            if fast_pair is not None:
                return fast_pair

        exif = cached_exif or self.exif_extractor.extract_exif_data(file_path)
        thumb = cached_thumb
        if thumb is None:
            max_dim = self._raw_thumbnail_extract_max_size(
                target_size, allow_heavy_fallback=allow_heavy_fallback
            )
            thumb = self.thumbnail_extractor.extract_thumbnail_from_raw(
                file_path,
                max_size=max_dim,
                allow_scan_fallback=True,
                raw_object=None,
            )
            if thumb is not None and exif:
                orientation = exif.get("orientation", 1)
                if orientation != 1:
                    thumb = self._apply_orientation_correction(
                        thumb, orientation, exif
                    )
        if thumb is not None and exif is not None:
            return exif, thumb

        # RAW Path: Open once, extract both
        import rawpy
        try:
            with rawpy.imread(file_path) as raw:
                # 1. Extract EXIF (verified with rawpy sensor sizes)
                exif = self.exif_extractor.extract_exif_data(file_path, raw_object=raw)
                
                # 2. Extract Thumbnail
                max_size_val = self._raw_thumbnail_extract_max_size(
                    target_size, allow_heavy_fallback=allow_heavy_fallback
                )
                thumb = self.thumbnail_extractor.extract_thumbnail_from_raw(
                    file_path,
                    max_size=max_size_val,
                    allow_scan_fallback=allow_heavy_fallback,
                    raw_object=raw
                )
                
                # Apply orientation to thumbnail if needed
                if thumb is not None and exif:
                    orientation = exif.get('orientation', 1)
                    if orientation != 1:
                        thumb = self._apply_orientation_correction(thumb, orientation, exif)
                
                return exif, thumb
        except Exception:
            # Fallback to sequential if single-pass fails
            exif = self.process_exif(file_path)
            thumb = self.process_thumbnail(file_path, allow_heavy_fallback, target_size)
            return exif, thumb
    
    def _apply_orientation_correction(
        self,
        image_array: np.ndarray,
        orientation: int,
        exif_data: Dict[str, Any] = None
    ) -> np.ndarray:
        """根據 EXIF Orientation 值修正影像方向

        Orientation Reference:
        1 = Normal
        2 = Mirrored horizontal
        3 = Rotated 180°
        4 = Mirrored vertical
        5 = Mirrored horizontal + Rotated 270° CW
        6 = Rotated 90° CW
        7 = Mirrored horizontal + Rotated 90° CW
        8 = Rotated 270° CW (i.e., 90° CCW)
        """
        if image_array is None:
            if _verbose_orientation_logs():
                # print(f"[ORIENTATION] Error: image_array is None in _apply_orientation_correction")
                pass
            return None
            
        original_shape = image_array.shape
        if _verbose_orientation_logs():
            # print(f"[ORIENTATION] Before correction: shape = {original_shape}")
            pass
        
        h, w = image_array.shape[:2]
        if exif_data:
            ow = int(exif_data.get("original_width") or 0)
            oh = int(exif_data.get("original_height") or 0)
            if ow > 0 and oh > 0:
                dw, dh = exif_display_dimensions(ow, oh, orientation)
                if (dh > dw) == (h > w):
                    return image_array

        if orientation == 1:
            return image_array

        # Rotation needed?
        import logging
        logger = logging.getLogger(__name__)

        if orientation == 2:
            # Mirror left-right
            if _verbose_orientation_logs():
                # print(f"[ORIENTATION] Numpy operation: np.fliplr(image_array) - Mirror left-right")
                pass
            result = np.fliplr(image_array)
        elif orientation == 3:
            # Rotate 180°
            if _verbose_orientation_logs():
                # print(f"[ORIENTATION] Numpy operation: np.rot90(image_array, 2) - Rotate 180°")
                pass
            result = np.rot90(image_array, 2)
        elif orientation == 4:
            # Mirror top-bottom
            if _verbose_orientation_logs():
                # print(f"[ORIENTATION] Numpy operation: np.flipud(image_array) - Mirror top-bottom")
                pass
            result = np.flipud(image_array)
        elif orientation == 5:
            # Mirror LR + rotate 270° CW (k=1 CCW)
            if _verbose_orientation_logs():
                # print(f"[ORIENTATION] Numpy operation: np.rot90(np.fliplr(image_array), 1) - Mirror LR + rotate 90° CCW")
                pass
            result = np.rot90(np.fliplr(image_array), 1)
        elif orientation == 6:
            # Orientation 6: Image is rotated 90° CW.
            # We need to rotate it 90° CW (k=3) to fix it.
            if _verbose_orientation_logs():
                # print(f"[ORIENTATION] Numpy operation: np.rot90(image_array, 3) - Rotate 90° CW")
                pass
            result = np.rot90(image_array, 3)
        elif orientation == 7:
            # Mirror LR + rotate 90° CW
            if _verbose_orientation_logs():
                # print(f"[ORIENTATION] Numpy operation: np.rot90(np.fliplr(image_array), 3) - Mirror LR + rotate 270° CCW (90° CW)")
                pass
            result = np.rot90(np.fliplr(image_array), 3)
        elif orientation == 8:
            # Orientation 8: Image is rotated 270° CW (90° CCW).
            # We need to rotate it 90° CCW (k=1) to fix it.
            if _verbose_orientation_logs():
                # print(f"[ORIENTATION] Numpy operation: np.rot90(image_array, 1) - Rotate 90° CCW")
                pass
            result = np.rot90(image_array, 1)
        else:
            # Unknown orientation
            result = image_array
        
        final_shape = result.shape
        if _verbose_orientation_logs():
            # print(f"[ORIENTATION] After correction: shape = {final_shape}")
            pass
        return result


