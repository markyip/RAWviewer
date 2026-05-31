"""
統一圖像處理器 - 處理所有圖像類型（RAW、JPEG、PNG等）

這個模組實現了重構提案中的 UnifiedImageProcessor，提供統一的接口
來處理所有圖像類型，內置快取檢查和錯誤處理。
"""

import os
import numpy as np
from typing import Optional, Dict, Any, Union, Tuple
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QPixmap, QImage
# PIL Image, rawpy, and exifread will be imported lazily to avoid import delays

from image_cache import get_image_cache
from enhanced_raw_processor import (
    ThumbnailExtractor, 
    EXIFExtractor, 
    OptimizedRAWProcessor
)
from common_image_loader import (
    is_raw_file,
    is_tiff_file,
    load_pixmap_safe,
    use_libraw_consistent_preview_first,
)


def _verbose_orientation_logs() -> bool:
    return os.environ.get("RAWVIEWER_VERBOSE_ORIENTATION_LOGS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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
    
    def process_thumbnail(self, file_path: str, allow_heavy_fallback: bool = True,
                          target_size: Optional[QSize] = None) -> Optional[np.ndarray]:
        """處理縮圖（統一接口）"""
        MAX_THUMB_DIM = 1024

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

            if cached is not None:
                # Keep cache hits cheap for gallery scrolling. The old self-healing orientation
                # check touched EXIF/SQLite on every hit; enable it only when repairing caches.
                if os.environ.get("RAWVIEWER_VALIDATE_THUMB_CACHE") == "1":
                    exif_data = self.exif_extractor.extract_exif_data(file_path)
                    orientation = exif_data.get('orientation', 1) if exif_data else 1
                    if orientation in (6, 8) and hasattr(cached, 'shape'):
                        h, w = cached.shape[:2]
                        if w > h:
                            import logging
                            logger = logging.getLogger(__name__)
                            logger.debug(
                                "Cached thumbnail orientation mismatch; invalidating %s",
                                os.path.basename(file_path),
                            )
                            self.cache.exif_cache.remove(file_path)
                            cached = None # Force re-processing
                
                if cached is not None:
                    return cached
        
        # 提取縮圖 (Max 512px for Gallery)
        is_raw = self._is_raw_file(file_path)
        if is_raw:
            thumbnail = self.thumbnail_extractor.extract_thumbnail_from_raw(
                file_path,
                max_size=MAX_THUMB_DIM,
                allow_scan_fallback=allow_heavy_fallback,
            )
        else:
            thumbnail = self.thumbnail_extractor.extract_thumbnail_from_image(
                file_path, 
                max_size=MAX_THUMB_DIM,
                target_size=target_size
            )
        
        if thumbnail is None:
            if allow_heavy_fallback:
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
        exif_data = self.exif_extractor.extract_exif_data(file_path)
        orientation = exif_data.get('orientation', 1) if exif_data else 1
        if orientation != 1:
            thumbnail = self._apply_orientation_correction(thumbnail, orientation, exif_data)
        
        # 快取縮圖（優化：Resize & Encode to JPEG）
        # NOTE: Only cache the standard 512px version to keep the persistent cache predictable.
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

            w, h = pil_img.size
            if w <= MAX_THUMB_DIM and h <= MAX_THUMB_DIM:
                thumbnail_small = thumbnail
                thumbnail_small_pil = pil_img
            else:
                scale = min(MAX_THUMB_DIM / w, MAX_THUMB_DIM / h)
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                thumbnail_small_pil = pil_img.resize((new_w, new_h), Image.Resampling.HAMMING)
                thumbnail_small = np.array(thumbnail_small_pil)
            
            buffer = io.BytesIO()
            thumbnail_small_pil.save(buffer, format='JPEG', quality=85)
            jpeg_data = buffer.getvalue()
            
            self.cache.put_thumbnail(file_path, thumbnail_small, jpeg_data)
            return thumbnail_small
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error processing thumbnail with PIL: {e}")
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
            return self._process_regular_image(file_path)
    
    def _process_raw_image(self, file_path: str, 
                          use_full_resolution: bool = False,
                          executor: Optional[Any] = None) -> Optional[np.ndarray]:
        """處理 RAW 圖像"""
        libraw_first = use_libraw_consistent_preview_first()
        try:
            # 獲取 EXIF 數據（用於處理參數）
            exif_data = self.exif_extractor.extract_exif_data(file_path)
            
            # 檢查文件大小以決定處理策略
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            use_fast_processing = file_size_mb > 20 and not use_full_resolution
            
            # 處理 RAW 文件
            # Check if we should use Preview (embedded JPEG) instead of processing RAW
            if not use_full_resolution and not libraw_first:
               # OPTIMIZATION: Try to get/create a high-quality preview (e.g. 1920px) 
               # instead of processing the RAW data (even at half size, raw processing is slow/heavy)
               
               # Check Preview Cache
               cached_preview = self.cache.get_preview(file_path)
               if cached_preview is not None:
                   if _verbose_orientation_logs():
                       print(f"[PREVIEW] Using cached preview for {os.path.basename(file_path)}")
                   return cached_preview
               
               # Extract Preview
               if _verbose_orientation_logs():
                   print(f"[PREVIEW] Extracting preview from RAW for {os.path.basename(file_path)}")
               preview = self.thumbnail_extractor.extract_preview_from_raw(file_path, max_size=1920)
               
               if preview is not None:
                   # Check if preview is large enough for a good "Fit" view
                   # Many phone DNGs have tiny previews (e.g. 1024px) that look bad
                   h, w = preview.shape[:2]
                   if max(h, w) >= 1600:
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
            if executor:
                # logger = logging.getLogger(__name__)
                # logger.info(f"[PIL/PROCESS_POOL] Offloading RAW postprocess to pool for {file_path}")
                future = executor.submit(decode_raw_file, file_path, params)
                rgb_image = future.result()
            else:
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
            logger.error(f"Error processing RAW image {file_path}: {e}", exc_info=True)
            # LibRaw failed entirely; try byte-scan for embedded JPEG (already used in thumbnail
            # path, but preview may have been skipped or a different limit may help).
            try:
                from enhanced_raw_processor import extract_embedded_jpeg_by_scan
                lim = 8192 if use_full_resolution else 1920
                scanned = extract_embedded_jpeg_by_scan(file_path, lim)
                if scanned is not None:
                    exif_data = self.exif_extractor.extract_exif_data(file_path)
                    orientation = exif_data.get("orientation", 1) if exif_data else 1
                    if orientation != 1:
                        scanned = self._apply_orientation_correction(
                            scanned, orientation, exif_data
                        )
                    if not use_full_resolution and not libraw_first:
                        self.cache.put_preview(file_path, scanned)
                    return scanned
            except Exception:
                pass
            return None
    
    def _process_regular_image(self, file_path: str) -> Optional[QPixmap]:
        """處理常規圖像（JPEG/PNG/WebP）"""
        try:
            # 使用共用載入函數
            pixmap = load_pixmap_safe(file_path)
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

        # RAW Path: Open once, extract both
        import rawpy
        try:
            with rawpy.imread(file_path) as raw:
                # 1. Extract EXIF (verified with rawpy sensor sizes)
                exif = self.exif_extractor.extract_exif_data(file_path, raw_object=raw)
                
                # 2. Extract Thumbnail
                thumb = self.thumbnail_extractor.extract_thumbnail_from_raw(
                    file_path,
                    max_size=512,
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
        
        if orientation == 1:
            if os.environ.get("RAWVIEWER_VERBOSE_ORIENTATION_LOGS") == "1":
                # print(f"[ORIENTATION] Numpy operation: No operation (orientation = 1)")
                pass
            return image_array
            
        # SAFETY CHECK: If the image dimensions already match the "corrected" dimensions
        # (e.g. it's already a portrait image but EXIF still says Orientation 6), 
        # it might have been pre-rotated by the loader/camera.
        h, w = image_array.shape[:2]
        if os.environ.get("RAWVIEWER_VERBOSE_ORIENTATION_LOGS") == "1":
            # print(f"[ORIENTATION] UnifiedImageProcessor: Applying correction for Orientation {orientation} to {w}x{h} image")
            pass

        if orientation in (5, 6, 7, 8) and h > w:
            if os.environ.get("RAWVIEWER_VERBOSE_ORIENTATION_LOGS") == "1":
                # print(f"[ORIENTATION] UnifiedImageProcessor: Image is already portrait ({w}x{h}), skipping manual rotation for Orientation {orientation}")
                pass
            return image_array
        if orientation in (3, 4) and h < w:
            # For 180 flips, this is less certain, but we keep it as a placeholder
            pass
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


