"""
統一圖像處理器 - 處理所有圖像類型（RAW、JPEG、PNG等）

這個模組實現了重構提案中的 UnifiedImageProcessor，提供統一的接口
來處理所有圖像類型，內置快取檢查和錯誤處理。
"""

import os
import numpy as np
from typing import Optional, Dict, Any, Union
from PyQt6.QtGui import QPixmap, QImage
# PIL Image, rawpy, and exifread will be imported lazily to avoid import delays

from image_cache import get_image_cache
from enhanced_raw_processor import (
    ThumbnailExtractor, 
    EXIFExtractor, 
    OptimizedRAWProcessor
)
from common_image_loader import is_raw_file, is_tiff_file, load_pixmap_safe


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
    
    def process_thumbnail(self, file_path: str) -> Optional[np.ndarray]:
        """處理縮圖（統一接口）"""
        # 檢查快取
        cached = self.cache.get_thumbnail(file_path)
        if cached is not None:
            return cached
        
        # 提取縮圖
        if self._is_raw_file(file_path):
            thumbnail = self.thumbnail_extractor.extract_thumbnail_from_raw(file_path)
        else:
            thumbnail = self.thumbnail_extractor.extract_thumbnail_from_image(file_path)
        
        # 應用方向校正到縮圖
        if thumbnail is not None:
            # 獲取 EXIF 數據以獲取 orientation
            exif_data = self.exif_extractor.extract_exif_data(file_path)
            orientation = exif_data.get('orientation', 1) if exif_data else 1
            if orientation != 1:
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"[UNIFIED_PROC] Applying orientation correction {orientation} to thumbnail for {os.path.basename(file_path)}")
                thumbnail = self._apply_orientation_correction(thumbnail, orientation, exif_data)
        
        # 快取縮圖（已應用 orientation）
        if thumbnail is not None:
            self.cache.put_thumbnail(file_path, thumbnail)
        
        return thumbnail
    
    def process_full_image(self, file_path: str, 
                          use_full_resolution: bool = False) -> Optional[Union[np.ndarray, QPixmap]]:
        """處理完整圖像（統一接口）"""
        # 檢查快取
        if use_full_resolution:
            cached_image = self.cache.get_full_image(file_path)
            if cached_image is not None:
                return cached_image
        
        cached_pixmap = self.cache.get_pixmap(file_path)
        if cached_pixmap is not None:
            return cached_pixmap
        
        # 處理圖像
        if self._is_raw_file(file_path):
            return self._process_raw_image(file_path, use_full_resolution)
        else:
            return self._process_regular_image(file_path)
    
    def _process_raw_image(self, file_path: str, 
                          use_full_resolution: bool = False) -> Optional[np.ndarray]:
        """處理 RAW 圖像"""
        try:
            # 獲取 EXIF 數據（用於處理參數）
            exif_data = self.exif_extractor.extract_exif_data(file_path)
            
            # 檢查文件大小以決定處理策略
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            use_fast_processing = file_size_mb > 20 and not use_full_resolution
            
            # 處理 RAW 文件
            import rawpy
            with rawpy.imread(file_path) as raw:
                # 獲取處理參數
                params = self.raw_processor.get_optimized_processing_params(
                    file_path, exif_data
                )
                
                # 根據需求調整參數
                if use_fast_processing:
                    params['half_size'] = True
                
                if use_full_resolution:
                    params['half_size'] = False
                    params['output_bps'] = 8  # 8-bit 足夠顯示
                
                # 處理 RAW
                rgb_image = raw.postprocess(**params)
                
                # 應用方向校正
                orientation = exif_data.get('orientation', 1) if exif_data else 1
                import logging
                logger = logging.getLogger(__name__)
                file_basename = os.path.basename(file_path)
                original_shape = rgb_image.shape if rgb_image is not None else None
                logger.info(f"[UNIFIED_PROC] Processing file: {file_basename}, Original shape: {original_shape}, Orientation: {orientation}")
                if orientation != 1:
                    logger.info(f"[UNIFIED_PROC] Applying orientation correction: {orientation}")
                    rgb_image = self._apply_orientation_correction(rgb_image, orientation, exif_data)
                    final_shape = rgb_image.shape if rgb_image is not None else None
                    logger.info(f"[UNIFIED_PROC] File: {file_basename}, Orientation correction applied, Final shape: {final_shape}")
                else:
                    logger.debug(f"[UNIFIED_PROC] File: {file_basename}, Orientation is 1 (normal), no correction needed")
                
                # 快取完整圖像
                if rgb_image is not None:
                    self.cache.put_full_image(file_path, rgb_image)
                
                return rgb_image
                
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error processing RAW image {file_path}: {e}", exc_info=True)
            return None
    
    def _process_regular_image(self, file_path: str) -> Optional[QPixmap]:
        """處理常規圖像（JPEG/PNG/WebP）"""
        try:
            # 使用共用載入函數
            pixmap = load_pixmap_safe(file_path)
            return pixmap if not pixmap.isNull() else None
                
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error processing regular image {file_path}: {e}", exc_info=True)
            return None
    
    def process_exif(self, file_path: str) -> Optional[Dict[str, Any]]:
        """處理 EXIF 數據（統一接口）"""
        # 檢查快取
        cached_exif = self.cache.get_exif(file_path)
        if cached_exif is not None:
            return cached_exif
        
        # 提取 EXIF 數據
        exif_data = self.exif_extractor.extract_exif_data(file_path)
        
        # 快取 EXIF 數據
        if exif_data:
            self.cache.put_exif(file_path, exif_data)
        
        return exif_data
    
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
        if orientation == 1:
            return image_array
        elif orientation == 2:
            # Mirror left-right
            return np.fliplr(image_array)
        elif orientation == 3:
            # Rotate 180°
            return np.rot90(image_array, 2)
        elif orientation == 4:
            # Mirror top-bottom
            return np.flipud(image_array)
        elif orientation == 5:
            # Mirror LR + rotate 270° CW (k=1 CCW)
            return np.rot90(np.fliplr(image_array), 1)
        elif orientation == 6:
            # Rotate 90° CW (k=3 CCW)
            return np.rot90(image_array, 3)
        elif orientation == 7:
            # Mirror LR + rotate 90° CW
            return np.rot90(np.fliplr(image_array), 3)
        elif orientation == 8:
            # Rotate 270° CW (90° CCW) - need to rotate 90° CW to correct
            # np.rot90 with k=3 rotates 270° CCW = 90° CW
            return np.rot90(image_array, 1)
        
        # Unknown orientation
        return image_array


