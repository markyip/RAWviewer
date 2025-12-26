"""
共用圖像載入函數 - 統一處理所有圖像類型的載入邏輯

這個模組提供統一的圖像載入函數，避免重複代碼。
"""

import os
from typing import Optional, Tuple
from PyQt6.QtGui import QPixmap, QImage
# PIL Image will be imported lazily to avoid import delays

from image_cache import get_image_cache


def is_raw_file(file_path: str) -> bool:
    """檢查是否為 RAW 文件"""
    raw_exts = {
        '.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef',
        '.srw', '.x3f', '.raf', '.3fr', '.fff', '.iiq', '.cap', '.erf',
        '.mef', '.mos', '.nrw', '.rwl', '.srf'
    }
    ext = os.path.splitext(file_path)[1].lower()
    return ext in raw_exts


def is_tiff_file(file_path: str) -> bool:
    """檢查是否為 TIFF 文件（包括錯誤擴展名的情況）"""
    file_ext = os.path.splitext(file_path)[1].lower()
    is_tiff = file_ext in ('.tiff', '.tif')
    
    # 檢查文件內容以檢測錯誤擴展名的 TIFF 文件
    if not is_tiff:
        try:
            from PIL import Image
            with Image.open(file_path) as test_img:
                if test_img.format in ('TIFF', 'TIF'):
                    is_tiff = True
        except:
            pass  # 不是 PIL 可讀文件或不是 TIFF
    
    return is_tiff


def load_pixmap_safe(file_path: str) -> QPixmap:
    """安全載入 QPixmap，對 TIFF 文件使用 PIL 以避免 Qt 警告"""
    cache = get_image_cache()
    
    # 檢查快取
    cached_pixmap = cache.get_pixmap(file_path)
    if cached_pixmap is not None and not cached_pixmap.isNull():
        return cached_pixmap
    
    # 對於 TIFF 文件，使用 PIL 避免 Qt 警告
    if is_tiff_file(file_path):
        try:
            from PIL import Image, ImageOps
            with Image.open(file_path) as pil_image:
                # Apply EXIF orientation correction
                pil_image = ImageOps.exif_transpose(pil_image)
                
                if pil_image.mode not in ('RGB', 'L'):
                    pil_image = pil_image.convert('RGB')
                
                width, height = pil_image.size
                if pil_image.mode == 'RGB':
                    qimage = QImage(pil_image.tobytes('raw', 'RGB'), 
                                   width, height, 
                                   QImage.Format.Format_RGB888)
                elif pil_image.mode == 'L':
                    qimage = QImage(pil_image.tobytes('raw', 'L'), 
                                   width, height, 
                                   QImage.Format.Format_Grayscale8)
                else:
                    rgb_pil = pil_image.convert('RGB')
                    qimage = QImage(rgb_pil.tobytes('raw', 'RGB'), 
                                   width, height, 
                                   QImage.Format.Format_RGB888)
                
                if not qimage.isNull():
                    pixmap = QPixmap.fromImage(qimage)
                    cache.put_pixmap(file_path, pixmap)
                    return pixmap
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"PIL fallback failed for TIFF: {os.path.basename(file_path)}: {e}")
            # 對於 TIFF 文件，永遠不要使用 QPixmap(file_path)，因為會觸發警告
            return QPixmap()
    
    # 對於其他格式（JPEG, PNG, etc.），使用 QImageReader 並啟用自動方向轉換
    try:
        from PyQt6.QtGui import QImageReader
        reader = QImageReader(file_path)
        reader.setAutoTransform(True)  # 自動應用 EXIF 方向
        pixmap = QPixmap.fromImageReader(reader)
        if not pixmap.isNull():
            cache.put_pixmap(file_path, pixmap)
            return pixmap
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"QImageReader failed for {os.path.basename(file_path)}: {e}")
    
    # 回退到直接使用 QPixmap（不會應用方向，但比沒有好）
    pixmap = QPixmap(file_path)
    if not pixmap.isNull():
        cache.put_pixmap(file_path, pixmap)
    return pixmap


def get_image_aspect_ratio(file_path: str) -> float:
    """獲取圖像寬高比（不載入完整圖像）"""
    cache = get_image_cache()
    
    # 嘗試從 EXIF 快取獲取尺寸
    exif_data = cache.get_exif(file_path)
    if exif_data:
        original_width = exif_data.get('original_width')
        original_height = exif_data.get('original_height')
        if original_width and original_height and original_height > 0:
            return original_width / original_height
    
    # 對於 TIFF 文件，使用 PIL
    if is_tiff_file(file_path):
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                width, height = img.size
                if height > 0:
                    return width / height
        except:
            pass
    
    # 對於其他格式，嘗試 QImageReader（但不包括 RAW 和 TIFF）
    if not is_raw_file(file_path):
        try:
            from PyQt6.QtGui import QImageReader
            reader = QImageReader(file_path)
            # Enable automatic EXIF orientation transformation
            reader.setAutoTransform(True)
            size = reader.size()
            if size.isValid() and size.height() > 0:
                return size.width() / size.height()
        except:
            pass
    
    return 1.0  # 默認寬高比


def check_cache_for_image(file_path: str, use_full_resolution: bool = False) -> Tuple[Optional[object], str]:
    """
    統一檢查快取
    
    返回: (cached_item, cache_type)
    cache_type: 'full_image', 'pixmap', 'thumbnail', 'exif', 或 None
    """
    cache = get_image_cache()
    
    # 檢查完整圖像快取
    if use_full_resolution:
        cached_image = cache.get_full_image(file_path)
        if cached_image is not None:
            return cached_image, 'full_image'
    
    # 檢查 QPixmap 快取（用於非 RAW 文件）
    cached_pixmap = cache.get_pixmap(file_path)
    if cached_pixmap is not None and not cached_pixmap.isNull():
        return cached_pixmap, 'pixmap'
    
    # 檢查縮圖快取
    cached_thumbnail = cache.get_thumbnail(file_path)
    if cached_thumbnail is not None:
        return cached_thumbnail, 'thumbnail'
    
    # 檢查 EXIF 快取
    cached_exif = cache.get_exif(file_path)
    if cached_exif is not None:
        return cached_exif, 'exif'
    
    return None, None

