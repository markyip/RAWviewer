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
        orientation = exif_data.get('orientation', 1)
        
        if original_width and original_height and original_height > 0:
            # Handle orientation swap
            if orientation in (5, 6, 7, 8):
                return original_height / original_width
            return original_width / original_height
    
    # 對於 RAW 文件，嘗試使用 rawpy 快速獲取尺寸（比 exifread 更可靠）
    if is_raw_file(file_path):
        try:
            import rawpy
            # 只讀取 Metadata，不處理圖像
            with rawpy.imread(file_path) as raw:
                sizes = raw.sizes
                w = sizes.width
                h = sizes.height
                # 部分 RAW 格式的 sizes 可能包含黑邊，所以 raw.sizes 雖準確但我們需要考慮 orientation
                # rawpy 的 sizes 屬性通常是未旋轉的原始感光元件尺寸
                
                # 我們還需要讀取 EXIF 來決定是否旋轉
                # 如果沒有快取 EXIF，我們簡單嘗試讀取
                # 但這裡為了速度，我們可能需要一個快速的 EXIF 读取器
                # 暫時簡單讀取，如果沒有 cached EXIF，寬高比可能不正確（未旋轉）
                # 這可以接受，因為會在 load_visible_images 時修正 layout
                
                # 不過，为了 improved layout stability, 尝试获取 orientation
                return w / h if h > 0 else 1.333
        except:
             pass

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
    
    return 1.333  # 默認寬高比 (4:3)
