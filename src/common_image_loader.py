"""
共用圖像載入函數 - 統一處理所有圖像類型的載入邏輯

這個模組提供統一的圖像載入函數，避免重複代碼。
"""

import os
from typing import Optional, Tuple, Any
from PyQt6.QtGui import QPixmap, QImage
# PIL Image will be imported lazily to avoid import delays

from image_cache import get_image_cache

def check_cache_for_image(file_path: str, use_full_resolution: bool = False) -> Tuple[Optional[Any], Optional[str]]:
    """
    檢查快取中是否存在圖像或相關數據。
    
    返回: (數據, 快取類型) 或 (None, None)
    類型包括: 'full_image', 'pixmap', 'thumbnail', 'exif'
    """
    cache = get_image_cache()
    
    # 1. 如果請求全解析度，優先檢查全圖像快取
    if use_full_resolution:
        full_image = cache.get_full_image(file_path)
        if full_image is not None:
            return full_image, 'full_image'
            
    # 2. 檢查 Pixmap 快取 (常用於 QSS/一般顯示)
    pixmap = cache.get_pixmap(file_path)
    if pixmap is not None and not pixmap.isNull():
        return pixmap, 'pixmap'
        
    # 3. 如果不是必須全解析度，檢查記憶體縮圖快取
    if not use_full_resolution:
        thumbnail = cache.get_thumbnail(file_path)
        if thumbnail is not None:
            return thumbnail, 'thumbnail'
            
    # 4. 檢查 EXIF 元數據快取
    exif_data = cache.get_exif(file_path)
    if exif_data:
        return exif_data, 'exif'
        
    return None, None

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
    
    # 對於 RAW 文件，嘗試從快取獲取尺寸，否則使用 rawpy 快速獲取
    if is_raw_file(file_path):
        # 優先嘗試 EXIF 快取（最快且包含方向）
        if exif_data:
            orig_w = exif_data.get('original_width')
            orig_h = exif_data.get('original_height')
            orient = exif_data.get('orientation', 1)
            if orig_w and orig_h and orig_h > 0:
                if orient in (5, 6, 7, 8):
                    return orig_h / orig_w
                return orig_w / orig_h
                
        try:
            import rawpy
            # 只讀取 Metadata，不處理圖像
            with rawpy.imread(file_path) as raw:
                sizes = raw.sizes
                w = sizes.width
                h = sizes.height
                
                # Consider orientation for RAW files
                # raw.sizes.flip is the orientation (same as EXIF)
                # 0: normal, 1: flip horizontal, 2: rotate 180, 3: flip vertical,
                # 4: mirror horizontal + rotate 270 CW, 5: rotate 90 CW, 
                # 6: mirror horizontal + rotate 90 CW, 7: rotate 270 CW
                
                # However, rawpy's flip mapping is slightly different or can be used with postprocess(user_flip=X)
                # For aspect ratio, we just need to know if it's 90/270 rotated.
                # raw.sizes.iwidth and iheight are the dimensions AFTER rotation if flip != 0?
                # Actually, raw.sizes.width/height is sensor.
                # Let's use the same logic as EXIF orientation if we can get it.
                
                # Try to use rawpy's own orientation (flip)
                flip = raw.sizes.flip
                if flip in (5, 6, 7): # 90 or 270 degree rotations in rawpy
                    if w > 0: return h / w
                
                if h > 0:
                    return w / h
                return 1.333
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
                width = size.width()
                height = size.height()
                
                # Check transformation to see if dimensions need swapping
                # QImageReader.size() returns the size *before* transformation in some versions,
                # so we need to manually swap if the transformation involves 90 degree rotation.
                transformation = reader.transformation()
                from PyQt6.QtGui import QImageIOHandler
                
                # Check for 90 or 270 degree rotation
                # QImageIOHandler.Transformation.TransformationRotate90 = 2
                # QImageIOHandler.Transformation.TransformationRotate270 = 4
                # We also need to consider mirrored versions if they involve 90 deg rotation
                # TransformationMirrorAndRotate90 = 6, TransformationMirrorAndRotate270 = 8 is not a standard enum value in older Qt?
                # Let's stick to the basic check:
                
                # Simple check: if orientation is 5, 6, 7, 8 (which correspond to rotations), we swap.
                # However, QImageReader.transformation() returns a Transformation enum.
                # TransformationRotate90 (2), TransformationRotate270 (4), 
                # TransformationMirrorAndRotate90 (6), TransformationFlipAndRotate90 (6?)
                
                t_val = transformation.value
                
                # Qt::ImageTransformation:
                # 0: None
                # 1: Mirror
                # 2: Flip
                # 3: Rotate180
                # 4: Rotate90
                # 5: MirrorAndRotate90
                # 6: FlipAndRotate90
                # 7: Rotate270
                
                # So 4, 5, 6, 7 imply 90 degree component (swapped dimensions)
                if t_val >= 4:
                    width, height = height, width
                    
                return width / height
        except Exception as e:
            # import logging
            # logging.getLogger(__name__).debug(f"Aspect ratio check failed: {e}")
            pass
    
    return 1.333  # 默認寬高比 (4:3)
