
"""Common image loader: QImage/QPixmap/PIL/HDR/EDR decode entry points."""

from __future__ import annotations

import io
import os

import numpy as np
from PyQt6.QtCore import QLoggingCategory, QSize
from PyQt6.QtGui import QImage, QPixmap

from image_cache import get_image_cache

# Silence qt.imageformats warnings (e.g. missing TIFF tag warnings on RAW files)
QLoggingCategory.setFilterRules("qt.imageformats.warning=false\\nqt.imageformats.tiff.warning=false")
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

import contextlib

from .edr import (
    hlg_to_linear,
    is_macos_edr_enabled,
    linear_to_srgb,
    pq_to_linear,
    tone_map_reinhard,
)
from .exif_display import (
    exif_display_aspect_ratio,
    finalize_index_thumbnail_array,
    mark_exif_pixels_display_oriented,
)
from .helpers import (
    _regular_image_max_edge,
    is_raw_file,
    is_tiff_file,
)


def decode_embedded_jpeg_bytes(
    jpeg_bytes: bytes,
    max_size: int = 0,
) -> np.ndarray | None:
    """Decode embedded JPEG to RGB; applies the JPEG segment's own EXIF orientation."""
    import logging
    logger = logging.getLogger(__name__)
    try:
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        im = Image.open(io.BytesIO(jpeg_bytes))
        # Fast path: JPEG DCT-domain scaled decode (1/2, 1/4, 1/8). Decoding a
        # 30MP+ embedded preview at reduced scale instead of full size then
        # downsizing dominates both gallery (~512) and single-view display-tier
        # (~1920ΓÇô2304) cost. draft() never upsizes and keeps pixel orientation
        # identical to a full decode. Kill-switch: RAWVIEWER_JPEG_DRAFT_DECODE=0.
        draft_cap = 1024
        try:
            from image_cache import memory_preview_max_edge

            draft_cap = max(1024, int(memory_preview_max_edge()))
        except Exception:
            draft_cap = 2304
        if (
            0 < max_size <= draft_cap
            and im.format == "JPEG"
            and max(im.size) > max_size * 2
            and os.environ.get("RAWVIEWER_JPEG_DRAFT_DECODE", "1").strip().lower()
            not in {"0", "false", "no", "off"}
        ):
            with contextlib.suppress(Exception):
                im.draft("RGB", (max_size, max_size))
        if im.mode != "RGB":
            im = im.convert("RGB")
        w, h = im.size
        if max_size > 0 and (w > max_size or h > max_size):
            im.thumbnail((max_size, max_size), Image.Resampling.HAMMING)
        return np.array(im)
    except Exception as e:
        logger.error("[DECODE] Error decoding embedded JPEG bytes: %s", e)
        return None
def orientation_from_embedded_jpeg_bytes(jpeg_bytes: bytes) -> int:
    """EXIF Orientation (1ΓÇô8) from an embedded JPEG; portrait pixels ΓåÆ 6 if tag missing."""
    try:
        from PIL import Image

        im = Image.open(io.BytesIO(jpeg_bytes))
        exif = im.getexif()
        if exif:
            o = exif.get(274)
            if o is None:
                o = exif.get(0x0112)
            if isinstance(o, int) and 1 <= o <= 8:
                return o
        w, h = im.size
        if h > w:
            return 6
    except Exception:
        pass
    return 1
def load_qimage_safe(file_path: str, max_edge: int = 0):
    """QImage twin of load_pixmap_safe, for WORKER THREADS.

    QPixmap is GUI-thread-only in Qt -- it is backed by platform paint
    resources -- but ImageLoadWorker reaches load_pixmap_safe through
    process_full_image -> _process_regular_image for every non-RAW file, so
    a QPixmap was being constructed off the GUI thread on every JPEG, PNG,
    WebP and TIFF, then handed across via pixmap_ready. QImage has no such
    restriction: it is plain pixel data, safe to build anywhere and cheap to
    convert once it reaches the GUI thread.

    Deliberately narrower than load_pixmap_safe: QImageReader for the common
    formats, PIL for TIFF and as the fallback. HDR/EDR paths are NOT
    replicated -- they set module state about the last load and belong on the
    GUI thread -- so this returns a null QImage for anything it cannot handle
    plainly, and the caller decides what to do.

    Returns a null QImage on failure; never raises.
    """
    from PyQt6.QtGui import QImageReader

    try:
        if is_tiff_file(file_path):
            return _pil_file_to_qimage(file_path, max_edge=max_edge)

        reader = QImageReader(file_path)
        reader.setAutoTransform(True)
        if max_edge > 0:
            size = reader.size()
            w, h = size.width(), size.height()
            if w > 0 and h > 0 and max(w, h) > max_edge:
                if w >= h:
                    sw, sh = max_edge, max(1, int(h * max_edge / max(w, 1)))
                else:
                    sh, sw = max_edge, max(1, int(w * max_edge / max(h, 1)))
                reader.setScaledSize(QSize(sw, sh))
        img = reader.read()
        if img is not None and not img.isNull():
            return img
    except Exception:
        pass
    try:
        return _pil_file_to_qimage(file_path, max_edge=max_edge)
    except Exception:
        return QImage()
def _pil_file_to_qimage(file_path: str, max_edge: int = 0):
    """PIL load -> QImage, orientation applied. The QPixmap-free half of
    _pil_file_to_qpixmap, so worker threads can use it."""
    from PIL import Image, ImageFile, ImageOps

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = 200_000_000
    with Image.open(file_path) as pil_image:
        pil_image = ImageOps.exif_transpose(pil_image)
        if max_edge > 0:
            w, h = pil_image.size
            if max(w, h) > max_edge:
                pil_image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        width, height = pil_image.size
        import numpy as np

        arr = np.ascontiguousarray(np.asarray(pil_image))
        qimage = QImage(
            arr.data, width, height, width * 3, QImage.Format.Format_RGB888
        )
        # copy(): the QImage above borrows the numpy buffer, and this one
        # crosses a thread boundary where the array's lifetime is not ours.
        return qimage.copy()
def _pil_file_to_qpixmap(file_path: str, max_edge: int = 0) -> QPixmap:
    """Load JPEG/PNG/WebP via PIL with EXIF orientation; optional downscale."""
    import logging

    logger = logging.getLogger(__name__)
    try:
        from PIL import Image, ImageFile, ImageOps

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        Image.MAX_IMAGE_PIXELS = 200_000_000
        with Image.open(file_path) as pil_image:
            pil_image = ImageOps.exif_transpose(pil_image)
            if max_edge > 0:
                w, h = pil_image.size
                if max(w, h) > max_edge:
                    pil_image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            if pil_image.mode not in ("RGB", "L"):
                pil_image = pil_image.convert("RGB")
            if pil_image.mode == "L":
                pil_image = pil_image.convert("RGB")
            width, height = pil_image.size
            import numpy as np
            arr = np.asarray(pil_image)
            qimage = QImage(
                arr.data,
                width,
                height,
                width * 3,
                QImage.Format.Format_RGB888,
            )
            qimage.ndarr = arr  # Keep array alive
            if qimage.isNull():
                return QPixmap()
            return QPixmap.fromImage(qimage)
    except Exception as e:
        logger.debug(
            "PIL load failed for %s: %s", os.path.basename(file_path), e
        )
        return QPixmap()
def pixmap_covers_requested_edge(file_path: str, pixmap, max_edge: int) -> bool:
    """True when a cached ``pixmap`` satisfies a load request of ``max_edge``.

    ``max_edge <= 0`` means "native resolution": the pixmap must reach the
    file's header-probed dimensions (compared sorted, so an EXIF-rotated
    pixmap still matches). A pixmap at/above the app's own safety cap
    (_regular_image_max_edge) is always accepted -- it is the largest this
    app will ever produce, so rejecting it would re-run an expensive decode
    into the same result. Unknown native size accepts the cache.

    Without this check, the fit view's preview-capped pixmap (same cache key)
    satisfied every later full-resolution request: a 32888x8470 panorama
    cached at 2304px was re-delivered forever, so zooming to 100% showed a
    blurry upscale and the display never upgraded.
    """
    try:
        have = sorted((int(pixmap.width()), int(pixmap.height())))
        if have[1] >= _regular_image_max_edge():
            return True
        if max_edge > 0 and have[1] >= max_edge - 2:
            # Fast accept without a header probe -- the common (gallery/fit)
            # case stays as cheap as the unconditional hit it replaces.
            return True
        # Cached pixmap is smaller than the request: it still satisfies it if
        # the FILE is just natively small (never force a redecode that cannot
        # produce more pixels). One header-only probe, no pixel decode.
        from PyQt6.QtGui import QImageReader

        sz = QImageReader(file_path).size()
        if not sz.isValid() or sz.width() <= 0 or sz.height() <= 0:
            return True
        need = sorted((int(sz.width()), int(sz.height())))
        if max_edge > 0:
            scale = min(1.0, max_edge / max(need[1], 1))
            need = [int(need[0] * scale), int(need[1] * scale)]
        return have[0] >= need[0] - 2 and have[1] >= need[1] - 2
    except Exception:
        return True
def load_pixmap_safe(file_path: str, max_edge: int = 0) -> QPixmap:
    """σ«ëσà¿Φ╝ëσàÑ QPixmap∩╝îσ░ì TIFF µûçΣ╗╢Σ╜┐τö¿ PIL Σ╗ÑΘü┐σàì Qt Φ¡ªσæè"""
    global _last_load_used_edr
    _last_load_used_edr = False
    cache = get_image_cache()

    # µ¬óµƒÑσ┐½σÅû (size-aware: a preview-capped entry must not satisfy a
    # larger/native-resolution request -- see pixmap_covers_requested_edge)
    cached_pixmap = cache.get_pixmap(file_path)
    if cached_pixmap is not None and not cached_pixmap.isNull() and pixmap_covers_requested_edge(file_path, cached_pixmap, max_edge):
            return cached_pixmap

    if is_raw_file(file_path):
        preview_max = max_edge if max_edge > 0 else 2048
        return load_raw_preview_pixmap(file_path, max_size=preview_max)

    # Check if this is an HDR HEIF/AVIF or TIFF
    hdr_pixmap = try_load_hdr_image_pixmap(file_path, max_edge=max_edge)
    if hdr_pixmap is not None:
        _last_load_used_edr = is_macos_edr_enabled()
        cache.put_pixmap(file_path, hdr_pixmap)
        return hdr_pixmap
    # σ░ìµû╝ TIFF µûçΣ╗╢∩╝îΣ╜┐τö¿ PIL Θü┐σàì Qt Φ¡ªσæè
    if is_tiff_file(file_path):
        try:
            from PIL import Image, ImageOps
            with Image.open(file_path) as pil_image:
                # Apply EXIF orientation correction
                pil_image = ImageOps.exif_transpose(pil_image)

                if pil_image.mode not in ('RGB', 'L'):
                    pil_image = pil_image.convert('RGB')
                if pil_image.mode == 'L':
                    pil_image = pil_image.convert('RGB')

                width, height = pil_image.size
                import numpy as np
                arr = np.asarray(pil_image)
                qimage = QImage(arr.data,
                               width, height, width * 3,
                               QImage.Format.Format_RGB888)
                qimage.ndarr = arr  # Keep array alive

                if not qimage.isNull():
                    pixmap = QPixmap.fromImage(qimage)
                    cache.put_pixmap(file_path, pixmap)
                    return pixmap
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"PIL fallback failed for TIFF: {os.path.basename(file_path)}: {e}")
            # σ░ìµû╝ TIFF µûçΣ╗╢∩╝îµ░╕ΘüáΣ╕ìΦªüΣ╜┐τö¿ QPixmap(file_path)∩╝îσ¢áτé║µ£âΦº╕τÖ╝Φ¡ªσæè
            return QPixmap()

    # σ░ìµû╝σà╢Σ╗ûµá╝σ╝Å∩╝êJPEG, PNG, etc.∩╝ë∩╝îΣ╜┐τö¿ QImageReader Σ╕ªσòƒτö¿Φç¬σïòµû╣σÉæΦ╜ëµÅ¢
    try:
        from PyQt6.QtCore import QSize
        from PyQt6.QtGui import QImageReader

        reader = QImageReader(file_path)
        reader.setAutoTransform(True)  # Φç¬σïòµçëτö¿ EXIF µû╣σÉæ
        if max_edge > 0:
            original_size = reader.size()
            if original_size.isValid():
                w, h = original_size.width(), original_size.height()
                if max(w, h) > max_edge:
                    if w >= h:
                        sw = max_edge
                        sh = max(1, int(h * max_edge / max(w, 1)))
                    else:
                        sh = max_edge
                        sw = max(1, int(w * max_edge / max(h, 1)))
                    reader.setScaledSize(QSize(sw, sh))
        pixmap = QPixmap.fromImageReader(reader)
        if not pixmap.isNull():
            try:
                cached_exif = cache.get_exif(file_path)
                if cached_exif and int(cached_exif.get("orientation", 1) or 1) != 1:
                    cache.put_exif(file_path, mark_exif_pixels_display_oriented(cached_exif))
            except Exception:
                pass
            cache.put_pixmap(file_path, pixmap)
            return pixmap
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"QImageReader failed for {os.path.basename(file_path)}: {e}")

    pixmap = _pil_file_to_qpixmap(file_path, max_edge=max_edge)
    if not pixmap.isNull():
        cache.put_pixmap(file_path, pixmap)
        return pixmap

    # σ¢₧ΘÇÇσê░τ¢┤µÄÑΣ╜┐τö¿ QPixmap∩╝êΣ╕ìµ£âµçëτö¿µû╣σÉæ∩╝îΣ╜åµ»öµ▓Æµ£ëσÑ╜∩╝ë
    pixmap = QPixmap(file_path)
    if not pixmap.isNull():
        cache.put_pixmap(file_path, pixmap)
        return pixmap

    if max_edge <= 0:
        safety_edge = _regular_image_max_edge()
        pixmap = _pil_file_to_qpixmap(file_path, max_edge=safety_edge)
        if not pixmap.isNull():
            cache.put_pixmap(file_path, pixmap)
    return pixmap
def rgb_array_to_qpixmap(rgb_image: np.ndarray) -> QPixmap:
    """Convert oriented uint8 RGB/grayscale ndarray to QPixmap.

    Builds the QImage from the array's own buffer (no .tobytes() copy):
    QPixmap.fromImage() below deep-copies pixel data into the QPixmap's own
    native storage before this function returns, so the QImage never
    outlives rgb_image's scope -- safe to reference directly rather than
    paying for an extra full-frame copy on every conversion (this path runs
    on every gallery/thumbnail/display repaint).
    """
    if rgb_image is None or not hasattr(rgb_image, "shape"):
        return QPixmap()
    if rgb_image.dtype != np.uint8:
        rgb_image = rgb_image.astype(np.uint8, copy=False)
    if not rgb_image.flags["C_CONTIGUOUS"]:
        rgb_image = np.ascontiguousarray(rgb_image)
    h, w = rgb_image.shape[:2]
    if len(rgb_image.shape) == 2:
        qimg = QImage(
            rgb_image.data, w, h, w, QImage.Format.Format_Grayscale8
        )
    elif rgb_image.shape[2] >= 3:
        channels = int(rgb_image.shape[2])
        q_format = (
            QImage.Format.Format_RGB888
            if channels == 3
            else QImage.Format.Format_RGBA8888
        )
        qimg = QImage(
            rgb_image.data, w, h, channels * w, q_format
        )
    else:
        return QPixmap()

    qimg.ndarr = rgb_image  # Keep array alive explicitly
    return QPixmap.fromImage(qimg) if not qimg.isNull() else QPixmap()
def load_raw_preview_array(
    file_path: str,
    max_size: int = 2048,
    *,
    apply_container_orientation: bool = True,
) -> np.ndarray | None:
    """Single RAW preview path: cache ΓåÆ ThumbnailExtractor ΓåÆ container EXIF (once)."""
    cache = get_image_cache()
    for getter in (cache.get_thumbnail, cache.get_preview):
        try:
            cached = getter(file_path)
        except Exception:
            cached = None
        if cached is not None and hasattr(cached, "shape"):
            arr = np.asarray(cached, dtype=np.uint8)
            if is_raw_file(file_path):
                return finalize_index_thumbnail_array(file_path, arr, cache=cache)
            return arr

    from enhanced_raw_processor import ThumbnailExtractor

    thumb = ThumbnailExtractor().extract_thumbnail_from_raw(
        file_path, max_size=max_size, allow_scan_fallback=True
    )
    if thumb is None:
        return None

    if hasattr(thumb, "width"):
        from enhanced_raw_processor import _qimage_to_rgb_array

        thumb = _qimage_to_rgb_array(thumb)
    if thumb is None or not isinstance(thumb, np.ndarray):
        return None

    if not apply_container_orientation:
        return thumb

    return finalize_index_thumbnail_array(file_path, thumb, cache=cache)
def load_raw_preview_pixmap(file_path: str, max_size: int = 2048) -> QPixmap:
    """Oriented RAW preview as QPixmap (shared by single view fallbacks and gallery)."""
    cache = get_image_cache()
    cached_pixmap = cache.get_pixmap(file_path)
    if cached_pixmap is not None and not cached_pixmap.isNull():
        return cached_pixmap

    arr = load_raw_preview_array(file_path, max_size=max_size)
    if arr is None:
        return QPixmap()
    pixmap = rgb_array_to_qpixmap(arr)
    if not pixmap.isNull():
        cache.put_pixmap(file_path, pixmap)
    return pixmap
def get_image_aspect_ratio(file_path: str) -> float:
    """τì▓σÅûσ£ûσâÅσ»¼Θ½ÿµ»ö∩╝êΣ╕ìΦ╝ëσàÑσ«îµò┤σ£ûσâÅ∩╝ë"""
    cache = get_image_cache()

    # σÿùΦ⌐ªσ╛₧ EXIF σ┐½σÅûτì▓σÅûσ░║σ»╕
    exif_data = cache.get_exif(file_path)
    if exif_data:
        original_width = exif_data.get('original_width')
        original_height = exif_data.get('original_height')
        orientation = exif_data.get('orientation', 1)

        if original_width and original_height and original_height > 0:
            return exif_display_aspect_ratio(
                int(original_width), int(original_height), int(orientation or 1)
            )

    # σ░ìµû╝ RAW µûçΣ╗╢∩╝îσÿùΦ⌐ªσ╛₧σ┐½σÅûτì▓σÅûσ░║σ»╕∩╝îσÉªσëçΣ╜┐τö¿ rawpy σ┐½ΘÇƒτì▓σÅû
    if is_raw_file(file_path):
        # σä¬σàêσÿùΦ⌐ª EXIF σ┐½σÅû∩╝êµ£Çσ┐½Σ╕öσîàσÉ½µû╣σÉæ∩╝ë
        if exif_data:
            orig_w = exif_data.get('original_width')
            orig_h = exif_data.get('original_height')
            orient = exif_data.get('orientation', 1)
            if orig_w and orig_h and orig_h > 0:
                return exif_display_aspect_ratio(
                    int(orig_w), int(orig_h), int(orient or 1)
                )

        try:
            # σÅ¬Φ«ÇσÅû Metadata∩╝îΣ╕ìΦÖòτÉåσ£ûσâÅ
            # Lazy: a module-level import of processing.* creates a cycle.
            from processing.raw_decode.gil import rawpy_imread_warm

            with rawpy_imread_warm(file_path) as raw:
                sizes = raw.sizes
                # Use iwidth and iheight as they already account for orientation (rotation)
                # sizes.iwidth and iheight are internal dimensions after rotation
                # HOWEVER, for absolute safety with all rawpy versions,
                # let's check flip and swap sensor dimensions manually if needed.
                w = sizes.width
                h = sizes.height
                flip = sizes.flip

                if flip in (5, 6, 7, 8):
                    w, h = h, w

                if h > 0:
                    return w / h
                return 1.333
        except Exception:
             pass

    # σ░ìµû╝ TIFF µûçΣ╗╢∩╝îΣ╜┐τö¿ PIL
    if is_tiff_file(file_path):
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                width, height = img.size
                if height > 0:
                    return width / height
        except Exception:
            pass

    # σ░ìµû╝σà╢Σ╗ûµá╝σ╝Å∩╝îσÿùΦ⌐ª QImageReader∩╝êΣ╜åΣ╕ìσîàµï¼ RAW σÆî TIFF∩╝ë
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
        except Exception:
            # import logging
            # logging.getLogger(__name__).debug(f"Aspect ratio check failed: {e}")
            pass

    return 1.333  # Θ╗ÿΦ¬ìσ»¼Θ½ÿµ»ö (4:3)
def try_load_hdr_image_pixmap(file_path: str, max_edge: int = 0) -> QPixmap | None:
    """Try to decode HEIF/AVIF/TIFF HDR images with EDR on macOS or SDR tone-mapping."""
    import os

    from PIL import Image

    ext = file_path.lower()

    is_heif_or_avif = ext.endswith(('.heic', '.heif', '.avif'))
    is_tiff = ext.endswith(('.tif', '.tiff'))

    if not (is_heif_or_avif or is_tiff):
        return None

    try:
        bit_depth = 8
        transfer = 2
        icc_profile = b""
        arr = None
        channels = 3
        h, w = 0, 0

        if is_heif_or_avif:
            import pillow_heif
            heif_file = pillow_heif.read_heif(file_path)
            bit_depth = heif_file.info.get("bit_depth", 8)
            nclx = heif_file.info.get("nclx", {})
            if nclx:
                transfer = nclx.get("transfer_characteristics", 2)
            icc_profile = heif_file.info.get("icc_profile", b"")
            w, h = heif_file.size
            channels = 4 if heif_file.mode == 'RGBA' else 3

            # Check if HDR
            is_hdr = (bit_depth > 8) or (transfer in (16, 18))
            if not is_hdr:
                return None

            if bit_depth > 8:
                arr = np.frombuffer(heif_file.data, dtype=np.uint16)
                arr = arr.reshape((h, w, channels))
            else:
                arr = np.frombuffer(heif_file.data, dtype=np.uint8)
                arr = arr.reshape((h, w, channels))

        elif is_tiff:
            with Image.open(file_path) as pil_img:
                # Check BitsPerSample tag 258
                bits = pil_img.tag.get(258)
                is_16bit = False
                if bits:
                    is_16bit = any(b > 8 for b in bits) if isinstance(bits, tuple) else bits > 8
                if not is_16bit:
                    is_16bit = pil_img.mode in ("I;16", "I;16L", "I;16B", "RGBA;16", "RGB;16")

                if not is_16bit:
                    return None
                icc_profile = pil_img.info.get("icc_profile", b"")
                w, h = pil_img.size

            import cv2
            arr = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if arr is None:
                return None
            bit_depth = 16

            if len(arr.shape) == 3:
                channels = arr.shape[2]
                if channels == 3:
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                elif channels == 4:
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
            else:
                channels = 1

        if arr is None:
            return None

        # Determine the [0,1] float denominator from the FULL-resolution
        # array before any downsizing. INTER_AREA block-averaging can only
        # ever pull arr.max() down (never up), so computing this heuristic
        # AFTER a downsize can misjudge a true-16-bit image (native peak just
        # above 4095) as 10/12-bit if its bright content is sparse enough to
        # get averaged below the threshold -- picking a denominator dozens of
        # times too small and rendering the preview badly over-bright.
        max_val = (65535.0 if arr.max() > 4095 else 2 ** bit_depth - 1) if arr.dtype == np.uint16 else 255.0

        # Perform downsizing early to save memory and processing time
        if max_edge > 0 and max(w, h) > max_edge:
            import cv2
            if w >= h:
                sw = max_edge
                sh = max(1, int(h * max_edge / max(w, 1)))
            else:
                sh = max_edge
                sw = max(1, int(w * max_edge / max(h, 1)))
            arr = cv2.resize(arr, (sw, sh), interpolation=cv2.INTER_AREA)
            w, h = sw, sh
            if len(arr.shape) == 2:
                arr = arr.reshape((h, w, 1))

        # Determine float representation [0.0, 1.0]
        img_float = arr.astype(np.float32) / max_val

        from PyQt6.QtGui import QColorSpace

        # 1. macOS Native EDR path
        if is_macos_edr_enabled():
            # Pad to 4 channels for QImage RGBX64 or RGBA64
            if channels == 3:
                padded = np.empty((h, w, 4), dtype=np.uint16)
                if arr.dtype == np.uint16:
                    padded[:, :, :3] = arr
                else:
                    padded[:, :, :3] = (img_float[:, :, :3] * 65535.0).astype(np.uint16)
                padded[:, :, 3] = 65535
                arr_64 = padded
                q_format = QImage.Format.Format_RGBX64
            elif channels == 4:
                arr_64 = arr if arr.dtype == np.uint16 else (img_float * 65535.0).astype(np.uint16)
                q_format = QImage.Format.Format_RGBA64
            else:  # Grayscale -> RGBX64
                padded = np.empty((h, w, 4), dtype=np.uint16)
                val_16 = (img_float[:, :, 0] * 65535.0).astype(np.uint16) if len(img_float.shape) == 3 else (img_float * 65535.0).astype(np.uint16)
                padded[:, :, 0] = val_16
                padded[:, :, 1] = val_16
                padded[:, :, 2] = val_16
                padded[:, :, 3] = 65535
                arr_64 = padded
                q_format = QImage.Format.Format_RGBX64

            arr_64 = np.ascontiguousarray(arr_64)
            qimage = QImage(
                arr_64.data,
                w,
                h,
                w * 8,
                q_format
            )
            qimage.ndarr = arr_64  # Keep array alive

            # Set the color space so Quartz/Metal rendering will treat values correctly
            if icc_profile:
                cs = QColorSpace.fromIccProfile(icc_profile)
                if cs.isValid():
                    qimage.setColorSpace(cs)
            else:
                # Fallback: assume standard PQ/HLG space
                if transfer == 16:
                    # PQ
                    qimage.setColorSpace(QColorSpace(QColorSpace.NamedColorSpace.SRgb))
                else:
                    # default sRGB
                    qimage.setColorSpace(QColorSpace(QColorSpace.NamedColorSpace.SRgb))

        # 2. Windows / SDR Tone-mapping path
        else:
            rgb_float = img_float[:, :, :3] if channels >= 3 else (np.stack([img_float]*3, axis=-1) if len(img_float.shape) == 2 else np.stack([img_float[:, :, 0]]*3, axis=-1))

            # Linearize
            if transfer == 16:
                linear = pq_to_linear(rgb_float)
            elif transfer == 18:
                linear = hlg_to_linear(rgb_float)
            else:
                linear = np.power(rgb_float, 2.2)

            # Tone-map using Reinhard
            sdr_linear = tone_map_reinhard(linear, exposure=1.0)
            # Gamma correct to sRGB
            sdr_rgb = linear_to_srgb(sdr_linear)
            sdr_uint8 = (sdr_rgb * 255.0).astype(np.uint8)

            if channels == 4:
                sdr_rgba = np.empty((h, w, 4), dtype=np.uint8)
                sdr_rgba[:, :, :3] = sdr_uint8
                sdr_rgba[:, :, 3] = (img_float[:, :, 3] * 255.0).astype(np.uint8)
                out_arr = sdr_rgba
                q_format = QImage.Format.Format_RGBA8888
            else:
                out_arr = sdr_uint8
                q_format = QImage.Format.Format_RGB888

            out_arr = np.ascontiguousarray(out_arr)
            qimage = QImage(out_arr.data, w, h, w * out_arr.shape[2], q_format)
            qimage.ndarr = out_arr  # Keep array alive
            qimage.setColorSpace(QColorSpace(QColorSpace.NamedColorSpace.SRgb))

        return QPixmap.fromImage(qimage)

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to load HDR image {os.path.basename(file_path)}: {e}")
        return None

