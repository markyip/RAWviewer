"""
Embedded JPEG extraction, TIFF parsing, orientation helpers, and module-level
state for the enhanced RAW processor package.
"""
from __future__ import annotations

import logging
import os
import threading
import warnings
from typing import Any

import numpy as np
from PyQt6.QtCore import QLoggingCategory, QSize
from PyQt6.QtGui import QImage, QImageReader

from common_image_loader import (
    decode_embedded_jpeg_bytes,
)
from metadata import backend as metadata_backend
from processing.raw_decode.gil import rawpy_imread_warm

# Silence qt.imageformats warnings (e.g. missing TIFF tag warnings on RAW files)
QLoggingCategory.setFilterRules("qt.imageformats*=false")

logging.getLogger("exifread").setLevel(logging.ERROR)

# Suppress exifread warnings for unsupported file formats (e.g., video files)
warnings.filterwarnings('ignore', category=UserWarning, module='exifread')

# Cached EXIF rows without this version used embedded-JPEG dimensions as "original" (e.g. 1920x1080).
# Cached EXIF rows without this version used buggy orientation logic (e.g. LibRaw 5 mis-mapped, Sony MakerNote missing, or Silent Failures).
# v9: invalidate rows that stored orientation=1 for Canon CR2/CR3 while LibRaw flip / embedded JPEG said otherwise.
# v10: unified make-independent orientation authority (removed make-gated skips that left
#      non-Sony pre-rotated cameras sideways); force re-extraction so cache self-heal has
#      correct sensor dims/orientation to re-orient any stale wrongly-rotated thumbnails.
# v11: added nef_he_compressed (proactive Nikon HE/HE* detection); force re-extraction so
#      already-cached NEF rows pick up the new field instead of staying permanently unknown.
RAW_EXIF_SENSOR_META_VER = 11

_heavy_fallback_semaphore = threading.Semaphore(8)

# ========== concurrency gate ==========

def _rawpy_unpack_concurrency() -> int:
    """Concurrent in-process rawpy/LibRaw contexts allowed through the global gate.

    LibRaw is documented thread-safe per-context (each ``rawpy.imread`` creates
    an independent decoder). Unpack is typically ~80% of CPU Fast RAW cost; a
    fully serialized gate therefore caps sustained navigation at ~1/unpack_time
    even when ImageLoadManager has spare worker slots.

    Default is adaptive (not hard-coded 1):
      - explicit ``RAWVIEWER_UNPACK_CONCURRENCY=1..4`` always wins
      - else 3 on >=12 logical CPUs, 2 on >=6, else 1
      - soft RAM caps: <10 GB -> 1; <16 GB -> max 2

    Each concurrent unpack holds ~2 bytes/sensor-pixel of uint16 mosaic, so the
    ceiling stays at 4. Every call site shares this one gate (never a mixed
    Lock/Semaphore state).
    """
    raw = os.environ.get("RAWVIEWER_UNPACK_CONCURRENCY", "").strip()
    if raw:
        try:
            return max(1, min(4, int(raw)))
        except ValueError:
            return 1

    cpu = os.cpu_count() or 4
    if cpu >= 12:
        base = 3
    elif cpu >= 6:
        base = 2
    else:
        base = 1

    # Soft RAM cap: <10 GB -> 1; <16 GB -> max 2
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        if ram_gb < 10:
            base = min(base, 1)
        elif ram_gb < 16:
            base = min(base, 2)
    except ImportError:
        pass
    return base


_unpack_concurrency_cap = _rawpy_unpack_concurrency()
_unpack_concurrency_lock = threading.Lock()

def set_unpack_concurrency_cap(cap: int) -> int:
    """Override the adaptive concurrency cap at runtime.

    ``cap`` is clamped to [1..4].  Returns the new cap value.
    """
    global _unpack_concurrency_cap  # noqa: F821
    cap = max(1, min(4, cap))
    _unpack_concurrency_cap = cap
    return _unpack_concurrency_cap


# ========== byte-scan toggle ==========

_BYTESCAN_FIRST_EXTS = frozenset({
    # "Modern" RAW wrappers: Canon CR3 uses ISO Base Media File Format (BMFF),
    # DNG is TIFF-based but the embedded JPEG segment is expensive to extract
    # via LibRaw if the DNG is a "raw-embedded-into-a-container" variant.
    ".cr3", ".cr2", ".dng", ".tiff", ".tif",
    # Nikon NEF is TIFF-based; embedded JPEG is in SubIFDs.
    ".nef", ".nrw",
    # Sony ARW is TIFF-based with a full-size JPEG preview.
    ".arw",
    # Fujifilm RAF is TIFF-based (strips of raw data + a JPEG preview IFD).
    ".raf",
})

def _bytescan_first_enabled() -> bool:
    """Whether the byte-scan path (TIFF without LibRaw) is active."""
    raw = os.environ.get("RAWVIEWER_BYTESCAN_FIRST", "")
    if raw:
        return raw.lower() in ("1", "true", "yes")
    return True


_rawpy_global_lock = threading.Lock()
_embedded_scan_miss_cache: set[str] = set()
_embedded_scan_miss_lock = threading.Lock()
_embedded_scan_miss_cache_max = 5000


# ========== QImage helpers ==========

def _qimage_to_rgb_array(image: QImage) -> np.ndarray | None:
    """Convert a QImage to an RGB numpy array (HxWx3 uint8)."""
    try:
        if image.isNull():
            return None
        if image.format() != QImage.Format.Format_RGB888:
            image = image.convertToFormat(QImage.Format.Format_RGB888)
        w = image.width()
        h = image.height()
        if w <= 0 or h <= 0:
            return None
        bits = image.bits()
        bits.setsize(h * image.bytesPerLine())
        arr = np.frombuffer(bits, dtype=np.uint8).reshape(h, image.bytesPerLine())
        arr = arr[:, : w * 3].reshape(h, w, 3)
        return arr.copy()
    except Exception:
        return None


# ========== JPEG helpers ==========

def get_jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    """Parse JPEG SOF marker to get image dimensions without full decode."""
    if not data or len(data) < 4:
        return None
    i = 0
    while i < len(data) - 1:
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker == 0xD8:
            i += 2
            continue
        if marker in (0xC0, 0xC1, 0xC2):
            if i + 11 > len(data):
                return None
            height = (data[i + 5] << 8) | data[i + 6]
            width = (data[i + 7] << 8) | data[i + 8]
            return width, height
        if marker == 0xD9:
            return None
        seg_len = (data[i + 2] << 8) | data[i + 3]
        if seg_len < 2:
            return None
        i += 2 + seg_len
    return None


# ========== Embedded JPEG extraction ==========

def _largest_jpeg_from_blob(blob: bytes, max_size: int) -> np.ndarray | None:
    """Find the largest embedded JPEG in *blob* and decode+resize it.

    ``blob`` is a raw binary chunk (a file or an IFD ``Strip`` /
    ``PreviewImage`` value).  We scan it for JPEG SOI (0xFF 0xD8) markers,
    attempt to decode each, and return the decoded array of the largest by
    pixel count, resized to *max_size*.

    Returns ``None`` if no decodable JPEG is found.
    """
    if not blob or len(blob) < 4:
        return None
    best: np.ndarray | None = None
    best_pixels = 0
    i = 0
    while i < len(blob) - 1:
        if blob[i] != 0xFF:
            i += 1
            continue
        if blob[i + 1] != 0xD8:
            i += 1
            continue
        # found a JPEG SOI — find the EOI
        end = blob.find(b'\xff\xd9', i + 2)
        if end == -1:
            i += 2
            continue
        jpeg_bytes = blob[i:end + 2]
        try:
            decoded = decode_embedded_jpeg_bytes(jpeg_bytes, max_size)
            if decoded is not None:
                px = decoded.shape[0] * decoded.shape[1]
                if px > best_pixels:
                    best = decoded
                    best_pixels = px
        except Exception:
            pass
        i = end + 2
    return best


def _extract_bmff_uuid_embedded_jpeg(file_path: str, max_size: int) -> np.ndarray | None:
    """CR3-specific: scan BMFF structure for ``crx_preview`` / ``P prp`` UUID box.

    ISO Base Media File Format (BMFF) used by Canon CR3 wraps the embedded
    JPEG inside a UUID box (box-type ``uuid``) with a Canon-specific
    sub-type.  This is *much* faster than going through LibRaw because we
    skip raw decode entirely.

    Returns decoded+resized array or None.
    """
    try:
        import struct
        _BMFF_UUID_CANON_PREVIEW = bytes.fromhex(
            "85c0b687820f11e0ba5908001dce00b0"
        )
        with open(file_path, "rb") as f:
            data = f.read()
        i = 0
        while i + 8 <= len(data):
            box_size = struct.unpack_from(">I", data, i)[0]
            box_type = data[i + 4:i + 8]
            if box_type == b"uuid" and box_size >= 24:
                uuid_field = data[i + 8:i + 24]
                if uuid_field == _BMFF_UUID_CANON_PREVIEW:
                    jpeg_data = data[i + 24:i + box_size]
                    decoded = decode_embedded_jpeg_bytes(jpeg_data, max_size)
                    return decoded
            if box_size == 0:
                break
            i += max(box_size, 8)
        return None
    except Exception:
        return None


# ========== RAW orientation probing ==========

def probe_effective_raw_orientation(
    file_path: str,
) -> tuple[int, int, int]:
    """Return ``(orientation, original_width, original_height)`` for *file_path*.

    Uses LibRaw's ``sizes.flip`` as the authoritative sensor orientation,
    then converts it to our EXIF-orientation model.  Falls back to embedded
    JPEG orientation only when LibRaw can't open the file.
    """
    orientation = 1
    original_width = 0
    original_height = 0
    try:
        with rawpy_imread_warm(file_path) as raw:
            raw_sizes = raw.sizes
            original_width = int(raw_sizes.width)
            original_height = int(raw_sizes.height)
            flip = int(getattr(raw_sizes, "flip", 0) or 0)
            libraw_orientations = {0: 1, 3: 3, 5: 8, 6: 6}
            orientation = int(libraw_orientations.get(flip, flip))
    except Exception:
        orientation, original_width, original_height = (
            _orientation_from_embedded_preview(file_path)
        )
    return orientation, original_width, original_height


def cached_raw_exif_orientation_trustworthy(file_path: str, cached: dict[str, Any] | None = None) -> bool:
    """Whether *file_path* has a cached EXIF row with verified orientation.

    Short-circuits the expensive orientation probe for files that have
    already been verified. Accepts an optional *cached* EXIF dict from the
    caller (avoids a redundant DB fetch when the caller already has it).
    """
    try:
        row = cached if cached is not None else metadata_backend.fetch_exif_by_path(file_path)
        if row is not None:
            meta_ver = row.get("_sensor_meta_ver", 0)
            if (
                meta_ver >= RAW_EXIF_SENSOR_META_VER
                and row.get("verified_orientation", False)
            ):
                return True
    except Exception:
        pass
    return False


def _orientation_from_embedded_preview(
    file_path: str,
) -> tuple[int, int, int]:
    """Fallback: determine orientation from embedded JPEG preview dimensions.

    Returns ``(orientation, original_width, original_height)``.
    """
    orientation = 1
    original_width = 0
    original_height = 0
    try:
        from PIL import Image
        with Image.open(file_path) as img:
            exif = img.getexif()
            orientation = int(exif.get(0x0112, 1))
            original_width, original_height = img.size
    except Exception:
        pass
    return orientation, original_width, original_height


# ========== QImage reader thumbnail ==========

def _thumbnail_via_qimage_reader(
    file_path: str, max_size: int, auto_transform: bool = True
) -> np.ndarray | None:
    """Decode thumbnail using QImageReader, returning an RGB ndarray."""
    try:
        reader = QImageReader(file_path)
        reader.setAutoTransform(auto_transform)
        size = reader.size()
        if size.isValid() and size.width() > 0 and size.height() > 0:
            w, h = size.width(), size.height()
            if max_size > 0 and (w > max_size or h > max_size):
                scale = min(max_size / w, max_size / h)
                reader.setScaledSize(
                    QSize(max(1, int(w * scale)), max(1, int(h * scale)))
                )
            image = reader.read()
            return _qimage_to_rgb_array(image)
    except Exception:
        pass
    return None


# ========== TIFF parse helpers ==========

_BIG_ENDIAN = b"MM"
_LITTLE_ENDIAN = b"II"
TIFF_ENDIAN_MAP = {_BIG_ENDIAN: ">", _LITTLE_ENDIAN: "<"}

def extract_previews_via_tiff_parse(file_path: str, max_size: int = 0) -> list[bytes]:
    """Parse TIFF IFD chain at low level, collecting JPEG preview segments.

    Returns a list of raw JPEG byte blobs (without Exif container headers),
    ordered from the *largest* SubIFD JPEG preview first, then optionally
    the main IFD thumbnail if no SubIFD JPEG was found.

    Works on any TIFF-based RAW (CR2, NEF, ARW, RAF, DNG, etc.).  *max_size*
    is the pixel-count limit (0 = unlimited) for the returned segments.
    """
    try:
        from struct import unpack_from

        previews: list[bytes] = []
        with open(file_path, "rb") as f:
            header = f.read(8)
            if len(header) < 8 or header[:2] not in TIFF_ENDIAN_MAP:
                return previews
            endian = TIFF_ENDIAN_MAP[header[:2]]
            ifd_offset = unpack_from(endian + "I", header, 4)[0]
            if ifd_offset < 8:
                return previews

            # Walk the IFD chain looking for JPEG preview tags (SubIFDs and main IFD).
            while ifd_offset > 7:
                f.seek(ifd_offset)
                ifd_header = f.read(2)
                if len(ifd_header) < 2:
                    break
                num_entries = unpack_from(endian + "H", ifd_header)[0]
                entries_size = 2 + num_entries * 12 + 4
                f.seek(ifd_offset)
                ifd_data = f.read(entries_size)
                if len(ifd_data) < entries_size:
                    break

                next_ifd = unpack_from(endian + "I", ifd_data, 2 + num_entries * 12)[0]

                # Look for SubIFDs tag (0x014a)
                for entry_idx in range(num_entries):
                    entry_off = 2 + entry_idx * 12
                    tag = unpack_from(endian + "H", ifd_data, entry_off)[0]
                    type_ = unpack_from(endian + "H", ifd_data, entry_off + 2)[0]
                    count = unpack_from(endian + "I", ifd_data, entry_off + 4)[0]
                    val_offset = unpack_from(endian + "I", ifd_data, entry_off + 8)[0]

                    if tag == 0x014a and type_ == 13:  # SubIFDs offsets
                        sub_ifd_count = count
                        for sub_i in range(sub_ifd_count):
                            sub_ifd_offset = unpack_from(endian + "I", ifd_data, entry_off + 8 + sub_i * 4)[0]
                            try:
                                f.seek(sub_ifd_offset)
                                sub_data = f.read(4096)
                                jpeg = _scan_for_jpeg_segments(sub_data)
                                if jpeg:
                                    previews.extend(jpeg)
                            except Exception:
                                pass

                    # Also collect JPEGInterchangeFormat / StripOffsets from main IFD
                    if tag in (0x0201, 0x0117, 0x0100, 0x0101):
                        pass

                # Check main IFD for JPEG preview (tag 0x0201 = JPEGInterchangeFormat)
                for entry_idx in range(num_entries):
                    entry_off = 2 + entry_idx * 12
                    tag = unpack_from(endian + "H", ifd_data, entry_off)[0]
                    type_ = unpack_from(endian + "H", ifd_data, entry_off + 2)[0]
                    count = unpack_from(endian + "I", ifd_data, entry_off + 4)[0]
                    val_offset = unpack_from(endian + "I", ifd_data, entry_off + 8)[0]

                    if tag == 0x0201 and type_ == 7:  # JPEGInterchangeFormat
                        f.seek(val_offset)
                        jpeg_data = f.read(min(count, 10 * 1024 * 1024))
                        if jpeg_data and jpeg_data[:2] == b'\xff\xd8':
                            decoded = decode_embedded_jpeg_bytes(jpeg_data, max_size)
                            if decoded is not None:
                                previews.append(jpeg_data)

                ifd_offset = next_ifd
        return previews
    except Exception:
        return []


def _scan_for_jpeg_segments(data: bytes) -> list[bytes]:
    """Scan *data* for JPEG SOI/EOI markers and return all complete segments."""
    segments = []
    i = 0
    while i < len(data) - 1:
        if data[i] != 0xFF or data[i + 1] != 0xD8:
            i += 1
            continue
        end = data.find(b'\xff\xd9', i + 2)
        if end == -1:
            i += 1
            continue
        segments.append(data[i:end + 2])
        i = end + 2
    return segments


# ========== NEF HE compression detection ==========

def _detect_nef_he_compression(file_path: str) -> bool | None:
    """Proactively detect Nikon HE / HE* compressed NEF before processing.

    HE (High Efficiency) NEF files need extra half-size handling because
    lossy-compressed CFA data can cause rawpy dcraw-based postprocessing to
    produce green-tinted artefacts.  Returns True if HE, False if standard
    NEF, or None if not a NEF or read failed.

    Uses the byte-scan heuristic: HE NEFs have ``Nikon`` followed by
    ``HE`` / ``HE\x00`` in the first few kilobytes; standard NEFs have
    ``Nikon`` followed by the model name.
    """
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in (".nef", ".nrw"):
            return None
        with open(file_path, "rb") as f:
            head = f.read(8192)
        # Look for "Nikon" maker note
        nikon_pos = head.find(b"Nikon")
        if nikon_pos == -1:
            return None
        # After "Nikon" we expect model info; HE/HE* NEFs have "HE" shortly after
        after_nikon = head[nikon_pos:nikon_pos + 512]
        if b"HE\x00" in after_nikon or b" HE" in after_nikon:
            return True
        # Standard NEF has model name after "Nikon"
        if b"NEF" in after_nikon:
            return False
        return None
    except Exception:
        return None


# ========== IFD reader and embedded JPEG scan ==========

def read_ifd(offset):
    """Read an IFD entry at *offset* from current file position.

    Legacy helper used by ``extract_embedded_jpeg_by_scan``.
    """
    pass


def extract_embedded_jpeg_by_scan(file_path: str, max_size: int) -> np.ndarray | None:
    """Extract embedded JPEG from RAW by scanning for JPEG markers.

    High-speed path that scans the file for JPEG SOI markers and decodes
    the largest embedded JPEG.  Works on any TIFF-based RAW format.
    """
    try:
        # Check miss cache first
        miss_key = (file_path, max_size)
        with _embedded_scan_miss_lock:
            if miss_key in _embedded_scan_miss_cache:
                return None

        # Try BMFF (CR3) first
        if os.path.splitext(file_path)[1].lower() == ".cr3":
            result = _extract_bmff_uuid_embedded_jpeg(file_path, max_size)
            if result is not None:
                return result

        # Try TIFF parse for all other formats
        result = _extract_embedded_jpeg_by_scan_impl(file_path, max_size)
        if result is not None:
            return result

        return None
    except Exception:
        return None


def _extract_embedded_jpeg_by_scan_impl(file_path: str, max_size: int) -> np.ndarray | None:
    """Low-level TIFF scan for embedded JPEG."""
    from struct import unpack_from

    try:
        with open(file_path, "rb") as f:
            header = f.read(8)
            if len(header) < 8 or header[:2] not in TIFF_ENDIAN_MAP:
                return None
            endian = TIFF_ENDIAN_MAP[header[:2]]
            ifd_offset = unpack_from(endian + "I", header, 4)[0]
            if ifd_offset < 8:
                return None

            best_arr: np.ndarray | None = None
            best_pixels = 0
            visited_ifds: set[int] = set()

            while ifd_offset > 7 and ifd_offset not in visited_ifds:
                visited_ifds.add(ifd_offset)
                f.seek(ifd_offset)
                ifd_header = f.read(2)
                if len(ifd_header) < 2:
                    break
                num_entries = unpack_from(endian + "H", ifd_header)[0]
                if num_entries > 100:
                    break
                entries_size = 2 + num_entries * 12 + 4
                f.seek(ifd_offset)
                ifd_data = f.read(entries_size)
                if len(ifd_data) < entries_size:
                    break

                next_ifd = unpack_from(endian + "I", ifd_data, 2 + num_entries * 12)[0]

                for entry_idx in range(num_entries):
                    entry_off = 2 + entry_idx * 12
                    tag = unpack_from(endian + "H", ifd_data, entry_off)[0]
                    type_ = unpack_from(endian + "H", ifd_data, entry_off + 2)[0]
                    count_ = unpack_from(endian + "I", ifd_data, entry_off + 4)[0]
                    val_offset = unpack_from(endian + "I", ifd_data, entry_off + 8)[0]

                    # StripOffsets (0x0117) or JPEGInterchangeFormat (0x0201)
                    if tag == 0x0201 and type_ == 7:  # JPEGInterchangeFormat
                        f.seek(val_offset)
                        jpeg_data = f.read(min(count_, 10 * 1024 * 1024))
                        if jpeg_data[:2] == b'\xff\xd8':
                            decoded = decode_embedded_jpeg_bytes(jpeg_data, max_size)
                            if decoded is not None:
                                px = decoded.shape[0] * decoded.shape[1]
                                if px > best_pixels:
                                    best_arr = decoded
                                    best_pixels = px

                    # Also scan SubIFDs (0x014a)
                    if tag == 0x014a and type_ == 13:
                        for si in range(count_):
                            sub_off = unpack_from(endian + "I", ifd_data, entry_off + 8 + si * 4)[0]
                            try:
                                f.seek(sub_off)
                                sub_data = f.read(4096)
                                sj = _scan_for_jpeg_segments(sub_data)
                                for seg in sj:
                                    decoded = decode_embedded_jpeg_bytes(seg, max_size)
                                    if decoded is not None:
                                        px = decoded.shape[0] * decoded.shape[1]
                                        if px > best_pixels:
                                            best_arr = decoded
                                            best_pixels = px
                            except Exception:
                                pass

                ifd_offset = next_ifd

            return best_arr
    except Exception:
        return None


# ========== Module-level state (exported) ==========

# The following exports are accessed by sibling modules:
# _rawpy_global_lock
# _heavy_fallback_semaphore
# _bytescan_first_enabled
# _BYTESCAN_FIRST_EXTS
# _extract_bmff_uuid_embedded_jpeg
# _qimage_to_rgb_array
# _thumbnail_via_qimage_reader
# get_jpeg_dimensions
# _largest_jpeg_from_blob
# extract_previews_via_tiff_parse
# extract_embedded_jpeg_by_scan
# _extract_embedded_jpeg_by_scan_impl
# probe_effective_raw_orientation
# cached_raw_exif_orientation_trustworthy
# _orientation_from_embedded_preview
# _detect_nef_he_compression
# _rawpy_unpack_concurrency
# set_unpack_concurrency_cap
# RAW_EXIF_SENSOR_META_VER
