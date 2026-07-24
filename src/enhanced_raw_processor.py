"""
Enhanced RAW processor with smart thumbnail utilization and caching.

This module provides optimized RAW processing with immediate thumbnail display,
intelligent caching, and progressive image loading for maximum performance.
"""

import os
import time
import threading
import warnings
import numpy as np
import rawpy

from rawpy_gil import rawpy_imread_warm
import exifread
import sys
from typing import Optional, Dict, Any, Tuple, Union
from PyQt6.QtCore import QThread, pyqtSignal, QObject, QSize, QLoggingCategory
from PyQt6.QtGui import QPixmap, QImage, QImageReader
from PIL import Image
import io

# Silence qt.imageformats warnings (e.g. missing TIFF tag warnings on RAW files)
QLoggingCategory.setFilterRules("qt.imageformats*=false")

import logging
logging.getLogger("exifread").setLevel(logging.ERROR)

# Suppress exifread warnings for unsupported file formats (e.g., video files)
warnings.filterwarnings('ignore', category=UserWarning, module='exifread')

from image_cache import get_image_cache
from common_image_loader import (
    decode_embedded_jpeg_bytes,
    normalize_capture_time_string,
    orientation_from_embedded_jpeg_bytes,
    is_slow_storage_path,
)
import metadata_backend
from raw_file_extensions import RAW_FILE_EXTENSIONS

# Cached EXIF rows without this version used embedded-JPEG dimensions as "original" (e.g. 1920×1080).
# Cached EXIF rows without this version used buggy orientation logic (e.g. LibRaw 5 mis-mapped, Sony MakerNote missing, or Silent Failures).
# v9: invalidate rows that stored orientation=1 for Canon CR2/CR3 while LibRaw flip / embedded JPEG said otherwise.
# v10: unified make-independent orientation authority (removed make-gated skips that left
#      non-Sony pre-rotated cameras sideways); force re-extraction so cache self-heal has
#      correct sensor dims/orientation to re-orient any stale wrongly-rotated thumbnails.
# v11: added nef_he_compressed (proactive Nikon HE/HE* detection); force re-extraction so
#      already-cached NEF rows pick up the new field instead of staying permanently unknown.
RAW_EXIF_SENSOR_META_VER = 11

from image_load_manager import yield_if_current_task_active

def _rawpy_unpack_concurrency() -> int:
    """Concurrent in-process rawpy/LibRaw contexts allowed through the global gate.

    LibRaw is documented thread-safe per-context (each ``rawpy.imread`` creates
    an independent decoder). Unpack is typically ~80% of CPU Fast RAW cost; a
    fully serialized gate therefore caps sustained navigation at ~1/unpack_time
    even when ImageLoadManager has spare worker slots.

    Default is adaptive (not hard-coded 1):
      - explicit ``RAWVIEWER_UNPACK_CONCURRENCY=1..4`` always wins
      - else 3 on ≥12 logical CPUs, 2 on ≥6, else 1
      - soft RAM caps: <10 GB → 1; <16 GB → max 2

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
        default = 3
    elif cpu >= 6:
        default = 2
    else:
        default = 1
    try:
        import psutil

        ram_gb = float(psutil.virtual_memory().total) / (1024.0 ** 3)
        if ram_gb < 10.0:
            default = 1
        elif ram_gb < 16.0:
            default = min(default, 2)
    except Exception:
        pass
    return default


_UNPACK_CONCURRENCY = _rawpy_unpack_concurrency()
_rawpy_global_lock = (
    threading.Lock()
    if _UNPACK_CONCURRENCY == 1
    else threading.BoundedSemaphore(_UNPACK_CONCURRENCY)
)
_heavy_fallback_semaphore = threading.Semaphore(8)

# RAW formats where the lock-free byte-scan extractor reliably yields a correctly
# oriented embedded preview (finalize_index_thumbnail_array heals container EXIF).
# Verified vs LibRaw: Sony ARW 15/15, Nikon NEF 36/36, Canon CR3 23/23 (BMFF uuid
# box JPEG scan). Other formats (DNG, ORF, RAF, RW2, Hasselblad 3FR) still return
# None from byte-scan and fall through to LibRaw. Canon CR2 is not listed — its
# TIFF-parse previews can lack the orientation tag. LibRaw stays first for
# everything not listed here.
_BYTESCAN_FIRST_EXTS = frozenset({".arw", ".nef", ".cr3"})


def _bytescan_first_enabled() -> bool:
    """Default ON. Set RAWVIEWER_GALLERY_BYTESCAN_FIRST=0 to force the old LibRaw-first path."""
    return os.environ.get(
        "RAWVIEWER_GALLERY_BYTESCAN_FIRST", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}


def _qimage_to_rgb_array(image: QImage) -> Optional[np.ndarray]:
    """Convert a QImage into a contiguous RGB numpy array without creating a QPixmap."""
    img = image.convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    if w < 1 or h < 1:
        return None
    bpl = img.bytesPerLine()
    nbytes = h * bpl
    bits = img.constBits()
    if bits is None:
        return None
    try:
        if hasattr(bits, "asstring"):
            raw = bits.asstring(nbytes)
        else:
            raw = bytes(memoryview(bits)[:nbytes])
    except (BufferError, TypeError, AttributeError):
        out = np.empty((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                c = img.pixel(x, y)
                out[y, x, 0] = (c >> 16) & 0xFF
                out[y, x, 1] = (c >> 8) & 0xFF
                out[y, x, 2] = c & 0xFF
        return out
    arr = np.frombuffer(bytearray(raw), dtype=np.uint8).reshape(h, bpl)
    return np.ascontiguousarray(arr[:, : w * 3].reshape(h, w, 3))


_embedded_scan_miss_cache = set()
_embedded_scan_miss_lock = threading.Lock()
_embedded_scan_miss_cache_max = 4096
_embedded_scan_inflight: dict[tuple, threading.Event] = {}
_embedded_scan_results: dict[tuple, Optional[np.ndarray]] = {}
_embedded_scan_coalesce_lock = threading.Lock()
_embedded_scan_results_max = 512


def get_jpeg_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    """Parse JPEG header bytes to find image width and height without decoding."""
    if len(data) < 4 or not data.startswith(b'\xff\xd8'):
        return None
        
    offset = 2
    data_len = len(data)
    
    while offset < data_len:
        # Skip any 0xff padding bytes
        while offset < data_len and data[offset] == 0xff:
            offset += 1
            
        if offset >= data_len:
            break
            
        marker = data[offset]
        offset += 1
        
        # SOS (Start of Scan) or EOI (End of Image) mean header is over
        if marker == 0xda or marker == 0xd9:
            break
            
        # Standalone markers with no length/payload
        if marker in (0x01, 0xd0, 0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8):
            continue
            
        # Markers with length
        if offset + 2 > data_len:
            break
            
        segment_len = int.from_bytes(data[offset:offset+2], byteorder='big')
        
        # Check if it's an SOF marker (0xc0 - 0xcf except 0xc4, 0xc8, 0xcc)
        if 0xc0 <= marker <= 0xcf and marker not in (0xc4, 0xc8, 0xcc):
            if offset + 7 <= data_len:
                # SOF payload starts after the 2-byte length field:
                # offset + 2: precision (1 byte)
                # offset + 3: height (2 bytes)
                # offset + 5: width (2 bytes)
                height = int.from_bytes(data[offset+3:offset+5], byteorder='big')
                width = int.from_bytes(data[offset+5:offset+7], byteorder='big')
                return width, height
            break
            
        offset += segment_len
        
    return None


def _largest_jpeg_from_blob(blob: bytes, max_size: int) -> Optional[np.ndarray]:
    """Find a decodable JPEG segment inside a byte blob using fast header
    parsing -- the smallest one that still covers max_size (0 = full
    quality, i.e. the largest available), not always the largest.

    Canon CR3 (the caller, _extract_bmff_uuid_embedded_jpeg) embeds a
    near-sensor-resolution JPEG (20-40MP) alongside much smaller ones;
    decoding the huge one for a 512px gallery tile wastes ~10x the CPU for
    an identical downsampled result. Mirrors the equivalent fix already
    applied to the TIFF-based preview path (extract_previews_via_tiff_parse)
    for ARW/NEF -- this is the same bug, just never ported to CR3's BMFF
    parsing since it's a structurally different container format.
    """
    candidates: list[tuple[int, int, int, int, int]] = []  # (area, w, h, start, end)
    start = 0
    blob_len = len(blob)
    while True:
        idx = blob.find(b"\xff\xd8\xff", start)
        if idx < 0:
            break

        # Parse JPEG header from the next 64KB (or up to the end of the blob)
        header_chunk = blob[idx : min(blob_len, idx + 65536)]
        dims = get_jpeg_dimensions(header_chunk)
        if dims is not None:
            w, h = dims
            if w >= 32 and h >= 32:
                end_marker = blob.find(b"\xff\xd9", idx + 3)
                end_offset = end_marker + 2 if end_marker >= 0 else blob_len
                candidates.append((w * h, w, h, idx, end_offset))

        start = idx + 3

    if not candidates:
        return None

    if max_size > 0:
        sufficient = [c for c in candidates if max(c[1], c[2]) >= max_size]
        chosen = min(sufficient, key=lambda c: c[0]) if sufficient else max(candidates, key=lambda c: c[0])
    else:
        chosen = max(candidates, key=lambda c: c[0])

    _, _, _, best_offset, best_end_offset = chosen
    segment = blob[best_offset:best_end_offset]

    # Decode only the selected segment.
    try:
        return decode_embedded_jpeg_bytes(segment, max_size)
    except Exception:
        return None


def _extract_bmff_uuid_embedded_jpeg(file_path: str, max_size: int) -> Optional[np.ndarray]:
    """Canon CR3 stores grid/preview JPEG inside BMFF uuid boxes (ftyp crx), not TIFF IFD.

    All real preview content lives in the small ftyp/moov/uuid boxes at the
    START of the file (measured ~366KB on a real 40MB CR3); the sensor RAW
    data (mdat, the rest of the file) follows and is never touched by this
    function. Reads incrementally in chunks instead of loading the whole
    file upfront -- avoids ~40MB+ of unnecessary I/O per thumbnail/preview
    request on every CR3 file.
    """
    import struct

    CHUNK = 1 << 20  # 1MB steps; ftyp+moov+uuid boxes are typically well under this

    try:
        with open(file_path, "rb") as f:
            data = f.read(CHUNK)
            if len(data) < 12 or data[4:8] != b"ftyp":
                return None

            best: Optional[np.ndarray] = None
            best_area = 0
            off, n = 0, len(data)
            while True:
                if off + 8 > n:
                    more = f.read(CHUNK)
                    if not more:
                        break
                    data += more
                    n = len(data)
                    if off + 8 > n:
                        break
                size = struct.unpack(">I", data[off : off + 4])[0]
                typ = data[off + 4 : off + 8]
                if size < 8:
                    # size==1 (64-bit extended size -- always mdat/the sensor
                    # payload in practice) or a malformed box: nothing past
                    # here is a preview, so stop without reading the rest of
                    # the (often huge) file.
                    break
                if off + size > n:
                    more = f.read(off + size - n)
                    if more:
                        data += more
                        n = len(data)
                end = min(off + size, n)
                if typ == b"uuid":
                    arr = _largest_jpeg_from_blob(data[off:end], max_size)
                    if arr is not None and hasattr(arr, "shape") and len(arr.shape) >= 2:
                        h, w = arr.shape[:2]
                        area = w * h
                        if area > best_area:
                            best_area = area
                            best = arr
                off = end
            return best
    except Exception:
        return None


def probe_effective_raw_orientation(
    file_path: str, raw_object: Optional[rawpy.RawPy] = None
) -> int:
    """LibRaw flip + embedded-JPEG orientation when container EXIF is missing/wrong."""
    orientation = 1
    try:
        if raw_object is not None:
            flip = int(getattr(raw_object.sizes, "flip", 0) or 0)
            if flip != 0:
                flip_map = {0: 1, 3: 3, 5: 8, 6: 6}
                orientation = int(flip_map.get(flip, flip))
            if orientation <= 1:
                embedded_o = _orientation_from_embedded_preview(file_path, raw_object)
                if embedded_o not in (1, orientation):
                    orientation = embedded_o
            return int(orientation or 1)
        yield_if_current_task_active()
        with _rawpy_global_lock:
            with rawpy_imread_warm(file_path) as raw:
                return probe_effective_raw_orientation(file_path, raw)
    except Exception:
        if orientation <= 1:
            try:
                embedded_o = _orientation_from_embedded_preview(file_path, raw_object)
                if embedded_o not in (1, orientation):
                    orientation = embedded_o
            except Exception:
                pass
    return int(orientation or 1)


def cached_raw_exif_orientation_trustworthy(
    file_path: str, cached: Dict[str, Any]
) -> bool:
    """False when persisted EXIF orientation disagrees with LibRaw / embedded preview."""
    import common_image_loader

    if not common_image_loader.is_raw_file(file_path):
        return True
    if cached.get("verified_orientation"):
        return True
    cached_o = int(cached.get("orientation", 1) or 1)
    effective = probe_effective_raw_orientation(file_path)
    if effective <= 1:
        return True
    return cached_o == effective


def _orientation_from_embedded_preview(
    file_path: str, raw_object: Optional[rawpy.RawPy] = None
) -> int:
    """Orientation from LibRaw embedded JPEG EXIF (container tags are often missing/wrong)."""
    try:
        if raw_object is not None:
            thumb = raw_object.extract_thumb()
        else:
            yield_if_current_task_active()
            with _rawpy_global_lock:
                with rawpy_imread_warm(file_path) as raw:
                    thumb = raw.extract_thumb()
        if thumb is None or thumb.format != rawpy.ThumbFormat.JPEG:
            return 1
        return orientation_from_embedded_jpeg_bytes(thumb.data)
    except Exception:
        return 1


def _thumbnail_via_qimage_reader(file_path: str, max_size: int, auto_transform: bool = True) -> Optional[np.ndarray]:
    """OS codec fallback for RAW when LibRaw cannot open the container (common on Windows)."""
    try:
        from PyQt6.QtCore import Qt

        reader = QImageReader(file_path)
        if auto_transform:
            reader.setAutoTransform(True)
        size = reader.size()
        if max_size > 0 and size.isValid():
            if auto_transform:
                # Adjust target dimensions if rotation swaps width and height
                trans = reader.transformation()
                from PyQt6.QtGui import QImageIOHandler
                # Trans values 4, 5, 6, 7 in Qt involve 90 or 270 degree rotation
                if trans.value >= 4:
                    size = QSize(size.height(), size.width())
            reader.setScaledSize(
                size.scaled(max_size, max_size, Qt.AspectRatioMode.KeepAspectRatio)
            )
        qimg = reader.read()
        if qimg is None or qimg.isNull():
            return None
        return _qimage_to_rgb_array(qimg)
    except Exception:
        return None


def extract_previews_via_tiff_parse(file_path: str, max_size: int = 0) -> list[bytes]:
    """Parse TIFF structure to find embedded JPEG previews and return the one
    to use for `max_size` (0 = full quality, i.e. the largest available).

    Many RAW formats (e.g. Sony ARW) embed 2-3 previews of very different
    sizes (thumb / medium / near-full-sensor). Reading and decoding the
    largest one (often 20-40MP, several MB) to produce a 512px gallery tile
    wastes ~10x the decode time and several MB of I/O for no visual benefit
    -- it gets downsampled either way. So this only *fully* reads the JPEG
    segment it actually intends to return; every other candidate is sized
    up from a cheap <=64KB header read (JPEG SOF marker is always near the
    start of the segment).
    """
    import struct
    candidates: list[tuple[int, int, int, int, int, bytes]] = []
    try:
        with open(file_path, "rb") as f:
            header = f.read(8)
            if len(header) < 8:
                return []
            
            # Determine endianness
            if header[:2] == b"II":
                endian = "<"
            elif header[:2] == b"MM":
                endian = ">"
            else:
                return []
            
            magic = struct.unpack(endian + "H", header[2:4])[0]
            if magic != 42:
                return []
            
            first_ifd_offset = struct.unpack(endian + "I", header[4:8])[0]
            
            ifds_to_visit = [first_ifd_offset]
            visited_offsets = set()
            
            while ifds_to_visit:
                offset = ifds_to_visit.pop(0)
                if offset == 0 or offset in visited_offsets:
                    continue
                visited_offsets.add(offset)
                
                f.seek(offset)
                num_entries_bytes = f.read(2)
                if len(num_entries_bytes) < 2:
                    continue
                num_entries = struct.unpack(endian + "H", num_entries_bytes)[0]
                
                # Each entry is 12 bytes
                entry_data = f.read(num_entries * 12)
                if len(entry_data) < num_entries * 12:
                    continue
                
                next_ifd_bytes = f.read(4)
                if len(next_ifd_bytes) == 4:
                    next_ifd = struct.unpack(endian + "I", next_ifd_bytes)[0]
                    if next_ifd != 0:
                        ifds_to_visit.append(next_ifd)
                
                jpeg_offset = None
                jpeg_length = None
                sub_ifd_offsets = []
                
                for i in range(num_entries):
                    entry = entry_data[i*12 : (i+1)*12]
                    tag = struct.unpack(endian + "H", entry[0:2])[0]
                    type_val = struct.unpack(endian + "H", entry[2:4])[0]
                    count = struct.unpack(endian + "I", entry[4:8])[0]
                    value_offset = struct.unpack(endian + "I", entry[8:12])[0]
                    
                    if tag == 0x0201:  # JPEGInterchangeFormat
                        jpeg_offset = value_offset
                    elif tag == 0x0202:  # JPEGInterchangeFormatLength
                        jpeg_length = value_offset
                    elif tag == 0x014a:  # SubIFDs
                        # SubIFDs can be a list of offsets
                        if type_val == 4 or type_val == 13:  # LONG or IFD
                            if count == 1:
                                sub_ifd_offsets.append(value_offset)
                            elif count > 1:
                                # Read offsets from file
                                current_pos = f.tell()
                                f.seek(value_offset)
                                offsets_bytes = f.read(count * 4)
                                if len(offsets_bytes) == count * 4:
                                    offsets = list(struct.unpack(endian + count * "I", offsets_bytes))
                                    sub_ifd_offsets.extend(offsets)
                                f.seek(current_pos)
                
                if jpeg_offset is not None and jpeg_length is not None and jpeg_length > 0:
                    # Cheap sizing read only -- the full segment is read once,
                    # below, for whichever single candidate is actually chosen.
                    current_pos = f.tell()
                    f.seek(jpeg_offset)
                    header_chunk = f.read(min(jpeg_length, 65536))
                    f.seek(current_pos)
                    dims = get_jpeg_dimensions(header_chunk)
                    if dims is not None:
                        w, h = dims
                        candidates.append((w * h, w, h, jpeg_offset, jpeg_length, header_chunk))

                # Add sub-IFDs to visit list
                for sub_offset in sub_ifd_offsets:
                    if sub_offset != 0 and sub_offset not in visited_offsets:
                        ifds_to_visit.append(sub_offset)

            if not candidates:
                return []

            # max_size>0: cheapest preview that still covers the request (no
            # point decoding a 32MP preview for a 512px tile). max_size==0:
            # caller wants the best quality available (e.g. full single-image
            # preview), so keep the old "largest wins" behavior.
            if max_size > 0:
                sufficient = [c for c in candidates if max(c[1], c[2]) >= max_size]
                chosen = min(sufficient, key=lambda c: c[0]) if sufficient else max(candidates, key=lambda c: c[0])
            else:
                chosen = max(candidates, key=lambda c: c[0])

            _, _, _, jpeg_offset, jpeg_length, header_chunk = chosen
            if len(header_chunk) >= jpeg_length:
                return [header_chunk[:jpeg_length]]
            f.seek(jpeg_offset)
            jpeg_bytes = f.read(jpeg_length)
            if len(jpeg_bytes) == jpeg_length:
                return [jpeg_bytes]
            return []
    except Exception:
        pass
    return []


def _detect_nef_he_compression(file_path: str) -> Optional[bool]:
    """True if this NEF's NEFCompression value says High Efficiency (13) or
    High Efficiency* (14) -- the Nikon Z8/Z9/Z6III/Z50II codec LibRaw cannot
    decode (no upstream ETA). False if a compression value is found and it's
    something else. None if undetectable (non-Nikon, older maker-note
    format, parse failure) -- callers should treat that as "unknown, try the
    normal decode path," never as a hard error.

    Hand-rolled rather than using the `exifread` library's full maker-note
    parse (already a dependency, used elsewhere for AF regions) because
    that path was benchmarked at ~136ms/file -- too slow to run on every
    NEF. This walks IFD0 -> Exif IFD pointer 0x8769 -> MakerNote tag 0x927C
    -> Nikon's inner "TIFF-within-TIFF" (magic `Nikon\\0` + 2 version bytes
    + 2 padding bytes, then a second TIFF header at makernote_offset+10
    whose own IFD offsets are relative to that +10 base -- a well-
    documented Nikon quirk, not guesswork), then checks two locations for
    the compression value, in order:

    1. Tag 0x0093 (NEFCompression), a plain top-level int16u -- used by
       older/non-Z9-generation bodies.
    2. Tag 0x0051 (an opaque binary sub-block Nikon introduced for the
       Z8/Z9/Z6III generation, "MakerNotes0x51" per exiftool's Nikon.pm),
       byte offset 10 within that block's own data, also an int16u --
       exiftool's source notes NEFCompression was "relocated to
       MakerNotes_0x51 at offset x'0a (Z9)". Verified against 21 real
       Z8/Z9/Z6III NEF files (13 HE/HE*, 8 Lossless): matches LibRaw's own
       decode success/failure on every file, cross-checked directly against
       rawpy (HE/HE* raise LibRawFileUnsupportedError, Lossless decodes).

    Benchmarked at ~0.05ms/file (tag 0x0093 path) / ~0.1ms/file (tag 0x0051
    path) against real files -- both cheap enough to run unconditionally.
    """
    import struct

    try:
        with open(file_path, "rb") as f:
            header = f.read(8)
            if len(header) < 8:
                return None
            if header[:2] == b"II":
                e = "<"
            elif header[:2] == b"MM":
                e = ">"
            else:
                return None
            if struct.unpack(e + "H", header[2:4])[0] != 42:
                return None
            ifd0_offset = struct.unpack(e + "I", header[4:8])[0]

            def read_ifd(offset):
                f.seek(offset)
                n = struct.unpack(e + "H", f.read(2))[0]
                return n, f.read(n * 12)

            n, entries = read_ifd(ifd0_offset)
            if len(entries) < n * 12:
                return None
            exif_ifd_offset = None
            for i in range(n):
                ent = entries[i * 12:(i + 1) * 12]
                if struct.unpack(e + "H", ent[0:2])[0] == 0x8769:  # Exif IFD pointer
                    exif_ifd_offset = struct.unpack(e + "I", ent[8:12])[0]
                    break
            if exif_ifd_offset is None:
                return None

            n, entries = read_ifd(exif_ifd_offset)
            if len(entries) < n * 12:
                return None
            mn_offset = None
            for i in range(n):
                ent = entries[i * 12:(i + 1) * 12]
                if struct.unpack(e + "H", ent[0:2])[0] == 0x927C:  # MakerNote
                    mn_offset = struct.unpack(e + "I", ent[8:12])[0]
                    break
            if mn_offset is None:
                return None

            f.seek(mn_offset)
            blob_header = f.read(10)
            if len(blob_header) < 10 or blob_header[:6] != b"Nikon\x00":
                return None  # older-format maker note, not our concern

            inner_base = mn_offset + 10
            f.seek(inner_base)
            inner_hdr = f.read(8)
            if len(inner_hdr) < 8:
                return None
            if inner_hdr[:2] == b"II":
                ie = "<"
            elif inner_hdr[:2] == b"MM":
                ie = ">"
            else:
                return None
            if struct.unpack(ie + "H", inner_hdr[2:4])[0] != 42:
                return None
            inner_ifd_rel = struct.unpack(ie + "I", inner_hdr[4:8])[0]

            f.seek(inner_base + inner_ifd_rel)
            n2_bytes = f.read(2)
            if len(n2_bytes) < 2:
                return None
            n2 = struct.unpack(ie + "H", n2_bytes)[0]
            entries2 = f.read(n2 * 12)
            if len(entries2) < n2 * 12:
                return None

            tag51_value_offset = None
            tag51_count = None
            for i in range(n2):
                ent = entries2[i * 12:(i + 1) * 12]
                tag = struct.unpack(ie + "H", ent[0:2])[0]
                if tag == 0x0093:  # NEFCompression (older/non-Z9-gen bodies)
                    val = struct.unpack(ie + "H", ent[8:10])[0]
                    return val in (13, 14)
                if tag == 0x0051:  # MakerNotes0x51 (Z8/Z9/Z6III-gen bodies)
                    tag51_count = struct.unpack(ie + "I", ent[4:8])[0]
                    tag51_value_offset = struct.unpack(ie + "I", ent[8:12])[0]

            if tag51_value_offset is not None and tag51_count is not None and tag51_count > 4:
                # Compression int16u lives at byte offset 10 within this
                # sub-block's own data (see docstring); the block itself is
                # large enough that its value is stored via offset, not inline.
                f.seek(inner_base + tag51_value_offset + 10)
                val_bytes = f.read(2)
                if len(val_bytes) == 2:
                    val = struct.unpack(ie + "H", val_bytes)[0]
                    return val in (13, 14)

            return None
    except Exception:
        return None


def extract_embedded_jpeg_by_scan(file_path: str, max_size: int) -> Optional[np.ndarray]:
    """
    Scan for embedded JPEG (SOI … EOI) by first parsing TIFF directories,
    and falling back to sequential head/tail scanning if needed.
    """
    try:
        stat = os.stat(file_path)
        cache_key = (file_path, stat.st_size, stat.st_mtime, max_size)
    except Exception:
        return _extract_embedded_jpeg_by_scan_impl(file_path, max_size)

    with _embedded_scan_coalesce_lock:
        if cache_key in _embedded_scan_results:
            return _embedded_scan_results[cache_key]
        if cache_key in _embedded_scan_inflight:
            event = _embedded_scan_inflight[cache_key]
            leader = False
        else:
            event = threading.Event()
            _embedded_scan_inflight[cache_key] = event
            leader = True

    if not leader:
        event.wait(timeout=120)
        with _embedded_scan_coalesce_lock:
            return _embedded_scan_results.get(cache_key)

    try:
        result = _extract_embedded_jpeg_by_scan_impl(file_path, max_size)
        with _embedded_scan_coalesce_lock:
            if len(_embedded_scan_results) >= _embedded_scan_results_max:
                oldest_key = next(iter(_embedded_scan_results))
                _embedded_scan_results.pop(oldest_key, None)
            _embedded_scan_results[cache_key] = result
        return result
    finally:
        with _embedded_scan_coalesce_lock:
            _embedded_scan_inflight.pop(cache_key, None)
        event.set()


def _extract_embedded_jpeg_by_scan_impl(file_path: str, max_size: int) -> Optional[np.ndarray]:
    try:
        stat = os.stat(file_path)
        size = stat.st_size
        miss_key = (file_path, size, stat.st_mtime, max_size)
        with _embedded_scan_miss_lock:
            if miss_key in _embedded_scan_miss_cache:
                return None

        # Try TIFF parsing first
        tiff_previews = extract_previews_via_tiff_parse(file_path, max_size)
        if tiff_previews:
            candidates = []
            for jpeg_bytes in tiff_previews:
                dims = get_jpeg_dimensions(jpeg_bytes)
                if dims is not None:
                    w, h = dims
                    candidates.append((w * h, jpeg_bytes, w, h))
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                import logging
                logger = logging.getLogger(__name__)
                for area, segment, w, h in candidates:
                    decoded = decode_embedded_jpeg_bytes(segment, max_size)
                    if decoded is not None:
                        logger.info(
                            "[TIFF_PARSE] Found and successfully decoded preview: %dx%d for %s",
                            w, h, os.path.basename(file_path)
                        )
                        return decoded
        # Canon CR3 (BMFF ftyp crx): preview JPEG lives in uuid boxes, not TIFF IFD.
        if os.path.splitext(file_path)[1].lower() == ".cr3":
            cr3_thumb = _extract_bmff_uuid_embedded_jpeg(file_path, max_size)
            if cr3_thumb is not None:
                import logging
                logging.getLogger(__name__).info(
                    "[BMFF_UUID] Decoded CR3 preview for %s shape=%s",
                    os.path.basename(file_path),
                    getattr(cr3_thumb, "shape", cr3_thumb),
                )
                return cr3_thumb
        # TIFF / BMFF parse failed or yielded no decodable preview; cache as miss and return None
        with _embedded_scan_miss_lock:
            if len(_embedded_scan_miss_cache) >= _embedded_scan_miss_cache_max:
                _embedded_scan_miss_cache.clear()
            _embedded_scan_miss_cache.add(miss_key)
        return None
    except Exception:
        return None

class ThumbnailExtractor(QObject):
    """Fast thumbnail extractor for immediate display."""

    def __init__(self):
        super().__init__()

    def extract_thumbnail_from_raw(self, file_path: str, max_size: int = 1024,
                                   allow_scan_fallback: bool = True,
                                   raw_object: Optional[rawpy.RawPy] = None) -> Optional[np.ndarray]:
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
                with _rawpy_global_lock:
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
                            try:
                                os.remove(tmp_name)
                            except Exception: pass
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
                                 allow_scan_fallback: bool = True) -> Optional[np.ndarray]:
        """Extract high-quality preview from RAW file (embedded JPEG)."""
        # Reuse thumbnail logic but with larger size
        return self.extract_thumbnail_from_raw(
            file_path,
            max_size=max_size,
            allow_scan_fallback=allow_scan_fallback,
        )

    def extract_embedded_native_preview(
        self, file_path: str, allow_scan_fallback: bool = True
    ) -> Optional[np.ndarray]:
        """Extract embedded JPEG at native resolution (max_size=0 disables downscaling)."""
        return self.extract_thumbnail_from_raw(
            file_path,
            max_size=0,
            allow_scan_fallback=allow_scan_fallback,
        )

    def extract_thumbnail_from_image(self, file_path: str, max_size: int = 1024,
                                     target_size: Optional[QSize] = None) -> Optional[Union[np.ndarray, QImage]]:
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



class EXIFExtractor(QObject):
    """Fast EXIF data extractor with caching."""

    def __init__(self):
        super().__init__()
        self.cache = get_image_cache()

    def _persist_exif_result(self, file_path: str, result: Optional[Dict[str, Any]]) -> None:
        """Write freshly extracted metadata to memory + persistent EXIF cache."""
        if not file_path or not isinstance(result, dict) or not result:
            return
        try:
            self.cache.put_exif(file_path, result)
        except Exception:
            pass

    def build_minimal_raw_exif(
        self, file_path: str, raw_object: rawpy.RawPy
    ) -> Dict[str, Any]:
        """Fast sensor/orientation metadata from an open rawpy handle (no exiftool)."""
        orientation = 1
        original_width = 0
        original_height = 0
        try:
            sizes = raw_object.sizes
            original_width = int(sizes.width)
            original_height = int(sizes.height)
            if sizes.flip != 0:
                flip_map = {0: 1, 3: 3, 5: 8, 6: 6}
                orientation = flip_map.get(sizes.flip, sizes.flip)
        except Exception:
            pass
        if orientation <= 1:
            embedded_o = _orientation_from_embedded_preview(file_path, raw_object)
            if embedded_o not in (1, orientation):
                orientation = embedded_o
        return {
            "orientation": orientation,
            "camera_make": "",
            "camera_model": "",
            "exif_data": {},
            "original_width": original_width,
            "original_height": original_height,
            "capture_time": None,
            "focal_length": None,
            "aperture": None,
            "iso": None,
            "verified_orientation": True,
            "raw_exif_sensor_meta_ver": RAW_EXIF_SENSOR_META_VER,
            "minimal_preview_exif": True,
        }

    def extract_exif_data(self, file_path: str, raw_object: Optional[rawpy.RawPy] = None) -> Optional[Dict[str, Any]]:
        """Extract EXIF data from image file with RAW-specific orientation fallbacks."""
        # Check SQLite cache first (verify=True: this is the canonical
        # orientation source for callers that rotate pixels).
        cached = self.cache.get_exif(file_path, verify=True)
        if cached and not cached.get("minimal_preview_exif"):
            cached_ver = cached.get('raw_exif_sensor_meta_ver', 0)
            if cached_ver < RAW_EXIF_SENSOR_META_VER:
                if os.environ.get("RAWVIEWER_VERBOSE_ORIENTATION_LOGS") == "1":
                    # print(f"[ORIENTATION] EXIFExtractor: Stale cache version ({cached_ver} < {RAW_EXIF_SENSOR_META_VER}) for {os.path.basename(file_path)}, forcing re-extraction...")
                    pass
            elif cached_raw_exif_orientation_trustworthy(file_path, cached):
                if os.environ.get("RAWVIEWER_VERBOSE_ORIENTATION_LOGS") == "1":
                    # print(f"[ORIENTATION] EXIFExtractor: Found valid cached orientation={cached.get('orientation')} for {os.path.basename(file_path)}")
                    pass
                return cached
            else:
                try:
                    self.cache.exif_cache.remove(file_path)
                    self.cache.exif_memory_cache.remove(file_path)
                except Exception:
                    pass

        try:
            import metadata_backend

            tags = metadata_backend.process_file_from_path(file_path, details=False)
        except Exception as e:
            import logging
            from common_image_loader import is_emfile_error, note_emfile_pressure

            logging.getLogger(__name__).warning(f"metadata_backend failed on {file_path}: {e}")
            if is_emfile_error(e):
                note_emfile_pressure()
            tags = {}

        try:
            # Standard orientation tags
            orientation = 1
            orientation_tag_found = None
            
            # DEBUG: Log all potential orientation tags to help troubleshoot
            if os.environ.get("RAWVIEWER_VERBOSE_ORIENTATION_LOGS") == "1":
                orient_tags = {k: v for k, v in tags.items() if 'orient' in k.lower()}
                if orient_tags:
                    # print(f"[ORIENTATION] EXIFExtractor debug for {os.path.basename(file_path)}: Found orientation-like tags: {orient_tags}")
                    pass

            for tag_name in ('Image Orientation', 'EXIF Orientation', 'Orientation', 'MakerNote Orientation', 'EXIF SceneType', 'Sony Orientation', 'Sony Orientation 2'):
                tag = tags.get(tag_name)
                if tag:
                    # Try to get the numeric value directly
                    try:
                        if hasattr(tag, 'values') and tag.values:
                            val = tag.values[0]
                            if isinstance(val, int):
                                orientation = val
                                orientation_tag_found = tag_name
                                break
                    except Exception: pass
                    
                    # Fallback to string mapping (same as main.py)
                    orientation_str = str(tag).strip()
                    orientation_map = {
                        'Horizontal (normal)': 1,
                        'Mirrored horizontal': 2,
                        'Rotated 180': 3,
                        'Mirrored vertical': 4,
                        'Mirrored horizontal then rotated 90 CCW': 5,
                        'Rotated 90 CW': 6,
                        'Mirrored horizontal then rotated 90 CW': 7,
                        'Rotated 90 CCW': 8
                    }
                    if orientation_str in orientation_map:
                        orientation = orientation_map[orientation_str]
                        orientation_tag_found = tag_name
                        break
            
            if os.environ.get("RAWVIEWER_VERBOSE_ORIENTATION_LOGS") == "1" and orientation != 1:
                # print(f"[ORIENTATION] EXIFExtractor: Found orientation={orientation} via tag '{orientation_tag_found}' for {os.path.basename(file_path)}")
                pass
            
            camera_make = str(tags.get('Image Make', '')).strip()
            camera_model = str(tags.get('Image Model', '')).strip()
            
            exif_dict = {k: str(v) for k, v in tags.items()}
            
            # Dimensions
            original_width = 0
            original_height = 0
            
            # 1. Try DNG DefaultCropSize (often found in Apple ProRAW and Android DNGs)
            crop_size = tags.get('Image DefaultCropSize')
            if crop_size and hasattr(crop_size, 'values') and len(crop_size.values) >= 2:
                try:
                    original_width = int(crop_size.values[0])
                    original_height = int(crop_size.values[1])
                except Exception: pass

            # 2. Try DNG ActiveArea (Top, Left, Bottom, Right)
            if original_width <= 0 or original_height <= 0:
                active_area = tags.get('Image ActiveArea')
                if active_area and hasattr(active_area, 'values') and len(active_area.values) >= 4:
                    try:
                        # Values: Top, Left, Bottom, Right
                        top = int(active_area.values[0])
                        left = int(active_area.values[1])
                        bottom = int(active_area.values[2])
                        right = int(active_area.values[3])
                        original_height = bottom - top
                        original_width = right - left
                    except Exception: pass
            
            # 3. Fallback to standard EXIF width/height
            if original_width <= 0:
                for tag in ('EXIF ExifImageWidth', 'Image ImageWidth'):
                    if tag in tags:
                        try: 
                            original_width = int(tags[tag].values[0])
                            break
                        except Exception: pass
            
            if original_height <= 0:
                for tag in ('EXIF ExifImageLength', 'Image ImageLength'):
                    if tag in tags:
                        try:
                            original_height = int(tags[tag].values[0])
                            break
                        except Exception: pass

            # Second pass: If it's a RAW file, use rawpy to verify dimensions and
            # orientation (flip), and -- while the file is already open -- also
            # resolve the embedded-preview orientation fallback in the SAME
            # rawpy.imread() call. These used to be two separate opens (one here,
            # one in the block below), and each rawpy.imread() on a RAW file is a
            # full LibRaw parse (tens of ms even for embedded-preview-only use) --
            # doubling that cost on every cold gallery thumbnail for no reason.
            import common_image_loader
            if common_image_loader.is_raw_file(file_path):
                try:
                    if raw_object is not None:
                        sizes = raw_object.sizes
                        original_width = sizes.width
                        original_height = sizes.height
                        if orientation == 1 and sizes.flip != 0:
                            # Map LibRaw flip codes to EXIF Orientation
                            # LibRaw flip: 0=0, 3=180, 5=90CCW, 6=90CW
                            # EXIF Orient: 1=0, 3=180, 8=90CCW, 6=90CW
                            flip_map = {0: 1, 3: 3, 5: 8, 6: 6}
                            orientation = flip_map.get(sizes.flip, sizes.flip)
                        if orientation <= 1:
                            embedded_o = _orientation_from_embedded_preview(file_path, raw_object)
                            if embedded_o not in (1, orientation):
                                orientation = embedded_o
                    else:
                        yield_if_current_task_active()
                        with _rawpy_global_lock:
                            with rawpy_imread_warm(file_path) as raw:
                                sizes = raw.sizes
                                original_width = sizes.width
                                original_height = sizes.height
                                if orientation == 1 and sizes.flip != 0:
                                    flip_map = {0: 1, 3: 3, 5: 8, 6: 6}
                                    orientation = flip_map.get(sizes.flip, sizes.flip)
                                if orientation <= 1:
                                    embedded_o = _orientation_from_embedded_preview(file_path, raw)
                                    if embedded_o not in (1, orientation):
                                        orientation = embedded_o
                except Exception:
                    pass

            # Third pass: If dimensions are still 0 (e.g. non-RAW missing tags), use QImageReader (fast header read)
            if original_width <= 0 or original_height <= 0:
                try:
                    reader = QImageReader(file_path)
                    size = reader.size()
                    if size.isValid():
                        # QImageReader.size() takes autoTransform into account if set.
                        # Since we want to provide dimensions that will be combined with 
                        # 'orientation' later, we should be careful.
                        # But standard JPEGs with missing tags usually have orientation=1.
                        original_width = size.width()
                        original_height = size.height()
                except Exception:
                    pass

            # Fourth pass: If on macOS and missing dimensions, fallback to native sips
            if (original_width <= 0 or original_height <= 0) and sys.platform == 'darwin':
                try:
                    import subprocess
                    result = subprocess.run(['sips', '-g', 'pixelWidth', '-g', 'pixelHeight', '-g', 'orientation', file_path], 
                                          capture_output=True, text=True)
                    for line in result.stdout.splitlines():
                        if 'pixelWidth:' in line and original_width <= 0:
                            original_width = int(line.split(':')[1].strip())
                        elif 'pixelHeight:' in line and original_height <= 0:
                            original_height = int(line.split(':')[1].strip())
                        elif 'orientation:' in line and orientation == 1:
                            val = line.split(':')[1].strip()
                            # sips returns e.g. "1 (Normal)", "6 (Rotated 90 CW)", "8 (Rotated 90 CCW)"
                            import re
                            m = re.match(r'^(\d+)', val)
                            if m:
                                orientation = int(m.group(1))
                    if os.environ.get("RAWVIEWER_VERBOSE_ORIENTATION_LOGS") == "1":
                        print(f"[EXIF] Extracted via sips: {original_width}x{original_height}, orientation={orientation} for {os.path.basename(file_path)}")
                except Exception:
                    pass

            # Extract technical metadata for top-level cache columns
            focal_length = None
            aperture = None
            iso = None
            
            # Focal Length
            for tag in ('EXIF FocalLength', 'EXIF FocalLengthIn35mmFilm'):
                if tag in exif_dict:
                    try:
                        val = exif_dict[tag]
                        if '/' in val:
                            num, den = val.split('/')
                            focal_length = f"{round(float(num) / float(den))}mm"
                        else:
                            focal_length = f"{round(float(val))}mm"
                        break
                    except Exception: pass
            
            # Aperture
            for tag in ('EXIF FNumber', 'EXIF ApertureValue'):
                if tag in exif_dict:
                    try:
                        val = exif_dict[tag]
                        if '/' in val:
                            num, den = val.split('/')
                            aperture = f"f/{float(num) / float(den):.1f}"
                        else:
                            aperture = f"f/{float(val):.1f}"
                        break
                    except Exception: pass
            
            # ISO
            for tag in ('EXIF ISOSpeedRatings', 'EXIF ISO', 'EXIF PhotographicSensitivity'):
                if tag in exif_dict:
                    try:
                        val = exif_dict[tag]
                        if '/' in val:
                            num, den = val.split('/')
                            iso = f"ISO {int(float(num) / float(den))}"
                        else:
                            iso = f"ISO {val}"
                        break
                    except Exception: pass

            rating = 0
            for tag in ('XMP-xmp:Rating', 'XMP Rating', 'Rating', 'EXIF Rating'):
                if tag in tags:
                    try:
                        from raw_adjustments import _parse_rating_value

                        rating = _parse_rating_value(tags[tag])
                        if rating:
                            break
                    except Exception:
                        pass
                if tag in exif_dict:
                    try:
                        from raw_adjustments import _parse_rating_value

                        rating = _parse_rating_value(exif_dict[tag])
                        if rating:
                            break
                    except Exception:
                        pass
            if not rating:
                try:
                    from raw_adjustments import load_rating_for_file

                    rating = load_rating_for_file(file_path)
                except Exception:
                    pass

            # Build return dict
            result = {
                'orientation': orientation,
                'camera_make': camera_make,
                'camera_model': camera_model,
                'exif_data': exif_dict,
                'original_width': original_width,
                'original_height': original_height,
                'capture_time': (
                    normalize_capture_time_string(
                        exif_dict.get('EXIF DateTimeOriginal') or exif_dict.get('Image DateTime')
                    )
                    or exif_dict.get('EXIF DateTimeOriginal')
                    or exif_dict.get('Image DateTime')
                ),
                'focal_length': focal_length,
                'aperture': aperture,
                'iso': iso,
                'rating': rating,
                'verified_orientation': True,
                'raw_exif_sensor_meta_ver': RAW_EXIF_SENSOR_META_VER,
            }

            if file_path.lower().endswith(".nef"):
                # Proactive Nikon HE/HE* detection (see _detect_nef_he_compression):
                # lets callers route straight to the embedded-JPEG fallback instead
                # of paying for one failed LibRaw decode attempt per file.
                result['nef_he_compressed'] = _detect_nef_he_compression(file_path)
            
            if os.environ.get("RAWVIEWER_VERBOSE_ORIENTATION_LOGS") == "1":
                # print(f"[ORIENTATION] EXIFExtractor: Successfully returning metadata with orientation={orientation} for {os.path.basename(file_path)}")
                pass

            self._persist_exif_result(file_path, result)
            return result
            
        except Exception as e:
            if os.environ.get("RAWVIEWER_VERBOSE_ORIENTATION_LOGS") == "1":
                # print(f"[ORIENTATION] EXIFExtractor error for {os.path.basename(file_path)}: {e}")
                pass
            pass

        return {
            'orientation': orientation if 'orientation' in locals() else 1,
            'camera_make': camera_make if 'camera_make' in locals() else '',
            'camera_model': camera_model if 'camera_model' in locals() else '',
            'exif_data': {},
            'verified_orientation': True
        }


class OptimizedRAWProcessor(QObject):
    """Optimized RAW processor with format-specific optimizations."""

    def __init__(self):
        super().__init__()

    def is_canon_camera(self, file_path: str, exif_data: Dict[str, Any] = None) -> bool:
        """Check if this is a Canon camera."""
        # Check file extension first (faster)
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in ['.cr2', '.cr3']:
            return True

        # Check EXIF if available
        if exif_data and exif_data.get('camera_make'):
            return 'CANON' in exif_data['camera_make'].upper()

        return False

    def is_fujifilm_camera(self, file_path: str, exif_data: Dict[str, Any] = None) -> bool:
        """Check if this is a Fujifilm camera."""
        # Check file extension first (faster)
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in ['.raf']:
            return True

        # Check EXIF if available
        if exif_data and exif_data.get('camera_make'):
            make = exif_data['camera_make'].upper()
            return 'FUJIFILM' in make or 'FUJI' in make

        return False

    def is_sony_camera(self, file_path: str, exif_data: Dict[str, Any] = None) -> bool:
        """Check if this is a Sony camera."""
        # Check file extension first (faster)
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in ['.arw']:
            return True

        # Check EXIF if available
        if exif_data and exif_data.get('camera_make'):
            return 'SONY' in exif_data['camera_make'].upper()

        return False

    def get_optimized_processing_params(self, file_path: str, exif_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get optimized processing parameters based on camera and file."""
        # Default fast processing parameters
        params = {
            'use_camera_wb': True,
            'use_auto_wb': False,
            'output_bps': 8,  # 8-bit for speed
            'no_auto_bright': True,
            'gamma': (2.222, 4.5),  # Standard sRGB gamma
            'bright': 1.0,
            # 'highlight': 0,  # Removed for rawpy 0.25.0 compatibility
            # 'shadow': 0,  # Removed for rawpy 0.25.0 compatibility
            'user_flip': 0,  # Force rawpy to ignore EXIF orientation
            'demosaic_algorithm': rawpy.DemosaicAlgorithm.LINEAR, # MUCH faster than default for fast previews
        }

        # Get file size for processing decisions
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        except Exception:
            file_size_mb = 50  # Default assumption

        # Adjust for large files (use faster processing)
        if file_size_mb > 80:
            params.update({
                'half_size': True,  # Process at half resolution for speed
                'use_camera_wb': True,
                'no_auto_bright': True
            })

        # Camera-specific optimizations
        if self.is_canon_camera(file_path, exif_data):
            # Canon cameras benefit from camera white balance
            params.update({
                'use_camera_wb': True,
                'use_auto_wb': False
            })
        elif self.is_fujifilm_camera(file_path, exif_data):
            # Fujifilm X-Trans sensors
            # DCB is high quality but slower - disable for fast processing
            params.update({
                'use_camera_wb': True,
                'dcb_enhance': False if file_size_mb > 20 else True 
            })
        elif self.is_sony_camera(file_path, exif_data):
            # Sony cameras
            params.update({
                'use_camera_wb': True,
                'use_auto_wb': False
            })

        return params

    def process_raw_fast(self, file_path: str, exif_data: Dict[str, Any] = None) -> Optional[np.ndarray]:
        """Process RAW file with optimized parameters for speed."""
        try:
            yield_if_current_task_active()
            with _rawpy_global_lock:
                raw_ctx = rawpy_imread_warm(file_path)
            with raw_ctx as raw:
                params = self.get_optimized_processing_params(file_path, exif_data)
                with _heavy_fallback_semaphore:
                    rgb_image = raw.postprocess(**params)

                    return rgb_image

        except Exception as e:
            # Log the actual error for debugging
            # print(f"RAW processing error in process_raw_fast: {str(e)}")
            pass
            return None

    def process_raw_quality(self, file_path: str, exif_data: Dict[str, Any] = None) -> Optional[np.ndarray]:
        """Process RAW file with quality parameters (slower)."""
        try:
            yield_if_current_task_active()
            with _rawpy_global_lock:
                raw_ctx = rawpy_imread_warm(file_path)
            with raw_ctx as raw:
                # Quality processing parameters
                params = {
                    'use_camera_wb': True,
                    'output_bps': 16,  # 16-bit for quality
                    'gamma': (1, 1),   # Linear gamma for better quality
                    'no_auto_bright': True,
                    'bright': 1.0
                }

                # Camera-specific quality adjustments
                if self.is_canon_camera(file_path, exif_data):
                    params.update({
                        'use_camera_wb': True,
                        # 'highlight': 1,  # Removed for rawpy 0.25.0 compatibility
                    })
                elif self.is_fujifilm_camera(file_path, exif_data):
                    params.update({
                        'dcb_enhance': True,
                        'fbdd_noise_reduction': rawpy.FBDDNoiseReductionMode.Full
                    })

                with _heavy_fallback_semaphore:
                    rgb_image = raw.postprocess(**params)

                # Convert 16-bit to 8-bit for display
                if rgb_image is not None and hasattr(rgb_image, 'dtype') and rgb_image.dtype == np.uint16:
                    rgb_image = (rgb_image / 256).astype(np.uint8)

                return rgb_image

        except Exception as e:
            # Log the actual error for debugging
            # print(f"RAW processing error in process_raw_quality: {str(e)}")
            pass
            return None


class EnhancedRAWProcessor(QThread):
    """Enhanced RAW processor with progressive loading and caching."""

    # Signals
    thumbnail_ready = pyqtSignal(object)  # np.ndarray thumbnail
    image_processed = pyqtSignal(object)  # np.ndarray full image
    error_occurred = pyqtSignal(str)
    processing_progress = pyqtSignal(str)  # Status message
    exif_data_ready = pyqtSignal(dict)  # EXIF data dictionary

    def __init__(self, file_path: str, use_quality_processing: bool = False):
        super().__init__()
        self.file_path = file_path
        self.use_quality_processing = use_quality_processing
        self.cache = get_image_cache()
        self.thumbnail_extractor = ThumbnailExtractor()
        self.exif_extractor = EXIFExtractor()
        self.raw_processor = OptimizedRAWProcessor()

        # Determine if this is a RAW file
        self.is_raw_file = self._is_raw_file(file_path)

        # Processing state
        self._should_stop = threading.Event()

    def _is_raw_file(self, file_path: str) -> bool:
        """Check if file is a RAW format."""
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        return ext in RAW_FILE_EXTENSIONS

    def stop_processing(self):
        """Stop the processing thread."""
        self._should_stop.set()

    def run(self):
        """Main processing loop with progressive loading."""
        try:
            if not os.path.exists(self.file_path):
                self.error_occurred.emit(f"File not found: {self.file_path}")
                return

            # Step 1: Check cache for full image first
            cached_image = self.cache.get_full_image(self.file_path)
            if cached_image is not None and not self._should_stop.is_set():
                self.processing_progress.emit("Loaded from cache")
                self.image_processed.emit(cached_image)
                return

            # Step 2: Extract and emit thumbnail immediately
            if not self._should_stop.is_set():
                self._process_thumbnail()

            # Step 3: Extract EXIF data
            exif_data = None
            if not self._should_stop.is_set():
                self.processing_progress.emit("Reading metadata...")
                exif_data = self.exif_extractor.extract_exif_data(
                    self.file_path)

                # Emit EXIF data ready signal for immediate status bar update
                if exif_data:
                    self.exif_data_ready.emit(exif_data)

            # Step 4: Process full image
            if not self._should_stop.is_set():
                self._process_full_image(exif_data)

        except Exception as e:
            if not self._should_stop.is_set():
                self.error_occurred.emit(f"Processing error: {str(e)}")

    def _process_thumbnail(self):
        """Process and emit thumbnail."""
        # Check thumbnail cache first
        cached_thumbnail = self.cache.get_thumbnail(self.file_path)
        if cached_thumbnail is not None:
            self.thumbnail_ready.emit(cached_thumbnail)
            return

        self.processing_progress.emit("Extracting preview...")

        # Extract thumbnail
        thumbnail = None
        if self.is_raw_file:
            thumbnail = self.thumbnail_extractor.extract_thumbnail_from_raw(
                self.file_path)
        else:
            thumbnail = self.thumbnail_extractor.extract_thumbnail_from_image(
                self.file_path)

        if thumbnail is not None:
            # Cache the thumbnail
            self.cache.put_thumbnail(self.file_path, thumbnail, None)

            # Emit thumbnail
            self.thumbnail_ready.emit(thumbnail)
        else:
            # No thumbnail available, will wait for full processing
            pass

    def _process_full_image(self, exif_data: Dict[str, Any]):
        """Process full resolution image."""
        if self.is_raw_file:
            self._process_raw_image(exif_data)
        else:
            self._process_regular_image(exif_data)

    def _process_raw_image(self, exif_data: Dict[str, Any]):
        """Process RAW image file."""
        self.processing_progress.emit("Processing RAW image...")

        # Try to extract thumbnail as fallback first
        thumbnail_fallback = None
        try:
            thumbnail_fallback = self.thumbnail_extractor.extract_thumbnail_from_raw(
                self.file_path)
        except Exception:
            pass

        # Choose processing method based on quality setting
        try:
            if self.use_quality_processing:
                rgb_image = self.raw_processor.process_raw_quality(
                    self.file_path, exif_data)
            else:
                rgb_image = self.raw_processor.process_raw_fast(
                    self.file_path, exif_data)

            if rgb_image is not None:
                # Apply orientation correction
                orientation = exif_data.get(
                    'orientation', 1) if exif_data else 1
                if rgb_image is not None:
                    rgb_image = self._apply_orientation_correction(
                        rgb_image, orientation, exif_data)

                # Cache the processed image
                self.cache.put_full_image(self.file_path, rgb_image)

                # Emit the result
                self.processing_progress.emit("Processing complete")
                self.image_processed.emit(rgb_image)
            else:
                # RAW processing failed, try fallback
                if thumbnail_fallback is not None:
                    # Use thumbnail as fallback
                    orientation = exif_data.get(
                        'orientation', 1) if exif_data else 1
                    thumbnail_fallback = self._apply_orientation_correction(
                        thumbnail_fallback, orientation, exif_data)

                    self.processing_progress.emit(
                        "Using embedded thumbnail (RAW processing failed)")
                    self.image_processed.emit(thumbnail_fallback)
                else:
                    self.error_occurred.emit("Failed to process RAW file")

        except Exception as e:
            # Provide more specific error messages based on the error type
            error_msg = str(e)
            if "data corrupted" in error_msg.lower():
                error_msg = f"RAW processing failed due to LibRaw compatibility issue.\n\nThis is a known issue with LibRaw and certain RAW files.\nTry using a different RAW processor or contact the developer for updates.\n\nOriginal error: {error_msg}"
            elif "unsupported file format" in error_msg.lower():
                error_msg = f"This RAW file format may not be supported by your LibRaw version.\n\nOriginal error: {error_msg}"
            elif "input/output error" in error_msg.lower():
                error_msg = f"Cannot read the file. It may be corrupted or in use by another program.\n\nOriginal error: {error_msg}"
            elif "cannot allocate memory" in error_msg.lower():
                error_msg = f"Not enough memory to process this large RAW file.\n\nOriginal error: {error_msg}"

            # Try fallback if available
            if thumbnail_fallback is not None:
                orientation = exif_data.get(
                    'orientation', 1) if exif_data else 1
                thumbnail_fallback = self._apply_orientation_correction(
                    thumbnail_fallback, orientation, exif_data)

                self.processing_progress.emit(
                    "Using embedded thumbnail (RAW processing failed)")
                self.image_processed.emit(thumbnail_fallback)
            else:
                self.error_occurred.emit(
                    f"Error processing RAW file:\n{error_msg}")

    def _process_regular_image(self, exif_data: Dict[str, Any]):
        """Process regular image file (JPEG, TIFF, etc.)."""
        self.processing_progress.emit("Loading image...")

        # For non-RAW files, we emit None to let the main thread handle with QPixmap
        # This is more efficient for regular image formats
        self.image_processed.emit(None)

    def _apply_orientation_correction(self, image_array: np.ndarray, orientation: int, exif_data: Dict[str, Any] = None) -> np.ndarray:
        """Apply orientation correction to image array."""
        if image_array is None:
            return None
        from common_image_loader import apply_container_orientation_to_array

        # Make-independent: apply_container_orientation_to_array measures the pixels
        # against the expected display orientation and no-ops when they already match,
        # so pre-rotated cameras (Sony/Leica/etc.) need no special-case skip here.
        return apply_container_orientation_to_array(
            image_array, orientation, exif_data
        )

    def _is_camera_pre_rotated(self, exif_data: Dict[str, Any] = None) -> bool:
        """Check if camera stores data pre-rotated."""
        if not exif_data or not exif_data.get('camera_make'):
            return False

        make = exif_data['camera_make'].upper()
        # Sony, Leica, and Hasselblad often store RAW data pre-rotated
        return any(brand in make for brand in ['SONY', 'LEICA', 'HASSELBLAD'])


class PreloadManager(QObject):
    """Manages preloading of adjacent images for fast navigation using RAWProcessor (v0.5 style)."""

    def __init__(self, max_preload_threads: int = 2, processor_class=None):
        super().__init__()
        self.max_preload_threads = max_preload_threads
        self.active_threads = {}
        # Threads that were asked to stop but didn't finish in time. Keep them
        # referenced until they finish so their QThread wrapper is never
        # destroyed while still running (we never call QThread.terminate()).
        self._lingering_threads = []
        self.cache = get_image_cache()
        # Accept processor class as parameter to avoid circular imports
        # If None, will use EnhancedRAWProcessor as fallback
        self.processor_class = processor_class

    def _is_raw_file(self, file_path: str) -> bool:
        """Check if file is a RAW format."""
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        return ext in RAW_FILE_EXTENSIONS

    def preload_images(self, file_paths: list, priority_order: list = None):
        """Preload images in background threads using RAWProcessor (v0.5 style)."""
        if priority_order is None:
            priority_order = file_paths

        # Cancel existing preload threads
        self.cancel_all_preloads()

        # Use provided processor class or fallback to EnhancedRAWProcessor
        ProcessorClass = self.processor_class if self.processor_class else EnhancedRAWProcessor
        use_raw_processor = (self.processor_class is not None)

        # Start preloading up to max_preload_threads images
        for i, file_path in enumerate(priority_order[:self.max_preload_threads]):
            if file_path not in self.active_threads:
                # Check if already cached
                if self.cache.get_full_image(file_path) is not None:
                    continue

                # Start preload thread
                if use_raw_processor:
                    # Use RAWProcessor (v0.5 style)
                    is_raw = self._is_raw_file(file_path)
                    thread = ProcessorClass(
                        file_path, is_raw=is_raw, use_full_resolution=False)
                    # Connect signals - RAWProcessor uses image_processed signal
                    thread.image_processed.connect(
                        lambda img, fp=file_path: self._on_preload_complete(fp, img))
                    thread.error_occurred.connect(
                        lambda err, fp=file_path: self._on_preload_error(fp, err))
                else:
                    # Fallback to EnhancedRAWProcessor
                    thread = ProcessorClass(
                        file_path, use_quality_processing=False)
                    thread.image_processed.connect(
                        lambda img, fp=file_path: self._on_preload_complete(fp, img))
                
                thread.finished.connect(
                    lambda fp=file_path: self._cleanup_thread(fp))
                
                self.active_threads[file_path] = thread
                thread.start()

    def _on_preload_complete(self, file_path: str, rgb_image):
        """Handle preloaded image completion."""
        # Image is automatically cached by RAWProcessor/EnhancedRAWProcessor
        # Just clean up the thread reference
        if file_path in self.active_threads:
            # Thread will be cleaned up by finished signal
            pass

    def _on_preload_error(self, file_path: str, error_msg: str):
        """Handle preload error."""
        # Log error but don't show to user (preloading is background operation)
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Preload error for {os.path.basename(file_path)}: {error_msg}")
        # Clean up thread on error
        if file_path in self.active_threads:
            self._cleanup_thread(file_path)

    def cancel_all_preloads(self):
        """Cancel all active preload threads."""
        import logging
        logger = logging.getLogger(__name__)
        
        threads_to_cleanup = list(self.active_threads.items())
        logger.debug(f"cancel_all_preloads(): Cancelling {len(threads_to_cleanup)} preload threads")
        
        for file_path, thread in threads_to_cleanup:
            try:
                file_basename = os.path.basename(file_path)
                # Use cleanup() if available (RAWProcessor), otherwise use stop_processing()
                if hasattr(thread, 'cleanup'):
                    # cleanup() will handle thread stopping and rawpy handle closing safely
                    logger.debug(f"cancel_all_preloads(): Calling cleanup() for {file_basename}")
                    thread.cleanup()
                else:
                    # For EnhancedRAWProcessor, use stop_processing and wait
                    logger.debug(f"cancel_all_preloads(): Using stop_processing() for {file_basename}")
                    if hasattr(thread, 'stop_processing'):
                        thread.stop_processing()
                    
                    # OPTIMIZED: Use shorter initial wait, but allow longer if needed
                    # Preload threads are lower priority, so we can be more aggressive
                    if hasattr(thread, 'wait'):
                        wait_result = thread.wait(100)  # Initial 100ms wait (fast path)
                        logger.debug(f"cancel_all_preloads(): wait(100) returned {wait_result} for {file_basename}")
                        if not wait_result and hasattr(thread, 'isRunning') and thread.isRunning():
                            # Thread still running, wait longer for rawpy operations
                            additional_wait = thread.wait(200)  # Additional 200ms for rawpy operations
                            logger.debug(f"cancel_all_preloads(): Additional wait(200) returned {additional_wait} for {file_basename}")
                            if not additional_wait and thread.isRunning():
                                # Never terminate(): killing a thread inside rawpy/LibRaw
                                # native code can corrupt the heap or deadlock. Leave the
                                # thread to finish on its own and keep it referenced so the
                                # QThread wrapper isn't destroyed while running.
                                logger.debug(f"cancel_all_preloads(): Thread for {file_basename} still running; leaving it to finish")
                                self._lingering_threads.append(thread)
                                try:
                                    thread.finished.connect(lambda t=thread: self._discard_lingering(t))
                                except Exception:
                                    pass
            except Exception as e:
                logger.warning(f"cancel_all_preloads(): Error cleaning up thread for {os.path.basename(file_path)}: {e}", exc_info=True)
        
        self.active_threads.clear()
        logger.debug(f"cancel_all_preloads(): All preload threads cancelled")

    def _cleanup_thread(self, file_path: str):
        """Clean up finished thread."""
        if file_path in self.active_threads:
            del self.active_threads[file_path]

    def _discard_lingering(self, thread):
        """Drop a finished lingering thread (safe: it is no longer running)."""
        try:
            self._lingering_threads.remove(thread)
        except ValueError:
            pass
