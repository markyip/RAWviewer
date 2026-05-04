"""
Focus / subject hints for dashed overlay via **pyexiv2 (Exiv2)**.

No ExifTool subprocess: uses :func:`metadata_backend.read_exif_raw_string_dict` when
pyexiv2 is installed (same stack as general EXIF). Falls back to :mod:`exifread_af` in
the viewer when Exiv2 yields nothing.

Includes:
- CIPA SubjectArea / SubjectLocation
- Canon-style AF (AFImage*, AFArea*, AFAreaX/YPositions)
- Nikon contrast-detect AF (AFAreaXPosition center + AFAreaWidth/Height + AFImage*)
- Nikon **AFInfo2** binary (Exiv2 decimal byte string): scan for AF canvas + box → CDAF mapping
- Nikon AFAreaInitial* when values look like pixel rectangles
- Sony FocusLocation / FocusLocation2, FlexibleSpotPosition + FocusFrameSize

Coordinates are interpreted per CIPA EXIF 2.32:
- SubjectArea 2 values: width, height; region centered on image center.
- SubjectArea 3 values: center x, center y, diameter (circle → bounding square).
- SubjectArea 4 values: center x, center y, width, height.
- SubjectLocation: center x, y (deprecated); used as fallback with a small box.

Reference size uses ExifImageWidth/Height, then ImageWidth/Height, else the
display pixmap size (assumes 1:1). Preview rotation from EXIF Orientation is
not remapped here; the box may be offset on rotated JPEG previews.
"""

from __future__ import annotations

import re
import struct
import functools
import os
from typing import Any


def _pick(row: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


def _to_int(x: Any) -> int | None:
    try:
        if x is None:
            return None
        return int(round(float(x)))
    except (TypeError, ValueError):
        return None


def _numbers_from_tag_value(val: Any) -> list[int]:
    if val is None:
        return []
    if isinstance(val, (int, float)):
        v = _to_int(val)
        return [v] if v is not None else []
    if isinstance(val, list):
        out: list[int] = []
        for it in val:
            out.extend(_numbers_from_tag_value(it))
        return out
    s = str(val).strip()
    if not s:
        return []
    # Handle ratios like "1/2" or space-separated "1/2 3/4"
    parts = re.split(r"[\s,]+", s)
    nums: list[int] = []
    for p in parts:
        if "/" in p:
            try:
                a, b = p.split("/", 1)
                nums.append(int(round(float(a) / float(b))))
            except (ValueError, ZeroDivisionError):
                pass
            continue
        m = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", p)
        if m:
            try:
                nums.append(int(round(float(m.group(0)))))
            except ValueError:
                continue
    return nums


def _reference_dimensions(row: dict[str, Any]) -> tuple[int, int]:
    # EXIF standard dimensions.
    ew = _to_int(_pick(row, "ExifImageWidth", "PixelXDimension"))
    eh = _to_int(_pick(row, "ExifImageLength", "PixelYDimension"))
    # Maker-specific or RAW sensor dimensions often provide the true full-res canvas.
    sw = _to_int(_pick(row, "SensorWidth", "RawImageWidth", "AFImageWidth", "ImageWidth"))
    sh = _to_int(_pick(row, "SensorHeight", "RawImageHeight", "AFImageHeight", "ImageHeight"))
    return max(ew or 0, sw or 0), max(eh or 0, sh or 0)


def _rotate_rect(
    rect: tuple[int, int, int, int], iw: int, ih: int, orientation: int
) -> tuple[int, int, int, int]:
    """Rotate (l, t, w, h) from sensor space (iw x ih) to display space."""
    if orientation <= 1:
        return rect
    l, t, w, h = rect
    if orientation == 6:  # 90° CW
        return (ih - t - h, l, h, w)
    if orientation == 8:  # 90° CCW
        return (t, iw - l - w, h, w)
    if orientation == 3:  # 180°
        return (iw - l - w, ih - t - h, w, h)
    if orientation == 2:  # Flip H
        return (iw - l - w, t, w, h)
    if orientation == 4:  # Flip V
        return (l, ih - t - h, w, h)
    if orientation == 5:  # 90° CCW + Flip H
        return (ih - t - h, iw - l - w, h, w)
    if orientation == 7:  # 90° CW + Flip H
        return (t, l, h, w)
    return rect


def _clamp_rect(left: float, top: float, wi: float, hi: float, iw: int, ih: int) -> tuple[int, int, int, int]:
    iw_f, ih_f = float(max(iw, 1)), float(max(ih, 1))
    l = max(0.0, min(left, iw_f - 1.0))
    t = max(0.0, min(top, ih_f - 1.0))
    wcl = max(1.0, min(wi, iw_f - l))
    hcl = max(1.0, min(hi, ih_f - t))
    return int(l), int(t), int(round(wcl)), int(round(hcl))


def _rect_ref_from_subject_area(nums: list[int], iw: int, ih: int) -> tuple[int, int, int, int] | None:
    if len(nums) < 2:
        return None
    if len(nums) == 2:
        w, h = nums[0], nums[1]
        if w < 4 or h < 4:
            return None
        cx, cy = iw / 2.0, ih / 2.0
        return _clamp_rect(cx - w / 2.0, cy - h / 2.0, float(w), float(h), iw, ih)
    if len(nums) == 3:
        cx, cy, d = nums[0], nums[1], nums[2]
        if d < 4:
            return None
        return _clamp_rect(cx - d / 2.0, cy - d / 2.0, float(d), float(d), iw, ih)
    # 4+
    cx, cy, w, h = nums[0], nums[1], nums[2], nums[3]
    if w < 2 or h < 2:
        return None
    return _clamp_rect(cx - w / 2.0, cy - h / 2.0, float(w), float(h), iw, ih)


def _rect_ref_from_subject_location(nums: list[int], iw: int, ih: int) -> tuple[int, int, int, int] | None:
    if len(nums) < 2:
        return None
    cx, cy = nums[0], nums[1]
    side = max(24, min(iw, ih) // 14)
    return _clamp_rect(cx - side / 2.0, cy - side / 2.0, float(side), float(side), iw, ih)


def _split_xy_pairs(seq: list[int]) -> tuple[list[int], list[int]]:
    if len(seq) < 2:
        return [], []
    return list(seq[0::2]), list(seq[1::2])


def _first_size_from_tag(val: Any, minimum: int = 2) -> int | None:
    """First integer in a space- or comma-separated tag (e.g. Canon ``AFAreaWidths``)."""
    for n in _numbers_from_tag_value(val):
        if n >= minimum:
            return n
    return None


def _rect_ref_canon_style_af(row: dict[str, Any], ref_w: int, ref_h: int) -> tuple[int, int, int, int] | None:
    """Canon-style AF rectangle in reference (Exif image) pixel space."""
    fw_t = _pick(row, "AFImageWidth")
    fh_t = _pick(row, "AFImageHeight")
    # Exiv2 uses plural names on many bodies (CR3 / EOS R).
    aw_t = _pick(row, "AFAreaWidth", "AFAreaWidths")
    ah_t = _pick(row, "AFAreaHeight", "AFAreaHeights")
    xp_t = _pick(row, "AFAreaXPositions", "AFXPositions")
    yp_t = _pick(row, "AFAreaYPositions", "AFYPositions")

    fw = _to_int(fw_t) or ref_w
    fh = _to_int(fh_t) or ref_h
    aw = _to_int(aw_t)
    if aw is None or aw < 2:
        aw = _first_size_from_tag(aw_t, 2)
    ah = _to_int(ah_t)
    if ah is None or ah < 2:
        ah = _first_size_from_tag(ah_t, 2)
    if aw is None or ah is None or aw < 2 or ah < 2:
        return None

    xs = _numbers_from_tag_value(xp_t)
    ys = _numbers_from_tag_value(yp_t)
    if len(xs) >= 2 and len(ys) < 1:
        xs, ys = _split_xy_pairs(xs)
    elif len(ys) >= 2 and len(xs) < 1:
        xs, ys = _split_xy_pairs(ys)
    if len(xs) < 1 or len(ys) < 1:
        return None

    x0, y0 = int(xs[0]), int(ys[0])
    # Heuristic for absolute vs center-relative:
    # 1. Negative values are always relative.
    # 2. Small AFImageWidth (100, 1000) is almost always absolute normalized.
    # 3. If SubjectArea is present, use it as a reference for the true center.
    is_absolute = False
    if all(x >= 0 for x in xs) and all(y >= 0 for y in ys):
        if fw in (100, 1000, 128, 512):
            is_absolute = True
        else:
            sa = row.get("SubjectArea")
            if sa:
                sa_nums = _numbers_from_tag_value(sa)
                if len(sa_nums) >= 2:
                    sa_cx, sa_cy = sa_nums[0], sa_nums[1]
                    # Compare distance from SubjectArea center
                    dist_abs = (x0 - sa_cx)**2 + (y0 - sa_cy)**2
                    dist_rel = (fw*0.5 + x0 - sa_cx)**2 + (fh*0.5 + y0 - sa_cy)**2
                    if dist_abs < dist_rel:
                        is_absolute = True
            elif x0 > fw * 0.25 and y0 > fh * 0.25:
                # If no SubjectArea, only assume absolute if far from center
                is_absolute = True
    
    if is_absolute:
        cx, cy = float(x0), float(y0)
    else:
        cx = fw * 0.5 + x0
        cy = fh * 0.5 + y0

    if not (0 <= cx <= fw and 0 <= cy <= fh):
        cx = max(0.0, min(float(cx), float(fw)))
        cy = max(0.0, min(float(cy), float(fh)))

    left = cx - aw / 2.0
    top = cy - ah / 2.0
    sx = ref_w / float(max(fw, 1))
    sy = ref_h / float(max(fh, 1))
    return _clamp_rect(left * sx, top * sy, aw * sx, ah * sy, ref_w, ref_h)


def _rect_ref_nikon_cdaf(row: dict[str, Any], ref_w: int, ref_h: int) -> tuple[int, int, int, int] | None:
    """Nikon: AFAreaX/YPosition are center of AF box in AFImageWidth/Height coordinates."""
    fw = _to_int(_pick(row, "AFImageWidth"))
    fh = _to_int(_pick(row, "AFImageHeight"))
    cx = _to_int(_pick(row, "AFAreaXPosition"))
    cy = _to_int(_pick(row, "AFAreaYPosition"))
    aw = _to_int(_pick(row, "AFAreaWidth"))
    ah = _to_int(_pick(row, "AFAreaHeight"))
    if not (fw and fh and cx is not None and cy is not None and aw and ah):
        return None
    if aw < 2 or ah < 2 or fw < 8 or fh < 8:
        return None
    left = cx - aw / 2.0
    top = cy - ah / 2.0
    sx = ref_w / float(fw)
    sy = ref_h / float(fh)
    return _clamp_rect(left * sx, top * sy, aw * sx, ah * sy, ref_w, ref_h)


def _rect_ref_nikon_initial(row: dict[str, Any], ref_w: int, ref_h: int) -> tuple[int, int, int, int] | None:
    """Nikon AFAreaInitial* when they look like pixel coordinates in full reference space."""
    ix = _to_int(_pick(row, "AFAreaInitialXPosition"))
    iy = _to_int(_pick(row, "AFAreaInitialYPosition"))
    iw = _to_int(_pick(row, "AFAreaInitialWidth"))
    ih = _to_int(_pick(row, "AFAreaInitialHeight"))
    if ix is None or iy is None or iw is None or ih is None:
        return None
    if iw < 2 or ih < 2:
        return None
    # Heuristic: signed small values are not pixel boxes; skip unless plausible pixels
    if abs(ix) < 400 and abs(iy) < 400 and iw <= 256 and ih <= 256:
        if ix >= 0 and iy >= 0 and ix + iw <= ref_w + 4 and iy + ih <= ref_h + 4:
            return _clamp_rect(float(ix), float(iy), float(iw), float(ih), ref_w, ref_h)
    return None


def _bytes_from_space_separated_uint8_string(s: str) -> bytes | None:
    """Exiv2 often prints binary tags as space-separated decimal bytes (0–255)."""
    out = bytearray()
    for p in re.split(r"[\s,;]+", s.strip()):
        if not p:
            continue
        try:
            v = int(p, 10)
        except ValueError:
            continue
        if v < 0 or v > 255:
            return None
        out.append(v)
    return bytes(out) if out else None


def _nikon_afinfo2_unpack_candidate(blob: bytes, i: int) -> tuple[int, int, int, int, int, int] | None:
    try:
        fw, fh, cx, cy, aw, ah = struct.unpack_from("<6H", blob, i)
    except struct.error:
        return None
    if fw < 160 or fh < 120 or aw < 2 or ah < 2:
        return None
    if cx < -400 or cy < -400 or cx > fw + 400 or cy > fh + 400:
        return None
    if aw > fw + 64 or ah > fh + 64:
        return None
    return fw, fh, cx, cy, aw, ah


def _nikon_afinfo2_rect_and_reference(
    blob: bytes, ref_w: int, ref_h: int
) -> tuple[tuple[int, int, int, int], int, int] | None:
    """
    Nikon MakerNote **AFInfo2** packs AF canvas size and a selected AF rectangle.

    Returns ``(rect, ref_space_w, ref_space_h)`` where *rect* is in that reference
    pixel space. When EXIF reference is missing or is only a thumbnail (small),
    falls back to the AF canvas ``fw×fh`` from the blob so the box still maps to
    the preview pixmap by aspect ratio.
    """
    if len(blob) < 74:
        return None
    lim = min(len(blob) - 12, 128)

    def row_from(u: tuple[int, int, int, int, int, int]) -> dict[str, Any]:
        fw, fh, cx, cy, aw, ah = u
        return {
            "AFImageWidth": fw,
            "AFImageHeight": fh,
            "AFAreaXPosition": cx,
            "AFAreaYPosition": cy,
            "AFAreaWidth": aw,
            "AFAreaHeight": ah,
        }

    # Phase A: EXIF / IFD0 reference looks like full-frame (not a tiny preview only).
    if ref_w >= 2000 and ref_h >= 1200:
        tol_w = max(120, ref_w // 12)
        tol_h = max(120, ref_h // 12)
        for i in range(0, lim, 2):
            u = _nikon_afinfo2_unpack_candidate(blob, i)
            if u is None:
                continue
            fw, fh = u[0], u[1]
            if abs(fw - ref_w) > tol_w or abs(fh - ref_h) > tol_h:
                continue
            r = _rect_ref_nikon_cdaf(row_from(u), ref_w, ref_h)
            if r is not None:
                return r, ref_w, ref_h

    # Phase B: pick the largest plausible full-frame sextuple inside the blob.
    best: tuple[int, tuple[int, int, int, int], int, int] | None = None
    for i in range(0, lim, 2):
        u = _nikon_afinfo2_unpack_candidate(blob, i)
        if u is None:
            continue
        fw, fh = u[0], u[1]
        if fw < 1600 or fh < 1000:
            continue
        ar = fw / float(max(fh, 1))
        if ar < 1.2 or ar > 2.2:
            continue
        r = _rect_ref_nikon_cdaf(row_from(u), fw, fh)
        if r is None:
            continue
        area = fw * fh
        if best is None or area > best[0]:
            best = (area, r, fw, fh)
    if best is None:
        return None
    return best[1], best[2], best[3]




def _rect_ref_sony(row: dict[str, Any], ref_w: int, ref_h: int) -> tuple[int, int, int, int] | None:
    """
    Sony ``FocusLocation`` (Exiv2 / Sony1 or Sony2):

    - **Two values**: center ``(x, y)``.
    - **Four values**: either two corners of a rectangle **or** (common on ILCE ARW)
      ``image_width image_height focus_x focus_y`` — the latter yields an impossible
      corner box (wider than ``ref_w``); we detect that and treat the last pair as the
      center with ``FocusFrameSize`` for the box.
    - **FlexibleSpotPosition** + ``FocusFrameSize`` as fallback.
    """

    def _focus_frame_wh_from_row() -> tuple[float, float] | None:
        fs = _numbers_from_tag_value(_pick(row, "FocusFrameSize"))
        # Sony often pads with zeros, e.g. ``153 0 156 0 1 1``
        pos = [float(x) for x in fs if x >= 2.0][:2]
        if len(pos) >= 2:
            return pos[0], pos[1]
        if len(pos) == 1:
            s = pos[0]
            return s, s
        return None

    def _center_box(cx: float, cy: float) -> tuple[int, int, int, int]:
        wh = _focus_frame_wh_from_row()
        if wh is not None:
            w, h = wh
        else:
            w = h = float(max(24, min(ref_w, ref_h) // 18))
        return _clamp_rect(cx - w / 2.0, cy - h / 2.0, w, h, ref_w, ref_h)

    fl = _pick(row, "FocusLocation2") or _pick(row, "FocusLocation")
    nums = _numbers_from_tag_value(fl)
    if len(nums) >= 4:
        a, b, c, d = nums[0], nums[1], nums[2], nums[3]
        if not (a == 0 and b == 0 and c == 0 and d == 0):
            # ILCE ARW / Sony2: ``image_width image_height focus_x focus_y`` (first pair ≈ sensor size).
            # Must run **before** corner-rectangle logic: otherwise ``wi`` can still be ``<= ref_w``
            # while spanning almost the full frame (bogus corner interpretation).
            if (
                ref_w > 32
                and ref_h > 32
                and abs(a - ref_w) <= max(200, ref_w // 8)
                and abs(b - ref_h) <= max(200, ref_h // 8)
                and 0 <= c <= ref_w + 64
                and 0 <= d <= ref_h + 64
            ):
                return _center_box(float(c), float(d))
            left = float(min(a, c))
            top = float(min(b, d))
            wi = abs(float(c - a)) + 1.0
            hi = abs(float(d - b)) + 1.0
            # Axis-aligned rectangle from two corners, plausible AF size (not strip-wide).
            if (
                wi >= 2.0
                and hi >= 2.0
                and wi <= float(ref_w) + 4.0
                and hi <= float(ref_h) + 4.0
                and wi <= float(ref_w) * 0.65
                and hi <= float(ref_h) * 0.65
            ):
                return _clamp_rect(left, top, wi, hi, ref_w, ref_h)
    # Two-value center (Sony1 tag 0x2027 per Exiv2 docs); do not reuse first two of a failed 4-tuple.
    if 2 <= len(nums) <= 3 and (nums[0] != 0 or nums[1] != 0):
        return _center_box(float(nums[0]), float(nums[1]))

    fp = _numbers_from_tag_value(_pick(row, "FlexibleSpotPosition"))
    if len(fp) >= 2 and (fp[0] != 0 or fp[1] != 0):
        return _center_box(float(fp[0]), float(fp[1]))
    return None


def _ltwh_in_pixmap(
    rect_ref: tuple[int, int, int, int],
    ref_w: int,
    ref_h: int,
    pw: int,
    ph: int,
) -> tuple[int, int, int, int]:
    if ref_w <= 0 or ref_h <= 0:
        return rect_ref
    sx = pw / float(ref_w)
    sy = ph / float(ref_h)
    l, t, w, h = rect_ref
    return _clamp_rect(l * sx, t * sy, w * sx, h * sy, pw, ph)


def _row_from_pyexiv2(path: str) -> dict[str, Any] | None:
    """Flatten ``read_exif_raw_string_dict`` keys to Exiv2 leaf names (last segment)."""
    try:
        from metadata_backend import read_exif_raw_string_dict
    except ImportError:
        return None
    raw = read_exif_raw_string_dict(path)
    if not raw:
        return None
    by_leaf: dict[str, list[tuple[str, str]]] = {}
    for k, v in raw.items():
        v2 = (v or "").strip()
        if not v2:
            continue
        leaf = k.rsplit(".", 1)[-1]
        by_leaf.setdefault(leaf, []).append((k, v2))
    row: dict[str, Any] = {}
    for leaf, pairs in by_leaf.items():
        if len(pairs) > 1:
            # For SubjectArea / Location, prefer the CIPA standard EXIF tags (Photo)
            # over MakerNote duplicates which may use different coordinate spaces.
            if leaf in ("SubjectArea", "SubjectLocation"):
                standard = [kv for kv in pairs if "Exif.Photo." in kv[0] or "Exif.Image." in kv[0]]
                if standard:
                    row[leaf] = standard[0][1]
                    continue
        # Otherwise pick the longest key path (deepest leaf).
        pairs.sort(key=lambda kv: len(kv[0]), reverse=True)
        row[leaf] = pairs[0][1]
    return row


@functools.lru_cache(maxsize=1024)
def _get_focus_hint_sensor_space(path: str) -> tuple[tuple[int, int, int, int], str, int, int] | None:
    """Internal: returns ((l,t,w,h) in sensor space, source, ref_w, ref_h) or None."""
    row = _row_from_pyexiv2(path)
    if not row:
        # Fallback to exifread: only SubjectArea for non-RAW to keep it fast.
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        is_raw = ext in ("cr2", "cr3", "nef", "arw", "dng", "orf", "rw2", "raf")
        
        try:
            from exifread_af import (
                pixmap_ltwh_subject_cipa_from_exifread,
                pixmap_ltwh_af_from_exifread
            )
            # Use 1000x1000 as virtual pixmap to get a normalized rect_ref
            lt_sub = pixmap_ltwh_subject_cipa_from_exifread(path, 1000, 1000)
            if lt_sub:
                return lt_sub, "exif_subject", 1000, 1000
            
            if is_raw:
                lt_af = pixmap_ltwh_af_from_exifread(path, 1000, 1000)
                if lt_af:
                    return lt_af, "maker_af", 1000, 1000
        except:
            pass
        return None

    ref_w, ref_h = _reference_dimensions(row)
    if ref_w <= 0 or ref_h <= 0:
        return None

    # Priority 1: Maker-specific AF tags
    for fn in (
        _rect_ref_canon_style_af,
        _rect_ref_nikon_cdaf,
        _rect_ref_nikon_initial,
        _rect_ref_sony,
    ):
        rect_ref = fn(row, ref_w, ref_h)
        if rect_ref is not None:
            return rect_ref, "maker_af", ref_w, ref_h

    hint_af2 = _rect_ref_from_nikon_afinfo2_row_sensor_space(row, ref_w, ref_h)
    if hint_af2:
        return hint_af2

    # Priority 2: CIPA SubjectArea / SubjectLocation
    sa = row.get("SubjectArea")
    if sa is not None:
        nums = _numbers_from_tag_value(sa)
        rect_ref = _rect_ref_from_subject_area(nums, ref_w, ref_h)
        if rect_ref:
            return rect_ref, "exif_subject", ref_w, ref_h
    
    sl = row.get("SubjectLocation")
    if sl is not None:
        nums = _numbers_from_tag_value(sl)
        rect_ref = _rect_ref_from_subject_location(nums, ref_w, ref_h)
        if rect_ref:
            return rect_ref, "exif_subject", ref_w, ref_h

    return None


def _rect_ref_from_nikon_afinfo2_row_sensor_space(row: dict[str, Any], ref_w: int, ref_h: int) -> tuple[tuple[int, int, int, int], str, int, int] | None:
    raw_s = _pick(row, "AFInfo2")
    if raw_s is None:
        return None
    b = _bytes_from_space_separated_uint8_string(str(raw_s))
    if not b or len(b) < 74:
        return None
    parsed = _nikon_afinfo2_rect_and_reference(b, ref_w, ref_h)
    if parsed:
        rect_ref, rw, rh = parsed
        return rect_ref, "maker_af", rw, rh
    return None


def pixmap_ltwh_focus_hint(
    path: str,
    pixmap_w: int,
    pixmap_h: int,
    orientation: int = 1,
) -> tuple[tuple[int, int, int, int], str] | None:
    """
    Main entry: (left, top, width, height) in pixmap pixels, plus source name.
    Accounts for EXIF orientation when mapping from sensor to display space.
    Results are cached for performance.
    """
    if not path or pixmap_w < 4 or pixmap_h < 4:
        return None
    
    hint_data = _get_focus_hint_sensor_space(path)
    if not hint_data:
        return None
    
    rect_ref, source, ref_w, ref_h = hint_data
    
    # Rotate rect before mapping to final pixmap if display is rotated
    rect_rot = _rotate_rect(rect_ref, ref_w, ref_h, orientation)
    # Swap ref_w/ref_h if 90deg rotated
    rw_rot, rh_rot = (ref_h, ref_w) if orientation in (5, 6, 7, 8) else (ref_w, ref_h)
    
    return (
        _ltwh_in_pixmap(rect_rot, rw_rot, rh_rot, pixmap_w, pixmap_h),
        source,
    )


def pixmap_ltwh_subject_region(
    path: str,
    pixmap_w: int,
    pixmap_h: int,
    orientation: int = 1,
) -> tuple[int, int, int, int] | None:
    """
    Return (left, top, width, height) in **pixmap** pixel coordinates, or None.

    SubjectArea / SubjectLocation only. Prefer :func:`pixmap_ltwh_focus_hint` for RAW
    maker AF (Canon / Sony / Nikon).
    """
    hint = pixmap_ltwh_focus_hint(path, pixmap_w, pixmap_h, orientation)
    if hint is None:
        return None
    ltwh, src = hint
    return ltwh if src == "exif_subject" else None
