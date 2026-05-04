"""
Autofocus / subject region hints via exifread (no ExifTool).

- ``pixmap_ltwh_af_from_exifread``: Canon MakerNote AF (needs ``details=True``).
- ``pixmap_ltwh_subject_cipa_from_exifread``: CIPA **SubjectArea** / **SubjectLocation**
  for JPEG and others (``details=False`` is enough).
"""

from __future__ import annotations

from typing import Any


def _ref_dims(tags: dict[str, Any], fallback_w: int, fallback_h: int) -> tuple[int, int]:
    def _first_positive_int(*keys: str) -> int | None:
        for k in keys:
            t = tags.get(k)
            v = getattr(t, "values", None) if t else None
            if v is None:
                continue
            try:
                x = int(v[0]) if isinstance(v, (list, tuple)) and len(v) > 0 else int(v)
            except (TypeError, ValueError, IndexError):
                continue
            if x > 0:
                return x
        return None

    w = _first_positive_int(
        "EXIF ExifImageWidth",
        "EXIF PixelXDimension",
        "Image ImageWidth",
    )
    h = _first_positive_int(
        "EXIF ExifImageLength",
        "EXIF PixelYDimension",
        "Image ImageLength",
    )
    if w and h:
        return w, h
    if fallback_w > 0 and fallback_h > 0:
        return fallback_w, fallback_h
    return max(1, w or 1), max(1, h or 1)


def _flatten_ints(tag: Any) -> list[int]:
    """Flatten IfdTag values to ints (nested lists / ratios / bytes as 0..255)."""
    if tag is None:
        return []
    v = getattr(tag, "values", None)
    out: list[int] = []

    def walk(x: Any) -> None:
        if x is None:
            return
        if isinstance(x, bool):
            return
        if isinstance(x, int):
            out.append(int(x))
            return
        num = getattr(x, "num", None)
        den = getattr(x, "den", None)
        if num is not None and den is not None:
            try:
                di = int(den)
                out.append(int(round(int(num) / di)) if di else int(num))
            except (TypeError, ValueError):
                pass
            return
        if isinstance(x, bytes):
            for b in x:
                out.append(int(b))
            return
        if isinstance(x, (list, tuple)):
            for item in x:
                walk(item)

    walk(v)
    return out


def _split_xy(seq: list[int]) -> tuple[list[int], list[int]]:
    if len(seq) < 2:
        return [], []
    xs = list(seq[0::2])
    ys = list(seq[1::2])
    return xs, ys


def _clamp_ltwh(left: float, top: float, w: float, h: float, iw: int, ih: int) -> tuple[int, int, int, int]:
    iw_f, ih_f = float(max(iw, 1)), float(max(ih, 1))
    l = max(0.0, min(left, iw_f - 1.0))
    t = max(0.0, min(top, ih_f - 1.0))
    wc = max(1.0, min(w, iw_f - l))
    hc = max(1.0, min(h, ih_f - t))
    return int(l), int(t), int(round(wc)), int(round(hc))


def _scale_to_pixmap(
    ltwh: tuple[int, int, int, int],
    ref_w: int,
    ref_h: int,
    pixmap_w: int,
    pixmap_h: int,
) -> tuple[int, int, int, int] | None:
    l, t, ww, hh = ltwh
    if ref_w <= 0 or ref_h <= 0 or pixmap_w < 4 or pixmap_h < 4:
        return None
    sx = pixmap_w / float(ref_w)
    sy = pixmap_h / float(ref_h)
    return _clamp_ltwh(l * sx, t * sy, ww * sx, hh * sy, pixmap_w, pixmap_h)


def _canon_af_ltwh_in_ref(
    tags: dict[str, Any], ref_w: int, ref_h: int
) -> tuple[int, int, int, int] | None:
    """AF rectangle in EXIF / sensor reference pixel space."""
    fw_t = tags.get("MakerNote AFImageWidth")
    fh_t = tags.get("MakerNote AFImageHeight")
    aw_t = tags.get("MakerNote AFAreaWidth")
    ah_t = tags.get("MakerNote AFAreaHeight")
    xp_t = tags.get("MakerNote AFAreaXPositions")
    yp_t = tags.get("MakerNote AFAreaYPositions")

    def _one_int(tag: Any) -> int | None:
        if tag is None:
            return None
        v = getattr(tag, "values", None)
        if v is None:
            return None
        try:
            if isinstance(v, (list, tuple)) and len(v) >= 1:
                return int(v[0])
            return int(v)
        except (TypeError, ValueError, IndexError):
            return None

    fw = _one_int(fw_t) or ref_w
    fh = _one_int(fh_t) or ref_h
    aw = _one_int(aw_t)
    ah = _one_int(ah_t)
    if aw is None or ah is None or aw < 2 or ah < 2:
        return None

    xs = _flatten_ints(xp_t)
    ys = _flatten_ints(yp_t)
    if len(xs) >= 2 and len(ys) < 1:
        xs, ys = _split_xy(xs)
    elif len(ys) >= 2 and len(xs) < 1:
        xs, ys = _split_xy(ys)
    if len(xs) < 1 or len(ys) < 1:
        return None

    x0, y0 = int(xs[0]), int(ys[0])

    cx = fw * 0.5 + x0
    cy = fh * 0.5 + y0
    if not (0 <= cx <= fw and 0 <= cy <= fh):
        cx, cy = float(x0), float(y0)
    if not (0 <= cx <= fw and 0 <= cy <= fh):
        cx = max(0.0, min(float(cx), float(fw)))
        cy = max(0.0, min(float(cy), float(fh)))

    left = cx - aw / 2.0
    top = cy - ah / 2.0
    # Map from AF window to full reference image (often 1:1 with JPEG)
    sx = ref_w / float(max(fw, 1))
    sy = ref_h / float(max(fh, 1))
    return _clamp_ltwh(
        left * sx, top * sy, aw * sx, ah * sy, ref_w, ref_h
    )


def _canon_tag_0x0012_fallback(
    tags: dict[str, Any], ref_w: int, ref_h: int
) -> tuple[int, int, int, int] | None:
    """If sub-AF tags were not expanded, use raw short array layout (best-effort)."""
    t = tags.get("MakerNote Tag 0x0012")
    if t is None:
        return None
    v = getattr(t, "values", None)
    if not isinstance(v, (list, tuple)) or len(v) < 10:
        return None
    try:
        nums = [int(x) for x in v]
    except (TypeError, ValueError):
        return None
    if len(nums) < 10:
        return None
    fw = max(1, nums[4])
    fh = max(1, nums[5])
    aw = max(2, nums[6])
    ah = max(2, nums[7])
    x0 = nums[8]
    y0 = nums[9]
    cx = fw * 0.5 + x0
    cy = fh * 0.5 + y0
    if not (0 <= cx <= fw and 0 <= cy <= fh):
        cx, cy = float(x0), float(y0)
    left = cx - aw / 2.0
    top = cy - ah / 2.0
    sx = ref_w / float(fw)
    sy = ref_h / float(fh)
    return _clamp_ltwh(left * sx, top * sy, aw * sx, ah * sy, ref_w, ref_h)


def _rect_ref_subject_area_nums(
    nums: list[int], iw: int, ih: int
) -> tuple[int, int, int, int] | None:
    """SubjectArea in reference pixel space (CIPA EXIF)."""
    if len(nums) < 2:
        return None
    if len(nums) == 2:
        w, h = nums[0], nums[1]
        if w < 4 or h < 4:
            return None
        cx, cy = iw / 2.0, ih / 2.0
        return _clamp_ltwh(cx - w / 2.0, cy - h / 2.0, float(w), float(h), iw, ih)
    if len(nums) == 3:
        cx, cy, d = nums[0], nums[1], nums[2]
        if d < 4:
            return None
        return _clamp_ltwh(cx - d / 2.0, cy - d / 2.0, float(d), float(d), iw, ih)
    cx, cy, w, h = nums[0], nums[1], nums[2], nums[3]
    if w < 2 or h < 2:
        return None
    return _clamp_ltwh(cx - w / 2.0, cy - h / 2.0, float(w), float(h), iw, ih)


def _rect_ref_subject_location_nums(
    nums: list[int], iw: int, ih: int
) -> tuple[int, int, int, int] | None:
    if len(nums) < 2:
        return None
    cx, cy = nums[0], nums[1]
    side = max(24, min(iw, ih) // 14)
    return _clamp_ltwh(cx - side / 2.0, cy - side / 2.0, float(side), float(side), iw, ih)


def pixmap_ltwh_subject_cipa_from_exifread(
    path: str, pixmap_w: int, pixmap_h: int
) -> tuple[int, int, int, int] | None:
    """
    CIPA **SubjectArea** / **SubjectLocation** → (left, top, width, height) in pixmap pixels.

    Uses ``exifread.process_file(..., details=False)`` (no MakerNote decode). Fills the gap
    when pyexiv2 is missing or does not expose Photo tags the same way for JPEG.
    """
    if not path or pixmap_w < 4 or pixmap_h < 4:
        return None
    try:
        import exifread
    except ImportError:
        return None
    try:
        with open(path, "rb") as f:
            tags = exifread.process_file(
                f,
                details=False,
                extract_thumbnail=False,
            )
    except OSError:
        return None
    if not tags:
        return None
    ref_w, ref_h = _ref_dims(tags, pixmap_w, pixmap_h)
    if ref_w <= 0 or ref_h <= 0:
        ref_w, ref_h = pixmap_w, pixmap_h

    sa = tags.get("EXIF SubjectArea")
    if sa is not None:
        nums = _flatten_ints(sa)
        lt = _rect_ref_subject_area_nums(nums, ref_w, ref_h)
        if lt is not None:
            return _scale_to_pixmap(lt, ref_w, ref_h, pixmap_w, pixmap_h)
    sl = tags.get("EXIF SubjectLocation")
    if sl is not None:
        nums = _flatten_ints(sl)
        lt = _rect_ref_subject_location_nums(nums, ref_w, ref_h)
        if lt is not None:
            return _scale_to_pixmap(lt, ref_w, ref_h, pixmap_w, pixmap_h)
    return None


def pixmap_ltwh_af_from_exifread(
    path: str, pixmap_w: int, pixmap_h: int
) -> tuple[int, int, int, int] | None:
    """
    Return (left, top, width, height) in **pixmap** coordinates from MakerNote AF,
    or None if unavailable.
    """
    if not path or pixmap_w < 4 or pixmap_h < 4:
        return None
    try:
        import exifread
    except ImportError:
        return None
    try:
        with open(path, "rb") as f:
            tags = exifread.process_file(
                f,
                details=True,
                extract_thumbnail=False,
            )
    except OSError:
        return None
    if not tags:
        return None
    ref_w, ref_h = _ref_dims(tags, pixmap_w, pixmap_h)
    lt = _canon_af_ltwh_in_ref(tags, ref_w, ref_h)
    if lt is None:
        lt = _canon_tag_0x0012_fallback(tags, ref_w, ref_h)
    if lt is None:
        return None
    return _scale_to_pixmap(lt, ref_w, ref_h, pixmap_w, pixmap_h)
