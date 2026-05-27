"""
EXIF read path: prefer pyexiv2 (Exiv2), fall back to exifread.

Environment:
  RAWVIEWER_EXIF_BACKEND=auto|pyexiv2|exifread
    auto   — try pyexiv2 first when installed, else exifread (default)
    pyexiv2 — only pyexiv2 (raises / empty dict if unusable)
    exifread — only exifread

``details=True`` is not supported here (no MakerNote decode parity). Maker AF
overlays prefer :func:`read_exif_raw_string_dict` + ``exif_subject_area``; exifread
fallback is in ``exifread_af``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, BinaryIO, Mapping

# pyexiv2 will be imported lazily within functions to avoid startup overhead
_HAS_PYEXIV2 = None
_pyexiv2 = None  # type: ignore

def _ensure_pyexiv2():
    global _HAS_PYEXIV2, _pyexiv2
    if _HAS_PYEXIV2 is not None:
        return _HAS_PYEXIV2
    
    try:
        import pyexiv2 as _p
        _pyexiv2 = _p
        _HAS_PYEXIV2 = True
    except Exception:
        _pyexiv2 = None
        _HAS_PYEXIV2 = False
    return _HAS_PYEXIV2

_BACKEND = os.environ.get("RAWVIEWER_EXIF_BACKEND", "auto").strip().lower()


def exif_backend_mode() -> str:
    return _BACKEND


def has_pyexiv2() -> bool:
    return _ensure_pyexiv2()


def read_exif_raw_string_dict(path: str) -> dict[str, str] | None:
    """
    Full Exiv2 Exif key → string value (e.g. ``Exif.Photo.SubjectArea``,
    ``Exif.Canon.AFImageWidth``). No exifread-style renaming.

    Returns ``None`` when pyexiv2 is unavailable or the file cannot be read.
    """
    if not has_pyexiv2() or _pyexiv2 is None:
        return None
    try:
        with _pyexiv2.Image(path) as img:
            raw = img.read_exif()
    except Exception:
        return None
    if not raw:
        return None
    out: dict[str, str] = {}
    for k, v in raw.items():
        out[str(k)] = "" if v is None else str(v)
    return out


# EXIF Orientation after a further 90° clockwise rotation (metadata-only; pixels unchanged).
_EXIF_ORIENTATION_AFTER_CW90: dict[int, int] = {
    1: 6,
    2: 5,
    3: 8,
    4: 7,
    5: 2,
    6: 3,
    7: 4,
    8: 1,
}


def exif_orientation_after_cw90(current: int) -> int:
    o = int(current) if current else 1
    if o < 1 or o > 8:
        o = 1
    return _EXIF_ORIENTATION_AFTER_CW90[o]


def _parse_exif_orientation_string(val: str | None) -> int | None:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    first = s.split()[0]
    try:
        o = int(first)
    except ValueError:
        return None
    return o if 1 <= o <= 8 else None


def read_exif_orientation(path: str) -> int:
    """Return EXIF Orientation 1–8 from IFD0 (``Exif.Image.Orientation``), default 1."""
    raw = read_exif_raw_string_dict(path)
    if not raw:
        return 1
    for key in ("Exif.Image.Orientation", "Exif.Photo.Orientation"):
        o = _parse_exif_orientation_string(raw.get(key))
        if o is not None:
            return o
    for k, v in raw.items():
        if not k.endswith(".Orientation") or "Thumbnail" in k:
            continue
        o = _parse_exif_orientation_string(v)
        if o is not None:
            return o
    return 1


def rotate_exif_orientation_meta_cw90(path: str) -> None:
    """
    Advance EXIF Orientation by 90° clockwise (metadata only; RAW sensor data unchanged).

    Uses pyexiv2 / Exiv2. Raises ``RuntimeError`` if pyexiv2 is unavailable or the write fails.
    """
    if not has_pyexiv2() or _pyexiv2 is None:
        raise RuntimeError(
            "Rotating RAW files updates metadata only (sensor pixels are unchanged) and "
            "requires pyexiv2 (Exiv2). Install pyexiv2 for your platform (see RAWviewer "
            "requirements / build scripts)."
        )
    o = read_exif_orientation(path)
    new_o = exif_orientation_after_cw90(o)
    try:
        with _pyexiv2.Image(path) as img:
            img.modify_exif({"Exif.Image.Orientation": str(new_o)})
    except Exception as e:
        raise RuntimeError(
            f"Could not write EXIF Orientation to {path!r}: {e}"
        ) from e


@dataclass
class IfdTagLite:
    """Minimal exifread.IfdTag compatibility: .printable, .values, str()."""

    printable: str
    values: list[Any] | None = None

    def __post_init__(self) -> None:
        if self.values is None:
            self.values = _infer_values_from_printable(self.printable)

    def __str__(self) -> str:
        return self.printable


def _infer_values_from_printable(printable: str) -> list[Any]:
    s = (printable or "").strip()
    if not s:
        return []
    try:
        from exifread.utils import Ratio

        if re.fullmatch(r"-?\d+", s):
            return [int(s)]
        if re.fullmatch(r"-?\d+/\d+", s):
            a, b = s.split("/", 1)
            return [Ratio(int(a), int(b))]
    except Exception:
        pass
    return [s]


def _gps_rational_values(s: str) -> list[Any] | None:
    """Build [deg, min, sec] as Ratio list for semantic_search._gps_to_decimal."""
    try:
        from exifread.utils import Ratio
    except Exception:
        return None
    parts = str(s).split()
    out: list[Any] = []
    for tok in parts:
        if "/" in tok:
            a, b = tok.split("/", 1)
            try:
                out.append(Ratio(int(a), int(b)))
            except (ValueError, ZeroDivisionError):
                return None
        else:
            try:
                out.append(Ratio(int(tok), 1))
            except ValueError:
                return None
    if len(out) >= 3:
        return out[:3]
    return None


def _map_exiv2_key_to_exifread(key: str) -> str | None:
    if not key.startswith("Exif."):
        return None
    parts = key.split(".")
    if len(parts) < 3:
        return None
    group, rest = parts[1], parts[2:]
    name = ".".join(rest)
    if group == "Image":
        return f"Image {name}"
    if group == "Photo":
        return f"EXIF {name}"
    if group == "GPSInfo":
        return f"GPS {name}"
    if group == "Thumbnail":
        return f"Thumbnail {name}"
    if group == "SubIFD":
        return f"EXIF {name}"
    # Maker / vendor namespaces (Canon, Nikon, Olympus, Sony, Fujifilm, …)
    if group not in ("Image", "Photo", "GPSInfo", "Thumbnail", "SubIFD"):
        return f"MakerNote {name}"
    return None


def _maybe_gps_tag_lite(exiv_key: str, val: str) -> IfdTagLite:
    s = str(val).strip()
    if "GPSInfo.GPSLatitude" in exiv_key or "GPSInfo.GPSLongitude" in exiv_key:
        rats = _gps_rational_values(s)
        if rats is not None:
            return IfdTagLite(printable=s, values=rats)
    if "GPSInfo.GPSAltitude" in exiv_key:
        rats = _gps_rational_values(s)
        if rats is not None and rats:
            return IfdTagLite(printable=s, values=[rats[0]])
    return IfdTagLite(printable=s)


def _pyexiv2_tags(path: str) -> dict[str, Any] | None:
    if not has_pyexiv2() or _pyexiv2 is None:
        return None
    try:
        with _pyexiv2.Image(path) as img:
            raw = img.read_exif()
    except Exception:
        return None
    if not raw:
        return None
    out: dict[str, Any] = {}
    for exiv_k, exiv_v in raw.items():
        if not isinstance(exiv_v, str):
            exiv_v = str(exiv_v)
        ek = _map_exiv2_key_to_exifread(exiv_k)
        if not ek:
            continue
        tag = _maybe_gps_tag_lite(exiv_k, exiv_v)
        out[ek] = tag

    for ek, tag in list(out.items()):
        if not ek.startswith("MakerNote "):
            continue
        tail = ek[len("MakerNote ") :]
        leaf = tail.split(".")[-1]
        if leaf:
            alias = f"MakerNote {leaf}"
            if alias not in out:
                out[alias] = tag
    return out if out else None


def _exifread_process_path(
    path: str,
    *,
    details: bool,
    stop_tag: str | None,
) -> dict[str, Any]:
    import warnings

    import exifread

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with open(path, "rb") as f:
            if stop_tag is not None:
                return exifread.process_file(f, details=details, stop_tag=stop_tag)
            return exifread.process_file(f, details=details)


def process_file_from_path(
    path: str,
    *,
    details: bool = False,
    stop_tag: str | None = None,
) -> dict[str, Any]:
    """
    Return an exifread-like tag dict (IfdTagLite or native IfdTag values).

    When ``details=True``, always uses exifread (pyexiv2 path skips MakerNote parity).
    """
    if details:
        return _exifread_process_path(path, details=True, stop_tag=stop_tag)

    mode = _BACKEND if _BACKEND in ("auto", "pyexiv2", "exifread") else "auto"

    if mode == "exifread":
        return _exifread_process_path(path, details=False, stop_tag=stop_tag)

    if mode == "pyexiv2":
        tags = _pyexiv2_tags(path)
        if tags:
            return tags
        return _exifread_process_path(path, details=False, stop_tag=stop_tag)

    # auto
    if mode == "auto":
        # Check if the file is a RAW/DNG file. If so, prefer the high-performance exifread path,
        # which reads only headers and avoids pyexiv2/Exiv2 loading the huge raw sensor data (700ms -> 1ms).
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        from raw_file_extensions import RAW_FILE_EXTENSIONS
        if ext in RAW_FILE_EXTENSIONS:
            return _exifread_process_path(path, details=False, stop_tag=stop_tag)

        if has_pyexiv2():
            tags = _pyexiv2_tags(path)
            if tags and len(tags) >= 2:
                return tags
    return _exifread_process_path(path, details=False, stop_tag=stop_tag)


def process_file(
    fh: BinaryIO,
    *,
    details: bool = False,
    stop_tag: str | None = None,
    strict: bool = False,
    debug: bool = False,
    truncate_tags: bool = True,
    auto_seek: bool = True,
    extract_thumbnail: bool = True,
    builtin_types: bool = False,
) -> dict[str, Any]:
    """
    Drop-in-ish replacement for ``exifread.process_file`` when the caller has a
    file handle: reads current path from ``fh.name`` when available, else falls
    back to exifread on the stream.
    """
    name = getattr(fh, "name", None)
    if isinstance(name, str) and name and os.path.isfile(name):
        return process_file_from_path(
            name, details=details, stop_tag=stop_tag
        )
    import warnings

    import exifread

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if auto_seek:
            try:
                fh.seek(0)
            except Exception:
                pass
        if stop_tag is not None:
            return exifread.process_file(
                fh,
                details=details,
                stop_tag=stop_tag,
                strict=strict,
                debug=debug,
                truncate_tags=truncate_tags,
                auto_seek=False,
                extract_thumbnail=extract_thumbnail,
                builtin_types=builtin_types,
            )
        return exifread.process_file(
            fh,
            details=details,
            strict=strict,
            debug=debug,
            truncate_tags=truncate_tags,
            auto_seek=False,
            extract_thumbnail=extract_thumbnail,
            builtin_types=builtin_types,
        )


def tag_dict_as_str_dict(tags: Mapping[str, Any]) -> dict[str, str]:
    """Serialize tag dict to plain strings (e.g. EXIF cache)."""
    out: dict[str, str] = {}
    for k, v in tags.items():
        try:
            out[str(k)] = str(v)
        except Exception:
            continue
    return out
