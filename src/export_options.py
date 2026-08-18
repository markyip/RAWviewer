"""Export filename templates, signature+remark watermarks, and GPS stripping."""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Optional

DEFAULT_FILENAME_TEMPLATE = "{name}_edited.{ext}"
EXPORT_SETTINGS_PREFIX = "export/"

_TOKEN_RE = re.compile(r"\{([a-zA-Z_]+)\}")
_UNSAFE_CHARS = re.compile(r'[<>:"/\\\\|?*]')


def sanitize_filename_component(text: str) -> str:
    cleaned = _UNSAFE_CHARS.sub("_", str(text or "").strip())
    cleaned = cleaned.strip(" .")
    return cleaned or "untitled"


def _capture_datetime(context: dict[str, Any]) -> datetime:
    raw = context.get("capture_time") or context.get("datetime")
    if isinstance(raw, datetime):
        return raw
    text = str(raw or "").strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text[:19], fmt)
        except ValueError:
            continue
    mtime = context.get("mtime")
    if mtime:
        try:
            return datetime.fromtimestamp(float(mtime))
        except (OSError, ValueError, TypeError):
            pass
    return datetime.now()


def template_context_for_path(
    source_path: str,
    *,
    export_format: str,
    seq: int = 1,
    rating: int = 0,
    label: str = "",
    capture_time: str = "",
) -> dict[str, Any]:
    stem, _src_ext = os.path.splitext(os.path.basename(source_path or "image"))
    fmt = (export_format or "jpeg").strip().lower()
    ext = {"jpeg": "jpg", "jpg": "jpg", "webp": "webp", "tiff16": "tif", "tif": "tif", "tiff": "tif"}.get(
        fmt, "jpg"
    )
    try:
        mtime = os.path.getmtime(source_path) if source_path and os.path.isfile(source_path) else 0.0
    except OSError:
        mtime = 0.0
    return {
        "name": stem,
        "ext": ext,
        "seq": seq,
        "rating": int(rating or 0),
        "label": str(label or ""),
        "capture_time": capture_time,
        "mtime": mtime,
        "source_path": source_path,
    }


def format_export_filename(template: str, context: dict[str, Any]) -> str:
    """Replace ``{name} {ext} {seq} {yyyy} {mm} {dd} {hh} {min} {ss} {rating} {label}``."""
    tpl = (template or DEFAULT_FILENAME_TEMPLATE).strip() or DEFAULT_FILENAME_TEMPLATE
    dt = _capture_datetime(context)
    seq = int(context.get("seq") or 1)
    values = {
        "name": sanitize_filename_component(str(context.get("name") or "image")),
        "ext": sanitize_filename_component(str(context.get("ext") or "jpg")).lstrip("."),
        "seq": f"{seq:03d}",
        "yyyy": f"{dt.year:04d}",
        "yy": f"{dt.year % 100:02d}",
        "mm": f"{dt.month:02d}",
        "dd": f"{dt.day:02d}",
        "hh": f"{dt.hour:02d}",
        "min": f"{dt.minute:02d}",
        "ss": f"{dt.second:02d}",
        "rating": str(int(context.get("rating") or 0)),
        "label": sanitize_filename_component(str(context.get("label") or "")),
    }

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        return values.get(key, match.group(0))

    name = _TOKEN_RE.sub(_replace, tpl)
    name = name.replace("/", "_").replace("\\", "_")
    if not os.path.splitext(name)[1]:
        name = f"{name}.{values['ext']}"
    return name


MAX_SIGNATURE_NAME = 50
MAX_WATERMARK_REMARK = 120
DEFAULT_SIGNATURE_FONT = "snell"
DEFAULT_WATERMARK_CORNER = "br"
WATERMARK_CORNERS = ("tl", "tr", "bl", "br")
_WATERMARK_CORNER_LABELS = (
    ("tl", "Top left"),
    ("tr", "Top right"),
    ("bl", "Bottom left"),
    ("br", "Bottom right"),
)

# Built-in menu ids only — never store a user font path. macOS script faces
# first; DejaVu last so Linux CI / machines without Apple fonts still draw.
_SIGNATURE_FONTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "snell",
        "Snell Roundhand",
        (
            "/System/Library/Fonts/Supplemental/SnellRoundhand.ttc",
            "/System/Library/Fonts/Supplemental/Snell Roundhand.ttf",
        ),
    ),
    (
        "brush",
        "Brush Script",
        (
            "/System/Library/Fonts/Supplemental/Brush Script.ttf",
            "/Library/Fonts/Brush Script.ttf",
        ),
    ),
    (
        "chancery",
        "Apple Chancery",
        (
            "/System/Library/Fonts/Apple Chancery.ttf",
            "/System/Library/Fonts/Supplemental/Apple Chancery.ttf",
        ),
    ),
    (
        "zapfino",
        "Zapfino",
        (
            "/System/Library/Fonts/Supplemental/Zapfino.ttf",
            "/System/Library/Fonts/Zapfino.ttf",
            "/System/Library/Fonts/Supplemental/Zapfino.ttc",
        ),
    ),
    (
        "marker",
        "Marker Felt",
        (
            "/System/Library/Fonts/MarkerFelt.ttc",
            "/System/Library/Fonts/Supplemental/MarkerFelt.ttc",
        ),
    ),
    (
        "dejavu",
        "Plain",
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "DejaVuSans.ttf",
        ),
    ),
)


def list_signature_fonts() -> list[tuple[str, str]]:
    """``(font_id, display_name)`` pairs for the export dialog menu."""
    return [(fid, label) for fid, label, _paths in _SIGNATURE_FONTS]


def normalize_signature_font_id(font_id: str) -> str:
    ids = {fid for fid, _label, _paths in _SIGNATURE_FONTS}
    fid = str(font_id or "").strip().lower()
    return fid if fid in ids else DEFAULT_SIGNATURE_FONT


def _try_truetype(path: str, size: int):
    from PIL import ImageFont

    try:
        return ImageFont.truetype(path, size)
    except (OSError, ValueError):
        return None


def load_signature_font(font_id: str, size: int):
    """Load a script face by id; fall through to DejaVu / Pillow default."""
    from PIL import ImageFont

    by_id = {fid: paths for fid, _label, paths in _SIGNATURE_FONTS}
    preferred = normalize_signature_font_id(font_id)
    order = [preferred] + [fid for fid, _label, _paths in _SIGNATURE_FONTS if fid != preferred]
    for fid in order:
        for path in by_id[fid]:
            font = _try_truetype(path, size)
            if font is not None:
                return font
    return ImageFont.load_default()


def list_watermark_corners() -> list[tuple[str, str]]:
    return [(cid, label) for cid, label in _WATERMARK_CORNER_LABELS]


def normalize_watermark_corner(corner: str) -> str:
    cid = str(corner or "").strip().lower()
    return cid if cid in WATERMARK_CORNERS else DEFAULT_WATERMARK_CORNER


def _load_remark_font(size: int):
    from PIL import ImageFont

    for candidate in (
        "Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans.ttf",
    ):
        font = _try_truetype(candidate, size)
        if font is not None:
            return font
    return ImageFont.load_default()


def _text_metrics(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1], bbox[0], bbox[1]


def _draw_line(draw, text, font, x, y, left, top, fill, shadow_fill):
    ox, oy = x - left, y - top
    draw.text((ox + 1, oy + 1), text, font=font, fill=shadow_fill)
    draw.text((ox, oy), text, font=font, fill=fill)


def apply_export_watermark_rgb(
    rgb,
    *,
    remark: str = "",
    signature: str = "",
    font_id: str = "",
    corner: str = "",
    text: str = "",
    kind: str = "",
    remark_opacity: float = 0.7,
    signature_opacity: float = 0.82,
):
    """Draw signature and/or remark as one corner block. Returns a copy.

    Composites only a small corner patch — not two full-frame RGBA buffers.
    ``text`` / ``kind`` remain accepted for older callers.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    remark_s = str(remark or "").strip()[:MAX_WATERMARK_REMARK]
    signature_s = str(signature or "").strip()[:MAX_SIGNATURE_NAME]
    if not remark_s and not signature_s:
        leftover = str(text or "").strip()
        k = str(kind or "").strip().lower()
        if k == "signature":
            signature_s = leftover[:MAX_SIGNATURE_NAME]
        elif leftover:
            remark_s = leftover[:MAX_WATERMARK_REMARK]
    if rgb is None or (not remark_s and not signature_s):
        return rgb
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return rgb
    h, w = int(arr.shape[0]), int(arr.shape[1])
    corner_id = normalize_watermark_corner(corner)
    sig_size = max(28, min(110, h // 14))
    rem_size = max(16, min(48, h // 36))
    sig_font = load_signature_font(font_id, sig_size) if signature_s else None
    rem_font = _load_remark_font(rem_size) if remark_s else None
    scratch = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    measure = ImageDraw.Draw(scratch)
    sig_w = sig_h = sig_left = sig_top = 0
    rem_w = rem_h = rem_left = rem_top = 0
    if signature_s and sig_font is not None:
        sig_w, sig_h, sig_left, sig_top = _text_metrics(measure, signature_s, sig_font)
    if remark_s and rem_font is not None:
        rem_w, rem_h, rem_left, rem_top = _text_metrics(measure, remark_s, rem_font)
    gap = max(4, rem_size // 4) if (signature_s and remark_s) else 0
    shadow = 2
    margin = max(12, min(w, h) // 80)
    content_w = max(sig_w, rem_w)
    content_h = sig_h + gap + rem_h
    patch_w = max(1, min(w, content_w + margin + shadow))
    patch_h = max(1, min(h, content_h + margin + shadow))
    overlay = Image.new("RGBA", (patch_w, patch_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    right = corner_id in ("tr", "br")
    top = corner_id in ("tl", "tr")

    def _x_for(tw: int) -> int:
        if right:
            return max(0, patch_w - margin - shadow - tw)
        return margin

    sig_alpha = max(0, min(255, int(round(float(signature_opacity) * 255))))
    rem_alpha = max(0, min(255, int(round(float(remark_opacity) * 255))))
    if top:
        y = margin
        if signature_s and sig_font is not None:
            _draw_line(
                draw,
                signature_s,
                sig_font,
                _x_for(sig_w),
                y,
                sig_left,
                sig_top,
                (28, 22, 18, sig_alpha),
                (255, 255, 255, int(sig_alpha * 0.45)),
            )
            y += sig_h + gap
        if remark_s and rem_font is not None:
            _draw_line(
                draw,
                remark_s,
                rem_font,
                _x_for(rem_w),
                y,
                rem_left,
                rem_top,
                (255, 255, 255, rem_alpha),
                (0, 0, 0, int(rem_alpha * 0.55)),
            )
    else:
        y_rem = patch_h - margin - shadow - rem_h if remark_s else patch_h - margin
        if remark_s and rem_font is not None:
            _draw_line(
                draw,
                remark_s,
                rem_font,
                _x_for(rem_w),
                y_rem,
                rem_left,
                rem_top,
                (255, 255, 255, rem_alpha),
                (0, 0, 0, int(rem_alpha * 0.55)),
            )
        if signature_s and sig_font is not None:
            y_sig = (y_rem - gap - sig_h) if remark_s else (patch_h - margin - shadow - sig_h)
            _draw_line(
                draw,
                signature_s,
                sig_font,
                _x_for(sig_w),
                y_sig,
                sig_left,
                sig_top,
                (28, 22, 18, sig_alpha),
                (255, 255, 255, int(sig_alpha * 0.45)),
            )

    out = np.array(arr[..., :3], copy=True)
    ox = 0 if corner_id in ("tl", "bl") else w - patch_w
    oy = 0 if corner_id in ("tl", "tr") else h - patch_h
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(w, ox + patch_w), min(h, oy + patch_h)
    if x1 <= x0 or y1 <= y0:
        return out
    sx0, sy0 = x0 - ox, y0 - oy
    base = Image.fromarray(out[y0:y1, x0:x1], mode="RGB").convert("RGBA")
    ov = overlay.crop((sx0, sy0, sx0 + (x1 - x0), sy0 + (y1 - y0)))
    composed = Image.alpha_composite(base, ov).convert("RGB")
    out[y0:y1, x0:x1] = np.asarray(composed, dtype=out.dtype)
    return out


def apply_text_watermark_rgb(rgb, text: str, *, opacity: float = 0.7, corner: str = ""):
    """Draw a short remark caption. Returns a copy."""
    return apply_export_watermark_rgb(
        rgb, remark=text, corner=corner or DEFAULT_WATERMARK_CORNER, remark_opacity=opacity
    )


def apply_signature_watermark_rgb(
    rgb, name: str, *, font_id: str = "", opacity: float = 0.82, corner: str = ""
):
    """Draw a typed signature name. Returns a copy."""
    return apply_export_watermark_rgb(
        rgb,
        signature=name,
        font_id=font_id,
        corner=corner or DEFAULT_WATERMARK_CORNER,
        signature_opacity=opacity,
    )


def strip_gps_from_pil_exif(exif) -> None:
    """Remove GPS IFD pointer and GPS tags from a Pillow Exif object (in place)."""
    if exif is None:
        return
    gps_ifd = 0x8825
    try:
        exif.pop(gps_ifd, None)
    except Exception:
        pass
    try:
        if hasattr(exif, "get_ifd"):
            gps = exif.get_ifd(gps_ifd)
            if gps is not None:
                for key in list(gps.keys()):
                    gps.pop(key, None)
    except Exception:
        pass


def strip_gps_from_xmp_text(xml_text: str) -> str:
    """Drop GPS RDF descriptions / photoshop / exif GPS tags from sidecar XML."""
    if not xml_text or "GPS" not in xml_text and "gps" not in xml_text:
        return xml_text
    # Attribute form: exif:GPSLatitude="..."
    xml_text = re.sub(
        r'\s+(?:exif|tiff|crs|xmp):GPS[A-Za-z0-9]+="[^"]*"',
        "",
        xml_text,
        flags=re.IGNORECASE,
    )
    # Element form: <exif:GPSLatitude>...</exif:GPSLatitude>
    xml_text = re.sub(
        r"<[^>]*(?:exif:)?GPS[A-Za-z0-9]+[^>]*>.*?</[^>]*GPS[A-Za-z0-9]+>",
        "",
        xml_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return xml_text


def _read_source_exif(source_path: str):
    if not source_path or not os.path.isfile(source_path):
        return None
    from PIL import Image

    candidates = []
    stem, ext = os.path.splitext(source_path)
    if ext.lower() not in (".jpg", ".jpeg", ".tif", ".tiff", ".webp"):
        for jpeg_ext in (".JPG", ".jpg", ".JPEG", ".jpeg"):
            companion = stem + jpeg_ext
            if os.path.isfile(companion):
                candidates.append(companion)
                break
    candidates.append(source_path)
    for path in candidates:
        try:
            with Image.open(path) as im:
                exif = im.getexif()
                if exif is None:
                    continue
                if len(exif) == 0:
                    continue
                return exif
        except Exception:
            continue
    return None


def load_export_prefs(settings) -> dict[str, Any]:
    p = EXPORT_SETTINGS_PREFIX

    def _b(key: str, default: bool) -> bool:
        try:
            raw = settings.value(p + key, default)
            if isinstance(raw, str):
                return raw.strip().lower() not in ("0", "false", "no", "off")
            return bool(raw) if raw is not None else default
        except Exception:
            return default

    def _s(key: str, default: str) -> str:
        try:
            raw = settings.value(p + key, default)
            return str(raw) if raw is not None else default
        except Exception:
            return default

    signature = (_s("watermark_signature", "") or _s("watermark_name", "")).strip()[
        :MAX_SIGNATURE_NAME
    ]
    remark = (_s("watermark_remark", "") or _s("watermark_text", "")).strip()[:MAX_WATERMARK_REMARK]
    enabled = bool(signature or remark)
    return {
        "filename_template": _s("filename_template", DEFAULT_FILENAME_TEMPLATE),
        "strip_gps": _b("strip_gps", True),
        "copy_metadata": _b("copy_metadata", True),
        "watermark_kind": "off" if not enabled else "on",
        "watermark_enabled": enabled,
        "watermark_text": remark,
        "watermark_signature": signature,
        "watermark_name": signature,
        "watermark_font": normalize_signature_font_id(_s("watermark_font", DEFAULT_SIGNATURE_FONT)),
        "watermark_corner": normalize_watermark_corner(_s("watermark_corner", DEFAULT_WATERMARK_CORNER)),
        "last_directory": _s("last_directory", ""),
        "queue_format": _s("queue_format", "jpeg"),
    }


def watermark_kwargs_from_prefs(prefs: dict[str, Any]) -> dict[str, str]:
    """Map saved export prefs to pipeline watermark arguments."""
    signature = str(
        prefs.get("watermark_signature") or prefs.get("watermark_name") or ""
    ).strip()[:MAX_SIGNATURE_NAME]
    remark = str(
        prefs.get("watermark_remark") or prefs.get("watermark_text") or ""
    ).strip()[:MAX_WATERMARK_REMARK]
    return {
        "watermark_text": remark,
        "watermark_signature": signature,
        "watermark_font": normalize_signature_font_id(str(prefs.get("watermark_font") or ""))
        if signature
        else "",
        "watermark_corner": normalize_watermark_corner(str(prefs.get("watermark_corner") or "")),
    }


def save_export_prefs(settings, prefs: dict[str, Any]) -> None:
    p = EXPORT_SETTINGS_PREFIX
    for key, value in prefs.items():
        settings.setValue(p + key, value)
