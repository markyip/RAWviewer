"""
RAW adjustments parser and NumPy math helper.
Applies Exposure, Contrast, Highlights, Shadows, Whites, Blacks, Temp, Tint, Saturation, and Vibrance.
"""

from __future__ import annotations

import os
import tempfile
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

# Per-sidecar-path locks serializing write_xmp_rating/write_xmp_adjustments.
# Both read the existing file, compute a full replacement in memory, then
# os.replace() it -- os.replace() is atomic against corruption but not
# against a lost update, so two writers to the SAME sidecar racing (e.g. a
# background export-time adjustments save and a same-moment rating click)
# could each read-before-the-other's-write and one would silently clobber
# the other's change. Not capped/evicted like other per-path caches in this
# codebase: a Lock is a few hundred bytes (thousands of files in a session
# is still negligible), and evicting one while another thread might still
# hold a reference to it would defeat the whole point -- a second thread
# creating a fresh Lock for the same path after eviction would no longer be
# synchronized with the first.
_xmp_write_locks: Dict[str, threading.Lock] = {}
_xmp_write_locks_guard = threading.Lock()


def _xmp_write_lock(xmp_path: str) -> threading.Lock:
    key = os.path.normcase(os.path.abspath(xmp_path)) if xmp_path else ""
    with _xmp_write_locks_guard:
        lock = _xmp_write_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _xmp_write_locks[key] = lock
        return lock


DEFAULT_ADJUSTMENTS: Dict[str, float] = {
    "Exposure2012": 0.0,
    "Contrast2012": 0.0,
    "Highlights2012": 0.0,
    "Shadows2012": 0.0,
    "Whites2012": 0.0,
    "Blacks2012": 0.0,
    "Temperature": 5500.0,
    "Tint": 0.0,
    "Saturation": 0.0,
    "Vibrance": 0.0,
    "Sharpness": 0.0,
    "Clarity2012": 0.0,
    "Defringe": 0.0,
    "ColorNoiseReduction": 0.0,
    "LuminanceNoiseReduction": 0.0,
    "ParametricShadows": 0.0,
    "ParametricDarks": 0.0,
    "ParametricLights": 0.0,
    "ParametricHighlights": 0.0,
    # Off by default -- an automatic geometry change should never apply
    # silently. Baked into the decoded edit base at decode time (see
    # unified_image_processor.decode_raw_edit_base), not a per-tick pipeline
    # adjustment -- see raw_lens_correction.py.
    "LensCorrectionEnabled": 0.0,
    "DenoiseMethod": 0.0,
    # Geometry (see raw_transform.py): straighten + keystone perspective +
    # per-edge crop insets. Applied once at the head of both pipelines;
    # always stays within the original pixel frame (auto inscribed-rect crop).
    "CropAngle": 0.0,
    "PerspectiveVertical": 0.0,
    "PerspectiveHorizontal": 0.0,
    # Dodge & burn stops-per-mask-unit (see raw_dodge_burn.py). The mask
    # itself (a base64 PNG blob, potentially large) is NOT a plain numeric
    # attribute -- it's stored as its own XMP child element, mirroring
    # ToneCurvePV2012 (see write_xmp_adjustments / parse_dodge_burn_mask_from_xmp).
    "DodgeBurnStrength": 1.75,
}

from raw_hsl import HSL_COLOR_NAMES  # noqa: E402

for _hsl_color in HSL_COLOR_NAMES:
    DEFAULT_ADJUSTMENTS[f"HueAdjustment{_hsl_color}"] = 0.0
    DEFAULT_ADJUSTMENTS[f"SaturationAdjustment{_hsl_color}"] = 0.0
    DEFAULT_ADJUSTMENTS[f"LuminanceAdjustment{_hsl_color}"] = 0.0

RELEVANT_ADJUSTMENT_KEYS = frozenset(DEFAULT_ADJUSTMENTS.keys())
AS_SHOT_TEMP_KEY = "AsShotTemperature"
CHROMA_NR_ON_VALUE = 50.0
RECOVERY_BASELINE_KEY = "_recovery_baseline"
# UI hints when recovery look is on (local S/H recovery ≈ these PV2012 readouts).
RECOVERY_BASELINE_SHADOWS2012 = 40.0
RECOVERY_BASELINE_HIGHLIGHTS2012 = -35.0

_PV2012_TONE_SLIDER_KEYS = frozenset(
    {
        "Contrast2012",
        "Highlights2012",
        "Shadows2012",
        "Whites2012",
        "Blacks2012",
        "ParametricShadows",
        "ParametricDarks",
        "ParametricLights",
        "ParametricHighlights",
    }
)

# Sliders that disable recovery tone when moved (Highlights/Shadows are hint-only while recovery on).
_PV2012_RECOVERY_EXCLUSIVE_KEYS = _PV2012_TONE_SLIDER_KEYS - {
    "Highlights2012",
    "Shadows2012",
}


def uses_recovery_tone_map(adj: dict[str, float] | None) -> bool:
    """True when adjust should use P-key recovery tone instead of Reinhard + PV2012."""
    if not adj:
        return False
    if float(adj.get(RECOVERY_BASELINE_KEY, 0.0)) <= 0.5:
        return False
    from raw_tone_curve import CHANNEL_CURVE_KEYS, TONE_CURVE_SERIAL_KEY

    if str(adj.get(TONE_CURVE_SERIAL_KEY, "") or "").strip():
        return False
    for key in CHANNEL_CURVE_KEYS:
        if str(adj.get(key, "") or "").strip():
            return False
    for key in _PV2012_RECOVERY_EXCLUSIVE_KEYS:
        default = float(DEFAULT_ADJUSTMENTS.get(key, 0.0))
        if abs(float(adj.get(key, default)) - default) > 1e-4:
            return False
    return True


def recovery_baseline_slider_hints() -> dict[str, float]:
    """PV2012 Highlights/Shadows readouts shown when recovery look is enabled."""
    return {
        "Shadows2012": RECOVERY_BASELINE_SHADOWS2012,
        "Highlights2012": RECOVERY_BASELINE_HIGHLIGHTS2012,
    }


def is_pv2012_tone_slider(key: str) -> bool:
    return key in _PV2012_TONE_SLIDER_KEYS

CRS_NS = "http://adobe.com/camera-raw-settings/1.0/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
X_NS = "adobe:ns:meta/"
XMP_NS = "http://ns.adobe.com/xap/1.0/"


@dataclass(frozen=True)
class SliderSpec:
    key: str
    label: str
    minimum: int
    maximum: int
    default_value: float
    single_step: int
    slider_to_value: Callable[[int], float]
    value_to_slider: Callable[[float], int]
    format_value: Callable[[float], str]


def _slider_linear(
    key: str,
    label: str,
    minimum: int,
    maximum: int,
    default: float,
    *,
    scale: float = 1.0,
    fmt: Callable[[float], str] | None = None,
) -> SliderSpec:
    def slider_to_value(v: int) -> float:
        return float(v) * scale

    def value_to_slider(v: float) -> int:
        return int(round(float(v) / scale))

    return SliderSpec(
        key=key,
        label=label,
        minimum=minimum,
        maximum=maximum,
        default_value=default,
        single_step=max(1, int(round(scale * 10)) if scale < 1 else 1),
        slider_to_value=slider_to_value,
        value_to_slider=value_to_slider,
        format_value=fmt or (lambda x: f"{x:+.0f}"),
    )


SLIDER_SPECS: tuple[SliderSpec, ...] = (
    _slider_linear("Exposure2012", "Exposure", -500, 500, 0.0, scale=0.01, fmt=lambda x: f"{x:+.2f}"),
    _slider_linear("Contrast2012", "Contrast", -100, 100, 0.0),
    _slider_linear("Highlights2012", "Highlights", -100, 100, 0.0),
    _slider_linear("Shadows2012", "Shadows", -100, 100, 0.0),
    _slider_linear("Whites2012", "Whites", -100, 100, 0.0),
    _slider_linear("Blacks2012", "Blacks", -100, 100, 0.0),
    _slider_linear("ParametricShadows", "PV Shad", -100, 100, 0.0),
    _slider_linear("ParametricDarks", "PV Dark", -100, 100, 0.0),
    _slider_linear("ParametricLights", "PV Light", -100, 100, 0.0),
    _slider_linear("ParametricHighlights", "PV High", -100, 100, 0.0),
    SliderSpec(
        key="Temperature",
        label="Temp",
        minimum=2000,
        maximum=12000,
        default_value=5500.0,
        single_step=50,
        slider_to_value=lambda v: float(v),
        value_to_slider=lambda v: int(round(float(v))),
        format_value=lambda x: f"{int(round(x))}K",
    ),
    _slider_linear("Tint", "Tint", -150, 150, 0.0),
    _slider_linear("Saturation", "Saturation", -100, 100, 0.0),
    _slider_linear("Vibrance", "Vibrance", -100, 100, 0.0),
    _slider_linear("Sharpness", "Sharpness", 0, 150, 0.0, fmt=lambda x: f"{x:.0f}"),
    _slider_linear("Clarity2012", "Clarity", -100, 100, 0.0),
    _slider_linear("Defringe", "Defringe", 0, 100, 0.0),
    _slider_linear("LuminanceNoiseReduction", "Luma NR", 0, 100, 0.0, fmt=lambda x: f"{x:.0f}"),
    # Transform (raw_transform.py). Straighten in 0.1-degree steps; keystone
    # sliders mirror Lightroom's slider-based Transform panel; crop insets as
    # per-edge percentages (stored as fractions).
    _slider_linear("CropAngle", "Straighten", -450, 450, 0.0, scale=0.1, fmt=lambda x: f"{x:+.1f}°"),
    _slider_linear("PerspectiveVertical", "Vertical", -100, 100, 0.0),
    _slider_linear("PerspectiveHorizontal", "Horizontal", -100, 100, 0.0),
    # Per-edge crop-inset sliders were removed by request: cropping stays out
    # of the UI until a proper interactive overlay (visible crop rectangle
    # with drag handles) exists. raw_transform.apply_geometry still honors
    # the keys, so the future overlay only needs to write them.
)


def resolve_xmp_path(image_path: str) -> str:
    """Sidecar path next to the image (Lightroom-style basename.xmp)."""
    if not image_path:
        return ""
    companion = image_path + ".xmp"
    if os.path.isfile(companion):
        return companion
    return os.path.splitext(image_path)[0] + ".xmp"


def editing_features_enabled() -> bool:
    """Whether the Adjust panel, XMP writes, and edit export are available.

    On by default on the development branch
    (``RAWVIEWER_ENABLE_EDITING=0`` to disable). Rating read/write and plain
    browse/export-without-adjustments stay available either way.
    """
    return os.environ.get("RAWVIEWER_ENABLE_EDITING", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def sidecar_adjustments_enabled() -> bool:
    """Whether browse/full-res display applies saved XMP edit sliders to pixels.

    On by default when editing is enabled (RAWVIEWER_SIDECAR_ADJUST=0 to
    disable). It shipped default-off while edits were invisible everywhere
    outside the Adjust panel; now that gallery tiles and the fit preview
    render saved edits too (edited_previews_enabled), leaving the full-res
    tier unedited would make the edits visibly VANISH on zoom -- consistency
    across tiers matters more than the apply cost, which only edited files
    ever pay. Requires `editing_features_enabled()` -- browse-only builds
    never pay the apply cost even if the env var is set. Explicit
    `apply_sidecar_adjustments=True` callers are unaffected.
    """
    if not editing_features_enabled():
        return False
    return os.environ.get("RAWVIEWER_SIDECAR_ADJUST", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def edited_previews_enabled() -> bool:
    """Whether gallery tiles and the fit preview render saved XMP edits.

    On by default when editing is enabled (RAWVIEWER_EDITED_PREVIEWS=0 to
    disable). Applied at display/delivery time only -- adjusted pixels are
    never written back to any pixel cache, so there is no stale-thumbnail
    invalidation problem: re-editing simply changes what the next delivery
    applies. Cost (measured): ~5ms per 320px gallery tile, ~35ms per 720px
    tile, ~215ms for a 2304px fit preview -- worker-thread, edited files only.
    """
    if not editing_features_enabled():
        return False
    return os.environ.get("RAWVIEWER_EDITED_PREVIEWS", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


_SRGB_DECODE_LUT = None


def _decode_srgb_uint8_to_linear(arr: np.ndarray) -> np.ndarray:
    """Standard sRGB EOTF, uint8 [0,255] -> linear [0,1] float32.

    Gallery/thumbnail buffers for RAW files are the camera's embedded JPEG
    (extract_embedded_jpeg_by_scan / rawpy.extract_thumb) or, absent that,
    LibRaw's own raw.postprocess() output -- both approximately sRGB gamma
    (~2.2 with a linear toe), NOT the app's own dcraw-style BT.709
    linear16->uint8 encode from fast_raw_decode._gamma_lut8. This function
    used to invert THAT LUT (see git history), which assumes an input this
    code path never actually receives: the flatter BT.709 shadow curve,
    inverted, overestimates linear values versus the steeper true sRGB toe,
    so the reconstructed "linear" buffer was too bright even before edits --
    then process_linear_edit_buffer's exposure/tone stage (built for a
    genuinely linear-RAW input, e.g. decode_raw_edit_base) amplified that
    overestimate further, making edited gallery thumbnails visibly brighter
    than the same edit applied in single view (which decodes true
    scene-linear RAW, never through an 8-bit round trip at all). Report:
    "thumbnail preview brightness significantly higher than edited image."
    """
    global _SRGB_DECODE_LUT
    if _SRGB_DECODE_LUT is None:
        c = np.arange(256, dtype=np.float64) / 255.0
        lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        _SRGB_DECODE_LUT = lin.astype(np.float32)
    return _SRGB_DECODE_LUT[arr]


def apply_saved_edits_for_display(file_path: str, arr):
    """Apply saved XMP edits to a display-bound uint8 RGB ndarray.

    Returns the input unchanged when previews-with-edits is disabled, the
    file is not RAW, the file has no (non-default) saved adjustments, or the
    buffer is not an ndarray (QImage branches skip). Never raises. Callers
    must NOT persist the result into pixel caches -- caches hold the
    unadjusted base.

    RAW-only by contract: sidecars are resolved by BASENAME (Lightroom
    convention), so a JPEG sharing its basename with an edited RAW resolves
    to the RAW's sidecar -- without this gate the JPEG's gallery tile showed
    the RAW's edits applied to it ("also flagged as edited" report). Edits
    can only ever be created for RAW files (the Adjust panel requires a
    demosaiced RAW base), so non-RAW is always pass-through.

    Renders through the REAL scene-linear pipeline (sRGB decode ->
    process_linear_edit_buffer -> linear_to_display_uint8), not the legacy
    gamma-space approximation: the gamma path's tone math diverges visibly
    from the Adjust panel's render (reported as the gallery/fit preview
    being "way off" from the edited image). Working from an already
    tone-curved camera JPEG is still an approximation of the true RAW edit
    base, but the tone/color processing applied on top is now the same code
    the editor runs. The uint8->linear decode uses the standard sRGB EOTF,
    matching the actual gamma of the embedded-JPEG/LibRaw-postprocess
    thumbnail buffers this function receives (see
    _decode_srgb_uint8_to_linear's docstring for why the BT.709 LUT this
    used previously was the wrong inverse and made edited thumbnails read
    brighter than the true edited image).
    """
    try:
        if arr is None or not hasattr(arr, "shape") or not edited_previews_enabled():
            return arr
        from common_image_loader import is_raw_file

        if not is_raw_file(file_path):
            return arr
        # Sidecar existence FIRST (two isfile checks, ~0.05ms). Going straight
        # to load_adjustments_for_file ran read_as_shot_temperature -- a full
        # EXIF parse of the RAW file -- for EVERY delivered gallery tile,
        # edited or not, before the is_default early-out could save it. On a
        # cold session that extra per-tile file-open competed with thumbnail
        # extraction on the same disk: "gallery scrolling slower than before".
        xmp_path = resolve_xmp_path(file_path)
        if not (xmp_path and os.path.isfile(xmp_path)):
            return arr
        adj = load_adjustments_for_file(file_path)
        if is_default_adjustments(adj):
            return arr
        from raw_edit_pipeline import linear_to_display_uint8, process_linear_edit_buffer

        lin = _decode_srgb_uint8_to_linear(np.ascontiguousarray(arr))
        out = process_linear_edit_buffer(lin, adj, preview=True)
        return linear_to_display_uint8(out, adj)
    except Exception:
        return arr



def _parse_rating_value(raw: object) -> int:
    if raw is None:
        return 0
    try:
        text = str(raw).strip()
        if not text:
            return 0
        return max(0, min(5, int(float(text.split()[0]))))
    except (TypeError, ValueError):
        return 0


def parse_xmp_rating(xmp_path: str) -> int:
    """Parse xmp:Rating from a sidecar without loading adjustment sliders."""
    if not xmp_path or not os.path.isfile(xmp_path):
        return 0
    try:
        tree = ET.parse(xmp_path)
        root = tree.getroot()
        ns = {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "xmp": "http://ns.adobe.com/xap/1.0/",
        }
        for desc in root.findall(".//rdf:Description", ns):
            for key, val in desc.attrib.items():
                local_key = key.split("}", 1)[-1] if "}" in key else key.split(":")[-1]
                if local_key == "Rating":
                    rating = _parse_rating_value(val)
                    if rating:
                        return rating
            rating_el = desc.find("{http://ns.adobe.com/xap/1.0/}Rating")
            if rating_el is not None and rating_el.text:
                rating = _parse_rating_value(rating_el.text)
                if rating:
                    return rating
    except Exception:
        pass
    return 0


def load_rating_for_file(image_path: str) -> int:
    """Lightweight star rating from the image's XMP sidecar (no edit sliders)."""
    return parse_xmp_rating(resolve_xmp_path(image_path))


def write_xmp_rating(xmp_path: str, rating: int) -> None:
    """Write (or clear) xmp:Rating in a sidecar, preserving everything else.

    Independent of editing_features_enabled() -- ratings must persist even
    in browse-only builds that never touch the crs adjustment sliders.
    Edits the existing tree in place rather than rebuilding it (unlike
    write_xmp_adjustments) so a rating set here survives untouched even if
    the sidecar also holds crs sliders / tone curve / dodge-burn data from
    a build with editing enabled.

    Holds this sidecar's write lock for the whole read-modify-write so a
    concurrent write_xmp_adjustments() call (runs on the export worker
    thread) can't interleave with this one and lose either change -- see
    the _xmp_write_lock module comment.
    """
    if not xmp_path:
        raise ValueError("Invalid xmp path")
    with _xmp_write_lock(xmp_path):
        _write_xmp_rating_locked(xmp_path, rating)


def _write_xmp_rating_locked(xmp_path: str, rating: int) -> None:
    rating = max(0, min(5, int(rating)))

    ET.register_namespace("x", X_NS)
    ET.register_namespace("rdf", RDF_NS)
    ET.register_namespace("crs", CRS_NS)
    ET.register_namespace("xmp", XMP_NS)

    root = None
    if os.path.isfile(xmp_path):
        try:
            root = ET.parse(xmp_path).getroot()
        except Exception:
            root = None

    if root is None:
        if rating == 0:
            return
        root = ET.Element(f"{{{X_NS}}}xmpmeta")
        rdf = ET.SubElement(root, f"{{{RDF_NS}}}RDF")
        desc = ET.SubElement(rdf, f"{{{RDF_NS}}}Description")
        desc.set(f"{{{RDF_NS}}}about", "")
    else:
        rdf = root.find(f"{{{RDF_NS}}}RDF")
        if rdf is None:
            rdf = ET.SubElement(root, f"{{{RDF_NS}}}RDF")
        desc = rdf.find(f"{{{RDF_NS}}}Description")
        if desc is None:
            desc = ET.SubElement(rdf, f"{{{RDF_NS}}}Description")
            desc.set(f"{{{RDF_NS}}}about", "")

    rating_attr = f"{{{XMP_NS}}}Rating"
    if rating > 0:
        desc.set(rating_attr, str(rating))
    else:
        desc.attrib.pop(rating_attr, None)
        for child in list(desc):
            tag = child.tag
            local = tag.split("}", 1)[-1] if "}" in tag else tag
            if local == "Rating":
                desc.remove(child)

    # Nothing left worth keeping (no rating, no crs sliders, no child
    # elements like tone curve / dodge-burn mask) -- remove the sidecar
    # instead of littering the folder with an empty .xmp per photo.
    has_rating = rating_attr in desc.attrib
    has_crs = any(
        key.startswith(f"{{{CRS_NS}}}") for key in desc.attrib
    )
    has_children = len(list(desc)) > 0
    if not has_rating and not has_crs and not has_children:
        if os.path.isfile(xmp_path):
            os.remove(xmp_path)
        return

    tree = ET.ElementTree(root)
    parent = os.path.dirname(os.path.abspath(xmp_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    # Atomic write -- see write_xmp_adjustments for why (crash/race safety).
    fd, tmp_path = tempfile.mkstemp(
        dir=parent or None, prefix=os.path.basename(xmp_path) + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as f:
            tree.write(f, encoding="utf-8", xml_declaration=True)
        os.replace(tmp_path, xmp_path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def write_rating_for_file(image_path: str, rating: int) -> None:
    """Persist a star rating (0-5) to the image's XMP sidecar."""
    xmp_path = resolve_xmp_path(image_path)
    if not xmp_path:
        raise ValueError("Invalid image path")
    write_xmp_rating(xmp_path, rating)


# Per-path memo for read_as_shot_temperature: the value is a per-file constant
# (as-shot metadata), but computing it may need a rawpy.imread -- which ran on
# the GUI thread on EVERY panel sync (a 100-900ms stall per file open), and
# ran WITHOUT _rawpy_global_lock (racing worker decodes; every other imread in
# the app serializes on that lock). Session-scoped, tiny (one float per path).
_AS_SHOT_TEMP_CACHE: dict[str, float] = {}


def read_as_shot_temperature(image_path: str) -> float:
    """Best-effort as-shot CCT from EXIF or RAW metadata (not a UI preset)."""
    if not image_path:
        return DEFAULT_ADJUSTMENTS["Temperature"]
    cache_key = os.path.normcase(os.path.abspath(image_path))
    cached = _AS_SHOT_TEMP_CACHE.get(cache_key)
    if cached is not None:
        return cached
    result = DEFAULT_ADJUSTMENTS["Temperature"]
    resolved = False
    try:
        import metadata_backend

        tags = metadata_backend.process_file_from_path(image_path, details=False)
        for tag in (
            "EXIF ColorTemperature",
            "MakerNote ColorTemperature",
            "EXIF WB_RGGBLevelsAsShot",
        ):
            raw = tags.get(tag)
            if raw is None:
                continue
            if tag.endswith("ColorTemperature"):
                val = int(float(str(raw).split()[0]))
                if 2000 <= val <= 12000:
                    result = float(val)
                    resolved = True
                    break
    except Exception:
        pass
    if not resolved:
        try:
            import rawpy

            from enhanced_raw_processor import _rawpy_global_lock

            with _rawpy_global_lock:
                raw_ctx = rawpy.imread(image_path)
            with raw_ctx as raw:
                cam = np.array(raw.camera_whitebalance[:3], dtype=np.float64)
                day = np.array(raw.daylight_whitebalance[:3], dtype=np.float64)
                if np.all(cam > 0) and np.all(day > 0):
                    rb_cam = cam[0] / cam[2]
                    rb_day = day[0] / day[2]
                    ratio = rb_cam / max(rb_day, 1e-6)
                    est = 5500.0 * (ratio ** -0.35)
                    result = float(np.clip(est, 2000.0, 12000.0))
        except Exception:
            pass
    _AS_SHOT_TEMP_CACHE[cache_key] = result
    return result


def _parse_point_curve_from_xmp(xmp_path: str, tag_name: str) -> str:
    """Read a crs:<tag_name> point list → 'x,y;x,y' serialized string.

    Shared by the main luminance curve (ToneCurvePV2012) and the three
    Standard-mode channel curves (ToneCurvePV2012Red/Green/Blue) -- same
    XMP shape (an rdf:Seq of "x, y" li entries), different tag name.
    """
    if not xmp_path or not os.path.isfile(xmp_path):
        return ""
    from raw_tone_curve import serialize_tone_curve_points

    points: list[tuple[float, float]] = []
    try:
        tree = ET.parse(xmp_path)
        root = tree.getroot()
        ns = {
            "rdf": RDF_NS,
            "crs": CRS_NS,
        }
        for desc in root.findall(".//rdf:Description", ns):
            for child in desc:
                tag = child.tag
                local_key = tag.split("}")[-1] if "}" in tag else tag.split(":")[-1]
                if local_key != tag_name:
                    continue
                for li in child.findall(".//rdf:li", ns):
                    if not li.text:
                        continue
                    parts = [p.strip() for p in li.text.replace(" ", "").split(",")]
                    if len(parts) < 2:
                        continue
                    try:
                        points.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        continue
    except Exception:
        return ""
    return serialize_tone_curve_points(points)


def parse_tone_curve_pv2012_from_xmp(xmp_path: str) -> str:
    """Read crs:ToneCurvePV2012 point list → 'x,y;x,y' serialized string."""
    return _parse_point_curve_from_xmp(xmp_path, "ToneCurvePV2012")


def parse_dodge_burn_mask_from_xmp(xmp_path: str) -> str:
    """Read the custom DodgeBurnMask child element -> base64 PNG string."""
    if not xmp_path or not os.path.isfile(xmp_path):
        return ""
    try:
        tree = ET.parse(xmp_path)
        root = tree.getroot()
        ns = {"rdf": RDF_NS, "crs": CRS_NS}
        for desc in root.findall(".//rdf:Description", ns):
            for child in desc:
                tag = child.tag
                local_key = tag.split("}")[-1] if "}" in tag else tag.split(":")[-1]
                if local_key == "DodgeBurnMask" and child.text:
                    return child.text.strip()
    except Exception:
        pass
    return ""


def load_adjustments_for_file(image_path: str) -> Dict[str, float]:
    adj = dict(DEFAULT_ADJUSTMENTS)
    as_shot = read_as_shot_temperature(image_path)
    adj[AS_SHOT_TEMP_KEY] = as_shot
    xmp_path = resolve_xmp_path(image_path)
    has_xmp_temp = False
    if xmp_path and os.path.isfile(xmp_path):
        parsed = parse_xmp_adjustments(xmp_path)
        if "Temperature" in parsed:
            has_xmp_temp = True
        adj.update(parsed)
    if not has_xmp_temp:
        adj["Temperature"] = as_shot
    if xmp_path and os.path.isfile(xmp_path):
        from raw_tone_curve import TONE_CURVE_SERIAL_KEY

        serial = parse_tone_curve_pv2012_from_xmp(xmp_path)
        if serial:
            adj[TONE_CURVE_SERIAL_KEY] = serial
    if xmp_path and os.path.isfile(xmp_path):
        from raw_tone_curve import (
            TONE_CURVE_BLUE_KEY,
            TONE_CURVE_GREEN_KEY,
            TONE_CURVE_RED_KEY,
        )

        for key, tag in (
            (TONE_CURVE_RED_KEY, "ToneCurvePV2012Red"),
            (TONE_CURVE_GREEN_KEY, "ToneCurvePV2012Green"),
            (TONE_CURVE_BLUE_KEY, "ToneCurvePV2012Blue"),
        ):
            serial = _parse_point_curve_from_xmp(xmp_path, tag)
            if serial:
                adj[key] = serial
    if xmp_path and os.path.isfile(xmp_path):
        from raw_dodge_burn import MASK_KEY

        mask_serial = parse_dodge_burn_mask_from_xmp(xmp_path)
        if mask_serial:
            adj[MASK_KEY] = mask_serial
    return adj


def wb_reference_temperature(adj: Dict[str, float] | None) -> float:
    if not adj:
        return DEFAULT_ADJUSTMENTS["Temperature"]
    return float(adj.get(AS_SHOT_TEMP_KEY, DEFAULT_ADJUSTMENTS["Temperature"]))


def is_neutral_wb(adj: Dict[str, float] | None) -> bool:
    if not adj:
        return True
    ref = wb_reference_temperature(adj)
    temp = float(adj.get("Temperature", ref))
    tint = float(adj.get("Tint", 0.0))
    return abs(temp - ref) <= 0.5 and abs(tint) <= 1e-4


def solve_white_balance_from_sample(
    r: float, g: float, b: float, ref_temp: float
) -> tuple[float, float]:
    """
    White-balance dropper: solve Temperature/Tint so a sampled scene-linear RGB
    pixel (as-shot camera WB, pre Temperature/Tint) becomes neutral gray.

    Exact inverse of ``_apply_wb_tint`` rather than a new color model: Temperature
    scales R/B oppositely relative to G, so the R/B ratio depends only on
    Temperature (bisection, monotonic over the 2000-12000K slider range); Tint then
    scales G alone, solved algebraically once Temperature is fixed. Samples outside
    what the slider range / Tint's +/-10% G scaling can correct are clamped to the
    nearest achievable value rather than raising an error.
    """
    r = max(float(r), 1e-6)
    g = max(float(g), 1e-6)
    b = max(float(b), 1e-6)
    ref_temp = float(ref_temp) if ref_temp and ref_temp > 0 else DEFAULT_ADJUSTMENTS["Temperature"]
    r_ref, g_ref, b_ref = _kelvin_to_rgb(ref_temp)

    def rb_scale_ratio(t: float) -> float:
        r_t, g_t, b_t = _kelvin_to_rgb(t)
        scale_r = (r_t / r_ref) / (g_t / g_ref)
        scale_b = (b_t / b_ref) / (g_t / g_ref)
        return scale_r / scale_b

    target = b / r
    lo, hi = 2000.0, 12000.0
    ratio_lo, ratio_hi = rb_scale_ratio(lo), rb_scale_ratio(hi)  # monotonically decreasing
    if target >= ratio_lo:
        temperature = lo
    elif target <= ratio_hi:
        temperature = hi
    else:
        for _ in range(40):
            mid = (lo + hi) / 2.0
            if rb_scale_ratio(mid) > target:
                lo = mid
            else:
                hi = mid
        temperature = (lo + hi) / 2.0

    r_t, g_t, b_t = _kelvin_to_rgb(temperature)
    scale_r = (r_t / r_ref) / (g_t / g_ref)
    final_r = r * scale_r
    tint = ((1.0 - final_r / g) / 0.1) * 150.0 if g > 1e-6 else 0.0
    tint = max(-150.0, min(150.0, tint))
    return temperature, tint


def is_default_adjustments(adj: Dict[str, float] | None) -> bool:
    if not adj:
        return True
    from raw_dodge_burn import MASK_KEY
    from raw_tone_curve import CHANNEL_CURVE_KEYS, TONE_CURVE_SERIAL_KEY

    if str(adj.get(TONE_CURVE_SERIAL_KEY, "") or "").strip():
        return False
    for key in CHANNEL_CURVE_KEYS:
        if str(adj.get(key, "") or "").strip():
            return False
    if str(adj.get(MASK_KEY, "") or "").strip():
        return False
    ref_temp = wb_reference_temperature(adj)
    for key, default in DEFAULT_ADJUSTMENTS.items():
        av = float(adj.get(key, default))
        bv = float(default)
        if key == "Temperature":
            if abs(av - ref_temp) > 0.5:
                return False
        elif abs(av - bv) > 1e-4:
            return False
    return True


def adjustments_equal(a: Dict[str, float], b: Dict[str, float]) -> bool:
    ref_a = wb_reference_temperature(a)
    ref_b = wb_reference_temperature(b)
    for key, default in DEFAULT_ADJUSTMENTS.items():
        av = float(a.get(key, default))
        bv = float(b.get(key, default))
        if key == "Temperature":
            da = av - ref_a
            db = bv - ref_b
            if abs(da - db) > 0.5:
                return False
        elif abs(av - bv) > 1e-4:
            return False
    return True

def parse_xmp_adjustments(xmp_path: str) -> dict[str, float]:
    """Parse Lightroom-compatible crs adjustment sliders from an XMP sidecar file."""
    adjustments = {}
    try:
        tree = ET.parse(xmp_path)
        root = tree.getroot()
        
        ns = {
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'x': 'adobe:ns:meta/',
            'crs': 'http://adobe.com/camera-raw-settings/1.0/'
        }
        
        # 1. Search attributes on rdf:Description elements
        for desc in root.findall('.//rdf:Description', ns):
            for key, val in desc.attrib.items():
                local_key = key
                if '}' in key:
                    ns_uri, local_key = key.split('}', 1)
                    if ns_uri != '{' + ns['crs']:
                        continue
                elif ':' in key:
                    prefix, local_key = key.split(':', 1)
                    if prefix != 'crs':
                        continue
                
                try:
                    adjustments[local_key] = float(val)
                except ValueError:
                    pass
                    
        # 2. Search child elements in crs namespace under rdf:Description
        for desc in root.findall('.//rdf:Description', ns):
            for child in desc:
                tag = child.tag
                local_key = tag
                if '}' in tag:
                    ns_uri, local_key = tag.split('}', 1)
                    if ns_uri != '{' + ns['crs']:
                        continue
                elif ':' in tag:
                    prefix, local_key = tag.split(':', 1)
                    if prefix != 'crs':
                        continue
                if child.text:
                    try:
                        adjustments[local_key] = float(child.text.strip())
                    except ValueError:
                        pass
    except Exception:
        pass
    
    relevant_keys = RELEVANT_ADJUSTMENT_KEYS
    out = {k: v for k, v in adjustments.items() if k in relevant_keys}
    if "Defringe" not in out:
        purple = float(adjustments.get("DefringePurpleAmount", 0.0))
        green = float(adjustments.get("DefringeGreenAmount", 0.0))
        if purple > 0.0 or green > 0.0:
            out["Defringe"] = max(purple, green)
    return out


def write_xmp_adjustments(xmp_path: str, adj: Dict[str, float]) -> None:
    """Write Lightroom-compatible crs sliders to an XMP sidecar.

    Preserves an existing xmp:Rating -- this rebuilds the sidecar from
    scratch, and rating is written independently (write_xmp_rating, not
    gated by editing_features_enabled()), so a slider save must not blow
    away a star rating that was set earlier.

    Holds this sidecar's write lock for the whole read-modify-write --
    this runs on the export worker thread (see write_xmp_adjustments_for_file
    callers), so without it a same-moment write_xmp_rating() call from the
    main thread (rating a photo while it's exporting) could interleave and
    lose either change. See the _xmp_write_lock module comment.
    """
    if not editing_features_enabled():
        return
    with _xmp_write_lock(xmp_path):
        _write_xmp_adjustments_locked(xmp_path, adj)


def _write_xmp_adjustments_locked(xmp_path: str, adj: Dict[str, float]) -> None:
    existing_rating = parse_xmp_rating(xmp_path)
    merged = dict(DEFAULT_ADJUSTMENTS)
    merged.update(adj or {})
    if is_default_adjustments(merged):
        if existing_rating > 0:
            _write_xmp_rating_locked(xmp_path, existing_rating)
        elif os.path.isfile(xmp_path):
            os.remove(xmp_path)
        return

    ET.register_namespace("x", X_NS)
    ET.register_namespace("rdf", RDF_NS)
    ET.register_namespace("crs", CRS_NS)
    ET.register_namespace("xmp", XMP_NS)

    root = ET.Element(f"{{{X_NS}}}xmpmeta")
    rdf = ET.SubElement(root, f"{{{RDF_NS}}}RDF")
    desc = ET.SubElement(rdf, f"{{{RDF_NS}}}Description")
    desc.set(f"{{{RDF_NS}}}about", "")

    from raw_pv2012 import PROCESS_VERSION, TONE_CURVE_NAME_2012

    desc.set(f"{{{CRS_NS}}}ProcessVersion", PROCESS_VERSION)
    desc.set(f"{{{CRS_NS}}}ToneCurveName2012", TONE_CURVE_NAME_2012)
    if existing_rating > 0:
        desc.set(f"{{{XMP_NS}}}Rating", str(existing_rating))

    for key in sorted(RELEVANT_ADJUSTMENT_KEYS):
        if key == "Defringe":
            continue
        value = float(merged.get(key, DEFAULT_ADJUSTMENTS[key]))
        default = DEFAULT_ADJUSTMENTS[key]
        if key == "Temperature":
            ref = wb_reference_temperature(merged)
            if abs(value - ref) <= 0.5:
                continue
        elif abs(value - default) <= 1e-4:
            continue
        if key == "Temperature" and value <= 0:
            value = default
        desc.set(f"{{{CRS_NS}}}{key}", _format_xmp_value(key, value))

    defringe = float(merged.get("Defringe", 0.0))
    if defringe > 1e-4:
        amt = _format_xmp_value("Defringe", defringe)
        desc.set(f"{{{CRS_NS}}}DefringePurpleAmount", amt)
        desc.set(f"{{{CRS_NS}}}DefringeGreenAmount", amt)

    from raw_tone_curve import (
        TONE_CURVE_BLUE_KEY,
        TONE_CURVE_GREEN_KEY,
        TONE_CURVE_RED_KEY,
        TONE_CURVE_SERIAL_KEY,
        deserialize_tone_curve_points,
    )

    def _write_point_curve(key: str, tag: str) -> None:
        serial = str(merged.get(key, "") or "")
        points = deserialize_tone_curve_points(serial)
        if len(points) < 2:
            return
        tc = ET.SubElement(desc, f"{{{CRS_NS}}}{tag}")
        seq = ET.SubElement(tc, f"{{{RDF_NS}}}Seq")
        for x, y in points:
            li = ET.SubElement(seq, f"{{{RDF_NS}}}li")
            li.text = f"{int(round(x))}, {int(round(y))}"

    _write_point_curve(TONE_CURVE_SERIAL_KEY, "ToneCurvePV2012")
    _write_point_curve(TONE_CURVE_RED_KEY, "ToneCurvePV2012Red")
    _write_point_curve(TONE_CURVE_GREEN_KEY, "ToneCurvePV2012Green")
    _write_point_curve(TONE_CURVE_BLUE_KEY, "ToneCurvePV2012Blue")

    from raw_dodge_burn import MASK_KEY

    mask_serial = str(merged.get(MASK_KEY, "") or "")
    if mask_serial:
        db = ET.SubElement(desc, f"{{{CRS_NS}}}DodgeBurnMask")
        db.text = mask_serial

    tree = ET.ElementTree(root)
    parent = os.path.dirname(os.path.abspath(xmp_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    # Atomic write: write to a temp file in the same directory, then os.replace()
    # (atomic on POSIX and Windows). A direct tree.write(xmp_path, ...) could leave
    # a truncated/corrupted sidecar if interrupted (crash, force-quit, disk full)
    # mid-write, or if two writers race on the same path (e.g. the export worker
    # and a slider release firing concurrently).
    fd, tmp_path = tempfile.mkstemp(
        dir=parent or None, prefix=os.path.basename(xmp_path) + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as f:
            tree.write(f, encoding="utf-8", xml_declaration=True)
        os.replace(tmp_path, xmp_path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def write_xmp_adjustments_for_file(image_path: str, adj: Dict[str, float]) -> None:
    if not editing_features_enabled():
        return
    xmp_path = resolve_xmp_path(image_path)
    if not xmp_path:
        raise ValueError("Invalid image path")
    write_xmp_adjustments(xmp_path, adj)


def _format_xmp_value(key: str, value: float) -> str:
    if key == "Exposure2012":
        return f"{value:.2f}".rstrip("0").rstrip(".") if abs(value) < 10 else f"{value:.1f}"
    if key == "Temperature":
        return str(int(round(value)))
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def apply_adjustments_to_rgb(rgb_image: np.ndarray, adj: dict[str, float]) -> np.ndarray:
    """Apply adjustments; scene-linear uint16/float32 use the linear edit pipeline."""
    if rgb_image is None:
        return rgb_image
    if adj is None:
        return rgb_image
    if rgb_image.dtype in (np.uint16, np.float32):
        out = apply_adjustments_to_linear(rgb_image, adj)
        if out is not None and out.dtype in (np.uint16, np.float32):
            from raw_edit_pipeline import linear_to_display_uint8, process_linear_edit_buffer

            merged = dict(DEFAULT_ADJUSTMENTS)
            merged.update(adj or {})
            processed = process_linear_edit_buffer(rgb_image, merged, preview=True)
            out = linear_to_display_uint8(processed, merged)
        return out
    return _apply_adjustments_to_srgb(rgb_image, adj)


def apply_adjustments_to_linear(rgb_image: np.ndarray, adj: dict[str, float]) -> np.ndarray:
    """Scene-linear edit path: PV2012 + NLM chroma, tone-map, encode sRGB uint8."""
    from raw_edit_pipeline import linear_to_display_uint8, process_linear_edit_buffer

    if rgb_image is None:
        return rgb_image
    merged = dict(DEFAULT_ADJUSTMENTS)
    merged.update(adj or {})
    use_recovery = uses_recovery_tone_map(merged)
    if is_default_adjustments(merged) and not use_recovery:
        processed = process_linear_edit_buffer(rgb_image, {}, preview=True)
        return linear_to_display_uint8(processed, {})
    processed = process_linear_edit_buffer(rgb_image, merged, preview=True)
    return linear_to_display_uint8(processed, merged)


def apply_adjustments_to_linear_uint16(rgb_image: np.ndarray, adj: dict[str, float]) -> np.ndarray:
    """Full-quality export: PV2012 pipeline → 16-bit display-referred sRGB."""
    from raw_edit_pipeline import linear_to_export_uint16_srgb, process_linear_edit_buffer

    if rgb_image is None:
        return rgb_image
    merged = dict(DEFAULT_ADJUSTMENTS)
    merged.update(adj or {})
    processed = process_linear_edit_buffer(rgb_image, merged, preview=False, chroma_denoise=True)
    return linear_to_export_uint16_srgb(processed, merged)


def _apply_adjustments_to_srgb(rgb_image: np.ndarray, adj: dict[str, float]) -> np.ndarray:
    """Legacy gamma-space path for 8-bit sRGB buffers (cached previews / JPEG)."""
    if rgb_image is None or not adj:
        return rgb_image
    merged = dict(DEFAULT_ADJUSTMENTS)
    merged.update(adj)
    if is_default_adjustments(merged):
        return rgb_image

    # Geometry first (straighten/perspective/crop, raw_transform.py) so the
    # gamma path -- gallery tiles, fit previews, cached uint8 redisplay --
    # shows the same framing as the linear edit pipeline. Must run before the
    # row-band split below: it changes the buffer's shape.
    from raw_transform import apply_geometry

    rgb_image = apply_geometry(rgb_image, merged)

    # Row-band parallelism: every stage below is per-pixel except the small-
    # radius Gaussian blurs inside detail-enhance (Sharpness/Clarity/
    # Defringe), so splitting rows across threads with enough padding to
    # cover those blur radii gives byte-identical output, just computed
    # concurrently -- numpy/cv2 release the GIL for these array sizes, so a
    # plain ThreadPoolExecutor gets real parallelism without the IPC cost of
    # multiprocessing. Dodge & Burn is excluded (its mask needs its own
    # per-band slicing, not implemented here) and falls back to the
    # single-threaded path below.
    from raw_dodge_burn import MASK_KEY as _db_mask_key

    mask_serial = str(merged.get(_db_mask_key, "") or "")
    n_workers = _detail_pipeline_worker_count(rgb_image.shape[0])
    if not mask_serial and n_workers > 1:
        return _apply_adjustments_to_srgb_banded(rgb_image, merged, n_workers)

    img = rgb_image.astype(np.float32) / 255.0
    img = _apply_adjustments_to_srgb_core(img, merged)
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def banded_worker_count(height: int) -> int:
    """Row-band thread count for the banded pixel pipelines (shared by
    raw_adjustments and raw_edit_pipeline -- one tuning point, not several).
    """
    # Thread-pool overhead isn't worth it below a few hundred rows.
    if height < 512:
        return 1
    return min(8, os.cpu_count() or 1)


# Back-compat alias (older call sites/tests reference the private name).
_detail_pipeline_worker_count = banded_worker_count


def band_ranges(h: int, n_workers: int, pad_px: int = 0) -> list:
    """Split ``h`` rows into up to ``n_workers`` contiguous bands.

    Returns [(y0, y1, pad_top, pad_bot), ...] where pads are clamped to the
    image edges. Single source of truth for the band-boundary math used by
    every banded pipeline (a boundary off-by-one fixed here fixes all of
    them; previously three near-identical copies existed).
    """
    import math

    band_h = math.ceil(h / max(1, n_workers))
    bands = []
    for i in range(n_workers):
        y0 = i * band_h
        y1 = min(h, y0 + band_h)
        if y0 >= y1:
            break
        bands.append((y0, y1, min(pad_px, y0), min(pad_px, h - y1)))
    return bands


_BANDED_EXECUTOR = None
_BANDED_EXECUTOR_LOCK = None


def banded_executor():
    """Shared lazily-created pool for banded pixel stages.

    The banded functions used to build (spawn + join up to 8 OS threads) a
    fresh ThreadPoolExecutor per call -- on the Adjust-panel live-preview
    path that's pool churn on every slider tick. One idle pool costs nothing
    between calls; concurrent callers share it safely (Executor.map is
    thread-safe).
    """
    global _BANDED_EXECUTOR, _BANDED_EXECUTOR_LOCK
    import threading

    if _BANDED_EXECUTOR_LOCK is None:
        _BANDED_EXECUTOR_LOCK = threading.Lock()
    with _BANDED_EXECUTOR_LOCK:
        if _BANDED_EXECUTOR is None:
            from concurrent.futures import ThreadPoolExecutor

            _BANDED_EXECUTOR = ThreadPoolExecutor(
                max_workers=min(8, os.cpu_count() or 1),
                thread_name_prefix="banded-pixels",
            )
        return _BANDED_EXECUTOR


# Comfortably larger than 3x any Gaussian sigma used in apply_detail_enhancements
# (defringe 1.5, sharpness 0.9, clarity's downsampled effective ~10) -- a band's
# padded region always has enough real neighboring pixels for every blur stage
# to produce the same result as processing the full image in one pass.
_BAND_PAD_PX = 48


def _apply_adjustments_to_srgb_banded(
    rgb_image: np.ndarray, merged: dict[str, float], n_workers: int
) -> np.ndarray:
    """Row-band parallel version of _apply_adjustments_to_srgb's core pipeline."""
    h = rgb_image.shape[0]
    bands = band_ranges(h, n_workers, pad_px=_BAND_PAD_PX)

    def _process_band(band):
        y0, y1, pad_top, pad_bot = band
        src = rgb_image[y0 - pad_top : y1 + pad_bot]
        img = src.astype(np.float32) / 255.0
        out = _apply_adjustments_to_srgb_core(img, merged)
        out_u8 = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
        return out_u8[pad_top : pad_top + (y1 - y0)]

    import cv2

    prev_threads = cv2.getNumThreads()
    # Avoid oversubscription: cv2's own internal (IPP/TBB) threading would
    # otherwise compete with our outer thread pool for the same cores.
    cv2.setNumThreads(1)
    try:
        results = list(banded_executor().map(_process_band, bands))
    finally:
        cv2.setNumThreads(prev_threads)

    return np.concatenate(results, axis=0)


def _apply_adjustments_to_srgb_core(img: np.ndarray, merged: dict[str, float]) -> np.ndarray:
    """All per-pixel/local-only adjustment stages on a float32 [0,1] (H,W,3)
    buffer -- may be a padded row-band slice, see _apply_adjustments_to_srgb_banded.
    """
    # 1. Temperature & Tint (neutral gray preserved — scale R/B vs G)
    temp_val = float(merged.get("Temperature", DEFAULT_ADJUSTMENTS["Temperature"]))
    tint_val = float(merged.get("Tint", 0.0))
    if temp_val > 1000.0 and abs(temp_val - DEFAULT_ADJUSTMENTS["Temperature"]) > 1.0:
        r_tgt, g_tgt, b_tgt = _kelvin_to_rgb(temp_val)
        r_ref, g_ref, b_ref = _kelvin_to_rgb(DEFAULT_ADJUSTMENTS["Temperature"])
        if g_ref > 1e-5 and g_tgt > 1e-5:
            img[:, :, 0] *= (r_tgt / r_ref) / (g_tgt / g_ref)
            img[:, :, 2] *= (b_tgt / b_ref) / (g_tgt / g_ref)
    elif abs(temp_val) > 1e-4 and temp_val <= 1000.0:
        # Lightroom-style relative offset when not stored as Kelvin
        scale = temp_val / 100.0
        img[:, :, 0] *= 1.0 + scale * 0.1
        img[:, :, 2] *= 1.0 - scale * 0.1

    if abs(tint_val) > 1e-4:
        img[:, :, 1] *= 1.0 - (tint_val / 150.0) * 0.1

    # 2. Exposure
    exp_val = float(merged.get("Exposure2012", 0.0))
    if abs(exp_val) > 1e-4:
        img *= 2.0 ** exp_val

    # 2b. Dodge & burn (see raw_dodge_burn.py). This legacy gamma-space path
    # is used for cached uint8 full-image redisplay (browse revisit/zoom of
    # an edited photo, not the live Adjust-panel preview) -- applying the
    # mask here too keeps that path visually consistent with the scene-
    # linear pipeline's result (process_linear_edit_buffer).
    from raw_dodge_burn import MASK_KEY as _db_mask_key

    mask_serial = str(merged.get(_db_mask_key, "") or "")
    if mask_serial:
        from raw_dodge_burn import DEFAULT_STRENGTH as _db_default_strength
        from raw_dodge_burn import STRENGTH_KEY as _db_strength_key
        from raw_dodge_burn import _deserialize_mask_cached, apply_dodge_burn

        mask = _deserialize_mask_cached(mask_serial)
        if mask is not None:
            stops = float(merged.get(_db_strength_key, _db_default_strength))
            img = apply_dodge_burn(img, mask, stops)

    # 3. Contrast
    contrast_val = float(merged.get("Contrast2012", 0.0))
    if abs(contrast_val) > 1e-4:
        c = contrast_val / 100.0
        factor = (259.0 * (c * 100.0 + 255.0)) / (255.0 * (259.0 - c * 100.0))
        img = factor * (img - 0.5) + 0.5

    # 4. Highlights & Shadows — luminance-weighted (hue-preserving, PaintFE-style)
    hi_val = float(merged.get("Highlights2012", 0.0))
    sh_val = float(merged.get("Shadows2012", 0.0))
    if abs(hi_val) > 1e-4 or abs(sh_val) > 1e-4:
        img = _apply_highlights_shadows(img, hi_val, sh_val)

    # 5. Whites & Blacks
    white_val = float(merged.get("Whites2012", 0.0))
    black_val = float(merged.get("Blacks2012", 0.0))
    if abs(white_val) > 1e-4 or abs(black_val) > 1e-4:
        lum = _channel_luminance(img)
        if abs(black_val) > 1e-4:
            img = _apply_masked_luminance_adjust(
                img, lum, _black_region_weight(lum), black_val / 100.0, lift=True,
                lift_up_strength=0.03,
            )
            lum = _channel_luminance(img)
        if abs(white_val) > 1e-4:
            img = _apply_masked_luminance_adjust(
                img, lum, _highlight_region_weight(lum), white_val / 100.0, lift=True
            )

    # 6. Saturation & Vibrance
    sat_val = float(merged.get("Saturation", 0.0))
    vib_val = float(merged.get("Vibrance", 0.0))
    if abs(sat_val) > 1e-4 or abs(vib_val) > 1e-4:
        luma = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
        luma = np.expand_dims(luma, axis=-1)

        sat_scale = 1.0
        if abs(sat_val) > 1e-4:
            sat_scale += sat_val / 100.0

        if abs(vib_val) > 1e-4:
            # np.max/min(img, axis=-1) is ~8x slower than a direct elementwise
            # chain for a size-3 last axis (measured 356ms vs 45ms at 32MP).
            r_c, g_c, b_c = img[:, :, 0:1], img[:, :, 1:2], img[:, :, 2:3]
            max_val = np.maximum(np.maximum(r_c, g_c), b_c)
            min_val = np.minimum(np.minimum(r_c, g_c), b_c)
            s = (max_val - min_val) / (max_val + 1e-5)
            vib_factor = (vib_val / 100.0) * (1.0 - s)
            sat_scale += vib_factor

        img = luma + (img - luma) * sat_scale

    from raw_detail_enhance import apply_detail_enhancements

    img = np.clip(img, 0.0, 1.0)
    img = apply_detail_enhancements(img, merged)
    return img


def _linear_float_from_buffer(rgb_image: np.ndarray) -> np.ndarray:
    """Normalize uint16 linear or uint8 inputs to float32 scene-linear."""
    if rgb_image.dtype == np.uint16:
        return rgb_image.astype(np.float32) / 65535.0
    if rgb_image.dtype == np.float32:
        return np.clip(rgb_image, 0.0, None)
    return rgb_image.astype(np.float32) / 255.0


def _linear_buffer_to_display_uint8(rgb_image: np.ndarray) -> np.ndarray:
    if rgb_image is None:
        return rgb_image
    from raw_edit_pipeline import linear_to_display_uint8, process_linear_edit_buffer

    processed = process_linear_edit_buffer(rgb_image, {}, preview=True)
    return linear_to_display_uint8(processed)


def _tone_map_linear_to_srgb8(
    img: np.ndarray,
    encode_srgb8,
    luminance_fn,
) -> np.ndarray:
    """Luminance-preserving Reinhard shoulder, then sRGB encode."""
    lum = luminance_fn(img)
    mapped = lum / (1.0 + np.maximum(lum, 0.0))
    scale = mapped / np.maximum(lum, 1e-6)
    display = img * scale[..., np.newaxis]
    return encode_srgb8(display)


_SHADOW_LUM_MAX = 0.22
_HIGHLIGHT_LUM_MIN = 0.55
_BLACK_LUM_MAX = 0.07


def _channel_luminance(img: np.ndarray) -> np.ndarray:
    return 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]


def _shadow_region_weight(lum: np.ndarray) -> np.ndarray:
    return np.clip((_SHADOW_LUM_MAX - lum) / _SHADOW_LUM_MAX, 0.0, 1.0) ** 2.5


def _highlight_region_weight(lum: np.ndarray) -> np.ndarray:
    span = max(1.0 - _HIGHLIGHT_LUM_MIN, 1e-6)
    return np.clip((lum - _HIGHLIGHT_LUM_MIN) / span, 0.0, 1.0) ** 2.0


def _black_region_weight(lum: np.ndarray) -> np.ndarray:
    return np.clip((_BLACK_LUM_MAX - lum) / _BLACK_LUM_MAX, 0.0, 1.0) ** 2.0


def _apply_masked_luminance_adjust(
    img: np.ndarray,
    lum: np.ndarray,
    mask: np.ndarray,
    amount: float,
    *,
    lift: bool,
    lift_up_strength: float = 0.55,
) -> np.ndarray:
    """
    Hue-preserving tone tweak: scale RGB by a luminance ratio inside ``mask`` only.
    Positive amount lifts; negative amount darkens the masked region.

    ``lift_up_strength`` (the ``lift and amount > 0`` branch) is the only branch
    whose safe coefficient depends on how steep ``mask`` is: the shared 0.55
    default is safe for ``_highlight_region_weight`` (worst slope -1.0) but not
    for ``_shadow_region_weight`` (-12.36) or ``_black_region_weight`` (-29.57,
    a narrower 0.07-wide region) -- both produced verified tone-curve reversals
    (banding) well within slider range. Callers using those masks must pass a
    safe override (see call sites below); the other three branches are safe
    for every mask used in this module at their existing fixed coefficients.
    """
    if abs(amount) < 1e-6:
        return img
    if lift and amount > 0:
        lum_new = lum + mask * amount * (1.0 - lum) * lift_up_strength
    elif lift and amount < 0:
        lum_new = lum + mask * amount * lum * 0.45
    elif not lift and amount < 0:
        knee = np.maximum(lum - _HIGHLIGHT_LUM_MIN, 0.0)
        lum_new = lum + mask * amount * knee * 0.65
    else:
        lum_new = lum + mask * amount * (1.0 - lum) * 0.35
    ratio = lum_new / np.maximum(lum, 1e-6)
    # A per-channel loop with a sliced out= beats both a plain (H,W,3)*(H,W)
    # broadcast and pre-tiling ratio to (H,W,3) via np.repeat (measured at
    # 32MP: broadcast 142ms, repeat-tile 119ms, per-channel-out 91ms) --
    # numpy's broadcasting iterator doesn't take the fast contiguous SIMD
    # path for a trailing size-1 axis, and repeat pays its own full-size
    # allocation+copy. Bit-identical to the broadcast result.
    out = np.empty_like(img)
    for c in range(img.shape[-1]):
        np.multiply(img[:, :, c], ratio, out=out[:, :, c])
    return out


def _apply_highlights_shadows(img: np.ndarray, hi_val: float, sh_val: float) -> np.ndarray:
    """Gamma-space HS using the same masked luminance ratios."""
    lum = _channel_luminance(img)
    if abs(sh_val) > 1e-4:
        img = _apply_masked_luminance_adjust(
            img, lum, _shadow_region_weight(lum), sh_val / 100.0, lift=True,
            lift_up_strength=0.07,
        )
        lum = _channel_luminance(img)
    if abs(hi_val) > 1e-4:
        amt = hi_val / 100.0
        if amt < 0:
            img = _apply_masked_luminance_adjust(
                img, lum, _highlight_region_weight(lum), amt, lift=False
            )
        else:
            img = _apply_masked_luminance_adjust(
                img, lum, _highlight_region_weight(lum), amt, lift=True
            )
    return img


def _kelvin_to_rgb(k: float) -> tuple[float, float, float]:
    k = max(1000.0, min(40000.0, k))
    temp = k / 100.0
    
    # Red
    if temp <= 66.0:
        r = 255.0
    else:
        r = temp - 60.0
        r = 329.698727446 * (r ** -0.1332047592)
        r = max(0.0, min(255.0, r))
        
    # Green
    if temp <= 66.0:
        g = temp
        g_val = max(1.0, g)
        g = 99.4708025861 * np.log(g_val) - 161.1195681661
        g = max(0.0, min(255.0, g))
    else:
        g = temp - 60.0
        g = 288.1221695283 * (g ** -0.0755148492)
        g = max(0.0, min(255.0, g))
        
    # Blue
    if temp >= 66.0:
        b = 255.0
    else:
        if temp <= 19.0:
            b = 0.0
        else:
            b = temp - 10.0
            b_val = max(1.0, b)
            b = 138.5177312231 * np.log(b_val) - 305.0447927307
            b = max(0.0, min(255.0, b))
            
    return r / 255.0, g / 255.0, b / 255.0
