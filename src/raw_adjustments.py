"""
RAW adjustments parser and NumPy math helper.
Applies Exposure, Contrast, Highlights, Shadows, Whites, Blacks, Temp, Tint, Saturation, and Vibrance.
"""

from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

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
    from raw_tone_curve import TONE_CURVE_SERIAL_KEY

    if str(adj.get(TONE_CURVE_SERIAL_KEY, "") or "").strip():
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
)


def resolve_xmp_path(image_path: str) -> str:
    """Sidecar path next to the image (Lightroom-style basename.xmp)."""
    if not image_path:
        return ""
    companion = image_path + ".xmp"
    if os.path.isfile(companion):
        return companion
    return os.path.splitext(image_path)[0] + ".xmp"


def read_as_shot_temperature(image_path: str) -> float:
    """Best-effort as-shot CCT from EXIF or RAW metadata (not a UI preset)."""
    if not image_path:
        return DEFAULT_ADJUSTMENTS["Temperature"]
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
                    return float(val)
    except Exception:
        pass
    try:
        import rawpy

        with rawpy.imread(image_path) as raw:
            cam = np.array(raw.camera_whitebalance[:3], dtype=np.float64)
            day = np.array(raw.daylight_whitebalance[:3], dtype=np.float64)
            if np.all(cam > 0) and np.all(day > 0):
                rb_cam = cam[0] / cam[2]
                rb_day = day[0] / day[2]
                ratio = rb_cam / max(rb_day, 1e-6)
                est = 5500.0 * (ratio ** -0.35)
                return float(np.clip(est, 2000.0, 12000.0))
    except Exception:
        pass
    return DEFAULT_ADJUSTMENTS["Temperature"]


def parse_tone_curve_pv2012_from_xmp(xmp_path: str) -> str:
    """Read crs:ToneCurvePV2012 point list → 'x,y;x,y' serialized string."""
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
                if local_key != "ToneCurvePV2012":
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
    from raw_tone_curve import TONE_CURVE_SERIAL_KEY

    if str(adj.get(TONE_CURVE_SERIAL_KEY, "") or "").strip():
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
    """Write Lightroom-compatible crs sliders to an XMP sidecar."""
    merged = dict(DEFAULT_ADJUSTMENTS)
    merged.update(adj or {})
    if is_default_adjustments(merged):
        if os.path.isfile(xmp_path):
            os.remove(xmp_path)
        return

    ET.register_namespace("x", X_NS)
    ET.register_namespace("rdf", RDF_NS)
    ET.register_namespace("crs", CRS_NS)

    root = ET.Element(f"{{{X_NS}}}xmpmeta")
    rdf = ET.SubElement(root, f"{{{RDF_NS}}}RDF")
    desc = ET.SubElement(rdf, f"{{{RDF_NS}}}Description")
    desc.set(f"{{{RDF_NS}}}about", "")

    from raw_pv2012 import PROCESS_VERSION, TONE_CURVE_NAME_2012

    desc.set(f"{{{CRS_NS}}}ProcessVersion", PROCESS_VERSION)
    desc.set(f"{{{CRS_NS}}}ToneCurveName2012", TONE_CURVE_NAME_2012)

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

    from raw_tone_curve import TONE_CURVE_SERIAL_KEY, deserialize_tone_curve_points

    serial = str(merged.get(TONE_CURVE_SERIAL_KEY, "") or "")
    points = deserialize_tone_curve_points(serial)
    if len(points) >= 2:
        tc = ET.SubElement(desc, f"{{{CRS_NS}}}ToneCurvePV2012")
        seq = ET.SubElement(tc, f"{{{RDF_NS}}}Seq")
        for x, y in points:
            li = ET.SubElement(seq, f"{{{RDF_NS}}}li")
            li.text = f"{int(round(x))}, {int(round(y))}"

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

    # Convert image to float32 normalized [0.0, 1.0] for calculations
    img = rgb_image.astype(np.float32) / 255.0

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
            max_val = np.max(img, axis=-1, keepdims=True)
            min_val = np.min(img, axis=-1, keepdims=True)
            s = (max_val - min_val) / (max_val + 1e-5)
            vib_factor = (vib_val / 100.0) * (1.0 - s)
            sat_scale += vib_factor

        img = luma + (img - luma) * sat_scale

    from raw_detail_enhance import apply_detail_enhancements

    img = np.clip(img, 0.0, 1.0)
    img = apply_detail_enhancements(img, merged)
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


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
    return img * ratio[..., np.newaxis]


def _apply_highlights_shadows_linear(
    img: np.ndarray, hi_val: float, sh_val: float
) -> np.ndarray:
    """Shadows / highlights via masked luminance ratios (no global gray add)."""
    from raw_tone_recovery import _luminance

    lum = _luminance(img)
    if abs(sh_val) > 1e-4:
        img = _apply_masked_luminance_adjust(
            img, lum, _shadow_region_weight(lum), sh_val / 100.0, lift=True,
            lift_up_strength=0.07,
        )
        lum = _luminance(img)
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
