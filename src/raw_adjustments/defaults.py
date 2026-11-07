"""
RAW adjustments parser and NumPy math helper.
Applies Exposure, Contrast, Highlights, Shadows, Whites, Blacks, Temp, Tint, Saturation, and Vibrance.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass

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
_xmp_write_locks: dict[str, threading.Lock] = {}
_xmp_write_locks_guard = threading.Lock()


def _xmp_write_lock(xmp_path: str) -> threading.Lock:
    key = os.path.normcase(os.path.abspath(xmp_path)) if xmp_path else ""
    with _xmp_write_locks_guard:
        lock = _xmp_write_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _xmp_write_locks[key] = lock
        return lock


DEFAULT_ADJUSTMENTS: dict[str, float] = {
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
    # AI denoise detail preservation (export-only): 0 = maximum smoothness,
    # 100 = maximum original fine detail. Drives the half-res AI denoise
    # edge-weighted high-frequency restore gain in onnx_scunet.process().
    "DenoiseDetail": 50.0,
    # AI denoise amount (export-only): 0 = original untouched, 100 = full
    # model output. Blends the model result with the input in
    # onnx_scunet.process(strength=...), like Lightroom's Denoise Amount.
    "AIDenoiseAmount": 100.0,
    "ParametricShadows": 0.0,
    "ParametricDarks": 0.0,
    "ParametricLights": 0.0,
    "ParametricHighlights": 0.0,
    # Off by default -- an automatic geometry change should never apply
    # silently. Baked into the decoded edit base at decode time (see
    # processing.unified.decode_raw_edit_base), not a per-tick pipeline
    # adjustment -- see raw_lens_correction.py.
    "LensCorrectionEnabled": 0.0,
    "DenoiseMethod": 0.0,
    # Geometry (see raw_transform.py): straighten + keystone perspective +
    # per-edge crop insets. Applied once at the head of both pipelines;
    # always stays within the original pixel frame (auto inscribed-rect crop).
    "CropAngle": 0.0,
    "PerspectiveVertical": 0.0,
    "PerspectiveHorizontal": 0.0,
    "Distortion": 0.0,
    "AnamorphicRatio": 1.0,
    "CropLeft": 0.0,
    "CropRight": 0.0,
    "CropTop": 0.0,
    "CropBottom": 0.0,
    # Dodge & burn stops-per-mask-unit (see raw_dodge_burn.py). The mask
    # itself (a base64 PNG blob, potentially large) is NOT a plain numeric
    # attribute -- it's stored as its own XMP child element, mirroring
    # ToneCurvePV2012 (see write_xmp_adjustments / parse_dodge_burn_mask_from_xmp).
    "DodgeBurnStrength": 1.75,
    # Look / effects (see raw_effects.py) ΓÇö display-linear after tone-map.
    "PostCropVignetteAmount": 0.0,
    "PostCropVignetteMidpoint": 50.0,
    "Dehaze": 0.0,
    # Creative 3D LUT amount 0ΓÇô100 (see raw_lut.py). LUT basename is a string
    # key (CreativeLUTName), handled like tone-curve serials ΓÇö not in this float map.
    "CreativeLUTAmount": 0.0,
}

from raw_adjustments.hsl import HSL_COLOR_NAMES  # noqa: E402

for _hsl_color in HSL_COLOR_NAMES:
    DEFAULT_ADJUSTMENTS[f"HueAdjustment{_hsl_color}"] = 0.0
    DEFAULT_ADJUSTMENTS[f"SaturationAdjustment{_hsl_color}"] = 0.0
    DEFAULT_ADJUSTMENTS[f"LuminanceAdjustment{_hsl_color}"] = 0.0

RELEVANT_ADJUSTMENT_KEYS = frozenset(DEFAULT_ADJUSTMENTS.keys())
# Export-only denoise knobs: they change the exported pixels but are invisible
# in the always-shown preview, so they must not mark a file "edited" -- yet a
# change to any of them must still be persisted to the sidecar.
_EXPORT_ONLY_DENOISE_KEYS = frozenset({"DenoiseDetail", "AIDenoiseAmount"})
AS_SHOT_TEMP_KEY = "AsShotTemperature"

ANAMORPHIC_DESQUEEZE_PRESETS: tuple[tuple[str, float], ...] = (
    ("Off (1.0x)", 1.0),
    ("1.33x", 1.33),
    ("1.5x", 1.5),
    ("1.6x", 1.6),
    ("2.0x", 2.0),
)
_PROFILE_STALE_WARNED: set[str] = set()

# Common photographic white-balance presets (Kelvin). These are industry-
# standard illuminant targets (Lightroom-style), not EXIF enums ΓÇö cameras
# rarely store "Cloudy"/"Shade" as a named mode with a reliable K value.
# "As Shot" uses read_as_shot_temperature() / AsShotTemperature from EXIF.
WB_PRESETS: tuple[tuple[str, float | None], ...] = (
    ("As Shot", None),
    ("Daylight", 5500.0),
    ("Cloudy", 6500.0),
    ("Shade", 7500.0),
    ("Tungsten", 2850.0),
    ("Fluorescent", 3800.0),
    ("Flash", 5500.0),
)
CHROMA_NR_ON_VALUE = 50.0
RECOVERY_BASELINE_KEY = "_recovery_baseline"
# UI hints when recovery look is on (local S/H recovery Γëê these PV2012 readouts).
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
    _slider_linear("AIDenoiseAmount", "AI Amount", 0, 100, 100.0, fmt=lambda x: f"{x:.0f}"),
    _slider_linear("DenoiseDetail", "AI Detail", 0, 100, 50.0, fmt=lambda x: f"{x:.0f}"),
    # Transform (raw_transform.py). Straighten in 0.1-degree steps; keystone
    # sliders mirror Lightroom's slider-based Transform panel; crop insets as
    # per-edge percentages (stored as fractions).
    _slider_linear("CropAngle", "Straighten", -450, 450, 0.0, scale=0.1, fmt=lambda x: f"{x:+.1f}┬░"),
    _slider_linear("PerspectiveVertical", "Vertical", -100, 100, 0.0),
    _slider_linear("PerspectiveHorizontal", "Horizontal", -100, 100, 0.0),
    _slider_linear("Distortion", "Distortion", -100, 100, 0.0),
    # Per-edge crop-inset sliders were removed by request: cropping stays out
    # of the UI until a proper interactive overlay (visible crop rectangle
    # with drag handles) exists. raw_transform.apply_geometry still honors
    # the keys, so the future overlay only needs to write them.
    # (Crop overlay writes CropLeft/Right/Top/Bottom; no numeric sliders.)
    _slider_linear("PostCropVignetteAmount", "Vignette", -100, 100, 0.0),
    _slider_linear(
        "PostCropVignetteMidpoint",
        "Midpoint",
        0,
        100,
        50.0,
        fmt=lambda x: f"{x:.0f}",
    ),
    _slider_linear("Dehaze", "Dehaze", -100, 100, 0.0),
)


def _kelvin_to_rgb(k: float) -> tuple[float, float, float]:
    """Convert colour temperature (K) to linear sRGB multipliers (McCamry-style)."""
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
