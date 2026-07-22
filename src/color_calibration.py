"""Camera color calibration engine using standard 24-patch ColorChecker targets and EXIF model registry."""

from __future__ import annotations

import json
import logging
import os
import math
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Standard 24 ColorChecker Patch Reference Colors (sRGB 0-255)
# Row 1: 1.Dark Skin, 2.Light Skin, 3.Blue Sky, 4.Foliage, 5.Blue Flower, 6.Bluish Green
# Row 2: 7.Orange, 8.Purplish Blue, 9.Moderate Red, 10.Purple, 11.Yellow Green, 12.Orange Yellow
# Row 3: 13.Blue, 14.Green, 15.Red, 16.Yellow, 17.Magenta, 18.Cyan
# Row 4: 19.White (.05 D), 20.Neutral 8 (.23 D), 21.Neutral 6.5 (.44 D), 22.Neutral 5 (.70 D), 23.Neutral 3.5 (1.05 D), 24.Black (1.50 D)
COLORCHECKER_24_REF = np.array([
    [115, 82, 68],   [194, 150, 130], [98, 122, 157],  [87, 108, 67],   [133, 128, 177], [103, 189, 170],
    [214, 126, 44],  [80, 91, 166],   [193, 90, 99],   [94, 60, 108],   [157, 188, 64],  [224, 163, 46],
    [56, 61, 150],   [70, 148, 73],   [175, 54, 60],   [231, 199, 31],  [187, 86, 149],  [8, 133, 161],
    [243, 243, 242], [200, 200, 200], [160, 160, 160], [122, 122, 121], [85, 85, 85],    [52, 52, 52]
], dtype=np.float32)


def get_camera_profile_path() -> str:
    """Path to stored camera model profiles JSON file in user app directory."""
    app_dir = os.path.join(os.path.expanduser("~"), ".rawviewer")
    os.makedirs(app_dir, exist_ok=True)
    return os.path.join(app_dir, "camera_profiles.json")


def normalize_camera_key(make: str, model: str, iso: Optional[int] = None) -> str:
    """Generate normalized lookup key from camera Make, Model, and optional ISO level."""
    make_clean = (make or "").strip().lower()
    model_clean = (model or "").strip().lower()
    raw = f"{make_clean}_{model_clean}".strip("_")
    cleaned = "".join(c if c.isalnum() else "_" for c in raw)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    base_key = cleaned.strip("_") or "unknown_camera"
    if iso is not None and iso > 0:
        return f"{base_key}_iso{iso}"
    return base_key


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


def save_camera_profile(
    make: str,
    model: str,
    profile_data: Dict[str, Any],
    iso: Optional[int] = None,
) -> bool:
    """Save calibrated profile data for a specific camera model and optional ISO level."""
    key = normalize_camera_key(make, model, iso=iso)
    path = get_camera_profile_path()
    registry = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except Exception:
            registry = {}

    clean_data = _sanitize_for_json(profile_data)
    clean_data["make"] = make
    clean_data["model"] = model
    clean_data["iso"] = iso
    registry[key] = clean_data

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
        logger.info("Saved camera calibration profile for %s %s (ISO %s) -> key: %s", make, model, iso, key)
        return True
    except Exception as exc:
        logger.error("Failed to save camera profile: %s", exc)
        return False


def get_camera_profile(
    make: str,
    model: str,
    iso: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve calibrated profile data.
    
    Order of preference:
    1. Exact ISO match (e.g. sony_ilce_7rm4_iso400)
    2. Closest ISO match for this camera model
    3. General camera profile fallback (sony_ilce_7rm4)
    """
    base_key = normalize_camera_key(make, model, iso=None)
    path = get_camera_profile_path()
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            registry = json.load(f)

        # 1. Exact ISO match
        if iso is not None and iso > 0:
            exact_key = f"{base_key}_iso{iso}"
            if exact_key in registry:
                return registry[exact_key]

            # 2. Closest ISO match for this camera model
            iso_keys = [k for k in registry if k.startswith(f"{base_key}_iso")]
            if iso_keys:
                iso_pairs = []
                for k in iso_keys:
                    try:
                        k_iso = int(k.split("_iso")[-1])
                        iso_pairs.append((abs(k_iso - iso), k))
                    except ValueError:
                        pass
                if iso_pairs:
                    iso_pairs.sort(key=lambda x: x[0])
                    return registry[iso_pairs[0][1]]

        # 3. General fallback
        return registry.get(base_key)
    except Exception as exc:
        logger.warning("Error reading camera profile registry: %s", exc)
        return None


def delete_camera_profile(
    make: str,
    model: str,
    iso: Optional[int] = None,
) -> bool:
    """Remove a stored profile, reverting this camera to the LibRaw baseline.

    Deletes the exact ISO-specific key when ``iso`` is given, otherwise the
    general key for the model. Returns True if an entry was actually removed.
    Without this there is no way out of a bad calibration short of hand-editing
    camera_profiles.json, while ``apply_camera_profile_defaults`` keeps applying
    it to every future shot from the same body.
    """
    key = normalize_camera_key(make, model, iso=iso)
    path = get_camera_profile_path()
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            registry = json.load(f)
    except Exception as exc:
        logger.warning("Could not read camera profile registry for delete: %s", exc)
        return False

    if key not in registry:
        return False
    registry.pop(key, None)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
        logger.info("Deleted camera calibration profile key: %s", key)
        return True
    except Exception as exc:
        logger.error("Failed to delete camera profile: %s", exc)
        return False


def describe_camera_profile(
    make: str,
    model: str,
    iso: Optional[int] = None,
) -> Optional[str]:
    """Short human label for the profile that would apply, or None.

    Used by the Adjust panel so an auto-applied profile is visible rather than
    a silent colour shift the user cannot attribute.
    """
    prof = get_camera_profile(make, model, iso=iso)
    if not prof:
        return None
    name = f"{prof.get('make', make) or ''} {prof.get('model', model) or ''}".strip()
    prof_iso = prof.get("iso")
    if prof_iso:
        return f"{name or 'Camera'} (ISO {prof_iso})"
    return name or "Camera"


def has_factory_color_profile(file_path: str) -> Optional[bool]:
    """Whether LibRaw has a real colour matrix for this file's camera.

    True  -> LibRaw ships a camera-specific matrix; calibrating overrides
             colour science that is already working.
    False -> no matrix (all zeros); the file decodes but gets generic colour,
             which is the case manual calibration genuinely helps.
    None  -> could not be determined (unreadable / non-RAW); caller should not
             draw a conclusion either way.

    Note ``color_matrix`` is unset (all zeros) even on fully supported bodies --
    only ``rgb_xyz_matrix`` is a reliable signal.
    """
    try:
        import rawpy

        with rawpy.imread(file_path) as raw:
            mat = np.asarray(raw.rgb_xyz_matrix, dtype=np.float64)
    except Exception as exc:
        logger.debug("Could not read rgb_xyz_matrix for %s: %s", file_path, exc)
        return None
    if mat.size == 0:
        return None
    return bool(np.any(np.abs(mat) > 1e-6))


def extract_patch_colors(
    image: np.ndarray,
    corners: List[Tuple[float, float]],
) -> List[Tuple[float, float, float]]:
    """Extract average RGB colors for the 24 patches from 4 corner pins on image canvas."""
    if len(corners) != 4 or image is None or image.size == 0:
        return []

    pts_src = np.array(corners, dtype=np.float32)
    # Warped target grid coordinates (6 columns x 4 rows)
    width, height = 600, 400
    pts_dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(image, M, (width, height))

    sampled = []
    col_w = width / 6.0
    row_h = height / 4.0

    for r in range(4):
        for c in range(6):
            # Center 50% box of patch to avoid border glare
            x1 = int(c * col_w + col_w * 0.25)
            x2 = int(c * col_w + col_w * 0.75)
            y1 = int(r * row_h + row_h * 0.25)
            y2 = int(r * row_h + row_h * 0.75)
            
            patch = warped[y1:y2, x1:x2]
            mean_rgb = np.mean(patch, axis=(0, 1))
            sampled.append((float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])))

    return sampled


def validate_and_detect_color_checker(
    image: np.ndarray,
    corners: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[bool, str, List[Tuple[float, float, float]]]:
    """Validate whether a valid 24-patch ColorChecker chart is present in the specified region.
    
    Returns (is_valid, error_message, sampled_patches).
    """
    if image is None or image.size == 0:
        return False, "Invalid or empty image buffer for color calibration.", []

    if not corners or len(corners) != 4:
        return False, "Please select the 4 corners bounding the ColorChecker chart.", []

    sampled = extract_patch_colors(image, corners)
    if not sampled or len(sampled) != 24:
        return False, "Could not sample 24 patches from the selected region.", []

    # 1. Check Color & Luminance Variance across all 24 patches
    lums = [0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2] for p in sampled]
    std_lum = float(np.std(lums))
    if std_lum < 15.0:
        return (
            False,
            "No ColorChecker chart detected in the selected area (insufficient patch color variation).\n\n"
            "Please align the 4 corner handles over a valid 24-patch ColorChecker chart.",
            [],
        )

    # 2. Check Gray Ramp Monotonicity (Row 4, Patches 19-24)
    gray_lums = lums[18:24]
    if not (gray_lums[0] > gray_lums[3] > gray_lums[5]):
        return (
            False,
            "Chart validation failed: Selected area neutral gray scale does not match a standard ColorChecker layout.\n\n"
            "Please ensure the 4 corners accurately enclose the 24 patches with white on the bottom-left.",
            [],
        )

    return True, "", sampled


def _srgb_encode_0_255(rgb_0_255: np.ndarray) -> np.ndarray:
    """Scene-linear 0-255 -> sRGB display-encoded 0-255 (sRGB OETF)."""
    lin = np.clip(np.asarray(rgb_0_255, dtype=np.float32) / 255.0, 0.0, 1.0)
    enc = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1.0 / 2.4) - 0.055)
    return (enc * 255.0).astype(np.float32)


def _srgb_decode_0_255(rgb_0_255: np.ndarray) -> np.ndarray:
    """sRGB display-encoded 0-255 -> scene-linear 0-255 (inverse OETF)."""
    enc = np.clip(np.asarray(rgb_0_255, dtype=np.float32) / 255.0, 0.0, 1.0)
    lin = np.where(enc <= 0.04045, enc / 12.92, np.power((enc + 0.055) / 1.055, 2.4))
    return (lin * 255.0).astype(np.float32)


def _normalize_exposure(sampled: np.ndarray) -> np.ndarray:
    """Scale scene-linear patches so the white patch matches the reference white.

    The edit base is a scene-linear decode with no auto-brightening, so the
    chart's white patch lands wherever the exposure happened to put it (on the
    test CR3: 81/255 against a 243/255 reference, a 2.84x gap). Without this,
    the same camera calibrates differently from a darker vs brighter frame of
    the same chart, and every colour delta is dominated by exposure rather
    than by the camera's colour response.
    """
    white_lin = float(np.mean(sampled[18]))
    ref_white_lin = float(np.mean(_srgb_decode_0_255(COLORCHECKER_24_REF[18])))
    if white_lin <= 1e-3:
        return sampled
    return sampled * (ref_white_lin / white_lin)


def calibrate_camera_curves_and_hsl(
    sampled_rgb: List[Tuple[float, float, float]],
    wb_mode: str = "auto",  # "auto" (all 6 neutral patches), "white_19", "neutral_22"
) -> Dict[str, Any]:
    """Calculate White Balance shift and HSL deltas from 24 scene-linear patches.

    ``sampled_rgb`` is scene-linear 0-255 (see ``decode_raw_edit_base``). Each
    output is derived in the space it is *applied* in, which is not the same
    space for all of them:

    * Temperature / Tint -> applied in the scene-linear WB stage
      (``raw_edit_pipeline._apply_wb_tint``), so solved on linear ratios here.
    * HSL hue/sat/lum -> applied after the tone map, in display space, so
      solved on sRGB-encoded values against the sRGB reference chart.

    Deriving HSL on linear values against sRGB references biased every band the
    same way (measured on a neutral, correctly-exposed Canon CR3: lum +5.8 to
    +18.4 across all 8 bands, saturation -2.2 to -9.8, for a camera that needs
    almost no correction).
    """
    if len(sampled_rgb) != 24:
        raise ValueError(f"Expected 24 sampled RGB patches, got {len(sampled_rgb)}")

    sampled = np.array(sampled_rgb, dtype=np.float32)

    # 1. Neutral Gray Ramp Calibration (Patches 18-23, 0-indexed).
    # Stays scene-linear: WB is a ratio between channels and is applied in the
    # linear stage, so encoding first would skew it.
    gray_sampled = sampled[18:24]

    # White Balance calculation from neutral patches
    if wb_mode == "white_19":
        samp_wb = gray_sampled[0]
    elif wb_mode == "neutral_22":
        samp_wb = gray_sampled[3]
    else:  # "auto": robust average across all 6 neutral gray patches
        samp_wb = np.mean(gray_sampled, axis=0)

    g_val = max(1e-3, float(samp_wb[1]))
    r_ratio = float(samp_wb[0]) / g_val
    b_ratio = float(samp_wb[2]) / g_val

    # Calculate WB Temperature shift (in Kelvin) and Tint shift
    temp_shift = round((1.0 - r_ratio) * 2000.0, 1)
    tint_shift = round((1.0 - (g_val / ((float(samp_wb[0]) + float(samp_wb[2])) * 0.5 + 1e-4))) * 50.0, 1)

    # 2. HSL Color Patch Shifts (Patches 0-17).
    # Exposure-normalise, then sRGB-encode: HSL is applied post-tone-map, so the
    # deltas must be measured in that same display space (see docstring).
    display = _srgb_encode_0_255(_normalize_exposure(sampled))

    color_bands = ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"]

    # Map color patches to HSL bands
    patch_band_map = {
        0: "Red", 1: "Orange", 2: "Blue", 3: "Green", 4: "Purple", 5: "Aqua",
        6: "Orange", 7: "Blue", 8: "Red", 9: "Purple", 10: "Yellow", 11: "Yellow",
        12: "Blue", 13: "Green", 14: "Red", 15: "Yellow", 16: "Magenta", 17: "Aqua"
    }

    # Accumulate per band, then average. Bands do not get equal patch counts on
    # a 24-patch chart (Red/Blue/Yellow have 3 each, Magenta only 1), so summing
    # made a band's correction scale with how often it appears on the chart.
    acc: Dict[str, List[Tuple[float, float, float]]] = {b: [] for b in color_bands}

    for idx, band in patch_band_map.items():
        s_rgb = np.uint8([[np.clip(display[idx], 0, 255)]])
        r_rgb = np.uint8([[COLORCHECKER_24_REF[idx]]])

        s_hsv = cv2.cvtColor(s_rgb, cv2.COLOR_RGB2HSV)[0, 0]
        r_hsv = cv2.cvtColor(r_rgb, cv2.COLOR_RGB2HSV)[0, 0]

        # OpenCV 8-bit hue is 0-179 and wraps; take the shorter way round so a
        # red patch straddling 0/180 can't produce a ~180 degree "correction".
        dh = float(r_hsv[0]) - float(s_hsv[0])
        if dh > 90.0:
            dh -= 180.0
        elif dh < -90.0:
            dh += 180.0
        ds = (float(r_hsv[1]) - float(s_hsv[1])) / 255.0 * 20.0
        dl = (float(r_hsv[2]) - float(s_hsv[2])) / 255.0 * 20.0
        acc[band].append((dh, ds, dl))

    hue_shifts = {b: 0.0 for b in color_bands}
    sat_shifts = {b: 0.0 for b in color_bands}
    lum_shifts = {b: 0.0 for b in color_bands}
    for band, deltas in acc.items():
        if not deltas:
            continue
        arr = np.array(deltas, dtype=np.float32)
        hue_shifts[band] = round(float(arr[:, 0].mean()) * 0.5, 1)
        sat_shifts[band] = round(float(arr[:, 1].mean()) * 0.5, 1)
        lum_shifts[band] = round(float(arr[:, 2].mean()) * 0.5, 1)

    return {
        "temperature_shift": temp_shift,
        "tint_shift": tint_shift,
        "hsl_hue": hue_shifts,
        "hsl_sat": sat_shifts,
        "hsl_lum": lum_shifts,
    }
