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
    app_dir = os.path.join(os.path.expanduser("~"), ".gemini", "antigravity-ide")
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


def calibrate_camera_curves_and_hsl(
    sampled_rgb: List[Tuple[float, float, float]]
) -> Dict[str, Any]:
    """Calculate calibrated RGB curves, White Balance shift, and HSL deltas from 24 patches."""
    if len(sampled_rgb) != 24:
        raise ValueError(f"Expected 24 sampled RGB patches, got {len(sampled_rgb)}")

    sampled = np.array(sampled_rgb, dtype=np.float32)

    # 1. Neutral Gray Ramp Calibration (Patches 18-23, 0-indexed)
    gray_sampled = sampled[18:24]
    gray_ref = COLORCHECKER_24_REF[18:24]

    # Calculate White Balance Temperature/Tint shift from white patch #19
    w_samp = gray_sampled[0]  # White patch
    g_val = max(1e-3, w_samp[1])
    r_ratio = w_samp[0] / g_val
    b_ratio = w_samp[2] / g_val
    
    # Calculate WB Temperature shift (in Kelvin) and Tint shift
    temp_shift = round((1.0 - r_ratio) * 2000.0, 1)
    tint_shift = round((1.0 - (g_val / ((w_samp[0] + w_samp[2]) * 0.5 + 1e-4))) * 50.0, 1)

    # Calculate RGB Curves (cubic spline control points normalized 0.0 - 1.0)
    # Control points at x = 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    curve_r = [[0.0, 0.0], [0.2, 0.2], [0.4, 0.4], [0.6, 0.6], [0.8, 0.8], [1.0, 1.0]]
    curve_g = [[0.0, 0.0], [0.2, 0.2], [0.4, 0.4], [0.6, 0.6], [0.8, 0.8], [1.0, 1.0]]
    curve_b = [[0.0, 0.0], [0.2, 0.2], [0.4, 0.4], [0.6, 0.6], [0.8, 0.8], [1.0, 1.0]]

    for i in range(6):
        x = float(gray_sampled[5 - i, 1] / 255.0)  # Green channel as brightness reference
        target = float(gray_ref[5 - i, 0] / 255.0)
        
        r_target = float(gray_ref[5 - i, 0] / 255.0)
        g_target = float(gray_ref[5 - i, 1] / 255.0)
        b_target = float(gray_ref[5 - i, 2] / 255.0)

        curve_r[i] = [round(x, 3), round(min(1.0, max(0.0, r_target)), 3)]
        curve_g[i] = [round(x, 3), round(min(1.0, max(0.0, g_target)), 3)]
        curve_b[i] = [round(x, 3), round(min(1.0, max(0.0, b_target)), 3)]

    # 2. HSL Color Patch Shifts (Patches 0-17)
    color_bands = ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"]
    hue_shifts = {b: 0.0 for b in color_bands}
    sat_shifts = {b: 0.0 for b in color_bands}
    lum_shifts = {b: 0.0 for b in color_bands}

    # Map color patches to HSL bands
    patch_band_map = {
        0: "Red", 1: "Orange", 2: "Blue", 3: "Green", 4: "Purple", 5: "Aqua",
        6: "Orange", 7: "Blue", 8: "Red", 9: "Purple", 10: "Yellow", 11: "Yellow",
        12: "Blue", 13: "Green", 14: "Red", 15: "Yellow", 16: "Magenta", 17: "Aqua"
    }

    for idx, band in patch_band_map.items():
        s_rgb = np.uint8([[sampled[idx]]])
        r_rgb = np.uint8([[COLORCHECKER_24_REF[idx]]])
        
        s_hsv = cv2.cvtColor(s_rgb, cv2.COLOR_RGB2HSV)[0, 0]
        r_hsv = cv2.cvtColor(r_rgb, cv2.COLOR_RGB2HSV)[0, 0]

        dh = float(r_hsv[0]) - float(s_hsv[0])
        ds = (float(r_hsv[1]) - float(s_hsv[1])) / 255.0 * 20.0
        dl = (float(r_hsv[2]) - float(s_hsv[2])) / 255.0 * 20.0

        hue_shifts[band] = round(hue_shifts[band] + dh * 0.5, 1)
        sat_shifts[band] = round(sat_shifts[band] + ds * 0.5, 1)
        lum_shifts[band] = round(lum_shifts[band] + dl * 0.5, 1)

    return {
        "temperature_shift": temp_shift,
        "tint_shift": tint_shift,
        "curve_r": curve_r,
        "curve_g": curve_g,
        "curve_b": curve_b,
        "hsl_hue": hue_shifts,
        "hsl_sat": sat_shifts,
        "hsl_lum": lum_shifts,
    }
