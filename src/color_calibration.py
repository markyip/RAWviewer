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


def normalize_camera_key(make: str, model: str) -> str:
    """Generate normalized lookup key from camera Make & Model."""
    raw = f"{make or ''}_{model or ''}".strip().lower()
    cleaned = "".join(c if c.isalnum() else "_" for c in raw)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "unknown_camera"


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


def save_camera_profile(make: str, model: str, profile_data: Dict[str, Any]) -> bool:
    """Save calibrated profile data for a specific camera model."""
    key = normalize_camera_key(make, model)
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
    registry[key] = clean_data

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
        logger.info("Saved camera calibration profile for %s %s -> key: %s", make, model, key)
        return True
    except Exception as exc:
        logger.error("Failed to save camera profile: %s", exc)
        return False


def get_camera_profile(make: str, model: str) -> Optional[Dict[str, Any]]:
    """Retrieve calibrated profile data for a camera model if available."""
    key = normalize_camera_key(make, model)
    path = get_camera_profile_path()
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            registry = json.load(f)
        return registry.get(key)
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
