"""Camera color calibration engine using standard 24-patch ColorChecker targets and EXIF model registry."""

from __future__ import annotations

import contextlib
import json
import logging
import os
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

COLORCHECKER_24_REF = np.array(
    [
        [115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67],
        [133, 128, 177], [103, 189, 170], [214, 126, 44], [80, 91, 166],
        [193, 90, 99], [94, 60, 108], [157, 188, 64], [224, 163, 46],
        [56, 61, 150], [70, 148, 73], [175, 54, 60], [231, 199, 31],
        [187, 86, 149], [8, 133, 161], [243, 243, 242], [200, 200, 200],
        [160, 160, 160], [122, 122, 121], [85, 85, 85], [52, 52, 52],
    ],
    dtype=np.float32,
)


def get_camera_profile_path() -> str:
    app_dir = os.path.join(os.path.expanduser("~"), ".rawviewer")
    os.makedirs(app_dir, exist_ok=True)
    return os.path.join(app_dir, "camera_profiles.json")


def camera_identity_from_exif(exif: dict[str, Any] | None) -> tuple[str, str, int | None]:
    tags = exif or {}

    def _first(*names: str) -> str:
        for name in names:
            value = tags.get(name)
            if value:
                text = str(value).strip()
                if text:
                    return text
        return ""

    make = _first("Image Make", "Make", "EXIF Make")
    model = _first("Image Model", "Model", "EXIF Model")
    iso_val: int | None = None
    raw_iso = _first(
        "EXIF ISOSpeedRatings", "ISOSpeedRatings", "EXIF ISO", "ISO",
        "EXIF PhotographicSensitivity",
    )
    if raw_iso:
        try:
            digits = "".join(c for c in raw_iso.split(",")[0] if c.isdigit())
            if digits:
                iso_val = int(digits)
        except Exception:
            iso_val = None
    return make, model, iso_val


def camera_identity_for_file(path: str) -> tuple[str, str, int | None]:
    if not path:
        return ("", "", None)
    try:
        from metadata.backend import process_file_from_path

        return camera_identity_from_exif(process_file_from_path(path, details=False))
    except Exception:
        logger.debug("[CALIB] Could not read camera identity for %s", path, exc_info=True)
        return ("", "", None)


def normalize_camera_key(make: str, model: str, iso: int | None = None) -> str:
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
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


def current_decode_signature(file_path: str | None = None) -> dict[str, Any]:
    sig: dict[str, Any] = {"libraw": None, "wb_corrected": None}
    try:
        import rawpy

        version = getattr(rawpy, "libraw_version", None)
        if version is not None:
            sig["libraw"] = ".".join(str(int(v)) for v in version)
    except Exception:
        pass
    if file_path:
        try:
            from processing.raw_decode.fast import get_corrected_camera_wb, wb_sanity_enabled

            if wb_sanity_enabled():
                sig["wb_corrected"] = get_corrected_camera_wb(file_path) is not None
        except Exception:
            pass
    return sig


def _compare_signatures(saved: dict[str, Any], now: dict[str, Any]) -> str | None:
    saved_libraw, now_libraw = saved.get("libraw"), now.get("libraw")
    if saved_libraw and now_libraw and saved_libraw != now_libraw:
        return (
            f"Calibrated with LibRaw {saved_libraw}; this build uses "
            f"{now_libraw}. Re-shoot the chart if colour looks off."
        )
    saved_wb, now_wb = saved.get("wb_corrected"), now.get("wb_corrected")
    if saved_wb is not None and now_wb is not None and bool(saved_wb) != bool(now_wb):
        if saved_wb:
            return (
                "Calibrated while this camera's white balance was being "
                "auto-corrected; it no longer is. Re-shoot the chart."
            )
        return (
            "This camera's white balance is now being auto-corrected, which "
            "it was not when the chart was shot. Re-shoot the chart."
        )
    return None


def profile_decode_mismatch(
    profile: dict[str, Any] | None, file_path: str | None = None
) -> str | None:
    if not isinstance(profile, dict):
        return None
    saved = profile.get("decode_signature")
    if not isinstance(saved, dict):
        return None
    return _compare_signatures(saved, current_decode_signature(file_path))


def save_camera_profile(
    make: str, model: str, profile_data: dict[str, Any],
    iso: int | None = None, source_file: str | None = None,
) -> bool:
    key = normalize_camera_key(make, model, iso=iso)
    path = get_camera_profile_path()
    registry: dict[str, Any] = {}
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                registry = json.load(f)
        except Exception:
            registry = {}
    clean_data = _sanitize_for_json(profile_data)
    clean_data["make"] = make
    clean_data["model"] = model
    clean_data["iso"] = iso
    if "decode_signature" not in clean_data:
        clean_data["decode_signature"] = _sanitize_for_json(
            current_decode_signature(source_file)
        )
    registry[key] = clean_data
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
        return True
    except Exception as exc:
        logger.error("Failed to save camera profile: %s", exc)
        return False


def get_camera_profile(make: str, model: str, iso: int | None = None) -> dict[str, Any] | None:
    base_key = normalize_camera_key(make, model, iso=None)
    path = get_camera_profile_path()
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            registry = json.load(f)
        if iso is not None and iso > 0:
            exact_key = f"{base_key}_iso{iso}"
            if exact_key in registry:
                return registry[exact_key]
            iso_keys = [k for k in registry if k.startswith(f"{base_key}_iso")]
            iso_pairs = []
            for k in iso_keys:
                with contextlib.suppress(ValueError):
                    iso_pairs.append((abs(int(k.split("_iso")[-1]) - iso), k))
            if iso_pairs:
                iso_pairs.sort(key=lambda x: x[0])
                return registry[iso_pairs[0][1]]
        return registry.get(base_key)
    except Exception as exc:
        logger.warning("Error reading camera profile registry: %s", exc)
        return None


def delete_camera_profile(make: str, model: str, iso: int | None = None) -> bool:
    key = normalize_camera_key(make, model, iso=iso)
    path = get_camera_profile_path()
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            registry = json.load(f)
    except Exception:
        return False
    if key not in registry:
        return False
    registry.pop(key, None)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
        return True
    except Exception:
        return False


def describe_camera_profile(
    make: str, model: str, iso: int | None = None, file_path: str | None = None,
) -> str | None:
    prof = get_camera_profile(make, model, iso=iso)
    if not prof:
        return None
    name = f"{prof.get('make', make) or ''} {prof.get('model', model) or ''}".strip()
    prof_iso = prof.get("iso")
    label = f"{name or 'Camera'} (ISO {prof_iso})" if prof_iso else (name or "Camera")
    reason = profile_decode_mismatch(prof, file_path)
    return f"{label} — {reason}" if reason else label


def has_factory_color_profile(file_path: str) -> bool | None:
    try:
        import rawpy

        with rawpy.imread(file_path) as raw:
            mat = np.asarray(raw.rgb_xyz_matrix, dtype=np.float64)
    except Exception:
        return None
    if mat.size == 0:
        return None
    return bool(np.any(np.abs(mat) > 1e-6))


def extract_patch_colors(
    image: np.ndarray, corners: list[tuple[float, float]]
) -> list[tuple[float, float, float]]:
    if len(corners) != 4 or image is None or image.size == 0:
        return []
    pts_src = np.array(corners, dtype=np.float32)
    width, height = 600, 400
    pts_dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    sampled = []
    col_w, row_h = width / 6.0, height / 4.0
    for r in range(4):
        for c in range(6):
            x1, x2 = int(c * col_w + col_w * 0.25), int(c * col_w + col_w * 0.75)
            y1, y2 = int(r * row_h + row_h * 0.25), int(r * row_h + row_h * 0.75)
            mean_rgb = np.mean(warped[y1:y2, x1:x2], axis=(0, 1))
            sampled.append((float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])))
    return sampled


def validate_and_detect_color_checker(
    image: np.ndarray, corners: list[tuple[float, float]] | None = None
) -> tuple[bool, str, list[tuple[float, float, float]]]:
    if image is None or image.size == 0:
        return False, "Invalid or empty image buffer for color calibration.", []
    if not corners or len(corners) != 4:
        return False, "Please select the 4 corners bounding the ColorChecker chart.", []
    sampled = extract_patch_colors(image, corners)
    if not sampled or len(sampled) != 24:
        return False, "Could not sample 24 patches from the selected region.", []
    lums = [0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2] for p in sampled]
    if float(np.std(lums)) < 15.0:
        return False, "No ColorChecker chart detected in the selected area.", []
    gray_lums = lums[18:24]
    if not (gray_lums[0] > gray_lums[3] > gray_lums[5]):
        return False, "Chart validation failed: gray ramp does not match ColorChecker layout.", []
    return True, "", sampled


def _srgb_encode_0_255(rgb_0_255: np.ndarray) -> np.ndarray:
    lin = np.clip(np.asarray(rgb_0_255, dtype=np.float32) / 255.0, 0.0, 1.0)
    enc = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1.0 / 2.4) - 0.055)
    return (enc * 255.0).astype(np.float32)


def _srgb_decode_0_255(rgb_0_255: np.ndarray) -> np.ndarray:
    enc = np.clip(np.asarray(rgb_0_255, dtype=np.float32) / 255.0, 0.0, 1.0)
    lin = np.where(enc <= 0.04045, enc / 12.92, np.power((enc + 0.055) / 1.055, 2.4))
    return (lin * 255.0).astype(np.float32)


def _normalize_exposure(sampled: np.ndarray) -> np.ndarray:
    white_lin = float(np.mean(sampled[18]))
    ref_white_lin = float(np.mean(_srgb_decode_0_255(COLORCHECKER_24_REF[18])))
    if white_lin <= 1e-3:
        return sampled
    return sampled * (ref_white_lin / white_lin)


def calibrate_camera_curves_and_hsl(
    sampled_rgb: list[tuple[float, float, float]], wb_mode: str = "auto"
) -> dict[str, Any]:
    if len(sampled_rgb) != 24:
        raise ValueError(f"Expected 24 sampled RGB patches, got {len(sampled_rgb)}")
    sampled = np.array(sampled_rgb, dtype=np.float32)
    gray_sampled = sampled[18:24]
    if wb_mode == "white_19":
        samp_wb = gray_sampled[0]
    elif wb_mode == "neutral_22":
        samp_wb = gray_sampled[3]
    else:
        samp_wb = np.mean(gray_sampled, axis=0)
    g_val = max(1e-3, float(samp_wb[1]))
    r_ratio = float(samp_wb[0]) / g_val
    temp_shift = round((1.0 - r_ratio) * 2000.0, 1)
    tint_shift = round(
        (1.0 - (g_val / ((float(samp_wb[0]) + float(samp_wb[2])) * 0.5 + 1e-4))) * 50.0, 1
    )
    display = _srgb_encode_0_255(_normalize_exposure(sampled))
    color_bands = ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"]
    patch_band_map = {
        0: "Red", 1: "Orange", 2: "Blue", 3: "Green", 4: "Purple", 5: "Aqua",
        6: "Orange", 7: "Blue", 8: "Red", 9: "Purple", 10: "Yellow", 11: "Yellow",
        12: "Blue", 13: "Green", 14: "Red", 15: "Yellow", 16: "Magenta", 17: "Aqua",
    }
    acc: dict[str, list[tuple[float, float, float]]] = {b: [] for b in color_bands}
    for idx, band in patch_band_map.items():
        s_rgb = np.uint8([[np.clip(display[idx], 0, 255)]])
        r_rgb = np.uint8([[COLORCHECKER_24_REF[idx]]])
        s_hsv = cv2.cvtColor(s_rgb, cv2.COLOR_RGB2HSV)[0, 0]
        r_hsv = cv2.cvtColor(r_rgb, cv2.COLOR_RGB2HSV)[0, 0]
        dh = float(r_hsv[0]) - float(s_hsv[0])
        if dh > 90.0:
            dh -= 180.0
        elif dh < -90.0:
            dh += 180.0
        ds = (float(r_hsv[1]) - float(s_hsv[1])) / 255.0 * 20.0
        dl = (float(r_hsv[2]) - float(s_hsv[2])) / 255.0 * 20.0
        acc[band].append((dh, ds, dl))
    hue_shifts = dict.fromkeys(color_bands, 0.0)
    sat_shifts = dict.fromkeys(color_bands, 0.0)
    lum_shifts = dict.fromkeys(color_bands, 0.0)
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
