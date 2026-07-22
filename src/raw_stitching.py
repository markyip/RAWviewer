"""HDR and Panorama stitching engine for RAWviewer."""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Dict

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StitchResult:
    """Result of an HDR, Panorama, or HDR Panorama stitching operation."""
    success: bool
    image: Optional[np.ndarray] = None
    output_path: str = ""
    error_message: str = ""
    rejected_paths: List[str] = None

    def __post_init__(self):
        if self.rejected_paths is None:
            self.rejected_paths = []


def align_hdr_images(images: List[np.ndarray]) -> Tuple[List[np.ndarray], bool]:
    """Align handheld bracketed exposures using OpenCV Median Threshold Bitmap (MTB).
    
    Returns (aligned_images, success).
    """
    if not images or len(images) < 2:
        return images, True

    try:
        align_mtb = cv2.createAlignMTB(max_bits=4, exclude_range=4, cut=True)
        aligned = []
        align_mtb.process(images, aligned)
        return aligned, True
    except Exception as exc:
        logger.warning("MTB HDR alignment failed: %s", exc)
        return images, False


def crop_panorama_borders(img: np.ndarray) -> np.ndarray:
    """Crop out black/transparent border regions from a warped panorama to leave a clean rectangular frame."""
    if img is None or img.size == 0:
        return img
    try:
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY if img.shape[2] == 3 else cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img
        x, y, w, h = cv2.boundingRect(contours[0])
        cropped = img[y:y+h, x:x+w]
        
        # Inner bounding refinement for jagged warping edges
        sub_gray = gray[y:y+h, x:x+w]
        row_min = np.min(sub_gray, axis=1)
        col_min = np.min(sub_gray, axis=0)
        valid_rows = np.where(row_min > 0)[0]
        valid_cols = np.where(col_min > 0)[0]
        if len(valid_rows) > 0 and len(valid_cols) > 0:
            r_start, r_end = valid_rows[0], valid_rows[-1] + 1
            c_start, c_end = valid_cols[0], valid_cols[-1] + 1
            if (r_end - r_start) > h * 0.5 and (c_end - c_start) > w * 0.5:
                cropped = cropped[r_start:r_end, c_start:c_end]
        return cropped
    except Exception as exc:
        logger.warning("Auto-crop panorama borders failed: %s", exc)
        return img


def estimate_ev_offsets(images: List[np.ndarray]) -> List[float]:
    """Estimate relative EV offsets across a set of images based on mean luminance."""
    if not images:
        return []
    lum_list = []
    for img in images:
        if img.ndim == 3:
            lum = np.mean(0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2])
        else:
            lum = np.mean(img)
        lum_list.append(float(lum))
    
    mid_idx = len(lum_list) // 2
    ref_lum = max(1e-4, lum_list[mid_idx])
    
    ev_offsets = []
    for l in lum_list:
        ratio = max(1e-4, l) / ref_lum
        ev = math.log2(ratio)
        ev_offsets.append(round(ev, 1))
    return ev_offsets


def merge_hdr_exposure_fusion(
    images: List[np.ndarray],
    highlight_weight: float = 1.0,
    shadow_weight: float = 1.0,
    midtone_weight: float = 1.0,
    image_weights: Optional[List[float]] = None,
    contrast_weight: float = 1.0,
    saturation_weight: float = 1.0,
) -> np.ndarray:
    """Mertens Exposure Fusion with custom Highlight, Shadow, Midtone & Per-Image weights."""
    if not images:
        raise ValueError("No images provided for HDR merge")
    if len(images) == 1:
        return images[0].copy()

    is_uint8 = (images[0].dtype == np.uint8)
    norm_imgs = []
    for img in images:
        if img.dtype == np.uint8:
            norm_imgs.append(img.astype(np.float32) / 255.0)
        elif img.dtype == np.uint16:
            norm_imgs.append(img.astype(np.float32) / 65535.0)
        else:
            norm_imgs.append(np.clip(img.astype(np.float32), 0.0, 1.0))

    n = len(norm_imgs)
    if image_weights is None or len(image_weights) != n:
        image_weights = [1.0] * n

    weights_maps = []
    for idx, img in enumerate(norm_imgs):
        user_w = max(0.0, float(image_weights[idx]))
        if user_w < 1e-4:
            weights_maps.append(np.zeros((img.shape[0], img.shape[1]), dtype=np.float32))
            continue

        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        well_exp = np.exp(-0.5 * ((gray - 0.5) ** 2) / (0.2 ** 2)) ** midtone_weight
        hl_map = np.exp(-0.5 * ((gray - 0.15) ** 2) / (0.25 ** 2)) ** highlight_weight
        sh_map = np.exp(-0.5 * ((gray - 0.85) ** 2) / (0.25 ** 2)) ** shadow_weight

        exp_w = (well_exp + hl_map + sh_map) / 3.0

        if contrast_weight > 0.0:
            lap = np.abs(cv2.Laplacian((gray * 255).astype(np.uint8), cv2.CV_32F))
            contrast_w = (lap / (np.max(lap) + 1e-5)) ** contrast_weight
        else:
            contrast_w = 1.0

        if saturation_weight > 0.0:
            sat_w = np.std(img, axis=-1) ** saturation_weight
        else:
            sat_w = 1.0

        w_combined = (exp_w * contrast_w * sat_w * user_w) + 1e-12
        weights_maps.append(w_combined.astype(np.float32))

    sum_w = np.zeros_like(weights_maps[0])
    for w in weights_maps:
        sum_w += w
    sum_w = np.maximum(sum_w, 1e-12)

    for i in range(n):
        weights_maps[i] /= sum_w

    try:
        mertens = cv2.createMergeMertens(
            contrast_weight=float(contrast_weight),
            saturation_weight=float(saturation_weight),
            exposure_weight=float(midtone_weight),
        )
        uint8_inputs = [(img * 255.0).clip(0, 255).astype(np.uint8) for img in norm_imgs]
        res = mertens.process(uint8_inputs)
        res = np.clip(res, 0.0, 1.0)
    except Exception:
        res = np.zeros_like(norm_imgs[0])
        for i in range(n):
            w3 = np.expand_dims(weights_maps[i], axis=-1)
            res += norm_imgs[i] * w3
        res = np.clip(res, 0.0, 1.0)

    if is_uint8:
        return (res * 255.0).round().clip(0, 255).astype(np.uint8)
    return (res * 65535.0).round().clip(0, 65535).astype(np.uint16)


def stitch_panorama(
    images: List[np.ndarray],
    paths: Optional[List[str]] = None,
    auto_crop: bool = True,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> StitchResult:
    """Stitch images into a linear panorama with automatic rectangular border cropping."""
    if not images or len(images) < 2:
        return StitchResult(
            success=False,
            error_message="Panorama stitching requires at least 2 images."
        )

    n = len(images)
    paths = paths or [f"Image {i+1}" for i in range(n)]

    if progress_callback:
        progress_callback(10, "Detecting features & matching keypoints…")

    prep_imgs = []
    for img in images:
        if img.dtype == np.uint16:
            img_u8 = (img / 257.0).astype(np.uint8)
        elif img.dtype == np.float32:
            img_u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img_u8 = img.astype(np.uint8)
        prep_imgs.append(img_u8)

    try:
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

        if progress_callback:
            progress_callback(40, "Estimating camera poses & homography…")

        status, stitched = stitcher.stitch(prep_imgs)

        if status == cv2.Stitcher_OK and stitched is not None and stitched.size > 0:
            if auto_crop:
                if progress_callback:
                    progress_callback(85, "Auto-cropping warped black borders…")
                stitched = crop_panorama_borders(stitched)
            if progress_callback:
                progress_callback(95, "Finalizing panorama blend…")
            return StitchResult(
                success=True,
                image=stitched,
            )

        error_reasons = {
            cv2.Stitcher_ERR_NEED_MORE_IMGS: "Insufficient feature overlap between images (minimum 20% overlap required).",
            cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Camera homography estimation failed due to severe misalignment or parallax.",
            cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera pose parameter adjustment failed.",
        }
        err_detail = error_reasons.get(status, f"Stitching failed with status code {status}.")
        logger.warning("Stitcher failed for %d images: %s", n, err_detail)

        return StitchResult(
            success=False,
            error_message=f"Stitching failed: {err_detail}",
            rejected_paths=paths,
        )

    except Exception as exc:
        logger.exception("Panorama stitching exception: %s", exc)
        return StitchResult(
            success=False,
            error_message=f"Panorama stitching failed: {str(exc)}",
            rejected_paths=paths,
        )


def merge_hdr_panorama(
    images: List[np.ndarray],
    paths: List[str],
    weights: Optional[Dict[str, float]] = None,
    auto_crop: bool = True,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> StitchResult:
    """Combined Panorama HDR processing: fuses bracketed exposures then stitches panels."""
    if not images or len(images) < 4:
        return StitchResult(
            success=False,
            error_message="Panorama HDR requires at least 4 images (bracketed exposure pairs)."
        )

    weights = weights or {}
    hl_w = float(weights.get("highlight", 1.0))
    sh_w = float(weights.get("shadow", 1.0))
    mid_w = float(weights.get("midtone", 1.0))

    if progress_callback:
        progress_callback(10, "Aligning HDR exposure brackets…")

    n = len(images)
    group_size = 2 if n % 2 == 0 else 3

    panels = []
    panel_paths = []
    for i in range(0, n, group_size):
        chunk = images[i:i + group_size]
        chunk_paths = paths[i:i + group_size]
        if len(chunk) < 2:
            panels.append(chunk[0])
            panel_paths.append(chunk_paths[0])
            continue
        aligned, _ = align_hdr_images(chunk)
        hdr_panel = merge_hdr_exposure_fusion(
            aligned,
            highlight_weight=hl_w,
            shadow_weight=sh_w,
            midtone_weight=mid_w,
        )
        panels.append(hdr_panel)
        panel_paths.append(chunk_paths[0])

    if progress_callback:
        progress_callback(50, "Stitching HDR panels into panorama…")

    return stitch_panorama(
        panels,
        paths=panel_paths,
        auto_crop=auto_crop,
        progress_callback=progress_callback,
    )
