"""Focus stacking engine for RAWviewer.

Combines a set of photos of the same scene shot at different focal planes into a
single frame that is sharp everywhere, by picking the sharpest source per region
and blending the seams away.

Pipeline (mirrors raw_stitching.py's shape so the host worker/dialog reuse the
same StitchResult contract):

    decode (host)  -> align (ECC affine)  -> per-image sharpness map
                   -> Laplacian-pyramid fusion weighted by sharpness
                   -> optional auto-crop of the alignment border

Alignment note: focus stacks share exposure but NOT magnification -- as focus
shifts, the lens breathes (scale/perspective drift), so the MTB translation-only
aligner used for HDR brackets (raw_stitching.align_hdr_images) is wrong here.
We use findTransformECC with an affine model, which absorbs the small scale and
shift breathing introduces. Feature-based homography is the v2 path for heavier
handheld/parallax cases; see FocusStackResult.alignment_warnings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ProgressCb = Optional[Callable[[int, str], None]]


@dataclass
class FocusStackResult:
    """Result of a focus-stacking merge (shares StitchResult's success/image/
    error_message contract so the host worker can treat them interchangeably)."""

    success: bool
    image: Optional[np.ndarray] = None
    error_message: str = ""
    rejected_paths: List[str] = field(default_factory=list)
    alignment_warnings: List[str] = field(default_factory=list)


def _to_gray_u8(img: np.ndarray) -> np.ndarray:
    """Luma as uint8, whatever the source dtype (uint8/uint16/float)."""
    if img.dtype == np.uint8:
        base = img
    elif img.dtype == np.uint16:
        base = (img.astype(np.float32) / 257.0).astype(np.uint8)
    else:
        base = (np.clip(img.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
    if base.ndim == 3:
        return cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
    return base


def _as_float32(img: np.ndarray) -> np.ndarray:
    """Normalise any supported dtype to float32 in [0, 1] for blending."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    return np.clip(img.astype(np.float32), 0.0, 1.0)


def align_focus_stack(
    images: List[np.ndarray],
    progress_callback: ProgressCb = None,
) -> Tuple[List[np.ndarray], List[str]]:
    """Align every frame to the first using an ECC affine model.

    Returns (aligned_images, warnings). Frames that fail to converge are passed
    through unaligned with a warning rather than dropped -- a slightly
    misaligned frame still contributes sharp detail where it agrees, and losing
    a frame silently would leave an unexplained soft band in the result.
    """
    if len(images) < 2:
        return images, []

    ref_gray = _to_gray_u8(images[0]).astype(np.float32)
    h, w = ref_gray.shape
    # Work on a downscaled luma for the ECC solve (fast, and breathing is a
    # global transform so it is well-estimated at low res), then apply the
    # full-res warp.
    scale = min(1.0, 1024.0 / max(h, w))
    ref_small = cv2.resize(ref_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else ref_gray

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)
    aligned = [images[0]]
    warnings: List[str] = []

    for idx in range(1, len(images)):
        if progress_callback:
            progress_callback(10 + int(30 * idx / len(images)), f"Aligning frame {idx + 1}…")
        mov = images[idx]
        mov_gray = _to_gray_u8(mov).astype(np.float32)
        mov_small = (
            cv2.resize(mov_gray, (ref_small.shape[1], ref_small.shape[0]), interpolation=cv2.INTER_AREA)
            if scale < 1.0
            else mov_gray
        )
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(
                ref_small, mov_small, warp, cv2.MOTION_AFFINE, criteria, None, 5
            )
            # Rescale the translation terms from the downscaled solve to full res.
            if scale < 1.0:
                warp[0, 2] /= scale
                warp[1, 2] /= scale
            aligned_img = cv2.warpAffine(
                mov, warp, (w, h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT,
            )
            aligned.append(aligned_img)
        except cv2.error as exc:
            logger.warning("ECC align failed for frame %d: %s", idx, exc)
            warnings.append(f"Frame {idx + 1} could not be aligned; used as-is.")
            aligned.append(mov)

    return aligned, warnings


def _sharpness_map(gray_u8: np.ndarray, blur: int = 5) -> np.ndarray:
    """Local focus measure: |Laplacian|, smoothed so selection is region-wise.

    Response to noise is suppressed by averaging the absolute Laplacian over a
    small window (a cheap stand-in for a variance-of-Laplacian window), which
    keeps the weight map coherent instead of speckled per pixel.
    """
    lap = cv2.Laplacian(gray_u8, cv2.CV_32F, ksize=3)
    sharp = np.abs(lap)
    k = blur | 1  # odd
    return cv2.GaussianBlur(sharp, (k, k), 0)


def _fuse_pyramid(
    images: List[np.ndarray],
    weights: List[np.ndarray],
    levels: int = 5,
) -> np.ndarray:
    """Blend float32 images by per-pixel weight through a Laplacian pyramid.

    A single-level (hard argmax) selection leaves visible seams where the
    sharpest source switches; blending the Laplacian bands under a Gaussian
    pyramid of the weights carries fine detail from the sharp frame while
    cross-fading low frequencies, which hides the transitions.
    """
    # Normalise weights to sum to 1 per pixel.
    wsum = np.zeros(weights[0].shape, dtype=np.float32)
    for wmap in weights:
        wsum += wmap
    wsum = np.maximum(wsum, 1e-6)
    norm_w = [w / wsum for w in weights]

    h, w = images[0].shape[:2]
    max_levels = max(1, int(np.floor(np.log2(max(1, min(h, w))))) - 1)
    levels = max(1, min(levels, max_levels))

    def gauss_pyr(x: np.ndarray) -> List[np.ndarray]:
        pyr = [x]
        for _ in range(levels):
            pyr.append(cv2.pyrDown(pyr[-1]))
        return pyr

    def lap_pyr(x: np.ndarray) -> List[np.ndarray]:
        gp = gauss_pyr(x)
        lp = []
        for i in range(levels):
            up = cv2.pyrUp(gp[i + 1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
            lp.append(gp[i] - up)
        lp.append(gp[-1])
        return lp

    # Accumulate blended Laplacian bands across all sources.
    blended: Optional[List[np.ndarray]] = None
    for img, wmap in zip(images, norm_w):
        lp = lap_pyr(img)
        wp = gauss_pyr(wmap)
        contrib = []
        for band, wband in zip(lp, wp):
            wexp = wband[..., None] if band.ndim == 3 else wband
            contrib.append(band * wexp)
        if blended is None:
            blended = contrib
        else:
            blended = [b + c for b, c in zip(blended, contrib)]

    # Collapse the pyramid.
    out = blended[-1]
    for i in range(len(blended) - 2, -1, -1):
        out = cv2.pyrUp(out, dstsize=(blended[i].shape[1], blended[i].shape[0])) + blended[i]
    return np.clip(out, 0.0, 1.0)


def _auto_crop_border(img: np.ndarray, aligned_masks: List[np.ndarray]) -> np.ndarray:
    """Trim to the region every aligned frame covers (no reflect-padding edges)."""
    valid = np.ones(img.shape[:2], dtype=bool)
    for m in aligned_masks:
        valid &= m
    ys, xs = np.where(valid)
    if ys.size == 0:
        return img
    return img[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def focus_stack(
    images: List[np.ndarray],
    paths: Optional[List[str]] = None,
    align: bool = True,
    auto_crop: bool = True,
    progress_callback: ProgressCb = None,
) -> FocusStackResult:
    """Merge a focus bracket into one all-in-focus frame.

    ``images`` are same-scene frames at different focal planes (any of uint8 /
    uint16 / float32; the output matches the input bit depth family). Order does
    not matter -- selection is per-region by sharpness.
    """
    if not images or len(images) < 2:
        return FocusStackResult(False, error_message="Focus stacking needs at least 2 photos.")

    shapes = {im.shape[:2] for im in images}
    if len(shapes) != 1:
        return FocusStackResult(
            False,
            error_message="Selected photos are not the same dimensions; focus stacking needs identical frames.",
        )

    warnings: List[str] = []
    try:
        if align:
            images, warnings = align_focus_stack(images, progress_callback)

        if progress_callback:
            progress_callback(50, "Measuring sharpness…")
        weights = [_sharpness_map(_to_gray_u8(im)) for im in images]
        floats = [_as_float32(im) for im in images]

        if progress_callback:
            progress_callback(70, "Blending sharpest regions…")
        fused = _fuse_pyramid(floats, weights)

        if auto_crop:
            if progress_callback:
                progress_callback(92, "Cropping alignment border…")
            # A frame's valid region is where the warp did not pull in reflected
            # edge pixels; approximate with a full mask for the reference and
            # non-black luma for warped frames.
            masks = [np.ones(fused.shape[:2], dtype=bool)]
            for im in images[1:]:
                masks.append(_to_gray_u8(im) > 0)
            fused = _auto_crop_border(fused, masks)

        # Return in the input's bit-depth family.
        ref = images[0]
        if ref.dtype == np.uint16:
            out = (fused * 65535.0).astype(np.uint16)
        elif ref.dtype == np.uint8:
            out = (fused * 255.0).astype(np.uint8)
        else:
            out = fused.astype(np.float32)

        if progress_callback:
            progress_callback(100, "Focus stack complete")
        return FocusStackResult(True, image=out, alignment_warnings=warnings)
    except Exception as exc:
        logger.exception("Focus stacking failed")
        return FocusStackResult(False, error_message=f"Focus stacking failed: {exc}", alignment_warnings=warnings)
