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


def refine_alignment_flow(
    images: List[np.ndarray],
    ref_idx: int = 0,
    max_shift_frac: float = 0.03,
    progress_callback: ProgressCb = None,
) -> Tuple[List[np.ndarray], float]:
    """Non-rigid (per-pixel) refinement after the global affine align.

    A single affine cannot satisfy parallax -- near objects shift more than far
    ones between viewpoints, so the global solve leaves depth-dependent residual
    that makes the sharpness selector composite a shifted foreground edge from
    two positions (the classic doubled-edge ghost). Dense optical flow warps
    each frame's geometry onto the reference to absorb that.

    The flow is estimated on *heavily blurred* luma on purpose: focus stacks
    have regions sharp in one frame and soft in another, and raw flow would be
    dominated by that focus difference. Parallax is a low-frequency geometric
    offset, so blurring to low frequencies leaves the shared structure the flow
    should track and discards the focus mismatch. Vectors are clamped to
    ``max_shift_frac`` of the frame so textureless/out-of-focus areas cannot
    inject wild warps.

    Returns (refined_images, parallax_px) where parallax_px is the median flow
    magnitude across frames -- a cheap "how much parallax is present" signal for
    the caller's warning.
    """
    if len(images) < 2:
        return images, 0.0

    ref = images[ref_idx]
    ref_gray = _to_gray_u8(ref)
    h, w = ref_gray.shape
    sigma = max(2.0, min(h, w) / 200.0)
    ref_blur = cv2.GaussianBlur(ref_gray, (0, 0), sigma)
    max_shift = max_shift_frac * max(h, w)

    # Confidence: only warp where the REFERENCE has real structure to register
    # against. A focus stack's reference is sharp in some regions and soft in
    # others; warping a frame's SHARP region against a reference that is SOFT
    # there resamples good detail into mush (measured: it halved sharpness).
    # Raising the normalised contrast to a >1 power collapses weakly-structured
    # (blurred) regions toward zero confidence so their original pixels pass
    # through cleanly, while genuinely-detailed regions -- where parallax
    # actually shows -- keep near-full confidence. The 1.5 exponent balances
    # sharp-detail preservation on tripod stacks (~0.75 of a no-warp merge)
    # against parallax reduction on handheld ones (~0.5 residual); real stacks
    # will refine this.
    ref_struct = cv2.GaussianBlur(np.abs(cv2.Laplacian(ref_blur, cv2.CV_32F)), (0, 0), sigma)
    thresh = np.percentile(ref_struct, 78) + 1e-6
    confidence = np.clip(ref_struct / thresh, 0.0, 1.0) ** 1.5

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    refined = list(images)
    mags: List[float] = []
    for idx, img in enumerate(images):
        if idx == ref_idx:
            continue
        if progress_callback:
            progress_callback(40 + int(8 * idx / len(images)), f"Refining alignment {idx + 1}…")
        mov_blur = cv2.GaussianBlur(_to_gray_u8(img), (0, 0), sigma)
        flow = dis.calc(ref_blur, mov_blur, None)
        flow = np.clip(flow, -max_shift, max_shift)
        # Attenuate by confidence, then smooth so the warp stays continuous.
        flow[..., 0] = cv2.GaussianBlur(flow[..., 0] * confidence, (0, 0), sigma)
        flow[..., 1] = cv2.GaussianBlur(flow[..., 1] * confidence, (0, 0), sigma)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mags.append(float(np.median(mag[mag > 0.1])) if np.any(mag > 0.1) else 0.0)
        map_x = xx + flow[..., 0]
        map_y = yy + flow[..., 1]
        warped = cv2.remap(
            img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        # Only adopt the warped pixels where we trust the registration. remap's
        # sub-pixel resample softens detail even for tiny residual flow, so in
        # low-confidence (reference-blurred) regions keep the ORIGINAL pixels
        # untouched -- that is exactly where the sharp frame's data must survive.
        conf3 = confidence[..., None] if img.ndim == 3 else confidence
        blended = warped.astype(np.float32) * conf3 + img.astype(np.float32) * (1.0 - conf3)
        refined[idx] = blended.astype(img.dtype)

    return refined, (float(np.median(mags)) if mags else 0.0)


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


def _seam_aware_weights(
    sharp_maps: List[np.ndarray],
    guide_gray_u8: np.ndarray,
    radius: int = 24,
    eps: float = 1e-3,
) -> List[np.ndarray]:
    """Coherent per-source weights: which frame owns each region, seam-aware.

    Raw smoothed-sharpness weights switch source pixel-by-pixel, which under
    residual parallax picks a shifted edge from two frames at once. Instead take
    the per-pixel argmax to decide the sharpest source, then regularise each
    source's mask with a guided filter keyed on the reference luma: region
    boundaries snap to real image edges and small speckle is absorbed, so a
    contested (parallax) area is handed wholesale to one frame rather than
    interleaved. This is the same intent as a graph-cut seam, done with the
    edge-aware guided filter (robust and always available here).
    """
    stack = np.stack(sharp_maps, axis=0)  # (N, H, W)
    labels = np.argmax(stack, axis=0)
    weights: List[np.ndarray] = []
    for i in range(len(sharp_maps)):
        mask = (labels == i).astype(np.float32)
        smooth = cv2.ximgproc.guidedFilter(guide_gray_u8, mask, radius, eps)
        weights.append(np.clip(smooth, 0.0, 1.0))
    return weights


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


# Above this median residual flow (in pixels), the global affine left parallax
# the sharpness selector will ghost on; the caller surfaces it as a warning.
_PARALLAX_WARN_PX = 2.0


def focus_stack(
    images: List[np.ndarray],
    paths: Optional[List[str]] = None,
    align: bool = True,
    local_align: bool = True,
    seam_aware: bool = True,
    auto_crop: bool = True,
    progress_callback: ProgressCb = None,
) -> FocusStackResult:
    """Merge a focus bracket into one all-in-focus frame.

    ``images`` are same-scene frames at different focal planes (any of uint8 /
    uint16 / float32; the output matches the input bit depth family). Order does
    not matter -- selection is per-region by sharpness.

    ``local_align`` (v2) adds dense optical-flow refinement after the global
    affine to absorb parallax the affine cannot. ``seam_aware`` (v2) selects
    each region's owning frame coherently instead of per-pixel, so contested
    parallax areas are not composited from two frames at once.
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

        if local_align:
            images, parallax_px = refine_alignment_flow(images, progress_callback=progress_callback)
            if parallax_px > _PARALLAX_WARN_PX:
                warnings.append(
                    f"Significant parallax detected (~{parallax_px:.1f}px between frames). "
                    "Edges of near objects may still show softness; a tripod or focus rail "
                    "avoids this."
                )

        if progress_callback:
            progress_callback(50, "Measuring sharpness…")
        sharp_maps = [_sharpness_map(_to_gray_u8(im)) for im in images]
        floats = [_as_float32(im) for im in images]

        if seam_aware:
            weights = _seam_aware_weights(sharp_maps, _to_gray_u8(images[0]))
        else:
            weights = sharp_maps

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
