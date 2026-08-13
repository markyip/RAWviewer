"""Live proxy preview and per-frame controls for the merge dialog.

The merge engines are fast enough to re-run interactively *if* the expensive
stage is not repeated. For a focus stack that stage is alignment (ECC, then
dense optical flow); for an HDR merge it is the MTB registration; for a
panorama it is the stitch itself. Each is prepared once here and cached, so
moving a slider or placing a focus point only re-runs fusion.

Everything is computed on a downscaled proxy of the sources, decoded through
the same path the real merge uses (``decode_raw_edit_base`` ->
``to_display_referred``) so what the panel shows is what the merge produces,
only smaller.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from PyQt6.QtCore import QObject, QRunnable, Qt, QThreadPool, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy

logger = logging.getLogger(__name__)

# Long edge of the proxy the panel tunes on. Big enough to judge a seam or a
# focus transition, small enough that a fusion is tens of milliseconds.
PROXY_MAX_EDGE = 900


def realign_tuning_to_decoded(
    source_paths: List[str],
    valid_paths: List[str],
    image_weights: Optional[List[float]],
    focus_points: Optional[List[tuple]],
) -> Tuple[Optional[List[float]], Optional[List[tuple]]]:
    """Keep per-frame weights and focus points aligned after a decode skip.

    The merge worker drops frames that fail to decode, then used to pass the
    original-length ``image_weights`` / original frame indices into engines
    that require one entry per surviving image (otherwise HDR falls back to
    1.0 and focus points aim at the wrong frame).
    """
    if list(valid_paths) == list(source_paths):
        if image_weights is not None and len(image_weights) != len(valid_paths):
            return None, focus_points
        return image_weights, focus_points

    index_of = {p: i for i, p in enumerate(source_paths)}
    new_of = {p: i for i, p in enumerate(valid_paths)}

    new_weights: Optional[List[float]]
    if image_weights is None:
        new_weights = None
    elif len(image_weights) == len(source_paths):
        new_weights = [float(image_weights[index_of[p]]) for p in valid_paths]
    elif len(image_weights) == len(valid_paths):
        new_weights = list(image_weights)
    else:
        new_weights = None

    if not focus_points:
        return new_weights, focus_points

    remapped: List[tuple] = []
    for pt in focus_points:
        x, y, r, idx = pt[0], pt[1], pt[2], int(pt[3])
        if 0 <= idx < len(source_paths):
            ni = new_of.get(source_paths[idx])
            if ni is not None:
                remapped.append((x, y, r, ni, *pt[4:]))
    return new_weights, remapped

# Distinct hues for the source map -- "which photo is this region coming from".
_SOURCE_COLORS = [
    (217, 105, 30), (58, 140, 201), (94, 176, 92), (198, 87, 148),
    (204, 170, 54), (120, 108, 196), (86, 180, 176), (200, 96, 82),
]


def source_map_rgb(labels: np.ndarray, colors: List[Tuple[int, int, int]]) -> np.ndarray:
    """Colourise a per-pixel frame-index map for display.

    ``colors`` is indexed the way ``labels`` is -- by position in the *fused*
    set, not by position in the dialog's photo list. The caller passes the
    swatch colour of each ticked row so the map and the list agree even when
    photos in the middle of the set are switched off.
    """
    out = np.zeros(labels.shape + (3,), dtype=np.uint8)
    for i, color in enumerate(colors):
        out[labels == i] = color
    return out


def to_pixmap(img: np.ndarray) -> QPixmap:
    """Display-referred array -> QPixmap, at whatever depth it arrives."""
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    # QImage does not own the buffer: copy() before the ndarray goes out of
    # scope, or the pixmap renders freed memory.
    qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


class PreviewLabel(QLabel):
    """Preview surface that reports clicks in normalised image coordinates.

    Normalised because the click is placed on this proxy but applied to the
    full-resolution merge, which is a different pixel grid.
    """

    clicked = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(420, 320)
        self._pix: Optional[QPixmap] = None
        self._draw_rect = None

    def set_image(self, pix: Optional[QPixmap]) -> None:
        self._pix = pix
        self._rescale()

    def resizeEvent(self, event):  # noqa: N802 (Qt naming)
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pix is None or self._pix.isNull():
            self.setPixmap(QPixmap())
            self._draw_rect = None
            return
        scaled = self._pix.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)
        x0 = (self.width() - scaled.width()) // 2
        y0 = (self.height() - scaled.height()) // 2
        self._draw_rect = (x0, y0, scaled.width(), scaled.height())

    def mousePressEvent(self, event):  # noqa: N802 (Qt naming)
        if self._draw_rect is not None and event.button() == Qt.MouseButton.LeftButton:
            x0, y0, w, h = self._draw_rect
            px, py = event.position().x() - x0, event.position().y() - y0
            if 0 <= px < w and 0 <= py < h and w > 0 and h > 0:
                self.clicked.emit(px / w, py / h)
        super().mousePressEvent(event)


class TuningSignals(QObject):
    """Worker -> GUI thread. Kept on the dialog so it outlives the runnables."""

    proxies_ready = pyqtSignal(object, object)   # images, paths
    preview_ready = pyqtSignal(int, object, object)  # generation, rgb, labels
    failed = pyqtSignal(int, str)


def safe_emit(owner: QObject, signal_name: str, *args) -> None:
    """Emit unless the receiving object is already gone.

    A proxy decode or a preview render can outlive the dialog -- close it
    mid-render and the worker finishes into a QObject whose C++ side Qt has
    destroyed, which raises RuntimeError inside a thread-pool task where
    nothing is waiting to catch it.

    Takes the object and the signal's NAME rather than the bound signal:
    looking the attribute up on a deleted wrapper raises the same RuntimeError,
    so the lookup has to happen inside the guard too.
    """
    try:
        getattr(owner, signal_name).emit(*args)
    except RuntimeError:
        logger.debug("Merge tuning signal dropped: receiver already destroyed")


class _Task(QRunnable):
    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self):
        try:
            self._fn()
        except Exception:
            logger.exception("Merge tuning task failed")


def run_async(fn) -> None:
    QThreadPool.globalInstance().start(_Task(fn))


def load_proxies(paths: List[str]) -> Tuple[List[np.ndarray], List[str]]:
    """Decode every source small, in the merge's own colour domain.

    Uses the half-size edit base, which is LRU-cached, so opening the panel on
    a set the user has been browsing is close to free.
    """
    import cv2

    from raw_stitching import to_display_referred
    from unified_image_processor import UnifiedImageProcessor

    processor = UnifiedImageProcessor()
    out, kept = [], []
    for p in paths:
        base = processor.decode_raw_edit_base(p, use_full_resolution=False)
        if base is None:
            continue
        img = to_display_referred(base)
        del base
        h, w = img.shape[:2]
        scale = min(1.0, PROXY_MAX_EDGE / float(max(h, w)))
        if scale < 1.0:
            img = cv2.resize(
                img, (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        # 8-bit for the proxy: it is display-referred already, the panel only
        # has to look right, and this keeps a 30-frame stack in a few hundred MB.
        out.append((img >> 8).astype(np.uint8) if img.dtype == np.uint16 else img)
        kept.append(p)
    return out, kept


class MergePreviewEngine:
    """Caches the expensive per-mode stage and re-runs only what tuning changes.

    Not a QObject: it is called from worker threads and touches no widgets.
    """

    def __init__(self, mode: str):
        self.mode = mode
        self.images: List[np.ndarray] = []
        self._focus_prep = None
        self._focus_key: Optional[tuple] = None
        self._hdr_aligned: Dict[tuple, List[np.ndarray]] = {}
        self._pano_raw: Optional[np.ndarray] = None
        self._pano_key: Optional[tuple] = None

    def set_images(self, images: List[np.ndarray]) -> None:
        self.images = images
        self.invalidate()

    def invalidate(self) -> None:
        """Drop the prepared stages -- alignment options changed."""
        self._focus_prep = None
        self._focus_key = None
        self._hdr_aligned.clear()
        self._pano_raw = None
        self._pano_key = None

    def _active(self, active: List[int]) -> List[np.ndarray]:
        return [self.images[i] for i in active if 0 <= i < len(self.images)]

    def render(self, opts: Dict[str, Any]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """(rgb preview, per-pixel source labels or None) for these settings."""
        if self.mode == "focus_stack":
            return self._render_focus(opts)
        if self.mode == "panorama":
            return self._render_panorama(opts), None
        return self._render_hdr(opts)

    def _render_focus(self, opts):
        from focus_stacking import (
            fuse_prepared_stack,
            prepare_focus_stack,
            source_label_map,
        )

        active = opts["active"]
        key = (tuple(active), opts["align"], opts["parallax"], opts["auto_crop"])
        if self._focus_key != key:
            self._focus_prep = prepare_focus_stack(
                self._active(active),
                align=opts["align"],
                local_align=opts["parallax"],
                auto_crop=opts["auto_crop"],
            )
            self._focus_key = key
        prep = self._focus_prep
        weights = opts.get("frame_weights")
        points = opts.get("focus_points")
        fused = fuse_prepared_stack(
            prep, auto_crop=opts["auto_crop"], frame_weights=weights, focus_points=points
        )
        labels = source_label_map(
            prep,
            frame_weights=weights,
            focus_points=points,
            auto_crop=opts["auto_crop"],
        )
        return fused, labels

    def _render_hdr(self, opts):
        from raw_stitching import align_hdr_images, merge_hdr_exposure_fusion

        active = opts["active"]
        frames = self._active(active)
        weights = opts.get("frame_weights")
        if self.mode == "hdr_panorama":
            # One panel at a time: the stitch is seconds, the fusion is
            # milliseconds, and the exposure controls only affect the fusion.
            g = max(2, int(opts.get("group_size") or 2))
            start = int(opts.get("preview_panel", 0)) * g
            chunk = frames[start:start + g]
            if chunk:
                frames = chunk
            else:
                frames = frames[:g]
                start = 0
            if weights is not None and len(weights) >= start + len(frames):
                weights = weights[start:start + len(frames)]
            elif weights is not None and len(weights) != len(frames):
                weights = None
        key = (tuple(active), opts["align"], self.mode, opts.get("preview_panel"),
               opts.get("group_size"))
        if opts["align"]:
            cached = self._hdr_aligned.get(key)
            if cached is None:
                cached, _ok = align_hdr_images(frames)
                self._hdr_aligned[key] = cached
            frames = cached
        fused = merge_hdr_exposure_fusion(
            frames,
            highlight_weight=opts["highlight"],
            shadow_weight=opts["shadow"],
            midtone_weight=opts["midtone"],
            image_weights=weights,
        )
        return fused, None

    def _render_panorama(self, opts):
        from raw_stitching import crop_panorama_borders, stitch_panorama

        active = opts["active"]
        key = tuple(active)
        if self._pano_key != key:
            res = stitch_panorama(self._active(active), auto_crop=False)
            if not res.success:
                raise RuntimeError(res.error_message)
            self._pano_raw = res.image
            self._pano_key = key
        # Re-cropping the cached stitch is instant, so the crop toggle is live.
        return crop_panorama_borders(self._pano_raw) if opts["auto_crop"] else self._pano_raw
