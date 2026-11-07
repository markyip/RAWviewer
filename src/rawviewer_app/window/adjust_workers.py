"""Background workers for the adjust panel preview, masks and export."""
from __future__ import annotations

import contextlib
import os
import sys
from typing import Any

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal

from core.config import _env_int

from . import lazy
from .lazy import np

lazy.register(sys.modules[__name__])

__all__ = [
    "_AdjustEditBaseSignals",
    "_AdjustMaskLoadSignals",
    "_AdjustMaskLoadWorker",
    "_AdjustPreviewSignals",
    "_AdjustPreviewWorker",
    "_AdjustCompareOriginalSignals",
    "_AdjustCompareOriginalWorker",
    "_AdjustEditBaseSignals",
    "_AdjustEditBaseWorker",
    "_AdjustExportSignals",
    "_AdjustExportWorker",
]


class _AdjustEditBaseSignals(QObject):
    finished = pyqtSignal(str, object, bool, str)  # file_path, ndarray|None, lens_profile_available, lens_profile_name


class _AdjustMaskLoadSignals(QObject):
    # generation, file_path, dodge/burn mask|None, heal mask|None
    finished = pyqtSignal(int, str, object, object)


class _AdjustMaskLoadWorker(QRunnable):
    """Decode the (often multi-MB PNG) dodge/burn + heal mask blobs off the
    UI thread -- base64 + cv2.imdecode at edit-base resolution is the slow
    part of opening the Adjust panel for a file with existing local edits,
    not the (cheap) XMP XML parse that already ran synchronously."""

    def __init__(
        self,
        generation: int,
        file_path: str,
        db_serial: str,
        heal_serial: str,
        signals: _AdjustMaskLoadSignals,
    ):
        super().__init__()
        self.generation = generation
        self.file_path = file_path
        self.db_serial = db_serial
        self.heal_serial = heal_serial
        self.signals = signals

    def run(self) -> None:
        db_mask = None
        heal_mask = None
        try:
            from raw_adjustments.dodge_burn import deserialize_mask as deserialize_db_mask

            db_mask = deserialize_db_mask(self.db_serial)
        except Exception:
            db_mask = None
        try:
            from raw_adjustments.spot_heal import deserialize_mask as deserialize_heal_mask

            heal_mask = deserialize_heal_mask(self.heal_serial)
        except Exception:
            heal_mask = None
        self.signals.finished.emit(self.generation, self.file_path, db_mask, heal_mask)


class _AdjustPreviewSignals(QObject):
    finished = pyqtSignal(int, str, object)  # generation, file_path, ndarray|None
    ai_progress = pyqtSignal(int, float)  # generation, AI denoise fraction 0..1


class _AdjustPreviewWorker(QRunnable):
    """Apply slider adjustments off the GUI thread during live preview."""

    def __init__(
        self,
        generation: int,
        file_path: str,
        base_rgb: Any,
        adj: dict,
        signals: _AdjustPreviewSignals,
        stage_cache: Any = None,
        *,
        preview_lite: bool = False,
        settle_fast: bool = False,
        ai_denoise: bool = False,
    ):
        super().__init__()
        self.generation = int(generation)
        self.file_path = file_path
        self.base_rgb = base_rgb
        self.adj = dict(adj or {})
        self.signals = signals
        self.stage_cache = stage_cache
        self.preview_lite = bool(preview_lite)
        self.settle_fast = bool(settle_fast)
        # Live AI-denoise preview (settle passes only): same full-resolution
        # pipeline as export, so the preview matches the saved file. Bypasses
        # the staged cache, which has no denoise stage.
        self.ai_denoise = bool(ai_denoise)

    def run(self) -> None:
        out = None
        try:
            import time as _time

            from core.perf_metrics import perf_mark

            t0 = _time.perf_counter()
            dtype_name = str(getattr(self.base_rgb, "dtype", ""))
            if self.ai_denoise and dtype_name in ("uint16", "float32"):
                from raw_adjustments.pipeline import linear_to_display_uint8, run_edit_pipeline

                gen = self.generation

                def _ai_progress(frac: float) -> None:
                    self.signals.ai_progress.emit(gen, float(frac))

                linear = run_edit_pipeline(
                    self.base_rgb,
                    self.adj,
                    target="preview",
                    use_ai_denoise=True,
                    progress_cb=_ai_progress,
                )
                out = linear_to_display_uint8(linear, self.adj)
            elif dtype_name in ("uint16", "float32") and self.stage_cache is not None:
                # Staged/cached recompute: reuses cached intermediate buffers
                # across ticks when only later-stage slider keys changed
                # since the last call (see PreviewStageCache in
                # raw_edit_pipeline.py) -- this is what makes dragging a
                # single slider (e.g. Sharpness) skip re-running WB/exposure/
                # denoise/PV2012 tone/tonemap on every tick.
                # preview_lite: cheaper PV2012 on the downsampled drag base;
                # settle_fast: full base with lighter chroma damp.
                from raw_adjustments.pipeline import render_adjust_preview_uint8

                out = render_adjust_preview_uint8(
                    self.base_rgb,
                    self.adj,
                    self.stage_cache,
                    preview_lite=self.preview_lite,
                    settle_fast=self.settle_fast,
                )
            else:
                from raw_adjustments import apply_adjustments_to_rgb

                out = apply_adjustments_to_rgb(self.base_rgb, self.adj)
                if out is not None and str(getattr(out, "dtype", "")) in {
                    "uint16",
                    "float32",
                    "float64",
                }:
                    import logging

                    logging.getLogger(__name__).warning(
                        "Adjust preview returned non-display buffer (%s); forcing encode",
                        getattr(out, "dtype", None),
                    )
                    from raw_adjustments import apply_adjustments_to_linear

                    out = apply_adjustments_to_linear(self.base_rgb, self.adj)
            if out is not None and str(getattr(out, "dtype", "")) != "uint8":
                from raw_adjustments.tone_recovery import _encode_srgb8

                out = _encode_srgb8(np.clip(out.astype(np.float32), 0.0, None))
            try:
                from raw_adjustments.dodge_burn import MASK_KEY as _DB_MASK_KEY

                has_db = 1 if str(self.adj.get(_DB_MASK_KEY, "") or "").strip() else 0
            except Exception:
                has_db = 0
            shape = getattr(self.base_rgb, "shape", None)
            h = int(shape[0]) if shape is not None and len(shape) >= 2 else 0
            w = int(shape[1]) if shape is not None and len(shape) >= 2 else 0
            perf_mark(
                "adjust_preview",
                (_time.perf_counter() - t0) * 1000.0,
                self.file_path,
                h=h,
                w=w,
                lite=1 if self.preview_lite else 0,
                db=has_db,
            )
        except Exception:
            import logging

            logging.getLogger(__name__).exception(
                "Adjust preview worker failed for %s",
                os.path.basename(self.file_path or "") or "?",
            )
            out = None
        self.signals.finished.emit(self.generation, self.file_path, out)


class _AdjustCompareOriginalSignals(QObject):
    finished = pyqtSignal(str, object)  # file_path, ndarray|None


class _AdjustCompareOriginalWorker(QRunnable):
    """Render the unedited look for compare-with-original (tone defaults).

    Geometry (crop/straighten/keystone) from the live edit is applied so the
    original side matches the edited frame's framing; IgnoreAspectRatio stretch
    is no longer required when sizes already agree.
    """

    def __init__(
        self,
        file_path: str,
        base_rgb: Any,
        signals: _AdjustCompareOriginalSignals,
        geometry_adj: dict | None = None,
    ):
        super().__init__()
        self.file_path = file_path
        self.base_rgb = base_rgb
        self.signals = signals
        self.geometry_adj = dict(geometry_adj or {})

    def run(self) -> None:
        out = None
        try:
            from raw_adjustments import DEFAULT_ADJUSTMENTS, apply_adjustments_to_rgb
            from raw_adjustments.transform import TRANSFORM_KEYS

            adj = dict(DEFAULT_ADJUSTMENTS)
            for key in TRANSFORM_KEYS:
                if key in self.geometry_adj:
                    adj[key] = self.geometry_adj[key]
            out = apply_adjustments_to_rgb(self.base_rgb, adj)
            if out is not None and str(getattr(out, "dtype", "")) != "uint8":
                from raw_adjustments.tone_recovery import _encode_srgb8

                out = _encode_srgb8(np.clip(out.astype(np.float32), 0.0, None))
        except Exception:
            out = None
        self.signals.finished.emit(self.file_path, out)


_ADJUST_FAST_PREVIEW_MAX_EDGE = 640


def _adjust_fast_preview_max_edge() -> int:
    return max(
        320,
        min(
            _env_int(
                "RAWVIEWER_ADJUST_FAST_PREVIEW_MAX_EDGE",
                _ADJUST_FAST_PREVIEW_MAX_EDGE,
                minimum=320,
            ),
            2000,
        ),
    )


def _make_fast_preview_base(base: Any) -> Any:
    """
    Downsample a decoded RAW edit base for the continuous live-drag preview
    (see PreviewStageCache_fast in __init__). Keeps the base's own dtype
    (uint16 stays uint16) so it flows through the exact same
    render_adjust_preview_uint8 code path as the full-res base -- just with
    far fewer pixels. Returns the original array unchanged if it's already
    at or under the target size, or on any failure (never block the normal
    full-quality path over this).
    """
    try:
        if base is None or not hasattr(base, "shape"):
            return base
        import cv2

        h, w = int(base.shape[0]), int(base.shape[1])
        max_dim = max(h, w)
        target = _adjust_fast_preview_max_edge()
        if max_dim <= target:
            return base
        scale = target / max_dim
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        return cv2.resize(base, (nw, nh), interpolation=cv2.INTER_AREA)
    except Exception:
        return base


class _AdjustEditBaseWorker(QRunnable):
    """Background LibRaw demosaic for the adjust panel (never embedded JPEG)."""

    def __init__(
        self,
        file_path: str,
        process_pool: Any,
        signals: _AdjustEditBaseSignals,
        *,
        use_full_resolution: bool = False,
        apply_lens_correction: bool = False,
    ):
        super().__init__()
        self.file_path = file_path
        self.process_pool = process_pool
        self.signals = signals
        self.use_full_resolution = use_full_resolution
        self.apply_lens_correction = apply_lens_correction

    def run(self) -> None:
        base = None
        has_lens_profile = False
        lens_profile_name = ""
        try:
            from processing.unified import UnifiedImageProcessor

            processor = UnifiedImageProcessor()
            base = processor.decode_raw_edit_base(
                self.file_path,
                executor=self.process_pool,
                use_full_resolution=self.use_full_resolution,
                apply_lens_correction=self.apply_lens_correction,
            )
            has_lens_profile = processor.lens_profile_available(self.file_path)
            if has_lens_profile:
                from raw_adjustments.lens_correction import get_lens_profile_name
                exif_data = processor.exif_extractor.extract_exif_data(self.file_path)
                lens_profile_name = get_lens_profile_name(exif_data)
        except Exception:
            base = None
        self.signals.finished.emit(self.file_path, base, has_lens_profile, lens_profile_name)


class _AdjustExportSignals(QObject):
    # source, output, format, error|None, xmp_warning|None
    finished = pyqtSignal(str, str, str, object, object)
    # source, percent (0-100), stage label
    progress = pyqtSignal(str, int, str)


class _AdjustExportWorker(QRunnable):
    """Full-res export: baked TIFF/JPEG/WebP/DNG or RAW copy + XMP settings."""

    def __init__(
        self,
        file_path: str,
        output_path: str,
        adj: dict,
        export_format: str,
        process_pool: Any,
        signals: _AdjustExportSignals,
        use_ai_denoise: bool = False,
    ):
        super().__init__()
        self.file_path = file_path
        self.output_path = output_path
        self.adj = dict(adj or {})
        self.export_format = export_format
        self.process_pool = process_pool
        self.signals = signals
        # Export-only, not persisted to the XMP sidecar.
        self.use_ai_denoise = bool(use_ai_denoise)
        # Set from the export progress dialog's Cancel button (main thread);
        # polled between export stages and between NN-denoise tiles.
        import threading as _threading

        self.cancel_event = _threading.Event()
        # Set in run()'s finally so the UI can tell the worker is done without
        # depending on the queued `finished` signal being delivered.
        self.done_event = _threading.Event()

    def run(self) -> None:
        from rawviewer_app.services.batch import export_one_file

        def emit_progress(percent: int, stage: str = "") -> None:
            with contextlib.suppress(Exception):
                self.signals.progress.emit(self.file_path, percent, stage)

        err, xmp_warning = None, None
        try:
            err, xmp_warning = export_one_file(
                self.file_path,
                self.output_path,
                self.adj,
                self.export_format,
                self.process_pool,
                cancel_event=self.cancel_event,
                progress_cb=emit_progress,
                use_ai_denoise=self.use_ai_denoise,
            )
        except Exception as exc:  # never lose the completion signal
            err = exc
        finally:
            self.done_event.set()
        self.signals.finished.emit(
            self.file_path, self.output_path, self.export_format, err, xmp_warning
        )
