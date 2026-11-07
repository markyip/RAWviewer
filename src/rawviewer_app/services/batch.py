"""Off-UI-thread batch file operations: copy, trash, discard, edited export.

Workers receive an immutable snapshot of paths, run serially, and report
coalesced progress via Qt signals. Cancel is polled between files (and between
export stages inside a single file), so a cancel lands quickly. Nothing here
touches caches or QPixmaps — file-system spread only.
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal

logger = logging.getLogger(__name__)


def unique_destination(dest_dir: str, filename: str, reserved: set) -> str:
    """Never overwrite: pick filename, then _1/_2/… suffixes inside *reserved*."""
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(dest_dir, filename)
    n = 1
    while candidate in reserved or os.path.lexists(candidate):
        candidate = os.path.join(dest_dir, f"{base}_{n}{ext}")
        n += 1
    reserved.add(candidate)
    return candidate


# Export format → source file extensions that need no re-encode if the
# source is already in that format and carries no real edits.
_BYPASS_EXT = {
    "jpeg": frozenset({".jpg", ".jpeg"}),
    "webp": frozenset({".webp"}),
    "tiff16": frozenset({".tif", ".tiff"}),
}


def _should_skip_reencode(
    source_path: str,
    export_format: str,
    adj: dict | None,
    use_ai_denoise: bool,
) -> bool:
    """True when the source can be copied verbatim instead of decoded + re-encoded.

    Conditions: source extension already matches the target format, no AI
    denoise requested, and adjustments are effectively default (no real edits).
    Re-encoding a JPEG/WebP would generational-loss degrade it for no reason.
    """
    if use_ai_denoise:
        return False
    ext = os.path.splitext(source_path)[1].lower()
    if ext not in _BYPASS_EXT.get(export_format, frozenset()):
        return False
    from raw_adjustments import is_default_adjustments

    return is_default_adjustments(adj)


def copy_one_file(
    src: str,
    dest_dir: str,
    reserved: set,
    *,
    copy_sidecar: bool = True,
) -> str:
    """Copy a file preserving mtime (+ its XMP sidecar). Returns the dest path.

    Raises on failure; a failed sidecar copy is non-fatal (logged, skipped) —
    the image itself matters more than its saved edits.
    """
    from raw_adjustments import resolve_xmp_path

    dest = unique_destination(dest_dir, os.path.basename(src), reserved)
    shutil.copy2(src, dest, follow_symlinks=False)
    if copy_sidecar:
        try:
            xmp = resolve_xmp_path(src)
            if xmp and os.path.isfile(xmp):
                shutil.copy2(
                    xmp,
                    unique_destination(dest_dir, os.path.basename(xmp), reserved),
                    follow_symlinks=False,
                )
        except Exception as exc:
            logger.warning(
                "[BATCH] Sidecar copy skipped for %s: %s", os.path.basename(src), exc
            )
    return dest


def discard_one_file(src: str) -> str:
    """Move into '<folder>/Discard/<name>_discarded_N.<ext>' (same naming rule
    the single-image discard has always used). Returns the dest path."""
    folder_path = os.path.dirname(src)
    discard_folder = os.path.join(folder_path, "Discard")
    os.makedirs(discard_folder, exist_ok=True)
    filename = os.path.basename(src)
    base, ext = os.path.splitext(filename)
    target_path = os.path.join(discard_folder, filename)
    counter = 1
    while os.path.lexists(target_path):
        target_path = os.path.join(
            discard_folder, f"{base}_discarded_{counter}{ext}"
        )
        counter += 1
    os.rename(src, target_path)
    return target_path


class BatchFileOpSignals(QObject):
    # done (0-based completed count), total, current file basename
    progress = pyqtSignal(int, int, str)
    # completed, failed, cancelled, detail: list[(src, err|None, dest|None)]
    finished = pyqtSignal(int, int, bool, object)


class BatchFileOpWorker(QRunnable):
    """Serial copy/trash/discard over a snapshot of paths."""

    def __init__(
        self,
        paths: list[str],
        dest_dir: str = "",
        mode: str = "copy",
        signals: BatchFileOpSignals | None = None,
        cancel_event: threading.Event | None = None,
    ):
        super().__init__()
        self.paths = list(paths)
        self.dest_dir = dest_dir
        self.signals = signals or BatchFileOpSignals()
        self.mode = mode
        self.cancel_event = cancel_event or threading.Event()
        # Completion flag readable from the UI thread; set in run()'s finally
        # so a lost `finished` signal can never wedge the caller's gate.
        self.done_event = threading.Event()

    def run(self) -> None:
        from send2trash import send2trash

        total = len(self.paths)
        ok = 0
        failed = 0
        want_cancel = False
        detail: list[tuple[str, str | None, str | None]] = []
        reserved: set = set()
        try:
            for i, src in enumerate(self.paths):
                if self.cancel_event.is_set():
                    break
                name = os.path.basename(src)
                try:
                    self.signals.progress.emit(i, total, name)
                    if self.mode == "copy":
                        dest = copy_one_file(src, self.dest_dir, reserved)
                        detail.append((src, None, dest))
                    elif self.mode == "trash":
                        send2trash(os.path.normpath(src))
                        detail.append((src, None, None))
                    elif self.mode == "discard":
                        dest = discard_one_file(src)
                        detail.append((src, None, dest))
                    else:  # pragma: no cover - programming error
                        raise ValueError(f"unknown mode {self.mode!r}")
                    ok += 1
                except Exception as exc:  # keep going; per-file failure only
                    failed += 1
                    logger.error("[BATCH] %s failed on %s: %s", self.mode, src, exc)
                    detail.append((src, str(exc) or repr(exc), None))
                # note: progress of the *next* item is emitted at loop start
                self.signals.progress.emit(i + 1, total, name)
            want_cancel = self.cancel_event.is_set()
        except Exception as exc:  # worker-level failure (e.g. progress emit)
            logger.error("[BATCH] %s worker crashed: %s", self.mode, exc, exc_info=True)
            failed += 1
            want_cancel = True
            detail.append(("", f"worker crashed: {exc}", None))
        finally:
            self.done_event.set()
            self.signals.finished.emit(ok, failed, want_cancel, list(detail))


def export_one_file(
    file_path: str,
    output_path: str,
    adj: dict,
    export_format: str,
    process_pool: Any,
    *,
    cancel_event: threading.Event | None = None,
    progress_cb: Callable | None = None,  # (percent:int, stage:str)
    use_ai_denoise: bool = False,
    ai_amount: float | None = None,
) -> tuple[Exception | None, Exception | None]:
    """Full-res single-file export.

    Body moved verbatim from _AdjustExportWorker.run(): syncs the source XMP
    sidecar, decodes the full-resolution edit base, then renders/encodes via
    raw_edit_pipeline.export_adjusted_image. Returns (err, xmp_warning); a
    cancelled export raises ExportCancelled inside err and removes any partial
    output file.
    """
    logger = logging.getLogger(__name__)
    logger.info(
        "[EXPORT] export_one_file: format=%s source=%s output=%s",
        export_format,
        os.path.basename(file_path),
        output_path,
    )
    # Skip decode + re-encode when the source is already in the target format
    # and carries no real edits -- copy it verbatim to avoid generational loss.
    if _should_skip_reencode(file_path, export_format, adj, use_ai_denoise):
        try:
            shutil.copy2(file_path, output_path)
            logger.info(
                "[EXPORT] copied without re-encode (already %s, no edits): %s",
                export_format,
                os.path.basename(file_path),
            )
        except Exception as exc:
            return exc, None
        return None, None
    err = None
    xmp_warning = None
    cancelled = cancel_event if cancel_event is not None else threading.Event()
    try:
        from processing.unified import UnifiedImageProcessor
        from raw_adjustments import resolve_xmp_path, write_xmp_adjustments_for_file
        from raw_adjustments.pipeline import ExportCancelled, export_adjusted_image

        # Best-effort: keep the source file's sidecar in sync with what's
        # being baked/copied. This must NOT block the export itself.
        embed_xmp = None
        try:
            write_xmp_adjustments_for_file(file_path, adj)
        except Exception as xmp_exc:
            xmp_warning = xmp_exc
            logger.warning(
                "[EXPORT] Source XMP sidecar update failed (export continues): %s",
                xmp_exc,
            )
        else:
            xmp_path = resolve_xmp_path(file_path)
            embed_xmp = xmp_path if xmp_path and os.path.isfile(xmp_path) else None

        # If the batch export requested a specific AI amount, override each
        # photo's stored AIDenoiseAmount so the whole batch uses one strength.
        # Applied after the XMP write so the source sidecar keeps its own value.
        if ai_amount is not None:
            adj["AIDenoiseAmount"] = float(ai_amount)

        if cancelled.is_set():
            raise ExportCancelled()

        def emit_progress(percent: int, stage: str = "") -> None:
            try:
                if progress_cb is not None:
                    progress_cb(percent, stage)
            except Exception:
                pass

        emit_progress(5, "Decoding…")
        logger.info("[EXPORT] Decoding full-resolution edit base…")
        processor = UnifiedImageProcessor()
        last_decode_err = None
        try:
            base = processor.decode_raw_edit_base(
                file_path,
                executor=process_pool,
                use_full_resolution=True,
                apply_lens_correction=float(adj.get("LensCorrectionEnabled", 0.0) or 0.0) > 0.5,
            )
        except Exception as decode_exc:
            last_decode_err = decode_exc
            base = None
            logger.error(
                "[EXPORT] Full-resolution decode raised for %s",
                os.path.basename(file_path),
                exc_info=True,
            )
        if base is None:
            detail_bits = []
            try:
                if processor.is_libraw_unsupported(file_path):
                    detail_bits.append(
                        "LibRaw cannot demosaic this file and no usable "
                        "embedded JPEG was found."
                    )
            except Exception:
                pass
            if last_decode_err is not None:
                detail_bits.append(str(last_decode_err))
            detail = (" " + " ".join(detail_bits)) if detail_bits else ""
            raise RuntimeError(
                "Full-resolution decode failed — cannot export."
                f"{detail} "
                "Neither a LibRaw demosaic nor a usable embedded preview "
                "was available at export resolution. "
                "If the app vanished mid-export, the Mac may have run out of memory "
                "on a large full-res demosaic."
            )
        if cancelled.is_set():
            raise ExportCancelled()
        emit_progress(
            40 if use_ai_denoise else 70,
            "Denoising (AI)…" if use_ai_denoise else "Encoding…",
        )

        def _export_progress(frac: float, stage: str = "denoise") -> None:
            frac_val = max(0.0, min(1.0, float(frac)))
            if use_ai_denoise:
                emit_progress(40 + int(frac_val * 50), "Denoising (AI)…")
            else:
                emit_progress(70 + int(frac_val * 20), "Encoding…")

        export_adjusted_image(
            export_format,
            output_path=output_path,
            rgb_linear=base,
            adj=adj,
            embed_xmp_path=embed_xmp,
            cancel_check=cancelled.is_set,
            progress_cb=_export_progress if use_ai_denoise else None,
            use_ai_denoise=use_ai_denoise,
        )
        emit_progress(100, "Done")
    except Exception as exc:
        err = exc
        if type(exc).__name__ == "ExportCancelled":
            logger.info("[BATCH] export cancelled by user")
            try:
                if os.path.isfile(output_path):
                    os.remove(output_path)
            except Exception:
                pass
        else:
            logger.error("[BATCH] export failed", exc_info=True)
    return err, xmp_warning


class BatchExportWorker(QRunnable):
    """Export a snapshot of files as edited copies, strictly serial.

    Adjustments are loaded from each file's XMP sidecar (so everything the
    viewer remembers on disk is baked in). Output names are <stem>_edited.<ext>
    with collision suffixes; per-file failures are collected and the loop
    continues. Reuses the shared per-file export body.
    """

    def __init__(
        self,
        paths: list[str],
        dest_dir: str,
        export_format: str,
        process_pool: Any,
        signals: BatchFileOpSignals | None = None,
        cancel_event: threading.Event | None = None,
        name_suffix: str = "_edited",
        use_ai_denoise: bool = False,
        ai_amount: float | None = None,
    ):
        super().__init__()
        self.paths = list(paths)
        self.dest_dir = dest_dir
        self.export_format = export_format
        self.process_pool = process_pool
        self.signals = signals or BatchFileOpSignals()
        self.cancel_event = cancel_event or threading.Event()
        self.name_suffix = name_suffix
        self.use_ai_denoise = bool(use_ai_denoise)
        self.ai_amount = ai_amount
        self.done_event = threading.Event()

    _EXT_BY_FORMAT = {
        "tiff16": ".tif",
        "jpeg": ".jpg",
        "webp": ".webp",
    }

    def run(self) -> None:
        from raw_adjustments import load_adjustments_for_file

        total = len(self.paths)
        ok = 0
        failed = 0
        want_cancel = False
        detail: list[tuple[str, str | None, str | None]] = []
        reserved: set = set()
        try:
            for i, src in enumerate(self.paths):
                if self.cancel_event.is_set():
                    break
                name = os.path.basename(src)
                self.signals.progress.emit(i, total, name)
                base, _ = os.path.splitext(name)
                ext = self._EXT_BY_FORMAT.get(
                    (self.export_format or "").strip().lower(), ".jpg"
                )
                output_path = unique_destination(
                    self.dest_dir, f"{base}{self.name_suffix}{ext}", reserved
                )
                adj = {}
                try:
                    adj = dict(load_adjustments_for_file(src) or {})
                except Exception as adj_exc:
                    logger.warning(
                        "[BATCH] Adjustments load failed for %s (defaults): %s",
                        name,
                        adj_exc,
                    )
                err, _xmp_warning = export_one_file(
                    src,
                    output_path,
                    adj,
                    self.export_format,
                    self.process_pool,
                    cancel_event=self.cancel_event,
                    use_ai_denoise=self.use_ai_denoise,
                    ai_amount=self.ai_amount,
                )
                if err is None:
                    ok += 1
                    detail.append((src, None, output_path))
                else:
                    failed += 1
                    detail.append(
                        (src, type(err).__name__ + ": " + (str(err) or repr(err)), None)
                    )
                self.signals.progress.emit(i + 1, total, name)
            want_cancel = self.cancel_event.is_set()
        except Exception as exc:  # worker-level failure (never lose completion)
            logger.error("[BATCH] export worker crashed: %s", exc, exc_info=True)
            failed += 1
            want_cancel = True
            detail.append(("", f"worker crashed: {exc}", None))
        finally:
            self.done_event.set()
            self.signals.finished.emit(ok, failed, want_cancel, list(detail))
