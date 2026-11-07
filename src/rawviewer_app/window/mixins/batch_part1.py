
"""batch mixin, part 1 of 1 (15 methods, 669 lines)."""
from __future__ import annotations

import contextlib
import logging
import os
import sys

from PyQt6.QtCore import QTimer

from core.config import _norm_path

from .. import lazy
from ..adjust_workers import _AdjustExportWorker
from ..qt_helpers import _get_bg_thread_pool

lazy.register(sys.modules[__name__])


class BatchPart1:
    def _start_batch_removal(self, kind: str, paths: list[str]) -> None:
        """Trash/discard a selection via the serial off-thread worker.

        Pre-cancels in-flight loads (UI thread), snapshots the surviving scroll
        anchor, then hands the immutable path list to BatchFileOpWorker. The
        UI-only post-pass (cache invalidation, list/corpus updates, gallery
        resync) happens once in _on_batch_op_finished.
        """
        self._batch_removal_anchor = self._surviving_anchor_after_removal(paths)
        for file_path in paths:
            self._cancel_load_and_preload_for_path(file_path)
        self._start_batch_op(kind, paths, dest_dir="", reveal=False)

    def _any_batch_op_in_progress(self) -> bool:
        self._clear_finished_worker_flags()
        return bool(getattr(self, "_batch_op_in_progress", False)) or bool(
            getattr(self, "_adjust_export_in_progress", False)
        )

    def _batch_export_selected(self) -> None:
        """Top-bar Export…: pre-export dialog, then a serial off-thread batch run."""
        if self._any_batch_op_in_progress():
            if hasattr(self, "status_bar"):
                self.status_bar.showMessage(
                    "A batch operation is already running", 3500
                )
            return
        paths = self._gallery_selected_canonical_paths()
        if not paths:
            if hasattr(self, "status_bar"):
                self.status_bar.showMessage("No images selected in gallery", 2500)
            return
        self._stop_slideshow()

        from rawviewer_app.core.widgets import BatchExportDialog, _norm_dir_key

        source_dir_keys = {
            _norm_dir_key(os.path.dirname(os.path.abspath(p))) for p in paths
        }
        dlg = BatchExportDialog(
            parent=self,
            count=len(paths),
            source_dir_keys=source_dir_keys,
            writable_check=self._directory_looks_writable,
        )
        dlg.pick_directory = (
            lambda title="Choose Destination Folder", start=None:  # noqa: B008
            self._pick_directory_dialog(title, start or dlg.destination_edit.text())
        )
        dlg.exec()
        if not dlg.result_value or not dlg.result_config:
            return
        cfg = dlg.result_config
        settings = self.get_settings()
        settings.setValue("batch_export_mode", cfg["mode"])
        settings.setValue("batch_export_format", cfg["format"])
        settings.setValue("last_batch_export_dir", cfg["destination"])
        settings.setValue("batch_export_reveal", "1" if cfg["reveal"] else "0")
        settings.setValue("batch_export_ai_denoise", "1" if cfg.get("ai_denoise") else "0")
        try:
            ai_amount = max(0, min(100, int(cfg.get("ai_amount", 100))))
        except (TypeError, ValueError):
            ai_amount = 100
        settings.setValue("batch_export_ai_amount", str(ai_amount))

        if cfg["mode"] == "copy":
            self._start_batch_op("copy", paths, cfg["destination"], cfg["reveal"])
        else:
            self._start_batch_op(
                "edited", paths, cfg["destination"], cfg["reveal"],
                export_format=cfg["format"],
                use_ai_denoise=bool(cfg.get("ai_denoise")),
                ai_amount=ai_amount,
            )

    def _start_batch_op(
        self,
        kind: str,
        paths: list[str],
        dest_dir: str,
        reveal: bool,
        export_format: str = "",
        use_ai_denoise: bool = False,
        ai_amount: float | None = None,
    ) -> None:
        """Shared launcher: modal TaskProgressDialog + serial off-thread worker.

        kind 'copy'      → BatchFileOpWorker(mode='copy')
        kind 'trash'     → BatchFileOpWorker(mode='trash')   (used by Phase 3)
        kind 'discard'   → BatchFileOpWorker(mode='discard') (used by Phase 3)
        kind 'edited'    → BatchExportWorker
        """
        from rawviewer_app.core.widgets import TaskProgressDialog
        from rawviewer_app.services.batch import (
            BatchExportWorker,
            BatchFileOpSignals,
            BatchFileOpWorker,
        )

        self._batch_op_worker = None
        signals = BatchFileOpSignals()
        signals.progress.connect(self._on_batch_op_progress)
        signals.finished.connect(
            lambda done, failed, cancelled, detail, k=kind, d=dest_dir:
            self._on_batch_op_finished(done, failed, cancelled, detail, k, d, reveal)
        )
        if kind == "edited":
            pool = getattr(getattr(self, "image_manager", None), "_process_pool", None)
            worker = BatchExportWorker(
                paths, dest_dir, export_format, pool, signals=signals,
                use_ai_denoise=use_ai_denoise,
                ai_amount=ai_amount,
            )
            title = f"Export ({export_format.upper() if export_format else 'Edited'})"
        else:
            worker = BatchFileOpWorker(paths, dest_dir, mode=kind, signals=signals)
            title = {
                "copy": "Copying Images",
                "trash": "Deleting Images",
                "discard": "Moving to Discard",
            }.get(kind, "Batch Operation")

        self._batch_op_in_progress = True
        try:
            dlg = TaskProgressDialog(self, title=title)
            dlg.canceled.connect(worker.cancel_event.set)
            self._batch_op_in_progress_dialog: TaskProgressDialog | None = dlg

            def _show_dialog_if_current(d=dlg) -> None:
                # Skip the flash for fast batches (mirror of the single-export dialog).
                current = self._batch_op_in_progress_dialog
                if current is not None and current is d:
                    current.show()

            QTimer.singleShot(300, _show_dialog_if_current)
            # Hold a reference so the worker (and its signals object) cannot be
            # collected mid-run, and so the gate can check done_event.
            self._batch_op_worker = worker
            _get_bg_thread_pool().start(worker)
        except Exception:
            # A launcher failure must never leave the Export gate latched.
            self._batch_op_in_progress = False
            self._batch_op_worker = None
            self._close_batch_progress_dialog()
            raise

    def _on_batch_op_progress(self, done: int, total: int, basename: str) -> None:
        dlg = getattr(self, "_batch_op_in_progress_dialog", None)
        if dlg is None:
            return
        try:
            dlg.set_progress(int(done * 100 / total) if total else 100)
            dlg.set_message(f"{done} of {total} · {basename}")
        except Exception:
            pass

    def _close_batch_progress_dialog(self) -> None:
        dlg = getattr(self, "_batch_op_in_progress_dialog", None)
        if dlg is not None:
            self._batch_op_in_progress_dialog = None
            with contextlib.suppress(Exception):
                dlg.canceled.disconnect()
            try:
                dlg.close()
                dlg.deleteLater()
            except Exception:
                pass

    def _on_batch_op_finished(
        self,
        done: int,
        failed: int,
        cancelled: bool,
        detail: list[tuple],
        kind: str,
        dest_dir: str,
        reveal: bool,
    ) -> None:
        self._batch_op_in_progress = False
        self._batch_op_worker = None
        self._close_batch_progress_dialog()
        verb = {
            "copy": "Copied", "trash": "Deleted", "discard": "Discarded",
            "edited": "Exported",
        }.get(kind, "Processed")
        if kind in ("trash", "discard"):
            self._after_batch_removal(kind, detail)
            return
        if hasattr(self, "status_bar"):
            msg = f"{verb} {done} image{'s' if done != 1 else ''}"
            if kind in ("copy", "edited"):
                msg += f" to {os.path.basename(dest_dir.rstrip(os.sep)) or dest_dir}"
            if failed:
                msg += f" ({failed} failed)"
            if cancelled:
                msg += " -- cancelled"
            self.status_bar.showMessage(msg, 6000)
        # Files written into the currently open folder must enter the gallery,
        # exactly like single-image exports do.
        if kind in ("copy", "edited") and detail:
            folder = _norm_path(getattr(self, "current_folder", "") or "")
            for _src, _err, dest in detail:
                if dest and folder and _norm_path(os.path.dirname(dest)) == folder:
                    self.register_new_file(dest)
        if failed:
            failure = next((err for _s, err, _d in detail if err), "unknown error")
            with contextlib.suppress(Exception):
                self.show_error(
                    f"{verb} -- {failed} failed",
                    f"{failed} file(s) could not be processed.\nLast error: {failure}",
                )
        if reveal and not cancelled and os.path.isdir(dest_dir):
            self._open_folder_in_file_manager(dest_dir)

    def _after_batch_removal(self, kind: str, detail: list[tuple]) -> None:
        """UI-thread post-pass for trash/discard batches.

        detail: [(src, err|None, dest|None], one entry per attempted path --
        including files that were already gone, which the old inline loop also
        counted as removed.
        """
        from image_cache import get_image_cache

        cache = get_image_cache()
        removed_paths = []
        failed = 0
        last_err = None
        for src, err, dest in detail:
            if err:
                failed += 1
                last_err = err
                continue
            cache.invalidate_file(src)
            if kind == "discard" and dest:
                self._migrate_visual_rotation_path(src, dest)
            self._drop_discarded_from_semantic_corpus(src)
            removed_paths.append(src)
        if removed_paths:
            self._remove_files_from_active_image_list(removed_paths)
        verb = "Deleted" if kind == "trash" else "Discarded"
        removed = len(removed_paths)
        if hasattr(self, "status_bar") and self.status_bar is not None:
            msg = f"{verb} {removed} image{'s' if removed != 1 else ''}"
            if failed:
                msg += f" ({failed} failed)"
            self.status_bar.showMessage(msg, 5000)
        self._after_gallery_bulk_file_removal(
            anchor_path=getattr(self, "_batch_removal_anchor", None)
        )
        self._batch_removal_anchor = None
        self.schedule_save_session_state()
        if failed:
            with contextlib.suppress(Exception):
                self.show_error(
                    f"{verb} -- {failed} failed",
                    f"{failed} file(s) could not be {verb.lower()}.\nLast error: {last_err}",
                )

    def _preferred_writable_export_dir(self, preferred: str = "") -> str:
        """Pick a writable export folder; fall back to Pictures/Desktop/home."""
        preferred = self._sanitize_dialog_start_dir(preferred)
        if preferred and self._directory_looks_writable(preferred):
            return preferred
        home = os.path.expanduser("~")
        for candidate in (
            os.path.join(home, "Pictures"),
            os.path.join(home, "Desktop"),
            os.path.join(home, "Downloads"),
            home,
        ):
            if self._directory_looks_writable(candidate):
                return candidate
        return preferred or home

    def _export_destination_permission_error(self, output_path: str) -> str | None:
        """Return a user-facing message if output_path cannot be written, else None."""
        output_path = (output_path or "").strip()
        if not output_path:
            return "No export destination was chosen."
        parent = os.path.dirname(os.path.abspath(output_path))
        if not parent:
            return f"Invalid export path:\n{output_path}"
        if not os.path.isdir(parent):
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError as exc:
                return (
                    f"Cannot create export folder:\n{parent}\n\n{exc}\n\n"
                    "Choose Desktop, Pictures, or Downloads instead "
                    "(camera cards / network shares are often read-only)."
                )
        if os.path.exists(output_path) and not os.access(output_path, os.W_OK):
            return (
                f"Cannot overwrite (permission denied):\n{output_path}\n\n"
                "Pick another name or folder."
            )
        if not self._directory_looks_writable(parent):
            return (
                f"Cannot write to this folder (permission denied or read-only volume):\n"
                f"{parent}\n\n"
                "Choose Desktop, Pictures, or Downloads. "
                "SD cards and camera volumes are often mounted read-only."
            )
        return None

    def _cancel_display_pixmap_converter(self) -> None:
        converter = getattr(self, "_display_pixmap_converter", None)
        if converter is None:
            self._pending_display_pixmap_context = None
            return
        try:
            converter.stop_processing()
            with contextlib.suppress(Exception):
                converter.pixmap_ready.disconnect(self._on_display_numpy_pixmap_ready)
            converter.quit()
            if not converter.wait(500):
                converter.terminate()
                converter.wait(500)
        except Exception:
            pass
        self._display_pixmap_converter = None
        self._pending_display_pixmap_context = None

    def _on_adjust_panel_export_requested(
        self, export_format: str, adj: dict, use_ai_denoise: bool = False
    ) -> None:
        logger = logging.getLogger(__name__)
        path = getattr(self, "current_file_path", None)
        if not path:
            logger.info("[EXPORT] Requested but no current_file_path -- ignoring")
            return
        if getattr(self, "_adjust_export_in_progress", False):
            logger.warning(
                "[EXPORT] Requested (%s) while another export is still in progress"
                " for %s -- ignoring. If this persists across attempts, the previous"
                " export's worker never signalled completion.",
                export_format,
                os.path.basename(path),
            )
            return
        logger.info(
            "[EXPORT] Requested format=%s for %s (ai_denoise=%s)",
            export_format, os.path.basename(path), use_ai_denoise,
        )
        from PyQt6.QtWidgets import QFileDialog

        from raw_adjustments.pipeline import (
            EXPORT_FORMAT_JPEG,
            EXPORT_FORMAT_TIFF16,
            EXPORT_FORMAT_WEBP,
        )

        raw_fmt = (export_format or EXPORT_FORMAT_TIFF16).strip().lower()
        fmt = raw_fmt
        base = os.path.splitext(os.path.basename(path))[0]
        # Default next to the RAW when that folder is writable; otherwise jump to
        # Pictures/Desktop so users browsing a locked SD/camera card don't sit
        # on a read-only destination and fail after a long full-res decode.
        source_dir = os.path.dirname(os.path.abspath(path))
        start_dir = self._preferred_writable_export_dir(source_dir)
        if start_dir != self._sanitize_dialog_start_dir(source_dir):
            logger.info(
                "[EXPORT] Source folder not writable (%s) -- defaulting save dialog to %s",
                source_dir,
                start_dir,
            )

        nn_suffix = " + AI denoise" if use_ai_denoise else ""
        if use_ai_denoise:
            with contextlib.suppress(TypeError, ValueError, AttributeError):
                nn_suffix = (
                    " + AI denoise "
                    f"{int(round(float(adj.get('AIDenoiseAmount', 100.0))))}%"
                )
        if fmt == EXPORT_FORMAT_JPEG:
            default_name = f"{base}_edited.jpg"
            dialog_title = "Export JPEG" + nn_suffix
            file_filter = "JPEG (*.jpg *.jpeg)"
        elif fmt == EXPORT_FORMAT_WEBP:
            default_name = f"{base}_edited.webp"
            dialog_title = "Export WebP" + nn_suffix
            file_filter = "WebP (*.webp)"
        else:
            default_name = f"{base}_edited.tif"
            dialog_title = "Export 16-bit TIFF" + nn_suffix
            file_filter = "TIFF (*.tif *.tiff)"

        if fmt == EXPORT_FORMAT_JPEG:
            file_types = ["jpg", "jpeg"]
        elif fmt == EXPORT_FORMAT_WEBP:
            file_types = ["webp"]
        else:
            file_types = ["tif", "tiff"]

        # Plain QFileDialog.getSaveFileName() can silently return "" with no
        # dialog ever appearing when the app is launched from a Terminal shell
        # script (Qt's native macOS panel creation fails in that process
        # context) -- the same failure mode already worked around for the
        # Open dialog (_open_file_dialog_macos_native). Apply the identical
        # NSSavePanel-with-AppleScript-fallback treatment here.
        try:
            if self._use_qt_file_dialog():
                output_path, _ = QFileDialog.getSaveFileName(
                    self,
                    dialog_title,
                    os.path.join(start_dir, default_name),
                    file_filter,
                    options=QFileDialog.Option.DontUseNativeDialog,
                )
            elif sys.platform == "darwin":
                output_path = self._save_file_dialog_macos_native(
                    start_dir, default_name, dialog_title, file_types
                )
            else:
                self._prepare_for_modal_dialog()
                output_path, _ = QFileDialog.getSaveFileName(
                    self,
                    dialog_title,
                    os.path.join(start_dir, default_name),
                    file_filter,
                )
        except Exception as e:
            logger.error("[EXPORT] Save dialog failed: %s", e, exc_info=True)
            self.show_error("Export", f"Could not open save dialog:\n{e}")
            return

        if os.environ.get("RAWVIEWER_DEBUG", "").strip() == "1":
            backend = "qt" if self._use_qt_file_dialog() else (
                "macos-native" if sys.platform == "darwin" else "qfiledialog"
            )
            logger.info("[EXPORT] Save dialog done (backend=%s, selected=%r)", backend, output_path or "")

        if not output_path:
            logger.info("[EXPORT] Save dialog cancelled by user")
            if hasattr(self, "status_bar"):
                self.status_bar.showMessage("Export cancelled", 2500)
            return
        logger.info("[EXPORT] Save target chosen: %s", output_path)
        low = output_path.lower()
        if fmt == EXPORT_FORMAT_JPEG and not low.endswith((".jpg", ".jpeg")):
            output_path += ".jpg"
        elif fmt == EXPORT_FORMAT_WEBP and not low.endswith(".webp"):
            output_path += ".webp"
        elif fmt == EXPORT_FORMAT_TIFF16 and not low.endswith((".tif", ".tiff")):
            output_path += ".tif"

        perm_err = self._export_destination_permission_error(output_path)
        if perm_err:
            logger.warning("[EXPORT] Destination not writable: %s", output_path)
            self.show_error("Export -- permission denied", perm_err)
            return

        self._adjust_export_in_progress = True
        self._adjust_export_format = raw_fmt
        panel = getattr(self, "single_image_adjust_panel", None)
        if panel is not None and hasattr(panel, "set_export_enabled"):
            panel.set_export_enabled(False)
        if hasattr(self, "status_bar"):
            self.status_bar.showMessage(
                f"Exporting {os.path.basename(output_path)}…", 0
            )
        # adj comes straight from the panel's export_requested signal, which
        # (like every other adjustments snapshot) has no knowledge of the
        # dodge/burn / heal masks -- without this, exported files would
        # silently drop local brush work. See _dodge_burn_overlay_adj.
        # Export may run in a process pool, so replace the live mask object
        # with a PNG serial the worker can pickle/deserialize.
        try:
            from raw_adjustments.dodge_burn import MASK_KEY, MASK_OBJ_KEY, serialize_mask
            from raw_adjustments.spot_heal import (
                MASK_KEY as HEAL_MASK_KEY,
            )
            from raw_adjustments.spot_heal import (
                MASK_OBJ_KEY as HEAL_MASK_OBJ_KEY,
            )
            from raw_adjustments.spot_heal import serialize_mask as serialize_heal_mask

            adj = self._dodge_burn_overlay_adj(adj)
            adj = dict(adj)
            mask_obj = adj.pop(MASK_OBJ_KEY, None)
            if mask_obj is not None and not getattr(mask_obj, "is_empty", True):
                adj[MASK_KEY] = serialize_mask(mask_obj)
            heal_obj = adj.pop(HEAL_MASK_OBJ_KEY, None)
            if heal_obj is not None and not getattr(heal_obj, "is_empty", True):
                adj[HEAL_MASK_KEY] = serialize_heal_mask(heal_obj)
        except Exception:
            pass
        pool = getattr(getattr(self, "image_manager", None), "_process_pool", None)
        worker = _AdjustExportWorker(
            path,
            output_path,
            adj,
            raw_fmt,
            pool,
            self._adjust_export_signals,
            use_ai_denoise=bool(use_ai_denoise),
        )
        logger.info("[EXPORT] Handing off to background worker (format=%s)", raw_fmt)
        self._adjust_export_worker = worker
        # Modal progress dialog: freezes app interaction for the duration
        # (window-modal), offers Cancel (polled between export stages and
        # between AI-denoise tiles, so it lands within ~2s), and is dismissed
        # automatically by _on_adjust_export_finished. Frameless MD3 chrome
        # matching CustomConfirmDialog (stock QProgressDialog was OS-native
        # title bar + an indeterminate marquee with no real percentage).
        try:
            from rawviewer_app.core.widgets import TaskProgressDialog

            dlg = TaskProgressDialog(
                self, title="Export", message=f"Exporting {os.path.basename(output_path)}…"
            )
            dlg.canceled.connect(worker.cancel_event.set)
            self._adjust_export_dialog = dlg
            self._adjust_export_dialog_path = path
            # Skip the flash for fast exports, same as the old QProgressDialog's
            # setMinimumDuration(300) -- only show once the export has actually
            # been running for a moment.
            QTimer.singleShot(
                300,
                lambda d=dlg: d.show()
                if getattr(self, "_adjust_export_dialog", None) is d
                else None,
            )
        except Exception:
            self._adjust_export_dialog = None
            self._adjust_export_dialog_path = None
        _get_bg_thread_pool().start(worker)

    def _on_adjust_export_progress(self, source_path: str, percent: int, stage: str) -> None:
        dlg = getattr(self, "_adjust_export_dialog", None)
        if dlg is None or _norm_path(source_path) != _norm_path(
            getattr(self, "_adjust_export_dialog_path", "") or ""
        ):
            return
        try:
            dlg.set_progress(percent)
            dlg.set_message(f"Exporting {os.path.basename(source_path)}… {stage}")
        except Exception:
            pass

    def _on_adjust_export_finished(
        self,
        source_path: str,
        output_path: str,
        export_format: str,
        err: object,
        xmp_warning: object = None,
    ) -> None:
        logger = logging.getLogger(__name__)
        logger.info(
            "[EXPORT] _on_adjust_export_finished: source=%s output=%s err=%s",
            os.path.basename(source_path),
            output_path,
            err,
        )
        self._adjust_export_in_progress = False
        self._adjust_export_worker = None
        dlg = getattr(self, "_adjust_export_dialog", None)
        if dlg is not None:
            self._adjust_export_dialog = None
            self._adjust_export_dialog_path = None
            with contextlib.suppress(Exception):
                dlg.canceled.disconnect()
            try:
                dlg.close()
                dlg.deleteLater()
            except Exception:
                pass
        panel = getattr(self, "single_image_adjust_panel", None)
        if panel is not None and hasattr(panel, "set_export_enabled"):
            panel.set_export_enabled(True)
        if _norm_path(source_path) != _norm_path(
            getattr(self, "current_file_path", "") or ""
        ):
            # User navigated to a different image while this export was still
            # running in the background. The status bar now belongs to that
            # other image, so we don't overwrite it -- but the user who
            # requested this export still deserves to know whether it
            # succeeded, otherwise a slow export that finishes after they've
            # moved on looks exactly like nothing happened at all.
            logger.info(
                "[EXPORT] Current file changed since export was requested"
                " (now viewing %s) -- showing a transient notification instead"
                " of the per-file status bar",
                os.path.basename(getattr(self, "current_file_path", "") or "?"),
            )
            if hasattr(self, "status_bar"):
                name = os.path.basename(source_path)
                if err is not None:
                    if type(err).__name__ != "ExportCancelled":
                        msg = str(err) or repr(err)
                        self.status_bar.showMessage(
                            f"Export of {name} failed (while viewing another image): {msg}",
                            7000,
                        )
                        with contextlib.suppress(Exception):
                            self.show_error("Export failed", f"{name}: {msg}")
                else:
                    self.status_bar.showMessage(
                        f"Exported {name} to {os.path.basename(output_path)}"
                        " (while viewing another image)",
                        7000,
                    )
                    self.register_new_file(output_path)
            return
        if err is not None:
            if type(err).__name__ == "ExportCancelled":
                if hasattr(self, "status_bar"):
                    self.status_bar.showMessage("Export cancelled", 4000)
                return
            msg = str(err) or repr(err)
            if isinstance(err, (PermissionError, OSError)) and getattr(err, "errno", None) in (
                13,
                1,
                30,  # EROFS
            ):
                msg = (
                    f"{msg}\n\n"
                    "The export folder may be read-only (SD card / camera volume / "
                    "network share). Choose Desktop, Pictures, or Downloads and try again."
                )
            elif isinstance(err, PermissionError) or (
                isinstance(err, OSError) and "Permission" in msg
            ):
                msg = (
                    f"{msg}\n\n"
                    "Choose a writable folder (Desktop, Pictures, or Downloads) and try again."
                )
            if hasattr(self, "status_bar"):
                self.status_bar.showMessage(f"Export failed: {msg}", 7000)
            with contextlib.suppress(Exception):
                self.show_error("Export failed", msg)
            return
        fmt = (export_format or "tiff16").strip().lower()
        labels = {
            "tiff16": "16-bit TIFF",
            "jpeg": "JPEG",
            "webp": "WebP",
        }
        label = labels.get(fmt, "image")
        if hasattr(self, "status_bar"):
            if xmp_warning is not None:
                self.status_bar.showMessage(
                    f"Exported {label}: {os.path.basename(output_path)}"
                    f" (source XMP not updated: {xmp_warning})",
                    7000,
                )
            else:
                self.status_bar.showMessage(
                    f"Exported {label}: {os.path.basename(output_path)}", 5000
                )
        # Show it now if it landed in the folder being browsed, rather than
        # leaving it invisible until the folder is reopened.
        self.register_new_file(output_path)

    def _track_pixmap_converter(self, converter) -> None:
        """Keep a preload PixmapConverter (QThread) alive until it finishes,
        then drop it -- each one holds a full copy of the decoded RGB buffer
        (PixmapConverter.__init__ does rgb_image.copy()), so leaving it in a
        list that's never pruned leaked one full-resolution buffer per
        preloaded navigation for the rest of the session (confirmed via a
        260-image navigation stress test: RSS climbed from ~1GB to ~3.2GB and
        per-navigation latency degraded ~4x from first third to last third).
        Same live-set-with-discard-on-finished pattern as
        _live_display_pixmap_converters in _start_display_numpy_pixmap_worker.
        """
        live = getattr(self, "_pixmap_converters", None)
        if live is None:
            live = self._pixmap_converters = set()
        live.add(converter)
        converter.finished.connect(lambda c=converter: self._pixmap_converters.discard(c))
