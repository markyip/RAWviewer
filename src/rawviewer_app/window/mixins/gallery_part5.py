"""gallery mixin, part 5 of 5 (3 methods, 236 lines)."""
from __future__ import annotations

import contextlib
import os
import sys

from PyQt6.QtCore import QRunnable

from core.config import _env_int, _norm_path
from rawviewer_app.signals import FolderSortRefineSignals

from .. import lazy
from ..qt_helpers import _get_bg_thread_pool

lazy.register(sys.modules[__name__])


class GalleryPart5:
    def _schedule_folder_sort_refinement(self, token: int, file_stats: dict) -> None:
        """Refine mtime-based folder order to EXIF capture-time order in the background."""
        if token != getattr(self, "_folder_load_generation", None):
            return
        if getattr(self, "_folder_sort_refinement_token", None) == token:
            return
        paths = list(getattr(self, "image_files", []) or [])
        if len(paths) <= 1:
            return
        self._folder_sort_refinement_token = token
        newest_first = self.get_sort_preference()
        viewer = self

        signals = FolderSortRefineSignals()
        signals.ready.connect(viewer._apply_folder_sort_refinement)

        class _FolderSortRefineWorker(QRunnable):
            def __init__(self_inner):
                super().__init__()
                self_inner.signals = signals

            def run(self_inner):
                try:
                    folder_hint = (
                        os.path.dirname(os.path.abspath(paths[0])) if paths else None
                    )
                    # Refinement only reorders image_files; it does not need the
                    # full per-file EXIF blob (the gallery fetches that lazily on
                    # first open). Reading capture_time-only keeps this background
                    # sort worker from stealing the GIL via a JSON-decode-per-file
                    # burst while the current image is decoding/painting.
                    sorted_files, bulk_metadata = viewer.sort_files_by_capture_time(
                        paths,
                        newest_first=newest_first,
                        file_stats=file_stats,
                        folder_path=folder_hint,
                        capture_time_only=True,
                    )
                except Exception as e:
                    import logging
                    import traceback

                    from common_image_loader import is_emfile_error, note_emfile_pressure

                    logging.getLogger(__name__).error(
                        f"[SORT_REFINE] Exception in background refinement sort: {e}\n{traceback.format_exc()}"
                    )
                    if is_emfile_error(e):
                        note_emfile_pressure()
                        with contextlib.suppress(Exception):
                            viewer._enter_io_pressure_mode("sort_refine_emfile")
                    fallback = list(getattr(viewer, "image_files", []) or paths)
                    if fallback:
                        try:
                            from PyQt6 import sip
                            if not sip.isdeleted(self_inner.signals):
                                self_inner.signals.ready.emit(token, fallback, {})
                        except Exception:
                            pass
                    return

                try:
                    from PyQt6 import sip
                    if not sip.isdeleted(self_inner.signals):
                        self_inner.signals.ready.emit(token, sorted_files, bulk_metadata)
                except Exception:
                    pass

        def _start_worker():
            if token != getattr(viewer, "_folder_load_generation", None):
                return
            import logging

            logging.getLogger(__name__).info(
                "[SORT_REFINE] Starting background EXIF capture-time sort (%d files)",
                len(paths),
            )
            if hasattr(viewer, "status_bar") and viewer.status_bar:
                viewer.status_bar.showMessage(
                    "Sorting folder by capture time…", 0
                )
            _get_bg_thread_pool().start(_FolderSortRefineWorker())

        if getattr(self, "_folder_fast_open_token", None) == token:
            import logging

            defer_ms = _env_int("RAWVIEWER_FAST_OPEN_SORT_DEFER_MS", 2500, minimum=0)
            logging.getLogger(__name__).info(
                "[FOLDER] Deferring EXIF sort until first paint or %dms (fast-open)",
                defer_ms,
            )
            if defer_ms <= 0:
                _start_worker()
            else:
                self._defer_until_first_paint(_start_worker, fallback_ms=defer_ms)
        else:
            _start_worker()

    def _apply_folder_sort_refinement(self, token: int, sorted_files: list, bulk_metadata: dict) -> None:
        if token != getattr(self, "_folder_load_generation", None):
            return
        if self._is_semantic_search_filter_active():
            return
        if not sorted_files:
            return
        preserved = self.current_file_path
        self.image_files = sorted_files
        self._folder_sort_refinement_applied_token = token
        self._semantic_search_corpus_files = list(sorted_files)
        self._gallery_search_index_session_key = None
        if bulk_metadata:
            self._gallery_bulk_metadata = bulk_metadata
            self._bump_gallery_rating_presence_epoch()
        self._rebuild_burst_groups()
        if preserved:
            preserved_norm = _norm_path(preserved)
            for i, fp in enumerate(sorted_files):
                if _norm_path(fp) == preserved_norm:
                    self.current_file_index = i
                    self.current_file_path = fp
                    break
        self._sync_filmstrip_to_folder()
        self.update_status_bar()
        self._refresh_image_counter()
        if hasattr(self, "status_bar") and self.status_bar:
            self.status_bar.showMessage("Folder sorted by capture time (EXIF)", 3000)
        if getattr(self, "view_mode", "single") == "gallery":
            anchor = None
            gj = getattr(self, "gallery_justified", None)
            if gj is not None and hasattr(gj, "get_scroll_anchor_path"):
                try:
                    anchor = gj.get_scroll_anchor_path()
                except Exception:
                    anchor = None
            self._sync_gallery_to_folder_files(
                bulk_metadata,
                scroll_to_current=False,
                scroll_anchor_path=anchor,
                force=True,
            )

    def _sync_gallery_to_folder_files(
        self,
        bulk_metadata=None,
        *,
        scroll_to_current: bool = True,
        scroll_anchor_path=None,
        force: bool = False,
    ) -> None:
        """Apply ``image_files`` order to the gallery widget (even when gallery is hidden).

        When ``scroll_anchor_path`` is set (or captured from the gallery viewport), scroll
        restores that file after reorder instead of jumping to ``current_file_path``.
        """
        from PyQt6.QtCore import QTimer

        gj = getattr(self, "gallery_justified", None)
        if gj is None or not getattr(self, "image_files", None):
            return
        meta = bulk_metadata
        if meta is None:
            meta = getattr(self, "_gallery_bulk_metadata", None) or {}
        if (
            getattr(self, "view_mode", "") == "gallery"
            and getattr(self, "_gallery_bookmark_filter_active", False)
        ):
            files = self._gallery_files_for_display()
        else:
            files = list(self.image_files)
        gallery_order = list(getattr(gj, "images", []) or [])
        layout_seq = tuple(getattr(gj, "_layout_image_sequence", ()) or ())
        order_mismatch = gallery_order != files or layout_seq != tuple(files)
        if not force and not order_mismatch:
            return
        # A forced rebuild with no actual order change (e.g. a passive metadata-ready
        # re-sort) would tear down and rebuild every tile mid-load, interrupting
        # thumbnail decodes and causing scattered/blank tiles. Skip it while the
        # gallery is still warming up; the warmup/aspects-settle pass rebuilds anyway.
        if (
            force
            and not order_mismatch
            and getattr(self, "view_mode", "") == "gallery"
            and hasattr(gj, "_gallery_warmup_active")
            and gj._gallery_warmup_active()
        ):
            return
        # Prefer explicit anchor / saved anchor / current file -- never infer from scroll=0
        # (get_scroll_anchor_path() would return the first thumbnail, not the open image).
        # In gallery mode, prefer the open file / explicit anchor -- not scroll=0
        # (get_scroll_anchor_path() at y=0 is the first thumbnail, not the current image).
        anchor_state = None
        anchor = (
            scroll_anchor_path
            or getattr(self, "_gallery_scroll_anchor_path", None)
        )
        if anchor is None and scroll_to_current:
            anchor = getattr(self, "current_file_path", None)
        # While already in gallery during a passive resort, keep viewport position.
        if (
            anchor is None
            and getattr(self, "view_mode", "") == "gallery"
            and gj
            and not force
        ):
            anchor = gj.get_scroll_anchor_path()
        if gj is not None and hasattr(gj, "_capture_scroll_anchor_state"):
            try:
                if scroll_anchor_path:
                    anchor_state = {
                        "file_path": scroll_anchor_path,
                        "offset_px": 0,
                    }
                elif not scroll_to_current:
                    anchor_state = gj._capture_scroll_anchor_state()
            except Exception:
                anchor_state = None
        try:
            gj.set_images(
                files,
                meta if meta else None,
                force_rebuild=force or order_mismatch,
            )
        except TypeError:
            gj.set_images(files, meta if meta else None)
        if anchor_state and hasattr(gj, "_schedule_scroll_to_file_retry"):
            gj._pending_scroll_anchor_state = anchor_state
            gj._pending_scroll_to_path = anchor_state.get("file_path")
            gj._scroll_to_file_attempts = 0
            gj._schedule_scroll_to_file_retry(0)
        elif anchor and hasattr(gj, "scroll_to_file"):
            gj.scroll_to_file(anchor)
            self._gallery_scroll_anchor_path = None
        if hasattr(gj, "_sync_content_geometry"):
            QTimer.singleShot(0, gj._sync_content_geometry)
        if getattr(self, "view_mode", "") == "gallery" and hasattr(gj, "load_visible_images"):
            QTimer.singleShot(30, gj.load_visible_images)
            QTimer.singleShot(150, gj.load_visible_images)
