"""search mixin, part 4 of 4 (19 methods, 697 lines)."""
from __future__ import annotations

import contextlib
import logging
import os
import sys

from PyQt6.QtCore import QRunnable, QTimer

from core.config import _env_int, _norm_path
from rawviewer_app.signals import QuickFolderIndexSignals, SemanticIndexPrepSignals
from rawviewer_app.workers import _SemanticIndexPrepWorker

from .. import lazy
from ..lazy import metadata_index_idle_delay_ms
from ..qt_helpers import _get_bg_thread_pool

lazy.register(sys.modules[__name__])


class SearchPart4:
    def _maybe_resume_semantic_indexing(self) -> None:
        """Resume interrupted indexing only after the user opens gallery search."""
        try:
            logger = logging.getLogger(__name__)
            if not self._user_wants_semantic_embeddings_now():
                folder = getattr(self, "current_folder", None)
                if folder and self._is_semantic_index_incomplete(folder):
                    logger.info(
                        "[INDEX][RESUME] Deferred until user opens search (folder=%s)",
                        folder,
                    )
                return
            folder = getattr(self, "current_folder", None)
            corpus = list(getattr(self, "image_files", []) or [])
            if not folder or not corpus:
                logger.info(
                    "[INDEX][RESUME] Skip: missing folder/corpus "
                    f"(has_folder={bool(folder)}, corpus_size={len(corpus)})"
                )
                return
            if getattr(self, "_semantic_indexing_in_progress", False) or getattr(
                self, "_semantic_index_prep_in_progress", False
            ):
                logger.info(
                    "[INDEX][RESUME] Skip: indexing already active "
                    f"(semantic={getattr(self, '_semantic_indexing_in_progress', False)}, "
                    f"prep={getattr(self, '_semantic_index_prep_in_progress', False)})"
                )
                return
            if not self._is_semantic_index_incomplete(folder):
                logger.info(
                    "[INDEX][RESUME] Skip: folder not marked incomplete "
                    f"(folder={folder})"
                )
                return

            # Give folder view setup a short head start, then resume in background.
            logger.info(
                "[INDEX][RESUME] Schedule prep in 1200ms "
                f"(folder={folder}, corpus_size={len(corpus)})"
            )
            QTimer.singleShot(
                1200,
                lambda f=folder, c=list(corpus): self._trigger_resume_semantic_indexing(f, c),
            )
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"[INDEX][RESUME] Failed to schedule resume prep: {e}"
            )

    def _trigger_resume_semantic_indexing(self, folder, corpus):
        logger = logging.getLogger(__name__)
        # Double check folder hasn't changed during the delay
        if getattr(self, "current_folder", None) != folder:
            logger.info(
                "[INDEX][RESUME] Cancel: folder changed "
                f"(expected={folder}, current={getattr(self, 'current_folder', None)})"
            )
            return
        if self._semantic_indexing_in_progress or getattr(self, "_semantic_index_prep_in_progress", False):
            logger.info(
                "[INDEX][RESUME] Cancel: indexing already active "
                f"(semantic={self._semantic_indexing_in_progress}, "
                f"prep={getattr(self, '_semantic_index_prep_in_progress', False)})"
            )
            return
        try:
            index = self._get_semantic_index()
            if not index.semantic_backend_available():
                logger.info(
                    "[INDEX][RESUME] Cancel: semantic backend unavailable "
                    f"({index.semantic_backend_error()})"
                )
                return
            logger.info(
                "[INDEX][RESUME] Start prep worker "
                f"(folder={folder}, corpus_size={len(corpus)})"
            )
            # Spin up the background prep worker instead of doing get_index_coverage and get_pending_paths on UI thread!
            self._run_semantic_index_prep_for_resume(folder, corpus)
        except Exception as e:
            logger.warning(f"[INDEX][RESUME] Failed before prep worker start: {e}")

    def _run_semantic_index_prep_for_resume(self, folder, corpus_files):
        if getattr(self, "_semantic_index_prep_in_progress", False):
            return
        self._semantic_index_prep_in_progress = True

        index = self._get_semantic_index()
        signals = SemanticIndexPrepSignals()
        self._semantic_index_prep_signals = signals

        signals.done.connect(lambda coverage, pending, face_pending, extract_pending=None: self._on_semantic_index_resume_prep_done(folder, corpus_files, coverage, pending, face_pending))
        signals.error.connect(self._on_semantic_index_prep_error)

        worker = _SemanticIndexPrepWorker(corpus_files, index, signals)
        _get_bg_thread_pool().start(worker)

    def _on_semantic_index_resume_prep_done(self, folder, corpus_files, coverage, pending, face_pending):
        self._semantic_index_prep_in_progress = False
        logger = logging.getLogger(__name__)
        # Double check folder hasn't changed
        if getattr(self, "current_folder", None) != folder:
            logger.info(
                "[INDEX][RESUME] Prep done ignored: folder changed "
                f"(expected={folder}, current={getattr(self, 'current_folder', None)})"
            )
            return

        # If there are no pending files and no face pending files, mark complete
        needs_semantic = len(pending) > 0
        needs_face = face_pending > 0
        logger.info(
            "[INDEX][RESUME] Prep done "
            f"(pending_semantic={len(pending)}, pending_face={int(face_pending or 0)}, "
            f"needs_semantic={needs_semantic}, needs_face={needs_face})"
        )

        if not needs_semantic and not needs_face:
            self._set_semantic_index_incomplete(folder, False)
            logger.info("[INDEX][RESUME] Nothing pending; marked folder complete")
            return

        if needs_semantic:
            if self._user_wants_semantic_embeddings_now():
                from search import semantic_embeddings_enabled

                self._start_semantic_index_build_background(
                    corpus_files,
                    coverage=coverage,
                    pending_files=pending,
                    run_semantic_embeddings=semantic_embeddings_enabled(),
                    show_search_progress=bool(
                        getattr(self, "_gallery_search_show_index_progress", False)
                    ),
                )
            else:
                logger.info(
                    "[INDEX][RESUME] Semantic embeddings deferred until user opens search "
                    f"(pending={len(pending)})"
                )
                self._set_semantic_index_incomplete(folder, True)
            return

        if needs_face:
            self._start_face_index_background(corpus_files)
        else:
            self._set_semantic_index_incomplete(folder, False)

    def _resume_indexing_if_single_view(self):
        """Resume background indexing only if the user has settled in single view mode."""
        if getattr(self, "view_mode", "single") != "single":
            return
        if getattr(self, "_folder_fast_open_token", None) is not None:
            if hasattr(self, "_resume_indexing_timer") and self._resume_indexing_timer:
                self._resume_indexing_timer.start(3000)
            return
        if not getattr(self, "_single_view_first_render_logged", False):
            if hasattr(self, "_resume_indexing_timer") and self._resume_indexing_timer:
                self._resume_indexing_timer.start(3000)
            return
        try:
            self._get_semantic_index().pause_indexing(False)
            import logging

            logging.getLogger(__name__).info(
                "[VIEW_MODE] Background indexing resumed after idle period in single view"
            )
        except Exception:
            pass

    def _resume_indexing_if_gallery_idle(self):
        """Gallery browsing keeps background indexing paused; resume only when idle/settled."""
        if getattr(self, "view_mode", "") != "gallery":
            return
        import logging
        logger = logging.getLogger(__name__)

        # Check if there are active high-priority (CURRENT) tasks in the image manager
        if hasattr(self, "image_manager") and self.image_manager is not None:
            try:
                from image_load_manager import Priority
                with self.image_manager._queue_lock:
                    for task in self.image_manager._active_tasks.values():
                        if (
                            getattr(task, "priority", None) == Priority.CURRENT
                            and not task.is_cancelled()
                        ):
                            logger.debug(
                                "[VIEW_MODE] Deferring background indexing: high-priority gallery task active"
                            )
                            if hasattr(self, "_resume_indexing_timer") and self._resume_indexing_timer:
                                self._resume_indexing_timer.start(3000)
                            return
            except Exception as e:
                logger.error(f"Error checking active image manager tasks: {e}")

        try:
            self._get_semantic_index().pause_indexing(False)
            logger.info(
                "[VIEW_MODE] Background indexing resumed after idle period in gallery view"
            )
        except Exception as e:
            logger.error(f"Error resuming background indexing in gallery view: {e}")

    def _sorted_insert_index(self, files: list, path: str) -> int:
        """Where ``path`` belongs in ``files`` under the active sort.

        Falls back to the end rather than guessing when the order in use is
        not one this can reproduce -- an appended tile is mildly out of place;
        a wrongly-placed one looks like corruption.
        """
        try:
            newest_first = bool(getattr(self, "sort_newest_first", False))
            key = os.path.getmtime(path)
            times = []
            for p in files:
                try:
                    times.append(os.path.getmtime(p))
                except OSError:
                    times.append(None)
            if any(t is None for t in times):
                return len(files)
            ordered = times == sorted(times, reverse=newest_first)
            if not ordered:
                return len(files)
            for i, t in enumerate(times):
                if (key > t) if newest_first else (key < t):
                    return i
            return len(files)
        except Exception:
            return len(files)

    def _advance_index_past_unsupported_for_editing(self, step: int) -> None:
        """Editor-mode navigation: silently skip files the editor cannot open.

        Called after the nav index has been advanced by ``step`` (+1/-1).
        Keeps stepping in the same direction (with wraparound) until a
        supported file is found; if every file is unsupported, stays put.
        """
        if not self._adjust_panel_active():
            return
        n = len(self.image_files or [])
        if n <= 1:
            return
        import logging

        for _ in range(n - 1):
            path = self.image_files[self.current_file_index]
            if self._editing_supported_for_file(path):
                return
            logging.getLogger(__name__).info(
                "[ADJUST] Skipping %s during editor navigation (not editable)",
                os.path.basename(path),
            )
            self.current_file_index = (self.current_file_index + step) % n

    def _schedule_silent_metadata_index(self) -> None:
        """Defer silent metadata indexing after folder load (no search UI)."""
        from search import metadata_auto_index_enabled

        if not metadata_auto_index_enabled():
            return
        if getattr(self, "_metadata_index_run_done", False):
            return
        if self._semantic_indexing_in_progress or getattr(
            self, "_semantic_index_prep_in_progress", False
        ):
            return
        t = getattr(self, "_metadata_index_defer_timer", None)
        if t is None:
            return
        delay_ms = metadata_index_idle_delay_ms()
        t.start(delay_ms)

    def _maybe_start_background_metadata_index(self):
        """Timer callback to scan the current folder for missing metadata."""
        if getattr(self, "_metadata_index_prep_complete", False) and not getattr(
            self, "_metadata_index_run_done", False
        ):
            self._retry_deferred_metadata_index_worker()
            return
        if getattr(self, "_metadata_index_run_done", False):
            if getattr(self, "_gallery_search_user_wants_semantic", False):
                corpus = self._gallery_search_index_corpus()
                QTimer.singleShot(0, lambda: self._start_user_semantic_indexing(corpus))
            return
        if self._semantic_indexing_in_progress or getattr(self, "_semantic_index_prep_in_progress", False):
            return
        try:
            from common_image_loader import io_pressure_active

            if io_pressure_active():
                self._metadata_index_defer_timer.start(15000)
                return
        except Exception:
            pass

        # Hold metadata indexing while the gallery is still warming up or actively
        # painting tiles: the parallel EXIF/thumbnail extraction competes with
        # gallery thumbnail decodes for the same worker pool and (on external/USB
        # drives) the same slow disk, which stalls first paint and -- on Windows --
        # can drive concurrency high enough to crash. Reschedule instead of start.
        if self._gallery_blocks_background_indexing():
            import logging

            logging.getLogger(__name__).info(
                "[INDEX] Deferring background metadata indexing (gallery warming/loading)"
            )
            t = getattr(self, "_metadata_index_defer_timer", None)
            if t is not None:
                t.start(_env_int("RAWVIEWER_METADATA_INDEX_GALLERY_RETRY_MS", 2000, minimum=250))
            return

        if self._should_defer_metadata_index_for_single_view():
            import logging

            logging.getLogger(__name__).info(
                "[INDEX] Deferring background metadata indexing (single-view navigation/load pressure)"
            )
            t = getattr(self, "_metadata_index_defer_timer", None)
            if t is not None:
                t.start(
                    _env_int("RAWVIEWER_SINGLE_VIEW_INDEX_DEFER_MS", 4000, minimum=500)
                )
            return

        folder = getattr(self, "current_folder", None)
        corpus = getattr(self, "_semantic_search_corpus_files", None) or getattr(
            self, "image_files", None
        )
        if not folder or not corpus:
            return

        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[INDEX] Starting background metadata indexing for folder: {folder}")

        self._run_metadata_index_prep(folder, corpus)

    def _should_defer_metadata_index_for_gallery(self) -> bool:
        """True while gallery first-paint/warmup is in progress (avoid I/O contention).

        Only applies in gallery mode. Indexing must not start on a fixed timer while
        visible tiles are still blank -- that starves first paint on large folders.
        A long absolute ceiling still guarantees indexing eventually runs for search.
        """
        if getattr(self, "view_mode", "") != "gallery":
            return False
        import time

        gj = getattr(self, "gallery_justified", None)
        opened = float(getattr(self, "_gallery_deferral_opened_ts", 0.0) or 0.0)
        abs_ceiling_s = _env_int(
            "RAWVIEWER_GALLERY_INDEX_FORCE_AFTER_MS", 60000, minimum=10000
        ) / 1000.0
        if (
            gj is not None
            and not getattr(gj, "_first_thumb_ready_after_set", False)
            and opened
            and (time.time() - opened) < abs_ceiling_s
        ):
            return True

        if self._gallery_has_thumbnail_load_pressure():
            return True

        deferred_until = float(getattr(self, "_gallery_indexing_deferred_until", 0.0) or 0.0)
        if time.time() < deferred_until:
            return True

        # After first paint: soft ceiling so continuous scrolling does not defer forever.
        scroll_ceiling_s = _env_int(
            "RAWVIEWER_METADATA_INDEX_DEFER_MAX_MS", 30000, minimum=5000
        ) / 1000.0
        if opened and (time.time() - opened) > scroll_ceiling_s:
            return False
        return False

    def _gallery_blocks_background_indexing(self) -> bool:
        """Single gate shared by every background-indexing entry point (metadata scan,
        semantic index build, and their deferred retries) so they cannot drift out of
        sync on what "gallery first paint still in progress" means.
        """
        return self._should_defer_metadata_index_for_gallery()

    def _pause_single_view_prefetch_for_indexing(self) -> None:
        """Stop idle display prefetch while metadata indexing runs."""
        timer = getattr(self, "_idle_display_prefetch_timer", None)
        if timer is not None:
            timer.stop()

    def _retry_deferred_metadata_index_worker(self) -> None:
        """Start metadata index worker after prep completed but was deferred."""
        if getattr(self, "_metadata_index_run_done", False):
            return
        if not getattr(self, "_metadata_index_prep_complete", False):
            return
        if self._semantic_indexing_in_progress or getattr(
            self, "_semantic_index_prep_in_progress", False
        ):
            return
        pending_extract = int(
            getattr(self, "_metadata_index_extract_pending_count", 0) or 0
        )
        if getattr(self, "view_mode", "") == "gallery" and self._gallery_blocks_background_indexing():
            import logging

            logging.getLogger(__name__).info(
                "[INDEX] Deferring metadata worker after prep (gallery first-paint, pending=%d)",
                pending_extract,
            )
            t = getattr(self, "_metadata_index_defer_timer", None)
            if t is not None:
                t.start(
                    _env_int(
                        "RAWVIEWER_METADATA_INDEX_GALLERY_RETRY_MS", 1500, minimum=250
                    )
                )
            return
        if self._should_defer_metadata_index_for_single_view(
            pending_extract=pending_extract
        ):
            import logging

            logging.getLogger(__name__).info(
                "[INDEX] Deferring metadata worker after prep (single-view load pressure, pending=%d)",
                pending_extract,
            )
            t = getattr(self, "_metadata_index_defer_timer", None)
            if t is not None:
                t.start(
                    _env_int("RAWVIEWER_SINGLE_VIEW_INDEX_DEFER_MS", 4000, minimum=500)
                )
            return
        corpus = getattr(self, "_metadata_index_deferred_corpus", None)
        pending = getattr(self, "_metadata_index_deferred_pending", None)
        coverage = getattr(self, "_metadata_index_deferred_coverage", None)
        if not corpus or not pending:
            return
        self._start_semantic_index_build_background(
            corpus,
            coverage=coverage,
            pending_files=pending,
            run_semantic_embeddings=False,
            show_search_progress=False,
        )

    def _should_defer_metadata_index_for_single_view(
        self, *, pending_extract: int | None = None
    ) -> bool:
        """True while single-view navigation or heavy display loads are in flight."""
        if getattr(self, "view_mode", "single") != "single":
            return False
        import time

        if getattr(self, "_navigation_in_progress", False):
            return True
        if self._raw_recovery_preview_active():
            return True
        if not getattr(self, "fit_to_window", True):
            return True
        last_nav = float(getattr(self, "_last_navigation_finish_ts", 0.0) or 0.0)
        nav_grace_s = _env_int("RAWVIEWER_SINGLE_VIEW_INDEX_DEFER_AFTER_NAV_MS", 3500, minimum=0) / 1000.0
        if last_nav and (time.time() - last_nav) < nav_grace_s:
            return True
        last_activity = float(
            getattr(self, "_last_single_view_interactivity_ts", 0.0) or 0.0
        )
        idle_grace_s = _env_int("RAWVIEWER_SINGLE_VIEW_INDEX_IDLE_MS", 12000, minimum=0) / 1000.0
        if last_activity and (time.time() - last_activity) < idle_grace_s:
            return True
        if not getattr(self, "_single_view_first_render_logged", False):
            return True
        try:
            stats = self.image_manager.get_stats()
            active = int(stats.get("active_tasks", 0) or 0)
            queue = int(stats.get("queue_size", 0) or 0)
            threshold = _env_int("RAWVIEWER_SINGLE_VIEW_INDEX_DEFER_ACTIVE", 3, minimum=1)
            if active >= threshold or queue >= threshold:
                return True
        except Exception:
            pass
        if pending_extract is None:
            pending_extract = int(
                getattr(self, "_metadata_index_extract_pending_count", 0) or 0
            )
        huge_threshold = _env_int(
            "RAWVIEWER_SINGLE_VIEW_INDEX_HUGE_PENDING", 1000, minimum=200
        )
        if pending_extract >= huge_threshold:
            opened = float(
                getattr(self, "_single_view_index_defer_opened_ts", 0.0) or 0.0
            )
            if not opened:
                self._single_view_index_defer_opened_ts = time.time()
                opened = self._single_view_index_defer_opened_ts
            ceiling_s = _env_int(
                "RAWVIEWER_SINGLE_VIEW_INDEX_HUGE_CEILING_MS", 45000, minimum=5000
            ) / 1000.0
            if (time.time() - opened) < ceiling_s:
                return True
        return False

    def _run_metadata_index_prep(self, folder, corpus_files):
        if getattr(self, "_semantic_index_prep_in_progress", False):
            return
        self._semantic_index_prep_in_progress = True

        index = self._get_semantic_index()
        signals = SemanticIndexPrepSignals()
        self._semantic_index_prep_signals = signals

        signals.done.connect(
            lambda coverage, pending, face_pending, extract_pending=None: self._on_metadata_index_prep_done(
                folder, corpus_files, coverage, pending, face_pending, extract_pending
            )
        )
        signals.error.connect(self._on_semantic_index_prep_error)

        worker = _SemanticIndexPrepWorker(
            corpus_files, index, signals, metadata_only=True
        )
        _get_bg_thread_pool().start(worker)

    def _on_metadata_index_prep_done(
        self,
        folder,
        corpus_files,
        coverage,
        pending,
        face_pending,
        extract_pending=None,
    ):
        self._semantic_index_prep_in_progress = False
        if getattr(self, "current_folder", None) != folder:
            return

        import logging
        logger = logging.getLogger(__name__)
        # extract_pending MUST come from the background worker. Re-scanning
        # thousands of paths here freezes gallery tile paint (measured ~11s on
        # a 6880-file folder after Pending scan completed off-thread).
        if extract_pending is None:
            extract_pending = []
        logger.info(
            "[INDEX] Background metadata prep done. Broad pending=%d, need EXIF extract=%d",
            len(pending or []),
            len(extract_pending or []),
        )

        self._metadata_index_prep_complete = True
        self._metadata_index_extract_pending_count = len(extract_pending or [])

        if not extract_pending:
            self._metadata_index_run_done = True
            if not pending:
                logger.info("[INDEX] No pending metadata files. Index is up to date.")
            else:
                logger.info(
                    "[INDEX] Metadata-only pass: nothing to extract; skipping index worker (no ONNX probe)"
                )
            # Always finish the silent-metadata chain (marks session complete /
            # unlocks search). Do not leave this to a later embedding scan.
            self._finish_silent_metadata_without_worker(corpus_files)
            if getattr(self, "_gallery_search_user_wants_semantic", False):
                QTimer.singleShot(
                    0, lambda: self._start_user_semantic_indexing(corpus_files)
                )
            return

        self._metadata_index_deferred_corpus = corpus_files
        self._metadata_index_deferred_coverage = coverage
        self._metadata_index_deferred_pending = pending

        if self._should_defer_metadata_index_for_single_view(
            pending_extract=len(extract_pending)
        ):
            logger.info(
                "[INDEX] Deferring metadata worker after prep (single-view load pressure, pending=%d)",
                len(extract_pending),
            )
            t = getattr(self, "_metadata_index_defer_timer", None)
            if t is not None:
                t.start(
                    _env_int("RAWVIEWER_SINGLE_VIEW_INDEX_DEFER_MS", 4000, minimum=500)
                )
            return

        self._start_semantic_index_build_background(
            corpus_files=corpus_files,
            coverage=coverage,
            pending_files=pending,
            # Silent background pass should stay metadata-only; semantic pass starts
            # when user explicitly opens search.
            run_semantic_embeddings=False,
            show_search_progress=False,
        )
        if bool(getattr(self, "_gallery_search_user_wants_semantic", False)) or bool(
            getattr(self, "_defer_semantic_after_metadata", False)
        ):
            logger.info(
                "[INDEX] User requested semantic during metadata pass; semantic will start after metadata completion."
            )

    def _start_quick_folder_index(self, folder_path: str, start_path: str, token: int) -> None:
        """Background scandir to expand the fast-open navigation list before EXIF sort completes."""
        signals = QuickFolderIndexSignals()
        signals.ready.connect(self._on_quick_folder_index_ready)
        self._active_quick_folder_index_signals = signals
        viewer = self

        class _QuickFolderIndexWorker(QRunnable):
            def run(self_inner):
                import time

                try:
                    t0 = time.time()
                    files, stats, start_idx = viewer._build_quick_folder_image_list(
                        folder_path, start_path
                    )
                    scan_s = time.time() - t0
                    try:
                        from PyQt6 import sip
                        if not sip.isdeleted(signals):
                            signals.ready.emit(token, files, stats, start_idx, scan_s)
                    except Exception:
                        pass
                except Exception:
                    pass

        _get_bg_thread_pool().start(_QuickFolderIndexWorker())

    def _on_quick_folder_index_ready(
        self, token, image_files, file_stats, start_idx, scan_s
    ) -> None:
        if token != getattr(self, "_folder_load_generation", None):
            return
        if token != getattr(self, "_folder_fast_open_token", None):
            return
        if not image_files:
            return

        import logging

        logger = logging.getLogger(__name__)
        preserved = self.current_file_path
        self.image_files = list(image_files)
        self._semantic_search_corpus_files = list(image_files)
        self._gallery_search_index_session_key = None
        self._quick_folder_file_stats = file_stats
        self._rebuild_burst_groups()

        # --- OPTIMIZATION: INSTANT EXIF SORT IF FULLY CACHED ---
        try:
            idx = self._get_semantic_index()
            if idx and len(self.image_files) > 1:
                coverage = idx.get_index_coverage(self.image_files)
                if coverage and coverage.get("missing", 1) == 0:
                    logger.info("[FOLDER] Folder is 100% cached in database. Performing instant EXIF sort refinement on main thread.")
                    newest_first = self.get_sort_preference()
                    sorted_files, merged_meta = self.sort_files_by_capture_time(
                        self.image_files,
                        newest_first=newest_first,
                        file_stats=file_stats,
                        probe_uncached=False,
                    )
                    if sorted_files:
                        self.image_files = sorted_files
                        self._semantic_search_corpus_files = list(sorted_files)
                        self._folder_sort_refinement_applied_token = token
                        if merged_meta:
                            if not self._gallery_bulk_metadata:
                                self._gallery_bulk_metadata = {}
                            self._gallery_bulk_metadata.update(merged_meta)
                            self._bump_gallery_rating_presence_epoch()
        except Exception as e:
            logger.warning(f"[FOLDER] Instant EXIF sort failed: {e}")

        if preserved:
            preserved_norm = _norm_path(preserved)
            idx = start_idx
            for i, fp in enumerate(image_files):
                if _norm_path(fp) == preserved_norm:
                    idx = i
                    break
            self.current_file_index = idx
            self.current_file_path = image_files[idx]
        else:
            self.current_file_index = start_idx
            self.current_file_path = image_files[start_idx]

        self._sync_filmstrip_to_folder()
        self.update_status_bar()
        self._refresh_image_counter()
        logger.info(
            "[FOLDER] Quick folder index ready: %d images in %.3fs (navigation enabled)",
            len(image_files),
            scan_s,
        )
        if len(image_files) > 1:
            self._schedule_folder_sort_refinement(token, file_stats)
        if self._should_defer_nav_preload():
            self._schedule_post_session_restore_prefetch()
        else:
            with contextlib.suppress(Exception):
                QTimer.singleShot(
                    0,
                    lambda: self._queue_display_tier_prefetch_for_paths(
                        self._paths_within_nav_prefetch_radius(self.current_file_index)
                    ),
                )
