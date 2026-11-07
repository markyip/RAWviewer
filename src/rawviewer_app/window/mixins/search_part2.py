"""search mixin, part 2 of 4 (17 methods, 898 lines)."""
from __future__ import annotations

import contextlib
import logging
import sys
import time

from PyQt6.QtCore import QEasingCurve, QRunnable, QTimer

from core.config import _env_int, _env_true
from rawviewer_app.signals import SemanticIndexPrepSignals, SemanticIndexSignals
from rawviewer_app.workers import _SemanticIndexPrepWorker

from .. import lazy
from ..qt_helpers import _get_bg_thread_pool

lazy.register(sys.modules[__name__])


class SearchPart2:
    def _set_search_panel_expanded(self, expanded: bool, animate: bool = True):
        if not hasattr(self, "search_expand_container") or self.search_expand_container is None:
            return
        if expanded:
            full = (getattr(self, "_gallery_search_status_full", "") or "").strip()
            if full:
                self._search_panel_target_width = self._gallery_search_status_panel_width(full)
            else:
                self._search_panel_target_width = int(
                    getattr(self, "_search_panel_target_width_idle", 310) or 310
                )
        target = int(self._search_panel_target_width if expanded else 0)
        was_expanded = bool(getattr(self, "_search_panel_expanded", False))
        self._search_panel_expanded = bool(expanded)
        container = self.search_expand_container

        if not expanded and hasattr(self, "gallery_search_input") and self.gallery_search_input is not None:
                with contextlib.suppress(Exception):
                    self.gallery_search_input.clearFocus()

        if expanded:
            # Child min-width must not fight the 0→target slide.
            self._release_gallery_search_panel_width()
            if hasattr(self, "search_expand_layout") and hasattr(self, "gallery_search_panel"):
                with contextlib.suppress(Exception):
                    self.search_expand_layout.setCurrentWidget(self.gallery_search_panel)

        # Visible footprint is full content width; reveal amount is the mask.
        content_now = int(getattr(self, "_search_panel_anim_content_width", 0) or 0)
        if content_now <= 0:
            content_now = int(
                getattr(self, "_search_panel_target_width", 310) or 310
            )
        current_w = content_now if (container.isVisible() and was_expanded) else 0
        if (
            was_expanded == bool(expanded)
            and (
                (expanded and current_w == target)
                or (not expanded and not container.isVisible())
            )
            and not self._search_panel_animation_running()
        ):
            if not expanded:
                self._search_panel_anim_content_width = 0
                self._place_search_expand_overlay(0)
            return

        anim = getattr(self, "_search_panel_animation", None)
        if anim is not None:
            with contextlib.suppress(Exception):
                anim.finished.disconnect()
            with contextlib.suppress(Exception):
                anim.stop()
            self._search_panel_animation = None

        # Clip-reveal: keep the field laid out at the final width for the whole slide.
        if expanded:
            self._search_panel_anim_content_width = max(1, int(target))
        else:
            self._search_panel_anim_content_width = max(1, int(content_now))

        if not animate:
            if expanded:
                self._search_panel_anim_content_width = max(1, int(target))
                self._place_search_expand_overlay(target)
                self._search_panel_anim_content_width = 0
                self._reapply_gallery_search_panel_width_from_status()
                self._place_search_expand_overlay(target)
                with contextlib.suppress(Exception):
                    self.search_expand_container.clearMask()
            else:
                self._search_panel_anim_content_width = 0
                self._place_search_expand_overlay(0)
                self._release_gallery_search_panel_width()
                if not self._is_gallery_search_indexing_active():
                    self._reset_gallery_search_to_idle()
                QTimer.singleShot(0, self._restore_keyboard_focus)
            self._run_pending_search_index_after_expand()
            return

        from PyQt6.QtCore import QVariantAnimation

        start_w = current_w if was_expanded or container.isVisible() else 0
        if expanded and start_w <= 0:
            # Seed a zero-reveal mask so the first frame has a valid anchor.
            self._place_search_expand_overlay(0)
            start_w = 0
        elif not expanded and start_w <= 0:
            start_w = int(self._search_panel_anim_content_width)

        panel_anim = QVariantAnimation(self)
        panel_anim.setDuration(200)
        panel_anim.setStartValue(int(start_w))
        panel_anim.setEndValue(int(target))
        panel_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        panel_anim.valueChanged.connect(
            lambda v: self._place_search_expand_overlay(int(v))
        )
        self._search_panel_animation = panel_anim

        def _finish(exp=bool(expanded), tgt=int(target)):
            self._search_panel_animation = None
            if exp:
                final = max(
                    int(tgt),
                    int(getattr(self, "_search_panel_target_width", tgt) or tgt),
                )
                self._search_panel_target_width = final
                self._search_panel_anim_content_width = final
                self._place_search_expand_overlay(final)
                self._search_panel_anim_content_width = 0
                self._reapply_gallery_search_panel_width_from_status()
                final = int(
                    getattr(self, "_search_panel_target_width", final) or final
                )
                self._place_search_expand_overlay(final)
                with contextlib.suppress(Exception):
                    self.search_expand_container.clearMask()
                self._run_pending_search_index_after_expand()
            else:
                self._search_panel_anim_content_width = 0
                self._place_search_expand_overlay(0)
                self._release_gallery_search_panel_width()
                if not self._is_gallery_search_indexing_active():
                    self._reset_gallery_search_to_idle()
                QTimer.singleShot(0, self._restore_keyboard_focus)

        panel_anim.finished.connect(_finish)
        panel_anim.start()

    def _run_pending_search_index_after_expand(self) -> None:
        """Start indexing only after the open slide finishes (avoids mid-slide relayout)."""
        corpus = getattr(self, "_pending_search_index_corpus", None)
        self._pending_search_index_corpus = None
        if corpus is None:
            return
        if not corpus:
            self._set_gallery_search_status("No images available for semantic search")
            return
        self._start_user_semantic_indexing(corpus)

    def _is_semantic_index_ready(self, corpus_files):
        now = time.time()
        cache = getattr(self, "_semantic_coverage_cache", None)
        if cache and (now - getattr(self, "_semantic_coverage_cache_ts", 0.0)) < 2.0:
            same_folder = cache.get("folder") == getattr(self, "current_folder", None)
            same_count = cache.get("count") == len(corpus_files)
            if same_folder and same_count:
                return cache.get("coverage", {})
        idx = self._get_semantic_index()
        coverage = idx.get_index_coverage(corpus_files)
        self._semantic_coverage_cache = {
            "folder": getattr(self, "current_folder", None),
            "count": len(corpus_files),
            "coverage": coverage,
        }
        self._semantic_coverage_cache_ts = now
        return coverage

    def _run_semantic_index_prep(self, corpus_files):
        """Run coverage and pending files checks in a background prep worker."""
        if getattr(self, "_semantic_index_prep_in_progress", False):
            return

        self._semantic_index_prep_in_progress = True
        self.status_bar.showMessage("Preparing search index...")

        index = self._get_semantic_index()
        signals = SemanticIndexPrepSignals()
        self._semantic_index_prep_signals = signals

        signals.done.connect(lambda coverage, pending, face_pending, extract_pending=None: self._on_semantic_index_prep_done(corpus_files, coverage, pending, face_pending))
        signals.error.connect(self._on_semantic_index_prep_error)

        worker = _SemanticIndexPrepWorker(corpus_files, index, signals)
        _get_bg_thread_pool().start(worker)

    def _on_semantic_index_prep_done(self, corpus_files, coverage, pending, face_pending):
        self._semantic_index_prep_in_progress = False
        if bool(getattr(self, "_gallery_search_user_wants_semantic", False)) or bool(
            getattr(self, "_defer_semantic_after_metadata", False)
        ):
            self._on_user_semantic_prep_done(
                corpus_files, coverage, pending, face_pending
            )
            return

        # If there are no pending files and no face pending files, mark complete
        needs_semantic = len(pending) > 0
        needs_face = face_pending > 0

        if not needs_semantic and not needs_face:
            self._mark_gallery_search_index_session_complete(corpus_files)
            self._reset_gallery_search_to_idle()
            self.status_bar.showMessage("", 0)
            return

        if needs_semantic:
            user_semantic = self._user_wants_semantic_embeddings_now()
            from search import semantic_embeddings_enabled

            self._start_semantic_index_build_background(
                corpus_files,
                coverage=coverage,
                pending_files=pending,
                run_semantic_embeddings=user_semantic and semantic_embeddings_enabled(),
                show_search_progress=bool(
                    user_semantic
                    and getattr(self, "_gallery_search_show_index_progress", False)
                ),
            )
            return

        if needs_face:
            self._start_face_index_background(corpus_files)
        else:
            self._mark_gallery_search_index_session_complete(corpus_files)
            self._reset_gallery_search_to_idle()
            self.status_bar.showMessage("", 0)

    def _on_semantic_index_prep_error(self, err_msg):
        self._semantic_index_prep_in_progress = False
        self.status_bar.showMessage(f"Index preparation error: {err_msg}", 5000)

    def _start_semantic_index_build_background(
        self,
        corpus_files,
        coverage=None,
        pending_files=None,
        run_semantic_embeddings=True,
        *,
        show_search_progress: bool = False,
    ):
        if self._semantic_indexing_in_progress:
            return

        if self._gallery_blocks_background_indexing():
            import logging

            logging.getLogger(__name__).info(
                "[INDEX] Deferring semantic index build (gallery thumbnail load in progress)"
            )
            QTimer.singleShot(
                _env_int("RAWVIEWER_SEMANTIC_INDEX_GALLERY_RETRY_MS", 3000, minimum=500),
                lambda: self._start_semantic_index_build_background(
                    corpus_files,
                    coverage=coverage,
                    pending_files=pending_files,
                    run_semantic_embeddings=run_semantic_embeddings,
                    show_search_progress=show_search_progress,
                ),
            )
            return

        pending_extract = len(pending_files or [])
        if self._should_defer_metadata_index_for_single_view(
            pending_extract=pending_extract
        ):
            import logging

            logging.getLogger(__name__).info(
                "[INDEX] Deferring semantic index build (single-view navigation/load pressure, pending=%d)",
                pending_extract,
            )
            QTimer.singleShot(
                _env_int("RAWVIEWER_SINGLE_VIEW_INDEX_DEFER_MS", 4000, minimum=500),
                lambda: self._start_semantic_index_build_background(
                    corpus_files,
                    coverage=coverage,
                    pending_files=pending_files,
                    run_semantic_embeddings=run_semantic_embeddings,
                    show_search_progress=show_search_progress,
                ),
            )
            return

        # Phase A/B split (plan PR-3 step 3): even once the hard gate above clears,
        # a background (non-user-initiated) pass still downgrades to metadata-only
        # while gallery tiles are actively decoding, or when
        # RAWVIEWER_GALLERY_INDEX_CACHE_ONLY forces it. Metadata-only is cheap
        # (exifread, no RAW decode) and marks files semantic_ready=1,dim=0 via
        # _mark_metadata_ready_without_embedding, so the scheduled Phase B retry
        # below correctly re-picks them up for the heavy MobileCLIP/decode pass.
        # Skipped for user-initiated searches (show_search_progress=True) -- a user
        # actively waiting on the search progress bar shouldn't be silently delayed.
        # Bounded by the same absolute ceiling as the hard gate above (opened_ts +
        # RAWVIEWER_GALLERY_INDEX_FORCE_AFTER_MS) so continuous scrolling cannot
        # starve embeddings -- and thus search -- forever.
        gallery_deferral_opened = float(
            getattr(self, "_gallery_deferral_opened_ts", 0.0) or 0.0
        )
        within_force_ceiling = (
            not gallery_deferral_opened
            or (time.time() - gallery_deferral_opened)
            < _env_int("RAWVIEWER_GALLERY_INDEX_FORCE_AFTER_MS", 60000, minimum=10000) / 1000.0
        )
        if (
            run_semantic_embeddings
            and not show_search_progress
            and (
                _env_true("RAWVIEWER_GALLERY_INDEX_CACHE_ONLY")
                or (within_force_ceiling and self._gallery_has_thumbnail_load_pressure())
            )
        ):
            import logging

            logging.getLogger(__name__).info(
                "[INDEX] Phase A (metadata-only): gallery thumbnail load pressure; "
                "deferring semantic embeddings to a Phase B retry"
            )
            QTimer.singleShot(
                _env_int("RAWVIEWER_GALLERY_INDEX_PHASE_B_RETRY_MS", 4000, minimum=500),
                lambda: self._start_semantic_index_build_background(
                    corpus_files,
                    coverage=coverage,
                    pending_files=pending_files,
                    run_semantic_embeddings=True,
                    show_search_progress=show_search_progress,
                ),
            )
            run_semantic_embeddings = False

        if coverage is None or pending_files is None:
            self._run_semantic_index_prep(corpus_files)
            return

        if show_search_progress:
            self._gallery_search_show_index_progress = True
        if run_semantic_embeddings and show_search_progress:
            self._semantic_index_pass_kind = "user_semantic"
        elif not run_semantic_embeddings and not show_search_progress:
            self._semantic_index_pass_kind = "silent_metadata"
        else:
            self._semantic_index_pass_kind = "metadata_pass"
        with contextlib.suppress(Exception):
            self._get_semantic_index().pause_indexing(False)

        index = self._get_semantic_index()
        try:
            unique_corpus, _ = index._filter_duplicate_raw_companions(corpus_files)
            fallback_total = len(unique_corpus)
        except Exception:
            fallback_total = len(corpus_files)
        total_files = int(coverage.get("total", fallback_total))

        if not run_semantic_embeddings and pending_files:
            import logging

            _log = logging.getLogger(__name__)
            extract_pending = index.get_metadata_extraction_pending_paths(pending_files)
            if len(extract_pending) != len(pending_files):
                _log.info(
                    "[INDEX] Metadata-only pass: %d of %d pending paths need on-disk EXIF extract",
                    len(extract_pending),
                    len(pending_files),
                )
            if not extract_pending:
                _log.info(
                    "[INDEX] Metadata-only pass: nothing to extract; skipping index worker (no ONNX probe)"
                )
                self._finish_silent_metadata_without_worker(corpus_files)
                return
            pending_files = extract_pending

        indexed_files = max(0, total_files - len(pending_files))
        if not pending_files:
            self._clear_stale_gallery_search_index_progress()
            with contextlib.suppress(Exception):
                self._start_face_index_background(corpus_files)
            if not self._is_gallery_search_indexing_active():
                self._mark_gallery_search_index_session_complete(corpus_files)
                self._reset_gallery_search_to_idle()
                if (
                    getattr(self, "_search_panel_expanded", False)
                    and hasattr(self, "gallery_search_input")
                    and self.gallery_search_input is not None
                ):
                    self.gallery_search_input.setFocus()
                    self._sync_gallery_search_input_editable()
            self.status_bar.showMessage("", 0)
            self._invalidate_semantic_coverage_cache()
            return
        folder = getattr(self, "current_folder", None)
        if folder:
            self._set_semantic_index_incomplete(folder, True)
        self._metadata_index_run_done = True
        self._pause_single_view_prefetch_for_indexing()
        token = time.time_ns()
        self._semantic_index_active_token = token
        self._semantic_indexing_in_progress = True
        # The throttle is no longer engaged for the whole run. Instead the worker
        # acquires it only around heavy work (neural pass / thumbnail warm / face
        # scan) and drops it while paused, so a metadata-only or gallery-paused
        # index never serializes thumbnail decode. See SemanticImageIndex.
        self._wire_index_load_throttle(index)
        self._semantic_index_progress_total = total_files
        self._semantic_index_last_progress = None
        self._semantic_index_progress_base = 0
        self._semantic_index_run_semantic_embeddings = run_semantic_embeddings
        signals = SemanticIndexSignals()
        self._semantic_index_signals = signals
        signals.progress.connect(self._on_semantic_index_progress)
        signals.done.connect(self._on_semantic_index_done)
        signals.error.connect(self._on_semantic_index_error)

        class _SemanticIndexWorker(QRunnable):
            def __init__(
                self_inner,
                token,
                files,
                index,
                signals,
                album_total,
                album_indexed_base,
                run_semantic_embeddings=True,
            ):
                super().__init__()
                self_inner.token = token
                self_inner.files = files
                self_inner.index = index
                self_inner.signals = signals
                self_inner.album_total = album_total
                self_inner.album_indexed_base = album_indexed_base
                self_inner.run_semantic_embeddings = run_semantic_embeddings

            def run(self_inner):
                try:
                    def _progress(done, total, message):
                        try:
                            from PyQt6 import sip
                            if not sip.isdeleted(self_inner.signals):
                                self_inner.signals.progress.emit(
                                    self_inner.token, done, total, str(message)
                                )
                        except Exception:
                            pass

                    defer_faces = self_inner.index._defer_face_scan_during_build()
                    result = self_inner.index.build_index(
                        self_inner.files,
                        progress_callback=_progress,
                        album_total=self_inner.album_total,
                        album_indexed_base=self_inner.album_indexed_base,
                        run_face_scan=not defer_faces,
                        run_semantic_embeddings=self_inner.run_semantic_embeddings,
                    )
                    result["faces_deferred"] = bool(
                        result.get("faces_deferred")
                        or (
                            defer_faces
                            and result.get("faces_pending", 0) > 0
                        )
                    )
                    try:
                        from PyQt6 import sip
                        if not sip.isdeleted(self_inner.signals):
                            self_inner.signals.done.emit(self_inner.token, result)
                    except Exception:
                        pass
                except Exception as e:
                    try:
                        from PyQt6 import sip
                        if not sip.isdeleted(self_inner.signals):
                            self_inner.signals.error.emit(self_inner.token, str(e))
                    except Exception:
                        pass

        if hasattr(self, "gallery_search_input") and self.gallery_search_input is not None:
            ph = self.gallery_search_input.placeholderText() or "Search gallery"
            if self._should_show_search_index_progress(ph):
                ph = "Search gallery"
            if not getattr(self, "_gallery_search_placeholder_saved", ""):
                self._gallery_search_placeholder_saved = ph
        if show_search_progress:
            label = "Semantic" if run_semantic_embeddings else "Metadata"
            self._set_gallery_search_status(
                self._format_index_progress(
                    label,
                    indexed_files,
                    self._semantic_index_progress_total,
                )
            )
        self._begin_indexing_load_throttle()
        worker = _SemanticIndexWorker(
            token,
            list(pending_files),
            index,
            signals,
            self._semantic_index_progress_total,
            indexed_files,
            run_semantic_embeddings,
        )
        _get_bg_thread_pool().start(worker)

    def _format_index_progress(self, stage: str, done: int, total: int) -> str:
        try:
            from search import format_index_progress

            return format_index_progress(stage, done, total)
        except Exception:
            total_i = max(0, int(total))
            done_i = max(0, min(int(done), total_i))
            if total_i <= 0:
                return stage
            return f"{stage}: {done_i}/{total_i}"

    def _show_live_metadata_progress_in_search(self, fallback: str) -> None:
        """Switch an already-running background metadata pass to visible progress.

        Background passes keep "Metadata: X/Y" out of the search field by design
        (_set_gallery_search_status suppresses those prefixes when
        _gallery_search_show_index_progress is False). Once the user opens the
        search panel they are actively waiting, so flip the flag -- subsequent
        progress events (every ~10 files) update the field live -- and seed it
        immediately with the last known counts rather than a static message the
        user cannot estimate completion from.
        """
        self._gallery_search_show_index_progress = True
        last = getattr(self, "_semantic_index_last_progress", None)
        total = int(getattr(self, "_semantic_index_progress_total", 0) or 0)
        if last and last[1] > 0:
            self._set_gallery_search_status(
                self._format_index_progress("Metadata", last[0], last[1])
            )
        elif total > 0:
            # Pass started but extraction hasn't begun (prep/DB-scan phase).
            self._set_gallery_search_status(
                self._format_index_progress("Metadata", 0, total)
            )
        else:
            self._set_gallery_search_status(fallback)

    def _should_show_search_index_progress(self, message: str) -> bool:
        msg = (message or "").strip()
        if not msg:
            return False
        lowered = msg.lower()
        if lowered.startswith("preparing thumbnail"):
            return True
        if lowered.startswith("downloading"):
            return True
        # Only the three search-index phases belong in the search field.
        return lowered.startswith(
            ("metadata:", "semantic:", "face:")
        )

    def _on_semantic_index_progress(self, token, i, n, message):
        if token != self._semantic_index_active_token:
            return

        # Dynamically load newly warmed thumbnails into gallery view to speed up scrolling/display
        if getattr(self, "view_mode", "") == "gallery":
            gj = getattr(self, "gallery_justified", None)
            if gj is not None and hasattr(gj, "_request_load_visible_images"):
                import time
                now = time.time()
                last_time = getattr(self, "_last_semantic_progress_reload_time", 0.0)
                if now - last_time >= 2.0:
                    self._last_semantic_progress_reload_time = now
                    gj._request_load_visible_images(100)

        msg = str(message or "").strip()
        pass_kind = getattr(self, "_semantic_index_pass_kind", None)
        show_progress = getattr(self, "_gallery_search_show_index_progress", False)

        # Remember the latest counts so opening the search panel mid-pass can
        # immediately show "Metadata: 1234/6881" instead of a static message.
        with contextlib.suppress(Exception):
            self._semantic_index_last_progress = (max(0, int(i)), max(0, int(n)))

        if not show_progress:
            # If search UI doesn't want progress, route standard indexing/thumbnail messages to status bar only
            if msg and self._should_show_search_index_progress(msg):
                self._set_gallery_search_status(msg)
            else:
                total = max(1, int(n) if n else self._semantic_index_progress_total)
                done = min(total, max(0, int(i)))
                label = "Semantic" if getattr(self, "_semantic_index_run_semantic_embeddings", True) else "Metadata"
                self._set_gallery_search_status(
                    self._format_index_progress(label, done, total)
                )
            return

        # Below is when show_progress is True (e.g. search UI is expanded/active)
        if pass_kind == "user_semantic" and msg.lower().startswith("metadata:"):
            return
        if msg and self._should_show_search_index_progress(msg):
            if pass_kind == "user_semantic" and not msg.lower().startswith("semantic:") and not msg.lower().startswith("face:") and not msg.lower().startswith("preparing thumbnail"):
                return
            self._set_gallery_search_status(msg)
            return
        total = max(1, int(n) if n else self._semantic_index_progress_total)
        done = min(total, max(0, int(i)))
        if pass_kind == "user_semantic":
            label = "Semantic"
        else:
            label = "Semantic" if getattr(self, "_semantic_index_run_semantic_embeddings", True) else "Metadata"
        self._set_gallery_search_status(
            self._format_index_progress(label, done, total)
        )

    def _on_semantic_index_done(self, token, result):
        if token != self._semantic_index_active_token:
            return
        pass_kind = getattr(self, "_semantic_index_pass_kind", None)
        self._semantic_indexing_in_progress = False
        if not getattr(self, "_face_indexing_in_progress", False):
            self._end_indexing_load_throttle()
        self._semantic_index_active_token = None
        self._semantic_index_signals = None
        self._semantic_index_progress_base = 0
        self._semantic_index_progress_total = 0
        self._semantic_index_pass_kind = None
        folder = getattr(self, "current_folder", None)
        corpus = self._gallery_search_index_corpus()
        self._invalidate_semantic_coverage_cache()

        if pass_kind == "silent_metadata":
            self._finish_silent_metadata_without_worker(corpus_files=None, skipped=False)
            return

        if pass_kind == "user_semantic":
            if folder:
                self._set_semantic_index_incomplete(folder, False)
            pending_emb = 0
            face_pending = 0
            try:
                if corpus:
                    index = self._get_semantic_index()
                    pending_emb = len(index.get_pending_embedding_paths(corpus) or [])
                    face_pending = int(index.get_face_pending_count(corpus) or 0)
            except Exception:
                pass
            if pending_emb > 0:
                QTimer.singleShot(0, lambda: self._start_user_semantic_indexing(corpus))
            elif face_pending > 0:
                self._chain_face_index_after_semantic(corpus, face_pending=face_pending)
            elif self._is_gallery_semantic_search_ready(corpus):
                self._finish_gallery_search_unlock(corpus)
            else:
                self._sync_gallery_search_input_editable()
            self._gallery_search_user_collapsed_while_busy = False
            return

        if folder:
            self._set_semantic_index_incomplete(folder, False)
        face_pending = 0
        if corpus:
            try:
                face_pending = self._get_semantic_index().get_face_pending_count(corpus)
            except Exception:
                face_pending = 0
        will_run_face = face_pending > 0
        if not will_run_face and not (
            getattr(self, "_gallery_search_user_wants_semantic", False)
            or getattr(self, "_defer_semantic_after_metadata", False)
        ):
            self._mark_gallery_search_index_session_complete(corpus)
            self._reset_gallery_search_to_idle()
        getattr(self, "_gallery_search_user_collapsed_while_busy", False)
        self._gallery_search_user_collapsed_while_busy = False
        if getattr(self, "_search_panel_expanded", False):
            self._set_gallery_search_input_visible()
            if hasattr(self, "gallery_search_input") and self.gallery_search_input is not None:
                self.gallery_search_input.setFocus()
        else:
            self._ensure_gallery_search_collapsed()
        self._sync_gallery_search_input_editable()
        if will_run_face:
            show = bool(getattr(self, "_gallery_search_show_index_progress", False))
            self._start_face_index_background(corpus, show_search_progress=show)

    def _start_face_index_background(self, corpus_files, *, show_search_progress: bool = True) -> None:
        """Second pass: face_count backfill after semantic search is usable."""
        import time

        logger = logging.getLogger(__name__)
        from search import face_scan_enabled
        if not face_scan_enabled():
            logger.info("[INDEX][FACE] Skip start: face scan is disabled")
            self._face_indexing_in_progress = False
            self._face_index_active_token = None
            self._face_index_signals = None
            corpus = self._gallery_search_index_corpus()
            if self._is_gallery_semantic_search_ready(corpus):
                self._finish_gallery_search_unlock(corpus)
            else:
                self._reset_gallery_search_to_idle()
            return

        if getattr(self, "view_mode", "") == "gallery":
            defer_until = float(getattr(self, "_gallery_indexing_deferred_until", 0.0) or 0.0)
            gj = getattr(self, "gallery_justified", None)
            warmup = (
                gj is not None
                and hasattr(gj, "_gallery_warmup_active")
                and gj._gallery_warmup_active()
            )
            if warmup or time.time() < defer_until:
                logger.info("[INDEX][FACE] Skip start: gallery warmup in progress")
                return
        if getattr(self, "_face_indexing_in_progress", False):
            logger.info("[INDEX][FACE] Skip start: already running")
            return
        if getattr(self, "_semantic_indexing_in_progress", False):
            logger.info("[INDEX][FACE] Skip start: semantic indexing is running")
            return
        try:
            index = self._get_semantic_index()
            pending = index.get_face_pending_count(corpus_files)
            if pending <= 0:
                logger.info(
                    "[INDEX][FACE] Skip start: nothing pending "
                    f"(pending={int(pending or 0)}, corpus_size={len(corpus_files or [])})"
                )
                return
        except Exception as e:
            logger.warning(f"[INDEX][FACE] Failed to query pending face count: {e}")
            return

        token = time.time_ns()
        self._face_index_active_token = token
        self._face_indexing_in_progress = True
        self._begin_indexing_load_throttle()
        try:
            unique_corpus, _ = index._filter_duplicate_raw_companions(corpus_files)
            total_files = len(unique_corpus)
        except Exception:
            total_files = len(corpus_files)
        signals = SemanticIndexSignals()
        self._face_index_signals = signals
        signals.progress.connect(self._on_face_index_progress)
        signals.done.connect(self._on_face_index_done)
        signals.error.connect(self._on_face_index_error)

        class _FaceIndexWorker(QRunnable):
            def __init__(self_inner, token, files, index, signals, album_total):
                super().__init__()
                self_inner.token = token
                self_inner.files = files
                self_inner.index = index
                self_inner.signals = signals
                self_inner.album_total = album_total

            def run(self_inner):
                try:
                    def _progress(done, total, message):
                        try:
                            from PyQt6 import sip
                            if not sip.isdeleted(self_inner.signals):
                                self_inner.signals.progress.emit(
                                    self_inner.token, done, total, str(message)
                                )
                        except Exception:
                            pass

                    count = self_inner.index.backfill_face_counts(
                        self_inner.files,
                        progress_callback=_progress,
                        album_total=self_inner.album_total,
                        album_indexed_base=0,
                    )
                    try:
                        from PyQt6 import sip
                        if not sip.isdeleted(self_inner.signals):
                            self_inner.signals.done.emit(
                                self_inner.token, {"faces_scanned": count}
                            )
                    except Exception:
                        pass
                except Exception as e:
                    try:
                        from PyQt6 import sip
                        if not sip.isdeleted(self_inner.signals):
                            self_inner.signals.error.emit(self_inner.token, str(e))
                    except Exception:
                        pass

        self._set_gallery_search_status(
            self._format_index_progress("Face", 0, pending)
        ) if show_search_progress else None
        if not show_search_progress and hasattr(self, "status_bar"):
            with contextlib.suppress(Exception):
                self.status_bar.showMessage(
                    f"Face scan in background -- {pending} photos remaining",
                    4000,
                )
        self._face_index_show_search_progress = bool(show_search_progress)
        logger.info(
            "[INDEX][FACE] Start worker "
            f"(pending={int(pending or 0)}, corpus_size={len(corpus_files or [])})"
        )
        worker = _FaceIndexWorker(token, list(corpus_files), index, signals, total_files)
        _get_bg_thread_pool().start(worker)

    def _on_face_index_progress(self, token, i, n, message):
        if token != getattr(self, "_face_index_active_token", None):
            return
        if not getattr(self, "_face_index_show_search_progress", True):
            return

        msg = str(message or "").strip()
        if msg and self._should_show_search_index_progress(msg):
            self._set_gallery_search_status(msg)
            return
        total = max(1, int(n) if n else 1)
        done = min(total, max(0, int(i)))
        self._set_gallery_search_status(
            self._format_index_progress("Face", done, total)
        )

    def _on_face_index_done(self, token, result):
        logger = logging.getLogger(__name__)
        if token != getattr(self, "_face_index_active_token", None):
            return
        self._face_indexing_in_progress = False
        if not getattr(self, "_semantic_indexing_in_progress", False):
            self._end_indexing_load_throttle()
        self._face_index_active_token = None
        self._face_index_signals = None
        corpus = self._gallery_search_index_corpus()
        self._invalidate_semantic_coverage_cache()

        pending_emb = 0
        face_pending = 0
        try:
            index = self._get_semantic_index()
            pending_emb = len(index.get_pending_embedding_paths(corpus) or [])
            face_pending = int(index.get_face_pending_count(corpus) or 0)
        except Exception as exc:
            logger.warning("[INDEX][FACE] Done handler pending check failed: %s", exc)

        logger.info(
            "[INDEX][FACE] Done: pending_emb=%d pending_face=%d corpus=%d",
            pending_emb,
            face_pending,
            len(corpus or []),
        )

        if pending_emb > 0:
            self._gallery_search_show_index_progress = True
            self._start_user_semantic_indexing(corpus)
        elif self._is_gallery_semantic_search_ready(corpus):
            QTimer.singleShot(0, lambda c=list(corpus or []): self._finish_gallery_search_unlock(c))
        elif face_pending > 0:
            self._sync_gallery_search_input_editable()
        else:
            QTimer.singleShot(0, lambda c=list(corpus or []): self._finish_gallery_search_unlock(c))
        self._gallery_search_user_collapsed_while_busy = False

    def _on_face_index_error(self, token, error):
        if token != getattr(self, "_face_index_active_token", None):
            return
        self._face_indexing_in_progress = False
        if not getattr(self, "_semantic_indexing_in_progress", False):
            self._end_indexing_load_throttle()
        self._face_index_active_token = None
        self._face_index_signals = None
        with contextlib.suppress(Exception):
            self._reset_gallery_search_to_idle()
        self._sync_gallery_search_input_editable()
        self._invalidate_semantic_coverage_cache()
        self.status_bar.showMessage(f"Face search failed: {error}", 5000)

    def _on_semantic_index_error(self, token, error):
        if token != self._semantic_index_active_token:
            return
        self._semantic_indexing_in_progress = False
        if not getattr(self, "_face_indexing_in_progress", False):
            self._end_indexing_load_throttle()
        self._semantic_index_active_token = None
        self._semantic_index_signals = None
        self._semantic_index_progress_base = 0
        self._semantic_index_progress_total = 0
        with contextlib.suppress(Exception):
            self._restore_gallery_search_placeholder()
        user_hid = getattr(self, "_gallery_search_user_collapsed_while_busy", False)
        if not user_hid and getattr(self, "_search_panel_expanded", False):
            self._set_gallery_search_input_visible()
        self._set_gallery_search_status("Semantic indexing failed")
        self._gallery_search_user_collapsed_while_busy = False
        self._sync_gallery_search_input_editable()
        self.status_bar.showMessage(f"Semantic indexing failed: {error}", 5000)
