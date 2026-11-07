"""search mixin, part 3 of 4 (24 methods, 873 lines)."""
from __future__ import annotations

import contextlib
import logging
import os
import sys
import time

from PyQt6.QtCore import QRunnable
from PyQt6.QtWidgets import QInputDialog, QMessageBox

from core.config import safe_print
from rawviewer_app.signals import SemanticIndexPrepSignals, SemanticSearchQuerySignals, SemanticSearchResortSignals
from rawviewer_app.workers import SemanticAssetDownloadSignals, _SemanticIndexPrepWorker

from .. import lazy
from ..paths import _require_str_path
from ..qt_helpers import _get_bg_thread_pool

lazy.register(sys.modules[__name__])


class SearchPart3:
    def _start_semantic_asset_download_background(self, corpus_files):
        if self._semantic_asset_download_in_progress:
            return
        index = self._get_semantic_index()
        token = time.time_ns()
        self._semantic_asset_download_in_progress = True
        self._gallery_search_show_index_progress = True
        signals = SemanticAssetDownloadSignals()
        self._semantic_asset_download_signals = signals
        signals.progress.connect(self._on_semantic_asset_download_progress)
        signals.done.connect(self._on_semantic_asset_download_done)
        signals.error.connect(self._on_semantic_asset_download_error)

        class _SemanticAssetDownloadWorker(QRunnable):
            def __init__(self_inner, token, index, files, signals):
                super().__init__()
                self_inner.token = token
                self_inner.index = index
                self_inner.files = files
                self_inner.signals = signals

            def run(self_inner):
                try:
                    def _progress(pct, message=None):
                        from core.download_progress import download_status_text

                        pct_val = max(0, min(100, int(pct)))
                        text = (message or download_status_text(pct_val)).strip()
                        try:
                            from PyQt6 import sip
                            if not sip.isdeleted(self_inner.signals):
                                self_inner.signals.progress.emit(self_inner.token, pct_val, text)
                        except Exception:
                            pass

                    path = self_inner.index.download_semantic_backend_assets(
                        progress_callback=_progress
                    )
                    try:
                        from PyQt6 import sip
                        if not sip.isdeleted(self_inner.signals):
                            self_inner.signals.done.emit(self_inner.token, path, list(self_inner.files))
                    except Exception:
                        pass
                except Exception as e:
                    try:
                        from PyQt6 import sip
                        if not sip.isdeleted(self_inner.signals):
                            self_inner.signals.error.emit(self_inner.token, str(e))
                    except Exception:
                        pass

        self._set_gallery_search_status("Downloading... 0%")
        worker = _SemanticAssetDownloadWorker(token, index, list(corpus_files), signals)
        _get_bg_thread_pool().start(worker)

    def _on_semantic_asset_download_progress(self, token, pct, message):
        if not self._semantic_asset_download_in_progress:
            return
        short = (message or "").strip() or f"Downloading... {int(pct)}%"
        self._set_gallery_search_status(short)

    def _on_semantic_asset_download_done(self, token, asset_path, corpus_files):
        if not self._semantic_asset_download_in_progress:
            return
        self._semantic_asset_download_in_progress = False
        self._semantic_asset_download_signals = None
        dialog = getattr(self, "_mobileclip_download_dialog", None)
        if dialog is not None:
            dialog.show_download_complete()
        self._mobileclip_download_dialog = None
        if corpus_files:
            self._start_user_semantic_indexing(corpus_files)

    def _on_semantic_asset_download_error(self, token, error):
        if not self._semantic_asset_download_in_progress:
            return
        self._semantic_asset_download_in_progress = False
        self._semantic_asset_download_signals = None
        dialog = getattr(self, "_mobileclip_download_dialog", None)
        if dialog is not None:
            dialog.show_download_error(error)
        else:
            self._set_gallery_search_status("Asset download failed")
            self.status_bar.showMessage(f"Asset download failed: {error}", 7000)
        self._gallery_search_user_collapsed_while_busy = False

    def _semantic_disabled_hint_texts(self) -> tuple[str, str]:
        """(status-bar notice, tooltip) for the EXIF-only search state.

        Lite intentionally ships without AI search (no bundled models, no
        indexing overhead) -- never suggest the enable flag there. The env-var
        hint is only for Plus builds / dev runs where the flag is simply off.
        """
        try:
            from core.profile import is_lite_build

            lite = is_lite_build()
        except Exception:
            lite = False
        if lite:
            return (
                "Standard: EXIF, filename & place search -- no AI",
                "Standard searches EXIF metadata, filenames, and GPS places.\n"
                "Install Plus for AI semantic search.",
            )
        return (
            "EXIF, filename & place search -- AI off",
            "Queries match EXIF metadata, filenames, and GPS places.\n"
            "Enable RAWVIEWER_ENABLE_SEMANTIC_SEARCH=1 for AI search.",
        )

    def _notify_semantic_embeddings_disabled(self) -> None:
        """Tell the user (once per session) that search is EXIF-only.

        Fires when the user explicitly opens semantic search while
        RAWVIEWER_ENABLE_SEMANTIC_SEARCH is off. Without this, the search field
        unlocks looking fully functional and AI queries just return nothing.
        """
        import logging

        logging.getLogger(__name__).info(
            "[INDEX][USER] Semantic (AI) search disabled "
            "(RAWVIEWER_ENABLE_SEMANTIC_SEARCH is off); search is EXIF/filename-only"
        )
        if getattr(self, "_semantic_disabled_notice_shown", False):
            return
        self._semantic_disabled_notice_shown = True
        notice, tooltip = self._semantic_disabled_hint_texts()
        if hasattr(self, "status_bar"):
            with contextlib.suppress(Exception):
                self.status_bar.showMessage(notice, 10000)
        inp = getattr(self, "gallery_search_input", None)
        if inp is not None:
            with contextlib.suppress(Exception):
                inp.setToolTip(tooltip)

    def _start_user_semantic_indexing(self, corpus_files) -> None:
        """Run or resume semantic embeddings after the user opens search (metadata may already be silent)."""
        logger = logging.getLogger(__name__)
        corpus_files = list(corpus_files or self._gallery_search_index_corpus() or [])
        if not corpus_files:
            logger.info("[INDEX][USER] Skip semantic start: empty corpus")
            self._set_gallery_search_status("No images available for semantic search")
            return
        logger.info(
            "[INDEX][USER] Semantic start requested: corpus=%d wants=%s defer=%s semantic_in_progress=%s prep_in_progress=%s",
            len(corpus_files),
            bool(getattr(self, "_gallery_search_user_wants_semantic", False)),
            bool(getattr(self, "_defer_semantic_after_metadata", False)),
            bool(getattr(self, "_semantic_indexing_in_progress", False)),
            bool(getattr(self, "_semantic_index_prep_in_progress", False)),
        )

        from search import semantic_embeddings_enabled

        if not getattr(self, "_defer_semantic_after_metadata", False) and self._is_gallery_semantic_search_ready(corpus_files):
                logger.info("[INDEX][USER] Skip semantic start: already ready")
                self._finish_gallery_search_unlock(corpus_files)
                return

        if getattr(self, "_semantic_indexing_in_progress", False):
            if getattr(self, "_semantic_index_pass_kind", None) == "silent_metadata":
                self._show_live_metadata_progress_in_search(
                    "Preparing metadata in background…"
                )
            logger.info("[INDEX][USER] Skip semantic start: semantic indexing already in progress")
            self._sync_gallery_search_input_editable()
            return

        if getattr(self, "_semantic_index_prep_in_progress", False):
            logger.info("[INDEX][USER] Skip semantic start: prep worker already in progress")
            self._sync_gallery_search_input_editable()
            return

        if getattr(self, "_semantic_asset_download_in_progress", False):
            logger.info("[INDEX][USER] Skip semantic start: asset download in progress")
            self._sync_gallery_search_input_editable()
            return

        if getattr(self, "_face_indexing_in_progress", False):
            logger.info("[INDEX][USER] Skip semantic start: face indexing in progress")
            self._sync_gallery_search_input_editable()
            return

        if not semantic_embeddings_enabled():
            # The user explicitly asked for semantic search but embeddings are
            # disabled (RAWVIEWER_ENABLE_SEMANTIC_SEARCH unset/0 -- the default for
            # dev runs; only frozen/installed Full builds default it on). Say so
            # instead of silently unlocking an EXIF-only field: this exact silence
            # cost a debugging session when run_debug_full.bat lacked the flag and
            # 'lion'/'plane' queries just returned nothing with no explanation.
            self._notify_semantic_embeddings_disabled()
            if self._is_gallery_semantic_search_ready(corpus_files):
                self._finish_gallery_search_unlock(corpus_files)
            elif getattr(self, "_metadata_index_run_done", False):
                # Metadata may be complete while face indexing is still pending.
                try:
                    face_pending = int(self._get_semantic_index().get_face_pending_count(corpus_files) or 0)
                except Exception:
                    face_pending = 0
                if face_pending > 0:
                    self._gallery_search_show_index_progress = True
                    self._set_gallery_search_status(
                        self._format_index_progress("Face", 0, face_pending)
                    )
                    self._start_face_index_background(corpus_files)
                else:
                    self._finish_gallery_search_unlock(corpus_files)
            else:
                self._show_live_metadata_progress_in_search("Indexing metadata…")
                self._schedule_silent_metadata_index()
            self._sync_gallery_search_input_editable()
            return

        folder = getattr(self, "current_folder", None)
        if folder and self._is_semantic_index_incomplete(folder):
            self._set_semantic_index_incomplete(folder, True)

        try:
            index = self._get_semantic_index()
            if not index.semantic_backend_available():
                backend_error = index.semantic_backend_error()
                if "Missing MobileCLIP" in backend_error and index.mobileclip_supports_hub_download():
                    if not self._prompt_mobileclip_download(corpus_files):
                        self._set_gallery_search_status("EXIF search only (semantic models not downloaded)")
                else:
                    self._set_gallery_search_status(f"EXIF search only. Backend: {backend_error}")
                return
        except Exception as e:
            self._set_gallery_search_status(f"Search error: {e}")
            return

        self._run_semantic_index_prep_for_search(corpus_files)

    def _run_semantic_index_prep_for_search(self, corpus_files) -> None:
        if getattr(self, "_semantic_index_prep_in_progress", False):
            return
        self._semantic_index_prep_in_progress = True
        index = self._get_semantic_index()
        signals = SemanticIndexPrepSignals()
        self._semantic_index_prep_signals = signals
        signals.done.connect(
            lambda coverage, pending, face_pending, extract_pending=None: self._on_user_semantic_prep_done(
                corpus_files, coverage, pending, face_pending
            )
        )
        signals.error.connect(self._on_semantic_index_prep_error)
        _get_bg_thread_pool().start(
            _SemanticIndexPrepWorker(corpus_files, index, signals)
        )

    def _on_user_semantic_prep_done(self, corpus_files, coverage, pending, face_pending):
        logger = logging.getLogger(__name__)
        self._semantic_index_prep_in_progress = False
        wants = bool(getattr(self, "_gallery_search_user_wants_semantic", False))
        defer = bool(getattr(self, "_defer_semantic_after_metadata", False))
        if not (wants or defer):
            expanded = bool(getattr(self, "_search_panel_expanded", False))
            pending_n = len(pending or [])
            if not (expanded and pending_n > 0):
                logger.info(
                    "[INDEX][USER] Prep done ignored: wants=%s defer=%s expanded=%s pending=%d",
                    wants,
                    defer,
                    expanded,
                    pending_n,
                )
                return
            wants = True
            self._gallery_search_user_wants_semantic = True
        # Consume deferred intent only after prep callback is reached.
        if getattr(self, "_defer_semantic_after_metadata", False):
            self._defer_semantic_after_metadata = False

        from search import semantic_embeddings_enabled

        if not semantic_embeddings_enabled():
            self._notify_semantic_embeddings_disabled()
            if self._is_gallery_semantic_search_ready(corpus_files):
                self._finish_gallery_search_unlock(corpus_files)
                return
            if int(face_pending or 0) > 0:
                self._chain_face_index_after_semantic(
                    corpus_files, face_pending=int(face_pending)
                )
                return
            self._set_gallery_search_status("Indexing metadata…")
            self._sync_gallery_search_input_editable()
            return

        if not pending:
            if int(face_pending or 0) > 0:
                self._chain_face_index_after_semantic(
                    corpus_files, face_pending=int(face_pending)
                )
                return
            self._finish_gallery_search_unlock(corpus_files)
            return

        work = list(pending or [])
        if work:
            self._gallery_search_show_index_progress = True
            self._start_semantic_index_build_background(
                corpus_files,
                coverage=coverage,
                pending_files=work,
                run_semantic_embeddings=True,
                show_search_progress=True,
            )
            self._sync_gallery_search_input_editable()
            return

        self._finish_gallery_search_unlock(corpus_files)
        self._sync_gallery_search_input_editable()

    def _on_search_bottom_clicked(self):
        logger = logging.getLogger(__name__)
        safe_print(
            f"[SEARCH_DEBUG] Search button clicked. Current expansion state: {getattr(self, '_search_panel_expanded', False)}"
        )

        if self._search_panel_expanded:
            logger.info("[SEARCH][BUTTON] expanded -> collapse panel")
            safe_print("[SEARCH_DEBUG] Collapsing search panel.")
            self._pending_search_index_corpus = None
            if self._is_gallery_search_indexing_active():
                self._gallery_search_user_collapsed_while_busy = True
            self._gallery_search_user_wants_semantic = False
            self._defer_semantic_after_metadata = False
            self._set_search_panel_expanded(False, animate=True)
            self._sync_gallery_search_input_editable()
            return

        self._gallery_search_user_collapsed_while_busy = False
        self._gallery_search_user_wants_semantic = True
        self._defer_semantic_after_metadata = True
        self._gallery_search_index_session_key = None
        self._invalidate_semantic_coverage_cache()
        self._gallery_search_show_index_progress = True
        self._gallery_search_status_full = ""
        corpus_files = self._gallery_search_index_corpus()
        logger.info(
            "[SEARCH][BUTTON] collapsed -> expand and request semantic (corpus=%d)",
            len(corpus_files),
        )
        # Indexing/status text must wait until the open slide finishes -- otherwise
        # placeholder/width updates mid-animation create the halfway-pause look.
        self._pending_search_index_corpus = list(corpus_files or [])
        self._set_search_panel_expanded(True, animate=True)
        # Keep the width animation alone -- a second expand via
        # gallery-search input-visible would restart it and snap icons.
        self._sync_gallery_search_input_editable()

    def _build_semantic_index_current_folder(self):
        # Legacy menu path retained for compatibility; use same gallery button flow.
        self._on_search_bottom_clicked()

    def _semantic_search_current_folder(self):
        if not self.image_files:
            self.status_bar.showMessage("No images to search", 2500)
            return
        query, ok = QInputDialog.getText(
            self,
            "Semantic Search",
            (
                "Describe image + optional filters.\n"
                "Examples:\n"
                "  jet takeoff camera:sony iso<800 has:gps city:tokyo\n"
                "  date:2026:05 lens:70-200 country:jp\n"
                "  year>=2024 month=5"
            ),
        )
        if not ok:
            return
        query = (query or "").strip()
        if not query:
            return

        self._run_semantic_search_query(query)

    def _semantic_search_from_bar(self):
        if not self._is_gallery_semantic_search_ready():
            try:
                index = self._get_semantic_index()
                corpus = self._gallery_search_index_corpus()
                pending_emb = len(index.get_pending_embedding_paths(corpus))
                pending_face = int(index.get_face_pending_count(corpus) or 0)
                logging.getLogger(__name__).info(
                    "[SEARCH][READY] bar blocked: pending_embeddings=%d pending_face=%d corpus=%d",
                    pending_emb,
                    pending_face,
                    len(corpus or []),
                )
                if corpus and (pending_emb > 0 or pending_face > 0):
                    self._start_user_semantic_indexing(corpus)
            except Exception:
                pass
            self.status_bar.showMessage(
                "Semantic indexing in progress -- search unlocks when indexing completes.",
                5000,
            )
            return
        query = ""
        if hasattr(self, "gallery_search_input") and self.gallery_search_input is not None:
            query = (self.gallery_search_input.text() or "").strip()
        if not query:
            self._clear_semantic_search_results(silent=True)
            return
        if query == (self._last_semantic_query or ""):
            return
        self._run_semantic_search_query(query)

    def _on_gallery_search_text_changed(self, text):
        if (text or "").strip():
            return
        # Triggered by clear button "x" or manual delete-to-empty.
        if (
            self._semantic_search_backup_files
            or self._semantic_search_result_paths
            or (getattr(self, "_last_semantic_query", "") or "").strip()
        ):
            self._clear_semantic_search_results(silent=True)

    def _run_semantic_search_query(self, query: str):
        query = (query or "").strip()
        if not query:
            return
        if not self._is_gallery_semantic_search_ready():
            try:
                index = self._get_semantic_index()
                corpus = self._gallery_search_index_corpus()
                pending_emb = len(index.get_pending_embedding_paths(corpus))
                pending_face = int(index.get_face_pending_count(corpus) or 0)
                logging.getLogger(__name__).info(
                    "[SEARCH][READY] blocked: pending_embeddings=%d pending_face=%d corpus=%d",
                    pending_emb,
                    pending_face,
                    len(corpus or []),
                )
            except Exception:
                pass
            self.status_bar.showMessage(
                "Complete semantic indexing before searching the gallery.",
                5000,
            )
            return
        base_files = self._gallery_search_base_files()
        if not base_files:
            return
        if self._semantic_search_backup_files is None:
            self._semantic_search_backup_files = list(
                self._semantic_search_corpus_files
                if self._semantic_search_corpus_files
                else getattr(self, "image_files", []) or []
            )
        if getattr(self, "_semantic_indexing_in_progress", False):
            progress = (
                getattr(self, "_gallery_search_status_full", "") or ""
            ).strip()
            if progress:
                self.status_bar.showMessage(
                    f"Searching while indexing -- {progress}",
                    4500,
                )
            else:
                self.status_bar.showMessage(
                    "Searching while indexing -- EXIF/metadata filters apply now; "
                    "semantic ranking improves as indexing continues.",
                    4500,
                )
        else:
            self.status_bar.showMessage("Running search...")

        token = getattr(self, "_semantic_search_query_generation", 0) + 1
        self._semantic_search_query_generation = token

        signals = getattr(self, "_semantic_search_query_signals", None)
        if signals is None:
            signals = SemanticSearchQuerySignals()
            signals.ready.connect(self._apply_semantic_search_query_result)
            signals.error.connect(self._on_semantic_search_query_error)
            self._semantic_search_query_signals = signals

        index = self._get_semantic_index()
        sort_newest = self.get_sort_preference()

        class _SemanticSearchQueryWorker(QRunnable):
            def run(self_inner):
                try:
                    metadata_hits, semantic_query = index.search_metadata_text(
                        query,
                        base_files,
                        top_k=max(1, len(base_files)),
                        sort_newest=sort_newest,
                    )
                    from search import semantic_embeddings_enabled

                    used_semantic_backend = False
                    status_message = ""
                    hits = metadata_hits
                    if not semantic_query:
                        hits = metadata_hits
                    elif not semantic_embeddings_enabled():
                        hits = metadata_hits
                        if not hits:
                            status_message = (
                                "No matches -- this build supports metadata filters only "
                                "(e.g. camera:sony iso<800 city:tokyo)."
                            )
                    elif not index.semantic_backend_available():
                        hits = []
                        status_message = (
                            f"Semantic backend unavailable: {index.semantic_backend_error()}"
                        )
                    else:
                        metadata_candidate_paths = [h.file_path for h in metadata_hits]
                        if semantic_query and not metadata_candidate_paths:
                            metadata_candidate_paths = list(base_files)
                        hits = index.search_text(
                            semantic_query,
                            metadata_candidate_paths,
                            top_k=min(500, len(base_files)),
                            min_score=0.20,
                        )
                        used_semantic_backend = True

                    result = {
                        "hits": hits,
                        "semantic_query": semantic_query,
                        "used_semantic_backend": used_semantic_backend,
                        "sort_newest": sort_newest,
                        "status_message": status_message,
                        "backend_missing": bool(semantic_query)
                        and not index.semantic_backend_available(),
                    }
                    try:
                        from PyQt6 import sip

                        if not sip.isdeleted(signals):
                            signals.ready.emit(token, query, result)
                    except Exception:
                        pass
                except Exception as exc:
                    try:
                        from PyQt6 import sip

                        if not sip.isdeleted(signals):
                            signals.error.emit(token, query, str(exc))
                    except Exception:
                        pass

        _get_bg_thread_pool().start(_SemanticSearchQueryWorker())

    def _on_semantic_search_query_error(self, token: int, query: str, message: str) -> None:
        if token != getattr(self, "_semantic_search_query_generation", None):
            return
        if hasattr(self, "gallery_search_input") and self.gallery_search_input is not None:
            current = (self.gallery_search_input.text() or "").strip()
            if current and current != (query or "").strip():
                return
        QMessageBox.warning(self, "Semantic Search", message)

    def _apply_semantic_search_query_result(self, token: int, query: str, result: dict) -> None:
        if token != getattr(self, "_semantic_search_query_generation", None):
            return
        if hasattr(self, "gallery_search_input") and self.gallery_search_input is not None:
            current = (self.gallery_search_input.text() or "").strip()
            if current and current != (query or "").strip():
                return

        hits = result.get("hits") or []
        result.get("semantic_query")
        used_semantic_backend = bool(result.get("used_semantic_backend"))
        sort_newest = bool(result.get("sort_newest"))
        status_message = str(result.get("status_message") or "")
        backend_missing = bool(result.get("backend_missing"))

        if status_message:
            self.status_bar.showMessage(status_message, 6000 if "metadata filters only" in status_message else 5000)

        if not hits:
            self.image_files = []
            self._semantic_search_result_paths = []
            self.current_file_index = -1
            self.current_file_path = None
            empty_msg = (
                "Semantic search unavailable"
                if backend_missing
                else (
                    "No matching bookmarked images"
                    if getattr(self, "_gallery_bookmark_filter_active", False)
                    else "No matching images"
                )
            )
            self._switch_to_gallery_for_search()
            if hasattr(self, "gallery_justified") and self.gallery_justified:
                self.gallery_justified.clear_thumbnail_widgets()
                self.gallery_justified.set_images([])
                self.gallery_justified.show_empty_message(empty_msg)
            self._sync_gallery_scrollbar_policy()
            if hasattr(self, "status_counter_label"):
                self._update_gallery_counter()
            if not backend_missing and not status_message:
                self.status_bar.showMessage("No matching images", 3000)
            self._last_semantic_query = query
            return

        ranked_paths = [h.file_path for h in hits]
        pending_capture_sort = used_semantic_backend and len(ranked_paths) > 1
        if pending_capture_sort:
            self._last_semantic_query = query
            self._schedule_semantic_search_resort(
                query,
                ranked_paths,
                hits,
                used_semantic_backend,
                sort_newest,
            )
        if not ranked_paths:
            self.image_files = []
            self._semantic_search_result_paths = []
            self.current_file_index = -1
            self.current_file_path = None
            self._switch_to_gallery_for_search()
            empty_msg = (
                "No matching bookmarked images"
                if getattr(self, "_gallery_bookmark_filter_active", False)
                else "No matching images"
            )
            if getattr(self, "_gallery_bookmark_filter_active", False):
                self._apply_gallery_bookmark_filter()
                if hasattr(self, "gallery_justified") and self.gallery_justified:
                    self.gallery_justified.show_empty_message(empty_msg)
            elif hasattr(self, "gallery_justified") and self.gallery_justified:
                self.gallery_justified.clear_thumbnail_widgets()
                self.gallery_justified.set_images([])
                self.gallery_justified.show_empty_message(empty_msg)
                self._sync_gallery_scrollbar_policy()
                self._update_gallery_counter()
            self.status_bar.showMessage(empty_msg, 3000)
            self._last_semantic_query = query
            return

        self.image_files = ranked_paths
        self._semantic_search_result_paths = list(ranked_paths)
        self.current_file_index = 0
        self.current_file_path = self.image_files[0]

        self._switch_to_gallery_for_search()
        if getattr(self, "_gallery_bookmark_filter_active", False):
            self._apply_gallery_bookmark_filter()
        else:
            self._update_gallery_counter()
            if hasattr(self, "gallery_justified") and self.gallery_justified:
                self.gallery_justified.hide_empty_message()
            self._sync_gallery_scrollbar_policy()
            self._update_gallery_view()

        top = max(hits, key=lambda h: h.score) if hits else None
        effective_n = len(self._gallery_effective_files())
        if used_semantic_backend and top is not None:
            if getattr(self, "_gallery_bookmark_filter_active", False):
                message = (
                    f"Search: {effective_n} bookmarked match"
                    f"{'es' if effective_n != 1 else ''}"
                    f" | top score {top.score:.3f}"
                )
            else:
                message = f"Semantic search: {len(ranked_paths)} result(s) | top score {top.score:.3f}"
        elif getattr(self, "_gallery_bookmark_filter_active", False):
            message = f"EXIF search: {effective_n} bookmarked match{'es' if effective_n != 1 else ''}"
        else:
            message = f"EXIF search: {len(ranked_paths)} result(s)"
        self.status_bar.showMessage(message, 5000)
        self._last_semantic_query = query

    def _schedule_semantic_search_resort(
        self,
        query: str,
        ranked_paths: list,
        hits,
        used_semantic_backend: bool,
        sort_newest: bool,
    ) -> None:
        """Re-sort semantic search hits by capture time on a background worker."""
        token = getattr(self, "_semantic_search_resort_generation", 0) + 1
        self._semantic_search_resort_generation = token
        paths = list(ranked_paths)

        signals = getattr(self, "_semantic_search_resort_signals", None)
        if signals is None:
            signals = SemanticSearchResortSignals()
            signals.ready.connect(self._apply_semantic_search_resort)
            self._semantic_search_resort_signals = signals

        viewer = self
        folder_path = getattr(self, "current_folder", None)

        class _SemanticSearchResortWorker(QRunnable):
            def run(self_inner):
                bulk_meta: dict = {}
                sorted_paths = paths
                try:
                    sorted_paths, bulk_meta = viewer.sort_files_by_capture_time(
                        paths,
                        newest_first=sort_newest,
                        folder_path=folder_path,
                    )
                except Exception as e:
                    import logging
                    import traceback

                    logging.getLogger(__name__).error(
                        "[SEARCH] Background capture-time resort failed: %s\n%s",
                        e,
                        traceback.format_exc(),
                    )
                try:
                    from PyQt6 import sip

                    if not sip.isdeleted(signals):
                        signals.ready.emit(
                            token,
                            query,
                            sorted_paths,
                            hits,
                            used_semantic_backend,
                            bulk_meta,
                        )
                except Exception:
                    pass

        _get_bg_thread_pool().start(_SemanticSearchResortWorker())

    def _apply_semantic_search_resort(
        self,
        token: int,
        query: str,
        ranked_paths: list,
        hits,
        used_semantic_backend: bool,
        bulk_meta: dict,
    ) -> None:
        """Apply background capture-time re-sort for semantic search on the UI thread."""
        if token != getattr(self, "_semantic_search_resort_generation", None):
            return
        if (query or "").strip() != (getattr(self, "_last_semantic_query", "") or "").strip():
            return
        if not ranked_paths:
            return

        preserved = self.current_file_path
        self.image_files = ranked_paths
        self._semantic_search_result_paths = list(ranked_paths)
        if bulk_meta:
            self._gallery_bulk_metadata = bulk_meta
            self._bump_gallery_rating_presence_epoch()

        if preserved and preserved in self.image_files:
            self.current_file_index = self.image_files.index(preserved)
            self.current_file_path = preserved
        elif self.image_files:
            self.current_file_index = 0
            self.current_file_path = self.image_files[0]

        if getattr(self, "_gallery_bookmark_filter_active", False):
            self._apply_gallery_bookmark_filter()
        elif getattr(self, "view_mode", "single") == "gallery":
            self._update_gallery_counter()
            if hasattr(self, "gallery_justified") and self.gallery_justified:
                self.gallery_justified.hide_empty_message()
            self._sync_gallery_scrollbar_policy()
            self._update_gallery_view()

        top = max(hits, key=lambda h: h.score) if hits else None
        effective_n = len(self._gallery_effective_files())
        if used_semantic_backend and top is not None:
            if getattr(self, "_gallery_bookmark_filter_active", False):
                message = (
                    f"Search: {effective_n} bookmarked match"
                    f"{'es' if effective_n != 1 else ''}"
                    f" | top score {top.score:.3f}"
                )
            else:
                message = (
                    f"Semantic search: {len(ranked_paths)} result(s) "
                    f"| sorted by capture time | top score {top.score:.3f}"
                )
        elif getattr(self, "_gallery_bookmark_filter_active", False):
            message = f"EXIF search: {effective_n} bookmarked match{'es' if effective_n != 1 else ''}"
        else:
            message = f"EXIF search: {len(ranked_paths)} result(s)"
        self.status_bar.showMessage(message, 5000)

    def _clear_semantic_search_results(self, silent=False, exit_to_gallery=False):
        # Capture current gallery scroll anchor before clearing filter
        anchor_path = None
        if getattr(self, "view_mode", "single") == "gallery" and hasattr(self, "gallery_justified") and self.gallery_justified:
            anchor_path = self.gallery_justified.get_scroll_anchor_path()
            self._gallery_scroll_anchor_path = anchor_path

        if not self._semantic_search_backup_files and not self._semantic_search_corpus_files:
            if not silent:
                self.status_bar.showMessage("No active semantic search filter", 2500)
            return
        if self._semantic_search_corpus_files:
            restored = [p for p in self._semantic_search_corpus_files if os.path.isfile(p)]
        else:
            backup_files = self._semantic_search_backup_files or []
            restored = [p for p in backup_files if os.path.isfile(p)]
        self._semantic_search_backup_files = None
        self._semantic_search_result_paths = None
        self._last_semantic_query = ""
        if not restored:
            if not silent:
                self.status_bar.showMessage("Search filter cleared, but no files remain", 3000)
            return
        self.image_files = restored
        self._update_gallery_counter()
        if hasattr(self, "gallery_justified") and self.gallery_justified:
            gj = self.gallery_justified
            try:
                if getattr(gj, "load_manager", None) is not None:
                    if hasattr(gj.load_manager, "flush_queue"):
                        gj.load_manager.flush_queue()
                    else:
                        gj.load_manager.cancel_all_tasks()
            except Exception:
                pass
            gj.hide_empty_message()
        if self.current_file_path in self.image_files:
            self.current_file_index = self.image_files.index(
                _require_str_path(self.current_file_path)
            )
        else:
            self.current_file_index = 0
            self.current_file_path = self.image_files[0]
        if exit_to_gallery:
            self.view_mode = "gallery"
            self._stop_slideshow()
            self._show_gallery_view()
        elif getattr(self, "view_mode", "single") == "gallery":
            self._sync_gallery_scrollbar_policy()
            if getattr(self, "_gallery_bookmark_filter_active", False):
                self._apply_gallery_bookmark_filter()
            else:
                self._update_gallery_view()
        else:
            self.load_raw_image(self.current_file_path)
            self._sync_filmstrip_to_folder()
        if not silent:
            if getattr(self, "_gallery_bookmark_filter_active", False):
                n = len(self._gallery_effective_files())
                self.status_bar.showMessage(
                    f"Showing {n} bookmarked image{'s' if n != 1 else ''}",
                    3000,
                )
            else:
                self.status_bar.showMessage("Semantic search filter cleared", 3000)

    def _semantic_folder_settings_key(self, folder: str) -> str:
        import hashlib

        norm = os.path.normcase(os.path.normpath(folder or ""))
        digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:24]
        return f"semantic_index_incomplete/{digest}"

    def _set_semantic_index_incomplete(self, folder: str, incomplete: bool) -> None:
        if not folder:
            return
        settings = self.get_settings()
        key = self._semantic_folder_settings_key(folder)
        if incomplete:
            settings.setValue(key, True)
        else:
            settings.remove(key)

    def _is_semantic_index_incomplete(self, folder: str) -> bool:
        if not folder:
            return False
        return bool(
            self.get_settings().value(self._semantic_folder_settings_key(folder), False)
        )

    def _user_wants_semantic_embeddings_now(self) -> bool:
        """True only after the user opens gallery search (not on folder load alone)."""
        return bool(
            getattr(self, "_gallery_search_user_wants_semantic", False)
            or getattr(self, "_defer_semantic_after_metadata", False)
        )
