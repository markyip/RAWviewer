"""search mixin, part 1 of 4 (49 methods, 850 lines)."""
from __future__ import annotations

import contextlib
import logging
import sys
import time
from typing import Any, cast

from PyQt6.QtCore import QPoint, Qt, QTimer
from PyQt6.QtGui import QCursor, QRegion

from core.config import _norm_path

from .. import lazy
from ..macos import _macos_install_line_edit_completion_guard

SemanticImageIndex = None

lazy.register(sys.modules[__name__])


class SearchPart1:
    def _is_semantic_search_filter_active(self) -> bool:
        query = (getattr(self, "_last_semantic_query", "") or "").strip()
        return bool(query and getattr(self, "_semantic_search_backup_files", None) is not None)

    def _gallery_search_base_files(self) -> list[str]:
        """Files searched by the gallery bar (full folder/corpus; bookmark filter is display-only)."""
        if self._semantic_search_corpus_files:
            return list(self._semantic_search_corpus_files)
        return list(getattr(self, "image_files", []) or [])

    def _sync_filmstrip_index(self, center: bool | None = None) -> None:
        bar = self._filmstrip_bar()
        if bar is None or not bar.isEnabled():
            return
        if center is None:
            center = self._filmstrip_is_visible_enough()
        files = self._navigation_files()
        idx = getattr(self, "current_file_index", -1)
        path = getattr(self, "current_file_path", None)
        if path and files:
            try:
                target = _norm_path(path)
                for i, fp in enumerate(files):
                    if _norm_path(fp) == target:
                        idx = i
                        self.current_file_index = i
                        break
            except Exception:
                pass
        if idx >= 0:
            bar.set_current_index(idx, center=center)
            bar.refresh_visible_thumbnails(refresh_cache=True)

    def _get_semantic_index(self):
        global SemanticImageIndex
        if SemanticImageIndex is None:
            try:
                from search import SemanticImageIndex as _SemanticImageIndex

                SemanticImageIndex = _SemanticImageIndex
            except Exception as exc:
                raise RuntimeError(
                    "Semantic search module is unavailable. "
                    "Please ensure Plus-edition dependencies are installed."
                ) from exc
        if self._semantic_index is None:
            self._semantic_index = SemanticImageIndex()
            if getattr(self, "_pending_abort_semantic_index", False):
                self._semantic_index.request_abort_indexing()
                self._pending_abort_semantic_index = False
            elif getattr(self, "_pending_pause_semantic_index", False):
                self._semantic_index.pause_indexing(True)
                self._pending_pause_semantic_index = False
        return self._semantic_index

    def _pause_semantic_indexing_deferred(self) -> None:
        """Pause indexing without opening the semantic DB on the UI thread."""
        if self._semantic_index is not None:
            self._semantic_index.pause_indexing(True)
        else:
            self._pending_pause_semantic_index = True

    def _abort_semantic_indexing_deferred(self) -> None:
        """Abort in-flight indexing from a previous folder scope."""
        if self._semantic_index is not None:
            self._semantic_index.request_abort_indexing()
        else:
            self._pending_abort_semantic_index = True
            self._pending_pause_semantic_index = True

    def _wire_index_load_throttle(self, index) -> None:
        """Give the background index worker thread-safe ILM throttle toggles.

        The worker engages the throttle only around heavy work and releases it
        while paused, so an idle/metadata-only index never starves the gallery.
        """
        mgr = getattr(self, "image_manager", None)
        if mgr is None or index is None:
            return
        enter = getattr(mgr, "enter_semantic_indexing_throttle", None)
        exit_ = getattr(mgr, "exit_semantic_indexing_throttle", None)
        if not callable(enter) or not callable(exit_):
            return
        setter = getattr(index, "set_load_throttle_hooks", None)
        if callable(setter):
            with contextlib.suppress(Exception):
                setter(enter, exit_)

    def _begin_indexing_load_throttle(self) -> None:
        """Cut ILM concurrency while semantic or face indexing holds large RAM."""
        depth = int(getattr(self, "_indexing_load_throttle_depth", 0) or 0) + 1
        self._indexing_load_throttle_depth = depth
        if depth != 1:
            return
        mgr = getattr(self, "image_manager", None)
        if mgr is not None and hasattr(mgr, "enter_semantic_indexing_throttle"):
            with contextlib.suppress(Exception):
                mgr.enter_semantic_indexing_throttle()

    def _end_indexing_load_throttle(self) -> None:
        depth = int(getattr(self, "_indexing_load_throttle_depth", 0) or 0)
        if depth <= 0:
            return
        depth -= 1
        self._indexing_load_throttle_depth = depth
        if depth != 0:
            return
        mgr = getattr(self, "image_manager", None)
        if mgr is not None and hasattr(mgr, "exit_semantic_indexing_throttle"):
            with contextlib.suppress(Exception):
                mgr.exit_semantic_indexing_throttle()

    def _release_indexing_load_throttle(self) -> None:
        """Force-release semantic indexing I/O throttle (e.g. on folder change or gallery entry)."""
        self._indexing_load_throttle_depth = 0
        mgr = getattr(self, "image_manager", None)
        if mgr is not None and hasattr(mgr, "force_exit_semantic_indexing_throttle"):
            with contextlib.suppress(Exception):
                mgr.force_exit_semantic_indexing_throttle()

    def _defer_background_indexing_for_gallery(self) -> None:
        """Pause semantic/face background work while gallery thumbnails first paint."""
        import logging
        import time

        logger = logging.getLogger(__name__)
        file_count = len(getattr(self, "image_files", []) or [])
        # Short deferral only -- long warmup windows starve first paint on big folders.
        warmup_s = min(2.0, 0.5 + file_count / 4000.0)
        self._gallery_indexing_deferred_until = time.time() + warmup_s
        self._gallery_deferral_opened_ts = time.time()
        self._pause_semantic_indexing_deferred()
        if getattr(self, "_face_indexing_in_progress", False):
            self._face_index_active_token = None
            logger.info(
                "[VIEW_MODE] Face indexing deferred during gallery entry (%d files)",
                file_count,
            )

    def _end_gallery_indexing_deferral(self) -> None:
        """Restore background indexing / load concurrency after gallery warmup."""
        self._gallery_indexing_deferred_until = 0.0
        try:
            if hasattr(self, "image_manager") and self.image_manager is not None and hasattr(self.image_manager, "exit_gallery_warmup_throttle"):
                    self.image_manager.exit_gallery_warmup_throttle()
        except Exception:
            pass

    def _reset_semantic_search_for_new_folder(self):
        """Clear gallery search UI and stale query when the folder scope changes."""
        self._abort_semantic_indexing_deferred()
        self._semantic_search_backup_files = None
        self._semantic_search_result_paths = None
        self._last_semantic_query = ""
        self._semantic_indexing_in_progress = False
        self._semantic_index_active_token = None
        self._semantic_index_signals = None
        self._semantic_index_progress_base = 0
        self._semantic_index_progress_total = 0
        self._semantic_coverage_cache = None
        self._semantic_coverage_cache_ts = 0.0
        self._metadata_index_run_done = False
        self._metadata_index_prep_complete = False
        self._metadata_index_extract_pending_count = 0
        self._metadata_index_deferred_corpus = None
        self._metadata_index_deferred_coverage = None
        self._metadata_index_deferred_pending = None
        self._single_view_index_defer_opened_ts = 0.0
        self._gallery_bookmark_filter_active = False
        if hasattr(self, "_metadata_index_defer_timer"):
            self._metadata_index_defer_timer.stop()
        self._gallery_search_user_collapsed_while_busy = False
        # Collapse first so clearing status does not re-trigger expand-with-new-width while still open.
        with contextlib.suppress(Exception):
            self._set_search_panel_expanded(False, animate=False)
        with contextlib.suppress(Exception):
            self._ensure_gallery_search_collapsed()
        with contextlib.suppress(Exception):
            self._set_gallery_search_status("")
        if hasattr(self, "gallery_search_input") and self.gallery_search_input is not None:
            try:
                self.gallery_search_input.blockSignals(True)
                self.gallery_search_input.clear()
            finally:
                self.gallery_search_input.blockSignals(False)
            ph = getattr(self, "_gallery_search_placeholder_saved", "") or "Search gallery"
            self.gallery_search_input.setPlaceholderText(ph)
            self._gallery_search_placeholder_saved = ""

    def _is_gallery_search_indexing_active(self) -> bool:
        return bool(
            getattr(self, "_semantic_indexing_in_progress", False)
            or getattr(self, "_semantic_index_prep_in_progress", False)
            or getattr(self, "_semantic_asset_download_in_progress", False)
            or getattr(self, "_face_indexing_in_progress", False)
        )

    def _invalidate_semantic_coverage_cache(self) -> None:
        self._semantic_coverage_cache = None
        self._semantic_coverage_cache_ts = 0.0
        self._gallery_search_ready_cache = None
        self._gallery_search_ready_cache_ts = 0.0

    def _gallery_search_semantic_work_active(self) -> bool:
        """True while semantic/metadata prep blocks typing in the search field."""
        return bool(
            getattr(self, "_semantic_indexing_in_progress", False)
            or getattr(self, "_semantic_index_prep_in_progress", False)
            or getattr(self, "_semantic_asset_download_in_progress", False)
        )

    def _gallery_search_index_corpus(self) -> list:
        """Full folder corpus for indexing -- not filtered search results."""
        corpus = getattr(self, "_semantic_search_corpus_files", None)
        if corpus:
            return list(corpus)
        return list(getattr(self, "image_files", []) or [])

    def _gallery_search_index_session_key_for(self, corpus_files) -> str:
        import hashlib

        folder = getattr(self, "current_folder", "") or ""
        try:
            model = self._get_semantic_index().model_name or ""
        except Exception:
            model = ""
        digest = hashlib.sha256("\n".join(sorted(corpus_files)).encode()).hexdigest()[:16]
        return f"{folder}|{model}|{digest}"

    def _mark_gallery_search_index_session_complete(self, corpus_files) -> None:
        if corpus_files:
            self._gallery_search_index_session_key = (
                self._gallery_search_index_session_key_for(corpus_files)
            )

    def _is_gallery_search_index_session_complete(self, corpus_files) -> bool:
        key = getattr(self, "_gallery_search_index_session_key", None)
        if not key or not corpus_files:
            return False
        return key == self._gallery_search_index_session_key_for(corpus_files)

    def _gallery_search_needs_index_work(self, corpus_files) -> tuple[bool, bool]:
        """Return (needs_semantic, needs_face) for the given corpus."""
        if getattr(self, "view_mode", "") == "gallery" and self._gallery_has_thumbnail_load_pressure():
            # Avoid a 1k+ path DB scan on the UI thread during gallery first paint.
            return True, False
        try:
            from search import semantic_embeddings_enabled

            index = self._get_semantic_index()
            if semantic_embeddings_enabled() and not index.semantic_backend_available():
                return True, False
            if index.get_pending_paths(corpus_files):
                return True, False
            return False, index.get_face_pending_count(corpus_files) > 0
        except Exception:
            return False, False

    def _is_gallery_search_fully_indexed(self, corpus_files) -> bool:
        if not corpus_files:
            return False
        needs_semantic, needs_face = self._gallery_search_needs_index_work(corpus_files)
        return not needs_semantic and not needs_face

    def _gallery_search_idle_tooltip(self) -> str:
        """Default search-field tooltip when no indexing progress is in the placeholder."""
        try:
            from search import semantic_embeddings_enabled

            if semantic_embeddings_enabled():
                return ""
        except Exception:
            pass
        return self._semantic_disabled_hint_texts()[1]

    def _reset_gallery_search_to_idle(self) -> None:
        """Restore the search field to the normal idle placeholder (no index progress)."""
        # Keep user intent while semantic is still pending (defer survives stale-progress cleanup).
        if getattr(self, "_defer_semantic_after_metadata", False):
            return
        if getattr(self, "_gallery_search_user_wants_semantic", False):
            return
        if (
            getattr(self, "_search_panel_expanded", False)
            and not self._is_gallery_semantic_search_ready()
        ):
            return
        self._gallery_search_show_index_progress = False
        self._gallery_search_user_wants_semantic = False
        self._gallery_search_status_full = ""
        self._gallery_search_placeholder_saved = ""
        inp = getattr(self, "gallery_search_input", None)
        if inp is not None:
            inp.setPlaceholderText("Search gallery")
            inp.setToolTip(self._gallery_search_idle_tooltip())
            self._reset_gallery_search_panel_width()
        self._sync_gallery_search_input_editable()

    def _restore_gallery_search_placeholder(self) -> None:
        inp = getattr(self, "gallery_search_input", None)
        if inp is None:
            return
        ph = getattr(self, "_gallery_search_placeholder_saved", "") or "Search gallery"
        if self._should_show_search_index_progress(ph):
            ph = "Search gallery"
        inp.setPlaceholderText(ph)
        self._gallery_search_placeholder_saved = ""

    def _clear_stale_gallery_search_index_progress(self) -> None:
        """Drop leftover index progress when no background pass is running."""
        if self._is_gallery_search_indexing_active():
            return
        if getattr(self, "_defer_semantic_after_metadata", False):
            return
        if getattr(self, "_gallery_search_user_wants_semantic", False):
            return
        if (
            getattr(self, "_search_panel_expanded", False)
            and not self._is_gallery_semantic_search_ready()
        ):
            return
        full = (getattr(self, "_gallery_search_status_full", "") or "").strip()
        if full and self._should_show_search_index_progress(full):
            self._reset_gallery_search_to_idle()
            return
        inp = getattr(self, "gallery_search_input", None)
        if inp is None:
            return
        ph = (inp.placeholderText() or "").strip()
        if ph and self._should_show_search_index_progress(ph):
            self._reset_gallery_search_to_idle()

    def _is_gallery_semantic_search_ready(self, corpus_files=None) -> bool:
        """True when semantic embeddings and face scan (if enabled) are complete."""
        corpus = list(corpus_files or self._gallery_search_index_corpus() or [])
        if not corpus:
            return False
        if getattr(self, "_defer_semantic_after_metadata", False):
            # Deferred user semantic must not trust a session marked ready before
            # metadata-only background indexing finished.
            self._gallery_search_index_session_key = None
            self._gallery_search_ready_cache = None
            self._gallery_search_ready_cache_ts = 0.0
        elif self._is_gallery_search_index_session_complete(corpus):
            return True
        if (
            getattr(self, "_semantic_indexing_in_progress", False)
            and getattr(self, "_semantic_index_active_token", None) is None
        ):
            logging.getLogger(__name__).warning(
                "[SEARCH][READY] Clearing stale semantic indexing flag (no active token)."
            )
            self._semantic_indexing_in_progress = False
        if getattr(self, "_semantic_index_prep_in_progress", False):
            return False
        if getattr(self, "_semantic_indexing_in_progress", False):
            return False
        if getattr(self, "_face_indexing_in_progress", False):
            return False

        now = time.time()
        cache = getattr(self, "_gallery_search_ready_cache", None)
        if cache and (now - getattr(self, "_gallery_search_ready_cache_ts", 0.0)) < 3.0 and (
            cache.get("folder") == getattr(self, "current_folder", None)
            and cache.get("count") == len(corpus)
        ):
            return bool(cache.get("ready", False))

        from search import face_scan_enabled, semantic_embeddings_enabled

        ready = False
        if not semantic_embeddings_enabled():
            ready = bool(getattr(self, "_metadata_index_run_done", False))
        else:
            try:
                index = self._get_semantic_index()
                if index.semantic_backend_available():
                    # Metadata-only rows (semantic_ready=1, dim=0) must still block search.
                    ready = self._is_gallery_search_fully_indexed(corpus)
                    if ready:
                        cov = index.get_semantic_embedding_coverage(corpus)
                        if cov["total"] > 0 and cov["embeddable"] == 0:
                            ready = False
            except Exception:
                ready = False

        if ready and face_scan_enabled():
            try:
                ready = self._get_semantic_index().get_face_pending_count(corpus) == 0
            except Exception:
                ready = False

        self._gallery_search_ready_cache = {
            "folder": getattr(self, "current_folder", None),
            "count": len(corpus),
            "ready": ready,
        }
        self._gallery_search_ready_cache_ts = now
        if ready and not getattr(self, "_defer_semantic_after_metadata", False):
            self._mark_gallery_search_index_session_complete(corpus)
        return ready

    def _chain_face_index_after_semantic(
        self, corpus_files, *, face_pending: int | None = None
    ) -> None:
        """Start face scan after semantic pass -- search field stays locked until done."""
        from search import face_scan_enabled

        if not face_scan_enabled():
            self._finish_gallery_search_unlock(corpus_files)
            return
        if getattr(self, "_face_indexing_in_progress", False):
            self._sync_gallery_search_input_editable()
            return
        corpus = list(corpus_files or self._gallery_search_index_corpus() or [])
        if not corpus:
            return
        try:
            pending = (
                int(face_pending)
                if face_pending is not None
                else int(self._get_semantic_index().get_face_pending_count(corpus) or 0)
            )
        except Exception:
            return
        if pending <= 0:
            self._finish_gallery_search_unlock(corpus)
            return
        self._gallery_search_show_index_progress = True
        self._set_gallery_search_status(
            self._format_index_progress("Face", 0, pending)
        )
        self._invalidate_semantic_coverage_cache()
        self._start_face_index_background(corpus, show_search_progress=True)
        self._sync_gallery_search_input_editable()

    def _finish_gallery_search_unlock(self, corpus_files=None) -> None:
        """Semantic + face indexing complete -- enable typing in the search field."""
        corpus = corpus_files or self._gallery_search_index_corpus()
        if corpus:
            self._mark_gallery_search_index_session_complete(corpus)
            self._gallery_search_ready_cache = {
                "folder": getattr(self, "current_folder", None),
                "count": len(corpus),
                "ready": True,
            }
            self._gallery_search_ready_cache_ts = time.time()
        folder = getattr(self, "current_folder", None)
        if folder:
            self._set_semantic_index_incomplete(folder, False)
        self._gallery_search_user_wants_semantic = False
        self._gallery_search_show_index_progress = False
        self._gallery_search_status_full = ""
        inp = getattr(self, "gallery_search_input", None)
        if inp is not None:
            from search import semantic_embeddings_enabled

            if semantic_embeddings_enabled():
                inp.setPlaceholderText("Search gallery")
            else:
                # EXIF/location unlock: short placeholder; AI limitation is in the tooltip.
                inp.setPlaceholderText("Search EXIF, name, place")
            inp.setToolTip(self._gallery_search_idle_tooltip())
        self._reset_gallery_search_panel_width()
        self._sync_gallery_search_input_editable()
        if (
            getattr(self, "_search_panel_expanded", False)
            and inp is not None
        ):
            inp.setFocus()

    def _sync_gallery_search_input_editable(self) -> None:
        """Lock search input until semantic embeddings and face scan (if enabled) finish."""
        inp = getattr(self, "gallery_search_input", None)
        if inp is None:
            return
        expanded = bool(getattr(self, "_search_panel_expanded", False))
        if (
            self._gallery_search_semantic_work_active()
            or getattr(self, "_face_indexing_in_progress", False)
        ):
            ready = False
        else:
            ready = self._is_gallery_semantic_search_ready()
        locked = expanded and not ready
        inp.setReadOnly(locked)
        prev_locked = getattr(self, "_gallery_search_input_locked_prev", None)
        if prev_locked is None or bool(prev_locked) != bool(locked):
            logging.getLogger(__name__).debug(
                "[SEARCH][INPUT] lock_state changed: locked=%s expanded=%s ready=%s semantic_work=%s face_work=%s",
                locked,
                expanded,
                ready,
                self._gallery_search_semantic_work_active(),
                getattr(self, "_face_indexing_in_progress", False),
            )
        self._gallery_search_input_locked_prev = bool(locked)
        inp.setCursor(
            QCursor(Qt.CursorShape.ArrowCursor if locked else Qt.CursorShape.IBeamCursor)
        )

    def _apply_gallery_search_status_placeholder(self) -> None:
        """Show indexing progress inside the search field placeholder."""
        inp = getattr(self, "gallery_search_input", None)
        lab = getattr(self, "gallery_search_status_label", None)
        if lab is not None:
            lab.hide()
            lab.setText("")
            lab.setToolTip("")
        if inp is None:
            return
        full = (getattr(self, "_gallery_search_status_full", None) or "").strip()
        indexing = self._is_gallery_search_indexing_active()
        if full and self._should_show_search_index_progress(full) and not indexing:
            full = ""
            self._gallery_search_status_full = ""
        if full:
            inp.setPlaceholderText(full)
            cap = int(getattr(self, "_search_input_width_max", 500) or 500)
            w = self._gallery_search_input_width_for_text(full)
            inp.setToolTip(full if w >= cap else self._gallery_search_idle_tooltip())
            return
        inp.setToolTip(self._gallery_search_idle_tooltip())
        if not indexing:
            ph = getattr(self, "_gallery_search_placeholder_saved", "") or "Search gallery"
            if self._should_show_search_index_progress(ph):
                ph = "Search gallery"
            inp.setPlaceholderText(ph)

    def _gallery_search_panel_is_visible(self) -> bool:
        return bool(
            getattr(self, "_search_panel_expanded", False)
            and not getattr(self, "_gallery_search_user_collapsed_while_busy", False)
        )

    def _release_gallery_search_panel_width(self) -> None:
        """Let the expand strip collapse to zero without child min-width fighting layout."""
        inp = getattr(self, "gallery_search_input", None)
        panel = getattr(self, "gallery_search_panel", None)
        if inp is not None:
            inp.setMinimumWidth(0)
            inp.setMaximumWidth(16777215)
        if panel is not None:
            panel.setMinimumWidth(0)
            panel.setMaximumWidth(16777215)

    def _reapply_gallery_search_panel_width_from_status(self) -> None:
        """Restore search-field width after the expand strip opens."""
        panel = getattr(self, "gallery_search_panel", None)
        inp = getattr(self, "gallery_search_input", None)
        if panel is not None:
            panel.setMinimumWidth(0)
            panel.setMaximumWidth(16777215)
        if inp is None:
            return
        full = (getattr(self, "_gallery_search_status_full", "") or "").strip()
        if full:
            w = self._gallery_search_input_width_for_text(full)
            if inp.minimumWidth() == w and inp.maximumWidth() == w:
                return
            self._apply_gallery_search_input_width(w, progress=True)
            self._search_panel_target_width = self._gallery_search_status_panel_width(full)
        else:
            input_idle = int(getattr(self, "_search_input_width_idle", 300) or 300)
            if inp.minimumWidth() == input_idle and inp.maximumWidth() == input_idle:
                return
            self._reset_gallery_search_panel_width()

    def _apply_gallery_search_input_width(self, width: int, *, progress: bool = False) -> None:
        """Apply search-field width; pin min=max whenever progress or idle-expanded."""
        # Pinning mid-slide fights overlay geometry and looks like a halfway pause.
        if self._search_panel_animation_running():
            return
        inp = getattr(self, "gallery_search_input", None)
        if inp is None:
            return
        idle = int(getattr(self, "_search_input_width_idle", 300) or 300)
        w = max(140, int(width))
        pin = progress or w <= idle
        if pin:
            inp.setMinimumWidth(w)
            inp.setMaximumWidth(w)
        else:
            inp.setMinimumWidth(140)
            inp.setMaximumWidth(max(idle, w))

    def _gallery_search_input_width_for_text(self, text: str) -> int:
        """Fit placeholder progress text without shrinking font (13px)."""
        idle = int(getattr(self, "_search_input_width_idle", 300) or 300)
        cap = int(getattr(self, "_search_input_width_max", 500) or 500)
        display = (text or "").strip()
        if not display:
            return idle
        inp = getattr(self, "gallery_search_input", None)
        if inp is None:
            return idle
        from PyQt6.QtGui import QFontMetrics

        fm = QFontMetrics(inp.font())
        try:
            text_w = fm.horizontalAdvance(display)
        except AttributeError:
            text_w = cast(Any, fm).width(display)
        # Padding, border, and clear-button affordance (Windows needs extra room).
        needed = int(text_w) + 72
        return min(max(idle, needed), cap)

    def _gallery_search_status_panel_width(self, text: str) -> int:
        """Expandable strip width tracks the search field (progress in placeholder)."""
        input_w = self._gallery_search_input_width_for_text(text)
        return input_w + 10

    def _reset_gallery_search_panel_width(self) -> None:
        idle = int(getattr(self, "_search_panel_target_width_idle", 310) or 310)
        self._search_panel_target_width = idle
        inp = getattr(self, "gallery_search_input", None)
        if inp is not None:
            input_idle = int(getattr(self, "_search_input_width_idle", 300) or 300)
            self._apply_gallery_search_input_width(input_idle, progress=False)
            inp.setToolTip(self._gallery_search_idle_tooltip())

    def _set_gallery_search_status(
        self, message: str, animate: bool = False, *, relayout: bool = True
    ):
        has_msg = bool(message and str(message).strip())
        show_in_search = bool(getattr(self, "_gallery_search_show_index_progress", False))
        if has_msg and self._should_show_search_index_progress(message) and not show_in_search:
            # Background metadata/semantic passes stay silent in the search UI.
            if self._is_gallery_search_indexing_active() and hasattr(self, "status_bar"):
                with contextlib.suppress(Exception):
                    self.status_bar.showMessage(str(message).strip(), 0)
            has_msg = False
            message = ""
        self._gallery_search_status_full = (message or "").strip()

        busy = self._is_gallery_search_indexing_active()
        panel_visible = self._gallery_search_panel_is_visible()
        animating = self._search_panel_animation_running()
        inp = getattr(self, "gallery_search_input", None)
        if inp is not None:
            if has_msg:
                w = self._gallery_search_input_width_for_text(
                    self._gallery_search_status_full
                )
                self._search_panel_target_width = self._gallery_search_status_panel_width(
                    self._gallery_search_status_full
                )
                # Defer pinning until the open/close slide finishes.
                if panel_visible and not animating:
                    self._apply_gallery_search_input_width(w, progress=True)
            elif panel_visible and not animating:
                self._reset_gallery_search_panel_width()
            self._apply_gallery_search_status_placeholder()
            self._sync_gallery_search_input_editable()

        if not has_msg and hasattr(self, "status_bar") and self.status_bar is not None:
            with contextlib.suppress(Exception):
                self.status_bar.clearMessage()

        if not relayout:
            return

        if not has_msg:
            if not animating:
                self._reset_gallery_search_panel_width()
        else:
            self._search_panel_target_width = self._gallery_search_status_panel_width(
                self._gallery_search_status_full
            )

        if getattr(self, "view_mode", "single") != "gallery" and not getattr(self, "_search_panel_expanded", False):
                return

        user_collapsed = getattr(
            self, "_gallery_search_user_collapsed_while_busy", False
        )
        if user_collapsed:
            return
        if not getattr(self, "_search_panel_expanded", False):
            self._apply_search_expand_container_width(0, animate=False)
            return
        if has_msg or busy:
            self._apply_search_expand_container_width(
                self._search_panel_target_width, animate=animate
            )

    def _search_expand_overlay_anchor(self) -> tuple | None:
        """Top-left anchor (x, y, height) for the floating search field."""
        btn = getattr(self, "search_bottom_button", None)
        host = getattr(self, "status_bar_widget", None)
        if btn is None or host is None or not btn.isVisible():
            return None
        origin = btn.mapTo(host, QPoint(0, 0))
        gap = 8
        h = max(btn.height(), 28)
        return origin.x() + btn.width() + gap, origin.y(), h

    def _place_search_expand_overlay(self, width: int) -> None:
        """Position the floating search strip; width 0 hides it.

        Does not touch left_buttons / right_status layouts -- that in-row
        maximumWidth animation was what made trailing icons flicker.

        Open/close animate a clip mask over a fixed full-width field so
        layout never reflows mid-slide (the old per-frame resize caused the
        halfway-pause look on collapse).
        """
        container = getattr(self, "search_expand_container", None)
        host = getattr(self, "status_bar_widget", None)
        if container is None or host is None:
            return
        reveal_w = max(0, int(width))
        anchor = self._search_expand_overlay_anchor()
        if anchor is None:
            if reveal_w <= 0:
                with contextlib.suppress(Exception):
                    container.clearMask()
                container.hide()
            return
        x, y, h = anchor
        trailing = getattr(self, "right_status_trailing", None)
        max_w = None
        if trailing is not None and trailing.isVisible():
            try:
                right_limit = trailing.mapTo(host, QPoint(0, 0)).x() - 8
                max_w = max(0, int(right_limit) - int(x))
            except Exception:
                max_w = None
        if max_w is None:
            max_w = max(0, host.width() - int(x) - 8)

        content_w = int(getattr(self, "_search_panel_anim_content_width", 0) or 0)
        if content_w <= 0:
            content_w = int(getattr(self, "_search_panel_target_width", 310) or 310)
        content_w = min(max(1, content_w), int(max_w) if max_w > 0 else content_w)
        reveal_w = min(reveal_w, content_w)

        panel = getattr(self, "gallery_search_panel", None)
        inp = getattr(self, "gallery_search_input", None)
        if reveal_w > 0 or bool(getattr(self, "_search_panel_expanded", False)):
            if panel is not None:
                panel.setFixedHeight(int(h))
                panel.setFixedWidth(int(content_w))
            if inp is not None:
                inp.setFixedHeight(min(28, int(h)))
                inp.setFixedWidth(max(140, int(content_w) - 10))

        # Footprint stays at full content width; mask reveals/hides.
        geo_w = content_w if reveal_w > 0 else 0
        container.setGeometry(int(x), int(y), int(geo_w), int(h))
        container.setFixedHeight(int(h))
        container.setFixedWidth(int(geo_w) if geo_w > 0 else 0)
        if reveal_w > 0:
            if container.parent() is not host:
                container.setParent(host)
            with contextlib.suppress(Exception):
                container.setMask(QRegion(0, 0, int(reveal_w), int(h)))
            container.show()
            container.raise_()
        else:
            with contextlib.suppress(Exception):
                container.clearMask()
            container.hide()

    def _apply_search_expand_overlay_geometry(
        self, width: int, *, expanded: bool | None = None
    ) -> None:
        """Resize/place the floating search strip (legacy name kept for callers)."""
        if expanded is None:
            expanded = int(width) > 0
        if int(width) > 0:
            self._search_panel_target_width = int(width)
        self._set_search_panel_expanded(bool(expanded), animate=False)

    def _search_panel_animation_running(self) -> bool:
        anim = getattr(self, "_search_panel_animation", None)
        if anim is None:
            return False
        try:
            from PyQt6.QtCore import QAbstractAnimation

            return anim.state() == QAbstractAnimation.State.Running
        except Exception:
            return True

    def _sync_search_expand_overlay(self) -> None:
        # Re-anchor after status-bar relayout; never interrupt an in-flight slide.
        if self._search_panel_animation_running():
            return
        if not getattr(self, "_search_panel_expanded", False):
            self._place_search_expand_overlay(0)
            return
        self._place_search_expand_overlay(
            int(getattr(self, "_search_panel_target_width", 310) or 310)
        )

    def _schedule_search_expand_overlay_sync(self, delay_ms: int = 0) -> None:
        """Re-anchor the floating search field after the status bar relayouts."""
        timer = getattr(self, "_search_overlay_sync_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._sync_search_expand_overlay)
            self._search_overlay_sync_timer = timer
        timer.start(max(0, int(delay_ms)))

    def _apply_search_expand_container_width(self, target: int, *, animate: bool = False) -> None:
        """Resize the floating search strip; skip redundant updates."""
        if not hasattr(self, "search_expand_container") or self.search_expand_container is None:
            return
        target = max(0, int(target))
        expanded = target > 0
        container = self.search_expand_container
        current_w = container.width() if container.isVisible() else 0
        if (
            current_w == target
            and bool(getattr(self, "_search_panel_expanded", False)) == expanded
        ):
            return
        if not animate and self._search_panel_animation_running():
            if target > 0:
                self._search_panel_target_width = target
                # Extend the in-flight slide instead of restarting (avoids pause).
                anim = getattr(self, "_search_panel_animation", None)
                if anim is not None:
                    try:
                        if int(target) > int(anim.endValue()):
                            anim.setEndValue(int(target))
                    except Exception:
                        pass
            return
        if target > 0:
            self._search_panel_target_width = target
        self._set_search_panel_expanded(expanded, animate=animate)

    def _ensure_gallery_search_collapsed(self) -> None:
        """Keep the search field hidden until the user taps the search button."""
        if getattr(self, "_search_panel_expanded", False):
            return
        self._apply_search_expand_container_width(0, animate=False)

    def _set_gallery_search_input_visible(self, *, animate: bool = False):
        """Show the search panel beside the search icon (only when user expanded search)."""
        if hasattr(self, "search_expand_layout") and hasattr(self, "gallery_search_panel"):
            self.search_expand_layout.setCurrentWidget(self.gallery_search_panel)
        if not getattr(self, "_search_panel_expanded", False):
            self._ensure_gallery_search_collapsed()
            self._sync_gallery_search_input_editable()
            return
        self._clear_stale_gallery_search_index_progress()
        full = (getattr(self, "_gallery_search_status_full", "") or "").strip()
        if full:
            self._search_panel_target_width = self._gallery_search_status_panel_width(full)
        else:
            self._search_panel_target_width = int(
                getattr(self, "_search_panel_target_width_idle", 310) or 310
            )
        self._apply_search_expand_container_width(
            self._search_panel_target_width, animate=animate
        )
        self._sync_gallery_search_input_editable()
        if hasattr(self, "gallery_search_input") and self.gallery_search_input is not None:
            _macos_install_line_edit_completion_guard(self.gallery_search_input)

    def _switch_to_gallery_for_search(self):
        """Enter gallery mode to show semantic search results or empty state."""
        if getattr(self, "view_mode", "single") == "gallery":
            return
        self.view_mode = "gallery"
        self._stop_slideshow()
        self._show_gallery_view()
