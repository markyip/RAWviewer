
"""gallery mixin, part 4 of 5 (47 methods, 813 lines)."""
from __future__ import annotations

import contextlib
import logging
import os
import sys

from PyQt6.QtCore import QSize, Qt, QTimer

from core import theme
from core.config import _env_int, _norm_path

from .. import lazy
from ..lazy import is_raw_file, qta
from ..paths import _require_str_path
from ..qt_helpers import _qta_icon

lazy.register(sys.modules[__name__])


class GalleryPart4:
    def _gallery_shift_click_selection(self, file_path: str) -> None:
        """Shift+click range selection.

        A Shift+click only extends an inclusive range when the previous action was
        also a Shift+click. The first Shift+click after a Cmd/Ctrl+click (or any
        non-shift action) just sets a fresh range anchor and selects that one tile,
        so picking 4th→6th after Cmd+click 2nd won't pull in the 3rd.
        """
        canonical = self._gallery_canonical_path(file_path)
        if not canonical:
            return
        key = _norm_path(canonical)
        paths = self._gallery_selection_ordered_paths()
        norm_list = [_norm_path(p) for p in paths if p]
        if key not in norm_list:
            return
        selected = set(getattr(self, "_gallery_selected_paths", set()) or set())
        anchor = getattr(self, "_gallery_selection_anchor", None)
        last_action = getattr(self, "_gallery_last_select_action", None)
        if last_action != "shift" or not anchor or anchor not in norm_list:
            self._gallery_selection_anchor = key
            self._gallery_last_select_action = "shift"
            selected.add(key)
            self._gallery_set_selection_keys(selected)
            return
        try:
            i0 = norm_list.index(anchor)
            i1 = norm_list.index(key)
        except ValueError:
            self._gallery_selection_anchor = key
            self._gallery_last_select_action = "shift"
            selected.add(key)
            self._gallery_set_selection_keys(selected)
            return
        lo, hi = (i0, i1) if i0 <= i1 else (i1, i0)
        selected |= {norm_list[i] for i in range(lo, hi + 1)}
        self._gallery_last_select_action = "shift"
        self._gallery_set_selection_keys(selected)

    def _gallery_canonical_path(self, file_path: str) -> str | None:
        if not file_path or not self.image_files:
            return None
        target = _norm_path(file_path)
        for p in self.image_files:
            if _norm_path(p) == target:
                return p
        return None

    def _is_gallery_path_selected(self, file_path: str) -> bool:
        raw = set(getattr(self, "_gallery_selected_paths", set()) or set())
        if not raw:
            return False
        key = _norm_path(file_path)
        if key in raw:
            return True
        member_keys = self._gallery_selection_keys_for_tile(file_path)
        if len(member_keys) > 1:
            return member_keys <= raw
        return False

    def _gallery_has_selection(self) -> bool:
        return bool(getattr(self, "_gallery_selected_paths", None))

    def _clear_gallery_selection(self) -> None:
        selected = getattr(self, "_gallery_selected_paths", None)
        if selected:
            selected.clear()
        self._gallery_selection_anchor = None
        self._gallery_last_select_action = None
        self._gallery_refresh_selection_ui()

    def _request_clear_gallery_selection(self) -> None:
        self._clear_gallery_selection()

    def _load_gallery_pixmap_async(self, file_path: str) -> None:
        """Legacy justified-gallery async loader (no-op; external gallery handles loads)."""
        return

    def _gallery_select_all_displayed(self) -> None:
        """Select every image currently displayed (filters/search/burst view respected)."""
        if getattr(self, "view_mode", "") != "gallery":
            return
        files = self._gallery_files_for_display()
        self._gallery_set_selection_keys({_norm_path(p) for p in files if p})
        self._gallery_selection_anchor = None
        self._gallery_last_select_action = None

    def _prune_gallery_selection_to_displayed(self, display_files=None) -> None:
        """Drop selected keys that are not in the currently displayed set.

        Batch actions must never touch an image the user can no longer see:
        rating filters, semantic-search results and folder reloads can all
        narrow the displayed list while old keys linger in the selection set.
        """
        if display_files is None:
            if getattr(self, "view_mode", "") != "gallery":
                return
            display_files = self._gallery_files_for_display()
        display_keys = {_norm_path(p) for p in display_files if p}
        selected = getattr(self, "_gallery_selected_paths", None)
        if not selected:
            return
        if selected - display_keys:
            self._gallery_selected_paths = selected & display_keys
            self._gallery_refresh_selection_ui()

    def _sync_gallery_selection_bar(self, n: int, shown: int) -> None:
        bar = getattr(self, "gallery_selection_bar", None)
        if bar is None:
            return
        bar.setVisible(
            getattr(self, "view_mode", "single") == "gallery" and shown > 0
        )
        if not bar.isVisible():
            return
        label = getattr(self, "gallery_selection_count_label", None)
        select_all = getattr(self, "gallery_select_all_btn", None)
        clear_btn = getattr(self, "gallery_clear_selection_btn", None)
        delete_btn = getattr(self, "gallery_delete_selection_btn", None)
        discard_btn = getattr(self, "gallery_discard_selection_btn", None)
        export_btn = getattr(self, "gallery_export_selection_btn", None)
        if label is not None:
            label.setText(f"{n} selected" if n else "")
        for btn in (select_all, clear_btn):
            if btn is not None:
                btn.setEnabled(shown > 0)
        for btn in (delete_btn, discard_btn, export_btn):
            if btn is not None:
                btn.setEnabled(n > 0)

    def _sync_gallery_selection_chrome(self) -> None:
        """Update the gallery selection top bar; status text elsewhere."""
        vm = getattr(self, "view_mode", "single")
        n = len(self._gallery_selected_expanded_norm_keys())
        shown = len(self._gallery_files_for_display()) if vm == "gallery" else 0
        self._sync_gallery_selection_bar(n, shown)
        if n <= 0 or vm == "gallery":
            return
        if hasattr(self, "status_bar") and self.status_bar:
            hint = "Share -- open in editor, Delete/Down bulk actions"
            self.status_bar.showMessage(
                f"{n} image{'s' if n != 1 else ''} selected -- {hint}",
                5000,
            )

    def _gallery_has_bookmarks(self) -> bool:
        return bool(getattr(self, "_gallery_bookmarked_paths", None))

    def _is_gallery_path_bookmarked(self, file_path: str) -> bool:
        return _norm_path(file_path) in getattr(self, "_gallery_bookmarked_paths", set())

    def _gallery_toggle_path_bookmark(self, file_path: str) -> None:
        canonical = self._gallery_canonical_path(file_path)
        if not canonical:
            return
        key = _norm_path(canonical)
        bookmarked = getattr(self, "_gallery_bookmarked_paths", set())
        if key in bookmarked:
            bookmarked.discard(key)
            msg = "Bookmark removed"
        else:
            bookmarked.add(key)
            msg = "Bookmarked for opening in another app"
        self._gallery_bookmarked_paths = bookmarked
        self._apply_gallery_bookmark_visual_updates()
        if hasattr(self, "status_bar") and self.status_bar:
            self.status_bar.showMessage(msg, 2000)

    def _apply_gallery_bookmark_visual_updates(self) -> None:
        self._refresh_filmstrip_bookmark_visuals()
        self._refresh_gallery_bookmark_visuals()
        self._sync_bookmark_indicator()
        cleared_empty_filter = self._maybe_clear_empty_bookmark_filter()
        if (
            getattr(self, "view_mode", "") == "gallery"
            and getattr(self, "_gallery_bookmark_filter_active", False)
        ):
            self._apply_gallery_bookmark_filter()
        elif getattr(self, "view_mode", "") == "gallery":
            self._sync_share_button_visibility()
            if cleared_empty_filter:
                self._apply_gallery_bookmark_filter()
        if getattr(self, "view_mode", "single") == "single" and getattr(
            self, "_gallery_bookmark_filter_active", False
        ):
            self._sync_filmstrip_to_folder()
        self._save_persisted_gallery_bookmarks()

    def _gallery_toggle_selected_bookmarks(self) -> None:
        """Toggle bookmark state for each selected gallery image."""
        paths = self._gallery_selected_canonical_paths()
        if not paths:
            return
        bookmarked = getattr(self, "_gallery_bookmarked_paths", set())
        added = 0
        removed = 0
        for p in paths:
            canonical = self._gallery_canonical_path(p)
            if not canonical:
                continue
            key = _norm_path(canonical)
            if key in bookmarked:
                bookmarked.discard(key)
                removed += 1
            else:
                bookmarked.add(key)
                added += 1
        self._gallery_bookmarked_paths = bookmarked
        self._apply_gallery_bookmark_visual_updates()
        if hasattr(self, "status_bar") and self.status_bar:
            if added and not removed:
                msg = f"Bookmarked {added} image{'s' if added != 1 else ''}"
            elif removed and not added:
                msg = f"Removed bookmark from {removed} image{'s' if removed != 1 else ''}"
            else:
                msg = (
                    f"Updated bookmarks ({added} added, {removed} removed)"
                    if added or removed
                    else "No bookmarks changed"
                )
            self.status_bar.showMessage(msg, 2500)

    def _gallery_bookmarked_canonical_paths(self) -> list[str]:
        return self._filter_paths_to_bookmarked(list(getattr(self, "image_files", []) or []))

    def _gallery_files_for_display(self) -> list[str]:
        """Gallery file list, optionally narrowed to bookmarked images only."""
        if getattr(self, "view_mode", "") != "gallery":
            return list(getattr(self, "image_files", []) or [])
        return self._gallery_effective_files()

    def _gallery_viewport_scroll_path(self) -> str | None:
        """Resolve the path shown in the gallery grid for scroll-to-current."""
        path = (
            getattr(self, "current_file_path", None)
            or getattr(self, "_gallery_scroll_target_path", None)
        )
        if not path:
            return None
        gj = getattr(self, "gallery_justified", None)
        if gj is None or not hasattr(gj, "_resolve_gallery_path"):
            return path
        resolved = gj._resolve_gallery_path(path)
        if resolved:
            return resolved
        norm = _norm_path(path)
        path_to_group = getattr(self, "_burst_path_to_group", {}) or {}
        for group_id, members in path_to_group.items():
            if not any(_norm_path(m) == norm for m in members):
                continue
            covers = dict(getattr(self, "_burst_group_covers", {}) or {})
            covers.update(getattr(self, "_burst_stack_covers", {}) or {})
            cover = covers.get(group_id) or (members[0] if members else None)
            if cover:
                resolved = gj._resolve_gallery_path(cover)
                if resolved:
                    return resolved
            for member in members:
                resolved = gj._resolve_gallery_path(member)
                if resolved:
                    return resolved
            break
        return path

    def _scroll_gallery_to_current_file(self, *, backup: bool = True) -> None:
        """Scroll the gallery so the current single-view file is in the viewport."""
        from PyQt6.QtCore import QTimer

        gj = getattr(self, "gallery_justified", None)
        if gj is None or not hasattr(gj, "scroll_to_file"):
            return
        anchor = self._gallery_viewport_scroll_path()
        if not anchor:
            return
        gj.scroll_to_file(anchor)
        if backup:
            QTimer.singleShot(300, lambda p=anchor: gj.scroll_to_file(p))
            QTimer.singleShot(650, lambda p=anchor: gj.scroll_to_file(p))

    def _on_gallery_clear_filter_clicked(self) -> None:
        self._clear_gallery_bookmark_filter()
        self._clear_gallery_rating_filter()

    def _sync_gallery_clear_filter_button(self) -> None:
        btn = getattr(self, "gallery_clear_filter_button", None)
        if btn is None:
            return
        in_gallery = getattr(self, "view_mode", "") == "gallery"
        filter_active = bool(
            getattr(self, "_gallery_bookmark_filter_active", False)
        ) or int(getattr(self, "_gallery_rating_filter_min", 0) or 0) > 0
        btn.setVisible(in_gallery and filter_active)

    def _clear_gallery_bookmark_filter(self) -> bool:
        """Leave bookmark-only gallery view; keep active search results if any."""
        if getattr(self, "view_mode", "") != "gallery":
            return False
        if not getattr(self, "_gallery_bookmark_filter_active", False):
            return False
        self._gallery_bookmark_filter_active = False
        self._sync_bookmark_indicator()
        self._apply_gallery_bookmark_filter()
        return True

    def _gallery_scope_filter_qualifier(self) -> str:
        """Short qualifier for the active bookmark/rating gallery filter(s), e.g. 'bookmarked, 3★+'."""
        bookmark_active = bool(getattr(self, "_gallery_bookmark_filter_active", False))
        min_rating = int(getattr(self, "_gallery_rating_filter_min", 0) or 0)
        parts = []
        if bookmark_active:
            parts.append("bookmarked")
        if min_rating > 0:
            parts.append(f"{min_rating}★+")
        return ", ".join(parts)

    def _gallery_scope_filter_description(self) -> str:
        """Human-readable 'images' phrase for the active bookmark/rating gallery filter(s)."""
        qualifier = self._gallery_scope_filter_qualifier()
        return f"{qualifier} images" if qualifier else "images"

    def _apply_gallery_bookmark_filter(self) -> None:
        if getattr(self, "view_mode", "") != "gallery":
            return
        self._maybe_clear_empty_bookmark_filter()
        self._maybe_clear_empty_rating_filter()
        self._sync_gallery_rating_filter_stars()
        self._sync_gallery_clear_filter_button()
        filter_active = bool(
            getattr(self, "_gallery_bookmark_filter_active", False)
        ) or int(getattr(self, "_gallery_rating_filter_min", 0) or 0) > 0
        display_files = self._gallery_files_for_display()
        display_keys = {_norm_path(p) for p in display_files}
        selected = set(getattr(self, "_gallery_selected_paths", set()) or set())
        if selected - display_keys:
            self._gallery_selected_paths = selected & display_keys
            self._gallery_refresh_selection_ui()
        gj = getattr(self, "gallery_justified", None)
        if gj is not None:
            meta = getattr(self, "_gallery_bulk_metadata", None) or {}
            if filter_active:
                if not display_files:
                    scope_desc = self._gallery_scope_filter_description()
                    empty_msg = (
                        f"No matching {scope_desc}"
                        if self._is_semantic_search_filter_active()
                        else f"No {scope_desc}"
                    )
                    gj.set_images([], meta if meta else None)
                    gj.show_empty_message(empty_msg)
                else:
                    if hasattr(gj, "hide_empty_message"):
                        gj.hide_empty_message()
                    gj.set_images(display_files, meta if meta else None)
            else:
                if hasattr(gj, "hide_empty_message"):
                    gj.hide_empty_message()
                anchor = None
                if hasattr(gj, "get_scroll_anchor_path"):
                    try:
                        anchor = gj.get_scroll_anchor_path()
                    except Exception:
                        anchor = None
                display_files = list(getattr(self, "image_files", []) or [])
                if not display_files:
                    gj.set_images([], meta if meta else None)
                    empty_msg = (
                        "No matching images"
                        if self._is_semantic_search_filter_active()
                        else "No images"
                    )
                    gj.show_empty_message(empty_msg)
                else:
                    self._sync_gallery_to_folder_files(
                        meta if meta else None,
                        scroll_anchor_path=anchor,
                        force=True,
                    )
        self._sync_gallery_scrollbar_policy()
        self._update_gallery_counter()
        self._sync_share_button_visibility()
        self._sync_slideshow_button_visibility()
        if hasattr(self, "status_bar") and self.status_bar:
            if filter_active:
                n = len(display_files)
                scope_desc = self._gallery_scope_filter_description()
                if self._is_semantic_search_filter_active():
                    self.status_bar.showMessage(
                        f"Showing {n} matching {scope_desc}",
                        2500,
                    )
                else:
                    self.status_bar.showMessage(
                        f"Showing {n} {scope_desc}",
                        2500,
                    )
            elif self._is_semantic_search_filter_active():
                n = len(getattr(self, "image_files", []) or [])
                self.status_bar.showMessage(
                    f"Showing {n} search result{'s' if n != 1 else ''}",
                    2500,
                )
            else:
                self.status_bar.showMessage("Showing all images", 2000)

    def _refresh_filmstrip_bookmark_visuals(self) -> None:
        bar = self._filmstrip_bar()
        if bar is not None and hasattr(bar, "set_bookmarked_norm_paths"):
            bar.set_bookmarked_norm_paths(
                getattr(self, "_gallery_bookmarked_paths", set()) or set()
            )
        self._sync_bookmark_indicator()

    def _refresh_gallery_bookmark_visuals(self) -> None:
        gj = getattr(self, "gallery_justified", None)
        if gj is not None and hasattr(gj, "refresh_gallery_bookmark_visuals"):
            gj.refresh_gallery_bookmark_visuals()
        if gj is not None and hasattr(gj, "refresh_gallery_burst_visuals"):
            gj.refresh_gallery_burst_visuals()
        if gj is not None and hasattr(gj, "refresh_gallery_edited_visuals"):
            gj.refresh_gallery_edited_visuals()

    def _gallery_toggle_path_selection(self, file_path: str) -> None:
        canonical = self._gallery_canonical_path(file_path)
        if not canonical:
            return
        storage_key = _norm_path(canonical)
        member_keys = self._gallery_selection_keys_for_tile(file_path)
        selected = set(getattr(self, "_gallery_selected_paths", set()) or set())
        is_stack = len(member_keys) > 1
        if is_stack:
            is_on = storage_key in selected or member_keys <= selected
            if is_on:
                selected.discard(storage_key)
                selected -= member_keys
            else:
                selected.add(storage_key)
                selected |= member_keys
        elif storage_key in selected:
            selected.discard(storage_key)
        else:
            selected.add(storage_key)
        self._gallery_selection_anchor = storage_key
        self._gallery_last_select_action = "toggle"
        self._gallery_set_selection_keys(selected)

    def _gallery_selected_canonical_paths(self) -> list[str]:
        keys = self._gallery_selected_expanded_norm_keys()
        return self._gallery_selected_canonical_paths_from_keys(keys)

    def _after_gallery_bulk_file_removal(self, anchor_path: str | None = None) -> None:
        self._clear_gallery_selection()
        self._refresh_filmstrip_bookmark_visuals()
        self._refresh_gallery_bookmark_visuals()
        self._sync_share_button_visibility()
        self._save_persisted_gallery_bookmarks()
        if not self.image_files:
            had_semantic_scope = bool(
                self._semantic_search_backup_files or self._semantic_search_corpus_files
            )
            if had_semantic_scope:
                self._clear_semantic_search_results(silent=True, exit_to_gallery=True)
                inp = getattr(self, "gallery_search_input", None)
                if inp is not None:
                    inp.blockSignals(True)
                    try:
                        inp.clear()
                    finally:
                        inp.blockSignals(False)
            gj = getattr(self, "gallery_justified", None)
            if gj is not None:
                gj.set_images([])
                gj.show_empty_message("No images remaining in this folder")
            self.status_bar.showMessage("No images remaining in folder")
            return
        self._sync_gallery_to_folder_files(
            scroll_anchor_path=anchor_path,
            force=True,
        )
        gj = getattr(self, "gallery_justified", None)
        if gj is not None and hasattr(gj, "refresh_gallery_selection_visuals"):
            gj.refresh_gallery_selection_visuals()

    def delete_gallery_selection(self) -> None:
        paths = self._gallery_selected_canonical_paths()
        if not paths:
            self.status_bar.showMessage("No images selected in gallery", 2500)
            return
        self._stop_slideshow()
        if not self._confirm_bulk_deletion(len(paths), discard=False):
            return
        self._start_batch_removal("trash", paths)

    def discard_gallery_selection(self) -> None:
        paths = self._gallery_selected_canonical_paths()
        if not paths:
            self.status_bar.showMessage("No images selected in gallery", 2500)
            return
        self._stop_slideshow()
        if not self._confirm_bulk_deletion(len(paths), discard=True):
            return
        self._start_batch_removal("discard", paths)

    def _gallery_item_clicked(self, file_path):
        """Handle gallery item click - burst stack, single view, or selection."""
        mods = self._gallery_event_modifiers()
        if self._gallery_has_shift_modifier(mods) and not self._gallery_has_extend_modifier(mods):
            self._gallery_shift_click_selection(file_path)
            return
        if self._gallery_has_extend_modifier(mods):
            self._gallery_toggle_path_selection(file_path)
            return
        if (
            getattr(self, "view_mode", "") == "gallery"
            and not getattr(self, "_burst_group_view_active", False)
            and self._is_burst_grouping_enabled()
        ):
            members = self._burst_members_for_path(file_path)
            if members:
                self._enter_burst_group_view(members, cover=file_path)
                return
        self._clear_gallery_selection()
        # SYNC: Update current file index immediately to prevent navigation jumps
        target = _norm_path(file_path)
        if self.image_files:
            try:
                for i, p in enumerate(self.image_files):
                    if _norm_path(p) == target:
                        self.current_file_index = i
                        break
            except Exception:
                pass
        self.current_file_path = file_path

        # Mark that we're loading from gallery view - this will trigger full resolution load
        self._loading_from_gallery = True
        self._gallery_entry_accept_embedded = True
        # Reset the GPU view's OWN fit state BEFORE painting the instant tile:
        # the legacy variables below only cover the QLabel route. If the user
        # left the previous single image zoomed (fit_mode False), the tile
        # paint took set_pixmap's preserve-view branch and treated the NEW
        # image's small gallery tile as a same-content downgrade of the OLD
        # image -- s_new = s_old * old_w/tile_w blew up to ~14x ("clicking a
        # gallery image lands on a zoomed-in view", every image once the
        # stale non-fit state stuck).
        gv = getattr(self, "gpu_view", None)
        if gv is not None:
            gv._fit_mode = True
            gv._zoom_intent_100 = False
        # The canonical clear, not the partial hand-rolled reset below: the
        # PREVIOUS image's zoom-restore state (_restore_zoom_level,
        # _pending_zoom_toggle, _pending_gpu_zoom_point, ...) otherwise
        # re-applies its 100% to the NEWLY clicked image the moment its full
        # buffer lands ("clicking a gallery image zooms to 100%").
        self._clear_all_zoom_state()
        self._paint_instant_preview_for_path(file_path, prefer_gallery=True)

        # CRITICAL: Reset zoom state to fit-to-window when loading from gallery
        # This ensures we don't land on a zoomed-in view
        self.fit_to_window = True
        self.current_zoom_level = 1.0
        self.zoom_center_point = None
        # Clear any pending zoom state
        if hasattr(self, '_pending_zoom'):
            self._pending_zoom = False
        if hasattr(self, '_pending_zoom_restore'):
            self._pending_zoom_restore = False
        if hasattr(self, '_restore_zoom_center'):
            self._restore_zoom_center = None
        if hasattr(self, '_restore_zoom_level'):
            self._restore_zoom_level = None

        # Settle-to-fit guarantee. Multiple independent races have landed a
        # gallery click at 100% (stale-viewport fit misreads, leaked
        # zoom-restore state, late prefetched full-res buffers taking the
        # preserve path); each was fixed, and each time the field found
        # another. This closes the CLASS: shortly after the transition, if
        # the view ended non-fit and the user has not zoomed since (any user
        # zoom sets _zoom_intent_100 or a pending-zoom flag), force fit and
        # log which state needed correcting so the residual path stays
        # visible in logs instead of on screen.
        self._gallery_fit_token = token = object()

        def _enforce_gallery_fit() -> None:
            if getattr(self, "_gallery_fit_token", None) is not token:
                return
            g = getattr(self, "gpu_view", None)
            if g is None or not g.has_pixmap():
                return
            user_zoomed = bool(
                getattr(g, "_zoom_intent_100", False)
                or getattr(self, "_pending_zoom_toggle", False)
                or getattr(self, "_pending_gpu_zoom_point", None) is not None
                or getattr(self, "_pending_zoom", False)
            )
            if user_zoomed or g.is_fit_mode():
                return
            logging.getLogger(__name__).info(
                "[GALLERY_FIT] Correcting non-fit landing after gallery click "
                "(scale=%.4f fit=%.4f) -- residual zoom path still active",
                g.current_scale(), g.fit_scale(),
            )
            g.fit_to_window()

        # Two shots only: the transition's own paints land within ~400ms, and
        # a longer window risks undoing a genuine wheel zoom (which sets no
        # intent flag) made right after landing.
        QTimer.singleShot(80, _enforce_gallery_fit)
        QTimer.singleShot(450, _enforce_gallery_fit)

        self.view_mode = 'single'
        if hasattr(self, 'view_mode_button'):
            # Update icon if using qtawesome
            if qta is not None:
                try:
                    gallery_icon = _qta_icon('fa5s.th', color=theme.INK_MUTED)
                    self.view_mode_button.setIcon(gallery_icon)
                    self.view_mode_button.setIconSize(QSize(20, 20))
                    self.view_mode_button.setText("")  # Clear text if using icon
                except Exception:
                    self.view_mode_button.setText("Gallery")
            else:
                self.view_mode_button.setText("Gallery")
        self._show_single_view()

        # Reset orientation flag for new load from gallery
        self._orientation_already_applied = False

        def _load_gallery_thumbnail(self, file_path, label):
            """Load thumbnail for gallery item asynchronously"""
            # Try to get thumbnail from cache first
            try:
                # Check if we have a cached pixmap
                cached_pixmap = self.image_cache.get_pixmap(file_path)
                if cached_pixmap and not cached_pixmap.isNull():
                    # Scale to fit thumbnail size
                    scaled_pixmap = cached_pixmap.scaled(200, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    label.setPixmap(scaled_pixmap)
                    return

                # Try to load from file
                if os.path.exists(file_path):
                    # For non-RAW files, use safe loader (handles TIFF properly)
                    file_ext = os.path.splitext(file_path)[1].lower()
                    if file_ext not in ['.arw', '.cr2', '.nef', '.raf', '.orf', '.dng', '.cr3']:
                        pixmap = self._load_pixmap_safe(file_path)
                        if not pixmap.isNull():
                            # Ensure dimensions > 0 (already 200x150 here, but good for safety)
                            scaled_pixmap = pixmap.scaled(200, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                            label.setPixmap(scaled_pixmap)
                            return

                    # For RAW files, we'll need to extract thumbnail
                    # This is a placeholder - full implementation would use ThumbnailExtractor
                    label.setText("RAW\nLoading...")
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Error loading gallery thumbnail for {file_path}: {e}")
                label.setText("Error")

    def _update_sort_button_text(self):
        """Update the sort toggle button text based on current preference"""
        if hasattr(self, 'sort_toggle_button'):
            if self.get_sort_preference():
                self.sort_toggle_button.setText("⇅ Newest")
            else:
                self.sort_toggle_button.setText("⇅ Oldest")
        self._schedule_search_expand_overlay_sync()

    def _can_skip_gallery_to_single(self, requested_file_path):
        """Check if we can skip loading when transitioning from gallery to single view"""
        if not getattr(self, "_gallery_instant_display_quality", False):
            return False
        if _norm_path(getattr(self, "_displayed_content_path", "")) != _norm_path(requested_file_path):
            return False
        return self._single_view_pixels_on_screen(requested_file_path)

    def _finish_gallery_to_single_without_redisplay(self, requested_file_path):
        """Finish gallery to single view transition without redisplaying"""
        self._on_single_view_content_displayed()

    def on_thumbnail_fallback(self, message):
        """RAW fast preview (embedded JPEG / cache). Normal path -- not a user-facing warning."""
        import logging

        logger = logging.getLogger(__name__)
        logger.debug("RAW preview thumbnail: %s", message)
        # Optional dev-only status hint (default off -- avoids truncated "warning thumbnail…" text).
        if os.environ.get("RAWVIEWER_VERBOSE_RAW_PREVIEW", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ) and self.current_file_path:
            name = os.path.basename(_require_str_path(self.current_file_path))
            self.status_bar.showMessage(f"{name}: Loading preview…")

    def on_thumbnail_ready(self, thumbnail):
        """Handle when thumbnail is ready for immediate display."""
        if thumbnail is not None:
            # Smart thumbnail display: only show thumbnail if it makes sense
            if self._should_show_thumbnail() and not self._full_res_only_mode():
                self.display_numpy_image(thumbnail)
                self.status_bar.showMessage(
                    "Preview loaded - processing full image...")
            else:
                # Store thumbnail but don't display it yet
                self._pending_thumbnail = thumbnail
                self.status_bar.showMessage(
                    "Processing full image for quality evaluation...")

    def _should_show_thumbnail(self):
        """Determine if we should show thumbnail immediately or wait for full image."""
        if self._suppress_jpeg_interim():
            return False
        # If user explicitly wants thumbnails even when zoomed, always show
        if self.show_thumbnails_when_zoomed:
            return True

        # Don't show thumbnail if user is in 100% zoom mode (checking sharpness)
        if not self.fit_to_window:
            return False

        if getattr(self, "_preserve_nav_zoom_active", False):
            return False

        # Don't show thumbnail if we're maintaining zoom state from navigation
        # (user was previously at 100% zoom checking sharpness)
        if hasattr(self, '_maintain_zoom_on_navigation'):
            return False

        if getattr(self, "_pending_zoom_restore", False):
            return False

        # Don't show thumbnail if we're restoring zoom state to 100%
        if getattr(self, "_restore_zoom_center", None) is not None:
            return False
        return getattr(self, "_restore_zoom_level", None) is None

    def _gallery_slideshow_scope(self) -> list[str]:
        if getattr(self, "_gallery_bookmark_filter_active", False):
            return self._gallery_files_for_display()
        return list(getattr(self, "image_files", []) or [])

    def _abort_session_restore_heavy_for_gallery(self) -> None:
        """Drop session-restore staging when the user switches to gallery.

        Gallery tiles share the load pool; continuing deferred full decode /
        nav preload (or a forever-retry timer) starves visible-tile fill.
        Clearing the defer flags also unblocks idle CuPy import scheduling
        once the gallery is no longer active.
        """
        if not (
            getattr(self, "_session_restore_defer_preload", False)
            or getattr(self, "_session_restore_staged_scheduled", False)
        ):
            return
        import logging

        logging.getLogger(__name__).info(
            "[SESSION] Aborting staged restore loads for gallery entry"
        )
        self._session_restore_defer_preload = False
        self._session_restore_staged_scheduled = False
        self._session_restore_full_decode_allowed = False
        timer = getattr(self, "_gpu_backend_import_timer", None)
        if timer is not None:
            with contextlib.suppress(Exception):
                timer.stop()

    def _gallery_has_thumbnail_load_pressure(self) -> bool:
        """True when gallery thumbnail decodes are still in flight (defer heavy background work)."""
        if getattr(self, "view_mode", "") != "gallery":
            return False
        gj = getattr(self, "gallery_justified", None)
        if gj is None:
            return False
        if hasattr(gj, "_gallery_warmup_active") and gj._gallery_warmup_active():
            return True
        if not getattr(gj, "_first_thumb_ready_after_set", False):
            return True
        active = len(getattr(gj, "_active_tasks", {}) or {})
        # Default 1: any in-flight gallery thumb work should defer INDEX prep --
        # the old default of 16 let metadata prep start mid-fill and then freeze
        # the UI when the done slot re-scanned thousands of paths.
        threshold = _env_int("RAWVIEWER_GALLERY_INDEX_DEFER_ACTIVE", 1, minimum=1)
        return active >= threshold

    def _scroll_gallery_vertical(self, direction: int) -> bool:
        """Scroll gallery when in gallery mode. direction +1 = down, -1 = up."""
        if getattr(self, "view_mode", "single") != "gallery":
            return False
        gs = getattr(self, "gallery_scroll", None)
        if gs is None:
            return False
        sb = gs.verticalScrollBar()
        if sb is None:
            return False
        from rawviewer_ui.gallery_scroll import key_scroll_delta_px

        step = key_scroll_delta_px(sb.singleStep())
        nval = sb.value() + direction * step
        nval = max(sb.minimum(), min(sb.maximum(), nval))
        sb.setValue(nval)
        return True

    def _position_gallery_frame_grip(self) -> None:
        """Place the gallery right-edge resize overlay over the scrollbar frame margin."""
        overlay = getattr(self, "gallery_frame_gutter", None)
        container = getattr(self, "gallery_widget", None)
        if overlay is None or container is None:
            return
        in_gallery = getattr(self, "view_mode", "") == "gallery" and container.isVisible()
        if not in_gallery:
            overlay.hide()
            return
        strip = self._GALLERY_FRAME_RIGHT_GUTTER
        overlay.setGeometry(
            max(0, container.width() - strip),
            0,
            strip,
            container.height(),
        )
        overlay.show()
        overlay.raise_()

    def _gallery_frame_right_gutter(self) -> int:
        gutter = getattr(self, "gallery_frame_gutter", None)
        if gutter is not None and gutter.isVisible():
            return gutter.width()
        if getattr(self, "view_mode", "") == "gallery":
            return self._GALLERY_FRAME_RIGHT_GUTTER
        return 0

    def _mtime_sort_image_files(self, image_files, file_stats, newest_first=None):
        """Order folder entries by file mtime (and RAW/JPEG tie-breakers); no EXIF reads."""
        if not image_files:
            return []
        if newest_first is None:
            newest_first = self.get_sort_preference()
        sort_keys = {}
        for fp in image_files:
            mtime = file_stats.get(fp, (0, 0))[1]
            base_name = os.path.basename(fp).lower()
            stem = os.path.splitext(base_name)[0]
            ext = os.path.splitext(base_name)[1]
            raw_rank = 1 if is_raw_file(fp) else 0
            primary_ts = -mtime if newest_first else mtime
            sort_keys[fp] = (primary_ts, stem, raw_rank, ext, base_name)
        return sorted(image_files, key=lambda fp: sort_keys[fp])
