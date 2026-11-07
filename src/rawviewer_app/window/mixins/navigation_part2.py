"""navigation mixin, part 2 of 2 (9 methods, 336 lines)."""
from __future__ import annotations

import contextlib
import os
import sys

from core.config import _env_true, _norm_path

from .. import lazy
from ..paths import _SUBFOLDER_FALLBACK_MAX_DIRS, _SUBFOLDER_FALLBACK_SKIP

lazy.register(sys.modules[__name__])


class NavigationPart2:
    def scan_folder_for_images(self, file_path):
        """Scan the folder containing the given file for all image files"""
        import logging
        logger = logging.getLogger(__name__)

        try:
            # Get the folder path
            folder_path = os.path.abspath(os.path.dirname(file_path))
            prev_folder = getattr(self, "current_folder", None)
            # Already on screen -> just point at it. This also covers the
            # parent-fallback scope (files live one level down, so
            # dirname(file) != current_folder): never re-derive the folder
            # identity from a file path that is already loaded.
            target = _norm_path(file_path)
            if getattr(self, "image_files", None):
                for i, img_file in enumerate(self.image_files):
                    if _norm_path(img_file) == target:
                        self.current_file_index = i
                        self.current_file_path = img_file
                        return
            if (
                prev_folder is not None
                and _norm_path(prev_folder) == _norm_path(folder_path)
            ):
                # Unknown file in the same folder: nothing to update (no rescan).
                return
            if prev_folder is not None and _norm_path(prev_folder) != _norm_path(folder_path):
                self._reset_semantic_search_for_new_folder()

            # Validate folder
            if not os.path.isdir(folder_path):
                logger.warning(f"Folder does not exist: {folder_path}")
                self.show_error("Folder Not Found",
                              f"The folder does not exist:\n{folder_path}")
                return

            self.current_folder = folder_path
            self._sync_burst_rejected_for_current_folder()

            # Get supported extensions
            supported_extensions = self.get_supported_extensions()

            # Shared scanner: direct files, else one level of subfolders.
            image_files = []

            try:
                for ap, _stat in self._scan_folder_image_paths(folder_path):
                    image_files.append(ap)
            except OSError as e:
                error_msg = f"Cannot read folder contents:\n{str(e)}"
                logger.error(f"Error reading folder {folder_path}: {e}")
                self.show_error("Folder Access Error", error_msg)
                return

            # Check if any images were found
            if not image_files:
                logger.warning(f"No supported images found in folder: {folder_path}")
                # Display message in main viewing area instead of popup
                self.show_no_images_message(supported_extensions)
                # Reset state
                self.image_files = []
                self._semantic_search_corpus_files = []
                self._semantic_search_backup_files = None
                self._semantic_search_result_paths = None
                self._last_semantic_query = ""
                self._gallery_search_index_session_key = None
                self.current_file_index = -1
                self.current_file_path = None
                self.current_folder = None
                return

            # Sort files according to user preference
            # Ensure we only take the sorted file list (sort_image_files returns a tuple of (list, metadata_dict))
            sorted_files, _ = self.sort_image_files(image_files)
            self.image_files = sorted_files
            self._reconcile_burst_rejected_paths_with_folder()
            # Keep semantic search corpus aligned with the currently scanned folder.
            self._semantic_search_corpus_files = list(self.image_files)
            self._semantic_search_backup_files = None
            self._semantic_search_result_paths = None
            self._last_semantic_query = ""
            self._gallery_search_index_session_key = None

            # Find current file index
            self.current_file_index = -1

            # Normalize paths for comparison by converting to absolute paths
            # This handles both forward/backward slashes and case differences
            try:
                normalized_target = os.path.abspath(file_path)

                for i, img_file in enumerate(self.image_files):
                    normalized_img_file = os.path.abspath(img_file)
                    if normalized_target.lower() == normalized_img_file.lower():
                        self.current_file_index = i
                        break
            except Exception:
                # Fallback to original logic
                if file_path in self.image_files:
                    self.current_file_index = self.image_files.index(file_path)

            # Update status bar after scanning
            self.update_status_bar()

        except Exception as e:
            logger.error(f"Error scanning folder: {e}", exc_info=True)
            error_msg = f"Error scanning folder:\n{str(e)}"
            self.show_error("Folder Scan Error", error_msg)

    def _current_folder_label(self) -> str:
        """Human-readable current folder label for gallery top metadata."""
        folder = (getattr(self, "current_folder", None) or "").strip()
        if not folder:
            return ""
        norm = os.path.normpath(folder)
        base = os.path.basename(norm)
        return base or norm

    def move_current_image_to_discard(self):
        """Move the current image to a 'Discard' folder in the same directory"""
        if not self.current_file_path or not os.path.exists(self.current_file_path):
            self.show_error("Discard Error", "No image file to move.")
            return
        self._stop_slideshow()
        file_to_move = self.current_file_path
        self._move_path_to_discard(file_to_move)
        # --- Preserve zoom/pan state for next image (like navigation/discard) ---
        if not self.fit_to_window:
            self._preserve_nav_zoom_active = True
            self._maintain_zoom_on_navigation = True
            self._restore_zoom_center = self._zoom_anchor_for_navigation_restore()
            self._restore_zoom_level = self.current_zoom_level
            if getattr(self, "current_pixmap", None):
                self._restore_pixmap_size = self.current_pixmap.size()
            # Save current scroll position instead of start_scroll_x/y
            self._restore_start_scroll_x = self.scroll_area.horizontalScrollBar().value()
            self._restore_start_scroll_y = self.scroll_area.verticalScrollBar().value()
        else:
            self._preserve_nav_zoom_active = False
            if hasattr(self, "_maintain_zoom_on_navigation"):
                delattr(self, "_maintain_zoom_on_navigation")
            self._restore_zoom_center = None
            self._restore_zoom_level = None
            self._restore_start_scroll_x = None
            self._restore_start_scroll_y = None
        self.handle_post_deletion_navigation()

    def _scan_folder_generator(self, folder_path, extensions):
        """Yield (path, stat) for supported images directly in folder_path (no subfolders)."""
        try:
            with os.scandir(folder_path) as it:
                for entry in it:
                    if entry.name.startswith('.'):
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext not in extensions:
                        continue
                    if ext == ".dng":
                        from common_image_loader import dng_prefers_embedded_preview_first
                        if dng_prefers_embedded_preview_first(entry.path):
                            continue
                    try:
                        stat = entry.stat()
                        if stat.st_size > 0:
                            yield entry.path, stat
                    except OSError:
                        pass
        except (OSError, PermissionError):
            pass

    def _resolve_start_file_path(self, folder_path: str, start_file: str) -> str | None:
        """Resolve start_file (basename or path) to an absolute file path in folder_path."""
        if not folder_path or not start_file:
            return None
        if os.path.isabs(start_file) and os.path.isfile(start_file):
            return os.path.abspath(start_file)
        direct = os.path.join(folder_path, start_file)
        if os.path.isfile(direct):
            return os.path.abspath(direct)
        base_norm = os.path.normcase(os.path.basename(start_file))
        try:
            with os.scandir(folder_path) as it:
                for entry in it:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    if os.path.normcase(entry.name) == base_norm:
                        return os.path.abspath(entry.path)
        except OSError:
            pass
        return None

    def _scan_subfolder_dirs(self, folder_path: str) -> list[str]:
        """Immediate child directories eligible for the parent fallback.

        Depth 1 only: skip hidden dirs and ``Discard`` (rejected shots must not
        resurrect), never follow symlinks, sort deterministically, and cap the
        count so opening a drive root or a large tree stays fast.
        """
        try:
            with os.scandir(folder_path) as it:
                children = [
                    entry.path
                    for entry in it
                    if not entry.name.startswith(".")
                    and os.path.normcase(entry.name) not in _SUBFOLDER_FALLBACK_SKIP
                    and entry.is_dir(follow_symlinks=False)
                ]
        except (OSError, PermissionError):
            return []
        children.sort(key=os.path.normcase)
        return children[:_SUBFOLDER_FALLBACK_MAX_DIRS]

    def _folder_has_supported_images(self, folder_path: str) -> bool:
        """Cheap first-hit check mirroring ``_scan_folder_image_paths`` (incl. fallback).

        Used by the drop handler so "folder has images" and the later scan can
        never disagree (they must consult the same scanner).
        """
        extensions = set(self.get_supported_extensions())
        if next(self._scan_folder_generator(folder_path, extensions), None) is not None:
            return True
        for child in self._scan_subfolder_dirs(folder_path):
            if next(self._scan_folder_generator(child, extensions), None) is not None:
                return True
        return False

    def _scan_folder_image_paths(self, folder_path: str):
        """Yield absolute paths and file stats for supported images.

        Reads folder_path's direct files; when it has none, falls back one level
        to its immediate subfolders (Sony/card layout: DCIM/100MSDCF/*.ARW). The
        fallback is all-or-nothing - child files are never merged into a
        non-empty top level. Never deeper than one level.
        """
        extensions = set(self.get_supported_extensions())
        seen_paths = set()

        def _collect(path):
            for full_path, stat_info in self._scan_folder_generator(path, extensions):
                ap = os.path.abspath(full_path)
                if ap in seen_paths:
                    continue
                seen_paths.add(ap)
                yield ap, stat_info

        yield from _collect(folder_path)
        if seen_paths or _env_true("RAWVIEWER_NO_SUBFOLDER_FALLBACK"):
            return
        for child in self._scan_subfolder_dirs(folder_path):
            yield from _collect(child)

    def _apply_instant_single_file_open(
        self,
        folder_path: str,
        start_path: str,
        token: int,
        *,
        preserve_folder_index: bool = False,
    ) -> None:
        """Show the requested image immediately while the folder scan continues in background."""
        import logging

        logger = logging.getLogger(__name__)
        self._folder_fast_open_token = token
        self._folder_fast_open_preserved_path = start_path
        self.current_folder = folder_path
        self._sync_burst_rejected_for_current_folder()
        prev_path = getattr(self, "current_file_path", None)
        if preserve_folder_index and getattr(self, "image_files", None):
            start_norm = _norm_path(start_path)
            idx = 0
            for i, fp in enumerate(self.image_files):
                if _norm_path(fp) == start_norm:
                    idx = i
                    start_path = fp
                    break
            self.current_file_index = idx
            self.current_file_path = start_path
            self._semantic_search_corpus_files = list(self.image_files)
            self._folder_sort_refinement_applied_token = token
        else:
            self.image_files = [start_path]
            self._semantic_search_corpus_files = [start_path]
            self._gallery_search_index_session_key = None
            self.current_file_index = 0
            self.current_file_path = start_path
            self._quick_folder_file_stats = {}
            self._filmstrip_warmed_paths = set()
            self._gallery_bulk_metadata = {}
            self._bump_gallery_rating_presence_epoch()
        try:
            if getattr(self, "image_manager", None) is not None:
                if preserve_folder_index and prev_path:
                    self.image_manager.cancel_task(prev_path)
                else:
                    self.image_manager.cancel_all_tasks()
        except Exception:
            pass
        self._displayed_content_path = None
        self._manager_display_track_path = None
        self._manager_displayed_max_dim = 0
        self._last_loaded_path = None
        self._last_manager_exif_path = None
        self._last_manager_exif_ts = 0.0
        self._last_manager_exif_was_minimal = False
        self._defer_full_exif_path = None
        self._full_exif_queued_norm = None
        self.current_image = None
        self.current_pixmap = None
        self._is_half_size_displayed = False
        self._full_resolution_loading = False
        self._preserve_nav_zoom_active = False
        self._pending_zoom_restore = False
        self._restore_zoom_center = None
        self._restore_zoom_level = None
        if hasattr(self, "_maintain_zoom_on_navigation"):
            with contextlib.suppress(AttributeError):
                delattr(self, "_maintain_zoom_on_navigation")
        if hasattr(self, "image_label"):
            self.image_label.clear()
        if hasattr(self, "loading_overlay"):
            self.loading_overlay.show_loading("Loading image...")
        if getattr(self, "view_mode", "single") != "single":
            self.view_mode = "single"
        if preserve_folder_index:
            logger.info(
                "[FOLDER] Same-folder fast open: %s (keeping folder index, no rescan)",
                os.path.basename(start_path),
            )
        else:
            logger.info(
                "[FOLDER] Fast single-file open: %s (quick index + EXIF sort continue in background)",
                os.path.basename(start_path),
            )
        # Start preview decode before single-view UI work so it overlaps gallery teardown.
        self._skip_next_single_view_load = True
        self.load_raw_image(start_path)
        self._show_single_view()
        if not preserve_folder_index:
            self._start_quick_folder_index(folder_path, start_path, token)
