
"""misc mixin, part 2 of 4 (31 methods, 880 lines)."""
from __future__ import annotations

import contextlib
import logging
import os
import sys
import time

from PyQt6.QtCore import QPoint, QSize, Qt, QTimer
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QFileDialog

from core import theme
from core.config import _norm_path

from .. import lazy
from ..lazy import image_covers_sensor_resolution, is_raw_file
from ..paths import _is_in_folder_scope, _require_str_path

lazy.register(sys.modules[__name__])


class MiscPart2:
    def _update_raw_toggle_button_state(self):
        """Update the text, tooltip and styling of the RAW/JPEG toggle button."""
        if not hasattr(self, "raw_toggle_button"):
            return

        from PyQt6.QtGui import QIcon

        self.raw_toggle_button.setIcon(QIcon())
        self.raw_toggle_button.setText("RAW")
        self.raw_toggle_button.setFixedSize(44, 28)

        settings = self.get_settings()
        use_embedded = settings.value("use_embedded_jpeg_workflow", True, type=bool)

        btn_font = self.raw_toggle_button.font()
        btn_font.setBold(not use_embedded)
        btn_font.setPointSize(13 if use_embedded else 16)
        self.raw_toggle_button.setFont(btn_font)

        raw_btn_base = """
            QPushButton {
                padding: 0px;
                border: none;
                background: transparent;
                text-align: center;
                letter-spacing: 0.5px;
                min-width: 44px;
                max-width: 44px;
                min-height: 28px;
                max-height: 28px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.05);
                border-radius: 4px;
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 0.1);
            }
        """

        if use_embedded:
            self.raw_toggle_button.setToolTip(
                "Embedded JPEG workflow (Fast)\nClick to toggle to RAW image workflow (High Quality)"
            )
            self.raw_toggle_button.setStyleSheet(raw_btn_base + f"""
                QPushButton {{
                    color: {theme.INK_MUTED};
                    font-size: 13px;
                    font-weight: 500;
                }}
                QPushButton:hover {{
                    color: {theme.INK};
                }}
            """)
        else:
            self.raw_toggle_button.setToolTip(
                "RAW workflow (High Quality)\nClick to toggle to Embedded JPEG workflow (Fast)"
            )
            self.raw_toggle_button.setStyleSheet(raw_btn_base + f"""
                QPushButton {{
                    color: {theme.INK};
                    background-color: {theme.EMBER};
                    border-radius: 4px;
                    font-size: 16px;
                    font-weight: 700;
                }}
                QPushButton:hover {{
                    color: {theme.INK};
                    background-color: {theme.EMBER};
                }}
                QPushButton:pressed {{
                    color: {theme.INK};
                    background-color: {theme.EMBER};
                }}
            """)

    def _update_edr_toggle_button_state(self):
        """Update text/tooltip/styling of the EDR display toggle button."""
        btn = getattr(self, "edr_toggle_button", None)
        if btn is None:
            return
        btn.setText("EDR")
        settings = self.get_settings()
        enabled = settings.value("raw_edr_display_enabled", True, type=bool)
        base = """
            QPushButton {
                padding: 0px;
                border: none;
                background: transparent;
                text-align: center;
                letter-spacing: 0.5px;
                min-width: 44px;
                max-width: 44px;
                min-height: 28px;
                max-height: 28px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.05);
                border-radius: 4px;
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 0.1);
            }
        """
        if enabled:
            btn.setToolTip(
                "EDR display on (High Quality highlights, slower RAW browsing)\n"
                "Click to switch to standard display (Fast)"
            )
            btn.setStyleSheet(base + f"""
                QPushButton {{
                    color: {theme.INK_MUTED};
                    font-size: 13px;
                    font-weight: 700;
                }}
                QPushButton:hover {{
                    color: {theme.INK};
                    background-color: rgba(255, 255, 255, 0.08);
                }}
            """)
        else:
            btn.setToolTip(
                "EDR display off (Fast, standard dynamic range)\n"
                "Click to enable EDR highlight headroom (slower RAW browsing)"
            )
            btn.setStyleSheet(base + f"""
                QPushButton {{
                    color: {theme.INK_FAINT};
                    font-size: 13px;
                    font-weight: 500;
                }}
                QPushButton:hover {{
                    color: {theme.INK_MUTED};
                }}
            """)

    def is_animated_image(self, file_path: str) -> bool:
        if not file_path:
            return False
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.gif':
            return True
        if ext == '.webp':
            try:
                from PIL import Image
                with Image.open(file_path) as img:
                    return getattr(img, "is_animated", False) and int(getattr(img, "n_frames", 1)) > 1
            except Exception:
                return False
        return False

    def _stop_animations(self):
        if hasattr(self, "_current_movie") and self._current_movie:
            with contextlib.suppress(Exception):
                self._current_movie.stop()
            self.image_label.setMovie(None)
            self._current_movie = None

        if hasattr(self, "_webp_timer") and self._webp_timer:
            with contextlib.suppress(Exception):
                self._webp_timer.stop()
            self._webp_timer = None

        self._current_webp_frames = []
        self._current_webp_durations = []
        self._current_webp_idx = 0
        self._webp_decode_generation = getattr(self, "_webp_decode_generation", 0) + 1
        self._webp_decode_in_progress_path = None
        self._original_movie_size = None

    def _animated_playback_active(self, file_path: str | None = None) -> bool:
        """True when GIF/WebP animation is loaded or decoding for this path."""
        path = file_path or getattr(self, "current_file_path", None)
        if not path or not self.is_animated_image(path):
            return False
        if _norm_path(path) != _norm_path(getattr(self, "current_file_path", None) or ""):
            return False
        if getattr(self, "_current_movie", None) is not None:
            return True
        if getattr(self, "_current_webp_frames", None):
            return len(self._current_webp_frames) > 0
        pending = getattr(self, "_webp_decode_in_progress_path", None)
        return bool(pending and _norm_path(pending) == _norm_path(path))

    def _update_webp_scaling(self):
        if hasattr(self, "_current_webp_frames") and self._current_webp_frames:
            idx = (self._current_webp_idx - 1) % len(self._current_webp_frames)
            pixmap = self._current_webp_frames[idx]
            display_pixmap = pixmap
            if self.fit_to_window:
                v_size = self.scroll_area.viewport().size()
                display_pixmap = pixmap.scaled(v_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(display_pixmap)
            self.image_label.resize(display_pixmap.size())

    def _update_gif_scaling(self):
        if not hasattr(self, "_current_movie") or not self._current_movie:
            return
        orig_size = getattr(self, "_original_movie_size", None)
        if not orig_size or orig_size.isEmpty():
            return
        if self.fit_to_window:
            v_size = self.scroll_area.viewport().size()
            w_ratio = v_size.width() / orig_size.width()
            h_ratio = v_size.height() / orig_size.height()
            ratio = min(w_ratio, h_ratio)
            fitted_size = QSize(int(orig_size.width() * ratio), int(orig_size.height() * ratio))
            self._current_movie.setScaledSize(fitted_size)
            self.image_label.resize(fitted_size)
        else:
            self._current_movie.setScaledSize(orig_size)
            self.image_label.resize(orig_size)

    def _on_view_mode_button_clicked(self) -> None:
        if getattr(self, "view_mode", "") == "compare":
            self._exit_compare_mode()
            return
        if getattr(self, "_burst_group_view_active", False):
            if getattr(self, "view_mode", "") == "single":
                if not self._path_in_burst_group_view(getattr(self, "current_file_path", None)):
                    self._exit_burst_group_view()
                self.view_mode = "gallery"
                self._show_gallery_view()
                return
            self._exit_burst_group_view()
            return
        self.toggle_view_mode()

    def toggle_view_mode(self):
        """Toggle between single image view and gallery view"""
        if getattr(self, '_is_delete_dialog_open', False):
            return
        # Do not block gallery entry on EXIF sort. Large cold folders can take
        # minutes to probe; _apply_folder_sort_refinement still reorders later.
        if not self._is_exif_sort_ready():
            import logging
            logging.getLogger(__name__).info(
                "[VIEW_MODE] EXIF sort still running; entering gallery on provisional order"
            )
        if getattr(self, "view_mode", "") == "compare":
            self._exit_compare_mode()
            return
        self._last_view_mode_switch_ts = time.monotonic()
        import logging
        logger = logging.getLogger(__name__)
        logger.debug("[MODESWITCH] toggle_view_mode called; current=%s", self.view_mode)
        logger.info("[VIEW_MODE] ========== toggle_view_mode() STARTED ==========")

        if self.view_mode == 'single':
            logger.info("[VIEW_MODE] Switching from single to gallery mode")
            self._disable_raw_recovery()
            self._gallery_entry_accept_embedded = False
            self.view_mode = 'gallery'
            try:
                if hasattr(self, "_resume_indexing_timer") and self._resume_indexing_timer:
                    self._resume_indexing_timer.stop()
                self._pause_semantic_indexing_deferred()
            except Exception:
                pass
            if getattr(self, "_focus_subject_outline_active", False):
                self._focus_subject_outline_active = False
                self._focus_subject_rect_image = None
                self._focus_rect_source = None
        else:
            logger.info("[VIEW_MODE] Switching from gallery to single mode")
            self.view_mode = 'single'
            self._adopt_gallery_navigation_path()
            self._loading_from_gallery = True
            self._gallery_entry_accept_embedded = True

        logger.info("[VIEW_MODE] Mode changed, calling view method (elapsed: 0.000s)")
        if self.view_mode == 'gallery':
            self._stop_slideshow()
            self._show_gallery_view()
        else:
            self._show_single_view()
        logger.debug("[MODESWITCH] toggle_view_mode finished; current=%s", self.view_mode)

        logger.info("[VIEW_MODE] ========== toggle_view_mode() COMPLETED in 0.004s ==========")

    def register_new_file(self, path: str, *, select: bool = False) -> bool:
        """Bring a file this app just wrote into the open folder, live.

        Exports land beside the originals, but nothing told the gallery: there
        is no filesystem watcher anywhere in the app, and image_files is only
        ever built by a folder scan. So a new file stayed invisible until the
        folder was reopened.

        Insertion rather than a rescan: the app knows exactly what it wrote,
        so re-listing the directory would be strictly more work for strictly
        less certainty. Files written by other applications are still not
        noticed; that would need a watcher, which costs a descriptor per
        directory and behaves badly on network volumes.

        Returns whether the file was added.
        """
        try:
            if not path or not os.path.isfile(path):
                return False
            norm = _norm_path(path)
            files = list(getattr(self, "image_files", []) or [])
            if any(_norm_path(p) == norm for p in files):
                return False
            folder = os.path.dirname(os.path.abspath(path))
            current_folder = getattr(self, "current_folder", "") or ""
            # Parent-fallback scopes (card layouts) keep child-folder paths, so
            # accept anything still inside the open folder, not only siblings.
            if current_folder and not _is_in_folder_scope(folder, current_folder):
                # Written somewhere else; nothing on screen to update.
                return False

            # Place it where the current sort would have put it, so the tile
            # does not jump on the next real scan.
            insert_at = self._sorted_insert_index(files, path)
            files.insert(insert_at, path)
            self.image_files = files
            current = getattr(self, "current_file_path", None)
            if current:
                with contextlib.suppress(ValueError):
                    self.current_file_index = self.image_files.index(current)

            gj = getattr(self, "gallery_justified", None)
            if gj is not None and hasattr(gj, "set_images"):
                gj.set_images(list(self.image_files), force_rebuild=True)
            self._sync_filmstrip_files()
            if select:
                self._show_created_file(path)
            return True
        except Exception:
            import logging

            logging.getLogger(__name__).warning(
                "[GALLERY] could not register new file %s", path, exc_info=True
            )
            return False

    def _should_show_slideshow_button(self) -> bool:
        if getattr(self, "_burst_group_view_active", False):
            return False
        if not getattr(self, "image_files", None):
            return False
        if self._adjust_panel_active():
            return False
        if getattr(self, "view_mode", "single") == "single":
            return True
        return getattr(self, "view_mode", "") == "gallery"

    def _sync_slideshow_button_visibility(self) -> None:
        btn = getattr(self, "slideshow_bottom_button", None)
        if btn is None:
            return
        vis = self._should_show_slideshow_button()
        if btn.isVisible() != vis:
            btn.setVisible(vis)

    def _maybe_clear_empty_bookmark_filter(self) -> bool:
        """Leave bookmark-only view when the folder has no bookmarks left."""
        if not getattr(self, "_gallery_bookmark_filter_active", False):
            return False
        if getattr(self, "_gallery_bookmarked_paths", set()):
            return False
        self._gallery_bookmark_filter_active = False
        self._sync_bookmark_indicator()
        return True

    def _on_bookmark_slot_clicked(self) -> None:
        """Legacy bookmark chrome -- filter/toggle UI is retired in favor of ratings."""
        vm = getattr(self, "view_mode", "")
        if vm == "single":
            if hasattr(self, "status_bar") and self.status_bar is not None:
                self.status_bar.showMessage(
                    "Rate with 1–5 (0 clears). Bookmarks are no longer used for culling.",
                    3500,
                )
            return
        if vm != "gallery":
            return
        # Clear any leftover bookmark filter from older sessions; rating filter
        # is the supported way to narrow the gallery now.
        if getattr(self, "_gallery_bookmark_filter_active", False):
            self._clear_gallery_bookmark_filter()
            return
        if hasattr(self, "status_bar") and self.status_bar is not None:
            self.status_bar.showMessage(
                "Use the bottom stars to filter by rating (bookmarks retired).",
                3500,
            )

    def _remove_files_from_active_image_list(self, file_paths: list[str]) -> None:
        if not file_paths or not self.image_files:
            return
        remove_keys = {_norm_path(p) for p in file_paths if p}
        if not remove_keys:
            return
        self.image_files = [
            p for p in self.image_files if _norm_path(p) not in remove_keys
        ]
        bookmarked = getattr(self, "_gallery_bookmarked_paths", None)
        if bookmarked:
            bookmarked -= remove_keys
            self._gallery_bookmarked_paths = bookmarked
        selected = getattr(self, "_gallery_selected_paths", None)
        if selected:
            selected -= remove_keys
        if self.current_file_path and _norm_path(self.current_file_path) in remove_keys:
            if self.image_files:
                self.current_file_index = min(
                    self.current_file_index, len(self.image_files) - 1
                )
                if self.current_file_index < 0:
                    self.current_file_index = 0
                self.current_file_path = self.image_files[self.current_file_index]
            else:
                self.current_file_index = -1
                self.current_file_path = None

    def _surviving_anchor_after_removal(self, removed_paths) -> str | None:
        """Nearest file that will still exist after removing *removed_paths*.

        The scroll anchor passed to _sync_gallery_to_folder_files must be a
        surviving file: anchoring on a deleted path can never resolve, the
        scroll-to-file retries exhaust, and the gallery lands at the top.
        Prefers the first file after the earliest removed one (reading order),
        falling back to the nearest one before it.
        """
        removed = {os.path.normpath(p) for p in removed_paths if p}
        files = list(getattr(self, "image_files", None) or [])
        first = None
        for i, f in enumerate(files):
            if os.path.normpath(f) in removed:
                first = i
                break
        if first is None:
            return None
        for f in files[first:]:
            if os.path.normpath(f) not in removed:
                return f
        for f in reversed(files[:first]):
            if os.path.normpath(f) not in removed:
                return f
        return None

    def sort_image_files(self, file_paths, file_stats=None):
        """Sort files by capture time according to user preference (Newest/Oldest)"""
        newest_first = self.get_sort_preference()  # True = Newest first, False = Oldest first
        return self.sort_files_by_capture_time(file_paths, newest_first=newest_first, file_stats=file_stats)

    def _directory_looks_writable(self, directory: str) -> bool:
        """Best-effort write probe (POSIX permissions / read-only volumes).

        RAWviewer is not App-Sandboxed, so NSSavePanel does not grant extra TCC
        power -- export still needs a folder the process can create files in
        (SD cards and camera volumes are often mounted read-only).
        """
        directory = (directory or "").strip()
        if not directory or not os.path.isdir(directory):
            return False
        try:
            if not os.access(directory, os.W_OK):
                return False
        except OSError:
            return False
        probe = None
        try:
            import tempfile

            fd, probe = tempfile.mkstemp(prefix=".rawviewer_wprobe_", dir=directory)
            os.close(fd)
            return True
        except OSError:
            return False
        finally:
            if probe:
                with contextlib.suppress(OSError):
                    os.remove(probe)

    def _macos_nsopenpanel_reliable(self) -> bool:
        """False when NSOpenPanel/NSSavePanel construction is unsafe to try.

        A bare ``python3 main.py`` Terminal launch (``sys.frozen`` unset --
        no real .app bundle/Info.plist/bundle identifier) was observed via
        py-spy to sometimes hang *indefinitely* inside
        ``NSOpenPanel.alloc().init()`` on newer macOS instead of returning
        nil -- not consistently, so the try/except fallback to AppleScript
        cannot save us; a hang never raises. The built .app (``sys.frozen``
        True) has real bundle identity and does not hit this, so only skip
        NSPanel there when NOT frozen -- go straight to the AppleScript
        dialog, which is reliable everywhere.
        """
        return bool(getattr(sys, "frozen", False))

    def open_file(self):
        """Open an image file (and its containing folder).

        Matches GitHub HEAD (d122434): synchronous QFileDialog on the click stack.
        Deferred timers / processEvents() before the panel break macOS NSOpenPanel.
        """
        import logging

        logger = logging.getLogger(__name__)
        settings = self.get_settings()
        last_dir = settings.value("last_opened_dir", "")

        exts = self.get_supported_extensions()
        input_exts = " ".join([f"*{e}" for e in exts])
        filter_str = f"Images ({input_exts});;All Files (*)"

        try:
            if self._use_qt_file_dialog():
                file_path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Open File",
                    last_dir,
                    filter_str,
                    options=QFileDialog.Option.DontUseNativeDialog,
                )
            elif sys.platform == "darwin":
                file_path = self._open_file_dialog_macos_native(last_dir)
            else:
                self._prepare_for_modal_dialog()
                file_path, _ = QFileDialog.getOpenFileName(
                    self, "Open File", last_dir, filter_str
                )
        except Exception as e:
            logger.error("Open image dialog failed: %s", e, exc_info=True)
            self.show_error("Open Image", f"Could not open file chooser:\n{e}")
            return

        if os.environ.get("RAWVIEWER_DEBUG", "").strip() == "1":
            if self._use_qt_file_dialog():
                backend = "qt"
            elif sys.platform == "darwin":
                backend = "macos-native"
            else:
                backend = "qfiledialog"
            logger.info(
                "Open image dialog done (backend=%s, selected=%r)",
                backend,
                file_path or "",
            )

        if file_path:
            folder_path = os.path.dirname(file_path)
            base = os.path.basename(file_path)
            self.load_folder_images(
                folder_path, start_file=base, start_view="single"
            )
            settings.setValue("last_opened_dir", folder_path)

    def _no_image_loaded_help_text(self) -> str:
        """Empty-state instructions (single view / no folder loaded)."""
        return (
            "No image loaded\n\n"
            "Click 📁 or drag and drop a folder or image to load it\n"
            "← / → -- Previous / next image\n"
            "Space / Double-click -- Toggle fit-to-window / 100% zoom\n"
            "Click and drag to pan when zoomed\n"
            "Scroll wheel (fit-to-window): Scroll down = next image, Scroll up = previous image\n\n"
            "For all other shortcuts, click or hover the 'i' button in the bottom right."
        )

    def image_double_click_event(self, ev):
        try:
            if (
                hasattr(self, "current_file_path")
                and self.current_file_path
                and self.is_animated_image(_require_str_path(self.current_file_path))
            ):
                return
            if not self.current_pixmap:
                if hasattr(self, 'current_file_path') and self.current_file_path:
                    self._pending_zoom_toggle = True
                    self._pending_zoom_target_path = self.current_file_path
                    self._pending_zoom_pos = ev.pos()
                    import logging
                    logging.getLogger(__name__).info(
                        "Double-click recorded while image is loading; will zoom once ready."
                    )
                return
            if ev.button() == Qt.MouseButton.LeftButton:
                self._stop_slideshow()
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"[TRACK] User double-clicked to zoom - file: {os.path.basename(_require_str_path(self.current_file_path)) if hasattr(self, 'current_file_path') and self.current_file_path else 'Unknown'}")
                # Spacebar still jumps to EXIF focus box when outline is on; double-click
                # always zooms toward the clicked point (same as outline off).
                if (
                    getattr(self, "_focus_subject_outline_active", False)
                    and not self._single_view_is_fit_mode()
                ):
                    self.fit_to_window = True
                    self.current_zoom_level = 1.0
                    self.zoom_center_point = None
                    self.scale_image_to_fit()
                    self.image_label.setCursor(
                        QCursor(Qt.CursorShape.ArrowCursor)
                    )
                elif self._single_view_is_fit_mode():
                    # Zooming in from fit-to-window mode
                    self._recenter_on_focus_after_nav = False
                    click_pos = ev.pos()
                    displayed_pixmap = self.image_label.pixmap()

                    if displayed_pixmap:
                        # Calculate where the click occurred relative to the displayed image
                        label_size = self.image_label.size()
                        displayed_size = displayed_pixmap.size()

                        # Calculate image offset within the label (image is centered in label)
                        image_x_offset = (label_size.width() -
                                          displayed_size.width()) / 2
                        image_y_offset = (label_size.height() -
                                          displayed_size.height()) / 2

                        # Adjust click position relative to the displayed image
                        adjusted_click_x = click_pos.x() - image_x_offset
                        adjusted_click_y = click_pos.y() - image_y_offset

                        # Check if click is within the displayed image bounds
                        if (0 <= adjusted_click_x < displayed_size.width() and
                                0 <= adjusted_click_y < displayed_size.height()):

                            # Calculate the ratio of the click position within the displayed image
                            click_ratio_x = adjusted_click_x / displayed_size.width()
                            click_ratio_y = adjusted_click_y / displayed_size.height()

                            # Map this ratio to the full-size image coordinates
                            full_size = self.current_pixmap.size()
                            image_click_x = int(click_ratio_x * full_size.width())
                            image_click_y = int(click_ratio_y * full_size.height())

                            # Clamp to valid coordinates
                            image_click_x = max(
                                0, min(image_click_x, full_size.width() - 1))
                            image_click_y = max(
                                0, min(image_click_y, full_size.height() - 1))

                            self.zoom_center_point = QPoint(
                                image_click_x, image_click_y)
                        else:
                            # Click outside image, center on image center
                            self.zoom_center_point = QPoint(
                                self.current_pixmap.width() // 2,
                                self.current_pixmap.height() // 2)
                    else:
                        # No displayed pixmap, center on image center
                        self.zoom_center_point = QPoint(
                            self.current_pixmap.width() // 2,
                            self.current_pixmap.height() // 2)

                    self._zoom_in_to_image_point_finish()
                else:
                    # Zooming out to fit-to-window mode
                    self.fit_to_window = True
                    self.current_zoom_level = 1.0
                    self.zoom_center_point = None
                    self._clear_all_zoom_state()
                    self.scale_image_to_fit()
                    self.image_label.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

            self.update_status_bar()
            self.setFocus()
            ev.accept()
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Error in image_double_click_event: {e}", exc_info=True)

    def _schedule_interim_quality_upgrade(self, file_path: str) -> None:
        """After showing a low-res interim at zoomed framing, get sharp pixels.

        Deferred a tick so the current display pass settles first. Displays
        the cached sensor-res buffer when available (the interim upgrade path
        in _apply_gpu_view re-places it at the same framing), otherwise queues
        a sensor decode.
        """

        def _upgrade() -> None:
            if _norm_path(file_path) != _norm_path(
                getattr(self, "current_file_path", "") or ""
            ):
                return
            if not getattr(self, "_nav_zoom_interim_active", False):
                return
            try:
                cached_full = self.image_cache.get_full_image(file_path)
                if cached_full is not None and hasattr(cached_full, "shape"):
                    ch, cw = cached_full.shape[0], cached_full.shape[1]
                    exif = self.image_cache.get_exif(file_path)
                    if image_covers_sensor_resolution(cw, ch, exif):
                        self._orientation_already_applied = True
                        try:
                            self.display_numpy_image(cached_full)
                        finally:
                            self._orientation_already_applied = False
                        return
            except Exception:
                pass
            with contextlib.suppress(Exception):
                self._request_full_res_decode(file_path, priority_current=True, force=True)

        QTimer.singleShot(0, _upgrade)

    def _cached_full_covers_sensor(self, file_path: str, cached_full=None) -> bool:
        """True when a cached full-image buffer reaches sensor resolution.

        In RAW workflow, an embedded-JPEG stand-in never counts as covering
        (callers must still be able to queue a true LibRaw decode). In
        non-RAW workflow the same buffer *is* the final display tier.
        """
        if not file_path:
            return False
        try:
            raw_workflow = False
            try:
                from common_image_loader import use_libraw_consistent_preview_first

                raw_workflow = bool(use_libraw_consistent_preview_first(file_path))
            except Exception:
                raw_workflow = False
            if (
                cached_full is None
                and raw_workflow
                and self.image_cache.full_image_is_embedded_jpeg(file_path)
            ):
                return False
            buf = cached_full if cached_full is not None else self.image_cache.get_full_image(file_path)
            if buf is None:
                return False
            if hasattr(buf, "shape"):
                h, w = int(buf.shape[0]), int(buf.shape[1])
            elif hasattr(buf, "width") and hasattr(buf, "height"):
                w, h = int(buf.width()), int(buf.height())
            else:
                return False
            exif = self.image_cache.get_exif(file_path)
            return image_covers_sensor_resolution(w, h, exif)
        except Exception:
            return False

    def _is_raw_file(self, file_path):
        """Check if file is RAW format"""
        raw_extensions = {'.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2',
                         '.pef', '.srw', '.x3f', '.raf', '.3fr', '.fff', '.iiq',
                         '.cap', '.erf', '.mef', '.mos', '.nrw', '.rwl', '.srf'}
        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext in raw_extensions

    def _should_skip_redundant_load(self, requested_file_path, force_reload):
        """Check if we can skip redundant load (already on screen)"""
        if force_reload:
            return False
        if getattr(self, "view_mode", "single") != "single":
            return False
        if getattr(self, "_navigation_in_progress", False):
            return False
        if getattr(self, "_loading_from_gallery", False):
            return False
        if getattr(self, "_loading_from_filmstrip", False):
            return False
        if getattr(self, "_preserve_nav_zoom_active", False):
            return False
        if _norm_path(requested_file_path) != _norm_path(getattr(self, "current_file_path", "") or ""):
            return False
        if not getattr(self, "_single_view_first_render_logged", False):
            return False
        return self._single_view_pixels_on_screen(requested_file_path)

    def on_image_processed_enhanced(self, rgb_image):
        """Handle enhanced image processing results."""
        try:
            if rgb_image is None:
                # Non-RAW file - load with QPixmap and cache it
                pixmap = self._load_pixmap_safe(self.current_file_path)
                if pixmap.isNull():
                    self.show_error("Display Error",
                                    "Could not load image file.")
                    return

                # _load_pixmap_safe now uses QImageReader with setAutoTransform(True) for regular images
                # which automatically applies EXIF orientation. Orientation is already applied.
                # Set flag so display_pixmap doesn't apply it again
                self._orientation_already_applied = True

                # Cache the pixmap (already has correct orientation)
                self.image_cache.put_pixmap(self.current_file_path, pixmap)

                self.display_pixmap(pixmap)
            else:
                # RAW file - processed numpy array
                self.display_numpy_image(rgb_image)

            # Update UI state
            if self.current_file_path:
                self.scan_folder_for_images(self.current_file_path)

            # Update status bar with EXIF data instead of just showing "Loaded"
            self.update_status_bar()

            # Start preloading adjacent images
            self._start_preloading()

            # Clear any pending thumbnail since we now have the full image
            self._pending_thumbnail = None

        except Exception as e:
            error_msg = f"Error displaying image: {str(e)}"
            self.show_error("Display Error", error_msg)

        self.setFocus()

    def _start_memory_pressure_monitor(self) -> None:
        """Idle QTimer (every few seconds). Not on thumbnail paint."""
        try:
            from core.memory_pressure import backoff_enabled

            if not backoff_enabled():
                return
        except Exception:
            return
        timer = getattr(self, "_memory_pressure_timer", None)
        if timer is not None and not timer.isActive():
            timer.start()

    def _on_memory_pressure_tick(self) -> None:
        try:
            from core.memory_pressure import poll_and_apply

            poll_and_apply(viewer=self)
        except Exception:
            logging.getLogger(__name__).debug("memory-pressure poll failed", exc_info=True)

    def _suppress_jpeg_interim(self, file_path: str | None = None) -> bool:
        """True when embedded-JPEG interim tiers must not be painted (RAW workflow).

        The user-facing contract of the RAW (High Quality) workflow is "what
        you see is the LibRaw render" -- camera-JPEG tiers have different tone
        and flash on replacement. Suppressed only at fit view (the zoomed-nav
        interim keeps its own equivalent-framing tier) and only for files
        LibRaw can actually decode (unsupported files keep their embedded
        fallback, which is all they have).

        Exception: gallery → single always allows embedded/gallery tiers so the
        first paint does not wait on demosaic (RAW HQ / Adjust must not starve
        that transition). ``_gallery_entry_accept_embedded`` outlives the brief
        ``_loading_from_gallery`` flag cleared mid-load.
        """
        if getattr(self, "_loading_from_gallery", False) or getattr(
            self, "_gallery_entry_accept_embedded", False
        ):
            return False
        path = file_path or getattr(self, "current_file_path", None)
        if not path:
            return False

        # For non-RAW files, full decode is extremely fast. We suppress interim
        # blurry preview/thumbnail tiers to prevent a low-res flash.
        if not is_raw_file(path):
            return True
        if str(os.environ.get("RAWVIEWER_RAW_NO_JPEG_INTERIM", "1")).strip().lower() in (
            "0",
            "false",
            "no",
            "off",
        ):
            return False  # escape hatch: restore JPEG interim tiers in RAW mode
        from common_image_loader import use_libraw_consistent_preview_first

        if not use_libraw_consistent_preview_first(path):
            return False
        try:
            from processing.unified import _LIBRAW_UNSUPPORTED_PATHS

            if os.path.normcase(os.path.abspath(path)) in _LIBRAW_UNSUPPORTED_PATHS:
                return False
        except Exception:
            pass
        return True

    def _raw_edr_eligible_ignoring_idle(self, file_path: str | None = None) -> bool:
        """Settings/workflow gate for RAW EDR display, without the nav-idle debounce.

        Split out of :meth:`_should_use_raw_edr_display` so the idle-arming
        logic in ``display_numpy_image`` can ask "would EDR apply here at
        all" before deciding whether to pay for the decode immediately or
        defer it (see ``_maybe_arm_raw_edr_after_idle``).
        """
        path = file_path or getattr(self, "current_file_path", None)
        if not path or not is_raw_file(path):
            return False
        if getattr(self, "_raw_recovery_skip_display_cache", False):
            return False
        if getattr(self, "_raw_recovery_active", False):
            return False
        if getattr(self, "_raw_edr_fallback_active", False):
            return False
        from common_image_loader import use_raw_edr_for_display

        return use_raw_edr_for_display()
