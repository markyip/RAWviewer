
"""metadata mixin, part 1 of 2 (29 methods, 874 lines)."""
from __future__ import annotations

import datetime
import os
import sys

from PyQt6.QtCore import QPoint, QRect, Qt, QTimer
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap

from core.config import _env_int, _norm_path

from .. import lazy
from ..lazy import process_file_from_path
from ..paths import _require_str_path

lazy.register(sys.modules[__name__])


class MetadataPart1:
    def _draw_focus_subject_outline_on_base_painter(
        self, painter: QPainter, base_pm: QPixmap, rect_image_space: QRect
    ) -> None:
        """Dashed outline: MakerNote AF (amber) or EXIF Subject area (lime)."""
        if base_pm.isNull() or rect_image_space.isNull():
            return
        ow = self.current_pixmap.width() if self.current_pixmap else 0
        oh = self.current_pixmap.height() if self.current_pixmap else 0
        if ow <= 0 or oh <= 0:
            return
        sx = base_pm.width() / float(ow)
        sy = base_pm.height() / float(oh)
        rx = int(rect_image_space.left() * sx)
        ry = int(rect_image_space.top() * sy)
        rw = max(1, int(round(rect_image_space.width() * sx)))
        rh = max(1, int(round(rect_image_space.height() * sy)))
        bw, bh = base_pm.width(), base_pm.height()
        min_draw = max(12, min(bw, bh) // 50)
        if rw < min_draw or rh < min_draw:
            cx = rx + rw / 2.0
            cy = ry + rh / 2.0
            rw = max(rw, min_draw)
            rh = max(rh, min_draw)
            rx = int(round(cx - rw / 2.0))
            ry = int(round(cy - rh / 2.0))
            rx = max(0, min(rx, bw - 1))
            ry = max(0, min(ry, bh - 1))
            rw = max(1, min(rw, bw - rx))
            rh = max(1, min(rh, bh - ry))
        src = getattr(self, "_focus_rect_source", None) or ""
        col = QColor(255, 185, 45, 255) if src == "makernote_af" else QColor(165, 255, 95, 255)
        pen = QPen(col)
        pen.setWidth(max(2, min(base_pm.width(), base_pm.height()) // 380))
        pen.setCosmetic(True)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRect(rx, ry, rw, rh))

    def _focus_jump_to_subject_center(self) -> bool:
        """With focus outline on: center zoom on EXIF / maker AF rectangle (fit-to-window)."""
        if self._raw_recovery_preview_active():
            self._notify_raw_recovery_zoom_blocked()
            return True
        if not self.current_pixmap or self.current_pixmap.isNull():
            return False
        rect = getattr(self, "_focus_subject_rect_image", None)
        if rect is None or rect.isNull() or rect.width() < 1 or rect.height() < 1:
            # Fallback to standard zoom if no EXIF box
            return False
        cx = rect.left() + rect.width() // 2
        cy = rect.top() + rect.height() // 2
        self._stop_slideshow()
        self._recenter_on_focus_after_nav = False
        self.zoom_center_point = QPoint(cx, cy)
        self._pending_zoom_focus_subject = True
        self._focus_zoom_anchor_active = True
        gv = getattr(self, "gpu_view", None)
        if gv is not None and gv.has_pixmap():
            if gv.is_fit_mode():
                self._zoom_in_to_image_point_finish()
            else:
                gv.center_on_image_point(float(cx), float(cy))
            self.update_status_bar()
            self.setFocus()
            return True
        if self.fit_to_window:
            self._zoom_in_to_image_point_finish()
        else:
            self.apply_zoom_and_pan()
        self.update_status_bar()
        self.setFocus()
        return True

    def _toggle_focus_subject_outline(self) -> bool:
        """Return True if handled."""
        if getattr(self, "view_mode", "") == "compare":
            self._compare_focus_outline_active = not getattr(
                self, "_compare_focus_outline_active", False
            )
            if self._compare_focus_outline_active:
                self._compare_update_focus_overlays()
                self.status_bar.showMessage(
                    "Focus outline ON -- amber dashed = maker AF; lime = Subject / CIPA. "
                    "From fit: Space centers each pane on its focus box (frame center if none). F = off.",
                    6500,
                )
            else:
                self._compare_clear_focus_overlays()
                self.status_bar.showMessage("Focus outline off", 2000)
            return True
        if getattr(self, "view_mode", "single") != "single":
            return False

        self._focus_subject_outline_active = not getattr(
            self, "_focus_subject_outline_active", False
        )
        if self._focus_subject_outline_active:
            self._refresh_focus_subject_rect_from_exif()
            if getattr(self, "_focus_subject_rect_image", None):
                self.status_bar.showMessage(
                    "Focus outline ON -- amber dashed = maker AF; lime = Subject / CIPA. "
                    "From fit: Space centers on the box; double-click zooms to the click. F = off.",
                    6500,
                )
            else:
                # Always acknowledge the toggle; silent ON with no box looks like
                # F is broken (common when maker AF/SubjectArea is absent, or when
                # pyexiv2 failed to load in a packaged build).
                pyexiv2_ok = False
                try:
                    from metadata.backend import has_pyexiv2

                    pyexiv2_ok = bool(has_pyexiv2())
                except Exception:
                    pyexiv2_ok = False
                if pyexiv2_ok:
                    self.status_bar.showMessage(
                        "Focus outline ON -- no AF / SubjectArea box in this file’s metadata. "
                        "F = off.",
                        6500,
                    )
                else:
                    self.status_bar.showMessage(
                        "Focus outline ON -- EXIF engine (pyexiv2) unavailable; "
                        "maker AF points cannot be read. F = off.",
                        8000,
                    )
            self._redraw_single_view_pixmap_without_relayout()
        else:
            self._focus_subject_rect_image = None
            self._focus_rect_source = None
            self._redraw_single_view_pixmap_without_relayout()
            self.status_bar.showMessage("Focus outline off", 2000)
        self.update_status_bar()
        return True

    def _edr_metadata_label(self) -> str | None:
        if self._adjust_panel_active():
            return "RAW edit"
        from common_image_loader import format_edr_status_label

        return format_edr_status_label(
            display_mode=getattr(self, "_display_edr_mode", None),
            gpu_view=getattr(self, "gpu_view", None),
        )

    def _refresh_focus_subject_rect_from_exif(self) -> None:
        """Populate _focus_subject_rect_image from pyexiv2, then exifread Subject*, then Canon AF.

        The EXIF hint is derived in the image's *upright* (EXIF-oriented) space, so it is
        computed against the un-rotated base pixmap dimensions and then rotated by the
        user's non-destructive visual rotation so the box tracks the displayed image.
        """
        self._focus_subject_rect_image = None
        self._focus_rect_source = None
        path = getattr(self, "current_file_path", None)
        pm = getattr(self, "current_pixmap", None)
        if not path or pm is None or pm.isNull():
            return

        try:
            from metadata.subject_area import _rotate_rect, pixmap_ltwh_focus_hint

            orientation = 1
            cached_exif = self.image_cache.get_exif(path)
            if cached_exif:
                orientation = cached_exif.get("orientation", 1)

            # Visual rotation the user applied (clockwise degrees, non-destructive).
            degrees = self._get_visual_rotation_degrees()

            # Hint must be computed against the base (un-visually-rotated) pixmap size.
            base_pm = getattr(self, "_base_display_pixmap", None)
            if base_pm is not None and not base_pm.isNull():
                base_w, base_h = base_pm.width(), base_pm.height()
            elif degrees in (90, 270):
                # current_pixmap dims are swapped relative to the base when rotated 90/270.
                base_w, base_h = pm.height(), pm.width()
            else:
                base_w, base_h = pm.width(), pm.height()

            hint = pixmap_ltwh_focus_hint(path, base_w, base_h, orientation)
            if hint is not None:
                ltwh, src = hint
                # Map from base space into the displayed (visually rotated) pixmap space.
                if degrees:
                    ocode = {90: 6, 180: 3, 270: 8}.get(degrees, 1)
                    ltwh = _rotate_rect(ltwh, base_w, base_h, ocode)
                self._focus_subject_rect_image = QRect(
                    ltwh[0], ltwh[1], ltwh[2], ltwh[3]
                )
                self._focus_rect_source = (
                    "exif_subject" if src == "exif_subject" else "makernote_af"
                )
                return
        except ImportError:
            pass
        except Exception:
            pass

    def _sync_focus_subject_outline_after_display(self) -> None:
        if (
            not getattr(self, "_focus_subject_outline_active", False)
            or getattr(self, "view_mode", "single") != "single"
        ):
            return
        self._refresh_focus_subject_rect_from_exif()

        # Re-anchor to the EXIF focus box only after arrow-key navigation while zoomed.
        # Space and double-click set zoom_center_point explicitly -- do not override them here.
        if (
            not self._single_view_is_fit_mode()
            and getattr(self, "_recenter_on_focus_after_nav", False)
        ):
            self._recenter_on_focus_after_nav = False
            rect = getattr(self, "_focus_subject_rect_image", None)
            if rect is not None and not rect.isNull() and rect.width() >= 1 and rect.height() >= 1:
                cx = rect.left() + rect.width() // 2
                cy = rect.top() + rect.height() // 2
                self.zoom_center_point = QPoint(cx, cy)
                gv = getattr(self, "gpu_view", None)
                if gv is not None and gv.has_pixmap():
                    gv.center_on_image_point(float(cx), float(cy))
                else:
                    # Clear restored scroll positions to force re-centering on the new focus point
                    self._restore_start_scroll_x = None
                    self._restore_start_scroll_y = None

        self._redraw_single_view_pixmap_without_relayout()

    def _maybe_refresh_focus_subject_outline_after_display(self) -> None:
        if (
            getattr(self, "_focus_subject_outline_active", False)
            and getattr(self, "view_mode", "single") == "single"
        ):
            QTimer.singleShot(0, self._sync_focus_subject_outline_after_display)

    def _filter_paths_to_min_rating(self, files: list[str], min_rating: int) -> list[str]:
        if min_rating <= 0:
            return files
        return [p for p in files if p and self._gallery_path_rating(p) >= min_rating]

    def _capture_times_for_paths(self, paths: list[str]) -> dict:
        capture_times = {}
        try:
            capture_times = self.image_cache.get_capture_times_for_sort(paths)
        except Exception:
            capture_times = {}
        return capture_times or {}

    def _compare_exif_for_path(self, file_path: str) -> dict:
        exif = {}
        try:
            if hasattr(self, "image_cache") and self.image_cache is not None:
                exif = dict(self.image_cache.get_exif(file_path) or {})
        except Exception:
            exif = {}
        if not exif.get("orientation"):
            orientation = self.get_orientation_from_exif(file_path)
            if orientation != 1:
                exif["orientation"] = orientation
        return exif

    def _ensure_compare_exif_ready(self, file_path: str) -> None:
        if not file_path:
            return
        self._compare_exif_for_path(file_path)

    def _compare_pixmap_matches_exif(self, file_path: str, pm: QPixmap) -> bool:
        from common_image_loader import pixmap_matches_exif_display

        if pm is None or pm.isNull() or not file_path:
            return True
        exif = self._compare_exif_for_path(file_path)
        orientation = int(exif.get("orientation", 1) or 1)
        if orientation == 1:
            return True
        ow = int(exif.get("original_width") or 0)
        oh = int(exif.get("original_height") or 0)
        if ow <= 0 or oh <= 0:
            w, h = pm.width(), pm.height()
            if orientation in (6, 8):
                return h > w
            return True
        return pixmap_matches_exif_display(
            pm.width(), pm.height(), ow, oh, orientation
        )

    def on_manager_exif_ready(self, file_path: str, exif_data: dict):
        """處理 ImageLoadManager 的 EXIF 數據就緒信號"""
        import logging
        import time
        logger = logging.getLogger(__name__)

        if getattr(self, "_suppress_single_manager_callbacks", False):
            return
        if getattr(self, "view_mode", "single") != "single":
            return

        # Only handle current file's EXIF (normalize for Windows path format differences)
        if _norm_path(file_path) != _norm_path(getattr(self, "current_file_path", None)):
            return

        # De-bounce duplicate EXIF-ready signals for the same current file.
        # During rapid folder/view transitions multiple paths may request EXIF for the same file;
        # repeated immediate status rewrites can make folder switch appear "stuck".
        now = time.time()
        last_path = getattr(self, "_last_manager_exif_path", None)
        last_ts = float(getattr(self, "_last_manager_exif_ts", 0.0) or 0.0)
        is_minimal = isinstance(exif_data, dict) and bool(
            exif_data.get("minimal_preview_exif")
        )
        if (
            _norm_path(last_path) == _norm_path(file_path)
            and (now - last_ts) < 0.35
        ):
            if is_minimal and getattr(self, "_last_manager_exif_was_minimal", False):
                logger.debug(
                    "[MANAGER] Skipping duplicate minimal EXIF for %s",
                    os.path.basename(file_path),
                )
                return
            if not is_minimal and not getattr(
                self, "_last_manager_exif_was_minimal", False
            ):
                logger.debug(
                    f"[MANAGER] Skipping duplicate EXIF-ready burst for "
                    f"{os.path.basename(file_path)}"
                )
                return
        self._last_manager_exif_path = file_path
        self._last_manager_exif_ts = now
        self._last_manager_exif_was_minimal = is_minimal

        if is_minimal:
            logger.info(
                "[MANAGER] Minimal EXIF ready for %s (full tags deferred)",
                os.path.basename(file_path),
            )
        else:
            logger.info(f"[MANAGER] EXIF data ready for {os.path.basename(file_path)}")

        # Debug: Check exif_data structure
        if isinstance(exif_data, dict) and not is_minimal:
            has_exif_data_key = 'exif_data' in exif_data
            exif_tags_count = 0
            exif_tags_sample = []
            if has_exif_data_key and isinstance(exif_data.get('exif_data'), dict):
                exif_tags_dict = exif_data['exif_data']
                exif_tags_count = len(exif_tags_dict)
                # Log sample of EXIF tags to see what we have
                exif_tags_sample = list(exif_tags_dict.keys())[:10]
                logger.info(f"[MANAGER] EXIF data structure - has 'exif_data' key: {has_exif_data_key}, "
                           f"exif_tags_count: {exif_tags_count}, sample tags: {exif_tags_sample}")
            elif any(key.startswith('EXIF ') or key.startswith('Image ') for key in exif_data):
                exif_tags_count = len([k for k in exif_data if k.startswith('EXIF ') or k.startswith('Image ')])
                logger.info(f"[MANAGER] EXIF data structure - has 'exif_data' key: {has_exif_data_key}, "
                           f"exif_tags_count: {exif_tags_count}, top-level keys: {list(exif_data.keys())[:10]}")
            else:
                logger.info(f"[MANAGER] EXIF data structure - has 'exif_data' key: {has_exif_data_key}, "
                           f"exif_tags_count: {exif_tags_count}, top-level keys: {list(exif_data.keys())[:10]}")
        elif not isinstance(exif_data, dict):
            logger.warning(f"[MANAGER] EXIF data is not a dict, type: {type(exif_data)}")

        self._exif_data_ready = True
        if exif_data:
            self.image_cache.put_exif(
                file_path, exif_data, persist_disk=not is_minimal
            )
            logger.debug(
                "[MANAGER] Stored %s EXIF in cache for %s",
                "minimal" if is_minimal else "full",
                os.path.basename(file_path),
            )

        if getattr(self, "view_mode", "single") != "single":
            return

        if is_minimal:
            if self._single_view_pixels_on_screen(file_path):
                cached = self.image_cache.get_exif(file_path) or exif_data
                self._apply_preview_first_status_bar(cached)
            return

        tag_n = 0
        if isinstance(exif_data, dict):
            inner = exif_data.get("exif_data")
            if isinstance(inner, dict):
                tag_n = len(inner)
        logger.info(
            "[MANAGER] %s EXIF ready, updating status bar for %s",
            "Full" if tag_n >= 20 else "Cached",
            os.path.basename(file_path),
        )
        self.update_status_bar()
        if not self._single_view_pixels_on_screen(file_path):
            self._try_paint_display_quality_cache(file_path)

    def get_orientation_from_exif(self, file_path):
        """Extract orientation via EXIFExtractor (RAW fallbacks included)."""
        try:
            from enhanced_raw_processor import EXIFExtractor

            exif = EXIFExtractor().extract_exif_data(file_path)
            if exif:
                orientation = int(exif.get("orientation", 1) or 1)
                if 1 <= orientation <= 8:
                    return orientation
        except Exception:
            pass
        return 1

    def is_camera_pre_rotated(self):
        """Check if this camera stores image data pre-rotated for non-RAW files"""
        # CRITICAL: Only skip orientation correction for RAW files, not JPEG files
        # JPEG files always need orientation correction based on EXIF orientation tag
        # RAW files from certain cameras may be pre-rotated, but JPEG files are not
        try:
            # Check if this is a RAW file
            raw_extensions = {'.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2',
                             '.pef', '.srw', '.x3f', '.raf', '.3fr', '.fff', '.iiq',
                             '.cap', '.erf', '.mef', '.mos', '.nrw', '.rwl', '.srf'}
            file_ext = os.path.splitext(_require_str_path(self.current_file_path))[1].lower()

            # For JPEG and other non-RAW files, always apply orientation correction
            if file_ext not in raw_extensions:
                return False

            # For RAW files, check camera make
            tags = process_file_from_path(self.current_file_path, details=False)
            make = tags.get("Image Make")

            if make:
                make_str = str(make).upper()
                # Sony cameras often store RAW data pre-rotated (but not JPEG)
                if "SONY" in make_str:
                    return True

                # Leica cameras also store RAW data pre-rotated (but not JPEG)
                if "LEICA" in make_str:
                    return True

                # Hasselblad cameras also store RAW data pre-rotated (but not JPEG)
                if "HASSELBLAD" in make_str:
                    return True

        except Exception:
            pass

        return False

    def _finish_silent_metadata_without_worker(self, corpus_files=None, *, skipped: bool = True) -> None:
        """Complete silent metadata pass without further index work.

        When *skipped* is True, the DB was already current and no worker ran.
        When False, a metadata-only worker just finished normally.
        """
        import logging

        logger = logging.getLogger(__name__)
        folder = getattr(self, "current_folder", None)
        if folder:
            self._set_semantic_index_incomplete(folder, False)
        self._invalidate_semantic_coverage_cache()
        corpus = corpus_files or self._gallery_search_index_corpus()
        wants_semantic = bool(getattr(self, "_gallery_search_user_wants_semantic", False))
        deferred_semantic = bool(getattr(self, "_defer_semantic_after_metadata", False))
        if skipped:
            logger.info(
                "[INDEX][CHAIN] silent_metadata skipped (DB current): wants_semantic=%s deferred_semantic=%s corpus=%d",
                wants_semantic,
                deferred_semantic,
                len(corpus or []),
            )
        else:
            logger.info(
                "[INDEX][CHAIN] silent_metadata done: wants_semantic=%s deferred_semantic=%s corpus=%d",
                wants_semantic,
                deferred_semantic,
                len(corpus or []),
            )
        if wants_semantic or deferred_semantic:
            logger.info(
                "[INDEX][CHAIN] Triggering deferred semantic start after silent metadata."
            )
            self._gallery_search_index_session_key = None
            self._invalidate_semantic_coverage_cache()
            self._gallery_search_user_wants_semantic = True
            self._gallery_search_show_index_progress = True
            self._start_user_semantic_indexing(corpus)

    def _on_gallery_metadata_ready(self, meta, folder_at_request):
        """Thread-safe handler for metadata fetch completion"""
        try:
            # Clear active fetcher references now that it's done
            self._active_metadata_fetcher = None
            self._active_metadata_signals = None

            # Ignore if folder changed during fetch
            if getattr(self, "current_folder", None) != folder_at_request:
                return
            self._gallery_bulk_metadata = meta
            self._gallery_metadata_fetch_in_progress = False
            self._bump_gallery_rating_presence_epoch()
            self._rebuild_burst_groups()
            if getattr(self, "view_mode", "") == "gallery":
                self._sync_gallery_rating_filter_stars()

            if hasattr(self, 'gallery_justified') and self.gallery_justified and self.image_files:
                try:
                    import logging
                    logger = logging.getLogger(__name__)
                    current_file = getattr(self, "current_file_path", None)

                    # Only re-sort if the EXIF refinement hasn't already applied a correct order.
                    # The refinement uses the semantic index (6889 entries) and produces authentic
                    # capture-time order.  This fetch used probe_uncached=False (cache only, ~3 entries)
                    # so its sort would degrade to mtime order and overwrite the correct EXIF order.
                    current_gen = getattr(self, "_folder_load_generation", None)
                    refinement_applied = (
                        getattr(self, "_folder_sort_refinement_applied_token", None) == current_gen
                    )
                    refinement_pending = (
                        getattr(self, "_folder_sort_refinement_token", None) == current_gen
                        and not refinement_applied
                    )
                    if not refinement_applied and not refinement_pending:
                        newest_first = self.get_sort_preference()
                        file_stats = getattr(self, "_quick_folder_file_stats", None)
                        sorted_files, merged_meta = self.sort_files_by_capture_time(
                            self.image_files,
                            newest_first=newest_first,
                            file_stats=file_stats,
                            probe_uncached=False,
                            folder_path=folder_at_request,
                        )
                        if merged_meta:
                            meta.update(merged_meta)
                        if sorted_files != self.image_files:
                            self.image_files = sorted_files
                            if current_file in self.image_files:
                                self.current_file_index = self.image_files.index(current_file)
                            self.update_status_bar()
                    else:
                        logger.info(
                            "[GALLERY] Background metadata ready: deferring cache-only re-sort "
                            "(refinement applied=%s pending=%s)",
                            refinement_applied,
                            refinement_pending,
                        )

                    self._sync_gallery_to_folder_files(
                        meta,
                        scroll_to_current=getattr(self, "view_mode", "") != "gallery",
                        force=not refinement_pending,
                    )
                    logger.info(f"[GALLERY] Background metadata ready: {len(meta)} items, gallery refreshed")
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"[GALLERY] Error refreshing gallery with metadata: {e}")
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"[GALLERY] Critical error in _on_gallery_metadata_ready: {e}")
        finally:
            self._gallery_metadata_fetch_in_progress = False

    def _maybe_clear_empty_rating_filter(self) -> bool:
        """Reset the rating filter when no images meet the threshold anymore."""
        min_rating = int(getattr(self, "_gallery_rating_filter_min", 0) or 0)
        if min_rating <= 0:
            return False
        files = list(getattr(self, "image_files", []) or [])
        if any(self._gallery_path_rating(p) >= min_rating for p in files):
            return False
        # Memory-only ratings: while bulk metadata is still loading (or empty),
        # do not clear -- SQLite-only ratings land with the folder meta fetch.
        if getattr(self, "_gallery_metadata_fetch_in_progress", False):
            return False
        if not (getattr(self, "_gallery_bulk_metadata", None) or {}):
            return False
        self._gallery_rating_filter_min = 0
        self._sync_gallery_rating_filter_stars()
        return True

    def _clear_gallery_rating_filter(self) -> bool:
        """Clear the gallery rating filter; keep active search/bookmark filters if any."""
        if getattr(self, "view_mode", "") != "gallery":
            return False
        if int(getattr(self, "_gallery_rating_filter_min", 0) or 0) <= 0:
            return False
        self._gallery_rating_filter_min = 0
        self._sync_gallery_rating_filter_stars()
        self._apply_gallery_bookmark_filter()
        return True

    def _on_gallery_rating_star_clicked(self, n: int) -> None:
        if getattr(self, "view_mode", "") != "gallery":
            return
        current = int(getattr(self, "_gallery_rating_filter_min", 0) or 0)
        self._gallery_rating_filter_min = 0 if current == n else n
        self._sync_gallery_rating_filter_stars()
        self._apply_gallery_bookmark_filter()

    def _sync_gallery_rating_filter_stars(self) -> None:
        widget = getattr(self, "gallery_rating_filter_stars", None)
        if widget is None:
            return
        in_gallery = getattr(self, "view_mode", "") == "gallery"
        files = list(getattr(self, "image_files", None) or [])
        min_rating = int(getattr(self, "_gallery_rating_filter_min", 0) or 0)
        # Hide the star row entirely when nothing in the current gallery has
        # ever been rated -- clicking a star then would just narrow to zero
        # results (or immediately self-clear via _maybe_clear_empty_rating_
        # filter), a non-functional control with no visible feedback. Keep it
        # visible while a filter is already active (min_rating > 0) even if
        # this particular sync call races ahead of _maybe_clear_empty_rating_
        # filter -- that function is what actually resets the filter once it
        # observes zero qualifying images, and the star row must stay up
        # until then so the active-filter state stays visible/clearable.
        if not in_gallery or not files:
            widget.hide()
            return
        # Gallery-only: still O(n) over paths, but ratings are memory-only
        # (bulk meta / RAM EXIF) -- never per-file get_exif() disk/SQLite.
        any_rated = min_rating > 0 or self._gallery_folder_has_any_rating(files)
        if not any_rated:
            widget.hide()
            return
        widget.show()
        buttons = getattr(self, "_gallery_rating_filter_buttons", None) or []
        filled = getattr(self, "_rating_star_filled_icon", None)
        empty = getattr(self, "_rating_star_empty_icon", None)
        for i, btn in enumerate(buttons, start=1):
            btn.setIcon((filled if i <= min_rating else empty) or QIcon())
            btn.setToolTip(
                f"Show {i}★ and above (click again to clear)"
                if i != min_rating
                else "Click to clear rating filter"
            )

    def _sync_single_view_rating_stars(self) -> None:
        widget = getattr(self, "single_view_rating_stars", None)
        if widget is None:
            return
        in_single = getattr(self, "view_mode", "single") == "single"
        has_files = bool(getattr(self, "image_files", None))
        if not in_single or not has_files:
            widget.hide()
            return
        widget.show()
        path = getattr(self, "current_file_path", None)
        rating = self._gallery_path_rating(path) if path else 0
        self._set_rating_stars_display(rating)

    def _set_rating_stars_display(self, rating: int) -> None:
        buttons = getattr(self, "_rating_star_buttons", None)
        if not buttons:
            return
        r = max(0, min(5, int(rating or 0)))
        filled = getattr(self, "_rating_star_filled_icon", None)
        empty = getattr(self, "_rating_star_empty_icon", None)
        for i, btn in enumerate(buttons, start=1):
            btn.setIcon((filled if i <= r else empty) or QIcon())

    def _on_rating_star_clicked(self, n: int) -> None:
        path = getattr(self, "current_file_path", None)
        if not path:
            return
        current = self._gallery_path_rating(path)
        self.rate_current_image(0 if current == n else n)

    def _bump_gallery_rating_presence_epoch(self) -> None:
        """Invalidate cached 'any file rated?' answers for the star-filter row."""
        self._gallery_rating_presence_epoch = (
            int(getattr(self, "_gallery_rating_presence_epoch", 0) or 0) + 1
        )
        self._gallery_any_rated_cache = None

    def _gallery_path_rating(self, file_path: str) -> int:
        """0-5 star rating for gallery/filmstrip badges and rating filters.

        Memory-only: bulk metadata (folder scan / rate_current_image) then the
        in-RAM EXIF map. Never calls ``get_exif()`` -- that path stats + hits
        SQLite per file, and walking thousands of paths from status-bar /
        filter sync on a USB gallery froze the UI for 10s+. Ratings that only
        live in SQLite appear once bulk metadata (or a rate action) lands.
        """
        if not file_path:
            return 0
        meta = getattr(self, "_gallery_bulk_metadata", None) or {}
        row = meta.get(file_path)
        if row and row.get("rating"):
            try:
                return max(0, min(5, int(row["rating"])))
            except (TypeError, ValueError):
                pass
        try:
            cache = getattr(self, "image_cache", None)
            exif = (
                cache.get_exif_memory_only(file_path)
                if cache is not None and hasattr(cache, "get_exif_memory_only")
                else None
            )
        except Exception:
            exif = None
        if exif and exif.get("rating"):
            try:
                return max(0, min(5, int(exif["rating"])))
            except (TypeError, ValueError):
                pass
        return 0

    def _gallery_folder_has_any_rating(self, files) -> bool:
        """True if any path in ``files`` has a known in-memory rating > 0."""
        files = list(files or [])
        epoch = int(getattr(self, "_gallery_rating_presence_epoch", 0) or 0)
        bulk = getattr(self, "_gallery_bulk_metadata", None) or {}
        key = (
            getattr(self, "current_folder", None),
            len(files),
            epoch,
            id(bulk),
            len(bulk),
        )
        cached = getattr(self, "_gallery_any_rated_cache", None)
        if cached is not None and cached[0] == key:
            return bool(cached[1])
        any_rated = False
        for path in files:
            if path and self._gallery_path_rating(path) > 0:
                any_rated = True
                break
        self._gallery_any_rated_cache = (key, any_rated)
        return any_rated

    def get_image_capture_time(self, file_path):
        """Extract image capture time from EXIF data (DateTimeOriginal)"""
        try:
            # Try to get from cache first
            from image_cache import get_image_cache
            cache = get_image_cache()
            cached_exif = cache.get_exif(file_path)

            if cached_exif and cached_exif.get('capture_time'):
                from common_image_loader import parse_capture_time_to_timestamp
                ts = parse_capture_time_to_timestamp(cached_exif['capture_time'])
                if ts > 0:
                    return ts

            # If not in cache, try to extract from EXIF
            try:
                # Only attempt exifread on formats likely to have compatible EXIF
                # (JPEG, TIFF, and RAW formats)
                ext = os.path.splitext(file_path)[1].lower()
                exif_likely_exts = {'.jpg', '.jpeg', '.tif', '.tiff', '.arw', '.cr2', '.nef', '.raf', '.orf', '.dng', '.cr3'}

                if ext in exif_likely_exts:
                    tags = process_file_from_path(file_path, details=False)

                    # Try different datetime tags in order of preference
                    datetime_tags = [
                        "EXIF DateTimeOriginal",
                        "Image DateTime",
                        "EXIF DateTime",
                    ]
                    for tag_name in datetime_tags:
                        if tag_name in tags:
                            datetime_raw = tags[tag_name]
                            try:
                                datetime_str = str(datetime_raw)
                                # Parse datetime string (format: "YYYY:MM:DD HH:MM:SS")
                                dt = datetime.strptime(
                                    datetime_str, "%Y:%m:%d %H:%M:%S"
                                )
                                return dt.timestamp()
                            except (ValueError, AttributeError):
                                continue
            except Exception:
                pass

            # Fallback to file modification time if EXIF extraction fails
            try:
                return os.path.getmtime(file_path)
            except (OSError, AttributeError):
                # Last resort: use 0 (will sort to beginning/end depending on order)
                return 0
        except Exception:
            # If all else fails, use file modification time
            try:
                return os.path.getmtime(file_path)
            except (OSError, AttributeError):
                return 0

    def _parallel_probe_capture_times(
        self,
        file_paths,
        bulk_metadata,
        *,
        sample_path: str | None = None,
        conservative: bool = False,
    ) -> dict:
        """Probe capture timestamps in parallel for all paths not cached in bulk_metadata."""
        missing_paths = []
        for fp in file_paths:
            meta = bulk_metadata.get(fp) if bulk_metadata else None
            if not meta or not meta.get("capture_time"):
                missing_paths.append(fp)

        probed_timestamps: dict = {}
        if not missing_paths:
            return probed_timestamps

        import logging
        from concurrent.futures import ThreadPoolExecutor

        from common_image_loader import (
            io_pressure_active,
            probe_capture_timestamp_from_file,
            sort_probe_worker_count,
        )

        logger = logging.getLogger(__name__)
        hint = sample_path or missing_paths[0]
        conservative = conservative or io_pressure_active() or len(missing_paths) > 400
        workers = sort_probe_worker_count(hint, conservative=conservative)
        chunksize = max(1, len(missing_paths) // max(workers * 8, 1))
        logger.info(
            "[SORT] Probing %d uncached files (%d workers, chunksize=%d, conservative=%s)",
            len(missing_paths),
            workers,
            chunksize,
            conservative,
        )

        def _probe_one(p: str) -> tuple[str, float]:
            try:
                import time

                from image_load_manager import Priority, get_image_load_manager
                manager = get_image_load_manager()
                viewer = self
                # Yield only while arrow-key navigation is in flight so cold
                # folder sort does not steal the disk from the on-screen image.
                # Do NOT wait on PRELOAD_* (idle neighbor prefetch is almost
                # always queued) or on CURRENT once navigation has stopped --
                # full-res CURRENT upgrades stay active for seconds, and a
                # per-file 250ms wait × thousands of files permanently delayed
                # EXIF sort, keeping the Gallery button hidden.
                # Cap: RAWVIEWER_SORT_PROBE_MAX_YIELD_MS (0 = never yield).
                max_yield_ms = _env_int(
                    "RAWVIEWER_SORT_PROBE_MAX_YIELD_MS", 250, minimum=0
                )
                if max_yield_ms > 0:
                    yield_deadline = time.monotonic() + (max_yield_ms / 1000.0)
                    while time.monotonic() < yield_deadline:
                        if not getattr(viewer, "_navigation_in_progress", False):
                            break
                        has_current = False
                        if manager is not None:
                            with manager._queue_lock:
                                for t in manager._active_tasks.values():
                                    if t.is_cancelled():
                                        continue
                                    if getattr(t, "priority", None) == Priority.CURRENT:
                                        has_current = True
                                        break
                        if not has_current:
                            break
                        time.sleep(0.05)
            except Exception:
                pass

            try:
                return p, probe_capture_timestamp_from_file(p)
            except Exception:
                return p, 0.0


        with ThreadPoolExecutor(max_workers=workers) as pool:
            for p, ts in pool.map(_probe_one, missing_paths, chunksize=chunksize):
                if ts > 0:
                    probed_timestamps[p] = ts

        logger.info(
            "[SORT] Probed %d/%d capture timestamps",
            len(probed_timestamps),
            len(missing_paths),
        )

        if probed_timestamps:
            try:
                from image_cache import get_image_cache

                get_image_cache().record_probed_capture_times(probed_timestamps)
            except Exception:
                pass
        return probed_timestamps
