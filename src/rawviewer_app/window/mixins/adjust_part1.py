"""adjust mixin, part 1 of 5 (42 methods, 892 lines)."""
from __future__ import annotations

import contextlib
import os
import sys

from PyQt6.QtCore import QRect, Qt, QTimer
from PyQt6.QtGui import QColor, QPixmap, QTransform
from PyQt6.QtWidgets import QLabel, QWidget

from core import theme
from core.config import _norm_path

from .. import lazy
from ..lazy import is_raw_file, use_libraw_consistent_preview_first
from ..qt_helpers import _get_bg_thread_pool

lazy.register(sys.modules[__name__])


class AdjustPart1:
    def _preview_acceptable_for_single_view(
        self,
        image,
        *,
        file_path: str | None = None,
        context: str = "display",
    ) -> bool:
        """Whether a cached buffer may be painted (navigation=512+, display=1400+)."""
        from common_image_loader import dng_prefers_embedded_preview_first

        for_navigation = str(context or "").strip().lower() == "navigation"
        if file_path and dng_prefers_embedded_preview_first(file_path):
            min_dim = 1024
        else:
            min_dim = self._single_view_cache_min_dim(for_navigation=for_navigation)
        allow_sensor = bool(
            for_navigation
            and file_path
            and use_libraw_consistent_preview_first(file_path)
            and is_raw_file(file_path)
        )
        return self._cache_buffer_suitable_for_single_view(
            image,
            min_dim=min_dim,
            file_path=file_path,
            allow_sensor_resolution=allow_sensor,
        )

    def _raw_recovery_preview_active(self) -> bool:
        return bool(getattr(self, "_raw_recovery_active", False))

    def _adjust_panel_active(self) -> bool:
        from raw_adjustments import editing_features_enabled

        if not editing_features_enabled():
            return False
        if getattr(self, "view_mode", "") != "single":
            return False
        if not getattr(self, "_adjust_overlay_visible", False):
            return False
        panel = getattr(self, "single_image_adjust_panel", None)
        return panel is not None and panel.isVisible()

    def _toggle_auto_tone_preview(self) -> bool:
        return self._toggle_raw_recovery()

    def _refresh_auto_tone_display(self) -> None:
        self._refresh_neutral_raw_display()

    def _on_transform_interaction(self, active: bool) -> None:
        """Alignment grid while a Transform slider (straighten/perspective) is
        held: pressed shows a thirds grid over the image, released restores
        the user's own composition-grid setting. The user's persisted grid
        preference is never modified -- only the on-screen mode is overridden
        for the duration of the drag.

        Also enter geometry-overlay preview so rotation/keystone is shown on
        the full frame with the crop rectangle indicating the final outcome
        (instead of hard-cropping away context while the slider moves).
        """
        gv = getattr(self, "gpu_view", None)
        self._transform_interacting = bool(active)
        try:
            if active:
                self._enter_geometry_overlay_preview(refresh=True)
            else:
                panel = getattr(self, "single_image_adjust_panel", None)
                adj = panel.get_adjustments() if panel is not None else {}
                if not self._adj_has_warp(adj) and not (
                    panel is not None and panel.is_crop_mode()
                ):
                    self._exit_geometry_overlay_preview(refresh=True)
                elif panel is not None and not panel.is_crop_mode():
                    # Keep overlay while straighten/perspective remain non-zero;
                    # upgrade to settle quality after the slider releases.
                    self._enter_geometry_overlay_preview(refresh=False)
                    self._apply_adjust_panel_preview(full_quality=True)
        except Exception:
            pass
        if gv is None:
            return
        try:
            if active:
                self._transform_grid_saved_mode = getattr(
                    self, "_composition_grid_mode", "off"
                )
                # Fine 9x9 mesh: a 3x3 is too coarse to judge straightening /
                # keystone convergence against.
                gv.set_composition_grid_mode("fine")
            else:
                gv.set_composition_grid_mode(
                    getattr(self, "_transform_grid_saved_mode", None)
                    or getattr(self, "_composition_grid_mode", "off")
                )
            gv.viewport().update()
        except Exception:
            pass

    def _init_compare_view(self, main_layout) -> None:
        from PyQt6.QtWidgets import QHBoxLayout

        from rawviewer_ui.widgets.gpu import GpuImageView

        container = QWidget()
        container.setObjectName("rawviewer_compare_container")
        container.setStyleSheet(f"QWidget {{ background-color: {theme.VOID}; }}")
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        self.compare_select_view = GpuImageView(container)
        self.compare_candidate_view = GpuImageView(container)
        for view in (self.compare_select_view, self.compare_candidate_view):
            view._shortcut_handler = self._handle_app_shortcut
            view._wheel_navigate_in_fit = False
            view.wheelNavigate.connect(lambda _d, v=view: None)
            view.doubleClickedAt.connect(
                lambda pt, v=view: self._on_compare_double_clicked(v, pt)
            )
            view.zoomChanged.connect(
                lambda _s, v=view: self._on_compare_zoom_changed(v)
            )
            view.horizontalScrollBar().valueChanged.connect(
                lambda _v, src=view: self._on_compare_scroll_changed(src)
            )
            view.verticalScrollBar().valueChanged.connect(
                lambda _v, src=view: self._on_compare_scroll_changed(src)
            )
            row.addWidget(view, 1)

        self._compare_complete_banner = QLabel("", container)
        self._compare_complete_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._compare_complete_banner.setStyleSheet(
            f"color: {theme.INK}; font-size: 15px; font-weight: 500; background: rgba(0,0,0,0.55);"
            " padding: 12px 18px; border-radius: 8px;"
        )
        self._compare_complete_banner.hide()

        anchor = getattr(self, "single_view_container", None) or self.scroll_area
        scroll_index = main_layout.indexOf(anchor)
        main_layout.insertWidget(scroll_index + 1, container)
        self.compare_widget = container
        container.hide()

    def _burst_group_compare_plan(self):
        from core.burst_grouping import BURST_MIN_COUNT, sort_paths_by_capture_time
        from core.comparison_mode import CompareEntry

        if not getattr(self, "_burst_group_view_active", False):
            return None
        members = self._burst_group_view_visible_members()
        if len(members) < BURST_MIN_COUNT:
            return None
        capture_times = self._capture_times_for_paths(members)
        ordered = sort_paths_by_capture_time(members, capture_times)
        return ordered, CompareEntry.BURST_GROUP

    def _compare_entry_available(self) -> bool:
        if getattr(self, "view_mode", "") != "gallery":
            return False
        # Cheap gate: no plan/sort work unless at least two keys are picked.
        raw = getattr(self, "_gallery_selected_paths", None) or set()
        if len(raw) < 2:
            return False
        return len(self._gallery_selected_expanded_norm_keys()) >= 2

    def _compare_session_plan(self):
        if getattr(self, "_burst_group_view_active", False):
            return self._burst_group_compare_plan()
        from core.burst_grouping import (
            selection_covers_single_burst_group,
            sort_paths_by_capture_time,
        )
        from core.comparison_mode import CompareEntry

        keys = self._gallery_selected_expanded_norm_keys()
        if len(keys) < 2:
            return None
        path_to_group = getattr(self, "_burst_path_to_group", {}) or {}
        rejected = set(getattr(self, "_burst_rejected_paths", set()) or set())
        burst_members = selection_covers_single_burst_group(
            keys, path_to_group, rejected
        )
        if burst_members is not None:
            capture_times = self._capture_times_for_paths(burst_members)
            ordered = sort_paths_by_capture_time(burst_members, capture_times)
            return ordered, CompareEntry.BURST_GROUP
        paths = self._gallery_selected_canonical_paths_from_keys(keys)
        if len(paths) < 2:
            return None
        capture_times = self._capture_times_for_paths(paths)
        ordered = sort_paths_by_capture_time(paths, capture_times)
        return ordered, CompareEntry.GALLERY_MULTISELECT

    def _sync_compare_button_visibility(self) -> None:
        btn = getattr(self, "compare_bottom_button", None)
        if btn is None:
            return
        if getattr(self, "view_mode", "") == "compare":
            btn.hide()
            return
        available = self._compare_entry_available()
        btn.setVisible(available)
        btn.setEnabled(available)

    def _reject_compare_path(self, path: str) -> None:
        if not path:
            return
        key = _norm_path(path)
        rejected = getattr(self, "_burst_rejected_paths", set()) or set()
        rejected.add(key)
        self._burst_rejected_paths = rejected
        self._save_burst_rejected_paths()
        try:
            from metadata import backend as metadata_backend

            if metadata_backend.has_pyexiv2():
                pass  # H4 sidecar write TBC when M X2 lands
        except Exception:
            pass
        # Physically move the file to the Discard folder
        self._move_path_to_discard(path)
        self._rebuild_burst_groups()

    def _enter_compare_mode(self) -> None:
        from core.comparison_mode import ComparisonSession

        plan = self._compare_session_plan()
        if plan is None:
            return
        members, entry = plan
        if len(members) < 2:
            return
        session = ComparisonSession(members=members, entry=entry)
        session.restart_round()
        self._comparison_session = session
        self._compare_sync_mode = "A"
        self._compare_sync_paused = False
        self.view_mode = "compare"
        self._show_compare_view()

    def _show_compare_view(self) -> None:
        session = getattr(self, "_comparison_session", None)
        if session is None:
            return
        if hasattr(self.image_manager, "set_gallery_view_active"):
            self.image_manager.set_gallery_view_active(False)
        self._stop_slideshow()
        bar = self._filmstrip_bar()
        if bar is not None:
            bar.hide()
            bar.setEnabled(False)
        self._hide_filmstrip_chrome()
        if hasattr(self, "single_view_container") and self.single_view_container:
            self.single_view_container.hide()
        if hasattr(self, "gallery_widget") and self.gallery_widget:
            self.gallery_widget.hide()
        if hasattr(self, "compare_widget") and self.compare_widget:
            self.compare_widget.show()
            self.compare_widget.raise_()
        if hasattr(self, "sort_toggle_button"):
            self.sort_toggle_button.hide()
        if hasattr(self, "size_slider"):
            self.size_slider.hide()
        if hasattr(self, "compare_bottom_button"):
            self.compare_bottom_button.hide()
        self._stop_slideshow()
        self._suppress_single_manager_callbacks = False
        self._sync_burst_group_bottom_chrome()
        self._queue_compare_session_decodes()
        self._refresh_compare_panes()
        self._sync_share_button_visibility()
        self._sync_composition_grid_display()
        self._sync_clipping_overlay()
        self._update_compare_status_bar()
        self._restore_keyboard_focus()

    def _exit_compare_mode(self) -> None:
        session = getattr(self, "_comparison_session", None)
        entry = getattr(session, "entry", None) if session else None
        select_path = getattr(session, "select", None) if session else None
        remaining = session.visible_members() if session else []
        self._comparison_session = None
        self._compare_sync_mode = "A"
        self._compare_sync_paused = False
        self._compare_focus_outline_active = False
        self._compare_clear_focus_overlays()
        if hasattr(self, "_compare_complete_banner") and self._compare_complete_banner:
            self._compare_complete_banner.hide()
        if hasattr(self, "compare_widget") and self.compare_widget:
            self.compare_widget.hide()
            for view in (
                getattr(self, "compare_select_view", None),
                getattr(self, "compare_candidate_view", None),
            ):
                if view is not None:
                    view.clearFocus()
                    view.clear_clipping_overlay()
                    self._compare_stop_animation(view)
        self.view_mode = "gallery"
        from PyQt6.QtCore import QTimer

        from core.comparison_mode import CompareEntry

        if entry == CompareEntry.BURST_GROUP:
            if len(remaining) > 1:
                self._burst_group_view_active = True
                self._burst_group_view_members = remaining
                self._show_gallery_view()
            else:
                self._exit_burst_group_view(scroll_to=select_path or (remaining[0] if remaining else None))
                self._show_gallery_view()
        else:
            if getattr(self, "_burst_group_view_active", False):
                self._exit_burst_group_view()
            self._show_gallery_view()
            if select_path:
                gj = getattr(self, "gallery_justified", None)
                if gj is not None and hasattr(gj, "scroll_to_file"):
                    QTimer.singleShot(0, lambda: gj.scroll_to_file(select_path))
        self._sync_compare_button_visibility()
        self.update_status_bar()

        self._restore_keyboard_focus()
        QTimer.singleShot(0, self._restore_keyboard_focus)
        QTimer.singleShot(100, self._restore_keyboard_focus)

    def _refresh_compare_panes(self, changed=None) -> None:
        session = getattr(self, "_comparison_session", None)
        if session is None:
            return
        if hasattr(self, "_compare_complete_banner") and self._compare_complete_banner:
            if session.complete:
                self._compare_complete_banner.setText(
                    "Comparison complete. Press Esc to return."
                )
                self._compare_complete_banner.show()
                self._compare_complete_banner.raise_()
                self._position_compare_complete_banner()
            else:
                self._compare_complete_banner.hide()
        if changed != "candidate":
            self._load_compare_pane(
                session.select,
                self.compare_select_view,
                force=(changed == "both"),
            )
        if changed != "select":
            self._load_compare_pane(
                session.candidate,
                self.compare_candidate_view,
                force=(changed == "both"),
            )
        self._update_compare_status_bar()
        self._sync_clipping_overlay()

    def _position_compare_complete_banner(self) -> None:
        banner = getattr(self, "_compare_complete_banner", None)
        container = getattr(self, "compare_widget", None)
        if banner is None or container is None:
            return
        banner.adjustSize()
        bw = banner.sizeHint().width()
        bh = banner.sizeHint().height()
        banner.setGeometry(
            max(0, (container.width() - bw) // 2),
            max(0, (container.height() - bh) // 2),
            bw,
            bh,
        )

    def _compare_view_for_path(self, path: str):
        if not path:
            return None
        norm = _norm_path(path)
        for view in (
            getattr(self, "compare_select_view", None),
            getattr(self, "compare_candidate_view", None),
        ):
            if view is None:
                continue
            if _norm_path(getattr(view, "file_path", "") or "") == norm:
                return view
        return None

    def _compare_pane_max_dim(self, view) -> int:
        if view is None or not view.has_pixmap():
            return 0
        return max(int(view._img_w), int(view._img_h))

    def _compare_best_pixmap_for_path(self, file_path: str):
        """Return (pixmap, orientation_already_applied) for compare panes."""
        if not file_path:
            return None, True
        best_pm = None
        best_dim = 0
        best_oriented = True

        def _consider(candidate, *, oriented: bool) -> None:
            nonlocal best_pm, best_dim, best_oriented
            if candidate is None:
                return
            pm = candidate
            if hasattr(candidate, "isNull"):
                if candidate.isNull():
                    return
                dim = max(candidate.width(), candidate.height())
            else:
                try:
                    dim = max(int(candidate.shape[0]), int(candidate.shape[1]))
                except Exception:
                    return
                pm = self._numpy_to_qpixmap(candidate)
                if pm is None or pm.isNull():
                    return
            if dim > best_dim:
                best_dim = dim
                best_pm = pm
                best_oriented = oriented

        with contextlib.suppress(Exception):
            _consider(self.image_cache.get_full_image(file_path), oriented=True)
        with contextlib.suppress(Exception):
            _consider(self.image_cache.get_preview(file_path), oriented=True)
        with contextlib.suppress(Exception):
            _consider(self.image_cache.get_pixmap(file_path), oriented=False)
        _consider(self._gallery_preview_pixmap_for_path(file_path), oriented=True)
        if best_dim < 1920 and file_path:
            try:
                from common_image_loader import is_raw_file

                if is_raw_file(file_path):
                    embedded = self._extract_embedded_preview(file_path)
                    if embedded is not None and not embedded.isNull():
                        ed = max(embedded.width(), embedded.height())
                        if ed > best_dim:
                            return embedded, True
            except Exception:
                pass
        return best_pm, best_oriented

    def _compare_apply_visual_rotation_for_path(
        self, pixmap: QPixmap, file_path: str
    ) -> QPixmap:
        """Apply stored non-destructive visual rotation for a compare-pane pixmap."""
        if pixmap is None or pixmap.isNull() or not file_path:
            return pixmap
        degrees = self._get_visual_rotation_degrees(file_path)
        if degrees == 0:
            return pixmap
        transform = QTransform()
        transform.rotate(degrees)
        return pixmap.transformed(transform, Qt.TransformationMode.SmoothTransformation)

    def _compare_orient_pixmap_for_path(self, file_path: str, pm: QPixmap) -> QPixmap:
        # No make-based skip: _compare_pixmap_matches_exif below is the make-independent
        # dimension check that decides whether the pixmap is already display-oriented.
        if pm is None or pm.isNull() or not file_path:
            return pm
        if self._compare_pixmap_matches_exif(file_path, pm):
            return pm
        orientation = int(self._compare_exif_for_path(file_path).get("orientation", 1) or 1)
        if orientation != 1:
            return self.apply_orientation_to_pixmap(pm, orientation)
        return pm

    def _compare_prepare_pane_pixmap(self, file_path: str, pm: QPixmap) -> QPixmap:
        pm = self._compare_orient_pixmap_for_path(file_path, pm)
        return self._compare_apply_visual_rotation_for_path(pm, file_path)

    def _compare_display_pixmap_for_path(self, file_path: str):
        """Return compare-pane pixmap with correct orientation and visual rotation."""
        self._ensure_compare_exif_ready(file_path)
        pm, _orientation_applied = self._compare_best_pixmap_for_path(file_path)
        if pm is None or pm.isNull():
            return pm
        return self._compare_prepare_pane_pixmap(file_path, pm)

    def _compare_focus_rect_for_path(self, path: str, view):
        """Focus/subject rect in displayed compare-pane image coordinates."""
        if not path or view is None or not view.has_pixmap():
            return None, None
        try:
            from metadata.subject_area import _rotate_rect, pixmap_ltwh_focus_hint

            orientation = 1
            cached_exif = self.image_cache.get_exif(path)
            if cached_exif:
                orientation = cached_exif.get("orientation", 1)
            degrees = self._get_visual_rotation_degrees(path)
            disp_w, disp_h = int(view._img_w), int(view._img_h)
            if degrees in (90, 270):
                base_w, base_h = disp_h, disp_w
            else:
                base_w, base_h = disp_w, disp_h
            hint = pixmap_ltwh_focus_hint(path, base_w, base_h, orientation)
            if hint is None:
                return None, None
            ltwh, src = hint
            if degrees:
                ocode = {90: 6, 180: 3, 270: 8}.get(degrees, 1)
                ltwh = _rotate_rect(ltwh, base_w, base_h, ocode)
            return (
                QRect(ltwh[0], ltwh[1], ltwh[2], ltwh[3]),
                "makernote_af" if src == "makernote_af" else "exif_subject",
            )
        except Exception:
            return None, None

    def _compare_update_focus_overlay_for_view(self, view) -> None:
        if view is None:
            return
        if not getattr(self, "_compare_focus_outline_active", False):
            view.clear_overlay()
            return
        path = getattr(view, "file_path", None)
        if not path:
            view.clear_overlay()
            return
        rect, src = self._compare_focus_rect_for_path(str(path), view)
        if rect is not None and not rect.isNull():
            color = (
                QColor(255, 185, 45)
                if src == "makernote_af"
                else QColor(165, 255, 95)
            )
            view.set_overlay_rect(rect, color)
        else:
            view.clear_overlay()

    def _compare_update_focus_overlays(self) -> None:
        for view in (
            getattr(self, "compare_select_view", None),
            getattr(self, "compare_candidate_view", None),
        ):
            self._compare_update_focus_overlay_for_view(view)

    def _compare_clear_focus_overlays(self) -> None:
        for view in (
            getattr(self, "compare_select_view", None),
            getattr(self, "compare_candidate_view", None),
        ):
            if view is not None:
                view.clear_overlay()

    def _queue_compare_pane_decode(self, file_path: str) -> None:
        if not file_path:
            return
        try:
            from image_load_manager import Priority

            self.image_manager.load_image(
                file_path,
                priority=Priority.CURRENT,
                cancel_existing=False,
                use_full_resolution=True,
                stages={"exif", "thumbnail", "full"},
            )
        except Exception:
            pass

    def _queue_compare_session_decodes(self) -> None:
        session = getattr(self, "_comparison_session", None)
        if session is None:
            return
        for path in session.visible_members():
            if path:
                self._ensure_compare_exif_ready(path)
                self._queue_compare_pane_decode(path)

    def _compare_upgrade_view_image(self, file_path: str, image) -> bool:
        """Apply a higher-resolution buffer to the matching compare pane, if any."""
        if getattr(self, "view_mode", "") != "compare":
            return False
        view = self._compare_view_for_path(file_path)
        if view is None:
            return False
        from PyQt6.QtGui import QImage, QPixmap

        if hasattr(image, "shape"):
            new_dim = max(int(image.shape[0]), int(image.shape[1]))
            raw_pm = self._numpy_to_qpixmap(image)
        elif isinstance(image, QImage):
            new_dim = max(int(image.height()), int(image.width()))
            raw_pm = QPixmap.fromImage(image)
        elif hasattr(image, "width"):
            new_dim = max(int(image.height()), int(image.width()))
            raw_pm = image
        else:
            return False
        if raw_pm is None or raw_pm.isNull():
            return False
        self._ensure_compare_exif_ready(file_path)
        pm = self._compare_prepare_pane_pixmap(file_path, raw_pm)
        if pm is None or pm.isNull():
            return False
        old_dim = self._compare_pane_max_dim(view)
        size_changed = (
            view._img_w != pm.width()
            or view._img_h != pm.height()
        )
        if (
            old_dim > 0
            and new_dim <= int(old_dim * 1.02)
            and not size_changed
            and self._compare_pixmap_matches_exif(file_path, raw_pm)
        ):
            if getattr(self, "_compare_focus_outline_active", False):
                self._compare_update_focus_overlay_for_view(view)
            return True
        preserve = not view.is_fit_mode()
        view.set_pixmap(pm, preserve_view=preserve)
        if getattr(self, "_compare_focus_outline_active", False):
            self._compare_update_focus_overlay_for_view(view)
        self._compare_refresh_stale_pane_orientations(exclude_path=file_path)
        return True

    def _compare_refresh_stale_pane_orientations(self, *, exclude_path: str | None = None) -> None:
        """Re-render compare panes whose on-screen buffer no longer matches EXIF."""
        session = getattr(self, "_comparison_session", None)
        if session is None:
            return
        exclude = _norm_path(exclude_path) if exclude_path else ""
        for path, view in (
            (session.select, getattr(self, "compare_select_view", None)),
            (session.candidate, getattr(self, "compare_candidate_view", None)),
        ):
            if not path or view is None or _norm_path(path) == exclude:
                continue
            self._ensure_compare_exif_ready(path)
            correct = self._compare_display_pixmap_for_path(path)
            if correct is None or correct.isNull():
                continue
            if (
                not view.has_pixmap()
                or view._img_w != correct.width()
                or view._img_h != correct.height()
            ):
                preserve = not view.is_fit_mode()
                view.set_pixmap(correct, preserve_view=preserve)
                if getattr(self, "_compare_focus_outline_active", False):
                    self._compare_update_focus_overlay_for_view(view)
                continue
            best_pm, _ = self._compare_best_pixmap_for_path(path)
            if best_pm is not None and not best_pm.isNull() and not self._compare_pixmap_matches_exif(path, best_pm):
                    preserve = not view.is_fit_mode()
                    view.set_pixmap(correct, preserve_view=preserve)
                    if getattr(self, "_compare_focus_outline_active", False):
                        self._compare_update_focus_overlay_for_view(view)

    def _load_compare_pane(self, path: str, view, *, force: bool = False) -> None:
        if view is None:
            return
        if not path:
            view.set_pixmap(QPixmap())
            view.file_path = None
            self._compare_stop_animation(view)
            return

        # Stop existing animation before switching/loading path
        self._compare_stop_animation(view)

        norm_path = _norm_path(path)
        same_path = _norm_path(getattr(view, "file_path", "") or "") == norm_path
        if same_path and view.has_pixmap() and not force:
            self._ensure_compare_exif_ready(path)
            correct = self._compare_display_pixmap_for_path(path)
            if correct is not None and not correct.isNull():
                if view._img_w != correct.width() or view._img_h != correct.height():
                    preserve = not view.is_fit_mode()
                    view.set_pixmap(correct, preserve_view=preserve)
                else:
                    best_pm, _ = self._compare_best_pixmap_for_path(path)
                    if best_pm is not None and not self._compare_pixmap_matches_exif(path, best_pm):
                        preserve = not view.is_fit_mode()
                        view.set_pixmap(correct, preserve_view=preserve)
                    else:
                        self._compare_start_animation(view, path)
                        return
            else:
                self._compare_start_animation(view, path)
                return
        view.file_path = path
        self._ensure_compare_exif_ready(path)
        pm = self._compare_display_pixmap_for_path(path)
        if pm is not None and not pm.isNull():
            if same_path and not force and not view.is_fit_mode():
                view.set_pixmap(pm, preserve_view=True)
            else:
                view.set_pixmap(pm)
                view.fit_to_window()
        else:
            view.set_pixmap(QPixmap())
            view.set_placeholder_text(os.path.basename(path))
        self._queue_compare_pane_decode(path)
        if getattr(self, "_compare_focus_outline_active", False):
            self._compare_update_focus_overlay_for_view(view)

        # Start animation for the newly set path
        self._compare_start_animation(view, path)

    def _compare_stop_animation(self, view) -> None:
        anims = getattr(self, "_compare_view_animations", {})
        if view in anims:
            anim = anims[view]
            if anim["type"] == "gif":
                movie = anim["movie"]
                movie.stop()
                with contextlib.suppress(Exception):
                    movie.frameChanged.disconnect(anim["handler"])
            elif anim["type"] == "webp":
                timer = anim["timer"]
                timer.stop()
            del anims[view]

    def _compare_start_animation(self, view, path: str) -> None:
        if not path or not os.path.exists(path):
            return
        ext = os.path.splitext(path)[1].lower()
        if ext not in (".gif", ".webp"):
            return

        if not hasattr(self, "_compare_view_animations"):
            self._compare_view_animations = {}

        if ext == ".gif":
            from PyQt6.QtGui import QMovie
            movie = QMovie(path)
            # Create a safe closure/handler that checks if this view is still displaying this path
            def handler():
                if getattr(view, "file_path", "") == path:
                    pm = movie.currentPixmap()
                    if pm is not None and not pm.isNull():
                        view.set_pixmap(pm, preserve_view=True)
            movie.frameChanged.connect(handler)
            movie.start()
            self._compare_view_animations[view] = {
                "type": "gif",
                "movie": movie,
                "handler": handler
            }
        elif ext == ".webp":
            from PyQt6.QtCore import QObject, QRunnable, pyqtSignal

            class WebpCompareSignals(QObject):
                ready = pyqtSignal(list, list) # frames (QImage), durations

            class WebpCompareWorker(QRunnable):
                def __init__(self_inner):
                    super().__init__()
                    self_inner.signals = WebpCompareSignals()
                def run(self_inner):
                    try:
                        from PIL import Image
                        frames = []
                        durations = []
                        with Image.open(path) as img:
                            n_frames = getattr(img, "n_frames", 1)
                            if n_frames <= 1:
                                return # Static webp
                            for frame_idx in range(n_frames):
                                img.seek(frame_idx)
                                from PIL.ImageQt import ImageQt
                                qim = ImageQt(img.convert("RGBA"))
                                frames.append(qim.copy())
                                durations.append(img.info.get("duration", 100) or 100)
                        self_inner.signals.ready.emit(frames, durations)
                    except Exception:
                        pass

            worker = WebpCompareWorker()
            def on_ready(qimages, durations):
                if getattr(view, "file_path", "") != path:
                    return
                pixmaps = [QPixmap.fromImage(qim) for qim in qimages]
                timer = QTimer(self)
                timer.setSingleShot(True)

                state = {
                    "frames": pixmaps,
                    "durations": durations,
                    "idx": 0
                }

                def play_frame():
                    if getattr(view, "file_path", "") != path:
                        timer.stop()
                        return
                    idx = state["idx"]
                    pm = state["frames"][idx]
                    view.set_pixmap(pm, preserve_view=True)

                    duration = state["durations"][idx]
                    state["idx"] = (idx + 1) % len(state["frames"])
                    timer.start(duration)

                timer.timeout.connect(play_frame)
                play_frame()

                self._compare_view_animations[view] = {
                    "type": "webp",
                    "timer": timer
                }

            worker.signals.ready.connect(on_ready)
            _get_bg_thread_pool().start(worker)

    def _on_compare_double_clicked(self, source_view, scene_pt) -> None:
        if getattr(self, "view_mode", "") != "compare":
            return
        if getattr(self, "_compare_sync_mode", "A") != "A":
            return
        select_view = self.compare_select_view
        candidate_view = self.compare_candidate_view
        if select_view is None or candidate_view is None:
            return
        if not select_view.has_pixmap() or not candidate_view.has_pixmap():
            return
        w = max(1, source_view._img_w)
        h = max(1, source_view._img_h)
        u = max(0.0, min(1.0, float(scene_pt.x()) / w))
        v = max(0.0, min(1.0, float(scene_pt.y()) / h))
        self._compare_zoom_both_to_normalized(u, v)

    def _compare_relative_zoom(self, view) -> float:
        """Zoom level as a multiple of fit-to-window (1.0 == fit)."""
        if view is None or not view.has_pixmap():
            return 1.0
        fit = view.fit_scale()
        if fit <= 0:
            return 1.0
        return view.current_scale() / fit

    def _compare_set_relative_zoom(self, view, relative: float) -> None:
        """Apply zoom relative to this pane's own fit scale (handles mixed orientations)."""
        if view is None or not view.has_pixmap():
            return
        fit = view.fit_scale()
        eps = view._FIT_SCALE_EPS
        if relative <= eps:
            view.fit_to_window()
        else:
            view.set_scale(relative * fit)

    def _compare_zoom_both_to_normalized(self, u: float, v: float) -> None:
        self._compare_sync_guard = True
        try:
            for view in (self.compare_select_view, self.compare_candidate_view):
                if view is None or not view.has_pixmap():
                    continue
                x = u * max(0, view._img_w - 1)
                y = v * max(0, view._img_h - 1)
                if view.is_at_fit_scale() or view.is_fit_mode():
                    view.zoom_to_actual_at(x, y)
                else:
                    view.fit_to_window()
        finally:
            self._compare_sync_guard = False

    def _compare_toggle_zoom_mode_a(self) -> None:
        if getattr(self, "_compare_sync_mode", "A") != "A":
            return
        select_view = self.compare_select_view
        candidate_view = self.compare_candidate_view
        if select_view is None or candidate_view is None:
            return
        if (
            select_view.has_pixmap()
            and candidate_view.has_pixmap()
            and select_view.is_at_fit_scale()
            and candidate_view.is_at_fit_scale()
        ):
            if getattr(self, "_compare_focus_outline_active", False):
                self._compare_zoom_panes_to_focus_points()
            else:
                self._compare_zoom_both_to_normalized(0.5, 0.5)
        else:
            self._compare_sync_guard = True
            try:
                if select_view.has_pixmap():
                    select_view.fit_to_window()
                if candidate_view.has_pixmap():
                    candidate_view.fit_to_window()
            finally:
                self._compare_sync_guard = False
            self._compare_sync_paused = False

    def _compare_focus_point_for_path(self, path: str, view) -> tuple[float, float]:
        """Normalized (u, v) focus center in compare-pane coordinates; 0.5,0.5 if unknown."""
        if not path or view is None or not view.has_pixmap():
            return 0.5, 0.5
        rect, _src = self._compare_focus_rect_for_path(path, view)
        if rect is None or rect.isNull() or rect.width() < 1 or rect.height() < 1:
            return 0.5, 0.5
        cx = rect.left() + rect.width() * 0.5
        cy = rect.top() + rect.height() * 0.5
        return (
            max(0.0, min(1.0, cx / max(1, view._img_w))),
            max(0.0, min(1.0, cy / max(1, view._img_h))),
        )

    def _compare_zoom_panes_to_focus_points(self) -> None:
        """Zoom each compare pane to its own EXIF focus center (frame center when missing)."""
        session = getattr(self, "_comparison_session", None)
        select_view = self.compare_select_view
        candidate_view = self.compare_candidate_view
        if select_view is None or candidate_view is None:
            return
        pane_specs = (
            (getattr(session, "select", None), select_view),
            (getattr(session, "candidate", None), candidate_view),
        )
        self._compare_sync_guard = True
        try:
            for path, view in pane_specs:
                if view is None or not view.has_pixmap() or not path:
                    continue
                u, v = self._compare_focus_point_for_path(str(path), view)
                view.zoom_to_actual_at(
                    u * max(0, view._img_w - 1),
                    v * max(0, view._img_h - 1),
                )
        finally:
            self._compare_sync_guard = False
        # Panes land on different centers; pause mirror sync until fit again.
        self._compare_sync_paused = True
