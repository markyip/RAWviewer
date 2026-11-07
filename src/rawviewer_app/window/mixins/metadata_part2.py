"""metadata mixin, part 2 of 2 (30 methods, 708 lines)."""
from __future__ import annotations

import contextlib
import os
import platform
import sys
import threading
from typing import Any

from PyQt6.QtCore import QPoint, QTimer
from PyQt6.QtWidgets import QMessageBox

from core.config import _norm_path, safe_print

from .. import lazy
from ..lazy import exif_orientation_after_cw90
from ..paths import _require_str_path

lazy.register(sys.modules[__name__])


class MetadataPart2:
    def sort_files_by_capture_time(
        self,
        file_paths,
        newest_first=True,
        file_stats=None,
        *,
        probe_uncached: bool = True,
        folder_path: str | None = None,
        sort_probe_conservative: bool = False,
        capture_time_only: bool = False,
    ):
        """Sort files by capture time according to user preference (Newest/Oldest).

        capture_time_only: read ONLY the capture_time column (via
        get_capture_times_for_sort) instead of the full per-file EXIF blob.
        Sorting needs nothing else, and get_multiple_exif() JSON-decodes
        a ~97-tag EXIF dict per file -- on a background sort worker that
        burst holds the GIL and starves the on-screen image's
        decode/paint (measured stalling session-restore first-image display).
        In this mode bulk_metadata is returned EMPTY: the full per-file EXIF
        (gallery tile geometry) is fetched lazily when the gallery is actually
        opened (see the _GalleryMetadataFetch path), and returning {} here
        avoids both clobbering _gallery_bulk_metadata and suppressing that lazy
        fetch.
        """
        import logging
        import time
        logger = logging.getLogger(__name__)

        if not file_paths:
            return [], {}

        sort_start = time.time()

        # 1. Bulk fetch metadata for all files at once (MUCH faster than individual calls)
        from image_cache import get_image_cache
        cache = get_image_cache()

        # sort_meta holds only what the sort keys need (capture_time); it is what
        # we return as bulk_metadata EXCEPT in capture_time_only mode (see below).
        if capture_time_only:
            ct_map = cache.get_capture_times_for_sort(list(file_paths))
            sort_meta = {
                fp: {"capture_time": ct} for fp, ct in ct_map.items() if ct
            }
        else:
            # Pass the pre-collected file stats to avoid redundant os.stat calls
            sort_meta = cache.get_multiple_exif(file_paths, file_stats)

            # capture_time doesn't depend on RAW_EXIF_SENSOR_META_VER / orientation-trust
            # checks that get_multiple_exif gates the rest of a row on, so a previously
            # probed-and-persisted capture_time (record_probed_capture_times) can still
            # satisfy sorting even when the row it's stored alongside fails those
            # unrelated freshness checks and gets excluded from sort_meta above.
            # Without this, every folder re-open would re-probe the same files forever.
            needs_capture_time = [
                fp for fp in file_paths if not (sort_meta.get(fp) or {}).get("capture_time")
            ]
            if needs_capture_time:
                extra_capture_times = cache.get_capture_times_for_sort(needs_capture_time)
                for fp, ct in extra_capture_times.items():
                    if ct:
                        sort_meta[fp] = {**(sort_meta.get(fp) or {}), "capture_time": ct}

        # 2. Parallel probe for any uncached files (skip when caller only wants cache hits)
        probed_timestamps: dict = {}
        if probe_uncached:
            sample = folder_path
            if not sample and file_paths:
                sample = os.path.dirname(os.path.abspath(file_paths[0]))
            probed_timestamps = self._parallel_probe_capture_times(
                file_paths,
                sort_meta,
                sample_path=sample,
                conservative=sort_probe_conservative,
            )

        # 3. Pre-calculate sorting keys to avoid repeated strptime calls
        # This is a MASSIVE optimization for thousands of files
        from common_image_loader import resolve_folder_sort_timestamp

        sort_keys = {}
        for fp in file_paths:
            mtime = 0.0
            if file_stats and fp in file_stats:
                mtime = file_stats[fp][1]
            meta = sort_meta.get(fp) if sort_meta else None
            probed_ts = probed_timestamps.get(fp, 0.0)

            _has_capture, timestamp, _source = resolve_folder_sort_timestamp(
                fp,
                metadata=meta,
                file_mtime=mtime,
                probe_file=False,
                probed_capture_timestamp=probed_ts
            )
            sort_keys[fp] = (timestamp, os.path.basename(fp).lower())

        # 4. Perform the sort using pre-calculated keys
        sorted_files = sorted(
            file_paths,
            key=lambda fp: sort_keys[fp],
            reverse=newest_first
        )

        sort_time = time.time() - sort_start
        logger.info(
            "[SORT] %s fetch & sort of %d files took %.3fs",
            "Capture-time-only" if capture_time_only else "Bulk metadata",
            len(file_paths),
            sort_time,
        )
        safe_print(f"[PERF] 🔄 Sorted {len(file_paths)} files in {sort_time*1000:.1f}ms")
        # capture_time_only: bulk_metadata is intentionally empty so the gallery
        # lazily fetches full per-file EXIF on first open (see the docstring).
        return sorted_files, ({} if capture_time_only else sort_meta)

    def _flush_deferred_full_exif_after_first_paint(
        self, file_path: str | None = None
    ) -> None:
        fp = file_path or getattr(self, "_defer_full_exif_path", None)
        if not fp:
            return
        if _norm_path(fp) != _norm_path(getattr(self, "current_file_path", "") or ""):
            self._defer_full_exif_path = None
            return
        norm = _norm_path(fp)
        if getattr(self, "_full_exif_queued_norm", None) == norm:
            self._defer_full_exif_path = None
            return
        self._full_exif_queued_norm = norm
        self._defer_full_exif_path = None
        from image_load_manager import Priority as _BgPriority

        self.image_manager.load_image(
            fp,
            priority=_BgPriority.BACKGROUND,
            cancel_existing=False,
            use_full_resolution=False,
            stages={"exif"},
        )

    def _schedule_raw_sensor_exif_status_refresh(self) -> None:
        """Reload EXIF through EXIFExtractor so status shows sensor WxH."""
        fp = getattr(self, "current_file_path", None)
        if not fp or not os.path.isfile(fp):
            return
        if getattr(self, "view_mode", "single") != "single":
            return
        if getattr(self, "_raw_status_exif_refresh_path", None) == fp:
            return
        if getattr(self, "_raw_status_exif_refresh_scheduled", False):
            return
        self._raw_status_exif_refresh_scheduled = True

        def _job():
            self._raw_status_exif_refresh_scheduled = False
            if getattr(self, "current_file_path", None) != fp or not os.path.isfile(fp):
                return
            try:
                from enhanced_raw_processor import EXIFExtractor

                data = EXIFExtractor().extract_exif_data(fp)
                if data:
                    self.image_cache.put_exif(fp, data)
                    self._raw_status_exif_refresh_path = fp
                    self.update_status_bar()
            except Exception:
                pass

        QTimer.singleShot(0, _job)

    def on_exif_data_ready(self, exif_data):
        """Handle when EXIF data becomes available."""
        # Update status bar immediately with EXIF data when it becomes available
        # This ensures EXIF data is shown even in fit-to-window mode
        import logging
        logger = logging.getLogger(__name__)
        if self.current_file_path:
            # Set a flag to indicate EXIF data is ready
            self._exif_data_ready = True
            # Always update status bar when EXIF data is ready, even if pixmap is not yet available
            # This ensures original resolution is shown immediately
            logger.info(f"[EXIF] EXIF data ready, updating status bar for {os.path.basename(_require_str_path(self.current_file_path))}")
            self.update_status_bar()
            logger.info("[EXIF] Status bar updated")

    def _sync_single_image_histogram(self):
        """Refresh histogram from current_pixmap when in single-image mode."""
        w = getattr(self, "single_image_histogram", None)
        w_adj = getattr(self, "single_image_adjust_histogram", None)
        if w is None:
            return
        if getattr(self, "view_mode", "single") != "single":
            return
        pm = getattr(self, "current_pixmap", None)
        if pm is None or pm.isNull():
            w.clear()
            w.setEnabled(False)
            w.setVisible(False)
            if w_adj:
                w_adj.clear()
            c = getattr(self, "single_view_container", None)
            if c is not None and hasattr(c, "relayout_histogram"):
                c.relayout_histogram()
        else:
            w.setEnabled(True)
            if self._histogram_user_hidden:
                self._histogram_overlay_visible = False
            else:
                self._histogram_overlay_visible = True

            # Hide floating histogram if adjust panel is visible
            panel = getattr(self, "single_image_adjust_panel", None)
            if panel and panel.isVisible():
                w.setVisible(False)
            else:
                w.setVisible(getattr(self, "_histogram_overlay_visible", False))

            w.set_pixmap(pm)
            if w_adj:
                w_adj.set_pixmap(pm)
            c = getattr(self, "single_view_container", None)
            if c is not None and hasattr(c, "relayout_histogram"):
                c.relayout_histogram()
        self._sync_clipping_overlay()
        self._sync_location_map()
        self._sync_adjust_panel()

    def _map_adjust_display_point_to_buffer(
        self, pt, buffer_w: int, buffer_h: int
    ) -> tuple[float, float]:
        """Display-pixmap point → same-framing buffer (dodge/burn + WB pick)."""
        from raw_adjustments.transform import map_display_point_to_buffer

        ux, uy = self._display_point_to_unrotated(float(pt.x()), float(pt.y()))
        disp = self._unrotated_display_size()
        if disp is None:
            return ux, uy
        return map_display_point_to_buffer(
            ux,
            uy,
            display_w=disp[0],
            display_h=disp[1],
            buffer_w=int(buffer_w),
            buffer_h=int(buffer_h),
        )

    def _on_xmp_preset_applied(self) -> None:
        """Drop per-image local mask when a global XMP look is applied."""
        self._dodge_burn_mask = None
        self._dodge_burn_luma_guide = None
        self._spot_heal_mask = None
        panel = getattr(self, "single_image_adjust_panel", None)
        if panel is not None:
            panel.set_dodge_burn_mask_present(False)
            if hasattr(panel, "set_spot_heal_mask_present"):
                panel.set_spot_heal_mask_present(False)
        self._sync_dodge_burn_mask_overlay()

    def _camera_identity_for_file(self, path: str) -> tuple[str, str, int | None]:
        """(make, model, iso) from EXIF, or ("", "", None) if unreadable.

        Delegates to color_calibration so the profile lookup and this
        banner/dialog path can never disagree about a camera's identity.
        """
        from raw_adjustments.color_calibration import camera_identity_for_file

        return camera_identity_for_file(path)

    def _refresh_camera_profile_banner(self, panel: Any, path: str) -> None:
        """Show the panel banner when a saved profile is auto-applied to this file.

        Mirrors load_adjustments_for_file: the profile is only applied when
        the image has no XMP sidecar, so a file the user has already edited
        must not claim a profile is in effect.
        """
        try:
            from raw_adjustments import resolve_xmp_path
            from raw_adjustments.color_calibration import describe_camera_profile

            xmp = resolve_xmp_path(path)
            if xmp and os.path.isfile(xmp):
                panel.set_camera_profile_active("")
                return
            make, model, iso_val = self._camera_identity_for_file(path)
            if not (make or model):
                panel.set_camera_profile_active("")
                return
            label = describe_camera_profile(make, model, iso=iso_val, file_path=path)
            panel.set_camera_profile_active(
                label or "", make=make, model=model, iso=iso_val
            )
        except Exception:
            with contextlib.suppress(Exception):
                panel.set_camera_profile_active("")

    def _on_camera_profile_removed(self) -> None:
        """Re-decode defaults after a saved camera profile was deleted."""
        path = getattr(self, "current_file_path", None)
        if not path:
            return
        panel = getattr(self, "single_image_adjust_panel", None)
        if panel is None:
            return
        try:
            from raw_adjustments import load_adjustments_for_file

            panel.set_adjustments(load_adjustments_for_file(path))
        except Exception:
            pass
        self._apply_adjust_panel_preview(full_quality=True)
        if hasattr(self, "status_bar") and self.status_bar:
            self.status_bar.showMessage("Camera profile removed", 3000)

    def _on_calibrate_camera_requested(self) -> None:
        """Launch interactive camera color calibration dialog for ColorChecker targets."""
        # current_file_path is the viewer's attribute; self.file_path only
        # exists on worker classes, so the old getattr always returned None and
        # this menu entry could never open.
        # The dialog always offers "Use my own file" plus a DPReview browser
        # search -- do not require a ColorChecker to already be the current photo.
        target_path = getattr(self, "current_file_path", None)
        make, model, iso_val = "", "", None
        img = None
        factory = None
        source_file = None

        try:
            from PyQt6.QtWidgets import QDialog

            from processing.unified import UnifiedImageProcessor
            from rawviewer_ui.dialogs.color_calibration_dialog import ColorCalibrationDialog

            if target_path and os.path.isfile(target_path):
                make, model, iso_val = self._camera_identity_for_file(target_path)
                source_file = target_path
                img = getattr(self, "_adjust_preview_base_rgb", None)
                if img is None:
                    img = self._ensure_adjust_preview_base(target_path)
                if img is None:
                    proc = UnifiedImageProcessor()
                    img = proc.decode_raw_edit_base(target_path, use_full_resolution=False)
                if img is not None and not hasattr(img, "shape"):
                    img = None
                try:
                    from raw_adjustments.color_calibration import has_factory_color_profile

                    factory = has_factory_color_profile(target_path)
                except Exception:
                    factory = None
            elif hasattr(self, "status_bar") and self.status_bar:
                self.status_bar.showMessage(
                    "Choose a ColorChecker file, or find a DPReview sample.", 4000
                )

            dlg = ColorCalibrationDialog(
                img, make, model, iso=iso_val, parent=self,
                has_factory_profile=factory,
                source_file=source_file,
            )
            if dlg.exec() == QDialog.DialogCode.Accepted and dlg.calibrated_profile:
                profile = dlg.calibrated_profile
                panel = getattr(self, "single_image_adjust_panel", None)
                if panel is not None:
                    if "temperature_shift" in profile:
                        cur_temp = panel._sliders.get("Temperature")
                        if cur_temp:
                            cur_temp.setValue(cur_temp.value() + int(profile["temperature_shift"]))
                    if "tint_shift" in profile:
                        cur_tint = panel._sliders.get("Tint")
                        if cur_tint:
                            cur_tint.setValue(cur_tint.value() + int(profile["tint_shift"]))

                    # Surface the just-saved profile immediately, so the banner
                    # and its Remove button are available without a reload.
                    # Banner follows the photo in the viewer, not a DPReview
                    # sample the user picked only as a ColorChecker reference.
                    if target_path and os.path.isfile(target_path):
                        self._refresh_camera_profile_banner(panel, target_path)

                camera_name = f"{dlg.make} {dlg.model}".strip() or "Camera"
                iso_label = f" (ISO {dlg.iso})" if dlg.iso else ""
                if hasattr(self, "status_bar") and self.status_bar:
                    self.status_bar.showMessage(
                        f"Calibrated profile saved for {camera_name}{iso_label}", 4000
                    )

        except Exception as exc:
            QMessageBox.warning(
                self, "Calibration Error", f"Failed to launch calibration dialog: {exc}"
            )

    def _reencode_persisted_preview_for_sidecar(self, path: str, adj: dict | None = None) -> None:
        """Re-encode the persisted LibRaw preview tier with just-saved sidecar adjustments.

        The cached preview (see ``UnifiedImageProcessor._process_raw_image``'s
        ``_persist_libraw_preview``) is written once per full decode and
        otherwise sits untouched, so without this a revisit right after
        saving briefly flashes the pre-edit render before the adjusted full
        buffer lands and corrects it -- and would keep doing so on every
        later cache-cold revisit, since a save alone doesn't trigger a full
        re-decode. Runs off the UI thread; a no-op if the unadjusted full
        buffer isn't cached (nothing to re-encode from -- the next full
        decode persists the adjusted preview itself now that
        ``_persist_libraw_preview`` applies the sidecar at decode time too).

        ``adj`` may be None: the worker loads the sidecar itself so navigation
        never blocks the UI on exifread/rawpy as-shot temperature reads.
        """
        from common_image_loader import is_raw_file
        from raw_adjustments import sidecar_adjustments_enabled

        if not sidecar_adjustments_enabled():
            # Browse tiers show original pixels; never bake edits into the
            # persisted preview cache.
            return
        if not path or not is_raw_file(path):
            return
        base = self.image_cache.get_full_image(path)
        if base is None:
            base = getattr(self, "_adjust_preview_base_rgb", None)
        if base is None:
            return

        def _worker(fp=path, rgb=base, a=adj):
            try:
                import cv2

                from image_cache import memory_preview_max_edge
                from raw_adjustments import (
                    apply_adjustments_to_rgb,
                    is_default_adjustments,
                    load_adjustments_for_file,
                )

                a = load_adjustments_for_file(fp) if a is None else dict(a)
                if not is_default_adjustments(a):
                    rgb = apply_adjustments_to_rgb(rgb, a)
                h, w = rgb.shape[:2]
                cap = memory_preview_max_edge()
                if max(h, w) > cap:
                    sc = cap / max(h, w)
                    prev = cv2.resize(
                        rgb,
                        (max(1, int(w * sc)), max(1, int(h * sc))),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    prev = rgb
                ok, buf = cv2.imencode(
                    ".jpg",
                    cv2.cvtColor(prev, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92],
                )
                self.image_cache.put_preview(
                    fp,
                    prev,
                    jpeg_data=buf.tobytes() if ok else None,
                    libraw_rendered=True,
                )
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    def _on_adjust_lens_correction_toggled(self, checked: bool) -> None:
        """Re-decode the edit base with/without lens-profile correction baked in.

        Unlike every other Adjust toggle, this cannot just re-run the preview
        pipeline on the existing _adjust_preview_base_rgb -- the correction is
        applied once at decode time (see raw_lens_correction.py /
        UnifiedImageProcessor.decode_raw_edit_base), so the base itself has to
        be rebuilt. _begin_adjust_editing_session already does exactly that
        (reset base -> _request_adjust_edit_base, which reads the panel's
        current LensCorrectionEnabled value), also invalidating the compare
        cache and PreviewStageCache correctly via the existing base-identity
        reset points.
        """
        path = getattr(self, "current_file_path", None)
        if not path:
            return
        if hasattr(self, "status_bar"):
            self.status_bar.showMessage(
                "Re-decoding RAW with lens correction…" if checked else "Re-decoding RAW…",
                0,
            )
        self._begin_adjust_editing_session(path)

    def _clear_single_image_histogram(self):
        w = getattr(self, "single_image_histogram", None)
        if w is not None:
            w.clear()
        w_adj = getattr(self, "single_image_adjust_histogram", None)
        if w_adj is not None:
            w_adj.clear()
            w.setEnabled(False)
            w.setVisible(False)
            c = getattr(self, "single_view_container", None)
            if c is not None and hasattr(c, "relayout_histogram"):
                c.relayout_histogram()
        self._clear_location_map()
        panel = getattr(self, "single_image_adjust_panel", None)
        if panel is not None:
            panel.hide()

    def _sync_location_map(self) -> None:
        """Debounced refresh -- avoids canceling map load on every histogram tick."""
        timer = getattr(self, "_map_sync_timer", None)
        if timer is not None:
            timer.start(250)
            return
        self._sync_location_map_now()

    def _sync_location_map_now(self):
        """Refresh GPS map when in single-image mode (requires network + GPS)."""
        w = getattr(self, "single_image_location_map", None)
        if w is None:
            return
        if getattr(self, "view_mode", "single") != "single":
            w.clear()
            return
        path = getattr(self, "current_file_path", None)
        pm = getattr(self, "current_pixmap", None)
        if not path or pm is None or pm.isNull():
            w.clear()
            c = getattr(self, "single_view_container", None)
            if c is not None and hasattr(c, "relayout_map"):
                c.relayout_map()
            return
        if self._map_user_hidden or not getattr(self, "_map_overlay_visible", False):
            w.hide()
            c = getattr(self, "single_view_container", None)
            if c is not None and hasattr(c, "relayout_map"):
                c.relayout_map()
            return

        def _apply_online(ok: bool) -> None:
            self._map_online = ok
            if not ok:
                w.clear()
                return
            w.load_for_file(path)
            c = getattr(self, "single_view_container", None)
            if c is not None and hasattr(c, "relayout_map"):
                c.relayout_map()

        if self._map_online is None:
            w.ensure_online(_apply_online)
        else:
            _apply_online(self._map_online)

    def _clear_location_map(self):
        w = getattr(self, "single_image_location_map", None)
        if w is not None:
            w.clear()
            c = getattr(self, "single_view_container", None)
            if c is not None and hasattr(c, "relayout_map"):
                c.relayout_map()

    def _exif_orientation_after_cw90_meta_only(self, o: int) -> int:
        """EXIF Orientation tag after a further 90° clockwise rotation (metadata-only, pixels unchanged)."""
        return exif_orientation_after_cw90(o)

    def _update_single_view_rating_display(self, rating: int):
        """Update the bottom-bar star rating control for the current image."""
        self._set_rating_stars_display(rating)

    def _get_current_image_rating(self) -> int:
        """Current image's star rating (0 = unrated), same exif-cache lookup
        used when a file is first loaded (see _display_image's cached_exif
        read)."""
        path = getattr(self, "current_file_path", None)
        if not path:
            return 0
        cached_exif = self.image_cache.get_exif(path)
        if not cached_exif:
            return 0
        return cached_exif.get('rating', 0) or 0

    def _shortcut_activate_histogram(self) -> None:
        if getattr(self, "view_mode", "") != "single":
            return
        if getattr(self, "_adjust_overlay_visible", False):
            # QShortcut owns H -- arm Heal here (browse histogram is a no-op
            # while Adjust is open; see Local Heal tool).
            if self._shortcut_blocked_by_text_input():
                return
            panel = getattr(self, "single_image_adjust_panel", None)
            if panel is not None and hasattr(panel, "set_dodge_burn_mode"):
                panel.set_dodge_burn_mode("heal")
            return
        if self._shortcut_blocked_by_text_input():
            return
        self._histogram_overlay_visible = not getattr(
            self, "_histogram_overlay_visible", False
        )
        self._histogram_user_hidden = not self._histogram_overlay_visible
        if hasattr(self, "single_image_histogram"):
            pm = getattr(self, "current_pixmap", None)
            if pm is not None and not pm.isNull():
                self.single_image_histogram.setVisible(self._histogram_overlay_visible)
            else:
                self.single_image_histogram.setVisible(False)
        container = getattr(self, "single_view_container", None)
        if container is not None and hasattr(container, "relayout_histogram"):
            container.relayout_histogram()

    def _shortcut_activate_location_map(self) -> None:
        if getattr(self, "view_mode", "") != "single":
            return
        if getattr(self, "_adjust_overlay_visible", False):
            # QShortcut owns M -- toggle the brush Mask overlay here instead
            # (map overlay fights the Adjust panel layout; browse-only there).
            if self._shortcut_blocked_by_text_input():
                return
            panel = getattr(self, "single_image_adjust_panel", None)
            if panel is not None and hasattr(panel, "toggle_dodge_burn_show_mask"):
                panel.toggle_dodge_burn_show_mask()
            return
        if self._shortcut_blocked_by_text_input():
            return
        self._map_overlay_visible = not getattr(self, "_map_overlay_visible", False)
        self._map_user_hidden = not self._map_overlay_visible
        if self._map_overlay_visible and self._map_online is False:
            self._map_online = None
        self._sync_location_map()

    def _title_bar_corner_resize_at(self, pos: QPoint) -> bool:
        """Top corners of the custom title bar stay resizable on frameless windows."""
        tb = getattr(self, "title_bar", None)
        if tb is None or not tb.isVisible():
            return False
        if pos.y() >= tb.height():
            return False
        border = self._frame_resize_border()
        if pos.x() <= border and pos.y() <= border:
            return True
        return bool(pos.x() >= self.width() - border and pos.y() <= border)

    def _title_bar_blocks_resize_at(self, pos: QPoint) -> bool:
        if not (hasattr(self, "title_bar") and self.title_bar is not None):
            return False
        if pos.y() >= self.title_bar.height():
            return False
        return not self._title_bar_corner_resize_at(pos)

    def _is_title_bar_chrome_widget(self, obj) -> bool:
        from PyQt6.QtWidgets import QWidget

        tb = getattr(self, "title_bar", None)
        if tb is None:
            return False
        w = obj if isinstance(obj, QWidget) else None
        while w is not None:
            if w is tb:
                return True
            w = w.parentWidget()
        return False

    def extract_exif_data(self, file_path):
        """Extract HUD-oriented EXIF fields via the shared EXIFExtractor (cached)."""
        empty = {
            "focal_length": None,
            "aperture": None,
            "iso": None,
            "capture_time": None,
        }
        if not file_path:
            return empty
        try:
            from enhanced_raw_processor import EXIFExtractor

            data = EXIFExtractor().extract_exif_data(file_path)
            if not isinstance(data, dict):
                return empty
            return {
                "focal_length": data.get("focal_length"),
                "aperture": data.get("aperture"),
                "iso": data.get("iso"),
                "capture_time": data.get("capture_time"),
            }
        except Exception:
            return empty

    def _metadata_file_label(self, file_path: str) -> str:
        """Folder + filename prefix for the centered EXIF HUD."""
        if not file_path:
            return ""
        filename = os.path.basename(file_path)
        parent = os.path.dirname(os.path.normpath(file_path))
        folder_name = os.path.basename(parent) if parent else ""
        if folder_name and folder_name not in (".", filename):
            return f"{folder_name} / {filename}"
        return filename

    def _gallery_top_metadata_label(self) -> str:
        """Top metadata in gallery mode: burst stack stats or folder name."""
        if (
            getattr(self, "view_mode", "") == "gallery"
            and getattr(self, "_burst_group_view_active", False)
        ):
            return self._burst_group_view_status_label()
        return self._current_folder_label()

    def _apply_top_metadata_text(self, text: str) -> None:
        """Show EXIF / image info in the top bar (Windows: custom title bar; macOS: window title)."""
        display = (text or "").strip()
        if platform.system() == "Darwin":
            # Match Windows: metadata in the title area, not "filename -- RAWviewer".
            self.setWindowTitle(display if display else "RAWviewer")
            if hasattr(self, "top_metadata_bar") and self.top_metadata_bar is not None:
                self.top_metadata_bar.hide()
            return
        if hasattr(self, "title_bar") and self.title_bar is not None:
            self.title_bar.set_metadata(display)
        elif hasattr(self, "top_metadata_bar") and self.top_metadata_bar is not None:
            self.top_metadata_bar.set_metadata(display)
        elif hasattr(self, "status_metadata_label") and self.status_metadata_label is not None:
            self.status_metadata_label.setText(display)
            self.status_metadata_label.setVisible(bool(display))

    def _is_exif_sort_ready(self) -> bool:
        """Check if the EXIF capture-time sort is ready/applied for the current folder."""
        current_gen = getattr(self, "_folder_load_generation", None)
        refinement_applied = (
            getattr(self, "_folder_sort_refinement_applied_token", None) == current_gen
        )
        return refinement_applied or len(getattr(self, "image_files", [])) <= 1
