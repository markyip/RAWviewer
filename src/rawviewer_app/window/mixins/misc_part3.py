
"""misc mixin, part 3 of 4 (22 methods, 849 lines)."""
from __future__ import annotations

import contextlib
import os
import sys
import time

from PyQt6.QtCore import QPoint, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication

from core import theme
from core.config import _env_int, _find_file_index_in_list, _norm_path, safe_print

from .. import lazy
from ..lazy import image_covers_sensor_resolution, np
from ..paths import _require_str_path

lazy.register(sys.modules[__name__])


class MiscPart3:
    def _maybe_arm_raw_edr_after_idle(
        self, file_path: str, rgb_image, width: int, height: int
    ) -> None:
        """Schedule a delayed RAW EDR upgrade once nav settles on ``file_path``.

        Called from ``display_numpy_image`` when EDR would apply but hasn't
        been armed yet: the current frame paints through the fast SDR path,
        and if no further navigation happens within the idle window, this
        arms EDR for the file and re-dispatches the same buffer through the
        EDR converter -- upgrading the on-screen image in place instead of
        re-decoding on every arrow-key press.
        """
        from PyQt6.QtCore import QTimer

        if not file_path:
            return
        norm = _norm_path(file_path)
        gen = int(getattr(self, "_raw_edr_idle_generation", 0)) + 1
        self._raw_edr_idle_generation = gen
        delay_ms = _env_int("RAWVIEWER_RAW_EDR_IDLE_MS", 400, minimum=0)

        def _tick(g=gen, fp=file_path, n=norm, rgb=rgb_image, w=width, h=height) -> None:
            if g != int(getattr(self, "_raw_edr_idle_generation", 0)):
                return
            if _norm_path(getattr(self, "current_file_path", "") or "") != n:
                return
            self._raw_edr_armed_key = n
            if not self._should_use_raw_edr_display(fp):
                return
            exif_for_edr = None
            with contextlib.suppress(Exception):
                exif_for_edr = self.image_cache.get_exif(fp)
            is_half_size_edr = not image_covers_sensor_resolution(
                w, h, exif_for_edr
            )
            self._start_display_numpy_pixmap_worker(
                fp,
                rgb,
                width=w,
                height=h,
                is_half_size=is_half_size_edr,
                fallback_rgb=rgb,
            )

        QTimer.singleShot(delay_ms, _tick)

    def _nef_he_compressed(self, file_path: str) -> bool:
        """True if this NEF is Nikon HE/HE*-compressed.

        Uses the cached EXIF flag when present; otherwise a cheap direct
        maker-note probe so cold opens still block editing before the first
        browse decode has populated the EXIF cache.
        """
        if not file_path or not file_path.lower().endswith(".nef"):
            return False
        try:
            cached_exif = self.image_cache.get_exif(file_path)
        except Exception:
            cached_exif = None
        if cached_exif and cached_exif.get("nef_he_compressed") is True:
            return True
        if cached_exif and cached_exif.get("nef_he_compressed") is False:
            return False
        try:
            from enhanced_raw_processor import _detect_nef_he_compression

            return _detect_nef_he_compression(file_path) is True
        except Exception:
            return False

    def _path_marked_libraw_unsupported(self, file_path: str | None) -> bool:
        if not file_path:
            return False
        try:
            from processing.unified import _LIBRAW_UNSUPPORTED_PATHS

            return os.path.normcase(os.path.abspath(file_path)) in _LIBRAW_UNSUPPORTED_PATHS
        except Exception:
            return False

    def _mark_path_libraw_unsupported(self, file_path: str) -> None:
        try:
            from processing.unified import _mark_libraw_unsupported

            _mark_libraw_unsupported(os.path.normcase(os.path.abspath(file_path)))
        except Exception:
            try:
                from processing.unified import _LIBRAW_UNSUPPORTED_PATHS

                _LIBRAW_UNSUPPORTED_PATHS.add(os.path.normcase(os.path.abspath(file_path)))
            except Exception:
                pass

    def _auto_revert_to_embedded_jpeg_workflow(
        self, file_path: str, *, reason: str = ""
    ) -> bool:
        """Allow embedded pixels for *this* LibRaw-unsupported file only.

        Returns True the first time the path is marked so callers can reload.
        Does **not** flip the app-wide ``use_embedded_jpeg_workflow`` setting:
        leaving RAW (High Quality) mode globally after the first HE/HE* NEF
        silently forced every later ARW/CR3 onto embedded JPEG and starved
        GPU demosaic for the rest of the folder (and the Full+GPU autotest).
        Per-file ``_LIBRAW_UNSUPPORTED_PATHS`` already lets
        ``_suppress_jpeg_interim`` / ``_process_raw_image`` use embeds.
        """
        already = self._path_marked_libraw_unsupported(file_path)
        self._mark_path_libraw_unsupported(file_path)
        msg = (
            f"{os.path.basename(file_path)}: this file's RAW compression isn't "
            f"supported by the current decoder -- showing the embedded preview instead."
        )
        if reason:
            import logging

            logging.getLogger(__name__).info(
                "[WORKFLOW] Embedded fallback for unsupported RAW %s (%s) "
                "(RAW mode kept for other files)",
                os.path.basename(file_path),
                reason,
            )
        if hasattr(self, "status_bar"):
            self.status_bar.showMessage(msg, 8000)
        return not already

    def _show_auto_dismiss_notice(
        self,
        message: str,
        *,
        detail: str = "",
        duration_ms: int = 2000,
    ) -> None:
        """Non-modal centered toast that hides itself after ``duration_ms``."""
        from PyQt6.QtCore import Qt, QTimer
        from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout

        old = getattr(self, "_auto_dismiss_notice", None)
        if old is not None:
            try:
                old.hide()
                old.deleteLater()
            except Exception:
                pass
            self._auto_dismiss_notice = None

        frame = QFrame(self)
        frame.setObjectName("autoDismissNotice")
        frame.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        frame.setStyleSheet(
            f"""
            #autoDismissNotice {{
                background-color: rgba(29, 26, 22, 235);
                border: 1px solid {theme.LINE};
                border-radius: 10px;
            }}
            """
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setSpacing(6)

        msg = QLabel(message)
        msg.setWordWrap(True)
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg.setStyleSheet(
            f"color: {theme.INK}; font-size: 14px; font-weight: 600; background: transparent;"
        )
        layout.addWidget(msg)

        if detail:
            info = QLabel(detail)
            info.setWordWrap(True)
            info.setAlignment(Qt.AlignmentFlag.AlignCenter)
            info.setStyleSheet(
                f"color: {theme.INK_MUTED}; font-size: 12px; background: transparent;"
            )
            layout.addWidget(info)

        frame.adjustSize()
        # Cap width so long wrap text stays readable on large monitors.
        max_w = min(420, max(280, self.width() - 80))
        if frame.width() > max_w:
            frame.setFixedWidth(max_w)
            frame.adjustSize()
        x = max(0, (self.width() - frame.width()) // 2)
        y = max(24, (self.height() - frame.height()) // 3)
        frame.move(x, y)
        frame.show()
        frame.raise_()
        self._auto_dismiss_notice = frame

        def _hide(token=frame) -> None:
            if getattr(self, "_auto_dismiss_notice", None) is token:
                try:
                    token.hide()
                    token.deleteLater()
                except Exception:
                    pass
                self._auto_dismiss_notice = None

        QTimer.singleShot(max(400, int(duration_ms)), _hide)

    def _on_brush_tool_left_image(self) -> None:
        """Disarm Dodge/Burn/Eraser/Heal when the cursor leaves the photo.

        Only fires once the cursor has actually landed on the image at
        least once since arming (see gpu_image_view's
        ``_dodge_burn_confirmed_on_image`` / ``_maybe_emit_brush_tool_left_image``)
        -- arming from the toolbar or a hotkey while the cursor is still
        over the panel/elsewhere does not disarm on the next stray move;
        only a genuine on-image -> off-image transition does.
        """
        panel = getattr(self, "single_image_adjust_panel", None)
        if panel is not None and hasattr(panel, "disarm_dodge_burn") and panel.dodge_burn_mode() is not None:
            panel.disarm_dodge_burn()

    def _adj_has_warp(self, adj: dict | None) -> bool:
        """True when straighten / keystone would change the frame."""
        if not adj:
            return False
        try:
            return any(
                abs(float(adj.get(k, 0.0) or 0.0)) > 1e-4
                for k in (
                    "CropAngle",
                    "PerspectiveVertical",
                    "PerspectiveHorizontal",
                )
            )
        except Exception:
            return False

    def _on_auto_straighten_requested(self) -> None:
        """Auto straighten: Hough / gradient estimate → CropAngle slider."""
        panel = getattr(self, "single_image_adjust_panel", None)
        if panel is None:
            return
        from raw_adjustments.auto_adjust import estimate_straighten_angle

        # Estimate on a display buffer WITHOUT current CropAngle so the result
        # is an absolute straighten angle (not a residual on an already-warped
        # preview). Other adjustments can stay -- they help edges without
        # rotating the frame.
        rgb = None
        base = getattr(self, "_adjust_preview_base_rgb", None)
        if base is not None and hasattr(base, "shape"):
            try:
                from raw_adjustments.pipeline import linear_to_display_uint8

                adj0 = dict(panel.get_adjustments())
                adj0["CropAngle"] = 0.0
                adj0["PerspectiveVertical"] = 0.0
                adj0["PerspectiveHorizontal"] = 0.0
                rgb = linear_to_display_uint8(base, adj0)
            except Exception:
                rgb = None
        if rgb is None:
            last = getattr(self, "_adjust_last_render", None)
            if last is not None and last[0] is not None:
                rgb = last[0]
        if rgb is None:
            if hasattr(self, "status_bar"):
                self.status_bar.showMessage(
                    "Auto straighten needs a decoded image -- try again in a moment",
                    3000,
                )
            return
        angle = estimate_straighten_angle(rgb)
        if angle is None:
            if hasattr(self, "status_bar"):
                self.status_bar.showMessage(
                    "Auto straighten: no dominant horizon/vertical found",
                    3500,
                )
            return
        import numpy as np

        adj = panel.get_adjustments()
        new_angle = float(np.clip(angle, -45.0, 45.0))
        adj["CropAngle"] = new_angle
        panel.set_adjustments(adj)
        panel._emit_preview_and_save()
        if hasattr(self, "status_bar"):
            self.status_bar.showMessage(
                f"Auto straighten {new_angle:+.1f}°",
                3500,
            )

    def _slideshow_interval_ms(self) -> int:
        try:
            return max(500, int(os.environ.get("RAWVIEWER_SLIDESHOW_INTERVAL_MS", "5000")))
        except ValueError:
            return 5000

    def _slideshow_is_active(self) -> bool:
        t = getattr(self, "_slideshow_timer", None)
        return t is not None and t.isActive()

    def _stop_slideshow(self):
        t = getattr(self, "_slideshow_timer", None)
        if t is not None and t.isActive():
            t.stop()
        self._slideshow_scope_paths = None
        self._sync_slideshow_button_icon(False)

    def _slideshow_scope_start_path(self, scope: list[str]) -> str:
        scope_norm = {_norm_path(p): p for p in scope if p}
        cp = getattr(self, "current_file_path", None)
        if cp and _norm_path(cp) in scope_norm:
            return scope_norm[_norm_path(cp)]
        gj = getattr(self, "gallery_justified", None)
        anchor = None
        if gj is not None and hasattr(gj, "get_scroll_anchor_path"):
            try:
                anchor = gj.get_scroll_anchor_path()
            except Exception:
                anchor = None
        if anchor and _norm_path(anchor) in scope_norm:
            return scope_norm[_norm_path(anchor)]
        return scope[0]

    def _slideshow_advance(self, direction: str) -> None:
        scope = getattr(self, "_slideshow_scope_paths", None)
        if scope:
            files = [p for p in scope if p and os.path.isfile(p)]
            if len(files) <= 1:
                return
            cur = getattr(self, "current_file_path", None)
            idx = 0
            if cur:
                for i, p in enumerate(files):
                    if _norm_path(p) == _norm_path(cur):
                        idx = i
                        break
            idx = (idx + 1) % len(files) if direction == "next" else (idx - 1) % len(files)
            self._open_slideshow_image(files[idx])
            return
        if direction == "next":
            self.navigate_to_next_image()
        else:
            self.navigate_to_previous_image()

    def _on_slideshow_tick(self):
        if getattr(self, "view_mode", "single") != "single":
            self._stop_slideshow()
            return
        if not self._slideshow_is_active():
            return
        scope = getattr(self, "_slideshow_scope_paths", None)
        files = (
            [p for p in scope if p and os.path.isfile(p)]
            if scope
            else self._navigation_files()
        )
        cur = getattr(self, "current_file_path", None)
        if not files or not cur:
            self._stop_slideshow()
            return
        self._slideshow_force_fit_next = True
        self._slideshow_advance("next")

    def _start_slideshow_timer(self) -> None:
        if self._slideshow_timer is None:
            self._slideshow_timer = QTimer(self)
            self._slideshow_timer.timeout.connect(self._on_slideshow_tick)
        self._slideshow_timer.start(self._slideshow_interval_ms())
        self._sync_slideshow_button_icon(True)

    def _begin_slideshow_at_path(self, start: str, *, scope_paths: list[str] | None) -> None:
        idx = _find_file_index_in_list(self.image_files, start, default=-1)
        if idx >= 0:
            self.current_file_index = idx
        self.current_file_path = start
        self._slideshow_scope_paths = list(scope_paths) if scope_paths else None
        if getattr(self, "view_mode", "") != "single":
            self.view_mode = "single"
            self._show_single_view()
        else:
            self._open_slideshow_image(start)
            self._sync_filmstrip_to_folder()
        self.fit_to_window = True
        self.current_zoom_level = 1.0
        self.zoom_center_point = None
        if getattr(self, "current_pixmap", None) and not self.current_pixmap.isNull():
            self.scale_image_to_fit()
        self.update_status_bar()
        self._start_slideshow_timer()

    def _on_slideshow_bottom_clicked(self) -> None:
        if getattr(self, "_is_delete_dialog_open", False):
            return
        if self._slideshow_is_active():
            self._stop_slideshow()
            return
        vm = getattr(self, "view_mode", "")
        if vm == "gallery":
            scope = self._gallery_slideshow_scope()
            if not scope:
                return
            bookmark_scope = bool(getattr(self, "_gallery_bookmark_filter_active", False))
            start = self._slideshow_scope_start_path(scope)
            self._begin_slideshow_at_path(
                start,
                scope_paths=scope if bookmark_scope else None,
            )
            return
        if vm != "single" or not self.image_files:
            return
        if getattr(self, "_gallery_bookmark_filter_active", False):
            scope = self._gallery_bookmarked_canonical_paths()
            if scope:
                start = getattr(self, "current_file_path", None) or scope[0]
                if _norm_path(start) not in {_norm_path(p) for p in scope}:
                    start = scope[0]
                self._begin_slideshow_at_path(start, scope_paths=scope)
                return
        self._begin_slideshow_at_path(
            getattr(self, "current_file_path", None) or self.image_files[0],
            scope_paths=None,
        )

    def _format_app_name(self, app_path: str) -> str:
        name = os.path.basename(app_path.rstrip(os.sep))
        if name.lower().endswith(".app"):
            name = name[:-4]
        if name.lower().endswith(".exe"):
            name = name[:-4]
        return name or "selected app"

    def _session_restore_burst_active(self) -> bool:
        return bool(getattr(self, "_session_restore_defer_preload", False))

    def _try_import_gpu_backend(self) -> None:
        """Idle-gated main-thread torch import; retries while the user is busy."""
        retry_ms = _env_int("RAWVIEWER_GPU_IMPORT_RETRY_MS", 2000, minimum=200)
        zoom_quiet_ms = _env_int(
            "RAWVIEWER_GPU_IMPORT_ZOOM_QUIET_MS", 2500, minimum=0
        )
        nav_quiet_ms = _env_int(
            "RAWVIEWER_GPU_IMPORT_NAV_QUIET_MS", 2500, minimum=0
        )
        try:
            from bootstrap_pkg import torch as torch_bootstrap

            if torch_bootstrap.gpu_backend_ready():
                return
        except Exception:
            torch_bootstrap = None  # type: ignore

        gv = getattr(self, "gpu_view", None)
        if gv is not None and not gv.is_fit_mode():
            QTimer.singleShot(retry_ms, self._try_import_gpu_backend)
            return
        if getattr(self, "_pending_zoom", False) or getattr(
            self, "_full_resolution_loading", False
        ):
            QTimer.singleShot(retry_ms, self._try_import_gpu_backend)
            return
        zoom_ts = float(getattr(self, "_user_zoom_interaction_ts", 0.0) or 0.0)
        if zoom_ts > 0.0 and (time.monotonic() - zoom_ts) * 1000.0 < zoom_quiet_ms:
            QTimer.singleShot(retry_ms, self._try_import_gpu_backend)
            return
        nav_ts = float(getattr(self, "_last_navigation_finish_ts", 0.0) or 0.0)
        if nav_ts > 0.0 and (time.monotonic() - nav_ts) * 1000.0 < nav_quiet_ms:
            QTimer.singleShot(retry_ms, self._try_import_gpu_backend)
            return
        if self._should_defer_full_decode():
            QTimer.singleShot(retry_ms, self._try_import_gpu_backend)
            return
        if self._adjust_panel_active():
            QTimer.singleShot(retry_ms, self._try_import_gpu_backend)
            return
        # A view switch (esp. single -> gallery) queues a burst of main-thread
        # work; a multi-second CuPy/torch import landing then freezes the
        # gallery (observed: only a handful of tiles paint, then UI stalls
        # for the full import). Gallery thumbs use CPU paths -- never import
        # the GPU demosaic backend while gallery is active.
        switch_ts = float(getattr(self, "_last_view_mode_switch_ts", 0.0) or 0.0)
        if switch_ts > 0.0 and (time.monotonic() - switch_ts) * 1000.0 < nav_quiet_ms:
            QTimer.singleShot(retry_ms, self._try_import_gpu_backend)
            return
        if getattr(self, "view_mode", "single") == "gallery":
            QTimer.singleShot(retry_ms, self._try_import_gpu_backend)
            return
        if torch_bootstrap is None:
            return

        def _pump() -> None:
            app = QApplication.instance()
            if app is not None:
                app.processEvents()

        try:
            if hasattr(self, "status_bar") and self.status_bar:
                self.status_bar.showMessage("Loading GPU decode backend…", 8000)
        except Exception:
            pass
        torch_bootstrap.import_gpu_backend_on_main_thread(pump=_pump)
        try:
            if hasattr(self, "status_bar") and self.status_bar:
                self.status_bar.clearMessage()
        except Exception:
            pass

    def on_image_processed(self, rgb_image):
        if self._adjust_panel_blocks_display_refresh():
            return
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.hide_loading()

        import logging
        import time
        import traceback
        logger = logging.getLogger(__name__)

        process_start = time.time()

        # Defensive check: ensure we're still valid
        try:
            # Check if object is still valid
            if not hasattr(self, 'current_file_path'):
                logger.warning("[PROCESS] on_image_processed called but object may be invalid")
                return
        except Exception:
            logger.error("[PROCESS] on_image_processed called but object is invalid (access violation risk)")
            return

        # Get current file info for logging
        current_file = getattr(self, 'current_file_path', None)
        current_file_basename = os.path.basename(current_file) if current_file else 'N/A'

        logger.info(f"[PROCESS] ========== on_image_processed() STARTED at {process_start:.3f} ==========")
        logger.info(f"[PROCESS] Current file: {current_file_basename}")

        # Log image info
        if rgb_image is not None:
            image_shape = rgb_image.shape if hasattr(rgb_image, 'shape') else 'unknown'
            image_dtype = rgb_image.dtype if hasattr(rgb_image, 'dtype') else 'unknown'
            logger.info(f"[PROCESS] Image data - shape: {image_shape}, dtype: {image_dtype}")
        else:
            logger.info("[PROCESS] Image data is None")

        # Check if this signal is for the current file (important for rapid navigation)
        if current_file:
            # Try to get the file path from the processor if available
            processor_file = None
            if hasattr(self, 'current_processor') and self.current_processor is not None:
                processor_file = getattr(self.current_processor, 'file_path', None)
                logger.info(f"[PROCESS] Processor file: {os.path.basename(processor_file) if processor_file else 'None'}")

            if processor_file and processor_file != current_file:
                logger.warning(f"[PROCESS] Signal mismatch: processor file ({os.path.basename(processor_file)}) != current file ({current_file_basename}). Skipping processing to avoid displaying wrong image.")
                safe_print(f"[PERF] ⚠️  SKIP PROCESSING: File changed (processor: {os.path.basename(processor_file)}, current: {current_file_basename})")
                # Skip processing - this image is no longer relevant
                return

        try:
            if rgb_image is None:
                # Check if this is a RAW file - QPixmap cannot load RAW files directly
                raw_extensions = {'.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2',
                                 '.pef', '.srw', '.x3f', '.raf', '.3fr', '.fff', '.iiq',
                                 '.cap', '.erf', '.mef', '.mos', '.nrw', '.rwl', '.srf'}
                file_ext = os.path.splitext(_require_str_path(self.current_file_path))[1].lower()
                if file_ext in raw_extensions:
                    # This is expected behavior: RAW processing returned None (possibly cancelled or failed)
                    # QPixmap cannot load RAW files directly, so we skip the fallback
                    # Error handling is done via error_occurred signal, so this is just informational
                    logger.debug(f"RAW file processing returned None, cannot use QPixmap fallback: {os.path.basename(_require_str_path(self.current_file_path))}")
                    return

                # Non-RAW file: load with safe loader (handles TIFF properly)
                pixmap = self._load_pixmap_safe(self.current_file_path)
                if pixmap.isNull():
                    self.show_error("Display Error",
                                    "Could not load image file.")
                    return

                # _load_pixmap_safe now uses QImageReader with setAutoTransform(True) for regular images
                # and ImageOps.exif_transpose for TIFF/RAW JPEG thumbnails, which automatically applies EXIF orientation.
                # Orientation is already applied, so set flag so display_pixmap doesn't apply it again
                self._orientation_already_applied = True
                logger.debug("[PROCESS] Orientation already applied by _load_pixmap_safe (QImageReader/PIL)")

                self.current_pixmap = pixmap
                # Pixmap loaded from disk is full resolution for non-RAW files
                self._is_half_size_displayed = False
                logger.debug("[PROCESS] Setting _is_half_size_displayed=False for loaded pixmap")

                # CRITICAL: Check for zoom restoration FIRST before resetting fit_to_window
                # Preserve-nav / pinch paths may set level + anchor without a prior zoom_center_point.
                has_restore_zoom = (
                    getattr(self, "_preserve_nav_zoom_active", False)
                    or getattr(self, "_pending_zoom_restore", False)
                    or getattr(self, "_restore_zoom_center", None) is not None
                    or getattr(self, "_restore_zoom_level", None) is not None
                )
                has_maintain_zoom = hasattr(self, '_maintain_zoom_on_navigation')
                logger.debug(f"on_image_processed (QPixmap): has_restore_zoom={has_restore_zoom}, has_maintain_zoom={has_maintain_zoom}, current fit_to_window={self.fit_to_window}")

                if has_restore_zoom:
                    # Zoom restoration needed - don't reset fit_to_window
                    logger.info(f"on_image_processed (QPixmap): Restoring zoom state - center={self._restore_zoom_center}, level={getattr(self, '_restore_zoom_level', 'N/A')}")
                    self.fit_to_window = False
                    self.current_zoom_level = self._restore_zoom_level or 1.0
                    c = self._restore_zoom_center
                    self.zoom_center_point = c if c is not None else QPoint(pixmap.width() // 2, pixmap.height() // 2)

                    if hasattr(self, '_restore_start_scroll_x') and hasattr(self, '_restore_start_scroll_y'):
                        if self._restore_start_scroll_x is not None and self._restore_start_scroll_y is not None:
                            self.start_scroll_x = self._restore_start_scroll_x
                            self.start_scroll_y = self._restore_start_scroll_y
                    else:
                        self.start_scroll_x = self.scroll_area.horizontalScrollBar().value()
                        self.start_scroll_y = self.scroll_area.verticalScrollBar().value()

                    self.apply_zoom_and_pan()
                    self._restore_zoom_center = None
                    self._restore_zoom_level = None
                    self._restore_start_scroll_x = None
                    self._restore_start_scroll_y = None
                    # Clean up _maintain_zoom_on_navigation if it exists
                    if has_maintain_zoom:
                        delattr(self, '_maintain_zoom_on_navigation')
                    self._finish_nav_zoom_preserve()
                elif not has_maintain_zoom:
                    # No zoom restoration needed and not maintaining zoom - reset to fit-to-window
                    logger.debug("on_image_processed (QPixmap): No zoom state to restore, resetting to fit-to-window")
                    self.fit_to_window = True
                    self.current_zoom_level = 1.0
                    self.zoom_center_point = None
                    self.scale_image_to_fit()
                else:
                    # Maintaining zoom state from navigation
                    logger.debug(f"on_image_processed (QPixmap): Maintaining zoom state from navigation, fit_to_window={self.fit_to_window}")
                    if self.fit_to_window:
                        self.scale_image_to_fit()
                    else:
                        self.apply_zoom_and_pan()
                    delattr(self, '_maintain_zoom_on_navigation')

                if self.current_file_path:
                    self.scan_folder_for_images(self.current_file_path)
                # Don't pass dimensions - let update_status_bar use original dimensions from cache
                self.update_status_bar()
            else:
                # RAW: successful processing with numpy array
                try:
                    if hasattr(rgb_image, 'shape'):
                        shape = rgb_image.shape
                        height, width = shape[0], shape[1]
                        channels = shape[2] if len(shape) > 2 else 1
                    elif hasattr(rgb_image, 'width') and hasattr(rgb_image, 'height'):
                        height, width = rgb_image.height(), rgb_image.width()
                        channels = 3
                    else:
                        logger.error(f"Invalid image type in on_image_processed: {type(rgb_image)}")
                        return

                    bytes_per_line = channels * width

                    # Ensure the data is contiguous and convert to bytes for PyQt6 compatibility
                    if not rgb_image.flags['C_CONTIGUOUS']:
                        rgb_image = np.ascontiguousarray(rgb_image)

                    # Convert to bytes if needed (PyQt6 compatibility)
                    # Check if this is half_size image (for on-demand full resolution loading)
                    # Detect half_size by checking if dimensions are approximately half of expected full resolution
                    # Typical full resolution: 6000-8000px in largest dimension, half_size: 3000-4000px
                    # Check both width and height to handle both portrait and landscape orientations
                    max_dimension = max(width, height)
                    is_half_size = max_dimension < 3000  # Assume full resolution is typically >=3000px in largest dimension
                    self._is_half_size_displayed = is_half_size
                    import logging
                    logger = logging.getLogger(__name__)
                    if is_half_size:
                        logger.debug(f"Detected half_size image: {width}x{height} (max: {max_dimension}), will load full resolution on zoom")
                    else:
                        logger.info(f"Detected full resolution image: {width}x{height} (max: {max_dimension})")

                    # Check if we need to restore zoom - if so, skip half_size display and load full resolution directly
                    navigate_zoom_restore = (
                        getattr(self, "_preserve_nav_zoom_active", False)
                        or getattr(self, "_pending_zoom_restore", False)
                        or getattr(self, "_restore_zoom_center", None) is not None
                        or getattr(self, "_restore_zoom_level", None) is not None
                    )
                    if navigate_zoom_restore and is_half_size:
                            logger.debug(f"Zoom restoration needed: center={self._restore_zoom_center}, level={getattr(self, '_restore_zoom_level', 'N/A')}, skipping half_size display")
                            if not hasattr(self, '_full_resolution_loading') or not self._full_resolution_loading:
                                # Check if full resolution is already cached
                                cached_full = self.image_cache.get_full_image(self.current_file_path)
                                if cached_full is not None:
                                    cached_max_dim = max(cached_full.shape[1], cached_full.shape[0])
                                    if cached_max_dim >= 3000:
                                        logger.debug("Full resolution image already cached, loading immediately for zoom restoration...")
                                        self._orientation_already_applied = True
                                        self.display_numpy_image(cached_full)
                                        self._is_half_size_displayed = False
                                        self._full_resolution_loading = False
                                        return
                                else:
                                    # Start loading full resolution in background
                                    logger.debug("Starting full resolution load for zoom restoration - skipping half_size display")
                                    self._load_full_resolution_on_demand()
                                    # Store zoom restoration intent
                                    self._pending_zoom_restore = True
                                    self._pending_zoom_center = self._restore_zoom_center
                                    self._pending_zoom_level = self._restore_zoom_level
                                    self._restore_zoom_center = None
                                    self._restore_zoom_level = None
                                    # Don't display half_size image - wait for full resolution
                                    return

                    # Also check if we're already loading full resolution - if so, don't display half_size
                    if hasattr(self, '_full_resolution_loading') and self._full_resolution_loading:
                        logger.debug("Full resolution loading in progress, skipping half_size display")
                        return

                    # Also check if we have pending zoom restore - if so, don't display half_size
                    if hasattr(self, '_pending_zoom_restore') and self._pending_zoom_restore:
                        logger.debug("Pending zoom restore, skipping half_size display")
                        return

                    # Check if we have a cached pixmap first (faster path)
                    # BUT: if this is a full resolution image, don't use cached half_size pixmap
                    cached_pixmap = self.image_cache.get_pixmap(self.current_file_path)
                    if cached_pixmap is not None:
                        # Check if cached pixmap matches the current image size
                        # If current image is full resolution but cached pixmap is small, convert new one
                        # Check if cached pixmap is smaller than current image (use max dimension)
                        cached_max_dim = max(cached_pixmap.width(), cached_pixmap.height())
                        current_max_dim = max(width, height)
                        if not is_half_size and cached_max_dim < current_max_dim:
                            logger.info(f"[PROCESS] Cached pixmap is small ({cached_pixmap.width()}x{cached_pixmap.height()}, max: {cached_max_dim}) but current image is full resolution ({width}x{height}, max: {current_max_dim}), converting new pixmap")
                            # Convert to QPixmap for full resolution image
                            image_data = rgb_image.data.tobytes() if hasattr(
                                rgb_image.data, 'tobytes') else bytes(rgb_image.data)

                            q_format = QImage.Format.Format_RGB888
                            if channels == 1:
                                q_format = QImage.Format.Format_Grayscale8
                            elif channels == 4:
                                q_format = QImage.Format.Format_RGBA8888

                            q_image = QImage(image_data, width, height,
                                             bytes_per_line, q_format)
                            pixmap = QPixmap.fromImage(q_image)
                            self.current_pixmap = pixmap

                            # Cache the new full resolution pixmap (replace the old small one)
                            self.image_cache.put_pixmap(self.current_file_path, pixmap)
                            logger.info(f"[PROCESS] Full resolution pixmap cached: {pixmap.width()}x{pixmap.height()}")
                        else:
                            logger.debug("Using cached QPixmap for faster display")
                            pixmap = cached_pixmap
                            # Check if cached pixmap is half_size
                            pixmap_max_dim = max(pixmap.width(), pixmap.height())
                            if pixmap_max_dim < 3000:
                                logger.debug(f"Cached pixmap is half_size: {pixmap.width()}x{pixmap.height()}, will load full resolution on zoom")
                            self.current_pixmap = pixmap
                    else:
                        # Convert to QPixmap
                        logger.info(f"[PROCESS] Converting full resolution image to QPixmap: {width}x{height}")
                        image_data = rgb_image.data.tobytes() if hasattr(
                            rgb_image.data, 'tobytes') else bytes(rgb_image.data)

                        q_format = QImage.Format.Format_RGB888
                        if channels == 1:
                            q_format = QImage.Format.Format_Grayscale8
                        elif channels == 4:
                            q_format = QImage.Format.Format_RGBA8888

                        q_image = QImage(image_data, width, height,
                                         bytes_per_line, q_format)
                        pixmap = QPixmap.fromImage(q_image)

                        # RAWProcessor already oriented rgb_image; do not rotate again
                        self._orientation_already_applied = True

                        self.current_pixmap = pixmap

                        # Cache the pixmap for future use (faster than numpy->QPixmap conversion)
                        # This ensures next time we load this image, we can skip conversion
                        self.image_cache.put_pixmap(self.current_file_path, pixmap)
                        logger.info(f"[PROCESS] Full resolution pixmap cached: {pixmap.width()}x{pixmap.height()}")

                    # Check if we need to restore zoom - if so, skip half_size display and load full resolution directly
                    navigate_zoom_restore = (
                        getattr(self, "_preserve_nav_zoom_active", False)
                        or getattr(self, "_pending_zoom_restore", False)
                        or getattr(self, "_restore_zoom_center", None) is not None
                        or getattr(self, "_restore_zoom_level", None) is not None
                    )
                    if navigate_zoom_restore and is_half_size:
                            logger.debug("Zoom restoration needed: skipping half_size display (converted from numpy)")
                            if not hasattr(self, '_full_resolution_loading') or not self._full_resolution_loading:
                                # Check if full resolution is already cached
                                cached_full = self.image_cache.get_full_image(self.current_file_path)
                                if cached_full is not None:
                                    cached_max_dim = max(cached_full.shape[1], cached_full.shape[0])
                                    if cached_max_dim >= 3000:
                                        logger.debug("Full resolution image already cached, loading immediately for zoom restoration...")
                                        self._orientation_already_applied = True
                                        self.display_numpy_image(cached_full)
                                        self._is_half_size_displayed = False
                                        self._full_resolution_loading = False
                                        return
                                else:
                                    # Start loading full resolution in background
                                    logger.debug("Starting full resolution load for zoom restoration - skipping half_size display")
                                    self._load_full_resolution_on_demand()
                                    # Store zoom restoration intent
                                    self._pending_zoom_restore = True
                                    self._pending_zoom_center = self._restore_zoom_center
                                    self._pending_zoom_level = self._restore_zoom_level
                                    self._restore_zoom_center = None
                                    self._restore_zoom_level = None
                                    # Don't display half_size image - wait for full resolution
                                    return

                    # Use display_pixmap to handle zoom restoration if needed
                    self.display_pixmap(pixmap)

                    # Preload handled by _start_preloading() → ImageLoadManager
                    self._start_preloading()

                    # Calculate total time from navigation to display
                    # Update status bar to show original dimensions from cache
                    self.update_status_bar()

                    if hasattr(self, '_last_navigation_start'):
                        total_time = time.time() - self._last_navigation_start
                        logger.info(f"RAW image displayed successfully: {width}x{height} (total from navigation: {total_time:.3f}s)")
                        safe_print(f"[PERF] 🖼️  IMAGE DISPLAYED: {width}x{height} (total navigation time: {total_time*1000:.1f}ms)")
                    else:
                        logger.info(f"RAW image displayed successfully: {width}x{height}")
                        safe_print(f"[PERF] 🖼️  IMAGE DISPLAYED: {width}x{height}")
                except Exception as e:
                    import logging
                    import traceback
                    logger = logging.getLogger(__name__)
                    logger.error(f"Error processing RAW image in on_image_processed: {e}", exc_info=True)
                    logger.debug(f"RAW image processing error traceback: {traceback.format_exc()}")
                    # Try to show error to user
                    try:
                        self.show_error("Display Error", f"Error processing RAW image: {str(e)}")
                    except Exception as show_error_ex:
                        logger.error(f"Error showing error message: {show_error_ex}")
        except Exception as e:
            import traceback
            logger.error(f"Critical error in on_image_processed: {e}", exc_info=True)
            logger.debug(f"on_image_processed error traceback: {traceback.format_exc()}")
            error_msg = f"Error displaying image: {str(e)}"
            try:
                self.show_error("Display Error", error_msg)
            except Exception as show_error_ex:
                logger.error(f"Error showing error message: {show_error_ex}")
        finally:
            try:
                self.setFocus()
            except Exception as focus_error:
                logger.warning(f"Error setting focus: {focus_error}")
