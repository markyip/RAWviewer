
"""display mixin, part 4 of 6 (14 methods, 859 lines)."""
from __future__ import annotations

import contextlib
import os
import sys

from PyQt6.QtGui import QImage, QPixmap

from core.config import _norm_path, safe_print
from rawviewer_app.processing import PixmapConverter, RawEdrPixmapConverter

from .. import lazy
from ..lazy import image_covers_sensor_resolution, is_raw_file, np
from ..paths import _require_str_path

lazy.register(sys.modules[__name__])


class DisplayPart4:
    def _try_display_rgb_via_gl_tex(
        self,
        rgb_image,
        *,
        width: int,
        height: int,
        channels: int,
        is_half_size: bool,
    ) -> bool:
        """Phase 2b: paint full-res uint8 RGB via OpenGL drawImage (skip QPixmap).

        Enabled with ``RAWVIEWER_GPU_GL_TEX=1``. Falls back to the pixmap worker
        when the GPU view is missing or the buffer is not contiguous RGB888.
        """
        try:
            from rawviewer_ui.widgets.gpu import gpu_gl_tex_enabled

            if not gpu_gl_tex_enabled():
                return False
            gv = getattr(self, "gpu_view", None)
            if gv is None or channels != 3:
                return False
            import numpy as np

            arr = np.asarray(rgb_image)
            if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] < 3:
                return False
            arr = np.ascontiguousarray(arr)
            cur_path = getattr(self, "current_file_path", None) or ""
            preserve = bool(
                getattr(self, "_preserve_nav_zoom_active", False)
                or getattr(self, "_maintain_zoom_on_navigation", False)
                or not self._single_view_is_fit_mode()
            )
            gv.set_file_path(cur_path)
            if not gv.set_rgb_numpy(arr, preserve_view=preserve if preserve else None):
                return False

            # Bookkeeping mirrors display_pixmap without building a QPixmap first.
            self._orientation_already_applied = True
            self._is_half_size_displayed = bool(is_half_size)
            prev_displayed = getattr(self, "_displayed_content_path", None)
            self._displayed_content_path = cur_path
            # Lightweight 1x1 placeholder pixmap so null-checks don't clear state.
            # Real pixels live in the GL RGB item.
            if getattr(self, "current_pixmap", None) is None or (
                hasattr(self.current_pixmap, "isNull") and self.current_pixmap.isNull()
            ):
                from PyQt6.QtGui import QPixmap as _QP

                self.current_pixmap = _QP(1, 1)
            pm_max = max(int(width), int(height))
            if _norm_path(prev_displayed or "") != _norm_path(cur_path or ""):
                self._manager_displayed_max_dim = pm_max
            else:
                self._manager_displayed_max_dim = max(
                    getattr(self, "_manager_displayed_max_dim", 0), pm_max
                )
            with contextlib.suppress(Exception):
                self._sync_single_image_histogram()
            with contextlib.suppress(Exception):
                self._on_single_view_content_displayed()
            with contextlib.suppress(Exception):
                self._sync_gpu_single_view_visibility()
            import logging

            logging.getLogger(__name__).info(
                "[DISPLAY] GL-tex RGB path %dx%d (skipped QPixmap converter)",
                width,
                height,
            )
            return True
        except Exception:
            return False

    def _try_display_device_rgb_via_cuda_gl(
        self,
        device_rgb,
        *,
        width: int,
        height: int,
        channels: int,
        is_half_size: bool,
    ) -> bool:
        """Phase 2c: CUDA DeviceRgb → GL texture (no D2H for display)."""
        try:
            from processing.raw_decode.gpu_bridge import DeviceRgb, cuda_gl_enabled
            from rawviewer_ui.widgets.gpu import gpu_cuda_gl_enabled

            if not (cuda_gl_enabled() or gpu_cuda_gl_enabled()):
                return False
            if not isinstance(device_rgb, DeviceRgb):
                return False
            gv = getattr(self, "gpu_view", None)
            if gv is None or channels != 3:
                return False
            if not gv.is_opengl_viewport():
                return False
            cur_path = getattr(self, "current_file_path", None) or ""
            preserve = bool(
                getattr(self, "_preserve_nav_zoom_active", False)
                or getattr(self, "_maintain_zoom_on_navigation", False)
                or not self._single_view_is_fit_mode()
            )
            gv.set_file_path(cur_path)
            if not gv.set_device_rgb(
                device_rgb, preserve_view=preserve if preserve else None
            ):
                return False

            self._orientation_already_applied = True
            self._is_half_size_displayed = bool(is_half_size)
            prev_displayed = getattr(self, "_displayed_content_path", None)
            self._displayed_content_path = cur_path
            if getattr(self, "current_pixmap", None) is None or (
                hasattr(self.current_pixmap, "isNull") and self.current_pixmap.isNull()
            ):
                from PyQt6.QtGui import QPixmap as _QP

                self.current_pixmap = _QP(1, 1)
            pm_max = max(int(width), int(height))
            if _norm_path(prev_displayed or "") != _norm_path(cur_path or ""):
                self._manager_displayed_max_dim = pm_max
            else:
                self._manager_displayed_max_dim = max(
                    getattr(self, "_manager_displayed_max_dim", 0), pm_max
                )
            with contextlib.suppress(Exception):
                self._sync_single_image_histogram()
            with contextlib.suppress(Exception):
                self._on_single_view_content_displayed()
            with contextlib.suppress(Exception):
                self._sync_gpu_single_view_visibility()
            import logging

            logging.getLogger(__name__).info(
                "[DISPLAY] CUDA-GL DeviceRgb path %dx%d (skipped D2H + QPixmap)",
                width,
                height,
            )
            return True
        except Exception:
            return False

    def _start_display_numpy_pixmap_worker(
        self,
        file_path: str,
        rgb_image,
        *,
        width: int,
        height: int,
        is_half_size: bool,
        fallback_rgb=None,
    ) -> None:
        """Convert a full-resolution numpy buffer to QPixmap off the UI thread."""
        import logging

        logger = logging.getLogger(__name__)
        pending = getattr(self, "_pending_display_pixmap_context", None) or {}
        if (
            _norm_path(pending.get("file_path", "")) == _norm_path(file_path)
            and int(pending.get("width") or 0) == int(width)
            and int(pending.get("height") or 0) == int(height)
            and getattr(self, "_display_pixmap_converter", None) is not None
        ):
            logger.debug(
                "[DISPLAY] Skipping duplicate QPixmap conversion for %s (%dx%d)",
                os.path.basename(file_path),
                width,
                height,
            )
            return
        self._cancel_display_pixmap_converter()
        self._pending_display_pixmap_context = {
            "file_path": file_path,
            "width": width,
            "height": height,
            "is_half_size": is_half_size,
            "orientation_already_applied": getattr(
                self, "_orientation_already_applied", False
            ),
            "use_edr": self._should_use_raw_edr_display(file_path),
            "fallback_rgb": fallback_rgb,
        }
        ctx = self._pending_display_pixmap_context
        if ctx.get("use_edr"):
            exif_data = None
            with contextlib.suppress(Exception):
                exif_data = self.image_cache.get_exif(file_path)
            max_edge = max(width, height) if is_half_size else 0
            converter = RawEdrPixmapConverter(
                file_path,
                max_edge=max_edge,
                exif_data=exif_data,
            )
            log_label = "RAW EDR"
        else:
            converter = PixmapConverter(file_path, rgb_image, None)
            log_label = "QPixmap"
        converter.pixmap_ready.connect(self._on_display_numpy_pixmap_ready)
        # _display_pixmap_converter is a single slot: when full-res buffers
        # arrive in rapid succession (fast RAW decode + LibRaw-first mode),
        # assigning the new converter would drop the last reference to the
        # previous, still-running QThread -- Python GC then destroys a live
        # QThread and Qt aborts the process ("QThread: Destroyed while
        # thread '' is still running"). Keep every converter alive in a set
        # until its own finished signal fires.
        live = getattr(self, "_live_display_pixmap_converters", None)
        if live is None:
            live = self._live_display_pixmap_converters = set()
        live.add(converter)
        converter.finished.connect(
            lambda c=converter: self._live_display_pixmap_converters.discard(c)
        )
        self._display_pixmap_converter = converter
        logger.info(
            "[DISPLAY] Queued background %s conversion for %s (%dx%d)",
            log_label,
            os.path.basename(file_path),
            width,
            height,
        )
        converter.start()

    def _on_display_numpy_pixmap_ready(self, file_path: str, pixmap) -> None:
        """Finish display_numpy_image on the UI thread after background conversion."""
        import logging
        import time

        logger = logging.getLogger(__name__)
        ctx = getattr(self, "_pending_display_pixmap_context", None) or {}
        if _norm_path(ctx.get("file_path", "")) != _norm_path(file_path):
            return
        if _norm_path(file_path) != _norm_path(getattr(self, "current_file_path", None)):
            return
        if (
            getattr(self, "_raw_recovery_active", False)
            and is_raw_file(file_path)
        ):
            logger.info(
                "[DISPLAY] Skipping background pixmap -- RAW recovery preview active"
            )
            self._pending_display_pixmap_context = None
            self._display_pixmap_converter = None
            return

        width = int(ctx.get("width") or (pixmap.width() if pixmap else 0))
        height = int(ctx.get("height") or (pixmap.height() if pixmap else 0))
        is_half_size = bool(ctx.get("is_half_size"))
        self._pending_display_pixmap_context = None
        self._display_pixmap_converter = None

        if pixmap is None or pixmap.isNull():
            logger.warning("[DISPLAY] Background QPixmap conversion returned empty pixmap")
            if ctx.get("use_edr"):
                logger.info("[DISPLAY] RAW EDR failed; falling back to SDR display path")
                pending_rgb = ctx.get("fallback_rgb")
                if pending_rgb is not None:
                    self._pending_display_pixmap_context = None
                    self._display_pixmap_converter = None
                    self._raw_edr_fallback_active = True
                    try:
                        self.display_numpy_image(pending_rgb)
                    finally:
                        self._raw_edr_fallback_active = False
            return

        if ctx.get("use_edr"):
            self._display_edr_mode = "raw"
            logger.info("[DISPLAY] RAW EDR pixmap ready (%dx%d)", width, height)
        else:
            self._display_edr_mode = None

        if hasattr(self, "current_file_path") and self.current_file_path:
            orientation_already_applied = bool(
                ctx.get("orientation_already_applied")
            )
            if not orientation_already_applied:
                orientation = self.get_orientation_from_exif(self.current_file_path)
                if orientation != 1:
                    logger.info(f"[DISPLAY] Applying orientation correction: {orientation}")
                    pixmap = self.apply_orientation_to_pixmap(pixmap, orientation)

        self._is_half_size_displayed = is_half_size
        if is_half_size:
            logger.debug(
                f"[DISPLAY] Below sensor resolution on screen ({width}x{height})"
            )
        else:
            self._smooth_zoom_full_request_sent = False
            logger.info(
                f"[DISPLAY] Sensor-resolution buffer on screen ({width}x{height})"
            )

        if hasattr(self, "current_file_path") and self.current_file_path and not is_half_size:
            if not getattr(self, "_raw_recovery_skip_display_cache", False):
                logger.info("[DISPLAY] Caching FULL resolution pixmap")
                self.image_cache.put_pixmap(self.current_file_path, pixmap)
            else:
                logger.info("[DISPLAY] Skipping cache (RAW recovery preview)")
        else:
            logger.info("[DISPLAY] Skipping cache for thumbnail/half_size image")

        if not is_half_size and hasattr(self, "_pending_zoom_restore") and self._pending_zoom_restore:
            logger.info("[DISPLAY] Full resolution loaded, will restore zoom after display")

        pixmap_display_start = time.time()
        logger.info("[DISPLAY] Calling display_pixmap()")
        self.display_pixmap(pixmap)
        pixmap_display_time = time.time() - pixmap_display_start

        if not is_half_size and hasattr(self, "_pending_zoom_restore") and self._pending_zoom_restore:
            logger.info("[DISPLAY] Full resolution displayed, restoring zoom state")
            self._pending_zoom_restore = False
            self.fit_to_window = False
            pending_zoom_level = getattr(self, "_pending_zoom_level", None)
            pending_zoom_center = getattr(self, "_pending_zoom_center", None)
            self.current_zoom_level = pending_zoom_level or 1.0
            self.zoom_center_point = pending_zoom_center
            if hasattr(self, "_restore_start_scroll_x") and hasattr(self, "_restore_start_scroll_y"):
                self.start_scroll_x = self._restore_start_scroll_x
                self.start_scroll_y = self._restore_start_scroll_y
            self.apply_zoom_and_pan()
            self.update_status_bar()
            if hasattr(self, "_pending_zoom_center"):
                delattr(self, "_pending_zoom_center")
            if hasattr(self, "_pending_zoom_level"):
                delattr(self, "_pending_zoom_level")
            logger.info("[DISPLAY] Zoom state restored after full resolution display")
            self._finish_nav_zoom_preserve()

        logger.info(
            "[DISPLAY] Background QPixmap display complete: %dx%d (pixmap: %.3fs)",
            width,
            height,
            pixmap_display_time,
        )

    def display_numpy_image(self, rgb_image):
        """Display a numpy image array."""
        if self._adjust_panel_blocks_display_refresh() and not getattr(
            self, "_adjust_preview_painting", False
        ):
            return
        if hasattr(self, 'current_file_path') and self.is_animated_image(_require_str_path(self.current_file_path)):
            return
        # RAW workflow, JPEG-interim suppression: last line of defense against
        # sub-display-tier camera-JPEG buffers reaching the screen through any
        # of the many paint paths. LibRaw-rendered tiers (half decode, sensor
        # decode, persisted LibRaw preview) are always >= the memory preview
        # cap; 512px-class buffers here are embedded-JPEG pixels.
        try:
            cur = getattr(self, "current_file_path", None)
            if (
                rgb_image is not None
                and hasattr(rgb_image, "shape")
                and max(rgb_image.shape[0], rgb_image.shape[1]) < 1024
                and self._suppress_jpeg_interim(cur)
                and not self.image_cache.is_libraw_preview(cur)
            ):
                import logging as _logging

                _logging.getLogger(__name__).debug(
                    "[TIERSWAP] refusing %sx%s JPEG interim for %s (RAW workflow)",
                    rgb_image.shape[1],
                    rgb_image.shape[0],
                    os.path.basename(cur or ""),
                )
                # HE/HE* (and other LibRaw-unsupported RAW): the small embed may
                # be the only pixels we will ever get. Mark the path unsupported
                # so this buffer can paint without leaving RAW mode for other files.
                if cur and (
                    self._nef_he_compressed(cur)
                    or self._path_marked_libraw_unsupported(cur)
                ):
                    self._auto_revert_to_embedded_jpeg_workflow(
                        cur,
                        reason="libraw-unsupported RAW (e.g. Nikon HE/HE*)",
                    )
                    # Fall through and paint the embed below.
                else:
                    # A refused interim can be the ONLY delivery this file's "full"
                    # stage ever produces (e.g. under a RAW<->JPEG workflow toggle
                    # plus rapid navigation load-shedding on a slow volume/raw_limit=1
                    # path). With nothing else queued, the view sticks on whatever
                    # was on screen before forever -- the TTFR watchdog only logs a
                    # warning, it never retries. Re-request the real sensor decode
                    # once per file so a refused tier always gets a follow-up.
                    norm_cur = _norm_path(cur or "")
                    if norm_cur and norm_cur not in self._raw_workflow_refused_tier_paths:
                        self._raw_workflow_refused_tier_paths.add(norm_cur)
                        # force+bypass: a gallery 512px entry may already sit in
                        # full_image_cache and (with bogus 160x120 EXIF) look like
                        # it "covers sensor", short-circuiting a non-forced re-queue.
                        self._request_full_res_decode(
                            cur, priority_current=True, force=True
                        )
                    # Do NOT clear the viewport here. Clearing showed the empty-state
                    # "No image loaded" placeholder while LibRaw was still decoding,
                    # and left RAW-mode navigation on a blank page (HUD updated,
                    # pixels gone). Keep the previous frame until the real decode
                    # paints -- same contract as arrow-key nav elsewhere.
                    return
        except Exception:
            pass

        import logging
        import time
        logger = logging.getLogger(__name__)
        display_start = time.time()

        try:
            if hasattr(rgb_image, 'shape'):
                shape = rgb_image.shape
                height, width = shape[0], shape[1]
                channels = shape[2] if len(shape) > 2 else 1
            elif hasattr(rgb_image, 'width') and hasattr(rgb_image, 'height'):
                height, width = rgb_image.height(), rgb_image.width()
                channels = 3
            else:
                logger.error(f"Invalid image type in display_numpy_image: {type(rgb_image)}")
                return

            path = getattr(self, "current_file_path", None)
            recovery_preview = getattr(self, "_raw_recovery_skip_display_cache", False)

            max_dim = max(height, width)

            # Check for resolution downgrade for the CURRENT file
            if hasattr(self, 'current_file_path') and self.current_file_path:
                norm_current = _norm_path(self.current_file_path)
                if (
                    not recovery_preview
                    and not getattr(self, "_skip_resolution_downgrade_check", False)
                    and hasattr(self, "_file_max_dim_map")
                    and norm_current in self._file_max_dim_map
                    and max_dim < self._file_max_dim_map[norm_current] * 0.9
                    and self._single_view_pixels_on_screen(self.current_file_path)
                ):
                    logger.info(
                        f"[DISPLAY] Ignoring resolution downgrade for "
                        f"{os.path.basename(_require_str_path(self.current_file_path))}: {width}x{height} < cached max"
                    )
                    return

                if not hasattr(self, "_file_max_dim_map"):
                    self._file_max_dim_map = {}
                self._file_max_dim_map[norm_current] = max(
                    self._file_max_dim_map.get(norm_current, 0), max_dim
                )

            logger.info(f"[DISPLAY] ========== display_numpy_image() STARTED at {display_start:.3f} ==========")
            logger.info(f"[DISPLAY] Image dimensions: {width}x{height}, channels: {channels}")

            if (
                self._should_use_raw_edr_display(path)
                and hasattr(self, "current_file_path")
                and self.current_file_path
            ):
                exif_for_edr = None
                with contextlib.suppress(Exception):
                    exif_for_edr = self.image_cache.get_exif(self.current_file_path)
                is_half_size_edr = not image_covers_sensor_resolution(
                    width, height, exif_for_edr
                )
                self._start_display_numpy_pixmap_worker(
                    self.current_file_path,
                    rgb_image,
                    width=width,
                    height=height,
                    is_half_size=is_half_size_edr,
                    fallback_rgb=rgb_image,
                )
                return

            if (
                self._raw_edr_eligible_ignoring_idle(path)
                and hasattr(self, "current_file_path")
                and self.current_file_path
            ):
                # EDR applies to this file but hasn't settled yet: paint the
                # fast SDR buffer now (fall through below) and upgrade to
                # EDR in place if the user is still here after the idle
                # window -- see _should_use_raw_edr_display's docstring.
                self._maybe_arm_raw_edr_after_idle(
                    self.current_file_path, rgb_image, width, height
                )

            self._display_edr_mode = None

            # Collapse redundant full-resolution re-displays. Several session-
            # restore and cached-full fast-paths (the restored-zoom eager
            # decode, the deferred post-restore full decode, and assorted
            # "display cached_full" callers) feed the identical sensor-res
            # buffer back through here after it is already painted -- observed
            # 3x for a single session restore, each re-running a ~66MP QPixmap
            # conversion of a buffer that is already on screen. If an equal-or-
            # higher-res buffer for this exact file is already painted, there
            # is nothing to upgrade; skip the convert+display entirely. Gated
            # with the same escape hatches as the on_manager_image_ready
            # redundancy guard (_skip_resolution_downgrade_check for workflow/
            # recovery repaints, _loading_from_gallery for blur->sharp
            # upgrades) plus an adjust/edit-preview exclusion, and placed after
            # the EDR paths above so EDR upgrades are never suppressed.
            if (
                not recovery_preview
                and path
                and max(width, height) >= 3000
                and not getattr(self, "_skip_resolution_downgrade_check", False)
                and not getattr(self, "_loading_from_gallery", False)
                and not getattr(self, "_adjust_preview_painting", False)
                and not self._adjust_panel_active()
                and self._already_displaying_buffer_for_path(path, rgb_image)
            ):
                logger.info(
                    "[DISPLAY] Skipping redundant full-res re-display for %s "
                    "(%dx%d already on screen)",
                    os.path.basename(path),
                    width,
                    height,
                )
                return

            max_dimension = max(width, height)
            is_full_resolution = max_dimension >= 3000

            # Check if we have a cached pixmap first (fastest path)
            if hasattr(self, 'current_file_path') and self.current_file_path:
                logger.info(f"[DISPLAY] Full resolution check: {is_full_resolution} (max dimension: {max_dimension}, {width}x{height})")

                if is_full_resolution and not recovery_preview:
                    logger.info(f"[DISPLAY] Full resolution image ({width}x{height}), converting and displaying")
                elif recovery_preview:
                    logger.info(
                        "[DISPLAY] RAW recovery preview (%dx%d), bypassing cached pixmap",
                        width,
                        height,
                    )
                else:
                    logger.info("[DISPLAY] Checking for cached pixmap")
                    cached_pixmap = self.image_cache.get_pixmap(self.current_file_path)
                    if cached_pixmap is not None:
                        input_aspect = width / height
                        cached_aspect = cached_pixmap.width() / cached_pixmap.height()
                        aspect_mismatch = (input_aspect > 1.0 and cached_aspect < 1.0) or (input_aspect < 1.0 and cached_aspect > 1.0)

                        if aspect_mismatch:
                            logger.warning(f"[DISPLAY] Aspect ratio mismatch (input={input_aspect:.2f}, cached={cached_aspect:.2f}), ignoring cache")
                            cached_pixmap = None

                            logger.info(f"[DISPLAY] Using cached pixmap for {width}x{height} image")

                            # Cached pixmaps are already oriented
                            self._orientation_already_applied = True

                            self.display_pixmap(cached_pixmap)
                            return

            exif_for_flag = None
            if hasattr(self, "current_file_path") and self.current_file_path:
                with contextlib.suppress(Exception):
                    exif_for_flag = self.image_cache.get_exif(self.current_file_path)
            is_half_size = not image_covers_sensor_resolution(
                width, height, exif_for_flag
            )

            if (
                is_full_resolution
                and not recovery_preview
                and hasattr(self, "current_file_path")
                and self.current_file_path
            ):
                try:
                    from processing.raw_decode.gpu_bridge import DeviceRgb as _DeviceRgb

                    if isinstance(rgb_image, _DeviceRgb):
                        if self._try_display_device_rgb_via_cuda_gl(
                            rgb_image,
                            width=width,
                            height=height,
                            channels=channels,
                            is_half_size=is_half_size,
                        ):
                            return
                        # Interop failed: fall through with host numpy.
                        rgb_image = rgb_image.to_numpy()
                except Exception:
                    pass
                if self._try_display_rgb_via_gl_tex(
                    rgb_image,
                    width=width,
                    height=height,
                    channels=channels,
                    is_half_size=is_half_size,
                ):
                    return
                self._start_display_numpy_pixmap_worker(
                    self.current_file_path,
                    rgb_image,
                    width=width,
                    height=height,
                    is_half_size=is_half_size,
                )
                return

            # Convert to QPixmap (preview / sub-full-res path stays on UI thread)
            if rgb_image.dtype == np.uint16:
                rgb_image = (rgb_image / 257.0).astype(np.uint8)
            elif np.issubdtype(rgb_image.dtype, np.floating):
                from raw_adjustments.tone_recovery import _encode_srgb8
                rgb_image = _encode_srgb8(
                    np.clip(rgb_image.astype(np.float32), 0.0, None)
                )
            elif rgb_image.dtype != np.uint8:
                rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

            bytes_per_line = channels * width
            logger.info(f"[DISPLAY] Converting numpy array to QPixmap - bytes_per_line: {bytes_per_line}")

            if not rgb_image.flags['C_CONTIGUOUS']:
                logger.info("[DISPLAY] Making array contiguous")
                rgb_image = np.ascontiguousarray(rgb_image)

            conversion_start = time.time()

            q_format = QImage.Format.Format_RGB888
            if channels == 1:
                q_format = QImage.Format.Format_Grayscale8
            elif channels == 4:
                q_format = QImage.Format.Format_RGBA8888

            # Zero-copy QImage from numpy array
            q_image = QImage(rgb_image.data, width, height, bytes_per_line, q_format)
            q_image.ndarr = rgb_image  # Keep numpy array alive
            logger.info("[DISPLAY] QImage created, creating QPixmap")
            pixmap = QPixmap.fromImage(q_image)
            conversion_time = time.time() - conversion_start
            logger.info(f"[DISPLAY] QImage/QPixmap conversion completed in {conversion_time:.3f}s")

            # CRITICAL: Apply orientation correction only if not already applied
            # Images from UnifiedImageProcessor (via ImageLoadManager) already have orientation applied
            # Images from old RAWProcessor path need orientation correction here
            if hasattr(self, 'current_file_path') and self.current_file_path:
                # Check if orientation was already applied (e.g., by UnifiedImageProcessor)
                orientation_already_applied = getattr(self, '_orientation_already_applied', False)
                # Console log: _orientation_already_applied value in display_numpy_image
                # safe_print(f"[ORIENTATION] display_numpy_image: _orientation_already_applied = {orientation_already_applied}")
                if not orientation_already_applied:
                    orientation = self.get_orientation_from_exif(self.current_file_path)
                    if orientation != 1:
                        logger.info(f"[DISPLAY] Applying orientation correction: {orientation}")
                        pixmap = self.apply_orientation_to_pixmap(pixmap, orientation)
                    else:
                        logger.debug("[DISPLAY] Orientation is 1 (normal), no correction needed")
                else:
                    logger.debug("[DISPLAY] Orientation already applied by processor, skipping")
                    # safe_print(f"[ORIENTATION] display_numpy_image: Skipping orientation correction (already applied)")
            self._is_half_size_displayed = is_half_size
            if is_half_size:
                logger.debug(
                    f"[DISPLAY] Below sensor resolution on screen ({width}x{height})"
                )
            else:
                self._smooth_zoom_full_request_sent = False
                logger.info(
                    f"[DISPLAY] Sensor-resolution buffer on screen ({width}x{height})"
                )

            # Cache the pixmap for future use (after orientation correction)
            # ONLY cache if it's the full resolution image, NOT a thumbnail/half_size
            if hasattr(self, "current_file_path") and self.current_file_path and not is_half_size:
                if not getattr(self, "_raw_recovery_skip_display_cache", False):
                    logger.info("[DISPLAY] Caching FULL resolution pixmap")
                    self.image_cache.put_pixmap(self.current_file_path, pixmap)
                else:
                    logger.info("[DISPLAY] Skipping cache (RAW recovery preview)")
            else:
                logger.info("[DISPLAY] Skipping cache for thumbnail/half_size image")

            # Check if we need to restore zoom after displaying full resolution
            # This handles the case when navigating from a zoomed image
            if not is_half_size and hasattr(self, '_pending_zoom_restore') and self._pending_zoom_restore:
                logger.info("[DISPLAY] Full resolution loaded, will restore zoom after display")

            pixmap_display_start = time.time()
            logger.info("[DISPLAY] Calling display_pixmap()")
            self.display_pixmap(pixmap)
            pixmap_display_time = time.time() - pixmap_display_start

            # After displaying full resolution, check if we need to restore zoom
            if not is_half_size and hasattr(self, '_pending_zoom_restore') and self._pending_zoom_restore:
                logger.info("[DISPLAY] Full resolution displayed, restoring zoom state")
                self._pending_zoom_restore = False
                self.fit_to_window = False
                # Use getattr to safely get pending zoom parameters
                pending_zoom_level = getattr(self, '_pending_zoom_level', None)
                pending_zoom_center = getattr(self, '_pending_zoom_center', None)
                self.current_zoom_level = pending_zoom_level or 1.0
                self.zoom_center_point = pending_zoom_center
                # Restore scroll position if available
                if hasattr(self, '_restore_start_scroll_x') and hasattr(self, '_restore_start_scroll_y'):
                    self.start_scroll_x = self._restore_start_scroll_x
                    self.start_scroll_y = self._restore_start_scroll_y
                # Must use fractional zoom scaling (pinch/trackpad/Ctrl-wheel); zoom_to_point() pins to native pixmap.
                self.apply_zoom_and_pan()
                self.update_status_bar()
                # Clean up
                if hasattr(self, '_pending_zoom_center'):
                    delattr(self, '_pending_zoom_center')
                if hasattr(self, '_pending_zoom_level'):
                    delattr(self, '_pending_zoom_level')
                logger.info("[DISPLAY] Zoom state restored after full resolution display")
                self._finish_nav_zoom_preserve()
            total_time = time.time() - display_start
            logger.info(f"[DISPLAY] RAW image displayed successfully: {width}x{height} (pixmap display: {pixmap_display_time:.3f}s, total: {total_time:.3f}s)")
            safe_print(f"[PERF] 🖼️  DISPLAY COMPLETE: {width}x{height} (pixmap: {pixmap_display_time*1000:.1f}ms, total: {total_time*1000:.1f}ms)")

        except Exception as e:
            total_time = time.time() - display_start
            logger.error(f"[DISPLAY] ========== ERROR in display_numpy_image (at {time.time():.3f}, duration: {total_time:.3f}s) ==========")
            logger.error(f"[DISPLAY] Exception type: {type(e).__name__}, message: {e}", exc_info=True)
            error_msg = f"Error displaying numpy image: {str(e)}"
            self.show_error("Display Error", error_msg)

    def _single_view_has_display_image(self) -> bool:
        pm = getattr(self, "current_pixmap", None)
        if pm is not None and not pm.isNull():
            return True
        gv = getattr(self, "gpu_view", None)
        return bool(gv is not None and gv.has_pixmap())

    def _display_brightness_hwnd(self) -> int | None:
        """HWND of this window so Windows can raise the nearest monitor."""
        try:
            wid = int(self.effectiveWinId())
            return wid if wid else None
        except Exception:
            return None

    def _uses_gpu_single_view(self) -> bool:
        gv = getattr(self, "gpu_view", None)
        if gv is None:
            return False
        path = getattr(self, "current_file_path", None)
        return not (path and self.is_animated_image(_require_str_path(path)))

    def _unrotated_display_size(self) -> tuple[int, int] | None:
        """Display size in pre-visual-rotation space (matches edit/sample buffers)."""
        disp = self._adjust_display_pixel_size()
        if disp is None:
            return None
        deg = int(self._get_visual_rotation_degrees() or 0) % 360
        dw, dh = disp
        if deg in (90, 270):
            return dh, dw
        return dw, dh

    def _display_point_to_unrotated(self, x: float, y: float) -> tuple[float, float]:
        """Map a click on the visually-rotated pixmap back to buffer space."""
        deg = int(self._get_visual_rotation_degrees() or 0) % 360
        disp = self._adjust_display_pixel_size()
        if disp is None or deg == 0:
            return float(x), float(y)
        dw, dh = disp
        if deg == 90:
            return float(dh - 1 - y), float(x)
        if deg == 180:
            return float(dw - 1 - x), float(dh - 1 - y)
        if deg == 270:
            return float(y), float(dw - 1 - x)
        return float(x), float(y)

    def _widgets_with_main_mouse_event_filter(self) -> list:
        """Widgets where MainWindow.eventFilter handles mouse/gesture."""
        out: list = []
        sa = getattr(self, "scroll_area", None)
        if sa is not None:
            out.append(sa)
            out.append(sa.viewport())
        il = getattr(self, "image_label", None)
        if il is not None:
            out.append(il)
        gv = getattr(self, "gpu_view", None)
        if gv is not None:
            out.append(gv)
            out.append(gv.viewport())
        tb = getattr(self, "title_bar", None)
        if tb is not None:
            out.append(tb)
        svc = getattr(self, "single_view_container", None)
        if svc is not None:
            out.append(svc)
        drop_targets = getattr(self, "_file_drop_targets", None)
        if drop_targets:
            out.extend(list(drop_targets))
        seen = set()
        deduped: list = []
        for w in out:
            key = int(w.winId()) if hasattr(w, "winId") else id(w)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(w)
        return deduped

    def _displayed_base_pixmap_size(self, file_path: str | None = None) -> tuple[int, int] | None:
        """Un-rotated buffer size already on screen for ``file_path`` (if any)."""
        path = file_path or getattr(self, "current_file_path", None)
        if not path:
            return None
        cur = getattr(self, "current_file_path", None)
        if cur is None or _norm_path(cur) != _norm_path(path):
            return None
        displayed_for = getattr(self, "_displayed_content_path", None)
        if displayed_for is not None and _norm_path(displayed_for) != _norm_path(path):
            return None
        base = getattr(self, "_base_display_pixmap", None)
        if base is not None and not base.isNull():
            return base.width(), base.height()
        if displayed_for is None:
            return None
        pm = getattr(self, "current_pixmap", None)
        if pm is None or pm.isNull() or self._get_visual_rotation_degrees(path):
            return None
        return pm.width(), pm.height()

    def _manager_pixmap_is_duplicate_display(self, file_path: str, pixmap) -> bool:
        """True when manager delivered the same base buffer we already show."""
        if pixmap is None or pixmap.isNull():
            return False
        base_size = self._displayed_base_pixmap_size(file_path)
        if base_size is None:
            return False
        return base_size == (pixmap.width(), pixmap.height())

    def _resolve_display_pixmap_base(self, pixmap: QPixmap) -> QPixmap:
        """Ensure display_pixmap never treats an already rotated buffer as the base."""
        if pixmap is None or pixmap.isNull():
            return pixmap
        base = getattr(self, "_base_display_pixmap", None)
        current = getattr(self, "current_pixmap", None)
        if (
            base is not None
            and not base.isNull()
            and pixmap is current
            and pixmap is not base
            and self._get_visual_rotation_degrees()
        ):
            return base
        return pixmap
