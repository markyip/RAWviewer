"""image_load mixin, part 1 of 3 (21 methods, 756 lines)."""
from __future__ import annotations

import contextlib
import os
import sys
import time

from PyQt6.QtCore import QRunnable, QTimer
from PyQt6.QtGui import QImage, QPixmap

from core.config import _env_int, _norm_path
from rawviewer_app.signals import WebpDecodeSignals

from .. import lazy
from ..lazy import image_covers_sensor_resolution, is_raw_file
from ..qt_helpers import _get_bg_thread_pool

lazy.register(sys.modules[__name__])


class ImageLoadPart1:
    def _load_pixmap_safe(self, file_path):
        """Delegate to common_image_loader (RAW + TIFF + JPEG with shared orientation)."""
        from common_image_loader import load_pixmap_safe, take_last_load_edr_flag

        pixmap = load_pixmap_safe(file_path)
        if take_last_load_edr_flag():
            self._display_edr_mode = "hdr"
        return pixmap

    def _cache_buffer_max_dim(self, image, *, min_dim: int = 0) -> int:
        """Return max(width, height) for a cached numpy/QPixmap buffer (0 if unknown)."""
        try:
            if hasattr(image, "shape"):
                return max(int(image.shape[0]), int(image.shape[1]))
            if hasattr(image, "width"):
                return max(int(image.height()), int(image.width()))
        except Exception:
            pass
        return 0

    def _single_view_cache_min_dim(self, *, for_navigation: bool = False) -> int:
        """Minimum max-edge for painting a cached preview in single view.

        Navigation default is above gallery grid-tier (~512px) so a 341x512
        disk thumbnail cannot flash in after a failed full-decode retry
        (``_try_paint_display_quality_cache`` uses context=navigation).
        """
        if for_navigation:
            return _env_int("RAWVIEWER_NAV_PREVIEW_MIN_DIM", 768, minimum=513)
        return _env_int("RAWVIEWER_GALLERY_INSTANT_MIN_DIM", 1400, minimum=400)

    def _cache_buffer_suitable_for_single_view(
        self, image, *, min_dim: int | None = None, file_path: str | None = None,
        allow_sensor_resolution: bool = False,
    ) -> bool:
        """Reject gallery grid-tier disk previews (~512px); wait for embedded thumbnail/full.

        Sensor-resolution buffers in the preview cache are also rejected so EXIF-ready
        does not replace a fast 1920px first frame with a full LibRaw decode.
        Navigation may allow LibRaw half-res buffers when RAW workflow is active.
        """
        if image is None:
            return False
        if min_dim is None:
            min_dim = self._single_view_cache_min_dim(for_navigation=False)
        if self._cache_buffer_max_dim(image) < int(min_dim):
            return False
        if (
            not allow_sensor_resolution
            and file_path
            and image_covers_sensor_resolution is not None
        ):
            try:
                w = h = 0
                if hasattr(image, "shape"):
                    h, w = int(image.shape[0]), int(image.shape[1])
                elif hasattr(image, "width"):
                    w, h = int(image.width()), int(image.height())
                if w and h:
                    exif = None
                    if hasattr(self, "image_cache"):
                        exif = self.image_cache.get_exif(file_path)
                    if image_covers_sensor_resolution(w, h, exif):
                        return False
            except Exception:
                pass
        return True

    def _clipping_overlay_cache_token(self, pm) -> tuple:
        return (int(pm.cacheKey()), pm.width(), pm.height())

    def _on_raw_recovery_ready(self, file_path: str, rgb) -> None:
        norm = _norm_path(file_path)
        if getattr(self, "_raw_recovery_worker_path", None) == norm:
            self._raw_recovery_worker_path = None
        if rgb is not None:
            self._raw_recovery_cache[norm] = rgb
            if len(self._raw_recovery_cache) > 8:
                for old_key in list(self._raw_recovery_cache.keys())[:-8]:
                    self._raw_recovery_cache.pop(old_key, None)
        if not getattr(self, "_raw_recovery_active", False):
            return
        if norm != _norm_path(getattr(self, "current_file_path", None)):
            return
        if rgb is None:
            return
        import logging

        logging.getLogger(__name__).info(
            "RAW recovery ready for %s shape=%s",
            os.path.basename(file_path),
            getattr(rgb, "shape", None),
        )
        self._orientation_already_applied = True
        try:
            self._display_raw_recovery_buffer(file_path, rgb)
        finally:
            self._orientation_already_applied = False
        self._sync_raw_recovery_view_mode()
        self.status_bar.showMessage(
            "RAW recovery preview ready (local shadows/highlights). P=off",
            3500,
        )

    def _on_raw_recovery_failed(self, file_path: str, message: str) -> None:
        norm = _norm_path(file_path)
        if getattr(self, "_raw_recovery_worker_path", None) == norm:
            self._raw_recovery_worker_path = None
        import logging

        logging.getLogger(__name__).warning(
            "RAW recovery failed for %s: %s",
            os.path.basename(file_path),
            message,
        )
        if (
            getattr(self, "_raw_recovery_active", False)
            and _norm_path(file_path) == _norm_path(getattr(self, "current_file_path", None))
        ):
            self.status_bar.showMessage(
                f"RAW recovery failed -- showing standard preview ({message})",
                5000,
            )
            self._sync_raw_recovery_view_mode()

    def _enforce_disk_cache_budget_background(self) -> None:
        """Evict oldest disk-cache entries over budget, off the main thread.

        Unlike cleanup_old_cache() (a handful of cheap SQL DELETEs),
        enforce_disk_cache_budget() stats every cached file to total disk usage --
        potentially thousands of os.path.getsize() calls -- so it runs on a
        QThreadPool worker instead of blocking the Qt event loop.
        """
        cache = getattr(self, "image_cache", None)
        if cache is None:
            return

        class _DiskBudgetWorker(QRunnable):
            def run(self_inner):
                try:
                    cache.enforce_disk_cache_budget()
                except Exception as exc:
                    import logging

                    logging.getLogger(__name__).debug(
                        "Disk cache budget enforcement failed: %s", exc, exc_info=True
                    )

        _get_bg_thread_pool().start(_DiskBudgetWorker())

    def _connect_image_manager_signals(self):
        """連接 ImageLoadManager 的永久信號（事件驅動架構）"""
        # 縮圖就緒
        self.image_manager.thumbnail_ready.connect(self.on_manager_thumbnail_ready)
        # 完整圖像就緒
        self.image_manager.image_ready.connect(self.on_manager_image_ready)
        # QPixmap 就緒（非 RAW 文件）
        self.image_manager.pixmap_ready.connect(self.on_manager_pixmap_ready)
        self.image_manager.qimage_ready.connect(self.on_manager_qimage_ready)
        # EXIF 數據就緒
        self.image_manager.exif_data_ready.connect(self.on_manager_exif_ready)
        # 錯誤處理
        self.image_manager.error_occurred.connect(self.on_manager_error)
        # 進度更新
        self.image_manager.progress_updated.connect(self.on_manager_progress)

    def _load_burst_rejected_paths(self, folder: str) -> None:
        import json

        self._burst_rejected_paths = set()
        if not folder:
            return
        try:
            raw = self.get_settings().value(self._burst_rejected_settings_key(folder), "")
            if not raw:
                return
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", errors="ignore")
            data = json.loads(raw)
            if isinstance(data, list):
                self._burst_rejected_paths = {
                    _norm_path(str(item)) for item in data if item
                }
        except Exception:
            self._burst_rejected_paths = set()

    def _maybe_queue_background_full_decode(self, file_path: str) -> None:
        """Queue sensor-res decode only when the on-screen buffer is not already full resolution."""
        import logging

        logger = logging.getLogger(__name__)
        if self._should_defer_full_decode():
            logger.debug("[LOAD] _maybe_queue_background_full_decode deferred for %s", os.path.basename(file_path or ""))
            return
        if not file_path:
            return
        try:
            preview = self.image_cache.get_preview_memory_only(file_path)
            exif = self.image_cache.get_exif(file_path)
            if preview is not None and image_covers_sensor_resolution(
                preview.shape[1], preview.shape[0], exif
            ):
                logger.debug("[LOAD] _maybe_queue_background_full_decode: preview covers sensor for %s", os.path.basename(file_path))
                return
            if self._cached_full_covers_sensor(file_path):
                logger.debug("[LOAD] _maybe_queue_background_full_decode: cached full covers sensor for %s", os.path.basename(file_path))
                return
        except Exception:
            pass
        logger.info("[LOAD] _maybe_queue_background_full_decode: queueing for %s", os.path.basename(file_path))
        from image_load_manager import Priority as _Priority

        # On-screen image: CURRENT so GPU raw_limit reservation cannot let
        # neighbor PRELOAD full starve this demosaic.
        prio = _Priority.CURRENT
        cur = getattr(self, "current_file_path", None)
        if cur and _norm_path(cur) != _norm_path(file_path):
            prio = _Priority.BACKGROUND
            self.image_manager.load_image(
                file_path,
                priority=prio,
                cancel_existing=False,
                use_full_resolution=True,
                stages={"full"},
                bypass_cache=not self._cached_full_covers_sensor(file_path),
            )

    def _on_manager_image_ready_impl(self, file_path: str, image):
        """處理 ImageLoadManager 的完整圖像就緒信號"""
        if getattr(self, "view_mode", "") == "compare":
            self._compare_upgrade_view_image(file_path, image)
            return
        if getattr(self, "_suppress_single_manager_callbacks", False):
            return
        if self._adjust_panel_blocks_display_refresh():
            return
        if getattr(self, "view_mode", "single") != "single":
            return
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.hide_loading()

        import logging
        logger = logging.getLogger(__name__)

        # Only handle current file's image (normalize for Windows path format differences)
        if _norm_path(file_path) != _norm_path(getattr(self, "current_file_path", None)):
            logger.debug(f"[MANAGER] Image for different file: {os.path.basename(file_path)}")
            return

        logger.info(f"[MANAGER] Full image ready for {os.path.basename(file_path)}")

        # Reset the deferred-sensor-decode gate so a later navigation to the
        # same file is not permanently blocked ("stops loading full resolution
        # view of any image" symptom). The gate is set in
        # _flush_deferred_sensor_full_decode and was never cleared before.
        try:
            if _norm_path(file_path) == _norm_path(getattr(self, "current_file_path", "") or ""):
                self._sensor_full_decode_queued_norm = None
                self._defer_sensor_full_decode_path = None
        except Exception:
            pass

        if image is None:
            logger.error(f"[MANAGER] image is None in on_manager_image_ready for {file_path}")
            return
        # Full-res-only testing mode: ignore non-sensor preview tiers delivered
        # as image_ready; wait for the real sensor-resolution buffer.
        if self._full_res_only_mode():
            try:
                if hasattr(image, "shape"):
                    _h, _w = image.shape[0], image.shape[1]
                elif hasattr(image, "width") and hasattr(image, "height"):
                    _w, _h = image.width(), image.height()
                else:
                    _w = _h = 0
                _exif = self.image_cache.get_exif(file_path)
                if _w > 0 and _h > 0 and not image_covers_sensor_resolution(_w, _h, _exif):
                    logger.info(
                        "[FULL_RES_ONLY] Skipping non-sensor image_ready (%dx%d) for %s",
                        _w, _h, os.path.basename(file_path),
                    )
                    return
            except Exception:
                pass

        if self._should_hold_full_until_preview_paint(file_path):
            logger.info(
                "[MANAGER] Holding full-res paint until preview first-render for %s",
                os.path.basename(file_path),
            )
            return

        # Check resolution to see if this is "Full" or "Preview" -- we need this BEFORE
        # the redundancy check so full-res images are never skipped even when the on-screen
        # buffer has the same pixel dimensions (e.g. gallery-to-single transition can leave a
        # high-pixel-count but blurry embedded-JPEG preview on screen).
        if hasattr(image, 'shape'):
            shape = image.shape
            height, width = shape[0], shape[1]
            shape[2] if len(shape) > 2 else 1
        elif hasattr(image, 'width') and hasattr(image, 'height'):
            # It's likely a QPixmap or QImage
            height, width = image.height(), image.width()
        else:
            logger.error(f"[MANAGER] Invalid image object type: {type(image)}")
            return

        max_dim = max(height, width)
        exif_for_res = None
        with contextlib.suppress(Exception):
            exif_for_res = self.image_cache.get_exif(file_path)
        is_full_resolution = image_covers_sensor_resolution(width, height, exif_for_res)

        # Skip redundant redraws when the same path is already on screen at the same max
        # dimension. Gallery → single may still upgrade blur→sharp via _loading_from_gallery.
        skip_redundant = (
            not getattr(self, "_skip_resolution_downgrade_check", False)
            and self._already_displaying_buffer_for_path(file_path, image)
        )
        if skip_redundant and (
            not is_full_resolution
            or not getattr(self, "_loading_from_gallery", False)
        ):
            logger.info(
                "[MANAGER] Skipping redundant image_ready for %s (already on screen%s)",
                os.path.basename(file_path),
                ", full-res" if is_full_resolution else "",
            )
            self._full_resolution_loading = False
            self._finish_pending_zoom_after_full_load(file_path)
            return


        # Space/double-click deferred zoom: skip intermediate buffers (avoids a
        # one-frame fit transform on a large pixmap = upper-left flash).
        if (
            getattr(self, "_pending_zoom", False)
            and not is_full_resolution
            and _norm_path(file_path) == _norm_path(getattr(self, "current_file_path", "") or "")
        ):
            logger.info(
                "[MANAGER] Holding %dx%d buffer -- waiting for sensor-res before pending zoom (%s)",
                width,
                height,
                os.path.basename(file_path),
            )
            self._orientation_already_applied = False
            return

        if is_full_resolution:
            logger.info(f"[MANAGER] Full-resolution pixels loaded ({width}x{height}).")
            self._full_resolution_loading = False
            self._is_half_size_displayed = False
            # Only preserve zoom across a resolution swap when the user is actually zoomed in.
            # In fit-to-window mode, refitting the sharper buffer avoids a 100% flash then fit.
            if not self._single_view_is_fit_mode():
                self._maintain_zoom_on_navigation = True
        else:
            self._is_half_size_displayed = True

        # Mark that orientation is already applied (UnifiedImageProcessor applies it)
        # logger.debug(f"[MANAGER] Setting _orientation_already_applied = True before display_numpy_image")
        self._orientation_already_applied = True

        # CRITICAL: Prevent resolution downgrade within the SAME file: a late small preview must not
        # replace a higher-resolution image we already showed for this path. When switching files,
        # ``current_pixmap`` may still hold the *previous* image -- do not compare dimensions then.
        displayed_for = getattr(self, "_displayed_content_path", None)
        pixmap_is_for_this_file = (
            displayed_for is not None
            and _norm_path(displayed_for) == _norm_path(file_path)
        )
        current_max_dim = 0
        if self.current_pixmap and pixmap_is_for_this_file:
            current_max_dim = max(self.current_pixmap.width(), self.current_pixmap.height())

        if (
            not getattr(self, "_skip_resolution_downgrade_check", False)
            and pixmap_is_for_this_file
            and max_dim < current_max_dim
            and self._single_view_pixels_on_screen(file_path)
            and _norm_path(file_path) == _norm_path(getattr(self, "current_file_path", None))
        ):
            logger.info(f"[MANAGER] Ignoring lower-resolution preview ({width}x{height}) as higher-resolution image ({current_max_dim}px) is already displayed.")
            # Still update status bar and focus if needed, but don't redisplay
            self._orientation_already_applied = False
            self._full_resolution_loading = False
            self._finish_pending_zoom_after_full_load(file_path)
            return

        if (
            pixmap_is_for_this_file
            and current_max_dim > 0
            and max_dim > current_max_dim
            and not getattr(self, "_navigation_in_progress", False)
        ):
            self._mark_pending_resolution_crossfade()

        norm_fp = _norm_path(file_path)
        raw_recovery = (
            getattr(self, "_raw_recovery_active", False) and is_raw_file(file_path)
        )
        if raw_recovery:
            cached = self._raw_recovery_cache.get(norm_fp)
            if cached is not None:
                self._orientation_already_applied = True
                try:
                    self._display_raw_recovery_buffer(file_path, cached)
                finally:
                    self._orientation_already_applied = False
            else:
                self.display_numpy_image(image)
                self._start_raw_recovery(file_path)
                self._sync_raw_recovery_view_mode()
                self._orientation_already_applied = False
                self._full_resolution_loading = False
                self._finish_pending_zoom_after_full_load(file_path)
                return
        else:
            self.display_numpy_image(image)
        self._skip_resolution_downgrade_check = False
        self._on_single_view_content_displayed()

        # logger.debug(f"[MANAGER] Resetting _orientation_already_applied = False after display_numpy_image")
        self._orientation_already_applied = False  # Reset flag

        # Track the max dimension seen for this specific file path to prevent downgrades
        if not hasattr(self, "_file_max_dim_map"):
            self._file_max_dim_map = {}
        self._file_max_dim_map[_norm_path(file_path)] = max(
            self._file_max_dim_map.get(_norm_path(file_path), 0), max_dim
        )

        self.status_bar.showMessage(f"Loaded {os.path.basename(file_path)}")

        # Pending 100% zoom after half-res (double-click or spacebar) is applied inside
        # display_pixmap() using _pending_zoom_center / _pending_zoom_thumbnail_size.
        # Do not re-run zoom here: a stale pre-display _pending_zoom flag would wrongly
        # replace the click-based center with the image center.

        # CRITICAL: Ensure metadata is updated after image is displayed
        # Try to get original dimensions from cache to show in status bar immediately
        orig_w, orig_h = None, None
        cached_exif = self.image_cache.get_exif(file_path)
        if cached_exif:
            orig_w = cached_exif.get('original_width')
            orig_h = cached_exif.get('original_height')

        self.update_status_bar(width=orig_w, height=orig_h)

        self.setFocus()
        self.schedule_save_session_state()
        self._start_preloading()

    def _retry_display_tier_with_full_decode(
        self, file_path: str, error_message: str
    ) -> bool:
        """Backward-compatible alias for preview-tier auto full decode."""
        return self._retry_raw_preview_with_full_decode(file_path, error_message)

    def _load_and_start_animation(self, file_path):
        self._stop_animations()
        # Animated GIF/WebP always play fit-to-window; no 100% zoom or pan.
        self.fit_to_window = True
        self.current_zoom_level = 1.0
        self.zoom_center_point = None
        self._pending_zoom = False
        self._pending_zoom_center = None
        self._pending_zoom_thumbnail_size = None
        self._pending_zoom_toggle = False

        # Hide GPU view, show scroll area/image label (animated formats only)
        container = getattr(self, "single_view_container", None)
        if container is not None and hasattr(container, "set_gpu_view_active"):
            container.set_gpu_view_active(False)
        elif getattr(self, "gpu_view", None) is not None:
            self.gpu_view.hide()
            self.scroll_area.show()
        self.image_label.show()
        self.image_label.setText("")

        # Hide loading overlay if visible
        if hasattr(self, "loading_overlay") and self.loading_overlay:
            self.loading_overlay.hide_loading()

        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.gif':
            from PyQt6.QtGui import QMovie
            self._current_movie = QMovie(file_path)
            self.image_label.setMovie(self._current_movie)

            def on_frame_changed(frame_no):
                if not self._current_movie:
                    return
                with contextlib.suppress(Exception):
                    self._current_movie.frameChanged.disconnect(on_frame_changed)
                movie_size = self._current_movie.frameRect().size()
                self._original_movie_size = movie_size
                self._update_gif_scaling()

            self._current_movie.frameChanged.connect(on_frame_changed)
            self._current_movie.start()
            self._mark_animated_content_displayed(file_path)
            self.status_bar.showMessage(f"Playing GIF: {os.path.basename(file_path)}")
            return True
        elif ext == '.webp':
            token = getattr(self, "_webp_decode_generation", 0) + 1
            self._webp_decode_generation = token
            self._webp_decode_in_progress_path = file_path
            self._current_webp_frames = []
            self._current_webp_durations = []
            self._current_webp_idx = 0

            signals = getattr(self, "_webp_decode_signals", None)
            if signals is None:
                signals = WebpDecodeSignals()
                signals.ready.connect(self._on_webp_frames_decoded)
                signals.failed.connect(self._on_webp_decode_failed)
                self._webp_decode_signals = signals

            self.status_bar.showMessage(
                f"Decoding WebP: {os.path.basename(file_path)}"
            )

            class _WebpDecodeWorker(QRunnable):
                def run(self_inner):
                    try:
                        from PIL import Image

                        frames = []
                        durations = []
                        with Image.open(file_path) as img:
                            n_frames = getattr(img, "n_frames", 1)
                            if n_frames <= 1:
                                raise ValueError("WebP is not animated")
                            for frame_idx in range(n_frames):
                                img.seek(frame_idx)
                                frame_rgba = img.convert("RGBA")
                                w, h = frame_rgba.size
                                data = frame_rgba.tobytes("raw", "RGBA")
                                qim = QImage(
                                    data, w, h, QImage.Format.Format_RGBA8888
                                ).copy()
                                frames.append(qim)
                                duration = img.info.get("duration", 100)
                                if duration <= 0:
                                    duration = 100
                                durations.append(duration)
                    except Exception as e:
                        try:
                            from PyQt6 import sip

                            if not sip.isdeleted(signals):
                                signals.failed.emit(file_path, str(e))
                        except Exception:
                            pass
                        return

                    try:
                        from PyQt6 import sip

                        if not sip.isdeleted(signals):
                            signals.ready.emit(token, file_path, frames, durations)
                    except Exception:
                        pass

            _get_bg_thread_pool().start(_WebpDecodeWorker())
            return True
        return False

    def _on_webp_frames_decoded(
        self, token: int, file_path: str, qimages: list, durations: list
    ) -> None:
        if token != getattr(self, "_webp_decode_generation", None):
            return
        if _norm_path(file_path) != _norm_path(getattr(self, "current_file_path", None)):
            return
        if not qimages:
            return

        self._webp_decode_in_progress_path = None
        self._current_webp_frames = [QPixmap.fromImage(qim) for qim in qimages]
        self._current_webp_durations = list(durations)
        self._current_webp_idx = 0
        self._webp_timer = QTimer(self)
        self._webp_timer.setSingleShot(True)
        self._webp_timer.timeout.connect(self._play_next_webp_frame)
        self._play_next_webp_frame()
        self._mark_animated_content_displayed(file_path)
        self.status_bar.showMessage(f"Playing WebP: {os.path.basename(file_path)}")

    def _on_webp_decode_failed(self, file_path: str, error_message: str) -> None:
        if _norm_path(file_path) != _norm_path(getattr(self, "current_file_path", None)):
            return
        self._webp_decode_in_progress_path = None
        import logging

        logging.getLogger(__name__).error(
            "Error loading animated WebP: %s", error_message, exc_info=True
        )
        self.image_label.setText(f"Error loading WebP: {error_message}")

    def _clear_finished_worker_flags(self) -> None:
        """Un-wedge the export gates when a worker finished unobserved.

        Both gates used to be cleared only by the worker's queued Qt
        ``finished`` signal. If that emission was ever lost (worker raised
        before emitting, or its per-op signal object died with the runnable
        the thread pool owns), the flag stayed True forever and every later
        Export click returned early -- the pre-export dialog then silently
        never appeared again. Workers now set a plain threading.Event in a
        ``finally``; the UI thread can read that without waiting on signal
        delivery, so a stale gate heals on the next click.
        """
        batch_worker = getattr(self, "_batch_op_worker", None)
        if (
            batch_worker is not None
            and getattr(batch_worker, "done_event", None) is not None
            and batch_worker.done_event.is_set()
        ):
            self._batch_op_worker = None
            self._batch_op_in_progress = False
            self._close_batch_progress_dialog()
        export_worker = getattr(self, "_adjust_export_worker", None)
        if (
            export_worker is not None
            and getattr(export_worker, "done_event", None) is not None
            and export_worker.done_event.is_set()
        ):
            self._adjust_export_worker = None
            self._adjust_export_in_progress = False

    def _needs_full_resolution_upgrade(self, file_path: str | None = None) -> bool:
        """True when on-screen pixels are below sensor resolution (EXIF).

        In the RAW (High Quality) workflow, an embedded-JPEG stand-in also
        counts as needing upgrade even when its dimensions already reach
        sensor size -- camera JPEGs can be near-full-res but are not the
        app's LibRaw look (see image_cache.full_image_is_embedded_jpeg).

        In the non-RAW / embedded-JPEG workflow that stand-in *is* the
        intended full image: treating it as incomplete forced a LibRaw
        demosaic on zoom/fit/idle and made non-RAW browsing look like RAW
        (and feel slow).
        """
        fp = file_path or getattr(self, "current_file_path", None)
        pm = getattr(self, "current_pixmap", None)
        if not fp or pm is None or pm.isNull():
            return False
        raw_workflow = False
        try:
            from common_image_loader import use_libraw_consistent_preview_first

            raw_workflow = bool(use_libraw_consistent_preview_first(fp))
        except Exception:
            raw_workflow = False
        if raw_workflow and self.image_cache.full_image_is_embedded_jpeg(fp):
            return True
        try:
            exif = self.image_cache.get_exif(fp)
        except Exception:
            exif = None
        return not image_covers_sensor_resolution(pm.width(), pm.height(), exif)

    def _flush_deferred_sensor_full_decode(
        self, file_path: str | None = None, *, force: bool = False
    ) -> None:
        import logging

        logger = logging.getLogger(__name__)
        if self._adjust_panel_blocks_display_refresh():
            return
        if self._raw_recovery_preview_active():
            return
        if not force and self._should_defer_full_decode():
            logger.debug(
                "[LOAD] _flush_deferred_sensor_full_decode deferred for %s (session restore)",
                os.path.basename(file_path or ""),
            )
            return
        fp = file_path or getattr(self, "_defer_sensor_full_decode_path", None)
        if not fp:
            logger.debug("[LOAD] _flush_deferred_sensor_full_decode: no file path")
            return
        if _norm_path(fp) != _norm_path(getattr(self, "current_file_path", "") or ""):
            logger.debug(
                "[LOAD] _flush_deferred_sensor_full_decode: path mismatch %s vs %s",
                os.path.basename(fp),
                os.path.basename(getattr(self, "current_file_path", "") or ""),
            )
            self._defer_sensor_full_decode_path = None
            return
        norm = _norm_path(fp)
        if getattr(self, "_sensor_full_decode_queued_norm", None) == norm:
            logger.debug(
                "[LOAD] _flush_deferred_sensor_full_decode: already queued for %s",
                os.path.basename(fp),
            )
            self._defer_sensor_full_decode_path = None
            return
        self._sensor_full_decode_queued_norm = norm
        self._defer_sensor_full_decode_path = None
        logger.info(
            "[LOAD] _flush_deferred_sensor_full_decode: queueing sensor decode for %s (force=%s)",
            os.path.basename(fp),
            force,
        )
        try:
            cached_full = self.image_cache.get_full_image(fp)
            if cached_full is not None and self._needs_full_resolution_upgrade(fp):
                self.on_manager_image_ready(fp, cached_full)
                return
        except Exception:
            pass
        self._request_full_res_decode(fp, priority_current=True, force=force)

    def _allow_full_decode_for_user_zoom(self) -> None:
        """User zoom/100% must not wait on session-restore full-decode deferral."""
        if getattr(self, "_session_restore_defer_preload", False):
            self._session_restore_full_decode_allowed = True
        # Mark interaction so GPU torch import stays out of the way.
        self._user_zoom_interaction_ts = time.monotonic()
        # If deferred full was held for TTFR quiet time, kick it now.
        try:
            fp = getattr(self, "current_file_path", None)
            if fp and self._needs_full_resolution_upgrade(fp):
                self._flush_deferred_sensor_full_decode(fp, force=True)
        except Exception:
            pass

    def _maybe_request_full_res_for_smooth_zoom(self) -> None:
        if self._raw_recovery_preview_active():
            return
        # Adjust uses half-res / settle preview tiers -- never queue browse
        # sensor full-decode from pinch/100% while the editor is open.
        if self._adjust_panel_active():
            return
        fp = getattr(self, "current_file_path", None)
        if not fp:
            return
        self._allow_full_decode_for_user_zoom()
        # Prefer a covering cache hit even when current_pixmap is still the
        # preview tier (common race: full numpy landed, pixmap conversion still
        # queued). Avoids a redundant "starting full-resolution decode" path.
        if self._try_sync_full_before_zoom():
            return
        if not self._needs_full_resolution_upgrade():
            return
        if getattr(self, "_full_resolution_loading", False):
            return
        if getattr(self, "_smooth_zoom_full_request_sent", False):
            return
        norm = _norm_path(fp)
        if getattr(self, "_sensor_full_decode_queued_norm", None) == norm:
            # Already in flight from session restore / idle flush -- just wait.
            self._smooth_zoom_full_request_sent = True
            return
        # Latch the one-shot only after a real queue. Setting it before a
        # refused call (Adjust opened mid-gesture, etc.) left the flag True
        # with a dead else/`pass` that claimed retries were allowed -- so the
        # rest of the pinch gesture never asked again. Log only on success
        # to avoid the prior per-sample flood on no-op queues.
        if self._request_full_res_decode(fp, priority_current=True, urgent=True):
            self._smooth_zoom_full_request_sent = True
            self._full_resolution_loading = True
            import logging

            logging.getLogger(__name__).info(
                "[ZOOM] Preview below sensor resolution -- starting full-resolution decode"
            )
