"""display mixin, part 5 of 6 (4 methods, 891 lines)."""
from __future__ import annotations

import contextlib
import os
import sys
import time

from PyQt6.QtCore import QPoint, QTimer
from PyQt6.QtGui import QPixmap

from core.config import _norm_path

from .. import lazy
from ..lazy import image_covers_sensor_resolution, metadata_index_idle_delay_ms
from ..paths import _require_str_path

lazy.register(sys.modules[__name__])


class DisplayPart5:
    def display_pixmap(self, pixmap):
        """Display a QPixmap."""
        if self._adjust_panel_blocks_display_refresh() and not getattr(
            self, "_adjust_preview_painting", False
        ):
            return
        if hasattr(self, 'current_file_path') and self.is_animated_image(_require_str_path(self.current_file_path)):
            return
        # RAW workflow JPEG-interim suppression (see display_numpy_image) --
        # same guard for the QPixmap paint path.
        try:
            cur = getattr(self, "current_file_path", None)
            if (
                pixmap is not None
                and not pixmap.isNull()
                and max(pixmap.width(), pixmap.height()) < 1024
                and self._suppress_jpeg_interim(cur)
                and not self.image_cache.is_libraw_preview(cur)
            ):
                return
        except Exception:
            pass

        import logging
        import time
        logger = logging.getLogger(__name__)
        display_start = time.time()
        pixmap = self._resolve_display_pixmap_base(pixmap)

        # Get current file name for logging context
        current_file = os.path.basename(_require_str_path(self.current_file_path)) if hasattr(self, 'current_file_path') and self.current_file_path else "Unknown"
        logger.info(f"[DISPLAY_PIXMAP] ========== display_pixmap() STARTED at {display_start:.3f} ==========")
        logger.info(f"[DISPLAY_PIXMAP] File: {current_file}, Pixmap size: {pixmap.width()}x{pixmap.height()}")

        if pixmap is None or pixmap.isNull():
            logger.error(f"[DISPLAY_PIXMAP] Received null pixmap for {current_file}")
            self._set_single_view_pixmap(pixmap)
            if hasattr(self, "loading_overlay"):
                self.loading_overlay.hide_loading()
            return

        # New content is arriving (thumbnail/preview/full upgrade or navigation). Request a
        # one-shot cross-fade; it is consumed by the next _set_single_view_pixmap() in this
        # synchronous call chain. Reset defensively at the end so it can never leak to a
        # later zoom/pan repaint.
        self._crossfade_next_paint = True

        if getattr(self, "_slideshow_force_fit_next", False):
            if self._slideshow_is_active():
                self.fit_to_window = True
                self.current_zoom_level = 1.0
                self.zoom_center_point = None
            self._slideshow_force_fit_next = False

        # Keep the unrotated (display pipeline) pixmap so visual rotations can re-render instantly.
        self._base_display_pixmap = pixmap
        pixmap = self._apply_visual_rotation_for_current(pixmap)
        prev_pm = getattr(self, "current_pixmap", None)
        prev_path = getattr(self, "_displayed_content_path", None)
        pending_gallery_full = getattr(self, "_gallery_preview_pending_full", False)
        cur_path = getattr(self, "current_file_path", None)
        skip_crossfade = bool(getattr(self, "_skip_resolution_crossfade_once", False))
        if skip_crossfade:
            self._skip_resolution_crossfade_once = False
            self._gallery_preview_pending_full = False
            self._clear_pending_resolution_crossfade()
        needs_resolution_crossfade = (
            False
            if skip_crossfade
            else self._resolution_upgrade_needs_crossfade(
                prev_pm,
                pixmap,
                prev_path=prev_path,
                cur_path=cur_path,
            )
        )
        logging.getLogger(__name__).debug(
            "[TIERSWAP] display: prev=%s (%sx%s) -> new=%s (%sx%s) skip=%s fade=%s",
            os.path.basename(prev_path or "") or None,
            prev_pm.width() if prev_pm is not None and not prev_pm.isNull() else 0,
            prev_pm.height() if prev_pm is not None and not prev_pm.isNull() else 0,
            os.path.basename(cur_path or "") or None,
            pixmap.width(),
            pixmap.height(),
            skip_crossfade,
            needs_resolution_crossfade,
        )
        gv = getattr(self, "gpu_view", None)
        new_file = (
            gv is not None
            and _norm_path(cur_path) != _norm_path(getattr(self, "_gpu_last_path", None))
        )
        # is_nav_fade tracking is deliberately independent of gv/new_file
        # above (which only fires when a GPU view exists) -- it must also
        # work on the legacy QLabel fallback path (_set_single_view_pixmap,
        # used when gv is None) via self._resolution_crossfade_is_navigation.
        # Persists across the async decode wait between a navigation's first
        # paint (the nav preview) and its upgrade paint (full-res landing)
        # -- _navigation_in_progress itself resets to False within ~0.3-0.6s
        # of the keypress, long before a multi-second RAW decode completes,
        # so it cannot be read here to identify "this upgrade paint belongs
        # to a navigation".
        is_new_display_path = _norm_path(cur_path) != _norm_path(
            getattr(self, "_last_display_path_for_fade", None)
        )
        is_nav_fade = (not is_new_display_path) and bool(
            getattr(self, "_pending_nav_fade", False)
        )
        self._pending_nav_fade = is_new_display_path
        self._last_display_path_for_fade = cur_path
        self._resolution_crossfade_is_navigation = is_nav_fade

        viewport_fade_snapshot = None
        viewport_fade_pm = None
        resolution_fade_ms = None
        if needs_resolution_crossfade:
            viewport_fade_snapshot = self._grab_viewport_snapshot_for_crossfade()
            if viewport_fade_snapshot is None and prev_pm is not None and not prev_pm.isNull():
                viewport_fade_pm = QPixmap(prev_pm)
            if prev_pm is not None and not prev_pm.isNull():
                resolution_fade_ms = self._resolution_crossfade_duration_ms(
                    max(prev_pm.width(), prev_pm.height()),
                    max(pixmap.width(), pixmap.height()),
                    is_navigation=is_nav_fade,
                )
        if pending_gallery_full:
            self._gallery_preview_pending_full = False
            self._gallery_crossfade_next = True
        if needs_resolution_crossfade:
            self._clear_pending_resolution_crossfade()
            self._crossfade_use_resolution_duration = True
        elif getattr(self, "_pending_resolution_crossfade", False):
            self._clear_pending_resolution_crossfade()

        if new_file:
            self._cancel_content_crossfade()
            self._clear_pending_resolution_crossfade()
            self._clear_focus_zoom_anchor()
        pending_point_zoom = bool(
            getattr(self, "_pending_zoom", False)
            and not getattr(self, "_full_resolution_loading", False)
        )
        use_viewport_crossfade = (
            needs_resolution_crossfade
            and not new_file
            and gv is not None
            and (viewport_fade_snapshot is not None or viewport_fade_pm is not None)
            and not pending_point_zoom
        )
        defer_display_commit = use_viewport_crossfade
        apply_gen = int(getattr(self, "_single_view_display_generation", 0))

        def _commit_display_metadata() -> None:
            prev_displayed_path = getattr(self, "_displayed_content_path", None)
            self.current_pixmap = pixmap
            self._displayed_content_path = cur_path
            if getattr(self, "_focus_subject_outline_active", False):
                self._focus_subject_rect_image = None
                self._focus_rect_source = None
            self._sync_single_image_histogram()
            if pixmap is not None and not pixmap.isNull() and getattr(
                self, "current_file_path", None
            ):
                pm_max = max(pixmap.width(), pixmap.height())
                # _manager_displayed_max_dim is a per-file high-water-mark of the
                # on-screen resolution (a later lower-res delivery must not
                # downgrade the display). It MUST reset when the displayed file
                # changes: carrying a larger previous image's value forward makes
                # a smaller new image's full-res look "already displayed" to
                # _already_displaying_buffer_for_path, so its full decode is
                # skipped and it sticks at the blurry preview (regression: nav
                # from an 8256px image to a 6720px one). On a file change, seed
                # from this paint; otherwise keep the running max.
                if _norm_path(prev_displayed_path or "") != _norm_path(cur_path or ""):
                    self._manager_displayed_max_dim = pm_max
                else:
                    self._manager_displayed_max_dim = max(
                        getattr(self, "_manager_displayed_max_dim", 0), pm_max
                    )

        if not defer_display_commit:
            _commit_display_metadata()

        # Route B: GPU view owns fit/zoom/pan. Push the full pixmap (GPU scales it) and let
        # the QGraphicsView handle the transform. New file -> fit; same-file resolution
        # upgrade -> preserve framing while zoomed.
        if gv is not None:
            gv.set_file_path(cur_path)
            self._crossfade_next_paint = False
            (
                getattr(self, "_preserve_nav_zoom_active", False)
                or getattr(self, "_maintain_zoom_on_navigation", False)
            )

            def _apply_gpu_view() -> bool:
                if apply_gen != int(getattr(self, "_single_view_display_generation", 0)):
                    return False

                if getattr(self, "_workflow_toggle_fit_until_display", False):
                    gv._zoom_intent_100 = False
                    gv.set_pixmap(pixmap, preserve_view=False)
                    gv.fit_to_window()
                    self.fit_to_window = True
                    self.zoom_center_point = None
                    self._clear_focus_zoom_anchor()
                    self._schedule_workflow_toggle_fit_complete()
                    return False

                # Deferred zoom: single atomic paint only (never set_pixmap + fit first).
                if self._pending_zoom_ready_for_pixmap(pixmap, cur_path):
                    return self._apply_pending_gpu_point_zoom_in_place(pixmap, gv)

                pending_user_zoom = bool(
                    getattr(self, "_pending_zoom_toggle", False)
                    or getattr(self, "_pending_zoom", False)
                    or getattr(self, "_pending_gpu_zoom_point", None) is not None
                    or getattr(self, "_pending_zoom_focus_subject", False)
                    or getattr(self, "_focus_zoom_anchor_active", False)
                    or (
                        getattr(gv, "_zoom_intent_100", False)
                        and not getattr(self, "_workflow_toggle_fit_until_display", False)
                    )
                )
                nav_preserve = bool(
                    getattr(self, "_preserve_nav_zoom_active", False)
                    or getattr(self, "_maintain_zoom_on_navigation", False)
                )
                zoomed_in = not self._single_view_is_fit_mode()
                logging.getLogger(__name__).debug(
                    "[ZOOMNAV] apply_gpu_view: new_file=%s nav_preserve=%s zoomed_in=%s "
                    "pending_user_zoom=%s pm=%sx%s interim=%s path=%s",
                    new_file,
                    nav_preserve,
                    zoomed_in,
                    pending_user_zoom,
                    pixmap.width(),
                    pixmap.height(),
                    getattr(self, "_nav_zoom_interim_active", False),
                    os.path.basename(cur_path or ""),
                )

                if new_file and (nav_preserve or zoomed_in):
                    pm_max_dim = max(pixmap.width(), pixmap.height())
                    is_raw = os.path.splitext(cur_path or "")[-1].lower() in {
                        '.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2',
                        '.pef', '.srw', '.x3f', '.raf', '.3fr', '.fff', '.iiq',
                        '.cap', '.erf', '.mef', '.mos', '.nrw', '.rwl', '.srf'}
                    is_pixmap_half_size = (pm_max_dim < 3000) and is_raw
                    if is_pixmap_half_size:
                        # RAW workflow delivers small tiers first. Show the
                        # tier at the equivalent on-screen framing (blurry,
                        # sharpens in place when the sensor buffer lands via
                        # the same-file upgrade path below) instead of
                        # flashing out to fit and re-zooming much later.
                        # Gate only on genuine user zoom intents. A stale
                        # _pending_zoom (the older defer-until-full-res
                        # mechanism, which this restore supersedes) must not
                        # block it -- that left navigation stuck at fit until
                        # the sensor decode landed, or dropped to fit entirely
                        # when the stale center could not be resolved.
                        user_zoom_intent = bool(
                            getattr(self, "_pending_zoom_toggle", False)
                            or getattr(self, "_pending_gpu_zoom_point", None) is not None
                            or getattr(self, "_pending_zoom_focus_subject", False)
                            or getattr(self, "_focus_zoom_anchor_active", False)
                        )
                        if not user_zoom_intent and nav_preserve and self._apply_gpu_navigation_zoom_restore(
                                pixmap, gv, scale_level_to_pixmap=True
                            ):
                                # Supersede any stale deferred zoom and hold
                                # this frame against further sub-sensor tiers
                                # (their preserve math would wobble the
                                # framing); the sensor-res buffer clears it.
                                self._clear_pending_point_zoom_state()
                                self._nav_zoom_interim_active = True
                                # The on-screen buffer just regressed to a
                                # small tier; drop the per-file "already
                                # displayed at N px" bookkeeping or the
                                # re-decoded sensor buffer is discarded as a
                                # stale duplicate and the view stays blurry.
                                self._manager_displayed_max_dim = 0
                                with contextlib.suppress(Exception):
                                    self._file_max_dim_map.pop(
                                        _norm_path(cur_path), None
                                    )
                                # The interim must not get stuck blurry: some
                                # cache-hit loads (e.g. quality_preview) never
                                # queue a follow-up decode. Display the cached
                                # full buffer if it covers the sensor, else
                                # queue the sensor decode explicitly.
                                self._schedule_interim_quality_upgrade(cur_path)
                                return True
                        # Fallback: defer zoom until full resolution arrives
                        self._pending_zoom = True
                        self._pending_zoom_center = self.zoom_center_point
                        self._pending_zoom_thumbnail_size = pixmap.size()
                        gv.set_pixmap(pixmap, preserve_view=False)
                        if not pending_user_zoom:
                            gv.fit_to_window()
                            self.fit_to_window = True
                        else:
                            self.fit_to_window = False
                        return False

                    if self._apply_gpu_navigation_zoom_restore(pixmap, gv):
                        return True

                if new_file:
                    if not pending_user_zoom:
                        gv._zoom_intent_100 = False
                    gv.set_pixmap(pixmap, preserve_view=False)
                    if self._single_view_is_fit_mode() and not pending_user_zoom:
                        gv.fit_to_window()
                    return False

                # While the equivalent-framing interim tier is on screen
                # (zoomed RAW navigation), place each higher-res tier at the
                # same sensor-equivalent framing (derived on the fly from the
                # current view + EXIF sensor dims) instead of set_pixmap's
                # preserve math, whose >=100% snap would wobble the framing.
                # Lower-res or same-res tiers are dropped. The hold ends when
                # a sensor-covering buffer lands.
                if (
                    getattr(self, "_nav_zoom_interim_active", False)
                    and not self._single_view_is_fit_mode()
                    and gv.has_pixmap()
                ):
                    cur_max = max(int(gv._img_w or 0), int(gv._img_h or 0))
                    new_max = max(pixmap.width(), pixmap.height())
                    if new_max <= cur_max:
                        return False
                    placed = False
                    try:
                        if cur_max <= 0:
                            raise ValueError("no current image extent to scale from")
                        # new_level = scale * cur_max / new_max holds regardless
                        # of sensor dims -- in the original derivation
                        # (sens_level = scale*cur_max/s_max; new_level =
                        # sens_level*s_max/new_max) the s_max terms cancel
                        # algebraically, so the framing math never actually
                        # needed EXIF sensor dimensions. Compute it
                        # unconditionally: previously a still-uncached EXIF
                        # entry (plausible mid-navigation, e.g. fast culling
                        # through many files before EXIF finishes loading for
                        # each one) made `sdims` empty, silently skipped this
                        # whole placement, and fell back to set_pixmap's
                        # generic preserve math -- whose own comment says its
                        # >=100% snap "wobbles" the framing. That fallback is
                        # exactly the reported "zoom glitches during culling".
                        # Sensor dims are still used below, but only for the
                        # secondary (non-framing) decisions: when to consider
                        # this the final tier and whether to allow overscale.
                        new_level = float(gv.current_scale()) * cur_max / new_max
                        from common_image_loader import sensor_pixel_dimensions

                        sdims = sensor_pixel_dimensions(
                            self.image_cache.get_exif(cur_path)
                        )
                        s_max = max(sdims) if sdims else 0
                        c = gv.mapToScene(gv.viewport().rect().center())
                        fx = c.x() / max(1, gv._img_w)
                        fy = c.y() / max(1, gv._img_h)
                        gv.set_pixmap_zoomed_at(
                            pixmap,
                            fx * pixmap.width(),
                            fy * pixmap.height(),
                            scale=new_level,
                            allow_overscale=bool(s_max) and new_max < s_max * 0.98,
                        )
                        self.current_zoom_level = float(gv.current_scale())
                        self.fit_to_window = False
                        placed = True
                        if s_max and new_max >= s_max * 0.98:
                            self._nav_zoom_interim_active = False
                    except Exception:
                        placed = False
                    if placed:
                        return False
                    self._nav_zoom_interim_active = False

                # Same-file buffer upgrade (preview → sensor, etc.)
                if zoomed_in or nav_preserve or pending_user_zoom:
                    preserve = zoomed_in or nav_preserve or needs_resolution_crossfade
                    self._apply_gpu_pixmap_with_focus_anchor(
                        gv, pixmap, preserve=preserve
                    )
                    return False

                if self._single_view_is_fit_mode() and not pending_user_zoom:
                    gv.set_pixmap(pixmap, preserve_view=False)
                    gv.fit_to_window()
                    return False

                gv.set_pixmap(pixmap, preserve_view=False)
                return False

            pending_zoom_applied = False
            if use_viewport_crossfade:
                fade_pm = viewport_fade_snapshot or viewport_fade_pm

                def _apply_gpu_view_with_commit() -> None:
                    # A newer navigation/reload (e.g. RAW<->JPEG workflow toggle,
                    # or simply moving on to another file) may have bumped
                    # _single_view_display_generation while this crossfade was
                    # animating. _apply_gpu_view() already no-ops the actual
                    # paint in that case, but _commit_display_metadata() has no
                    # such guard on its own -- calling it anyway would mark
                    # this stale file/resolution as "displayed" even though the
                    # GPU view's pixmap was never touched, leaving the screen
                    # frozen on whatever was on it before while every bookkeeping
                    # field claims the new file is on screen (observed as
                    # "toggle RAW view, navigate away, still shows the same
                    # image"). Skip both together so a newer, valid paint is
                    # solely responsible for updating state.
                    if apply_gen != int(getattr(self, "_single_view_display_generation", 0)):
                        return
                    _commit_display_metadata()
                    _apply_gpu_view()

                def _on_viewport_crossfade_done() -> None:
                    if apply_gen == int(getattr(self, "_single_view_display_generation", 0)):
                        self._gpu_last_path = cur_path

                self._run_viewport_crossfade(
                    fade_pm,
                    pixmap_is_viewport_snapshot=viewport_fade_snapshot is not None,
                    apply_fn=_apply_gpu_view_with_commit,
                    duration_ms=resolution_fade_ms,
                    on_finished=_on_viewport_crossfade_done,
                )
            else:
                pending_zoom_applied = _apply_gpu_view()
                self._gpu_last_path = cur_path
            if (
                not pending_zoom_applied
                and self._single_view_is_fit_mode()
                and not getattr(self, "_preserve_nav_zoom_active", False)
            ):
                self._schedule_gpu_viewport_refit()

            # Focus/subject outline lives as a scene overlay; clear stale rect from the
            # previous image, then let the scheduled refresh re-derive it for this one.
            gv.clear_overlay()

            if hasattr(self, "_maintain_zoom_on_navigation"):
                with contextlib.suppress(AttributeError):
                    delattr(self, "_maintain_zoom_on_navigation")

            pending_gpu_pt = getattr(self, "_pending_gpu_zoom_point", None)
            if pending_gpu_pt is not None:
                with contextlib.suppress(AttributeError):
                    delattr(self, "_pending_gpu_zoom_point")
                self.zoom_center_point = QPoint(
                    int(pending_gpu_pt.x()), int(pending_gpu_pt.y())
                )
                self._zoom_in_to_image_point_finish()
            elif getattr(self, "_pending_zoom_toggle", False):
                self._pending_zoom_toggle = False
                if hasattr(self, "_pending_zoom_target_path"):
                    with contextlib.suppress(AttributeError):
                        delattr(self, "_pending_zoom_target_path")
                if getattr(self, "zoom_center_point", None) is not None:
                    self._gpu_zoom_in_to_point_finish()
                elif gv.wants_zoom_in_toggle():
                    gv.zoom_to_actual()
                    self.fit_to_window = False
                    self.current_zoom_level = float(gv.current_scale())
                    self._gpu_request_full_resolution_if_needed()
                else:
                    gv.fit_to_window()
                    self.fit_to_window = True
            if hasattr(self, "loading_overlay"):
                self.loading_overlay.hide_loading()
            self._sync_displayed_half_size_flag(cur_path)
            self._update_gpu_status()
            self._maybe_refresh_focus_subject_outline_after_display()
            self._sync_gpu_single_view_visibility()
            return

        # Handle pending zoom toggle from spacebar/double click (when pixmap wasn't ready)
        if hasattr(self, '_pending_zoom_toggle') and self._pending_zoom_toggle:
            logger.info("[DISPLAY_PIXMAP] Handling pending zoom toggle")
            self._pending_zoom_toggle = False
            if hasattr(self, "_pending_zoom_target_path"):
                with contextlib.suppress(AttributeError):
                    delattr(self, "_pending_zoom_target_path")

            if hasattr(self, '_pending_zoom_pos'):
                # Zoom to the recorded point instead of center
                delattr(self, '_pending_zoom_pos')
                # Need to wait until label layout is done to compute ratios reliably,
                # but we can try to guess or simply center for now if layout hasn't run.
                # Just call toggle_zoom() for spacebar or center, it's safer.
                # Actually, we can just use the toggle_zoom logic.
                self.toggle_zoom()
            else:
                self.toggle_zoom()

            # If we're on a half-size thumbnail, ensure full resolution preserves the zoom
            pm_max_dim = max(pixmap.width(), pixmap.height())
            if pm_max_dim < 3000 and not self.fit_to_window:
                self._pending_zoom_restore = True
                self._restore_zoom_center = self.zoom_center_point
                self._restore_zoom_level = self.current_zoom_level

            self._maybe_refresh_focus_subject_outline_after_display()
            if hasattr(self, "loading_overlay"):
                self.loading_overlay.hide_loading()
            return  # Don't continue with normal display logic

        # Zoom / navigation restore: _restore_zoom_center alone misses pinch (center may still be unset).
        has_restore_zoom = (
            getattr(self, "_preserve_nav_zoom_active", False)
            or getattr(self, "_pending_zoom_restore", False)
            or getattr(self, "_restore_zoom_center", None) is not None
            or getattr(self, "_restore_zoom_level", None) is not None
        )
        logger.info(
            f"[DISPLAY_PIXMAP] Checking zoom restoration - has_restore_zoom: {has_restore_zoom}, "
            f"preserve_nav_zoom: {getattr(self, '_preserve_nav_zoom_active', False)}, "
            f"pending_zoom_restore: {getattr(self, '_pending_zoom_restore', False)}, "
            f"_restore_zoom_center: {getattr(self, '_restore_zoom_center', None)}, "
            f"_restore_zoom_level: {getattr(self, '_restore_zoom_level', None)}, "
            f"fit_to_window: {self.fit_to_window}, pixmap_size: {pixmap.width()}x{pixmap.height()}"
        )
        if has_restore_zoom:
            if hasattr(self, "_maintain_zoom_on_navigation"):
                with contextlib.suppress(AttributeError):
                    delattr(self, "_maintain_zoom_on_navigation")
            logger.info("[DISPLAY_PIXMAP] Zoom restoration pending, preserving zoom state")
        elif needs_resolution_crossfade and self._single_view_is_fit_mode():
            # Fit-mode resolution upgrade: refit the sharper buffer once (crossfade handles the blend).
            if getattr(self, "gpu_view", None) is None:
                self.current_zoom_level = 1.0
                self.zoom_center_point = None
                self.scale_image_to_fit()
        elif not hasattr(self, '_maintain_zoom_on_navigation'):
            # CRITICAL: Check current fit_to_window state before resetting
            # If user has zoomed in (fit_to_window = False), preserve that state
            # This prevents zoom state from being lost when navigating from a zoomed image
            if self.fit_to_window:
                # User is in fit-to-window mode, safe to reset
                logger.debug("display_pixmap: fit_to_window=True, resetting to fit-to-window")
                self.current_zoom_level = 1.0
                self.zoom_center_point = None
                self.scale_image_to_fit()
            else:
                # User has zoomed in (fit_to_window = False), preserve zoom state
                # This happens when navigating from a zoomed image - the zoom state
                # will be saved by navigate_to_next_image() after load_raw_image() completes
                logger.info(f"[DISPLAY_PIXMAP] fit_to_window=False, preserving zoom state (zoom_level={self.current_zoom_level}, "
                           f"zoom_center_point={self.zoom_center_point})")
                # Check if this is a half_size image - if so, temporarily show fit-to-window
                # and wait for full resolution before applying zoom
                pixmap_max_dim = max(pixmap.width(), pixmap.height())
                raw_exts = {'.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2',
                            '.pef', '.srw', '.x3f', '.raf', '.3fr', '.fff', '.iiq',
                            '.cap', '.erf', '.mef', '.mos', '.nrw', '.rwl', '.srf'}
                is_raw = os.path.splitext(getattr(self, "current_file_path", ""))[-1].lower() in raw_exts
                is_pixmap_half_size = (pixmap_max_dim < 3000) and is_raw
                if is_pixmap_half_size:
                    logger.info(f"[DISPLAY_PIXMAP] Half-size image ({pixmap.width()}x{pixmap.height()}), "
                               "showing Fit preview until a full-resolution buffer arrives (avoid fake zoom upscale).")
                    if not getattr(self, "_pending_zoom", False):
                        self._pending_zoom = True
                        self._pending_zoom_center = self.zoom_center_point
                        self._pending_zoom_thumbnail_size = pixmap.size()
                    self.fit_to_window = True
                    self.current_zoom_level = 1.0
                    self.zoom_center_point = None
                    self.scale_image_to_fit()
                else:
                    # Full resolution image - apply zoom now
                    logger.info("[DISPLAY_PIXMAP] Full resolution image, applying zoom immediately")
                    self.apply_zoom_and_pan()
        else:
            if self.fit_to_window:
                self.scale_image_to_fit()
            else:
                self.apply_zoom_and_pan()
            delattr(self, '_maintain_zoom_on_navigation')

        # Handle zoom restoration
        # Handle zoom restoration
        if has_restore_zoom:
            self.fit_to_window = False
            logger.info(f"[DISPLAY_PIXMAP] Processing zoom restoration - half_size={getattr(self, '_is_half_size_displayed', False)}, "
                       f"pixmap_size={pixmap.width()}x{pixmap.height()}, "
                       f"_restore_zoom_center={self._restore_zoom_center}, "
                       f"_restore_zoom_level={getattr(self, '_restore_zoom_level', None)}")

            # If restoring zoom and currently displaying half_size, load full resolution FIRST
            pixmap_max_dim = max(pixmap.width(), pixmap.height())
            raw_exts = {'.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2',
                        '.pef', '.srw', '.x3f', '.raf', '.3fr', '.fff', '.iiq',
                        '.cap', '.erf', '.mef', '.mos', '.nrw', '.rwl', '.srf'}
            is_raw = os.path.splitext(getattr(self, "current_file_path", ""))[-1].lower() in raw_exts
            is_pixmap_half_size = (pixmap_max_dim < 3000) and is_raw

            if (hasattr(self, '_is_half_size_displayed') and self._is_half_size_displayed) or is_pixmap_half_size:
                if is_pixmap_half_size:
                    self._is_half_size_displayed = True

                if not hasattr(self, '_full_resolution_loading') or not self._full_resolution_loading:
                    # Check if full resolution is already cached
                    cached_full = self.image_cache.get_full_image(self.current_file_path)
                    if cached_full is not None:
                        cached_max_dim = max(cached_full.shape[1], cached_full.shape[0])
                        if cached_max_dim >= 3000:
                            logger.info("[DISPLAY_PIXMAP] Full resolution image already cached, loading immediately for zoom restoration...")
                            self._full_resolution_loading = True
                            # Store zoom restoration intent BEFORE displaying
                            self._pending_zoom_restore = True
                            self._pending_zoom_center = self._restore_zoom_center
                            self._pending_zoom_level = self._restore_zoom_level
                            self._restore_zoom_center = None
                            self._restore_zoom_level = None
                            self.display_numpy_image(cached_full)
                            self._is_half_size_displayed = False
                            self._full_resolution_loading = False
                            self._maybe_refresh_focus_subject_outline_after_display()
                            return # Zoom restoration handled by display_numpy_image
                        else:
                            # Start loading full resolution in background
                            logger.info("[DISPLAY_PIXMAP] Cached image is half_size, starting full resolution load for zoom restoration")
                            self._load_full_resolution_on_demand()
                            self._pending_zoom_restore = True
                            self._pending_zoom_center = self._restore_zoom_center
                            self._pending_zoom_level = self._restore_zoom_level
                            self._restore_zoom_center = None
                            self._restore_zoom_level = None
                            # Like 40b9ade: avoid a two-step zoomed-soft-thumb UX -- show preview fit until sharp buffer arrives.
                            if hasattr(self, '_maintain_zoom_on_navigation'):
                                delattr(self, '_maintain_zoom_on_navigation')
                            self.fit_to_window = True
                            self.current_zoom_level = 1.0
                            self.zoom_center_point = None
                            self.scale_image_to_fit()
                            logger.info(
                                "[DISPLAY_PIXMAP] Preview fit-to-window while full resolution loads for zoom restore "
                                f"(pending center={self._pending_zoom_center}, level={self._pending_zoom_level})"
                            )
                            if hasattr(self, "loading_overlay"):
                                self.loading_overlay.hide_loading()
                            return
                    else:
                        # Fallback: start loading full resolution
                        logger.info("[DISPLAY_PIXMAP] No cached full resolution, starting load for zoom restoration")
                        self._load_full_resolution_on_demand()
                        self._pending_zoom_restore = True
                        self._pending_zoom_center = self._restore_zoom_center
                        self._pending_zoom_level = self._restore_zoom_level
                        self._restore_zoom_center = None
                        self._restore_zoom_level = None
                        if hasattr(self, '_maintain_zoom_on_navigation'):
                            delattr(self, '_maintain_zoom_on_navigation')
                        self.fit_to_window = True
                        self.current_zoom_level = 1.0
                        self.zoom_center_point = None
                        self.scale_image_to_fit()
                        logger.info(
                            "[DISPLAY_PIXMAP] Preview fit-to-window while full resolution loads for zoom restore "
                            f"(pending center={self._pending_zoom_center}, level={self._pending_zoom_level})"
                        )
                        if hasattr(self, "loading_overlay"):
                            self.loading_overlay.hide_loading()
                        return

            # Effective zoom target: half->full deferral moves *_restore_* into _pending_* before this path.
            eff_level = getattr(self, "_restore_zoom_level", None)
            if eff_level is None:
                eff_level = getattr(self, "_pending_zoom_level", None)
            eff_center = getattr(self, "_restore_zoom_center", None)
            if eff_center is None:
                eff_center = getattr(self, "_pending_zoom_center", None)

            self.current_zoom_level = eff_level or 1.0

            # Scale zoom center point if pixmap size changed (half-size -> full-res)
            if hasattr(self, '_restore_pixmap_size') and self._restore_pixmap_size is not None and eff_center is not None:
                old_size = self._restore_pixmap_size
                new_size = pixmap.size()

                # Only scale if sizes are different
                if old_size.width() != new_size.width() or old_size.height() != new_size.height():
                    scale_x = new_size.width() / old_size.width() if old_size.width() > 0 else 1.0
                    scale_y = new_size.height() / old_size.height() if old_size.height() > 0 else 1.0

                    scaled_center = QPoint(
                        int(eff_center.x() * scale_x),
                        int(eff_center.y() * scale_y)
                    )
                    logger.debug(f"Scaled zoom center from {eff_center} ({old_size.width()}x{old_size.height()}) to {scaled_center} ({new_size.width()}x{new_size.height()})")
                    self.zoom_center_point = scaled_center
                else:
                    self.zoom_center_point = eff_center

                self._restore_pixmap_size = None
            else:
                c = eff_center
                self.zoom_center_point = (
                    c
                    if c is not None
                    else QPoint(max(0, pixmap.width() // 2), max(0, pixmap.height() // 2))
                )
            self._restore_zoom_center = None
            self._restore_zoom_level = None
            if getattr(self, "_pending_zoom_restore", False):
                self._pending_zoom_restore = False
            if hasattr(self, "_pending_zoom_center"):
                delattr(self, "_pending_zoom_center")
            if hasattr(self, "_pending_zoom_level"):
                delattr(self, "_pending_zoom_level")

            # Apply saved scroll position before applying zoom and pan
            if hasattr(self, '_restore_start_scroll_x') and hasattr(self, '_restore_start_scroll_y') and self._restore_start_scroll_x is not None and self._restore_start_scroll_y is not None:
                    self.start_scroll_x = self._restore_start_scroll_x
                    self.start_scroll_y = self._restore_start_scroll_y

            self.apply_zoom_and_pan()
            self._finish_nav_zoom_preserve()

        # Handle pending zoom from double-click on thumbnail
        # Check actual pixmap size to be extra safe
        pending_thumb_size = getattr(self, "_pending_zoom_thumbnail_size", None)
        old_max = max(pending_thumb_size.width(), pending_thumb_size.height()) if pending_thumb_size is not None else 0
        new_max = max(pixmap.width(), pixmap.height())
        size_upgraded = (old_max <= 0) or (new_max > int(old_max * 1.02))
        pending_zoom_ready = not getattr(self, "_full_resolution_loading", False)

        if (
            hasattr(self, '_pending_zoom')
            and self._pending_zoom
            and pending_zoom_ready
            and size_upgraded
        ):
            if getattr(self, "gpu_view", None) is not None:
                # Handled inside _apply_gpu_view for the GPU path to avoid asynchronous race conditions
                pass
            else:
                logger.info("[DISPLAY_PIXMAP] Handling pending zoom with full resolution image")
                target = self._resolve_pending_zoom_center_for_pixmap(pixmap)
                if target is not None:
                    self.zoom_center_point = target
                self.fit_to_window = False
                self.current_zoom_level = 1.0
                self.zoom_to_point()
                self._clear_pending_point_zoom_state()


        # Update status bar immediately with EXIF data
        # Don't pass dimensions - let update_status_bar use original dimensions from cache
        # update_status_bar will read EXIF data from cache automatically
        self._sync_displayed_half_size_flag(cur_path)
        self.update_status_bar()
        self._maybe_refresh_focus_subject_outline_after_display()
        if hasattr(self, "loading_overlay"):
            self.loading_overlay.hide_loading()

        # Track image fully loaded and rendered
        display_time = time.time() - display_start
        exif_track = None
        try:
            if cur_path:
                exif_track = self.image_cache.get_exif(cur_path)
        except Exception:
            pass
        is_full_res = image_covers_sensor_resolution(
            pixmap.width(), pixmap.height(), exif_track
        )
        logger.info(f"[TRACK] Image completely loaded and rendered - file: {current_file}, size: {pixmap.width()}x{pixmap.height()}, full_res: {is_full_res}, time: {display_time:.3f}s")

    def _on_single_view_content_displayed(self):
        """Called when a single-view image or preview is successfully painted on screen.
        Triggers deferred background indexing prep.
        """
        try:
            if not getattr(self, "_single_view_first_render_logged", False):
                req_ts = float(getattr(self, "_single_view_load_request_ts", 0.0) or 0.0)
                req_path = str(getattr(self, "_single_view_load_request_path", "") or "")
                cur_path = str(getattr(self, "current_file_path", "") or "")
                if req_ts > 0.0 and req_path and _norm_path(req_path) == _norm_path(cur_path):
                    pm = getattr(self, "current_pixmap", None)
                    pm_w = int(pm.width()) if pm is not None and not pm.isNull() else 0
                    pm_h = int(pm.height()) if pm is not None and not pm.isNull() else 0
                    if pm_w > 0 and pm_h > 0:
                        elapsed = max(0.0, time.time() - req_ts)
                        # Signal, not a poll: _on_first_render_ack stops the
                        # timeout timer and records TTFR/NAVTIME. The guard
                        # above (plus the flag set in the slot) keeps this
                        # idempotent, so a repeated paint cannot re-fire it.
                        self._single_view_first_render.emit(
                            cur_path, elapsed, pm_w, pm_h
                        )
        except Exception:
            pass

        # Restart the idle full-res timer for the current image.
        with contextlib.suppress(Exception):
            self._restart_full_res_idle_timer()

        if self._should_defer_nav_preload():
            self._schedule_post_session_restore_prefetch()
        elif not self._raw_recovery_preview_active():
            # Fit + embedded-JPEG workflow: do not pull sensor-covering 6K–8K
            # immediately after TTFR -- that saturates the drive and delays the
            # urgent ±1 neighbor prefetch that first arrow-key nav needs.
            # Zoomed / RAW-consistent paths still flush full right away.
            defer_full_for_neighbors = False
            try:
                from common_image_loader import use_libraw_consistent_preview_first

                fp_now = getattr(self, "current_file_path", None)
                defer_full_for_neighbors = bool(
                    fp_now
                    and self._single_view_is_fit_mode()
                    and not use_libraw_consistent_preview_first(fp_now)
                )
            except Exception:
                defer_full_for_neighbors = False
            if defer_full_for_neighbors:
                self._flush_deferred_full_exif_after_first_paint()
                self._schedule_idle_full_decode_after_nav_preview(
                    getattr(self, "current_file_path", None) or ""
                )
            else:
                self._flush_deferred_sensor_full_decode()
                self._flush_deferred_full_exif_after_first_paint()

        # Always warm next/prev immediately (even during session-restore defer).
        with contextlib.suppress(Exception):
            self._queue_urgent_neighbor_prefetch()

        self._schedule_deferred_filmstrip_flush()

        try:
            fp = getattr(self, "current_file_path", None)
            if fp:
                cached = self.image_cache.get_exif(fp)
                if cached and cached.get("minimal_preview_exif"):
                    self._apply_preview_first_status_bar(cached)
        except Exception:
            pass
        QTimer.singleShot(1500, self._prefetch_filmstrip_thumbnails)
        if not self._should_defer_nav_preload():
            self._schedule_idle_display_prefetch()
            with contextlib.suppress(Exception):
                self._queue_display_tier_prefetch_for_paths(
                    self._paths_within_nav_prefetch_radius()
                )
        from search import metadata_auto_index_enabled
        if metadata_auto_index_enabled() and not self._semantic_indexing_in_progress and not getattr(self, "_semantic_index_prep_in_progress", False) and not getattr(self, "_metadata_index_run_done", False):
                    delay_ms = metadata_index_idle_delay_ms()
                    self._metadata_index_defer_timer.start(delay_ms)

    def _touch_single_view_interactivity(self) -> None:
        """Mark recent single-view interaction (nav/zoom/recovery) for index deferral."""
        import time

        self._last_single_view_interactivity_ts = time.time()
        self._single_view_index_defer_opened_ts = 0.0

    def _restore_single_view_keyboard_focus(self) -> None:
        """Backward-compatible alias."""
        if getattr(self, "view_mode", "single") == "single":
            self._restore_keyboard_focus()
