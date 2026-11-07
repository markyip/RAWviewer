"""adjust mixin, part 3 of 5 (31 methods, 892 lines)."""
from __future__ import annotations

import contextlib
import os
import sys

from PyQt6.QtGui import QPixmap

from core.config import _norm_path

from .. import lazy
from ..adjust_workers import (
    _AdjustCompareOriginalWorker,
    _AdjustEditBaseWorker,
    _AdjustMaskLoadWorker,
    _make_fast_preview_base,
)
from ..lazy import np
from ..qt_helpers import _get_bg_thread_pool

lazy.register(sys.modules[__name__])


class AdjustPart3:
    def _close_adjust_panel_from_ui(self) -> None:
        """Close Adjust and restore browse workflow if the editor forced RAW.

        Esc used to call ``_set_adjust_panel_visible(False)`` only, which left
        ``use_embedded_jpeg_workflow=False`` behind -- the next gallery click
        then loaded in RAW mode. E-toggle already restored; share that path.

        Gallery close sets ``view_mode`` to gallery *before* this runs, so
        ``toggle_raw_jpeg_workflow()`` (which no-ops outside single view)
        cannot restore the setting -- write it directly in that case.
        """
        if not getattr(self, "_adjust_overlay_visible", False):
            return
        self._set_adjust_panel_visible(False)
        path = getattr(self, "current_file_path", None)
        if path:
            # Load sidecar on the reencode worker -- never on the UI thread.
            self._reencode_persisted_preview_for_sidecar(path, None)
            # Opening an already-edited photo and closing without touching
            # anything used to leave the gallery tile as the embedded JPEG:
            # the tile is only refreshed by the save-time bake, which needs an
            # actual edit, and every other refresh path is gated on
            # SIDECAR_ADJUST (off by default). So a photo edited in an earlier
            # session showed unedited in the grid no matter how often you
            # opened it. Bake on close instead, when there is a settled render
            # to bake from.
            self._bake_browse_caches_for_visited_edit(path)
        if getattr(self, "_adjust_forced_raw_workflow", False):
            self._adjust_forced_raw_workflow = False
            settings = self.get_settings()
            use_embedded = settings.value("use_embedded_jpeg_workflow", True, type=bool)
            if use_embedded:
                return
            if getattr(self, "view_mode", "single") == "single":
                self.toggle_raw_jpeg_workflow()
                return
            # Already left single view (e.g. Esc→gallery): flip the setting
            # without a single-view reload.
            settings.setValue("use_embedded_jpeg_workflow", True)
            try:
                from common_image_loader import invalidate_libraw_consistent_preview_settings

                invalidate_libraw_consistent_preview_settings()
            except Exception:
                pass
            self._update_raw_toggle_button_state()

    def _sync_adjust_panel_for_current_file(self, *, force: bool = False) -> None:
        panel = getattr(self, "single_image_adjust_panel", None)
        path = getattr(self, "current_file_path", None)
        if panel is None or not path:
            return
        norm = _norm_path(path)
        if not force and norm == getattr(self, "_adjust_panel_loaded_norm", None):
            return
        db_serial = ""
        heal_serial = ""
        has_edit = False
        try:
            from raw_adjustments import is_default_adjustments, load_adjustments_for_file
            from raw_adjustments.dodge_burn import MASK_KEY
            from raw_adjustments.spot_heal import MASK_KEY as HEAL_MASK_KEY

            # XML parse only -- cheap. The mask blobs (base64 + PNG, at edit
            # base resolution, can be multi-MB) are decoded further below on
            # a worker thread. That decode alone isn't the reason opening an
            # edited image is slow, though -- the dominant cost (measured:
            # 2.8-5.5s on a 32MP file) is the full adjustment render itself
            # (WB/tone/curves/denoise/dodge-burn) kicked off below, whether
            # or not the file has a dodge/burn or heal mask at all.
            loaded_adj = load_adjustments_for_file(path)
            has_edit = not is_default_adjustments(loaded_adj)
            panel.set_adjustments(loaded_adj)
            self._adjust_panel_loaded_norm = norm
            self._pending_adjust_preview = dict(panel.get_adjustments())
            self._dodge_burn_luma_guide = None

            db_serial = str(loaded_adj.get(MASK_KEY, "") or "")
            heal_serial = str(loaded_adj.get(HEAL_MASK_KEY, "") or "")
            # Always clear the PREVIOUS file's mask objects synchronously,
            # even when this file has its own mask to decode -- the actual
            # decode happens async further below (off the UI thread), and
            # without this clear, a render kicked off before that finishes
            # (e.g. _apply_adjust_panel_preview a few lines down, on a
            # cached edit base) would silently paint with the prior file's
            # stale mask instead of no mask / a "not yet loaded" state. That
            # was a real correctness bug, not just a missing loading
            # indicator: db_apply/db=1 was observed on the very first
            # preview render for a freshly-navigated file, before the async
            # decode could possibly have completed -- only explainable by a
            # leftover mask object from whichever file was open before.
            self._dodge_burn_mask = None
            self._spot_heal_mask = None
            panel.set_dodge_burn_mask_present(False)
            if hasattr(panel, "set_spot_heal_mask_present"):
                panel.set_spot_heal_mask_present(False)
            self._sync_dodge_burn_mask_overlay()
        except Exception:
            # Do NOT swallow silently: a failure here leaves the panel showing
            # default sliders for a file that has a sidecar -- the exact
            # "previous XMP didn't load" symptom -- with zero diagnostic trail.
            import logging

            logging.getLogger(__name__).warning(
                "[ADJUST] Failed to load XMP adjustments into panel for %s",
                os.path.basename(path),
                exc_info=True,
            )
        if panel.isVisible():
            self._request_adjust_edit_base(path)
            # Applying a saved edit means a full adjustment render (measured
            # 2.8-5.5s on a 32MP file) before the photo on screen reflects
            # the sidecar -- show an overlay over the image for that window
            # so it doesn't look like a stall / like the edit was lost.
            # Cleared in _on_adjust_preview_ready once a render that actually
            # carries this file's local masks lands -- NOT merely the first
            # render, which is kicked off below while the mask decode is
            # still in flight and therefore shows the photo without the
            # brushwork. Must also clear when the new file has NO edit,
            # otherwise the flag/overlay from a previously-loaded edited
            # file would linger with nothing left to dismiss it.
            self._adjust_loading_edit_wants_db = bool(db_serial)
            self._adjust_loading_edit_wants_heal = bool(heal_serial)
            # While a mask decode is in flight, suppress the full-quality
            # render that would otherwise fire here (and in
            # _on_adjust_edit_base_ready): it cannot include the masks yet,
            # so its multi-second result is thrown away the moment
            # _on_adjust_mask_load_ready lands and re-renders. Opening an
            # edited image was paying for TWO full renders back to back
            # (observed 5545ms then 2804ms on one file). The mask handler
            # clears this and issues the single render that counts.
            self._adjust_mask_load_pending = bool(db_serial or heal_serial)
            self._set_edit_loading_overlay(norm if has_edit else None)
            if (
                getattr(self, "_adjust_preview_base_rgb", None) is not None
                and norm == getattr(self, "_adjust_edit_base_path", None)
            ):
                # Cached base: immediately fetch and display lens status
                try:
                    from processing.unified import UnifiedImageProcessor
                    processor = UnifiedImageProcessor()
                    has_profile = processor.lens_profile_available(path)
                    profile_name = ""
                    if has_profile:
                        from raw_adjustments.lens_correction import get_lens_profile_name
                        exif_data = processor.exif_extractor.extract_exif_data(path)
                        profile_name = get_lens_profile_name(exif_data)
                    panel.set_lens_correction_available(has_profile, profile_name)
                except Exception:
                    pass
                if not getattr(self, "_adjust_mask_load_pending", False):
                    self._apply_adjust_panel_preview(full_quality=True)

        if db_serial or heal_serial:
            self._adjust_mask_load_gen = (
                int(getattr(self, "_adjust_mask_load_gen", 0)) + 1
            )
            gen = self._adjust_mask_load_gen
            worker = _AdjustMaskLoadWorker(
                gen, path, db_serial, heal_serial, self._adjust_mask_load_signals
            )
            _get_bg_thread_pool().start(worker)

    def _render_reflects_loaded_edit(self, adj_used) -> bool:
        """True when a finished render already carried the file's local masks.

        The first render after entering the editor is submitted while the
        mask PNG decode is still on a worker thread, so it paints the photo
        WITHOUT the brushwork -- dismissing the loading overlay on it would
        drop the overlay before the edit is actually on screen. Renders pick
        the masks up via _dodge_burn_overlay_adj, which injects the live mask
        objects, so their presence is what marks a render as complete.
        """
        if adj_used is None:
            return False
        from raw_adjustments.dodge_burn import MASK_OBJ_KEY
        from raw_adjustments.spot_heal import MASK_OBJ_KEY as HEAL_MASK_OBJ_KEY

        if getattr(self, "_adjust_loading_edit_wants_db", False) and adj_used.get(MASK_OBJ_KEY) is None:
            return False
        return not (getattr(self, "_adjust_loading_edit_wants_heal", False) and adj_used.get(HEAL_MASK_OBJ_KEY) is None)

    def _on_adjust_mask_load_ready(
        self, generation: int, file_path: str, db_mask, heal_mask
    ) -> None:
        """Background decode of the dodge/burn + heal mask blobs landed.

        Fills in the actual mask objects (superseded by a later open/nav if
        ``generation``/``file_path`` no longer match) and re-renders once,
        since the preview kicked off while this was still in flight ran
        without the local-edit masks baked in. Does NOT touch the "Loading
        previous edit..." status message -- that's owned by
        _on_adjust_preview_ready (cleared once the actual render this mask
        feeds into lands), since mask decode is typically much faster than
        the full adjustment render and clearing here could dismiss the
        message while the render is still genuinely in flight.
        """
        if int(generation) != int(getattr(self, "_adjust_mask_load_gen", 0)):
            return
        if _norm_path(file_path) != _norm_path(getattr(self, "current_file_path", "") or ""):
            return
        # Masks are in hand: renders may proceed again (they were suppressed
        # while this decode was in flight to avoid a throwaway full render).
        self._adjust_mask_load_pending = False
        self._dodge_burn_mask = db_mask
        self._spot_heal_mask = heal_mask
        # Reconcile what the overlay is still waiting for with what actually
        # decoded: a blob that failed to decode (or came back empty) will
        # never appear in a render, so leaving the flag set would strand the
        # loading overlay forever.
        if getattr(self, "_adjust_loading_edit_norm", None) is not None:
            self._adjust_loading_edit_wants_db = (
                db_mask is not None and not db_mask.is_empty
            )
            self._adjust_loading_edit_wants_heal = (
                heal_mask is not None and not heal_mask.is_empty
            )
        panel = getattr(self, "single_image_adjust_panel", None)
        if panel is not None:
            panel.set_dodge_burn_mask_present(
                db_mask is not None and not db_mask.is_empty
            )
            if hasattr(panel, "set_spot_heal_mask_present"):
                panel.set_spot_heal_mask_present(
                    heal_mask is not None and not heal_mask.is_empty
                )
        self._sync_dodge_burn_mask_overlay()
        if panel is not None and panel.isVisible():
            self._apply_adjust_panel_preview(full_quality=True)

    def _refresh_adjust_panel_for_navigation(self, file_path: str | None = None) -> None:
        """Reload XMP sliders and edit base when the current file changes with panel open."""
        if not self._adjust_panel_active():
            return
        path = file_path or getattr(self, "current_file_path", None)
        if not path:
            return
        self._pending_adjust_preview = None
        self._sync_adjust_panel_for_current_file(force=True)
        self._begin_adjust_editing_session(path)

    def _begin_adjust_editing_session(self, file_path: str) -> None:
        """Cancel competing decodes and load a fresh LibRaw half-res edit base."""
        if not file_path:
            return
        with contextlib.suppress(Exception):
            self.image_manager.cancel_task(file_path)
        # Cancelling the task above can strand the generic single-view
        # "Loading Image..." overlay forever: it was shown by the reload
        # that toggle_raw_jpeg_workflow() just dispatched (opening the
        # editor forces RAW mode first), and its own completion callback
        # -- the only thing that would hide it -- never fires once its
        # task is cancelled here. The edit-base decode reports its own
        # progress via the status bar, so the overlay isn't needed for it.
        if hasattr(self, "loading_overlay"):
            self.loading_overlay.hide_loading()
        self._defer_sensor_full_decode_path = None
        self._sensor_full_decode_queued_norm = None
        self._adjust_preview_base_rgb = None
        self._adjust_preview_base_rgb_fast = None
        self._adjust_edit_base_path = None
        self._adjust_edit_base_loading_norm = None
        self._reset_adjust_compare_state()
        self._request_adjust_edit_base(file_path)

    def _adjust_panel_blocks_display_refresh(self) -> bool:
        """While the adjust panel is open, block async loader repaints of the current file."""
        if not getattr(self, "_adjust_overlay_visible", False):
            return False
        panel = getattr(self, "single_image_adjust_panel", None)
        return panel is not None and panel.isVisible()

    def _should_defer_adjust_preview_paint(self, pixmap: QPixmap) -> bool:
        """True when a live-drag lite tier would trash a zoomed sharper buffer.

        Last-run log: Fit settle at 3514px → double-click 100% → slider ticks
        painted 640px lite previews with exact_framing, so 100% zoom showed a
        soft postage stamp until the next ~3s settle. Also blocks lite while
        the WB dropper is armed so pick coords stay on a settled frame.
        """
        if pixmap is None or pixmap.isNull():
            return False
        gv = getattr(self, "gpu_view", None)
        picking = bool(gv is not None and gv.is_color_pick_mode())
        if not picking and self._single_view_is_fit_mode():
            return False
        cur = getattr(self, "current_pixmap", None)
        if cur is None or cur.isNull():
            return False
        cur_max = max(int(cur.width()), int(cur.height()))
        new_max = max(int(pixmap.width()), int(pixmap.height()))
        return cur_max >= 1200 and new_max < int(cur_max * 0.55)

    def _adjust_display_pixel_size(self) -> tuple[int, int] | None:
        """Width/height of the Adjust frame currently on screen (any tier)."""
        pm = getattr(self, "current_pixmap", None)
        if pm is not None and not pm.isNull():
            return int(pm.width()), int(pm.height())
        last = getattr(self, "_adjust_last_render", None)
        if last is not None and last[0] is not None and getattr(last[0], "ndim", 0) == 3:
            return int(last[0].shape[1]), int(last[0].shape[0])
        gv = getattr(self, "gpu_view", None)
        if gv is not None and gv.has_pixmap():
            try:
                w, h = gv.image_size()
                if int(w) > 0 and int(h) > 0:
                    return int(w), int(h)
            except Exception:
                pass
        return None

    def _restore_adjust_hires_before_zoom(self) -> bool:
        """If Fit live-drag left a 640px buffer on screen, restore settle first.

        Last-run: status showed ``640x427 - 100%`` after double-click because
        lite had replaced the half-res settle while still in Fit.
        """
        if not self._adjust_panel_active():
            return False
        hires = getattr(self, "_adjust_hires_render", None)
        if hires is None:
            return False
        if _norm_path(getattr(self, "_adjust_hires_render_norm", "") or "") != _norm_path(
            getattr(self, "current_file_path", "") or ""
        ):
            return False
        arr = hires[0]
        if arr is None or getattr(arr, "ndim", 0) != 3:
            return False
        h, w = int(arr.shape[0]), int(arr.shape[1])
        if max(h, w) < 1200:
            return False
        cur = getattr(self, "current_pixmap", None)
        if cur is not None and not cur.isNull() and max(h, w) <= max(int(cur.width()), int(cur.height())) * 1.05:
                return False
        try:
            pixmap = self._numpy_to_qpixmap(arr)
        except Exception:
            return False
        if pixmap is None or pixmap.isNull():
            return False
        self._display_adjust_preview_pixmap(pixmap)
        return True

    def _display_adjust_preview_pixmap(self, pixmap: QPixmap) -> None:
        """Paint adjust live-preview without navigation cross-fades or loader races."""
        if pixmap is None or pixmap.isNull():
            return
        if self._should_defer_adjust_preview_paint(pixmap):
            return
        if getattr(self, "_raw_recovery_active", False):
            self._raw_recovery_active = False
            self._sync_raw_recovery_view_mode()
        # No defensive copy: every caller hands over a pixmap produced by
        # QPixmap.fromImage() (_numpy_to_qpixmap / converter workers), which
        # already deep-copies into pixmap-owned storage. The copy() here was
        # a third full-buffer memcpy (~50MB at half-size) on the UI thread
        # per preview tick.
        self._adjust_preview_painting = True
        try:
            self._orientation_already_applied = True
            self._base_display_pixmap = pixmap
            pixmap = self._apply_visual_rotation_for_current(pixmap)
            self.current_pixmap = pixmap
            path = getattr(self, "current_file_path", None)
            if path:
                self._displayed_content_path = path
            if self._uses_gpu_single_view():
                gv = getattr(self, "gpu_view", None)
                if gv is not None:
                    preserve = not self._single_view_is_fit_mode()
                    # exact_framing: adjust-preview tiers (full-res display
                    # buffer -> fast downsampled base -> half-size edit base)
                    # are same-content re-renders at different resolutions;
                    # without it, set_pixmap's snap-to-100% heuristic zooms
                    # the view out on every slider tick while zoomed in.
                    gv.set_pixmap(pixmap, preserve_view=preserve, exact_framing=True)
                    gv.viewport().update()
            else:
                self.image_label.setPixmap(pixmap)
                self.image_label.resize(pixmap.size())
            timer = getattr(self, "_adjust_hist_sync_timer", None)
            if timer is not None:
                # Debounce harder during Adjust paints (instant-gain / D&B /
                # lite ticks) -- 120ms still re-ran hist nearly every throttle.
                timer.start(220)
            else:
                self._sync_single_image_histogram()
        finally:
            self._adjust_preview_painting = False

    def _request_adjust_edit_base(self, file_path: str) -> None:
        """Load LibRaw demosaic buffer for adjust preview (not embedded JPEG)."""
        from common_image_loader import is_raw_file

        if not file_path:
            self._adjust_preview_base_rgb = None
            self._adjust_preview_base_rgb_fast = None
            self._adjust_edit_base_path = None
            return
        norm = _norm_path(file_path)
        if (
            norm == getattr(self, "_adjust_edit_base_path", None)
            and getattr(self, "_adjust_preview_base_rgb", None) is not None
        ):
            return
        if getattr(self, "_adjust_edit_base_loading_norm", None) == norm:
            return
        if not is_raw_file(file_path):
            base = self.image_cache.get_full_image(file_path)
            if base is not None and hasattr(base, "shape"):
                self._adjust_preview_base_rgb = base
                self._adjust_edit_base_path = norm
                self._apply_adjust_panel_preview(full_quality=True)
            return
        self._adjust_edit_base_loading_norm = norm
        self._adjust_preview_base_rgb = None
        self._adjust_preview_base_rgb_fast = None
        self._adjust_edit_base_path = None
        if hasattr(self, "status_bar"):
            self.status_bar.showMessage(
                f"Loading RAW for editing: {os.path.basename(file_path)}…", 0
            )
        pool = getattr(getattr(self, "image_manager", None), "_process_pool", None)
        panel = getattr(self, "single_image_adjust_panel", None)
        lens_correction_on = False
        if panel is not None:
            lens_correction_on = (
                float(panel.get_adjustments().get("LensCorrectionEnabled", 0.0) or 0.0) > 0.5
            )
        worker = _AdjustEditBaseWorker(
            file_path,
            pool,
            self._adjust_edit_base_signals,
            use_full_resolution=False,
            apply_lens_correction=lens_correction_on,
        )
        _get_bg_thread_pool().start(worker)

    def _on_adjust_edit_base_ready(self, file_path: str, base, has_lens_profile: bool, lens_profile_name: str) -> None:
        norm = _norm_path(file_path)
        # Only clear the in-flight guard when it belongs to THIS completion:
        # clearing unconditionally let a PREVIOUS image's late completion
        # unlock a duplicate decode request for the current one -- every
        # navigation with the panel open ran decode_raw_edit_base twice
        # (doubled "[EDIT] Linear edit base" log lines), and the redundant
        # unpacks queued behind the global rawpy lock inflated unpack times
        # to seconds.
        if getattr(self, "_adjust_edit_base_loading_norm", None) == norm:
            self._adjust_edit_base_loading_norm = None
        if norm != _norm_path(getattr(self, "current_file_path", "") or ""):
            return
        panel = getattr(self, "single_image_adjust_panel", None)
        if panel is not None:
            panel.set_lens_correction_available(has_lens_profile, lens_profile_name)
            self._refresh_camera_profile_banner(panel, norm)
        if base is None or not hasattr(base, "shape"):
            # Silent handling: an edit-base decode failure used to just show
            # a 5s error and leave the editor stuck on the broken file.
            # decode_raw_edit_base already tried LibRaw then the embedded-JPEG
            # fallback; None means both failed (corrupt file, no embed, …).
            # Mark it so browse skips a doomed LibRaw retry, then move on
            # instead of leaving the editor stuck.
            try:
                from processing.unified import _LIBRAW_UNSUPPORTED_PATHS

                _LIBRAW_UNSUPPORTED_PATHS.add(os.path.normcase(os.path.abspath(file_path)))
            except Exception:
                pass
            if hasattr(self, "status_bar"):
                self.status_bar.showMessage(
                    f"Skipped {os.path.basename(file_path)} (not editable)", 3000
                )
            # No render will follow for this file, so nothing else would ever
            # dismiss an "applying saved edit" overlay raised for it.
            if norm == getattr(self, "_adjust_loading_edit_norm", None):
                self._set_edit_loading_overlay(None)
            # (norm already matches current_file_path -- checked above.)
            if self._adjust_panel_active() and self.image_files and len(self.image_files) > 1:
                self.navigate_to_next_image()
            return
        self._adjust_preview_base_rgb = base
        self._adjust_edit_base_path = norm
        self._adjust_preview_base_rgb_fast = _make_fast_preview_base(base)
        if panel is not None and panel.isVisible():
            self._pending_adjust_preview = dict(panel.get_adjustments())
            # See _sync_adjust_panel_for_current_file: skip while the mask
            # decode is still running, otherwise this render lands without
            # the brushwork and is immediately superseded.
            if not getattr(self, "_adjust_mask_load_pending", False):
                self._apply_adjust_panel_preview(full_quality=True)

            # Kick off compare pre-render asynchronously
            geo = None
            try:
                geo = panel.get_adjustments()
            except Exception:
                geo = None
            worker = _AdjustCompareOriginalWorker(
                file_path, base, self._adjust_compare_original_signals, geometry_adj=geo
            )
            _get_bg_thread_pool().start(worker)
        if hasattr(self, "status_bar"):
            h, w = base.shape[0], base.shape[1]
            import numpy as _np

            if getattr(base, "dtype", None) == _np.uint8:
                # RAW demosaic returns float32/uint16; uint8 here means the
                # embedded-JPEG fallback was used instead (e.g. HE/HE*-
                # compressed NEF, or a linear/composite DNG panorama).
                self.status_bar.showMessage(
                    f"Editing embedded preview (RAW demosaic unavailable) ({w}×{h})", 4000
                )
            else:
                self.status_bar.showMessage(
                    f"Linear RAW edit base ready ({w}×{h})", 2500
                )
        self._push_lut_chooser_previews()

    def _push_lut_chooser_previews(self) -> None:
        """Feed Looks-list LUT thumbs from a tiny as-shot display of this photo."""
        panel = getattr(self, "single_image_adjust_panel", None)
        path = getattr(self, "current_file_path", None)
        if panel is None or not path or not hasattr(panel, "set_lut_preview_source"):
            return
        if (
            str(getattr(panel, "_lut_thumb_path", "") or "") == str(path)
            and getattr(panel, "_lut_thumb_display_u8", None) is not None
        ):
            return
        try:
            from raw_adjustments.lut import lut_chooser_as_shot_display_u8

            base = getattr(self, "_adjust_preview_base_rgb_fast", None)
            if base is None:
                base = getattr(self, "_adjust_preview_base_rgb", None)
            if base is None:
                return
            tiny = lut_chooser_as_shot_display_u8(base, None)
            if tiny is None:
                return
            panel.set_lut_preview_source(tiny, path, 0)
        except Exception:
            pass

    def _capture_adjust_preview_base(self, file_path: str | None) -> None:
        """Edit base comes from LibRaw decode when the adjust panel opens."""
        return

    def _ensure_adjust_preview_base(self, file_path: str | None):
        if (
            file_path
            and _norm_path(file_path) == getattr(self, "_adjust_edit_base_path", None)
            and getattr(self, "_adjust_preview_base_rgb", None) is not None
        ):
            return self._adjust_preview_base_rgb
        if file_path:
            self._request_adjust_edit_base(file_path)
        return getattr(self, "_adjust_preview_base_rgb", None)

    def _on_adjust_panel_preview_changed(self, adj: dict) -> None:
        # Continuous/throttled live-drag path -- the one caller that can use
        # the default full_quality=False (downsampled fast base). Every other
        # caller of _apply_adjust_panel_preview is a discrete, one-shot action
        # (file load, recovery-baseline click, post-drag settle) and passes
        # full_quality=True explicitly.
        #
        # Exception: Compare mode must always use full_quality here too.
        # GpuImageView._img_w/_img_h (used to crop the *original* comparison
        # pixmap to the correct width) are updated from whatever pixmap
        # set_pixmap() last received -- if a live-drag tick delivers a
        # downsampled fast-preview pixmap, _img_w/_img_h shrink to match it,
        # and the next _update_compare_overlay() crops the full-resolution
        # original using coordinates sized for that *much smaller* frame,
        # showing a tiny corner of it stretched across the divider (reported
        # as the original side appearing "zoomed in" while dragging the tone
        # curve with Compare on). The fast/full pixmap sizes only ever match
        # by construction when full_quality is used, so Compare mode simply
        # always gets the full-res path -- see docs/EDIT_PIPELINE.md.
        gv = getattr(self, "gpu_view", None)
        comparing = bool(gv is not None and gv.is_compare_mode())
        self._pending_adjust_preview = dict(adj or {})
        # Zero-latency gain path: when the delta vs the last exact render is
        # ONLY Exposure/Temperature/Tint, those are (near-)pure per-channel
        # gains in the BT.709 power region -- apply them as a 256-entry LUT
        # on the cached rendered frame and paint immediately, every tick,
        # with no worker round-trip or throttle. The exact render below
        # still runs (throttled) and replaces the approximation when it
        # lands, so the image converges to the true pipeline output.
        # Compare overlay already scales the original to the edited pixmap
        # size (_update_compare_overlay), so drag can stay on the 640px lite
        # path; settle still upgrades via editing_finished → full_quality.
        if not comparing:
            self._try_instant_gain_preview(adj)
            self._try_instant_transform_preview(adj)
        self._apply_adjust_panel_preview(full_quality=False)

    def _try_instant_gain_preview(self, adj: dict) -> bool:
        last = getattr(self, "_adjust_last_render", None)
        if last is None:
            return False
        last_arr, last_adj = last
        if last_arr is None or _norm_path(
            getattr(self, "current_file_path", "") or ""
        ) != getattr(self, "_adjust_last_render_norm", None):
            return False
        try:
            # Fold in the same local-mask keys the render path injects via
            # _dodge_burn_overlay_adj. panel.get_adjustments() never carries
            # them, but the cached last_adj (a snapshot of what was submitted
            # to the worker) always does -- so without this the diff below
            # trips on _dodge_burn_mask_obj / _dodge_burn_mask_v1 (None vs a
            # value) and bails on EVERY tick for any image with a dodge/burn
            # or heal mask, silently disabling the zero-latency path exactly
            # where the full render is slowest (measured 2.8-5.5s).
            # The mask objects themselves are skipped: identity is not
            # meaningful to compare and str()-ing one renders its whole
            # ndarray. The cheap mem:HxW:vN fingerprint beside them already
            # encodes shape + mutation version, so a mask edited since the
            # cached render still correctly fails the comparison and falls
            # through to a real re-render.
            with contextlib.suppress(Exception):
                adj = self._dodge_burn_overlay_adj(adj)
            from raw_adjustments.dodge_burn import MASK_OBJ_KEY as _DB_MASK_OBJ_KEY
            from raw_adjustments.spot_heal import MASK_OBJ_KEY as _HEAL_MASK_OBJ_KEY

            keys = (set(adj) | set(last_adj)) - {_DB_MASK_OBJ_KEY, _HEAL_MASK_OBJ_KEY}
            for k in keys:
                if k in self._INSTANT_GAIN_KEYS:
                    continue
                a, b = adj.get(k), last_adj.get(k)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    if abs(float(a) - float(b)) > 1e-4:
                        return False
                elif str(a or "") != str(b or ""):
                    return False

            import cv2

            from raw_adjustments import _kelvin_to_rgb

            inv_g = 1.0 / 2.222
            f = 2.0 ** (
                (float(adj.get("Exposure2012", 0.0)) - float(last_adj.get("Exposure2012", 0.0)))
                / 2.222
            )
            gains = [f, f, f]
            t_new = float(adj.get("Temperature", 0.0))
            t_old = float(last_adj.get("Temperature", 0.0))
            if t_new > 1000.0 and t_old > 1000.0 and abs(t_new - t_old) > 0.5:
                rn, gn, bn = _kelvin_to_rgb(t_new)
                ro, go, bo = _kelvin_to_rgb(t_old)
                if min(gn, go, ro, bo) > 1e-5:
                    gains[0] *= ((rn / ro) / (gn / go)) ** inv_g
                    gains[2] *= ((bn / bo) / (gn / go)) ** inv_g
            tint_new = float(adj.get("Tint", 0.0))
            tint_old = float(last_adj.get("Tint", 0.0))
            if abs(tint_new - tint_old) > 1e-4:
                denom = 1.0 - (tint_old / 150.0) * 0.1
                if abs(denom) > 1e-5:
                    gains[1] *= ((1.0 - (tint_new / 150.0) * 0.1) / denom) ** inv_g

            if all(abs(g - 1.0) < 1e-5 for g in gains):
                return True  # nothing visible changed; skip repaint
            ramp = np.arange(256, dtype=np.float64)
            lut = np.stack(
                [np.clip(ramp * g + 0.5, 0, 255).astype(np.uint8) for g in gains],
                axis=-1,
            ).reshape(256, 1, 3)
            approx = cv2.LUT(last_arr, lut)
            pixmap = self._numpy_to_qpixmap(approx)
            if pixmap is not None and not pixmap.isNull():
                self._display_adjust_preview_pixmap(pixmap)
                return True
        except Exception:
            return False
        return False

    def _try_instant_transform_preview(self, adj: dict) -> bool:
        """Warp a cached identity-geometry display frame on the UI thread.

        When only Transform keys changed, skip the full preview worker round-trip
        for immediate feedback; the throttled worker still converges to the
        exact pipeline output.
        """
        src = getattr(self, "_adjust_transform_source_u8", None)
        src_adj = getattr(self, "_adjust_transform_source_adj", None)
        if src is None or src_adj is None:
            return False
        if _norm_path(getattr(self, "current_file_path", "") or "") != getattr(
            self, "_adjust_transform_source_norm", None
        ):
            return False
        try:
            keys = set(adj) | set(src_adj)
            for k in keys:
                if k in self._INSTANT_TRANSFORM_KEYS:
                    continue
                a, b = adj.get(k), src_adj.get(k)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    if abs(float(a) - float(b)) > 1e-4:
                        return False
                elif str(a or "") != str(b or ""):
                    return False
            # Need at least one transform key to have moved.
            moved = False
            for k in self._INSTANT_TRANSFORM_KEYS:
                a, b = adj.get(k), src_adj.get(k)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    if abs(float(a) - float(b)) > 1e-4:
                        moved = True
                        break
                elif str(a or "") != str(b or ""):
                    moved = True
                    break
            if not moved:
                return False

            from raw_adjustments.transform import apply_geometry

            geo_adj = dict(adj)
            if getattr(self, "_crop_preview_uncropped", False):
                for k in ("CropLeft", "CropRight", "CropTop", "CropBottom"):
                    geo_adj[k] = 0.0
            out = apply_geometry(src, geo_adj, preview=True)
            if out is None or getattr(out, "size", 0) == 0:
                return False
            pixmap = self._numpy_to_qpixmap(out)
            if pixmap is not None and not pixmap.isNull():
                self._display_adjust_preview_pixmap(pixmap)
                return True
        except Exception:
            return False
        return False

    def _on_dodge_burn_mode_changed(self, mode) -> None:
        gv = getattr(self, "gpu_view", None)
        if gv is not None:
            gv.set_dodge_burn_mode(mode is not None)
            if mode is not None:
                self._sync_dodge_burn_brush_cursor()
        self._sync_dodge_burn_mask_overlay()

    def _sync_dodge_burn_brush_cursor(self) -> None:
        panel = getattr(self, "single_image_adjust_panel", None)
        gv = getattr(self, "gpu_view", None)
        if panel is None or gv is None or not hasattr(gv, "set_dodge_burn_brush_radius"):
            return
        gv.set_dodge_burn_brush_radius(panel.dodge_burn_brush_radius())
        if hasattr(gv, "set_dodge_burn_brush_flow"):
            gv.set_dodge_burn_brush_flow(panel.dodge_burn_brush_strength())

    def _on_dodge_burn_brush_size_wheel(self, wheel_delta: int) -> None:
        """Trackpad/wheel while Dodge/Burn is armed → Brush Size, not navigate."""
        panel = getattr(self, "single_image_adjust_panel", None)
        if panel is None or panel.dodge_burn_mode() is None:
            return
        if hasattr(panel, "nudge_dodge_burn_brush_size"):
            panel.nudge_dodge_burn_brush_size(int(wheel_delta))
        else:
            # Fallback if panel build is older than this hook.
            slider = getattr(panel, "_db_size_slider", None)
            if slider is None:
                return
            amount = 8 if wheel_delta > 0 else -8
            slider.setValue(
                max(slider.minimum(), min(slider.maximum(), slider.value() + amount))
            )

    def _on_adjust_ai_preview_requested(self) -> None:
        self._apply_adjust_panel_preview(full_quality=True, ai_preview=True)

    def _on_adjust_ai_denoise_toggled(self, enabled: bool) -> None:
        try:
            settings = self.get_settings()
            settings.setValue("ai_denoise_enabled", "1" if enabled else "0")
        except Exception:
            pass

    def _crop_keys_zeroed(self, adj: dict) -> dict:
        """Preview adj with crop insets cleared (full frame under the overlay)."""
        out = dict(adj)
        for k in ("CropLeft", "CropRight", "CropTop", "CropBottom"):
            out[k] = 0.0
        return out

    def _enter_geometry_overlay_preview(self, *, refresh: bool = True) -> None:
        """Show full-frame transform under the crop overlay (final outcome).

        Used while Crop mode is on, and whenever straighten/perspective is
        active so rotation does not hard-crop away context.
        """
        panel = getattr(self, "single_image_adjust_panel", None)
        gv = getattr(self, "gpu_view", None)
        if panel is None or gv is None:
            return
        insets = panel.get_crop_insets()
        self._crop_working_insets = insets
        self._crop_preview_uncropped = True
        # Live Transform drag stays on the lite path; settle upgrades later.
        fq = not bool(getattr(self, "_transform_interacting", False))
        if panel.is_crop_mode():
            if not gv.is_crop_mode():
                gv.set_crop_mode(True, insets=insets, aspect=None, refit=True)
            else:
                gv.set_crop_insets(*insets)
            if refresh:
                self._apply_adjust_panel_preview(full_quality=fq)
            return
        self._geometry_overlay_preview = True
        if not gv.is_crop_mode():
            gv.set_crop_mode(True, insets=insets, aspect=None, refit=True)
        else:
            gv.set_crop_insets(*insets)
        if refresh:
            self._apply_adjust_panel_preview(full_quality=fq)

    def _exit_geometry_overlay_preview(self, *, refresh: bool = True) -> None:
        """Leave temporary geometry overlay unless explicit Crop mode is on."""
        panel = getattr(self, "single_image_adjust_panel", None)
        gv = getattr(self, "gpu_view", None)
        if panel is not None and panel.is_crop_mode():
            return
        if not getattr(self, "_geometry_overlay_preview", False):
            self._crop_preview_uncropped = False
            if gv is not None and gv.is_crop_mode():
                gv.set_crop_mode(False)
            return
        self._geometry_overlay_preview = False
        self._crop_preview_uncropped = False
        if panel is not None:
            insets = getattr(self, "_crop_working_insets", None)
            if insets is not None:
                with contextlib.suppress(Exception):
                    panel.set_crop_insets(*insets, emit=False)
        if gv is not None:
            gv.set_crop_mode(False)
        if refresh:
            self._apply_adjust_panel_preview(full_quality=True)

    def _sync_geometry_overlay_with_adj(self, adj: dict | None) -> None:
        """Keep overlay preview in sync with warp keys outside explicit Crop mode."""
        panel = getattr(self, "single_image_adjust_panel", None)
        if panel is not None and panel.is_crop_mode():
            self._crop_preview_uncropped = True
            return
        if getattr(self, "_transform_interacting", False) or self._adj_has_warp(adj):
            if not getattr(self, "_geometry_overlay_preview", False):
                self._enter_geometry_overlay_preview(refresh=False)
            else:
                self._crop_preview_uncropped = True
        elif getattr(self, "_geometry_overlay_preview", False):
            self._exit_geometry_overlay_preview(refresh=False)

    def _on_crop_mode_changed(self, active: bool) -> None:
        panel = getattr(self, "single_image_adjust_panel", None)
        gv = getattr(self, "gpu_view", None)
        if panel is None or gv is None:
            return
        if active:
            self._geometry_overlay_preview = False
            panel.disarm_dodge_burn()
            insets = panel.get_crop_insets()
            self._crop_working_insets = insets
            gv.set_crop_mode(True, insets=insets, aspect=None)
            self._crop_preview_uncropped = True
            self._apply_adjust_panel_preview(full_quality=True)
            if hasattr(self, "status_bar"):
                self.status_bar.showMessage(
                    "Crop mode -- drag handles; pick a ratio to lock aspect; Apply to commit",
                    4000,
                )
        else:
            self._crop_preview_uncropped = False
            if gv.is_crop_mode() and not getattr(self, "_geometry_overlay_preview", False):
                gv.set_crop_mode(False)
            adj = panel.get_adjustments() if panel is not None else {}
            if self._adj_has_warp(adj):
                self._enter_geometry_overlay_preview(refresh=True)
            else:
                self._geometry_overlay_preview = False
                self._apply_adjust_panel_preview(full_quality=True)

    def _on_crop_aspect_changed(self, aspect) -> None:
        gv = getattr(self, "gpu_view", None)
        if gv is None or not gv.is_crop_mode():
            return
        if aspect == "original":
            w, h = gv.image_size()
            aspect = (float(w) / float(h)) if w > 0 and h > 0 else None
        gv.set_crop_aspect_ratio(aspect if isinstance(aspect, (int, float)) else None)
        self._crop_working_insets = gv.crop_insets()

    def _on_crop_insets_live(
        self, left: float, right: float, top: float, bottom: float
    ) -> None:
        self._crop_working_insets = (float(left), float(right), float(top), float(bottom))
        panel = getattr(self, "single_image_adjust_panel", None)
        # Geometry-overlay preview (straighten/perspective) is not explicit Crop
        # mode -- write insets through so the dimmed rect matches saved crop.
        if (
            panel is not None
            and getattr(self, "_geometry_overlay_preview", False)
            and not panel.is_crop_mode()
        ):
            with contextlib.suppress(Exception):
                panel.set_crop_insets(
                    float(left), float(right), float(top), float(bottom), emit=False
                )
