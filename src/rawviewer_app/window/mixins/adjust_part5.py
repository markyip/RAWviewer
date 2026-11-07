"""adjust mixin, part 5 of 5 (13 methods, 537 lines)."""
from __future__ import annotations

import contextlib
import os
import sys

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QTransform

from core.config import _norm_path

from .. import lazy
from ..adjust_workers import _AdjustCompareOriginalWorker, _make_fast_preview_base
from ..qt_helpers import _get_bg_thread_pool

lazy.register(sys.modules[__name__])


class AdjustPart5:
    def _on_adjust_panel_editing_finished(self, adj: dict) -> None:
        path = getattr(self, "current_file_path", None)
        if not path:
            return
        # The drag/click gesture just settled -- snap the preview to full
        # quality (undoes the fast/downsampled base used while it was still
        # live-dragging). Runs in the background same as any other preview
        # request; the (comparatively slow) XMP write below doesn't block it.
        self._apply_adjust_panel_preview(full_quality=True)
        try:
            from raw_adjustments import load_adjustments_for_file, write_xmp_adjustments_for_file
            from raw_adjustments.dodge_burn import (
                DEFAULT_STRENGTH,
                MASK_KEY,
                MASK_OBJ_KEY,
                STRENGTH_KEY,
                serialize_mask,
            )
            from raw_adjustments.spot_heal import (
                MASK_KEY as HEAL_MASK_KEY,
            )
            from raw_adjustments.spot_heal import (
                MASK_OBJ_KEY as HEAL_MASK_OBJ_KEY,
            )
            from raw_adjustments.spot_heal import serialize_mask as serialize_heal_mask

            # Loaded unconditionally: the no-change check below needs it, and
            # it used to be bound only inside the undo branch -- so with undo
            # tracking disabled the comparison would have raised NameError.
            old_adj = load_adjustments_for_file(path)
            if not getattr(self, "_disable_undo_tracking", False):
                if not hasattr(self, "_undo_stack"):
                    self._undo_stack = {}
                history = self._undo_stack.setdefault(path, [])
                if not history or history[-1] != old_adj:
                    history.append(old_adj)
                    if len(history) > 50:
                        history.pop(0)

            # Persist PNG for XMP only here -- preview ticks use an in-memory
            # fingerprint via _dodge_burn_overlay_adj (no encode per slider).
            adj = dict(adj)
            adj.pop(MASK_OBJ_KEY, None)
            adj.pop(HEAL_MASK_OBJ_KEY, None)
            mask = getattr(self, "_dodge_burn_mask", None)
            if mask is not None and not mask.is_empty:
                adj[MASK_KEY] = serialize_mask(mask)
                adj.setdefault(STRENGTH_KEY, DEFAULT_STRENGTH)
            else:
                adj.pop(MASK_KEY, None)

            heal = getattr(self, "_spot_heal_mask", None)
            if heal is not None and not heal.is_empty:
                adj[HEAL_MASK_KEY] = serialize_heal_mask(heal)
            else:
                adj.pop(HEAL_MASK_KEY, None)

            # Nothing actually changed? Then do not write, and above all do
            # not invalidate. editing_finished fires on opening and closing
            # the panel, not only on an edit, and the invalidate below drops
            # EVERY tier for this file including the full image -- so simply
            # visiting the editor and going back forced a fresh 33MP decode.
            # Measured on a real session: 24 full-image misses, every one of
            # them "invalidated", none from eviction, with the cache sitting
            # at 0/8.
            if self._adjustments_match(adj, old_adj):
                self._skip_editing_finished_side_effects = True
            else:
                self._skip_editing_finished_side_effects = False
                write_xmp_adjustments_for_file(path, adj)
        except Exception as exc:
            if hasattr(self, "status_bar"):
                self.status_bar.showMessage(
                    f"Could not save adjustments: {exc}", 5000
                )
            return
        # Disabled (2026-07): reported color artifacts in the re-encoded
        # preview. See _reencode_persisted_preview_for_sidecar's docstring;
        # re-enable once root-caused.
        # self._reencode_persisted_preview_for_sidecar(path, adj)
        from raw_adjustments import is_default_adjustments

        cleared = is_default_adjustments(adj)
        self._invalidate_gallery_edited_memo(path)
        if hasattr(self, "status_bar"):
            self.status_bar.showMessage(
                "Edits cleared" if cleared else "Adjustments saved to XMP sidecar",
                2500,
            )
        self._refresh_gallery_bookmark_visuals()
        # Every cache tier that can hold an edits-baked-in render of this file
        # (gallery's own composited thumbnail pixmap, and the shared
        # preview/grid/full-image/pixmap tiers _paint_instant_preview_for_path
        # reads for the gallery->single instant paint) must drop its entry
        # here. Without this the just-saved edits were invisible both back in
        # the gallery grid and on the FIRST click into single view -- both
        # read a pre-edit cached render until something else happened to
        # evict it.
        if not getattr(self, "_skip_editing_finished_side_effects", False):
            try:
                from image_cache import get_image_cache

                get_image_cache().invalidate_file(path)
            except Exception:
                pass
        # Bake editor-aligned browse thumbs AFTER invalidate so gallery /
        # single-view match the Adjust panel without re-running the linear
        # pipeline on the embedded JPEG (which diverges from the demosaic
        # path and must never touch companion JPEGs of the same basename).
        #
        # Skip bake on Reset / all-defaults: ``_apply_adjust_panel_preview``
        # above is async, so ``_adjust_last_render`` is still the *previous*
        # edited frame -- re-baking it would put the edited look back into
        # gallery/filmstrip after the XMP sidecar was cleared.
        if not cleared:
            # Defer bake until the full-quality settle lands in
            # _on_adjust_preview_ready (instant-gain / stale last_render
            # must not be written into browse caches).
            self._pending_browse_bake_path = path
        else:
            self._pending_browse_bake_path = None
            # Drop the stale edited frame so a later accidental bake cannot
            # resurrect it; gallery reload will fetch original thumbs.
            if _norm_path(getattr(self, "_adjust_last_render_norm", "") or "") == _norm_path(
                path
            ):
                self._adjust_last_render = None
                self._adjust_last_render_norm = None
            if _norm_path(getattr(self, "_adjust_hires_render_norm", "") or "") == _norm_path(
                path
            ):
                self._adjust_hires_render = None
                self._adjust_hires_render_norm = None
        gj = getattr(self, "gallery_justified", None)
        if gj is not None and hasattr(gj, "invalidate_thumbnails_for_path"):
            with contextlib.suppress(Exception):
                gj.invalidate_thumbnails_for_path(path)
        bar = self._filmstrip_bar() if hasattr(self, "_filmstrip_bar") else None
        if bar is not None and hasattr(bar, "invalidate_thumbnails_for_path"):
            with contextlib.suppress(Exception):
                bar.invalidate_thumbnails_for_path(path)
        elif bar is not None and hasattr(bar, "refresh_visible_thumbnails"):
            with contextlib.suppress(Exception):
                bar.refresh_visible_thumbnails(refresh_cache=True)
        QTimer.singleShot(0, self._restore_keyboard_focus)

    def _bake_browse_caches_for_visited_edit(self, path: str) -> None:
        """Refresh browse tiles after merely VIEWING an edited photo.

        _persist_editor_aligned_browse_caches only runs from editing_finished,
        so it needs a change to save. This covers the other case: a photo
        carrying edits from a previous session, opened and closed untouched.

        Deliberately narrow. It bakes only when a full-quality render for this
        exact file is already in hand -- no decoding, no pipeline pass, just
        the resize and encode that the save path performs -- and only when the
        sidecar actually holds non-default edits, so an unedited photo never
        rewrites its own cache.
        """
        try:
            from common_image_loader import is_raw_file

            if not path or not is_raw_file(path):
                return
            last = getattr(self, "_adjust_last_render", None)
            if not last or not isinstance(last, tuple) or last[0] is None:
                return
            if _norm_path(getattr(self, "_adjust_last_render_norm", "") or "") != _norm_path(path):
                return
            array = last[0]
            # Same floor the save-time bake uses: a small interim render must
            # not be written into browse caches as if it were the real thing.
            if getattr(array, "ndim", 0) != 3 or max(array.shape[0], array.shape[1]) < 1200:
                return

            from raw_adjustments import is_default_adjustments, load_adjustments_for_file

            if is_default_adjustments(load_adjustments_for_file(path)):
                return
            self._persist_editor_aligned_browse_caches(path)
            gj = getattr(self, "gallery_justified", None)
            if gj is not None and hasattr(gj, "invalidate_thumbnails_for_path"):
                gj.invalidate_thumbnails_for_path(path)
        except Exception:
            import logging

            logging.getLogger(__name__).debug(
                "[BAKE] visited-edit bake skipped for %s", path, exc_info=True
            )

    def _on_adjust_recovery_baseline_requested(self) -> None:
        path = getattr(self, "current_file_path", None)
        panel = getattr(self, "single_image_adjust_panel", None)
        if not path or panel is None:
            return
        from common_image_loader import is_raw_file

        if not is_raw_file(path):
            if hasattr(self, "status_bar"):
                self.status_bar.showMessage(
                    "Recovery baseline is only available for RAW/DNG files", 3500
                )
            return
        if getattr(self, "_raw_recovery_active", False):
            self._raw_recovery_active = False
            self._sync_raw_recovery_view_mode()
        panel.set_recovery_baseline(True)
        self._apply_adjust_panel_preview(full_quality=True)
        if hasattr(self, "status_bar"):
            self.status_bar.showMessage(
                "Recovery look on -- Shadows/Highlights readouts updated; "
                "tweak Exposure / Sat / HSL from here",
                6000,
            )
        QTimer.singleShot(0, self._restore_keyboard_focus)

    def _reset_adjust_compare_state(self) -> None:
        """Drop the cached compare-with-original render and exit split view.

        Called whenever the edit base changes (new file, or a fresh editing
        session) since a cached "original" render for the previous base would
        otherwise silently show the wrong image's before/after split.
        """
        self._adjust_compare_original_pixmap = None
        self._adjust_compare_original_path = None
        panel = getattr(self, "single_image_adjust_panel", None)
        if panel is not None:
            panel.set_compare_active(False)
        gv = getattr(self, "gpu_view", None)
        if gv is not None:
            gv.set_compare_mode(False)

    def _on_adjust_compare_toggled(self, checked: bool) -> None:
        """Arm/disarm the compare-with-original split view (GPU single view only)."""
        panel = getattr(self, "single_image_adjust_panel", None)
        gv = getattr(self, "gpu_view", None)
        if panel is None or gv is None:
            return
        if not checked:
            gv.set_compare_mode(False)
            return
        path = getattr(self, "current_file_path", None)
        base = getattr(self, "_adjust_preview_base_rgb", None)
        if not path or base is None or not hasattr(base, "shape"):
            panel.set_compare_active(False)
            if hasattr(self, "status_bar"):
                self.status_bar.showMessage(
                    "Compare needs a decoded RAW edit base -- try again in a moment",
                    3500,
                )
            return
        norm = _norm_path(path)
        cached = getattr(self, "_adjust_compare_original_pixmap", None)
        if cached is not None and getattr(self, "_adjust_compare_original_path", None) == norm:
            gv.set_compare_original_pixmap(cached)
            gv.set_compare_mode(True)
            return
        if hasattr(self, "status_bar"):
            self.status_bar.showMessage("Rendering original for comparison…", 0)
        geo = None
        try:
            geo = panel.get_adjustments() if panel is not None else None
        except Exception:
            geo = None
        worker = _AdjustCompareOriginalWorker(
            path, base, self._adjust_compare_original_signals, geometry_adj=geo
        )
        _get_bg_thread_pool().start(worker)

    def _on_adjust_compare_original_ready(self, file_path: str, array) -> None:
        panel = getattr(self, "single_image_adjust_panel", None)
        gv = getattr(self, "gpu_view", None)
        if gv is None:
            return
        if _norm_path(file_path) != _norm_path(getattr(self, "current_file_path", "") or ""):
            return
        if array is None:
            if panel is not None:
                panel.set_compare_active(False)
            if hasattr(self, "status_bar"):
                self.status_bar.showMessage("Could not render original for comparison", 3500)
            return
        pixmap = self._numpy_to_qpixmap(array)
        if pixmap is None or pixmap.isNull():
            return
        self._adjust_compare_original_pixmap = pixmap
        self._adjust_compare_original_path = _norm_path(file_path)
        gv.set_compare_original_pixmap(pixmap)

        is_compare_active = panel is not None and getattr(panel, "_compare_btn", None) is not None and panel._compare_btn.isChecked()
        if is_compare_active:
            gv.set_compare_mode(True)
            if hasattr(self, "status_bar"):
                self.status_bar.clearMessage()

    def _on_auto_tone_requested(self) -> None:
        """One Auto: WB + Light sliders + straighten, one undo, no colour grade.

        Saturation / Vibrance / HSL stay put so colour is edited by hand after
        Auto (Color / WB section and HSL mixer).
        """
        panel = getattr(self, "single_image_adjust_panel", None)
        base = getattr(self, "_adjust_preview_base_rgb", None)
        if panel is None or base is None or not hasattr(base, "shape"):
            if hasattr(self, "status_bar"):
                self.status_bar.showMessage(
                    "Auto needs a decoded edit base -- try again in a moment",
                    3000,
                )
            return
        from raw_adjustments import solve_white_balance_from_sample, wb_reference_temperature
        from raw_adjustments.auto_adjust import estimate_auto_tone, estimate_neutral_rgb, estimate_straighten_angle
        from raw_adjustments.pipeline import _linear_float_from_buffer, linear_to_display_uint8

        fast = _make_fast_preview_base(base)
        lin = _linear_float_from_buffer(fast)
        updates: dict[str, float] = {}
        parts: list[str] = []
        skipped: list[str] = []

        est = estimate_neutral_rgb(lin)
        if est is None:
            skipped.append("WB")
        else:
            r, g, b = est
            ref_temp = wb_reference_temperature(panel.get_adjustments())
            temperature, tint = solve_white_balance_from_sample(r, g, b, ref_temp)
            updates["Temperature"] = float(temperature)
            updates["Tint"] = float(tint)
            parts.append(f"WB {int(round(temperature))}K")

        tone = estimate_auto_tone(lin)
        if tone is None:
            skipped.append("tone")
        else:
            for key, value in tone.items():
                updates[key] = float(value)
            ev = float(tone.get("Exposure2012", 0.0))
            parts.append(f"tone {ev:+.2f} EV")

        rgb = None
        try:
            adj0 = dict(panel.get_adjustments())
            adj0.update(updates)
            adj0["CropAngle"] = 0.0
            adj0["PerspectiveVertical"] = 0.0
            adj0["PerspectiveHorizontal"] = 0.0
            rgb = linear_to_display_uint8(fast, adj0)
        except Exception:
            rgb = None
        if rgb is None:
            last = getattr(self, "_adjust_last_render", None)
            if last is not None and last[0] is not None:
                rgb = last[0]
        if rgb is None:
            skipped.append("straighten")
        else:
            angle = estimate_straighten_angle(rgb)
            if angle is None:
                skipped.append("straighten")
            else:
                import numpy as np

                new_angle = float(np.clip(angle, -45.0, 45.0))
                updates["CropAngle"] = new_angle
                parts.append(f"straighten {new_angle:+.1f}°")

        if not updates:
            if hasattr(self, "status_bar"):
                self.status_bar.showMessage(
                    "Auto: already balanced -- colour is in Color / WB and HSL",
                    3500,
                )
            return
        panel.apply_auto_tone(updates)
        msg = "Auto: " + ", ".join(parts)
        if skipped:
            msg += f" (skipped {', '.join(skipped)})"
        msg += " -- grade colour in Color / WB · Ctrl+Z undoes"
        if hasattr(self, "status_bar"):
            self.status_bar.showMessage(msg, 5000)

    def _sync_adjust_panel(self) -> None:
        panel = getattr(self, "single_image_adjust_panel", None)
        if panel is None:
            return
        if getattr(self, "view_mode", "") != "single":
            panel.hide()
            return
        pm = getattr(self, "current_pixmap", None)
        if (pm is None or pm.isNull()) and not self._single_view_has_display_image():
            panel.hide()
            return
        if getattr(self, "_adjust_user_hidden", True):
            self._adjust_overlay_visible = False
            panel.hide()
            return
        panel.setVisible(bool(getattr(self, "_adjust_overlay_visible", False)))
        container = getattr(self, "single_view_container", None)
        if container is not None and hasattr(container, "relayout_adjust"):
            container.relayout_adjust()

    def _get_visual_rotation_degrees(self, file_path=None) -> int:
        """Get current non-destructive clockwise visual rotation for a file."""
        fp = file_path or getattr(self, "current_file_path", None)
        if not fp:
            return 0
        k = _norm_path(fp)
        return int(self._visual_rotation_degrees.get(k, 0)) % 360

    def _apply_visual_rotation_for_current(self, pixmap: QPixmap) -> QPixmap:
        """Apply per-file visual rotation to the pixmap for on-screen display only."""
        if pixmap is None or pixmap.isNull():
            return pixmap
        degrees = self._get_visual_rotation_degrees()
        if degrees == 0:
            return pixmap
        transform = QTransform()
        # Stored ``degrees`` is clockwise (each button click +90° CW). pixmap.transformed()
        # with rotate(angle) yields a clockwise on-screen rotation for positive angle here.
        transform.rotate(degrees)
        return pixmap.transformed(transform, Qt.TransformationMode.SmoothTransformation)

    def _rotate_raster_pil_cw90(self, path: str) -> None:
        from PIL import Image, ImageOps

        tmp = path + ".rawviewer_rotate_tmp"
        im = None
        try:
            im = Image.open(path)
            im = ImageOps.exif_transpose(im)
            im = im.transpose(Image.Transpose.ROTATE_270)
            ext = os.path.splitext(path)[1].lower()
            save_kw = {}
            if ext in (".jpg", ".jpeg"):
                save_kw = {"quality": 95, "subsampling": 0, "optimize": True}
            if ext in (".jpg", ".jpeg"):
                im.save(tmp, "JPEG", **save_kw)
            elif ext in (".png",):
                im.save(tmp, "PNG", optimize=True)
            elif ext in (".webp",):
                im.save(tmp, "WEBP", quality=95, method=6)
            elif ext in (".bmp",):
                im.save(tmp, "BMP")
            elif ext in (".tif", ".tiff"):
                im.save(tmp, "TIFF", compression="tiff_lzw")
            else:
                im.save(tmp)
            os.replace(tmp, path)
        except Exception:
            if os.path.isfile(tmp):
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
            raise
        finally:
            if im is not None:
                im.close()

    def _rotate_raw_pyexiv2_meta_cw90(self, path: str) -> None:
        from metadata.backend import rotate_exif_orientation_meta_cw90

        rotate_exif_orientation_meta_cw90(path)

    def _rotate_current_image_clockwise_persist(self):
        """Rotate current image visually by 90° clockwise (non-destructive)."""
        path = getattr(self, "current_file_path", None)
        if not path or not os.path.isfile(path):
            self.status_bar.showMessage("No image to rotate", 2000)
            return
        self._stop_slideshow()
        k = _norm_path(path)
        current = int(self._visual_rotation_degrees.get(k, 0)) % 360
        updated = (current + 90) % 360
        if updated == 0:
            self._visual_rotation_degrees.pop(k, None)
        else:
            self._visual_rotation_degrees[k] = updated

        gj = getattr(self, "gallery_justified", None)
        if gj is not None:
            gallery_path = (
                gj._resolve_gallery_path(path)
                if hasattr(gj, "_resolve_gallery_path")
                else path
            )
            if hasattr(gj, "on_visual_rotation_changed"):
                with contextlib.suppress(Exception):
                    gj.on_visual_rotation_changed(gallery_path or path)
            elif hasattr(gj, "refresh_visible_tile_for_path"):
                with contextlib.suppress(Exception):
                    gj.refresh_visible_tile_for_path(gallery_path or path)

        bar = self._filmstrip_bar()
        if bar is not None:
            if hasattr(bar, "on_visual_rotation_changed"):
                with contextlib.suppress(Exception):
                    bar.on_visual_rotation_changed(path)
            elif hasattr(bar, "refresh_cell_for_path"):
                with contextlib.suppress(Exception):
                    bar.refresh_cell_for_path(path)

        # Re-render from the current base pixmap if available; otherwise load from cache/source.
        base_pixmap = getattr(self, "_base_display_pixmap", None)
        prev_w = prev_h = 0
        cur_pm = getattr(self, "current_pixmap", None)
        if cur_pm is not None and not cur_pm.isNull():
            prev_w, prev_h = cur_pm.width(), cur_pm.height()
        if base_pixmap is not None and not base_pixmap.isNull():
            self.display_pixmap(base_pixmap)
            pm_max = max(base_pixmap.width(), base_pixmap.height())
            self._manager_displayed_max_dim = max(
                getattr(self, "_manager_displayed_max_dim", 0), pm_max
            )
        else:
            self.load_raw_image(path)
        new_pm = getattr(self, "current_pixmap", None)
        aspect_swapped = (
            prev_w > 0
            and prev_h > 0
            and new_pm is not None
            and not new_pm.isNull()
            and (
                new_pm.width() != prev_w
                or new_pm.height() != prev_h
            )
        )
        if aspect_swapped:
            gv = getattr(self, "gpu_view", None)
            if gv is not None and gv.is_fit_mode():
                gv.fit_to_window()
            elif getattr(self, "fit_to_window", False):
                self.scale_image_to_fit()
        self._save_persisted_visual_rotations()
        self.status_bar.showMessage(f"Rotated view {updated}°", 1800)
