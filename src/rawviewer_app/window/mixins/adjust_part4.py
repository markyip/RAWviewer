"""adjust mixin, part 4 of 5 (19 methods, 894 lines)."""
from __future__ import annotations

import contextlib
import math
import sys
import time

from core.config import _norm_path

from .. import lazy
from ..adjust_workers import _AdjustPreviewWorker
from ..lazy import np
from ..qt_helpers import _get_bg_thread_pool

lazy.register(sys.modules[__name__])


class AdjustPart4:
    def _on_crop_apply_requested(self) -> None:
        panel = getattr(self, "single_image_adjust_panel", None)
        gv = getattr(self, "gpu_view", None)
        if panel is None:
            return
        insets = (
            gv.crop_insets()
            if gv is not None and gv.is_crop_mode()
            else getattr(self, "_crop_working_insets", panel.get_crop_insets())
        )
        self._geometry_overlay_preview = False
        self._crop_preview_uncropped = False
        panel.set_crop_mode_ui(False)
        if gv is not None:
            gv.set_crop_mode(False)
        panel.set_crop_insets(*insets, emit=True)
        adj = panel.get_adjustments()
        if self._adj_has_warp(adj):
            self._enter_geometry_overlay_preview(refresh=True)

    def _on_crop_cancel_requested(self) -> None:
        panel = getattr(self, "single_image_adjust_panel", None)
        gv = getattr(self, "gpu_view", None)
        self._geometry_overlay_preview = False
        self._crop_preview_uncropped = False
        if panel is not None:
            panel.set_crop_mode_ui(False)
        if gv is not None:
            gv.set_crop_mode(False)
        adj = panel.get_adjustments() if panel is not None else {}
        if self._adj_has_warp(adj):
            self._enter_geometry_overlay_preview(refresh=True)
        else:
            self._apply_adjust_panel_preview(full_quality=True)

    def _on_crop_reset_requested(self) -> None:
        panel = getattr(self, "single_image_adjust_panel", None)
        gv = getattr(self, "gpu_view", None)
        self._crop_working_insets = (0.0, 0.0, 0.0, 0.0)
        if gv is not None and gv.is_crop_mode():
            gv.set_crop_insets(0.0, 0.0, 0.0, 0.0)
        if panel is not None:
            panel.set_crop_insets(0.0, 0.0, 0.0, 0.0, emit=not panel.is_crop_mode())

    def _on_dodge_burn_mask_toggled(self, show: bool) -> None:
        self._sync_dodge_burn_mask_overlay()

    def _on_adjust_panel_reset(self) -> None:
        """The panel's Reset button rebuilds a clean adjustments dict, but
        local masks live here (not in that dict) and must be cleared too."""
        had_db = getattr(self, "_dodge_burn_mask", None) is not None
        had_heal = getattr(self, "_spot_heal_mask", None) is not None
        if not had_db and not had_heal:
            return
        self._dodge_burn_mask = None
        self._dodge_burn_luma_guide = None
        self._spot_heal_mask = None
        panel = getattr(self, "single_image_adjust_panel", None)
        if panel is not None:
            if had_db:
                panel.set_dodge_burn_mask_present(False)
            if had_heal and hasattr(panel, "set_spot_heal_mask_present"):
                panel.set_spot_heal_mask_present(False)
        self._sync_dodge_burn_mask_overlay()

    def _on_dodge_burn_clear_requested(self) -> None:
        if getattr(self, "_dodge_burn_mask", None) is None:
            return
        self._dodge_burn_mask = None
        self._dodge_burn_luma_guide = None
        panel = getattr(self, "single_image_adjust_panel", None)
        if panel is not None:
            panel.set_dodge_burn_mask_present(False)
        self._sync_dodge_burn_mask_overlay()
        # _on_adjust_panel_editing_finished already re-renders at full
        # quality; calling _apply_adjust_panel_preview here too raced with
        # it (the redundant in-flight request could tag the real one as
        # stale via the dirty-generation check in _on_adjust_preview_ready,
        # discarding its painted frame until an unrelated action requested
        # another render).
        if panel is not None:
            self._on_adjust_panel_editing_finished(panel.get_adjustments())

    def _on_spot_heal_clear_requested(self) -> None:
        if getattr(self, "_spot_heal_mask", None) is None:
            return
        self._spot_heal_mask = None
        panel = getattr(self, "single_image_adjust_panel", None)
        if panel is not None and hasattr(panel, "set_spot_heal_mask_present"):
            panel.set_spot_heal_mask_present(False)
        self._sync_dodge_burn_mask_overlay()
        # See _on_dodge_burn_clear_requested: skip the redundant direct
        # preview call, _on_adjust_panel_editing_finished already renders.
        if panel is not None:
            self._on_adjust_panel_editing_finished(panel.get_adjustments())

    def _active_local_mask_for_overlay(self):
        """Mask shown by the Mask overlay: heal when Heal is armed, else dodge/burn."""
        panel = getattr(self, "single_image_adjust_panel", None)
        mode = panel.dodge_burn_mode() if panel is not None else None
        if mode == "heal":
            return getattr(self, "_spot_heal_mask", None)
        return getattr(self, "_dodge_burn_mask", None)

    def _sync_dodge_burn_mask_overlay(self) -> None:
        """Push current mask (or empty) to the view when Show Mask is on."""
        gv = getattr(self, "gpu_view", None)
        if gv is None:
            return
        panel = getattr(self, "single_image_adjust_panel", None)
        show = bool(panel is not None and panel.dodge_burn_show_mask())
        gv.set_dodge_burn_mask_overlay(self._active_local_mask_for_overlay(), show)

    def _dodge_burn_mask_shape(self) -> tuple[int, int] | None:
        """Canonical (H, W) the mask is painted at: post-geometry edit frame.

        Matches the Adjust display framing (crop/straighten/keystone) so
        stroke mapping agrees with WB pick / on-screen pixels. Cached per
        base identity + transform keys.
        """
        base = getattr(self, "_adjust_preview_base_rgb", None)
        if base is None or not hasattr(base, "shape"):
            return None
        panel = getattr(self, "single_image_adjust_panel", None)
        adj = {}
        try:
            if panel is not None:
                adj = panel.get_adjustments() or {}
        except Exception:
            adj = {}
        from raw_adjustments.transform import TRANSFORM_KEYS, has_geometry

        uncropped = bool(getattr(self, "_crop_preview_uncropped", False))
        geo_key = tuple(float(adj.get(k, 0.0) or 0.0) for k in TRANSFORM_KEYS)
        cache_key = (id(base), geo_key, uncropped)
        if getattr(self, "_dodge_burn_mask_shape_cache_key", None) == cache_key:
            cached = getattr(self, "_dodge_burn_mask_shape_cached", None)
            if cached is not None:
                return cached
        h, w = int(base.shape[0]), int(base.shape[1])
        if uncropped or not has_geometry(adj):
            shape = (h, w)
        else:
            sample = self._wb_linear_sample_buffer(adj)
            if sample is None or not hasattr(sample, "shape"):
                shape = (h, w)
            else:
                shape = (int(sample.shape[0]), int(sample.shape[1]))
        self._dodge_burn_mask_shape_cache_key = cache_key
        self._dodge_burn_mask_shape_cached = shape
        return shape

    def _on_spot_heal_stroke(self, pt, pressure: float, is_end: bool) -> None:
        """Paint heal coverage; run full inpaint preview on stroke end."""
        panel = getattr(self, "single_image_adjust_panel", None)
        if panel is None:
            return
        mask_shape = self._dodge_burn_mask_shape()
        if mask_shape is None:
            return
        try:
            from raw_adjustments.spot_heal import HealMask, stamp_heal_brush

            mh, mw = mask_shape
            mask = getattr(self, "_spot_heal_mask", None)
            if mask is None or mask.data.shape != (mh, mw):
                mask = HealMask.empty(mh, mw)
                self._spot_heal_mask = mask

            mx, my = self._map_adjust_display_point_to_buffer(pt, mw, mh)
            disp = self._adjust_display_pixel_size()
            disp_w = disp[0] if disp is not None else mw
            sx = mw / max(1, disp_w)
            radius = max(2.0, panel.dodge_burn_brush_radius() * sx)
            strength = panel.dodge_burn_brush_strength() * max(0.05, float(pressure))
            stamp_heal_brush(mask, mx, my, radius, strength)
            if hasattr(panel, "set_spot_heal_mask_present"):
                panel.set_spot_heal_mask_present(not mask.is_empty)

            # Live: show green coverage while painting so the user sees
            # what will be filled (even if Mask is off). Restore Mask
            # preference on stroke end.
            gv = getattr(self, "gpu_view", None)
            if gv is not None:
                now = time.perf_counter()
                last = float(getattr(self, "_heal_mask_overlay_last", 0.0) or 0.0)
                if is_end or (now - last) >= 0.05:
                    self._heal_mask_overlay_last = now
                    wanted = bool(panel.dodge_burn_show_mask())
                    gv.set_dodge_burn_mask_overlay(
                        mask, True if not is_end else wanted
                    )

            if is_end:
                self._sync_dodge_burn_mask_overlay()
                self._apply_adjust_panel_preview(full_quality=True)
                self._on_adjust_panel_editing_finished(panel.get_adjustments())
                if hasattr(self, "status_bar") and self.status_bar:
                    self.status_bar.showMessage("Heal applied", 2000)
        except Exception:
            import logging

            logging.getLogger(__name__).warning(
                "[SPOT_HEAL] stroke handling failed", exc_info=True
            )

    def _on_dodge_burn_stroke(self, pt, pressure: float, is_end: bool) -> None:
        panel = getattr(self, "single_image_adjust_panel", None)
        if panel is None:
            return
        mode = panel.dodge_burn_mode()
        if mode == "heal":
            self._on_spot_heal_stroke(pt, pressure, is_end)
            return
        mask_shape = self._dodge_burn_mask_shape()
        if mode is None or mask_shape is None:
            if is_end:
                self._dodge_burn_stroke_active = False
                self._dodge_burn_stroke_baseline = None
                self._dodge_burn_stroke_work = None
                self._dodge_burn_stroke_pixmap = None
                self._dodge_burn_mask_at_stroke_start = None
            return
        try:
            from core.perf_metrics import perf_mark
            from raw_adjustments.dodge_burn import (
                DodgeBurnMask,
                edge_snap_region,
                erase_brush,
                stamp_brush,
            )
            from raw_adjustments.spot_heal import erase_heal_brush

            mh, mw = mask_shape
            mask = getattr(self, "_dodge_burn_mask", None)
            if mask is None or mask.data.shape != (mh, mw):
                mask = DodgeBurnMask.empty(mh, mw)
                self._dodge_burn_mask = mask

            # Map the stroke point from the CURRENTLY DISPLAYED pixmap's
            # coordinate space (whatever tier is on screen) down to the
            # fixed mask resolution.
            last = getattr(self, "_adjust_last_render", None)
            mx, my = self._map_adjust_display_point_to_buffer(pt, mw, mh)
            disp = self._adjust_display_pixel_size()
            disp_w = disp[0] if disp is not None else mw
            disp_h = disp[1] if disp is not None else mh
            sx = mw / max(1, disp_w)
            sy = mh / max(1, disp_h)
            # Brush size slider is in *display* pixels; convert to mask space.
            radius = max(2.0, panel.dodge_burn_brush_radius() * sx)
            strength_val = panel.dodge_burn_brush_strength()

            # Distance-based flow scaling: if the pointer barely moved since the last stamp point,
            # scale down delta to prevent stationary hesitation hot-spots.
            last_pt = getattr(self, "_dodge_burn_last_stamp_pt", None)
            dist_factor = 1.0
            if last_pt is not None:
                dist = math.hypot(mx - last_pt[0], my - last_pt[1])
                if dist < 0.25 * radius:
                    dist_factor = max(0.15, dist / (0.25 * radius))
            self._dodge_burn_last_stamp_pt = (mx, my)

            delta = strength_val * max(0.05, float(pressure)) * 0.28 * dist_factor
            max_stroke_cap = max(0.05, strength_val * max(0.05, float(pressure)) * 0.75)

            # Snapshot baseline once per stroke so live preview applies only
            # this stroke's mask delta (never re-bakes prior strokes into a
            # rectangular blit -- that path produced hard square blocks when
            # the mask patch resized poorly against the display buffer).
            if not getattr(self, "_dodge_burn_stroke_active", False):
                if last is not None:
                    self._dodge_burn_stroke_baseline = last[0].copy()
                    self._dodge_burn_stroke_baseline_adj = last[1]
                else:
                    self._dodge_burn_stroke_baseline = None
                    self._dodge_burn_stroke_baseline_adj = None
                self._dodge_burn_stroke_work = None
                self._dodge_burn_stroke_pixmap = None
                self._dodge_burn_mask_at_stroke_start = mask.data.copy()
                self._dodge_burn_stroke_dirty = None
                self._dodge_burn_last_stamp_pt = (mx, my)
                self._dodge_burn_stroke_perf = {
                    "n": 0,
                    "stamp_sum": 0.0,
                    "stamp_max": 0.0,
                    "patch_sum": 0.0,
                    "patch_max": 0.0,
                }

            # Edge-assist guide: cache luminance + chroma for the stroke so
            # each stamp stays O(brush area). Rebuild when the edit-base
            # shape changes. Chroma is a cheap opponent-color pair (R-Y,
            # B-Y), not a full colorspace conversion -- only relative
            # distances matter for the similarity gate in
            # raw_dodge_burn._edge_assist_gate.
            luma = getattr(self, "_dodge_burn_luma_guide", None)
            chroma_guide = getattr(self, "_dodge_burn_chroma_guide", None)
            if luma is None or getattr(luma, "shape", None) != (mh, mw):
                luma = None
                chroma_guide = None
                try:
                    base = getattr(self, "_adjust_preview_base_rgb", None)
                    if base is not None and hasattr(base, "shape"):
                        from raw_adjustments.pipeline import _linear_float_from_buffer

                        b = _linear_float_from_buffer(base)
                        if b.shape[0] != mh or b.shape[1] != mw:
                            import cv2

                            b = cv2.resize(
                                b, (mw, mh), interpolation=cv2.INTER_AREA
                            )
                        luma = (
                            0.2126 * b[..., 0]
                            + 0.7152 * b[..., 1]
                            + 0.0722 * b[..., 2]
                        ).astype(np.float32)
                        chroma_guide = np.stack(
                            [b[..., 0] - luma, b[..., 2] - luma], axis=-1
                        ).astype(np.float32)
                        self._dodge_burn_luma_guide = luma
                        self._dodge_burn_chroma_guide = chroma_guide
                except Exception:
                    luma = None
                    chroma_guide = None
                    self._dodge_burn_luma_guide = None
                    self._dodge_burn_chroma_guide = None

            edge_on = bool(panel.dodge_burn_edge_assist())
            t_stamp = time.perf_counter()
            if mode == "erase":
                # Flow × pressure → erase amount (1.0 clears center in one stamp).
                erase_amt = panel.dodge_burn_brush_strength() * max(
                    0.05, float(pressure)
                )
                bbox = erase_brush(
                    mask,
                    mx,
                    my,
                    radius,
                    erase_amt,
                    luminance=luma,
                    chroma=chroma_guide,
                    edge_assist=edge_on,
                )
                if panel is not None and hasattr(panel, "set_dodge_burn_mask_present"):
                    panel.set_dodge_burn_mask_present(not mask.is_empty)
                # Also erase heal coverage under the same brush.
                heal = getattr(self, "_spot_heal_mask", None)
                if heal is not None and heal.data.shape == (mh, mw):
                    erase_heal_brush(heal, mx, my, radius, erase_amt)
                    if panel is not None and hasattr(panel, "set_spot_heal_mask_present"):
                        panel.set_spot_heal_mask_present(not heal.is_empty)
            else:
                bbox = stamp_brush(
                    mask,
                    mx,
                    my,
                    radius,
                    delta,
                    dodge=(mode == "dodge"),
                    luminance=luma,
                    chroma=chroma_guide,
                    edge_assist=edge_on,
                    stroke_baseline=getattr(self, "_dodge_burn_mask_at_stroke_start", None),
                    max_stroke_delta=max_stroke_cap,
                )
            stamp_ms = (time.perf_counter() - t_stamp) * 1000.0
            perf_acc = getattr(self, "_dodge_burn_stroke_perf", None)
            if isinstance(perf_acc, dict):
                perf_acc["n"] = int(perf_acc.get("n", 0)) + 1
                perf_acc["stamp_sum"] = float(perf_acc.get("stamp_sum", 0.0)) + stamp_ms
                perf_acc["stamp_max"] = max(float(perf_acc.get("stamp_max", 0.0)), stamp_ms)
            # Avoid flooding the log at pointer sample rate; keep slow outliers.
            if stamp_ms >= 2.0:
                perf_mark(
                    "db_stamp",
                    stamp_ms,
                    getattr(self, "current_file_path", None),
                    w=mw,
                    h=mh,
                    mode=mode,
                    edge_assist=edge_on,
                )
            self._dodge_burn_stroke_active = True
            if mask.is_empty:
                panel.set_dodge_burn_mask_present(False)
            else:
                panel.set_dodge_burn_mask_present(True)

            # Keep Show Mask overlay in sync while painting (throttled).
            if panel.dodge_burn_show_mask():
                prev_dirty = getattr(self, "_dodge_burn_mask_overlay_dirty", None)
                bx0, by0, bx1, by1 = bbox
                if prev_dirty is not None:
                    bx0 = min(bx0, prev_dirty[0])
                    by0 = min(by0, prev_dirty[1])
                    bx1 = max(bx1, prev_dirty[2])
                    by1 = max(by1, prev_dirty[3])
                self._dodge_burn_mask_overlay_dirty = (bx0, by0, bx1, by1)
                now = time.perf_counter()
                last = float(getattr(self, "_dodge_burn_mask_overlay_last", 0.0) or 0.0)
                if is_end or (now - last) >= 0.05:
                    self._dodge_burn_mask_overlay_last = now
                    overlay_dirty = self._dodge_burn_mask_overlay_dirty
                    self._dodge_burn_mask_overlay_dirty = None
                    gv = getattr(self, "gpu_view", None)
                    if gv is not None:
                        gv.update_dodge_burn_mask(mask, dirty_bbox=overlay_dirty)

            baseline = getattr(self, "_dodge_burn_stroke_baseline", None)
            mask_start = getattr(self, "_dodge_burn_mask_at_stroke_start", None)
            if baseline is not None and mask_start is not None:
                import cv2

                t_patch = time.perf_counter()
                x0, y0, x1, y1 = bbox
                # Pad the stamp bbox slightly so soft edges aren't clipped.
                pad = 2
                x0 = max(0, x0 - pad)
                y0 = max(0, y0 - pad)
                x1 = min(mw, x1 + pad)
                y1 = min(mh, y1 + pad)
                # Accumulate stroke dirty rect so multi-point drags preview
                # the whole stroke, not only the latest stamp.
                prev = getattr(self, "_dodge_burn_stroke_dirty", None)
                if prev is not None:
                    x0 = min(x0, prev[0])
                    y0 = min(y0, prev[1])
                    x1 = max(x1, prev[2])
                    y1 = max(y1, prev[3])
                self._dodge_burn_stroke_dirty = (x0, y0, x1, y1)

                px0 = max(0, int(x0 / sx))
                py0 = max(0, int(y0 / sy))
                px1 = min(disp_w, int(np.ceil(x1 / sx)))
                py1 = min(disp_h, int(np.ceil(y1 / sy)))
                if px1 > px0 and py1 > py0:
                    m_x0 = max(0, int(px0 * sx))
                    m_y0 = max(0, int(py0 * sy))
                    m_x1 = min(mw, max(m_x0 + 1, int(np.ceil(px1 * sx))))
                    m_y1 = min(mh, max(m_y0 + 1, int(np.ceil(py1 * sy))))
                    d_patch = (mask.data[m_y0:m_y1, m_x0:m_x1] - mask_start[m_y0:m_y1, m_x0:m_x1])
                    ph, pw = py1 - py0, px1 - px0
                    if d_patch.size > 0 and min(d_patch.shape) >= 2:
                        if d_patch.shape != (ph, pw):
                            d_disp = cv2.resize(
                                d_patch.astype(np.float32),
                                (pw, ph),
                                interpolation=cv2.INTER_LINEAR,
                            )
                        else:
                            d_disp = d_patch.astype(np.float32)
                    else:
                        # Degenerate mask mapping -- stamp a display-space
                        # gaussian at the cursor instead of a flat block.
                        from raw_adjustments.dodge_burn import circular_brush_falloff

                        r_disp = max(2.0, float(panel.dodge_burn_brush_radius()))
                        d_disp = circular_brush_falloff(
                            0, ph, 0, pw, pt.x() - px0, pt.y() - py0, r_disp
                        ) * float(delta) * (1.0 if mode == "dodge" else -1.0)

                    stops = float(
                        (getattr(self, "_dodge_burn_stroke_baseline_adj", None) or {}).get(
                            "DodgeBurnStrength", 1.75
                        )
                    )
                    gain = np.exp2(d_disp * stops).astype(np.float32)
                    patch = baseline[py0:py1, px0:px1].astype(np.float32)
                    out8 = np.clip(patch * gain[..., np.newaxis] + 0.5, 0, 255).astype(
                        np.uint8
                    )
                    # One full-frame working buffer per stroke; blit the dirty
                    # patch into the existing QPixmap instead of baseline.copy()
                    # + full tobytes every pointer sample.
                    work = getattr(self, "_dodge_burn_stroke_work", None)
                    stroke_pm = getattr(self, "_dodge_burn_stroke_pixmap", None)
                    if (
                        work is None
                        or work.shape != baseline.shape
                        or stroke_pm is None
                        or stroke_pm.isNull()
                        or stroke_pm.width() != baseline.shape[1]
                        or stroke_pm.height() != baseline.shape[0]
                    ):
                        work = baseline.copy()
                        self._dodge_burn_stroke_work = work
                        stroke_pm = self._numpy_to_qpixmap(work)
                        self._dodge_burn_stroke_pixmap = stroke_pm
                    work[py0:py1, px0:px1] = out8
                    if stroke_pm is not None and not stroke_pm.isNull():
                        from PyQt6.QtGui import QImage, QPainter

                        ph, pw = out8.shape[0], out8.shape[1]
                        patch_img = QImage(
                            out8.data,
                            pw,
                            ph,
                            int(out8.strides[0]),
                            QImage.Format.Format_RGB888,
                        ).copy()
                        painter = QPainter(stroke_pm)
                        painter.drawImage(px0, py0, patch_img)
                        painter.end()
                        self._display_adjust_preview_pixmap(stroke_pm)
                        # Keep last_render aligned with the live stroke buffer.
                        self._adjust_last_render = (
                            work,
                            getattr(self, "_dodge_burn_stroke_baseline_adj", None)
                            or (last[1] if last is not None else {}),
                        )
                        self._adjust_last_render_norm = _norm_path(
                            getattr(self, "current_file_path", "") or ""
                        )
                patch_ms = (time.perf_counter() - t_patch) * 1000.0
                if isinstance(perf_acc, dict):
                    perf_acc["patch_sum"] = float(perf_acc.get("patch_sum", 0.0)) + patch_ms
                    perf_acc["patch_max"] = max(
                        float(perf_acc.get("patch_max", 0.0)), patch_ms
                    )
                if patch_ms >= 5.0:
                    perf_mark(
                        "db_live_patch",
                        patch_ms,
                        getattr(self, "current_file_path", None),
                        pw=max(0, px1 - px0),
                        ph=max(0, py1 - py0),
                    )

            if is_end:
                self._dodge_burn_stroke_active = False
                self._dodge_burn_stroke_baseline = None
                self._dodge_burn_stroke_work = None
                self._dodge_burn_stroke_pixmap = None
                self._dodge_burn_mask_at_stroke_start = None
                self._dodge_burn_stroke_dirty = None
                self._dodge_burn_last_stamp_pt = None
                luminance = getattr(self, "_dodge_burn_luma_guide", None)
                if luminance is None or getattr(luminance, "shape", None) != (mh, mw):
                    luminance = None
                    try:
                        base = getattr(self, "_adjust_preview_base_rgb", None)
                        if base is not None and hasattr(base, "shape"):
                            b = base.astype(np.float32)
                            luminance = (
                                0.2126 * b[..., 0]
                                + 0.7152 * b[..., 1]
                                + 0.0722 * b[..., 2]
                            )
                    except Exception:
                        luminance = None
                if luminance is not None:
                    t_snap = time.perf_counter()
                    edge_snap_region(mask, luminance, bbox)
                    perf_mark(
                        "db_edge_snap",
                        (time.perf_counter() - t_snap) * 1000.0,
                        getattr(self, "current_file_path", None),
                        mh=mh,
                        mw=mw,
                    )
                if isinstance(perf_acc, dict) and int(perf_acc.get("n", 0)) > 0:
                    n = int(perf_acc["n"])
                    perf_mark(
                        "db_stroke",
                        float(perf_acc.get("stamp_sum", 0.0))
                        + float(perf_acc.get("patch_sum", 0.0)),
                        getattr(self, "current_file_path", None),
                        n=n,
                        stamp_max=float(perf_acc.get("stamp_max", 0.0)),
                        stamp_avg=float(perf_acc.get("stamp_sum", 0.0)) / n,
                        patch_max=float(perf_acc.get("patch_max", 0.0)),
                        patch_avg=float(perf_acc.get("patch_sum", 0.0)) / n,
                    )
                self._dodge_burn_stroke_perf = None
                self._apply_adjust_panel_preview(full_quality=True)
                self._on_adjust_panel_editing_finished(panel.get_adjustments())
        except Exception:
            import logging

            logging.getLogger(__name__).warning(
                "[DODGE_BURN] stroke handling failed", exc_info=True
            )
            self._dodge_burn_stroke_active = False
            self._dodge_burn_stroke_baseline = None
            self._dodge_burn_stroke_work = None
            self._dodge_burn_stroke_pixmap = None
            self._dodge_burn_mask_at_stroke_start = None
            self._dodge_burn_stroke_dirty = None
            self._dodge_burn_stroke_perf = None

    def _on_adjust_preview_ready(self, generation: int, file_path: str, array) -> None:
        # Clear the AI busy state for the AI generation even when it is stale or
        # errored (array is None): the button must never stay stuck disabled.
        if getattr(self, "_ai_preview_gen", None) == int(generation):
            self._ai_preview_gen = None
            self._set_ai_preview_busy(False)
        self._adjust_preview_busy = False
        dirty_full_quality = bool(getattr(self, "_adjust_preview_dirty_full_quality", False))
        self._adjust_preview_dirty_full_quality = False
        dirty_ai = bool(getattr(self, "_adjust_preview_dirty_ai", False))
        self._adjust_preview_dirty_ai = False
        if int(generation) != int(getattr(self, "_adjust_preview_gen", 0)):
            if getattr(self, "_adjust_preview_dirty", False):
                self._apply_adjust_panel_preview(full_quality=dirty_full_quality, ai_preview=dirty_ai)
            return
        if _norm_path(file_path) != _norm_path(getattr(self, "current_file_path", "") or ""):
            if getattr(self, "_adjust_preview_dirty", False):
                self._apply_adjust_panel_preview(full_quality=dirty_full_quality, ai_preview=dirty_ai)
            return
        if array is not None:
            try:
                pixmap = self._numpy_to_qpixmap(array)
            except Exception:
                pixmap = None
            dirty_now = bool(getattr(self, "_adjust_preview_dirty", False))
            # A newer request is already queued -- do not paint/cache this stale
            # generation (was flashing the previous slider values, then settling).
            if dirty_now:
                if getattr(self, "_adjust_preview_dirty", False):
                    self._apply_adjust_panel_preview(full_quality=dirty_full_quality, ai_preview=dirty_ai)
                return
            if pixmap is not None and not pixmap.isNull():
                deferred = self._should_defer_adjust_preview_paint(pixmap)
                if not deferred:
                    self._display_adjust_preview_pixmap(pixmap)
                # Cache this exact render as the new instant-gain baseline
                # (see _try_instant_gain_preview): array is uint8 RGB, ready
                # for cv2.LUT with no further conversion. While zoomed, do not
                # replace a sharper settle baseline with a deferred lite tier
                # -- otherwise Exposure instant-gain and zoom-in restore both
                # collapse to 640px.
                submitted = getattr(self, "_adjust_preview_submitted_adj", None)
                adj_used = (submitted or {}).get(generation)
                if adj_used is not None and array.dtype == np.uint8 and array.ndim == 3:
                    skip_cache = False
                    prev = getattr(self, "_adjust_last_render", None)
                    if (
                        deferred
                        and prev is not None
                        and prev[0] is not None
                        and max(int(prev[0].shape[0]), int(prev[0].shape[1]))
                        > max(int(array.shape[0]), int(array.shape[1])) * 1.2
                    ):
                        skip_cache = True
                    if not skip_cache:
                        self._adjust_last_render = (array, adj_used)
                        self._adjust_last_render_norm = _norm_path(file_path)
                        # Identity-geometry display frame for instant Transform warps.
                        # Prefer uncropped (overlay) frames so straighten can show
                        # context outside the crop rect.
                        try:
                            warp_zero = not any(
                                abs(float(adj_used.get(k, 0.0) or 0.0)) > 1e-4
                                for k in (
                                    "CropAngle",
                                    "PerspectiveVertical",
                                    "PerspectiveHorizontal",
                                )
                            )
                            if warp_zero and array.dtype == np.uint8:
                                self._adjust_transform_source_u8 = array
                                self._adjust_transform_source_adj = dict(adj_used)
                                self._adjust_transform_source_norm = _norm_path(file_path)
                        except Exception:
                            pass
                    if max(int(array.shape[0]), int(array.shape[1])) >= 1200:
                        self._adjust_hires_render = (array, adj_used)
                        self._adjust_hires_render_norm = _norm_path(file_path)
                # Deferred browse bake after full-quality settle (editing_finished
                # used to bake the previous last_render while this worker ran).
                pending_bake = getattr(self, "_pending_browse_bake_path", None)
                if (
                    pending_bake
                    and _norm_path(pending_bake) == _norm_path(file_path)
                    and not deferred
                    and max(int(array.shape[0]), int(array.shape[1])) >= 1200
                ):
                    self._pending_browse_bake_path = None
                    self._persist_editor_aligned_browse_caches(file_path)
        # This generation is the accepted (non-superseded) render for the
        # current file. Drop the "Loading previous edit..." overlay only once
        # the render that landed actually carried this file's local masks --
        # the first render is submitted while the mask decode is still in
        # flight, so dismissing on it would hide the overlay while the photo
        # on screen is still missing the brushwork. A failed render (array
        # None) also releases the overlay so it cannot get stuck.
        if _norm_path(file_path) == getattr(self, "_adjust_loading_edit_norm", None):
            submitted_map = getattr(self, "_adjust_preview_submitted_adj", None)
            if array is None or self._render_reflects_loaded_edit(
                (submitted_map or {}).get(generation)
            ):
                self._set_edit_loading_overlay(None)
        if getattr(self, "_adjust_preview_dirty", False):
            self._apply_adjust_panel_preview(full_quality=dirty_full_quality, ai_preview=dirty_ai)

    def _dodge_burn_overlay_adj(self, adj: dict) -> dict:
        """Fold main.py's local-mask state into an adjustments dict.

        Dodge/burn and spot-heal masks live in main.py (painted via the GPU
        view), not on any panel widget -- ``panel.get_adjustments()`` has no
        way to include them. Preview/export paths inject a live mask object
        plus a cheap fingerprint for stage-cache keys; XMP save serializes
        PNG separately (see ``_on_adjust_panel_editing_finished``).
        """
        from raw_adjustments.dodge_burn import (
            DEFAULT_STRENGTH,
            MASK_KEY,
            MASK_OBJ_KEY,
            STRENGTH_KEY,
            mask_stage_fingerprint,
        )
        from raw_adjustments.spot_heal import (
            MASK_KEY as HEAL_MASK_KEY,
        )
        from raw_adjustments.spot_heal import (
            MASK_OBJ_KEY as HEAL_MASK_OBJ_KEY,
        )
        from raw_adjustments.spot_heal import mask_stage_fingerprint as heal_fingerprint

        mask = getattr(self, "_dodge_burn_mask", None)
        if mask is not None and not mask.is_empty:
            adj = dict(adj)
            adj[MASK_OBJ_KEY] = mask
            adj[MASK_KEY] = mask_stage_fingerprint(mask)
            adj.setdefault(STRENGTH_KEY, DEFAULT_STRENGTH)
        else:
            if MASK_KEY in adj or MASK_OBJ_KEY in adj:
                adj = dict(adj)
                adj.pop(MASK_KEY, None)
                adj.pop(MASK_OBJ_KEY, None)

        heal = getattr(self, "_spot_heal_mask", None)
        if heal is not None and not heal.is_empty:
            adj = dict(adj)
            adj[HEAL_MASK_OBJ_KEY] = heal
            adj[HEAL_MASK_KEY] = heal_fingerprint(heal)
        else:
            if HEAL_MASK_KEY in adj or HEAL_MASK_OBJ_KEY in adj:
                adj = dict(adj)
                adj.pop(HEAL_MASK_KEY, None)
                adj.pop(HEAL_MASK_OBJ_KEY, None)
        return adj

    def _apply_adjust_panel_preview(self, *, full_quality: bool = False, ai_preview: bool = False) -> None:
        panel = getattr(self, "single_image_adjust_panel", None)
        path = getattr(self, "current_file_path", None)
        if not path:
            return
        adj = panel.get_adjustments() if panel is not None else getattr(self, "_pending_adjust_preview", None)
        if adj is None:
            return
        with contextlib.suppress(Exception):
            adj = self._dodge_burn_overlay_adj(adj)
        self._sync_geometry_overlay_with_adj(adj)
        if getattr(self, "_crop_preview_uncropped", False):
            adj = self._crop_keys_zeroed(adj)
        self._pending_adjust_preview = dict(adj)
        norm = _norm_path(path)
        base_full = getattr(self, "_adjust_preview_base_rgb", None)
        if base_full is None or norm != getattr(self, "_adjust_edit_base_path", None):
            self._request_adjust_edit_base(path)
            return
        if getattr(self, "_adjust_preview_busy", False):
            self._adjust_preview_dirty = True
            # A pending full_quality/AI request must not be downgraded by a
            # subsequent fast one arriving while still queued -- OR-combine.
            self._adjust_preview_dirty_full_quality = (
                full_quality or getattr(self, "_adjust_preview_dirty_full_quality", False)
            )
            self._adjust_preview_dirty_ai = (
                ai_preview or getattr(self, "_adjust_preview_dirty_ai", False)
            )
            return
        self._adjust_preview_dirty = False
        self._adjust_preview_dirty_full_quality = False
        self._adjust_preview_dirty_ai = False
        self._adjust_preview_busy = True
        self._adjust_preview_gen = int(getattr(self, "_adjust_preview_gen", 0)) + 1
        gen = self._adjust_preview_gen
        base_fast = getattr(self, "_adjust_preview_base_rgb_fast", None)
        # AI denoise preview is on-demand (the panel's "Preview AI" button)
        # and runs at full resolution, the same base as export, so it shows
        # the real result. Slider changes never run AI.
        ai_preview = bool(
            ai_preview and base_full is not None and self._ai_denoise_preview_ready()
        )
        use_fast = not full_quality and base_fast is not None
        if ai_preview:
            base = base_full
            stage_cache = None
        elif use_fast:
            base = base_fast
            stage_cache = getattr(self, "_adjust_preview_stage_cache_fast", None)
        else:
            base = base_full
            stage_cache = getattr(self, "_adjust_preview_stage_cache", None)
        # Snapshot the adjustments this generation renders with, so
        # _on_adjust_preview_ready can cache (exact_array, adj) as the new
        # instant-gain baseline once the exact render lands. Small dict
        # pruned to the last few generations -- stale entries are harmless,
        # just unused memory.
        submitted = getattr(self, "_adjust_preview_submitted_adj", None)
        if submitted is None:
            submitted = self._adjust_preview_submitted_adj = {}
        submitted[gen] = dict(adj)
        for stale in [g for g in submitted if g < gen - 4]:
            del submitted[stale]
        if ai_preview:
            self._ai_preview_gen = gen
            self._set_ai_preview_busy(True)
        worker = _AdjustPreviewWorker(
            gen,
            path,
            base,
            dict(adj),
            self._adjust_preview_signals,
            stage_cache=stage_cache,
            preview_lite=use_fast,
            # Settle on half-res base: keep y0 guided filter, skip chroma-damp.
            settle_fast=bool(full_quality) and not use_fast,
            ai_denoise=ai_preview,
        )
        _get_bg_thread_pool().start(worker)

    def _ai_denoise_preview_ready(self) -> bool:
        """Panel AI toggle on AND the ONNX denoise model is actually present."""
        panel = getattr(self, "single_image_adjust_panel", None)
        if panel is None or not hasattr(panel, "ai_denoise_enabled"):
            return False
        try:
            if not panel.ai_denoise_enabled():
                return False
        except Exception:
            return False
        try:
            from raw_adjustments.pipeline import ai_denoise_model_present

            return ai_denoise_model_present()
        except Exception:
            return False

    def _apply_dodge_burn_mask_from_adj(self, adj: dict | None) -> None:
        """Restore the in-memory dodge/burn mask from an adjustments dict.

        Undo / file load push sliders via ``set_adjustments``, but the live
        mask lives on the main window (``_dodge_burn_mask``). Without this,
        Ctrl/Cmd+Z restored slider state then ``_on_adjust_panel_editing_finished``
        re-serialized the still-current painted mask and undo looked broken.
        """
        from raw_adjustments.dodge_burn import MASK_KEY, deserialize_mask

        serial = str((adj or {}).get(MASK_KEY, "") or "").strip()
        mask = None
        if serial and not serial.startswith("mem:"):
            try:
                mask = deserialize_mask(serial)
            except Exception:
                mask = None
        if mask is not None and getattr(mask, "is_empty", False):
            mask = None
        self._dodge_burn_mask = mask
        self._dodge_burn_luma_guide = None
        self._dodge_burn_mask_shape_cache_key = None
        self._dodge_burn_mask_shape_cached = None
        panel = getattr(self, "single_image_adjust_panel", None)
        if panel is not None:
            panel.set_dodge_burn_mask_present(mask is not None)
        self._sync_dodge_burn_mask_overlay()

    def _apply_spot_heal_mask_from_adj(self, adj: dict | None) -> None:
        """Restore the in-memory spot-heal mask from an adjustments dict."""
        from raw_adjustments.spot_heal import MASK_KEY, deserialize_mask

        serial = str((adj or {}).get(MASK_KEY, "") or "").strip()
        mask = None
        if serial and not serial.startswith("mem:"):
            try:
                mask = deserialize_mask(serial)
            except Exception:
                mask = None
        if mask is not None and getattr(mask, "is_empty", False):
            mask = None
        self._spot_heal_mask = mask
        panel = getattr(self, "single_image_adjust_panel", None)
        if panel is not None and hasattr(panel, "set_spot_heal_mask_present"):
            panel.set_spot_heal_mask_present(mask is not None)
        self._sync_dodge_burn_mask_overlay()

    def _adjustments_match(self, a: dict, b: dict) -> bool:
        """Whether two adjustment dicts describe the same edit.

        Live objects (keys ending in ``_obj``) are excluded: they are rebuilt
        on every panel bind, so comparing them by identity would report a
        change every time and defeat the point. Their serialized twins are in
        the dict and are compared.
        """
        def _clean(d):
            out = {}
            for k, v in (d or {}).items():
                if k.endswith("_obj"):
                    continue
                if isinstance(v, float):
                    out[k] = round(v, 6)
                else:
                    out[k] = v
            return out

        return _clean(a) == _clean(b)
