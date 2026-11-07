"""display mixin, part 1 of 6 (23 methods, 893 lines)."""
from __future__ import annotations

import contextlib
import os
import sys

from PyQt6.QtCore import QEasingCurve, QPoint, QPropertyAnimation, QRect, Qt, QTimer
from PyQt6.QtGui import QPainter, QPixmap
from PyQt6.QtWidgets import QGraphicsOpacityEffect, QLabel

from core.config import _norm_path

from .. import lazy
from ..lazy import check_memory_cache_for_image
from ..paths import _require_str_path

lazy.register(sys.modules[__name__])


class DisplayPart1:
    def _set_single_view_pixmap(self, base: QPixmap) -> None:
        """Set image_label pixmap with optional dashed focus / subject outline."""
        # Consume the one-shot cross-fade request. Only display_pixmap() (i.e. new content
        # arriving: thumbnail/preview/full upgrade or navigation) sets this flag. Zoom/pan/resize
        # repaints reach here without it, so they never fade.
        do_fade = bool(getattr(self, "_crossfade_next_paint", False))
        self._crossfade_next_paint = False

        if base is None or base.isNull():
            self._cancel_content_crossfade()
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("Failed to load image")
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return

        # Snapshot the currently displayed content so we can fade it out on top of the new image.
        old_pm = None
        if do_fade and getattr(self, "_crossfade_enabled", True):
            current = self.image_label.pixmap()
            if current is not None and not current.isNull():
                old_pm = QPixmap(current)  # detach from the label's live pixmap
        else:
            # Any non-fade paint (zoom/pan) must drop a still-running fade to avoid stale overlay.
            self._cancel_content_crossfade()

        blended = base
        ow = self.current_pixmap.width() if self.current_pixmap else 0
        oh = self.current_pixmap.height() if self.current_pixmap else 0
        subj = getattr(self, "_focus_subject_rect_image", None)
        draw_exif_subj = (
            getattr(self, "_focus_subject_outline_active", False)
            and subj is not None
            and isinstance(subj, QRect)
            and not subj.isNull()
            and self.current_pixmap is not None
            and not self.current_pixmap.isNull()
            and ow > 0
            and oh > 0
            and not base.isNull()
        )
        grid_mode = getattr(self, "_composition_grid_mode", "off")
        draw_grid = (
            grid_mode != "off"
            and self.current_pixmap is not None
            and not self.current_pixmap.isNull()
            and ow > 0
            and oh > 0
            and not base.isNull()
        )
        draw_clipping = (
            getattr(self, "_clipping_overlay_active", False)
            and self.current_pixmap is not None
            and not self.current_pixmap.isNull()
            and ow > 0
            and oh > 0
            and not base.isNull()
        )
        if draw_exif_subj or draw_grid or draw_clipping:
            blended = base.copy()
            p = QPainter(blended)
            if draw_exif_subj and subj is not None:
                self._draw_focus_subject_outline_on_base_painter(p, base, subj)
            if draw_grid:
                from rawviewer_ui.overlays.composition_grid import draw_composition_grid

                draw_composition_grid(p, ow, oh, grid_mode)
            if draw_clipping:
                self._ensure_clipping_overlay_cached()
                clip_pm = getattr(self, "_clipping_overlay_pm", None)
                if clip_pm is not None and not clip_pm.isNull():
                    p.drawPixmap(0, 0, clip_pm)
            p.end()
        pending_size = blended.size()

        if old_pm is not None and getattr(self, "_crossfade_use_resolution_duration", False):

            def _apply_label() -> None:
                self.image_label.setPixmap(blended)
                self.image_label.resize(pending_size)

            fade_ms = self._resolution_crossfade_duration_ms(
                max(old_pm.width(), old_pm.height()),
                max(blended.width(), blended.height()),
                is_navigation=bool(getattr(self, "_resolution_crossfade_is_navigation", False)),
            )
            self._crossfade_overlay_then_apply(
                self.image_label,
                self.image_label.rect(),
                old_pm,
                _apply_label,
                duration_ms=fade_ms,
            )
        else:
            self.image_label.setPixmap(blended)
            if old_pm is not None:
                def _finish_label_resize(_size=pending_size):
                    with contextlib.suppress(Exception):
                        self.image_label.resize(_size)

                try:
                    self._run_content_crossfade(
                        old_pm,
                        on_finished=_finish_label_resize,
                    )
                except Exception:
                    _finish_label_resize()
            else:
                self.image_label.resize(pending_size)

    def _cancel_content_crossfade(self) -> None:
        """Stop and remove any in-flight cross-fade overlay/animation."""
        anim = getattr(self, "_crossfade_anim", None)
        if anim is not None:
            with contextlib.suppress(Exception):
                anim.stop()
            self._crossfade_anim = None
        overlay = getattr(self, "_crossfade_overlay", None)
        if overlay is not None:
            with contextlib.suppress(Exception):
                overlay.deleteLater()
            self._crossfade_overlay = None

    def _resolution_crossfade_duration_ms(
        self, prev_max: int = 0, new_max: int = 0, *, is_navigation: bool = False
    ) -> int:
        base = int(getattr(self, "_resolution_crossfade_ms", 280))
        # Navigation's preview->full jump is routinely a huge ratio (a ~500px
        # nav preview to an 8000px+ sensor buffer) -- the biggest tier in this
        # scaling almost always fires here, adding up to 600ms of visible
        # settle time on every single navigation now that the crossfade
        # actually engages correctly (see the GL-viewport snapshot fix).
        # That reads as "loading got slower" even though decode/paint timing
        # is unchanged. Keep the up-scaling for other crossfade triggers
        # (gallery preview upgrade, zoom) where a slower fade is deliberate
        # polish, not a tax on every "next image" press.
        if is_navigation:
            return base
        if prev_max > 0 and new_max > prev_max:
            ratio = new_max / float(prev_max)
            if ratio >= 2.5:
                return min(600, int(base * 1.55))
            if ratio >= 1.5:
                return min(480, int(base * 1.28))
        return base

    def _single_view_pixels_on_screen(self, file_path: str | None = None) -> bool:
        """True when the current file's pixels are actually visible (GPU view or legacy label)."""
        path = file_path or getattr(self, "current_file_path", None)
        if not path:
            return False
        if self.is_animated_image(path):
            if _norm_path(getattr(self, "current_file_path", "")) != _norm_path(path):
                return False
            if getattr(self, "_current_movie", None) is not None:
                return True
            if getattr(self, "_current_webp_frames", None):
                return len(self._current_webp_frames) > 0
            label = getattr(self, "image_label", None)
            if label is not None:
                lp = label.pixmap()
                if lp is not None and not lp.isNull():
                    return True
            return False
        if _norm_path(getattr(self, "_displayed_content_path", "")) != _norm_path(path):
            return False
        gv = getattr(self, "gpu_view", None)
        if gv is not None:
            if not gv.has_pixmap():
                return False
            # CUDA-GL / GL-tex path: image_size is authoritative. display() may
            # keep a 1×1 placeholder in current_pixmap (avoids a huge empty
            # QPixmap alloc); do not treat that as a size mismatch / blank view.
            if getattr(gv, "_use_rgb_gl", False):
                try:
                    gw, gh = gv.image_size()
                    return gw > 0 and gh > 0
                except Exception:
                    return False
            pm = getattr(self, "current_pixmap", None)
            if pm is not None and not pm.isNull():
                try:
                    gw, gh = gv.image_size()
                    if gw > 0 and gh > 0 and (gw != pm.width() or gh != pm.height()):
                        return False
                except Exception:
                    pass
            return True
        pm = getattr(self, "current_pixmap", None)
        if pm is not None and not pm.isNull():
            return True
        label = getattr(self, "image_label", None)
        if label is not None:
            lp = label.pixmap()
            return lp is not None and not lp.isNull()
        return False

    def _try_paint_display_quality_cache(self, file_path: str) -> bool:
        if self._full_res_only_mode():
            return False
        """Paint display-quality preview from cache when metadata is ready but viewport is empty.

        Full-resolution buffers are excluded so EXIF-ready does not paint a multi-second
        LibRaw decode; full pixels are shown via ``on_manager_image_ready`` instead.
        """
        import logging

        logger = logging.getLogger(__name__)
        if not file_path or self._single_view_pixels_on_screen(file_path):
            return bool(file_path) and self._single_view_pixels_on_screen(file_path)
        try:
            preview = self.image_cache.get_preview_memory_only(file_path)
            if preview is not None and self._suppress_jpeg_interim(
                file_path
            ) and not self.image_cache.is_libraw_preview(file_path):
                # RAW workflow: embedded-JPEG preview; wait for LibRaw pixels.
                preview = None
            if preview is not None:
                if self._preview_acceptable_for_single_view(
                    preview, file_path=file_path, context="navigation"
                ):
                    return self._display_cached_numpy_for_path(
                        file_path, preview, from_cache="quality_preview"
                    )
                px = self._cache_buffer_max_dim(preview)
                nav_min = self._single_view_cache_min_dim(for_navigation=True)
                logger.info(
                    "[LOAD] Skipping low-res preview cache (%dpx < %d) for %s; "
                    "waiting for embedded thumbnail/full",
                    px,
                    nav_min,
                    os.path.basename(file_path),
                )
        except Exception:
            pass
        return False

    def _schedule_gpu_viewport_refit(self) -> None:
        """Refit GPU view after layout settles (first paint can run at viewport size 0)."""
        gv = getattr(self, "gpu_view", None)
        if gv is None:
            return
        from PyQt6.QtCore import QTimer

        def _refit() -> None:
            try:
                if gv.has_pixmap() and self._single_view_is_fit_mode():
                    gv.fit_to_window()
            except Exception:
                pass

        QTimer.singleShot(0, _refit)
        QTimer.singleShot(50, _refit)

    def _sync_gpu_single_view_visibility(self) -> None:
        """GPU view owns static-image input; legacy scroll is for GIF/WebP only."""
        gv = getattr(self, "gpu_view", None)
        if gv is None:
            return
        animated = bool(
            getattr(self, "current_file_path", None)
            and self.is_animated_image(_require_str_path(self.current_file_path))
        )
        container = getattr(self, "single_view_container", None)
        if container is not None and hasattr(container, "set_gpu_view_active"):
            container.set_gpu_view_active(not animated)
            return
        if animated:
            gv.hide()
            self.scroll_area.show()
        else:
            gv.show()
            self.scroll_area.hide()

    def _display_quality_buffer_cached(self, file_path: str, *, min_dim: int = 1400) -> bool:
        """True when a preview/full buffer suitable for single-view display is already cached."""
        if not file_path:
            return False
        try:
            preview = self.image_cache.get_preview_memory_only(file_path)
            if preview is not None and self._cache_buffer_suitable_for_single_view(
                preview, min_dim=min_dim, file_path=file_path
            ):
                return True
        except Exception:
            pass
        try:
            cached_item, cache_type = check_memory_cache_for_image(
                file_path, use_full_resolution=False
            )
        except Exception:
            return False
        if cached_item is None:
            return False
        # Gallery-tier (~512px) previews must not count as "display quality" --
        # non-RAW mode still stores them under preview_cache, and treating them
        # as warm used to skip resetting _manager_displayed_max_dim on nav.
        if cache_type in ("preview", "full_image"):
            return bool(
                self._cache_buffer_suitable_for_single_view(
                    cached_item, min_dim=min_dim, file_path=file_path
                )
            )
        if cache_type == "pixmap":
            try:
                if isinstance(cached_item, QPixmap) and not cached_item.isNull():
                    return max(cached_item.width(), cached_item.height()) >= min_dim
            except Exception:
                return False
            return False
        if cache_type == "thumbnail":
            try:
                if isinstance(cached_item, QPixmap):
                    return max(cached_item.width(), cached_item.height()) >= min_dim
                h, w = cached_item.shape[:2]
                return max(h, w) >= min_dim
            except Exception:
                pass
        return False

    def _display_cached_numpy_for_path(self, file_path: str, image, *, from_cache: str = "cache") -> bool:
        """Paint a cached numpy buffer and finish the load path."""
        import logging

        logger = logging.getLogger(__name__)
        if image is None:
            return False
        if (
            not getattr(self, "_skip_resolution_downgrade_check", False)
            and self._already_displaying_buffer_for_path(file_path, image)
        ):
            logger.info(
                f"[LOAD] Skipping redundant cache redraw for {os.path.basename(file_path)} "
                f"(already on screen)"
            )
            self._gallery_preview_pending_full = False
            self._loading_from_gallery = False
            self._on_single_view_content_displayed()
            return True
        filename = os.path.basename(file_path)
        logger.info(f"[LOAD] Cache hit ({from_cache}): displaying {filename}")
        self.status_bar.showMessage(f"Loaded {filename} from cache")
        self._orientation_already_applied = True
        if (
            getattr(self, "_gallery_instant_display_quality", False)
            or getattr(self, "_loading_from_gallery", False)
            or getattr(self, "_navigation_in_progress", False)
            or str(from_cache).startswith("nav_")
        ):
            self._skip_resolution_crossfade_once = True
            self._gallery_preview_pending_full = False
        try:
            self.display_numpy_image(image)
            self._skip_resolution_downgrade_check = False
            self._on_single_view_content_displayed()
        finally:
            self._orientation_already_applied = False
        self._restore_keyboard_focus()
        self.schedule_save_session_state()
        try:
            if self.image_files and file_path in self.image_files:
                self.current_file_index = self.image_files.index(file_path)
        except ValueError:
            pass
        self._start_preloading()
        if from_cache == "nav_full":
            self._maybe_queue_background_full_decode(file_path)
        elif from_cache == "nav_preview":
            self._schedule_idle_full_decode_after_nav_preview(file_path)
        elif from_cache == "preview":
            self._maybe_queue_background_full_decode(file_path)
        if hasattr(self, "loading_overlay"):
            self.loading_overlay.hide_loading()
        return True

    def _mark_pending_resolution_crossfade(self) -> None:
        self._pending_resolution_crossfade = True

    def _clear_pending_resolution_crossfade(self) -> None:
        self._pending_resolution_crossfade = False

    def _resolution_upgrade_needs_crossfade(
        self,
        prev_pm: QPixmap | None,
        new_pm: QPixmap | None,
        *,
        prev_path: str | None,
        cur_path: str | None,
    ) -> bool:
        if not getattr(self, "_crossfade_enabled", True):
            return False
        if prev_pm is None or prev_pm.isNull() or new_pm is None or new_pm.isNull():
            return False
        if _norm_path(prev_path or "") != _norm_path(cur_path or ""):
            return False
        if getattr(self, "_gallery_preview_pending_full", False):
            prev_max = max(prev_pm.width(), prev_pm.height())
            new_max = max(new_pm.width(), new_pm.height())
            if new_max > int(prev_max * 1.06):
                return True
            self._gallery_preview_pending_full = False
            return False
        prev_max = max(prev_pm.width(), prev_pm.height())
        new_max = max(new_pm.width(), new_pm.height())
        # Same file (guaranteed above) at essentially the same max dimension is a
        # redundant re-delivery of the buffer already on screen -- not a
        # resolution upgrade. Crossfading identical content just pulses (a visible
        # flash): observed when duplicate full-res deliveries race the viewport
        # crossfade's DEFERRED display-metadata commit -- the redundancy guards
        # read stale _manager_displayed_max_dim, let a second delivery through,
        # and each re-marks _pending_resolution_crossfade, so the second paint
        # fades a full-res buffer over an identical one. There is nothing to
        # dissolve, so never fade it, even if a resolution crossfade was pending.
        if new_max <= int(prev_max * 1.06):
            return False
        if getattr(self, "_pending_resolution_crossfade", False):
            return True
        return new_max > prev_max

    def _viewport_crossfade_parent_and_rect(self) -> tuple:
        container = getattr(self, "single_view_container", None)
        if container is None:
            return None, QRect()
        gv = getattr(self, "gpu_view", None)
        if gv is not None and gv.isVisible():
            if gv.parentWidget() is container:
                return container, gv.geometry()
            top_left = gv.mapTo(container, QPoint(0, 0))
            return container, QRect(top_left, gv.size())
        return container, container.rect()

    def _grab_viewport_snapshot_for_crossfade(self) -> QPixmap | None:
        gv = getattr(self, "gpu_view", None)
        if gv is not None and gv.isVisible() and gv.has_pixmap():
            try:
                snap = gv.capture_viewport_pixmap()
                if snap is not None and not snap.isNull():
                    return snap
            except Exception:
                pass
        target = gv.viewport() if gv is not None else None
        if target is None:
            scroll = getattr(self, "scroll_area", None)
            if scroll is not None:
                target = scroll.viewport()
        if target is None:
            return None
        # Grabbing a live QOpenGLWidget surface can abort the process on
        # Windows (SIGABRT / exit 3) -- the original reason for this guard.
        # Unconditional now (not just win32): capture_viewport_pixmap()'s own
        # grabFramebuffer() attempt above is *also* now unconditionally
        # skipped for GL viewports (macOS crash fix, see its docstring), which
        # means this target.grab() fallback -- previously reached rarely to
        # never on macOS, since grabFramebuffer() used to succeed first -- is
        # now the primary path there. QWidget.grab() forces a synchronous GPU
        # readback of a live GL/Metal surface; that's a plausible source of a
        # visible stutter, and if it reads back mid-composite, a black frame
        # used as the crossfade's fade-from source. Skip it here too and let
        # the caller fall back to the cached previous QPixmap (prev_pm) --
        # already-rendered content, no live GL touch, no readback stall.
        try:
            from PyQt6.QtOpenGLWidgets import QOpenGLWidget

            if isinstance(target, QOpenGLWidget):
                return None
        except Exception:
            pass
        try:
            snap = target.grab()
            if snap is not None and not snap.isNull():
                return snap
        except Exception:
            pass
        return None

    def _run_content_crossfade(
        self,
        old_pixmap: QPixmap,
        *,
        duration_ms: int | None = None,
        on_finished=None,
    ) -> None:
        """Fade the previously displayed content out on top of the freshly set image.

        The new (sharp) pixmap is already shown on image_label; we overlay the old pixmap
        and animate its opacity 1 -> 0 so the upgrade/navigation feels smooth rather than
        snapping. Cheap and isolated from the zoom/pan state machine.
        """
        if not getattr(self, "_crossfade_enabled", True):
            if on_finished is not None:
                on_finished()
            return
        if old_pixmap is None or old_pixmap.isNull():
            if on_finished is not None:
                on_finished()
            return
        try:
            self._cancel_content_crossfade()

            overlay = QLabel(self.image_label)
            overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            overlay.setGeometry(self.image_label.rect())
            overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
            overlay.setScaledContents(False)
            overlay.setPixmap(old_pixmap)
            overlay.show()
            overlay.raise_()

            effect = QGraphicsOpacityEffect(overlay)
            overlay.setGraphicsEffect(effect)
            effect.setOpacity(1.0)

            anim = QPropertyAnimation(effect, b"opacity", self)
            dur = self._crossfade_anim_duration_ms(duration_ms)
            anim.setDuration(dur)
            anim.setStartValue(1.0)
            anim.setEndValue(0.0)
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)

            def _cleanup(_overlay=overlay, _anim=anim, _done=on_finished):
                with contextlib.suppress(Exception):
                    _overlay.deleteLater()
                if getattr(self, "_crossfade_overlay", None) is _overlay:
                    self._crossfade_overlay = None
                if getattr(self, "_crossfade_anim", None) is _anim:
                    self._crossfade_anim = None
                if _done is not None:
                    with contextlib.suppress(Exception):
                        _done()

            anim.finished.connect(_cleanup)
            self._crossfade_overlay = overlay
            self._crossfade_anim = anim
            anim.start()
        except Exception:
            self._cancel_content_crossfade()
            if on_finished is not None:
                with contextlib.suppress(Exception):
                    on_finished()

    def _crossfade_anim_duration_ms(self, duration_ms: int | None) -> int:
        if duration_ms is not None:
            return int(duration_ms)
        if getattr(self, "_gallery_crossfade_next", False):
            self._gallery_crossfade_next = False
            return int(getattr(self, "_gallery_crossfade_duration_ms", 240))
        if getattr(self, "_crossfade_use_resolution_duration", False):
            self._crossfade_use_resolution_duration = False
            return self._resolution_crossfade_duration_ms()
        return int(getattr(self, "_crossfade_duration_ms", 160))

    def _fit_crossfade_overlay_pixmap(
        self, pixmap: QPixmap, geom: QRect, *, is_viewport_snapshot: bool
    ) -> QPixmap:
        if is_viewport_snapshot:
            return pixmap
        cw, ch = max(1, geom.width()), max(1, geom.height())
        return pixmap.scaled(
            cw,
            ch,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _crossfade_overlay_then_apply(
        self,
        parent,
        geom: QRect,
        overlay_pixmap: QPixmap,
        apply_fn,
        *,
        duration_ms: int | None = None,
        on_finished=None,
        pixmap_is_viewport_snapshot: bool = False,
    ) -> None:
        """Show an opaque snapshot overlay, apply the new view underneath, then fade the overlay out."""
        if not getattr(self, "_crossfade_enabled", True):
            with contextlib.suppress(Exception):
                apply_fn()
            if on_finished is not None:
                on_finished()
            return
        if parent is None or overlay_pixmap is None or overlay_pixmap.isNull():
            with contextlib.suppress(Exception):
                apply_fn()
            if on_finished is not None:
                on_finished()
            return
        try:
            self._cancel_content_crossfade()
            overlay = QLabel(parent)
            overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            overlay.setGeometry(geom)
            overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
            overlay.setPixmap(
                self._fit_crossfade_overlay_pixmap(
                    overlay_pixmap, geom, is_viewport_snapshot=pixmap_is_viewport_snapshot
                )
            )
            effect = QGraphicsOpacityEffect(overlay)
            overlay.setGraphicsEffect(effect)
            effect.setOpacity(1.0)
            overlay.show()
            overlay.raise_()
            self._crossfade_overlay = overlay

            def _apply_and_fade() -> None:
                with contextlib.suppress(Exception):
                    apply_fn()
                anim = QPropertyAnimation(effect, b"opacity", self)
                anim.setDuration(self._crossfade_anim_duration_ms(duration_ms))
                anim.setStartValue(1.0)
                anim.setEndValue(0.0)
                anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

                def _cleanup(_overlay=overlay, _anim=anim, _done=on_finished):
                    with contextlib.suppress(Exception):
                        _overlay.deleteLater()
                    if getattr(self, "_crossfade_overlay", None) is _overlay:
                        self._crossfade_overlay = None
                    if getattr(self, "_crossfade_anim", None) is _anim:
                        self._crossfade_anim = None
                    if _done is not None:
                        with contextlib.suppress(Exception):
                            _done()

                anim.finished.connect(_cleanup)
                self._crossfade_anim = anim
                anim.start()

            QTimer.singleShot(0, _apply_and_fade)
        except Exception:
            self._cancel_content_crossfade()
            with contextlib.suppress(Exception):
                apply_fn()
            if on_finished is not None:
                with contextlib.suppress(Exception):
                    on_finished()

    def _run_viewport_crossfade(
        self,
        old_pixmap: QPixmap | None,
        *,
        duration_ms: int | None = None,
        on_finished=None,
        pixmap_is_viewport_snapshot: bool = False,
        apply_fn=None,
    ) -> None:
        """Fade the previous viewport image out (GPU path / gallery preview upgrades)."""
        if old_pixmap is None or old_pixmap.isNull():
            if apply_fn is not None:
                apply_fn()
            elif on_finished is not None:
                on_finished()
            return
        if apply_fn is not None:
            parent, geom = self._viewport_crossfade_parent_and_rect()
            self._crossfade_overlay_then_apply(
                parent,
                geom,
                old_pixmap,
                apply_fn,
                duration_ms=duration_ms,
                on_finished=on_finished,
                pixmap_is_viewport_snapshot=pixmap_is_viewport_snapshot,
            )
            return
        if not getattr(self, "_crossfade_enabled", True):
            if on_finished is not None:
                on_finished()
            return
        parent, geom = self._viewport_crossfade_parent_and_rect()
        if parent is None or old_pixmap is None or old_pixmap.isNull():
            if on_finished is not None:
                on_finished()
            return
        try:
            self._cancel_content_crossfade()
            overlay = QLabel(parent)
            overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            overlay.setGeometry(geom)
            overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
            overlay.setPixmap(
                self._fit_crossfade_overlay_pixmap(
                    old_pixmap, geom, is_viewport_snapshot=pixmap_is_viewport_snapshot
                )
            )
            overlay.show()
            overlay.raise_()

            effect = QGraphicsOpacityEffect(overlay)
            overlay.setGraphicsEffect(effect)
            effect.setOpacity(1.0)

            anim = QPropertyAnimation(effect, b"opacity", self)
            anim.setDuration(self._crossfade_anim_duration_ms(duration_ms))
            anim.setStartValue(1.0)
            anim.setEndValue(0.0)
            anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

            def _cleanup(_overlay=overlay, _anim=anim, _done=on_finished):
                with contextlib.suppress(Exception):
                    _overlay.deleteLater()
                if getattr(self, "_crossfade_overlay", None) is _overlay:
                    self._crossfade_overlay = None
                if getattr(self, "_crossfade_anim", None) is _anim:
                    self._crossfade_anim = None
                if _done is not None:
                    with contextlib.suppress(Exception):
                        _done()

            anim.finished.connect(_cleanup)
            self._crossfade_overlay = overlay
            self._crossfade_anim = anim
            anim.start()
        except Exception:
            self._cancel_content_crossfade()
            if on_finished is not None:
                with contextlib.suppress(Exception):
                    on_finished()

    def _zoom_in_to_image_point_finish(self) -> None:
        """``zoom_center_point`` already set; match double-click zoom-in (full decode + 100%)."""
        import logging

        logger = logging.getLogger(__name__)
        fp = getattr(self, "current_file_path", None)
        if not fp:
            return
        gv = getattr(self, "gpu_view", None)
        pm = getattr(self, "current_pixmap", None)
        if gv is not None and gv.has_pixmap():
            self._allow_full_decode_for_user_zoom()
            self._try_sync_full_before_zoom()
            should_upgrade = self._needs_full_resolution_upgrade(fp)
            focus_jump = bool(
                getattr(self, "_pending_zoom_focus_subject", False)
                or getattr(self, "_focus_zoom_anchor_active", False)
            )
            if should_upgrade and not self._adjust_panel_active():
                self._pending_zoom = True
                self._pending_zoom_center = (
                    self.zoom_center_point
                    if getattr(self, "zoom_center_point", None) is not None
                    else None
                )
                self._pending_zoom_thumbnail_size = (
                    pm.size() if pm is not None and not pm.isNull() else None
                )
                gv._zoom_intent_100 = True
                self._request_full_res_decode(fp, priority_current=True, urgent=True)
                if not focus_jump:
                    self._gpu_zoom_in_to_point_finish()
                return
            self._restore_adjust_hires_before_zoom()
            self._gpu_zoom_in_to_point_finish()
            if self._adjust_panel_active():
                self._apply_adjust_panel_preview(full_quality=True)
            return
        if pm is None or pm.isNull():
            return

        should_upgrade = self._needs_full_resolution_upgrade(fp)

        if should_upgrade and self._adjust_panel_active():
            self._restore_adjust_hires_before_zoom()
            self.fit_to_window = False
            self.current_zoom_level = 1.0
            if self.current_pixmap:
                self.zoom_to_point()
            self._apply_adjust_panel_preview(full_quality=True)
            return

        if should_upgrade:
            if getattr(self, "_full_resolution_loading", False):
                self._pending_zoom = True
                self._pending_zoom_center = self.zoom_center_point
                self._pending_zoom_thumbnail_size = self.current_pixmap.size() if self.current_pixmap else None
                return
            if not getattr(self, "_full_resolution_loading", False):
                logger.info("Zoom-to-point -- triggering full resolution load path")
                cached_full = self.image_cache.get_full_image(self.current_file_path)
                if cached_full is not None:
                    if hasattr(cached_full, "shape"):
                        cached_h, cached_w = cached_full.shape[0], cached_full.shape[1]
                    elif hasattr(cached_full, "width") and hasattr(cached_full, "height"):
                        cached_h, cached_w = cached_full.height(), cached_full.width()
                    else:
                        cached_h, cached_w = 0, 0
                    cached_max_dim = max(cached_w, cached_h)
                    if cached_max_dim >= 3000:
                        self._full_resolution_loading = True
                        self._maintain_zoom_on_navigation = True
                        self._orientation_already_applied = True

                        old_current_size = (
                            self.current_pixmap.size() if self.current_pixmap else None
                        )

                        self.display_numpy_image(cached_full)
                        self._is_half_size_displayed = False
                        self._full_resolution_loading = False

                        if self.current_pixmap is not None and old_current_size is not None:
                            scale_x = (
                                self.current_pixmap.width() / old_current_size.width()
                                if old_current_size.width() > 0
                                else 1.0
                            )
                            scale_y = (
                                self.current_pixmap.height() / old_current_size.height()
                                if old_current_size.height() > 0
                                else 1.0
                            )
                            if getattr(self, "zoom_center_point", None):
                                self.zoom_center_point = QPoint(
                                    int(self.zoom_center_point.x() * scale_x),
                                    int(self.zoom_center_point.y() * scale_y),
                                )
                        if hasattr(self, "_maintain_zoom_on_navigation"):
                            delattr(self, "_maintain_zoom_on_navigation")
                        if getattr(self, "gpu_view", None) is None:
                            self.fit_to_window = False
                            self.current_zoom_level = 1.0
                            self.zoom_to_point()
                        return

                self._load_full_resolution_on_demand()
                self._pending_zoom = True
                self._pending_zoom_center = (
                    self.zoom_center_point
                    if getattr(self, "zoom_center_point", None)
                    else None
                )
                self._pending_zoom_thumbnail_size = (
                    self.current_pixmap.size() if self.current_pixmap else None
                )
                self.status_bar.showMessage("Loading full resolution for 100% zoom...")
                return

        if getattr(self, "gpu_view", None) is not None:
            return

        self.fit_to_window = False
        self.current_zoom_level = 1.0

        if self.current_pixmap:
            available_size = self.scroll_area.size()
            safe_width = max(1, available_size.width() - 20)
            safe_height = max(1, available_size.height() - 20)
            fit_scale_x = safe_width / max(1, self.current_pixmap.width())
            fit_scale_y = safe_height / max(1, self.current_pixmap.height())
            fit_scale = min(fit_scale_x, fit_scale_y)
            if self.current_zoom_level <= fit_scale * 1.05:
                self.current_zoom_level = fit_scale * 2.0

        if self.current_zoom_level > 1.05:
            self.apply_zoom_and_pan()
        else:
            self.zoom_to_point()

    def _single_view_is_fit_mode(self) -> bool:
        gv = getattr(self, "gpu_view", None)
        if gv is not None:
            gv._sync_fit_mode_flag()
            return gv.wants_zoom_in_toggle()
        if bool(getattr(self, "fit_to_window", True)):
            return True
        pm = getattr(self, "current_pixmap", None)
        if pm is None or pm.isNull():
            return True
        fit_scale = self._fit_zoom_level()
        zoom = float(getattr(self, "current_zoom_level", 1.0))
        return zoom <= fit_scale * 1.05

    def _single_view_is_zoomed_in(self) -> bool:
        return not self._single_view_is_fit_mode()

    def _preserve_zoom_navigation_active(self) -> bool:
        """True while arrow/filmstrip nav should keep zoom and avoid soft preview tiers."""
        return bool(
            getattr(self, "_preserve_nav_zoom_active", False)
            or getattr(self, "_maintain_zoom_on_navigation", False)
        )
