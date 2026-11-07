"""Gallery mixin part 2 of 7 for JustifiedGallery."""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np
from PyQt6.QtCore import QSize, QTimer
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QScrollArea

from .helpers import _env_int, _focus_gallery_switch_logs, _orient_debug_enabled

logger = logging.getLogger(__name__)

class GalleryPart2:

    def _get_first_visible_image_index(self) -> int:
        idx = self._get_upper_left_visible_image_index()
        if idx >= 0:
            return idx
        scroll = self._scroll_area_widget()
        if scroll is None:
            return -1
        scroll_y = scroll.verticalScrollBar().value()
        for i, item in enumerate(self._gallery_layout_items):
            if item["rect"].bottom() > scroll_y:
                return i
        return -1

    def begin_zoom_session(self) -> bool:
        """Begin interactive zoom; capture upper-left anchor for scroll restore."""
        if not self._gallery_layout_items:
            return False
        scroll = self._scroll_area_widget()
        scroll_y = (
            int(scroll.verticalScrollBar().value()) if scroll is not None else 0
        )
        idx = self._get_upper_left_visible_image_index()
        if idx < 0:
            idx = self._get_first_visible_image_index()

        self._is_zooming = True
        if 0 <= idx < len(self._gallery_layout_items):
            item = self._gallery_layout_items[idx]
            rect = item["rect"]
            self._zoom_anchor_state = {
                "file_path": item.get("file_path"),
                "offset_y_px": int(rect.top() - scroll_y),
            }
        else:
            self._zoom_anchor_state = self._capture_scroll_anchor_state()

        self._stop_layout_rebuild_timers()
        self.hide_loading_message()
        return True

    def apply_gallery_zoom(self, row_height: int) -> None:
        """Justified relayout at the new row height; keep the zoom anchor fixed vertically."""
        row_height = max(220, min(500, int(row_height)))
        if not self._is_zooming:
            self.begin_zoom_session()
        self.TARGET_ROW_HEIGHT = row_height
        self._gallery_zoom_rebuild = True
        try:
            self.build_gallery(force=True)
        finally:
            self._gallery_zoom_rebuild = False
        self._request_load_visible_images(0)

    def end_zoom_session(self) -> None:
        """Commit zoom session; defer any missing thumbs until after slider release."""
        if not self._is_zooming:
            return
        self._is_zooming = False
        self._zoom_anchor_state = None
        self._last_layout_content_height = self._total_content_height
        self._last_build_ts = time.time()
        self._request_load_visible_images(30)

    def _current_image_layout_index(self) -> int | None:
        """Layout index for the viewer's current single-view / navigation file."""
        pv = getattr(self, "parent_viewer", None)
        if pv is None:
            return None
        current = getattr(pv, "current_file_path", None)
        if not current or not self._gallery_layout_items:
            return None
        resolved = self._resolve_gallery_path(current)
        if not resolved:
            return None
        indices = self._path_to_indices.get(resolved)
        if not indices:
            return None
        return indices[0]

    def _layout_index_at_scroll_y(self, anchor_y: int) -> int | None:
        items = self._gallery_layout_items
        if not items:
            return None
        low, high = 0, len(items) - 1
        idx = 0
        while low <= high:
            mid = (low + high) // 2
            if items[mid]["rect"].bottom() < anchor_y:
                low = mid + 1
            else:
                idx = mid
                high = mid - 1
        if idx < 0 or idx >= len(items):
            return None
        return idx

    def _gallery_prefetch_center_index(self, scroll_y: int, viewport_h: int) -> int | None:
        """Center for bidirectional JPEG prefetch: current image, else viewport center."""
        current_idx = self._current_image_layout_index()
        viewport_idx = self._layout_index_at_scroll_y(scroll_y + max(1, viewport_h) // 2)
        if current_idx is None:
            return viewport_idx
        if getattr(self, "_pending_scroll_to_path", None):
            return current_idx
        pending = self._entry_anchor_path()
        if (
            pending
            and not getattr(self, "_entry_prefetch_abandoned", False)
            and self._entry_prefetch_active(current_idx, scroll_y, viewport_h)
        ):
            return current_idx
        if viewport_idx is None:
            return current_idx
        if abs(current_idx - viewport_idx) <= self._gallery_center_prefetch_radius():
            return current_idx
        return viewport_idx

    def _entry_anchor_path(self) -> str | None:
        """File path used to restore scroll when entering gallery."""
        pending = getattr(self, "_pending_scroll_to_path", None)
        if pending:
            return pending
        pv = getattr(self, "parent_viewer", None)
        if pv is not None:
            current = getattr(pv, "current_file_path", None)
            if current:
                return current
            target = getattr(pv, "_gallery_scroll_target_path", None)
            if target:
                return target
        return None

    def _entry_anchor_layout_index(self) -> int | None:
        if not self._gallery_layout_items:
            return None
        path = self._entry_anchor_path()
        if not path:
            return None
        resolved = self._resolve_gallery_path(path)
        if not resolved:
            return None
        indices = self._path_to_indices.get(resolved)
        if not indices:
            return None
        return indices[0]

    def _layout_indices_near(self, center_idx: int, radius: int) -> list[int]:
        n = len(self._gallery_layout_items)
        if n <= 0:
            return []
        center_idx = max(0, min(int(center_idx), n - 1))
        ordered: list[int] = []
        seen: set[int] = set()
        for offset in range(int(radius) + 1):
            for idx in (center_idx + offset, center_idx - offset):
                if 0 <= idx < n and idx not in seen:
                    seen.add(idx)
                    ordered.append(idx)
        return ordered

    def _gallery_center_prefetch_radius(self) -> int:
        return _env_int("RAWVIEWER_GALLERY_ENTRY_PREFETCH_RADIUS", 48, minimum=4)

    def _entry_prefetch_radius(self) -> int:
        return self._gallery_center_prefetch_radius()

    def _entry_prefetch_active(
        self, anchor_idx: int | None, scroll_y: int, viewport_h: int
    ) -> bool:
        """True when scroll has not yet reached the entry thumbnail neighborhood."""
        if anchor_idx is None:
            return False
        if getattr(self, "_pending_scroll_to_path", None):
            return True
        if anchor_idx < 0 or anchor_idx >= len(self._gallery_layout_items):
            return False
        rect = self._gallery_layout_items[anchor_idx]["rect"]
        margin = max(viewport_h // 3, 80)
        view_top = scroll_y - margin
        view_bottom = scroll_y + viewport_h + margin
        return rect.bottom() < view_top or rect.top() > view_bottom

    def _prioritize_entry_paths(self, paths: list[str], cap: int | None = None) -> list[str]:
        """Order paths so EXIF/layout work near the entry file runs first."""
        if not paths:
            return []
        anchor_path = self._entry_anchor_path()
        if not anchor_path or not self.images:
            return paths[:cap] if cap else list(paths)
        resolved = self._resolve_gallery_path(anchor_path)
        if not resolved:
            return paths[:cap] if cap else list(paths)
        try:
            anchor_list_idx = self.images.index(resolved)
        except ValueError:
            return paths[:cap] if cap else list(paths)

        path_set = set(paths)
        seed_radius = min(max(self.BUILD_EXIF_PREFETCH_MAX // 2, 16), 96)
        ordered: list[str] = []
        seen: set[str] = set()
        for offset in range(seed_radius + 1):
            for idx in (anchor_list_idx + offset, anchor_list_idx - offset):
                if 0 <= idx < len(self.images):
                    p = self.images[idx]
                    if p in path_set and p not in seen:
                        seen.add(p)
                        ordered.append(p)
        for p in paths:
            if p not in seen:
                ordered.append(p)
        if cap is not None:
            return ordered[:cap]
        return ordered

    def _entry_prefetch_paths(self, anchor_idx: int) -> set[str]:
        return self._bidirectional_jpeg_paths_from_center(anchor_idx)

    def _bidirectional_jpeg_paths_from_center(self, center_idx: int) -> set[str]:
        """Paths within ±radius layout indices of center (up/down in the gallery grid)."""
        paths: set[str] = set()
        for idx in self._layout_indices_near(center_idx, self._gallery_center_prefetch_radius()):
            path = self._gallery_layout_items[idx].get("file_path")
            if path:
                paths.add(path)
        return paths

    def _missing_thumbnail_stage(self, path: str) -> bool:
        return not self._thumbnail_cache.get((path, self._thumb_base_key))

    def _missing_load_stages_for_path(
        self, path: str, *, jpeg_only: bool = False
    ) -> set[str]:
        stages: set[str] = set()
        if self._missing_thumbnail_stage(path):
            stages.add("thumbnail")
        if jpeg_only:
            return stages
        meta = self._metadata_cache.get(path)
        if not meta or meta.get("original_width") is None or meta.get("original_height") is None:
            stages.add("exif")
        return stages

    def _path_layout_distance(self, path: str, center_idx: int | None) -> int:
        if center_idx is None or not path:
            return 1_000_000
        indices = self._path_to_indices.get(path)
        if not indices:
            return 1_000_000
        return min(abs(i - center_idx) for i in indices)

    def set_images(
        self,
        images: list[str],
        bulk_metadata: dict[str, Any] | None = None,
        *,
        force_rebuild: bool = False,
    ):
        """
        Public API used by `main.py` to populate/refresh the gallery.
        Kept compatible with the legacy gallery interface.
        """
        try:
            if self.load_manager is None:
                # Ensure worker wiring exists before first render pass.
                self._post_init()
            new_images = list(images or [])
            # Drop caches keyed by path when the set of paths changes. Both
            # were only ever added to: _metadata_cache holds a full EXIF dict
            # per file and had no cap at all, so browsing folder after folder
            # grew it for the life of the session. The thumbnail cache is
            # byte-budgeted, but keeping another folder's tiles just spends
            # that budget on pixmaps nothing will ask for again.
            _prev = set(self.images or ())
            _next = set(new_images)
            if _prev and _prev != _next:
                self._prune_path_caches(_next)
            self._gallery_folder_token = self._current_folder_load_token()
            if new_images:
                self.hide_empty_message()
            same_order = (
                len(new_images) == len(self.images)
                and all(a == b for a, b in zip(new_images, self.images, strict=False))
            )
            if _focus_gallery_switch_logs():
                logger.debug(
                    "[MODESWITCH] gallery.set_images called; count=%d load_manager=%s force=%s same_order=%s",
                    len(new_images),
                    self.load_manager is not None,
                    force_rebuild,
                    same_order,
                )
            if not force_rebuild and same_order and self._layout_matches_current_images(new_images):
                if bulk_metadata:
                    self._metadata_cache.update(bulk_metadata)

                # Fetch updated metadata near the entry anchor only — never
                # walk all ~N missing paths on the UI thread (large folders).
                if self.parent_viewer and hasattr(self.parent_viewer, "image_cache"):
                    paths = [img for img in self.images if isinstance(img, str)]
                    missing_paths = [
                        p
                        for p in paths
                        if p not in self._metadata_cache
                        or self._metadata_cache[p].get("minimal_preview_exif")
                    ]
                    if missing_paths:
                        fetch_paths = self._prioritize_entry_paths(
                            missing_paths, cap=self.BUILD_EXIF_PREFETCH_MAX
                        )
                        bulk_fetched = self.parent_viewer.image_cache.get_multiple_exif(
                            fetch_paths
                        )
                        if bulk_fetched:
                            self._metadata_cache.update(bulk_fetched)
                            # If any aspect ratios/rotations changed, trigger a layout rebuild
                            layout_changed = False
                            for path, _exif in bulk_fetched.items():
                                if self.on_visual_rotation_changed(path, defer_rebuild=True):
                                    layout_changed = True
                            if layout_changed:
                                self.build_gallery(force=True)
                                return

                self._reposition_thumbnail_widgets()
                # Same-order re-entry: layout/scroll are already valid. Do not
                # park tile loads behind scroll_to_file retries — that left the
                # gallery blank for seconds on large folders (no load_visible).
                if self._pending_scroll_to_path:
                    QTimer.singleShot(0, self._apply_pending_scroll_to_file)
                self._request_load_visible_images(0)
                return

            self._gallery_generation += 1
            self.images = new_images
            self._rebuild_normalized_map()
            if not same_order or force_rebuild:
                self._reset_gallery_scroll_position()
            if not same_order or force_rebuild:
                # Order changed: invalidate layout so build_gallery cannot skip via count-only checks.
                self._building = False
                self._pending_gallery_build = False
                self._pending_build_metadata = None
                self._pending_build_force = False
                self.clear_thumbnail_widgets()
                self._gallery_layout_items.clear()
                self._path_to_indices.clear()
                self._layout_image_sequence = ()
                self._last_layout_content_height = 0
            self._active_tasks.clear()
            self._gallery_set_images_ts = time.time()
            self._first_thumb_ready_after_set = False

            if bulk_metadata:
                self._metadata_cache.update(bulk_metadata)

            # Cancel/clear outstanding thumbnail work for previous generation
            try:
                for fp in list(self._requested_thumbnail_paths):
                    try:
                        if self.load_manager is not None:
                            self.load_manager.cancel_task(fp)
                    except Exception:
                        pass
            finally:
                self._requested_thumbnail_paths.clear()
                self._placeholder_thumb_paths.clear()

            if not self.images:
                self._building = False
                self._pending_gallery_build = False
                self._pending_build_metadata = None
                self._pending_build_force = False
                self.clear_thumbnail_widgets(destroy=True)
                self._gallery_layout_items.clear()
                self._path_to_indices.clear()
                parent_scroll = self._scroll_area
                if parent_scroll is None:
                    parent_scroll = self.parent()
                    while parent_scroll and not isinstance(parent_scroll, QScrollArea):
                        parent_scroll = parent_scroll.parent()
                viewport_h = parent_scroll.viewport().height() if isinstance(parent_scroll, QScrollArea) else 0
                self._total_content_height = max(1, int(viewport_h))
                self.setMinimumHeight(self._total_content_height)
                self.resize(max(self.width(), self._get_viewport_width()), self._total_content_height)
                if isinstance(parent_scroll, QScrollArea):
                    parent_scroll.verticalScrollBar().setValue(parent_scroll.verticalScrollBar().minimum())
                self.update()
                return

            # Rebuild on next event loop tick (avoid doing heavy UI work inside caller)
            build_gen = self._gallery_generation
            need_force = force_rebuild or not same_order
            build_done = {"gen": -1}

            def _run_build():
                if self._gallery_generation != build_gen:
                    return
                if build_done["gen"] == build_gen:
                    return
                if self._building:
                    return
                if not need_force and self._layout_matches_current_images():
                    build_done["gen"] = build_gen
                    self._reposition_thumbnail_widgets()
                    self._request_load_visible_images(20)
                    return
                self.build_gallery(bulk_metadata=bulk_metadata, force=need_force)
                build_done["gen"] = build_gen

            QTimer.singleShot(0, _run_build)
            # Backup: mode-switch bursts can delay the first tick on large folders.
            QTimer.singleShot(200, _run_build)
        except Exception:
            logger.exception("[GALLERY] set_images() failed")

    def _rebuild_normalized_map(self):
        """Rebuild the O(1) normalized to canonical path mapping."""
        self._normalized_to_canonical = {}
        for img in self.images:
            if isinstance(img, str):
                try:
                    norm = os.path.normcase(os.path.abspath(img))
                except OSError:
                    norm = os.path.normcase(img)
                self._normalized_to_canonical[norm] = img

    def _resolve_gallery_path(self, file_path: str | None) -> str | None:
        """Match file_path to the canonical path string used in ``self.images``."""
        if not file_path or not self.images:
            return None
        try:
            target = os.path.normcase(os.path.abspath(file_path))
        except OSError:
            target = os.path.normcase(file_path)

        if hasattr(self, "_normalized_to_canonical") and target in self._normalized_to_canonical:
            return self._normalized_to_canonical[target]

        target_base = os.path.normcase(os.path.basename(file_path))
        for img in self.images:
            if not isinstance(img, str):
                continue
            try:
                if os.path.normcase(os.path.abspath(img)) == target:
                    return img
            except OSError:
                pass
            if os.path.normcase(os.path.basename(img)) == target_base:
                return img
        return None

    def scroll_to_file(self, file_path: str | None):
        """Scroll so file_path is visible after the justified layout is built."""
        if not file_path:
            return
        self._pending_scroll_to_path = file_path
        self._pending_scroll_anchor_state = {
            "file_path": file_path,
            "offset_px": 0,
        }
        self._scroll_to_file_attempts = 0
        self._schedule_scroll_to_file_retry(0)

    def _schedule_scroll_to_file_retry(self, delay_ms: int) -> None:
        QTimer.singleShot(max(0, int(delay_ms)), self._apply_pending_scroll_to_file)

    def _abandon_pending_scroll_to_file(self) -> None:
        """Give up on the entry scroll and unblock tile loading at the current offset."""
        self._pending_scroll_to_path = None
        self._pending_scroll_anchor_state = None
        self._scroll_to_file_attempts = 0
        logger.info("[GALLERY] scroll_to_file abandoned after max retries; loading at current offset")
        self._request_load_visible_images(20)

    def _apply_pending_scroll_to_file(self):
        anchor_state = getattr(self, "_pending_scroll_anchor_state", None)
        path = self._pending_scroll_to_path
        if not anchor_state and not path:
            return

        # Guard 1: Verify the justified layout is fully built/updated for the current images
        if not self._layout_image_sequence or len(self._layout_image_sequence) != len(self.images):
            attempts = getattr(self, "_scroll_to_file_attempts", 0) + 1
            self._scroll_to_file_attempts = attempts
            if attempts < 40:
                self._schedule_scroll_to_file_retry(50)
            else:
                self._abandon_pending_scroll_to_file()
            return

        if anchor_state:
            if self._restore_scroll_anchor_state(anchor_state):
                self._pending_scroll_to_path = None
                self._pending_scroll_anchor_state = None
                self._scroll_to_file_attempts = 0
                self._request_load_visible_images(20)
                return
            attempts = getattr(self, "_scroll_to_file_attempts", 0) + 1
            self._scroll_to_file_attempts = attempts
            if attempts < 40:
                self._schedule_scroll_to_file_retry(50)
            else:
                self._abandon_pending_scroll_to_file()
            return

        if not path:
            return

        resolved = self._resolve_gallery_path(path)
        if not resolved:
            pv = getattr(self, "parent_viewer", None)
            if pv is not None and hasattr(pv, "_gallery_viewport_scroll_path"):
                try:
                    alt = pv._gallery_viewport_scroll_path()
                    if alt:
                        resolved = self._resolve_gallery_path(alt)
                except Exception:
                    pass
        if not resolved:
            pv = getattr(self, "parent_viewer", None)
            cfi = getattr(pv, "current_file_index", -1) if pv is not None else -1
            images = self.images or []
            if 0 <= cfi < len(images):
                resolved = images[cfi]
            elif pv is not None:
                resolved = self._resolve_gallery_path(
                    getattr(pv, "current_file_path", None)
                )
        if not resolved:
            attempts = getattr(self, "_scroll_to_file_attempts", 0) + 1
            self._scroll_to_file_attempts = attempts
            if attempts < 40:
                self._schedule_scroll_to_file_retry(50)
            else:
                if attempts == 40:
                    logger.warning(
                        "[GALLERY] scroll_to_file gave up: %s not in layout (%d images)",
                        os.path.basename(path) if path else "?",
                        len(self.images or []),
                    )
                self._abandon_pending_scroll_to_file()
            return
        indices = self._path_to_indices.get(resolved)
        if not indices:
            attempts = getattr(self, "_scroll_to_file_attempts", 0) + 1
            self._scroll_to_file_attempts = attempts
            if attempts < 40:
                self._schedule_scroll_to_file_retry(50)
            else:
                self._abandon_pending_scroll_to_file()
            return
        idx = indices[0]
        if idx < 0 or idx >= len(self._gallery_layout_items):
            attempts = getattr(self, "_scroll_to_file_attempts", 0) + 1
            self._scroll_to_file_attempts = attempts
            if attempts < 40:
                self._schedule_scroll_to_file_retry(50)
            else:
                self._abandon_pending_scroll_to_file()
            return
        p = self._scroll_area
        if p is None:
            p = self.parent()
            while p and not isinstance(p, QScrollArea):
                p = p.parent()
        if not isinstance(p, QScrollArea):
            attempts = getattr(self, "_scroll_to_file_attempts", 0) + 1
            self._scroll_to_file_attempts = attempts
            if attempts < 40:
                self._schedule_scroll_to_file_retry(50)
            else:
                self._abandon_pending_scroll_to_file()
            return

        # Guard 2: Verify the scrollbar's maximum range has updated to match content_h
        sb = p.verticalScrollBar()
        expected_max = self._total_content_height - p.viewport().height()
        if expected_max > 0 and sb.maximum() < expected_max - 50:
            attempts = getattr(self, "_scroll_to_file_attempts", 0) + 1
            self._scroll_to_file_attempts = attempts
            if attempts < 40:
                self._schedule_scroll_to_file_retry(50)
            else:
                self._abandon_pending_scroll_to_file()
            return

        rect = self._gallery_layout_items[idx]["rect"]
        target = max(sb.minimum(), min(rect.top(), sb.maximum()))
        sb.setValue(target)
        self._pending_scroll_to_path = None
        self._pending_scroll_anchor_state = None
        self._scroll_to_file_attempts = 0
        self._request_load_visible_images(20)

    def _scaled_cache_key(self, file_path: str, target_size: QSize, fast: bool = False):
        """Cache key for crop-to-fit thumbnails at exact tile physical size."""
        th = int(target_size.height())
        tw = int(target_size.width())
        rot = self._get_rotation_degrees_for_path(file_path)
        if th <= 0 or tw <= 0:
            return (file_path, self._thumb_base_key, rot, "fast") if fast else (file_path, self._thumb_base_key, rot)
        return (file_path, tw, th, rot, "fast") if fast else (file_path, tw, th, rot)

    def _pixmap_logical_size(pixmap: QPixmap, dpr: float) -> QSize:
        safe_dpr = dpr if dpr > 0 else 1.0
        return QSize(
            int(round(pixmap.width() / safe_dpr)),
            int(round(pixmap.height() / safe_dpr)),
        )

    def _gallery_base_meets_tile(self, base: QPixmap, physical_size: QSize) -> bool:
        """True when cached base pixmap is sharp enough for this tile."""
        if base is None or base.isNull():
            return False
        try:
            from image_load_manager import _gallery_grid_min_dim

            dpr = float(base.devicePixelRatio() or 1.0)
            effective = max(int(base.width() * dpr), int(base.height() * dpr))
            need = max(int(physical_size.width()), int(physical_size.height()))
            floor = min(_gallery_grid_min_dim(), int(need * 0.85))
            return effective >= floor
        except Exception:
            return True

    def _fit_tile_pixmap(
        self, file_path: str, base: QPixmap, physical_size: QSize, dpr: float, fast: bool = False
    ) -> QPixmap:
        fitted = self._fit_rotated_thumbnail(file_path, base, physical_size, fast=fast)
        fitted.setDevicePixelRatio(dpr)
        self._thumbnail_cache.put(self._scaled_cache_key(file_path, physical_size, fast=fast), fitted)
        self._enforce_thumbnail_cache_budget()
        return fitted

    def _get_rotation_degrees_for_path(self, file_path: str) -> int:
        """Read per-file visual rotation from parent viewer (0/90/180/270)."""
        pv = getattr(self, "parent_viewer", None)
        if pv is None:
            return 0
        getter = getattr(pv, "_get_visual_rotation_degrees", None)
        if getter is None:
            return 0
        try:
            return int(getter(file_path)) % 360
        except Exception:
            return 0

    def _display_aspect(self, file_path: str, base_aspect: float) -> float:
        """Aspect ratio after non-destructive visual rotation."""
        if base_aspect <= 0:
            return base_aspect
        if self._get_rotation_degrees_for_path(file_path) in (90, 270):
            return 1.0 / base_aspect
        return base_aspect

    def _metadata_display_aspect(self, file_path: str) -> float | None:
        """Layout aspect from EXIF display dimensions (matches index thumbnail pixels)."""
        m = self._metadata_exif_for_path(file_path)
        if not m or not m.get("original_width") or not m.get("original_height"):
            return None
        from common_image_loader import exif_display_dimensions

        w, h = int(m["original_width"]), int(m["original_height"])
        o = int(m.get("orientation", 1) or 1)
        dw, dh = exif_display_dimensions(w, h, o)
        if dh <= 0:
            return None
        return self._display_aspect(file_path, dw / dh)

    def _aspect_orientation_flag(self, ar: float) -> int:
        """-1 portrait, 0 near-square, 1 landscape."""
        if ar <= 0:
            return 0
        if ar < 0.92:
            return -1
        if ar > 1.08:
            return 1
        return 0

    def _raw_aspect_undo_user_rotation(self, file_path: str, display_ar: float) -> float:
        if display_ar <= 0:
            return display_ar
        if self._get_rotation_degrees_for_path(file_path) in (90, 270):
            return 1.0 / display_ar
        return display_ar

    def _raw_aspect_from_pixmap(self, file_path: str, pixmap: QPixmap) -> float | None:
        """Measured width/height for layout, reconciled with EXIF when orientation disagrees."""
        if pixmap is None or pixmap.isNull() or pixmap.height() <= 0:
            return None
        px_ar = pixmap.width() / pixmap.height()
        meta_ar = self._metadata_display_aspect(file_path)
        if meta_ar is not None and meta_ar > 0:
            px_o = self._aspect_orientation_flag(px_ar)
            meta_o = self._aspect_orientation_flag(meta_ar)
            if px_o != 0 and meta_o != 0 and px_o != meta_o:
                return self._raw_aspect_undo_user_rotation(file_path, meta_ar)
            if abs(px_ar - meta_ar) / max(meta_ar, 0.01) > 0.35:
                return self._raw_aspect_undo_user_rotation(file_path, meta_ar)
        return px_ar

    def _apply_edits_to_tile(self, file_path: str, arr):
        """Render saved XMP edits into a tile buffer at delivery time.

        ndarray in, ndarray out (QImage and other inputs pass through). Cost
        is ~5ms per 320px tile / ~35ms at 720px rows, paid only for files
        with non-default saved edits (see raw_adjustments.apply_saved_edits_
        for_display); nothing is written back to the thumbnail caches.
        """
        try:
            from raw_adjustments import apply_saved_edits_for_display

            return apply_saved_edits_for_display(file_path, arr)
        except Exception:
            return arr

    def _orient_gallery_thumbnail_array(self, file_path: str, arr: np.ndarray) -> np.ndarray:
        try:
            from common_image_loader import finalize_index_thumbnail_array
            from image_cache import get_image_cache

            fixed = finalize_index_thumbnail_array(file_path, arr, cache=get_image_cache())
            return fixed if fixed is not None else arr
        except Exception:
            return arr

    def _metadata_exif_for_path(self, file_path: str) -> dict:
        """EXIF dict for layout/orientation — fall back to global cache when local cache is cold."""
        m = self._metadata_cache.get(file_path)
        if m and m.get("original_width") and m.get("original_height"):
            return m
        if file_path in self._metadata_fetch_attempted:
            # build_gallery() already bulk-fetched this folder's EXIF once via
            # get_multiple_exif(); if it's still missing here there's nothing new
            # in the global cache to find. Falling through to get_exif() below would
            # re-run a per-path SQLite SELECT for every still-missing file on every
            # rebuild — with thousands of paths on a large folder, and the background
            # indexer concurrently writing the same DB, that turned build_gallery's
            # per-item layout loop into multi-second stalls (measured 7-13s for 6881
            # items). Wait for the metadata-ready event to populate _metadata_cache.
            return m or {}
        try:
            from image_cache import get_image_cache

            cached = get_image_cache().get_exif(file_path)
            if cached:
                return cached
        except Exception:
            pass
        return m or {}

    def _orient_gallery_base_pixmap(self, file_path: str, pixmap: QPixmap) -> QPixmap:
        """Rotate base thumbnail to match container EXIF display orientation."""
        if pixmap is None or pixmap.isNull():
            return pixmap
        m = self._metadata_exif_for_path(file_path)
        if not m:
            if _orient_debug_enabled():
                logger.info(
                    "[ORIENT] base file=%s pix=%dx%d(%s) EXIF=MISSING -> no rotation",
                    os.path.basename(file_path), pixmap.width(), pixmap.height(),
                    "P" if pixmap.height() > pixmap.width() else "L",
                )
            return pixmap
        from common_image_loader import apply_container_orientation_to_pixmap

        if _orient_debug_enabled():
            from common_image_loader import (
                exif_pixels_display_oriented,
                exif_rotation_degrees_for_pixmap,
                pixmap_matches_exif_display,
            )
            ow = int(m.get("original_width") or 0)
            oh = int(m.get("original_height") or 0)
            o = int(m.get("orientation", 1) or 1)
            _deg = exif_rotation_degrees_for_pixmap(pixmap.width(), pixmap.height(), ow, oh, o)
            _match = pixmap_matches_exif_display(
                pixmap.width(), pixmap.height(), ow, oh, o,
                pixels_display_oriented=exif_pixels_display_oriented(m),
            )
            logger.info(
                "[ORIENT] base file=%s pix=%dx%d(%s) exif(o=%s ow=%d oh=%d) match=%s deg=%s",
                os.path.basename(file_path), pixmap.width(), pixmap.height(),
                "P" if pixmap.height() > pixmap.width() else "L", o, ow, oh, _match, _deg,
            )
        return apply_container_orientation_to_pixmap(pixmap, m)
