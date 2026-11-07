"""Gallery mixin part 6 of 7 for JustifiedGallery."""
from __future__ import annotations

import contextlib
import logging
import os
import time

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QScrollArea

from common_image_loader import is_raw_file

from .helpers import (
    _batch_tile_apply_enabled,
    _focus_gallery_switch_logs,
    _gallery_idle_load_priority,
    _gallery_idle_preload_batch,
    _gallery_idle_preload_ms,
    _orient_debug_enabled,
    _thumbnail_data_to_base_pixmap,
    _tile_apply_batch_size,
)

logger = logging.getLogger(__name__)

class GalleryPart6:

    def pause_idle_preload(self):
        """Immediately pause idle background thumbnail generation."""
        if hasattr(self, "_idle_preload_timer") and self._idle_preload_timer.isActive():
            self._idle_preload_timer.stop()
        if self.load_manager:
            idle_priority = _gallery_idle_load_priority()
            self.load_manager.cancel_tasks_by_priority(idle_priority)

    def _preload_remaining_thumbnails_background(self):
        """Silently preload thumbnails for out-of-viewport images during idle stages to smooth out future scrolling."""
        if self._building or not self.images or self.load_manager is None:
            return

        try:
            from common_image_loader import io_pressure_active

            if io_pressure_active():
                return
        except Exception:
            pass

        # Runs in gallery view AND while idle in single view: warming the
        # thumbnail caches during single-view idle makes a later gallery
        # switch paint instantly instead of re-triggering a folder-wide
        # thumbnail burst. Tasks go out at the lowest (idle) priority on the
        # shared load manager, so any navigation/decode work always preempts
        # them in queue order.
        pv = getattr(self, "parent_viewer", None)
        view_mode = getattr(pv, "view_mode", "single") if pv else "single"
        if view_mode not in ("gallery", "single"):
            return
        if len(self._active_tasks) > 0:
            return
        if view_mode == "single":
            # Extra single-view guard: yield to in-flight navigation work.
            # Idle priority already queues behind it, but a submitted
            # thumbnail read cannot be aborted mid-I/O, and fresh RAW
            # decodes are I/O-bound on the same volume -- so don't even
            # submit while the manager has a meaningful backlog; retry
            # after the normal idle interval instead.
            try:
                if pv.image_manager.get_stats().get("queue_size", 0) > 2:
                    self._idle_preload_timer.start(_gallery_idle_preload_ms())
                    return
            except Exception:
                pass

        # Bidirectional embedded-JPEG prefetch from current image / viewport center.
        start_index = 0
        center_idx = self._gallery_prefetch_center_index(scroll_y=0, viewport_h=400)
        try:
            p = self._scroll_area
            if p is None:
                p = self.parent()
                while p and not isinstance(p, QScrollArea):
                    p = p.parent()
            scroll_y = 0
            viewport_h = 400
            if isinstance(p, QScrollArea):
                scroll_y = p.verticalScrollBar().value()
                viewport_h = max(1, p.viewport().height())
            center_idx = self._gallery_prefetch_center_index(scroll_y, viewport_h)
            if center_idx is not None:
                start_index = center_idx
            elif isinstance(p, QScrollArea):
                for idx, item in enumerate(self._gallery_layout_items):
                    if item["rect"].bottom() > scroll_y:
                        start_index = idx
                        break
        except Exception:
            if center_idx is not None:
                start_index = center_idx

        # Search outwards from start_index: alternately checking next and previous indices
        n_items = len(self._gallery_layout_items)
        outwards_indices = []
        for offset in range(1, n_items):
            nxt = start_index + offset
            prev = start_index - offset
            if nxt < n_items:
                outwards_indices.append(nxt)
            if prev >= 0:
                outwards_indices.append(prev)

        from image_cache import get_image_cache
        global_cache = get_image_cache()

        preload_batch = []
        max_preload_batch = _gallery_idle_preload_batch()
        if len(self.images or []) > 500:
            max_preload_batch = min(max_preload_batch, 24)

        for idx in outwards_indices:
            item = self._gallery_layout_items[idx]
            path = item.get("file_path")
            if not path:
                continue

            # Skip if already in memory thumbnail cache
            if self._thumbnail_cache.get((path, self._thumb_base_key)) is not None:
                continue

            # Skip if already warm in memory (never touch disk on this UI timer).
            if global_cache.has_warm_thumbnail_in_memory(path):
                continue

            # Skip if already being processed or active
            if path in self._active_tasks:
                continue

            preload_batch.append(path)
            if len(preload_batch) >= max_preload_batch:
                break

        if preload_batch:
            idle_priority = _gallery_idle_load_priority()
            # Embedded JPEG only — fast bidirectional scroll cushion above/below center.
            for path in preload_batch:
                self.load_manager.load_image(
                    path,
                    priority=idle_priority,
                    cancel_existing=False,
                    stages={"thumbnail"},
                    gallery_thumbnail=True,
                )
                self._mark_active_task(path, idle_priority)

            # Schedule next batch after a short delay (progressive background loading)
            self._idle_preload_timer.start(_gallery_idle_preload_ms())

    def warm_thumbnails_from_global_cache(self, paths: list[str]) -> int:
        """Seed gallery pixmap cache from global ImageCache (e.g. after film strip)."""
        if not paths:
            return 0
        from image_cache import get_image_cache

        global_cache = get_image_cache()
        warmed = 0
        image_set = set(self.images) if self.images else set()
        for path in paths:
            if not path or path not in image_set:
                continue
            if self._thumbnail_cache.get((path, self._thumb_base_key)):
                continue
            thumb = global_cache.get_thumbnail_memory_only(path)
            if thumb is None:
                thumb = global_cache.get_grid_memory_only(path)
            if thumb is None:
                continue
            thumb = self._orient_gallery_thumbnail_array(path, thumb)
            thumb = self._apply_edits_to_tile(path, thumb)
            pixmap = _thumbnail_data_to_base_pixmap(thumb)
            if pixmap is None or pixmap.isNull():
                continue
            pixmap = self._store_oriented_base_pixmap(path, pixmap)
            self._reconcile_tile_aspect(path, pixmap)
            warmed += 1
        if warmed and _focus_gallery_switch_logs():
            logger.debug(
                "[GALLERY] Warmed %d tile(s) from global thumbnail cache", warmed
            )
        return warmed

    def on_thumbnail_ready(self, file_path, thumbnail_data):
        if self._gallery_folder_superseded():
            return
        # Mark path as no longer in-flight regardless of whether we can render it now.
        resolved = self._resolve_gallery_path(file_path)
        if resolved:
            file_path = resolved
        if file_path in self._active_tasks:
            del self._active_tasks[file_path]
        # Single-image view uses RAWImageViewer.on_manager_thumbnail_ready; skip gallery
        # work to avoid duplicate handling and any re-entrant UI paths on the same signal.
        if self.parent_viewer is not None and getattr(
            self.parent_viewer, "view_mode", "gallery"
        ) != "gallery":
            return
        if file_path not in self._path_to_indices:
            return
        if self._building or self._gallery_warmup_active() or self._thumb_ready_depth > 0:
            # Coalesce into the drain timer (one event) instead of one
            # QTimer.singleShot(0) per completed tile.
            self._pending_thumb_ready[file_path] = thumbnail_data
            if not self._tile_apply_timer.isActive():
                self._tile_apply_timer.start(0)
            return
        self._apply_thumbnail_ready(file_path, thumbnail_data)

    def _apply_thumbnail_ready(self, file_path, thumbnail_data):
        if self._thumb_ready_depth > 0:
            self._pending_thumb_ready[file_path] = thumbnail_data
            if not self._tile_apply_timer.isActive():
                self._tile_apply_timer.start(0)
            return
        self._thumb_ready_depth += 1
        try:
            self._apply_thumbnail_ready_impl(file_path, thumbnail_data)
        finally:
            self._thumb_ready_depth = max(0, self._thumb_ready_depth - 1)

    def _apply_thumbnail_ready_impl(self, file_path, thumbnail_data):
        resolved = self._resolve_gallery_path(file_path)
        if not resolved or resolved not in self._path_to_indices:
            return
        file_path = resolved

        thumbnail_data = self._apply_edits_to_tile(file_path, thumbnail_data)
        pixmap = _thumbnail_data_to_base_pixmap(thumbnail_data)

        if not pixmap or pixmap.isNull():
            self.on_thumbnail_error(file_path, "Null pixmap in on_thumbnail_ready")
            return

        pixmap = self._orient_gallery_base_pixmap(file_path, pixmap)

        existing = self._thumbnail_cache.get((file_path, self._thumb_base_key))
        if existing and not existing.isNull():
            old_dim = max(existing.width(), existing.height())
            new_dim = max(pixmap.width(), pixmap.height())
            if old_dim > new_dim:
                pixmap = existing

        meta_ar = self._metadata_display_aspect(file_path)
        if meta_ar is not None and pixmap.height() > 0:
            px_ar = pixmap.width() / pixmap.height()
            if abs(px_ar - meta_ar) / max(meta_ar, 0.01) > 0.35:
                existing = self._thumbnail_cache.get((file_path, self._thumb_base_key))
                if existing and not existing.isNull() and existing.height() > 0:
                    ex_ar = existing.width() / existing.height()
                    if abs(ex_ar - meta_ar) < abs(px_ar - meta_ar):
                        pixmap = existing
        pixmap = self._store_oriented_base_pixmap(file_path, pixmap)
        self._thumb_fail_counts.pop(file_path, None)
        self._base_fetch_miss_ts.pop(file_path, None)
        if not self._first_thumb_ready_after_set:
            self._first_thumb_ready_after_set = True
            self._end_gallery_load_warmup_throttle()

        time.time()
        if self._thumb_first_after_scroll_t is not None:
            self._thumb_first_after_scroll_t = None
        if self._thumb_first_after_settle_t is not None and self._last_scroll_settle_t > 0:
            self._thumb_first_after_settle_t = None

        # Record tile geometry from the oriented thumbnail pixels — ground truth, like the
        # filmstrip. This MUST run before the fast-scroll early-return: with fast byte-scan
        # decode many thumbnails complete mid-fast-scroll, and skipping it leaves a portrait
        # image in a landscape frame (the initial metadata/default aspect) after the tile is
        # later filled on settle. Recording the aspect + arming the (debounced) rebuild timer
        # is cheap; _handle_metadata_rebuild defers itself while fast-scrolling, so the actual
        # relayout still waits for scroll to settle.
        self._reconcile_tile_aspect(file_path, pixmap)

        frame_mismatch = self._any_visible_tile_frame_mismatch(file_path, pixmap)

        # Fast scroll used to drop completed thumbnails entirely (fill on settle).
        # That was masked while GIL stalls throttled the scroll rate; with the
        # stalls fixed, sustained scrolling keeps _is_scrolling_fast true the
        # whole time and NOTHING painted until keys/momentum stopped — the
        # gallery read as frozen (2026-07-20, macOS + Windows). Applies are
        # batched and budget-bounded in _drain_tile_applies, so paint through
        # fast mode instead of skipping.
        if frame_mismatch:
            if _orient_debug_enabled():
                logger.info(
                    "[ORIENT] defer paint file=%s until layout rebuild (frame mismatch)",
                    os.path.basename(file_path),
                )
            if not self._metadata_rebuild_timer.isActive():
                self._metadata_rebuild_timer.start(0 if not self._is_scrolling_fast else 250)
            return

        # Apply the pixmap to visible tiles. Under batching, defer the (scale/crop +
        # setPixmap) work to the drain timer so a burst of completions can't stall a
        # scroll frame; the base pixmap is already cached above, so drain re-reads it.
        if _batch_tile_apply_enabled():
            self._pending_tile_applies[file_path] = True
            if not self._tile_apply_timer.isActive():
                self._tile_apply_timer.start(0)
        else:
            self._apply_tile_pixmap_now(file_path)

        # Continue scheduling when new thumbnails arrive so visible blanks are filled quickly.
        if not self._is_scrolling_fast:
            self._request_load_visible_images(25)

    def _apply_tile_pixmap_now(self, file_path):
        """Scale/crop the cached base pixmap onto every visible tile showing file_path."""
        base = self._thumbnail_cache.get((file_path, self._thumb_base_key))
        if not base or base.isNull():
            return
        indices = self._path_to_indices.get(file_path, [])
        if not indices:
            return
        dpr = self.devicePixelRatio()
        for idx in indices:
            if idx in self._visible_widgets:
                w = self._visible_widgets[idx]
                if w.file_path != file_path:
                    continue
                logical_size = w.size()
                physical_size = QSize(int(logical_size.width() * dpr), int(logical_size.height() * dpr))
                if not self._frame_matches_pixmap_orientation(
                    self._gallery_layout_items[idx].get("rect")
                    if idx < len(self._gallery_layout_items)
                    else None,
                    base,
                ):
                    if _orient_debug_enabled():
                        logger.info(
                            "[ORIENT] skip paint file=%s idx=%d (frame mismatch, awaiting rebuild)",
                            os.path.basename(file_path),
                            idx,
                        )
                    continue
                fitted = self._fit_tile_pixmap(file_path, base, physical_size, dpr)
                if _orient_debug_enabled():
                    try:
                        rect = None
                        if idx < len(self._gallery_layout_items):
                            rect = self._gallery_layout_items[idx].get("rect")
                        b_o = "P" if base.height() > base.width() else "L"
                        r_o = "?" if rect is None else ("P" if rect.height() > rect.width() else "L")
                        f_o = "P" if fitted.height() > fitted.width() else "L"
                        flag = " FRAME_MISMATCH" if (rect is not None and r_o != b_o) else ""
                        logger.info(
                            "[ORIENT] paint file=%s base=%dx%d(%s) rect=%s(%s) fitted=%dx%d(%s)%s",
                            os.path.basename(file_path), base.width(), base.height(), b_o,
                            "None" if rect is None else f"{rect.width()}x{rect.height()}", r_o,
                            fitted.width(), fitted.height(), f_o, flag,
                        )
                    except Exception:
                        pass
                w.setPixmap(fitted)
                w.setText("")

    def _drain_tile_applies(self):
        """Apply a bounded batch of pending thumb-ready + tile pixmaps per tick."""
        if self._gallery_folder_superseded():
            self._pending_tile_applies.clear()
            self._pending_thumb_ready.clear()
            return
        batch = _tile_apply_batch_size()
        # Smaller bites while fast-scrolling so paints can't eat scroll frames,
        # but never zero — skipping entirely left the gallery blank for the
        # whole gesture once GIL stalls stopped throttling the scroll rate.
        if self._is_scrolling_fast:
            batch = max(2, batch // 4)
        count = 0
        # First: coalesce deferred thumbnail_ready payloads (was one singleShot each).
        while self._pending_thumb_ready and count < batch:
            file_path = next(iter(self._pending_thumb_ready))
            thumbnail_data = self._pending_thumb_ready.pop(file_path)
            self._apply_thumbnail_ready(file_path, thumbnail_data)
            count += 1
        # Then: paint cached base pixmaps onto visible tiles.
        while self._pending_tile_applies and count < batch:
            file_path = next(iter(self._pending_tile_applies))
            del self._pending_tile_applies[file_path]
            self._apply_tile_pixmap_now(file_path)
            count += 1
        if self._pending_thumb_ready or self._pending_tile_applies:
            self._tile_apply_timer.start(0)

    def on_task_completed(self, file_path):
        """Always clear active tasks when the background worker finishes."""
        resolved = self._resolve_gallery_path(file_path)
        if resolved:
            file_path = resolved
        if file_path in self._active_tasks:
            del self._active_tasks[file_path]
        if not self._building and not self._is_scrolling_fast:
            self._request_load_visible_images(40)

    def _on_retry_timer_timeout(self):
        self._max_retry_attempt = 1
        self._request_load_visible_images(40)

    def retry_placeholder_thumbnails(self, paths=None) -> int:
        """Let failed tiles try again instead of staying grey all session.

        A thumbnail that failed its retries got a grey square written into the
        normal cache, which then satisfied every later lookup -- so a
        transient failure was indistinguishable from a permanent one and the
        tile never recovered without reopening the folder.

        Clears the placeholder, the fail count and the request record for the
        given paths (default: all placeholders), so the next visibility pass
        requests them for real. Returns how many were reset.
        """
        targets = set(paths) if paths is not None else set(self._placeholder_thumb_paths)
        targets &= self._placeholder_thumb_paths
        if not targets:
            return 0
        cache = getattr(self, "_thumbnail_cache", None)
        for path in targets:
            self._placeholder_thumb_paths.discard(path)
            self._thumb_fail_counts.pop(path, None)
            self._requested_thumbnail_paths.discard(path)
            if cache is not None:
                try:
                    with cache.lock:
                        for key in [
                            k for k in list(cache.cache.keys())
                            if (k[0] if isinstance(k, tuple) and k else k) == path
                        ]:
                            cache.cache.pop(key, None)
                except Exception:
                    pass
        return len(targets)

    def on_thumbnail_error(self, file_path, error_msg):
        resolved = self._resolve_gallery_path(file_path)
        if resolved:
            file_path = resolved
        if file_path in self._active_tasks:
            del self._active_tasks[file_path]

        if file_path not in self.images:
            return

        lower_err = (error_msg or "").lower()
        clearly_unsupported = any(
            term in lower_err
            for term in (
                "unsupported file format",
                "not recognized",
                "unsupported or corrupt",
            )
        )
        attempt = self._thumb_fail_counts.get(file_path, 0) + 1
        # Cap the fail map so a long session browsing many folders cannot
        # grow without bound (folder switch clears it; this is a backstop).
        if len(self._thumb_fail_counts) >= 4096 and file_path not in self._thumb_fail_counts:
            # Drop an arbitrary oldest-ish entry (dict insertion order).
            self._thumb_fail_counts.pop(next(iter(self._thumb_fail_counts)))
        self._thumb_fail_counts[file_path] = attempt
        max_retries = 3

        is_emfile = "too many open files" in lower_err or "emfile" in lower_err
        if is_emfile:
            pv = self.parent_viewer
            if pv is not None and hasattr(pv, "_enter_io_pressure_mode"):
                pv._enter_io_pressure_mode("gallery_thumb_emfile")
            self._max_retry_attempt = max(getattr(self, "_max_retry_attempt", 1), 6)
        else:
            self._max_retry_attempt = max(getattr(self, "_max_retry_attempt", 1), attempt)

        # JPEG/PNG/WebP: keep in gallery on thumbnail-only failures; retry transient decode errors.
        if not is_raw_file(file_path):
            if attempt <= max_retries:
                logger.debug(
                    "[GALLERY] Thumbnail failed for %s (attempt %d/%d): %s",
                    os.path.basename(file_path),
                    attempt,
                    max_retries,
                    error_msg,
                )
                if not self._retry_timer.isActive():
                    self._retry_timer.start(min(500 * self._max_retry_attempt, 3000))
                return

            null_pixmap = QPixmap(100, 100)
            null_pixmap.fill(Qt.GlobalColor.darkGray)
            self._thumbnail_cache.put((file_path, self._thumb_base_key), null_pixmap)
            # Remember that this grey square is a PLACEHOLDER, not a
            # thumbnail. Cached under the normal key it is indistinguishable
            # from a real one, so every later lookup hit it and the tile
            # stayed grey for the rest of the session -- even though the
            # failure may have been transient (an EMFILE burst, IO pressure,
            # a volume that briefly went away). retry_placeholder_thumbnails()
            # clears these so a later pass can decode for real.
            self._placeholder_thumb_paths.add(file_path)
            logger.warning(
                "[GALLERY] Keeping %s after repeated thumbnail failures: %s",
                os.path.basename(file_path),
                error_msg,
            )
            return

        if attempt < max_retries and not clearly_unsupported:
            logger.debug(
                "[GALLERY] RAW thumbnail failed for %s (attempt %d/%d): %s",
                os.path.basename(file_path),
                attempt,
                max_retries,
                error_msg,
            )
            if not self._retry_timer.isActive():
                self._retry_timer.start(min(500 * self._max_retry_attempt, 3000))
            return

        if not clearly_unsupported and attempt < max_retries + 1:
            null_pixmap = QPixmap(100, 100)
            null_pixmap.fill(Qt.GlobalColor.darkGray)
            self._thumbnail_cache.put((file_path, self._thumb_base_key), null_pixmap)
            # Remember that this grey square is a PLACEHOLDER, not a
            # thumbnail. Cached under the normal key it is indistinguishable
            # from a real one, so every later lookup hit it and the tile
            # stayed grey for the rest of the session -- even though the
            # failure may have been transient (an EMFILE burst, IO pressure,
            # a volume that briefly went away). retry_placeholder_thumbnails()
            # clears these so a later pass can decode for real.
            self._placeholder_thumb_paths.add(file_path)
            logger.warning(
                "[GALLERY] Keeping RAW %s after thumbnail failures: %s",
                os.path.basename(file_path),
                error_msg,
            )
            return

        # Only drop RAW files that are clearly unsupported after retries are exhausted.
        removed = False
        if file_path in self.images:
            self.images.remove(file_path)
            removed = True

        pv = self.parent_viewer
        if pv is not None:
            if hasattr(pv, "image_files") and file_path in pv.image_files:
                try:
                    pv.image_files.remove(file_path)
                    removed = True
                except ValueError:
                    pass
            if removed and hasattr(pv, "_update_gallery_counter"):
                with contextlib.suppress(Exception):
                    pv._update_gallery_counter()

        if removed:
            logger.info(
                "[GALLERY] Removed unsupported RAW from list: %s (%s)",
                file_path,
                error_msg,
            )
            self.build_gallery(force=True)

        if self.parent_viewer is not None and getattr(
            self.parent_viewer, "view_mode", "gallery"
        ) == "gallery" and not self._retry_timer.isActive():
            self._retry_timer.start(100)

    def on_exif_ready(self, file_path, exif_data):
        """Update aspect ratio in layout when metadata arrives."""
        resolved = self._resolve_gallery_path(file_path)
        if not resolved:
            return

        if not exif_data:
            # Save dummy data so we don't infinitely request it
            self._metadata_cache[resolved] = {"original_width": 0, "original_height": 0}
            return

        # Store in local metadata cache (keep container orientation + sensor dimensions).
        self._metadata_cache[resolved] = exif_data
        try:
            from image_cache import get_image_cache

            cache = get_image_cache()
            # Workers put_exif to the memory cache before emitting exif_ready;
            # re-putting here made the UI thread take the exif disk-write lock
            # (caught mid-stall in trackpad scroll profiling). Persist only
            # records the memory cache doesn't already hold.
            if cache.get_exif_memory_only(file_path) is not exif_data:
                cache.put_exif(file_path, exif_data)
        except Exception:
            pass

        # Calculate real aspect ratio
        w = exif_data.get("original_width")
        h = exif_data.get("original_height")
        if not w or not h or h <= 0:
            return

        aspect = self._metadata_display_aspect(file_path)
        if aspect is None:
            return

        raw_meta = self._raw_aspect_undo_user_rotation(file_path, aspect)
        stored_raw = self._measured_raw_aspects.get(resolved)
        if stored_raw is None or (
            self._aspect_orientation_flag(stored_raw) != 0
            and self._aspect_orientation_flag(raw_meta) != 0
            and self._aspect_orientation_flag(stored_raw)
            != self._aspect_orientation_flag(raw_meta)
        ):
            self._measured_raw_aspects[resolved] = raw_meta
            base = self._thumbnail_cache.get((resolved, self._thumb_base_key))
            if base and not base.isNull():
                self._store_oriented_base_pixmap(resolved, base)

        # Check if we need to update layout (if it differs from default 1.333 or previous cache)
        # Find all occurrences of this path in layout
        indices = self._path_to_indices.get(file_path, [])
        changed = False
        geometry_fix = False
        for idx in indices:
            if idx < len(self._gallery_layout_items):
                old_aspect = self._gallery_layout_items[idx].get("aspect", 1.5)
                if abs(old_aspect - aspect) > 0.05:
                    if self._layout_aspect_needs_rebuild(old_aspect, aspect):
                        geometry_fix = True
                    self._gallery_layout_items[idx]["aspect"] = aspect
                    changed = True

        if changed:
            self.refresh_visible_tile_for_path(file_path)
            self._add_metadata_changed_path(file_path)
            if self._gallery_warmup_active() or self._building:
                return

            large = len(self.images) > 800
            rebuild_threshold = 12 if large else (5 if len(self.images) < 100 else 15)

            if geometry_fix and not self._metadata_rebuild_timer.isActive():
                debounce = 2500 if large else 400
                logger.debug(
                    "[GALLERY_DEBUG] Timer started by on_exif_ready (geometry) for %s",
                    file_path,
                )
                self._metadata_rebuild_timer.start(debounce)
            elif not self._metadata_rebuild_timer.isActive():
                if large:
                    debounce = 4000
                elif len(self._metadata_cache) < (len(self.images) * 0.5):
                    debounce = 2000
                else:
                    debounce = 800
                logger.debug(f"[GALLERY_DEBUG] Timer started by on_exif_ready (long) for {file_path}")
                self._metadata_rebuild_timer.start(debounce)
            elif len(self._metadata_changed_paths) >= rebuild_threshold:
                logger.debug(f"[GALLERY_DEBUG] Timer started by on_exif_ready (short) for {file_path}")
                self._metadata_rebuild_timer.start(1500 if large else 500)

    def _schedule_aspects_settle_rebuild(self, delay_ms: int = 3500) -> None:
        """One deferred full rebuild so justified rows match final EXIF/thumbnail aspects."""
        if len(self.images) < 50:
            return
        if not self._metadata_changed_paths:
            return
        n = len(self.images)
        if n >= 1000:
            delay_ms = max(delay_ms, 12000)
        elif n >= 500:
            delay_ms = max(delay_ms, 8000)
        elif n >= 200:
            delay_ms = max(delay_ms, 5000)
        if self._gallery_warmup_active():
            warmup_left_ms = int(
                max(0.0, float(self._gallery_warmup_until) - time.time()) * 1000
            )
            delay_ms = max(delay_ms, warmup_left_ms + 2000)
        if self._aspects_settle_timer.isActive():
            return
        self._aspects_settle_timer.start(max(500, int(delay_ms)))

    def _on_aspects_settle_rebuild(self) -> None:
        if not self.images or self._building:
            self._schedule_aspects_settle_rebuild(2000)
            return
        if self._is_scrolling_fast:
            self._schedule_aspects_settle_rebuild(1500)
            return
        if self._gallery_warmup_active():
            self._schedule_aspects_settle_rebuild(2000)
            return
        if not self._metadata_changed_paths:
            return
        if _focus_gallery_switch_logs():
            logger.debug(
                "[GALLERY] aspects settle rebuild (%d aspect drift)",
                len(self._metadata_changed_paths),
            )
        self._metadata_changed_paths.clear()
        self.build_gallery(force=True)

    def _image_index_for_path(self, file_path: str) -> int:
        for i, item in enumerate(self.images):
            if item == file_path:
                return i
        return -1

    def _layout_row_start_index(self, layout_idx: int) -> int:
        """First layout-item index in the same justified row as *layout_idx*."""
        if layout_idx <= 0 or layout_idx >= len(self._gallery_layout_items):
            return max(0, layout_idx)
        row_top = self._gallery_layout_items[layout_idx]["rect"].top()
        start = layout_idx
        for i in range(layout_idx - 1, -1, -1):
            if self._gallery_layout_items[i]["rect"].top() < row_top:
                break
            start = i
        return start

    def _partial_rebuild_start_image_index(self, changed_paths: set[str]) -> int | None:
        """Earliest image index whose justified row must be recomputed."""
        if not changed_paths:
            return None
        min_image_idx = len(self.images)
        for path in changed_paths:
            idx = self._image_index_for_path(path)
            if idx < 0:
                continue
            layout_indices = self._path_to_indices.get(path) or []
            if not layout_indices:
                min_image_idx = min(min_image_idx, idx)
                continue
            row_start = self._layout_row_start_index(min(layout_indices))
            row_path = self._gallery_layout_items[row_start].get("file_path")
            if isinstance(row_path, str):
                row_idx = self._image_index_for_path(row_path)
                if row_idx >= 0:
                    min_image_idx = min(min_image_idx, row_idx)
        if min_image_idx >= len(self.images):
            return None
        return min_image_idx
