"""Unified mixin part 6 of 7 for UnifiedImageProcessor."""
from __future__ import annotations

import os
from typing import Any

import numpy as np
from PyQt6.QtGui import QPixmap

from .helpers import (
    _verbose_orientation_logs,
    image_covers_sensor_resolution,
    memory_preview_max_edge,
    use_full_embedded_raw_preview,
)

_preview_executor = None


class UnifiedPart6:
    def process_full_image(self, file_path: str,
                          use_full_resolution: bool = False,
                          executor: Any | None = None,
                          apply_sidecar_adjustments: bool | None = None) -> np.ndarray | QPixmap | None:
        """處理完整圖像（統一接口）"""
        if apply_sidecar_adjustments is None:
            from raw_adjustments import sidecar_adjustments_enabled

            apply_sidecar_adjustments = sidecar_adjustments_enabled()
        # 檢查快取
        # CRITICAL: For RAW files, only check full_image cache, not pixmap cache
        # This ensures RAW files always go through proper processing with orientation correction
        is_raw = self._is_raw_file(file_path)

        if use_full_resolution:
            cached_image = self.cache.get_full_image(file_path)
            if cached_image is not None:
                # Orientation verification for cache hit -- use cached EXIF;
                # re-extracting on every worker hit was adding avoidable delay.
                exif_data = self.cache.get_exif(file_path, verify=False)
                if exif_data is None:
                    exif_data = self.exif_extractor.extract_exif_data(file_path)
                orientation = exif_data.get('orientation', 1) if exif_data else 1

                if hasattr(cached_image, 'shape'):
                    h, w = cached_image.shape[:2]

                    # Check for orientation mismatch (Portrait metadata but Landscape cached image)
                    if orientation in (6, 8) and w > h:
                        if _verbose_orientation_logs():
                            # print(f"[ORIENTATION] UnifiedImageProcessor: Cached full_image for {os.path.basename(file_path)} is UNROTATED (stale). Re-processing.")
                            pass
                        cached_image = None

                    # RAW files: a prior half-size decode must never satisfy a
                    # full-resolution request. Half tiles go to preview/grid
                    # only (see _process_raw_image); if a stale half buffer
                    # somehow remains under this key, reject it here.
                    if cached_image is not None and is_raw and not image_covers_sensor_resolution(w, h, exif_data):
                        if _verbose_orientation_logs():
                            pass
                        cached_image = None

                    # An embedded-JPEG stand-in (see _try_full_embedded_raw_
                    # preview) can reach/exceed sensor resolution in pixel
                    # count while still not being the app's own RAW-decoded
                    # pixels -- by dimensions alone it is indistinguishable
                    # from a true decode, which previously made this cache
                    # hit permanently short-circuit every subsequent
                    # use_full_resolution=True request (idle-decode-after-nav
                    # -pause, zoom-to-actual, ...) with the JPEG forever,
                    # never reaching the real fast_raw_decode/rawpy pipeline
                    # below. Reject it here exactly once so the request falls
                    # through to _process_raw_image, which (via the same
                    # full_image_is_embedded_jpeg flag) knows this is a
                    # repeat ask and skips its own JPEG shortcut to produce
                    # the true decode instead.
                    if cached_image is not None and is_raw and self.cache.full_image_is_embedded_jpeg(file_path):
                            cached_image = None

                    if cached_image is not None:
                        if _verbose_orientation_logs():
                            # print(f"[ORIENTATION] Using VALID cached full_image for {os.path.basename(file_path)}. Shape: {w}x{h}")
                            pass
                        if apply_sidecar_adjustments:
                            return self._apply_sidecar_if_needed(file_path, cached_image)
                        return cached_image
                else:
                    return cached_image


        # Only check pixmap cache for non-RAW files
        if not is_raw:
            cached_pixmap = self.cache.get_pixmap(file_path)
            if cached_pixmap is not None and (not use_full_resolution or self._pixmap_covers_file_resolution(
                file_path, cached_pixmap
            )):
                return cached_pixmap
        else:
            # For RAW files, don't use cached pixmap - always process fresh
            if _verbose_orientation_logs():
                # print(f"[ORIENTATION] RAW file {os.path.basename(file_path)} - skipping pixmap cache, will process as RAW")
                pass

        # 處理圖像
        if is_raw:
            base = self._process_raw_image(file_path, use_full_resolution, executor=executor)
            if apply_sidecar_adjustments:
                return self._apply_sidecar_if_needed(file_path, base)
            return base
        else:
            return self._process_regular_image(
                file_path, use_full_resolution=use_full_resolution
            )

    def _pixmap_covers_file_resolution(self, file_path: str, pixmap) -> bool:
        """True when ``pixmap`` is at (or near) the file's native resolution.

        Native size comes from a header-only QImageReader probe (~ms, no pixel
        decode). Dimensions are compared sorted so an EXIF-rotated pixmap
        (swapped w/h) still matches. Unknown native size accepts the cache
        (never force a redecode on a file we cannot even probe), and a pixmap
        already at the loader's own safety cap (_regular_image_max_edge) is
        accepted too -- it is the largest this app will ever produce for the
        file, so rejecting it would just re-run an expensive decode into the
        same result.
        """
        try:
            from common_image_loader import pixmap_covers_requested_edge

            return pixmap_covers_requested_edge(file_path, pixmap, 0)
        except Exception:
            return True

    def _try_full_embedded_raw_preview(
        self, file_path: str, exif_data: dict[str, Any] | None
    ) -> np.ndarray | None:
        """
        Return oriented embedded JPEG when it covers sensor resolution; cache as preview + full.
        """
        if not use_full_embedded_raw_preview():
            return None

        for cached in (
            self.cache.get_full_image(file_path),
            self.cache.get_preview(file_path),
        ):
            if cached is not None:
                h, w = cached.shape[:2]
                if image_covers_sensor_resolution(w, h, exif_data):
                    return cached

        embedded = self.thumbnail_extractor.extract_embedded_native_preview(file_path)
        if embedded is None:
            return None

        h, w = embedded.shape[:2]
        if not image_covers_sensor_resolution(w, h, exif_data):
            return None

        orientation = exif_data.get("orientation", 1) if exif_data else 1
        if orientation != 1:
            embedded = self._apply_orientation_correction(embedded, orientation, exif_data)

        self.cache.put_full_image(file_path, embedded)
        self.cache.mark_full_image_source(file_path, is_embedded_jpeg=True)

        # The downsampled "preview tier" cache entry isn't needed for this
        # call's return value (the caller wants the full-res array right
        # now) -- computing it (a PIL HAMMING resize of the full embedded
        # JPEG, ~200ms for a 60MP source) was previously blocking the
        # return, adding that entire cost to time-to-first-pixel for no
        # benefit. Populate it in the background instead; LRUCache.put is
        # thread-safe, and this only ever produces a lower-priority cache
        # entry nothing is blocked waiting on.
        def _populate_preview_tier_async(arr: np.ndarray = embedded) -> None:
            try:
                preview_tier = self._preview_tier_from_embedded(arr)
                self.cache.put_preview(file_path, preview_tier)
            except Exception:
                pass

        import concurrent.futures

        # Lazy initialization of module-level executor to prevent unbounded thread churn
        global _preview_executor
        if _preview_executor is None:
            _preview_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="preview-tier")

        _preview_executor.submit(_populate_preview_tier_async, embedded)

        import logging

        logging.getLogger(__name__).info(
            "[PREVIEW] Using full-resolution embedded JPEG for %s (%dx%d)",
            os.path.basename(file_path),
            embedded.shape[1],
            embedded.shape[0],
        )
        return embedded

    def _preview_tier_from_embedded(self, embedded: np.ndarray) -> np.ndarray:
        """Store a memory-budget preview tier separately from the sensor-sized buffer."""
        max_edge = memory_preview_max_edge()
        h, w = embedded.shape[:2]
        if max(h, w) <= max_edge:
            return embedded
        scale = max_edge / float(max(h, w))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        try:
            from PIL import Image

            pil = Image.fromarray(embedded)
            pil = pil.resize((new_w, new_h), Image.Resampling.HAMMING)
            return np.asarray(pil)
        except Exception:
            return embedded
