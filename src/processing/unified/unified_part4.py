"""Unified mixin part 4 of 7 for UnifiedImageProcessor."""
from __future__ import annotations

import os
from typing import Any

import numpy as np
from PyQt6.QtCore import QSize

from .helpers import (
    _LIBRAW_UNSUPPORTED_PATHS,
    disk_preview_max_edge,
    is_raw_file,
    memory_preview_max_edge,
    rawpy_imread_warm,
)


class UnifiedPart4:
    def lens_profile_available(self, file_path: str) -> bool:
        """
        Whether a lensfun profile matches this file's camera+lens -- gates
        whether the Adjust panel's lens-correction toggle is shown at all.
        Cheap on repeat calls: EXIF is SQLite-cached in extract_exif_data.
        """
        try:
            exif_data = self.exif_extractor.extract_exif_data(file_path)
            from raw_adjustments.lens_correction import has_lens_profile

            return has_lens_profile(exif_data)
        except Exception:
            return False

    def _is_raw_file(self, file_path: str) -> bool:
        """檢查是否為 RAW 文件"""
        return is_raw_file(file_path)

    def _raw_thumbnail_extract_max_size(
        self,
        target_size: QSize | None,
        *,
        allow_heavy_fallback: bool,
    ) -> int:
        """Gallery grid uses disk tier (~512px); single-view CURRENT uses memory preview tier (~1920px)."""
        if target_size is not None:
            tw = int(target_size.width()) if target_size.width() > 0 else 512
            th = int(target_size.height()) if target_size.height() > 0 else 512
            return max(256, min(1024, tw, th))
        if allow_heavy_fallback:
            return memory_preview_max_edge()
        return disk_preview_max_edge()

    def _single_view_display_thumbnail(self, target_size: QSize | None, allow_heavy_fallback: bool) -> bool:
        return target_size is None and bool(allow_heavy_fallback)

    def _cached_preview_meets_display_tier(self, cached_thumb) -> bool:
        if cached_thumb is None:
            return False
        try:
            min_px = int(memory_preview_max_edge() * 0.85)
            if hasattr(cached_thumb, "shape"):
                return max(int(cached_thumb.shape[0]), int(cached_thumb.shape[1])) >= min_px
            if hasattr(cached_thumb, "width"):
                return max(int(cached_thumb.height()), int(cached_thumb.width())) >= min_px
        except Exception:
            pass
        return False

    def _cache_display_tier_result(
        self,
        file_path: str,
        thumbnail: np.ndarray,
        exif_data: dict[str, Any] | None = None,
    ) -> None:
        """Memory-cache display-tier preview for preview-first warm opens only.

        Single ``put_preview`` copy (no duplicate thumbnail_cache write) so
        thumb+exif / full_pipeline cold paths are not penalized.
        """
        if thumbnail is None:
            return
        try:
            if self._cached_preview_meets_display_tier(
                self.cache.get_preview(file_path)
            ):
                return
            self.cache.put_preview(file_path, thumbnail)
        except Exception:
            pass
        if exif_data:
            try:
                cur = self.cache.get_exif(file_path)
                if cur is not None and not cur.get("minimal_preview_exif"):
                    return
                self.cache.put_exif(
                    file_path,
                    exif_data,
                    persist_disk=not bool(exif_data.get("minimal_preview_exif")),
                )
            except Exception:
                pass

    def cache_index_source_mipmap_tiers(
        self,
        file_path: str,
        thumbnail,
        *,
        exif_data: dict[str, Any] | None = None,
    ) -> None:
        """Backfill preview/grid/thumbnail tiers so gallery/film strip/semantic share one decode.

        Thin wrapper over the shared ``publish_mipmap_tiers`` (common_image_loader.py) —
        kept as a method for call-site compatibility (image_load_manager.py). Unlike the
        old skip-if-any-tier-exists version, missing tiers are now backfilled
        independently (thumbnail-quality-tier PR-1: see docs/thumbnail-cache-unification-plan.md).
        """
        from common_image_loader import publish_mipmap_tiers

        publish_mipmap_tiers(
            file_path,
            thumbnail,
            exif_data=exif_data,
            cache=self.cache,
            source="gallery_or_single",
        )

    def _extract_raw_preview_before_full_exiftool(
        self,
        file_path: str,
        cached_exif: dict[str, Any] | None,
        cached_thumb,
    ) -> tuple[dict[str, Any] | None, np.ndarray | None] | None:
        """Single rawpy open: embedded preview first, defer exiftool to a follow-up task."""
        if cached_exif is not None and self._cached_preview_meets_display_tier(cached_thumb):
            return cached_exif, cached_thumb

        try:
            from enhanced_raw_processor import _rawpy_global_lock
            with _rawpy_global_lock, rawpy_imread_warm(file_path) as raw:
                max_size_val = self._raw_thumbnail_extract_max_size(
                    None, allow_heavy_fallback=True
                )
                thumb = cached_thumb
                if not self._cached_preview_meets_display_tier(thumb):
                    thumb = self.thumbnail_extractor.extract_thumbnail_from_raw(
                        file_path,
                        max_size=max_size_val,
                        allow_scan_fallback=True,
                        raw_object=raw,
                    )
                if thumb is None:
                    return None
                exif = cached_exif
                if exif is None or exif.get("minimal_preview_exif"):
                    exif = self.exif_extractor.build_minimal_raw_exif(file_path, raw)
                if exif and exif.get("orientation", 1) != 1:
                    thumb = self._apply_orientation_correction(
                        thumb, exif["orientation"], exif
                    )
                return exif, thumb
        except Exception:
            return None

    def ensure_display_tier_preview(
        self, file_path: str, thumbnail: np.ndarray | None
    ) -> np.ndarray | None:
        """Upgrade sub-1400px worker thumbs to ~1920 embedded preview when possible."""
        if thumbnail is None or not self._is_raw_file(file_path):
            return thumbnail
        from common_image_loader import dng_prefers_embedded_preview_first
        is_pano_dng = dng_prefers_embedded_preview_first(file_path)
        try:
            min_px = 1024 if is_pano_dng else int(memory_preview_max_edge() * 0.85)
            if hasattr(thumbnail, "shape"):
                md = max(int(thumbnail.shape[0]), int(thumbnail.shape[1]))
            elif hasattr(thumbnail, "width"):
                md = max(int(thumbnail.width()), int(thumbnail.height()))
            else:
                return thumbnail
            if md >= min_px:
                return thumbnail
        except Exception:
            return thumbnail
        import logging

        logger = logging.getLogger(__name__)
        target = memory_preview_max_edge()
        try:
            from enhanced_raw_processor import extract_embedded_jpeg_by_scan

            scanned = extract_embedded_jpeg_by_scan(file_path, target)
            if scanned is not None:
                smd = max(int(scanned.shape[0]), int(scanned.shape[1]))
                if smd >= min_px:
                    # extract_embedded_jpeg_by_scan decodes the TIFF-IFD preview's raw
                    # pixels only -- the segment itself carries no orientation tag of
                    # its own (the container's main EXIF Orientation applies), so it
                    # must be applied here or this preview displays sideways/upside
                    # down whenever Orientation != 1.
                    try:
                        exif_data = self.cache.exif_cache.get(file_path)
                        if exif_data is None:
                            exif_data = self.exif_extractor.extract_exif_data(file_path)
                        orientation = (exif_data or {}).get("orientation", 1)
                        if orientation != 1:
                            scanned = self._apply_orientation_correction(
                                scanned, orientation, exif_data
                            )
                    except Exception:
                        pass
                    logger.info(
                        "[PREVIEW] Scan fallback raised preview to %dpx for %s",
                        smd,
                        os.path.basename(file_path),
                    )
                    return scanned
        except Exception:
            pass
        pair = self._extract_raw_preview_before_full_exiftool(file_path, None, None)
        if pair is not None and pair[1] is not None:
            thumb = pair[1]
            if hasattr(thumb, "shape"):
                smd = max(int(thumb.shape[0]), int(thumb.shape[1]))
                if smd >= min_px:
                    logger.info(
                        "[PREVIEW] Rawpy fallback raised preview to %dpx for %s",
                        smd,
                        os.path.basename(file_path),
                    )
                    return thumb
        half = self.try_half_size_display_preview(file_path)
        if half is not None:
            try:
                smd = max(int(half.shape[0]), int(half.shape[1]))
                if smd >= min_px:
                    logger.info(
                        "[PREVIEW] Half-size fallback raised preview to %dpx for %s",
                        smd,
                        os.path.basename(file_path),
                    )
                    return half
            except Exception:
                pass
        logger.warning(
            "[PREVIEW] Could not reach display tier (%dpx) for %s (stuck at %dpx)",
            min_px,
            os.path.basename(file_path),
            self._preview_buffer_max_dim(thumbnail),
        )
        return thumbnail

    def is_libraw_unsupported(self, file_path: str) -> bool:
        from common_image_loader import dng_prefers_embedded_preview_first
        if dng_prefers_embedded_preview_first(file_path):
            return True
        skip_key = os.path.normcase(os.path.abspath(file_path))
        return skip_key in _LIBRAW_UNSUPPORTED_PATHS

    def _preview_buffer_max_dim(self, buf) -> int:
        try:
            if hasattr(buf, "shape"):
                return max(int(buf.shape[0]), int(buf.shape[1]))
            if hasattr(buf, "width") and hasattr(buf, "height"):
                return max(int(buf.width()), int(buf.height()))
        except Exception:
            pass
        return 0

    def try_half_size_display_preview(self, file_path: str) -> np.ndarray | None:
        """Fast LibRaw half-size decode when embedded JPEG tiers are unusable."""
        if not self._is_raw_file(file_path):
            return None
        import logging

        logger = logging.getLogger(__name__)
        target = memory_preview_max_edge()
        try:
            from enhanced_raw_processor import _rawpy_global_lock
            from processing.raw_decode.fast import (
                decode_half_from_unpacked,
                params_supported_half,
                unpack_raw,
            )

            # Prefer the fast unpack path so a later full decode can reuse
            # the mosaic (same stash as fit-view half in _process_raw_image).
            half_params = {
                "half_size": True,
                "use_camera_wb": True,
                "no_auto_bright": False,
                "output_bps": 8,
                "user_flip": 0,
            }
            rgb = None
            if params_supported_half(half_params):
                unpacked = unpack_raw(file_path, rawpy_lock=_rawpy_global_lock)
                if unpacked is not None:
                    try:
                        rgb = decode_half_from_unpacked(unpacked)
                    except Exception:
                        rgb = None
                    if rgb is not None:
                        self._stash_unpacked_raw(file_path, unpacked)
            if rgb is None:

                with _rawpy_global_lock, rawpy_imread_warm(file_path) as raw:
                    rgb = raw.postprocess(
                        half_size=True,
                        use_camera_wb=True,
                        no_auto_bright=False,
                        output_bps=8,
                    )
            if rgb is None or rgb.size == 0:
                return None
            from PIL import Image

            pil = Image.fromarray(rgb.astype("uint8"), "RGB")
            w, h = pil.size
            scale = min(target / max(w, h, 1), 1.0)
            if scale < 1.0:
                pil = pil.resize(
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    Image.Resampling.BILINEAR,
                )
            return np.array(pil)
        except Exception as exc:
            logger.debug(
                "Half-size display preview failed for %s: %s",
                os.path.basename(file_path),
                exc,
            )
            return None
