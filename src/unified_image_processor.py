"""
統一圖像處理器 - 處理所有圖像類型（RAW、JPEG、PNG等）

這個模組實現了重構提案中的 UnifiedImageProcessor，提供統一的接口
來處理所有圖像類型，內置快取檢查和錯誤處理。
"""

import os
import numpy as np
from typing import Optional, Dict, Any, Union, Tuple
from PyQt6.QtCore import QSize, QLoggingCategory
from PyQt6.QtGui import QPixmap, QImage
# Silence qt.imageformats warnings (e.g. missing TIFF tag warnings on RAW files)
QLoggingCategory.setFilterRules("qt.imageformats*=false")

import logging
logging.getLogger("exifread").setLevel(logging.ERROR)
# PIL Image, rawpy, and exifread will be imported lazily to avoid import delays

from image_cache import (
    disk_preview_max_edge,
    get_image_cache,
    memory_preview_max_edge,
)
from enhanced_raw_processor import (
    ThumbnailExtractor, 
    EXIFExtractor, 
    OptimizedRAWProcessor
)
from common_image_loader import (
    array_matches_exif_display,
    image_covers_sensor_resolution,
    is_raw_file,
    load_pixmap_safe,
    use_full_embedded_raw_preview,
    use_libraw_consistent_preview_first,
)


def _verbose_orientation_logs() -> bool:
    return os.environ.get("RAWVIEWER_VERBOSE_ORIENTATION_LOGS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


# LibRaw cannot open some composite HDR DNGs; avoid repeated preview/decode attempts.
_LIBRAW_UNSUPPORTED_PATHS: set[str] = set()


def decode_raw_file(file_path: str, params: Dict[str, Any]) -> np.ndarray:
    """
    Helper function for process pool to decode RAW files.
    Must be at top level for pickling.
    """
    import rawpy
    from enhanced_raw_processor import _rawpy_global_lock, _heavy_fallback_semaphore
    with _rawpy_global_lock:
        raw_ctx = rawpy.imread(file_path)
    with raw_ctx as raw:
        with _heavy_fallback_semaphore:
            return raw.postprocess(**params)


class UnifiedImageProcessor:
    """統一的圖像處理器，處理所有圖像類型"""
    
    def __init__(self):
        self.cache = get_image_cache()
        self.thumbnail_extractor = ThumbnailExtractor()
        self.exif_extractor = EXIFExtractor()
        self.raw_processor = OptimizedRAWProcessor()
        # Single-slot LibRaw unpack reuse (see fast_raw_decode.UnpackedRaw):
        # the fit-view half decode stashes its unpacked mosaic here so the
        # deferred full decode of the same file skips the second unpack (the
        # 100-900ms dominant cost). One slot only (~2 bytes/sensor-pixel);
        # replaced on the next stash, consumed by the full decode.
        import threading as _threading

        self._unpacked_raw_lock = _threading.Lock()
        self._unpacked_raw_slot: Optional[tuple] = None  # (norm_path, UnpackedRaw)
        # Sidecar-adjusted full-buffer memo: applying XMP adjustments to a
        # sensor-resolution buffer costs ~2.1s at 45MP (measured), and
        # _apply_sidecar_if_needed used to re-run it on EVERY
        # process_full_image() fetch of an edited file (each zoom, revisit,
        # repaint). Two slots (current + previous file) keyed by
        # (norm_path, adjustment fingerprint); the fingerprint makes stale
        # entries self-invalidating when the sidecar changes.
        self._adjusted_full_lock = _threading.Lock()
        self._adjusted_full_slots: list = []  # [(norm_path, adj_key, np.ndarray)]

    def _stash_unpacked_raw(self, file_path: str, unpacked) -> None:
        key = os.path.normcase(os.path.abspath(file_path))
        with self._unpacked_raw_lock:
            self._unpacked_raw_slot = (key, unpacked)

    def _take_unpacked_raw(self, file_path: str):
        key = os.path.normcase(os.path.abspath(file_path))
        with self._unpacked_raw_lock:
            if self._unpacked_raw_slot and self._unpacked_raw_slot[0] == key:
                unpacked = self._unpacked_raw_slot[1]
                self._unpacked_raw_slot = None
                return unpacked
        return None

    def _apply_sidecar_if_needed(
        self, file_path: str, rgb_image: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Apply XMP sidecar adjustments on top of a base (unadjusted) RGB buffer.

        Memoized per (file, adjustment fingerprint, base buffer shape): see
        _adjusted_full_slots in __init__ for why.
        """
        if rgb_image is None:
            return None
        try:
            from raw_adjustments import (
                apply_adjustments_to_rgb,
                is_default_adjustments,
                load_adjustments_for_file,
            )

            adj = load_adjustments_for_file(file_path)
            if is_default_adjustments(adj):
                return rgb_image
            norm = os.path.normcase(os.path.abspath(file_path))
            # Fingerprint includes the base buffer's shape so a half-size
            # cached result can never be served for a full-res request.
            from raw_dodge_burn import MASK_KEY as _db_mask_key

            adj_key = (
                tuple(sorted((k, round(float(v), 4)) for k, v in adj.items()
                             if isinstance(v, (int, float)))),
                str(adj.get("_tone_curve_pv2012", "") or ""),
                str(adj.get(_db_mask_key, "") or ""),
                rgb_image.shape,
            )
            with self._adjusted_full_lock:
                for n, k, buf in self._adjusted_full_slots:
                    if n == norm and k == adj_key:
                        return buf
            import time as _time

            from perf_metrics import perf_mark

            _t0 = _time.perf_counter()
            out = apply_adjustments_to_rgb(rgb_image, adj)
            perf_mark(
                "sidecar_apply",
                (_time.perf_counter() - _t0) * 1000.0,
                file_path,
                mp=rgb_image.shape[0] * rgb_image.shape[1] / 1e6,
            )
            with self._adjusted_full_lock:
                self._adjusted_full_slots = [
                    e for e in self._adjusted_full_slots if e[0] != norm
                ][-1:] + [(norm, adj_key, out)]
            return out
        except Exception:
            return rgb_image

    def decode_raw_edit_base(
        self,
        file_path: str,
        executor: Optional[Any] = None,
        use_full_resolution: bool = False,
        apply_lens_correction: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Demosaic from RAW sensor data for the adjust panel.
        Returns 16-bit scene-linear RGB (never embedded JPEG; not cached for browse).
        Defaults to LibRaw half-size for responsive live editing.

        ``apply_lens_correction``: if True, undistort using a lensfun profile
        matched from this file's EXIF (see raw_lens_correction.py) -- applied
        once here, after orientation correction (so the Modifier sees the
        buffer's final width/height), not as a per-tick pipeline adjustment.
        """
        skip_key = os.path.normcase(os.path.abspath(file_path))
        from common_image_loader import dng_prefers_embedded_preview_first
        from raw_tone_recovery import edit_base_decode_params

        try:
            exif_data = self.exif_extractor.extract_exif_data(file_path)
            half_size = not use_full_resolution
            # Browse-brightness decode (Blend highlights, no exp_shift) with
            # the corrected-WB handling built in; see edit_base_decode_params.
            edit_params = edit_base_decode_params(
                half_size=half_size, demosaic="AHD", for_file=file_path
            )

            rgb_image: Optional[np.ndarray] = None
            if skip_key not in _LIBRAW_UNSUPPORTED_PATHS:
                try:
                    import rawpy
                    from enhanced_raw_processor import (
                        _heavy_fallback_semaphore,
                        _rawpy_global_lock,
                    )
                    from fast_raw_decode import try_fast_raw_decode

                    rgb_image = try_fast_raw_decode(
                        file_path, 
                        edit_params, 
                        rawpy_lock=_rawpy_global_lock,
                        return_linear=True
                    )

                    if rgb_image is None:
                        with _rawpy_global_lock:
                            raw_ctx = rawpy.imread(file_path)
                        with raw_ctx as raw:
                            cam_mul_for_scale = list(raw.camera_whitebalance or [])
                            with _heavy_fallback_semaphore:
                                rgb_image = raw.postprocess(**edit_params)
                    else:
                        cam_mul_for_scale = None
                    # Restore browse (min-normalized) brightness: dcraw
                    # divides the WB multipliers by their MAX whenever
                    # highlight_mode > 0 (measured: a uniform 1.74x darkening
                    # on a cam_mul of [1780,1024,1786]), which made the
                    # editor's default rendering visibly darker than browse.
                    # Rescale to float32 scene-linear where 1.0 = camera clip
                    # point: unclipped tones land exactly on browse scale,
                    # and values > 1.0 are the Blend-reconstructed highlight
                    # headroom that Exposure/Highlights can pull back down
                    # (display clips them to white, matching browse).
                    try:
                        from fast_raw_decode import get_corrected_camera_wb

                        mul = get_corrected_camera_wb(file_path) or cam_mul_for_scale
                        mul3 = [float(m) for m in (mul or [])[:3] if float(m) > 0]
                        dmax_over_dmin = (max(mul3) / min(mul3)) if len(mul3) == 3 else 1.0
                    except Exception:
                        dmax_over_dmin = 1.0
                    if rgb_image is not None and rgb_image.dtype == np.uint16:
                        rgb_image = rgb_image.astype(np.float32) * (
                            dmax_over_dmin / 65535.0
                        )
                except Exception as libraw_err:
                    is_unsupported = (
                        "unsupported" in str(libraw_err).lower()
                        or "not recognized" in str(libraw_err).lower()
                    )
                    if is_unsupported:
                        _LIBRAW_UNSUPPORTED_PATHS.add(skip_key)
                    logging.getLogger(__name__).warning(
                        "[EDIT] LibRaw linear decode failed for %s: %s",
                        os.path.basename(file_path),
                        libraw_err,
                    )
                    rgb_image = None

            if rgb_image is not None:
                if not exif_data or exif_data.get("orientation", 1) == 1:
                    exif_data = self.exif_extractor.extract_exif_data(file_path)
                orientation = exif_data.get("orientation", 1) if exif_data else 1
                if orientation != 1:
                    rgb_image = self._apply_orientation_correction(
                        rgb_image, orientation, exif_data
                    )
                if apply_lens_correction:
                    from raw_lens_correction import apply_lens_correction as _apply_lens_correction

                    rgb_image = _apply_lens_correction(rgb_image, exif_data)
                logging.getLogger(__name__).info(
                    "[EDIT] Linear edit base for %s (%dx%d, uint16=%s, half_size=%s)",
                    os.path.basename(file_path),
                    rgb_image.shape[1],
                    rgb_image.shape[0],
                    rgb_image.dtype == np.uint16,
                    half_size,
                )
                return rgb_image

            if dng_prefers_embedded_preview_first(file_path):
                try:
                    from PIL import Image

                    with Image.open(file_path) as im:
                        best_w = best_h = 0
                        best_idx = -1
                        n_frames = getattr(im, "n_frames", 1)
                        for idx in range(n_frames):
                            im.seek(idx)
                            w, h = im.size
                            if w * h > best_w * best_h:
                                best_w, best_h = w, h
                                best_idx = idx
                        if best_idx >= 0 and best_w >= 1024:
                            im.seek(best_idx)
                            rgb_im = im.convert("RGB")
                            full_pixels = np.array(rgb_im)
                            orientation = exif_data.get("orientation", 1) if exif_data else 1
                            if orientation != 1:
                                full_pixels = self._apply_orientation_correction(
                                    full_pixels, orientation, exif_data
                                )
                            return full_pixels
                except Exception:
                    pass
                logging.getLogger(__name__).warning(
                    "[EDIT] LibRaw unsupported for %s; no RAW demosaic base available",
                    os.path.basename(file_path),
                )
                return None

            if file_path.lower().endswith(".nef"):
                # Nikon HE/HE*-compressed NEF: LibRaw can't demosaic it at all.
                # Use the app's own byte-scan embedded-JPEG extractor (verified
                # 36/36 against LibRaw for NEF) rather than the DNG branch's PIL
                # multi-frame trick above -- for NEF the largest-pixel-count TIFF
                # "frame" is frequently the actual compressed Bayer sensor data,
                # not the embedded preview, which PIL can't decode correctly.
                try:
                    from enhanced_raw_processor import extract_embedded_jpeg_by_scan

                    embedded = extract_embedded_jpeg_by_scan(file_path, 0)
                    if embedded is not None:
                        orientation = exif_data.get("orientation", 1) if exif_data else 1
                        if orientation != 1:
                            embedded = self._apply_orientation_correction(
                                embedded, orientation, exif_data
                            )
                        logging.getLogger(__name__).info(
                            "[EDIT] Embedded-JPEG edit base for HE-compressed NEF %s (%dx%d)",
                            os.path.basename(file_path),
                            embedded.shape[1],
                            embedded.shape[0],
                        )
                        return embedded
                except Exception:
                    pass
                logging.getLogger(__name__).warning(
                    "[EDIT] LibRaw unsupported for %s; no embedded-JPEG edit base available either",
                    os.path.basename(file_path),
                )
                return None

            return None
        except Exception as e:
            logging.getLogger(__name__).error(
                "[EDIT] RAW edit base decode failed for %s: %s",
                os.path.basename(file_path),
                e,
                exc_info=True,
            )
            return None

    def lens_profile_available(self, file_path: str) -> bool:
        """
        Whether a lensfun profile matches this file's camera+lens -- gates
        whether the Adjust panel's lens-correction toggle is shown at all.
        Cheap on repeat calls: EXIF is SQLite-cached in extract_exif_data.
        """
        try:
            exif_data = self.exif_extractor.extract_exif_data(file_path)
            from raw_lens_correction import has_lens_profile

            return has_lens_profile(exif_data)
        except Exception:
            return False

    def _is_raw_file(self, file_path: str) -> bool:
        """檢查是否為 RAW 文件"""
        return is_raw_file(file_path)

    def _raw_thumbnail_extract_max_size(
        self,
        target_size: Optional[QSize],
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

    def _single_view_display_thumbnail(self, target_size: Optional[QSize], allow_heavy_fallback: bool) -> bool:
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
        exif_data: Optional[Dict[str, Any]] = None,
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
        exif_data: Optional[Dict[str, Any]] = None,
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
        cached_exif: Optional[Dict[str, Any]],
        cached_thumb,
    ) -> Optional[Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]]:
        """Single rawpy open: embedded preview first, defer exiftool to a follow-up task."""
        if cached_exif is not None and self._cached_preview_meets_display_tier(cached_thumb):
            return cached_exif, cached_thumb
        import rawpy

        try:
            from enhanced_raw_processor import _rawpy_global_lock
            with _rawpy_global_lock:
                with rawpy.imread(file_path) as raw:
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
        self, file_path: str, thumbnail: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
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

    @staticmethod
    def _preview_buffer_max_dim(buf) -> int:
        try:
            if hasattr(buf, "shape"):
                return max(int(buf.shape[0]), int(buf.shape[1]))
            if hasattr(buf, "width") and hasattr(buf, "height"):
                return max(int(buf.width()), int(buf.height()))
        except Exception:
            pass
        return 0

    def try_half_size_display_preview(self, file_path: str) -> Optional[np.ndarray]:
        """Fast LibRaw half-size decode when embedded JPEG tiers are unusable."""
        if not self._is_raw_file(file_path):
            return None
        import logging

        logger = logging.getLogger(__name__)
        target = memory_preview_max_edge()
        try:
            import rawpy

            from enhanced_raw_processor import _rawpy_global_lock
            with _rawpy_global_lock:
                with rawpy.imread(file_path) as raw:
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
    
    def process_thumbnail(self, file_path: str, allow_heavy_fallback: bool = True,
                          target_size: Optional[QSize] = None) -> Optional[np.ndarray]:
        """處理縮圖（統一接口）"""
        MAX_THUMB_DIM = 1024
        single_view_preview = self._single_view_display_thumbnail(
            target_size, allow_heavy_fallback
        )

        if not single_view_preview:
            grid = self.cache.get_grid(file_path)
            if grid is not None:
                return grid

        # 檢查快取
        cached = self.cache.get_thumbnail(file_path)
        if cached is not None:
            # Hard safety: never return an oversized "thumbnail" (e.g. RAW BITMAP thumb)
            try:
                if hasattr(cached, 'shape'):
                    h0, w0 = cached.shape[:2]
                    if max(h0, w0) > MAX_THUMB_DIM:
                        cached = None
            except Exception:
                cached = None

            if cached is not None and hasattr(cached, "shape"):
                exif_data = self.cache.get_exif(file_path)
                if exif_data is None:
                    exif_data = self.exif_extractor.extract_exif_data(file_path)
                if exif_data and not array_matches_exif_display(
                    cached.shape[1], cached.shape[0], exif_data
                ):
                    import logging

                    logging.getLogger(__name__).debug(
                        "Thumbnail orientation mismatch; re-extracting %s",
                        os.path.basename(file_path),
                    )
                    self.cache.thumbnail_cache.remove(file_path)
                    self.cache.disk_thumbnail_cache.remove(file_path)
                    cached = None

            if cached is not None:
                if single_view_preview and hasattr(cached, "shape"):
                    try:
                        if max(int(cached.shape[0]), int(cached.shape[1])) < int(
                            memory_preview_max_edge() * 0.85
                        ):
                            cached = None
                    except Exception:
                        pass
                elif hasattr(cached, "shape"):
                    try:
                        if max(int(cached.shape[0]), int(cached.shape[1])) < int(
                            disk_preview_max_edge() * 0.85
                        ):
                            cached = None
                    except Exception:
                        pass
                if cached is not None:
                    return cached

        # Gallery grid: disk tier (~512px). Single-view CURRENT: memory preview tier (~1920px).
        is_raw = self._is_raw_file(file_path)
        skip_key = os.path.normcase(os.path.abspath(file_path))
        if is_raw and skip_key in _LIBRAW_UNSUPPORTED_PATHS:
            return None
        if is_raw:
            max_size_val = self._raw_thumbnail_extract_max_size(
                target_size, allow_heavy_fallback=allow_heavy_fallback
            )
            thumbnail = self.thumbnail_extractor.extract_thumbnail_from_raw(
                file_path,
                max_size=max_size_val,
                allow_scan_fallback=allow_heavy_fallback,
            )
        else:
            thumbnail = self.thumbnail_extractor.extract_thumbnail_from_image(
                file_path, 
                max_size=MAX_THUMB_DIM,
                target_size=target_size
            )
        
        if thumbnail is None:
            if allow_heavy_fallback and is_raw:
                # Fallback: If no embedded thumbnail found, or it was rejected as too small,
                # do a fast half-size RAW decode to get a high-quality thumbnail.
                try:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.debug(f"No usable thumbnail for {file_path}, falling back to fast RAW decode")
                    
                    # Use OptimizedRAWProcessor for a fast decode
                    params = self.raw_processor.get_optimized_processing_params(file_path, None)
                    params['half_size'] = True
                    params['user_flip'] = 0 # Handle orientation manually
                    
                    import rawpy
                    from enhanced_raw_processor import _rawpy_global_lock, _heavy_fallback_semaphore
                    with _rawpy_global_lock:
                        raw_ctx = rawpy.imread(file_path)
                    with raw_ctx as raw:
                        with _heavy_fallback_semaphore:
                            thumbnail = raw.postprocess(**params)
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    _LIBRAW_UNSUPPORTED_PATHS.add(skip_key)
                    logger.warning(f"Heavy thumbnail fallback failed for {file_path}: {e}")
                    return None
            else:
                return None

        # Non-RAW reader path can return QImage directly.
        from_qimage_autotransform = False
        if isinstance(thumbnail, QImage):
            from enhanced_raw_processor import _qimage_to_rgb_array

            from_qimage_autotransform = True
            rgb = _qimage_to_rgb_array(thumbnail)
            if rgb is None:
                return None
            thumbnail = rgb

        # For RAW or fallback paths that return np.ndarray:
        # 獲取 EXIF 數據以獲取 orientation
        exif_data = self.cache.get_exif(file_path)
        if exif_data is None or exif_data.get("minimal_preview_exif"):
            if single_view_preview and self._is_raw_file(file_path):
                try:
                    import rawpy

                    from enhanced_raw_processor import _rawpy_global_lock
                    with _rawpy_global_lock:
                        with rawpy.imread(file_path) as raw:
                            exif_data = self.exif_extractor.build_minimal_raw_exif(
                                file_path, raw
                            )
                except Exception:
                    exif_data = None
            else:
                exif_data = None
        if exif_data is None:
            exif_data = self.exif_extractor.extract_exif_data(file_path)
        decoder_baked_exif = from_qimage_autotransform or (
            not is_raw and thumbnail is not None
        )
        if decoder_baked_exif and exif_data:
            from common_image_loader import mark_exif_pixels_display_oriented

            exif_data = mark_exif_pixels_display_oriented(exif_data)
            self.cache.put_exif(file_path, exif_data, persist_disk=False)
        orientation = exif_data.get('orientation', 1) if exif_data else 1
        if orientation != 1 and not decoder_baked_exif:
            thumbnail = self._apply_orientation_correction(thumbnail, orientation, exif_data)
        if exif_data and exif_data.get("minimal_preview_exif"):
            self.cache.put_exif(file_path, exif_data, persist_disk=False)
        
        try:
            from PIL import Image
            import io
            
            if not isinstance(thumbnail, np.ndarray):
                return thumbnail
                
            if thumbnail.dtype != np.uint8:
                thumbnail = thumbnail.astype(np.uint8)
            
            if len(thumbnail.shape) == 2:
                pil_img = Image.fromarray(thumbnail, 'L').convert('RGB')
            elif len(thumbnail.shape) == 3:
                pil_img = Image.fromarray(thumbnail, 'RGB')
            else:
                return thumbnail

            # --- Structured Mipmap (Multi-Scale) Caching Pipeline ---
            w, h = pil_img.size

            pv_max = (
                memory_preview_max_edge()
                if single_view_preview
                else disk_preview_max_edge()
            )
            if w <= pv_max and h <= pv_max:
                preview_pil = pil_img
            else:
                scale_1 = min(pv_max / w, pv_max / h)
                new_w1 = max(1, int(w * scale_1))
                new_h1 = max(1, int(h * scale_1))
                preview_pil = pil_img.resize((new_w1, new_h1), Image.Resampling.HAMMING)
            self.cache.put_preview(file_path, np.array(preview_pil))

            grid_max = disk_preview_max_edge()
            if w <= grid_max and h <= grid_max:
                grid_pil = pil_img
            else:
                scale_2 = min(grid_max / w, grid_max / h)
                new_w2 = max(1, int(w * scale_2))
                new_h2 = max(1, int(h * scale_2))
                grid_pil = pil_img.resize((new_w2, new_h2), Image.Resampling.HAMMING)

            from common_image_loader import encode_tile_bytes

            self.cache.put_grid(file_path, np.array(grid_pil), encode_tile_bytes(grid_pil))

            if w <= 256 and h <= 256:
                thumb_pil = pil_img
            else:
                scale_3 = min(256 / w, 256 / h)
                new_w3 = max(1, int(w * scale_3))
                new_h3 = max(1, int(h * scale_3))
                thumb_pil = pil_img.resize((new_w3, new_h3), Image.Resampling.HAMMING)

            thumb_np = np.array(thumb_pil)
            self.cache.put_thumbnail(file_path, thumb_np, encode_tile_bytes(thumb_pil))

            # Single-view: return screen-preview tier; gallery/preload: return grid tier.
            if single_view_preview:
                return np.array(preview_pil)
            return np.array(grid_pil)
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error processing structured mipmaps with PIL: {e}")
            self.cache.put_thumbnail(file_path, thumbnail)
            return thumbnail

    
    def process_full_image(self, file_path: str, 
                          use_full_resolution: bool = False,
                          executor: Optional[Any] = None,
                          apply_sidecar_adjustments: bool = True) -> Optional[Union[np.ndarray, QPixmap]]:
        """處理完整圖像（統一接口）"""
        # 檢查快取
        # CRITICAL: For RAW files, only check full_image cache, not pixmap cache
        # This ensures RAW files always go through proper processing with orientation correction
        is_raw = self._is_raw_file(file_path)
        
        if use_full_resolution:
            cached_image = self.cache.get_full_image(file_path)
            if cached_image is not None:
                # Orientation verification for cache hit
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

                    # RAW files: full_image_cache is also written by the
                    # half-size (fit-view) decode -- see _process_raw_image's
                    # unconditional put_full_image() call -- so a cache hit
                    # here is not necessarily sensor-resolution. Without this
                    # check, a use_full_resolution=True request that lands
                    # right after a half decode wrote this same key returns
                    # the half-size buffer forever: the image never upgrades
                    # to full resolution on zoom/idle-decode, and looks
                    # "stuck"/blank when a caller expects full-res dimensions
                    # (e.g. gallery -> single view). Only applies to RAW;
                    # non-RAW full_image_cache entries are always full-res.
                    if cached_image is not None and is_raw:
                        if not image_covers_sensor_resolution(w, h, exif_data):
                            if _verbose_orientation_logs():
                                pass
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
            if cached_pixmap is not None:
                if _verbose_orientation_logs():
                    # print(f"[ORIENTATION] Using cached pixmap for {os.path.basename(file_path)} (non-RAW file)")
                    pass
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
    
    def _try_full_embedded_raw_preview(
        self, file_path: str, exif_data: Optional[Dict[str, Any]]
    ) -> Optional[np.ndarray]:
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

        import threading

        threading.Thread(target=_populate_preview_tier_async, daemon=True).start()

        import logging

        logging.getLogger(__name__).info(
            "[PREVIEW] Using full-resolution embedded JPEG for %s (%dx%d)",
            os.path.basename(file_path),
            embedded.shape[1],
            embedded.shape[0],
        )
        return embedded

    @staticmethod
    def _preview_tier_from_embedded(embedded: np.ndarray) -> np.ndarray:
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

    def _process_raw_image(self, file_path: str, 
                          use_full_resolution: bool = False,
                          executor: Optional[Any] = None) -> Optional[np.ndarray]:
        """處理 RAW 圖像"""
        libraw_first = use_libraw_consistent_preview_first()
        skip_key = os.path.normcase(os.path.abspath(file_path))
        from common_image_loader import dng_prefers_embedded_preview_first
        if dng_prefers_embedded_preview_first(file_path):
            _LIBRAW_UNSUPPORTED_PATHS.add(skip_key)
        elif file_path.lower().endswith(".nef"):
            # Proactive Nikon HE/HE* detection: skip straight to the embedded-
            # JPEG fallback ladder below instead of a doomed LibRaw decode.
            cached_exif = self.cache.get_exif(file_path)
            he = cached_exif.get("nef_he_compressed") if cached_exif else None
            if he is None:
                # The EXIF row (with the proactive HE flag from EXIFExtractor)
                # may not be cached yet when the FIRST full-decode task for this
                # file runs -- e.g. a stages={'full'} nav-preload racing the
                # combined load that populates EXIF. Without this, that early
                # task takes the LibRaw path, fails to demosaic HE/HE*, and
                # returns None -> a spurious "unsupported or corrupt" dialog for
                # a file that a slightly later task displays fine from its
                # embedded JPEG. Detect directly (~0.05ms) so every task routes
                # to the embedded ladder regardless of EXIF-cache timing.
                try:
                    from enhanced_raw_processor import _detect_nef_he_compression

                    he = _detect_nef_he_compression(file_path)
                except Exception:
                    he = None
            if he is True:
                _LIBRAW_UNSUPPORTED_PATHS.add(skip_key)
        if skip_key in _LIBRAW_UNSUPPORTED_PATHS:
            cached = self.cache.get_preview(file_path)
            if cached is None:
                cached = self.cache.get_full_image(file_path)
            if cached is not None:
                cached_dim = max(cached.shape[0], cached.shape[1]) if hasattr(cached, "shape") else 0
                if cached_dim >= 1024:
                    return cached
            if use_full_resolution:
                exif_data = self.exif_extractor.extract_exif_data(file_path)
                
                # First try decoding the full image via PIL TIFF reader (useful for linear/composite DNG panoramas)
                try:
                    from PIL import Image
                    with Image.open(file_path) as im:
                        best_w = best_h = 0
                        best_idx = -1
                        n_frames = getattr(im, "n_frames", 1)
                        for idx in range(n_frames):
                            im.seek(idx)
                            w, h = im.size
                            if w * h > best_w * best_h:
                                best_w, best_h = w, h
                                best_idx = idx
                        if best_idx >= 0 and best_w >= 1024:
                            im.seek(best_idx)
                            rgb_im = im.convert("RGB")
                            full_pixels = np.array(rgb_im)
                            orientation = exif_data.get("orientation", 1) if exif_data else 1
                            if orientation != 1:
                                full_pixels = self._apply_orientation_correction(
                                    full_pixels, orientation, exif_data
                                )
                            self.cache.put_full_image(file_path, full_pixels)
                            self.cache.put_preview(file_path, full_pixels)
                            import logging
                            logging.getLogger(__name__).info(
                                "[FULL_RES] Successfully loaded full resolution DNG panorama via PIL TIFF reader: %dx%d",
                                best_w, best_h
                            )
                            return full_pixels
                except Exception as pil_e:
                    import logging
                    logging.getLogger(__name__).warning(
                        "[FULL_RES] PIL TIFF decode fallback failed/unsupported for %s: %s",
                        os.path.basename(file_path), pil_e
                    )

                full_embedded = self._try_full_embedded_raw_preview(file_path, exif_data)
                if full_embedded is not None:
                    return full_embedded
                
                # If that didn't work (due to resolution coverage reject or workflow toggle),
                # extract the largest native preview since LibRaw can't demosaic this file anyway.
                native_preview = self.thumbnail_extractor.extract_embedded_native_preview(file_path)
                if native_preview is not None:
                    orientation = exif_data.get("orientation", 1) if exif_data else 1
                    if orientation != 1:
                        native_preview = self._apply_orientation_correction(
                            native_preview, orientation, exif_data
                        )
                    self.cache.put_preview(file_path, native_preview)
                    self.cache.put_full_image(file_path, native_preview)
                    return native_preview
                
                mem_max = memory_preview_max_edge()
                preview = self.thumbnail_extractor.extract_preview_from_raw(
                    file_path, max_size=mem_max
                )
                if preview is not None:
                    orientation = (
                        exif_data.get("orientation", 1) if exif_data else 1
                    )
                    if orientation != 1:
                        preview = self._apply_orientation_correction(
                            preview, orientation, exif_data
                        )
                    self.cache.put_preview(file_path, preview)
                    return preview
                try:
                    from enhanced_raw_processor import extract_embedded_jpeg_by_scan

                    scanned = extract_embedded_jpeg_by_scan(file_path, mem_max)
                    if scanned is not None:
                        orientation = (
                            exif_data.get("orientation", 1) if exif_data else 1
                        )
                        if orientation != 1:
                            scanned = self._apply_orientation_correction(
                                scanned, orientation, exif_data
                            )
                        self.cache.put_preview(file_path, scanned)
                        return scanned
                except Exception:
                    pass
                return None
            return None
        try:
            # 獲取 EXIF 數據（用於處理參數）
            exif_data = self.exif_extractor.extract_exif_data(file_path)

            full_embedded = self._try_full_embedded_raw_preview(file_path, exif_data)
            if full_embedded is not None:
                return full_embedded

            # 檢查文件大小以決定處理策略
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            use_fast_processing = file_size_mb > 20 and not use_full_resolution
            
            # 處理 RAW 文件
            # Check if we should use Preview (embedded JPEG) instead of processing RAW
            if not use_full_resolution and not libraw_first:
               if skip_key in _LIBRAW_UNSUPPORTED_PATHS:
                   return self.cache.get_preview(file_path)
               # OPTIMIZATION: Try embedded / cached preview instead of full RAW demosaic.
               
               # Check Preview Cache
               cached_preview = self.cache.get_preview(file_path)
               if cached_preview is not None:
                   if _verbose_orientation_logs():
                       print(f"[PREVIEW] Using cached preview for {os.path.basename(file_path)}")
                   return cached_preview
               
               # Extract Preview
               if _verbose_orientation_logs():
                   print(f"[PREVIEW] Extracting preview from RAW for {os.path.basename(file_path)}")
               mem_max = memory_preview_max_edge()
               preview = self.thumbnail_extractor.extract_preview_from_raw(
                   file_path, max_size=mem_max
               )
               
               if preview is not None:
                   # Check if preview is large enough for a good "Fit" view
                   # Many phone DNGs have tiny previews (e.g. 1024px) that look bad
                   h, w = preview.shape[:2]
                   fit_min = max(1024, int(mem_max * 0.75))
                   if max(h, w) >= fit_min:
                       # Apply Orientation to Preview
                       orientation = exif_data.get('orientation', 1) if exif_data else 1
                       if orientation != 1:
                           preview = self._apply_orientation_correction(preview, orientation, exif_data)
                       
                       # Cache Preview
                       self.cache.put_preview(file_path, preview)
                       return preview
                   else:
                       if _verbose_orientation_logs():
                           print(f"[PREVIEW] Embedded preview is too small ({w}x{h}), falling back to RAW processing for better quality")
               
               # Fallback to RAW processing if preview extraction fails
               if _verbose_orientation_logs():
                   print("[PREVIEW] Preview extraction failed, falling back to RAW processing")

            # 獲取處理器參數
            params = self.raw_processor.get_optimized_processing_params(
                file_path, exif_data
            )
            
            # 根據需求調整參數
            if use_fast_processing:
                params['half_size'] = True

            # LibRaw-first: first fit view uses same pipeline as zoom (half-res decode is faster than full demosaic)
            if libraw_first and not use_full_resolution:
                params['half_size'] = True
            
            if use_full_resolution:
                params['half_size'] = False
                params['output_bps'] = 8  # 8-bit 足夠顯示
            
            # CRITICAL: Force rawpy to ignore EXIF orientation
            params['user_flip'] = 0
            
            # Fast full-res decode: LibRaw unpack + cv2 SIMD pixel math with
            # exact LibRaw color parity (see src/fast_raw_decode.py; gate:
            # scripts/fast_raw_decode_parity_gate.py). 1.4-1.7x faster than
            # the rawpy LINEAR path below with better demosaic quality (EA
            # vs bilinear). Returns None for unsupported sensors (X-Trans,
            # ...) or mismatched params -- everything below is the unchanged
            # fallback. RAWVIEWER_FAST_RAW_DECODE=0 disables.
            rgb_image = None
            if use_full_resolution or params.get('half_size'):
                from enhanced_raw_processor import _rawpy_global_lock as _fast_lock
                from fast_raw_decode import (
                    DecodeCancelled,
                    decode_half_from_unpacked,
                    finish_full_decode,
                    params_supported,
                    params_supported_half,
                    try_fast_raw_decode,
                    unpack_raw,
                )

                def _load_task_cancelled() -> bool:
                    try:
                        from image_load_manager import worker_thread_local

                        t = getattr(worker_thread_local, "task", None)
                        return bool(t is not None and t.is_cancelled())
                    except Exception:
                        return False

                try:
                    if use_full_resolution:
                        # Reuse the unpack stashed by this file's fit-view
                        # half decode (if any): skips the second LibRaw
                        # unpack, leaving only demosaic + color math.
                        unpacked = self._take_unpacked_raw(file_path)
                        if unpacked is not None and params_supported(params):
                            import logging

                            logging.getLogger(__name__).info(
                                "[FAST_RAW] full tier from stashed unpack: %s",
                                os.path.basename(file_path),
                            )
                            try:
                                rgb_image = finish_full_decode(
                                    unpacked, cancel_check=_load_task_cancelled
                                )
                            except DecodeCancelled:
                                raise
                            except Exception as e:
                                # finish_full_decode() has no internal
                                # safety net (unlike try_fast_raw_decode
                                # below) -- a demosaic/matrix failure on the
                                # stashed unpack (e.g. an odd-dimension or
                                # unusual-CFA sensor that unpack_raw's own
                                # checks let through) must not propagate and
                                # abort the whole decode with nothing
                                # displayed; fall through to try_fast_raw_
                                # decode / rawpy below instead.
                                logging.getLogger(__name__).warning(
                                    "[FAST_RAW] finish_full_decode failed for %s, "
                                    "falling back: %s",
                                    os.path.basename(file_path),
                                    e,
                                )
                                rgb_image = None
                        if rgb_image is None:
                            rgb_image = try_fast_raw_decode(
                                file_path,
                                params,
                                rawpy_lock=_fast_lock,
                                cancel_check=_load_task_cancelled,
                            )
                    elif params_supported_half(params):
                        # Fit-view half tier via the fast pixel path so the
                        # unpack can be stashed for the deferred full decode.
                        # Marginal cost over rawpy half_size is ~0 (both are
                        # unpack-bound); any failure falls through to the
                        # unchanged rawpy half decode below.
                        unpacked = unpack_raw(file_path, rawpy_lock=_fast_lock)
                        if unpacked is not None:
                            try:
                                rgb_image = decode_half_from_unpacked(
                                    unpacked, cancel_check=_load_task_cancelled
                                )
                            except DecodeCancelled:
                                raise
                            except Exception as e:
                                # Same rationale as finish_full_decode above:
                                # decode_half_from_unpacked() has no internal
                                # safety net either. Leave rgb_image None so
                                # the unchanged rawpy half decode below runs.
                                import logging

                                logging.getLogger(__name__).warning(
                                    "[FAST_RAW] decode_half_from_unpacked failed "
                                    "for %s, falling back: %s",
                                    os.path.basename(file_path),
                                    e,
                                )
                                rgb_image = None
                            if rgb_image is not None:
                                self._stash_unpacked_raw(file_path, unpacked)
                except DecodeCancelled:
                    # The owning prefetch task was cancelled (e.g. user
                    # navigated); abort the whole decode instead of falling
                    # back to a slower rawpy decode nobody wants anymore.
                    import logging

                    logging.getLogger(__name__).info(
                        "[FAST_RAW] decode aborted mid-flight (task cancelled): %s",
                        os.path.basename(file_path),
                    )
                    return None

            # 處理 RAW - 使用 Process Pool 繞過 GIL
            if rgb_image is None and executor:
                try:
                    # logger = logging.getLogger(__name__)
                    # logger.info(f"[PIL/PROCESS_POOL] Offloading RAW postprocess to pool for {file_path}")
                    future = executor.submit(decode_raw_file, file_path, params)
                    rgb_image = future.result()
                except Exception as pool_err:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"[PROCESS_POOL] Process pool decode failed for {os.path.basename(file_path)}: {pool_err}. "
                        "Falling back to in-process RAW decode."
                    )
            
            if rgb_image is None:
                import rawpy
                import time as _time
                from enhanced_raw_processor import _rawpy_global_lock, _heavy_fallback_semaphore
                # Keep the rawpy fallback consistent with the fast path's
                # embedded-JPEG WB sanity correction (misparsed as-shot WB on
                # bodies newer than the bundled LibRaw, e.g. EOS R6 Mark III).
                try:
                    from fast_raw_decode import get_corrected_camera_wb

                    _corrected_wb = get_corrected_camera_wb(file_path)
                    if _corrected_wb:
                        params = dict(params)
                        params['use_camera_wb'] = False
                        params['user_wb'] = list(_corrected_wb)
                except Exception:
                    pass
                _t_dec = _time.perf_counter()
                with _rawpy_global_lock:
                    raw_ctx = rawpy.imread(file_path)
                with raw_ctx as raw:
                    with _heavy_fallback_semaphore:
                        rgb_image = raw.postprocess(**params)
                from perf_metrics import perf_mark

                perf_mark(
                    "decode_rawpy",
                    (_time.perf_counter() - _t_dec) * 1000.0,
                    file_path,
                    tier="full" if use_full_resolution else "half",
                )
            
            # 應用方向校正 - 確保使用最新的 EXIFExtractor 邏輯
            if not exif_data or exif_data.get('orientation', 1) == 1:
                # Last-minute check: if we think it's 1, try a deep extraction just in case
                exif_data = self.exif_extractor.extract_exif_data(file_path)
            
            orientation = exif_data.get('orientation', 1) if exif_data else 1
            if os.environ.get("RAWVIEWER_VERBOSE_ORIENTATION_LOGS") == "1":
                # print(f"[ORIENTATION] UnifiedImageProcessor (RAW): {os.path.basename(file_path)} final_orientation={orientation}")
                pass
                
            if orientation != 1 and rgb_image is not None:
                rgb_image = self._apply_orientation_correction(rgb_image, orientation, exif_data)

            # Cache unadjusted base; sidecar is applied in process_full_image().
            if rgb_image is not None:
                self.cache.put_full_image(file_path, rgb_image)
                if libraw_first:
                    # RAW workflow: persist a LibRaw-rendered display tier
                    # (memory + disk JPEG). Every pre-decode tier is embedded-
                    # JPEG-colored, so without this a revisit whose full
                    # buffer was evicted repaints camera-JPEG colors and then
                    # visibly shifts when the fresh LibRaw render lands. With
                    # it, revisits paint consistent colors from the first
                    # frame. Workflow toggles already purge these caches, so
                    # embedded mode is unaffected. Runs on a daemon thread --
                    # the resize+JPEG encode costs ~110ms and must not delay
                    # the image-ready callback.
                    import threading

                    def _persist_libraw_preview(rgb=rgb_image, fp=file_path):
                        try:
                            import cv2

                            # NOTE: previously applied sidecar adjustments here
                            # so the persisted preview reflected edits, not
                            # just the raw base -- reverted (2026-07) after a
                            # report of color artifacts in the re-encoded
                            # preview. Persists the unadjusted base again
                            # until root-caused; see _reencode_persisted_
                            # preview_for_sidecar in main.py (also disabled).
                            h, w = rgb.shape[:2]
                            cap = memory_preview_max_edge()
                            if max(h, w) > cap:
                                sc = cap / max(h, w)
                                prev = cv2.resize(
                                    rgb,
                                    (max(1, int(w * sc)), max(1, int(h * sc))),
                                    interpolation=cv2.INTER_AREA,
                                )
                            else:
                                prev = rgb
                            ok, buf = cv2.imencode(
                                ".jpg",
                                cv2.cvtColor(prev, cv2.COLOR_RGB2BGR),
                                [int(cv2.IMWRITE_JPEG_QUALITY), 92],
                            )
                            self.cache.put_preview(
                                fp,
                                prev,
                                jpeg_data=buf.tobytes() if ok else None,
                                libraw_rendered=True,
                            )
                        except Exception:
                            pass

                    threading.Thread(
                        target=_persist_libraw_preview, daemon=True
                    ).start()

            return rgb_image
                
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            
            is_unsupported = "unsupported" in str(e).lower() or "not recognized" in str(e).lower()
            if is_unsupported:
                _LIBRAW_UNSUPPORTED_PATHS.add(skip_key)
                logger.warning(
                    f"RAW file unsupported by LibRaw (e.g., composite DNG): {os.path.basename(file_path)}. Error: {e}"
                )
            else:
                logger.error(f"Error processing RAW image {file_path}: {e}", exc_info=True)

            # LibRaw failed entirely; try byte-scan for embedded JPEG (already used in thumbnail
            # path, but preview may have been skipped or a different limit may help).
            try:
                from enhanced_raw_processor import extract_embedded_jpeg_by_scan
                lim = 8192 if use_full_resolution else memory_preview_max_edge()
                scanned = extract_embedded_jpeg_by_scan(file_path, lim)
                if scanned is not None:
                    exif_data = self.exif_extractor.extract_exif_data(file_path)
                    orientation = exif_data.get("orientation", 1) if exif_data else 1
                    if orientation != 1:
                        scanned = self._apply_orientation_correction(
                            scanned, orientation, exif_data
                        )
                    self.cache.put_preview(file_path, scanned)
                    if use_full_resolution:
                        self.cache.put_full_image(file_path, scanned)
                    return scanned
            except Exception:
                pass
            return None
    
    def _process_regular_image(
        self, file_path: str, use_full_resolution: bool = False
    ) -> Optional[QPixmap]:
        """處理常規圖像（JPEG/PNG/WebP）"""
        try:
            from image_cache import memory_preview_max_edge

            max_edge = 0 if use_full_resolution else memory_preview_max_edge()
            pixmap = load_pixmap_safe(file_path, max_edge=max_edge)
            if not pixmap.isNull():
                self.cache.put_pixmap(file_path, pixmap)
                return pixmap
            return None
                
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error processing regular image {file_path}: {e}", exc_info=True)
            return None
    
    def process_exif(self, file_path: str) -> Optional[Dict[str, Any]]:
        """處理 EXIF 數據（統一接口）"""
        return self.exif_extractor.extract_exif_data(file_path)

    def process_metadata_and_thumbnail(self, file_path: str, allow_heavy_fallback: bool = True,
                                      target_size: Optional[QSize] = None) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]:
        """
        COMBINED OPTIMIZATION: Extract both metadata and thumbnail in a single pass.
        This is crucial for RAW files to avoid opening the large file twice.
        """
        is_raw = self._is_raw_file(file_path)
        
        if not is_raw:
            # For non-RAW, standard sequential calls are fine as they use different backends
            exif = self.process_exif(file_path)
            thumb = self.process_thumbnail(file_path, allow_heavy_fallback, target_size)
            return exif, thumb

        # Gallery tiles: try cache + embedded JPEG without parsing the full RAW container.
        cached_exif = self.cache.get_exif(file_path)
        cached_thumb = self.cache.get_thumbnail(file_path)
        if cached_thumb is None and target_size is None:
            cached_thumb = self.cache.get_preview(file_path)

        if cached_thumb is not None and target_size is None and allow_heavy_fallback:
            try:
                if hasattr(cached_thumb, "shape"):
                    if max(int(cached_thumb.shape[0]), int(cached_thumb.shape[1])) < int(
                        memory_preview_max_edge() * 0.85
                    ):
                        cached_thumb = None
                elif hasattr(cached_thumb, "width"):
                    if max(int(cached_thumb.height()), int(cached_thumb.width())) < int(
                        memory_preview_max_edge() * 0.85
                    ):
                        cached_thumb = None
            except Exception:
                pass

        if cached_thumb is not None and target_size is None:
            try:
                min_px = int(memory_preview_max_edge() * 0.85)
                if hasattr(cached_thumb, "shape"):
                    if max(int(cached_thumb.shape[0]), int(cached_thumb.shape[1])) < min_px:
                        cached_thumb = None
                elif hasattr(cached_thumb, "width"):
                    if max(int(cached_thumb.height()), int(cached_thumb.width())) < min_px:
                        cached_thumb = None
            except Exception:
                pass

        if cached_exif is not None and cached_thumb is not None:
            if not cached_exif.get("minimal_preview_exif") and self._cached_preview_meets_display_tier(
                cached_thumb
            ):
                return cached_exif, cached_thumb

        if allow_heavy_fallback and target_size is None:
            fast_pair = self._extract_raw_preview_before_full_exiftool(
                file_path, cached_exif, cached_thumb
            )
            if fast_pair is not None:
                return fast_pair

        exif = cached_exif or self.exif_extractor.extract_exif_data(file_path)
        thumb = cached_thumb
        if thumb is None:
            max_dim = self._raw_thumbnail_extract_max_size(
                target_size, allow_heavy_fallback=allow_heavy_fallback
            )
            thumb = self.thumbnail_extractor.extract_thumbnail_from_raw(
                file_path,
                max_size=max_dim,
                allow_scan_fallback=True,
                raw_object=None,
            )
            if thumb is not None and exif:
                orientation = exif.get("orientation", 1)
                if orientation != 1:
                    thumb = self._apply_orientation_correction(
                        thumb, orientation, exif
                    )
        if thumb is not None and exif is not None:
            return exif, thumb

        # RAW Path: Open once, extract both
        import rawpy
        try:
            from enhanced_raw_processor import _rawpy_global_lock
            with _rawpy_global_lock:
                with rawpy.imread(file_path) as raw:
                    # 1. Extract EXIF (verified with rawpy sensor sizes)
                    exif = self.exif_extractor.extract_exif_data(file_path, raw_object=raw)
                
                    # 2. Extract Thumbnail
                    max_size_val = self._raw_thumbnail_extract_max_size(
                        target_size, allow_heavy_fallback=allow_heavy_fallback
                    )
                    thumb = self.thumbnail_extractor.extract_thumbnail_from_raw(
                        file_path,
                        max_size=max_size_val,
                        allow_scan_fallback=allow_heavy_fallback,
                        raw_object=raw
                    )
                
                    # Apply orientation to thumbnail if needed
                    if thumb is not None and exif:
                        orientation = exif.get('orientation', 1)
                        if orientation != 1:
                            thumb = self._apply_orientation_correction(thumb, orientation, exif)
                
                    return exif, thumb
        except Exception:
            # Fallback to sequential if single-pass fails
            exif = self.process_exif(file_path)
            thumb = self.process_thumbnail(file_path, allow_heavy_fallback, target_size)
            return exif, thumb
    
    def _apply_orientation_correction(
        self,
        image_array: np.ndarray,
        orientation: int,
        exif_data: Dict[str, Any] = None
    ) -> np.ndarray:
        """Apply container EXIF orientation (delegates to common_image_loader)."""
        from common_image_loader import apply_container_orientation_to_array

        return apply_container_orientation_to_array(
            image_array, orientation, exif_data
        )
