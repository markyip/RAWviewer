"""Unified mixin part 7 of 7 for UnifiedImageProcessor."""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QPixmap

from .helpers import (
    _LIBRAW_UNSUPPORTED_PATHS,
    _is_libraw_unsupported_error,
    _mark_libraw_unsupported,
    _on_gui_thread,
    _submit_persist,
    _verbose_orientation_logs,
    decode_raw_file,
    load_pixmap_safe,
    memory_preview_max_edge,
    rawpy_imread_warm,
    use_libraw_consistent_preview_first,
)


class UnifiedPart7:
    def _process_raw_image(self, file_path: str,
                          use_full_resolution: bool = False,
                          executor: Any | None = None) -> np.ndarray | None:
        """處理 RAW 圖像"""
        libraw_first = use_libraw_consistent_preview_first(file_path)
        skip_key = os.path.normcase(os.path.abspath(file_path))
        from common_image_loader import dng_prefers_embedded_preview_first
        if dng_prefers_embedded_preview_first(file_path):
            _mark_libraw_unsupported(skip_key)
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
                _mark_libraw_unsupported(skip_key)
        if skip_key in _LIBRAW_UNSUPPORTED_PATHS:
            cached = self.cache.get_preview(file_path)
            if cached is None:
                cached = self.cache.get_full_image(file_path)
            if cached is not None:
                cached_dim = max(cached.shape[0], cached.shape[1]) if hasattr(cached, "shape") else 0
                if cached_dim >= 1024:
                    return cached
            exif_data = self.exif_extractor.extract_exif_data(file_path)

            # Full-resolution request: prefer sensor-covering embeds / PIL TIFF.
            # Fit-view request: still must extract a displayable embed -- LibRaw
            # can never demosaic these files. Returning None here was a regression
            # for HE/HE* NEF: a stages={'full'} fit-view task (e.g. after
            # quality_on_screen or RAW-mode consistent-fit) emitted a false
            # "unsupported or corrupt" error when the in-memory cache had been
            # evicted by gallery preload, even though the embedded JPEG decodes
            # fine on the next paint.
            if use_full_resolution:
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
                            self.cache.mark_full_image_source(file_path, is_embedded_jpeg=False)
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
            #
            # NOTE: deliberately NOT marked via mark_full_image_source here.
            # This whole branch is reached only for skip_key already in
            # _LIBRAW_UNSUPPORTED_PATHS (X-Trans, NEF HE, corrupt/composite
            # DNG, ...) -- the embedded preview is the permanent, only
            # available "full" tier for these files; there is no true
            # decode to ever upgrade to, so tagging it as an embedded-JPEG
            # stand-in would just make the idle-decode-after-nav-pause
            # timer re-queue a pointless (if cheap, cache-hit) background
            # request forever with nothing to gain.
            native_preview = self.thumbnail_extractor.extract_embedded_native_preview(file_path)
            if native_preview is not None:
                orientation = exif_data.get("orientation", 1) if exif_data else 1
                if orientation != 1:
                    native_preview = self._apply_orientation_correction(
                        native_preview, orientation, exif_data
                    )
                self.cache.put_preview(file_path, native_preview)
                if use_full_resolution:
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
        try:
            # 獲取 EXIF 數據（用於處理參數）
            exif_data = self.exif_extractor.extract_exif_data(file_path)

            # 處理文件大小以決定處理策略
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            use_fast_processing = file_size_mb > 20 and not use_full_resolution

            # Full-resolution path only: accept sensor-covering embedded JPEG.
            # For fit-view / non-full, skip (would pin 30–60MP arrays in RAM).
            #
            # Only take the embedded-JPEG shortcut the FIRST time this file's
            # full tier is requested. A repeat use_full_resolution=True
            # request for a file whose full_image is already an embedded-JPEG
            # stand-in (full_image_is_embedded_jpeg) means the caller
            # deliberately wants the true pixels now -- but ONLY in the RAW
            # (libraw_first) workflow. In the non-RAW / embedded-JPEG workflow
            # that stand-in *is* the intended full image; demosaicing on
            # zoom/idle made non-RAW browsing look like LibRaw and feel slow.
            if use_full_resolution:
                if not self.cache.full_image_is_embedded_jpeg(file_path):
                    full_embedded = self._try_full_embedded_raw_preview(file_path, exif_data)
                    if full_embedded is not None:
                        return full_embedded
                elif not libraw_first:
                    cached_full = self.cache.get_full_image(file_path)
                    if cached_full is not None:
                        return cached_full
                    full_embedded = self._try_full_embedded_raw_preview(file_path, exif_data)
                    if full_embedded is not None:
                        return full_embedded
            else:
                # Prefer preview-max-edge embedded/scanned JPEG before demosaic.
                mem_max = memory_preview_max_edge()
                cached_preview = self.cache.get_preview(file_path)
                if cached_preview is not None:
                    return cached_preview
                if not libraw_first:
                    preview = self.thumbnail_extractor.extract_preview_from_raw(
                        file_path, max_size=mem_max
                    )
                    if preview is not None:
                        h, w = preview.shape[:2]
                        fit_min = max(1024, int(mem_max * 0.75))
                        if max(h, w) >= fit_min:
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

            # 處理 RAW 文件
            # Check if we should use Preview (embedded JPEG) instead of processing RAW
            if not use_full_resolution and not libraw_first:
               if skip_key in _LIBRAW_UNSUPPORTED_PATHS:
                   return self.cache.get_preview(file_path)
               # Remaining demosaic fallthrough when embed was too small / missing.
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
            # scripts/test/fast_raw_decode_parity_gate.py). 1.4-1.7x faster than
            # the rawpy LINEAR path below with better demosaic quality (EA
            # vs bilinear). Returns None for unsupported sensors (X-Trans,
            # ...) or mismatched params -- everything below is the unchanged
            # fallback. RAWVIEWER_FAST_RAW_DECODE=0 disables.
            rgb_image = None

            def _load_task_cancelled() -> bool:
                try:
                    from image_load_manager import worker_thread_local

                    t = getattr(worker_thread_local, "task", None)
                    return bool(t is not None and t.is_cancelled())
                except Exception:
                    return False

            if use_full_resolution or params.get('half_size'):
                from enhanced_raw_processor import _rawpy_global_lock as _fast_lock
                from processing.raw_decode.fast import (
                    DecodeCancelled,
                    decode_half_from_unpacked,
                    finish_full_decode,
                    params_supported,
                    params_supported_half,
                    prefer_gpu_decode_enabled,
                    try_fast_raw_decode,
                    unpack_raw,
                )

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
                                    unpacked,
                                    cancel_check=_load_task_cancelled,
                                    prefer_gpu=prefer_gpu_decode_enabled(),
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
                            elif unpacked is not None:
                                # Decode failed but unpack succeeded — keep
                                # mosaic for a later full-tier finish.
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
                if _load_task_cancelled():
                    return None
                try:
                    import concurrent.futures

                    future = executor.submit(decode_raw_file, file_path, params)
                    while True:
                        if _load_task_cancelled():
                            future.cancel()
                            return None
                        try:
                            rgb_image = future.result(timeout=0.05)
                            break
                        except concurrent.futures.TimeoutError:
                            continue
                except Exception as pool_err:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"[PROCESS_POOL] Process pool decode failed for {os.path.basename(file_path)}: {pool_err}. "
                        "Falling back to in-process RAW decode."
                    )

            if rgb_image is None:
                if _load_task_cancelled():
                    return None
                import time as _time

                from enhanced_raw_processor import _heavy_fallback_semaphore, _rawpy_global_lock
                # Keep the rawpy fallback consistent with the fast path's
                # embedded-JPEG WB sanity correction (misparsed as-shot WB on
                # bodies newer than the bundled LibRaw, e.g. EOS R6 Mark III).
                try:
                    from processing.raw_decode.fast import get_corrected_camera_wb

                    _corrected_wb = get_corrected_camera_wb(file_path)
                    if _corrected_wb:
                        params = dict(params)
                        params['use_camera_wb'] = False
                        params['user_wb'] = list(_corrected_wb)
                except Exception:
                    pass
                _t_dec = _time.perf_counter()
                with _rawpy_global_lock:
                    if _load_task_cancelled():
                        return None
                    raw_ctx = rawpy_imread_warm(file_path)
                with raw_ctx as raw:
                    if _load_task_cancelled():
                        return None
                    with _heavy_fallback_semaphore:
                        if _load_task_cancelled():
                            return None
                        rgb_image = raw.postprocess(**params)
                from core.perf_metrics import perf_mark

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
                try:
                    from processing.raw_decode.gpu_bridge import DeviceRgb, apply_orientation_device_rgb

                    if isinstance(rgb_image, DeviceRgb):
                        rgb_image = apply_orientation_device_rgb(rgb_image, orientation)
                    else:
                        rgb_image = self._apply_orientation_correction(
                            rgb_image, orientation, exif_data
                        )
                        # rot90/flip often yield non-contiguous views; compact
                        # once here so put_full_image(copy=False) does not pay
                        # a second full-frame copy.
                        if use_full_resolution and hasattr(rgb_image, "flags"):
                            rgb_image = np.ascontiguousarray(rgb_image)
                except Exception:
                    try:
                        from processing.raw_decode.gpu_bridge import DeviceRgb as _DR

                        host = (
                            rgb_image.to_numpy()
                            if isinstance(rgb_image, _DR)
                            else rgb_image
                        )
                        rgb_image = self._apply_orientation_correction(
                            host, orientation, exif_data
                        )
                    except Exception:
                        pass

            # Cache unadjusted base; sidecar is applied in process_full_image().
            # Half-size fit decodes must NOT enter full_image_cache (pollutes
            # the sensor-res tier and forces a useless multi-hundred-MB copy);
            # LibRaw half still persists via put_preview below.
            if rgb_image is not None:
                # One generation for this decode's deferred put_full / preview
                # persist so a later decode of the same path can invalidate both.
                cache_gen = self._bump_async_cache_gen(file_path)
                try:
                    from processing.raw_decode.gpu_bridge import DeviceRgb

                    if use_full_resolution:
                        if isinstance(rgb_image, DeviceRgb):
                            # Defer host D2H so CUDA-GL display can paint first.
                            def _cache_device_host(
                                dev=rgb_image, fp=file_path, expected=cache_gen
                            ):
                                try:
                                    if self._async_cache_gen_current(fp) != expected:
                                        return
                                    host = dev.to_numpy()
                                    if self._async_cache_gen_current(fp) != expected:
                                        return
                                    self.cache.put_full_image(fp, host, copy=False)
                                    self.cache.mark_full_image_source(
                                        fp, is_embedded_jpeg=False
                                    )
                                except Exception:
                                    pass

                            _submit_persist(_cache_device_host)
                            host = None
                        else:
                            self.cache.put_full_image(
                                file_path, rgb_image, copy=False
                            )
                            self.cache.mark_full_image_source(
                                file_path, is_embedded_jpeg=False
                            )
                            host = rgb_image
                    else:
                        host = (
                            None
                            if isinstance(rgb_image, DeviceRgb)
                            else rgb_image
                        )
                except Exception:
                    host = rgb_image if hasattr(rgb_image, "shape") else None
                    if (
                        use_full_resolution
                        and host is not None
                        and not hasattr(host, "tensor")
                    ):
                        try:
                            self.cache.put_full_image(file_path, host, copy=False)
                            self.cache.mark_full_image_source(
                                file_path, is_embedded_jpeg=False
                            )
                        except Exception:
                            pass
                if libraw_first:
                    # RAW workflow: persist a LibRaw-rendered display tier
                    # (memory + disk JPEG). Every pre-decode tier is embedded-
                    # JPEG-colored, so without this a revisit whose full
                    # buffer was evicted repaints camera-JPEG colors and then
                    # visibly shifts when the fresh LibRaw render lands. With
                    # it, revisits paint consistent colors from the first
                    # frame. Workflow toggles already purge these caches, so
                    # embedded mode is unaffected.
                    #
                    # Hybrid bake of browse-edits (when enabled):
                    #   1) Warm full-res adjusted memo -- resize+encode only
                    #      (UI already paid the multi-second apply).
                    #   2) Memo miss -- resize FIRST, then apply at preview
                    #      scale (never a second full-res apply just for a
                    #      2304px JPEG; spatial tools run at preview scale).
                    # Default (no browse-edits): persist the unadjusted base.

                    def _persist_libraw_preview(
                        rgb=rgb_image, fp=file_path, expected=cache_gen
                    ):
                        try:
                            if self._async_cache_gen_current(fp) != expected:
                                return
                            import cv2

                            from processing.raw_decode.gpu_bridge import DeviceRgb as _DR

                            if isinstance(rgb, _DR):
                                rgb = rgb.to_numpy()
                            if rgb is None:
                                return
                            if self._async_cache_gen_current(fp) != expected:
                                return

                            from raw_adjustments import (
                                apply_adjustments_to_rgb,
                                is_default_adjustments,
                                load_adjustments_for_file,
                                sidecar_adjustments_enabled,
                            )

                            source = rgb
                            already_preview_sized = False
                            if sidecar_adjustments_enabled():
                                adj = load_adjustments_for_file(fp)
                                if adj and not is_default_adjustments(adj):
                                    memo = self._peek_adjusted_full(
                                        fp, adj, rgb.shape
                                    )
                                    if memo is not None:
                                        # Path 1: warm memo -- no re-apply.
                                        source = memo
                                    else:
                                        # Path 2: miss -- downscale then apply.
                                        h0, w0 = rgb.shape[:2]
                                        cap0 = memory_preview_max_edge()
                                        if max(h0, w0) > cap0:
                                            sc0 = cap0 / max(h0, w0)
                                            small = cv2.resize(
                                                rgb,
                                                (
                                                    max(1, int(w0 * sc0)),
                                                    max(1, int(h0 * sc0)),
                                                ),
                                                interpolation=cv2.INTER_AREA,
                                            )
                                        else:
                                            small = rgb
                                        if (
                                            self._async_cache_gen_current(fp)
                                            != expected
                                        ):
                                            return
                                        source = apply_adjustments_to_rgb(
                                            small, adj
                                        )
                                        already_preview_sized = True
                            if self._async_cache_gen_current(fp) != expected:
                                return
                            if already_preview_sized:
                                prev = source
                            else:
                                h, w = source.shape[:2]
                                cap = memory_preview_max_edge()
                                if max(h, w) > cap:
                                    sc = cap / max(h, w)
                                    prev = cv2.resize(
                                        source,
                                        (
                                            max(1, int(w * sc)),
                                            max(1, int(h * sc)),
                                        ),
                                        interpolation=cv2.INTER_AREA,
                                    )
                                else:
                                    prev = source
                            if self._async_cache_gen_current(fp) != expected:
                                return
                            ok, buf = cv2.imencode(
                                ".jpg",
                                cv2.cvtColor(prev, cv2.COLOR_RGB2BGR),
                                [int(cv2.IMWRITE_JPEG_QUALITY), 92],
                            )
                            if self._async_cache_gen_current(fp) != expected:
                                return
                            self.cache.put_preview(
                                fp,
                                prev,
                                jpeg_data=buf.tobytes() if ok else None,
                                libraw_rendered=True,
                            )
                        except Exception:
                            pass

                    _submit_persist(_persist_libraw_preview)

            return rgb_image

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)

            is_unsupported = _is_libraw_unsupported_error(e)
            if is_unsupported:
                _mark_libraw_unsupported(skip_key)
                logger.warning(
                    f"RAW file unsupported by LibRaw (e.g., composite DNG): {os.path.basename(file_path)}. Error: {e}"
                )
            else:
                logger.error(f"Error processing RAW image {file_path}: {e}", exc_info=True)

            # LibRaw failed entirely; try byte-scan for embedded JPEG (already used in thumbnail
            # path, but preview may have been skipped or a different limit may help).
            # NOTE: deliberately not marked via mark_full_image_source (see the
            # matching note on the _LIBRAW_UNSUPPORTED_PATHS branch above) --
            # LibRaw failed here, so a retry will predictably fail the same
            # way; there is no true decode to ever upgrade to.
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
    ) -> QPixmap | None:
        """處理常規圖像（JPEG/PNG/WebP）"""
        try:
            from image_cache import memory_preview_max_edge

            max_edge = 0 if use_full_resolution else memory_preview_max_edge()

            # QPixmap is GUI-thread-only in Qt. This runs on an ImageLoadWorker
            # for every non-RAW file, so off the GUI thread it returns a
            # QImage instead and the conversion happens once the buffer has
            # been delivered. The pixmap cache is skipped there too: the GUI
            # thread reads it, and populating it from a worker was the second
            # half of the same problem.
            if not _on_gui_thread():
                from common_image_loader import load_qimage_safe

                qimg = load_qimage_safe(file_path, max_edge=max_edge)
                if qimg is not None and not qimg.isNull():
                    return qimg
                # Nothing safe to return from here; the caller treats None as
                # "no full-res buffer", which it already handles.
                return None

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

    def process_exif(self, file_path: str) -> dict[str, Any] | None:
        """處理 EXIF 數據（統一接口）"""
        return self.exif_extractor.extract_exif_data(file_path)

    def process_metadata_and_thumbnail(self, file_path: str, allow_heavy_fallback: bool = True,
                                      target_size: QSize | None = None) -> tuple[dict[str, Any] | None, np.ndarray | None]:
        """
        COMBINED OPTIMIZATION: Extract both metadata and thumbnail in a single pass.
        This is crucial for RAW files to avoid opening the large file twice.
        """
        if self.cache.is_libraw_preview(file_path):
            cached_exif = self.cache.get_exif(file_path, verify=True)
            prev = self.cache.get_preview(file_path)
            if prev is not None:
                from raw_adjustments import mark_array_as_already_adjusted
                if target_size is not None:
                    h, w = prev.shape[:2]
                    scale = min(target_size.width() / w, target_size.height() / h)
                    if scale < 1.0:
                        import cv2
                        prev = cv2.resize(
                            prev,
                            (max(1, int(w * scale)), max(1, int(h * scale))),
                            interpolation=cv2.INTER_AREA,
                        )
                mark_array_as_already_adjusted(prev)
                return cached_exif, prev

        is_raw = self._is_raw_file(file_path)

        if not is_raw:
            # For non-RAW, standard sequential calls are fine as they use different backends
            exif = self.process_exif(file_path)
            thumb = self.process_thumbnail(file_path, allow_heavy_fallback, target_size)
            return exif, thumb

        # Gallery tiles: try cache + embedded JPEG without parsing the full RAW container.
        # verify=True: this record feeds preview extraction/orientation (worker thread).
        cached_exif = self.cache.get_exif(file_path, verify=True)
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
                elif hasattr(cached_thumb, "width") and max(int(cached_thumb.height()), int(cached_thumb.width())) < int(
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
                elif hasattr(cached_thumb, "width") and max(int(cached_thumb.height()), int(cached_thumb.width())) < min_px:
                    cached_thumb = None
            except Exception:
                pass

        if cached_exif is not None and cached_thumb is not None and not cached_exif.get("minimal_preview_exif") and self._cached_preview_meets_display_tier(
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
        try:
            from enhanced_raw_processor import _rawpy_global_lock
            with _rawpy_global_lock, rawpy_imread_warm(file_path) as raw:
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

    def _embedded_jpeg_edit_base(
        self, file_path: str, *, use_full_resolution: bool
    ) -> np.ndarray | None:
        """Largest embedded preview as an 8-bit RGB edit base, or None.

        Used when LibRaw cannot demosaic (Nikon HE/HE*, Sony ARW6 before the
        packaging LibRaw pin, …). Returns uint8 RGB already orientation-corrected
        so the edit pipeline can divide by 255 the same way it does for a JPEG.

        Half-res matches the LibRaw path's ``half_size`` budget (long edge ≤2048)
        so live Adjust stays responsive regardless of base kind.

        Extract via byte-scan — do not route through ``process_full_image`` /
        ``process_thumbnail``, which refuse LibRaw-unsupported RAW or decode a
        full-size JPEG only to downscale it for live Adjust.
        """
        try:
            from enhanced_raw_processor import extract_embedded_jpeg_by_scan

            # 0 = largest embed (export); 2048 = live-Adjust budget.
            max_size = 0 if use_full_resolution else 2048
            img = extract_embedded_jpeg_by_scan(file_path, max_size)
            if img is None:
                return None
            img = np.asarray(img)
            if img.ndim != 3 or img.shape[2] < 3:
                return None
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            img = np.ascontiguousarray(img[..., :3])

            # Scan returns TIFF-IFD pixels without the container Orientation.
            try:
                exif_data = self.exif_extractor.extract_exif_data(file_path)
                orientation = (exif_data or {}).get("orientation", 1)
                if orientation != 1:
                    img = self._apply_orientation_correction(
                        img, orientation, exif_data
                    )
            except Exception:
                pass
            return img
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Embedded-JPEG edit base failed for %s: %s", file_path, exc
            )
            return None

    def _apply_orientation_correction(
        self,
        image_array: np.ndarray,
        orientation: int,
        exif_data: dict[str, Any] = None
    ) -> np.ndarray:
        """Apply container EXIF orientation (delegates to common_image_loader)."""
        from common_image_loader import apply_container_orientation_to_array

        return apply_container_orientation_to_array(
            image_array, orientation, exif_data
        )
