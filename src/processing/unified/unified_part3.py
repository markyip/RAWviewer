"""Unified mixin part 3 of 7 for UnifiedImageProcessor."""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from .helpers import (
    _EDIT_BASE_INFLIGHT,
    _EDIT_BASE_INFLIGHT_LOCK,
    _EDIT_BASE_LRU,
    _EDIT_BASE_LRU_LOCK,
    _LIBRAW_UNSUPPORTED_PATHS,
    _edit_base_file_ident,
    _edit_base_lru_max,
    _edit_base_max_edge,
    _is_libraw_unsupported_error,
    _mark_libraw_unsupported,
    cap_working_edit_base,
    decode_raw_file,
    rawpy_imread_warm,
)


class UnifiedPart3:
    def decode_raw_edit_base(
        self,
        file_path: str,
        executor: Any | None = None,
        use_full_resolution: bool = False,
        apply_lens_correction: bool = False,
    ) -> np.ndarray | None:
        """
        Edit-panel working base for a RAW file.

        Prefer LibRaw scene-linear demosaic (16-bit / float). When LibRaw cannot
        demosaic (HE/HE* NEF, ARW6 before the packaging pin, …), fall back to
        the largest embedded JPEG as an 8-bit display-referred base — same
        contract the pipeline already uses for uint8 inputs. Not cached for
        browse. Defaults to half-size for responsive live editing.

        ``apply_lens_correction``: if True, undistort using a lensfun profile
        matched from this file's EXIF (see raw_lens_correction.py) -- applied
        once here, after orientation correction (so the Modifier sees the
        buffer's final width/height), not as a per-tick pipeline adjustment.
        Only applied on the LibRaw demosaic path (embedded JPEG is already
        camera-processed).
        """
        skip_key = os.path.normcase(os.path.abspath(file_path))
        # Half-size only: full-res export bases are too large to keep around.
        # Ident includes mtime/size so a replaced file on Windows/macOS
        # invalidates without relying on path alone.
        file_ident = _edit_base_file_ident(file_path)
        max_edge = 0 if use_full_resolution else _edit_base_max_edge()
        cache_key = (
            skip_key,
            bool(use_full_resolution),
            bool(apply_lens_correction),
            file_ident,
            int(max_edge),
        )
        if not use_full_resolution:
            with _EDIT_BASE_LRU_LOCK:
                cached = _EDIT_BASE_LRU.get(cache_key)
                if cached is not None:
                    _EDIT_BASE_LRU.move_to_end(cache_key)
                    return cached

        # Module-level in-flight dedup: concurrent identical requests (the
        # panel's base request racing a preview worker's, each on its OWN
        # UnifiedImageProcessor instance) share one decode instead of queueing
        # a redundant multi-hundred-ms unpack behind the global rawpy lock.
        dedup_key = (
            skip_key,
            bool(use_full_resolution),
            bool(apply_lens_correction),
            int(max_edge),
        )
        import threading as _threading

        mine = False
        while True:
            with _EDIT_BASE_INFLIGHT_LOCK:
                slot = _EDIT_BASE_INFLIGHT.get(dedup_key)
                if slot is None:
                    slot = {"event": _threading.Event(), "result": None}
                    _EDIT_BASE_INFLIGHT[dedup_key] = slot
                    mine = True
            if mine:
                break
            slot["event"].wait(timeout=60.0)
            if slot["result"] is not None:
                return slot["result"]
            # Owner failed or timed out: loop and try to become the owner.
            with _EDIT_BASE_INFLIGHT_LOCK:
                if _EDIT_BASE_INFLIGHT.get(dedup_key) is slot:
                    _EDIT_BASE_INFLIGHT.pop(dedup_key, None)
        try:
            result = self._decode_raw_edit_base_impl(
                file_path,
                skip_key,
                executor=executor,
                use_full_resolution=use_full_resolution,
                apply_lens_correction=apply_lens_correction,
            )
            if result is not None and not use_full_resolution:
                result = cap_working_edit_base(result)
                with _EDIT_BASE_LRU_LOCK:
                    _EDIT_BASE_LRU[cache_key] = result
                    _EDIT_BASE_LRU.move_to_end(cache_key)
                    while len(_EDIT_BASE_LRU) > _edit_base_lru_max():
                        _EDIT_BASE_LRU.popitem(last=False)
            slot["result"] = result
            return result
        finally:
            with _EDIT_BASE_INFLIGHT_LOCK:
                _EDIT_BASE_INFLIGHT.pop(dedup_key, None)
            slot["event"].set()

    def _decode_raw_edit_base_impl(
        self,
        file_path: str,
        skip_key: str,
        executor: Any | None = None,
        use_full_resolution: bool = False,
        apply_lens_correction: bool = False,
    ) -> np.ndarray | None:
        from common_image_loader import dng_prefers_embedded_preview_first
        from raw_adjustments.tone_recovery import edit_base_decode_params

        try:
            exif_data = self.exif_extractor.extract_exif_data(file_path)
            half_size = not use_full_resolution
            # Browse-brightness decode (Blend highlights, no exp_shift) with
            # the corrected-WB handling built in; see edit_base_decode_params.
            edit_params = edit_base_decode_params(
                half_size=half_size, demosaic="AHD", for_file=file_path
            )

            rgb_image: np.ndarray | None = None
            if skip_key not in _LIBRAW_UNSUPPORTED_PATHS:
                try:
                    from enhanced_raw_processor import (
                        _heavy_fallback_semaphore,
                        _rawpy_global_lock,
                    )
                    from processing.raw_decode.fast import (
                        decode_half_from_unpacked,
                        params_supported_half,
                        try_fast_raw_decode,
                    )

                    # Prefer browse half-unpack stash (peek, don't consume) so
                    # opening Adjust skips a second LibRaw unpack when possible.
                    if half_size:
                        stashed = self._peek_unpacked_raw(file_path)
                        if (
                            stashed is not None
                            and params_supported_half(edit_params, return_linear=True)
                        ):
                            try:
                                rgb_image = decode_half_from_unpacked(
                                    stashed, return_linear=True
                                )
                            except Exception:
                                rgb_image = None

                    if rgb_image is None:
                        rgb_image = try_fast_raw_decode(
                            file_path,
                            edit_params,
                            rawpy_lock=_rawpy_global_lock,
                            return_linear=True,
                        )

                    cam_mul_for_scale = None
                    if rgb_image is None and executor is not None:
                        # Windows/Linux default: LibRaw process pool bypasses
                        # the GIL for the AHD/rawpy fallback. macOS usually
                        # has executor=None (pool off). Same helper as browse.
                        try:
                            future = executor.submit(
                                decode_raw_file, file_path, edit_params
                            )
                            rgb_image = future.result()
                        except Exception as pool_err:
                            logging.getLogger(__name__).warning(
                                "[EDIT] Process pool edit-base decode failed "
                                "for %s: %s; falling back in-process",
                                os.path.basename(file_path),
                                pool_err,
                            )
                            rgb_image = None

                    if rgb_image is None:
                        with _rawpy_global_lock:
                            raw_ctx = rawpy_imread_warm(file_path)
                        with raw_ctx as raw:
                            cam_mul_for_scale = list(raw.camera_whitebalance or [])
                            with _heavy_fallback_semaphore:
                                rgb_image = raw.postprocess(**edit_params)
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
                        from processing.raw_decode.fast import get_corrected_camera_wb

                        mul = get_corrected_camera_wb(file_path) or cam_mul_for_scale
                        if mul is None and edit_params.get("user_wb"):
                            mul = edit_params.get("user_wb")
                        mul3 = [float(m) for m in (mul or [])[:3] if float(m) > 0]
                        dmax_over_dmin = (
                            (max(mul3) / min(mul3)) if len(mul3) == 3 else 1.0
                        )
                    except Exception:
                        dmax_over_dmin = 1.0
                    if rgb_image is not None and rgb_image.dtype == np.uint16:
                        rgb_image = rgb_image.astype(np.float32) * (
                            dmax_over_dmin / 65535.0
                        )
                except Exception as libraw_err:
                    if _is_libraw_unsupported_error(libraw_err):
                        _mark_libraw_unsupported(skip_key)
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
                    from raw_adjustments.lens_correction import apply_lens_correction as _apply_lens_correction

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
                # Browse uses embedded/TIFF frames; Adjust requires demosaic.
                logging.getLogger(__name__).warning(
                    "[EDIT] LibRaw unsupported for %s; no RAW demosaic base available",
                    os.path.basename(file_path),
                )
                return None

            # LibRaw cannot demosaic (Nikon HE/HE*, Sony ARW6 before the
            # packaging LibRaw pin, exotic compressions, …): fall back to the
            # largest embedded JPEG as an 8-bit display-referred edit base.
            # WB / highlight recovery have far less headroom than a real RAW
            # base, but tone/colour/masks/crop/detail match the JPEG path the
            # pipeline already supports via uint8→linear.
            embedded = self._embedded_jpeg_edit_base(
                file_path, use_full_resolution=use_full_resolution
            )
            if embedded is not None:
                logging.getLogger(__name__).info(
                    "[EDIT] Embedded-JPEG edit base for %s (%dx%d, 8-bit) "
                    "-- no LibRaw demosaic available",
                    os.path.basename(file_path),
                    embedded.shape[1],
                    embedded.shape[0],
                )
                return embedded
            logging.getLogger(__name__).warning(
                "[EDIT] No demosaic base and no usable embedded JPEG for %s",
                os.path.basename(file_path),
            )
            return None
        except Exception as e:
            logging.getLogger(__name__).error(
                "[EDIT] RAW edit base decode failed for %s: %s",
                os.path.basename(file_path),
                e,
                exc_info=True,
            )
            return None
