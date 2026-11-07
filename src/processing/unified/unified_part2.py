"""Unified mixin part 2 of 7 for UnifiedImageProcessor."""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np


class UnifiedPart2:
    def _apply_sidecar_if_needed(
        self,
        file_path: str,
        rgb_image: np.ndarray | None,
        *,
        force: bool = False,
    ) -> np.ndarray | None:
        """Apply XMP sidecar adjustments on top of a base (unadjusted) RGB buffer.

        Memoized per (file, adjustment fingerprint, base buffer shape): see
        _adjusted_full_slots in __init__ for why.

        ``force=True`` skips the libraw-preview short-circuit so a deliberately
        downscaled interim buffer (progressive full apply) still gets edits.
        """
        if rgb_image is None:
            return None
        try:
            from raw_adjustments import is_array_already_adjusted

            if is_array_already_adjusted(rgb_image):
                return rgb_image
        except Exception:
            pass
        if not force:
            try:
                if self.cache.is_libraw_preview(file_path):
                    h, w = rgb_image.shape[:2]
                    from image_cache import memory_preview_max_edge
                    if max(h, w) <= memory_preview_max_edge():
                        return rgb_image
            except Exception:
                pass
        try:
            from processing.raw_decode.gpu_bridge import DeviceRgb

            if isinstance(rgb_image, DeviceRgb):
                # Editing still runs on host today; download only when needed.
                from raw_adjustments import (
                    is_default_adjustments,
                    load_adjustments_for_file,
                )

                adj = load_adjustments_for_file(file_path)
                if is_default_adjustments(adj):
                    return rgb_image
                rgb_image = rgb_image.to_numpy()
        except Exception:
            pass
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
            adj_key = self._adjustment_fingerprint(adj, rgb_image.shape)
            import threading as _threading

            inflight_key = (norm, adj_key)
            mine = False
            while True:
                with self._adjusted_full_lock:
                    for n, k, buf in self._adjusted_full_slots:
                        if n == norm and k == adj_key:
                            return buf
                    waiter = self._adjusted_inflight.get(inflight_key)
                    if waiter is None:
                        self._adjusted_inflight[inflight_key] = _threading.Event()
                        mine = True
                if mine:
                    break
                # Another thread is computing this exact apply (same file,
                # same adjustments, same base shape): wait for it rather than
                # burning a second multi-second pass on a 32MP buffer.
                waiter.wait(timeout=60.0)
                # Loop: either the memo now has it, or the owner failed and we
                # become the owner on the next pass.

            try:
                import time as _time

                from core.perf_metrics import perf_mark

                _t0 = _time.perf_counter()
                out = apply_adjustments_to_rgb(rgb_image, adj)
                perf_mark(
                    "sidecar_apply",
                    (_time.perf_counter() - _t0) * 1000.0,
                    file_path,
                    mp=rgb_image.shape[0] * rgb_image.shape[1] / 1e6,
                )
                with self._adjusted_full_lock:
                    # Keep one entry per file (the newest adj/shape) across up
                    # to 6 files: the previous 3-slot global list was evicted
                    # wholesale by neighbor prefetch, so revisits re-paid the
                    # full multi-second apply.
                    # In place: rebinding self._adjusted_full_slots would give
                    # this instance a private list again and unshare the memo.
                    kept = [
                        e for e in self._adjusted_full_slots if e[0] != norm
                    ][-5:]
                    kept.append((norm, adj_key, out))
                    self._adjusted_full_slots[:] = kept
                return out
            finally:
                with self._adjusted_full_lock:
                    ev = self._adjusted_inflight.pop(inflight_key, None)
                if ev is not None:
                    ev.set()
        except Exception:
            return rgb_image

    def apply_sidecar_progressive(
        self,
        file_path: str,
        rgb_image: np.ndarray | None,
        *,
        cancel_check: Any | None = None,
        on_interim: Any | None = None,
    ) -> np.ndarray | None:
        """Apply XMP edits with an optional preview-sized interim callback.

        For large bases (above ``memory_preview_max_edge``), downscale + apply
        first (~hundreds of ms) and invoke ``on_interim`` so the UI can paint
        demosaic-accurate edited pixels while the full-resolution apply
        (often 1–2s at 45MP) continues. Prefetch / default-adj callers should
        not use this — keep using ``_apply_sidecar_if_needed`` directly.

        Never writes adjusted pixels into ``full_image_cache`` (unadjusted
        base stays path-keyed; memo lives in ``_adjusted_full_slots``).
        Cross-platform: same path/normcase fingerprinting as the sync apply.
        """
        if rgb_image is None:
            return None

        def _cancelled() -> bool:
            try:
                return bool(cancel_check is not None and cancel_check())
            except Exception:
                return False

        try:
            from raw_adjustments import (
                is_default_adjustments,
                load_adjustments_for_file,
            )

            adj = load_adjustments_for_file(file_path)
            if is_default_adjustments(adj):
                return rgb_image
        except Exception:
            pass

        host = rgb_image
        try:
            from processing.raw_decode.gpu_bridge import DeviceRgb

            if isinstance(host, DeviceRgb):
                host = host.to_numpy()
        except Exception:
            pass

        if _cancelled():
            return None

        try:
            from image_cache import memory_preview_max_edge

            h, w = int(host.shape[0]), int(host.shape[1])
            cap = int(memory_preview_max_edge())
            if (
                on_interim is not None
                and cap > 0
                and max(h, w) > cap
            ):
                import cv2

                scale = cap / float(max(h, w))
                nw = max(1, int(round(w * scale)))
                nh = max(1, int(round(h * scale)))
                small = cv2.resize(host, (nw, nh), interpolation=cv2.INTER_AREA)
                if _cancelled():
                    return None
                interim = self._apply_sidecar_if_needed(
                    file_path, small, force=True
                )
                if interim is not None and not _cancelled():
                    try:
                        on_interim(interim)
                    except Exception:
                        logging.getLogger(__name__).debug(
                            "sidecar interim callback failed",
                            exc_info=True,
                        )
        except Exception:
            logging.getLogger(__name__).debug(
                "sidecar progressive interim failed",
                exc_info=True,
            )

        if _cancelled():
            return None
        return self._apply_sidecar_if_needed(file_path, host, force=True)
