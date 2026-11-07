"""Unified mixin part 1 of 7 for UnifiedImageProcessor."""
from __future__ import annotations

import os

import numpy as np

from .helpers import (
    _ADJUSTED_FULL_LOCK,
    _ADJUSTED_FULL_SLOTS,
    _ADJUSTED_INFLIGHT,
    _ASYNC_CACHE_GEN,
    _ASYNC_CACHE_GEN_LOCK,
    _ASYNC_CACHE_GEN_MAX,
    _UNPACKED_RAW_LOCK,
    _UNPACKED_RAW_SLOTS,
    EXIFExtractor,
    OptimizedRAWProcessor,
    ThumbnailExtractor,
    _unpack_stash_max_slots,
    get_image_cache,
)


class UnifiedPart1:
    def __init__(self):
        self.cache = get_image_cache()
        self.thumbnail_extractor = ThumbnailExtractor()
        self.exif_extractor = EXIFExtractor()
        self.raw_processor = OptimizedRAWProcessor()
        # LibRaw unpack reuse (see fast_raw_decode.UnpackedRaw): the fit-view
        # half decode stashes its unpacked mosaic so the deferred full decode
        # of the same file skips the second unpack (the 100-900ms dominant
        # cost). Small LRU (~2 bytes/sensor-pixel per slot) so A↔B revisit
        # and neighbor prefetch do not immediately evict the current file.
        # Consumed by the full decode via _take_unpacked_raw.
        #
        # MODULE-level, not per instance. ImageLoadWorker is a one-shot
        # QRunnable that builds its own UnifiedImageProcessor, so an instance
        # stash died with the worker and never served a hit across tasks.
        self._unpacked_raw_lock = _UNPACKED_RAW_LOCK
        self._unpacked_raw_slots = _UNPACKED_RAW_SLOTS
        # Sidecar-adjusted full-buffer memo: applying XMP adjustments to a
        # sensor-resolution buffer costs ~1.8-2.1s at 45MP even with the
        # row-band threading in raw_edit_pipeline.py, and _apply_sidecar_if_
        # needed used to re-run it on EVERY process_full_image() fetch of an
        # edited file (each zoom, revisit, repaint). Two slots (current +
        # previous file) keyed by (norm_path, adjustment fingerprint); the
        # fingerprint makes stale entries self-invalidating when the sidecar
        # changes. (This was briefly replaced with a no-op under the
        # assumption this build has no editing UI -- wrong: the Adjust panel,
        # Key_E shortcut, and XMP-saving are all still live in main.py, so
        # that no-op silently hid saved edits from the full-res view while
        # the live panel/preview tier still showed them. Restored.)
        self._adjusted_full_lock = _ADJUSTED_FULL_LOCK
        # Module-level, for the same reason the unpack stash and the in-flight
        # registry are: ImageLoadWorker is a one-shot QRunnable that builds its
        # own UnifiedImageProcessor, so an instance memo died with the worker
        # and an A-B-A revisit of an edited file never hit it -- re-paying a
        # multi-second full-res apply each time, which is the entire reason
        # this memo exists.
        self._adjusted_full_slots = _ADJUSTED_FULL_SLOTS
        # In-flight sidecar applies keyed (norm_path, adj_key): a second
        # thread asking for the same apply WAITS for the first instead of
        # recomputing -- overlapping duplicate applies of a 32MP buffer
        # measured as stacked 3.5s+7.2s entries in the same second.
        self._adjusted_inflight = _ADJUSTED_INFLIGHT
        # Async-persist gen tokens: module-level (see _ASYNC_CACHE_GEN).
        self._async_cache_lock = _ASYNC_CACHE_GEN_LOCK
        self._async_cache_gen = _ASYNC_CACHE_GEN

    def _bump_async_cache_gen(self, file_path: str) -> int:
        key = os.path.normcase(os.path.abspath(file_path or ""))
        with self._async_cache_lock:
            gen = int(self._async_cache_gen.get(key, 0)) + 1
            self._async_cache_gen[key] = gen
            self._async_cache_gen.move_to_end(key)
            while len(self._async_cache_gen) > _ASYNC_CACHE_GEN_MAX:
                self._async_cache_gen.popitem(last=False)
            return gen

    def _async_cache_gen_current(self, file_path: str) -> int:
        key = os.path.normcase(os.path.abspath(file_path or ""))
        with self._async_cache_lock:
            return int(self._async_cache_gen.get(key, 0))

    def _stash_unpacked_raw(self, file_path: str, unpacked) -> None:
        key = os.path.normcase(os.path.abspath(file_path))
        max_slots = _unpack_stash_max_slots()
        with self._unpacked_raw_lock:
            if key in self._unpacked_raw_slots:
                del self._unpacked_raw_slots[key]
            self._unpacked_raw_slots[key] = unpacked
            while len(self._unpacked_raw_slots) > max_slots:
                self._unpacked_raw_slots.popitem(last=False)

    def _peek_unpacked_raw(self, file_path: str):
        """Return stashed unpack without consuming it (Adjust can share browse stash)."""
        key = os.path.normcase(os.path.abspath(file_path))
        with self._unpacked_raw_lock:
            unpacked = self._unpacked_raw_slots.get(key)
            if unpacked is not None:
                self._unpacked_raw_slots.move_to_end(key)
            return unpacked

    def _take_unpacked_raw(self, file_path: str):
        key = os.path.normcase(os.path.abspath(file_path))
        with self._unpacked_raw_lock:
            return self._unpacked_raw_slots.pop(key, None)

    def _adjustment_fingerprint(self, adj: dict, shape) -> tuple:
        """Fingerprint for the full-res adjusted memo and its peekers.

        Includes base-buffer shape so a half-size cache entry can never be
        served for a full-res request. Every non-numeric edit is folded in
        (hashed): a hand-picked key list lagged behind channel curves / heal /
        mask layers. Serials (mask alpha as base64 PNG) are hashed, not stored.
        """
        import hashlib as _hashlib

        numeric = []
        strings = _hashlib.sha1()
        for _k in sorted(adj.keys()):
            _v = adj[_k]
            if isinstance(_v, bool):
                strings.update(f"{_k}={int(_v)}\x00".encode())
            elif isinstance(_v, (int, float)):
                numeric.append((_k, round(float(_v), 4)))
            elif isinstance(_v, str):
                if _v:
                    strings.update(f"{_k}={_v}\x00".encode())
            elif _v is not None:
                # Live objects (MaskLayerStack under _mask_layers_obj): serialized
                # twin is already in the dict; identity notices a swap.
                strings.update(f"{_k}=obj{id(_v)}\x00".encode())
        return (tuple(numeric), strings.hexdigest(), shape)

    def _peek_adjusted_full(
        self, file_path: str, adj: dict, shape
    ) -> np.ndarray | None:
        """Return a warm full-res adjusted buffer, or None. Non-blocking."""
        if adj is None or shape is None:
            return None
        norm = os.path.normcase(os.path.abspath(file_path or ""))
        adj_key = self._adjustment_fingerprint(adj, shape)
        with self._adjusted_full_lock:
            for n, k, buf in self._adjusted_full_slots:
                if n == norm and k == adj_key:
                    return buf
        return None
