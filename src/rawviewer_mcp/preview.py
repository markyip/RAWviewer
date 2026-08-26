"""In-process linear edit-base cache and small JPEG preview encode."""

from __future__ import annotations

import io
import os
from collections import OrderedDict
from typing import Any, Callable

import numpy as np

PREVIEW_LONG_EDGE = 1000  # ~800–1200 as specified

_DecodeFn = Callable[..., Any]

# Full-resolution linear buffers can run ~250-300MB each (24MP RAW). The MCP
# server is a long-lived process with no GUI memory-pressure monitor behind
# it, so this cache must bound itself: keep only the most recently used
# photos rather than growing without limit across a session.
_LINEAR_BASE_CACHE_MAX = 3


class LinearBaseCache:
    """LRU cache of decoded linear (or uint8 JPEG-base) RGB per path + mtime + flags."""

    def __init__(self, maxsize: int = _LINEAR_BASE_CACHE_MAX) -> None:
        self._store: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._maxsize = max(1, int(maxsize))

    def clear(self) -> None:
        self._store.clear()

    def get(
        self,
        path: str,
        *,
        use_full_resolution: bool = False,
        apply_lens_correction: bool = False,
        decode_fn: _DecodeFn | None = None,
    ) -> np.ndarray:
        ident = _file_ident(path)
        key = (
            os.path.normcase(os.path.abspath(path)),
            ident,
            bool(use_full_resolution),
            bool(apply_lens_correction),
        )
        cached = self._store.get(key)
        if cached is not None:
            self._store.move_to_end(key)
            return cached
        fn = decode_fn or decode_edit_base
        arr = fn(
            path,
            use_full_resolution=use_full_resolution,
            apply_lens_correction=apply_lens_correction,
        )
        if arr is None:
            raise RuntimeError(f"Failed to decode edit base: {path}")
        self._store[key] = arr
        self._store.move_to_end(key)
        while len(self._store) > self._maxsize:
            self._store.popitem(last=False)
        return arr


def _file_ident(path: str) -> tuple[int, int]:
    st = os.stat(path)
    return int(st.st_mtime_ns), int(st.st_size)


_processor = None


def _unified_processor():
    global _processor
    if _processor is None:
        from unified_image_processor import UnifiedImageProcessor

        _processor = UnifiedImageProcessor()
    return _processor


def decode_edit_base(
    path: str,
    *,
    use_full_resolution: bool = False,
    apply_lens_correction: bool = False,
) -> np.ndarray:
    """Half-res edit base via the app pipeline (no QApplication)."""
    proc = _unified_processor()
    return proc.decode_raw_edit_base(
        path,
        use_full_resolution=use_full_resolution,
        apply_lens_correction=apply_lens_correction,
    )


def render_preview_jpeg(
    base: np.ndarray,
    adj: dict,
    *,
    long_edge: int = PREVIEW_LONG_EDGE,
    quality: int = 85,
) -> tuple[bytes, int, int]:
    """Apply current edits, downscale, encode JPEG. Returns (bytes, w, h)."""
    from raw_edit_pipeline import linear_to_display_uint8, process_linear_edit_buffer

    processed = process_linear_edit_buffer(base, adj or {}, preview=True)
    rgb = linear_to_display_uint8(processed, adj)
    rgb = _downscale_uint8(rgb, long_edge)
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    from PIL import Image

    im = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=int(quality), subsampling=0)
    return buf.getvalue(), w, h


def _downscale_uint8(rgb: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if long_edge <= 0 or longest <= long_edge:
        return rgb
    scale = float(long_edge) / float(longest)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    try:
        import cv2

        return cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    except Exception:
        from PIL import Image

        im = Image.fromarray(rgb, mode="RGB")
        return np.asarray(im.resize((nw, nh), Image.Resampling.LANCZOS))
