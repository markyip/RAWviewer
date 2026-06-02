"""
Windows Shell property helpers — **POC / not used in production**.

POC on a 6889-file folder (Japan Trip): Shell DateTaken was ~2.4× slower than
metadata_backend probe with no ≥1s timestamp differences vs EXIF; sort order
only diverged on sub-second burst shots. Folder sort stays on EXIF cache/probe
(see common_image_loader.resolve_folder_sort_timestamp).

Uses IPropertyStore (System.Photo.DateTaken) for dev comparisons only —
scripts/compare_shell_capture_times.py
"""

from __future__ import annotations

import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_SHELL_DATE_CACHE: Dict[str, float] = {}


def shell_sort_timestamps_enabled() -> bool:
    if sys.platform != "win32":
        return False
    flag = os.environ.get("RAWVIEWER_USE_SHELL_SORT_DATES", "1").strip().lower()
    return flag not in ("0", "false", "no", "off")


def _filetime_to_timestamp(ft_low: int, ft_high: int) -> float:
    if not ft_low and not ft_high:
        return 0.0
    # FILETIME is 100-ns intervals since 1601-01-01 UTC
    ticks = (int(ft_high) << 32) + int(ft_low)
    if ticks <= 0:
        return 0.0
    return ticks / 10_000_000.0 - 11644473600.0


def _propvariant_to_timestamp(prop) -> float:
    try:
        # propsys returns PyPROPVARIANT; GetValue() yields aware datetime for DateTaken.
        val = prop.GetValue()
        if hasattr(val, "timestamp"):
            return float(val.timestamp())
    except Exception:
        pass
    try:
        import pywintypes  # type: ignore
        import pythoncom  # type: ignore

        if prop.vt == pythoncom.VT_FILETIME:
            dt = pywintypes.Time(prop)
            return float(dt.timestamp())
        if prop.vt in (pythoncom.VT_DATE,):
            raw = prop.GetValue()
            if hasattr(raw, "timestamp"):
                return float(raw.timestamp())
    except Exception:
        pass
    try:
        text = str(prop).strip()
        if not text:
            return 0.0
        from datetime import datetime

        for fmt in ("%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(text[:19], fmt).timestamp()
            except ValueError:
                continue
    except Exception:
        pass
    return 0.0


def shell_date_taken_timestamp(file_path: str) -> float:
    """Read Explorer-aligned capture/modified time for one file (0 if unavailable)."""
    if not shell_sort_timestamps_enabled():
        return 0.0
    norm = os.path.normcase(os.path.abspath(file_path))
    cached = _SHELL_DATE_CACHE.get(norm)
    if cached is not None:
        return cached

    ts = 0.0
    try:
        import pythoncom  # type: ignore
        from win32com.propsys import propsys, pscon  # type: ignore

        pythoncom.CoInitialize()
        try:
            store = propsys.SHGetPropertyStoreFromParsingName(norm)
            for key in (
                getattr(pscon, "PKEY_Photo_DateTaken", None),
                getattr(pscon, "PKEY_Media_DateTaken", None),
            ):
                if key is None:
                    continue
                try:
                    prop = store.GetValue(key)
                    ts = _propvariant_to_timestamp(prop)
                    if ts > 0:
                        break
                except Exception:
                    continue
        finally:
            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass
    except Exception as exc:
        logger.debug("[SHELL] Property read failed for %s: %s", os.path.basename(norm), exc)

    if ts <= 0:
        ts = 0.0

    _SHELL_DATE_CACHE[norm] = ts
    return ts


def bulk_shell_date_taken_timestamps(
    file_paths,
    *,
    max_workers: Optional[int] = None,
) -> Dict[str, float]:
    """Batch-read shell dates for sorting (keys are normalized absolute paths)."""
    if not shell_sort_timestamps_enabled() or not file_paths:
        return {}

    paths = [os.path.normcase(os.path.abspath(p)) for p in file_paths if p]
    out: Dict[str, float] = {}
    missing = [p for p in paths if p not in _SHELL_DATE_CACHE]
    for p in paths:
        if p in _SHELL_DATE_CACHE:
            out[p] = _SHELL_DATE_CACHE[p]

    if not missing:
        return out

    workers = max_workers
    if workers is None:
        workers = min(8, max(2, (os.cpu_count() or 4)))

    def _one(p: str) -> tuple[str, float]:
        return p, shell_date_taken_timestamp(p)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, p) for p in missing]
        for fut in as_completed(futures):
            try:
                p, ts = fut.result()
                if ts > 0:
                    out[p] = ts
            except Exception:
                continue
    return out


def clear_shell_date_cache() -> None:
    _SHELL_DATE_CACHE.clear()
