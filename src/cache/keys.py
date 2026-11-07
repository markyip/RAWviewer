"""Cache key, path, hash and environment helper functions for RAWviewer."""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
from typing import Any

import psutil

logger = logging.getLogger(__name__)

_psutil_vm_fallback_logged = False
_darwin_total_ram_cached: int | None = None


def _is_libraw_native_name(name: str) -> bool:
    lower = name.lower()
    return lower == "raw_r.dll" or (
        lower.startswith("libraw_r") and lower.endswith((".dylib", ".so", ".dll"))
    )


def _libraw_native_search_dirs() -> list[str]:
    """Directories that may hold the packaged LibRaw shared library.

    Dev installs keep ``raw_r.dll`` beside the rawpy package. Frozen / installer
    layouts often put it next to the exe or under ``_internal/`` (PyInstaller
    ``sys._MEIPASS``). Searching only rawpy's folder made the stamp report
    ``nolib`` after packaging, so a later pin swap would not invalidate caches.
    """
    dirs: list[str] = []
    try:
        import rawpy

        rawpy_dir = os.path.dirname(os.path.abspath(rawpy.__file__))
        dirs.append(rawpy_dir)
        parent = os.path.dirname(rawpy_dir)
        if parent:
            dirs.append(parent)
    except Exception:
        pass
    try:
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        dirs.append(exe_dir)
        dirs.append(os.path.join(exe_dir, "_internal"))
        dirs.append(os.path.join(exe_dir, "_internal", "rawpy"))
    except Exception:
        pass
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        dirs.append(meipass)
        dirs.append(os.path.join(meipass, "rawpy"))
    try:
        if getattr(sys, "frozen", False) and sys.argv:
            root = os.path.dirname(os.path.abspath(sys.argv[0]))
            dirs.append(root)
            dirs.append(os.path.join(root, "_internal"))
            dirs.append(os.path.join(root, "_internal", "rawpy"))
    except Exception:
        pass
    out: list[str] = []
    seen: set[str] = set()
    for d in dirs:
        if not d:
            continue
        key = os.path.normcase(os.path.abspath(d))
        if key in seen:
            continue
        if os.path.isdir(d):
            seen.add(key)
            out.append(d)
    return out


def _libraw_native_fingerprint() -> str:
    candidates: list[str] = []
    for directory in _libraw_native_search_dirs():
        try:
            for name in os.listdir(directory):
                if _is_libraw_native_name(name):
                    candidates.append(os.path.join(directory, name))
        except Exception:
            continue
    if not candidates:
        return "nolib"
    primary = None
    for path in candidates:
        base = os.path.basename(path).lower()
        if base == "raw_r.dll" or base.startswith("libraw_r.") or base == "libraw_r.dylib":
            primary = path
            break
    path = primary or sorted(candidates)[0]
    st = os.stat(path)
    return f"{os.path.basename(path)}:{st.st_size}:{int(st.st_mtime)}"


def libraw_runtime_stamp() -> str:
    """Fingerprint the LibRaw binary backing rawpy (version + native lib file).

    Packaging may swap ``raw_r.dll`` / ``libraw_r*.dylib`` while leaving
    ``rawpy.libraw_version`` unchanged (master still reports 0.22.0). Size and
    mtime of the bundled native library catch that swap so demosaic caches
    invalidate. Returns "" if rawpy is unavailable.
    """
    try:
        import rawpy
    except Exception:
        return ""
    try:
        ver = getattr(rawpy, "libraw_version", None) or ()
        ver_s = ".".join(str(int(x)) for x in ver[:3]) if ver else "unknown"
    except Exception:
        ver_s = "unknown"
    try:
        lib_s = _libraw_native_fingerprint()
    except Exception:
        lib_s = "nolib"
    return f"{ver_s}|{lib_s}"


def _encode_exif_blob(data: dict[str, Any] | None) -> bytes:
    """Serialize EXIF blob as UTF-8 JSON (safe; no pickle on write)."""
    payload = data if isinstance(data, dict) else {}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _decode_exif_blob(blob: Any) -> dict[str, Any]:
    """Deserialize EXIF blob as JSON.

    Legacy pickle rows are intentionally NOT decoded: unpickling data from an
    on-disk cache file would be arbitrary code execution for anyone with
    write access to the cache directory. Non-JSON rows are treated as a cache
    miss (empty dict); the caller re-extracts and rewrites the row as JSON.
    """
    if not blob:
        return {}
    try:
        if isinstance(blob, str):
            out = json.loads(blob)
            return out if isinstance(out, dict) else {}
        raw = bytes(blob)
        if raw[:1] in (b"{", b"["):
            out = json.loads(raw.decode("utf-8"))
            return out if isinstance(out, dict) else {}
    except Exception:
        pass
    return {}


def _safe_path_under_root(path: str | None, root: str | None) -> str | None:
    """Return realpath of ``path`` only if it resolves under ``root``."""
    if not path or not root:
        return None
    try:
        root_real = os.path.realpath(root)
        candidate = os.path.realpath(path)
    except OSError:
        return None
    if candidate == root_real or candidate.startswith(root_real + os.sep):
        return candidate
    return None


def _darwin_os_proc_available_memory() -> int | None:
    """Jetsam headroom via os_proc_available_memory (no subprocess)."""
    if platform.system() != "Darwin":
        return None
    try:
        import ctypes

        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        lib.os_proc_available_memory.restype = ctypes.c_uint64
        lib.os_proc_available_memory.argtypes = []
        n = int(lib.os_proc_available_memory())
        return n if n > 0 else None
    except Exception:
        return None


def _sysctl_hw_memsize_bytes() -> int | None:
    """Physical RAM via sysctlbyname (cached); subprocess only if ctypes fails."""
    global _darwin_total_ram_cached
    if platform.system() != "Darwin":
        return None
    if _darwin_total_ram_cached:
        return _darwin_total_ram_cached
    try:
        import ctypes

        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        val = ctypes.c_uint64(0)
        size = ctypes.c_size_t(ctypes.sizeof(val))
        lib.sysctlbyname.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        lib.sysctlbyname.restype = ctypes.c_int
        rc = lib.sysctlbyname(
            b"hw.memsize", ctypes.byref(val), ctypes.byref(size), None, 0
        )
        if rc == 0 and int(val.value) > 0:
            _darwin_total_ram_cached = int(val.value)
            return _darwin_total_ram_cached
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"],
            text=True,
            timeout=2,
        ).strip()
        n = int(out)
        if n > 0:
            _darwin_total_ram_cached = n
            return n
    except Exception:
        return None
    return None


def _darwin_page_size_bytes() -> int:
    if platform.system() != "Darwin":
        return 4096
    try:
        out = subprocess.check_output(["pagesize"], text=True, timeout=2).strip()
        return max(4096, int(out))
    except Exception:
        return 4096


def _parse_vm_stat_page_count(vm_stat_output: str, label: str) -> int:
    prefix = f"{label}:"
    for line in vm_stat_output.splitlines():
        if not line.startswith(prefix):
            continue
        raw = line.split(":", 1)[1].strip().rstrip(".")
        try:
            return int(raw.replace(",", ""))
        except ValueError:
            return 0
    return 0


def _darwin_memory_stats_from_vm_stat() -> tuple[int, int, float] | None:
    """Installed RAM + available bytes. Prefer os_proc_available_memory (jetsam).

    Never shells out to the ``memory_pressure`` CLI. ``vm_stat`` is a last
    resort when jetsam headroom is unavailable.
    """
    if platform.system() != "Darwin":
        return None
    total = _sysctl_hw_memsize_bytes()
    if not total:
        return None
    jetsam = _darwin_os_proc_available_memory()
    if jetsam is not None:
        available = min(int(total), int(jetsam))
        used = max(0, int(total) - available)
        percent = min(100.0, max(0.0, (used / float(total)) * 100.0))
        return int(total), available, percent
    try:
        vm_out = subprocess.check_output(["vm_stat"], text=True, timeout=2)
    except Exception:
        return None

    page_size = _darwin_page_size_bytes()
    free_pages = _parse_vm_stat_page_count(vm_out, "Pages free")
    inactive_pages = _parse_vm_stat_page_count(vm_out, "Pages inactive")
    speculative_pages = _parse_vm_stat_page_count(vm_out, "Pages speculative")
    available_pages = free_pages + inactive_pages + speculative_pages
    available = min(total, available_pages * page_size)
    used = max(0, total - available)
    percent = min(100.0, max(0.0, (used / total) * 100.0))
    return total, available, percent


def system_memory_total_bytes() -> int | None:
    """Total installed RAM in bytes (sysctl on macOS, then psutil)."""
    if platform.system() == "Darwin":
        total = _sysctl_hw_memsize_bytes()
        if total:
            return int(total)
    try:
        return int(psutil.virtual_memory().total)
    except Exception:
        return _sysctl_hw_memsize_bytes()


def _safe_virtual_memory() -> Any:
    """Memory stats with macOS sysctl/vm_stat first (avoids psutil HOST_VM_INFO64 issues)."""
    global _psutil_vm_fallback_logged
    if platform.system() == "Darwin":
        fallback = _darwin_memory_stats_from_vm_stat()
        if fallback is not None:
            from types import SimpleNamespace

            total, available, percent = fallback
            return SimpleNamespace(total=total, available=available, percent=percent)

    try:
        return psutil.virtual_memory()
    except Exception as exc:
        if not _psutil_vm_fallback_logged:
            logger.debug(
                "psutil.virtual_memory() unavailable (%s); using fallback memory stats",
                exc,
            )
            _psutil_vm_fallback_logged = True
        fallback = _darwin_memory_stats_from_vm_stat()
        from types import SimpleNamespace

        if fallback is not None:
            total, available, percent = fallback
            return SimpleNamespace(total=total, available=available, percent=percent)

        total = _sysctl_hw_memsize_bytes() or (8 * 1024**3)
        available = total // 4
        return SimpleNamespace(
            total=total,
            available=available,
            percent=min(100.0, max(0.0, ((total - available) / total) * 100.0)),
        )


def disk_preview_max_edge() -> int:
    """Max long edge for JPEG files under ~/.rawviewer_cache/previews (grid tier is separate)."""
    raw = os.environ.get("RAWVIEWER_DISK_PREVIEW_MAX", "512").strip()
    try:
        v = int(raw)
    except Exception:
        v = 512
    return max(256, min(v, 1024))


def memory_preview_max_edge() -> int:
    """Max in-memory preview for progressive single-image RAW display (not disk size)."""
    raw = os.environ.get("RAWVIEWER_MEMORY_PREVIEW_MAX", "1920").strip()
    try:
        v = int(raw)
    except Exception:
        v = 1920
    return max(512, min(v, 4096))


def _jpeg_bytes_max_edge(jpeg_data: bytes, max_edge: int) -> bytes:
    """Downscale JPEG bytes so disk cache entries stay thumbnail-sized."""
    if not jpeg_data or max_edge <= 0:
        return jpeg_data
    try:
        import io

        from PIL import Image, ImageOps

        pil_image = Image.open(io.BytesIO(jpeg_data))
        pil_image = ImageOps.exif_transpose(pil_image)
        w, h = pil_image.size
        if max(w, h) <= max_edge:
            # We must re-save to strip EXIF orientation and ensure it stays rotated.
            out = io.BytesIO()
            pil_image.save(out, format="JPEG", quality=85)
            return out.getvalue()
        scale = max_edge / float(max(w, h))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = pil_image.resize((new_w, new_h), Image.Resampling.HAMMING)
        if resized.mode != "RGB":
            resized = resized.convert("RGB")
        out = io.BytesIO()
        resized.save(out, format="JPEG", quality=85)
        return out.getvalue()
    except Exception:
        return jpeg_data


def _exif_cache_path_key(file_path: str) -> str:
    """Normalized path for exif_cache row keys and lookups (case- and slash-tolerant on Windows)."""
    if not file_path:
        return ""
    try:
        p = file_path if os.path.isabs(file_path) else os.path.abspath(file_path)
        key = os.path.normcase(os.path.normpath(p))
    except OSError:
        key = os.path.normcase(os.path.normpath(file_path))
    if os.sep == "\\":
        key = key.replace("/", "\\")
    return key


def _cache_path_key(file_path: str) -> str:
    """Normalized path key for thumbnail/grid/preview caches (matches semantic canonical paths on Windows)."""
    return _exif_cache_path_key(file_path)


def _exif_sql_path_expr(column: str = "file_path") -> str:
    """SQL expression that normalizes slashes for path comparisons."""
    if os.sep == "\\":
        return f"lower(replace({column}, '/', '\\\\'))"
    return f"lower(replace({column}, '\\\\', '/'))"


def _exif_sql_folder_like_pattern(folder_path: str) -> str:
    """LIKE prefix for all files under folder_path (matches stored canonical keys)."""
    root = _exif_cache_path_key(os.path.abspath(folder_path)).rstrip("\\/")
    if os.sep == "\\":
        return root + "\\%"
    return root.replace("\\", "/") + "/%"


def _exif_sql_folder_like_patterns(folder_path: str) -> list[str]:
    """Candidate LIKE prefixes (drive letter vs UNC realpath may differ in the DB)."""
    patterns: list[str] = []
    try:
        primary = _exif_sql_folder_like_pattern(folder_path)
        if primary:
            patterns.append(primary)
        try:
            real_root = os.path.realpath(os.path.abspath(folder_path))
        except OSError:
            real_root = None
        if real_root:
            alt = _exif_sql_folder_like_pattern(real_root)
            if alt and alt not in patterns:
                patterns.append(alt)
    except OSError:
        pass
    return patterns


def _mtime_matches(cached_mtime: float, current_mtime: float) -> bool:
    """Allow sub-second drift (network shares / FAT timestamp granularity)."""
    try:
        return abs(float(cached_mtime) - float(current_mtime)) < 1.0
    except (TypeError, ValueError):
        return cached_mtime == current_mtime


def _capture_time_from_exif_row(row: Any, exif_data: dict[str, Any] | None = None) -> str | None:
    ct = row[7] if len(row) > 7 else None
    if ct and str(ct).strip():
        return str(ct)
    if exif_data and exif_data.get("capture_time"):
        return str(exif_data["capture_time"])
    return None


def _file_stats_row(
    file_stats: dict[str, tuple[int, float]] | None, *paths: str
) -> tuple[int, float] | None:
    if not file_stats:
        return None
    for path in paths:
        if not path:
            continue
        if path in file_stats:
            return file_stats[path]
        nk = _exif_cache_path_key(path)
        for key, row in file_stats.items():
            if _exif_cache_path_key(key) == nk:
                return row
    return None


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, "").strip()
    if raw:
        try:
            return max(minimum, min(int(raw), maximum))
        except ValueError:
            pass
    return max(minimum, min(default, maximum))


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name, "").strip()
    if raw:
        try:
            return max(minimum, min(float(raw), maximum))
        except ValueError:
            pass
    return max(minimum, min(default, maximum))


def _preview_cache_adaptive_enabled() -> bool:
    raw = os.environ.get("RAWVIEWER_PREVIEW_CACHE_ADAPTIVE", "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


def recommended_preview_cache_items(available_gb: float | None = None) -> int:
    """Preview LRU capacity from free RAM, preview resolution, and build profile."""
    explicit = os.environ.get("RAWVIEWER_PREVIEW_CACHE_ITEMS", "").strip()
    if explicit:
        return _env_int("RAWVIEWER_PREVIEW_CACHE_ITEMS", 10, minimum=6, maximum=128)

    if not _preview_cache_adaptive_enabled():
        return 10

    if available_gb is None:
        try:
            available_gb = float(_safe_virtual_memory().available) / (1024**3)
        except Exception:
            return 10

    edge = memory_preview_max_edge()
    preview_mb = max(6.0, (edge / 1920.0) ** 2 * 12.0)

    budget_fraction = 0.12  # Increased from 0.08 for more cache on modern systems
    budget_cap_mb = 2400.0  # Increased from 1200
    try:
        from core.profile import indexing_loads_compete, is_lite_build

        if not indexing_loads_compete():
            budget_fraction = 0.18
            budget_cap_mb = 3200.0
        if is_lite_build():
            budget_fraction = max(budget_fraction, 0.20)
            budget_cap_mb = 3600.0
    except Exception:
        pass

    budget_mb = min(float(available_gb) * 1024.0 * budget_fraction, budget_cap_mb)
    items = int(budget_mb / preview_mb)

    if available_gb >= 32:
        floor = 32
    elif available_gb >= 24:
        floor = 24
    elif available_gb >= 16:
        floor = 20
    elif available_gb >= 10:
        floor = 16
    elif available_gb >= 6:
        floor = 14
    elif available_gb >= 3:
        floor = 10
    else:
        floor = 8

    min_items = _env_int("RAWVIEWER_PREVIEW_CACHE_ITEMS_MIN", 6, minimum=4, maximum=32)
    max_items = _env_int("RAWVIEWER_PREVIEW_CACHE_ITEMS_MAX", 64, minimum=8, maximum=128)
    return max(min_items, min(max(floor, items), max_items))
