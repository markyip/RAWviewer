#!/usr/bin/env python3
"""LibRaw runtime stamp invalidates demosaic pixel caches.

Packaging may swap raw_r.dll / libraw_r*.dylib while rawpy.libraw_version
stays at 0.22.x. The stamp must include the native library fingerprint so
persisted thumb/grid/preview tiers are purged on upgrade.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from image_cache import ImageCache, _libraw_native_search_dirs, libraw_runtime_stamp
    import unified_image_processor as uip

    stamp = libraw_runtime_stamp()
    check("libraw_runtime_stamp is non-empty when rawpy imports", bool(stamp))
    check("stamp contains a version segment", "|" in stamp and stamp.split("|")[0] != "")

    check(
        "release_edit_base_cache is available",
        callable(getattr(uip, "release_edit_base_cache", None)),
    )

    # Marker hygiene: a changed stamp clears disk tiers once.
    tmp = tempfile.mkdtemp()
    try:
        cache = ImageCache(tmp, persistent_cache_enabled=True)
        marker = os.path.join(tmp, "libraw_pixel_ver.txt")
        check("libraw stamp marker written on init", os.path.isfile(marker))
        with open(marker, "r", encoding="utf-8") as fh:
            written = fh.read().strip()
        check("marker matches current stamp", written == stamp)

        # Simulate an older LibRaw install.
        with open(marker, "w", encoding="utf-8") as fh:
            fh.write("0.0.0|oldlib")
        # Close SQLite handles before re-init / rmtree (Windows file locks).
        cache.close()
        del cache
        cache2 = ImageCache(tmp, persistent_cache_enabled=True)
        with open(marker, "r", encoding="utf-8") as fh:
            written2 = fh.read().strip()
        check("stale stamp is rewritten to current", written2 == stamp)
        cache2.close()
        del cache2
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)

    # Source-level: ImageCache wires the purge on persistent init.
    import inspect

    init_src = inspect.getsource(ImageCache.__init__)
    check(
        "ImageCache init calls _purge_stale_libraw_pixel_caches",
        "_purge_stale_libraw_pixel_caches" in init_src,
    )

    purge_src = inspect.getsource(ImageCache._purge_stale_libraw_pixel_caches)
    check(
        "stamp change clears the LibRaw unsupported set",
        "prune_libraw_unsupported_paths" in purge_src and "clear_all=True" in purge_src,
    )
    check(
        "stamp change clears libraw_preview_paths",
        "libraw_preview_paths.clear" in purge_src,
    )

    helper_src = inspect.getsource(_libraw_native_search_dirs)
    check(
        "stamp search includes packaged _internal / MEIPASS layouts",
        "_internal" in helper_src and "_MEIPASS" in helper_src,
        "PyInstaller / installer DLL is not only beside rawpy",
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
