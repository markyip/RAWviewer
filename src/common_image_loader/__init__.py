"""Unified common image loader: helpers, loaders, EXIF, mipmap, runtime, EDR."""

from . import edr, exif_display, helpers, loaders, mipmap, runtime

_SUBMODULES = (helpers, loaders, exif_display, mipmap, runtime, edr)


def __getattr__(name: str):
    """Live module-attribute fallback (PEP 562).

    The facade resolves every loader symbol through the submodules so that
    module-level state which is rebound in place (EDR flag, caches) is always
    read through the owning submodule.
    """
    for mod in _SUBMODULES:
        obj = getattr(mod, name, None)
        if obj is not None:
            return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    names = set(globals())
    for mod in _SUBMODULES:
        names.update(n for n in dir(mod) if not n.startswith("_"))
    return sorted(names)


__all__ = [
    "FULL_EMBEDDED_SENSOR_COVERAGE",
    "apply_container_orientation_to_array",
    "apply_container_orientation_to_pixmap",
    "array_matches_exif_display",
    "check_cache_for_image",
    "check_memory_cache_for_image",
    "decode_embedded_jpeg_bytes",
    "decode_raw_edr_pixmap",
    "dng_prefers_embedded_preview_first",
    "embedded_preview_rarely_covers_sensor",
    "encode_tile_bytes",
    "exif_display_aspect_ratio",
    "exif_display_dimensions",
    "finalize_index_thumbnail_array",
    "folder_sort_key_tuple",
    "format_edr_status_label",
    "get_image_aspect_ratio",
    "hlg_to_linear",
    "image_covers_sensor_resolution",
    "index_thumbnail_needs_orient_fix",
    "invalidate_libraw_consistent_preview_settings",
    "invalidate_volume_speed_cache",
    "io_pressure_active",
    "is_emfile_error",
    "is_external_or_network_volume",
    "is_macos_edr_enabled",
    "is_raw_file",
    "is_slow_storage_path",
    "is_tiff_file",
    "load_pixmap_safe",
    "load_qimage_safe",
    "load_raw_preview_array",
    "load_raw_preview_pixmap",
    "linear_edr_float_to_qpixmap",
    "linear_to_srgb",
    "mark_exif_pixels_display_oriented",
    "moderate_external_cap_enabled",
    "note_emfile_pressure",
    "normalize_capture_time_string",
    "orientation_from_embedded_jpeg_bytes",
    "parse_capture_time_to_timestamp",
    "pil_downscale_resample",
    "pixmap_covers_requested_edge",
    "pixmap_matches_exif_display",
    "pq_to_linear",
    "probe_capture_timestamp_from_file",
    "probe_volume_speed",
    "process_pool_mp_context",
    "process_pool_python_executable",
    "process_pool_worker_count",
    "raw_concurrent_load_limit",
    "raw_load_limit_aligned_to_gpu",
    "record_volume_speed",
    "resolve_container_exif_for_file",
    "resolve_folder_sort_timestamp",
    "rgb_array_to_qpixmap",
    "sensor_pixel_dimensions",
    "sort_probe_worker_count",
    "summarize_macos_edr_startup",
    "take_last_load_edr_flag",
    "tone_map_reinhard",
    "try_load_hdr_image_pixmap",
    "use_full_embedded_raw_preview",
    "use_libraw_consistent_preview_first",
    "use_progressive_raw_loading",
    "use_raw_edr_display",
    "use_raw_edr_for_display",
    "use_raw_process_pool",
    "volume_speed_tier",
]
