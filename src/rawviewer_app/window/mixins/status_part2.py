"""status mixin, part 2 of 2 (1 methods, 695 lines)."""
from __future__ import annotations

import os
import sys

from core import theme

from .. import lazy
from ..lazy import is_raw_file, process_file_from_path
from ..paths import _require_str_path
from ..qt_helpers import _qta_icon

lazy.register(sys.modules[__name__])


class StatusPart2:
    def _update_status_bar_now(self, width=None, height=None):
        """Update status bar with comprehensive information including EXIF data

        Args:
            width: Image width (optional)
            height: Image height (optional)
            exif_data: Direct EXIF data dict to use instead of cache (optional)
        """
        if not hasattr(self, 'status_metadata_label') or not hasattr(self, 'status_counter_label'):
            # Fallback to old method if UI components not initialized
            if not self.current_file_path:
                self.status_bar.showMessage("")  # Empty message when no image loaded
                return
            self.status_bar.showMessage("")
            return

        if self.view_mode == "compare":
            self._update_compare_status_bar()
            self._sync_compare_button_visibility()
            self._sync_share_button_visibility()
            if hasattr(self, "status_counter_label"):
                self._refresh_image_counter()
            return

        if hasattr(self, "shortcuts_hint_button"):
            self.shortcuts_hint_button.setToolTip(self._keyboard_shortcuts_tooltip())

        # Show metadata, counter, and sort button when there are files.
        if self.view_mode != "single":
            self._apply_top_metadata_text(self._gallery_top_metadata_label())
        # Show/hide sort button based on view mode
        self._sync_burst_group_bottom_chrome()
        self._sync_compare_button_visibility()
        if hasattr(self, 'size_slider'):
            if self.view_mode == 'gallery':
                self.size_slider.setVisible(True)
            else:
                self.size_slider.setVisible(False)

        # Gallery toggle in single mode; burst group view shows it as back-to-main-gallery.
        if hasattr(self, 'view_mode_button'):
            burst_back = (
                getattr(self, "_burst_group_view_active", False)
                and self.view_mode == "gallery"
            )
            self.view_mode_button.setVisible(
                burst_back
                or (
                    bool(self.image_files)
                    and self.view_mode == "single"
                    and self._is_exif_sort_ready()
                )
            )
        if hasattr(self, "share_bottom_button"):
            self._sync_share_button_visibility()
            self._sync_slideshow_button_visibility()
            cp = getattr(self, "current_file_path", None)
            show_rotate = bool(
                self.image_files
                and self.view_mode == "single"
                and cp
                and os.path.isfile(cp)
                and not self._adjust_panel_active()
            )
            if hasattr(self, "rotate_bottom_button"):
                self.rotate_bottom_button.setVisible(show_rotate)
            if hasattr(self, "adjust_bottom_button"):
                from raw_adjustments import editing_features_enabled

                # Unlike rotate, stays visible while the panel is open -- it is
                # the panel's on-screen toggle (EMBER icon = active).
                show_adjust = bool(
                    self.image_files
                    and self.view_mode == "single"
                    and cp
                    and os.path.isfile(cp)
                    and editing_features_enabled()
                )
                self.adjust_bottom_button.setVisible(show_adjust)
                if show_adjust:
                    adjust_active = self._adjust_panel_active()
                    self.adjust_bottom_button.setIcon(
                        _qta_icon(
                            "fa5s.sliders-h",
                            color=theme.EMBER if adjust_active else theme.INK_MUTED,
                        )
                    )
                    self.adjust_bottom_button.setToolTip(
                        "Close Adjust (E)"
                        if adjust_active
                        else "Adjust -- develop this photo (E)"
                    )
            if hasattr(self, "raw_toggle_button"):
                is_raw = bool(show_rotate and is_raw_file(cp))
                self.raw_toggle_button.setVisible(is_raw)
                if is_raw:
                    self._update_raw_toggle_button_state()
                if hasattr(self, "edr_toggle_button"):
                    # EDR only applies in the RAW (LibRaw) workflow; hide the
                    # toggle when browsing embedded JPEG previews.
                    use_embedded = self.get_settings().value(
                        "use_embedded_jpeg_workflow", True, type=bool
                    )
                    show_edr = bool(is_raw and not use_embedded)
                    self.edr_toggle_button.setVisible(show_edr)
                    if show_edr:
                        self._update_edr_toggle_button_state()
        self._sync_bookmark_indicator()
        if hasattr(self, "search_bottom_button") and not getattr(
            self, "_burst_group_view_active", False
        ):
            self.search_bottom_button.setVisible(
                bool(self.image_files)
                and self._is_exif_sort_ready()
                and not self._adjust_panel_active()
            )
        if getattr(self, "_search_panel_expanded", False):
            self._schedule_search_expand_overlay_sync()

        # Gallery mode should stay lightweight: avoid expensive per-image EXIF/status
        # recomputation (fallback reads can block UI and delay gallery paint).
        if self.view_mode != 'single':
            self._apply_top_metadata_text(self._gallery_top_metadata_label())
            if hasattr(self, 'status_counter_label'):
                self._refresh_image_counter()
            self._sync_burst_group_bottom_chrome()
            return

        if not self.current_file_path:
            # Clear top metadata when no image is loaded
            if self.view_mode == 'single':
                self._apply_top_metadata_text("")
            else:
                self._apply_top_metadata_text("")
            if hasattr(self, 'status_counter_label'):
                self.status_counter_label.setText("")
            return

        # Get filename (folder + name shown only in the centered EXIF HUD)
        file_label = self._metadata_file_label(self.current_file_path)

        # Get image dimensions - ALWAYS prefer original dimensions from cache
        # This ensures we show the original resolution even when displaying a thumbnail
        original_width = None
        original_height = None
        display_width = None
        display_height = None

        # Get original dimensions from cache (authoritative source)
        cached_exif = self.image_cache.get_exif_for_ui(self.current_file_path)
        if cached_exif and cached_exif.get("minimal_preview_exif"):
            self._apply_preview_first_status_bar(cached_exif)
            if hasattr(self, "status_counter_label"):
                self._refresh_image_counter()
            return
        if cached_exif:
            # CRITICAL: Always check for original_width and original_height in cache
            original_width = cached_exif.get('original_width')
            original_height = cached_exif.get('original_height')

            # Sanity check: if RAW but dimensions are suspiciously small (<=1920),
            # they are likely from a preview. Force re-extraction.
            raw_map = {'.arw', '.cr2', '.cr3', '.nef', '.dng', '.raf', '.orf', '.rw2'}
            is_raw_preview = os.path.splitext(_require_str_path(self.current_file_path))[1].lower() in raw_map

            if is_raw_preview and original_width and original_height and max(original_width, original_height) <= 1920:
                import logging

                logging.getLogger(__name__).info(
                    f"[STATUS] Suspiciously small dimensions ({original_width}x{original_height}) for RAW file, forcing re-extraction"
                )
                original_width = None
                original_height = None

        if not original_width or not original_height:
            raw_map = {'.arw', '.cr2', '.cr3', '.nef', '.dng', '.raf', '.orf', '.rw2'}
            is_raw = os.path.splitext(_require_str_path(self.current_file_path))[1].lower() in raw_map

            if is_raw:
                # Resolve true sensor dimensions OFF the main thread; the HUD refreshes
                # via _on_sensor_dims_resolved once ready. (rawpy.imread on an external
                # drive used to block the UI on every navigation.)
                self._request_sensor_dimensions_async(self.current_file_path, cached_exif)
            else:
                # Non-RAW: lightweight EXIF header read is cheap enough inline.
                try:
                    tags = process_file_from_path(
                        self.current_file_path, details=False
                    )
                    for w_tag in (
                        "EXIF ExifImageWidth",
                        "Image ImageWidth",
                        "Image Width",
                    ):
                        if w_tag in tags:
                            original_width = int(str(tags[w_tag]))
                            break
                    for h_tag in (
                        "EXIF ExifImageLength",
                        "Image ImageLength",
                        "Image Height",
                        "Image Length",
                        "EXIF ExifImageHeight",
                    ):
                        if h_tag in tags:
                            original_height = int(str(tags[h_tag]))
                            break

                    if original_width and original_height and cached_exif:
                        cached_exif["original_width"] = original_width
                        cached_exif["original_height"] = original_height
                        self.image_cache.put_exif(
                            self.current_file_path, cached_exif
                        )
                except Exception:
                    pass
        # Get current display dimensions (from pixmap)
        if self.current_pixmap:
            display_width = self.current_pixmap.width()
            display_height = self.current_pixmap.height()

        if is_raw_file(self.current_file_path) and original_width and original_height and display_width and display_height:
            om = max(int(original_width), int(original_height))
            dm = max(int(display_width), int(display_height))
            if getattr(self, '_is_half_size_displayed', False) and dm > 0 and abs(om - dm) <= max(2.0, dm * 0.015):
                import logging

                logging.getLogger(__name__).info(
                    f"[STATUS] Cached dimensions match preview pixmap ({original_width}x{original_height}); using display dims"
                )
                original_width = original_height = None

        # Use provided dimensions if available, otherwise use original dimensions
        # CRITICAL: Always prioritize original_width/original_height over display dimensions
        if width is None or height is None:
            if original_width and original_height:
                width = original_width
                height = original_height
            elif display_width and display_height:
                width = display_width
                height = display_height
            else:
                width = height = 0

        # RAW: cache often holds embedded‑preview WxH until EXIFExtractor fills sensor size -- kick async refresh early.
        _fp_sb = getattr(self, "current_file_path", None)
        if _fp_sb and is_raw_file(_fp_sb):
            _dm_sb = max(display_width or 0, display_height or 0)
            _ow_sb = original_width if original_width else 0
            _oh_sb = original_height if original_height else 0
            _om_sb = max(_ow_sb, _oh_sb) if (_ow_sb and _oh_sb) else 0
            _needs_sensor_exif = False
            if not original_width or not original_height or (
                getattr(self, "_is_half_size_displayed", False)
                and _dm_sb > 0
                and _om_sb > 0
                and _om_sb <= int(_dm_sb * 1.02 + 0.5)
            ):
                _needs_sensor_exif = True
            elif (
                _dm_sb > 0
                and _om_sb > 0
                and _dm_sb > int(_om_sb * 1.08 + 0.5)
            ):
                # Decoded pixmap is visibly larger than cached "original" (stale embedded preview dims).
                _needs_sensor_exif = True
            if _needs_sensor_exif:
                self._schedule_raw_sensor_exif_status_refresh()

        # Determine if we're displaying a thumbnail (for status bar indication)
        if original_width and original_height and display_width and display_height:
            # Check if display size is significantly smaller than original
            original_max = max(original_width, original_height)
            display_max = max(display_width, display_height)
            # Consider it a thumbnail if display is less than 80% of original
            if display_max < original_max * 0.8:
                pass

        # Get zoom level
        if self._raw_recovery_preview_active():
            zoom_level = "Recovery preview (fit only)"
        else:
            gv = getattr(self, "gpu_view", None)
            if gv is not None and gv.has_pixmap():
                zoom_level = "Fit" if gv.is_fit_mode() else f"{int(round(gv.current_scale() * 100))}%"
            elif self.fit_to_window:
                zoom_level = "Fit"
            else:
                zoom_level = f"{int(self.current_zoom_level * 100)}%"

        # --- Status bar dedup / debounce -----------------------------------
        # update_status_bar() is called many times per image (thumbnail ready,
        # full image ready, display_pixmap, minimal+full EXIF ready, etc.).
        # Re-parsing EXIF and re-applying identical HUD text on every call wastes
        # main-thread cycles and floods the log with duplicate [STATUS]/[TRACK]
        # lines. When nothing user-visible changed, re-apply the cached text
        # silently and skip the heavy EXIF parse + logging.
        if self.view_mode == 'single':
            _dedup_exif = self.image_cache.get_exif_for_ui(self.current_file_path)
            _dedup_exif_len = 0
            if _dedup_exif:
                _ed = _dedup_exif.get('exif_data')
                if isinstance(_ed, dict):
                    _dedup_exif_len = len(_ed)
            status_signature = (
                self.current_file_path,
                zoom_level,
                original_width, original_height,
                width, height,
                display_width, display_height,
                bool(getattr(self, '_is_half_size_displayed', False)),
                bool(self._raw_recovery_preview_active()),
                getattr(self, "_display_edr_mode", None),
                _dedup_exif_len,
            )
            if (
                getattr(self, '_last_status_signature', None) == status_signature
                and getattr(self, '_last_status_metadata_text', None) is not None
            ):
                self._apply_top_metadata_text(self._last_status_metadata_text)
                if hasattr(self, "status_counter_label"):
                    self._refresh_image_counter()
                return
        else:
            status_signature = None

        # Try to get EXIF data from cache first (faster) - matching reference version
        exif_info = []
        cached_exif = self.image_cache.get_exif_for_ui(self.current_file_path)
        import logging
        import time
        logger = logging.getLogger(__name__)
        exif_tags = None
        metadata_extracted_from_cache = False
        # Avoid repeatedly running expensive fallback EXIF probes in UI thread
        # when the file has sparse/unsupported EXIF fields.
        now_ts = time.time()
        _probe_ts = getattr(self, "_status_exif_probe_ts", {})
        last_probe_ts = float(_probe_ts.get(self.current_file_path, 0.0) or 0.0)
        allow_expensive_exif_probe = (now_ts - last_probe_ts) >= 8.0

        if cached_exif:
            exif_data_dict = cached_exif.get('exif_data')
            if exif_data_dict and isinstance(exif_data_dict, dict) and len(exif_data_dict) > 0:
                # Use cached EXIF data to build info string
                exif_tags = exif_data_dict
                # Log sample of tags to see what we have
                sample_tags = list(exif_tags.keys())[:10]
                logger.info(f"[STATUS] Using cached EXIF data with {len(exif_tags)} tags, sample: {sample_tags}")
                metadata_extracted_from_cache = True
            else:
                logger.debug(f"[STATUS] Cached EXIF data missing or empty - cached_exif keys: {list(cached_exif.keys()) if cached_exif else None}, "
                           f"exif_data type: {type(exif_data_dict)}, exif_data len: {len(exif_data_dict) if isinstance(exif_data_dict, dict) else 'N/A'}")
        else:
            logger.debug(f"[STATUS] No cached EXIF data found for {os.path.basename(_require_str_path(self.current_file_path))}")

        if exif_tags:
            # Extract EXIF info from exif_tags

            # Extract focal length (only if not 0 or null)
            # Try multiple possible tag keys for focal length
            focal_length_tags = ['EXIF FocalLength', 'EXIF FocalLengthIn35mmFilm']
            focal_length = None
            for tag_name in focal_length_tags:
                if tag_name in exif_tags:
                    focal_length_raw = exif_tags[tag_name]
                    try:
                        focal_str = str(focal_length_raw)
                        if '/' in focal_str:
                            num, den = focal_str.split('/')
                            focal_length = round(float(num) / float(den))
                        else:
                            focal_length = round(float(focal_str))
                        # Only add if focal length is not 0 or null
                        if focal_length and focal_length > 0:
                            exif_info.append(f"{focal_length}mm")
                            logger.debug(f"[STATUS] Found focal length from tag '{tag_name}': {focal_length}mm")
                            break
                    except (ValueError, AttributeError, ZeroDivisionError) as e:
                        logger.debug(f"[STATUS] Failed to parse focal length from tag '{tag_name}': {e}")
                        continue
            if not focal_length:
                # Try to find any tag containing "Focal" in the name
                for tag_key in exif_tags:
                    if 'Focal' in tag_key and tag_key not in focal_length_tags:
                        try:
                            focal_str = str(exif_tags[tag_key])
                            if '/' in focal_str:
                                num, den = focal_str.split('/')
                                focal_length = round(float(num) / float(den))
                            else:
                                focal_length = round(float(focal_str))
                            if focal_length and focal_length > 0:
                                exif_info.append(f"{focal_length}mm")
                                logger.debug(f"[STATUS] Found focal length from alternative tag '{tag_key}': {focal_length}mm")
                                break
                        except (ValueError, AttributeError, ZeroDivisionError):
                            continue
                if not focal_length:
                    logger.debug(f"[STATUS] No focal length found in tags. Available focal length tags: {[tag for tag in focal_length_tags if tag in exif_tags]}, "
                               f"all tags with 'Focal': {[tag for tag in exif_tags if 'Focal' in tag]}")

            # Extract aperture (only if not 0 or null)
            # Try multiple possible tag keys for aperture
            aperture_tags = ['EXIF FNumber', 'EXIF ApertureValue']
            aperture = None
            for tag_name in aperture_tags:
                if tag_name in exif_tags:
                    aperture_raw = exif_tags[tag_name]
                    try:
                        aperture_str = str(aperture_raw)
                        if '/' in aperture_str:
                            num, den = aperture_str.split('/')
                            aperture = float(num) / float(den)
                        else:
                            aperture = float(aperture_str)
                        # Only add if aperture is not 0 or null
                        if aperture and aperture > 0:
                            exif_info.append(f"f/{aperture:.1f}")
                            logger.debug(f"[STATUS] Found aperture from tag '{tag_name}': f/{aperture:.1f}")
                            break
                    except (ValueError, AttributeError, ZeroDivisionError) as e:
                        logger.debug(f"[STATUS] Failed to parse aperture from tag '{tag_name}': {e}")
                        continue
            if not aperture:
                # Try to find any tag containing "FNumber" or "Aperture" in the name
                for tag_key in exif_tags:
                    if ('FNumber' in tag_key or 'Aperture' in tag_key) and tag_key not in aperture_tags:
                        try:
                            aperture_str = str(exif_tags[tag_key])
                            if '/' in aperture_str:
                                num, den = aperture_str.split('/')
                                aperture = float(num) / float(den)
                            else:
                                aperture = float(aperture_str)
                            if aperture and aperture > 0:
                                exif_info.append(f"f/{aperture:.1f}")
                                logger.debug(f"[STATUS] Found aperture from alternative tag '{tag_key}': f/{aperture:.1f}")
                                break
                        except (ValueError, AttributeError, ZeroDivisionError):
                            continue
                if not aperture:
                    logger.debug(f"[STATUS] No aperture found in tags. Available aperture tags: {[tag for tag in aperture_tags if tag in exif_tags]}, "
                               f"all tags with 'FNumber' or 'Aperture': {[tag for tag in exif_tags if 'FNumber' in tag or 'Aperture' in tag]}")

            # Extract ISO
            # Try multiple possible tag keys for ISO
            iso_tags = ['EXIF ISOSpeedRatings', 'EXIF ISO', 'EXIF PhotographicSensitivity']
            iso = None
            for tag_name in iso_tags:
                if tag_name in exif_tags:
                    iso_raw = exif_tags[tag_name]
                    try:
                        iso_str = str(iso_raw)
                        # Handle fraction format for ISO
                        if '/' in iso_str:
                            num, den = iso_str.split('/')
                            iso = int(float(num) / float(den))
                        else:
                            iso = int(iso_str)
                        if iso and iso > 0:
                            exif_info.append(f"ISO {iso}")
                            logger.debug(f"[STATUS] Found ISO from tag '{tag_name}': ISO {iso}")
                            break
                    except (ValueError, AttributeError, ZeroDivisionError) as e:
                        logger.debug(f"[STATUS] Failed to parse ISO from tag '{tag_name}': {e}")
                        continue
            if not iso:
                # Try to find any tag containing "ISO" in the name
                for tag_key in exif_tags:
                    if 'ISO' in tag_key.upper() and tag_key not in iso_tags:
                        try:
                            iso_str = str(exif_tags[tag_key])
                            if '/' in iso_str:
                                num, den = iso_str.split('/')
                                iso = int(float(num) / float(den))
                            else:
                                iso = int(iso_str)
                            if iso and iso > 0:
                                exif_info.append(f"ISO {iso}")
                                logger.debug(f"[STATUS] Found ISO from alternative tag '{tag_key}': ISO {iso}")
                                break
                        except (ValueError, AttributeError, ZeroDivisionError):
                            continue
                if not iso:
                    logger.debug(f"[STATUS] No ISO found in tags. Available ISO tags: {[tag for tag in iso_tags if tag in exif_tags]}, "
                               f"all tags with 'ISO': {[tag for tag in exif_tags if 'ISO' in tag.upper()]}")

            # Extract capture time
            datetime_tags = ['EXIF DateTimeOriginal',
                             'Image DateTime', 'EXIF DateTime']
            for tag_name in datetime_tags:
                if tag_name in exif_tags:
                    datetime_raw = exif_tags[tag_name]
                    try:
                        datetime_str = str(datetime_raw)
                        from datetime import datetime
                        dt = datetime.strptime(
                            datetime_str, "%Y:%m:%d %H:%M:%S")
                        exif_info.append(dt.strftime("%H:%M:%S %Y-%m-%d"))
                        break
                    except (ValueError, AttributeError):
                        continue

            # Check if we successfully extracted any metadata from cached tags
            if not exif_info:
                logger.warning("[STATUS] No metadata extracted from cached EXIF tags. Attempting direct EXIF extraction as fallback.")
                metadata_extracted_from_cache = False

        # Fallback: If no exif_tags OR if we didn't extract any metadata from cache, try direct extraction
        # Only allow expensive synchronous extraction if not in fast_open startup/navigation phase to keep main thread fluid
        is_fast_open_active = getattr(self, "_folder_fast_open_token", None) is not None
        if (not exif_tags or not metadata_extracted_from_cache) and allow_expensive_exif_probe and not is_fast_open_active:
            _probe_ts[self.current_file_path] = now_ts
            self._status_exif_probe_ts = _probe_ts
            logger.info(f"[STATUS] Using fallback EXIF extraction - exif_tags: {exif_tags is not None}, metadata_extracted: {metadata_extracted_from_cache}")
            try:
                # Try direct EXIF extraction from file
                exif_data = self.extract_exif_data(self.current_file_path)
                if exif_data:
                    # Only add metadata that wasn't already extracted
                    if not any('mm' in item for item in exif_info) and exif_data.get('focal_length'):
                        exif_info.append(exif_data['focal_length'])
                        logger.info(f"[STATUS] Added focal length from fallback: {exif_data['focal_length']}")
                    if not any('f/' in item for item in exif_info) and exif_data.get('aperture'):
                        exif_info.append(exif_data['aperture'])
                        logger.info(f"[STATUS] Added aperture from fallback: {exif_data['aperture']}")
                    if not any('ISO' in item for item in exif_info) and exif_data.get('iso'):
                        exif_info.append(exif_data['iso'])
                        logger.info(f"[STATUS] Added ISO from fallback: {exif_data['iso']}")
                    if not any(':' in item and '-' in item for item in exif_info) and exif_data.get('capture_time'):
                        exif_info.append(exif_data['capture_time'])
                        logger.info(f"[STATUS] Added capture time from fallback: {exif_data['capture_time']}")

                    # If we got new data, try to update cache for future use
                    if exif_data and any([exif_data.get('focal_length'), exif_data.get('aperture'),
                                         exif_data.get('iso'), exif_data.get('capture_time')]):
                        logger.info("[STATUS] Fallback extraction found metadata, updating cache")
                        # Try to merge with existing cache or create new entry
                        if not cached_exif:
                            cached_exif = {}
                        # Update cache with extracted data
                        if 'exif_data' not in cached_exif or not cached_exif.get('exif_data'):
                            # Create a basic exif_data dict from extracted values
                            cached_exif['exif_data'] = {}
                        # Store the extracted values in a format that can be read later
                        # Note: This is a simplified cache update - full EXIF extraction would be better
                        self.image_cache.put_exif(self.current_file_path, cached_exif)
            except Exception as e:
                logger.warning(f"[STATUS] Fallback EXIF extraction failed: {e}")
        elif not exif_tags or not metadata_extracted_from_cache:
            logger.debug(
                "[STATUS] Skip fallback EXIF probe for %s (cooldown active)",
                os.path.basename(_require_str_path(self.current_file_path)),
            )

        # Final fallback: direct EXIF read (pyexiv2 preferred, else exifread)
        # Skip this heavy fallback synchronous IO on the main thread during fast-open loading
        if not exif_info and self.current_file_path and allow_expensive_exif_probe and not is_fast_open_active:
            logger.warning(
                "[STATUS] No metadata from cache/fallback; direct EXIF read "
                "(metadata_backend)."
            )
            try:
                tags = process_file_from_path(
                    self.current_file_path, details=False
                )

                # Search all tags for focal length
                for tag_key in tags:
                    if "Focal" in tag_key and not any(
                        "mm" in item for item in exif_info
                    ):
                        try:
                            focal_str = str(tags[tag_key])
                            if "/" in focal_str:
                                num, den = focal_str.split("/")
                                focal_length = round(float(num) / float(den))
                            else:
                                focal_length = round(float(focal_str))
                            if focal_length and focal_length > 0:
                                exif_info.append(f"{focal_length}mm")
                                logger.info(
                                    "[STATUS] Found focal length from direct "
                                    f"read, tag '{tag_key}': {focal_length}mm"
                                )
                                break
                        except Exception:
                            continue

                # Search all tags for ISO
                for tag_key in tags:
                    if "ISO" in tag_key.upper() and not any(
                        "ISO" in item for item in exif_info
                    ):
                        try:
                            iso_str = str(tags[tag_key])
                            if "/" in iso_str:
                                num, den = iso_str.split("/")
                                iso = int(float(num) / float(den))
                            else:
                                iso = int(iso_str)
                            if iso and iso > 0:
                                exif_info.append(f"ISO {iso}")
                                logger.info(
                                    "[STATUS] Found ISO from direct read, "
                                    f"tag '{tag_key}': ISO {iso}"
                                )
                                break
                        except Exception:
                            continue

                # Search all tags for aperture
                for tag_key in tags:
                    if (
                        "FNumber" in tag_key or "Aperture" in tag_key
                    ) and not any("f/" in item for item in exif_info):
                        try:
                            aperture_str = str(tags[tag_key])
                            if "/" in aperture_str:
                                num, den = aperture_str.split("/")
                                aperture = float(num) / float(den)
                            else:
                                aperture = float(aperture_str)
                            if aperture and aperture > 0:
                                exif_info.append(f"f/{aperture:.1f}")
                                logger.info(
                                    "[STATUS] Found aperture from direct read, "
                                    f"tag '{tag_key}': f/{aperture:.1f}"
                                )
                                break
                        except Exception:
                            continue
            except Exception as e:
                logger.warning(
                    f"[STATUS] Direct EXIF read (metadata_backend) failed: {e}"
                )

        # Construct metadata text (center label)
        metadata_parts = []

        # Add filename and dimensions - ALWAYS show original resolution
        # If displaying thumbnail, indicate it but still show original resolution
        # CRITICAL: Use original_width and original_height if available, not display dimensions
        display_width_final = original_width if original_width else width
        display_height_final = original_height if original_height else height

        if display_width_final > 0 and display_height_final > 0:
            # Show original resolution (no thumbnail indicator needed)
            metadata_parts.append(f"{file_label} - {display_width_final}x{display_height_final}")
        else:
            metadata_parts.append(file_label)

        # Add zoom level
        metadata_parts.append(zoom_level)

        edr_label = self._edr_metadata_label()
        if edr_label:
            metadata_parts.append(edr_label)

        # Add EXIF info if available
        if exif_info:
            metadata_parts.extend(exif_info)

        # Join all parts with separator
        metadata_text = " - ".join(metadata_parts)
        # Show and set metadata text in single view mode (top bar, centered).
        if self.view_mode == 'single':
            self._apply_top_metadata_text(metadata_text)
            # Remember signature + rendered text so identical follow-up calls can
            # short-circuit (see dedup guard above) instead of re-parsing/re-logging.
            self._last_status_signature = status_signature
            self._last_status_metadata_text = metadata_text
            # Track metadata display (logged once per distinct render thanks to dedup)
            import logging
            logger = logging.getLogger(__name__)
            # Extract individual EXIF fields for tracking
            has_focal = any('mm' in str(item) for item in exif_info) if exif_info else False
            has_aperture = any('f/' in str(item) for item in exif_info) if exif_info else False
            has_iso = any('ISO' in str(item) for item in exif_info) if exif_info else False
            has_datetime = any(':' in str(item) and '-' in str(item) for item in exif_info) if exif_info else False
            logger.info(f"[TRACK] Metadata displayed - file: {file_label}, full_text: {metadata_text}, "
                      f"exif_fields: {len(exif_info)}, focal: {has_focal}, aperture: {has_aperture}, iso: {has_iso}, datetime: {has_datetime}")
        else:
            self._apply_top_metadata_text("")

        # Update image counter (hidden until EXIF sort refinement is applied)
        if hasattr(self, "status_counter_label"):
            self._refresh_image_counter()
