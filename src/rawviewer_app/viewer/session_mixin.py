"""Folder load and session persistence (mixin for RAWImageViewer)."""
import logging
import os
import time
import traceback

from PyQt6.QtCore import QRunnable, QThreadPool, QTimer

from rawviewer_app.env import _env_int, _norm_path
from rawviewer_app.signals import FolderLoadSignals


class SessionMixin:
    def load_folder_images(self, folder_path, start_file=None, start_view=None):
        """Load images from a folder without blocking the UI during scan/sort."""
        import logging
        logger = logging.getLogger(__name__)
        folder_path = os.path.abspath(folder_path)
        if hasattr(self, "image_manager") and self.image_manager is not None:
            self.image_manager.update_volume_throttling(folder_path)
            # One-shot lightweight read-speed probe for this volume (off the main
            # thread). Fast external drives (TB3/4, USB4, NVMe) keep full
            # concurrency; only confirmed-slow drives get throttled.
            self.image_manager.prime_volume_speed_async(folder_path, start_file)
        same_folder = bool(
            getattr(self, "current_folder", None)
            and _norm_path(getattr(self, "current_folder")) == _norm_path(folder_path)
        )
        fast_open = self._should_fast_open_single_file(start_file, start_view)
        preserve_folder_index = bool(
            same_folder
            and fast_open
            and start_file
            and len(getattr(self, "image_files", []) or []) > 1
        )
        if not same_folder:
            self._gallery_bookmarked_paths = self._load_persisted_gallery_bookmarks(folder_path)
            # Drop full/preview/pixmap RAM and prune LibRaw skip-set when leaving a folder.
            try:
                cache = getattr(self, "image_cache", None)
                if cache is not None and hasattr(cache, "on_folder_changed"):
                    keep = start_file
                    if keep and not os.path.isabs(keep):
                        keep = os.path.join(folder_path, keep)
                    cache.on_folder_changed(keep_path=keep if keep and os.path.isfile(keep) else None)
            except Exception:
                pass
        # Folder scope changed: clear search bar, indexing UI, and filter snapshot.
        if not preserve_folder_index:
            self._reset_semantic_search_for_new_folder()

        try:
            if not folder_path:
                self.show_error("Invalid Folder", "No folder path provided")
                self._hide_all_loading_indicators()
                return
            if not os.path.exists(folder_path):
                self.show_error("Folder Not Found", f"The folder does not exist:\n{folder_path}")
                self._hide_all_loading_indicators()
                return
            if not os.path.isdir(folder_path):
                self.show_error("Invalid Path", f"The path is not a folder:\n{folder_path}")
                self._hide_all_loading_indicators()
                return

            if start_view in ("gallery", "single"):
                self.view_mode = start_view
            elif start_file and getattr(self, "view_mode", None) == "gallery":
                # Legacy behavior: opening a specific file from gallery focuses single view.
                self.view_mode = "single"

            self._folder_load_generation = getattr(self, "_folder_load_generation", 0) + 1
            token = self._folder_load_generation
            if not preserve_folder_index and hasattr(self, "_cancel_stale_folder_async_work"):
                self._cancel_stale_folder_async_work(
                    f"load_folder_images -> {os.path.basename(folder_path)}"
                )
            self._folder_sort_refinement_applied_token = None
            extensions = set(self.get_supported_extensions())
            newest_first = self.get_sort_preference()

            if fast_open:
                start_path = self._resolve_start_file_path(folder_path, start_file)
                if start_path:
                    self._apply_instant_single_file_open(
                        folder_path,
                        start_path,
                        token,
                        preserve_folder_index=preserve_folder_index,
                    )
                    if preserve_folder_index:
                        self._sync_filmstrip_to_folder()
                        return
                else:
                    fast_open = False

            if hasattr(self, 'view_mode'):
                if self.view_mode == 'gallery':
                    self._ensure_gallery_widget()
                    if self.gallery_justified:
                        self.gallery_justified.set_images([])
                        self.gallery_justified.show_loading_message("Scanning folder...")
                    # Hide single view immediately so gallery mode never stacks two panes.
                    self._show_gallery_view()
                else:
                    if hasattr(self, 'loading_overlay'):
                        if not fast_open:
                            self.loading_overlay.show_loading("Scanning folder...")
                    if hasattr(self, 'image_label') and not fast_open:
                        self.image_label.clear()

            signals = FolderLoadSignals()
            signals.ready.connect(self._on_folder_load_ready)
            signals.error.connect(self._on_folder_load_error)
            self._active_folder_load_signals = signals
            self._folder_sort_refinement_token = None
            viewer = self

            class _FolderLoadWorker(QRunnable):
                def __init__(self_inner, token, folder_path, extensions, newest_first,
                             start_file, start_view, signals):
                    super().__init__()
                    self_inner.token = token
                    self_inner.folder_path = folder_path
                    self_inner.extensions = extensions
                    self_inner.newest_first = newest_first
                    self_inner.start_file = start_file
                    self_inner.start_view = start_view
                    self_inner.signals = signals

                def _scan_top_level_only(self_inner, path):
                    """List image files in path only; do not descend into subfolders."""
                    with os.scandir(path) as it:
                        for entry in it:
                            if entry.name.startswith('.'):
                                continue
                            try:
                                if not entry.is_file(follow_symlinks=False):
                                    continue
                                ext = os.path.splitext(entry.name)[1].lower()
                                if ext not in self_inner.extensions:
                                    continue
                                stat = entry.stat()
                                if stat.st_size > 0:
                                    yield entry.path, stat
                            except (OSError, PermissionError):
                                continue

                def run(self_inner):
                    import time
                    from datetime import datetime
                    try:
                        scan_start = time.time()
                        image_files = []
                        file_stats = {}
                        for ap, stat_info in viewer._scan_folder_image_paths(
                            self_inner.folder_path
                        ):
                            image_files.append(ap)
                            file_stats[ap] = (stat_info.st_size, stat_info.st_mtime)

                        scan_time = time.time() - scan_start

                        sort_start = time.time()
                        bulk_metadata = {}
                        if image_files:
                            from common_image_loader import capture_timestamp_for_sort

                            if self_inner.start_file:
                                image_files = viewer._mtime_sort_image_files(
                                    image_files, file_stats, self_inner.newest_first
                                )
                            else:
                                from image_cache import get_image_cache
                                cache = get_image_cache()
                                bulk_metadata = cache.get_multiple_exif(image_files, file_stats)
                                probed_timestamps = viewer._parallel_probe_capture_times(image_files, bulk_metadata)

                                sort_keys = {}
                                for fp in image_files:
                                    meta = bulk_metadata.get(fp)
                                    mtime = file_stats.get(fp, (0, 0))[1]
                                    probed_ts = probed_timestamps.get(fp, 0.0)
                                    from common_image_loader import resolve_folder_sort_timestamp, is_raw_file
                                    _has_capture, timestamp, _source = resolve_folder_sort_timestamp(
                                        fp,
                                        metadata=meta,
                                        file_mtime=mtime,
                                        probe_file=False,
                                        probed_capture_timestamp=probed_ts
                                    )
                                    base_name = os.path.basename(fp).lower()
                                    stem = os.path.splitext(base_name)[0]
                                    ext = os.path.splitext(base_name)[1]
                                    # Keep DNG+JPEG backup pairs adjacent while preferring
                                    # display-friendly non-RAW variants first.
                                    raw_rank = 1 if is_raw_file(fp) else 0
                                    primary_ts = -timestamp if self_inner.newest_first else timestamp
                                    sort_keys[fp] = (primary_ts, stem, raw_rank, ext, base_name)

                                image_files = sorted(
                                    image_files,
                                    key=lambda fp: sort_keys[fp],
                                )
                        sort_time = time.time() - sort_start

                        try:
                            from PyQt6 import sip
                            if not sip.isdeleted(self_inner.signals):
                                self_inner.signals.ready.emit(
                                    self_inner.token,
                                    image_files,
                                    bulk_metadata,
                                    file_stats,
                                    self_inner.folder_path,
                                    self_inner.start_file,
                                    self_inner.start_view,
                                    scan_time,
                                    sort_time,
                                )
                        except Exception:
                            pass
                    except OSError as e:
                        try:
                            from PyQt6 import sip
                            if not sip.isdeleted(self_inner.signals):
                                self_inner.signals.error.emit(
                                    self_inner.token,
                                    "Folder Access Error",
                                    f"Cannot read folder contents:\n{str(e)}",
                                )
                        except Exception:
                            pass
                    except Exception as e:
                        try:
                            from PyQt6 import sip
                            if not sip.isdeleted(self_inner.signals):
                                self_inner.signals.error.emit(
                                    self_inner.token,
                                    "Folder Load Error",
                                    f"Unexpected error loading folder:\n{str(e)}",
                                )
                        except Exception:
                            pass

            worker = _FolderLoadWorker(token, folder_path, extensions, newest_first, start_file, start_view, signals)
            self._active_folder_load_worker = worker
            if fast_open:
                defer_ms = _env_int("RAWVIEWER_FAST_OPEN_FOLDER_LOAD_DEFER_MS", 2500, minimum=0)

                def _deferred_start(w=worker):
                    if token == getattr(viewer, "_folder_load_generation", None):
                        QThreadPool.globalInstance().start(w, priority=-1)

                logger.info(
                    "[FOLDER] Background folder load deferred until first paint or %dms",
                    defer_ms,
                )
                if defer_ms <= 0:
                    _deferred_start()
                else:
                    viewer._defer_until_first_paint(_deferred_start, fallback_ms=defer_ms)
            else:
                QThreadPool.globalInstance().start(worker)
                logger.info("[FOLDER] Background folder load started for %s", folder_path)
        except Exception as e:
            logger.error(f"Unexpected error in load_folder_images for {folder_path}: {e}", exc_info=True)
            self.show_error("Folder Load Error", f"Unexpected error loading folder:\n{str(e)}")
            self._hide_all_loading_indicators()

    def schedule_save_session_state(self) -> None:
        """Coalesce frequent QSettings writes (e.g. rapid Down-arrow discard) onto a debounced timer."""
        t = getattr(self, "_save_session_debounce_timer", None)
        if t is None:
            self.save_session_state()
            return
        t.start()

    def _cancel_load_and_preload_for_path(self, file_path: str) -> None:
        """Stop in-flight loads for a path we're about to rename/delete (keeps discard responsive)."""
        if not file_path:
            return
        try:
            if getattr(self, "image_manager", None) is not None:
                self.image_manager.cancel_task(file_path)
        except Exception:
            pass
        preload = getattr(self, "preload_manager", None)
        if preload is None or not hasattr(preload, "active_threads"):
            return
        if file_path not in preload.active_threads:
            return
        thread = preload.active_threads.pop(file_path, None)
        if thread is None:
            return
        try:
            if hasattr(thread, "cleanup"):
                thread.cleanup()
            elif hasattr(thread, "stop_processing"):
                thread.stop_processing()
                thread.quit()
                thread.wait(40)
        except Exception:
            pass

    def _remove_file_from_active_image_list(self, file_path: str) -> None:
        """Remove ``file_path`` from ``image_files`` with a cheap path when it's the current index."""
        if not file_path or not self.image_files:
            return
        i = self.current_file_index
        if 0 <= i < len(self.image_files) and self.image_files[i] == file_path:
            del self.image_files[i]
            return
        try:
            self.image_files.remove(file_path)
        except ValueError:
            pass

    def _drop_discarded_from_semantic_corpus(self, file_path: str) -> None:
        lst = getattr(self, "_semantic_search_corpus_files", None)
        if not lst:
            return
        try:
            lst.remove(file_path)
        except ValueError:
            pass

    def save_session_state(self):
        settings = self.get_settings()
        
        # Always save window geometry and state
        settings.setValue("window_geometry", self.saveGeometry())
        settings.setValue("window_state", self.saveState())
        
        if self.current_folder and self.current_file_index >= 0 and self.image_files:
            filename = os.path.basename(
                self.image_files[self.current_file_index])
            settings.setValue("last_session_folder", self.current_folder)
            settings.setValue("last_session_file", filename)
            # Save view mode so we can restore it
            if hasattr(self, 'view_mode'):
                settings.setValue("last_session_view_mode", self.view_mode)
        else:
            settings.remove("last_session_folder")
            settings.remove("last_session_file")
            settings.remove("last_session_view_mode")
        self._save_persisted_visual_rotations()
        self._save_persisted_gallery_bookmarks()
        self._save_persisted_overlay_positions()

    def _load_persisted_overlay_positions(self) -> None:
        """Load histogram overlay positions saved from a prior session."""
        import json

        self._pending_overlay_positions = None
        try:
            raw = self.get_settings().value("overlay_positions", "")
            if not raw:
                return
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", errors="ignore")
            data = json.loads(raw)
            if isinstance(data, dict) and data:
                self._pending_overlay_positions = data
        except Exception:
            pass

    def _try_apply_pending_overlay_positions(self) -> None:
        data = getattr(self, "_pending_overlay_positions", None)
        if not data:
            return
        container = getattr(self, "single_view_container", None)
        if container is None or container.width() < 80 or container.height() < 80:
            return
        if hasattr(container, "apply_overlay_session_positions"):
            container.apply_overlay_session_positions(data)
        self._pending_overlay_positions = None

    def _save_persisted_overlay_positions(self) -> None:
        import json

        container = getattr(self, "single_view_container", None)
        if container is None or not hasattr(container, "overlay_session_snapshot"):
            return
        snap = container.overlay_session_snapshot()
        try:
            if snap:
                self.get_settings().setValue("overlay_positions", json.dumps(snap))
            else:
                self.get_settings().remove("overlay_positions")
        except Exception:
            pass

    def restore_session_state(self):
        """Restore the last session's folder and file, with error handling for unavailable drives"""
        import logging
        logger = logging.getLogger(__name__)
        
        settings = self.get_settings()
        folder = settings.value("last_session_folder", None)
        file = settings.value("last_session_file", None)
        
        if folder and file:
            try:
                # Check if folder exists and is accessible
                if not os.path.isdir(folder):
                    logger.warning(f"[SESSION] Last session folder not found or not accessible: {folder}")
                    # Clear invalid session state
                    settings.remove("last_session_folder")
                    settings.remove("last_session_file")
                    return False
                
                # Try to list files in the folder
                # Lazy import natsort to avoid import delays
                from natsort import natsorted
                try:
                    files = [f for f in natsorted(os.listdir(folder))
                             if os.path.splitext(f)[1].lower() in self.get_supported_extensions()]
                except (PermissionError, OSError) as e:
                    # Handle cases where drive/folder is not accessible (e.g., disconnected network drive, USB drive)
                    logger.warning(f"[SESSION] Cannot access last session folder '{folder}': {e}")
                    # Clear invalid session state
                    settings.remove("last_session_folder")
                    settings.remove("last_session_file")
                    return False
                
                if file in files:
                    try:
                        # Restore view mode before loading folder
                        # If last view mode was 'gallery', open in 'single' mode instead
                        # to avoid slow gallery loading on large folders
                        view_mode = settings.value("last_session_view_mode", "single")
                        if view_mode == 'gallery':
                            # Force single view mode for better launch experience
                            self.view_mode = 'single'
                        elif view_mode in ('single', 'gallery'):
                            self.view_mode = view_mode
                        else:
                            self.view_mode = 'single'
                        self._orientation_already_applied = False # Reset flag on mode switch to be safe

                        self._session_restore_defer_preload = True
                        self._session_restore_staged_scheduled = False
                        self._session_restore_full_decode_allowed = False
                        logger.info(
                            "[SESSION] Deferring heavy loads until first paint "
                            "(full decode ~2.5s later; "
                            "RAWVIEWER_SESSION_RESTORE_DEFER_PRELOAD=0 to disable)"
                        )
                        try:
                            lm = getattr(self, "image_manager", None)
                            if lm is not None and hasattr(lm, "enter_gallery_warmup_throttle"):
                                lm.enter_gallery_warmup_throttle()
                        except Exception:
                            pass
                        self.load_folder_images(folder, start_file=file)
                        return True
                    except (PermissionError, OSError) as e:
                        logger.warning(f"[SESSION] Cannot load folder '{folder}': {e}")
                        # Clear invalid session state
                        settings.remove("last_session_folder")
                        settings.remove("last_session_file")
                        return False
                else:
                    logger.debug(f"[SESSION] Last session file '{file}' not found in folder '{folder}'")
                    return False
            except Exception as e:
                # Catch any other unexpected errors
                logger.error(f"[SESSION] Error restoring session state: {e}", exc_info=True)
                # Clear invalid session state to prevent repeated errors
                try:
                    settings.remove("last_session_folder")
                    settings.remove("last_session_file")
                except:
                    pass
                return False
        
        return False

