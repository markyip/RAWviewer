"""
Manages preloading of adjacent images for fast navigation.
"""
from __future__ import annotations

import os

from PyQt6.QtCore import QObject

from image_cache import get_image_cache
from raw_adjustments.file_extensions import RAW_FILE_EXTENSIONS

from .enhanced_processor import EnhancedRAWProcessor


class PreloadManager(QObject):
    """Manages preloading of adjacent images for fast navigation using RAWProcessor (v0.5 style)."""

    def __init__(self, max_preload_threads: int = 2, processor_class=None):
        super().__init__()
        self.max_preload_threads = max_preload_threads
        self.active_threads = {}
        self.cache = get_image_cache()
        # Accept processor class as parameter to avoid circular imports
        # If None, will use EnhancedRAWProcessor as fallback
        self.processor_class = processor_class

    def _is_raw_file(self, file_path: str) -> bool:
        """Check if file is a RAW format."""
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        return ext in RAW_FILE_EXTENSIONS

    def preload_images(self, file_paths: list[str], priority_order: list[str] | None = None):
        """Preload images in background threads using RAWProcessor (v0.5 style)."""
        if priority_order is None:
            priority_order = file_paths

        # Cancel existing preload threads
        self.cancel_all_preloads()

        # Use provided processor class or fallback to EnhancedRAWProcessor
        ProcessorClass = self.processor_class if self.processor_class else EnhancedRAWProcessor
        use_raw_processor = (self.processor_class is not None)

        # Start preloading up to max_preload_threads images
        for _i, file_path in enumerate(priority_order[:self.max_preload_threads]):
            if file_path not in self.active_threads:
                # Check if already cached
                if self.cache.get_full_image(file_path) is not None:
                    continue

                # Start preload thread
                if use_raw_processor:
                    # Use RAWProcessor (v0.5 style)
                    is_raw = self._is_raw_file(file_path)
                    thread = ProcessorClass(
                        file_path, is_raw=is_raw, use_full_resolution=False)
                    # Connect signals - RAWProcessor uses image_processed signal
                    thread.image_processed.connect(
                        lambda img, fp=file_path: self._on_preload_complete(fp, img))
                    thread.error_occurred.connect(
                        lambda err, fp=file_path: self._on_preload_error(fp, err))
                else:
                    # Fallback to EnhancedRAWProcessor
                    thread = ProcessorClass(
                        file_path, use_quality_processing=False)
                    thread.image_processed.connect(
                        lambda img, fp=file_path: self._on_preload_complete(fp, img))

                thread.finished.connect(
                    lambda fp=file_path: self._cleanup_thread(fp))

                self.active_threads[file_path] = thread
                thread.start()

    def _on_preload_complete(self, file_path: str, rgb_image):
        """Handle preloaded image completion."""
        # Image is automatically cached by RAWProcessor/EnhancedRAWProcessor
        # Just clean up the thread reference
        if file_path in self.active_threads:
            # Thread will be cleaned up by finished signal
            pass

    def _on_preload_error(self, file_path: str, error_msg: str):
        """Handle preload error."""
        # Log error but don't show to user (preloading is background operation)
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Preload error for {os.path.basename(file_path)}: {error_msg}")
        # Clean up thread on error
        if file_path in self.active_threads:
            self._cleanup_thread(file_path)

    def cancel_all_preloads(self):
        """Cancel all active preload threads."""
        import logging
        logger = logging.getLogger(__name__)

        threads_to_cleanup = list(self.active_threads.items())
        logger.debug(f"cancel_all_preloads(): Cancelling {len(threads_to_cleanup)} preload threads")

        for file_path, thread in threads_to_cleanup:
            try:
                file_basename = os.path.basename(file_path)
                # Use cleanup() if available (RAWProcessor), otherwise use stop_processing()
                if hasattr(thread, 'cleanup'):
                    # cleanup() will handle thread stopping and rawpy handle closing safely
                    logger.debug(f"cancel_all_preloads(): Calling cleanup() for {file_basename}")
                    thread.cleanup()
                else:
                    # For EnhancedRAWProcessor, use stop_processing and wait
                    logger.debug(f"cancel_all_preloads(): Using stop_processing() for {file_basename}")
                    if hasattr(thread, 'stop_processing'):
                        thread.stop_processing()

                    # OPTIMIZED: Use shorter initial wait, but allow longer if needed
                    # Preload threads are lower priority, so we can be more aggressive
                    if hasattr(thread, 'wait'):
                        wait_result = thread.wait(100)  # Initial 100ms wait (fast path)
                        logger.debug(f"cancel_all_preloads(): wait(100) returned {wait_result} for {file_basename}")
                        if not wait_result and hasattr(thread, 'isRunning') and thread.isRunning():
                            # Thread still running, wait longer for rawpy operations
                            additional_wait = thread.wait(200)  # Additional 200ms for rawpy operations
                            logger.debug(f"cancel_all_preloads(): Additional wait(200) returned {additional_wait} for {file_basename}")
                            if not additional_wait and thread.isRunning():
                                logger.debug(f"cancel_all_preloads(): Terminating thread for {file_basename}")
                                thread.terminate()
                                thread.wait(50)  # Short wait after terminate
            except Exception as e:
                logger.warning(f"cancel_all_preloads(): Error cleaning up thread for {os.path.basename(file_path)}: {e}", exc_info=True)

        self.active_threads.clear()
        logger.debug("cancel_all_preloads(): All preload threads cancelled")

    def _cleanup_thread(self, file_path: str):
        """Clean up finished thread."""
        if file_path in self.active_threads:
            del self.active_threads[file_path]
