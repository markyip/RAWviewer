import time
import os
import threading
from typing import List, Dict, Any, Optional

from PyQt6.QtWidgets import QWidget, QFrame, QHBoxLayout, QApplication, QLabel, QScrollArea
from PyQt6.QtCore import Qt, QTimer, QRect, QThreadPool, QSize
from PyQt6.QtGui import QPixmap, QImage, QFont, QImageReader

from src.ui.widgets import ThumbnailLabel, ImageLoaded
from src.image_cache import LRUCache, get_image_cache
from src.common_image_loader import get_image_aspect_ratio, load_pixmap_safe, is_raw_file

class ImageLoadTask(threading.Thread):
    """Background task to load and scale images"""
    
    def __init__(self, index, file_path, target_width, target_height, signal, parent_viewer=None, generation=0):
        super().__init__()
        self.index = index
        self.file_path = file_path
        self.target_width = target_width
        self.target_height = target_height
        self.signal = signal
        self.parent_viewer = parent_viewer
        self.generation = generation
        self._cancelled = False
        self._lock = threading.Lock()
        
    def cancel(self):
        """Cancel the task"""
        with self._lock:
            self._cancelled = True
            
    def is_cancelled(self):
        """Check if task is cancelled"""
        with self._lock:
            return self._cancelled
            
    def run(self):
        """Load and scale image in worker thread - returns QImage, not QPixmap"""
        if self.is_cancelled():
            return
            
        try:
            # OPTIMIZATION: Check if parent viewer has image_cache
            # If so, check for thumbnail or full image cache first
            if self.parent_viewer and hasattr(self.parent_viewer, 'image_cache'):
                cache = self.parent_viewer.image_cache
                
                # Check for cached thumbnail first (fastest)
                # Note: We need to be careful with scaling if using cached thumbnail
                cached_thumb = cache.get_thumbnail(self.file_path)
                if cached_thumb is not None:
                     # Convert numpy to QImage
                     from PyQt6.QtGui import QImage
                     h, w, c = cached_thumb.shape
                     bytes_per_line = 3 * w
                     q_img = QImage(cached_thumb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                     
                     if not q_img.isNull():
                        # Scale if needed
                        if w != self.target_width or h != self.target_height:
                            q_img = q_img.scaled(self.target_width, self.target_height, 
                                               Qt.AspectRatioMode.KeepAspectRatio, 
                                               Qt.TransformationMode.SmoothTransformation)
                        
                        if not self.is_cancelled():
                            try:
                                self.signal.loaded.emit(self.index, q_img, self.generation)
                                return
                            except RuntimeError:
                                pass
            
            # Normal loading path (if not cached)
            is_raw = is_raw_file(self.file_path)
            
            if is_raw:
                # Use load_pixmap_safe logic or specialized RAW loader?
                # For thumbnails in gallery, we want speed.
                # EnhancedRAWProcessor's ThumbnailExtractor is best here.
                from src.enhanced_raw_processor import ThumbnailExtractor
                extractor = ThumbnailExtractor()
                thumb = extractor.extract_thumbnail_from_raw(self.file_path)
                
                if thumb is not None and not self.is_cancelled():
                     h, w, c = thumb.shape
                     bytes_per_line = 3 * w
                     q_img = QImage(thumb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                     
                     # Ensure orientation matches (extractor might return unrotated numpy array for some raw implementations)
                     # But ThumbnailExtractor usually handles logic.
                     # Let's verify orientation just in case.
                     
                     # Scale
                     if not q_img.isNull():
                        q_img = q_img.scaled(self.target_width, self.target_height, 
                                           Qt.AspectRatioMode.KeepAspectRatio, 
                                           Qt.TransformationMode.SmoothTransformation)
                        try:
                            self.signal.loaded.emit(self.index, q_img, self.generation)
                            return
                        except RuntimeError:
                            pass
            
            # Fallback / Regular Image
            reader = QImageReader(self.file_path)
            
            if self.is_cancelled():
                return

            original_size = reader.size()
            
            if not original_size.isValid():
                # FALLBACK: Try PIL if QImageReader fails to get size
                try:
                    from PIL import Image, ImageOps
                    with Image.open(self.file_path) as img:
                        img = ImageOps.exif_transpose(img)
                        w, h = img.size
                        # Manual scale
                        aspect = w / h if h > 0 else 1.0
                        sw = int(self.target_height * aspect)
                        sh = self.target_height
                        if sw > self.target_width:
                            sw = self.target_width
                            sh = int(self.target_width / aspect) if aspect > 0 else self.target_height
                        
                        img = img.resize((sw, sh), Image.Resampling.LANCZOS)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Convert to QImage
                        image_bytes = img.tobytes('raw', 'RGB')
                        scaled_image = QImage(image_bytes, sw, sh, sw * 3, QImage.Format.Format_RGB888)
                        
                        if not scaled_image.isNull():
                            try:
                                self.signal.loaded.emit(self.index, scaled_image, self.generation)
                            except RuntimeError:
                                pass # Object deleted
                            return
                except Exception:
                    pass
                return

            aspect = original_size.width() / original_size.height() if original_size.height() > 0 else 1.0
            scaled_width = int(self.target_height * aspect)
            scaled_height = self.target_height
            
            # Ensure we don't exceed target width
            if scaled_width > self.target_width:
                scaled_width = self.target_width
                scaled_height = int(self.target_width / aspect) if aspect > 0 else self.target_height
            
            # Set scaled size - this makes QImageReader decode at target size directly
            reader.setScaledSize(QSize(scaled_width, scaled_height))
            reader.setAutoTransform(True)  # Handle EXIF orientation
            
            scaled_image = reader.read()
            
            if self.is_cancelled():
                return
                
            try:
                self.signal.loaded.emit(self.index, scaled_image, self.generation)
            except RuntimeError:
                pass
            
        except Exception as e:
            pass

class JustifiedGallery(QWidget):
    """
    Adaptive justified gallery layout.
    Uses adaptive row height with ±25% tolerance for better space utilization.
    Stores original pixmaps and rebuilds layout on resize.
    Implements lazy loading for faster initial display.
    """
    TARGET_ROW_HEIGHT = 200
    HEIGHT_TOLERANCE = 0.25  # allow ±25% adjustment
    MIN_SPACING = 4
    
    def __init__(self, images, parent=None):
        super().__init__(parent)
        import logging
        self.logger = logging.getLogger(__name__)
        self.parent_viewer = parent  # Reference to RAWImageViewer for loading images
        
        # Store original pixmaps (never scale twice)
        self.images = images  # List of file paths or pixmaps
        
        # Virtualization and Layout Layout
        self._gallery_layout_items = []  # List of {rect, file_path, aspect}
        self._visible_widgets = {}  # {file_path: ThumbnailLabel}
        self._widget_pool = []  # List of unused ThumbnailLabel widgets
        self._total_content_height = 0
        
        # Recursion protection
        self._building = False
        self._build_count = 0 
        self._resize_in_progress = False
        self._last_viewport_width = None
        self._ignore_resize_events = False
        
        # Persistent Metadata Cache
        self._metadata_cache = {} 
        
        # Thread pool for background image loading
        self.thread_pool = QThreadPool()
        # Increase thread count for faster parallel loading
        self.thread_pool.setMaxThreadCount(16)
        
        # Signal for image loading
        self.loader_signal = ImageLoaded()
        self.loader_signal.loaded.connect(self.apply_thumbnail)
        
        # Generation counter
        self._gallery_generation = 0
        self._active_tasks = {} # Track active tasks for cancellation
        
        # Monitoring and Batching
        self._load_timer = None
        self._resize_timer = None
        self._loading_tiles = set()
        self._background_loading_active = False 
        
        # Rate limiting
        self._load_queue = []
        self._priority_queue = [] 
        self._loads_per_second = 8
        self._batch_size = 8
        self._priority_batch_size = 20
        
        # Scroll Optimization
        self._last_scroll_y = -1
        self._last_scroll_time = 0
        self._current_scroll_speed = 0
        self._scroll_check_timer = None
        self._is_scrolling_fast = False
        self._scroll_optimize_threshold = 2000 # pixels/sec
        self._scroll_settle_timer = None
        
        # Cache
        self._thumbnail_cache = LRUCache(2000)
        self._row_height_buckets = [160, 200, 240, 280, 320]
        
        # Transparent overlay for loading message
        self._loading_label = None
        
        # Initialize widget attributes
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        QTimer.singleShot(100, self._delayed_build)
        
    def _setup_scroll_tracking(self):
        """Connect to parent scrollbar for speed tracking"""
        try:
            parent_scroll = self.parent_viewer.scroll_area if hasattr(self.parent_viewer, 'scroll_area') else None
            if parent_scroll:
                scrollbar = parent_scroll.verticalScrollBar()
                # Disconnect first to avoid duplicates
                try: scrollbar.valueChanged.disconnect(self._on_scroll)
                except: pass
                scrollbar.valueChanged.connect(self._on_scroll)
        except Exception as e:
            self.logger.error(f"[GALLERY] Failed to setup scroll tracking: {e}")

    def _on_scroll(self, value):
        """Track scroll speed and optimize loading"""
        import time
        now = time.time()
        
        # Update loading label position if visible
        if self._loading_label and self._loading_label.isVisible():
             self._update_loading_label_geometry()
        
        if self._last_scroll_y >= 0:
            dy = abs(value - self._last_scroll_y)
            dt = now - self._last_scroll_time
            
            if dt > 0.01: # Avoid division by zero
                current_speed = dy / dt
                self._current_scroll_speed = (self._current_scroll_speed * 0.3) + (current_speed * 0.7)
                
                was_fast = self._is_scrolling_fast
                self._is_scrolling_fast = self._current_scroll_speed > self._scroll_optimize_threshold
                
                if self._is_scrolling_fast != was_fast:
                    if not self._is_scrolling_fast:
                        QTimer.singleShot(50, self.load_visible_images)
        
        self._last_scroll_y = value
        self._last_scroll_time = now
        
        # Reset settle timer
        if self._scroll_settle_timer:
            self._scroll_settle_timer.stop()
        
        self._scroll_settle_timer = QTimer()
        self._scroll_settle_timer.setSingleShot(True)
        self._scroll_settle_timer.timeout.connect(self._on_scroll_settled)
        self._scroll_settle_timer.start(150) # 150ms settle time
        
    def _on_scroll_settled(self):
        """Called when scrolling stops"""
        self._current_scroll_speed = 0
        self._is_scrolling_fast = False
        self.load_visible_images()

    def _delayed_build(self):
        """Delayed initial build to ensure widget has proper size"""
        self._setup_scroll_tracking() 
        if self.width() > 0:
            QApplication.processEvents()
            self.build_gallery()
        else:
            QTimer.singleShot(100, self._delayed_build)
    
    def show_loading_message(self, message="Loading gallery..."):
        """Show loading message overlay"""
        if self._loading_label:
            self._loading_label.setText(message)
            self._loading_label.adjustSize()
            self._update_loading_label_geometry()
            return
        
        self._loading_label = QLabel(message, self)
        self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._loading_label.setStyleSheet("""
            QLabel {
                background-color: rgba(20, 20, 20, 200);
                color: rgba(255, 255, 255, 220);
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 12px;
            }
        """)
        font = QFont()
        font.setPointSize(10)
        self._loading_label.setFont(font)
        self._loading_label.show()
        self._loading_label.raise_()
        
        self._update_loading_label_geometry()
    
    def _update_loading_label_geometry(self):
        """Update loading label geometry - Bottom Center"""
        if self._loading_label and self.parent_viewer and self.parent_viewer.width() > 0:
            self._loading_label.adjustSize()
            w = self._loading_label.width()
            h = self._loading_label.height()
            
            parent_scroll = self.parent_viewer.scroll_area if hasattr(self.parent_viewer, 'scroll_area') else None
            if parent_scroll:
                 viewport_h = parent_scroll.viewport().height()
                 scroll_y = parent_scroll.verticalScrollBar().value()
                 y = scroll_y + viewport_h - h - 20
                 x = (self.width() - w) // 2
                 self._loading_label.move(x, int(y))
            else:
                 x = (self.width() - w) // 2
                 y = self.height() - h - 20
                 self._loading_label.move(x, y)
    
    def hide_loading_message(self):
        if self._loading_label:
            self._loading_label.hide()
            self._loading_label.deleteLater()
            self._loading_label = None
    
    def _get_viewport_width(self):
        """Helper method to get the correct viewport width from parent scroll area"""
        parent = self.parent()
        scroll_area = None
        
        while parent and not isinstance(parent, QScrollArea):
            parent = parent.parent()
        
        if parent and isinstance(parent, QScrollArea):
            scroll_area = parent
        
        if scroll_area:
            viewport = scroll_area.viewport()
            if viewport:
                return max(300, viewport.width() - 16)
        
        return max(300, self.width() - 16)
    
    def build_gallery(self, bulk_metadata=None):
        """Build true justified layout (Google Photos style) - virtualized version"""
        if self._building:
            return
        
        self._building = True
        self._build_count += 1
        
        try:
            # 1. Reset state
            for label in self._visible_widgets.values():
                label.hide()
                self._widget_pool.append(label)
            self._visible_widgets = {}
            self._gallery_layout_items = []
            
            # 2. Metadata handling
            if bulk_metadata:
                self._metadata_cache.update(bulk_metadata)
            
            cached_metadata = self._metadata_cache
            
            if not cached_metadata and self.parent_viewer and hasattr(self.parent_viewer, 'image_cache'):
                file_paths = [img for img in self.images if isinstance(img, str)]
                if file_paths:
                    cached_metadata = self.parent_viewer.image_cache.get_multiple_exif(file_paths)
                    self._metadata_cache = cached_metadata
            
            # 3. Layout Constants
            viewport_width = self._get_viewport_width()
            net_width = viewport_width - 16  # margins
            if net_width <= 0:
                self.hide_loading_message()
                self._building = False
                return

            current_y = 8
            current_row = []
            current_aspect_sum = 0
            
            def commit_row(row, aspect_sum, is_last=False):
                nonlocal current_y
                if not row or aspect_sum == 0:
                    return
                
                total_spacing = (len(row) - 1) * self.MIN_SPACING
                if not is_last:
                    row_h = (net_width - total_spacing) / aspect_sum
                    row_h = max(self.TARGET_ROW_HEIGHT * 0.5, min(self.TARGET_ROW_HEIGHT * 2.0, row_h))
                else:
                    row_h = self.TARGET_ROW_HEIGHT
                
                curr_x = 8
                for i, (item, aspect) in enumerate(row):
                    w = int(row_h * aspect)
                    if not is_last and i == len(row) - 1:
                        w = net_width - (curr_x - 8)
                    
                    rect = QRect(curr_x, int(current_y), int(w), int(row_h))
                    self._gallery_layout_items.append({
                        'rect': rect,
                        'file_path': item if isinstance(item, str) else None,
                        'aspect': aspect
                    })
                    curr_x += w + self.MIN_SPACING
                
                current_y += row_h + self.MIN_SPACING

            # 4. Greedy Row Partitioning
            PROCESS_BATCH_SIZE = 100 
            
            for idx, item in enumerate(self.images):
                if idx > 0 and idx % PROCESS_BATCH_SIZE == 0:
                    QApplication.processEvents()
                
                aspect = 1.333
                
                if isinstance(item, str):
                    m = cached_metadata.get(item)
                    if m and m.get('original_width') and m.get('original_height'):
                        w = m['original_width']
                        h = m['original_height']
                        orientation = m.get('orientation', 1)
                        if orientation in (5, 6, 7, 8):
                            w, h = h, w
                        aspect = w / h
                    else:
                        try:
                            aspect = get_image_aspect_ratio(item)
                            if aspect <= 0 or aspect > 10:
                                aspect = 1.333
                        except Exception:
                            aspect = 1.333
                else:
                    aspect = item.width() / item.height() if item.height() > 0 else 1.333
                
                current_row.append((item, aspect))
                current_aspect_sum += aspect
                
                ideal_width_at_target = current_aspect_sum * self.TARGET_ROW_HEIGHT + (len(current_row)-1)*self.MIN_SPACING
                
                if ideal_width_at_target >= net_width:
                    commit_row(current_row, current_aspect_sum, False)
                    current_row = []
                    current_aspect_sum = 0
            
            if current_row:
                commit_row(current_row, current_aspect_sum, True)
            
            self._total_content_height = int(current_y + 8)
            self.setMinimumHeight(self._total_content_height)
            self.update() 
            
            QApplication.processEvents() # Final flush
            
        except Exception as e:
            self.logger.error(f"[JUSTIFIED_GALLERY] Build error: {e}", exc_info=True)
        finally:
            self._building = False
            self.hide_loading_message() 
            QTimer.singleShot(0, self.load_visible_images)
            QTimer.singleShot(100, self._check_and_hide_loading_if_visible_loaded)

    def paintEvent(self, event):
        """Standard paint event"""
        pass # Widgets are handled by Qt
        # However, we implement virtualization in load_visible_images, which populates widgets

    def resizeEvent(self, event):
        """Handle resize with debounce"""
        if self._ignore_resize_events:
            return
            
        # Only rebuild if width changed significantly
        new_width = self.width()
        if self._last_viewport_width == new_width:
            return
            
        self._last_viewport_width = new_width
        
        self.show_loading_message("Resizing...")
        
        if self._resize_timer:
            self._resize_timer.stop()
            
        self._resize_timer = QTimer()
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(lambda: self.build_gallery())
        self._resize_timer.start(200) # 200ms debounce
        
        super().resizeEvent(event)

    def load_visible_images(self):
        """Load images visible in the viewport"""
        if self._building:
            return

        if self._is_scrolling_fast:
            return # Don't load while scrolling fast
            
        parent_scroll = self.parent()
        while parent_scroll and not isinstance(parent_scroll, QScrollArea):
             parent_scroll = parent_scroll.parent()
             
        if not isinstance(parent_scroll, QScrollArea):
            return # Can't determine visibility
            
        viewport_height = parent_scroll.viewport().height()
        scroll_y = parent_scroll.verticalScrollBar().value()
        
        # Buffer zone (1.5 screens)
        buffer = viewport_height * 0.5
        visible_top = scroll_y - buffer
        visible_bottom = scroll_y + viewport_height + buffer
        
        # Identify visible items
        visible_indices = []
        for i, item in enumerate(self._gallery_layout_items):
            rect = item['rect']
            if rect.bottom() >= visible_top and rect.top() <= visible_bottom:
                visible_indices.append(i)
                
        # 1. Recycling/Cleanup: Hide widgets that are no longer visible
        visible_keys = set()
        for i in visible_indices:
             item = self._gallery_layout_items[i]
             fp = item['file_path']
             if fp: visible_keys.add(fp)
             
        for fp, widget in list(self._visible_widgets.items()):
            if fp not in visible_keys:
                widget.hide()
                # Recycle widget
                self._widget_pool.append(widget)
                del self._visible_widgets[fp]
                
        # 2. Create/Reuse widgets for visible items
        for i in visible_indices:
            item = self._gallery_layout_items[i]
            file_path = item['file_path']
            rect = item['rect']
            
            if not file_path:
                continue # Skip non-file items if any
            
            # Check if widget already exists
            if file_path in self._visible_widgets:
                w = self._visible_widgets[file_path]
                w.setGeometry(rect)
                w.show()
                # Ensure it has content if available (might have been recycled)
                continue
                
            # Get a widget from pool or create new
            if self._widget_pool:
                widget = self._widget_pool.pop()
            else:
                widget = ThumbnailLabel(self)
                # Configure widget style/callbacks if needed
                # widget.mousePressEvent = ... 
                # (Assuming click handling is done elsewhere or we need to add it)
                # We'll attach click handler here
                widget.mousePressEvent = lambda e, fp=file_path: self._on_thumbnail_click(fp)

            widget.setGeometry(rect)
            widget.show()
            self._visible_widgets[file_path] = widget
            
            # 3. Request Image Load
            # Check cache first
            bucket = self._get_best_bucket(rect.height())
            cache_key = (file_path, bucket)
            
            # Use unified cache check using file_path
            # But here we need specific size for gallery.
            # We implemented specialized LRUCache with tuples for gallery in main.py, 
            # let's stick to using standard cache but scale, OR use the _thumbnail_cache
            # The _thumbnail_cache usage in original code was valid.
            
            cached_pixmap = self._thumbnail_cache.get(cache_key)
            if cached_pixmap:
                 widget.setPixmap(cached_pixmap)
            else:
                 # Check main cache
                 if self.parent_viewer and hasattr(self.parent_viewer, 'image_cache'):
                     # Check thumbnail
                     thumb = self.parent_viewer.image_cache.get_thumbnail(file_path)
                     if thumb is not None:
                         # We have a numpy thumbnail, convert and scale on UI thread (fast enough for small thumbnails)
                         # OR queue it to worker?
                         # Better queue for smooth scroll.
                         self._queue_image_load(i, file_path, rect.width(), rect.height())
                         continue
                 
                 # Queue load
                 self._queue_image_load(i, file_path, rect.width(), rect.height())
                 
        self._check_and_hide_loading_if_visible_loaded()
        self._continue_loading_remaining_images()

    def _get_best_bucket(self, height):
        """Find best bucket for height"""
        for b in self._row_height_buckets:
            if b >= height:
                return b
        return self._row_height_buckets[-1]

    def _queue_image_load(self, index, file_path, w, h):
        """Queue task to load image"""
        if file_path in self._loading_tiles:
            return
            
        self._loading_tiles.add(file_path)
        
        # Create Task
        task = ImageLoadTask(index, file_path, w, h, self.loader_signal, self.parent_viewer, self._gallery_generation)
        self._active_tasks[file_path] = task
        
        # Start in thread pool
        self.thread_pool.start(task)

    def apply_thumbnail(self, index, q_image_or_pixmap, generation):
        """Slot: Image loaded from thread"""
        if generation != self._gallery_generation:
            return
            
        try:
            if index < 0 or index >= len(self._gallery_layout_items):
                return
                
            item = self._gallery_layout_items[index]
            file_path = item['file_path']
            
            # Remove from loading set
            if file_path in self._loading_tiles:
                self._loading_tiles.remove(file_path)
            if file_path in self._active_tasks:
                del self._active_tasks[file_path]
            
            # Convert QImage to QPixmap if needed
            if isinstance(q_image_or_pixmap, QImage):
                 pixmap = QPixmap.fromImage(q_image_or_pixmap)
            else:
                 pixmap = q_image_or_pixmap
                 
            if not pixmap or pixmap.isNull():
                 return

            # Update widget if visible
            if file_path in self._visible_widgets:
                self._visible_widgets[file_path].setPixmap(pixmap)
            
            # Cache it
            bucket = self._get_best_bucket(pixmap.height())
            self._thumbnail_cache.put((file_path, bucket), pixmap)
            
            self._check_and_hide_loading_if_visible_loaded()
            
        except Exception as e:
            self.logger.error(f"Error applying thumbnail: {e}")

    def _on_thumbnail_click(self, file_path):
        """Handle click on thumbnail"""
        if self.parent_viewer:
             # Call parent viewer's open method
             if hasattr(self.parent_viewer, 'load_image'):
                 self.parent_viewer.load_image(file_path)
                 
    def _check_and_hide_loading_if_visible_loaded(self):
        """Check if all visible images are loaded and hide loading message if so"""
        if not self._gallery_layout_items:
            self.hide_loading_message()
            return

        all_loaded = True
        has_visible = False
        
        if self._visible_widgets:
            for file_path, widget in self._visible_widgets.items():
                if not widget.isVisible():
                    continue
                has_visible = True
                # If widget has no pixmap (pixmap() returns None or null), it's not loaded
                if not widget.pixmap() or widget.pixmap().isNull():
                    # We can also check loading_tiles
                    if file_path in self._loading_tiles:
                         all_loaded = False
                         break
        else:
             if self._gallery_layout_items:
                 all_loaded = False
        
        if all_loaded:
            self.hide_loading_message()

    def _continue_loading_remaining_images(self):
         # Background loading logic - reduced priority
         # Implementation similar to original but using thread pool
         pass
