import time
import os
import threading
import logging
import numpy as np
from typing import List, Dict, Any, Optional

from PyQt6.QtWidgets import QWidget, QFrame, QHBoxLayout, QVBoxLayout, QApplication, QLabel, QScrollArea, QSizePolicy
from PyQt6.QtCore import Qt, QTimer, QRect, QThreadPool, QSize, QRunnable, QObject, pyqtSignal, QPoint
from PyQt6.QtGui import QPixmap, QImage, QFont, QImageReader, QPainter, QBrush, QColor

from ui.widgets import ThumbnailLabel, ImageLoaded
from image_cache import LRUCache, get_image_cache
from image_load_manager import get_image_load_manager, Priority
from common_image_loader import get_image_aspect_ratio, load_pixmap_safe, is_raw_file

logger = logging.getLogger(__name__)

# -----------------------------
# Main Gallery Widget
# -----------------------------
class JustifiedGallery(QWidget):
    """
    Adaptive justified gallery layout with high-performance virtualization.
    Optimized for large directories using binary search and widget pooling.
    """
    TARGET_ROW_HEIGHT = 220 # Slightly larger for better viewing
    HEIGHT_TOLERANCE = 0.25
    MIN_SPACING = 6 # Slightly more breathability
    
    def __init__(self, images, parent=None):
        super().__init__(parent)
        self.parent_viewer = parent
        self.images = images
        
        # Virtualization Data
        self._gallery_layout_items = []  # List of {rect, file_path, aspect}
        self._visible_widgets = {}  # {file_path: ThumbnailLabel}
        self._widget_pool = []
        self._total_content_height = 0
        self._loading_label = None
        
        # State Management
        self._building = False
        self._gallery_generation = 0
        self._metadata_cache = {}    # path -> {width, height, aspect, orientation}
        self._path_to_index = {}
        
        # Performance/Caching
        self._thumbnail_cache = LRUCache(max_size=300) 
        self._row_height_buckets = [150, 200, 250, 300, 400] # Standard height buckets for caching
        self._active_tasks = set()
        
        # Threading: Use global manager
        self.load_manager = get_image_load_manager()
        self.load_manager.thumbnail_ready.connect(self.on_thumbnail_ready)
        self.load_manager.error_occurred.connect(self.on_load_error)
        
        # Timer Management (Reused to avoid GC)
        self._load_timer = QTimer(self)
        self._load_timer.setSingleShot(True)
        self._load_timer.timeout.connect(self.load_visible_images)
        
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._handle_resize_rebuild)
        
        # Metadata update rebuild timer
        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.timeout.connect(lambda: self.build_gallery())
        
        # Scroll Performance
        self._last_scroll_y = -1
        self._last_scroll_time = 0
        self._current_scroll_speed = 0
        self._is_scrolling_fast = False
        self._scroll_optimize_threshold = 3000 # px/s
        
        # Performance Tracking
        self._gallery_init_time = None
        self._gallery_build_time = None
        self._first_widget_time = None
        
        # Setup UI
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._loading_label = None
        
        QTimer.singleShot(50, self._delayed_initial_build)

    def _delayed_initial_build(self):
        """Initial build after widget measurements are valid"""
        self._setup_scroll_tracking()
        if self.width() > 0:
            QApplication.processEvents()
            self.build_gallery()
        else:
            QTimer.singleShot(50, self._delayed_initial_build)

    def _setup_scroll_tracking(self):
        """Connect to parent scrollbar"""
        try:
            # Find scroll area
            p = self.parent()
            while p and not isinstance(p, QScrollArea): p = p.parent()
            if p:
                scrollbar = p.verticalScrollBar()
                try: scrollbar.valueChanged.disconnect(self._on_scroll)
                except: pass
                scrollbar.valueChanged.connect(self._on_scroll)
        except: pass

    def _on_scroll(self, value):
        """Track scroll speed and prioritize recycling"""
        now = time.time()
        if self._last_scroll_y >= 0:
            dt = now - self._last_scroll_time
            if dt > 0.01:
                speed = abs(value - self._last_scroll_y) / dt
                self._current_scroll_speed = (self._current_scroll_speed * 0.4) + (speed * 0.6)
                self._is_scrolling_fast = self._current_scroll_speed > self._scroll_optimize_threshold
        
        self._last_scroll_y = value
        self._last_scroll_time = now
        
        # OPTIMIZATION: Use shorter delays for smoother scrolling
        # 16ms = ~60fps, 50ms for fast scrolling to reduce load
        self._load_timer.start(16 if not self._is_scrolling_fast else 50)

    def build_gallery(self, bulk_metadata=None):
        """Build layout coordinates for all images (Justified)"""
        if self._building or not self.images: return
        self._building = True
        _actual_build_start = time.time()
        
        try:
            # POOLING FIX: Return current visible widgets to pool before clearing dict
            for w in self._visible_widgets.values():
                w.hide()
                self._widget_pool.append(w)
            self._visible_widgets.clear()
            self._gallery_layout_items.clear()
            
            # Metadata Prep
            if bulk_metadata: self._metadata_cache.update(bulk_metadata)
            if not self._metadata_cache and self.parent_viewer and hasattr(self.parent_viewer, 'image_cache'):
                paths = [img for img in self.images if isinstance(img, str)]
                if paths:
                    self._metadata_cache = self.parent_viewer.image_cache.get_multiple_exif(paths)

            viewport_width = self._get_viewport_width()
            net_width = viewport_width - (self.MIN_SPACING * 2) - 16
            
            # Debug removed
            
            if net_width <= 0: 
                return

            current_y = 10
            row = []
            aspect_sum = 0
            
            def commit_row(r, a_sum, is_last):
                nonlocal current_y
                if not r: return
                spacing = (len(r) - 1) * self.MIN_SPACING
                row_h = self.TARGET_ROW_HEIGHT if is_last else (net_width - spacing) / a_sum
                row_h = max(self.TARGET_ROW_HEIGHT * 0.5, min(self.TARGET_ROW_HEIGHT * 2.0, row_h))
                
                curr_x = 10
                for i, (item, aspect) in enumerate(r):
                    w = int(row_h * aspect)
                    # MINIMUM WIDTH: Ensure even portrait images are visible
                    w = max(80, w)  # Minimum 80px width for visibility
                    if not is_last and i == len(r) - 1: w = net_width - (curr_x - 10)
                    
                    self._gallery_layout_items.append({
                        'rect': QRect(curr_x, int(current_y), int(w), int(row_h)),
                        'file_path': item if isinstance(item, str) else None,
                        'aspect': aspect
                    })
                    curr_x += w + self.MIN_SPACING
                current_y += row_h + self.MIN_SPACING

            # Layout Loop
            for item in self.images:
                aspect = 1.5  # Default 3:2 aspect ratio (common for cameras)
                if isinstance(item, str):
                    m = self._metadata_cache.get(item)
                    if m and m.get('original_width') and m.get('original_height'):
                        w, h = m['original_width'], m['original_height']
                        orient = m.get('orientation', 1)
                        if orient in (5, 6, 7, 8): 
                            w, h = h, w
                        aspect = w/h
                        
                        # CONSTRAINT: Limit extreme aspect ratios for proper layout
                        aspect = max(0.33, min(3.0, aspect))
                    # OPTIMIZATION: Don't call get_image_aspect_ratio() if no metadata
                    # This would block for every image without cached metadata
                    # Instead, use default and let thumbnails adjust when they load
                else: 
                    aspect = item.width()/item.height() if item.height() > 0 else 1.5
                    aspect = max(0.33, min(3.0, aspect))  # Apply constraint to QPixmap items too
                
                row.append((item, aspect))
                aspect_sum += aspect
                if (aspect_sum * self.TARGET_ROW_HEIGHT + (len(row)-1)*self.MIN_SPACING) >= net_width:
                    commit_row(row, aspect_sum, False)
                    row, aspect_sum = [], 0
            
            if row: commit_row(row, aspect_sum, True)
            
            # Update path lookup
            self._path_to_index = {item['file_path']: i for i, item in enumerate(self._gallery_layout_items)}
            
            self._total_content_height = int(current_y + 20)
            self.setMinimumHeight(self._total_content_height)
            self.update()
            
            # PERFORMANCE: Track build completion time
            actual_build_duration = (time.time() - _actual_build_start) * 1000
            print(f"[PERF][GALLERY] Gallery layout built in {actual_build_duration:.1f}ms", flush=True)
            
            if self._gallery_init_time:
                self._gallery_build_time = time.time()
                total_duration = (self._gallery_build_time - self._gallery_init_time) * 1000
                # logger.debug(f"[PERF][GALLERY] Time since folder load: {total_duration:.1f}ms")
            
        finally:
            self._building = False
            # Call this AFTER clearing building flag, otherwise it returns early
            self.load_visible_images()

    def _get_visible_range(self, buffer_rect):
        """High-performance binary search for visible range in layout items"""
        items = self._gallery_layout_items
        if not items: return []
        
        # Use binary search to find the first item that could be in buffer
        # This is VASTLY faster for large galleries
        import bisect
        
        # Key for bisect: items are sorted by rect.top()
        low = 0
        high = len(items) - 1
        start_idx = 0
        
        # Find first item where bottom > buffer.top
        y_top = buffer_rect.top()
        # BOUNDARY FIX: Ensure we don't skip items at the very top
        y_top = max(0, y_top - 50)  # Larger buffer to catch edge cases and prevent blank views
        while low <= high:
            mid = (low + high) // 2
            if items[mid]['rect'].bottom() < y_top:
                low = mid + 1
            else:
                start_idx = mid
                high = mid - 1
        
        visible = []
        for i in range(start_idx, len(items)):
            item = items[i]
            if item['rect'].top() > buffer_rect.bottom():
                break # We've passed the buffer zone
            visible.append((i, item))
        return visible

    def _get_cache_key(self, file_path, height):
        """Map height to closest bucket for better cache hit rate"""
        bucket = min(self._row_height_buckets, key=lambda x: abs(x - height))
        return (file_path, bucket)

    def load_visible_images(self, force_all=False):
        """Create/update widgets for visible area and request image loads"""
        if self._building and not force_all: return # Fast return during construction
        
        # Find Scroll Area Parent
        p = self.parent()
        while p and not isinstance(p, QScrollArea): p = p.parent()
        if not p: return # No scroll area parent found
        
        v_port = p.viewport()
        v_w = v_port.width() if v_port and v_port.width() > 0 else self.width()
        v_h = v_port.height() if v_port and v_port.height() > 0 else 800
        
        scroll_y = p.verticalScrollBar().value()
        visible_rect = QRect(0, scroll_y, v_w, v_h)
        
        # Buffer: Instantiate widgets for 50px (boundary safety)
        # Prefetch: Identify images to load for 2.0 screen heights
        buffer_rect = visible_rect.adjusted(0, -50, 0, 50) 
        prefetch_rect = visible_rect.adjusted(0, -v_h, 0, v_h * 2) 
        
        visible_items = self._get_visible_range(buffer_rect)
        prefetch_items = self._get_visible_range(prefetch_rect)
        
        visible_paths = {item['file_path'] for idx, item in visible_items}
        
        # 1. Recycling: Hide widgets that are no longer in buffer
        to_remove = []
        for path, w in self._visible_widgets.items():
            if path not in visible_paths:
                to_remove.append(path)
        
        for path in to_remove:
            w = self._visible_widgets.pop(path)
            w.hide()
            self._widget_pool.append(w)
            
        # 2. Display: Ensure widgets exist and are correctly positioned
        self.setUpdatesEnabled(False)
        try:
            for idx, item in visible_items:
                path = item['file_path']
                rect = item['rect']
                h = rect.height()
                
                if not path: continue
                
                if path not in self._visible_widgets:
                    w = self._widget_pool.pop() if self._widget_pool else ThumbnailLabel(self)
                    w.file_path = path
                    w.mousePressEvent = lambda e, p=path: self.parent_viewer._gallery_item_clicked(p)
                    self._visible_widgets[path] = w
                else:
                    w = self._visible_widgets[path]
                
                # Common update for both new and existing
                w.setGeometry(rect)
                w.setFixedSize(rect.size())
                if not w.isVisible():
                    w.show()
                
                # Apply cached image immediately if available
                cache_key = self._get_cache_key(path, h)
                cached = self._thumbnail_cache.get(cache_key)
                if cached:
                    w.setPixmap(cached)
                    w.setText("")
                else:
                    w.setText("Loading...")
                
                # PERFORMANCE: Track first widget display time (Time-to-Interactive)
                if self._gallery_init_time and not self._first_widget_time:
                    self._first_widget_time = time.time()
                    tti_duration = (self._first_widget_time - self._gallery_init_time) * 1000
                    print(f"[PERF][GALLERY] ✅ First widget displayed in {tti_duration:.1f}ms (TTI: folder load -> visible)", flush=True)
                    self.hide_loading_message()
        finally:
            self.setUpdatesEnabled(True)
            
        # 3. Loading: Prioritized background load for prefetch area
        manager = get_image_load_manager()
        for idx, item in prefetch_items:
            file_path = item['file_path']
            rect = item['rect']
            h = rect.height()
            
            cache_key = self._get_cache_key(file_path, h)
            if file_path not in self._active_tasks and not self._thumbnail_cache.get(cache_key):
                self._active_tasks.add(file_path)
                # PRIORITY: Visible images get high priority
                priority = Priority.CURRENT if visible_rect.intersects(rect) else Priority.BACKGROUND
                manager.load_image(file_path, priority, load_thumbnail_only=True)

    def on_thumbnail_ready(self, file_path, thumbnail_data):
        """Handle thumbnail signal from central manager"""
        # CRITICAL SAFEGARD: Ensure widget is still valid and image is relevant
        if not hasattr(self, '_path_to_index') or self._path_to_index is None:
            return
        if file_path not in self._path_to_index:
            return
        
        # Guard against processing if the widget is being destroyed
        try:
            # Convert numpy to QPixmap if needed
            if isinstance(thumbnail_data, np.ndarray):
                h, w = thumbnail_data.shape[:2]
                # CRITICAL FIX: Convert memoryview to bytes for QImage constructor and ensure 24-bit RGB
                bytes_per_line = 3 * w
                qimg = QImage(thumbnail_data.data.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
                pixmap = QPixmap.fromImage(qimg)
            else:
                pixmap = thumbnail_data # Assume already QPixmap
                
            if not pixmap or pixmap.isNull(): return
            
            # ASPECT RATIO UPDATE: Use thumbnail as the source of truth
            pix_w, pix_h = pixmap.width(), pixmap.height()
            if pix_h > 0:
                new_aspect = pix_w / pix_h
                old_m = self._metadata_cache.get(file_path, {})
                old_aspect = old_m.get('aspect', 1.5)
                
                # If aspect ratio changed significantly (e.g. from 1.5 landscape guess to 0.66 portrait)
                if abs(new_aspect - old_aspect) > 0.1:
                    self._metadata_cache[file_path] = {
                        'original_width': pix_w,
                        'original_height': pix_h,
                        'aspect': new_aspect,
                        'orientation': 1
                    }
                    self._rebuild_timer.start(300)
            
            # Cache for all buckets
            for b in self._row_height_buckets:
                self._thumbnail_cache.put((file_path, b), pixmap)
                
            # Update widget if visible
            if hasattr(self, '_visible_widgets') and file_path in self._visible_widgets:
                w = self._visible_widgets[file_path]
                if w and w.file_path == file_path:
                    w.setPixmap(pixmap)
                    w.setText("")
        except RuntimeException:
            # Widget might be deleted
            return
        except Exception as e:
            logger.debug(f"[GALLERY] Error in on_thumbnail_ready: {e}")
        finally:
            if hasattr(self, '_active_tasks') and file_path in self._active_tasks: 
                self._active_tasks.discard(file_path)

    def on_load_error(self, file_path, error_msg):
        """Handle failure to load an image"""
        if file_path in self._active_tasks:
            self._active_tasks.discard(file_path)
            
    def center_on_image(self, file_path):
        """Sync scroll position to a specific image"""
        item = next((x for x in self._gallery_layout_items if x['file_path'] == file_path), None)
        if item:
            p = self.parent()
            while p and not isinstance(p, QScrollArea): p = p.parent()
            if p:
                p.verticalScrollBar().setValue(item['rect'].top() - self.MIN_SPACING)

    def set_images(self, images, bulk_metadata=None):
        """Update image list and force rebuild"""
        print(f"[DEBUG][JustifiedGallery] set_images called with {len(images)} images", flush=True)
        
        # PERFORMANCE: Track gallery initialization time
        self._gallery_init_time = time.time()
        print(f"[PERF][GALLERY] Gallery view initiated at {self._gallery_init_time:.3f}", flush=True)
        
        self.images = images
        self._gallery_generation += 1
        
        # Stop existing tasks in manager to prioritize new folder
        manager = get_image_load_manager()
        manager.cancel_all_tasks()
        self._active_tasks.clear()
        
        # OPTIMIZATION: Don't clear metadata cache - we want to preserve it for fast rebuilds
        # self._metadata_cache.clear()  # REMOVED - this was causing slow rebuilds
        
        # Reset performance tracking for new gallery view
        self._gallery_build_time = None
        self._first_widget_time = None
        
        # REMOVED: Don't reset scroll position when returning to gallery
        # This preserves the user's viewing position when switching between views
        # Robustly find parent scroll area using parent_viewer reference or hierarchy
        # parent_scroll = None
        # if hasattr(self, 'parent_viewer') and self.parent_viewer and hasattr(self.parent_viewer, 'gallery_scroll'):
        #     parent_scroll = self.parent_viewer.gallery_scroll
        # 
        # if not parent_scroll:
        #     try:
        #         # Fallback: traverse up
        #         p = self.parent()
        #         while p and not isinstance(p, QScrollArea): p = p.parent()
        #         if p: parent_scroll = p
        #     except: pass
        #         
        # if parent_scroll and isinstance(parent_scroll, QScrollArea):
        #     parent_scroll.verticalScrollBar().setValue(0)
        #     parent_scroll.horizontalScrollBar().setValue(0)
        #     print(f"[DEBUG][JustifiedGallery] Scroll reset complete", flush=True)
        # else:
        #     print(f"[DEBUG][JustifiedGallery] Scroll reset SKIPPED (parent_scroll={parent_scroll})", flush=True)
              
        print(f"[DEBUG][JustifiedGallery] Calling build_gallery...", flush=True)
        self.build_gallery(bulk_metadata)

    def _get_viewport_width(self):
        p = self.parent()
        while p and not isinstance(p, QScrollArea): p = p.parent()
        w = p.viewport().width() if p else self.width()
        print(f"[DEBUG][JustifiedGallery] _get_viewport_width: {w} (parent={bool(p)}, self.width={self.width()})", flush=True)
        return w

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_timer.start(100) # Fast debounce

    def _handle_resize_rebuild(self):
        if self.width() > 0: self.build_gallery()

    def paintEvent(self, event):
        """Draw ultra-fast dark placeholders while scrolling"""
        painter = QPainter(self)
        brush = QBrush(QColor(30, 30, 30))
        for item in self._gallery_layout_items:
            rect = item['rect']
            if event.rect().intersects(rect):
                if item['file_path'] not in self._visible_widgets:
                    painter.fillRect(rect, brush)

    def wheelEvent(self, event):
        super().wheelEvent(event)
        self._load_timer.start(50)

    # -----------------------------
    # Loading Overlay
    # -----------------------------
    def show_loading_message(self, message):
        """Show loading overlay with robustness"""
        try:
            if not hasattr(self, '_loading_label') or not self._loading_label:
                self._loading_label = QLabel(self)
                self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self._loading_label.setStyleSheet("""
                    QLabel {
                        background-color: rgba(30, 30, 30, 220);
                        color: white;
                        border-radius: 8px;
                        padding: 15px 25px;
                        font-size: 14px;
                        font-weight: bold;
                        border: 1px solid rgba(255, 255, 255, 0.1);
                    }
                """)
                self._loading_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            
            self._loading_label.setText(message)
            self._loading_label.adjustSize()
            
            # Center the label relative to visible area or widget
            x = (self.width() - self._loading_label.width()) // 2
            
            # Use visible region for better vertical centering in large scroll area
            v_rect = self.visibleRegion().boundingRect()
            if not v_rect.isEmpty():
                y = v_rect.y() + (v_rect.height() - self._loading_label.height()) // 2
            else:
                y = (self.height() - self._loading_label.height()) // 2
                
            self._loading_label.move(x, max(0, y))
            self._loading_label.show()
            self._loading_label.raise_()
        except Exception as e:
            print(f"[GALLERY] Error showing loading message: {e}")

    def hide_loading_message(self):
        """Hide loading overlay safely"""
        try:
            if hasattr(self, '_loading_label') and self._loading_label:
                self._loading_label.hide()
        except:
            pass
