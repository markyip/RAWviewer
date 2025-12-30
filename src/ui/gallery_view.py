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
        
        # State Management
        self._building = False
        self._gallery_generation = 0
        self._active_tasks = {}
        self._metadata_cache = {}
        self._thumbnail_cache = LRUCache(10000) # Increased for smoother scrolling
        self._row_height_buckets = [180, 220, 260, 300, 340]
        self._path_to_index = {} # For faster lookup from signal
        
        # Threading: Use global manager
        self.load_manager = get_image_load_manager()
        self.load_manager.thumbnail_ready.connect(self.on_thumbnail_ready)
        
        # Timer Management (Reused to avoid GC)
        self._load_timer = QTimer(self)
        self._load_timer.setSingleShot(True)
        self._load_timer.timeout.connect(self.load_visible_images)
        
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._handle_resize_rebuild)
        
        # Scroll Performance
        self._last_scroll_y = -1
        self._last_scroll_time = 0
        self._current_scroll_speed = 0
        self._is_scrolling_fast = False
        self._scroll_optimize_threshold = 3000 # px/s
        
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
        
        # CRITICAL: Always run recycling, even if fast scrolling
        # but defer heavy image loading
        self._load_timer.start(50 if not self._is_scrolling_fast else 150)

    def build_gallery(self, bulk_metadata=None):
        """Build layout coordinates for all images (Justified)"""
        if self._building or not self.images: return
        self._building = True
        
        try:
            # Cleanup current state
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
            if net_width <= 0: return

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
                aspect = 1.333
                if isinstance(item, str):
                    m = self._metadata_cache.get(item)
                    if m and m.get('original_width') and m.get('original_height'):
                        w, h = m['original_width'], m['original_height']
                        orient = m.get('orientation', 1)
                        if orient in (5, 6, 7, 8): 
                            w, h = h, w
                        aspect = w/h
                        
                        # SANITY CHECK: If aspect is > 1 but we suspect it should be portrait
                        # (This is rare but helps with cache poisoning)
                        if aspect > 1.2 and orient == 1 and is_raw_file(item):
                             # Trust get_image_aspect_ratio more for RAW fallback
                             aspect = get_image_aspect_ratio(item)
                    else: 
                        aspect = get_image_aspect_ratio(item)
                else: aspect = item.width()/item.height() if item.height() > 0 else 1.333
                
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
            self.load_visible_images()
            
        finally:
            self._building = False

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

    def load_visible_images(self):
        """Virtualized loading with priority for scrolling smoothness and aggressive pre-fetching"""
        if self._building: return
        
        # Find Scroll Area Parent
        p = self.parent()
        while p and not isinstance(p, QScrollArea): p = p.parent()
        if not p: return

        v_port = p.viewport()
        scroll_y = p.verticalScrollBar().value()
        v_h = v_port.height()
        
        # 1. VISIBLE RANGE: What's on screen now
        buffer_rect = QRect(0, scroll_y, v_port.width(), v_h)
        visible_indices_items = self._get_visible_range(buffer_rect)
        visible_paths = {item['file_path'] for idx, item in visible_indices_items}
        
        # 2. PRE-FETCH RANGE: Aggressive look-ahead
        # Buffer 2 screens below and 1 screen above
        prefetch_rect = QRect(0, scroll_y - v_h, v_port.width(), v_h * 4)
        prefetch_indices_items = self._get_visible_range(prefetch_rect)
        
        # 3. Recycling Phase: Remove widgets far from view
        for path in list(self._visible_widgets.keys()):
            if path not in visible_paths:
                # Keep a small buffer of widgets near the screen
                w = self._visible_widgets.pop(path)
                w.hide()
                self._widget_pool.append(w)
                
        # 4. Defer heavy work if scrolling TOO fast, but still recycle
        if self._is_scrolling_fast:
            return

        # 5. Activation Phase for VISIBLE items
        load_tasks = []
        for idx, item in visible_indices_items:
            path = item['file_path']
            rect = item['rect']
            if not path: continue
            
            if path not in self._visible_widgets:
                w = self._widget_pool.pop() if self._widget_pool else ThumbnailLabel(self)
                w.file_path = path
                w.setGeometry(rect)
                w.setFixedSize(rect.size())
                w.mousePressEvent = lambda e, p=path: self.parent_viewer._gallery_item_clicked(p)
                
                cache_hit = False
                closest_bucket = min(self._row_height_buckets, key=lambda x: abs(x - rect.height()))
                cached = self._thumbnail_cache.get((path, closest_bucket))
                
                if cached:
                    w.setPixmap(cached)
                    w.setText("")
                    cache_hit = True
                else:
                    w.setText("Loading...")
                
                w.show()
                self._visible_widgets[path] = w
                
                if not cache_hit:
                    load_tasks.append((path, Priority.CURRENT))

        # 6. Aggressive Pre-fetch Phase
        for idx, item in prefetch_indices_items:
            path = item['file_path']
            if not path or path in visible_paths: continue
            
            # Only pre-fetch if not already in cache
            closest_bucket = min(self._row_height_buckets, key=lambda x: abs(x - item['rect'].height()))
            if not self._thumbnail_cache.get((path, closest_bucket)):
                load_tasks.append((path, Priority.PRELOAD_NEXT))

        # Start loading through central manager
        for path, priority in load_tasks[:48]: # Batch size
            self.load_manager.load_image(path, priority=priority, cancel_existing=False)

    def on_thumbnail_ready(self, file_path, thumbnail_data):
        """Handle thumbnail signal from central manager"""
        if file_path not in self._path_to_index: return
        
        # Convert numpy to QPixmap if needed
        if isinstance(thumbnail_data, np.ndarray):
            h, w = thumbnail_data.shape[:2]
            qimg = QImage(thumbnail_data.data, w, h, w * 3, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
        else:
            pixmap = thumbnail_data # Assume already QPixmap
            
        if not pixmap: return
        
        # Cache for all buckets to reduce future loads
        for b in self._row_height_buckets:
            self._thumbnail_cache.put((file_path, b), pixmap)
            
        # Update widget if visible
        if file_path in self._visible_widgets:
            w = self._visible_widgets[file_path]
            if w.file_path == file_path:
                w.setPixmap(pixmap)
                w.setText("")

        if file_path in self._active_tasks: del self._active_tasks[file_path]

    def scroll_to_image(self, file_path):
        """Sync scroll position to a specific image"""
        item = next((x for x in self._gallery_layout_items if x['file_path'] == file_path), None)
        if item:
            p = self.parent()
            while p and not isinstance(p, QScrollArea): p = p.parent()
            if p:
                p.verticalScrollBar().setValue(item['rect'].top() - self.MIN_SPACING)

    def set_images(self, images, bulk_metadata=None):
        """Update image list and force rebuild"""
        self.images = images
        self._gallery_generation += 1
        
        # Stop existing tasks in manager if needed (though manager handles concurrency)
        # self.load_manager.cancel_all_tasks() # Usually too aggressive for gallery
        self._active_tasks.clear()
        
        self._metadata_cache.clear()
        if hasattr(self.parent_viewer, 'scroll_area'):
             self.parent_viewer.scroll_area.verticalScrollBar().setValue(0)
             
        self.build_gallery(bulk_metadata)

    def _get_viewport_width(self):
        p = self.parent()
        while p and not isinstance(p, QScrollArea): p = p.parent()
        return p.viewport().width() if p else self.width()

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
