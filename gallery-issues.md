# JustifiedGallery Implementation Review

## Critical Issues

### 1. **Resize Width Update Timing (Already Identified)**
The `_last_viewport_width` update happens in the wrong place, causing resize detection to fail.

**Fix:** Move the update to `_handle_resize_rebuild()` after layout clearing but before rebuild.

### 2. **Missing `QRect` Import**
```python
visible_rect = QRect(scroll_x, scroll_y, viewport_width, viewport_height)
```
`QRect` is used but never imported. This will cause a `NameError`.

**Fix:** Add at the top of `load_visible_images()`:
```python
from PyQt6.QtCore import QRect
```

### 3. **Infinite Retry Loop in `load_visible_images()`**
At the end of `_process_load_queue()`:
```python
QTimer.singleShot(200, self.load_visible_images)
```
This creates an infinite loop that continuously calls `load_visible_images()` even when there's nothing to load.

**Fix:** Only retry if there are unloaded tiles with valid geometry:
```python
# Only retry if there are still unloaded tiles
unloaded_count = sum(1 for _, fp, _, _ in self.tiles if fp and fp not in self._loading_tiles)
if unloaded_count > 0:
    QTimer.singleShot(200, self.load_visible_images)
```

### 4. **Race Condition: Rebuild During Load**
When `set_images()` or resize triggers a rebuild, it clears `self.tiles` while background threads are still loading and trying to call `apply_thumbnail()` with old indices.

**Fix:** Add a generation counter to invalidate stale loads:
```python
# In __init__:
self._gallery_generation = 0

# In build_gallery() or set_images():
self._gallery_generation += 1
current_gen = self._gallery_generation

# When creating ImageLoadTask, pass generation:
task = ImageLoadTask(..., generation=current_gen)

# In apply_thumbnail():
def apply_thumbnail(self, index, image, generation):
    if generation != self._gallery_generation:
        return  # Stale load, ignore
```

## Moderate Issues

### 5. **Duplicate Detection Logic**
Multiple places check for duplicates (in queue, in loading set, in seen files). This is fragile and error-prone.

**Fix:** Centralize duplicate checking in `_process_load_queue()`:
```python
def _add_to_queue(self, index, file_path, target_width, target_height, is_priority):
    """Centralized method to add items to queue with duplicate checking"""
    if file_path in self._loading_tiles:
        return False
    if any(item[1] == file_path for item in self._load_queue):
        return False
    self._load_queue.append((index, file_path, target_width, target_height, is_priority))
    return True
```

### 6. **Cooldown Logic May Skip Necessary Rebuilds**
The cooldown in `resizeEvent()` blocks expansions but allows shrinks. However, if the user rapidly expands and shrinks, the expansion might be permanently lost.

**Fix:** Instead of blocking, accumulate the pending width:
```python
# Always update pending width (don't block)
self._pending_viewport_width = new_viewport_width

# But respect cooldown for actually triggering rebuild
if time_since_rebuild < 0.5:
    logger.debug(f"Rebuild cooldown active, will use latest width when timer fires")
    # Timer will pick up _pending_viewport_width when it fires
    return
```

### 7. **Memory Leak: Old Labels Not Properly Deleted**
When clearing layouts in `_handle_resize_rebuild()`, widgets are deleted but the `self.tiles` list still references them. Accessing deleted labels causes `RuntimeError`.

**Fix:** Clear tiles list when clearing layout:
```python
# In _handle_resize_rebuild():
# Clear tiles list BEFORE clearing layout
old_tiles = self.tiles
self.tiles = []

# Then clear layout...
```

### 8. **Geometry Validity Check Too Late**
In `load_visible_images()`, geometry is checked only after trying to access it:
```python
label_rect = label.geometry()
if label_rect.width() <= 0 or label_rect.height() <= 0:
    continue
```

**Fix:** Check if label is visible before accessing geometry:
```python
if not label.isVisible():
    continue
label_rect = label.geometry()
```

## Minor Issues

### 9. **Excessive Logging**
Many debug logs at INFO level will spam the console. Use DEBUG level:
```python
logger.debug(f"Processing load queue: {len(self._load_queue)} items")
```

### 10. **Magic Numbers**
Several hardcoded values without constants:
- `10` (threshold for width change)
- `200` (debounce delay)
- `120` (load timer delay)
- `500` (cache size limit)

**Fix:** Define as class constants:
```python
WIDTH_CHANGE_THRESHOLD = 10
RESIZE_DEBOUNCE_MS = 200
LOAD_DEBOUNCE_MS = 120
MAX_CACHE_SIZE = 500
```

### 11. **Inconsistent Error Handling**
Some methods use `try-except` extensively, others don't. Some log with `exc_info=True`, others don't.

**Fix:** Establish consistent error handling patterns.

### 12. **`_continue_processing_queue()` Called Redundantly**
Every exit point in `apply_thumbnail()` calls `_continue_processing_queue()`, even when nothing changed.

**Fix:** Call once at the end, or only when actually needed.

## Performance Concerns

### 13. **Repeated Queue Sorting**
`_process_load_queue()` sorts the entire queue on every call. For large queues, this is expensive.

**Fix:** Use two separate queues (priority and normal) or use `heapq` for efficient priority queue.

### 14. **Cache Lookup Inefficiency**
Cache key calculation happens multiple times for the same file:
```python
cache_key = self._get_cache_key(file_path, target_height)
if cache_key in self._thumbnail_cache:
    cached_pixmap = self._thumbnail_cache[cache_key]
```

**Fix:** Cache the cache key itself during layout phase.

### 15. **Synchronous Layout Clearing**
The while loop in `_handle_resize_rebuild()` blocks the UI thread:
```python
while self.container.count():
    item = self.container.takeAt(0)
    # ... more operations
```

**Fix:** Consider batching or deferring deletion to avoid UI freezes.

## Recommended Priority

1. **Critical:** Fix #2 (missing import), #3 (infinite loop), #4 (race condition)
2. **High:** Fix #1 (resize width timing), #7 (memory leak)
3. **Medium:** Fix #5 (duplicate detection), #6 (cooldown logic), #8 (geometry check)
4. **Low:** Issues #9-15 (code quality and optimization)

## Quick Win: Most Impactful Fix

Start with **Issue #3** (infinite loop) and **Issue #4** (race condition) - these likely cause the most visible problems and performance issues.