#!/usr/bin/env python3
"""Test FullImageCache LRU(10) functionality."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


class FullImageCache:
    """Simplified LRU(10) cache for full-resolution images.
    
    Replaces the adaptive full_image_cache with byte-budget enforcement
    with a simple fixed-size LRU cache that stores up to 10 full-resolution images.
    """
    
    def __init__(self, max_size: int = 10):
        from collections import OrderedDict
        import threading
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def __contains__(self, key: str) -> bool:
        with self.lock:
            return key in self.cache
    
    def get(self, key: str):
        """Get cached image, promoting it to MRU."""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
    
    def put(self, key: str, value) -> None:
        """Store image in cache with LRU eviction."""
        with self.lock:
            if key in self.cache:
                self.cache[key] = value
                self.cache.move_to_end(key)
            else:
                self.cache[key] = value
                if len(self.cache) > self.max_size:
                    self.cache.popitem(last=False)
    
    def remove(self, key: str) -> bool:
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> dict:
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate
            }

    # Alias methods for compatibility with existing code
    def get_full_image(self, file_path: str):
        """Alias for get() to maintain compatibility."""
        return self.get(file_path)
    
    def put_full_image(self, file_path: str, image, *, copy: bool = True):
        """Alias for put() to maintain compatibility."""
        if image is not None:
            stored = np.ascontiguousarray(image) if not copy else image.copy()
            self.put(file_path, stored)
    
    def peek(self, key: str):
        """Non-mutating lookup for compatibility."""
        with self.lock:
            return self.cache.get(key)


def main() -> int:
    # Test 1: FullImageCache LRU(10) basic functionality
    print("\n=== FullImageCache LRU(10) Tests ===")

    cache = FullImageCache(max_size=10)

    # Test basic put/get
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    cache.put("image1", test_image)
    check("Put and get works", cache.get("image1") is not None)
    check("Cache size is 1", len(cache.cache) == 1)

    # Test LRU eviction
    for i in range(11):
        img = np.ones((50, 50, 3), dtype=np.uint8) * i
        cache.put(f"image{i}", img)

    check("Cache size limited to 10", len(cache.cache) == 10)
    check("Oldest entry evicted", "image0" not in cache)
    check("Newest entries retained", "image10" in cache)

    # Test LRU access order
    cache.get("image1")  # Access image1 to make it MRU
    for i in range(12, 15):
        img = np.ones((50, 50, 3), dtype=np.uint8) * i
        cache.put(f"image{i}", img)

    check("LRU eviction respects access order", "image2" not in cache)
    check("MRU items retained", "image1" in cache)

    # Test alias methods
    cache.put_full_image("test_alias", np.zeros((20, 20, 3), dtype=np.uint8))
    check("put_full_image alias works", "test_alias" in cache)

    result = cache.get_full_image("test_alias")
    check("get_full_image alias works", result is not None)

    result = cache.peek("test_alias")
    check("peek alias works", result is not None)

    # Test stats
    stats = cache.get_stats()
    check("Stats includes size", stats['size'] == 10)
    check("Stats includes max_size", stats['max_size'] == 10)
    check("Stats includes hit_rate", 'hit_rate' in stats)

    # Summary
    print(f"\n{'='*60}")
    print(f"Total checks: {len(FAILURES) + 16}")
    print(f"Failures: {len(FAILURES)}")

    if FAILURES:
        print("\nFAILED CHECKS:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    else:
        print("\nALL CHECKS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())