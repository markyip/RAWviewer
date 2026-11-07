"""JustifiedGallery: adaptive justified gallery layout, assembled from mixin parts."""
from __future__ import annotations

from PyQt6.QtWidgets import QWidget

from .gallery_part1 import GalleryPart1
from .gallery_part2 import GalleryPart2
from .gallery_part3 import GalleryPart3
from .gallery_part4 import GalleryPart4
from .gallery_part5 import GalleryPart5
from .gallery_part6 import GalleryPart6
from .gallery_part7 import GalleryPart7


class JustifiedGallery(GalleryPart1, GalleryPart2, GalleryPart3, GalleryPart4, GalleryPart5, GalleryPart6, GalleryPart7, QWidget):
    """Adaptive justified gallery layout with high-performance virtualization.

    Optimized for large directories using binary search and widget pooling.
    """
