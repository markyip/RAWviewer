"""Composite ImageLoadManager assembling the manager mixin parts."""

from __future__ import annotations

from .manager_part1 import ImageLoadManagerPart1
from .manager_part2 import ImageLoadManagerPart2
from .manager_part3 import ImageLoadManagerPart3


class ImageLoadManager(ImageLoadManagerPart1, ImageLoadManagerPart2, ImageLoadManagerPart3):
    """Unified image load queue manager (composite of manager parts)."""
