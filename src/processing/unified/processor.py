"""UnifiedImageProcessor: RAW/JPEG processing pipeline, assembled from mixin parts."""
from __future__ import annotations

from .unified_part1 import UnifiedPart1
from .unified_part2 import UnifiedPart2
from .unified_part3 import UnifiedPart3
from .unified_part4 import UnifiedPart4
from .unified_part5 import UnifiedPart5
from .unified_part6 import UnifiedPart6
from .unified_part7 import UnifiedPart7


class UnifiedImageProcessor(UnifiedPart1, UnifiedPart2, UnifiedPart3, UnifiedPart4, UnifiedPart5, UnifiedPart6, UnifiedPart7):
    """Unified RAW/JPEG processing pipeline."""
