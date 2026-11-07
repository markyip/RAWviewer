"""SemanticImageIndex: SQLite-backed CLIP embedding index, assembled from mixin parts."""
from __future__ import annotations

from .engine_part1 import EnginePart1
from .engine_part2 import EnginePart2
from .engine_part3 import EnginePart3
from .engine_part4 import EnginePart4
from .engine_part5 import EnginePart5
from .engine_part6 import EnginePart6
from .engine_part7 import EnginePart7


class SemanticImageIndex(EnginePart1, EnginePart2, EnginePart3, EnginePart4, EnginePart5, EnginePart6, EnginePart7):
    """SQLite-backed CLIP embedding index for local semantic search."""
