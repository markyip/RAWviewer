"""Domain-specific exceptions raised across RAWviewer."""

from __future__ import annotations


class RAWViewerError(Exception):
    """Base class for all RAWviewer domain errors."""


class DecodeCancelled(RAWViewerError):
    """Raised mid-decode when the owning load task was cancelled.

    The chunked stages check ``cancel_check`` between chunks so a cancelled
    background/prefetch decode releases its RAW worker slot within ~10ms
    instead of running its remaining ~1s to completion -- this is what lets
    a navigation's CURRENT-priority decode start immediately.
    """


class IndexAborted(RAWViewerError):
    """Raised when background indexing is aborted (e.g. folder scope changed)."""


class ExportCancelled(RAWViewerError):
    """Raised when the user cancels an in-progress export."""
