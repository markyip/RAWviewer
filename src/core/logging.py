"""Shared logging helpers for RAWviewer."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured under the RAWviewer hierarchy.

    Args:
        name: Logger name (conventionally the calling module name).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)


def share_logger() -> logging.Logger:
    """Return the shared ``rawviewer.share`` logger."""
    return logging.getLogger("rawviewer.share")
