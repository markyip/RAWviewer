"""Core infrastructure for RAWviewer.

Public facade re-exporting the foundational building blocks used across the
application: configuration, domain errors, Qt event/signal contracts, and logging.
"""

from core import config, errors, events, logging

__all__ = ["config", "errors", "events", "logging"]
