"""Processing package facade.

Provides access to the unified RAW/JPEG processing pipeline.

The unified pipeline is resolved lazily (PEP 562). Eagerly importing it here
made every ``processing.<submodule>`` import (e.g. ``processing.raw_decode.gil``)
load the whole pipeline, which re-entered ``enhanced_raw_processor`` and
``common_image_loader`` while they were still partially initialized.
"""

import importlib


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    unified = importlib.import_module("processing.unified")
    try:
        return getattr(unified, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
