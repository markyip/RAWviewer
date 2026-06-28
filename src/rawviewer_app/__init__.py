"""RAWviewer application modules (split from monolithic main.py).

RAWImageViewer remains in ``main.py`` (PyInstaller entry point). Import submodules
directly: ``rawviewer_app.signals``, ``processing``, ``workers``, ``widgets``.
"""

__all__ = ["env", "signals", "processing", "workers", "widgets"]
