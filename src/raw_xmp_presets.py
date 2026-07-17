"""Managed XMP presets (apply-from-file), analogous to Creative LUT management.

``.xmp`` files live under the app support ``presets/`` directory. Selecting one
in the Adjust panel copies its global crs: settings onto the current image
(and writes that image's sidecar). Unlike LUTs, the preset name is not stored
on the image — apply is a one-shot load of slider values.
"""

from __future__ import annotations

import os
import shutil
from typing import Dict


def presets_dir() -> str:
    root = ""
    try:
        from PyQt6.QtCore import QStandardPaths

        root = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppDataLocation
        )
    except Exception:
        root = ""
    if not root or os.path.basename(root.rstrip(os.sep)) in {"", "-", "-c"}:
        root = os.path.join(os.path.expanduser("~"), ".rawviewer")
    path = os.path.join(root, "presets")
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        path = os.path.join(os.path.expanduser("~"), ".rawviewer", "presets")
        os.makedirs(path, exist_ok=True)
    return path


def list_managed_presets() -> list[str]:
    """Basenames of managed ``.xmp`` files (sorted)."""
    d = presets_dir()
    names = [f for f in os.listdir(d) if f.lower().endswith(".xmp")]
    return sorted(names, key=str.lower)


def managed_preset_path(name: str) -> str:
    return os.path.join(presets_dir(), os.path.basename(name or ""))


def import_xmp_preset(src_path: str) -> str:
    """Copy an ``.xmp`` into the managed folder; return basename."""
    if not src_path or not os.path.isfile(src_path):
        raise FileNotFoundError(src_path)
    if not src_path.lower().endswith(".xmp"):
        raise ValueError("Only .xmp preset files are supported")
    # Validate: must parse as adjustments (empty is ok — still a valid XMP).
    from raw_adjustments import load_adjustments_from_xmp

    load_adjustments_from_xmp(src_path)
    dest_name = os.path.basename(src_path)
    dest = os.path.join(presets_dir(), dest_name)
    if os.path.abspath(src_path) != os.path.abspath(dest):
        shutil.copy2(src_path, dest)
    return dest_name


def remove_managed_preset(name: str) -> None:
    path = managed_preset_path(name)
    if os.path.isfile(path):
        os.remove(path)


def load_preset_adjustments(name: str) -> Dict[str, float]:
    """Load global edit settings from a managed preset basename."""
    from raw_adjustments import load_adjustments_from_xmp

    return load_adjustments_from_xmp(managed_preset_path(name))
