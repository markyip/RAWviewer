"""Staging for generative results, so nothing reaches the user's folder unasked.

A generation used to write a file next to the original the moment it came
back. That put every experiment -- including the ones a photographer takes one
look at and rejects -- into the folder they have to clean up later, and made
"try again" mean "produce a second file".

Here a result is staged in a temp directory instead. It is visible, it can be
generated from again, and it becomes a real file only when the user exports
it. Nothing else in the app sees a staged result: the folder listing, the
gallery and the sidecar layer all still deal exclusively in exported files.

Two rules give the model its shape:

* **A chain, not a file.** Generating again uses the newest staged image as
  its source, so instructions accumulate ("remove the bin", then "warm it
  up") the way a person expects, rather than each one restarting from the
  original. The chain is per source image, so switching photos never mixes
  one image's experiments into another's.
* **Staged means unsaved.** The temp directory is per app run and is deleted
  on exit. Anything not exported is gone, which is the right default for
  scratch work but has to be said out loud in the UI -- see
  ``has_unexported_work``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# Lossless, and the same format the export path writes, so exporting is a copy
# of pixels that already exist rather than a re-encode.
_STAGE_EXT = ".tif"


@dataclass
class GenerativeStep:
    """One staged result: the image, and what was asked for to get it."""

    path: str
    instruction: str
    index: int  # 1-based position in its chain
    provenance: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"Version {self.index}"


class GenerativeSession:
    """Per-app-run staging area for generative results.

    Thread-safe: results are staged from the worker thread while the GUI
    thread reads the chain to draw the page.
    """

    def __init__(self, root: Optional[str] = None):
        self._root = root
        self._chains: Dict[str, List[GenerativeStep]] = {}
        self._lock = threading.RLock()

    # -- storage ------------------------------------------------------

    @property
    def root(self) -> str:
        """The temp directory, created on first use rather than at startup."""
        with self._lock:
            if not self._root:
                self._root = tempfile.mkdtemp(prefix="rawviewer-generative-")
            os.makedirs(self._root, exist_ok=True)
            return self._root

    def cleanup(self) -> None:
        """Drop every staged result. Called on app exit."""
        with self._lock:
            root, self._root = self._root, None
            self._chains.clear()
        if root and os.path.isdir(root):
            shutil.rmtree(root, ignore_errors=True)

    # -- chains -------------------------------------------------------

    @staticmethod
    def _key(source_path: str) -> str:
        return os.path.abspath(source_path or "")

    def chain(self, source_path: str) -> List[GenerativeStep]:
        """Staged results for this image, oldest first. Never None."""
        with self._lock:
            return list(self._chains.get(self._key(source_path), ()))

    def latest(self, source_path: str) -> Optional[GenerativeStep]:
        chain = self.chain(source_path)
        return chain[-1] if chain else None

    def has_unexported_work(self) -> bool:
        """Whether anything staged would be lost by quitting now."""
        with self._lock:
            return any(self._chains.values())

    def stage(
        self,
        source_path: str,
        image: np.ndarray,
        instruction: str,
        provenance: Optional[dict] = None,
    ) -> GenerativeStep:
        """Write a result to the staging area and append it to the chain."""
        from generative_derived_file import write_derived_image

        key = self._key(source_path)
        with self._lock:
            index = len(self._chains.get(key, ())) + 1
        stem = os.path.splitext(os.path.basename(source_path or "image"))[0]
        # The key is hashed into the name so two files with the same basename
        # in different folders cannot collide in the one flat staging dir.
        name = f"{stem}-{abs(hash(key)) % 0xFFFFFF:06x}-v{index}{_STAGE_EXT}"
        path = os.path.join(self.root, name)
        write_derived_image(image, path)

        step = GenerativeStep(
            path=path,
            instruction=(instruction or "").strip(),
            index=index,
            provenance=dict(provenance or {}),
        )
        with self._lock:
            self._chains.setdefault(key, []).append(step)
        return step

    def discard(self, source_path: str) -> None:
        """Throw away this image's chain; the original becomes the source again."""
        key = self._key(source_path)
        with self._lock:
            steps = self._chains.pop(key, [])
        for step in steps:
            try:
                os.remove(step.path)
            except OSError:
                pass

    def undo_last(self, source_path: str) -> Optional[GenerativeStep]:
        """Drop the newest result, returning the one now current (or None)."""
        key = self._key(source_path)
        with self._lock:
            steps = self._chains.get(key)
            if not steps:
                return None
            dropped = steps.pop()
            if not steps:
                self._chains.pop(key, None)
        try:
            os.remove(dropped.path)
        except OSError:
            pass
        return self.latest(source_path)

    # -- export -------------------------------------------------------

    def export(self, source_path: str, dest_path: str) -> str:
        """Promote the newest staged result to a real file the user chose.

        Copies rather than re-encodes -- the staged file is already the
        lossless TIFF the export path would have written. The provenance
        sidecar is written next to the destination so an exported file keeps
        its lineage, exactly as an auto-created derived file used to.
        """
        from generative_derived_file import write_derived_sidecar

        step = self.latest(source_path)
        if step is None:
            raise ValueError("Nothing staged to export.")
        dest_dir = os.path.dirname(os.path.abspath(dest_path))
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)

        # Same write-then-rename discipline as write_derived_image: a folder
        # scan must never catch a half-copied file and cache it as corrupt.
        partial = dest_path + ".part"
        shutil.copyfile(step.path, partial)
        os.replace(partial, dest_path)

        if step.provenance:
            try:
                write_derived_sidecar(dest_path, step.provenance)
            except Exception:
                pass
        return dest_path

    def suggested_export_path(self, source_path: str) -> str:
        """Where Export should point by default: beside the original."""
        from generative_derived_file import derived_path_for

        return derived_path_for(source_path, ext=_STAGE_EXT)
