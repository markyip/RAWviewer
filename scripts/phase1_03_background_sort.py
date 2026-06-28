"""Phase 1 item 3: background folder resort + semantic search capture-time resort."""
from __future__ import annotations

import os
import re
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PyQt6.QtCore import QCoreApplication, QThreadPool
from PyQt6.QtWidgets import QApplication

app = QApplication([]) if QCoreApplication.instance() is None else QApplication.instance()

from rawviewer_app.signals import FolderResortSignals, SemanticSearchResortSignals


def test_signal_signatures_include_generation_token() -> None:
    fr = FolderResortSignals()
    assert "int" in fr.ready.signal
    sr = SemanticSearchResortSignals()
    assert "int" in sr.ready.signal


def test_main_py_wiring() -> None:
    main_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
    with open(main_path, encoding="utf-8") as f:
        src = f.read()
    assert "def _apply_folder_resort" in src
    assert "def _schedule_semantic_search_resort" in src
    assert "def _apply_semantic_search_resort" in src
    assert "QThreadPool.globalInstance().start(_FolderResortWorker())" in src
    assert "newest_first = self.get_sort_preference()" in src
    assert "viewer.sort_files_by_capture_time(" in src
    assert "viewer.sort_image_files(" not in src.split("class _FolderResortWorker")[1].split("QThreadPool.globalInstance().start(_FolderResortWorker())")[0]
    assert "QThreadPool.globalInstance().start(_SemanticSearchResortWorker())" in src
    assert "FolderResortSignals" in src
    assert "SemanticSearchResortSignals" in src
    # Sync sort on UI thread during semantic search should be gone.
    body_start = src.index("ranked_paths = [h.file_path for h in hits]")
    body_end = src.index("if not ranked_paths:", body_start)
    block = src[body_start:body_end]
    assert "sort_files_by_capture_time" not in block


def test_folder_resort_worker_emits() -> None:
    received = []

    class ViewerStub:
        def sort_files_by_capture_time(self, paths, *, newest_first=True, file_stats=None, folder_path=None):
            time.sleep(0.05)
            ordered = list(reversed(paths)) if newest_first else list(paths)
            return ordered, {"meta": True}

    signals = FolderResortSignals()
    signals.ready.connect(
        lambda token, files, meta: received.append((token, list(files), dict(meta)))
    )
    viewer = ViewerStub()
    paths = ["a.cr3", "b.cr3", "c.cr3"]
    token = 1

    class Worker:
        def run(self):
            sorted_files, bulk_metadata = viewer.sort_files_by_capture_time(
                paths, newest_first=True
            )
            signals.ready.emit(token, sorted_files, bulk_metadata)

    from PyQt6.QtCore import QRunnable

    class R(QRunnable):
        def run(self):
            Worker().run()

    QThreadPool.globalInstance().start(R())
    deadline = time.time() + 3.0
    while not received and time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
    assert received, "folder resort worker did not emit"
    assert received[0][0] == 1
    assert received[0][1] == ["c.cr3", "b.cr3", "a.cr3"]


if __name__ == "__main__":
    test_signal_signatures_include_generation_token()
    test_main_py_wiring()
    test_folder_resort_worker_emits()
    print("OK: Phase 1 item 3 (background sort) tests passed")
