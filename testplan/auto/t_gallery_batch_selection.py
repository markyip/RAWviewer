"""Gallery batch selection: check overlay + select all/prune/sync-bar core.

Phase 1 pin. Covers:
- ThumbnailLabel check overlay: hit-test true over the top-left circle, false
  in the center; selection state is painted (set_gallery_selected must NOT call
  setStyleSheet); state survives pool-style rebinding (bool toggles back).
- Select all displayed / count; selection pruned when displayed set narrows.
- O(n) selection math: select-all on 10k paths stays well under 0.5 s
  (guards the O(n^2) norm-key expansion regression).
"""

import os
import sys
import threading
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QPoint
    from rawviewer_ui.widgets import ThumbnailLabel
    from main import RAWImageViewer, _norm_path

    app = QApplication.instance() or QApplication([])  # noqa: F841 (keeps the app alive)
    assert app is not None

    # --- overlay geometry / painting ---------------------------------------
    w = ThumbnailLabel()
    w.resize(200, 140)
    rect = w.selection_check_rect()
    check(
        "check rect is in the top-left corner",
        rect.left() <= 10 and rect.top() <= 10 and 12 <= rect.width() <= 28,
        str(rect),
    )
    check(
        "hit-test true over the circle center",
        w.hits_selection_check(rect.center()),
    )
    px = (rect.left(), 140 // 2)
    check(
        "hit-test false far from the circle",
        not w.hits_selection_check(QPoint(px[0], px[1])),
    )

    class _StyleCountLabel(ThumbnailLabel):
        setstyle_calls = 0

        def setStyleSheet(self, style):  # noqa: N802
            _StyleCountLabel.setstyle_calls += 1
            super().setStyleSheet(style)

    sc = _StyleCountLabel()
    base_calls = _StyleCountLabel.setstyle_calls
    sc.set_gallery_selected(True)
    sc.set_gallery_selected(False)
    check(
        "selection painting uses no setStyleSheet",
        _StyleCountLabel.setstyle_calls == base_calls,
    )
    sc._gallery_selected = False
    try:
        from PyQt6.QtGui import QPixmap

        sc.setPixmap(QPixmap())
        sc._gallery_selected = True
        sc.repaint()
        painted_ok = True
    except Exception as exc:  # pragma: no cover
        painted_ok = False
        check("painted overlay renders without error", False, str(exc))
    if painted_ok:
        check("painted overlay renders without error", True)
    sc._gallery_selected = False

    # --- selection core via borrowed methods --------------------------------

    class _GalleryStub:
        def refresh_gallery_selection_visuals(self):
            self.refreshed = True

    class _Host:
        def __init__(self, files, display=None):
            self.view_mode = "gallery"
            self.image_files = list(files)
            self._gallery_selected_paths = set()
            self._gallery_selection_anchor = None
            self._gallery_last_select_action = None
            self._display = list(display) if display is not None else list(files)
            self.gallery_justified = _GalleryStub()

        def _gallery_files_for_display(self):
            return list(self._display)

        def _is_burst_grouping_enabled(self):
            return False

    def borrow(name):
        return RAWImageViewer.__dict__[name]

    host = _Host([])
    host._gallery_set_selection_keys = types.MethodType(
        borrow("_gallery_set_selection_keys"), host
    )
    host._gallery_select_all_displayed = types.MethodType(
        borrow("_gallery_select_all_displayed"), host
    )
    host._prune_gallery_selection_to_displayed = types.MethodType(
        borrow("_prune_gallery_selection_to_displayed"), host
    )
    host._sync_gallery_selection_chrome = types.MethodType(
        borrow("_sync_gallery_selection_chrome"), host
    )
    host._sync_gallery_selection_bar = types.MethodType(
        borrow("_sync_gallery_selection_bar"), host
    )
    host._gallery_selected_expanded_norm_keys = types.MethodType(
        borrow("_gallery_selected_expanded_norm_keys"), host
    )
    host._gallery_selected_canonical_paths = types.MethodType(
        borrow("_gallery_selected_canonical_paths"), host
    )
    host._gallery_selected_canonical_paths_from_keys = types.MethodType(
        borrow("_gallery_selected_canonical_paths_from_keys"), host
    )
    host._sync_share_button_visibility = types.MethodType(
        borrow("_sync_share_button_visibility"), host
    )
    host._sync_compare_button_visibility = types.MethodType(
        borrow("_sync_compare_button_visibility"), host
    )
    host._gallery_has_selection = types.MethodType(
        borrow("_gallery_has_selection"), host
    )
    host._gallery_refresh_selection_ui = types.MethodType(
        borrow("_gallery_refresh_selection_ui"), host
    )

    files = [f"C:\\x\\f{i:04d}.jpg" for i in range(120)]
    host.image_files = list(files)
    host._display = list(files)
    host._gallery_select_all_displayed()
    check(
        "select all selects every displayed path",
        host._gallery_selected_paths == {_norm_path(p) for p in files},
    )

    # prune: displayed set narrows to a rating filter subset
    subset = files[:40]
    host._display = subset
    host._prune_gallery_selection_to_displayed()
    check(
        "selection pruned to displayed set",
        host._gallery_selected_paths == {_norm_path(p) for p in subset},
    )

    # canonical paths round-trip via O(n) map
    keys = host._gallery_selected_expanded_norm_keys()
    canonical = host._gallery_selected_canonical_paths()
    check(
        "canonical paths resolve all pruned selection",
        sorted(os.path.basename(p) for p in canonical)
        == sorted(os.path.basename(p) for p in subset),
        f"{len(canonical)}",
    )

    # --- perf pin: 10k select-all < 0.5 s -----------------------------------
    big = [f"C:\\x\\big\\i{i:05d}.jpg" for i in range(10000)]
    host.image_files = big
    host._display = big
    t0 = time.perf_counter()
    host._gallery_select_all_displayed()
    keys = host._gallery_selected_expanded_norm_keys()
    canonical = host._gallery_selected_canonical_paths()
    elapsed = time.perf_counter() - t0
    check("10k select-all+expand under 0.5s", elapsed < 0.5, f"{elapsed*1000:.0f} ms")
    check("10k count = 10000", len(keys) == 10000 and len(canonical) == 10000)

    # clear empties and bar sync does not explode
    host._gallery_set_selection_keys(set())
    check("clear empties selection", not host._gallery_has_selection())
    t0 = time.perf_counter()
    host._sync_gallery_selection_chrome()
    check(
        "bar sync cheap", time.perf_counter() - t0 < 0.5,
    )

    # --- source pins --------------------------------------------------------
    src = open(
        os.path.join(os.path.dirname(__file__), "..", "..", "src",
                     "rawviewer_ui", "widgets", "__init__.py"),
        encoding="utf-8",
    ).read()
    check(
        "no _STYLE_SELECTED / setStyleSheet in selection path",
        "_STYLE_SELECTED" not in src and "self.setStyleSheet(\n            self._STYLE_SELECTED" not in src,
    )
    bops_src = open(
        os.path.join(os.path.dirname(__file__), "..", "..", "src",
                     "rawviewer_app", "services", "batch.py"),
        encoding="utf-8",
    ).read()
    check(
        "no UI/cache imports in batch_ops (QPixmaps stay on UI thread)",
        "from PyQt6.QtGui import" not in bops_src and "from image_cache" not in bops_src,
    )
    from _source_utils import main_source

    vp = main_source()
    check(
        "TaskProgressDialog rename complete (no ExportProgressDialog left)",
        "ExportProgressDialog" not in vp
        and open(os.path.join(os.path.dirname(__file__), "..", "..", "src",
                              "rawviewer_app", "core", "widgets.py"),
                 encoding="utf-8").read().count("ExportProgressDialog") == 0,
    )
    check(
        "single export reuses export_one_file (no duplicated decode body)",
        "decode_raw_edit_base" not in vp or "export_one_file" in vp,
    )

    # --- batch_ops unit checks ----------------------------------------------
    import tempfile
    from rawviewer_app.services.batch import (
        BatchFileOpSignals, BatchFileOpWorker, BatchExportWorker,
        copy_one_file, discard_one_file, unique_destination,
    )

    with tempfile.TemporaryDirectory() as tmp:
        src_dir = os.path.join(tmp, "src")
        dst_dir = os.path.join(tmp, "dst")
        os.makedirs(src_dir)
        os.makedirs(dst_dir)
        a = os.path.join(src_dir, "a.arw")  # RAW name: sidecar resolution applies
        x = os.path.join(src_dir, "a.xmp")
        with open(a, "wb") as fh:
            fh.write(b"imagepayload")
        with open(x, "wb") as fh:
            fh.write(b"<xmp/>")
        os.utime(a, (1500000000, 1500000000))

        r = set()
        d1 = unique_destination(dst_dir, "a.jpg", r)
        d2 = unique_destination(dst_dir, "a.jpg", r)
        check(
            "unique_destination never repeats",
            os.path.basename(d1) == "a.jpg" and os.path.basename(d2) == "a_1.jpg",
            f"{os.path.basename(d1)}, {os.path.basename(d2)}",
        )

        copied = copy_one_file(a, dst_dir, set())
        check(
            "copy preserves mtime and source untouched",
            os.path.getmtime(copied) == 1500000000
            and os.path.isfile(a)
            and open(copied, "rb").read() == b"imagepayload",
        )

        copied_sidecar = copy_one_file(a, dst_dir, set())  # will copy a.xmp alongside
        check(
            "sidecar copied alongside",
            os.path.isfile(os.path.join(dst_dir, os.path.splitext(
                os.path.basename(copied_sidecar))[0] + ".xmp")),
        )

        disc = discard_one_file(a)
        check(
            "discard keeps naming rule and removes source",
            os.path.basename(disc) == "a.arw"
            and "Discard" in disc
            and not os.path.exists(a),
        )
        # second discard of a same-named file gets _discarded suffix
        with open(a, "wb") as fh:
            fh.write(b"x")
        disc2 = discard_one_file(a)
        check(
            "discard collision gets _discarded_1",
            os.path.basename(disc2) == "a_discarded_1.arw",
        )

        # worker over the pool: copy order + progress totals
        files = []
        for i in range(4):
            p = os.path.join(src_dir, f"b{i}.jpg")
            with open(p, "wb") as fh:
                fh.write(b"x")
            files.append(p)

        sig = BatchFileOpSignals()
        seen = []
        from PyQt6.QtCore import QEventLoop, QTimer

        loop = QEventLoop()
        state = {"done": None}

        def _fin(done, failed, cancelled, detail):
            state["done"] = (done, failed, cancelled, detail)
            loop.quit()

        sig.progress.connect(lambda d, t, n: seen.append((d, t, n)))
        sig.finished.connect(_fin)
        worker = BatchFileOpWorker(files, dst_dir, mode="copy", signals=sig)
        from PyQt6.QtCore import QThreadPool

        QThreadPool.globalInstance().start(worker)
        QTimer.singleShot(5000, loop.quit)
        loop.exec()
        done, failed, cancelled, detail = state["done"]
        check(
            "copy worker copies all + no failures",
            done == 4 and failed == 0 and not cancelled,
        )
        check("progress reached total", seen[-1][0] == 4 and seen[-1][1] == 4)
        check(
            "detail carries dest paths",
            all(d is not None and os.path.isfile(d) for _s, _e, d in detail),
        )

        # pre-cancelled run
        sig2 = BatchFileOpSignals()
        loop2 = QEventLoop()
        state2 = {}

        def _fin2(done, failed, cancelled, detail):
            state2["done"] = (done, failed, cancelled)
            loop2.quit()

        sig2.finished.connect(_fin2)
        ev = threading.Event()
        ev.set()
        worker2 = BatchFileOpWorker(files, dst_dir, mode="copy", signals=sig2, cancel_event=ev)
        QThreadPool.globalInstance().start(worker2)
        QTimer.singleShot(5000, loop2.quit)
        loop2.exec()
        done2, failed2, cancelled2 = state2["done"]
        check(
            "pre-cancelled worker does nothing",
            done2 == 0 and failed2 == 0 and cancelled2,
        )

        # batch export with stubbed per-file export: order + continue-on-error
        import rawviewer_app.services.batch as bops
        real_export = bops.export_one_file
        calls = []
        bops.export_one_file = lambda src, out, adj, fmt, pool, **kw: (
            (calls.append(src), RuntimeError("boom") if src.endswith("1.jpg") else None)[1],
            None,
        )
        sig3 = BatchFileOpSignals()
        loop3 = QEventLoop()
        state3 = {}

        def _fin3(done, failed, cancelled, detail):
            state3["done"] = (done, failed, cancelled)
            loop3.quit()

        sig3.finished.connect(_fin3)
        worker3 = BatchExportWorker(files, dst_dir, "jpeg", None, signals=sig3)
        QThreadPool.globalInstance().start(worker3)
        QTimer.singleShot(5000, loop3.quit)
        loop3.exec()
        bops.export_one_file = real_export
        done3, failed3, cancelled3 = state3["done"]
        check("export worker processes all in order", calls == files, str(len(calls)))
        check("export worker continues on error", done3 == 3 and failed3 == 1)

    # --- removal post-pass + pre-wiring --------------------------------------
    import image_cache as _image_cache

    class _StubCache:
        def __init__(self):
            self.invalidated = []

        def invalidate_file(self, p):
            self.invalidated.append(p)

    class _RemovalHost:
        def __init__(self, files):
            self.image_files = list(files)
            self.current_file_path = files[0] if files else None
            self.current_file_index = 0
            self._gallery_bookmarked_paths = set()
            self.status_bar = None
            self._batch_removal_anchor = files[0]
            self.corpus_dropped = []
            self.bulk_calls = []
            self.saves = 0

        def _drop_discarded_from_semantic_corpus(self, p):
            self.corpus_dropped.append(p)

        def _after_gallery_bulk_file_removal(self, anchor_path=None):
            self.bulk_calls.append(anchor_path)

        def schedule_save_session_state(self):
            self.saves += 1

        def show_error(self, *a, **k):
            self._errors = a

    rhost = _RemovalHost(files)
    orig_get_cache = _image_cache.get_image_cache
    _image_cache.get_image_cache = lambda: _StubCache()
    try:
        rhost._remove_files_from_active_image_list = types.MethodType(
            borrow("_remove_files_from_active_image_list"), rhost
        )
        after_removal = types.MethodType(borrow("_after_batch_removal"), rhost)
        detail = [
            (files[0], None, None),
            (files[1], None, None),
            (files[2], "boom", None),
            (files[3], None, None),
        ]
        after_removal("trash", detail)
    finally:
        _image_cache.get_image_cache = orig_get_cache
    check(
        "removal post-pass skips failed path, removes the rest",
        _norm_path(files[2]) in {
            _norm_path(p) for p in rhost.image_files
        }
        and all(
            _norm_path(p) != _norm_path(f)
            for p in rhost.image_files
            for f in (files[0], files[1], files[3])
        ),
        str([os.path.basename(p) for p in rhost.image_files]),
    )
    check(
        "failed path skipped for corpus drop",
        files[2] not in rhost.corpus_dropped and files[0] in rhost.corpus_dropped,
    )
    check("bulk removal ran with surviving anchor", rhost.bulk_calls)
    check("session saved once after removal", rhost.saves == 1)

    # pre-cancel + snapshot handoff before worker start
    phost = _RemovalHost(files)
    phost._stop_slideshow = lambda: None
    phost._batch_removal_anchor = None
    phost._cancelled = []
    phost._cancel_load_and_preload_for_path = lambda p: phost._cancelled.append(p)
    recorded = {}

    def _fake_start(kind, paths, dest_dir="", reveal=False, **kw):
        recorded["args"] = (kind, list(paths), dest_dir, reveal)

    phost._start_batch_op = _fake_start
    types.MethodType(borrow("_surviving_anchor_after_removal"), phost)
    phost._surviving_anchor_after_removal = types.MethodType(
        borrow("_surviving_anchor_after_removal"), phost
    )
    types.MethodType(borrow("_start_batch_removal"), phost)("discard", files)
    kind, rpaths, rdest, rreveal = recorded["args"]
    check(
        "pre-cancel touches every path before the worker starts",
        sorted(map(os.path.basename, phost._cancelled))
        == sorted(map(os.path.basename, files)),
        str(len(phost._cancelled)),
    )
    check(
        "removed snapshot handed to worker (no dest/reveal)",
        kind == "discard" and rdest == "" and rreveal is False and len(rpaths) == 4,
    )

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failing check(s)")
        return 1
    print("All checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
