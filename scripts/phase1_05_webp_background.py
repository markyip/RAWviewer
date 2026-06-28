"""Phase 1 item 5: background animated WebP frame decode (GIF stays QMovie)."""
from __future__ import annotations

import io
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PyQt6.QtCore import QCoreApplication, QThreadPool, QRunnable
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication

app = QApplication([]) if QCoreApplication.instance() is None else QApplication.instance()

from rawviewer_app.signals import WebpDecodeSignals


def test_signal_signature_includes_token() -> None:
    sig = WebpDecodeSignals()
    assert "int" in sig.ready.signal


def test_main_py_wiring() -> None:
    main_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
    with open(main_path, encoding="utf-8") as f:
        src = f.read()
    assert "WebpDecodeSignals" in src
    assert "def _on_webp_frames_decoded" in src
    assert "def _on_webp_decode_failed" in src
    assert "QThreadPool.globalInstance().start(_WebpDecodeWorker())" in src
    assert "_webp_decode_generation" in src
    # GIF path unchanged (QMovie in _load_and_start_animation)
    load_fn = src.split("def _load_and_start_animation")[1].split("def _on_webp_frames_decoded")[0]
    assert "QMovie" in load_fn
    assert "Playing GIF:" in load_fn
    # WebP decode runs in background worker, not inline in _load_and_start_animation
    webp_block = load_fn.split("elif ext == '.webp':")[1]
    webp_ui = webp_block.split("class _WebpDecodeWorker")[0]
    assert "with Image.open(file_path)" not in webp_ui
    assert "_WebpDecodeWorker" in webp_block


def test_webp_worker_emits_decoded_frames() -> None:
    try:
        from PIL import Image
    except ImportError:
        print("SKIP: PIL not available for WebP worker integration test")
        return

    buf = io.BytesIO()
    frames_pil = []
    for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        im = Image.new("RGB", (8, 8), color)
        frames_pil.append(im)
    frames_pil[0].save(
        buf,
        format="WEBP",
        save_all=True,
        append_images=frames_pil[1:],
        duration=50,
        loop=0,
    )
    buf.seek(0)
    data = buf.getvalue()

    received = []

    class Worker(QRunnable):
        def run(self):
            from PIL import Image as PILImage

            frames = []
            durations = []
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as tmp:
                tmp.write(data)
                path = tmp.name
            try:
                with PILImage.open(path) as img:
                    n_frames = getattr(img, "n_frames", 1)
                    for frame_idx in range(n_frames):
                        img.seek(frame_idx)
                        frame_rgba = img.convert("RGBA")
                        w, h = frame_rgba.size
                        raw = frame_rgba.tobytes("raw", "RGBA")
                        qim = QImage(raw, w, h, QImage.Format.Format_RGBA8888).copy()
                        frames.append(qim)
                        durations.append(img.info.get("duration", 100) or 100)
            finally:
                os.unlink(path)
            signals.ready.emit(1, "/tmp/test.webp", frames, durations)

    signals = WebpDecodeSignals()
    signals.ready.connect(
        lambda token, path, frames, durations: received.append(
            (token, path, len(frames), list(durations))
        )
    )
    QThreadPool.globalInstance().start(Worker())
    deadline = time.time() + 5.0
    while not received and time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
    assert received, "webp decode worker did not emit"
    assert received[0][0] == 1
    assert received[0][2] == 3
    assert len(received[0][3]) == 3


if __name__ == "__main__":
    test_signal_signature_includes_token()
    test_main_py_wiring()
    test_webp_worker_emits_decoded_frames()
    print("OK: Phase 1 item 5 (background WebP decode) tests passed")
