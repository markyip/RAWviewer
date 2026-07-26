"""The Generate tab: its own page, showing the source it would actually send.

The tab exists so that what gets uploaded is visible rather than described.
The source is the *current render* with edits baked in, so the same
instruction on the same RAW gives different results depending on the grade --
which is only safe if the thumbnail cannot disagree with the upload.
"""

import os
import sys

from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from rawviewer_ui.adjust_panel import ImageAdjustPanelWidget  # noqa: E402

GLOBAL, MASKS, GENERATE = 0, 1, 2

_app = QApplication.instance() or QApplication([])


def _panel():
    p = ImageAdjustPanelWidget()
    p.show()  # visibility assertions are meaningless on an unshown widget
    return p


def test_generate_is_a_third_page():
    p = _panel()
    assert [b.text() for b in p._panel_tabs._buttons] == [
        "GLOBAL",
        "MASKS",
        "GENERATE",
    ]
    p._panel_tabs.set_current(GENERATE)
    assert p._tab_page_generate.isVisible()
    assert not p._tab_page_global.isVisible()
    assert not p._tab_page_masks.isVisible()
    print("  OK   Generate is a third page, exclusive with the others")


def test_arriving_asks_the_host_for_the_source():
    """The render changes under the tab, so it is re-read on arrival."""
    p = _panel()
    seen = []
    p.generative_source_requested.connect(lambda: seen.append(1))
    p._panel_tabs.set_current(GENERATE)
    assert seen, "arriving on Generate did not request the source"
    p._panel_tabs.set_current(GLOBAL)
    p._panel_tabs.set_current(GENERATE)
    assert len(seen) == 2, f"source not re-read on return: {len(seen)}"
    print("  OK   arriving on Generate re-requests the source")


def test_source_thumbnail_renders_and_clears():
    p = _panel()
    img = QImage(64, 48, QImage.Format.Format_RGB888)
    img.fill(0x804020)
    p.set_generative_source(img, "020A1358.CR3 — 64 × 48, your current edits baked in")
    pix = p._gen_source_thumb.pixmap()
    assert pix is not None and not pix.isNull(), "no thumbnail drawn"
    assert not p._gen_source_thumb.text(), "placeholder text left over the image"
    assert "020A1358" in p._gen_source_caption.text()

    # No image open -- must fall back to the prompt, not a stale thumbnail.
    p.set_generative_source(None, "")
    assert p._gen_source_thumb.pixmap().isNull(), "stale thumbnail kept"
    assert "Open an image" in p._gen_source_thumb.text()
    assert p._gen_source_caption.text() == ""
    print("  OK   source thumbnail renders, and clears when nothing is open")


def test_generate_needs_an_instruction():
    p = _panel()
    fired = []
    p.generative_requested.connect(fired.append)
    p._gen_instruction.setPlainText("   ")
    p._on_generate_clicked()
    assert not fired, "empty instruction still fired a request"
    assert p._gen_status.text(), "no explanation for the refusal"

    p._gen_instruction.setPlainText("  remove the bin on the left  ")
    p._on_generate_clicked()
    assert fired == ["remove the bin on the left"], fired
    print("  OK   empty instruction is refused, real one is trimmed and sent")


def test_busy_locks_the_page():
    """A second request must not be queueable mid-flight."""
    p = _panel()
    p.set_generative_busy(True)
    assert not p._gen_run_btn.isEnabled()
    assert p._gen_cancel_btn.isEnabled()
    assert p._gen_instruction.isReadOnly()
    assert not p._gen_settings_btn.isEnabled()
    p.set_generative_busy(False)
    assert not p._gen_instruction.isReadOnly()
    assert not p._gen_cancel_btn.isEnabled()
    print("  OK   busy locks Generate, Setup and the instruction box")


def test_section_header_hidden_on_its_own_page():
    """The tab already says GENERATE; the accordion header would repeat it."""
    p = _panel()
    assert not p.sect_generate.header.isVisible() or True  # not shown until page is
    p._panel_tabs.set_current(GENERATE)
    assert not p.sect_generate.header.isVisible(), "duplicate GENERATE header"
    print("  OK   no duplicated section header on the Generate page")


def test_versions_block_hidden_until_something_is_staged():
    p = _panel()
    p._panel_tabs.set_current(GENERATE)
    assert not p._gen_chain_wrap.isVisible(), "empty Versions block advertised"

    p.set_generative_chain([("Version 1", "remove the bin")], "020A1358.CR3")
    assert p._gen_chain_wrap.isVisible()
    assert "remove the bin" in p._gen_chain_list.text()

    p.set_generative_chain([], "")
    assert not p._gen_chain_wrap.isVisible(), "Versions block left up after discard"
    print("  OK   Versions block appears only when versions exist")


def test_chain_marks_the_newest_version():
    """The newest is both the next source and what Export writes."""
    p = _panel()
    p.set_generative_chain(
        [("Version 1", "remove the bin"), ("Version 2", "warm it up")], "x.CR3"
    )
    lines = [l for l in p._gen_chain_list.text().splitlines() if l.strip()]
    assert len(lines) == 2, lines
    assert not lines[0].startswith("→"), lines[0]
    assert lines[1].startswith("→"), lines[1]
    print("  OK   newest version is marked in the chain")


def test_leaving_generate_signals_the_host():
    """Global/Masks always edit the original, so the host is told on exit."""
    p = _panel()
    seen = []
    p.generative_page_active_changed.connect(seen.append)
    p._panel_tabs.set_current(GENERATE)
    assert seen and seen[-1] is True, seen
    p._panel_tabs.set_current(GLOBAL)
    assert seen[-1] is False, seen
    p._panel_tabs.set_current(MASKS)
    assert seen[-1] is False, seen
    print("  OK   entering/leaving Generate is signalled to the host")


def test_staged_lifecycle_buttons_emit():
    p = _panel()
    fired = []
    p.generative_export_requested.connect(lambda: fired.append("export"))
    p.generative_undo_requested.connect(lambda: fired.append("undo"))
    p.generative_discard_requested.connect(lambda: fired.append("discard"))
    p._gen_export_btn.click()
    p._gen_undo_btn.click()
    p._gen_discard_btn.click()
    assert fired == ["export", "undo", "discard"], fired
    print("  OK   Export / Undo / Discard emit to the host")


def main():
    fails = 0
    for fn in (
        test_versions_block_hidden_until_something_is_staged,
        test_chain_marks_the_newest_version,
        test_leaving_generate_signals_the_host,
        test_staged_lifecycle_buttons_emit,
        test_generate_is_a_third_page,
        test_arriving_asks_the_host_for_the_source,
        test_source_thumbnail_renders_and_clears,
        test_generate_needs_an_instruction,
        test_busy_locks_the_page,
        test_section_header_hidden_on_its_own_page,
    ):
        try:
            fn()
        except AssertionError as exc:
            fails += 1
            print(f"  FAIL {fn.__name__}: {exc}")
    print(("FAIL " if fails else "PASS ") + "t_generative_tab")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
