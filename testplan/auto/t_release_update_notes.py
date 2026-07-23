"""Update popup shows a short 'What's new' distilled from GitHub release notes."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "rawviewer_ui"))

from release_update import mock_update_result, summarize_release_notes  # noqa: E402


def test_summarize_strips_markdown_and_bullets() -> None:
    body = (
        "## What's Changed\n"
        "- **Focus Stacking** all-in-focus merge\n"
        "* Camera `color` calibration\n"
        "\n"
        "### Fixes\n"
        "- Fixed HDR dialog crash\n"
    )
    out = summarize_release_notes(body)
    assert out, "expected non-empty summary"
    for line in out.splitlines():
        assert line.startswith("• "), f"line not bulleted: {line!r}"
    assert "**" not in out and "`" not in out and "#" not in out, out
    assert "Focus Stacking all-in-focus merge" in out
    assert "Fixed HDR dialog crash" in out


def test_summarize_drops_noise_and_truncates() -> None:
    body = "\n".join(
        ["## Notes", "---", "![img](x.png)", "<!-- comment -->",
         "**Full Changelog**: https://x/compare/a...b"]
        + [f"- item {i}" for i in range(40)]
    )
    out = summarize_release_notes(body, max_lines=8, max_chars=600)
    assert out.count("\n") + 1 <= 8, "exceeded max_lines"
    assert "Full Changelog" not in out and "![" not in out and "<!--" not in out
    assert len(out) <= 601


def test_summarize_empty_is_empty() -> None:
    assert summarize_release_notes("") == ""
    assert summarize_release_notes("\n\n---\n\n") == ""


def test_mock_result_carries_notes_from_env() -> None:
    os.environ["RAWVIEWER_MOCK_UPDATE_NOTES"] = "- New thing\n- Another thing"
    try:
        res = mock_update_result("9.9.9", "1.0.0")
        assert "New thing" in res["release_notes"]
        assert res["release_notes"].startswith("• ")
    finally:
        del os.environ["RAWVIEWER_MOCK_UPDATE_NOTES"]


def test_dialog_shows_notes_section_only_when_present() -> None:
    from PyQt6.QtWidgets import QApplication, QLabel
    from rawviewer_ui.release_update_dialog import ReleaseUpdateDialog

    _app = QApplication.instance() or QApplication([])

    def has_whats_new(dlg) -> bool:
        return any(
            isinstance(w, QLabel) and w.text() == "What's new"
            for w in dlg.findChildren(QLabel)
        )

    with_notes = ReleaseUpdateDialog(
        current="3.0.2", latest="3.1.0", release_notes="• A\n• B"
    )
    assert has_whats_new(with_notes)
    assert any(
        isinstance(w, QLabel) and "A" in w.text() and "B" in w.text()
        for w in with_notes.findChildren(QLabel)
    )

    without = ReleaseUpdateDialog(current="3.0.2", latest="3.1.0", release_notes="")
    assert not has_whats_new(without), "empty notes must omit the section"


def main() -> int:
    test_summarize_strips_markdown_and_bullets()
    test_summarize_drops_noise_and_truncates()
    test_summarize_empty_is_empty()
    test_mock_result_carries_notes_from_env()
    test_dialog_shows_notes_section_only_when_present()
    print("PASS t_release_update_notes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
