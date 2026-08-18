"""Update popup shows a short 'What's new' distilled from GitHub release notes."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "rawviewer_ui"))

from release_update import (  # noqa: E402
    fallback_whats_new,
    mock_update_result,
    notes_from_release,
    result_from_github_release,
    summarize_release_notes,
)


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
    for line in out.rstrip("…").splitlines():
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
    assert out.endswith("…")
    assert len(out) <= 601


def test_summarize_default_is_a_few_bullets() -> None:
    body = "\n".join(f"- feature {i}" for i in range(20))
    out = summarize_release_notes(body)
    assert out.endswith("…")
    assert out.count("• ") <= 5
    assert "feature 19" not in out


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


def test_mock_json_body_feeds_truncated_notes() -> None:
    """CI: GitHub-shaped JSON with a long body → short bullets, no network."""
    payload = {
        "tag_name": "v3.2.0",
        "name": "RAWviewer 3.2.0",
        "html_url": "https://github.com/markyip/RAWviewer/releases/tag/v3.2.0",
        "published_at": "2026-08-01T00:00:00Z",
        "body": "\n".join(
            ["## What's Changed", "- **Alpha** lands", "- Bravo fix"]
            + [f"- extra {i}" for i in range(30)]
        ),
    }
    res = result_from_github_release("3.0.2", payload)
    assert res["update_available"] is True
    assert res["latest"] == "v3.2.0"
    assert res["release_notes"].startswith("• ")
    assert "Alpha lands" in res["release_notes"]
    assert "extra 29" not in res["release_notes"]
    assert res["release_notes"].endswith("…")
    assert res["release_notes"].count("• ") <= 5


def test_whats_new_field_is_accepted() -> None:
    notes = notes_from_release({"tag_name": "v9.0.0", "whats_new": "- From JSON field"})
    assert "From JSON field" in notes


def test_empty_body_uses_310_fallback_not_32() -> None:
    assert "stability" in fallback_whats_new("3.1.0").lower()
    assert "3.0.2" in fallback_whats_new("v3.1.0")
    assert fallback_whats_new("3.2.0") == ""
    live = notes_from_release({"tag_name": "v3.1.0", "body": "- Live GitHub note"})
    assert "Live GitHub note" in live
    empty = notes_from_release({"tag_name": "v3.1.0", "body": ""})
    assert "stability" in empty.lower()
    empty_32 = result_from_github_release(
        "3.1.0", {"tag_name": "v3.2.0", "body": ""}
    )
    assert empty_32["release_notes"] == ""


def test_dialog_shows_notes_section_only_when_present() -> None:
    from PyQt6.QtWidgets import QApplication, QLabel
    from rawviewer_ui.release_update_dialog import ReleaseUpdateDialog

    _app = QApplication.instance() or QApplication([])

    def has_whats_new(dlg) -> bool:
        return any(
            isinstance(w, QLabel) and w.text() == "What's new"
            for w in dlg.findChildren(QLabel)
        )

    def has_notes_link(dlg) -> bool:
        return any(
            isinstance(w, QLabel) and "Release notes" in w.text()
            for w in dlg.findChildren(QLabel)
        )

    with_notes = ReleaseUpdateDialog(
        current="3.0.2", latest="3.1.0", release_notes="• A\n• B"
    )
    assert has_whats_new(with_notes)
    assert has_notes_link(with_notes)
    assert any(
        isinstance(w, QLabel) and w.objectName() == "releaseUpdateWhatsNew"
        and "A" in w.text() and "B" in w.text()
        for w in with_notes.findChildren(QLabel)
    )

    without = ReleaseUpdateDialog(current="3.0.2", latest="3.1.0", release_notes="")
    assert not has_whats_new(without), "empty notes must omit the section"
    assert has_notes_link(without)


def main() -> int:
    test_summarize_strips_markdown_and_bullets()
    test_summarize_drops_noise_and_truncates()
    test_summarize_default_is_a_few_bullets()
    test_summarize_empty_is_empty()
    test_mock_result_carries_notes_from_env()
    test_mock_json_body_feeds_truncated_notes()
    test_whats_new_field_is_accepted()
    test_empty_body_uses_310_fallback_not_32()
    test_dialog_shows_notes_section_only_when_present()
    print("PASS t_release_update_notes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
