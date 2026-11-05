"""Gallery copy-path button: geometry, hit-test, painted hover, clipboard.

Covers:
- copy_path_rect sits in the bottom-right and never overlaps the selection check
- hit-test is true over the button, false at the tile centre
- hover painting renders without error and never re-polishes styles
- leaving the tile clears the hover state
- the cached copy glyph is a real pixmap
- RAWImageViewer._copy_paths_to_clipboard puts the absolute path on the clipboard
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QPoint, QEvent
    from PyQt6.QtGui import QPixmap, QColor

    app = QApplication.instance() or QApplication([])  # noqa: F841 (keeps the app alive)
    assert app is not None

    from rawviewer_ui.widgets import ThumbnailLabel

    w = ThumbnailLabel()
    w.resize(200, 140)

    rect = w.copy_path_rect()
    check(
        "copy button sits in the bottom-right corner",
        rect.right() <= 200
        and rect.bottom() <= 140
        and rect.left() > 200 // 2
        and rect.top() > 140 // 2,
        str(rect),
    )
    check(
        "copy button does not overlap the selection check",
        not rect.intersects(w.selection_check_rect()),
        f"{rect} vs {w.selection_check_rect()}",
    )
    check(
        "hit-test true over the button centre",
        w.hits_copy_path_button(rect.center()),
    )
    check(
        "hit-test false at the tile centre",
        not w.hits_copy_path_button(QPoint(100, 70)),
    )

    # --- painted hover ------------------------------------------------------
    class _StyleCountLabel(ThumbnailLabel):
        setstyle_calls = 0

        def setStyleSheet(self, style):  # noqa: N802
            _StyleCountLabel.setstyle_calls += 1
            super().setStyleSheet(style)

    sc = _StyleCountLabel()
    sc.resize(200, 140)
    pm = QPixmap(200, 140)
    pm.fill(QColor("#333333"))
    sc.set_original_pixmap(pm)
    sc.file_path = "C:/photos/sample.ARW"
    base_calls = _StyleCountLabel.setstyle_calls
    sc._set_tile_hovered(True)
    sc._set_copy_hover(True)
    try:
        sc.repaint()
        painted_ok = True
    except Exception as exc:  # pragma: no cover
        painted_ok = False
        check("painted copy button renders without error", False, str(exc))
    if painted_ok:
        check("painted copy button renders without error", True)
    check(
        "hover painting uses no setStyleSheet",
        _StyleCountLabel.setstyle_calls == base_calls,
    )
    check(
        "hover state is on while the tile is hovered",
        sc._tile_hovered and sc._copy_hover,
    )

    # --- leave clears the hover state --------------------------------------
    sc.leaveEvent(QEvent(QEvent.Type.Leave))
    check(
        "leaving the tile clears copy hover state",
        not sc._tile_hovered and not sc._copy_hover,
    )

    # --- cached glyph -------------------------------------------------------
    icon = w._copy_icon(16)
    check(
        "cached copy glyph is a real pixmap",
        icon is not None
        and not icon.isNull()
        and icon.width() == 16
        and icon.height() == 16,
        f"{icon.width()}x{icon.height()}" if icon is not None else "None",
    )
    check(
        "copy glyph is cached across calls",
        w._copy_icon(16) is icon,
    )

    # --- clipboard ----------------------------------------------------------
    from main import RAWImageViewer

    target = os.path.join(os.getcwd(), "sample.ARW")
    RAWImageViewer._copy_paths_to_clipboard(None, [target])
    clip = QApplication.clipboard().text()
    check(
        "clipboard holds the absolute file path",
        clip == os.path.abspath(target),
        repr(clip),
    )

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
