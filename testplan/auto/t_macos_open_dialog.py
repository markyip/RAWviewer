#!/usr/bin/env python3
"""Regression guard: macOS Open File / Open Folder must not hang the app.

Root cause (macOS 27.0 beta, build 26A5388g): NSOpenPanel.alloc().init()
never returns for a process without a real app-bundle identity. Reproduced
with a minimal AppKit script containing no Qt and no RAWviewer code, and NOT
avoidable by setting NSApplicationActivationPolicy (tested: policy flips
2 -> 0 successfully, the very next alloc().init() still hangs). Nothing in
RAWviewer's folder-open code has changed since v2.5; the failure began the
day after the macOS 27.0 update.

Mitigation under test: when not running as a packaged .app (sys.frozen
unset), skip the native panel entirely and use a Finder-driven AppleScript
picker. The `activate` must sit inside a `tell application "Finder"` block --
a bare top-level `activate` leaves the dialog created but invisible (observed
live: osascript alive and waiting, no dialog on any Space).

Run:  pixi run python testplan/auto/t_macos_open_dialog.py
Live NSOpenPanel probe (hangs by design on an affected OS, so opt-in):
      RAWVIEWER_TEST_NSPANEL_PROBE=1 pixi run python testplan/auto/t_macos_open_dialog.py
"""
import os
import subprocess
import sys
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


class StubViewer:
    """Stand-in for RAWImageViewer.

    The dialog helpers need only a couple of collaborators, so the real (very
    heavy) QMainWindow is never constructed. Methods are pulled off the real
    class below, so this exercises shipping code rather than a copy of it.
    """

    def __init__(self, mainmod, extensions=(".arw", ".jpg")):
        self._mainmod = mainmod
        self._extensions = list(extensions)

    def get_supported_extensions(self):
        return list(self._extensions)

    def _sanitize_dialog_start_dir(self, last_dir):
        return self._mainmod.RAWImageViewer._sanitize_dialog_start_dir(self, last_dir)


def main() -> int:
    if sys.platform != "darwin":
        print("SKIP  macOS-only suite")
        return 0

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import main as mainmod

    bind = lambda name: getattr(mainmod.RAWImageViewer, name)  # noqa: E731
    viewer = StubViewer(mainmod)

    # --- 1. Native-panel gate ------------------------------------------------
    had_frozen = hasattr(sys, "frozen")
    old_frozen = getattr(sys, "frozen", None)
    if had_frozen:
        del sys.frozen
    try:
        check(
            "unbundled source launch skips NSOpenPanel",
            bind("_macos_nsopenpanel_reliable")(viewer) is False,
            "(this is the configuration that hangs on macOS 27 beta)",
        )
    finally:
        if had_frozen:
            sys.frozen = old_frozen

    sys.frozen = True
    try:
        check(
            "packaged .app keeps native NSOpenPanel",
            bind("_macos_nsopenpanel_reliable")(viewer) is True,
        )
    finally:
        if old_frozen is None:
            del sys.frozen
        else:
            sys.frozen = old_frozen

    # --- 2. AppleScript fallback shape --------------------------------------
    captured = []
    orig_run = mainmod._run_applescript
    mainmod._run_applescript = lambda script: (captured.append(script) or "")
    try:
        for label, method, picker in (
            ("open-file", "_open_file_dialog_macos_applescript", "choose file"),
            ("open-folder", "_open_folder_dialog_macos_applescript", "choose folder"),
        ):
            captured.clear()
            bind(method)(viewer, "")
            script = captured[0] if captured else ""
            ok = (
                'tell application "Finder"' in script
                and "end tell" in script
                and picker in script
                and "activate" in script
                and script.index("activate") < script.index(picker)
            )
            check(
                f"{label}: picker runs inside `tell application \"Finder\"` with activate first",
                ok,
                "(bare top-level activate leaves the dialog invisible)",
            )

            # osacompile is ground truth for AppleScript syntax.
            proc = subprocess.run(
                ["osacompile", "-o", "/dev/null", "-e", script],
                capture_output=True,
                text=True,
            )
            check(
                f"{label}: generated script compiles",
                proc.returncode == 0,
                proc.stderr.strip()[:120],
            )

        # Quote in a folder name must not terminate the AppleScript literal.
        tricky = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'we"ird')
        os.makedirs(tricky, exist_ok=True)
        try:
            captured.clear()
            bind("_open_folder_dialog_macos_applescript")(viewer, tricky)
            script = captured[0] if captured else ""
            check("start directory with a quote is escaped", 'we\\"ird' in script)
        finally:
            os.rmdir(tricky)
    finally:
        mainmod._run_applescript = orig_run

    # --- 3. Live NSOpenPanel probe (opt-in) ---------------------------------
    if os.environ.get("RAWVIEWER_TEST_NSPANEL_PROBE") == "1":
        probe = textwrap.dedent(
            """
            import AppKit, objc
            app = objc.lookUpClass("NSApplication").sharedApplication()
            print("policy", app.activationPolicy())
            panel = objc.lookUpClass("NSOpenPanel").alloc().init()
            print("PANEL", panel is not None)
            """
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", probe],
                capture_output=True,
                text=True,
                timeout=25,
            )
            check(
                "live probe: NSOpenPanel construction returns",
                "PANEL" in proc.stdout,
                "OS appears FIXED -- the gate above may be relaxable",
            )
        except subprocess.TimeoutExpired:
            check(
                "live probe: NSOpenPanel construction returns",
                False,
                "HUNG >25s -- OS still affected, gate must stay",
            )
    else:
        print("SKIP  live NSOpenPanel probe (set RAWVIEWER_TEST_NSPANEL_PROBE=1)")

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} -> {', '.join(FAILURES)}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
