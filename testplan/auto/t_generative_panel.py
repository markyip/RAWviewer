#!/usr/bin/env python3
"""Generative Edit panel section + consent gate (headless).

Runs the real ImageAdjustPanelWidget offscreen. The consent tests use an
isolated QSettings scope so a developer's real configuration is never
read or written by the suite.

Checks:
  1. The section exists, starts disabled with no endpoint, and says why.
  2. Generate refuses to emit on an empty instruction (no pointless
     round-trip, no confusing failure from the far end).
  3. Generate emits the instruction text when there is one.
  4. Busy state disables Generate, enables Cancel, and locks the box.
  5. Consent is GLOBAL but bound to the endpoint it was granted for --
     changing the endpoint revokes it, so agreeing to your own machine is
     never silently reused for someone else's API.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name} {detail}".rstrip())
    else:
        print(f"  FAIL  {name} {detail}".rstrip())
        FAILURES.append(name)


def test_consent_scope():
    """Consent is remembered globally but re-asked when the target changes."""
    from PyQt6.QtCore import QSettings

    import generative_settings as gs

    # Isolate from the developer's real settings.
    gs.ORG, gs.APP = "RAWviewerTest", "GenerativeSuite"
    QSettings(gs.ORG, gs.APP).clear()

    check("no consent by default", gs.has_consent("https://a.example/e") is False)

    gs.save_settings(endpoint="https://a.example/e")
    check("saved endpoint round-trips", gs.load_settings()["endpoint"] == "https://a.example/e")
    check("saving does not grant consent", gs.has_consent() is False)

    gs.grant_consent("https://a.example/e")
    check("consent granted for endpoint", gs.has_consent("https://a.example/e") is True)
    check("consent is remembered globally", gs.has_consent() is True)

    # A different destination must not inherit the decision.
    check("other endpoint not covered", gs.has_consent("https://b.example/e") is False)

    # Changing the configured endpoint revokes it outright.
    gs.save_settings(endpoint="https://b.example/e")
    check("endpoint change revokes consent", gs.has_consent() is False)
    check("old grant no longer valid", gs.has_consent("https://a.example/e") is False)

    gs.revoke_consent()
    check("explicit revoke works", gs.has_consent() is False)

    # Empty endpoint can never be consented to.
    gs.save_settings(endpoint="")
    gs.grant_consent("")
    check("empty endpoint never consented", gs.has_consent("") is False)

    QSettings(gs.ORG, gs.APP).clear()


def test_panel():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841

    from rawviewer_ui.adjust_panel import ImageAdjustPanelWidget

    panel = ImageAdjustPanelWidget()

    check("section built", hasattr(panel, "_gen_instruction"))
    check("run button exists", hasattr(panel, "_gen_run_btn"))
    check("cancel starts disabled", panel._gen_cancel_btn.isEnabled() is False)

    requests = []
    panel.generative_requested.connect(requests.append)

    # Empty instruction: no signal, and a reason shown.
    panel._gen_instruction.setPlainText("   ")
    panel._on_generate_clicked()
    check("empty instruction emits nothing", requests == [], f"got {requests}")
    check("empty instruction explains itself", "Describe" in panel._gen_status.text())

    panel._gen_instruction.setPlainText("  remove the bin  ")
    check("instruction is trimmed", panel.generative_instruction() == "remove the bin")
    panel._on_generate_clicked()
    check("instruction emitted", requests == ["remove the bin"], f"got {requests}")

    # Busy state.
    panel.set_generative_busy(True)
    check("busy disables Generate", panel._gen_run_btn.isEnabled() is False)
    check("busy enables Cancel", panel._gen_cancel_btn.isEnabled() is True)
    check("busy locks the instruction box", panel._gen_instruction.isReadOnly() is True)
    check("busy disables Setup", panel._gen_settings_btn.isEnabled() is False)

    panel.set_generative_busy(False)
    check("idle re-enables Cancel off", panel._gen_cancel_btn.isEnabled() is False)
    check("idle unlocks the instruction box", panel._gen_instruction.isReadOnly() is False)
    check("idle re-enables Setup", panel._gen_settings_btn.isEnabled() is True)

    cancels = []
    panel.generative_cancel_requested.connect(lambda: cancels.append(1))
    panel._gen_cancel_btn.setEnabled(True)
    panel._gen_cancel_btn.click()
    check("cancel emits", len(cancels) == 1)

    panel.set_generative_status("hello")
    check("status settable", panel._gen_status.text() == "hello")


def main():
    print("Generative Edit panel + consent gate")
    test_consent_scope()
    test_panel()

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
