"""The AI mask wait is visible, cancellable, and gives up on its own.

Smart Object and AI Selection run a segmentation net that can take many
seconds. All the user got was a status-bar line and a disabled button, which
is what a hung app also looks like -- there was no way to tell a slow model
from a stuck one, and no way out.

Two things have to hold. The dialog goes up for the wait and comes down on
every exit (finished, cancelled, timed out). And abandoning a run has to
actually abandon it: a QRunnable already inside the model cannot be stopped,
so cancel and timeout move a token and the late result is dropped when it
arrives. Without that, cancelling and starting again would let the first
run's mask land on top of the second's.

The timeout deliberately does NOT apply while the weights are downloading.
First use pulls 214 MB, which no sane timeout covers, and a timeout that
made first use impossible would be worse than the hang it was added for.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QWidget  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


_app = QApplication.instance() or QApplication([])


class _Host(QWidget):
    """Only what the busy-dialog helpers touch.

    They call each other, so borrow the real implementations rather than
    re-stating them here -- a stub of _end_ai_mask_busy would let the
    "never stack two" behaviour pass without being exercised.
    """

    def __init__(self):
        super().__init__()
        import main as _m

        cls = _m.RAWImageViewer
        self._end_ai_mask_busy = lambda: cls._end_ai_mask_busy(self)
        self._begin_ai_mask_busy = lambda k, **kw: cls._begin_ai_mask_busy(self, k, **kw)
        self._on_ai_mask_cancelled = lambda: cls._on_ai_mask_cancelled(self)
        self._ai_mask_busy = True
        self._ai_mask_token = 0
        self._ai_mask_dialog = None
        self.status = []
        self.buttons_enabled = False

    def _show_status(self, text, ms=0):
        self.status.append(text)

    def _set_ai_mask_buttons_enabled(self, on):
        self.buttons_enabled = bool(on)


def main() -> int:
    import main as mainmod

    M = mainmod.RAWImageViewer
    begin = M._begin_ai_mask_busy
    end = M._end_ai_mask_busy
    cancel = M._on_ai_mask_cancelled
    timeout = M._on_ai_mask_timeout

    check(
        "the timeout is 15s as asked",
        mainmod._AI_MASK_TIMEOUT_MS == 15_000,
        f"{mainmod._AI_MASK_TIMEOUT_MS} ms",
    )

    # --- the dialog goes up, and says which tool is running ---
    h = _Host()
    begin(h, "subject", downloading=False)
    dlg = h._ai_mask_dialog
    check("a dialog goes up for the wait", dlg is not None)
    check("it names the tool", "subject" in dlg._message_label.text().lower(),
          dlg._message_label.text())
    check("and it can be cancelled", dlg.cancel_btn.isEnabled())

    # --- cancel: dialog down, buttons back, token moved ---
    before = h._ai_mask_token
    cancel(h)
    check("cancel takes the dialog down", h._ai_mask_dialog is None)
    check("cancel clears busy", h._ai_mask_busy is False)
    check("cancel re-enables the buttons", h.buttons_enabled is True)
    check(
        "cancel moves the token",
        h._ai_mask_token != before,
        "otherwise the abandoned run's result would still be accepted",
    )

    # --- a late result from a cancelled run must be dropped ---
    h2 = _Host()
    begin(h2, "subject", downloading=False)
    stale = h2._ai_mask_token
    cancel(h2)
    h2._ai_mask_busy = True  # a second run is now in flight
    landed = []
    h2._add_mask_layer_from_alpha = lambda *a, **k: landed.append(1)
    h2.single_image_adjust_panel = None
    M._on_ai_mask_finished(h2, stale, "", "subject", object())
    check(
        "a cancelled run's late result is ignored",
        not landed and h2._ai_mask_busy is True,
        "it would otherwise drop a mask on the user after they said stop",
    )

    # --- timeout only fires for the run it was armed for ---
    h3 = _Host()
    begin(h3, "sky", downloading=False)
    armed = h3._ai_mask_token
    timeout(h3, armed - 1, "sky")  # an older run's timer
    check(
        "a stale timer does not touch the current run",
        h3._ai_mask_dialog is not None and h3._ai_mask_busy is True,
    )

    timeout(h3, armed, "sky")
    check("the run's own timer gives up", h3._ai_mask_dialog is None)
    check("and clears busy", h3._ai_mask_busy is False)
    check("and re-enables the buttons", h3.buttons_enabled is True)
    check(
        "and says so rather than failing silently",
        any("15" in s for s in h3.status),
        str(h3.status),
    )

    # --- a timer that fires after a clean finish is harmless ---
    h4 = _Host()
    begin(h4, "sky", downloading=False)
    tok = h4._ai_mask_token
    h4._ai_mask_busy = False  # finished normally
    h4.status.clear()
    timeout(h4, tok, "sky")
    check(
        "a timer firing after success says nothing",
        h4.status == [],
        "the work already landed; a timeout message would be a lie",
    )

    # --- downloading: no timeout, because 214 MB will not fit in 15s ---
    h5 = _Host()
    begin(h5, "subject", downloading=True)
    dlg5 = h5._ai_mask_dialog
    check("the download says what it is doing", "download" in dlg5._message_label.text().lower())
    check("and mentions it happens once", "once" in dlg5._message_label.text().lower())

    import inspect

    src = inspect.getsource(M._begin_ai_mask_busy)
    check(
        "no timer is armed while downloading",
        "if not downloading:" in src,
        "a 15s timeout would make first use impossible",
    )

    # --- starting a second run never stacks two dialogs ---
    h6 = _Host()
    begin(h6, "subject", downloading=False)
    first = h6._ai_mask_dialog
    begin(h6, "sky", downloading=False)
    check("a second run replaces the dialog", h6._ai_mask_dialog is not first)
    check("and only one is left", h6._ai_mask_dialog is not None)
    end(h6)
    check("end takes it down", h6._ai_mask_dialog is None)
    end(h6)  # must be safe twice
    check("end is safe to call twice", h6._ai_mask_dialog is None)

    # --- the bar is a marquee, not an invented percentage ---
    from rawviewer_app.widgets import ExportProgressDialog

    d = ExportProgressDialog(None, title="t", message="m")
    d.set_indeterminate(True)
    check(
        "an unknowable wait shows a marquee",
        (d._bar.minimum(), d._bar.maximum()) == (0, 0),
        "a percentage that stalls at 80% is worse than an honest marquee",
    )
    check("and hides the percentage", not d._percent_label.isVisible())
    d.set_progress(40)
    check("determinate still works for export", d._bar.maximum() == 100)
    d.close()

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
