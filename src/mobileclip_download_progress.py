"""Short MobileCLIP / Hugging Face download progress helpers (installer + in-app)."""

from __future__ import annotations

from typing import Callable, Optional

INSTALLER_PROGRESS_PREFIX = "@RAWVIEWER_PROGRESS"

_last_installer_progress_pct: int | None = None


def download_status_text(pct: int) -> str:
    pct = max(0, min(100, int(pct)))
    return f"Downloading... {pct}%"


def reset_installer_progress_state() -> None:
    """Call at the start of each installer model download run."""
    global _last_installer_progress_pct
    _last_installer_progress_pct = None


def emit_installer_progress(pct: int) -> None:
    """Machine-readable line for bootstrap.py (Windows installer)."""
    global _last_installer_progress_pct
    pct = max(0, min(100, int(pct)))
    if _last_installer_progress_pct == pct:
        return
    _last_installer_progress_pct = pct
    msg = download_status_text(pct)
    print(f"{INSTALLER_PROGRESS_PREFIX} pct={pct} message={msg}", flush=True)


def make_byte_progress_tqdm(
    stage_start: int,
    stage_end: int,
    on_pct: Callable[[int], None],
    *,
    silent: bool = False,
):
    """tqdm class that maps byte progress to an overall 0–100 pct callback."""
    from tqdm.auto import tqdm

    class ReportingTqdm(tqdm):
        def __init__(self, *args, **kwargs):
            if silent:
                kwargs["disable"] = True
            super().__init__(*args, **kwargs)

        def _emit_overall(self) -> None:
            if self.total and self.total > 0:
                frac = min(1.0, self.n / self.total)
                overall = stage_start + int(frac * (stage_end - stage_start))
                on_pct(overall)

        def update(self, n=1):
            result = super().update(n)
            self._emit_overall()
            return result

        def refresh(self, nolock=False, lock_args=None):
            result = super().refresh(nolock=nolock, lock_args=lock_args)
            self._emit_overall()
            return result

    return ReportingTqdm


def report_progress(
    progress_callback: Optional[Callable[..., None]],
    pct: int,
    *,
    installer: bool = False,
) -> None:
    """Notify installer stdout or in-app callback with a short status line."""
    pct = max(0, min(100, int(pct)))
    if installer:
        emit_installer_progress(pct)
        return
    if progress_callback is None:
        return
    text = download_status_text(pct)
    try:
        progress_callback(pct, text)
    except TypeError:
        progress_callback(text)
