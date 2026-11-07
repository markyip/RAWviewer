"""macOS-specific integration: sharing, App Nap and AppKit tweaks."""
from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import sys
import weakref

from PyQt6.QtCore import QEvent, QObject, QPoint, Qt, QTimer

from core.config import _env_true

from .logging_setup import _share_log
from .qt_helpers import _qt_app

__all__ = [
    "_macos_share_items_array",
    "_macos_write_sharing_content_scope",
    "_macos_register_sharing_service_delegate_metadata",
    "_get_macos_sharing_service_delegate_class",
    "_get_macos_share_picker_delegate_class",
    "_activate_macos_foreground_for_share",
    "_activate_macos_foreground_app",
    "_disable_macos_app_nap",
    "_applescript_escape",
    "_run_applescript",
    "_macos_try_force_dark_titlebar",
    "_macos_try_force_dark_titlebar_for_window",
    "_macos_apply_ns_text_completion_off",
    "_macos_disable_line_edit_system_completion",
    "_macos_schedule_line_edit_system_completion_off",
    "_macos_install_line_edit_completion_guard",
]


def _macos_share_items_array(urls) -> object:
    """Build AppKit items array (NSURL). Empty/invalid arrays make picker spin forever."""
    if urls is None:
        return None
    try:
        from Foundation import NSArray
        items_list = list(urls) if isinstance(urls, (list, tuple)) else [urls]

        arr = NSArray.arrayWithArray_(items_list)
        if arr is None:
            _share_log(logging.WARNING, "NSArray.arrayWithArray_ returned None")
            return None
        try:
            count = int(arr.count())
        except Exception:
            count = len(arr) if hasattr(arr, "__len__") else 0
        if count < 1:
            _share_log(logging.WARNING, "share items array count=0")
            return None
        return arr
    except Exception as exc:
        _share_log(logging.WARNING, "NSArray build failed: %s", exc)
        return None


_macos_share_picker_delegate_cls = None


_macos_sharing_service_delegate_cls = None


def _macos_write_sharing_content_scope(scope, value: int = 0) -> None:
    """Write NSSharingContentScopeItem (0) into AppKit out-parameter (^q). Module-level -- not on NSObject."""
    if scope is None:
        return
    try:
        scope[0] = value
        return
    except Exception as exc:
        _share_log(logging.DEBUG, "sharingContentScope scope[0]: %s", exc)
    try:
        import ctypes

        ptr = int(scope) if hasattr(scope, "__index__") else scope
        ctypes.cast(ptr, ctypes.POINTER(ctypes.c_long))[0] = value
    except Exception as exc:
        _share_log(logging.DEBUG, "sharingContentScope ctypes: %s", exc)


def _macos_register_sharing_service_delegate_metadata(delegate_cls) -> None:
    """Tell PyObjC that argument 4 is out-pointer (^q) so scope[0] works (no ObjCPointerWarning)."""
    try:
        import objc

        scope_out_type = objc._C_OUT + objc._C_PTR + objc._C_LNG
        meta = {"arguments": {4: {"type": scope_out_type}}}
        sel = b"sharingService:sourceWindowForShareItems:sharingContentScope:"
        for cls_name in (delegate_cls.__name__.encode(), b"NSObject"):
            with contextlib.suppress(Exception):
                objc.registerMetaDataForSelector(cls_name, sel, meta)
    except Exception as exc:
        _share_log(logging.DEBUG, "sharing delegate metadata registration: %s", exc)


def _get_macos_sharing_service_delegate_class():
    """NSSharingServiceDelegate -- AirDrop needs source NSWindow from Qt (Mail does not)."""
    global _macos_sharing_service_delegate_cls
    if _macos_sharing_service_delegate_cls is not None:
        return _macos_sharing_service_delegate_cls
    import objc
    from AppKit import NSMakeRect, NSObject, NSScreen

    class MacSharingServiceDelegate(NSObject):
        def initWithViewer_(self, viewer):
            self = objc.super(MacSharingServiceDelegate, self).init()
            if self is None:
                return None
            self._viewer = viewer
            return self

        def sharingService_sourceWindowForShareItems_sharingContentScope_(
            self, service, items, scope
        ):
            """Return NSWindow; set sharingContentScope or AirDrop gets ObjCPointerWarning (^q)."""
            _macos_write_sharing_content_scope(scope, 0)
            viewer = getattr(self, "_viewer", None)
            if viewer is None:
                return None
            try:
                win = viewer._macos_share_ns_window()
                if win is None:
                    _share_log(logging.WARNING, "AirDrop delegate: no source NSWindow")
                return win
            except Exception as exc:
                _share_log(logging.DEBUG, "sourceWindow delegate: %s", exc)
                return None

        def sharingService_willShareItems_(self, service, items):
            _share_log(logging.INFO, "sharing service will share items: %s", service)

        def sharingService_sourceFrameOnScreenForShareItem_(self, service, item):
            viewer = getattr(self, "_viewer", None)
            btn = getattr(viewer, "share_bottom_button", None) if viewer else None
            if btn is None:
                return NSMakeRect(0, 0, 1, 1)
            try:
                screen = NSScreen.mainScreen()
                if screen is None:
                    return NSMakeRect(0, 0, max(1, btn.width()), max(1, btn.height()))
                frame = screen.frame()
                top_left = btn.mapToGlobal(QPoint(0, 0))
                w = max(1, btn.width())
                h = max(1, btn.height())
                x = float(top_left.x())
                y = float(frame.size.height) - float(top_left.y()) - h
                return NSMakeRect(x, y, w, h)
            except Exception:
                return NSMakeRect(0, 0, max(1, btn.width()), max(1, btn.height()))

        def sharingService_didShareItems_(self, service, items):
            viewer = getattr(self, "_viewer", None)
            if viewer is not None:
                viewer._macos_share_airdrop_completed = True
                _share_log(logging.INFO, "sharing service shared items successfully")
                QTimer.singleShot(30000, viewer._macos_cleanup_share_temp)
                QTimer.singleShot(0, lambda: viewer._macos_share_end_picker_session("service_success"))

        def sharingService_didFailToShareItems_error_(self, service, items, error):
            viewer = getattr(self, "_viewer", None)
            if viewer is not None:
                viewer._macos_share_airdrop_completed = True
                _share_log(logging.WARNING, "sharing service failed to share items: %s", error)
                QTimer.singleShot(30000, viewer._macos_cleanup_share_temp)
                QTimer.singleShot(0, lambda: viewer._macos_share_end_picker_session("service_fail"))

    _macos_register_sharing_service_delegate_metadata(MacSharingServiceDelegate)
    _macos_sharing_service_delegate_cls = MacSharingServiceDelegate
    return MacSharingServiceDelegate


def _get_macos_share_picker_delegate_class():
    """Lazy NSObject delegate so NSSharingServicePicker can populate and we resume Qt filters on dismiss."""
    global _macos_share_picker_delegate_cls
    if _macos_share_picker_delegate_cls is not None:
        return _macos_share_picker_delegate_cls
    import objc
    from AppKit import NSObject

    class MacSharePickerDelegate(NSObject):
        def initWithViewer_(self, viewer):
            self = objc.super(MacSharePickerDelegate, self).init()
            if self is None:
                return None
            self._viewer = viewer
            return self

        def sharingServicePicker_sharingServicesForItems_proposedSharingServices_(
            self, picker, items, proposed
        ):
            viewer = getattr(self, "_viewer", None)
            cached = getattr(viewer, "_macos_share_cached_services", None) if viewer else None
            if cached:
                try:
                    n = len(cached)
                except Exception:
                    n = -1
                _share_log(logging.INFO, "picker delegate: returning %d cached service(s)", n)
                return cached
            try:
                n = len(proposed or [])
            except Exception:
                n = -1
            _share_log(logging.INFO, "picker delegate: returning %d proposed service(s)", n)
            return proposed

        def sharingServicePicker_delegateForSharingService_(self, picker, service):
            viewer = getattr(self, "_viewer", None)
            if viewer is None:
                return None
            try:
                delegate_cls = _get_macos_sharing_service_delegate_class()
                delegate = delegate_cls.alloc().initWithViewer_(viewer)
                # Keep reference to prevent GC
                viewer._macos_active_service_delegate = delegate
                return delegate
            except Exception as exc:
                _share_log(logging.WARNING, "failed to create service delegate: %s", exc)
                return None

        def sharingServicePicker_didChooseSharingService_(self, picker, service):
            viewer = getattr(self, "_viewer", None)
            if viewer is not None:
                _share_log(logging.INFO, "picker didChooseSharingService: %s", service)
                if service is None:
                    QTimer.singleShot(0, viewer._macos_cleanup_share_temp)
                else:
                    QTimer.singleShot(30000, viewer._macos_cleanup_share_temp)
                reason = "picker_cancel" if service is None else "picker_chose"
                QTimer.singleShot(0, lambda r=reason: viewer._macos_share_end_picker_session(r))
            return None

    _macos_share_picker_delegate_cls = MacSharePickerDelegate
    return MacSharePickerDelegate


def _activate_macos_foreground_for_share() -> None:
    """Bring app forward for NSSharingService (AirDrop panel needs a key window)."""
    if sys.platform != "darwin":
        return
    try:
        import objc

        ns_app = objc.lookUpClass("NSApplication").sharedApplication()
        if ns_app is not None:
            ns_app.setActivationPolicy_(0)
            ns_app.activateIgnoringOtherApps_(True)
    except Exception as exc:
        _share_log(logging.DEBUG, "share foreground activate: %s", exc)


def _activate_macos_foreground_app() -> None:
    """Opt-in NSApplication foreground (RAWVIEWER_MACOS_FORCE_FOREGROUND=1).

    Default off: v2.1 dev never needed this; brute-force activation can fight AppKit
    sheets (e.g. NSSharingServicePicker). Use Qt raise/activateWindow in dialogs instead.
    """
    if sys.platform != "darwin" or not _env_true("RAWVIEWER_MACOS_FORCE_FOREGROUND"):
        return
    try:
        import objc

        ns_app = objc.lookUpClass("NSApplication").sharedApplication()
        if ns_app is not None:
            ns_app.setActivationPolicy_(0)
            ns_app.activateIgnoringOtherApps_(True)
            _share_log(logging.DEBUG, "NSApplication activateIgnoringOtherApps OK")
    except Exception as exc:
        _share_log(logging.WARNING, "NSApplication foreground activation failed: %s", exc)


_macos_app_nap_activity_token = None


def _disable_macos_app_nap() -> None:
    """Hold an NSActivity for the process lifetime so App Nap cannot throttle QTimers.

    Without this, launching unfocused (or idling briefly) coalesces timer firing
    to ~10s ticks: the 800ms session-restore full-decode timer was observed
    firing 9-15s late, leaving the first image at preview tier until the user
    interacted. Uses UserInitiatedAllowingIdleSystemSleep so system idle sleep
    is unaffected -- this only opts out of timer/App Nap throttling.
    """
    global _macos_app_nap_activity_token
    if sys.platform != "darwin" or _macos_app_nap_activity_token is not None:
        return
    if _env_true("RAWVIEWER_ALLOW_APP_NAP"):
        return
    try:
        import objc

        process_info = objc.lookUpClass("NSProcessInfo").processInfo()
        # NSActivityUserInitiatedAllowingIdleSystemSleep
        options = 0x00FFFFFF & ~(1 << 20)
        _macos_app_nap_activity_token = process_info.beginActivityWithOptions_reason_(
            options, "RAWviewer image pipeline timers"
        )
    except Exception:
        _macos_app_nap_activity_token = None


def _applescript_escape(value: str) -> str:
    return (value or "").replace("\\", "\\\\").replace('"', '\\"')


def _run_applescript(script: str) -> str:
    """Run AppleScript; return stdout on success, empty string on cancel/error."""

    try:
        proc = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if proc.returncode != 0:
        return ""
    return (proc.stdout or "").strip()


def _macos_try_force_dark_titlebar():
    """
    Best-effort: request Dark Aqua appearance on macOS so the native title bar
    matches our dark UI.

    Uses PyObjC (already a project dependency, see _bring_app_to_foreground
    above) rather than hand-rolled ctypes objc_msgSend calls. The previous
    ctypes version called Objective-C methods that return autoreleased
    objects (stringWithUTF8String:, appearanceNamed:, sharedApplication)
    with no autorelease pool management and no retain -- a real, reproduced
    SIGSEGV in +[NSAppearance appearanceNamed:] was traced back to this via
    a live crash-report/faulthandler capture during scroll-stutter
    investigation (the crash was incidental to scrolling; this function
    fires once at startup via QTimer.singleShot). PyObjC manages pool/
    lifetime correctly for the exact same calls.
    """
    if sys.platform != "darwin":
        return False
    try:
        import objc

        NSAppearance = objc.lookUpClass("NSAppearance")
        NSApplication = objc.lookUpClass("NSApplication")

        appearance = NSAppearance.appearanceNamed_("NSAppearanceNameDarkAqua")
        app = NSApplication.sharedApplication()
        if app is None or appearance is None:
            return False

        app.setAppearance_(appearance)
        return True
    except Exception:
        # Never fail startup due to cosmetic integration.
        return False


def _macos_try_force_dark_titlebar_for_window(widget):
    """
    Best-effort: set Dark Aqua on the NSWindow backing a Qt widget.
    This is more reliable than setting NSApp appearance alone for the title bar.
    """
    if sys.platform != "darwin":
        return False
    if widget is None:
        return False
    try:
        import objc

        NSAppearance = objc.lookUpClass("NSAppearance")
        appearance = NSAppearance.appearanceNamed_("NSAppearanceNameDarkAqua")
        if appearance is None:
            return False

        # Obtain NSView* from Qt widget. On macOS, QWidget.winId() is a pointer;
        # wrap it as a PyObjC object rather than dereferencing it via raw ctypes.
        view_ptr = int(widget.winId())
        if not view_ptr:
            return False

        ns_view = objc.objc_object(c_void_p=view_ptr)
        ns_window = ns_view.window()
        if ns_window is None:
            return False

        ns_window.setAppearance_(appearance)
        return True
    except Exception:
        return False


_macos_completion_guarded_edits: weakref.WeakSet = weakref.WeakSet()


def _macos_apply_ns_text_completion_off(node) -> None:
    """Turn off AppKit completion/spellcheck hooks on one NSView/NSTextField."""
    for setter, value in (
        ("setAutomaticTextCompletionEnabled:", False),
        ("setAutomaticSpellingCorrectionEnabled:", False),
        ("setContinuousSpellCheckingEnabled:", False),
        ("setGrammarCheckingEnabled:", False),
        ("setSmartInsertDeleteEnabled:", False),
        ("setAutomaticDashSubstitutionEnabled:", False),
        ("setAutomaticQuoteSubstitutionEnabled:", False),
        ("setAutomaticLinkDetectionEnabled:", False),
    ):
        try:
            if node.respondsToSelector_(setter):
                getattr(node, setter.replace(":", "_"))(value)
        except Exception:
            pass
    try:
        if node.respondsToSelector_("setFocusRingType:"):
            try:
                import AppKit  # type: ignore[import-not-found]

                node.setFocusRingType_(AppKit.NSFocusRingTypeNone)
            except Exception:
                node.setFocusRingType_(1)  # NSFocusRingTypeNone
    except Exception:
        pass


def _macos_disable_line_edit_system_completion(line_edit) -> None:
    """
    Disable macOS NSTextField automatic completion/spellcheck on a Qt QLineEdit.

    Newer macOS builds host SPCompletionListServiceViewController inside Qt via
    ViewBridge (NSRemoteView). When that XPC view fails to configure, the process
    aborts (SIGABRT) -- seen with gallery search on macOS 26 + Qt 6.11.
    """
    if sys.platform != "darwin" or line_edit is None:
        return
    if os.environ.get("RAWVIEWER_MACOS_LINEEDIT_COMPLETION", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    with contextlib.suppress(Exception):
        line_edit.setInputMethodHints(Qt.InputMethodHint.ImhNoPredictiveText)
    try:

        import objc
    except ImportError:
        return
    try:
        view_ptr = int(line_edit.winId())
        if not view_ptr:
            return
        view = objc.objc_object(c_void_p=view_ptr)
        stack = [view]
        seen: set[int] = set()
        while stack:
            node = stack.pop()
            node_id = int(node) if node is not None else 0
            if node_id and node_id in seen:
                continue
            if node_id:
                seen.add(node_id)
            _macos_apply_ns_text_completion_off(node)
            try:
                subs = list(node.subviews() or [])
            except Exception:
                subs = []
            stack.extend(subs)
            try:
                parent = node.superview()
                if parent is not None:
                    stack.append(parent)
            except Exception:
                pass
        try:
            if view.respondsToSelector_("window"):
                win = view.window()
                if win is not None and win.respondsToSelector_("fieldEditor:forObject:"):
                    editor = win.fieldEditor_forObject_(True, view)
                    if editor is not None:
                        _macos_apply_ns_text_completion_off(editor)
        except Exception:
            pass
    except Exception:
        pass


def _macos_schedule_line_edit_system_completion_off(line_edit, delay_ms: int = 0) -> None:
    """Apply _macos_disable_line_edit_system_completion after winId() is valid."""
    if sys.platform != "darwin" or line_edit is None:
        return

    def _apply() -> None:
        _macos_disable_line_edit_system_completion(line_edit)

    QTimer.singleShot(max(0, int(delay_ms)), _apply)


class _MacLineEditCompletionGuard(QObject):
    """Re-disable AppKit text completion when the Qt line edit is shown or focused."""

    def eventFilter(self, watched, event):
        try:
            if event.type() in (
                QEvent.Type.FocusIn,
                QEvent.Type.Show,
                QEvent.Type.Polish,
                QEvent.Type.WindowActivate,
            ):
                _macos_disable_line_edit_system_completion(watched)
        except Exception:
            pass
        return False


def _macos_install_line_edit_completion_guard(line_edit) -> None:
    """Install focus/show hooks plus wake-from-sleep re-apply for a QLineEdit."""
    if sys.platform != "darwin" or line_edit is None:
        return
    if os.environ.get("RAWVIEWER_MACOS_LINEEDIT_COMPLETION", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    _macos_completion_guarded_edits.add(line_edit)
    guard = line_edit.findChild(_MacLineEditCompletionGuard)
    if guard is None:
        guard = _MacLineEditCompletionGuard(line_edit)
        line_edit.installEventFilter(guard)
    for delay in (0, 50, 250, 500):
        _macos_schedule_line_edit_system_completion_off(line_edit, delay)
    app = _qt_app()
    if app is not None and not getattr(app, "_macos_completion_reapply_hook", False):
        app._macos_completion_reapply_hook = True

        def _reapply_on_app_active(state) -> None:
            if state != Qt.ApplicationState.ApplicationActive:
                return
            for edit in list(_macos_completion_guarded_edits):
                _macos_disable_line_edit_system_completion(edit)

        app.applicationStateChanged.connect(_reapply_on_app_active)
