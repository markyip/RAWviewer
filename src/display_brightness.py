"""Best-effort display brightness control while Adjust is open.

macOS uses private CoreDisplay Get/SetUserBrightness (built-in panels).
Windows tries, in order: DDC/CI (dxva2) on the monitor nearest the window,
legacy ``\\\\.\\LCD`` IOCTL, then WMI ``WmiMonitorBrightness`` (laptop
backlights). External monitors without DDC/CI stay a no-op.

Never uses ``SetDeviceGammaRamp`` — that would distort colour management.
Callers must restore any saved value when leaving Adjust.
"""

from __future__ import annotations

import sys
from typing import Optional, Protocol

_core_display = None
_cg = None
_mac_ready = False
_active_backend: Optional["_BrightnessBackend"] = None


class _BrightnessBackend(Protocol):
    def get(self) -> Optional[float]:
        ...

    def set(self, value: float) -> bool:
        ...

    def close(self) -> None:
        ...


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _from_min_max(current: int, mn: int, mx: int) -> Optional[float]:
    if mx <= mn:
        return None
    return _clamp01((int(current) - int(mn)) / float(int(mx) - int(mn)))


def _to_min_max(unit: float, mn: int, mx: int) -> int:
    span = int(mx) - int(mn)
    if span <= 0:
        return int(mn)
    return int(mn) + int(round(_clamp01(unit) * span))


def _ctl_code(device_type: int, function: int, method: int = 0, access: int = 0) -> int:
    return (int(device_type) << 16) | (int(access) << 14) | (int(function) << 2) | int(method)


# ntddvdeo.h FILE_DEVICE_VIDEO=0x23, QUERY=0x126, SET=0x127, METHOD_BUFFERED.
IOCTL_VIDEO_QUERY_DISPLAY_BRIGHTNESS = _ctl_code(0x23, 0x126)
IOCTL_VIDEO_SET_DISPLAY_BRIGHTNESS = _ctl_code(0x23, 0x127)
DISPLAYPOLICY_BOTH = 3


def _close_backend(backend: Optional[_BrightnessBackend]) -> None:
    if backend is None:
        return
    closer = getattr(backend, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:
            pass


def _set_active_backend(backend: Optional[_BrightnessBackend]) -> None:
    global _active_backend
    if _active_backend is backend:
        return
    _close_backend(_active_backend)
    _active_backend = backend


def _ensure_mac_loaded() -> bool:
    global _core_display, _cg, _mac_ready
    if _mac_ready:
        return _core_display is not None and _cg is not None
    _mac_ready = True
    if sys.platform != "darwin":
        return False
    try:
        from ctypes import CDLL, c_double, c_uint32

        _core_display = CDLL(
            "/System/Library/Frameworks/CoreDisplay.framework/CoreDisplay"
        )
        _cg = CDLL(
            "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
        )
        _core_display.CoreDisplay_Display_GetUserBrightness.argtypes = [c_uint32]
        _core_display.CoreDisplay_Display_GetUserBrightness.restype = c_double
        _core_display.CoreDisplay_Display_SetUserBrightness.argtypes = [
            c_uint32,
            c_double,
        ]
        _cg.CGMainDisplayID.restype = c_uint32
        return True
    except Exception:
        _core_display = None
        _cg = None
        return False


def main_display_id() -> Optional[int]:
    if not _ensure_mac_loaded():
        return None
    try:
        return int(_cg.CGMainDisplayID())
    except Exception:
        return None


class _MacBackend:
    def __init__(self, display_id: Optional[int] = None) -> None:
        self._display_id = display_id

    def _did(self) -> Optional[int]:
        if self._display_id is not None:
            return int(self._display_id)
        return main_display_id()

    def get(self) -> Optional[float]:
        did = self._did()
        if did is None:
            return None
        try:
            from ctypes import c_uint32

            val = float(
                _core_display.CoreDisplay_Display_GetUserBrightness(c_uint32(int(did)))
            )
            if val < 0.0 or val > 1.0:
                return None
            return val
        except Exception:
            return None

    def set(self, value: float) -> bool:
        did = self._did()
        if did is None:
            return False
        try:
            from ctypes import c_double, c_uint32

            _core_display.CoreDisplay_Display_SetUserBrightness(
                c_uint32(int(did)), c_double(_clamp01(value))
            )
            return True
        except Exception:
            return False

    def close(self) -> None:
        return


class _Dxva2Backend:
    def __init__(self, hphys, array, count: int) -> None:
        self._hphys = hphys
        self._array = array
        self._count = int(count)
        self._mn = 0
        self._mx = 100

    def get(self) -> Optional[float]:
        try:
            import ctypes
            from ctypes import wintypes

            dxva2 = ctypes.WinDLL("dxva2", use_last_error=True)
            mn = wintypes.DWORD()
            cur = wintypes.DWORD()
            mx = wintypes.DWORD()
            if not dxva2.GetMonitorBrightness(
                self._hphys, ctypes.byref(mn), ctypes.byref(cur), ctypes.byref(mx)
            ):
                return None
            self._mn = int(mn.value)
            self._mx = int(mx.value)
            return _from_min_max(int(cur.value), self._mn, self._mx)
        except Exception:
            return None

    def set(self, value: float) -> bool:
        try:
            import ctypes
            from ctypes import wintypes

            if self._mx <= self._mn:
                return False
            dxva2 = ctypes.WinDLL("dxva2", use_last_error=True)
            level = _to_min_max(value, self._mn, self._mx)
            return bool(dxva2.SetMonitorBrightness(self._hphys, wintypes.DWORD(level)))
        except Exception:
            return False

    def close(self) -> None:
        try:
            import ctypes

            dxva2 = ctypes.WinDLL("dxva2", use_last_error=True)
            dxva2.DestroyPhysicalMonitors(self._count, self._array)
        except Exception:
            pass


class _LcdIoctlBackend:
    def __init__(self, handle) -> None:
        self._handle = handle

    def get(self) -> Optional[float]:
        try:
            import ctypes
            from ctypes import wintypes

            class _DISPLAY_BRIGHTNESS(ctypes.Structure):
                _fields_ = [
                    ("ucDisplayPolicy", ctypes.c_ubyte),
                    ("ucACBrightness", ctypes.c_ubyte),
                    ("ucDCBrightness", ctypes.c_ubyte),
                ]

            buf = _DISPLAY_BRIGHTNESS()
            returned = wintypes.DWORD()
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            ok = kernel32.DeviceIoControl(
                self._handle,
                IOCTL_VIDEO_QUERY_DISPLAY_BRIGHTNESS,
                None,
                0,
                ctypes.byref(buf),
                ctypes.sizeof(buf),
                ctypes.byref(returned),
                None,
            )
            if not ok:
                return None
            pct = int(buf.ucACBrightness or buf.ucDCBrightness)
            if pct < 0 or pct > 100:
                return None
            return pct / 100.0
        except Exception:
            return None

    def set(self, value: float) -> bool:
        try:
            import ctypes
            from ctypes import wintypes

            class _DISPLAY_BRIGHTNESS(ctypes.Structure):
                _fields_ = [
                    ("ucDisplayPolicy", ctypes.c_ubyte),
                    ("ucACBrightness", ctypes.c_ubyte),
                    ("ucDCBrightness", ctypes.c_ubyte),
                ]

            pct = int(round(_clamp01(value) * 100.0))
            buf = _DISPLAY_BRIGHTNESS()
            buf.ucDisplayPolicy = DISPLAYPOLICY_BOTH
            buf.ucACBrightness = pct
            buf.ucDCBrightness = pct
            returned = wintypes.DWORD()
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            return bool(
                kernel32.DeviceIoControl(
                    self._handle,
                    IOCTL_VIDEO_SET_DISPLAY_BRIGHTNESS,
                    ctypes.byref(buf),
                    ctypes.sizeof(buf),
                    None,
                    0,
                    ctypes.byref(returned),
                    None,
                )
            )
        except Exception:
            return False

    def close(self) -> None:
        try:
            import ctypes

            ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(self._handle)
        except Exception:
            pass


class _WmiBackend:
    def __init__(self, methods, monitor, keep) -> None:
        self._methods = methods
        self._monitor = monitor
        self._keep = keep  # locator + service must outlive the query objects

    def get(self) -> Optional[float]:
        try:
            pct = int(self._monitor.CurrentBrightness)
            if pct < 0 or pct > 100:
                return None
            return pct / 100.0
        except Exception:
            return None

    def set(self, value: float) -> bool:
        pct = int(round(_clamp01(value) * 100.0))
        try:
            # MSDN: WmiSetBrightness(Timeout, Brightness). Timeout 0 = now.
            self._methods.WmiSetBrightness(0, pct)
            return True
        except Exception:
            try:
                self._methods.WmiSetBrightness(Timeout=0, Brightness=pct)
                return True
            except Exception:
                return False

    def close(self) -> None:
        return


def _iter_swbem(objset) -> list:
    try:
        return list(objset)
    except Exception:
        pass
    try:
        count = int(objset.Count)
        items = []
        for i in range(count):
            try:
                items.append(objset.ItemIndex(i))
            except Exception:
                items.append(objset.Item(i + 1))
        return items
    except Exception:
        return []


def _win_hmonitors(hwnd: Optional[int]) -> list:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    if hwnd:
        try:
            user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
            user32.MonitorFromWindow.restype = wintypes.HMONITOR
            hmon = user32.MonitorFromWindow(int(hwnd), 2)  # MONITOR_DEFAULTTONEAREST
            if hmon:
                return [hmon]
        except Exception:
            pass
    found: list = []

    @ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(wintypes.RECT),
        wintypes.LPARAM,
    )
    def _cb(hmon, _hdc, _lprect, _lparam):
        found.append(hmon)
        return True

    try:
        user32.EnumDisplayMonitors(None, None, _cb, 0)
    except Exception:
        return []
    return found


def _try_dxva2(hwnd: Optional[int]) -> Optional[_Dxva2Backend]:
    try:
        import ctypes
        from ctypes import wintypes

        class PHYSICAL_MONITOR(ctypes.Structure):
            _fields_ = [
                ("hPhysicalMonitor", wintypes.HANDLE),
                ("szPhysicalMonitorDescription", wintypes.WCHAR * 128),
            ]

        dxva2 = ctypes.WinDLL("dxva2", use_last_error=True)
        dxva2.GetNumberOfPhysicalMonitorsFromHMONITOR.argtypes = [
            wintypes.HMONITOR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        dxva2.GetNumberOfPhysicalMonitorsFromHMONITOR.restype = wintypes.BOOL
        dxva2.GetPhysicalMonitorsFromHMONITOR.restype = wintypes.BOOL
        dxva2.GetMonitorBrightness.restype = wintypes.BOOL
        dxva2.SetMonitorBrightness.restype = wintypes.BOOL
        dxva2.DestroyPhysicalMonitors.restype = wintypes.BOOL

        for hmon in _win_hmonitors(hwnd):
            count = wintypes.DWORD()
            if not dxva2.GetNumberOfPhysicalMonitorsFromHMONITOR(
                hmon, ctypes.byref(count)
            ):
                continue
            if int(count.value) <= 0:
                continue
            arr = (PHYSICAL_MONITOR * int(count.value))()
            if not dxva2.GetPhysicalMonitorsFromHMONITOR(hmon, count, arr):
                continue
            backend = _Dxva2Backend(arr[0].hPhysicalMonitor, arr, int(count.value))
            if backend.get() is not None:
                return backend
            backend.close()
    except Exception:
        return None
    return None


def _try_lcd_ioctl() -> Optional[_LcdIoctlBackend]:
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        kernel32.CreateFileW.restype = wintypes.HANDLE
        GENERIC_READ = 0x80000000
        GENERIC_WRITE = 0x40000000
        FILE_SHARE_READ = 1
        FILE_SHARE_WRITE = 2
        OPEN_EXISTING = 3
        INVALID = wintypes.HANDLE(-1).value
        handle = kernel32.CreateFileW(
            "\\\\.\\LCD",
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            None,
            OPEN_EXISTING,
            0,
            None,
        )
        if not handle or int(handle) == int(INVALID):
            return None
        backend = _LcdIoctlBackend(handle)
        if backend.get() is not None:
            return backend
        backend.close()
    except Exception:
        return None
    return None


def _try_wmi() -> Optional[_WmiBackend]:
    try:
        import comtypes.client
    except Exception:
        return None
    try:
        locator = comtypes.client.CreateObject("WbemScripting.SWbemLocator")
        svc = locator.ConnectServer(".", "root\\wmi")
        monitors = _iter_swbem(svc.ExecQuery("SELECT * FROM WmiMonitorBrightness"))
        methods = _iter_swbem(
            svc.ExecQuery("SELECT * FROM WmiMonitorBrightnessMethods")
        )
        if not monitors or not methods:
            return None

        def _iname(obj) -> str:
            try:
                return str(getattr(obj, "InstanceName", "") or "")
            except Exception:
                return ""

        method_by_name = {_iname(m): m for m in methods if _iname(m)}
        for mon in monitors:
            try:
                if hasattr(mon, "Active") and not bool(mon.Active):
                    continue
            except Exception:
                pass
            matched = method_by_name.get(_iname(mon)) or methods[0]
            backend = _WmiBackend(matched, mon, (locator, svc, monitors, methods))
            if backend.get() is not None:
                return backend
    except Exception:
        return None
    return None


def _win_pick_backend(hwnd: Optional[int] = None) -> Optional[_BrightnessBackend]:
    for factory in (
        lambda: _try_dxva2(hwnd),
        _try_lcd_ioctl,
        _try_wmi,
    ):
        try:
            backend = factory()
        except Exception:
            backend = None
        if backend is not None:
            return backend
    return None


def get_brightness(
    display_id: Optional[int] = None, hwnd: Optional[int] = None
) -> Optional[float]:
    """Return current user brightness in 0..1, or None if unavailable."""
    if sys.platform == "darwin":
        if not _ensure_mac_loaded():
            return None
        backend = _MacBackend(display_id)
        val = backend.get()
        if val is not None:
            _set_active_backend(backend)
        return val
    if sys.platform == "win32":
        backend = _win_pick_backend(hwnd)
        if backend is None:
            _set_active_backend(None)
            return None
        val = backend.get()
        if val is None:
            _close_backend(backend)
            _set_active_backend(None)
            return None
        _set_active_backend(backend)
        return val
    return None


def set_brightness(
    value: float,
    display_id: Optional[int] = None,
    hwnd: Optional[int] = None,
) -> bool:
    """Set user brightness to ``value`` (clamped 0..1). Returns True on success."""
    v = _clamp01(value)
    if _active_backend is not None:
        try:
            if _active_backend.set(v):
                return True
        except Exception:
            pass
        _set_active_backend(None)
    if sys.platform == "darwin":
        if not _ensure_mac_loaded():
            return False
        backend = _MacBackend(display_id)
        if backend.set(v):
            _set_active_backend(backend)
            return True
        return False
    if sys.platform == "win32":
        backend = _win_pick_backend(hwnd)
        if backend is None:
            return False
        ok = bool(backend.set(v))
        if ok:
            _set_active_backend(backend)
        else:
            _close_backend(backend)
        return ok
    return False
