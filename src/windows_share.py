"""Windows system share sheet via DataTransferManager + IDataTransferManagerInterop."""

from __future__ import annotations

import asyncio
import ctypes
import os
import sys
import threading
from ctypes import (
    HRESULT,
    POINTER,
    Structure,
    WINFUNCTYPE,
    byref,
    c_uint,
    c_uint32,
    c_ulong,
    c_ushort,
    c_void_p,
    c_wchar_p,
)
from typing import Callable, Optional

if sys.platform != "win32":
    _share_file_windows = None  # type: ignore[misc, assignment]
else:
    import comtypes
    from comtypes import COMMETHOD, GUID, IUnknown

    _WINRT_INIT_LOCK = threading.Lock()
    _WINRT_INITIALIZED = False

    _DTM_IID = GUID("{A5CAEE9B-8708-49D1-8D36-67D25A8DA00C}")
    _INTEROP_IID = GUID("{3A3DCD6C-3EAB-43DC-BCDE-45671CE800C8}")
    _DTM_CLASS = "Windows.ApplicationModel.DataTransfer.DataTransferManager"

    class _WinGuid(Structure):
        _fields_ = [
            ("Data1", c_ulong),
            ("Data2", c_ushort),
            ("Data3", c_ushort),
            ("Data4", ctypes.c_ubyte * 8),
        ]

        @classmethod
        def from_str(cls, value: str) -> "_WinGuid":
            import uuid

            u = uuid.UUID(value)
            g = cls()
            g.Data1 = u.time_low
            g.Data2 = u.time_mid
            g.Data3 = u.time_hi_version
            for i, b in enumerate(u.bytes[8:]):
                g.Data4[i] = b
            return g

    class IDataTransferManagerInterop(IUnknown):
        _iid_ = _INTEROP_IID
        _methods_ = [
            COMMETHOD(
                [],
                HRESULT,
                "GetForWindow",
                (["in"], c_void_p, "appWindow"),
                (["in"], POINTER(GUID), "riid"),
                (["out"], POINTER(c_void_p), "dataTransferManager"),
            ),
            COMMETHOD(
                [],
                HRESULT,
                "ShowShareUIForWindow",
                (["in"], c_void_p, "appWindow"),
            ),
        ]

    class _ShareSession:
        __slots__ = ("hwnd", "path", "token", "handler", "dtm")

        def __init__(self, hwnd: int) -> None:
            self.hwnd = hwnd
            self.path = ""
            self.token = None
            self.handler: Optional[Callable] = None
            self.dtm = None

    _SESSIONS: dict[int, _ShareSession] = {}
    _SESSIONS_LOCK = threading.Lock()
    _INTEROP_CACHE: Optional[object] = None

    def _ensure_winrt_apartment() -> None:
        global _WINRT_INITIALIZED
        with _WINRT_INIT_LOCK:
            if _WINRT_INITIALIZED:
                return
            from winrt.runtime import STA, init_apartment

            init_apartment(STA)
            _WINRT_INITIALIZED = True

    def _get_interop() -> object:
        global _INTEROP_CACHE
        if _INTEROP_CACHE is not None:
            return _INTEROP_CACHE

        combase = ctypes.windll.combase
        WindowsCreateString = combase.WindowsCreateString
        WindowsCreateString.argtypes = [c_wchar_p, c_uint32, POINTER(c_void_p)]
        WindowsCreateString.restype = HRESULT
        WindowsDeleteString = combase.WindowsDeleteString
        WindowsDeleteString.argtypes = [c_void_p]
        WindowsDeleteString.restype = HRESULT
        RoGetActivationFactory = combase.RoGetActivationFactory
        RoGetActivationFactory.argtypes = [c_void_p, POINTER(_WinGuid), POINTER(c_void_p)]
        RoGetActivationFactory.restype = HRESULT

        hstring = c_void_p()
        hr = WindowsCreateString(_DTM_CLASS, len(_DTM_CLASS), byref(hstring))
        if hr & 0xFFFFFFFF:
            raise OSError(hr, "WindowsCreateString failed")

        raw = c_void_p()
        try:
            hr = RoGetActivationFactory(
                hstring,
                byref(_WinGuid.from_str("{3A3DCD6C-3EAB-43DC-BCDE-45671CE800C8}")),
                byref(raw),
            )
            if hr & 0xFFFFFFFF:
                raise OSError(hr, "RoGetActivationFactory failed")
        finally:
            WindowsDeleteString(hstring)

        interop = comtypes.cast(raw, POINTER(IUnknown)).QueryInterface(
            IDataTransferManagerInterop
        )
        _INTEROP_CACHE = interop
        return interop

    def _object_from_abi_ptr(ptr: int, target_type: type) -> object:
        """Best-effort wrap of a WinRT ABI pointer as a pywinrt projected type."""
        if not ptr:
            raise ValueError("null ABI pointer")

        # pywinrt 3.x: _from expects an existing projected Object; try known fallbacks.
        try:
            from winrt.system import Object as WinrtObject

            return target_type._from(WinrtObject(ptr))  # type: ignore[attr-defined]
        except Exception:
            pass

        try:
            return target_type._from(ptr)  # type: ignore[attr-defined]
        except Exception:
            pass

        unk = comtypes.cast(ptr, POINTER(IUnknown))
        try:
            return target_type._from(unk)  # type: ignore[attr-defined]
        except Exception as exc:
            raise TypeError(f"unable to project {target_type.__name__} from ABI pointer") from exc

    def _make_data_requested_handler(session: _ShareSession):
        from winrt.windows.applicationmodel.datatransfer import DataPackageOperation
        from winrt.windows.storage import StorageFile

        async def _fill_request(args) -> None:
            deferral = args.request.get_deferral()
            try:
                storage_file = await StorageFile.get_file_from_path_async(session.path)
                args.request.data.set_storage_items([storage_file])
                args.request.data.properties.title = os.path.basename(session.path)
                args.request.data.requested_operation = DataPackageOperation.COPY
            finally:
                deferral.complete()

        def _on_data_requested(_sender, args) -> None:
            try:
                asyncio.run(_fill_request(args))
            except RuntimeError:
                # If an event loop is already running in this thread, spin a fresh one.
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(_fill_request(args))
                finally:
                    loop.close()

        return _on_data_requested

    def _ensure_session(hwnd: int) -> _ShareSession:
        with _SESSIONS_LOCK:
            session = _SESSIONS.get(hwnd)
            if session is not None:
                return session

            _ensure_winrt_apartment()
            from winrt.windows.applicationmodel.datatransfer import DataTransferManager

            interop = _get_interop()
            dtm_ptr = interop.GetForWindow(hwnd, _DTM_IID)
            dtm = _object_from_abi_ptr(int(dtm_ptr), DataTransferManager)

            session = _ShareSession(hwnd)
            session.dtm = dtm
            session.handler = _make_data_requested_handler(session)
            session.token = dtm.add_data_requested(session.handler)
            _SESSIONS[hwnd] = session
            return session

    def share_file_windows(path: str, owner_hwnd: int = 0) -> bool:
        """Open the Windows share UI for a file path anchored to owner_hwnd."""
        abs_path = os.path.abspath(path)
        if not owner_hwnd or not os.path.isfile(abs_path):
            return False

        try:
            _ensure_winrt_apartment()
            session = _ensure_session(int(owner_hwnd))
            session.path = abs_path
            interop = _get_interop()
            interop.ShowShareUIForWindow(int(owner_hwnd))
            return True
        except Exception:
            return False

    _share_file_windows = share_file_windows

if sys.platform != "win32":

    def share_file_windows(path: str, owner_hwnd: int = 0) -> bool:
        return False

else:
    share_file_windows = _share_file_windows  # type: ignore[misc, assignment]
