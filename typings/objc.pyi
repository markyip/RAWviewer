"""Minimal PyObjC stub for cross-platform type checking (macOS-only at runtime)."""

from typing import Any, ContextManager, TypeVar

_C_OUT: str
_C_PTR: str
_C_LNG: str

def registerMetaDataForSelector(cls_name: bytes, sel: bytes, meta: dict[str, Any]) -> None: ...

_T = TypeVar("_T")

class _SuperProxy:
    def init(self) -> Any: ...

def super(cls: type, self: Any) -> _SuperProxy: ...

def lookUpClass(name: str) -> Any: ...

def objc_object(*, c_void_p: int) -> Any: ...

def autorelease_pool() -> ContextManager[None]: ...
