from typing import Any, BinaryIO, Self, Tuple
from pathlib import Path

from numpy.typing import NDArray

MAX_IMAGE_PIXELS: int | None

class Transpose:
    FLIP_LEFT_RIGHT: int
    FLIP_TOP_BOTTOM: int
    ROTATE_90: int
    ROTATE_180: int
    ROTATE_270: int
    TRANSPOSE: int
    TRANSVERSE: int

class Resampling:
    NEAREST: int
    BOX: int
    BILINEAR: int
    HAMMING: int
    BICUBIC: int
    LANCZOS: int

class Image:
    size: Tuple[int, int]
    mode: str
    info: dict[str, Any]

    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...

    def convert(self, mode: str) -> Image: ...
    def copy(self) -> Image: ...
    def seek(self, frame: int) -> None: ...
    def getexif(self) -> Any: ...
    def tobytes(self, encoder_name: str = ..., *args: Any) -> bytes: ...
    def resize(
        self,
        size: Tuple[int, int],
        resample: int = ...,
        box: Tuple[int, int, int, int] | None = None,
    ) -> Image: ...
    def thumbnail(
        self,
        size: Tuple[int, int],
        resample: int = ...,
    ) -> None: ...
    def transpose(self, method: int) -> Image: ...
    def save(
        self,
        fp: str | bytes | Path | BinaryIO,
        format: str | None = ...,
        **params: Any,
    ) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None: ...

class ImageFile(Image):
    n_frames: int
    is_animated: bool

    def _getexif(self) -> dict[Any, Any] | None: ...

_FilePath = str | bytes | Path

def open(
    fp: _FilePath | BinaryIO,
    mode: str = "r",
    formats: list[str] | tuple[str, ...] | None = None,
) -> ImageFile: ...

def fromarray(obj: NDArray[Any], mode: str | None = None) -> Image: ...
