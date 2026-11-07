"""
Lowercase RAW file extensions (without leading dot) treated as camera/LibRaw RAW in RAWviewer.

Used by gallery `format:raw` filtering and `is_raw_file()`. Aligned with README “Supported
Formats” plus common LibRaw-supported types (not every exotic variant).
"""

from __future__ import annotations

RAW_FILE_EXTENSIONS: frozenset[str] = frozenset(
    {
        # README-listed & app paths
        "3fr",
        "arw",
        "cr2",
        "cr3",
        "dng",
        "erf",
        "iiq",
        "nef",
        "nrw",
        "orf",
        "pef",
        "raf",
        "rw2",
        "srw",
        "x3f",
        # common_image_loader / main.py historical set
        "cap",
        "fff",
        "mef",
        "mos",
        "rwl",
        "srf",
        # Widely used LibRaw / manufacturer extensions
        "ari",
        "arq",
        "bay",
        "bmq",
        "braw",
        "c1a",
        "c1b",
        "crw",
        "crm",
        "cs1",
        "dc2",
        "dcr",
        "dcs",
        "drf",
        "eip",
        "gpr",
        "j6c",
        "k25",
        "kdc",
        "mdc",
        "mfw",
        "mrw",
        "ori",
        "ptx",
        "pxn",
        "qtk",
        "r3d",
        "raw",
        "rwz",
        "sr2",
        "sti",
    }
)

# Standard (non-RAW) image extensions opened by the viewer (without leading dot).
STANDARD_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {
        "jpeg",
        "jpg",
        "png",
        "gif",
        "webp",
        "heif",
        "heic",
        "avif",
        "tif",
        "tiff",
    }
)


def get_supported_extensions() -> list[str]:
    """Supported file extensions with leading dot (RAW + standard images)."""
    return [
        # RAW formats (aligned with main.py / README)
        ".cr2",
        ".cr3",
        ".nef",
        ".arw",
        ".dng",
        ".orf",
        ".rw2",
        ".pef",
        ".srw",
        ".x3f",
        ".raf",
        ".3fr",
        ".fff",
        ".iiq",
        ".cap",
        ".erf",
        ".mef",
        ".mos",
        ".nrw",
        ".rwl",
        ".srf",
        # Standard image formats
        ".jpeg",
        ".jpg",
        ".png",
        ".gif",
        ".webp",
        ".heif",
        ".heic",
        ".avif",
        ".tif",
        ".tiff",
    ]
