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
