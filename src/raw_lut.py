"""Creative 3D LUT (.cube) load / apply for the Adjust pipeline.

LUTs are managed as files under the app support ``luts/`` directory. The
active look is referenced from XMP via ``CreativeLUTName`` (basename),
``CreativeLUTAmount`` (0–100 strength; 0 = off), and
``CreativeLUTWorkingSpace`` (``Rec709`` | ``Linear``).

Default working space is **Rec709**: apply after the BT.709/dcraw gamma
encode so typical creative .cube files (Resolve / PS display-referred)
land in the domain they expect. **Linear** restores the older
display-linear (pre-encode) apply for scene-referred / identity cubes.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from typing import Optional

import numpy as np

LUT_NAME_KEY = "CreativeLUTName"
LUT_AMOUNT_KEY = "CreativeLUTAmount"
LUT_SPACE_KEY = "CreativeLUTWorkingSpace"
LUT_SPACE_REC709 = "Rec709"
LUT_SPACE_LINEAR = "Linear"
LUT_SPACE_DEFAULT = LUT_SPACE_REC709
LUT_KEYS = (LUT_NAME_KEY, LUT_AMOUNT_KEY, LUT_SPACE_KEY)


@dataclass
class CubeLUT:
    title: str
    size: int
    data: np.ndarray  # (N,N,N,3) float32 RGB, index order [b,g,r] per .cube convention
    domain_min: np.ndarray
    domain_max: np.ndarray


_LUT_CACHE: dict[str, CubeLUT] = {}


def luts_dir() -> str:
    root = ""
    try:
        from PyQt6.QtCore import QStandardPaths

        root = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppDataLocation
        )
    except Exception:
        root = ""
    # Offscreen / unnamed QApplication can yield a useless path like "/…/-c".
    if not root or os.path.basename(root.rstrip(os.sep)) in {"", "-", "-c"}:
        root = os.path.join(os.path.expanduser("~"), ".rawviewer")
    path = os.path.join(root, "luts")
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        path = os.path.join(os.path.expanduser("~"), ".rawviewer", "luts")
        os.makedirs(path, exist_ok=True)
    return path


def list_managed_luts() -> list[str]:
    """Basenames of managed ``.cube`` files (sorted)."""
    d = luts_dir()
    names = [f for f in os.listdir(d) if f.lower().endswith(".cube")]
    return sorted(names, key=str.lower)


def import_cube_file(src_path: str) -> str:
    """Copy a ``.cube`` into the managed folder; return basename."""
    if not src_path or not os.path.isfile(src_path):
        raise FileNotFoundError(src_path)
    if not src_path.lower().endswith(".cube"):
        raise ValueError("Only .cube LUT files are supported")
    # Validate parse before copying.
    parse_cube_file(src_path)
    dest_name = os.path.basename(src_path)
    dest = os.path.join(luts_dir(), dest_name)
    if os.path.abspath(src_path) != os.path.abspath(dest):
        shutil.copy2(src_path, dest)
    _LUT_CACHE.pop(dest, None)
    return dest_name


def remove_managed_lut(name: str) -> None:
    path = os.path.join(luts_dir(), os.path.basename(name))
    if os.path.isfile(path):
        os.remove(path)
    _LUT_CACHE.pop(path, None)


def managed_lut_path(name: str) -> str:
    return os.path.join(luts_dir(), os.path.basename(name or ""))


def parse_cube_file(path: str) -> CubeLUT:
    """Parse an Adobe/.cube 3D LUT file."""
    size = None
    title = os.path.splitext(os.path.basename(path))[0]
    dmin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    dmax = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    rows: list[list[float]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            up = line.upper()
            if up.startswith("TITLE"):
                m = re.search(r'TITLE\s+"([^"]*)"', line, re.I)
                if m:
                    title = m.group(1)
                continue
            if up.startswith("LUT_3D_SIZE"):
                size = int(line.split()[-1])
                continue
            if up.startswith("DOMAIN_MIN"):
                parts = line.split()
                dmin = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)
                continue
            if up.startswith("DOMAIN_MAX"):
                parts = line.split()
                dmax = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)
                continue
            if up.startswith("LUT_1D_SIZE"):
                raise ValueError("1D .cube LUTs are not supported (need LUT_3D_SIZE)")
            parts = line.split()
            if len(parts) >= 3:
                try:
                    rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue
    if not size or size < 2:
        raise ValueError(f"Invalid or missing LUT_3D_SIZE in {path}")
    expected = size * size * size
    if len(rows) < expected:
        raise ValueError(f"Expected {expected} LUT rows, got {len(rows)}")
    data = np.asarray(rows[:expected], dtype=np.float32).reshape(size, size, size, 3)
    return CubeLUT(title=title, size=size, data=data, domain_min=dmin, domain_max=dmax)


def load_managed_lut(name: str) -> Optional[CubeLUT]:
    path = managed_lut_path(name)
    if not name or not os.path.isfile(path):
        return None
    cached = _LUT_CACHE.get(path)
    if cached is not None:
        return cached
    lut = parse_cube_file(path)
    _LUT_CACHE[path] = lut
    return lut


def lut_working_space(adj: dict | None) -> str:
    """Return ``Rec709`` (default) or ``Linear`` from adjustments."""
    if not adj:
        return LUT_SPACE_DEFAULT
    raw = str(adj.get(LUT_SPACE_KEY, "") or "").strip()
    if not raw:
        return LUT_SPACE_DEFAULT
    key = raw.lower().replace(".", "").replace("_", "").replace("-", "")
    if key in ("linear", "lin", "scene", "scenelinear"):
        return LUT_SPACE_LINEAR
    if key in ("rec709", "709", "bt709", "gamma", "display", "srgb"):
        return LUT_SPACE_REC709
    return LUT_SPACE_DEFAULT


def apply_cube_lut(img: np.ndarray, lut: CubeLUT, amount: float = 100.0) -> np.ndarray:
    """Apply a 3D LUT to RGB float [0,1] (any transfer the caller chose)."""
    if img is None or lut is None:
        return img
    a = float(amount) / 100.0
    if a <= 1e-4:
        return img
    a = min(1.0, max(0.0, a))
    rgb = np.clip(img.astype(np.float32, copy=False), 0.0, 1.0)
    # Map into LUT domain.
    span = np.maximum(lut.domain_max - lut.domain_min, 1e-6)
    x = (rgb - lut.domain_min) / span
    x = np.clip(x, 0.0, 1.0)
    n = lut.size
    # .cube data order: R changes fastest, then G, then B → reshape (B,G,R,3)
    # After reshape(size,size,size,3) with rows in that order, index [b,g,r].
    grid = lut.data
    coords = x * (n - 1)
    r0 = np.floor(coords[..., 0]).astype(np.int32)
    g0 = np.floor(coords[..., 1]).astype(np.int32)
    b0 = np.floor(coords[..., 2]).astype(np.int32)
    r1 = np.minimum(r0 + 1, n - 1)
    g1 = np.minimum(g0 + 1, n - 1)
    b1 = np.minimum(b0 + 1, n - 1)
    fr = coords[..., 0] - r0
    fg = coords[..., 1] - g0
    fb = coords[..., 2] - b0

    def sample(bi, gi, ri):
        return grid[bi, gi, ri]

    c000 = sample(b0, g0, r0)
    c100 = sample(b0, g0, r1)
    c010 = sample(b0, g1, r0)
    c110 = sample(b0, g1, r1)
    c001 = sample(b1, g0, r0)
    c101 = sample(b1, g0, r1)
    c011 = sample(b1, g1, r0)
    c111 = sample(b1, g1, r1)
    fr = fr[..., None]
    fg = fg[..., None]
    fb = fb[..., None]
    c00 = c000 * (1 - fr) + c100 * fr
    c10 = c010 * (1 - fr) + c110 * fr
    c01 = c001 * (1 - fr) + c101 * fr
    c11 = c011 * (1 - fr) + c111 * fr
    c0 = c00 * (1 - fg) + c10 * fg
    c1 = c01 * (1 - fg) + c11 * fg
    mapped = c0 * (1 - fb) + c1 * fb
    if a >= 1.0 - 1e-4:
        return np.clip(mapped, 0.0, 1.0)
    return np.clip(rgb * (1.0 - a) + mapped * a, 0.0, 1.0)


def apply_creative_lut(img: np.ndarray, adj: dict | None) -> np.ndarray:
    """Apply managed LUT to float RGB [0,1] (caller chooses Linear vs encoded domain)."""
    if img is None or not adj:
        return img
    name = str(adj.get(LUT_NAME_KEY, "") or "").strip()
    amount = float(adj.get(LUT_AMOUNT_KEY, 0.0) or 0.0)
    if not name or abs(amount) < 1e-3:
        return img
    lut = load_managed_lut(name)
    if lut is None:
        return img
    return apply_cube_lut(img, lut, amount)


def apply_creative_lut_encoded(
    encoded: np.ndarray, adj: dict | None, peak: float
) -> np.ndarray:
    """Apply LUT on display-encoded pixels (uint8/uint16), return same dtype."""
    if encoded is None or not adj:
        return encoded
    name = str(adj.get(LUT_NAME_KEY, "") or "").strip()
    amount = float(adj.get(LUT_AMOUNT_KEY, 0.0) or 0.0)
    if not name or abs(amount) < 1e-3:
        return encoded
    peak_f = float(peak) if peak else 255.0
    f = encoded.astype(np.float32) / peak_f
    out = apply_creative_lut(f, adj)
    if encoded.dtype == np.uint8 or peak_f <= 255.5:
        return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    max_v = float(np.iinfo(encoded.dtype).max) if np.issubdtype(encoded.dtype, np.integer) else peak_f
    return np.clip(out * peak_f + 0.5, 0, max_v).astype(encoded.dtype)


def apply_pipeline_lut_linear(display_linear: np.ndarray, adj: dict | None) -> np.ndarray:
    """Pre-encode apply when working space is Linear."""
    if lut_working_space(adj) != LUT_SPACE_LINEAR:
        return display_linear
    return apply_creative_lut(display_linear, adj)


def apply_pipeline_lut_encoded(
    encoded: np.ndarray, adj: dict | None, peak: float
) -> np.ndarray:
    """Post-gamma apply when working space is Rec709 (default)."""
    if lut_working_space(adj) == LUT_SPACE_LINEAR:
        return encoded
    return apply_creative_lut_encoded(encoded, adj, peak)
