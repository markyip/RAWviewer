"""Windows app discovery: classic EXE installs and Microsoft Store / MSIX packages."""

from __future__ import annotations

import glob
import os
import subprocess
import sys
from typing import Callable, List, Optional, Sequence

# Returned when a Store package is present but no stable public exe path is available.
INSTALLED_MARKER = "__installed__"


def first_existing(candidates: Sequence[str]) -> Optional[str]:
    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    return None


def glob_first(patterns: Sequence[str]) -> Optional[str]:
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return os.path.abspath(matches[-1])
    return None


def program_files_paths(*relative: str) -> List[str]:
    roots: List[str] = []
    for env in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(env)
        if root:
            roots.append(root)
    return [os.path.join(root, rel) for root in roots for rel in relative]


def local_appdata_paths(*relative: str) -> List[str]:
    root = os.environ.get("LocalAppData", "")
    if not root:
        return []
    return [os.path.join(root, rel) for rel in relative]


def appdata_paths(*relative: str) -> List[str]:
    root = os.environ.get("APPDATA", "")
    if not root:
        return []
    return [os.path.join(root, rel) for rel in relative]


def windowsapps_alias_root() -> str:
    return os.path.join(
        os.environ.get("LocalAppData", ""),
        "Microsoft",
        "WindowsApps",
    )


def packages_root() -> str:
    return os.path.join(os.environ.get("LocalAppData", ""), "Packages")


def windowsapps_aliases(*names: str) -> Optional[str]:
    """MSIX app execution aliases (may be 0-byte reparse points; still launchable)."""
    root = windowsapps_alias_root()
    if not root or not os.path.isdir(root):
        return None
    for name in names:
        path = os.path.join(root, name)
        if os.path.isfile(path):
            return os.path.abspath(path)
    return None


def windowsapps_package_exe(*glob_patterns: str) -> Optional[str]:
    root = windowsapps_alias_root()
    if not root or not os.path.isdir(root):
        return None
    for pattern in glob_patterns:
        for path in sorted(glob.glob(os.path.join(root, pattern))):
            if os.path.isfile(path):
                return os.path.abspath(path)
    return None


def program_files_windowsapps_exe(*glob_patterns: str) -> Optional[str]:
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    if not pf:
        return None
    patterns = [
        os.path.join(pf, "WindowsApps", pattern) for pattern in glob_patterns
    ]
    return glob_first(patterns)


def store_packages_installed(*glob_patterns: str) -> bool:
    root = packages_root()
    if not root or not os.path.isdir(root):
        return False
    for pattern in glob_patterns:
        if glob.glob(os.path.join(root, pattern)):
            return True
    return False


def detect_windows_app(
    *,
    file_paths: Sequence[str] = (),
    program_files_rel: Sequence[str] = (),
    local_appdata_rel: Sequence[str] = (),
    appdata_rel: Sequence[str] = (),
    glob_patterns: Sequence[str] = (),
    windowsapps_aliases_names: Sequence[str] = (),
    windowsapps_package_globs: Sequence[str] = (),
    windowsapps_pf_globs: Sequence[str] = (),
    packages_globs: Sequence[str] = (),
) -> Optional[str]:
    """
    Find a Windows app executable, checking classic installs then Store / MSIX.

    Returns an absolute exe path, or INSTALLED_MARKER when a Store package exists
    without a readable exe (menu still shows the app; caller may use Windows Share).
    """
    path = first_existing(file_paths)
    if path:
        return path

    path = first_existing(program_files_paths(*program_files_rel))
    if path:
        return path

    path = first_existing(local_appdata_paths(*local_appdata_rel))
    if path:
        return path

    path = first_existing(appdata_paths(*appdata_rel))
    if path:
        return path

    path = glob_first(glob_patterns)
    if path:
        return path

    path = windowsapps_aliases(*windowsapps_aliases_names)
    if path:
        return path

    path = windowsapps_package_exe(*windowsapps_package_globs)
    if path:
        return path

    path = program_files_windowsapps_exe(*windowsapps_pf_globs)
    if path:
        return path

    if packages_globs and store_packages_installed(*packages_globs):
        return INSTALLED_MARKER

    return None


def is_installed_marker(path: Optional[str]) -> bool:
    return path == INSTALLED_MARKER


def is_usable_exe(path: Optional[str]) -> bool:
    """True when *path* can be passed to CreateProcess / QFileIconProvider."""
    return bool(path) and not is_installed_marker(path) and os.path.isfile(path)


def shell_open_with_files(exe: str, paths: Sequence[str]) -> bool:
    """Launch *exe* with file path arguments (CreateProcess on Windows; MSIX aliases need shell_open_documents)."""
    valid = [os.path.abspath(p) for p in paths if p and os.path.isfile(p)]
    if not is_usable_exe(exe) or not valid:
        return False
    try:
        subprocess.Popen(
            [exe, *valid],
            cwd=os.path.dirname(exe) or None,
            close_fds=True,
        )
        return True
    except Exception:
        return False


def shell_open_documents(paths: Sequence[str]) -> bool:
    """Open files with their registered handler (ShellExecute verb=open, lpFile=document)."""
    valid = [os.path.abspath(p) for p in paths if p and os.path.isfile(p)]
    if not valid:
        return False
    if sys.platform != "win32":
        return False
    opened = False
    try:
        import ctypes

        for path in valid:
            work_dir = os.path.dirname(path) or None
            ret = int(
                ctypes.windll.shell32.ShellExecuteW(None, "open", path, None, work_dir, 1)
            )
            if ret > 32:
                opened = True
    except Exception:
        return False
    return opened


def icon_path_for_detected(path: Optional[str]) -> str:
    """Path suitable for QFileIconProvider; empty when unknown."""
    if is_usable_exe(path):
        return path or ""
    return ""


def detect_first(*detectors: Callable[[], Optional[str]]) -> Optional[str]:
    for detect in detectors:
        path = detect()
        if path:
            return path
    return None
