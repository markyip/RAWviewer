"""Apps that accept multiple image paths on the command line (batch open)."""

from __future__ import annotations

import glob
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from win_app_detect import (
    detect_windows_app,
    first_existing,
    glob_first,
    icon_path_for_detected,
    is_installed_marker,
    is_usable_exe,
    local_appdata_paths,
    program_files_paths,
    shell_open_documents,
    shell_open_with_files,
)


def _mac_app(*names: str) -> Optional[str]:
    for name in names:
        path = os.path.join("/Applications", name)
        if os.path.isdir(path):
            return path
    return None


@dataclass(frozen=True)
class BatchOpenAppDef:
    id: str
    display_name: str
    detect: Callable[[], Optional[str]]
    launch: Callable[[str, Sequence[str]], bool]


def _win_program_files(*rel: str) -> List[str]:
    return program_files_paths(*rel)


def _detect_win_lightroom_classic() -> Optional[str]:
    return first_existing(
        _win_program_files(
            r"Adobe\Adobe Lightroom Classic\Lightroom.exe",
            r"Adobe\Adobe Lightroom Classic CC\Lightroom.exe",
        )
    )


def _detect_win_lightroom_cc() -> Optional[str]:
    return first_existing(
        _win_program_files(
            r"Adobe\Adobe Lightroom CC\Lightroom.exe",
            r"Adobe\Adobe Lightroom\Lightroom.exe",
        )
        + local_appdata_paths(r"Programs\Adobe Lightroom CC\Lightroom.exe")
    )


def _detect_win_photoshop() -> Optional[str]:
    pf_roots = [
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
    ]
    patterns = [
        os.path.join(root, "Adobe", "Adobe Photoshop *", "Photoshop.exe")
        for root in pf_roots
        if root
    ]
    return glob_first(patterns)


def _detect_win_capture_one() -> Optional[str]:
    return first_existing(
        _win_program_files(
            r"Capture One\Capture One.exe",
            r"Phase One\Capture One\Capture One.exe",
        )
    ) or glob_first(
        [
            os.path.join(
                os.environ.get("ProgramFiles", r"C:\Program Files"),
                "Capture One",
                "Capture One*.exe",
            )
        ]
    )


def _detect_win_dxo_photolab() -> Optional[str]:
    patterns = []
    for root in (
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
    ):
        if not root:
            continue
        patterns.extend(
            [
                os.path.join(root, "DxO", "DxO PhotoLab *", "DxO.PhotoLab.exe"),
                os.path.join(root, "DxO", "DxO PhotoLab *", "PhotoLab.exe"),
            ]
        )
    return glob_first(patterns)


def _detect_win_on1_photo_raw() -> Optional[str]:
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    if not pf:
        return None
    on1_root = os.path.join(pf, "ON1")
    if not os.path.isdir(on1_root):
        return None
    patterns: List[str] = []
    for folder in sorted(glob.glob(os.path.join(on1_root, "ON1 Photo RAW*"))):
        patterns.extend(
            glob.glob(os.path.join(folder, "ON1 Photo RAW *.exe"))
            + glob.glob(os.path.join(folder, "ON1 Photo RAW.exe"))
        )
    if not patterns:
        return None
    return os.path.abspath(sorted(patterns)[-1])


def _detect_win_affinity(
    *,
    folder_name: str,
    exe_name: str,
    alias_exe: str,
    package_prefix: str,
) -> Optional[str]:
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    product_glob = folder_name.rsplit(" ", 1)[0] + " *"
    return detect_windows_app(
        program_files_rel=(
            f"Serif\\{folder_name}\\{exe_name}",
            f"Affinity\\{folder_name}\\{exe_name}",
            f"Affinity\\{folder_name}\\{folder_name}.exe",
        ),
        glob_patterns=[
            os.path.join(pf, "Serif", product_glob, exe_name),
            os.path.join(pf, "Affinity", product_glob, exe_name),
            os.path.join(pf, "Affinity", product_glob, f"{folder_name}.exe"),
        ],
        windowsapps_aliases_names=(alias_exe,),
        windowsapps_package_globs=(f"{package_prefix}_*\\{alias_exe}",),
        windowsapps_pf_globs=(
            f"{package_prefix}_*\\App\\{exe_name}",
            f"*{alias_exe.replace('.exe', '')}*\\App\\{exe_name}",
        ),
        packages_globs=(f"{package_prefix}_*",),
    )


def _detect_win_affinity_photo_2() -> Optional[str]:
    return _detect_win_affinity(
        folder_name="Affinity Photo 2",
        exe_name="Photo.exe",
        alias_exe="AffinityPhoto2.exe",
        package_prefix="SerifEuropeLtd.AffinityPhoto2",
    )


def _detect_win_affinity_designer_2() -> Optional[str]:
    return _detect_win_affinity(
        folder_name="Affinity Designer 2",
        exe_name="Designer.exe",
        alias_exe="AffinityDesigner2.exe",
        package_prefix="SerifEuropeLtd.AffinityDesigner2",
    )


def _detect_win_affinity_publisher_2() -> Optional[str]:
    return _detect_win_affinity(
        folder_name="Affinity Publisher 2",
        exe_name="Publisher.exe",
        alias_exe="AffinityPublisher2.exe",
        package_prefix="SerifEuropeLtd.AffinityPublisher2",
    )


def _canva_affinity_exe() -> Optional[str]:
    path = os.path.join(
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        "Affinity",
        "Affinity",
        "Affinity.exe",
    )
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return os.path.abspath(path)
    return None


def _win_affinity_launch_exe(app_id: str) -> Optional[str]:
    """
    Real executable that accepts file paths on the command line.

    MSIX app execution aliases (0-byte reparse points under WindowsApps) return
    ERROR_ACCESS_DENIED when invoked with file arguments; prefer classic installs
    or the Canva unified Affinity binary for Photo.
    """
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    rel_by_id = {
        "affinity_photo_2": (
            r"Serif\Affinity Photo 2\Photo.exe",
            r"Affinity\Affinity Photo 2\Photo.exe",
        ),
        "affinity_designer_2": (
            r"Serif\Affinity Designer 2\Designer.exe",
            r"Affinity\Affinity Designer 2\Designer.exe",
        ),
        "affinity_publisher_2": (
            r"Serif\Affinity Publisher 2\Publisher.exe",
            r"Affinity\Affinity Publisher 2\Publisher.exe",
        ),
    }
    for rel in rel_by_id.get(app_id, ()):
        path = os.path.join(pf, rel)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return os.path.abspath(path)
    if app_id == "affinity_photo_2":
        return _canva_affinity_exe()
    return None


def _launch_exe(exe: str, paths: Sequence[str]) -> bool:
    return shell_open_with_files(exe, paths)


def _launch_affinity_win(app_id: str, paths: Sequence[str]) -> bool:
    exe = _win_affinity_launch_exe(app_id)
    if exe:
        return shell_open_with_files(exe, paths)
    # MSIX-only install: classic exe missing — open via file association (verb=open).
    return shell_open_documents(paths)


def _launch_mac_app(app_path: str, paths: Sequence[str]) -> bool:
    valid = [os.path.abspath(p) for p in paths if p and os.path.isfile(p)]
    if not app_path or not os.path.isdir(app_path) or not valid:
        return False
    try:
        subprocess.Popen(["open", "-a", app_path, *valid], close_fds=True)
        return True
    except Exception:
        return False


def _mac_detectors() -> List[BatchOpenAppDef]:
    specs = [
        (
            "lightroom_classic",
            "Lightroom Classic",
            lambda: _mac_app(
                "Adobe Lightroom Classic.app",
                "Adobe Lightroom Classic CC.app",
            ),
        ),
        (
            "lightroom_cc",
            "Lightroom",
            lambda: _mac_app("Adobe Lightroom.app", "Adobe Lightroom CC.app"),
        ),
        (
            "photoshop",
            "Photoshop",
            lambda: glob_first(["/Applications/Adobe Photoshop *.app"])
            or _mac_app("Adobe Photoshop 2024.app", "Adobe Photoshop 2025.app"),
        ),
        (
            "capture_one",
            "Capture One",
            lambda: _mac_app("Capture One.app")
            or glob_first(["/Applications/Capture One*.app"]),
        ),
        (
            "dxo_photolab",
            "DxO PhotoLab",
            lambda: glob_first(["/Applications/DxO PhotoLab *.app"])
            or _mac_app(
                "DxO PhotoLab 8.app",
                "DxO PhotoLab 7.app",
                "DxO PhotoLab 6.app",
            ),
        ),
        (
            "on1_photo_raw",
            "ON1 Photo RAW",
            lambda: glob_first(["/Applications/ON1 Photo RAW*.app"])
            or _mac_app(
                "ON1 Photo RAW 2026.app",
                "ON1 Photo RAW 2025.app",
                "ON1 Photo RAW MAX 2026.app",
            ),
        ),
        (
            "affinity_photo_2",
            "Affinity Photo 2",
            lambda: _mac_app("Affinity Photo 2.app", "Affinity Photo.app")
            or glob_first(["/Applications/Affinity Photo*.app"]),
        ),
        (
            "affinity_designer_2",
            "Affinity Designer 2",
            lambda: _mac_app("Affinity Designer 2.app", "Affinity Designer.app")
            or glob_first(["/Applications/Affinity Designer*.app"]),
        ),
        (
            "affinity_publisher_2",
            "Affinity Publisher 2",
            lambda: _mac_app("Affinity Publisher 2.app", "Affinity Publisher.app")
            or glob_first(["/Applications/Affinity Publisher*.app"]),
        ),
    ]
    return [
        BatchOpenAppDef(
            id=app_id,
            display_name=label,
            detect=detect,
            launch=_launch_mac_app,
        )
        for app_id, label, detect in specs
    ]


def _win_detectors() -> List[BatchOpenAppDef]:
    specs = [
        ("lightroom_classic", "Lightroom Classic", _detect_win_lightroom_classic),
        ("lightroom_cc", "Lightroom", _detect_win_lightroom_cc),
        ("photoshop", "Photoshop", _detect_win_photoshop),
        ("capture_one", "Capture One", _detect_win_capture_one),
        ("dxo_photolab", "DxO PhotoLab", _detect_win_dxo_photolab),
        ("on1_photo_raw", "ON1 Photo RAW", _detect_win_on1_photo_raw),
        ("affinity_photo_2", "Affinity Photo 2", _detect_win_affinity_photo_2),
        ("affinity_designer_2", "Affinity Designer 2", _detect_win_affinity_designer_2),
        ("affinity_publisher_2", "Affinity Publisher 2", _detect_win_affinity_publisher_2),
    ]
    return [
        BatchOpenAppDef(
            id=app_id,
            display_name=label,
            detect=detect,
            launch=_launch_exe,
        )
        for app_id, label, detect in specs
    ]


def _platform_defs() -> List[BatchOpenAppDef]:
    if sys.platform == "darwin":
        return _mac_detectors()
    if sys.platform == "win32":
        return _win_detectors()
    return []


@dataclass(frozen=True)
class InstalledBatchOpenApp:
    app_id: str
    display_name: str
    executable: str


def list_installed_batch_open_apps() -> List[InstalledBatchOpenApp]:
    """Return configured batch-open apps that appear to be installed on this machine."""
    out: List[InstalledBatchOpenApp] = []
    for spec in _platform_defs():
        detected = spec.detect()
        if not detected:
            continue
        if is_installed_marker(detected) and not (
            sys.platform == "win32" and spec.id.startswith("affinity_")
        ):
            continue
        out.append(
            InstalledBatchOpenApp(
                app_id=spec.id,
                display_name=spec.display_name,
                executable=icon_path_for_detected(detected) or detected,
            )
        )
    return out


def launch_batch_open_app(app_id: str, file_paths: Sequence[str]) -> bool:
    """Open *file_paths* in the app identified by *app_id*."""
    for spec in _platform_defs():
        if spec.id != app_id:
            continue
        if sys.platform == "win32" and app_id.startswith("affinity_"):
            return _launch_affinity_win(app_id, file_paths)
        exe = spec.detect()
        if not is_usable_exe(exe):
            return False
        return spec.launch(exe, file_paths)
    return False
