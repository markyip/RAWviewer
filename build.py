#!/usr/bin/env python3
"""
Build script for RAW Image Viewer Windows/macOS executable
Handles dependency installation and executable creation.
"""

VERSION = "2.3.2"

import os
import subprocess
import platform
import shutil
import time
import sys
import argparse
from pathlib import Path

# Repository root (directory containing this script)
REPO_ROOT = Path(__file__).resolve().parent

_WINDOWS_LITE_PIXI_SKIP = (
    "huggingface_hub",
    "onnxruntime-gpu",
    "onnxruntime-directml",
    "opencv-python-headless",
    "requests",
)


def write_build_profile(profile: str) -> None:
    profile = profile.strip().lower()
    if profile not in ("full", "lite"):
        raise ValueError(f"Unknown build profile: {profile}")
    path = REPO_ROOT / "src" / "build_profile.py"
    path.write_text(
        '"""Baked at build time by build.py (--profile full|lite). Dev default: full."""\n\n'
        f'PROFILE = "{profile}"\n',
        encoding="utf-8",
    )
    print(f"[INFO] Build profile: {profile} -> {path}")


def _project_venv_python() -> Path:
    if platform.system() == "Windows":
        return REPO_ROOT / "rawviewer_env" / "Scripts" / "python.exe"
    return REPO_ROOT / "rawviewer_env" / "bin" / "python3"


def _running_inside_project_venv() -> bool:
    try:
        return Path(sys.executable).resolve() == _project_venv_python().resolve()
    except OSError:
        return False


def _is_externally_managed_python() -> bool:
    """True for Homebrew / Debian PEP 668 installs where ``pip install`` to system is blocked."""
    return (Path(sys.prefix) / "EXTERNALLY-MANAGED").is_file()


def _should_use_project_venv_for_build() -> bool:
    """
    Prefer ./rawviewer_env so ``pip install`` / PyInstaller do not hit system Python limits.

    - macOS: always (matches ``scripts/Launch/shell/build_macos.sh``; Homebrew 3.14 may block pip without an
      ``EXTERNALLY-MANAGED`` file under ``sys.prefix``).
    - Linux: when PEP 668 marker is present.
    Set ``RAWVIEWER_USE_SYSTEM_PYTHON_BUILD=1`` to skip and use the current interpreter.
    """
    if os.environ.get("RAWVIEWER_USE_SYSTEM_PYTHON_BUILD", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return False
    if _running_inside_project_venv():
        return False
    if platform.system() == "Darwin":
        return True
    if _is_externally_managed_python():
        return True
    return False


def ensure_project_venv_and_reexec() -> None:
    """
    Create ./rawviewer_env if needed and re-exec this script with that interpreter.

    Skips when already using ./rawviewer_env (e.g. ``scripts/Launch/shell/build_macos.sh``) or when
    ``RAWVIEWER_USE_SYSTEM_PYTHON_BUILD=1``.
    """
    if not _should_use_project_venv_for_build():
        return
    vpy = _project_venv_python()
    venv_dir = REPO_ROOT / "rawviewer_env"
    if not vpy.is_file():
        if platform.system() == "Darwin":
            venv_msg = (
                "[INFO] Creating ./rawviewer_env — macOS builds default to an isolated venv "
                "(reliable pip/PyInstaller vs Homebrew Python). "
                "Set RAWVIEWER_USE_SYSTEM_PYTHON_BUILD=1 to opt out."
            )
        else:
            venv_msg = (
                "[INFO] Creating ./rawviewer_env — system Python is PEP 668 externally managed; "
                "pip cannot install into it."
            )
        print(venv_msg)
        rc = subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            check=False,
        ).returncode
        if rc != 0 or not vpy.is_file():
            print(
                "[ERROR] Could not create ./rawviewer_env. From the repo root try:\n"
                "  ./scripts/Launch/shell/build_macos.sh\n"
                "or:  python3 -m venv rawviewer_env && ./rawviewer_env/bin/python3 -m pip install -U pip && "
                "./rawviewer_env/bin/python3 build.py"
            )
            sys.exit(1)
    script = Path(__file__).resolve()
    argv = [str(vpy), str(script), *sys.argv[1:]]
    print(f"[INFO] Re-running build with project venv: {vpy}")
    os.execv(str(vpy), argv)


def run_command(cmd):
    # Support both string commands and lists
    if isinstance(cmd, list):
        result = subprocess.run(cmd)
    else:
        result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def _macos_app_bundle_name(profile: str = "full") -> str:
    if profile.strip().lower() == "lite":
        return "RAWviewer_Lite"
    return "RAWviewer"


def update_macos_plist(app_path, profile: str = "full"):
    """Update Info.plist in macOS app bundle to add file associations"""
    plist_path = os.path.join(app_path, 'Contents', 'Info.plist')
    if not os.path.exists(plist_path):
        print(f"[WARNING] Info.plist not found at {plist_path}")
        return False
        
    try:
        import plistlib
        with open(plist_path, 'rb') as f:
            plist = plistlib.load(f)
            
        # Define supported extensions
        image_extensions = [
            'jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp', 'tif', 'tiff', 'heic',
            'cr2', 'cr3', 'nef', 'arw', 'dng', 'raf', 'orf', 'rw2', 'pef', 'srw', 'crw', 'mef', 'mrw'
        ]
        
        # Add CFBundleDocumentTypes if not present
        if 'CFBundleDocumentTypes' not in plist:
            plist['CFBundleDocumentTypes'] = []
            
        # Check if our document type is already defined
        doc_type_exists = any(
            doc.get('CFBundleTypeName') == 'Image File' for doc in plist['CFBundleDocumentTypes']
        )
        
        if not doc_type_exists:
            doc_type = {
                'CFBundleTypeName': 'Image File',
                'CFBundleTypeRole': 'Viewer',
                'LSHandlerRank': 'Alternate',
                'LSItemContentTypes': [
                    'public.image',
                    'public.camera-raw-image'
                ],
                'CFBundleTypeExtensions': image_extensions
            }
            plist['CFBundleDocumentTypes'].append(doc_type)
            
        bundle_name = _macos_app_bundle_name(profile)
        is_lite = profile.strip().lower() == "lite"
        # Set a unique Bundle Identifier
        plist['CFBundleIdentifier'] = (
            'com.markyip.rawviewer.lite' if is_lite else 'com.markyip.rawviewer'
        )
        plist['CFBundleName'] = bundle_name
        plist['CFBundleDisplayName'] = 'RAWviewer Lite' if is_lite else 'RAW Image Viewer'
        plist['CFBundleExecutable'] = bundle_name
        plist['CFBundlePackageType'] = 'APPL'
        plist['CFBundleShortVersionString'] = VERSION
        plist['CFBundleVersion'] = VERSION
        
        # Add macOS permission usage descriptions
        plist['NSDesktopFolderUsageDescription'] = 'RAWviewer needs access to your Desktop to display images.'
        plist['NSDocumentsFolderUsageDescription'] = 'RAWviewer needs access to your Documents folder to display images.'
        plist['NSDownloadsFolderUsageDescription'] = 'RAWviewer needs access to your Downloads folder to display images.'
        plist['NSRemovableVolumesUsageDescription'] = 'RAWviewer needs access to external volumes to display images from cameras or cards.'
        plist['NSPhotoLibraryUsageDescription'] = 'RAWviewer needs access to your photo library to display images.'
        plist['NSAppleEventsUsageDescription'] = 'RAWviewer needs to receive file open events from the system.'
        
        # macOS specific flags
        plist['LSMinimumSystemVersion'] = '13.0'
        plist['NSHighResolutionCapable'] = True
        plist['LSSupportsOpeningDocumentsInPlace'] = True
        plist['LSApplicationCategoryType'] = 'public.app-category.photography'

        with open(plist_path, 'wb') as f:
            plistlib.dump(plist, f)
        print("[SUCCESS] Updated Info.plist with Bundle ID, file associations and usage descriptions")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to update Info.plist: {e}")
        return False


def install_dependencies(windows_accel: str = "cuda", *, profile: str = "full"):
    """Install required dependencies"""
    lite = profile.strip().lower() == "lite"
    print("Installing/upgrading dependencies...")
    system_name = platform.system()
    dependencies = [
        'PyQt6',
        'rawpy',
        'send2trash',
        'pyinstaller',
        'natsort',
        'exifread',
        'Pillow',  # Added for NEF thumbnail fallback
        'psutil',  # Added for system memory info in image_cache
        'numpy',   # Required for image processing (used in all modules)
        'qtawesome', # Required for icons in main.py
        'pyqtgraph',  # Optional/Future dependency included in requirements.txt
        'reverse-geocoder',  # Offline city/country lookup from GPS EXIF
        'pycountry',  # ISO country code -> full country name
    ]

    if system_name in ("Darwin", "Windows"):
        dependencies.append("certifi")
        if not lite:
            dependencies.append("requests")
    if system_name == "Windows":
        if not lite:
            # Windows semantic backend uses ONNX with selectable acceleration backend.
            accel = (windows_accel or "cuda").strip().lower()
            if accel == "cuda":
                dependencies.append('onnxruntime-gpu')
            elif accel == "directml":
                dependencies.append('onnxruntime-directml')
            else:
                print(f"[ERROR] Unsupported Windows acceleration backend: {windows_accel}")
                return False
            dependencies.append('opencv-python-headless')
            dependencies.append('huggingface-hub')
    elif system_name == "Darwin":
        dependencies.append('scipy')  # reverse_geocoder GPS lookup (scipy.spatial.cKDTree)
        if not lite:
            dependencies.append('huggingface-hub')
            dependencies.append('pyobjc-framework-CoreML')
            dependencies.append('pyobjc-framework-Vision')
        dependencies.append('pyobjc-framework-Quartz')
    if system_name in ("Darwin", "Windows"):
        dependencies.append("pyexiv2")

    for dep in dependencies:
        print(f"Installing {dep}...")
        if not run_command([sys.executable, "-m", "pip", "install", "--upgrade", dep]):
            print(f"[ERROR] Failed to install {dep}")
            return False

    if system_name == "Darwin":
        print("Installing pyobjc-framework-Cocoa (macOS share sheet)...")
        if not run_command(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pyobjc-framework-Cocoa"]
        ):
            print("[WARNING] pyobjc-framework-Cocoa install failed; Share may not work in the built app.")
    elif system_name == "Windows":
        print("Installing pywin32 (Windows Share verb)...")
        if not run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pywin32"]):
            print("[WARNING] pywin32 install failed; Share may not work in the built app.")

    print("Dependencies installed successfully!")
    return True


def _darwin_ensure_homebrew_pyexiv2_libs() -> None:
    """
    pyexiv2's bundled libexiv2.dylib expects Homebrew libinih / gettext on the build machine
    (see https://github.com/LeoHsiao1/pyexiv2/blob/master/docs/Tutorial.md FAQ).
    """
    if platform.system() != "Darwin":
        return
    brew = shutil.which("brew")
    if not brew:
        print(
            "[INFO] Homebrew (`brew`) not on PATH. If `import pyexiv2` fails, install "
            "https://brew.sh then run: brew install inih gettext"
        )
        return
    for formula in ("inih", "gettext"):
        listed = subprocess.run(
            [brew, "list", formula],
            capture_output=True,
        )
        if listed.returncode != 0:
            print(
                f"[INFO] Installing Homebrew `{formula}` (native dependency for pyexiv2 / Exiv2)..."
            )
            subprocess.run([brew, "install", formula], check=False)


def _darwin_preflight_pyexiv2_import() -> None:
    """Fail fast with a clear message before PyInstaller touches pyexiv2."""
    if platform.system() != "Darwin":
        return
    try:
        import pyexiv2  # noqa: F401
    except Exception as e:
        print(
            "[ERROR] pyexiv2 failed to import (required for this macOS build).\n"
            "  Install native libraries, then re-run:\n"
            "    brew install inih gettext\n"
            f"  Underlying error: {e}"
        )
        sys.exit(1)


def _prepare_windows_pixi_manifest(accel: str, *, profile: str = "full") -> Path:
    """
    Create a backend-specific pixi.toml copy for Windows installer payload.
    The bundled bootstrap always copies this file as `pixi.toml`.
    """
    src = REPO_ROOT / "pixi.toml"
    raw = src.read_text(encoding="utf-8")
    lite = profile.strip().lower() == "lite"
    if lite:
        kept_lines = []
        for line in raw.splitlines():
            if any(token in line for token in _WINDOWS_LITE_PIXI_SKIP):
                continue
            kept_lines.append(line)
        raw = "\n".join(kept_lines) + "\n"
        raw = raw.replace(
            "Professional RAW Image Viewer with Semantic Search",
            "Professional RAW Image Viewer (Lite — viewing, EXIF/GPS search)",
        )
    elif accel == "cuda":
        raw = raw.replace("onnxruntime-directml", "onnxruntime-gpu")
    elif accel == "directml":
        raw = raw.replace("onnxruntime-gpu", "onnxruntime-directml")
    else:
        raise ValueError(f"Unsupported Windows acceleration backend: {accel}")
    tmp_dir = REPO_ROOT / "build" / "_pixi_variants"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    suffix = "lite" if lite else accel
    out = tmp_dir / f"pixi-{suffix}.toml"
    out.write_text(raw, encoding="utf-8")
    return out


def _windows_setup_exe_name(accel: str, *, profile: str = "full") -> str:
    if profile.strip().lower() == "lite":
        return "RAWviewer_Setup_Lite"
    suffix = "CUDA" if accel == "cuda" else "DirectML"
    return f"RAWviewer_Setup_{suffix}"


def build_windows_launcher_stub(icon_path: str | None) -> Path:
    """Build a tiny RAWviewer.exe that launches pixi in the install directory."""
    stub_root = REPO_ROOT / "build" / "launcher_stub"
    dist_dir = stub_root / "dist"
    work_dir = stub_root / "build"
    spec_dir = stub_root
    stub_root.mkdir(parents=True, exist_ok=True)

    for path in (dist_dir, work_dir):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)

    launcher_src = REPO_ROOT / "src" / "win_launcher.py"
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--windowed",
        "--noconfirm",
        "--name",
        "RAWviewer",
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(work_dir),
        "--specpath",
        str(spec_dir),
        "--exclude-module",
        "PyQt6",
        "--exclude-module",
        "numpy",
        "--exclude-module",
        "PIL",
        "--exclude-module",
        "rawpy",
        "--exclude-module",
        "onnxruntime",
        str(launcher_src),
    ]
    if icon_path and os.path.isfile(icon_path):
        cmd.extend(["--icon", icon_path])

    print("[INFO] Building Windows launcher stub (RAWviewer.exe)...")
    if not run_command(cmd):
        print("[ERROR] Launcher stub build failed.")
        sys.exit(1)

    out = dist_dir / "RAWviewer.exe"
    if not out.is_file():
        print(f"[ERROR] Launcher stub was not created: {out}")
        sys.exit(1)
    print(f"[INFO] Launcher stub ready: {out} ({out.stat().st_size:,} bytes)")
    return out


def main():
    parser = argparse.ArgumentParser(description="Build RAWviewer executable")
    parser.add_argument(
        "--windows-accel",
        choices=["cuda", "directml"],
        default=os.environ.get("RAWVIEWER_WINDOWS_ACCEL", "cuda").strip().lower() or "cuda",
        help="Windows ONNX acceleration backend (default: cuda). Ignored for --profile lite.",
    )
    parser.add_argument(
        "--profile",
        choices=["full", "lite"],
        default=os.environ.get("RAWVIEWER_BUILD_PROFILE", "full").strip().lower() or "full",
        help="Build profile: full (AI search + face) or lite (viewing + EXIF/GPS search only).",
    )
    parser.add_argument(
        "--keep-dist",
        action="store_true",
        help="Windows: keep other dist/*.exe outputs (build both CUDA and DirectML in one session).",
    )
    args = parser.parse_args()
    if args.profile not in ("full", "lite"):
        args.profile = "full"

    ensure_project_venv_and_reexec()

    write_build_profile(args.profile)
    is_lite = args.profile == "lite"

    system_name = platform.system()
    if is_lite and system_name == "Windows":
        print("[INFO] Windows lite: no semantic/face AI / map overlay; EXIF+GPS gallery search only.")
    elif is_lite and system_name == "Darwin":
        print("[INFO] macOS lite: semantic/face AI / map overlay off; EXIF+GPS metadata search only.")
    if system_name == 'Windows':
        print("RAWviewer Windows Build Script")
        print(f"[INFO] Windows acceleration backend: {args.windows_accel}")
    elif system_name == 'Darwin':
        print(f"RAWviewer macOS Build Script v{VERSION}")
    else:
        print(f"RAWviewer Build Script v{VERSION} ({system_name})")
    print("==============================")
    print("")

    # Install dependencies first
    if not install_dependencies(windows_accel=args.windows_accel, profile=args.profile):
        print("[ERROR] Dependency installation failed.")
        sys.exit(1)

    if platform.system() == "Darwin":
        _darwin_ensure_homebrew_pyexiv2_libs()
        _darwin_preflight_pyexiv2_import()

    print("")
    print("Building RAWviewer executable...")

    # Import PyQt6 after installation
    try:
        import PyQt6
    except ImportError:
        print("[ERROR] PyQt6 not available after installation")
        sys.exit(1)

    # Clean previous builds
    print("Cleaning previous builds...")
    windows_target_setup = (
        "RAWviewer_Setup"
        if platform.system() == "Windows"
        else None
    )
    
    # Try to kill any running RAWviewer.exe processes on Windows
    if platform.system() == 'Windows':
        for image in (
            "RAWviewer.exe",
            "RAWviewer_Setup.exe",
            "RAWviewer_Setup_CUDA.exe",
            "RAWviewer_Setup_DirectML.exe",
            "RAWviewer_Setup_Lite.exe",
        ):
            try:
                result = subprocess.run(
                    ['taskkill', '/F', '/IM', image, '/T'],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    print(f"Closed running {image} instances")
            except Exception as e:
                print(f"[WARNING] Could not close running {image} instances: {e}")
        time.sleep(1)
    
    # Clean build directory
    if os.path.exists('build'):
        try:
            print("Cleaning build directory...")
            shutil.rmtree('build')
        except PermissionError as e:
            print(f"[WARNING] Could not delete build directory: {e}")
            print("  Continuing anyway...")
        except Exception as e:
            print(f"[WARNING] Error cleaning build directory: {e}")
    
    # Clean dist directory (try to delete specific files first)
    if os.path.exists('dist'):
        try:
            if args.keep_dist and platform.system() == "Windows" and windows_target_setup:
                target_file = os.path.join("dist", f"{windows_target_setup}.exe")
                if os.path.exists(target_file):
                    print(f"Cleaning dist directory (keeping other installers)...")
                    try:
                        os.remove(target_file)
                        print(f"  Removed {windows_target_setup}.exe")
                    except PermissionError:
                        print(f"[ERROR] Cannot delete {windows_target_setup}.exe - it may be running.")
                        print("  Please close RAWviewer and try again.")
                        sys.exit(1)
                    except Exception as e:
                        print(f"[WARNING] Could not delete {windows_target_setup}.exe: {e}")
            else:
                print("Cleaning dist directory...")
                if platform.system() == "Windows":
                    for name in (
                        "RAWviewer.exe",
                        "RAWviewer_Setup.exe",
                        "RAWviewer_Setup_CUDA.exe",
                        "RAWviewer_Setup_DirectML.exe",
                        "RAWviewer_Setup_Lite.exe",
                    ):
                        candidate = os.path.join("dist", name)
                        if os.path.exists(candidate):
                            try:
                                os.remove(candidate)
                                print(f"  Removed {name}")
                            except PermissionError:
                                print(f"[ERROR] Cannot delete {name} - it may be running.")
                                print("  Please close RAWviewer and try again.")
                                sys.exit(1)
                            except Exception as e:
                                print(f"[WARNING] Could not delete {name}: {e}")
                elif platform.system() == "Darwin":
                    for bundle in ("RAWviewer.app", "RAWviewer_Lite.app"):
                        candidate = os.path.join("dist", bundle)
                        if os.path.exists(candidate):
                            try:
                                shutil.rmtree(candidate)
                                print(f"  Removed {bundle}")
                            except PermissionError:
                                print(f"[ERROR] Cannot delete {bundle} - it may be running.")
                                print("  Please close RAWviewer and try again.")
                                sys.exit(1)
                            except Exception as e:
                                print(f"[WARNING] Could not delete {bundle}: {e}")
                else:
                    exe_name = "RAWviewer"
                    exe_path = os.path.join("dist", exe_name)
                    if os.path.exists(exe_path):
                        try:
                            os.remove(exe_path)
                            print(f"  Removed {exe_name}")
                        except PermissionError:
                            print(f"[ERROR] Cannot delete {exe_name} - it may be running.")
                            print("  Please close RAWviewer and try again.")
                            sys.exit(1)
                        except Exception as e:
                            print(f"[WARNING] Could not delete {exe_name}: {e}")

                # Try to remove the entire dist directory
                try:
                    shutil.rmtree('dist')
                except PermissionError:
                    print("[WARNING] Some files in dist directory are locked, but continuing...")
                except Exception as e:
                    print(f"[WARNING] Could not fully clean dist directory: {e}")
        except Exception as e:
            print(f"[WARNING] Error cleaning dist directory: {e}")

    # Prevent stale local logs from being packed into installer payload ("src;src").
    logs_dir = Path("src") / "logs"
    if logs_dir.exists():
        try:
            print("Cleaning src/logs before packaging...")
            shutil.rmtree(logs_dir)
        except Exception as e:
            print(f"[WARNING] Could not clean src/logs: {e}")
    # Platform-agnostic icon
    if platform.system() == 'Windows':
        icon_file = os.path.join('icons', 'appicon.ico')
    elif platform.system() == 'Darwin':
        icon_file = os.path.join('icons', 'appicon.icns')
    else:
        icon_file = os.path.join('icons', 'appicon.ico')  # fallback
    icon_path = os.path.abspath(icon_file)
    if not os.path.exists(icon_path):
        print(f"[WARNING] Icon file not found: {icon_path}")
        icon_arg = ''
    else:
        icon_arg = f'--icon "{icon_path}"'
    # Find PyQt6 imageformats plugin path
    pyqt_path = os.path.dirname(PyQt6.__file__)
    if platform.system() == 'Windows':
        imageformats_src = os.path.join(
            pyqt_path, 'Qt6', 'plugins', 'imageformats')
        add_data_sep = ';'
    elif platform.system() == 'Darwin':
        imageformats_src = os.path.join(
            pyqt_path, 'Qt6', 'plugins', 'imageformats')
        add_data_sep = ':'
    else:
        imageformats_src = os.path.join(
            pyqt_path, 'Qt6', 'plugins', 'imageformats')
        add_data_sep = ':'
    # Add --add-data for imageformats and icons directory
    add_data_args = [
        f'--add-data "{imageformats_src}{add_data_sep}imageformats"',
        f'--add-data "icons{add_data_sep}icons"'
    ]
    app_bundle_name = _macos_app_bundle_name(args.profile)
    windows_pixi_manifest = None
    if platform.system() == "Darwin":
        print(f"[INFO] macOS app bundle: dist/{app_bundle_name}.app")
        if is_lite:
            print("[INFO] macOS lite release: no MobileCLIP / face AI / map overlay; EXIF+GPS gallery search only.")
        else:
            print("[INFO] macOS release: MobileCLIP Core ML models are NOT bundled; users download in-app on first use.")
    elif platform.system() == "Windows":
        add_data_args.append('--add-data "uninstall.bat;."')
        add_data_args.append('--add-data "scripts;scripts"')
        
        # Prepare all three variants of pixi.toml for the unified installer
        cuda_manifest = _prepare_windows_pixi_manifest("cuda", profile="full")
        dml_manifest = _prepare_windows_pixi_manifest("directml", profile="full")
        lite_manifest = _prepare_windows_pixi_manifest("directml", profile="lite")
        
        cuda_str = str(cuda_manifest).replace("\\", "/")
        dml_str = str(dml_manifest).replace("\\", "/")
        lite_str = str(lite_manifest).replace("\\", "/")
        
        add_data_args.append(f'--add-data "{cuda_str};."')
        add_data_args.append(f'--add-data "{dml_str};."')
        add_data_args.append(f'--add-data "{lite_str};."')
        
        app_bundle_name = "RAWviewer_Setup"
        launcher_stub = build_windows_launcher_stub(
            icon_path if os.path.exists(icon_path) else None
        )
        add_data_args.append(f'--add-data "{launcher_stub.resolve()};."')
        print(f"[INFO] Windows installer output: dist\\{app_bundle_name}.exe")
        print("[INFO] Bundling all Windows environment manifests (CUDA, DirectML, and Lite).")
    add_data_arg_str = " ".join(add_data_args)

    src_path = os.path.abspath('src')
    
    cmd_base = [
        sys.executable, "-m", "PyInstaller",
        "--windowed",
        "--paths", src_path,
        "--hidden-import", "rawviewer_ui.gallery_view",
        "--hidden-import", "rawviewer_ui.widgets",
        "--hidden-import", "rawviewer_ui.mobileclip_download_dialog",
        "--hidden-import", "natsort",
        "--hidden-import", "send2trash",
        "--hidden-import", "metadata_backend",
        "--hidden-import", "rawviewer_profile",
        "--hidden-import", "build_profile",
        "--name", app_bundle_name
    ]
    try:
        import pyexiv2  # noqa: F401

        if platform.system() in ("Darwin", "Windows"):
            cmd_base.extend(["--hidden-import", "pyexiv2", "--collect-all", "pyexiv2"])
            print("[INFO] PyInstaller: bundling pyexiv2 with --collect-all (native Exiv2 libs).")
    except ImportError:
        print(
            "[WARNING] pyexiv2 not importable; build continues without pyexiv2 bundling. "
            "Install pyexiv2 before packaging for EXIF read/write in the app."
        )

    if platform.system() == "Darwin":
        cmd_base.extend([
            "--hidden-import", "objc",
            "--hidden-import", "AppKit",
            "--hidden-import", "Foundation",
            "--hidden-import", "Quartz",
            "--hidden-import", "certifi",
            "--hidden-import", "ssl_certs",
            "--collect-data", "certifi",
            "--hidden-import", "reverse_geocoder",
            "--hidden-import", "scipy.spatial.cKDTree",
            "--exclude-module", "coremltools",
            "--exclude-module", "torch",
            "--exclude-module", "torchvision",
            "--exclude-module", "sentence_transformers",
            "--exclude-module", "transformers",
            "--exclude-module", "sklearn",
            "--exclude-module", "tokenizers",
            "--exclude-module", "safetensors",
        ])
        if is_lite:
            cmd_base.extend([
                "--exclude-module", "huggingface_hub",
                "--exclude-module", "httpx",
                "--exclude-module", "CoreML",
                "--exclude-module", "Vision",
                "--exclude-module", "location_map_engine",
                "--exclude-module", "rawviewer_ui.location_map_overlay",
                "--exclude-module", "gps_neighbors",
            ])
        else:
            cmd_base.extend([
                "--hidden-import", "requests",
                "--hidden-import", "CoreML",
                "--hidden-import", "Vision",
                "--hidden-import", "huggingface_hub",
            ])
    elif platform.system() == "Windows":
        cmd_base.extend([
            "--hidden-import", "win32com.client",
            "--hidden-import", "pythoncom",
            "--hidden-import", "pywintypes",
        ])
        # Windows uses Pixi bootstrap. Exclude everything except PyQt6 and standard libs
        cmd_base.extend([
            "--exclude-module", "torch",
            "--exclude-module", "torchvision",
            "--exclude-module", "tensorboard",
            "--exclude-module", "sentence_transformers",
            "--exclude-module", "transformers",
            "--exclude-module", "scipy",
            "--exclude-module", "matplotlib",
            "--exclude-module", "onnxruntime",
            "--exclude-module", "rawpy",
            "--exclude-module", "numpy",
            "--exclude-module", "PIL",
            "--exclude-module", "pyexiv2",
            "--exclude-module", "pyqtgraph",
            "--exclude-module", "natsort",
            "--exclude-module", "send2trash",
            "--exclude-module", "exifread",
        ])
        
        add_data_args.append('--add-data "src;src"')
    
    if platform.system() == 'Darwin':
        cmd_base.append("--onedir")
        macos_pool_hook = os.path.join(src_path, "pyi_rth_macos_process_pool.py")
        release_defaults_hook = os.path.join(src_path, "pyi_rth_release_defaults.py")
        ssl_certs_hook = os.path.join(src_path, "pyi_rth_ssl_certs.py")
        profile_hook = os.path.join(src_path, "pyi_rth_profile_defaults.py")
        if os.path.isfile(ssl_certs_hook):
            cmd_base.extend(["--runtime-hook", ssl_certs_hook])
        if os.path.isfile(profile_hook):
            cmd_base.extend(["--runtime-hook", profile_hook])
        if os.path.isfile(macos_pool_hook):
            cmd_base.extend(["--runtime-hook", macos_pool_hook])
        if os.path.isfile(release_defaults_hook):
            cmd_base.extend(["--runtime-hook", release_defaults_hook])
        bundle_id = "com.markyip.rawviewer.lite" if is_lite else "com.markyip.rawviewer"
        cmd_base.extend(["--osx-bundle-identifier", bundle_id])
    else:
        cmd_base.append("--onefile")
        release_defaults_hook = os.path.join(src_path, "pyi_rth_release_defaults.py")
        profile_hook = os.path.join(src_path, "pyi_rth_profile_defaults.py")
        if os.path.isfile(profile_hook):
            cmd_base.extend(["--runtime-hook", profile_hook])
        if os.path.isfile(release_defaults_hook):
            cmd_base.extend(["--runtime-hook", release_defaults_hook])
        
    if icon_arg:
        if platform.system() == 'Windows':
            cmd_base.extend(["--icon", icon_path])
        else:
            cmd_base.extend(["--icon", icon_path])
            
    # Add data
    for arg in add_data_args:
        cmd_base.extend(["--add-data", arg.split('--add-data ')[-1].strip('"')])
        
    if platform.system() == 'Windows':
        cmd_base.append("src/bootstrap.py")
    else:
        cmd_base.append("src/main.py")

    print(f"Running: {' '.join(cmd_base)}")
    if not run_command(cmd_base):
        print("[ERROR] Build failed.")
        sys.exit(1)
    if platform.system() == 'Windows':
        exe_path = Path("dist") / f"{app_bundle_name}.exe"
    elif platform.system() == 'Darwin':
        exe_path = Path("dist") / f"{app_bundle_name}.app"
    else:
        exe_path = Path("dist") / app_bundle_name
    if exe_path.exists():
        print(f"[SUCCESS] Executable created: {exe_path}")
        if platform.system() == 'Darwin':
            print("Patching macOS Info.plist...")
            update_macos_plist(str(exe_path), profile=args.profile)
            print("Re-signing macOS app bundle (ad-hoc)...")
            run_command(['codesign', '--force', '--deep', '-s', '-', str(exe_path)])
            print("Clearing macOS quarantine attribute...")
            run_command(['xattr', '-cr', str(exe_path)])
    else:
        print("[ERROR] Executable was not created!")
        
    if platform.system() == 'Windows' and exe_path.exists():
        print("Build completed successfully.")
        print(f"  Installer: {exe_path}")
        print("  (Installs a separate RAWviewer.exe launcher stub into the chosen folder.)")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"[CRITICAL ERROR] Build script crashed: {e}")
        sys.exit(1)
    
    # If main() returns None/0 but we want to check if it really succeeded
    # Actually main() doesn't return anything, so we should check for failures inside main()
