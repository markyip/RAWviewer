#!/usr/bin/env python3
"""
Build script for RAW Image Viewer Windows/macOS executable
Handles dependency installation and executable creation.
"""

VERSION = "1.7.1"

import os
import subprocess
import platform
import shutil
import time
import sys
from pathlib import Path


def run_command(cmd):
    # Support both string commands and lists
    if isinstance(cmd, list):
        result = subprocess.run(cmd)
    else:
        result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def update_macos_plist(app_path):
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
            
        # Set a unique Bundle Identifier
        plist['CFBundleIdentifier'] = 'com.markyip.rawviewer'
        plist['CFBundleName'] = 'RAWviewer'
        plist['CFBundleDisplayName'] = 'RAW Image Viewer'
        plist['CFBundleExecutable'] = 'RAWviewer'
        plist['CFBundlePackageType'] = 'APPL'
        plist['CFBundleShortVersionString'] = VERSION
        
        # Add macOS permission usage descriptions
        plist['NSDesktopFolderUsageDescription'] = 'RAWviewer needs access to your Desktop to display images.'
        plist['NSDocumentsFolderUsageDescription'] = 'RAWviewer needs access to your Documents folder to display images.'
        plist['NSDownloadsFolderUsageDescription'] = 'RAWviewer needs access to your Downloads folder to display images.'
        plist['NSRemovableVolumesUsageDescription'] = 'RAWviewer needs access to external volumes to display images from cameras or cards.'
        plist['NSPhotoLibraryUsageDescription'] = 'RAWviewer needs access to your photo library to display images.'
        plist['NSAppleEventsUsageDescription'] = 'RAWviewer needs to receive file open events from the system.'
        
        # macOS specific flags
        plist['LSMinimumSystemVersion'] = '10.15.0'
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


def install_dependencies():
    """Install required dependencies"""
    print("Installing/upgrading dependencies...")
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
        'pyqtgraph'  # Optional/Future dependency included in requirements.txt
    ]

    for dep in dependencies:
        print(f"Installing {dep}...")
        if not run_command([sys.executable, "-m", "pip", "install", "--upgrade", dep]):
            print(f"[ERROR] Failed to install {dep}")
            return False

    if platform.system() == "Darwin":
        print("Installing pyobjc-framework-Cocoa (macOS share sheet)...")
        if not run_command(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pyobjc-framework-Cocoa"]
        ):
            print("[WARNING] pyobjc-framework-Cocoa install failed; Share may not work in the built app.")
    elif platform.system() == "Windows":
        print("Installing pywin32 (Windows Share verb)...")
        if not run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pywin32"]):
            print("[WARNING] pywin32 install failed; Share may not work in the built app.")

    print("Dependencies installed successfully!")
    return True


def main():
    system_name = platform.system()
    if system_name == 'Windows':
        print("RAWviewer Windows Build Script")
    elif system_name == 'Darwin':
        print(f"RAWviewer macOS Build Script v{VERSION}")
    else:
        print(f"RAWviewer Build Script v{VERSION} ({system_name})")
    print("==============================")
    print("")

    # Install dependencies first
    if not install_dependencies():
        print("[ERROR] Dependency installation failed.")
        return

    print("")
    print("Building RAWviewer executable...")

    # Import PyQt6 after installation
    try:
        import PyQt6
    except ImportError:
        print("[ERROR] PyQt6 not available after installation")
        return

    # Clean previous builds
    print("Cleaning previous builds...")
    
    # Try to kill any running RAWviewer.exe processes on Windows
    if platform.system() == 'Windows':
        try:
            result = subprocess.run(
                ['taskkill', '/F', '/IM', 'RAWviewer.exe', '/T'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("Closed running RAWviewer.exe instances")
                time.sleep(1)  # Wait a moment for file handles to release
        except Exception as e:
            print(f"[WARNING] Could not close running instances: {e}")
    
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
            print("Cleaning dist directory...")
            # Try to delete the exe file specifically first
            exe_name = 'RAWviewer.exe' if platform.system() == 'Windows' else 'RAWviewer'
            exe_path = os.path.join('dist', exe_name)
            if os.path.exists(exe_path):
                try:
                    os.remove(exe_path)
                    print(f"  Removed {exe_name}")
                except PermissionError:
                    print(f"[ERROR] Cannot delete {exe_name} - it may be running.")
                    print("  Please close RAWviewer and try again.")
                    return
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
    add_data_arg_str = " ".join(add_data_args)

    src_path = os.path.abspath('src')
    
    cmd_base = [
        sys.executable, "-m", "PyInstaller",
        "--windowed",
        "--paths", src_path,
        "--hidden-import", "rawviewer_ui.gallery_view",
        "--hidden-import", "rawviewer_ui.widgets",
        "--hidden-import", "natsort",
        "--hidden-import", "send2trash",
        "--name", "RAWviewer"
    ]
    if platform.system() == "Darwin":
        cmd_base.extend([
            "--hidden-import", "objc",
            "--hidden-import", "AppKit",
            "--hidden-import", "Foundation",
        ])
    elif platform.system() == "Windows":
        cmd_base.extend([
            "--hidden-import", "win32com.client",
            "--hidden-import", "pythoncom",
            "--hidden-import", "pywintypes",
        ])
    
    if platform.system() == 'Darwin':
        cmd_base.append("--onedir")
        cmd_base.extend(["--osx-bundle-identifier", "com.markyip.rawviewer"])
    else:
        cmd_base.append("--onefile")
        
    if icon_arg:
        if platform.system() == 'Windows':
            cmd_base.extend(["--icon", icon_path])
        else:
            cmd_base.extend(["--icon", icon_path])
            
    # Add data
    for arg in add_data_args:
        cmd_base.extend(["--add-data", arg.split('--add-data ')[-1].strip('"')])
        
    cmd_base.append("src/main.py")

    print(f"Running: {' '.join(cmd_base)}")
    if not run_command(cmd_base):
        print("[ERROR] Build failed.")
        return
    if platform.system() == 'Windows':
        exe_path = Path('dist/RAWviewer.exe')
    else:
        exe_path = Path('dist/RAWviewer.app')
    if exe_path.exists():
        print(f"[SUCCESS] Executable created: {exe_path}")
        if platform.system() == 'Darwin':
            print("Patching macOS Info.plist...")
            update_macos_plist(str(exe_path))
            print("Re-signing macOS app bundle (ad-hoc)...")
            run_command(['codesign', '--force', '--deep', '-s', '-', str(exe_path)])
            print("Clearing macOS quarantine attribute...")
            run_command(['xattr', '-cr', str(exe_path)])
    else:
        print("[ERROR] Executable was not created!")


if __name__ == '__main__':
    main()
