#!/usr/bin/env python3
"""
Build script for RAW Image Viewer Windows/macOS executable
Handles dependency installation and executable creation.
"""

import os
import subprocess
import platform
import shutil
import time
from pathlib import Path


import sys

def run_command(cmd):
    # Use explicit shell execution
    print(f"Running command: {cmd}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def install_dependencies():
    """Install required dependencies"""
    print("Installing/upgrading dependencies...")
    dependencies = [
        'PyQt6',
        'send2trash',
        'pyinstaller',
        'natsort',
        'exifread',
        'Pillow',  # Added for NEF thumbnail fallback
        'psutil',  # Added for system memory info in image_cache
        'numpy'    # Required for image processing (used in all modules)
    ]

    python_exe = sys.executable
    print(f"Using Python: {python_exe}")
    
    for dep in dependencies:
        print(f"Installing {dep}...")
        # Use sys.executable to ensure we install to the running python environment
        if not run_command(f'"{python_exe}" -m pip install --upgrade {dep}'):
            print(f"[ERROR] Failed to install {dep}")
            return False

    print("Dependencies installed successfully!")
    return True


def main():
    system_name = platform.system()
    if system_name == 'Windows':
        print("RAWviewer Windows Build Script")
    elif system_name == 'Darwin':
        print("RAWviewer macOS Build Script")
    else:
        print(f"RAWviewer Build Script ({system_name})")
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

    # Minimal PyInstaller command using python module to ensure correct environment
    build_command = (
        f'"{sys.executable}" -m PyInstaller --onefile --windowed {icon_arg} '
        f'{add_data_arg_str} '
        f'--hidden-import=enhanced_raw_processor '
        f'--hidden-import=image_cache '
        f'--hidden-import=image_load_manager '
        f'--hidden-import=unified_image_processor '
        f'--hidden-import=common_image_loader '
        f'--hidden-import=ui.widgets '
        f'--hidden-import=ui.gallery_view '
        f'src/main.py --name RAWviewer --paths src'
    )
    print(f"Running: {build_command}")
    if not run_command(build_command):
        print("[ERROR] Build failed.")
        return
    if platform.system() == 'Windows':
        exe_path = Path('dist/RAWviewer.exe')
    else:
        exe_path = Path('dist/RAWviewer')
    if exe_path.exists():
        print(f"[SUCCESS] Executable created: {exe_path}")
    else:
        print("[ERROR] Executable was not created!")


if __name__ == '__main__':
    main()
