#!/usr/bin/env python3
"""
Build script for RAW Image Viewer Windows/macOS executable
Handles dependency installation and executable creation.
"""

import os
import subprocess
import platform
from pathlib import Path


def run_command(cmd):
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


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
        'numpy'    # Required for image processing (used in all modules)
    ]

    for dep in dependencies:
        print(f"Installing {dep}...")
        if not run_command(f'pip install --upgrade {dep}'):
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
    for directory in ['build', 'dist']:
        if os.path.exists(directory):
            print(f"Cleaning previous {directory} directory...")
            import shutil
            shutil.rmtree(directory)
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

    # Minimal PyInstaller command
    build_command = (
        f'pyinstaller --onefile --windowed {icon_arg} '
        f'{add_data_arg_str} src/main.py --name RAWviewer'
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
