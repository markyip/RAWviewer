#!/usr/bin/env python3
"""
Minimal build script for RAW Image Viewer Windows/macOS executable
Only outputs the path to the created executable.
"""

import os
import subprocess
import platform
from pathlib import Path
import sys
import PyQt6

def run_command(cmd):
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def main():
    print("Building RAWviewer executable...")
    # Clean previous builds
    for directory in ['build', 'dist']:
        if os.path.exists(directory):
            print(f"Cleaning previous {directory} directory...")
            import shutil
            shutil.rmtree(directory)
    # Platform-agnostic icon
    if platform.system() == 'Windows':
        icon_file = 'appicon.ico'
    elif platform.system() == 'Darwin':
        icon_file = 'appicon.icns'
    else:
        icon_file = 'appicon.ico'  # fallback
    icon_path = os.path.abspath(icon_file)
    if not os.path.exists(icon_path):
        print(f"[WARNING] Icon file not found: {icon_path}")
        icon_arg = ''
    else:
        icon_arg = f'--icon "{icon_path}"'
    # Find PyQt6 imageformats plugin path
    pyqt_path = os.path.dirname(PyQt6.__file__)
    if platform.system() == 'Windows':
        imageformats_src = os.path.join(pyqt_path, 'Qt6', 'plugins', 'imageformats')
        add_data_sep = ';'
    elif platform.system() == 'Darwin':
        imageformats_src = os.path.join(pyqt_path, 'Qt6', 'plugins', 'imageformats')
        add_data_sep = ':'
    else:
        imageformats_src = os.path.join(pyqt_path, 'Qt6', 'plugins', 'imageformats')
        add_data_sep = ':'
    # Add --add-data for imageformats
    add_data_arg = f'--add-data "{imageformats_src}{add_data_sep}imageformats"'
    # Minimal PyInstaller command
    build_command = f'pyinstaller --onefile --windowed {icon_arg} {add_data_arg} src/main.py --name RAWviewer'
    print(f"Running: {build_command}")
    if not run_command(build_command):
        print("[ERROR] Build failed.")
        return
    exe_path = Path('dist/RAWviewer.exe') if platform.system() == 'Windows' else Path('dist/RAWviewer')
    if exe_path.exists():
        print(f"[SUCCESS] Executable created: {exe_path}")
    else:
        print("[ERROR] Executable was not created!")

if __name__ == '__main__':
    main()