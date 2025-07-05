#!/usr/bin/env python3
"""
Minimal build script for RAW Image Viewer Windows executable
Only outputs the path to the created executable.
"""

import os
import subprocess
from pathlib import Path

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
    # Minimal PyInstaller command
    build_command = 'pyinstaller --onefile --windowed --icon "D:/Development/RAWviewer/appicon.ico" src/main.py --name RAWviewer'
    print(f"Running: {build_command}")
    if not run_command(build_command):
        print("[ERROR] Build failed.")
        return
    exe_path = Path('dist/RAWviewer.exe')
    if exe_path.exists():
        print(f"[SUCCESS] Executable created: {exe_path}")
    else:
        print("[ERROR] Executable was not created!")

if __name__ == '__main__':
    main()