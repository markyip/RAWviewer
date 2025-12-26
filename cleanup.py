#!/usr/bin/env python3
"""
Cleanup script for RAWviewer project.
Removes build artifacts, cache files, and old logs.
"""

import os
import shutil
import glob
from pathlib import Path
from datetime import datetime, timedelta

def remove_directory(path):
    """Remove a directory if it exists."""
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            print(f"[OK] Removed: {path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to remove {path}: {e}")
            return False
    return False

def remove_file(path):
    """Remove a file if it exists."""
    if os.path.exists(path):
        try:
            os.remove(path)
            print(f"[OK] Removed: {path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to remove {path}: {e}")
            return False
    return False

def cleanup_old_logs(log_dir, days=7):
    """Remove log files older than specified days."""
    if not os.path.exists(log_dir):
        return 0
    
    cutoff_time = datetime.now() - timedelta(days=days)
    removed_count = 0
    
    for log_file in glob.glob(os.path.join(log_dir, "*.log")):
        try:
            file_time = datetime.fromtimestamp(os.path.getmtime(log_file))
            if file_time < cutoff_time:
                os.remove(log_file)
                removed_count += 1
                print(f"[OK] Removed old log: {os.path.basename(log_file)}")
        except Exception as e:
            print(f"[ERROR] Failed to remove {log_file}: {e}")
    
    return removed_count

def main():
    print("RAWviewer Cleanup Script")
    print("=" * 50)
    print()
    
    # Clean build directories
    print("Cleaning build artifacts...")
    remove_directory("build")
    remove_directory("dist")
    print()
    
    # Clean Python cache
    print("Cleaning Python cache...")
    remove_directory("src/__pycache__")
    # Also check for any other __pycache__ directories
    for pycache in Path(".").rglob("__pycache__"):
        if "rawviewer_env" not in str(pycache):  # Don't remove venv cache
            remove_directory(str(pycache))
    print()
    
    # Remove PyInstaller spec file (contains hardcoded paths)
    print("Removing generated spec file...")
    remove_file("RAWviewer.spec")
    print()
    
    # Clean old logs (keep last 7 days)
    print("Cleaning old log files (keeping last 7 days)...")
    removed_logs = cleanup_old_logs("src/logs", days=7)
    if removed_logs == 0:
        print("  No old log files to remove")
    print()
    
    print("=" * 50)
    print("Cleanup completed!")
    print()
    print("Note: Virtual environment (rawviewer_env/) was not touched.")
    print("Note: Source code and documentation files were preserved.")

if __name__ == "__main__":
    main()

