#!/usr/bin/env python3
"""
Automate building a MobileCLIP2 wheel for Windows distribution.

This script:
1. Clones Apple's ml-mobileclip repository.
2. Applies necessary patches for standalone usage / Windows compatibility.
3. Builds a .whl file that can be bundled into RAWviewer.
"""

import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path

REPO_URL = "https://github.com/apple/ml-mobileclip.git"
REPO_DIR = Path("ml-mobileclip-src")
WHEELS_DIR = Path("wheels")

def run_command(cmd, cwd=None):
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, cwd=cwd, shell=True if isinstance(cmd, str) else False)
    return result.returncode == 0

def main():
    if platform.system() != "Windows":
        # We can build the wheel on Mac/Linux but it might produce a cross-platform wheel 
        # or a platform-specific one depending on setup.py. 
        # MobileCLIP is mostly pure Python + torch, so it should be 'any' platform.
        print("[INFO] Running on non-Windows platform. Resulting wheel should be platform-independent if no C extensions.")

    # 1. Clone or Update
    if REPO_DIR.exists():
        print(f"[INFO] Updating {REPO_DIR}...")
        run_command(["git", "pull"], cwd=REPO_DIR)
    else:
        print(f"[INFO] Cloning {REPO_URL}...")
        if not run_command(["git", "clone", REPO_URL, str(REPO_DIR)]):
            print("[ERROR] Failed to clone repository.")
            return 1

    # 2. Patching (Re-exports for MobileCLIP2)
    # The Apple repo structure sometimes hides the MobileCLIP2 models unless explicitly re-exported
    # as noted in RAWviewer's export scripts.
    init_path = REPO_DIR / "mobileclip" / "__init__.py"
    if init_path.exists():
        with open(init_path, "r") as f:
            content = f.read()
        
        # Check if we need to add the MobileCLIP2 re-exports
        if "fastvit_mci3" not in content:
            print("[INFO] Patching mobileclip/__init__.py for MobileCLIP2 support...")
            # Very basic injection - the Apple repo may have changed, but we follow the known requirement
            addition = "\n# RAWviewer patch for MobileCLIP2 re-exports\nfrom .modules.common.mobileone import reparameterize_model\n"
            with open(init_path, "a") as f:
                f.write(addition)

    # 3. Build Wheel
    print("[INFO] Building wheel...")
    WHEELS_DIR.mkdir(exist_ok=True)
    
    # Ensure build tools are present
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "wheel", "setuptools"])

    if not run_command([sys.executable, "setup.py", "bdist_wheel", "--dist-dir", str(WHEELS_DIR.resolve())], cwd=REPO_DIR):
        print("[ERROR] Failed to build wheel.")
        return 1

    # 4. Cleanup source (optional)
    # shutil.rmtree(REPO_DIR)
    
    print(f"\n[SUCCESS] MobileCLIP wheel built in {WHEELS_DIR}/")
    for whl in WHEELS_DIR.glob("*.whl"):
        print(f"  - {whl.name}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
