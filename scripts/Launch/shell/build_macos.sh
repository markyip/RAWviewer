#!/bin/bash
# Build RAWviewer on macOS (full or lite profile).
# Repo root: scripts/Launch/shell -> ../../..
# Usage: build_macos.sh [full|lite]

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

PROFILE="${1:-${RAWVIEWER_BUILD_PROFILE:-full}}"
PROFILE="$(echo "$PROFILE" | tr '[:upper:]' '[:lower:]')"
if [[ "$PROFILE" != "full" && "$PROFILE" != "lite" ]]; then
    echo "[ERROR] Unknown profile: $PROFILE (use full or lite)"
    exit 1
fi
export RAWVIEWER_BUILD_PROFILE="$PROFILE"

if [[ "$PROFILE" == "lite" ]]; then
    APP_BUNDLE="RAWviewer_Lite.app"
    RELEASE_SUFFIX="macOS-Lite"
else
    APP_BUNDLE="RAWviewer.app"
    RELEASE_SUFFIX="macOS"
fi

VERSION="$(grep -E '^VERSION = ' "$REPO_ROOT/build.py" | sed -E 's/.*"([^"]+)".*/\1/')"
VERSION="${VERSION:-unknown}"
echo "RAWviewer macOS Build Script (v${VERSION}, profile: ${PROFILE})"
echo "======================================"
echo ""

if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "[ERROR] This script is designed for macOS only."
    echo "Current OS: $OSTYPE"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 is not installed or not in PATH"
    echo "Please install Python 3.10 or higher from https://www.python.org/"
    exit 1
fi

if [ ! -f "icons/appicon.icns" ]; then
    echo "[WARNING] Icon file not found: icons/appicon.icns"
    echo "The app will be built without a custom icon."
fi

# Prefer the Pixi environment: it is the interpreter + exact package set the
# app is developed and tested with, so the frozen .app matches dev behavior.
# The venv fallback pip-installs unpinned latest versions on a different
# Python — only used when Pixi isn't available.
PIXI_PYTHON="$REPO_ROOT/.pixi/envs/default/bin/python"
if [ -x "$PIXI_PYTHON" ] && "$PIXI_PYTHON" -c "import PyInstaller" >/dev/null 2>&1; then
    PYTHON_BIN="$PIXI_PYTHON"
    # Without this, build.py re-execs itself into ./rawviewer_env and the
    # Pixi interpreter choice is silently discarded.
    export RAWVIEWER_USE_SYSTEM_PYTHON_BUILD=1
    echo "Using Pixi environment interpreter: $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"
else
    VENV_DIR="$REPO_ROOT/rawviewer_env"
    PYTHON_BIN="$VENV_DIR/bin/python3"

    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
        echo "Virtual environment created."
    fi

    if [ ! -x "$PYTHON_BIN" ]; then
        echo "[ERROR] Missing venv interpreter: $PYTHON_BIN"
        echo "Remove the broken folder and re-run: rm -rf rawviewer_env"
        exit 1
    fi

    echo "Using virtual environment: $VENV_DIR"

    if command -v brew >/dev/null 2>&1; then
        echo "Checking Homebrew dependencies for pyexiv2 (inih, gettext)..."
        brew list inih &>/dev/null || brew install inih
        brew list gettext &>/dev/null || brew install gettext
    else
        echo "[INFO] brew not on PATH. If the build fails on pyexiv2: install Homebrew, then: brew install inih gettext"
    fi

    echo "Upgrading pip..."
    "$PYTHON_BIN" -m pip install --upgrade pip

    echo "Installing core dependencies..."
    CORE_DEPS="PyQt6 rawpy send2trash pyinstaller natsort exifread Pillow psutil numpy qtawesome pycountry certifi pyobjc-framework-Cocoa pyobjc-framework-Quartz opencv-python-headless pillow-heif tifffile lensfunpy"
    if [[ "$PROFILE" == "full" ]]; then
        CORE_DEPS="$CORE_DEPS requests huggingface-hub pyobjc-framework-CoreML pyobjc-framework-Vision"
    fi
    "$PYTHON_BIN" -m pip install --upgrade $CORE_DEPS

    echo "Installing required dependency: pyexiv2..."
    if ! "$PYTHON_BIN" -m pip install --upgrade pyexiv2; then
        echo "[ERROR] pyexiv2 install failed (required for macOS release builds)."
        echo "  Install native libraries, then re-run:"
        echo "    brew install inih gettext"
        exit 1
    fi
    echo "[INFO] pyexiv2 installed (Exiv2 / focus-point path enabled)."

    "$PYTHON_BIN" -m pip uninstall -y sentence-transformers torch torchvision transformers scikit-learn tokenizers safetensors coremltools >/dev/null 2>&1 || true
fi

echo "Cleaning previous builds..."
chmod -R u+w build dist 2>/dev/null || true
rm -rf build || true
chmod -R u+w dist 2>/dev/null || true
rm -rf dist || true
rm -f *.spec

echo "Building RAWviewer (${PROFILE})..."
if "$PYTHON_BIN" build.py --profile "$PROFILE"; then
    echo ""
    echo "[SUCCESS] Build completed!"
    echo ""

    if [ -d "dist/${APP_BUNDLE}" ]; then
        echo "macOS App Bundle created: dist/${APP_BUNDLE} (v${VERSION}, ${PROFILE})"
        echo ""
        echo "Packaging release zip (app + installer)..."
        RELEASE_NAME="RAWviewer-v${VERSION}-${RELEASE_SUFFIX}"
        RELEASE_DIR="dist/${RELEASE_NAME}"
        rm -rf "${RELEASE_DIR}"
        mkdir -p "${RELEASE_DIR}"
        ditto "dist/${APP_BUNDLE}" "${RELEASE_DIR}/${APP_BUNDLE}"
        cp "${REPO_ROOT}/scripts/Launch/shell/install_macos_app.sh" "${RELEASE_DIR}/"
        cp "${REPO_ROOT}/scripts/Launch/shell/remove_macos_quarantine.sh" "${RELEASE_DIR}/"
        cp "${REPO_ROOT}/scripts/Launch/shell/uninstall_macos_app.sh" "${RELEASE_DIR}/"
        cp "${REPO_ROOT}/scripts/Launch/shell/Uninstall RAWviewer.command" "${RELEASE_DIR}/"
        cp "${REPO_ROOT}/scripts/Launch/shell/macos_release_readme.txt" "${RELEASE_DIR}/Start Here.txt"
        chmod +x "${RELEASE_DIR}/install_macos_app.sh" "${RELEASE_DIR}/remove_macos_quarantine.sh"
        chmod +x "${RELEASE_DIR}/uninstall_macos_app.sh" "${RELEASE_DIR}/Uninstall RAWviewer.command"
        rm -f "dist/${RELEASE_NAME}.zip"
        ditto -c -k --sequesterRsrc --keepParent "${RELEASE_DIR}" "dist/${RELEASE_NAME}.zip"
        echo "Release zip: dist/${RELEASE_NAME}.zip"
        echo ""
        echo "End-user install: cd to extracted folder, then: bash install_macos_app.sh"
        echo "Smoke test:"
        echo "  unzip -q dist/${RELEASE_NAME}.zip -d /tmp && open \"/tmp/${RELEASE_NAME}\""
        echo "  # Terminal: cd /tmp/${RELEASE_NAME} && bash install_macos_app.sh"
        echo ""
        echo "Local dev run (no download quarantine):"
        echo "  open dist/${APP_BUNDLE}"
    elif [ -f "dist/RAWviewer" ]; then
        echo "Executable created: dist/RAWviewer"
        echo ""
        echo "To run the app:"
        echo "  ./dist/RAWviewer"
    else
        echo "[WARNING] Build completed but output files not found in expected location"
        echo "Check the dist/ directory for output files"
    fi
else
    echo ""
    echo "[ERROR] Build failed. Check the error messages above."
    exit 1
fi
