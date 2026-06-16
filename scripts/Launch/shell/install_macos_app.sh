#!/bin/bash
# Install RAWviewer.app to /Applications and clear macOS download quarantine.
# End users: open Terminal in the extracted zip folder and run: bash install_macos_app.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
APP_NAME="RAWviewer.app"
APP_SRC="${SCRIPT_DIR}/${APP_NAME}"
APP_DEST="/Applications/${APP_NAME}"

# Allow double-click via .command from a quarantined download folder.
xattr -cr "${SCRIPT_DIR}" 2>/dev/null || true

find_app_bundle() {
    if [[ -d "${APP_SRC}" ]]; then
        return 0
    fi
    local found
    found="$(find "${SCRIPT_DIR}" -maxdepth 2 -name "${APP_NAME}" -type d 2>/dev/null | head -1)"
    if [[ -n "${found}" ]]; then
        APP_SRC="${found}"
        return 0
    fi
    # Developer: script run from repo (dist/RAWviewer.app after build.py).
    local repo_dist
    repo_dist="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd)/dist/${APP_NAME}"
    if [[ -d "${repo_dist}" ]]; then
        APP_SRC="${repo_dist}"
        return 0
    fi
    return 1
}

confirm_install() {
    local version_label="$1"
    if command -v osascript >/dev/null 2>&1; then
        osascript <<EOF
set v to "${version_label}"
display dialog "RAWviewer" & v & " was downloaded outside the App Store, so macOS blocks it until the download quarantine is cleared.

This installer will:
• Clear that block on RAWviewer
• Copy it to your Applications folder
• Offer to open it for you

Click Install to continue (no Terminal needed)." with title "RAWviewer Installer" buttons {"Cancel", "Install"} default button "Install" with icon note
EOF
        return $?
    fi
    echo "Install RAWviewer${version_label} to Applications? [y/N]"
    read -r ans
    [[ "${ans}" =~ ^[yY] ]]
}

confirm_replace() {
    local version_label="$1"
    if command -v osascript >/dev/null 2>&1; then
        osascript <<EOF
set v to "${version_label}"
display dialog "RAWviewer" & v & " is already in Applications. Replace the existing copy?" with title "RAWviewer Installer" buttons {"Cancel", "Replace"} default button "Replace" with icon caution
EOF
        return $?
    fi
    echo "Replace existing RAWviewer in Applications? [y/N]"
    read -r ans
    [[ "${ans}" =~ ^[yY] ]]
}

confirm_launch() {
    local version_label="$1"
    if command -v osascript >/dev/null 2>&1; then
        osascript <<EOF
set v to "${version_label}"
display dialog "RAWviewer" & v & " is ready in Applications. Open it now?" with title "RAWviewer Installer" buttons {"Not now", "Open"} default button "Open" with icon note
EOF
        return $?
    fi
    echo "Open RAWviewer now? [Y/n]"
    read -r ans
    [[ ! "${ans}" =~ ^[nN] ]]
}

fail() {
    local msg="$1"
    echo "[ERROR] ${msg}"
    if command -v osascript >/dev/null 2>&1; then
        osascript -e "display alert \"RAWviewer Installer\" message \"${msg}\" as critical" 2>/dev/null || true
    fi
    exit 1
}

if ! find_app_bundle; then
    fail "Could not find RAWviewer.app next to this installer. Extract the full release zip first."
fi

VERSION=""
if [[ -f "${APP_SRC}/Contents/Info.plist" ]]; then
    VERSION="$(/usr/libexec/PlistBuddy -c 'Print CFBundleShortVersionString' "${APP_SRC}/Contents/Info.plist" 2>/dev/null || true)"
fi
VERSION_LABEL="${VERSION:+ v${VERSION}}"

if [[ -d "${APP_DEST}" ]]; then
    confirm_replace "${VERSION_LABEL}" || { echo "Installation cancelled."; exit 0; }
else
    confirm_install "${VERSION_LABEL}" || { echo "Installation cancelled."; exit 0; }
fi

echo "Clearing quarantine on ${APP_SRC}..."
xattr -cr "${APP_SRC}" 2>/dev/null || true

echo "Installing to ${APP_DEST}..."
if [[ -d "${APP_DEST}" ]]; then
    rm -rf "${APP_DEST}"
fi
ditto "${APP_SRC}" "${APP_DEST}"
xattr -cr "${APP_DEST}" 2>/dev/null || true

echo "RAWviewer installed successfully."

if confirm_launch "${VERSION_LABEL}"; then
    open "${APP_DEST}"
fi
