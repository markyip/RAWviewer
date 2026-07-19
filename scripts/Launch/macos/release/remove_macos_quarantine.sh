#!/bin/bash
# Clear download quarantine on RAWviewer.app in this folder (no copy to Applications).
# End users: bash remove_macos_quarantine.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
APP_NAME="RAWviewer.app"

# Unblock this script and siblings when the zip was downloaded from the web.
xattr -cr "${SCRIPT_DIR}" 2>/dev/null || true

find_app_bundle() {
    local app_src="${SCRIPT_DIR}/${APP_NAME}"
    if [[ -d "${app_src}" ]]; then
        APP_SRC="${app_src}"
        return 0
    fi
    local found
    found="$(find "${SCRIPT_DIR}" -maxdepth 2 -name "${APP_NAME}" -type d 2>/dev/null | head -1)"
    if [[ -n "${found}" ]]; then
        APP_SRC="${found}"
        return 0
    fi
    return 1
}

dialog_ok() {
    local message="$1"
    if command -v osascript >/dev/null 2>&1; then
        osascript <<EOF
display dialog "${message}" with title "RAWviewer" buttons {"OK"} default button "OK" with icon note
EOF
        return 0
    fi
    echo "${message}"
}

dialog_open() {
    local message="$1"
    if command -v osascript >/dev/null 2>&1; then
        osascript <<EOF
display dialog "${message}" with title "RAWviewer" buttons {"Not now", "Open"} default button "Open" with icon note
EOF
        return $?
    fi
    echo "${message}"
    read -r -p "Open RAWviewer now? [Y/n] " ans
    [[ ! "${ans}" =~ ^[nN] ]]
}

fail() {
    local msg="$1"
    echo "[ERROR] ${msg}"
    if command -v osascript >/dev/null 2>&1; then
        osascript -e "display alert \"RAWviewer\" message \"${msg}\" as critical" 2>/dev/null || true
    fi
    exit 1
}

if ! find_app_bundle; then
    fail "Could not find RAWviewer.app. Extract the full release zip first."
fi

VERSION=""
if [[ -f "${APP_SRC}/Contents/Info.plist" ]]; then
    VERSION="$(/usr/libexec/PlistBuddy -c 'Print CFBundleShortVersionString' "${APP_SRC}/Contents/Info.plist" 2>/dev/null || true)"
fi
VERSION_LABEL="${VERSION:+ (v${VERSION})}"

echo "Removing download quarantine from ${APP_SRC}..."
xattr -cr "${APP_SRC}" 2>/dev/null || true

dialog_ok "Download quarantine removed for RAWviewer${VERSION_LABEL}.

You can open RAWviewer.app here, or run: bash install_macos_app.sh"

if dialog_open "Open RAWviewer${VERSION_LABEL} from this folder now?"; then
    open "${APP_SRC}"
fi
