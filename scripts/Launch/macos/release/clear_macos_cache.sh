#!/bin/bash
# End-user cache wipe for installed / release-zip RAWviewer (macOS).
# Removes: ~/.rawviewer_cache, logs, QSettings preference plists.
# Does NOT remove: Applications/RAWviewer.app, photos, or XMP sidecars.
#
# Usage (from this folder):
#   bash clear_macos_cache.sh
# Or double-click Clear Cache.command

set -u

CLEARED=0
FAILED=0

pause_if_interactive() {
    if [ -t 0 ]; then
        read -r -p "Press Enter to close this session..."
    fi
}

echo
echo "========================================"
echo " RAWviewer — clear cache and session"
echo "========================================"
echo
echo "This removes local photo cache and app preferences so upgrades"
echo "can use the newer, faster search/index defaults."
echo "Your photos and XMP sidecars are NOT deleted."
echo

kill_app() {
    echo "Closing RAWviewer if running..."
    killall RAWviewer 2>/dev/null || true
    killall "RAWviewer" 2>/dev/null || true
    sleep 2
}

remove_tree() {
    local target="$1"
    local label="$2"
    if [ ! -e "$target" ]; then
        return 0
    fi
    echo "Removing ${label}:"
    echo "  ${target}"
    local tries=0
    while [ "$tries" -lt 5 ]; do
        tries=$((tries + 1))
        rm -rf "$target" 2>/dev/null || true
        if [ ! -e "$target" ]; then
            CLEARED=1
            return 0
        fi
        sleep 1
    done
    echo "  WARNING: Could not fully remove — file may be locked."
    FAILED=1
}

clear_qsettings() {
    echo "Clearing app preferences (window, last folder, bookmarks flags)..."
    local removed=0
    local plists=(
        "${HOME}/Library/Preferences/com.RAWviewer.RAWviewer.plist"
        "${HOME}/Library/Preferences/RAWviewer.plist"
        "${HOME}/Library/Preferences/com.rawviewer.RAWviewer.plist"
    )
    for plist in "${plists[@]}"; do
        if [ -f "$plist" ]; then
            rm -f "$plist" && removed=1 && echo "  Removed ${plist}"
        fi
    done
    shopt -s nullglob
    for plist in "${HOME}/Library/Preferences/"*[Rr][Aa][Ww]viewer*.plist; do
        if [ -f "$plist" ]; then
            rm -f "$plist" && removed=1 && echo "  Removed ${plist}"
        fi
    done
    shopt -u nullglob
    for domain in com.RAWviewer.RAWviewer RAWviewer com.rawviewer.RAWviewer; do
        if defaults read "$domain" &>/dev/null; then
            if defaults delete "$domain" 2>/dev/null; then
                removed=1
                echo "  Removed defaults domain ${domain}"
            else
                echo "  WARNING: Could not delete defaults domain ${domain}"
                FAILED=1
            fi
        fi
    done
    if [ "$removed" -eq 0 ]; then
        echo "  Preferences already clean"
    else
        CLEARED=1
    fi
}

kill_app
remove_tree "${HOME}/.rawviewer_cache" "image / EXIF / semantic / thumbnail cache"
remove_tree "${HOME}/Library/Application Support/RAWviewer/logs" "runtime logs"
remove_tree "${HOME}/Library/Logs/RAWviewer" "system logs"
remove_tree "${HOME}/Library/Caches/RAWviewer" "map tiles / app caches"
clear_qsettings

echo
if [ "$FAILED" -ne 0 ]; then
    echo "Finished with warnings — close RAWviewer fully and run again if needed."
    pause_if_interactive
    exit 1
fi
if [ "$CLEARED" -eq 0 ]; then
    echo "Nothing to clear — cache was already clean."
else
    echo "Done. Reopen RAWviewer and your folder; the first open may rebuild cache."
fi
pause_if_interactive
exit 0
