#!/bin/bash
# Full cache + session wipe for RAWviewer (dev and installed builds).
# Removes: ~/.rawviewer_cache, logs, dev logs, macOS QSettings (Preferences plists).
# Does NOT remove: /Applications/RAWviewer.app or repo models/ (bundled MobileCLIP assets).
#
# Repo root: scripts/Launch/shell -> ../../..

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

CLEARED=0
FAILED=0

pause_if_interactive() {
    if [ -t 0 ]; then
        read -r -p "Press Enter to close this session..."
    fi
}

echo
echo "========================================"
echo " RAWviewer - clear ALL cache and state"
echo "========================================"
echo

kill_app() {
    echo "Closing RAWviewer if running..."
    killall RAWviewer 2>/dev/null || true
    killall "RAWviewer" 2>/dev/null || true
    # Dev runs via python main.py
    pkill -f "[p]ython.*main\.py" 2>/dev/null || true
    pkill -f "[p]ython.*RAWviewer" 2>/dev/null || true
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
    echo "  WARNING: Could not fully remove - file may be locked."
    FAILED=1
}

clear_qsettings() {
    echo "Clearing QSettings / session (window, sort, last folder, rotations, semantic flags)..."
    local removed=0

    # Qt QSettings("RAWviewer", "RAWviewer") on macOS (NativeFormat)
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

    # Any other RAWviewer-related preference plists (Qt version / casing differences)
    shopt -s nullglob
    for plist in "${HOME}/Library/Preferences/"*[Rr][Aa][Ww]viewer*.plist; do
        if [ -f "$plist" ]; then
            rm -f "$plist" && removed=1 && echo "  Removed ${plist}"
        fi
    done
    shopt -u nullglob

    # Also clear via defaults domain when registered
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

verify_cold_state() {
    echo
    echo "Verifying cold-start state..."
    local issues=0

    if [ -d "${HOME}/.rawviewer_cache" ]; then
        echo "  WARNING: ~/.rawviewer_cache still exists (session/index cache not fully cleared)"
        issues=1
    else
        echo "  OK: ~/.rawviewer_cache removed"
    fi

    shopt -s nullglob
    local leftover_plists=("${HOME}/Library/Preferences/"*[Rr][Aa][Ww]viewer*.plist)
    shopt -u nullglob
    if [ "${#leftover_plists[@]}" -gt 0 ]; then
        echo "  WARNING: RAWviewer preference plists remain:"
        for p in "${leftover_plists[@]}"; do
            echo "    ${p}"
        done
        issues=1
    else
        echo "  OK: no RAWviewer Preferences plists"
    fi

    if defaults read com.RAWviewer.RAWviewer last_session_folder 2>/dev/null | grep -q .; then
        echo "  WARNING: last_session_folder still set in com.RAWviewer.RAWviewer"
        issues=1
    else
        echo "  OK: no last_session_folder in defaults"
    fi

    if [ "$issues" -eq 1 ]; then
        echo
        echo "Cold-start verification FAILED — fix warnings above, then run clear_cache.sh again."
        FAILED=1
        return 1
    fi
    echo "  Cold-start verification passed."
    return 0
}

mark_next_launch_cold_start() {
  # launch_dev.sh reads this once and sets RAWVIEWER_DISABLE_SESSION_RESTORE=1
  : > "${REPO_ROOT}/.rawviewer_cold_start"
  echo "  Next ./scripts/Launch/shell/launch_dev.sh will skip session restore (one launch)."
}

kill_app

remove_tree "${HOME}/.rawviewer_cache" "image/EXIF/semantic/thumbnail cache"
remove_tree "${HOME}/Library/Application Support/RAWviewer/logs" "runtime logs (~/Library/Application Support/RAWviewer/logs)"
remove_tree "${HOME}/Library/Logs/RAWviewer" "system logs (~/Library/Logs/RAWviewer)"
remove_tree "${REPO_ROOT}/src/logs" "repository dev logs (src/logs)"

# Optional cache subfolders under Application Support (if present)
remove_tree "${HOME}/Library/Application Support/RAWviewer/cache" "install cache folder"
remove_tree "${HOME}/Library/Application Support/RAWviewer/CrashDumps" "crash dumps"

clear_qsettings

verify_cold_state || true
if [ "$FAILED" -eq 0 ]; then
    mark_next_launch_cold_start
fi

echo
if [ "$FAILED" -eq 1 ]; then
    echo "Finished with warnings. Close RAWviewer and any Python dev instance, then run again."
elif [ "$CLEARED" -eq 1 ]; then
    echo "Cache and session state cleared."
    echo "For a true UI cold start: use ./scripts/Launch/shell/launch_dev.sh next"
    echo "(session restore is disabled for that one launch)."
else
    echo "Nothing found to clear (already clean)."
    mark_next_launch_cold_start
fi
echo
echo "Not removed: /Applications/RAWviewer.app or repo models/ (bundled assets)."
echo "Manual override: RAWVIEWER_DISABLE_SESSION_RESTORE=1 ./scripts/Launch/shell/launch_dev.sh"
echo

pause_if_interactive
