RAWviewer for macOS — START HERE
================================

Requires macOS 13 Ventura or newer.

INSTALL
-------

1. Extract this zip.

2. Open **Terminal** (Applications → Utilities → Terminal).

3. Go to this folder — type `cd ` (with a space), drag this folder
   onto Terminal, press Return.

4. Run:

   bash install_macos_app.sh

5. Click **Install**, then **Open** in the dialogs.

After that, open RAWviewer from **Applications** like any other app.


CLEAR CACHE (keep the app — recommended after upgrades)
-------------------------------------------------------

Use this when thumbnails, search, or gallery feel wrong/slow after an
upgrade. It does **not** delete your photos, XMP sidecars, or the app
in Applications.

What it removes:
  • Photo cache (~/.rawviewer_cache)
  • App logs and saved window/session preferences

Steps:

1. Quit RAWviewer if it is open.

2. Easiest: double-click **Clear Cache.command**
   (if macOS blocks it: right-click → Open → Open).

   Or in Terminal (from this extracted folder):

   bash clear_macos_cache.sh

3. Open RAWviewer again from Applications. The first folder open may
   rebuild thumbnails — that is normal.


UNINSTALL (remove the app and all local data)
---------------------------------------------

Dragging RAWviewer to the Trash alone is **not** enough — cache and
preferences stay on your Mac. Use the uninstall script below.

What it removes:
  • RAWviewer.app / RAWviewer_Lite.app from Applications
  • Photo cache (~/.rawviewer_cache)
  • Map tiles, logs, optional AI models, and app preferences

Keep this extracted folder (or re-download the zip from GitHub Releases)
so you can uninstall later.

Steps:

1. Quit RAWviewer if it is open.

2. Easiest: double-click **Uninstall RAWviewer.command**
   (if macOS blocks it: right-click → Open → Open).

   Or in Terminal (from this extracted folder):

   bash uninstall_macos_app.sh

3. Confirm **Uninstall** in the dialog.

4. Done — the app and local cache/preferences are removed.
   Your photo files are never deleted.

Optional: if you already dragged the app to Trash, still run the
uninstall script once to clear leftover cache and preferences.


CLEAR CACHE vs UNINSTALL
------------------------

  Clear Cache     Keep app; wipe cache/preferences only
  Uninstall       Remove app + cache + preferences


SEMANTIC SEARCH (Full builds — first use)
---------------------------------------

**Lite** builds skip this — they use metadata search only.

When you open gallery search for the first time on a **Full** build,
RAWviewer may ask to download the offline AI models
(~150 MB, one-time, needs internet). Click Download. Official **Full**
release zips built with build_macos.sh already include these models;
older or custom builds may need this step.


RUN FROM THIS FOLDER (no Applications copy)
-------------------------------------------

  bash remove_macos_quarantine.sh


More help: https://github.com/markyip/RAWviewer
