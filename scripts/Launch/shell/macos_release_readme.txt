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

SEMANTIC SEARCH (first use)
---------------------------

When you open gallery search for the first time, RAWviewer may ask to
download the offline AI models (~150 MB, one-time, needs internet).
Click Download. Official release zips built with build_macos.sh already
include these models; older or custom builds may need this step.

RUN FROM THIS FOLDER (no Applications copy)
-------------------------------------------

  bash remove_macos_quarantine.sh

More help: https://github.com/markyip/RAWviewer
