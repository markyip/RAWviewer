RAWviewer for macOS — START HERE
================================

Requires macOS 13 Ventura or newer.

WHAT TO DOUBLE-CLICK
--------------------

Most users (recommended):
  → Install RAWviewer.command
    Installs RAWviewer to your Applications folder and removes the
    macOS "downloaded from the internet" block. Then you can open it
    from Applications or Launchpad like any other app.

Run from this folder only (USB, Desktop, etc.):
  → Remove Quarantine.command
    Removes the download block so you can double-click RAWviewer.app
    here without moving it to Applications.

You do NOT need to run both .command files. Pick one path above.

THE .SH FILES (ignore unless you use Terminal)
----------------------------------------------
  install_macos_app.sh      — same as Install RAWviewer.command
  remove_macos_quarantine.sh — same as Remove Quarantine.command

IF macOS BLOCKS A .command FILE THE FIRST TIME
----------------------------------------------
Right-click the file → Open → Open (once). After that, double-click works.

IF THE APP STILL SAYS "DAMAGED" OR WON'T OPEN
---------------------------------------------
Run Remove Quarantine.command again, or in Terminal:
  xattr -cr /path/to/RAWviewer.app

More help: https://github.com/markyip/RAWviewer
