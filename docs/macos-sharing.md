# macOS and Windows share

**Share / Open** lists system share targets for the current photo (single view)
or the gallery selection.

## What each target receives

| Target | Payload |
|--------|---------|
| WhatsApp, Signal, and similar messaging apps | JPEG. Camera RAW / TIFF / DNG are not in those extensions' accepted types, so the menu would omit them if we offered the original. The proxy is the embedded JPEG preview when LibRaw has one, otherwise a re-encode. |
| AirDrop | **Original file** (the RAW, not the JPEG proxy). |
| Mail and other apps that accept the original | Original when `canPerformWithItems` says so; otherwise the JPEG proxy. |

JPEG / PNG / HEIC / WebP / GIF / BMP are shared as-is.

Files under `/Volumes/` are copied to a local temp first so share extensions
that refuse network volumes still see a path.

## macOS

The default path is a **Qt menu** of `NSSharingService` titles
(`RAWVIEWER_SHARE_MENU=1`). The native `NSSharingServicePicker` popover often
spins empty in this Qt 6 host; keep it opt-in
(`RAWVIEWER_SHARE_TRY_NATIVE_PICKER=1`).

**AirDrop** is in that menu. Choosing it:

1. Runs `osascript` (JavaScript) so the person picker is hosted **outside**
   Qt — `NSSharingService.performWithItems` on a Qt `NSWindow` usually
   returns success with no panel.
2. If that fails, in-process `com.apple.share.AirDrop.send` (disable with
   `RAWVIEWER_AIRDROP_PERFORM=0`).
3. If the panel still does not appear, Reveal in Finder with a status hint.

Hide AirDrop with `RAWVIEWER_SHARE_SHOW_AIRDROP=0`.

WhatsApp / Signal appear only when those apps are installed and their share
extensions are enabled in System Settings.

## Windows

The WinRT share flyout / clipboard HDROP path uses the same JPEG proxy for
RAW so WhatsApp and Signal can be offered. Multi-select still copies paths
when a flyout is not available.

## Code

| File | Role |
|------|------|
| `src/share_payload.py` | `needs_jpeg_proxy` / `write_jpeg_proxy` |
| `src/main.py` | Menu, AirDrop osascript, Windows chain |
| `src/windows_share.py` | WinRT DataTransferManager helper |
| `testplan/auto/t_share_payload.py` | Proxy vs original |

Older v2.1–v2.2 picker experiments lived in a since-removed
`macos-sharing-v21-v22.md`. AirDrop is no longer hidden by default.
