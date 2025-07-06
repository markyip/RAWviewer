# Changelog

## [Unreleased]

### Fixed
- **Zoom functionality**: Fixed double-click and space bar zoom not centering on the clicked area
  - Double-click now properly zooms to the clicked location when in fit-to-window mode
  - Space bar zoom now centers on the image center for consistent behavior
  - Improved coordinate calculation for accurate zoom targeting
  - Fixed `zoom_to_point()` method to actually call `_complete_zoom_to_point()` for proper centering

### Added
- **macOS support**: Added full macOS compatibility
  - Created macOS app bundle (.app) and command-line executable
  - Added macOS icon (.icns) support
  - Cross-platform icon detection (Windows .ico, macOS .icns, fallback .png)
  - Added `build_macos.sh` script for easy macOS builds
  - Updated README with macOS build instructions

### Changed
- **Cross-platform compatibility**: Updated application to work on both Windows and macOS
- **Build system**: Enhanced build script to detect platform and use appropriate icons
- **Documentation**: Updated README to reflect cross-platform support
- **Icon management**: Moved all icon files to an `icons/` folder; updated all code and build scripts to use the new location.
- **.gitignore updates**: Added `icon.iconset/` and `tests/` directories to `.gitignore` to keep the repository clean.
- **Test and asset cleanup**: Removed `icon.iconset/` and `tests/` from the repository; they are now ignored by git.
- **Zoom/pan state preservation**: After deleting or discarding an image, the next image now preserves the previous zoom and pan state for a smoother workflow.

## [Unreleased] 