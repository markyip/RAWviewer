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