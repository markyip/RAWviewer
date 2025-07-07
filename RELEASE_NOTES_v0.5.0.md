# RAWviewer v0.5.0 Release Notes

## 🚀 Major Performance & User Experience Improvements

### ⚡ Enhanced RAW Processing Engine
- **New Enhanced RAW Processor** with progressive loading
- **Smart thumbnail utilization** for instant preview
- **Intelligent image caching system** with memory management
- **Performance optimizations** for large RAW files (>80MB)
- **Camera-specific processing** for Canon, Fujifilm, and Sony

### 🔧 EXIF Data Improvements
- **Immediate EXIF display** in fit-to-window mode
- **Real-time EXIF data loading** via new signal system
- **Enhanced status bar** with comprehensive image information
- **Cached EXIF extraction** for faster subsequent loads

### 🎯 Smart Features
- **Progressive image loading** - see thumbnails while full image processes
- **Preload management** for adjacent images
- **Memory-aware caching** with automatic cleanup
- **Intelligent thumbnail display** logic

### 🛠️ Technical Improvements
- **Multi-threaded processing** for better responsiveness
- **Optimized memory usage** with configurable cache budgets
- **Enhanced error handling** with fallback mechanisms
- **Improved file format support** and compatibility

## 📋 What's New

### Performance Features
- **Enhanced RAW Processor**: New threaded processor with progressive loading
- **Image Cache System**: Intelligent caching for thumbnails and full images
- **Smart Thumbnail Logic**: Only shows thumbnails when appropriate
- **Memory Management**: Automatic cache cleanup based on system memory

### User Experience
- **Instant EXIF Data**: Shows camera settings immediately in fit-to-window mode
- **Progressive Loading**: See preview while full image processes
- **Better Status Information**: Enhanced status bar with comprehensive details
- **Improved Navigation**: Maintains zoom state when navigating between images

### Technical Enhancements
- **Camera-Specific Processing**: Optimized parameters for different camera brands
- **Fallback Mechanisms**: Uses embedded thumbnails when RAW processing fails
- **Error Recovery**: Better handling of processing errors
- **Code Cleanup**: Removed development files for cleaner codebase

## 🔧 System Requirements

- **macOS**: 10.14 or later (tested on macOS 15.6)
- **Architecture**: Apple Silicon (ARM64) native
- **Memory**: 4GB+ RAM recommended for large RAW files
- **Storage**: 100MB+ free space for cache

## 📦 Installation

1. Download `RAWviewer-v0.5.0-macOS.zip`
2. Extract the archive
3. Move `RAWviewer.app` to your Applications folder
4. Right-click and select "Open" on first launch (macOS security)

## 🐛 Known Issues

- Some older RAW formats may fall back to embedded thumbnails
- Large RAW files (>100MB) may take longer to process initially
- Cache directory is created in user's home directory

## 🔄 Upgrade Notes

- **Cache System**: New cache will be created on first launch
- **Settings**: Previous settings are preserved
- **Performance**: Significant speed improvements for repeated file access

## 📈 Performance Improvements

- **Up to 5x faster** thumbnail display
- **Instant EXIF data** in fit-to-window mode
- **Reduced memory usage** with smart caching
- **Better responsiveness** during RAW processing

## 🙏 Acknowledgments

Thanks to all users who provided feedback on v0.4.1 performance issues. This release addresses the major performance concerns while maintaining image quality.

---

**Download**: [RAWviewer-v0.5.0-macOS.zip](https://github.com/markyip/RAWviewer/releases/tag/v0.5.0)

**Previous Release**: [v0.4.1 Release Notes](RELEASE_NOTES_v0.4.1.md) 