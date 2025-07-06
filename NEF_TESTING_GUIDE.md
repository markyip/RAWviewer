# NEF File Testing Guide - RAWviewer v0.4.0

## 🎯 **NEF Compatibility Fix Summary**

RAWviewer v0.4.0 now includes **smart NEF compatibility** with automatic thumbnail fallback for problematic NEF files affected by LibRaw 0.21.3 compatibility issues.

## 📦 **Built Executables**

### Location: `dist/`
- **Command-line**: `RAWviewer` (39.5 MB)
- **macOS App Bundle**: `RAWviewer.app`
- **Distribution Package**: `RAWviewer-v0.4.0-macOS-NEF-Fixed.zip` (77 MB)

### Architecture
- **Platform**: macOS (Apple Silicon ARM64)
- **Minimum OS**: macOS 11.0 (Big Sur)
- **Dependencies**: Self-contained (no external dependencies required)

## 🧪 **Testing Instructions**

### 1. **Test NEF Files**
The following NEF files were confirmed to work with thumbnail fallback:
- `DSC_0057.NEF` - Triggers thumbnail fallback (5504×8256 resolution)
- `DSC_0907.NEF` - Triggers thumbnail fallback (5504×8256 resolution)

### 2. **Expected Behavior**
When opening problematic NEF files:
1. **Status message**: "⚠️ Using embedded thumbnail due to LibRaw compatibility issue - Image quality may be reduced"
2. **Image loads successfully** with high-quality embedded thumbnail
3. **Full functionality maintained** (zoom, pan, navigation, EXIF data)
4. **No error dialogs** or application crashes

### 3. **Testing Commands**

#### Command-line executable:
```bash
# Test individual NEF file
./dist/RAWviewer /path/to/your/nef/file.NEF

# Test folder with NEF files
./dist/RAWviewer /path/to/folder/with/nef/files/
```

#### App bundle:
```bash
# Test individual NEF file
open dist/RAWviewer.app --args /path/to/your/nef/file.NEF

# Test by double-clicking RAWviewer.app and using File > Open
```

### 4. **User Interface Testing**
- ✅ **File menu**: Open NEF files via File > Open
- ✅ **Drag & drop**: Drag NEF files onto the application window
- ✅ **Folder navigation**: Open folder containing NEF files
- ✅ **Keyboard shortcuts**: Space for zoom, arrows for navigation
- ✅ **Status bar**: Shows thumbnail fallback warning when applicable

## 🔍 **Technical Details**

### **Thumbnail Fallback System**
1. **Pre-extraction**: High-resolution thumbnails extracted before RAW processing
2. **Automatic fallback**: Seamlessly switches to thumbnail when RAW processing fails
3. **Quality preservation**: Uses embedded JPEG thumbnails (typically 8256×5504)
4. **Error handling**: Graceful degradation with informative user feedback

### **Supported NEF Processing**
- **Full RAW processing**: For compatible NEF files
- **Thumbnail fallback**: For NEF files with LibRaw 0.21.3 issues
- **Error reporting**: Clear messages for unsupported or corrupted files

### **Dependencies Included**
- PyQt6 (GUI framework)
- rawpy 0.25.0 (RAW processing with LibRaw 0.21.3)
- Pillow (thumbnail image processing)
- NumPy (image array handling)
- Additional supporting libraries

## ✅ **Verification Checklist**

### **Basic Functionality**
- [ ] Application launches without errors
- [ ] Can open regular RAW files (CR2, ARW, etc.)
- [ ] Can open JPEG files
- [ ] Keyboard shortcuts work correctly
- [ ] Zoom and pan functionality works

### **NEF-Specific Testing**
- [ ] NEF files open successfully (with or without fallback)
- [ ] Status bar shows appropriate messages
- [ ] Image quality is acceptable when using thumbnails
- [ ] Navigation between NEF files works
- [ ] EXIF data displays correctly for NEF files

### **Error Handling**
- [ ] Graceful handling of corrupted files
- [ ] Informative error messages
- [ ] No application crashes with problematic files

## 🐛 **Known Limitations**

1. **Thumbnail Quality**: When fallback is used, image quality depends on embedded thumbnail resolution
2. **LibRaw Compatibility**: Some very new NEF formats may still have issues
3. **Processing Speed**: Thumbnail extraction may be slightly slower than full RAW processing

## 📋 **Troubleshooting**

### **If NEF files don't open:**
1. Check if the file is actually a NEF file
2. Try with a different NEF file to isolate the issue
3. Check console output for specific error messages

### **If app won't launch:**
1. Right-click RAWviewer.app → Open (to bypass Gatekeeper)
2. Check System Preferences → Security & Privacy for blocked apps
3. Ensure macOS version is 11.0 or later

### **Performance issues:**
1. Close other memory-intensive applications
2. Ensure sufficient free disk space
3. Try with smaller NEF files first

## 🎉 **Success Criteria**

The NEF compatibility fix is successful if:
- ✅ Previously "corrupted" NEF files now open and display
- ✅ User receives clear feedback about thumbnail fallback usage
- ✅ Image quality is acceptable for viewing and evaluation
- ✅ All other RAW formats continue to work normally
- ✅ Application stability is maintained

---

**Built on**: January 6, 2025  
**Version**: RAWviewer v0.4.0 with NEF Thumbnail Fallback  
**Architecture**: macOS ARM64 (Apple Silicon)  
**LibRaw Version**: 0.21.3 with smart compatibility handling 