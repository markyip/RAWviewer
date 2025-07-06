# Automatic Image Orientation Correction

RAWviewer now includes automatic image orientation correction that ensures your photos display correctly regardless of how you held the camera when taking the shot.

## How It Works

### EXIF Orientation Reading
- RAWviewer reads the EXIF "Orientation" tag from both RAW and JPEG files
- This tag contains information about how the camera was oriented when the photo was taken
- The orientation is recorded as a value from 1-8, representing different rotations and flips

### Automatic Correction
- **RAW Files**: Orientation correction is applied after RAW processing but before display
- **JPEG/HEIF Files**: Orientation correction is applied to the loaded QPixmap
- **Thumbnail Fallback**: When using NEF thumbnail fallback, orientation is also corrected

### Supported Orientations

| Value | Description | Correction Applied |
|-------|-------------|-------------------|
| 1 | Normal (landscape) | None |
| 2 | Mirrored horizontal | Horizontal flip |
| 3 | Rotated 180° | 180° rotation |
| 4 | Mirrored vertical | Vertical flip |
| 5 | Mirrored horizontal + 90° CCW | Horizontal flip + 90° CCW |
| 6 | Rotated 90° CW | 90° clockwise rotation |
| 7 | Mirrored horizontal + 90° CW | Horizontal flip + 90° CW |
| 8 | Rotated 90° CCW | 90° counter-clockwise rotation |

## Benefits

### For Photographers
- **Portrait photos display correctly** - No more sideways images
- **Mixed orientation shoots** - Seamlessly view landscape and portrait images in sequence
- **Consistent viewing experience** - Images appear as intended, not as captured by sensor

### For Workflow
- **Faster culling** - No need to manually rotate images during review
- **Better composition assessment** - See images in their intended orientation immediately
- **Reduced post-processing** - Orientation is handled automatically

## Technical Implementation

### RAW Processing
```python
# Get orientation from EXIF
orientation = self.get_orientation_from_exif(file_path)

# Process RAW image
rgb_image = raw.postprocess()

# Apply orientation correction
rgb_image = self.apply_orientation_correction(rgb_image, orientation)
```

### Non-RAW Processing
```python
# Load image
pixmap = QPixmap(file_path)

# Get orientation and apply correction
orientation = self.get_orientation_from_exif(file_path)
pixmap = self.apply_orientation_to_pixmap(pixmap, orientation)
```

## Compatibility

### Camera Support
- **All major camera brands** - Canon, Nikon, Sony, Fujifilm, etc.
- **RAW and JPEG formats** - Works with any format that includes EXIF orientation
- **Modern and legacy cameras** - Supports EXIF orientation standard used since early 2000s

### Format Support
- **RAW formats**: CR2, CR3, NEF, ARW, DNG, ORF, RW2, PEF, SRW, X3F, RAF, and more
- **Standard formats**: JPEG, HEIF
- **Thumbnail fallback**: NEF thumbnails also respect orientation

## Error Handling

- **Missing EXIF data**: Defaults to normal orientation (no rotation)
- **Corrupted orientation tag**: Falls back to normal orientation
- **Invalid orientation values**: Treats as normal orientation
- **File read errors**: Gracefully handles EXIF reading failures

## Performance

- **Minimal overhead**: Orientation reading adds negligible processing time
- **Efficient rotation**: Uses optimized NumPy operations for array rotation
- **Memory efficient**: No additional memory copies for normal orientation (value 1)
- **GPU acceleration**: QTransform rotations can use GPU acceleration when available

## User Experience

### Automatic Operation
- **No user intervention required** - Works transparently
- **No settings to configure** - Always enabled
- **Consistent behavior** - Same logic for all image types

### Visual Feedback
- **Immediate correction** - Images display in correct orientation from the start
- **Preserved aspect ratios** - Rotation maintains image quality
- **Smooth navigation** - Orientation correction doesn't affect navigation speed

## Future Enhancements

### Potential Improvements
- **Manual override option** - Allow users to disable automatic correction
- **Orientation indicator** - Show when correction has been applied
- **Batch orientation fixes** - Apply orientation corrections to multiple files
- **Custom rotation** - Allow manual rotation independent of EXIF data

### Technical Considerations
- **Metadata preservation** - Ensure EXIF data is preserved during any file operations
- **Performance optimization** - Cache orientation data for faster repeated access
- **Memory management** - Optimize rotation operations for large RAW files

---

This feature significantly improves the user experience by ensuring all images display correctly regardless of camera orientation, making RAWviewer more intuitive and professional for photography workflows. 