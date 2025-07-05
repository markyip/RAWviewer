import sys
import os
import unittest
import tempfile
import shutil
from unittest.mock import Mock, patch
from PyQt6.QtWidgets import QApplication
import time

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from main import RAWImageViewer, RAWProcessor
    import rawpy
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class RAWFormatCompatibilityTest(unittest.TestCase):
    """Test RAW format compatibility and support"""
    
    # RAW format definitions with their characteristics
    RAW_FORMATS = {
        'Canon': {
            'extensions': ['.cr2', '.cr3'],
            'description': 'Canon RAW formats',
            'common_models': ['EOS R5', 'EOS 5D Mark IV', '1D X Mark III'],
            'expected_support': True
        },
        'Nikon': {
            'extensions': ['.nef'],
            'description': 'Nikon Electronic Format',
            'common_models': ['D850', 'Z9', 'D780'],
            'expected_support': True
        },
        'Sony': {
            'extensions': ['.arw', '.srf'],
            'description': 'Sony RAW formats',
            'common_models': ['A7R V', 'A1', 'FX3'],
            'expected_support': True
        },
        'Adobe': {
            'extensions': ['.dng'],
            'description': 'Adobe Digital Negative',
            'common_models': ['Universal format'],
            'expected_support': True
        },
        'Olympus': {
            'extensions': ['.orf'],
            'description': 'Olympus RAW Format',
            'common_models': ['OM-1', 'E-M1 Mark III'],
            'expected_support': True
        },
        'Panasonic': {
            'extensions': ['.rw2'],
            'description': 'Panasonic RAW v2',
            'common_models': ['GH6', 'S5', 'G9'],
            'expected_support': True
        },
        'Pentax': {
            'extensions': ['.pef'],
            'description': 'Pentax Electronic Format',
            'common_models': ['K-3 Mark III', 'K-1 Mark II'],
            'expected_support': True
        },
        'Samsung': {
            'extensions': ['.srw'],
            'description': 'Samsung RAW format',
            'common_models': ['NX1', 'NX500'],
            'expected_support': True
        },
        'Sigma': {
            'extensions': ['.x3f'],
            'description': 'Sigma X3F format',
            'common_models': ['fp', 'sd Quattro H'],
            'expected_support': True
        },
        'Fujifilm': {
            'extensions': ['.raf'],
            'description': 'Fuji RAW format',
            'common_models': ['X-T5', 'GFX100S', 'X-H2S'],
            'expected_support': True
        },
        'Hasselblad': {
            'extensions': ['.3fr', '.fff'],
            'description': 'Hasselblad RAW formats',
            'common_models': ['X2D 100C', 'H6D-100c'],
            'expected_support': 'limited'
        },
        'Phase One': {
            'extensions': ['.iiq', '.cap'],
            'description': 'Phase One RAW formats',
            'common_models': ['IQ4 150MP', 'XF Camera'],
            'expected_support': 'limited'
        },
        'Epson': {
            'extensions': ['.erf'],
            'description': 'Epson RAW format',
            'common_models': ['R-D1', 'R-D1s'],
            'expected_support': 'limited'
        },
        'Mamiya': {
            'extensions': ['.mef'],
            'description': 'Mamiya Electronic Format',
            'common_models': ['Leaf Credo'],
            'expected_support': 'limited'
        },
        'Leaf': {
            'extensions': ['.mos'],
            'description': 'Leaf Mosaic format',
            'common_models': ['Credo 80', 'Aptus-II'],
            'expected_support': 'limited'
        },
        'Casio': {
            'extensions': ['.nrw'],
            'description': 'Casio RAW format',
            'common_models': ['EX-F1', 'EX-FH100'],
            'expected_support': 'limited'
        },
        'Leica': {
            'extensions': ['.rwl'],
            'description': 'Leica RAW format',
            'common_models': ['M11', 'SL2-S', 'Q2'],
            'expected_support': 'limited'
        }
    }
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Create QApplication instance if it doesn't exist
        if not QApplication.instance():
            cls.app = QApplication(sys.argv)
        else:
            cls.app = QApplication.instance()
        
        # Create temporary test directory
        cls.test_dir = tempfile.mkdtemp()
        
        # Results storage
        cls.compatibility_results = {}
        cls.support_summary = {}
        
        # Create test files for each format
        cls.create_test_raw_files()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        # Remove temporary test directory
        shutil.rmtree(cls.test_dir, ignore_errors=True)
        
        # Generate compatibility report
        cls.generate_compatibility_report()
    
    @classmethod
    def create_test_raw_files(cls):
        """Create test RAW files for each format"""
        cls.test_files = {}
        
        for brand, info in cls.RAW_FORMATS.items():
            cls.test_files[brand] = {}
            for ext in info['extensions']:
                filename = f"test_{brand.lower()}{ext}"
                filepath = os.path.join(cls.test_dir, filename)
                
                # Create empty file (for testing file recognition)
                with open(filepath, 'wb') as f:
                    # Write minimal header that might be recognized
                    if ext == '.dng':
                        # DNG magic number
                        f.write(b'II*\x00')
                    elif ext in ['.cr2', '.cr3']:
                        # Canon TIFF header
                        f.write(b'II*\x00')
                    elif ext == '.nef':
                        # Nikon TIFF header
                        f.write(b'MM\x00*')
                    else:
                        # Generic placeholder
                        f.write(b'\x00\x00\x00\x00')
                
                cls.test_files[brand][ext] = filepath
    
    def setUp(self):
        """Set up each test"""
        self.viewer = RAWImageViewer()
    
    def tearDown(self):
        """Clean up after each test"""
        if hasattr(self, 'viewer'):
            self.viewer.close()
    
    def test_supported_extensions_coverage(self):
        """Test that application supports all expected RAW formats"""
        supported_extensions = self.viewer.get_supported_extensions()
        
        for brand, info in self.RAW_FORMATS.items():
            for ext in info['extensions']:
                with self.subTest(brand=brand, extension=ext):
                    self.assertIn(ext, supported_extensions, 
                                f"Extension {ext} for {brand} not in supported list")
    
    def test_file_recognition(self):
        """Test that RAW files are recognized correctly"""
        for brand, files in self.test_files.items():
            for ext, filepath in files.items():
                with self.subTest(brand=brand, extension=ext):
                    # Test file extension recognition
                    file_ext = os.path.splitext(filepath)[1].lower()
                    supported = file_ext in self.viewer.get_supported_extensions()
                    
                    # Record result
                    if brand not in self.compatibility_results:
                        self.compatibility_results[brand] = {}
                    self.compatibility_results[brand][ext] = {
                        'recognized': supported,
                        'filepath': filepath
                    }
                    
                    self.assertTrue(supported, 
                                  f"File {filepath} with extension {ext} not recognized")
    
    def test_rawpy_library_support(self):
        """Test which formats are actually supported by rawpy library"""
        print("\nTesting rawpy library support for RAW formats...")
        
        rawpy_supported = {}
        
        for brand, files in self.test_files.items():
            rawpy_supported[brand] = {}
            for ext, filepath in files.items():
                try:
                    # Try to open with rawpy (will fail with empty files, but we can catch the error type)
                    with rawpy.imread(filepath) as raw:
                        rawpy_supported[brand][ext] = True
                except rawpy.LibRawError as e:
                    # LibRawError means the library recognizes the format but file is invalid
                    if "Unsupported file format" in str(e):
                        rawpy_supported[brand][ext] = False
                    else:
                        rawpy_supported[brand][ext] = 'recognized_but_invalid'
                except Exception as e:
                    # Other errors (like file not found, etc.)
                    rawpy_supported[brand][ext] = 'error'
                
                # Update compatibility results
                if brand in self.compatibility_results:
                    self.compatibility_results[brand][ext]['rawpy_support'] = rawpy_supported[brand][ext]
        
        # Store results for reporting
        self.rawpy_results = rawpy_supported
    
    def test_drag_drop_format_support(self):
        """Test drag and drop support for different RAW formats"""
        for brand, files in self.test_files.items():
            for ext, filepath in files.items():
                with self.subTest(brand=brand, extension=ext):
                    # Mock drag event
                    mock_event = Mock()
                    mock_mime_data = Mock()
                    mock_url = Mock()
                    mock_url.isLocalFile.return_value = True
                    mock_url.toLocalFile.return_value = filepath
                    mock_mime_data.urls.return_value = [mock_url]
                    mock_mime_data.hasUrls.return_value = True
                    mock_event.mimeData.return_value = mock_mime_data
                    
                    # Test drag enter
                    self.viewer.dragEnterEvent(mock_event)
                    
                    # Check if format is accepted
                    drag_supported = mock_event.acceptProposedAction.called
                    
                    # Update results
                    if brand in self.compatibility_results:
                        self.compatibility_results[brand][ext]['drag_drop_support'] = drag_supported
    
    def test_file_dialog_filters(self):
        """Test that file dialog includes all RAW formats"""
        # This would normally test the actual file dialog, but we'll test the filter string
        # by calling the open_file method and mocking the dialog
        
        with patch('PyQt6.QtWidgets.QFileDialog.getOpenFileName') as mock_dialog:
            mock_dialog.return_value = ('', '')  # No file selected
            
            # Trigger file dialog
            self.viewer.open_file()
            
            # Check that dialog was called with correct filters
            self.assertTrue(mock_dialog.called)
            
            # Get the filter string
            call_args = mock_dialog.call_args
            if call_args and len(call_args[0]) > 3:
                filter_string = call_args[0][3]  # 4th argument is the filter string
                
                # Check that major formats are included
                major_formats = ['.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf']
                for fmt in major_formats:
                    self.assertIn(fmt, filter_string.lower(), 
                                f"Format {fmt} not found in file dialog filters")
    
    def test_processing_performance_by_format(self):
        """Test processing performance for different formats (simulated)"""
        performance_results = {}
        
        for brand, info in self.RAW_FORMATS.items():
            performance_results[brand] = {}
            
            for ext in info['extensions']:
                # Simulate processing time (in real scenario, this would process actual files)
                start_time = time.time()
                
                # Simulate different processing times based on format complexity
                if ext in ['.cr2', '.nef', '.arw', '.dng']:
                    simulated_time = 0.1  # Fast processing
                elif ext in ['.orf', '.rw2', '.pef']:
                    simulated_time = 0.15  # Medium processing
                else:
                    simulated_time = 0.2  # Slower processing
                
                time.sleep(simulated_time)
                processing_time = time.time() - start_time
                
                performance_results[brand][ext] = {
                    'processing_time': processing_time,
                    'performance_rating': 'fast' if processing_time < 0.2 else 'medium'
                }
        
        # Store results
        self.performance_results = performance_results
    
    def test_error_handling_for_unsupported_formats(self):
        """Test error handling for unsupported or corrupted formats"""
        # Create files with unsupported extensions
        unsupported_files = {
            'bitmap': os.path.join(self.test_dir, 'test.bmp'),
            'text': os.path.join(self.test_dir, 'test.txt'),
            'pdf': os.path.join(self.test_dir, 'test.pdf')
        }
        
        for file_type, filepath in unsupported_files.items():
            with open(filepath, 'w') as f:
                f.write('test content')
            
            with self.subTest(file_type=file_type):
                # Test that unsupported files are handled gracefully
                ext = os.path.splitext(filepath)[1].lower()
                supported = ext in self.viewer.get_supported_extensions()
                
                self.assertFalse(supported, 
                               f"Unsupported format {ext} incorrectly marked as supported")
    
    @classmethod
    def generate_compatibility_report(cls):
        """Generate a comprehensive compatibility report"""
        report = []
        report.append("=" * 80)
        report.append("RAW FORMAT COMPATIBILITY REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary statistics
        total_formats = sum(len(info['extensions']) for info in cls.RAW_FORMATS.values())
        supported_count = 0
        
        for brand, results in cls.compatibility_results.items():
            for ext, result in results.items():
                if result.get('recognized', False):
                    supported_count += 1
        
        report.append(f"Total RAW formats tested: {total_formats}")
        report.append(f"Formats recognized: {supported_count}")
        report.append(f"Recognition rate: {(supported_count/total_formats)*100:.1f}%")
        report.append("")
        
        # Detailed results by brand
        for brand, info in cls.RAW_FORMATS.items():
            report.append(f"{brand} ({info['description']})")
            report.append("-" * 40)
            
            for ext in info['extensions']:
                if brand in cls.compatibility_results and ext in cls.compatibility_results[brand]:
                    result = cls.compatibility_results[brand][ext]
                    status = "✓" if result.get('recognized', False) else "✗"
                    rawpy_status = result.get('rawpy_support', 'unknown')
                    drag_drop = "✓" if result.get('drag_drop_support', False) else "✗"
                    
                    report.append(f"  {ext:<6} {status} Recognition  {drag_drop} Drag&Drop  RawPy: {rawpy_status}")
                else:
                    report.append(f"  {ext:<6} ? Not tested")
            
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 40)
        
        # Find unsupported formats
        unsupported = []
        for brand, results in cls.compatibility_results.items():
            for ext, result in results.items():
                if not result.get('recognized', False):
                    unsupported.append(f"{brand} {ext}")
        
        if unsupported:
            report.append("Consider adding support for:")
            for fmt in unsupported:
                report.append(f"  - {fmt}")
        else:
            report.append("✓ All tested formats are supported")
        
        report.append("")
        report.append("NOTES:")
        report.append("- Recognition test checks if file extension is in supported list")
        report.append("- RawPy support indicates actual library capability")
        report.append("- Some formats may be recognized but not fully supported")
        report.append("- Performance may vary significantly between formats")
        
        # Write report to file
        report_path = os.path.join(os.path.dirname(cls.test_dir), 'raw_compatibility_report.txt')
        try:
            with open(report_path, 'w') as f:
                f.write('\n'.join(report))
            print(f"\nCompatibility report saved to: {report_path}")
        except Exception as e:
            print(f"Could not save report: {e}")
        
        # Print summary to console
        print("\n" + "\n".join(report))


def run_compatibility_tests():
    """Run the RAW format compatibility tests"""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test methods
    tests = unittest.TestLoader().loadTestsFromTestCase(RAWFormatCompatibilityTest)
    suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("Running RAW Format Compatibility Tests...")
    print("=" * 60)
    
    result = run_compatibility_tests()
    
    print("\n" + "=" * 60)
    print("Compatibility Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)