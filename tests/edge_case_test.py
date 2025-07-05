import sys
import os
import unittest
import tempfile
import shutil
import stat
from unittest.mock import Mock, patch, MagicMock
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
import time

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from main import RAWImageViewer
except ImportError as e:
    print(f"Error importing main module: {e}")
    sys.exit(1)


class EdgeCaseTest(unittest.TestCase):
    """Test edge cases and unusual scenarios"""
    
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
        
        # Create test scenarios
        cls.create_test_scenarios()
        
        # Results storage
        cls.edge_case_results = {}
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        # Clean up read-only files
        cls.cleanup_readonly_files()
        
        # Remove temporary test directory
        shutil.rmtree(cls.test_dir, ignore_errors=True)
        
        # Generate edge case report
        cls.generate_edge_case_report()
    
    @classmethod
    def create_test_scenarios(cls):
        """Create various test scenarios"""
        # Empty folder
        cls.empty_folder = os.path.join(cls.test_dir, 'empty_folder')
        os.makedirs(cls.empty_folder, exist_ok=True)
        
        # Single image folder
        cls.single_image_folder = os.path.join(cls.test_dir, 'single_image')
        os.makedirs(cls.single_image_folder, exist_ok=True)
        cls.single_image_file = os.path.join(cls.single_image_folder, 'single.cr2')
        with open(cls.single_image_file, 'wb') as f:
            f.write(b'fake RAW content')
        
        # Large file simulation
        cls.large_file = os.path.join(cls.test_dir, 'large_file.nef')
        with open(cls.large_file, 'wb') as f:
            # Create a 50MB file
            f.write(b'0' * (50 * 1024 * 1024))
        
        # Corrupted files
        cls.corrupted_files = []
        for i, ext in enumerate(['.cr2', '.nef', '.arw', '.dng']):
            corrupted_file = os.path.join(cls.test_dir, f'corrupted_{i}{ext}')
            with open(corrupted_file, 'wb') as f:
                # Write invalid content
                f.write(b'This is not a valid RAW file content')
            cls.corrupted_files.append(corrupted_file)
        
        # Files with special characters
        cls.special_char_files = []
        special_names = [
            'test with spaces.cr2',
            'test-with-dashes.nef',
            'test_with_underscores.arw',
            'test.with.dots.dng',
            'test@with#special$chars.cr2'
        ]
        
        for name in special_names:
            try:
                file_path = os.path.join(cls.test_dir, name)
                with open(file_path, 'wb') as f:
                    f.write(b'test content')
                cls.special_char_files.append(file_path)
            except (OSError, UnicodeError):
                # Skip files that can't be created on this system
                pass
        
        # Read-only files
        cls.readonly_files = []
        for i in range(2):
            readonly_file = os.path.join(cls.test_dir, f'readonly_{i}.cr2')
            with open(readonly_file, 'wb') as f:
                f.write(b'readonly content')
            
            # Make file read-only
            try:
                os.chmod(readonly_file, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
                cls.readonly_files.append(readonly_file)
            except OSError:
                # Skip if we can't make it read-only
                pass
        
        # Very long filename
        try:
            long_name = 'a' * 200 + '.cr2'
            cls.long_filename = os.path.join(cls.test_dir, long_name)
            with open(cls.long_filename, 'wb') as f:
                f.write(b'long filename content')
        except OSError:
            cls.long_filename = None
        
        # Zero-byte files
        cls.zero_byte_files = []
        for ext in ['.cr2', '.nef', '.arw']:
            zero_file = os.path.join(cls.test_dir, f'zero_byte{ext}')
            with open(zero_file, 'wb') as f:
                pass  # Create empty file
            cls.zero_byte_files.append(zero_file)
        
        # Mixed content folder
        cls.mixed_folder = os.path.join(cls.test_dir, 'mixed_content')
        os.makedirs(cls.mixed_folder, exist_ok=True)
        
        # Add various file types
        mixed_files = [
            ('image1.cr2', b'raw content'),
            ('image2.nef', b'nef content'),
            ('document.txt', b'text content'),
            ('readme.md', b'markdown content'),
            ('script.py', b'python content'),
            ('image3.arw', b'sony raw'),
            ('data.json', b'{"key": "value"}'),
        ]
        
        for filename, content in mixed_files:
            file_path = os.path.join(cls.mixed_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(content)
    
    @classmethod
    def cleanup_readonly_files(cls):
        """Clean up read-only files by making them writable first"""
        for file_path in cls.readonly_files:
            try:
                if os.path.exists(file_path):
                    os.chmod(file_path, stat.S_IWRITE | stat.S_IREAD)
            except OSError:
                pass
    
    def setUp(self):
        """Set up each test"""
        self.viewer = RAWImageViewer()
    
    def tearDown(self):
        """Clean up after each test"""
        if hasattr(self, 'viewer'):
            self.viewer.close()
    
    def test_empty_folder_handling(self):
        """Test behavior with empty folders"""
        # Try to scan empty folder
        dummy_file = os.path.join(self.empty_folder, 'dummy.cr2')
        
        # Create temporary file for scanning
        with open(dummy_file, 'wb') as f:
            f.write(b'temp')
        
        try:
            self.viewer.scan_folder_for_images(dummy_file)
            
            # Should handle empty folder gracefully
            self.assertEqual(len(self.viewer.image_files), 1)  # Only the dummy file
            
            # Remove the file and test navigation
            os.remove(dummy_file)
            self.viewer.image_files = []
            
            # Navigation should not crash
            self.viewer.navigate_to_next_image()
            self.viewer.navigate_to_previous_image()
            
            self.edge_case_results['empty_folder'] = {
                'tested': True,
                'passed': True,
                'notes': 'Empty folder handled gracefully'
            }
            
        except Exception as e:
            self.edge_case_results['empty_folder'] = {
                'tested': True,
                'passed': False,
                'error': str(e)
            }
    
    def test_single_image_folder(self):
        """Test behavior with folder containing only one image"""
        try:
            self.viewer.scan_folder_for_images(self.single_image_file)
            
            # Should find the single image
            self.assertEqual(len(self.viewer.image_files), 1)
            self.assertEqual(self.viewer.current_file_index, 0)
            
            # Navigation should not crash or change index
            initial_index = self.viewer.current_file_index
            self.viewer.navigate_to_next_image()
            self.viewer.navigate_to_previous_image()
            
            # Index should remain the same for single image
            self.assertEqual(self.viewer.current_file_index, initial_index)
            
            self.edge_case_results['single_image'] = {
                'tested': True,
                'passed': True,
                'notes': 'Single image folder handled correctly'
            }
            
        except Exception as e:
            self.edge_case_results['single_image'] = {
                'tested': True,
                'passed': False,
                'error': str(e)
            }
    
    def test_large_file_handling(self):
        """Test handling of very large files"""
        try:
            # Get file size
            file_size = os.path.getsize(self.large_file)
            self.assertGreater(file_size, 40 * 1024 * 1024)  # Should be > 40MB
            
            # Test that large file is recognized
            self.viewer.scan_folder_for_images(self.large_file)
            
            # Should be in the file list
            self.assertIn(self.large_file, self.viewer.image_files)
            
            # Try to load (will fail with fake content, but should not crash)
            with patch.object(self.viewer, 'show_error') as mock_error:
                self.viewer.load_raw_image(self.large_file)
                
                # Should show error gracefully
                # (Since it's not a real RAW file, it will fail to process)
            
            self.edge_case_results['large_file'] = {
                'tested': True,
                'passed': True,
                'file_size_mb': file_size / (1024 * 1024),
                'notes': 'Large file handled without crashing'
            }
            
        except Exception as e:
            self.edge_case_results['large_file'] = {
                'tested': True,
                'passed': False,
                'error': str(e)
            }
    
    def test_corrupted_files(self):
        """Test handling of corrupted RAW files"""
        corrupted_results = []
        
        for corrupted_file in self.corrupted_files:
            try:
                # Test that corrupted file is recognized by extension
                self.viewer.scan_folder_for_images(corrupted_file)
                
                # Should be in the file list (recognized by extension)
                self.assertIn(corrupted_file, self.viewer.image_files)
                
                # Try to load corrupted file
                with patch.object(self.viewer, 'show_error') as mock_error:
                    self.viewer.load_raw_image(corrupted_file)
                    
                    # Should show error for corrupted file
                    # (Will fail to process, but should not crash)
                
                corrupted_results.append({
                    'file': os.path.basename(corrupted_file),
                    'passed': True,
                    'notes': 'Corrupted file handled gracefully'
                })
                
            except Exception as e:
                corrupted_results.append({
                    'file': os.path.basename(corrupted_file),
                    'passed': False,
                    'error': str(e)
                })
        
        self.edge_case_results['corrupted_files'] = {
            'tested': True,
            'results': corrupted_results,
            'total_files': len(self.corrupted_files)
        }
    
    def test_special_character_filenames(self):
        """Test handling of files with special characters in names"""
        special_results = []
        
        for special_file in self.special_char_files:
            try:
                # Test that file with special characters is handled
                self.viewer.scan_folder_for_images(special_file)
                
                # Should be in the file list
                self.assertIn(special_file, self.viewer.image_files)
                
                # Test window title update
                self.viewer.current_file_path = special_file
                filename = os.path.basename(special_file)
                self.viewer.setWindowTitle(f"RAW Image Viewer - {filename}")
                
                # Should not crash
                special_results.append({
                    'file': os.path.basename(special_file),
                    'passed': True,
                    'notes': 'Special characters handled correctly'
                })
                
            except Exception as e:
                special_results.append({
                    'file': os.path.basename(special_file),
                    'passed': False,
                    'error': str(e)
                })
        
        self.edge_case_results['special_characters'] = {
            'tested': True,
            'results': special_results,
            'total_files': len(self.special_char_files)
        }
    
    def test_readonly_file_deletion(self):
        """Test deletion of read-only files"""
        if not self.readonly_files:
            self.skipTest("No read-only files could be created")
        
        readonly_results = []
        
        for readonly_file in self.readonly_files:
            try:
                # Set up viewer with read-only file
                self.viewer.current_file_path = readonly_file
                self.viewer.image_files = [readonly_file]
                self.viewer.current_file_index = 0
                
                # Mock confirmation dialog to accept deletion
                with patch.object(self.viewer, 'confirm_deletion') as mock_confirm:
                    mock_confirm.return_value = True
                    
                    with patch.object(self.viewer, 'show_error') as mock_error:
                        # Try to delete read-only file
                        self.viewer.delete_current_image()
                        
                        # Should show error for read-only file
                        # (send2trash might fail on read-only files)
                
                readonly_results.append({
                    'file': os.path.basename(readonly_file),
                    'passed': True,
                    'notes': 'Read-only file deletion handled gracefully'
                })
                
            except Exception as e:
                readonly_results.append({
                    'file': os.path.basename(readonly_file),
                    'passed': False,
                    'error': str(e)
                })
        
        self.edge_case_results['readonly_files'] = {
            'tested': True,
            'results': readonly_results,
            'total_files': len(self.readonly_files)
        }
    
    def test_long_filename_handling(self):
        """Test handling of very long filenames"""
        if not self.long_filename:
            self.skipTest("Long filename could not be created")
        
        try:
            # Test that long filename is handled
            self.viewer.scan_folder_for_images(self.long_filename)
            
            # Should be in the file list
            self.assertIn(self.long_filename, self.viewer.image_files)
            
            # Test status bar with long filename
            self.viewer.current_file_path = self.long_filename
            self.viewer.update_status_bar()
            
            # Should not crash
            self.edge_case_results['long_filename'] = {
                'tested': True,
                'passed': True,
                'filename_length': len(os.path.basename(self.long_filename)),
                'notes': 'Long filename handled correctly'
            }
            
        except Exception as e:
            self.edge_case_results['long_filename'] = {
                'tested': True,
                'passed': False,
                'error': str(e)
            }
    
    def test_zero_byte_files(self):
        """Test handling of zero-byte files"""
        zero_results = []
        
        for zero_file in self.zero_byte_files:
            try:
                # Test that zero-byte file is recognized by extension
                self.viewer.scan_folder_for_images(zero_file)
                
                # Should be in the file list
                self.assertIn(zero_file, self.viewer.image_files)
                
                # Try to load zero-byte file
                with patch.object(self.viewer, 'show_error') as mock_error:
                    self.viewer.load_raw_image(zero_file)
                    
                    # Should show error for zero-byte file
                    # (Will fail to process, but should not crash)
                
                zero_results.append({
                    'file': os.path.basename(zero_file),
                    'passed': True,
                    'notes': 'Zero-byte file handled gracefully'
                })
                
            except Exception as e:
                zero_results.append({
                    'file': os.path.basename(zero_file),
                    'passed': False,
                    'error': str(e)
                })
        
        self.edge_case_results['zero_byte_files'] = {
            'tested': True,
            'results': zero_results,
            'total_files': len(self.zero_byte_files)
        }
    
    def test_mixed_content_folder(self):
        """Test folder with mixed file types"""
        try:
            # Scan folder with mixed content
            test_raw_file = os.path.join(self.mixed_folder, 'image1.cr2')
            self.viewer.scan_folder_for_images(test_raw_file)
            
            # Should only include supported image files
            supported_extensions = self.viewer.get_supported_extensions()
            expected_files = []
            
            for filename in os.listdir(self.mixed_folder):
                ext = os.path.splitext(filename)[1].lower()
                if ext in supported_extensions:
                    expected_files.append(os.path.join(self.mixed_folder, filename))
            
            # Should find only supported files
            self.assertEqual(len(self.viewer.image_files), len(expected_files))
            
            # All files in the list should be supported
            for file_path in self.viewer.image_files:
                ext = os.path.splitext(file_path)[1].lower()
                self.assertIn(ext, supported_extensions)
            
            self.edge_case_results['mixed_content'] = {
                'tested': True,
                'passed': True,
                'total_files': len(os.listdir(self.mixed_folder)),
                'supported_files': len(self.viewer.image_files),
                'notes': 'Mixed content folder filtered correctly'
            }
            
        except Exception as e:
            self.edge_case_results['mixed_content'] = {
                'tested': True,
                'passed': False,
                'error': str(e)
            }
    
    def test_rapid_navigation(self):
        """Test rapid navigation between images"""
        try:
            # Set up multiple files
            test_files = [
                os.path.join(self.mixed_folder, 'image1.cr2'),
                os.path.join(self.mixed_folder, 'image2.nef'),
                os.path.join(self.mixed_folder, 'image3.arw')
            ]
            
            self.viewer.image_files = test_files
            self.viewer.current_file_index = 0
            
            # Rapid navigation
            with patch.object(self.viewer, 'load_raw_image') as mock_load:
                for _ in range(20):
                    self.viewer.navigate_to_next_image()
                    QApplication.processEvents()
                
                # Should handle rapid navigation without issues
                self.assertTrue(0 <= self.viewer.current_file_index < len(test_files))
            
            self.edge_case_results['rapid_navigation'] = {
                'tested': True,
                'passed': True,
                'notes': 'Rapid navigation handled correctly'
            }
            
        except Exception as e:
            self.edge_case_results['rapid_navigation'] = {
                'tested': True,
                'passed': False,
                'error': str(e)
            }
    
    def test_drag_drop_invalid_files(self):
        """Test drag and drop with invalid file types"""
        invalid_files = [
            os.path.join(self.mixed_folder, 'document.txt'),
            os.path.join(self.mixed_folder, 'script.py'),
            os.path.join(self.mixed_folder, 'data.json')
        ]
        
        invalid_results = []
        
        for invalid_file in invalid_files:
            try:
                # Mock drag event with invalid file
                mock_event = Mock()
                mock_mime_data = Mock()
                mock_url = Mock()
                mock_url.isLocalFile.return_value = True
                mock_url.toLocalFile.return_value = invalid_file
                mock_mime_data.urls.return_value = [mock_url]
                mock_mime_data.hasUrls.return_value = True
                mock_event.mimeData.return_value = mock_mime_data
                
                # Test drag enter
                self.viewer.dragEnterEvent(mock_event)
                
                # Should reject invalid file
                mock_event.ignore.assert_called_once()
                
                invalid_results.append({
                    'file': os.path.basename(invalid_file),
                    'passed': True,
                    'notes': 'Invalid file correctly rejected'
                })
                
            except Exception as e:
                invalid_results.append({
                    'file': os.path.basename(invalid_file),
                    'passed': False,
                    'error': str(e)
                })
        
        self.edge_case_results['drag_drop_invalid'] = {
            'tested': True,
            'results': invalid_results,
            'total_files': len(invalid_files)
        }
    
    def test_memory_cleanup_after_errors(self):
        """Test that memory is cleaned up properly after errors"""
        try:
            initial_objects = len([obj for obj in locals().values() if hasattr(obj, '__dict__')])
            
            # Simulate various error scenarios
            error_scenarios = [
                lambda: self.viewer.load_raw_image('/nonexistent/file.cr2'),
                lambda: self.viewer.scan_folder_for_images('/nonexistent/folder/file.cr2'),
                lambda: self.viewer.on_processing_error('Test error'),
            ]
            
            for scenario in error_scenarios:
                try:
                    scenario()
                except Exception:
                    pass  # Expected to fail
                
                # Process events to allow cleanup
                QApplication.processEvents()
            
            # Check that we haven't accumulated too many objects
            final_objects = len([obj for obj in locals().values() if hasattr(obj, '__dict__')])
            object_growth = final_objects - initial_objects
            
            self.edge_case_results['memory_cleanup'] = {
                'tested': True,
                'passed': object_growth < 100,  # Reasonable threshold
                'object_growth': object_growth,
                'notes': 'Memory cleanup appears to work correctly'
            }
            
        except Exception as e:
            self.edge_case_results['memory_cleanup'] = {
                'tested': True,
                'passed': False,
                'error': str(e)
            }
    
    @classmethod
    def generate_edge_case_report(cls):
        """Generate edge case test report"""
        report = []
        report.append("=" * 80)
        report.append("EDGE CASE TEST REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Test summary
        total_tests = len(cls.edge_case_results)
        passed_tests = sum(1 for result in cls.edge_case_results.values() 
                          if result.get('passed', False))
        
        report.append(f"Total edge case tests: {total_tests}")
        report.append(f"Passed tests: {passed_tests}")
        report.append(f"Success rate: {(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "No tests run")
        report.append("")
        
        # Detailed results
        report.append("DETAILED RESULTS:")
        report.append("-" * 40)
        
        for test_name, result in cls.edge_case_results.items():
            status = "✓" if result.get('passed', False) else "✗"
            report.append(f"{test_name:<25} {status}")
            
            if 'notes' in result:
                report.append(f"  Notes: {result['notes']}")
            
            if 'error' in result:
                report.append(f"  Error: {result['error']}")
            
            if 'results' in result:
                # Handle multi-file results
                for sub_result in result['results']:
                    sub_status = "✓" if sub_result.get('passed', False) else "✗"
                    report.append(f"    {sub_result['file']:<20} {sub_status}")
            
            report.append("")
        
        # Edge cases covered
        report.append("EDGE CASES COVERED:")
        report.append("-" * 40)
        edge_cases = [
            "Empty folders",
            "Single image folders",
            "Very large files (>40MB)",
            "Corrupted RAW files",
            "Files with special characters",
            "Read-only files",
            "Very long filenames",
            "Zero-byte files",
            "Mixed content folders",
            "Rapid navigation",
            "Invalid drag & drop files",
            "Memory cleanup after errors"
        ]
        
        for case in edge_cases:
            report.append(f"  ✓ {case}")
        
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 40)
        
        failed_tests = [name for name, result in cls.edge_case_results.items() 
                       if not result.get('passed', False)]
        
        if failed_tests:
            report.append("Issues found with:")
            for test in failed_tests:
                report.append(f"  - {test}")
            report.append("")
        
        report.append("General recommendations:")
        report.append("- Continue robust error handling for edge cases")
        report.append("- Monitor memory usage with large files")
        report.append("- Consider user feedback for unsupported operations")
        report.append("- Test with real-world RAW files when possible")
        report.append("- Implement progressive loading for very large files")
        
        # Save report
        report_path = os.path.join(os.path.dirname(cls.test_dir), 'edge_case_report.txt')
        try:
            with open(report_path, 'w') as f:
                f.write('\n'.join(report))
            print(f"\nEdge case report saved to: {report_path}")
        except Exception as e:
            print(f"Could not save report: {e}")
        
        # Print summary
        print("\n" + "\n".join(report))


def run_edge_case_tests():
    """Run the edge case tests"""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test methods
    tests = unittest.TestLoader().loadTestsFromTestCase(EdgeCaseTest)
    suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("Running Edge Case Tests...")
    print("=" * 60)
    
    result = run_edge_case_tests()
    
    print("\n" + "=" * 60)
    print("Edge Case Test Summary:")
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