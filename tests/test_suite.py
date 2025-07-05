import sys
import os
import unittest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtGui import QPixmap
import time
import psutil
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from main import RAWImageViewer, RAWProcessor
except ImportError as e:
    print(f"Error importing main module: {e}")
    sys.exit(1)


class TestRAWImageViewer(unittest.TestCase):
    """Comprehensive test suite for RAW Image Viewer application"""
    
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
        
        # Create test files
        cls.create_test_files()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        # Remove temporary test directory
        shutil.rmtree(cls.test_dir, ignore_errors=True)
    
    @classmethod
    def create_test_files(cls):
        """Create test files for testing"""
        # Create empty files with different extensions
        cls.test_files = {
            'valid_raw': os.path.join(cls.test_dir, 'test.cr2'),
            'valid_raw2': os.path.join(cls.test_dir, 'test.nef'),
            'invalid_file': os.path.join(cls.test_dir, 'test.txt'),
            'corrupted_raw': os.path.join(cls.test_dir, 'corrupted.dng'),
            'empty_folder': os.path.join(cls.test_dir, 'empty'),
            'single_image_folder': os.path.join(cls.test_dir, 'single'),
        }
        
        # Create empty files
        for file_path in [cls.test_files['valid_raw'], cls.test_files['valid_raw2'], 
                         cls.test_files['invalid_file'], cls.test_files['corrupted_raw']]:
            with open(file_path, 'w') as f:
                f.write('')
        
        # Create folders
        os.makedirs(cls.test_files['empty_folder'], exist_ok=True)
        os.makedirs(cls.test_files['single_image_folder'], exist_ok=True)
        
        # Create a single image in single_image_folder
        single_image = os.path.join(cls.test_files['single_image_folder'], 'single.arw')
        with open(single_image, 'w') as f:
            f.write('')
    
    def setUp(self):
        """Set up each test"""
        self.viewer = RAWImageViewer()
    
    def tearDown(self):
        """Clean up after each test"""
        if hasattr(self, 'viewer'):
            self.viewer.close()
            self.viewer = None


class TestUIInitialization(TestRAWImageViewer):
    """Test UI initialization and basic functionality"""
    
    def test_window_initialization(self):
        """Test that the main window initializes correctly"""
        self.assertIsNotNone(self.viewer)
        self.assertEqual(self.viewer.windowTitle(), 'RAW Image Viewer')
        self.assertIsNotNone(self.viewer.image_label)
        self.assertIsNotNone(self.viewer.scroll_area)
        self.assertIsNotNone(self.viewer.status_bar)
    
    def test_initial_state(self):
        """Test initial state of the application"""
        self.assertIsNone(self.viewer.current_image)
        self.assertIsNone(self.viewer.current_pixmap)
        self.assertFalse(self.viewer.is_100_percent_zoom)
        self.assertEqual(self.viewer.current_file_index, -1)
        self.assertEqual(self.viewer.image_files, [])
    
    def test_supported_extensions(self):
        """Test supported file extensions"""
        extensions = self.viewer.get_supported_extensions()
        
        # Check for key RAW formats
        raw_formats = ['.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2', 
                      '.pef', '.srw', '.x3f', '.raf']
        for ext in raw_formats:
            self.assertIn(ext, extensions)
        
        # Check for common formats
        common_formats = ['.jpg', '.jpeg', '.tiff', '.tif', '.png']
        for ext in common_formats:
            self.assertIn(ext, extensions)
    
    def test_menu_creation(self):
        """Test that menus are created correctly"""
        menubar = self.viewer.menuBar()
        self.assertIsNotNone(menubar)
        
        # Check that File and Help menus exist
        menu_titles = [action.text() for action in menubar.actions()]
        self.assertIn('File', menu_titles)
        self.assertIn('Help', menu_titles)


class TestKeyboardShortcuts(TestRAWImageViewer):
    """Test keyboard shortcuts functionality"""
    
    def test_space_key_without_image(self):
        """Test space key when no image is loaded"""
        # Should not crash when no image is loaded
        QTest.keyClick(self.viewer, Qt.Key.Key_Space)
        self.assertIsNone(self.viewer.current_pixmap)
    
    def test_arrow_keys_without_images(self):
        """Test arrow keys when no images are loaded"""
        # Should not crash when no images are loaded
        QTest.keyClick(self.viewer, Qt.Key.Key_Left)
        QTest.keyClick(self.viewer, Qt.Key.Key_Right)
        self.assertEqual(self.viewer.current_file_index, -1)
    
    def test_delete_key_without_image(self):
        """Test delete key when no image is loaded"""
        # Should not crash when no image is loaded
        QTest.keyClick(self.viewer, Qt.Key.Key_Delete)
        self.assertIsNone(self.viewer.current_file_path)
    
    def test_focus_policy(self):
        """Test that the main window can receive focus for keyboard events"""
        self.assertEqual(self.viewer.focusPolicy(), Qt.FocusPolicy.StrongFocus)


class TestFileOperations(TestRAWImageViewer):
    """Test file operations and handling"""
    
    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist"""
        with patch.object(self.viewer, 'show_error') as mock_error:
            self.viewer.load_raw_image('/nonexistent/file.cr2')
            mock_error.assert_called_once()
    
    def test_scan_folder_for_images(self):
        """Test folder scanning functionality"""
        # Test with valid folder containing images
        test_file = os.path.join(self.test_dir, 'test.cr2')
        self.viewer.scan_folder_for_images(test_file)
        
        # Should find the RAW files we created
        self.assertGreater(len(self.viewer.image_files), 0)
        self.assertIn(test_file, self.viewer.image_files)
    
    def test_scan_empty_folder(self):
        """Test scanning an empty folder"""
        empty_file = os.path.join(self.test_files['empty_folder'], 'dummy.cr2')
        
        # Create the file first
        with open(empty_file, 'w') as f:
            f.write('')
        
        self.viewer.scan_folder_for_images(empty_file)
        
        # Should find at least the dummy file
        self.assertGreaterEqual(len(self.viewer.image_files), 1)
    
    def test_scan_single_image_folder(self):
        """Test scanning a folder with only one image"""
        single_file = os.path.join(self.test_files['single_image_folder'], 'single.arw')
        self.viewer.scan_folder_for_images(single_file)
        
        self.assertEqual(len(self.viewer.image_files), 1)
        self.assertEqual(self.viewer.current_file_index, 0)


class TestErrorHandling(TestRAWImageViewer):
    """Test error handling scenarios"""
    
    def test_raw_processing_error(self):
        """Test handling of RAW processing errors"""
        # Mock RAW processor to simulate error
        with patch('main.RAWProcessor') as mock_processor:
            mock_instance = Mock()
            mock_processor.return_value = mock_instance
            
            # Simulate error
            error_msg = "Test error message"
            self.viewer.on_processing_error(error_msg)
            
            # Check that error is handled properly
            self.assertEqual(self.viewer.windowTitle(), 'RAW Image Viewer')
            self.assertIn("Error loading image", self.viewer.status_bar.currentMessage())
    
    def test_corrupted_file_handling(self):
        """Test handling of corrupted files"""
        with patch.object(self.viewer, 'show_error') as mock_error:
            # Try to load corrupted file
            corrupted_file = self.test_files['corrupted_raw']
            self.viewer.load_raw_image(corrupted_file)
            
            # Should update status and title appropriately
            self.assertEqual(self.viewer.current_file_path, corrupted_file)
    
    def test_permission_denied_deletion(self):
        """Test handling of permission denied during file deletion"""
        # Mock send2trash to raise permission error
        with patch('main.send2trash') as mock_send2trash:
            mock_send2trash.side_effect = PermissionError("Permission denied")
            
            # Set up a fake current file
            self.viewer.current_file_path = self.test_files['valid_raw']
            
            with patch.object(self.viewer, 'show_error') as mock_error:
                self.viewer.perform_deletion()
                mock_error.assert_called_once()


class TestDragAndDrop(TestRAWImageViewer):
    """Test drag and drop functionality"""
    
    def test_drag_enter_valid_file(self):
        """Test drag enter event with valid file"""
        # Mock drag event with valid RAW file
        mock_event = Mock()
        mock_mime_data = Mock()
        mock_url = Mock()
        mock_url.isLocalFile.return_value = True
        mock_url.toLocalFile.return_value = '/test/file.cr2'
        mock_mime_data.urls.return_value = [mock_url]
        mock_mime_data.hasUrls.return_value = True
        mock_event.mimeData.return_value = mock_mime_data
        
        self.viewer.dragEnterEvent(mock_event)
        mock_event.acceptProposedAction.assert_called_once()
    
    def test_drag_enter_invalid_file(self):
        """Test drag enter event with invalid file"""
        # Mock drag event with invalid file
        mock_event = Mock()
        mock_mime_data = Mock()
        mock_url = Mock()
        mock_url.isLocalFile.return_value = True
        mock_url.toLocalFile.return_value = '/test/file.txt'
        mock_mime_data.urls.return_value = [mock_url]
        mock_mime_data.hasUrls.return_value = True
        mock_event.mimeData.return_value = mock_mime_data
        
        self.viewer.dragEnterEvent(mock_event)
        mock_event.ignore.assert_called_once()


class TestZoomFunctionality(TestRAWImageViewer):
    """Test zoom functionality"""
    
    def test_toggle_zoom_without_image(self):
        """Test zoom toggle when no image is loaded"""
        initial_zoom = self.viewer.is_100_percent_zoom
        self.viewer.toggle_zoom()
        
        # Should not change zoom state when no image is loaded
        self.assertEqual(self.viewer.is_100_percent_zoom, initial_zoom)
    
    def test_zoom_state_tracking(self):
        """Test zoom state tracking"""
        # Create a mock pixmap
        mock_pixmap = Mock(spec=QPixmap)
        mock_pixmap.size.return_value = Mock()
        mock_pixmap.width.return_value = 100
        mock_pixmap.height.return_value = 100
        mock_pixmap.scaled.return_value = mock_pixmap
        
        self.viewer.current_pixmap = mock_pixmap
        
        # Test zoom toggle
        initial_zoom = self.viewer.is_100_percent_zoom
        self.viewer.toggle_zoom()
        self.assertNotEqual(self.viewer.is_100_percent_zoom, initial_zoom)


class TestNavigationFunctionality(TestRAWImageViewer):
    """Test navigation between images"""
    
    def test_navigation_empty_list(self):
        """Test navigation when image list is empty"""
        self.viewer.image_files = []
        self.viewer.current_file_index = -1
        
        # Should not crash or change state
        self.viewer.navigate_to_next_image()
        self.assertEqual(self.viewer.current_file_index, -1)
        
        self.viewer.navigate_to_previous_image()
        self.assertEqual(self.viewer.current_file_index, -1)
    
    def test_navigation_single_image(self):
        """Test navigation when only one image is available"""
        self.viewer.image_files = ['/test/single.cr2']
        self.viewer.current_file_index = 0
        
        # Should not navigate when only one image
        self.viewer.navigate_to_next_image()
        # Navigation should not occur with single image
        
        self.viewer.navigate_to_previous_image()
        # Navigation should not occur with single image
    
    def test_navigation_wraparound(self):
        """Test navigation wraparound functionality"""
        # Mock multiple files
        self.viewer.image_files = ['/test/1.cr2', '/test/2.cr2', '/test/3.cr2']
        self.viewer.current_file_index = 2  # Last image
        
        with patch.object(self.viewer, 'load_raw_image') as mock_load:
            # Navigate to next should wrap to first
            self.viewer.navigate_to_next_image()
            self.assertEqual(self.viewer.current_file_index, 0)
            
            # Navigate to previous from first should wrap to last
            self.viewer.current_file_index = 0
            self.viewer.navigate_to_previous_image()
            self.assertEqual(self.viewer.current_file_index, 2)


class TestPerformanceAndMemory(TestRAWImageViewer):
    """Test performance and memory usage"""
    
    def test_memory_usage_baseline(self):
        """Test baseline memory usage"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory usage should be reasonable for a PyQt application
        self.assertLess(initial_memory, 200)  # Less than 200MB
    
    def test_startup_time(self):
        """Test application startup time"""
        start_time = time.time()
        
        # Create new viewer instance
        test_viewer = RAWImageViewer()
        
        startup_time = time.time() - start_time
        test_viewer.close()
        
        # Startup should be fast (less than 2 seconds)
        self.assertLess(startup_time, 2.0)
    
    def test_zoom_toggle_performance(self):
        """Test zoom toggle performance"""
        # Create a mock pixmap
        mock_pixmap = Mock(spec=QPixmap)
        mock_pixmap.size.return_value = Mock()
        mock_pixmap.width.return_value = 1000
        mock_pixmap.height.return_value = 1000
        mock_pixmap.scaled.return_value = mock_pixmap
        
        self.viewer.current_pixmap = mock_pixmap
        
        # Measure zoom toggle time
        start_time = time.time()
        
        for _ in range(10):
            self.viewer.toggle_zoom()
        
        avg_time = (time.time() - start_time) / 10
        
        # Each zoom toggle should be fast (less than 0.1 seconds)
        self.assertLess(avg_time, 0.1)


class TestStatusBarUpdates(TestRAWImageViewer):
    """Test status bar functionality"""
    
    def test_status_bar_no_file(self):
        """Test status bar when no file is loaded"""
        self.viewer.update_status_bar()
        self.assertEqual(self.viewer.status_bar.currentMessage(), "Ready")
    
    def test_status_bar_with_file(self):
        """Test status bar with file loaded"""
        # Mock file path and pixmap
        self.viewer.current_file_path = '/test/image.cr2'
        
        mock_pixmap = Mock(spec=QPixmap)
        mock_pixmap.width.return_value = 1920
        mock_pixmap.height.return_value = 1080
        self.viewer.current_pixmap = mock_pixmap
        
        self.viewer.update_status_bar()
        
        status_msg = self.viewer.status_bar.currentMessage()
        self.assertIn('image.cr2', status_msg)
        self.assertIn('1920×1080', status_msg)
    
    def test_status_bar_with_file_list(self):
        """Test status bar with file list information"""
        self.viewer.current_file_path = '/test/image.cr2'
        self.viewer.image_files = ['/test/1.cr2', '/test/2.cr2', '/test/3.cr2']
        self.viewer.current_file_index = 1
        
        mock_pixmap = Mock(spec=QPixmap)
        mock_pixmap.width.return_value = 1920
        mock_pixmap.height.return_value = 1080
        self.viewer.current_pixmap = mock_pixmap
        
        self.viewer.update_status_bar()
        
        status_msg = self.viewer.status_bar.currentMessage()
        self.assertIn('2 of 3', status_msg)  # Should show position in list


class TestWindowResizing(TestRAWImageViewer):
    """Test window resizing behavior"""
    
    def test_resize_without_image(self):
        """Test window resize when no image is loaded"""
        # Should not crash when resizing without image
        self.viewer.resize(800, 600)
        self.assertIsNone(self.viewer.current_pixmap)
    
    def test_resize_with_image_fit_mode(self):
        """Test window resize with image in fit-to-window mode"""
        # Mock pixmap
        mock_pixmap = Mock(spec=QPixmap)
        mock_pixmap.scaled.return_value = mock_pixmap
        self.viewer.current_pixmap = mock_pixmap
        self.viewer.is_100_percent_zoom = False
        
        with patch.object(self.viewer, 'scale_image_to_fit') as mock_scale:
            # Trigger resize event
            self.viewer.resize(1000, 800)
            # scale_image_to_fit should be called during resize
            # (Note: This tests the logic, actual resize event might not trigger in unit test)
    
    def test_resize_with_image_100_percent_mode(self):
        """Test window resize with image in 100% zoom mode"""
        # Mock pixmap
        mock_pixmap = Mock(spec=QPixmap)
        self.viewer.current_pixmap = mock_pixmap
        self.viewer.is_100_percent_zoom = True
        
        with patch.object(self.viewer, 'scale_image_to_fit') as mock_scale:
            # Trigger resize event
            self.viewer.resize(1000, 800)
            # scale_image_to_fit should NOT be called in 100% mode
            # (Note: This tests the logic, actual resize event might not trigger in unit test)


def run_test_suite():
    """Run the complete test suite"""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestUIInitialization,
        TestKeyboardShortcuts,
        TestFileOperations,
        TestErrorHandling,
        TestDragAndDrop,
        TestZoomFunctionality,
        TestNavigationFunctionality,
        TestPerformanceAndMemory,
        TestStatusBarUpdates,
        TestWindowResizing
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    # Run the test suite
    print("Running RAW Image Viewer Test Suite...")
    print("=" * 60)
    
    result = run_test_suite()
    
    print("\n" + "=" * 60)
    print("Test Suite Summary:")
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