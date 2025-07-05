import sys
import os
import unittest
import tempfile
import shutil
import time
import psutil
import threading
from unittest.mock import Mock, patch
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from main import RAWImageViewer, RAWProcessor
except ImportError as e:
    print(f"Error importing main module: {e}")
    sys.exit(1)


class PerformanceTest(unittest.TestCase):
    """Performance and memory usage testing"""
    
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
        
        # Create test files of various sizes
        cls.create_test_files()
        
        # Results storage
        cls.performance_results = {}
        
        # Get initial system state
        cls.process = psutil.Process()
        cls.initial_memory = cls.process.memory_info().rss / 1024 / 1024  # MB
        cls.initial_cpu_percent = cls.process.cpu_percent()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        # Remove temporary test directory
        shutil.rmtree(cls.test_dir, ignore_errors=True)
        
        # Generate performance report
        cls.generate_performance_report()
    
    @classmethod
    def create_test_files(cls):
        """Create test files of various sizes"""
        cls.test_files = {}
        
        # Small file (1MB)
        cls.test_files['small'] = os.path.join(cls.test_dir, 'small.cr2')
        cls.create_test_file(cls.test_files['small'], 1024 * 1024)
        
        # Medium file (10MB)
        cls.test_files['medium'] = os.path.join(cls.test_dir, 'medium.nef')
        cls.create_test_file(cls.test_files['medium'], 10 * 1024 * 1024)
        
        # Large file (50MB)
        cls.test_files['large'] = os.path.join(cls.test_dir, 'large.arw')
        cls.create_test_file(cls.test_files['large'], 50 * 1024 * 1024)
        
        # Extra large file (100MB)
        cls.test_files['extra_large'] = os.path.join(cls.test_dir, 'extra_large.dng')
        cls.create_test_file(cls.test_files['extra_large'], 100 * 1024 * 1024)
        
        # Multiple files for batch testing
        cls.test_files['batch'] = []
        for i in range(20):
            batch_file = os.path.join(cls.test_dir, f'batch_{i}.cr2')
            cls.create_test_file(batch_file, 5 * 1024 * 1024)  # 5MB each
            cls.test_files['batch'].append(batch_file)
    
    @classmethod
    def create_test_file(cls, file_path, size_bytes):
        """Create a test file of specified size"""
        with open(file_path, 'wb') as f:
            # Write in chunks to avoid memory issues
            chunk_size = 1024 * 1024  # 1MB chunks
            written = 0
            
            while written < size_bytes:
                chunk = min(chunk_size, size_bytes - written)
                f.write(b'0' * chunk)
                written += chunk
    
    def setUp(self):
        """Set up each test"""
        self.viewer = RAWImageViewer()
        
        # Record initial state
        self.start_memory = self.process.memory_info().rss / 1024 / 1024
        self.start_time = time.time()
    
    def tearDown(self):
        """Clean up after each test"""
        if hasattr(self, 'viewer'):
            self.viewer.close()
        
        # Force garbage collection
        import gc
        gc.collect()
    
    def measure_performance(self, operation_name, operation_func):
        """Measure performance of an operation"""
        # Record initial state
        start_memory = self.process.memory_info().rss / 1024 / 1024
        start_time = time.time()
        
        # Run operation
        try:
            result = operation_func()
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        # Record final state
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Calculate metrics
        execution_time = end_time - start_time
        memory_delta = end_memory - start_memory
        
        performance_data = {
            'operation': operation_name,
            'execution_time': execution_time,
            'memory_delta': memory_delta,
            'start_memory': start_memory,
            'end_memory': end_memory,
            'success': success,
            'error': error,
            'result': result
        }
        
        return performance_data
    
    def test_application_startup_time(self):
        """Test application startup performance"""
        def startup_operation():
            # Create new viewer instance
            test_viewer = RAWImageViewer()
            test_viewer.show()
            QApplication.processEvents()
            test_viewer.close()
            return True
        
        perf_data = self.measure_performance('startup', startup_operation)
        
        # Startup should be fast
        self.assertLess(perf_data['execution_time'], 5.0, "Startup time too slow")
        
        # Memory usage should be reasonable
        self.assertLess(perf_data['end_memory'], 200, "Initial memory usage too high")
        
        self.performance_results['startup'] = perf_data
    
    def test_file_scanning_performance(self):
        """Test folder scanning performance"""
        def scan_operation():
            self.viewer.scan_folder_for_images(self.test_files['small'])
            return len(self.viewer.image_files)
        
        perf_data = self.measure_performance('file_scanning', scan_operation)
        
        # Scanning should be fast
        self.assertLess(perf_data['execution_time'], 2.0, "File scanning too slow")
        
        self.performance_results['file_scanning'] = perf_data
    
    def test_zoom_toggle_performance(self):
        """Test zoom toggle performance"""
        # Set up mock pixmap
        mock_pixmap = Mock(spec=QPixmap)
        mock_pixmap.size.return_value = Mock()
        mock_pixmap.width.return_value = 4000
        mock_pixmap.height.return_value = 3000
        mock_pixmap.scaled.return_value = mock_pixmap
        
        self.viewer.current_pixmap = mock_pixmap
        
        def zoom_operation():
            # Perform multiple zoom toggles
            for _ in range(10):
                self.viewer.toggle_zoom()
                QApplication.processEvents()
            return True
        
        perf_data = self.measure_performance('zoom_toggle', zoom_operation)
        
        # Zoom operations should be fast
        avg_time_per_toggle = perf_data['execution_time'] / 10
        self.assertLess(avg_time_per_toggle, 0.1, "Zoom toggle too slow")
        
        self.performance_results['zoom_toggle'] = perf_data
    
    def test_navigation_performance(self):
        """Test navigation performance"""
        # Set up multiple files
        self.viewer.image_files = self.test_files['batch'][:10]  # First 10 files
        self.viewer.current_file_index = 0
        
        def navigation_operation():
            # Navigate through all files
            with patch.object(self.viewer, 'load_raw_image') as mock_load:
                for _ in range(20):  # Navigate more than available files to test wraparound
                    self.viewer.navigate_to_next_image()
                    QApplication.processEvents()
            return True
        
        perf_data = self.measure_performance('navigation', navigation_operation)
        
        # Navigation should be fast
        avg_time_per_nav = perf_data['execution_time'] / 20
        self.assertLess(avg_time_per_nav, 0.05, "Navigation too slow")
        
        self.performance_results['navigation'] = perf_data
    
    def test_memory_usage_with_large_files(self):
        """Test memory usage with large files"""
        def large_file_operation():
            # Process information about large files
            large_files = [
                self.test_files['large'],
                self.test_files['extra_large']
            ]
            
            for file_path in large_files:
                # Simulate file processing (without actual RAW processing)
                self.viewer.scan_folder_for_images(file_path)
                QApplication.processEvents()
            
            return len(large_files)
        
        perf_data = self.measure_performance('large_files', large_file_operation)
        
        # Memory increase should be reasonable
        self.assertLess(perf_data['memory_delta'], 100, "Memory usage too high with large files")
        
        self.performance_results['large_files'] = perf_data
    
    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated operations"""
        def memory_leak_operation():
            initial_memory = self.process.memory_info().rss / 1024 / 1024
            
            # Perform many operations that might leak memory
            for i in range(50):
                # Create and destroy viewers
                temp_viewer = RAWImageViewer()
                temp_viewer.scan_folder_for_images(self.test_files['small'])
                temp_viewer.close()
                
                # Process events
                QApplication.processEvents()
                
                # Force garbage collection every 10 iterations
                if i % 10 == 0:
                    import gc
                    gc.collect()
            
            final_memory = self.process.memory_info().rss / 1024 / 1024
            return final_memory - initial_memory
        
        perf_data = self.measure_performance('memory_leak', memory_leak_operation)
        
        # Memory growth should be minimal
        memory_growth = perf_data.get('result', 0)
        self.assertLess(memory_growth, 50, "Potential memory leak detected")
        
        self.performance_results['memory_leak'] = perf_data
    
    def test_cpu_usage_monitoring(self):
        """Test CPU usage during operations"""
        def cpu_intensive_operation():
            # Perform CPU-intensive operations
            cpu_percentages = []
            
            for _ in range(10):
                start_cpu = self.process.cpu_percent()
                
                # Simulate image processing work
                with patch.object(self.viewer, 'on_image_processed') as mock_processed:
                    # Create mock image data
                    mock_image = np.random.randint(0, 255, (2000, 3000, 3), dtype=np.uint8)
                    self.viewer.on_image_processed(mock_image)
                    QApplication.processEvents()
                
                time.sleep(0.1)  # Allow CPU measurement
                cpu_percentages.append(self.process.cpu_percent())
            
            return sum(cpu_percentages) / len(cpu_percentages)
        
        perf_data = self.measure_performance('cpu_usage', cpu_intensive_operation)
        
        # CPU usage should be reasonable
        avg_cpu = perf_data.get('result', 0)
        self.assertLess(avg_cpu, 80, "CPU usage too high")
        
        self.performance_results['cpu_usage'] = perf_data
    
    def test_ui_responsiveness(self):
        """Test UI responsiveness during operations"""
        def responsiveness_operation():
            response_times = []
            
            # Test UI responsiveness while performing operations
            for _ in range(10):
                start_time = time.time()
                
                # Perform UI operation
                self.viewer.update_status_bar()
                QApplication.processEvents()
                
                response_time = time.time() - start_time
                response_times.append(response_time)
            
            return sum(response_times) / len(response_times)
        
        perf_data = self.measure_performance('ui_responsiveness', responsiveness_operation)
        
        # UI should be responsive
        avg_response_time = perf_data.get('result', 0)
        self.assertLess(avg_response_time, 0.01, "UI not responsive enough")
        
        self.performance_results['ui_responsiveness'] = perf_data
    
    def test_concurrent_operations(self):
        """Test performance with concurrent operations"""
        def concurrent_operation():
            # Simulate multiple operations happening concurrently
            operations_completed = 0
            
            def background_operation():
                nonlocal operations_completed
                for _ in range(10):
                    self.viewer.update_status_bar()
                    time.sleep(0.01)
                    operations_completed += 1
            
            # Start background thread
            thread = threading.Thread(target=background_operation)
            thread.start()
            
            # Perform foreground operations
            for _ in range(10):
                self.viewer.scan_folder_for_images(self.test_files['small'])
                QApplication.processEvents()
            
            thread.join()
            return operations_completed
        
        perf_data = self.measure_performance('concurrent_operations', concurrent_operation)
        
        # Operations should complete successfully
        self.assertTrue(perf_data['success'], "Concurrent operations failed")
        
        self.performance_results['concurrent_operations'] = perf_data
    
    def test_resource_cleanup(self):
        """Test proper resource cleanup"""
        def cleanup_operation():
            initial_handles = len(self.process.open_files())
            
            # Perform operations that create resources
            for _ in range(10):
                temp_viewer = RAWImageViewer()
                temp_viewer.scan_folder_for_images(self.test_files['small'])
                temp_viewer.close()
                
                # Force cleanup
                QApplication.processEvents()
            
            import gc
            gc.collect()
            
            final_handles = len(self.process.open_files())
            return final_handles - initial_handles
        
        perf_data = self.measure_performance('resource_cleanup', cleanup_operation)
        
        # Resource usage should not grow significantly
        handle_growth = perf_data.get('result', 0)
        self.assertLess(handle_growth, 20, "Resource cleanup not working properly")
        
        self.performance_results['resource_cleanup'] = perf_data
    
    def test_batch_processing_performance(self):
        """Test performance when processing multiple files"""
        def batch_operation():
            # Process multiple files in sequence
            for file_path in self.test_files['batch'][:5]:  # Process first 5 files
                self.viewer.scan_folder_for_images(file_path)
                QApplication.processEvents()
            
            return len(self.test_files['batch'][:5])
        
        perf_data = self.measure_performance('batch_processing', batch_operation)
        
        # Batch processing should be reasonable
        files_processed = perf_data.get('result', 0)
        if files_processed > 0:
            time_per_file = perf_data['execution_time'] / files_processed
            self.assertLess(time_per_file, 1.0, "Batch processing too slow")
        
        self.performance_results['batch_processing'] = perf_data
    
    @classmethod
    def generate_performance_report(cls):
        """Generate performance test report"""
        report = []
        report.append("=" * 80)
        report.append("PERFORMANCE TEST REPORT")
        report.append("=" * 80)
        report.append("")
        
        # System information
        report.append("SYSTEM INFORMATION:")
        report.append("-" * 40)
        report.append(f"CPU Count: {psutil.cpu_count()}")
        report.append(f"Total Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
        report.append(f"Available Memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
        report.append(f"Initial Memory Usage: {cls.initial_memory:.2f} MB")
        report.append("")
        
        # Performance results
        report.append("PERFORMANCE RESULTS:")
        report.append("-" * 40)
        
        for test_name, result in cls.performance_results.items():
            status = "✓" if result['success'] else "✗"
            report.append(f"{test_name.replace('_', ' ').title():<25} {status}")
            report.append(f"  Execution Time: {result['execution_time']:.3f} seconds")
            report.append(f"  Memory Delta:   {result['memory_delta']:+.2f} MB")
            report.append(f"  Start Memory:   {result['start_memory']:.2f} MB")
            report.append(f"  End Memory:     {result['end_memory']:.2f} MB")
            
            if result['error']:
                report.append(f"  Error:          {result['error']}")
            
            report.append("")
        
        # Performance benchmarks
        report.append("PERFORMANCE BENCHMARKS:")
        report.append("-" * 40)
        
        benchmarks = [
            ("Startup Time", "< 5.0 seconds"),
            ("File Scanning", "< 2.0 seconds"),
            ("Zoom Toggle", "< 0.1 seconds per toggle"),
            ("Navigation", "< 0.05 seconds per navigation"),
            ("Memory Usage", "< 200 MB initial, < 100 MB growth"),
            ("CPU Usage", "< 80% average"),
            ("UI Responsiveness", "< 0.01 seconds response time"),
        ]
        
        for benchmark, target in benchmarks:
            report.append(f"  {benchmark:<20} {target}")
        
        report.append("")
        
        # Performance summary
        total_tests = len(cls.performance_results)
        passed_tests = sum(1 for result in cls.performance_results.values() if result['success'])
        
        report.append("PERFORMANCE SUMMARY:")
        report.append("-" * 40)
        report.append(f"Total tests: {total_tests}")
        report.append(f"Passed tests: {passed_tests}")
        report.append(f"Success rate: {(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "No tests run")
        
        # Find slowest operations
        if cls.performance_results:
            slowest_op = max(cls.performance_results.items(), 
                           key=lambda x: x[1]['execution_time'])
            highest_memory = max(cls.performance_results.items(), 
                               key=lambda x: x[1]['memory_delta'])
            
            report.append(f"Slowest operation: {slowest_op[0]} ({slowest_op[1]['execution_time']:.3f}s)")
            report.append(f"Highest memory usage: {highest_memory[0]} ({highest_memory[1]['memory_delta']:+.2f} MB)")
        
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 40)
        
        slow_operations = [name for name, result in cls.performance_results.items() 
                          if result['execution_time'] > 2.0]
        
        if slow_operations:
            report.append("Consider optimizing these slow operations:")
            for op in slow_operations:
                report.append(f"  - {op}")
        
        high_memory_ops = [name for name, result in cls.performance_results.items() 
                          if result['memory_delta'] > 50]
        
        if high_memory_ops:
            report.append("Consider optimizing memory usage for:")
            for op in high_memory_ops:
                report.append(f"  - {op}")
        
        report.append("")
        report.append("General recommendations:")
        report.append("- Monitor memory usage with very large RAW files")
        report.append("- Consider implementing progressive loading for large files")
        report.append("- Use image caching for frequently accessed files")
        report.append("- Implement background processing for CPU-intensive operations")
        report.append("- Regular profiling with real-world usage patterns")
        
        # Save report
        report_path = os.path.join(os.path.dirname(cls.test_dir), 'performance_report.txt')
        try:
            with open(report_path, 'w') as f:
                f.write('\n'.join(report))
            print(f"\nPerformance report saved to: {report_path}")
        except Exception as e:
            print(f"Could not save report: {e}")
        
        # Print summary
        print("\n" + "\n".join(report))


def run_performance_tests():
    """Run the performance tests"""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test methods
    tests = unittest.TestLoader().loadTestsFromTestCase(PerformanceTest)
    suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("Running Performance Tests...")
    print("=" * 60)
    
    result = run_performance_tests()
    
    print("\n" + "=" * 60)
    print("Performance Test Summary:")
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