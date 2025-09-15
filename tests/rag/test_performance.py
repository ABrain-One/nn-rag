"""
Performance tests for the RAG system.
Tests performance characteristics, scalability, and resource usage.
"""

import pytest
import tempfile
import shutil
import json
import os
import time
import psutil
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import threading
import concurrent.futures

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ab.rag.extract_blocks import BlockExtractor
from ab.rag.block_validator import BlockValidator


class TestPerformance:
    """Performance test suite for the RAG system."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def performance_environment(self, temp_dir):
        """Set up a performance test environment."""
        generated_dir = os.path.join(temp_dir, "generated_packages")
        block_dir = os.path.join(temp_dir, "block")
        config_dir = os.path.join(temp_dir, "config")
        
        os.makedirs(generated_dir, exist_ok=True)
        os.makedirs(block_dir, exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)
        
        return {
            "temp_dir": temp_dir,
            "generated_dir": generated_dir,
            "block_dir": block_dir,
            "config_dir": config_dir
        }
    
    def test_single_block_extraction_performance(self, performance_environment):
        """Test performance of single block extraction."""
        env = performance_environment
        
        with patch('ab.rag.extract_blocks.BlockExtractor.warm_index_once', return_value=True), \
             patch('ab.rag.extract_blocks.BlockExtractor.extract_block') as mock_extract, \
             patch('ab.rag.extract_blocks.BlockExtractor.validate_block') as mock_validate:
            
            # Mock fast extraction
            mock_extract.return_value = {"success": True, "block_name": "TestBlock"}
            mock_validate.return_value = {"status": "valid", "moved_to_block_dir": True}
            
            extractor = BlockExtractor()
            
            # Measure extraction time
            start_time = time.time()
            result = extractor.extract_single_block("TestBlock")
            end_time = time.time()
            
            extraction_time = end_time - start_time
            
            # Verify performance
            assert result["success"] is True
            assert extraction_time < 1.0  # Should complete within 1 second
            print(f"Single block extraction time: {extraction_time:.3f}s")
    
    def test_batch_extraction_performance(self, performance_environment):
        """Test performance of batch extraction."""
        env = performance_environment
        
        # Create test JSON with many blocks
        json_path = os.path.join(env["config_dir"], "performance_blocks.json")
        test_blocks = [f"PerformanceBlock{i}" for i in range(100)]
        with open(json_path, 'w') as f:
            json.dump(test_blocks, f)
        
        with patch('ab.rag.extract_blocks.BlockExtractor.warm_index_once', return_value=True), \
             patch('ab.rag.extract_blocks.BlockExtractor.extract_blocks_from_file') as mock_extract:
            
            # Mock batch extraction
            mock_extract.return_value = {
                "success": True,
                "processed": 100,
                "ok": 95,
                "fail": 5
            }
            
            extractor = BlockExtractor()
            
            # Measure batch extraction time
            start_time = time.time()
            result = extractor.auto_extract_all_blocks(json_path=json_path)
            end_time = time.time()
            
            batch_time = end_time - start_time
            
            # Verify performance
            assert result["success"] is True
            assert batch_time < 5.0  # Should complete within 5 seconds
            print(f"Batch extraction (100 blocks) time: {batch_time:.3f}s")
    
    def test_validation_performance(self, performance_environment):
        """Test performance of block validation."""
        env = performance_environment
        
        # Create many test files
        block_count = 50
        for i in range(block_count):
            file_path = os.path.join(env["generated_dir"], f"ValidationBlock{i}.py")
            with open(file_path, 'w') as f:
                f.write(f'''
import torch.nn as nn
class ValidationBlock{i}(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
    
    def forward(self, x):
        return self.conv(x)
''')
        
        validator = BlockValidator(
            generated_dir=env["generated_dir"],
            block_dir=env["block_dir"]
        )
        
        # Measure validation time
        start_time = time.time()
        block_names = [f"ValidationBlock{i}" for i in range(block_count)]
        results = validator.validate_all_blocks()
        end_time = time.time()
        
        validation_time = end_time - start_time
        
        # Verify performance
        assert len(results) == block_count
        assert all(result[0] is True for result in results.values())  # is_valid
        assert validation_time < 10.0  # Should complete within 10 seconds
        print(f"Validation (50 blocks) time: {validation_time:.3f}s")
    
    def test_memory_usage_during_extraction(self, performance_environment):
        """Test memory usage during extraction."""
        env = performance_environment
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with patch('ab.rag.extract_blocks.BlockExtractor.warm_index_once', return_value=True), \
             patch('ab.rag.extract_blocks.BlockExtractor.extract_blocks_from_file') as mock_extract:
            
            # Mock extraction with large results
            mock_extract.return_value = {
                "success": True,
                "processed": 1000,
                "ok": 950,
                "fail": 50,
                "results": {f"Block{i}": {"success": True} for i in range(1000)}
            }
            
            extractor = BlockExtractor()
            
            # Run extraction
            result = extractor.auto_extract_all_blocks()
            
            # Get peak memory usage
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            
            # Verify memory usage
            assert result["success"] is True
            # Use environment variable or 25% of total system memory as threshold
            max_memory_increase_mb = float(os.environ.get("MAX_MEMORY_INCREASE_MB", psutil.virtual_memory().total / 1024 / 1024 * 0.25))
            assert memory_increase < max_memory_increase_mb, f"Memory increase ({memory_increase:.1f}MB) exceeded threshold ({max_memory_increase_mb:.1f}MB)"
            print(f"Memory increase during extraction: {memory_increase:.1f}MB (threshold: {max_memory_increase_mb:.1f}MB)")
    
    def test_concurrent_validation_performance(self, performance_environment):
        """Test performance of concurrent validation."""
        env = performance_environment
        
        # Create test files
        block_count = 20
        for i in range(block_count):
            file_path = os.path.join(env["generated_dir"], f"ConcurrentBlock{i}.py")
            with open(file_path, 'w') as f:
                f.write(f'''
import torch.nn as nn
class ConcurrentBlock{i}(nn.Module):
    def forward(self, x): return x
''')
        
        validator = BlockValidator(
            generated_dir=env["generated_dir"],
            block_dir=env["block_dir"]
        )
        
        # Test sequential validation
        start_time = time.time()
        block_names = [f"ConcurrentBlock{i}" for i in range(block_count)]
        sequential_results = {}
        for name in block_names:
            sequential_results[name] = validator.validate_single_block(name)
        sequential_time = time.time() - start_time
        
        # Reset for concurrent test
        shutil.rmtree(env["block_dir"])
        os.makedirs(env["block_dir"])
        
        # Test concurrent validation
        start_time = time.time()
        concurrent_results = validator.validate_all_blocks()
        concurrent_time = time.time() - start_time
        
        # Verify results
        assert len(sequential_results) == len(concurrent_results)
        assert all(result[0] is True for result in concurrent_results.values())  # is_valid
        
        # Concurrent should be faster (or at least not significantly slower)
        print(f"Sequential validation time: {sequential_time:.3f}s")
        print(f"Concurrent validation time: {concurrent_time:.3f}s")
        print(f"Speedup: {sequential_time / concurrent_time:.2f}x")
    
    def test_large_file_handling_performance(self, performance_environment):
        """Test performance with large files."""
        env = performance_environment
        
        # Create large files
        large_content = "x" * 100000  # 100KB content
        file_path = os.path.join(env["generated_dir"], "LargeFileBlock.py")
        with open(file_path, 'w') as f:
            f.write(f'''
import torch.nn as nn
class LargeFileBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.data = "{large_content}"
    
    def forward(self, x):
        return x
''')
        
        validator = BlockValidator(
            generated_dir=env["generated_dir"],
            block_dir=env["block_dir"]
        )
        
        # Measure validation time for large file
        start_time = time.time()
        result = validator.validate_single_block("LargeFileBlock")
        end_time = time.time()
        
        validation_time = end_time - start_time
        
        # Verify performance
        assert result[0] is True  # is_valid
        assert validation_time < 2.0  # Should complete within 2 seconds
        print(f"Large file validation time: {validation_time:.3f}s")
    
    def test_scalability_with_block_count(self, performance_environment):
        """Test scalability with increasing block count."""
        env = performance_environment
        
        block_counts = [10, 50, 100, 200]
        times = []
        
        for count in block_counts:
            # Create test files
            for i in range(count):
                file_path = os.path.join(env["generated_dir"], f"ScalabilityBlock{i}.py")
                with open(file_path, 'w') as f:
                    f.write(f'''
import torch.nn as nn
class ScalabilityBlock{i}(nn.Module):
    def forward(self, x): return x
''')
            
            validator = BlockValidator(
                generated_dir=env["generated_dir"],
                block_dir=env["block_dir"]
            )
            
            # Measure validation time
            start_time = time.time()
            block_names = [f"ScalabilityBlock{i}" for i in range(count)]
            results = validator.validate_all_blocks()
            end_time = time.time()
            
            validation_time = end_time - start_time
            times.append(validation_time)
            
            # Verify results
            assert len(results) == count
            assert all(result[0] is True for result in results.values())  # is_valid
            
            # Clean up for next iteration
            shutil.rmtree(env["block_dir"])
            os.makedirs(env["block_dir"])
        
        # Verify scalability (should be roughly linear)
        print("Scalability test results:")
        for count, time_taken in zip(block_counts, times):
            print(f"  {count} blocks: {time_taken:.3f}s")
        
        # Check that time increases roughly linearly
        if len(times) >= 2:
            ratio = times[-1] / times[0]
            block_ratio = block_counts[-1] / block_counts[0]
            # Time should not increase more than 2x the block count ratio
            assert ratio < 2 * block_ratio
    
    def test_cpu_usage_during_processing(self, performance_environment):
        """Test CPU usage during processing."""
        env = performance_environment
        
        # Create test files
        block_count = 30
        for i in range(block_count):
            file_path = os.path.join(env["generated_dir"], f"CPUBlock{i}.py")
            with open(file_path, 'w') as f:
                f.write(f'''
import torch.nn as nn
class CPUBlock{i}(nn.Module):
    def forward(self, x): return x
''')
        
        validator = BlockValidator(
            generated_dir=env["generated_dir"],
            block_dir=env["block_dir"]
        )
        
        # Monitor CPU usage during processing
        cpu_usage = []
        block_names = [f"CPUBlock{i}" for i in range(block_count)]
        
        def monitor_cpu():
            while True:
                cpu_percent = psutil.cpu_percent()
                cpu_usage.append(cpu_percent)
                time.sleep(0.1)
        
        # Start CPU monitoring
        monitor_thread = threading.Thread(target=monitor_cpu, daemon=True)
        monitor_thread.start()
        
        # Run validation
        start_time = time.time()
        results = validator.validate_all_blocks()
        end_time = time.time()
        
        # Stop monitoring
        time.sleep(0.5)  # Let monitoring finish
        
        # Analyze CPU usage
        avg_cpu = sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0
        max_cpu = max(cpu_usage) if cpu_usage else 0
        
        # Verify results
        assert len(results) == block_count
        assert all(result[0] is True for result in results.values())  # is_valid
        
        print(f"Average CPU usage: {avg_cpu:.1f}%")
        print(f"Peak CPU usage: {max_cpu:.1f}%")
        
        # CPU usage should be reasonable (not 100%)
        assert max_cpu < 90  # Should not max out CPU
    
    def test_disk_io_performance(self, performance_environment):
        """Test disk I/O performance during file operations."""
        env = performance_environment
        
        # Create many files to test disk I/O
        block_count = 100
        for i in range(block_count):
            file_path = os.path.join(env["generated_dir"], f"DiskIOBlock{i}.py")
            with open(file_path, 'w') as f:
                f.write(f'''
import torch.nn as nn
class DiskIOBlock{i}(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
    
    def forward(self, x):
        return self.conv(x)
''')
        
        validator = BlockValidator(
            generated_dir=env["generated_dir"],
            block_dir=env["block_dir"]
        )
        
        # Measure disk I/O time
        start_time = time.time()
        block_names = [f"DiskIOBlock{i}" for i in range(block_count)]
        results = validator.validate_all_blocks()
        end_time = time.time()
        
        io_time = end_time - start_time
        
        # Verify results
        assert len(results) == block_count
        assert all(result[0] is True for result in results.values())  # is_valid
        
        # Calculate I/O rate
        files_per_second = block_count / io_time
        print(f"Disk I/O rate: {files_per_second:.1f} files/second")
        
        # Should process at least 10 files per second
        assert files_per_second > 10
    
    def test_memory_leak_detection(self, performance_environment):
        """Test for memory leaks during repeated operations."""
        env = performance_environment
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple iterations
        iterations = 10
        for iteration in range(iterations):
            # Create test files
            for i in range(10):
                file_path = os.path.join(env["generated_dir"], f"LeakTestBlock{i}.py")
                with open(file_path, 'w') as f:
                    f.write(f'''
import torch.nn as nn
class LeakTestBlock{i}(nn.Module):
    def forward(self, x): return x
''')
            
            validator = BlockValidator(
                generated_dir=env["generated_dir"],
                block_dir=env["block_dir"]
            )
            
            # Run validation
            block_names = [f"LeakTestBlock{i}" for i in range(10)]
            results = validator.validate_all_blocks()
            
            # Clean up
            shutil.rmtree(env["block_dir"])
            os.makedirs(env["block_dir"])
            
            # Check memory every few iterations
            if iteration % 3 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory
                print(f"Iteration {iteration}: Memory increase: {memory_increase:.1f}MB")
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_memory_increase = final_memory - initial_memory
        
        print(f"Total memory increase after {iterations} iterations: {total_memory_increase:.1f}MB")
        
        # Memory increase should be reasonable (less than 100MB)
        assert total_memory_increase < 100


if __name__ == "__main__":
    pytest.main([__file__])
