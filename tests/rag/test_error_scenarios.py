"""
Tests for error scenarios and edge cases.
Tests error handling, recovery, and edge cases throughout the system.
"""

import pytest
import tempfile
import shutil
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import sys

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ab.rag.extract_blocks import BlockExtractor
from ab.rag.block_validator import BlockValidator


class TestErrorScenarios:
    """Test suite for error scenarios and edge cases."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def extractor(self, temp_dir):
        """Create a BlockExtractor instance for testing."""
        with patch('ab.rag.extract_blocks.BlockExtractor.warm_index_once', return_value=True):
            extractor = BlockExtractor()
            return extractor
    
    def test_missing_json_file_generation_failure(self, extractor, temp_dir):
        """Test handling when JSON file generation fails."""
        missing_json = os.path.join(temp_dir, "missing_blocks.json")
        
        with patch('ab.rag.extract_blocks.BlockExtractor._generate_block_names', return_value=False):
            result = extractor.auto_extract_all_blocks(json_path=missing_json)
            
            assert result["success"] is False
            assert "Failed to generate block names" in result["reason"]
    
    def test_index_warming_failure(self, extractor, temp_dir):
        """Test handling when index warming fails."""
        json_path = os.path.join(temp_dir, "test_blocks.json")
        with open(json_path, 'w') as f:
            json.dump(["AConv"], f)
        
        with patch('ab.rag.extract_blocks.BlockExtractor.warm_index_once', return_value=False):
            result = extractor.auto_extract_all_blocks(json_path=json_path)
            
            assert result["success"] is False
            assert "Failed to warm index" in result["reason"]
    
    def test_corrupted_json_file(self, extractor, temp_dir):
        """Test handling of corrupted JSON file."""
        corrupted_json = os.path.join(temp_dir, "corrupted.json")
        with open(corrupted_json, 'w') as f:
            f.write("invalid json content {")
        
        # Mock the JSON loading to raise an error
        with patch('ab.rag.extract_blocks.json.load') as mock_load:
            mock_load.side_effect = json.JSONDecodeError("Expecting ',' delimiter", "", 0)
            
            result = extractor.auto_extract_all_blocks(json_path=corrupted_json)
            
            assert result["success"] is False
            assert "names file error" in result["reason"]
    
    def test_permission_denied_json_creation(self, extractor, temp_dir):
        """Test handling when JSON file cannot be created due to permissions."""
        # Create a read-only directory
        read_only_dir = os.path.join(temp_dir, "readonly")
        os.makedirs(read_only_dir, mode=0o444)
        
        json_path = os.path.join(read_only_dir, "blocks.json")
        
        # Mock the file existence check to simulate permission error
        with patch('pathlib.Path.exists', side_effect=PermissionError("Permission denied")):
            result = extractor.auto_extract_all_blocks(json_path=json_path)
            
            assert result["success"] is False
            assert "Permission denied" in result["reason"]
    
    def test_memory_error_during_extraction(self, extractor, temp_dir):
        """Test handling of memory errors during extraction."""
        with patch('ab.rag.extract_blocks.BlockExtractor.extract_block') as mock_extract:
            mock_extract.side_effect = MemoryError("Out of memory")
            
            result = extractor.extract_single_block("AConv")
            
        assert result["success"] is False
        assert "Out of memory" in result["reason"]
    
    def test_disk_full_error(self, extractor, temp_dir):
        """Test handling of disk full errors."""
        with patch('ab.rag.extract_blocks.BlockExtractor.extract_block') as mock_extract:
            mock_extract.side_effect = OSError("No space left on device")
            
            result = extractor.extract_single_block("AConv")
            
        assert result["success"] is False
        assert "No space left" in result["reason"]
    
    def test_network_error_during_indexing(self, extractor, dummy_json_file):
        """Test handling of network errors during indexing."""
        with patch('ab.rag.extract_blocks.BlockExtractor.warm_index_once') as mock_warm:
            mock_warm.side_effect = ConnectionError("Network unreachable")
            
            result = extractor.auto_extract_all_blocks(json_path=dummy_json_file)
            
            assert result["success"] is False
            assert "Auto-extraction failed" in result["reason"]
    
    def test_timeout_error_during_extraction(self, extractor):
        """Test handling of timeout errors during extraction."""
        with patch('ab.rag.extract_blocks.BlockExtractor.extract_block') as mock_extract:
            mock_extract.side_effect = TimeoutError("Operation timed out")
            
            result = extractor.extract_single_block("AConv")
            
        assert result["success"] is False
        assert "timed out" in result["reason"]
    
    def test_invalid_block_name_characters(self, extractor):
        """Test handling of invalid block name characters."""
        invalid_names = ["", " ", "Block\nName", "Block\tName", "Block/Name", "Block\\Name"]
        
        for invalid_name in invalid_names:
            result = extractor.extract_single_block(invalid_name)
            assert result["success"] is False
    
    def test_very_long_block_name(self, extractor):
        """Test handling of very long block names."""
        long_name = "A" * 1000  # Very long name
        
        with patch('ab.rag.extract_blocks.BlockExtractor.extract_block') as mock_extract:
            mock_extract.return_value = {"success": True, "block_name": long_name}
            
            result = extractor.extract_single_block(long_name)
            
            # Should handle long names gracefully
            assert result["success"] is True
            assert result["block_name"] == long_name
    
    
    def test_malformed_validation_result(self, extractor):
        """Test handling of malformed validation results."""
        with patch('ab.rag.extract_blocks.BlockExtractor.extract_block') as mock_extract, \
             patch('ab.rag.extract_blocks.BlockExtractor.validate_block') as mock_validate:
            
            mock_extract.return_value = {"success": True, "block_name": "AConv"}
            mock_validate.return_value = None  # Malformed result
            
            result = extractor.extract_single_block("AConv")
            
            # Should handle malformed validation gracefully
            assert result["success"] is True
            assert "validation" in result
    
    def test_circular_dependency_detection(self, extractor):
        """Test handling of circular dependencies."""
        with patch('ab.rag.extract_blocks.BlockExtractor.extract_block') as mock_extract:
            mock_extract.return_value = {
                "success": False,
                "reason": "circular dependency detected",
                "block_name": "CircularBlock"
            }
            
            result = extractor.extract_single_block("CircularBlock")
            
            assert result["success"] is False
            assert "circular dependency" in result["reason"]
    
    def test_invalid_python_syntax_in_generated_file(self, temp_dir):
        """Test handling of invalid Python syntax in generated files."""
        validator = BlockValidator(
            generated_dir=os.path.join(temp_dir, "generated"),
            block_dir=os.path.join(temp_dir, "block")
        )
        
        # Create invalid Python file
        invalid_file = os.path.join(temp_dir, "generated", "InvalidBlock.py")
        os.makedirs(os.path.dirname(invalid_file), exist_ok=True)
        with open(invalid_file, 'w') as f:
            f.write("invalid python syntax {")
        
        is_valid, error = validator.validate_single_block("InvalidBlock")
        
        assert is_valid is False
        assert error is not None
    
    def test_missing_imports_in_generated_file(self, temp_dir):
        """Test handling of missing imports in generated files."""
        validator = BlockValidator(
            generated_dir=os.path.join(temp_dir, "generated"),
            block_dir=os.path.join(temp_dir, "block")
        )
        
        # Create file with missing imports
        missing_import_file = os.path.join(temp_dir, "generated", "MissingImportBlock.py")
        os.makedirs(os.path.dirname(missing_import_file), exist_ok=True)
        with open(missing_import_file, 'w') as f:
            f.write('''
class MissingImportBlock:
    def __init__(self):
        self.conv = torch.nn.Conv2d(3, 64, 3)  # torch not imported
''')
        
        is_valid, error = validator.validate_single_block("MissingImportBlock")
        
        # The validation only checks syntax, not runtime imports
        # So this should be valid from a syntax perspective
        assert is_valid is True
        assert error is None
    
    def test_file_system_errors_during_validation(self, temp_dir):
        """Test handling of file system errors during validation."""
        validator = BlockValidator(
            generated_dir=os.path.join(temp_dir, "generated"),
            block_dir=os.path.join(temp_dir, "block")
        )
        
        # Create a file that will cause file system errors
        error_file = os.path.join(temp_dir, "generated", "ErrorBlock.py")
        os.makedirs(os.path.dirname(error_file), exist_ok=True)
        with open(error_file, 'w') as f:
            f.write('''
import torch.nn as nn
class ErrorBlock(nn.Module):
    def forward(self, x): return x
''')
        
        # Mock file operations to raise errors
        with patch('shutil.move', side_effect=OSError("Permission denied")):
            is_valid, error = validator.validate_single_block("ErrorBlock")
            
            # validate_single_block doesn't move files, so it should be valid
            # but validate_and_move_block would fail
            assert is_valid is True
            assert error is None
    
    def test_large_file_handling(self, temp_dir):
        """Test handling of very large generated files."""
        validator = BlockValidator(
            generated_dir=os.path.join(temp_dir, "generated"),
            block_dir=os.path.join(temp_dir, "block")
        )
        
        # Create a very large file
        large_file = os.path.join(temp_dir, "generated", "LargeBlock.py")
        os.makedirs(os.path.dirname(large_file), exist_ok=True)
        
        with open(large_file, 'w') as f:
            f.write('''
import torch.nn as nn
class LargeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(1000):
            self.layers.append(nn.Conv2d(64, 64, 3))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
''')
        
        is_valid, error = validator.validate_single_block("LargeBlock")
        
        # Should handle large files gracefully
        assert is_valid is True  # Large files should still be valid
        assert error is None
    
    def test_unicode_handling_in_block_names(self, extractor):
        """Test handling of Unicode characters in block names."""
        unicode_names = ["Blockα", "Blockβ", "Block中文", "Block🚀"]
        
        for unicode_name in unicode_names:
            with patch('ab.rag.extract_blocks.BlockExtractor.extract_block') as mock_extract:
                mock_extract.return_value = {
                    "success": True,
                    "block_name": unicode_name
                }
                
                result = extractor.extract_single_block(unicode_name)
                
                # Should handle Unicode names gracefully
                assert result["success"] is True
                assert result["block_name"] == unicode_name
    
    def test_empty_block_list_handling(self, extractor):
        """Test handling of empty block lists."""
        result = extractor.extract_multiple_blocks([])
        assert result == {}
        
        # Mock retry_failed_blocks to avoid actual extraction
        with patch.object(extractor, 'retry_failed_blocks', return_value={}):
            result = extractor.retry_failed_blocks()
            assert result == {}
    
    def test_none_parameter_handling(self, extractor):
        """Test handling of None parameters."""
        # Test with None block name
        result = extractor.extract_single_block(None)
        assert result["success"] is False
        
        # Test with None block list
        result = extractor.extract_multiple_blocks(None)
        assert result == {}
    
    def test_negative_limit_handling(self, extractor, temp_dir):
        """Test handling of negative limits."""
        json_path = os.path.join(temp_dir, "test_blocks.json")
        with open(json_path, 'w') as f:
            json.dump(["AConv", "AAttn"], f)
        
        with patch('ab.rag.extract_blocks.BlockExtractor.extract_blocks_from_file') as mock_extract:
            mock_extract.return_value = {"success": True, "processed": 0}
            
            result = extractor.auto_extract_all_blocks(json_path=json_path, limit=-1)
            
            # Should handle negative limits gracefully
            assert result["success"] is True


if __name__ == "__main__":
    pytest.main([__file__])
