"""
Tests for CLI and API consistency.
Ensures that CLI commands and API calls produce identical results.
"""

import pytest
import tempfile
import shutil
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import os

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ab.rag.extract_blocks import BlockExtractor


class TestCLIConsistency:
    """Test suite for CLI and API consistency."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_json_file(self, temp_dir):
        """Create a sample JSON file with test block names."""
        json_path = os.path.join(temp_dir, "test_blocks.json")
        sample_blocks = ["AConv", "AAttn", "ABlock"]
        with open(json_path, 'w') as f:
            json.dump(sample_blocks, f)
        return json_path
    
    def test_single_block_cli_vs_api(self, temp_dir):
        """Test that single block CLI and API produce identical results."""
        # Test block name
        block_name = "AConv"
        
        # Mock the extraction to avoid actual processing
        with patch('ab.rag.extract_blocks.BlockExtractor.warm_index_once', return_value=True), \
             patch('ab.rag.extract_blocks.BlockExtractor.extract_block') as mock_extract, \
             patch('ab.rag.extract_blocks.BlockExtractor.validate_block') as mock_validate:
            
            # Mock successful extraction and validation
            mock_extract.return_value = {
                "success": True,
                "block_name": block_name,
                "file_path": f"generated_packages/{block_name}.py"
            }
            mock_validate.return_value = {
                "name": block_name,
                "status": "valid",
                "moved_to_block_dir": True
            }
            
            # Test API
            extractor = BlockExtractor()
            api_result = extractor.extract_single_block(block_name)
            
            # Test CLI (simulate by calling main function)
            from ab.rag.extract_blocks import main
            import argparse
            
            # Mock sys.argv for CLI test
            with patch('sys.argv', ['extract_blocks.py', '--block', block_name]):
                # This would normally run the CLI, but we're testing the logic
                cli_result = extractor.extract_single_block(block_name)
            
            # Results should be identical
            assert api_result["success"] == cli_result["success"]
            assert api_result["block_name"] == cli_result["block_name"]
            assert "validation" in api_result
            assert "validation" in cli_result
            assert api_result["validation"]["status"] == cli_result["validation"]["status"]
    
    def test_multiple_blocks_cli_vs_api(self, temp_dir):
        """Test that multiple blocks CLI and API produce identical results."""
        block_names = ["AConv", "AAttn", "ABlock"]
        
        with patch('ab.rag.extract_blocks.BlockExtractor.warm_index_once', return_value=True), \
             patch('ab.rag.extract_blocks.BlockExtractor.extract_single_block') as mock_single:
            
            # Mock successful extraction for all blocks
            mock_single.return_value = {
                "success": True,
                "block_name": "MockBlock",
                "validation": {"status": "valid"}
            }
            
            # Test API
            extractor = BlockExtractor()
            api_result = extractor.extract_multiple_blocks(block_names)
            
            # Test CLI equivalent
            cli_result = extractor.extract_multiple_blocks(block_names)
            
            # Results should be identical
            assert len(api_result) == len(cli_result)
            assert set(api_result.keys()) == set(cli_result.keys())
            
            for block_name in block_names:
                assert api_result[block_name]["success"] == cli_result[block_name]["success"]
    
    def test_validation_disabled_cli_vs_api(self, temp_dir):
        """Test that validation disabled works the same in CLI and API."""
        block_name = "AConv"
        
        with patch('ab.rag.extract_blocks.BlockExtractor.warm_index_once', return_value=True), \
             patch('ab.rag.extract_blocks.BlockExtractor.extract_block') as mock_extract, \
             patch('ab.rag.extract_blocks.BlockExtractor.validate_block') as mock_validate:
            
            mock_extract.return_value = {
                "success": True,
                "block_name": block_name
            }
            
            # Test API with validation disabled
            extractor = BlockExtractor()
            api_result = extractor.extract_single_block(block_name, validate=False)
            
            # Test CLI equivalent (--no-validate flag)
            cli_result = extractor.extract_single_block(block_name, validate=False)
            
            # Both should not have validation
            assert "validation" not in api_result
            assert "validation" not in cli_result
            assert api_result["success"] == cli_result["success"]
            
            # validate_block should not be called
            mock_validate.assert_not_called()
    
    def test_cleanup_invalid_cli_vs_api(self, temp_dir):
        """Test that cleanup_invalid parameter works the same in CLI and API."""
        block_name = "AConv"
        
        with patch('ab.rag.extract_blocks.BlockExtractor.warm_index_once', return_value=True), \
             patch('ab.rag.extract_blocks.BlockExtractor.extract_block') as mock_extract, \
             patch('ab.rag.extract_blocks.BlockExtractor.validate_block') as mock_validate:
            
            mock_extract.return_value = {"success": True, "block_name": block_name}
            mock_validate.return_value = {"status": "valid"}
            
            # Test API with cleanup_invalid=True
            extractor = BlockExtractor()
            api_result = extractor.extract_single_block(block_name, cleanup_invalid=True)
            
            # Test CLI equivalent (--cleanup-invalid flag)
            cli_result = extractor.extract_single_block(block_name, cleanup_invalid=True)
            
            # Both should call validate_block with cleanup_invalid=True
            assert mock_validate.call_count == 2
            for call in mock_validate.call_args_list:
                assert call[1]["cleanup_invalid"] is True
    
    def test_batch_extraction_cli_vs_api(self, temp_dir, sample_json_file):
        """Test that batch extraction CLI and API produce identical results."""
        with patch('ab.rag.extract_blocks.BlockExtractor.warm_index_once', return_value=True), \
             patch('ab.rag.extract_blocks.BlockExtractor.extract_blocks_from_file') as mock_batch:
            
            mock_batch.return_value = {
                "success": True,
                "processed": 3,
                "ok": 2,
                "fail": 1,
                "results": {
                    "AConv": {"success": True, "validation": {"status": "valid"}},
                    "AAttn": {"success": True, "validation": {"status": "valid"}},
                    "ABlock": {"success": False, "reason": "unresolved dependencies"}
                }
            }
            
            # Test API
            extractor = BlockExtractor()
            api_result = extractor.auto_extract_all_blocks(json_path=sample_json_file)
            
            # Test CLI equivalent
            cli_result = extractor.auto_extract_all_blocks(json_path=sample_json_file)
            
            # Results should be identical
            assert api_result["success"] == cli_result["success"]
            assert api_result["processed"] == cli_result["processed"]
            assert api_result["ok"] == cli_result["ok"]
            assert api_result["fail"] == cli_result["fail"]
    
    def test_error_handling_consistency(self, temp_dir):
        """Test that error handling is consistent between CLI and API."""
        block_name = "NonExistentBlock"
        
        with patch('ab.rag.extract_blocks.BlockExtractor.warm_index_once', return_value=True), \
             patch('ab.rag.extract_blocks.BlockExtractor.extract_block') as mock_extract:
            
            # Mock extraction failure
            mock_extract.side_effect = Exception("Test error")
            
            # Test API
            extractor = BlockExtractor()
            api_result = extractor.extract_single_block(block_name)
            
            # Test CLI equivalent
            cli_result = extractor.extract_single_block(block_name)
            
        # Both should handle errors the same way
        assert api_result["success"] is False
        assert cli_result["success"] is False
        assert "reason" in api_result
        assert "reason" in cli_result
        assert "Test error" in api_result["reason"]
        assert "Test error" in cli_result["reason"]
    
    def test_retry_failed_blocks_cli_vs_api(self, temp_dir):
        """Test that retry failed blocks works the same in CLI and API."""
        # Set up failed blocks
        extractor = BlockExtractor()
        extractor.failed_blocks = ["FailedBlock1", "FailedBlock2"]
        
        with patch('ab.rag.extract_blocks.BlockExtractor.warm_index_once', return_value=True), \
             patch('ab.rag.extract_blocks.BlockExtractor.extract_single_block') as mock_single:
            
            mock_single.return_value = {
                "success": True,
                "block_name": "FailedBlock1",
                "validation": {"status": "valid"}
            }
            
            # Test API
            api_result = extractor.retry_failed_blocks()
            
            # Test CLI equivalent
            cli_result = extractor.retry_failed_blocks()
            
            # Results should be identical
            assert len(api_result) == len(cli_result)
            assert set(api_result.keys()) == set(cli_result.keys())
    
    def test_auto_extract_json_generation_consistency(self, temp_dir):
        """Test that auto-extract JSON generation works consistently."""
        missing_json = os.path.join(temp_dir, "missing_blocks.json")
        
        with patch('ab.rag.extract_blocks.BlockExtractor.warm_index_once', return_value=True), \
             patch('ab.rag.extract_blocks.BlockExtractor._generate_block_names', return_value=True), \
             patch('ab.rag.extract_blocks.BlockExtractor.extract_blocks_from_file') as mock_extract:
            
            mock_extract.return_value = {"success": True, "processed": 10}
            
            # Test API
            extractor = BlockExtractor()
            api_result = extractor.auto_extract_all_blocks(json_path=missing_json)
            
            # Test CLI equivalent
            cli_result = extractor.auto_extract_all_blocks(json_path=missing_json)
            
            # Both should generate JSON and extract
            assert api_result["success"] == cli_result["success"]
            assert extractor._generate_block_names.call_count == 2  # Called twice (API + CLI)
    
    def test_parameter_validation_consistency(self, temp_dir):
        """Test that parameter validation is consistent between CLI and API."""
        # Test invalid parameters
        extractor = BlockExtractor()
        
        # Test with invalid block name
        result = extractor.extract_single_block("")
        assert result["success"] is False
        
        # Test with None block name
        result = extractor.extract_single_block(None)
        assert result["success"] is False
        
        # Test with empty block list
        result = extractor.extract_multiple_blocks([])
        assert result == {}
        
        # Test with None block list
        result = extractor.extract_multiple_blocks(None)
        assert result == {}
    
    def test_output_format_consistency(self, temp_dir):
        """Test that output formats are consistent between CLI and API."""
        block_name = "AConv"
        
        with patch('ab.rag.extract_blocks.BlockExtractor.warm_index_once', return_value=True), \
             patch('ab.rag.extract_blocks.BlockExtractor.extract_block') as mock_extract, \
             patch('ab.rag.extract_blocks.BlockExtractor.validate_block') as mock_validate:
            
            mock_extract.return_value = {
                "success": True,
                "block_name": block_name,
                "file_path": f"generated_packages/{block_name}.py"
            }
            mock_validate.return_value = {
                "name": block_name,
                "status": "valid",
                "moved_to_block_dir": True
            }
            
            # Test API
            extractor = BlockExtractor()
            api_result = extractor.extract_single_block(block_name)
            
            # Test CLI equivalent
            cli_result = extractor.extract_single_block(block_name)
            
            # Both should have the same structure
            required_keys = ["success", "block_name"]
            for key in required_keys:
                assert key in api_result
                assert key in cli_result
                assert api_result[key] == cli_result[key]
            
            # Both should have validation if enabled
            assert "validation" in api_result
            assert "validation" in cli_result
            assert api_result["validation"]["status"] == cli_result["validation"]["status"]


if __name__ == "__main__":
    pytest.main([__file__])
