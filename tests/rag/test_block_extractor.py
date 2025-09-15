"""
Comprehensive tests for BlockExtractor functionality.
Tests all API methods, validation, error handling, and edge cases.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import os
import sys

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ab.rag.extract_blocks import BlockExtractor


class TestBlockExtractor:
    """Test suite for BlockExtractor class."""
    
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
            # Mock the results file to use temp directory
            return extractor
    
    @pytest.fixture
    def sample_json_file(self, temp_dir):
        """Create a sample JSON file with test block names."""
        json_path = os.path.join(temp_dir, "test_blocks.json")
        sample_blocks = ["AConv", "AAttn", "ABlock", "ACM", "APNB"]
        with open(json_path, 'w') as f:
            json.dump(sample_blocks, f)
        return json_path
    
    def test_initialization(self, extractor):
        """Test BlockExtractor initialization."""
        assert extractor.max_workers > 0
        assert extractor.max_retries == 2
        assert extractor.index_mode == "missing"
        assert hasattr(extractor, 'extracted_blocks')
        assert hasattr(extractor, 'failed_blocks')
        assert hasattr(extractor, 'skipped_blocks')
    
    def test_extract_single_block_with_validation(self, extractor):
        """Test single block extraction with validation enabled."""
        with patch.object(extractor, 'extract_block') as mock_extract, \
             patch.object(extractor, 'validate_block') as mock_validate:
            
            # Mock successful extraction
            mock_extract.return_value = {
                "success": True,
                "block_name": "AConv",
                "file_path": "generated_packages/AConv.py"
            }
            
            # Mock successful validation
            mock_validate.return_value = {
                "name": "AConv",
                "status": "valid",
                "moved_to_block_dir": True
            }
            
            result = extractor.extract_single_block("AConv")
            
            assert result["success"] is True
            assert "validation" in result
            assert result["validation"]["status"] == "valid"
            mock_extract.assert_called_once_with("AConv")
            mock_validate.assert_called_once_with("AConv", cleanup_invalid=False)
    
    def test_extract_single_block_without_validation(self, extractor):
        """Test single block extraction with validation disabled."""
        with patch.object(extractor, 'extract_block') as mock_extract, \
             patch.object(extractor, 'validate_block') as mock_validate:
            
            mock_extract.return_value = {
                "success": True,
                "block_name": "AConv"
            }
            
            result = extractor.extract_single_block("AConv", validate=False)
            
            assert result["success"] is True
            assert "validation" not in result
            mock_extract.assert_called_once_with("AConv")
            mock_validate.assert_not_called()
    
    def test_extract_single_block_failure(self, extractor):
        """Test single block extraction failure."""
        with patch.object(extractor, 'extract_block') as mock_extract:
            mock_extract.side_effect = Exception("Test error")
            
            result = extractor.extract_single_block("InvalidBlock")
            
            assert result["success"] is False
            assert "reason" in result
            assert "Test error" in result["reason"]
    
    def test_extract_multiple_blocks_with_validation(self, extractor):
        """Test multiple block extraction with validation."""
        with patch.object(extractor, 'extract_single_block') as mock_single:
            mock_single.return_value = {
                "success": True,
                "block_name": "AConv",
                "validation": {"status": "valid"}
            }
            
            results = extractor.extract_multiple_blocks(["AConv", "AAttn"])
            
            assert len(results) == 2
            assert "AConv" in results
            assert "AAttn" in results
            assert all(r["success"] for r in results.values())
            assert mock_single.call_count == 2
    
    def test_extract_multiple_blocks_without_validation(self, extractor):
        """Test multiple block extraction without validation."""
        with patch.object(extractor, 'extract_single_block') as mock_single:
            mock_single.return_value = {
                "success": True,
                "block_name": "AConv"
            }
            
            results = extractor.extract_multiple_blocks(["AConv", "AAttn"], validate=False)
            
            assert len(results) == 2
            mock_single.assert_any_call("AConv", validate=False, cleanup_invalid=False)
            mock_single.assert_any_call("AAttn", validate=False, cleanup_invalid=False)
    
    def test_retry_failed_blocks_with_validation(self, extractor):
        """Test retry failed blocks with validation."""
        # Add some failed blocks
        extractor.failed_blocks = ["FailedBlock1", "FailedBlock2"]
        
        with patch.object(extractor, 'extract_single_block') as mock_single:
            mock_single.return_value = {
                "success": True,
                "block_name": "FailedBlock1",
                "validation": {"status": "valid"}
            }
            
            results = extractor.retry_failed_blocks()
            
            assert len(results) == 2
            assert "FailedBlock1" in results
            assert "FailedBlock2" in results
            mock_single.assert_any_call("FailedBlock1", validate=True, cleanup_invalid=False)
            mock_single.assert_any_call("FailedBlock2", validate=True, cleanup_invalid=False)
    
    def test_retry_failed_blocks_without_validation(self, extractor):
        """Test retry failed blocks without validation."""
        extractor.failed_blocks = ["FailedBlock1"]
        
        with patch.object(extractor, 'extract_single_block') as mock_single:
            mock_single.return_value = {"success": True}
            
            results = extractor.retry_failed_blocks(validate=False)
            
            assert len(results) == 1
            mock_single.assert_called_once_with("FailedBlock1", validate=False, cleanup_invalid=False)
    
    def test_auto_extract_all_blocks_with_existing_json(self, extractor, sample_json_file):
        """Test auto-extract with existing JSON file."""
        with patch.object(extractor, 'warm_index_once', return_value=True), \
             patch.object(extractor, 'extract_blocks_from_file') as mock_extract:
            
            mock_extract.return_value = {
                "success": True,
                "processed": 5,
                "ok": 4,
                "fail": 1
            }
            
            result = extractor.auto_extract_all_blocks(json_path=sample_json_file)
            
            assert result["success"] is True
            assert result["processed"] == 5
            mock_extract.assert_called_once()
    
    def test_auto_extract_all_blocks_with_missing_json(self, extractor, temp_dir):
        """Test auto-extract with missing JSON file (should generate it)."""
        json_path = os.path.join(temp_dir, "missing_blocks.json")
        
        with patch.object(extractor, 'warm_index_once', return_value=True), \
             patch.object(extractor, '_generate_block_names', return_value=True), \
             patch.object(extractor, 'extract_blocks_from_file') as mock_extract:
            
            mock_extract.return_value = {"success": True, "processed": 10}
            
            result = extractor.auto_extract_all_blocks(json_path=json_path)
            
            assert result["success"] is True
            extractor._generate_block_names.assert_called_once()
            mock_extract.assert_called_once()
    
    def test_auto_extract_all_blocks_json_generation_failure(self, extractor, temp_dir):
        """Test auto-extract when JSON generation fails."""
        json_path = os.path.join(temp_dir, "missing_blocks.json")
        
        with patch.object(extractor, '_generate_block_names', return_value=False):
            result = extractor.auto_extract_all_blocks(json_path=json_path)
            
            assert result["success"] is False
            assert "Failed to generate block names" in result["reason"]
    
    def test_auto_extract_all_blocks_index_warming_failure(self, extractor, sample_json_file):
        """Test auto-extract when index warming fails."""
        with patch.object(extractor, 'warm_index_once', return_value=False):
            result = extractor.auto_extract_all_blocks(json_path=sample_json_file)
            
            assert result["success"] is False
            assert "Failed to warm index" in result["reason"]
    
    def test_generate_block_names_success(self, extractor, temp_dir):
        """Test successful block names generation."""
        output_path = os.path.join(temp_dir, "generated_blocks.json")
        
        with patch('ab.rag.make_blocks_name.discover_nn_block_names') as mock_discover:
            mock_discover.return_value = ["Block1", "Block2", "Block3"]
            
            result = extractor._generate_block_names(Path(output_path))
            
            assert result is True
            assert os.path.exists(output_path)
            
            with open(output_path, 'r') as f:
                data = json.load(f)
                assert data == ["Block1", "Block2", "Block3"]
    
    def test_generate_block_names_failure(self, extractor, temp_dir):
        """Test block names generation failure."""
        output_path = os.path.join(temp_dir, "generated_blocks.json")
        
        with patch('ab.rag.make_blocks_name.discover_nn_block_names', side_effect=Exception("Discovery failed")):
            result = extractor._generate_block_names(Path(output_path))
            
            assert result is False
            assert not os.path.exists(output_path)
    
    def test_get_extraction_stats(self, extractor):
        """Test extraction statistics."""
        # Set up some test data
        extractor.extracted_blocks = [
            {"block_name": "Block1"},
            {"block_name": "Block2"}
        ]
        extractor.failed_blocks = ["Block3"]
        extractor.skipped_blocks = ["Block4"]
        
        stats = extractor.get_extraction_stats()
        
        assert stats["extracted_count"] == 2
        assert stats["failed_count"] == 1
        assert stats["skipped_count"] == 1
        assert stats["extracted_blocks"] == ["Block1", "Block2"]
        assert stats["failed_blocks"] == ["Block3"]
        assert stats["skipped_blocks"] == ["Block4"]
    
    def test_get_extraction_stats_empty(self, extractor):
        """Test extraction statistics with empty data."""
        # Create a fresh extractor to get empty stats
        with patch('ab.rag.extract_blocks.BlockExtractor.warm_index_once', return_value=True):
            fresh_extractor = BlockExtractor()
            # Clear the data to test empty stats
            fresh_extractor.extracted_blocks = []
            fresh_extractor.failed_blocks = []
            fresh_extractor.skipped_blocks = []
            
            stats = fresh_extractor.get_extraction_stats()
            
            assert stats["extracted_count"] == 0
            assert stats["failed_count"] == 0
            assert stats["skipped_count"] == 0
            assert stats["extracted_blocks"] == []
            assert stats["failed_blocks"] == []
            assert stats["skipped_blocks"] == []
    
    def test_cleanup_invalid_parameter(self, extractor):
        """Test cleanup_invalid parameter is passed correctly."""
        with patch.object(extractor, 'extract_block') as mock_extract, \
             patch.object(extractor, 'validate_block') as mock_validate:
            
            mock_extract.return_value = {"success": True}
            mock_validate.return_value = {"status": "valid"}
            
            # Test with cleanup_invalid=True
            extractor.extract_single_block("AConv", cleanup_invalid=True)
            mock_validate.assert_called_with("AConv", cleanup_invalid=True)
            
            # Test with cleanup_invalid=False
            extractor.extract_single_block("AAttn", cleanup_invalid=False)
            mock_validate.assert_called_with("AAttn", cleanup_invalid=False)


if __name__ == "__main__":
    pytest.main([__file__])
