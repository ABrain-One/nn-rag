"""
Simple test to verify the test framework is working.
"""

import pytest
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ab.rag.extract_blocks import BlockExtractor


class TestSimple:
    """Simple test class to verify test framework."""
    
    def test_import_works(self):
        """Test that imports work correctly."""
        from ab.rag.extract_blocks import BlockExtractor
        assert BlockExtractor is not None
    
    def test_block_extractor_creation(self):
        """Test that BlockExtractor can be created."""
        # This should work without errors
        extractor = BlockExtractor()
        assert extractor is not None
        assert hasattr(extractor, 'max_workers')
        assert hasattr(extractor, 'extracted_blocks')
    
    def test_simple_math(self):
        """Test basic functionality."""
        assert 1 + 1 == 2
        assert "hello" + " world" == "hello world"
    
    def test_list_operations(self):
        """Test list operations."""
        test_list = [1, 2, 3, 4, 5]
        assert len(test_list) == 5
        assert 3 in test_list
        assert 6 not in test_list


if __name__ == "__main__":
    pytest.main([__file__])
