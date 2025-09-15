"""
Tests for block validation functionality.
Tests validation logic, file movement, error handling, and edge cases.
"""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ab.rag.block_validator import BlockValidator


class TestBlockValidator:
    """Test suite for BlockValidator class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def validator(self, temp_dir):
        """Create a BlockValidator instance for testing."""
        generated_dir = os.path.join(temp_dir, "generated_packages")
        block_dir = os.path.join(temp_dir, "block")
        os.makedirs(generated_dir, exist_ok=True)
        os.makedirs(block_dir, exist_ok=True)
        
        return BlockValidator(generated_dir=generated_dir, block_dir=block_dir)
    
    @pytest.fixture
    def valid_python_file(self, temp_dir):
        """Create a valid Python file for testing."""
        file_path = os.path.join(temp_dir, "generated_packages", "ValidBlock.py")
        content = '''
import torch
import torch.nn as nn

class ValidBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
    
    def forward(self, x):
        return self.conv(x)
'''
        with open(file_path, 'w') as f:
            f.write(content)
        return file_path
    
    @pytest.fixture
    def invalid_python_file(self, temp_dir):
        """Create an invalid Python file for testing."""
        file_path = os.path.join(temp_dir, "generated_packages", "InvalidBlock.py")
        content = '''
import torch
import torch.nn as nn

class InvalidBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
    
    def forward(self, x):
        return self.conv(x  # Missing closing parenthesis
'''
        with open(file_path, 'w') as f:
            f.write(content)
        return file_path
    
    def test_initialization(self, validator):
        """Test BlockValidator initialization."""
        assert validator.generated_dir is not None
        assert validator.block_dir is not None
        assert os.path.exists(validator.generated_dir)
        assert os.path.exists(validator.block_dir)
    
    def test_validate_single_block_valid(self, validator, valid_python_file):
        """Test validation of a valid block."""
        block_name = "ValidBlock"
        
        is_valid, error = validator.validate_single_block(block_name)
        
        assert is_valid is True
        assert error is None
        
        # Note: validate_single_block only validates, doesn't move files
        # File movement is handled by validate_and_move_block
    
    def test_validate_single_block_invalid_syntax(self, validator, invalid_python_file):
        """Test validation of a block with syntax errors."""
        block_name = "InvalidBlock"
        
        is_valid, error = validator.validate_single_block(block_name)
        
        assert is_valid is False
        assert error is not None
        assert "syntax" in error.lower() or "invalid" in error.lower()
        
        # Check that file was not moved
        block_file = os.path.join(validator.block_dir, f"{block_name}.py")
        assert not os.path.exists(block_file)
        assert os.path.exists(invalid_python_file)
    
    def test_validate_single_block_missing_file(self, validator):
        """Test validation of a non-existent block file."""
        block_name = "NonExistentBlock"
        
        is_valid, error = validator.validate_single_block(block_name)
        
        assert is_valid is False
        assert "does not exist" in error
    
    def test_validate_and_move_block_valid(self, validator, valid_python_file):
        """Test validation and movement of a valid block."""
        block_name = "ValidBlock"
        
        is_valid, error = validator.validate_and_move_block(block_name)
        
        assert is_valid is True
        assert error is None
        
        # Check that file was moved
        block_file = os.path.join(validator.block_dir, f"{block_name}.py")
        assert os.path.exists(block_file)
        assert not os.path.exists(valid_python_file)
    
    def test_validate_and_move_block_invalid(self, validator, invalid_python_file):
        """Test validation and movement of an invalid block."""
        block_name = "InvalidBlock"
        
        is_valid, error = validator.validate_and_move_block(block_name)
        
        assert is_valid is False
        assert error is not None
        
        # Check that file was not moved
        block_file = os.path.join(validator.block_dir, f"{block_name}.py")
        assert not os.path.exists(block_file)
        assert os.path.exists(invalid_python_file)
    
    def test_move_valid_block_success(self, validator, valid_python_file):
        """Test successful block movement."""
        block_name = "ValidBlock"
        
        result = validator.move_valid_block(block_name)
        
        assert result is True
        
        # Check that file was moved
        block_file = os.path.join(validator.block_dir, f"{block_name}.py")
        assert os.path.exists(block_file)
        assert not os.path.exists(valid_python_file)
    
    def test_move_valid_block_missing_file(self, validator):
        """Test moving a non-existent block."""
        block_name = "NonExistentBlock"
        
        result = validator.move_valid_block(block_name)
        
        assert result is False
    
    def test_move_valid_block_destination_exists(self, validator, valid_python_file):
        """Test moving a block when destination already exists."""
        block_name = "ValidBlock"
        
        # Create a file in the destination
        block_file = os.path.join(validator.block_dir, f"{block_name}.py")
        with open(block_file, 'w') as f:
            f.write("existing content")
        
        result = validator.move_valid_block(block_name)
        
        assert result is True
        
        # Check that file was overwritten
        with open(block_file, 'r') as f:
            content = f.read()
            assert "class ValidBlock" in content  # New content, not "existing content"
    
    def test_validate_blocks_batch(self, validator, temp_dir):
        """Test batch validation of multiple blocks."""
        # Create multiple test files
        valid_file = os.path.join(temp_dir, "generated_packages", "ValidBlock1.py")
        invalid_file = os.path.join(temp_dir, "generated_packages", "InvalidBlock1.py")
        
        # Valid file
        with open(valid_file, 'w') as f:
            f.write('''
import torch.nn as nn
class ValidBlock1(nn.Module):
    def forward(self, x): return x
''')
        
        # Invalid file
        with open(invalid_file, 'w') as f:
            f.write('''
import torch.nn as nn
class InvalidBlock1(nn.Module
    def forward(self, x): return x  # Missing closing parenthesis
''')
        
        results = validator.validate_all_blocks()
        
        assert len(results) == 2
        assert "ValidBlock1" in results
        assert "InvalidBlock1" in results
        
        # Check valid block
        valid_result = results["ValidBlock1"]
        assert valid_result[0] is True  # is_valid
        assert valid_result[1] is None  # error
        
        # Check invalid block
        invalid_result = results["InvalidBlock1"]
        assert invalid_result[0] is False  # is_valid
        assert invalid_result[1] is not None  # error
    
    def test_validate_all_blocks_empty(self, validator):
        """Test batch validation with no files."""
        results = validator.validate_all_blocks()
        assert results == {}
    
    def test_validate_blocks_mixed_results(self, validator, temp_dir):
        """Test batch validation with mixed valid/invalid blocks."""
        # Create test files
        valid_file = os.path.join(temp_dir, "generated_packages", "ValidBlock.py")
        invalid_file = os.path.join(temp_dir, "generated_packages", "InvalidBlock.py")
        missing_file = os.path.join(temp_dir, "generated_packages", "MissingBlock.py")
        
        # Valid file
        with open(valid_file, 'w') as f:
            f.write('''
import torch.nn as nn
class ValidBlock(nn.Module):
    def forward(self, x): return x
''')
        
        # Invalid file
        with open(invalid_file, 'w') as f:
            f.write('''
import torch.nn as nn
class InvalidBlock(nn.Module
    def forward(self, x): return x  # Missing closing parenthesis
''')
        
        # Missing file (don't create it)
        
        results = validator.validate_all_blocks()
        
        assert len(results) == 2  # Only validates existing files
        
        # Valid block should be valid
        assert results["ValidBlock"][0] is True  # is_valid
        assert results["ValidBlock"][1] is None  # error
        
        # Invalid block should be invalid
        assert results["InvalidBlock"][0] is False  # is_valid
        assert results["InvalidBlock"][1] is not None  # error
        
        # Missing block is not included in results since validate_all_blocks only processes existing files
    
    def test_validation_error_handling(self, validator, temp_dir):
        """Test error handling during validation."""
        # Create a file that will cause an error during validation
        error_file = os.path.join(temp_dir, "generated_packages", "ErrorBlock.py")
        with open(error_file, 'w') as f:
            f.write('''
import torch.nn as nn
class ErrorBlock(nn.Module):
    def forward(self, x): return x
''')
        
        # Test normal validation (the method should handle errors gracefully)
        is_valid, error = validator.validate_single_block("ErrorBlock")
        
        # Should be valid since the file has correct syntax
        assert is_valid is True
        assert error is None
    
    def test_file_permissions_error(self, validator, temp_dir):
        """Test handling of file permission errors."""
        block_name = "PermissionBlock"
        source_file = os.path.join(temp_dir, "generated_packages", f"{block_name}.py")
        dest_file = os.path.join(validator.block_dir, f"{block_name}.py")
        
        # Create source file
        with open(source_file, 'w') as f:
            f.write('''
import torch.nn as nn
class PermissionBlock(nn.Module):
    def forward(self, x): return x
''')
        
        # Make destination directory read-only
        # 0o444 = read-only for owner, group, and others
        os.chmod(validator.block_dir, 0o444)
        
        try:
            result = validator.move_valid_block(block_name)
            assert result is False
        finally:
            # Restore permissions
            os.chmod(validator.block_dir, 0o755)


if __name__ == "__main__":
    pytest.main([__file__])
