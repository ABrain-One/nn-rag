"""
Pytest configuration and shared fixtures for the RAG test suite.
"""

import pytest
import tempfile
import shutil
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a session-scoped test data directory."""
    test_dir = tempfile.mkdtemp(prefix="nn_rag_test_")
    yield test_dir
    shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def clean_environment():
    """Ensure a clean environment for each test."""
    # Store original environment
    original_env = os.environ.copy()
    
    # Set test environment variables
    os.environ["PYTHONPATH"] = str(Path(__file__).parent.parent)
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(scope="function")
def mock_repositories():
    """Mock repository data for testing."""
    return {
        "pytorch/vision": {
            "blocks": ["ResNet", "VGG", "AlexNet"],
            "files": 1000
        },
        "huggingface/pytorch-image-models": {
            "blocks": ["EfficientNet", "ConvNeXt", "VisionTransformer"],
            "files": 500
        }
    }


@pytest.fixture(scope="function")
def sample_block_data():
    """Sample block data for testing."""
    return {
        "AConv": {
            "source": '''
import torch.nn as nn
class AConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3)
    
    def forward(self, x):
        return self.conv(x)
''',
            "dependencies": ["torch", "torch.nn"],
            "is_valid": True
        },
        "AAttn": {
            "source": '''
import torch.nn as nn
class AAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, 8)
    
    def forward(self, x):
        return self.attention(x, x, x)[0]
''',
            "dependencies": ["torch", "torch.nn"],
            "is_valid": True
        },
        "InvalidBlock": {
            "source": '''
import torch.nn as nn
class InvalidBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
    
    def forward(self, x):
        return self.conv(x  # Missing closing parenthesis
''',
            "dependencies": ["torch", "torch.nn"],
            "is_valid": False
        }
    }


@pytest.fixture(scope="function")
def dummy_json_file(tmp_path):
    """Create a dummy JSON file with minimal test blocks."""
    dummy_json = tmp_path / "dummy_blocks.json"
    test_blocks = {
        "blocks": [
            {
                "name": "TestBlock1",
                "file": "test_file1.py",
                "class": "TestBlock1"
            },
            {
                "name": "TestBlock2", 
                "file": "test_file2.py",
                "class": "TestBlock2"
            }
        ]
    }
    
    import json
    with open(dummy_json, 'w') as f:
        json.dump(test_blocks, f, indent=2)
    
    return str(dummy_json)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add markers based on test file names
        if "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "test_performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        elif "test_error_scenarios" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        else:
            item.add_marker(pytest.mark.unit)
