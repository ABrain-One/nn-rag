"""
Neural Retrieval-Augmented Generation for GitHub code blocks.

This package provides tools for extracting and validating code blocks from GitHub repositories.
"""

# Version
__version__ = "1.0.3"

# Import main components
from .extract_blocks import BlockExtractor
from .block_validator import BlockValidator

# Post-install setup - this runs when the package is imported
def _setup_package_data():
    """Setup package data on first import if not already done."""
    from .utils.path_resolver import get_package_root, get_cache_dir, ensure_package_data_exists, ensure_cache_structure
    
    package_dir = get_package_root()
    cache_dir = get_cache_dir()
    index_db = cache_dir / "index.db"
    
    # Ensure cache structure exists
    ensure_cache_structure()
    
    # Check if package data is already set up
    if cache_dir.exists() and index_db.exists():
        return
    
    # Ensure essential config files exist
    ensure_package_data_exists()
    
    print("Package data setup completed. The package will clone repos on first use.")

# Only run setup if this is a fresh installation (no cache directory)
def _check_and_setup():
    from .utils.path_resolver import get_cache_dir, ensure_package_data_exists
    
    cache_dir = get_cache_dir()
    
    # Only run setup if cache directory doesn't exist at all
    if not cache_dir.exists():
        _setup_package_data()

# Run setup only if needed
_check_and_setup()

__all__ = ["BlockExtractor", "BlockValidator", "__version__"]