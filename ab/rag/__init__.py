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
    import os
    import site
    from pathlib import Path
    
    # Find the package directory
    package_dir = None
    for path in site.getsitepackages() + [site.getusersitepackages()]:
        ab_rag_path = os.path.join(path, 'ab', 'rag')
        if os.path.exists(ab_rag_path):
            package_dir = Path(ab_rag_path)
            break
    
    if package_dir is None:
        package_dir = Path(__file__).parent
    
    cache_dir = package_dir / ".cache"
    index_db = cache_dir / "index.db"
    
    # Check if package data is already set up
    if cache_dir.exists() and index_db.exists():
        return
    
    # Run setup if needed
    try:
        setup_script = package_dir / "setup_data.py"
        if setup_script.exists():
            import subprocess
            import sys
            print("Setting up package data on first use...")
            result = subprocess.run([
                sys.executable, str(setup_script)
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("Package data setup completed successfully!")
            else:
                print("Package data setup failed, will clone repos on first use.")
    except Exception as e:
        print(f"Package data setup failed: {e}")
        print("The package will work but will clone repos on first use.")

# Only run setup if this is a fresh installation (no cache directory)
def _check_and_setup():
    import os
    import site
    from pathlib import Path
    
    # Find the package directory
    package_dir = None
    for path in site.getsitepackages() + [site.getusersitepackages()]:
        ab_rag_path = os.path.join(path, 'ab', 'rag')
        if os.path.exists(ab_rag_path):
            package_dir = Path(ab_rag_path)
            break
    
    if package_dir is None:
        package_dir = Path(__file__).parent
    
    cache_dir = package_dir / ".cache"
    
    # Only run setup if cache directory doesn't exist at all
    if not cache_dir.exists():
        _setup_package_data()

# Run setup only if needed
_check_and_setup()

__all__ = ["BlockExtractor", "BlockValidator", "__version__"]