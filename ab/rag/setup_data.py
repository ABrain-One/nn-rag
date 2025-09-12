#!/usr/bin/env python3
"""
Pre-installation data setup for nn-rag package.

This script clones repositories and builds the index during package installation
to reduce runtime overhead. If this fails or is skipped, the package will
fall back to runtime cloning and indexing.
"""

import os
import sys
from pathlib import Path
import subprocess
import json
import logging

# Add the package to the path
package_dir = Path(__file__).parent
sys.path.insert(0, str(package_dir.parent.parent))

from ab.rag.extract_blocks import BlockExtractor

def setup_package_data():
    """
    Clone repositories and build index during package installation.
    Returns (success: bool, message: str)
    """
    try:
        # Create package cache directory
        cache_dir = package_dir / ".cache"
        cache_dir.mkdir(exist_ok=True)
        
        print("Setting up nn-rag package data...")
        print(f"Cache directory: {cache_dir}")
        
        # Initialize extractor with package-local paths
        extractor = BlockExtractor(index_mode="force")
        
        # Clone and index all repositories
        print("Cloning repositories and building index...")
        success = extractor.warm_index_once()
        
        if success:
            print("Package data setup completed successfully!")
            return True, "Package data initialized successfully"
        else:
            print("Package data setup failed, will use runtime setup")
            return False, "Package data setup failed, will fall back to runtime setup"
            
    except Exception as e:
        print(f"Package data setup failed: {e}")
        return False, f"Package data setup failed: {e}"

def check_package_data():
    """
    Check if package data is available and valid.
    Returns (available: bool, message: str)
    """
    try:
        cache_dir = package_dir / ".cache"
        index_db = cache_dir / "index.db"
        
        if not cache_dir.exists():
            return False, "Cache directory not found"
            
        if not index_db.exists():
            return False, "Index database not found"
            
        # Check if index has data by checking if any repos are indexed
        from ab.rag.file_index import FileIndexStore
        index = FileIndexStore(db_path=index_db)
        
        # Check if we have any indexed repositories by testing a known repo
        test_repos = ["pytorch/pytorch", "huggingface/transformers", "pytorch/vision"]
        indexed_count = sum(1 for repo in test_repos if index.repo_has_index(repo))
        
        if indexed_count == 0:
            return False, "No repositories indexed"
            
        return True, f"Package data available with {indexed_count} repositories"
        
    except Exception as e:
        return False, f"Package data check failed: {e}"

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        # Check mode
        available, message = check_package_data()
        print(f"Package data check: {message}")
        sys.exit(0 if available else 1)
    else:
        # Setup mode
        success, message = setup_package_data()
        print(f"Setup result: {message}")
        sys.exit(0 if success else 1)
