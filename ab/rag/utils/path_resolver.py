"""
Centralized path resolution utility for the nn-rag package.

This module provides functions to resolve paths to package resources regardless of whether
the package is running from source or installed via pip.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union


def get_package_root() -> Path:
    """
    Get the root directory of the nn-rag package.
    
    This works both in development (source) and when installed via pip.
    """
    # First, try to find the package in site-packages
    import site
    for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
        package_path = Path(site_dir) / "ab" / "rag"
        if package_path.exists() and (package_path / "__init__.py").exists():
            return package_path
    
    # Fallback to the directory containing this file
    return Path(__file__).parent.parent


def get_package_data_dir() -> Path:
    """
    Get the directory containing package data files (config, etc.).
    """
    return get_package_root() / "config"


def get_cache_dir() -> Path:
    """
    Get the cache directory for the package.
    Creates the directory if it doesn't exist.
    
    Uses a user-writable location (~/.cache/nn-rag/) to ensure it works
    when the package is installed via pip (where package directories may be read-only).
    """
    # Use user's home directory cache to ensure writability when package is installed
    cache_dir = Path.home() / ".cache" / "nn-rag"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def ensure_cache_structure() -> bool:
    """
    Ensure the complete cache directory structure exists.
    
    Returns:
        True if the structure was created successfully
    """
    try:
        cache_dir = get_cache_dir()
        repo_cache_dir = cache_dir / "repo_cache"
        repo_cache_dir.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Warning: Could not create cache structure: {e}")
        return False


def get_config_file_path(filename: str) -> Path:
    """
    Get the full path to a configuration file.
    
    Returns the path in the package data directory (writable location).
    The actual config loading logic will check package resources first.
    
    Args:
        filename: Name of the config file (e.g., "repo_config.json")
    
    Returns:
        Path to the configuration file location (where it can be written)
    """
    # Return writable location - actual loading will check package resources first
    return get_package_data_dir() / filename


def get_resource_path(relative_path: Union[str, Path]) -> Path:
    """
    Get the full path to a package resource.
    
    Args:
        relative_path: Path relative to the package root
    
    Returns:
        Full path to the resource
    """
    if isinstance(relative_path, str):
        relative_path = Path(relative_path)
    
    return get_package_root() / relative_path


def ensure_package_data_exists() -> bool:
    """
    Ensure that essential package data files exist.
    If they don't exist, create them with default content.
    
    Returns:
        True if all data files exist or were created successfully
    """
    try:
        config_dir = get_package_data_dir()
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for essential config files
        repo_config = config_dir / "repo_config.json"
        nn_blocks = config_dir / "nn_block_names.json"
        
        if not repo_config.exists():
            # Try to copy from package data first (if available in installed package)
            try:
                from importlib.resources import files
                package_config = files('ab.rag.config').joinpath('repo_config.json')
                if package_config.is_file():
                    # Copy the full config from package
                    repo_config.write_text(package_config.read_text(encoding='utf-8'), encoding='utf-8')
                    return True
            except (ImportError, FileNotFoundError, AttributeError):
                pass
            
            # Fallback: Create minimal default repo config only if package config not found
            default_repo_config = {
                "huggingface/pytorch-image-models": {
                    "url": "https://github.com/huggingface/pytorch-image-models.git",
                    "priority": 1,
                    "paths": ["timm/models", "timm/layers"],
                    "description": "Essential for model/layer blocks"
                },
                "pytorch/vision": {
                    "url": "https://github.com/pytorch/vision.git",
                    "priority": 1,
                    "paths": ["torchvision/models", "torchvision/ops"],
                    "description": "TorchVision models & ops"
                }
            }
            import json
            repo_config.write_text(json.dumps(default_repo_config, indent=2), encoding="utf-8")
        
        if not nn_blocks.exists():
            # Create minimal nn_blocks file
            default_blocks = ["BasicBlock", "Bottleneck", "ResNet", "VGG", "AlexNet"]
            import json
            nn_blocks.write_text(json.dumps(default_blocks, indent=2), encoding="utf-8")
        
        return True
    except Exception as e:
        print(f"Warning: Could not ensure package data exists: {e}")
        return False


def get_development_root() -> Optional[Path]:
    """
    Get the development root directory if running from source.
    Returns None if not in development mode.
    """
    # Check if we're in a development environment
    current_file = Path(__file__)
    potential_root = current_file.parent.parent.parent
    
    # Look for development indicators
    if (potential_root / "setup.py").exists() or (potential_root / "pyproject.toml").exists():
        return potential_root
    
    return None


def is_development_mode() -> bool:
    """
    Check if the package is running in development mode.
    """
    return get_development_root() is not None


def get_generated_packages_dir() -> Path:
    """
    Get the directory for generated packages.
    """
    if is_development_mode():
        # In development, use the project root
        dev_root = get_development_root()
        return dev_root / "ab" / "rag" / "generated_packages"
    else:
        # When installed, use package directory
        return get_package_root() / "generated_packages"


def get_blocks_dir(project_dir: Optional[Path] = None) -> Path:
    """
    Get the directory for blocks.
    This should always be in the user's project directory, not the package directory.
    
    Args:
        project_dir: Optional project directory. If not provided, auto-detects the project directory.
    
    Returns:
        Path to the blocks directory
    """
    if is_development_mode():
        # In development, use the project root
        dev_root = get_development_root()
        return dev_root / "blocks"
    else:
        # When installed, use the provided project directory or auto-detect
        if project_dir is not None:
            return project_dir / "blocks"
        else:
            # Auto-detect project directory
            detected_project = _detect_project_directory()
            return detected_project / "blocks"


def _detect_project_directory() -> Path:
    """
    Auto-detect the user's project directory.
    Looks for common project indicators and returns the most appropriate directory.
    """
    import os
    from pathlib import Path
    
    current_dir = Path(os.getcwd())
    
    # Look for project indicators in current directory and parents
    for check_dir in [current_dir] + list(current_dir.parents):
        # Check for common project indicators
        project_indicators = [
            "requirements.txt",
            "pyproject.toml", 
            "setup.py",
            "package.json",
            "Cargo.toml",
            "go.mod",
            ".git",
            "src",
            "lib",
            "app"
        ]
        
        # Count how many indicators are present
        indicator_count = sum(1 for indicator in project_indicators if (check_dir / indicator).exists())
        
        # If we find multiple indicators, this is likely a project directory
        if indicator_count >= 2:
            return check_dir
        
        # If we find a .git directory, this is definitely a project
        if (check_dir / ".git").exists():
            return check_dir
    
    # If no project directory is detected, use current working directory
    return current_dir
