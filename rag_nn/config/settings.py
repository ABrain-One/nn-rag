"""
Configuration settings for RAG-NN pipeline
"""

import os
from dataclasses import dataclass
from typing import List, Optional

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

@dataclass
class PipelineConfig:
    """Pipeline configuration settings"""
    
    # GitHub API settings
    github_token: Optional[str] = None
    max_results_per_block: int = 3
    rate_limit_delay: float = 1.0
    
    # Search preferences
    prefer_pytorch: bool = True
    prefer_less_dependencies: bool = True
    min_stars: int = 10
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    max_concurrent_requests: int = 5
    
    # Output settings
    output_directory: str = "output"
    blocks_directory: str = "blocks"
    health_check_enabled: bool = True
    
    # File processing
    max_file_size_kb: int = 500
    encoding_fallback: str = "latin-1"
    
    def __post_init__(self):
        """Load environment variables after initialization"""
        if not self.github_token:
            self.github_token = os.getenv('GITHUB_PAT') or os.getenv('GITHUB_TOKEN')
        
        # Create directories if they don't exist
        os.makedirs(self.output_directory, exist_ok=True)
        os.makedirs(self.blocks_directory, exist_ok=True)

@dataclass
class SearchConfig:
    """Search-specific configuration"""
    
    # CV NN specific terms
    cv_keywords: List[str] = None
    nn_keywords: List[str] = None
    
    # Framework preferences
    preferred_frameworks: List[str] = None
    
    def __post_init__(self):
        if self.cv_keywords is None:
            self.cv_keywords = [
                'conv', 'convolution', 'resnet', 'vgg', 'alexnet',
                'mobilenet', 'efficientnet', 'vision', 'image', 'detection'
            ]
        
        if self.nn_keywords is None:
            self.nn_keywords = [
                'neural', 'network', 'model', 'layer', 'activation',
                'pooling', 'dropout', 'batch_norm', 'linear'
            ]
        
        if self.preferred_frameworks is None:
            self.preferred_frameworks = ['pytorch', 'torch', 'pytorch-lightning']

# Default configuration instance
default_config = PipelineConfig()
default_search_config = SearchConfig()
