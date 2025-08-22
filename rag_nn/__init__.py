"""
RAG-NN: Intelligent Neural Network Code Retrieval Pipeline
A high-performance system for fetching and validating CV NN code blocks from GitHub
"""

__version__ = "1.0.0"
__author__ = "RAG-NN Team"

from .core.pipeline import SmartRAGPipeline
from .core.health_checker import HealthChecker

__all__ = [
    'SmartRAGPipeline',
    'HealthChecker'
]
