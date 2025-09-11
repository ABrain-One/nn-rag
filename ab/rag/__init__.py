"""
Neural Network RAG - Block Extraction and Retrieval System

This package provides tools for extracting, analyzing, and retrieving neural network blocks
from various PyTorch-based repositories.

Main Components:
- BlockExtractor: Main class for extracting neural network blocks with dependency resolution
- BlockValidator: Validates extracted blocks and moves them to the block directory
- Various utilities for package generation, repository caching, and definition resolution

API Usage:
    from ab.rag.extract_blocks import BlockExtractor
    
    # Initialize extractor
    extractor = BlockExtractor()
    
    # Extract a single block
    result = extractor.extract_single_block("ResNet")
    
    # Extract multiple blocks
    results = extractor.extract_multiple_blocks(["ResNet", "VGG", "DenseNet"])
    
    # Extract from JSON file
    results = extractor.extract_blocks_from_file("nn_block_names.json")
    
    # Validate a block
    validation = extractor.validate_block("ResNet")
    
    # Get extraction statistics
    stats = extractor.get_extraction_stats()
"""

from .extract_blocks import BlockExtractor
from .block_validator import BlockValidator

__all__ = ["BlockExtractor", "BlockValidator"]

