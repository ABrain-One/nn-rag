"""
Block Validator Module

This module validates generated Python blocks by checking if they compile successfully.
Valid blocks are moved to a "block" directory for further use.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import ast
import importlib.util

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BlockValidator:
    """Validates generated Python blocks and moves valid ones to a 'block' directory."""
    
    def __init__(self, generated_dir: str = "generated_packages", block_dir: str = "block"):
        """
        Initialize the BlockValidator.
        
        Args:
            generated_dir: Directory containing generated packages
            block_dir: Directory where valid blocks will be moved
        """
        self.generated_dir = Path(generated_dir)
        self.block_dir = Path(block_dir)
        self._ensure_directories()
        
    def _ensure_directories(self):
        """Ensure both generated and block directories exist."""
        self.generated_dir.mkdir(exist_ok=True)
        self.block_dir.mkdir(exist_ok=True)
        logger.info(f"Generated directory: {self.generated_dir}")
        logger.info(f"Block directory: {self.block_dir}")
    
    def validate_single_block(self, block_name: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a single block by checking if it compiles.
        
        Args:
            block_name: Name of the block to validate (without .py extension)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        block_file = self.generated_dir / f"{block_name}.py"
        
        if not block_file.exists():
            return False, f"Block file {block_file} does not exist"
        
        try:
            # Read the file content
            with open(block_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if content is empty
            if not content.strip():
                return False, "Block file is empty"
            
            # Try to parse with AST first (syntax check)
            try:
                ast.parse(content)
            except SyntaxError as e:
                return False, f"Syntax error: {e}"
            
            # Try to compile the code
            try:
                compile(content, str(block_file), 'exec')
            except Exception as e:
                return False, f"Compilation error: {e}"
            
            # Additional validation: check for basic Python structure
            if not self._has_valid_structure(content):
                return False, "Block lacks valid Python structure (no classes/functions)"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def _has_valid_structure(self, content: str) -> bool:
        """
        Check if the content has valid Python structure.
        
        Args:
            content: Python code content
            
        Returns:
            True if content has valid structure
        """
        try:
            tree = ast.parse(content)
            
            # Check if there are any class or function definitions
            has_classes = any(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
            has_functions = any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
            
            # Also allow modules with just imports and constants
            has_imports = any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))
            has_assignments = any(isinstance(node, ast.Assign) for node in ast.walk(tree))
            
            return has_classes or has_functions or (has_imports and has_assignments)
            
        except Exception:
            return False
    
    def move_valid_block(self, block_name: str) -> bool:
        """
        Move a valid block to the block directory.
        
        Args:
            block_name: Name of the block to move
            
        Returns:
            True if move was successful
        """
        source_file = self.generated_dir / f"{block_name}.py"
        target_file = self.block_dir / f"{block_name}.py"
        
        try:
            # Move the file to block directory (not copy)
            shutil.move(str(source_file), str(target_file))
            logger.info(f"Successfully moved {block_name} to block directory")
            return True
        except Exception as e:
            logger.error(f"Failed to move {block_name}: {e}")
            return False
    
    def validate_and_move_block(self, block_name: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a block and move it if valid.
        
        Args:
            block_name: Name of the block to validate and move
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        is_valid, error = self.validate_single_block(block_name)
        
        if is_valid:
            if self.move_valid_block(block_name):
                return True, None
            else:
                return False, "Block is valid but failed to move"
        else:
            return False, error
    
    def validate_all_blocks(self) -> Dict[str, Tuple[bool, Optional[str]]]:
        """
        Validate all blocks in the generated directory.
        
        Returns:
            Dictionary mapping block names to (is_valid, error_message) tuples
        """
        results = {}
        
        for py_file in self.generated_dir.glob("*.py"):
            block_name = py_file.stem
            logger.info(f"Validating block: {block_name}")
            
            is_valid, error = self.validate_and_move_block(block_name)
            results[block_name] = (is_valid, error)
            
            if is_valid:
                logger.info(f"✓ {block_name}: Valid and moved to block directory")
            else:
                logger.warning(f"✗ {block_name}: {error}")
        
        return results
    
    def get_validation_summary(self, results: Dict[str, Tuple[bool, Optional[str]]]) -> Dict[str, int]:
        """
        Get a summary of validation results.
        
        Args:
            results: Results from validate_all_blocks()
            
        Returns:
            Summary dictionary with counts
        """
        total = len(results)
        valid = sum(1 for is_valid, _ in results.values() if is_valid)
        invalid = total - valid
        
        return {
            "total": total,
            "valid": valid,
            "invalid": invalid,
            "success_rate": (valid / total * 100) if total > 0 else 0
        }
    
    def cleanup_invalid_blocks(self, results: Dict[str, Tuple[bool, Optional[str]]]) -> int:
        """
        Remove invalid blocks from the generated directory.
        
        Args:
            results: Results from validate_all_blocks()
            
        Returns:
            Number of blocks removed
        """
        removed_count = 0
        
        for block_name, (is_valid, _) in results.items():
            if not is_valid:
                block_file = self.generated_dir / f"{block_name}.py"
                try:
                    block_file.unlink()
                    logger.info(f"Removed invalid block: {block_name}")
                    removed_count += 1
                except Exception as e:
                    logger.error(f"Failed to remove invalid block {block_name}: {e}")
        
        return removed_count


def main():
    """Main function for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate generated Python blocks")
    parser.add_argument("--generated-dir", default="generated_packages", 
                       help="Directory containing generated packages")
    parser.add_argument("--block-dir", default="block", 
                       help="Directory where valid blocks will be moved")
    parser.add_argument("--cleanup", action="store_true", 
                       help="Remove invalid blocks after validation")
    
    args = parser.parse_args()
    
    validator = BlockValidator(args.generated_dir, args.block_dir)
    
    print("Starting block validation...")
    results = validator.validate_all_blocks()
    
    # Print summary
    summary = validator.get_validation_summary(results)
    print(f"\nValidation Summary:")
    print(f"Total blocks: {summary['total']}")
    print(f"Valid blocks: {summary['valid']}")
    print(f"Invalid blocks: {summary['invalid']}")
    print(f"Success rate: {summary['success_rate']:.1f}%")
    
    # Cleanup if requested
    if args.cleanup:
        removed = validator.cleanup_invalid_blocks(results)
        print(f"Removed {removed} invalid blocks")


if __name__ == "__main__":
    main()
