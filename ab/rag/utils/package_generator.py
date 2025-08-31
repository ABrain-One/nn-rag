"""
Minimal Package Generator for PyTorch Block Extractor
Cache-based approach only
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from ab.rag.config import OUTPUT_CONFIG
import json
from datetime import datetime

class PackageGenerator:
    """Minimal package generator for extracted PyTorch blocks."""
    
    def __init__(self, base_output_dir: Optional[str] = None):
        self.base_output_dir = base_output_dir or OUTPUT_CONFIG['base_dir']
        self.metadata_file = OUTPUT_CONFIG['metadata_file']
        self.create_init = OUTPUT_CONFIG['init_template']
        self.create_requirements = OUTPUT_CONFIG['requirements_file']
        
        # Ensure output directory exists
        Path(self.base_output_dir).mkdir(parents=True, exist_ok=True)
    
    def generate_package(self, block_name: str, 
                        source_info: Dict[str, Any],
                        dependencies: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a single Python file for an extracted block.
        
        Args:
            block_name: Name of the block to generate package for
            source_info: Source file information
            dependencies: Resolved dependencies information
            
        Returns:
            Dictionary with package generation results
        """
        print(f"Generating single Python file for {block_name}")
        
        # Create output directory
        output_dir = Path(self.base_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate single Python file
        package_info = self._generate_single_python_file(
            output_dir, block_name, source_info, dependencies
        )
        
        print(f"+ Single Python file generated: {package_info['file_path']}")
        
        return {
            'block_name': block_name,
            'file_path': package_info['file_path'],
            'files_generated': package_info['files_generated'],
            'total_files': package_info['total_files']
        }
    
    def _generate_single_python_file(self, output_dir: Path, block_name: str,
                                   source_info: Dict[str, Any], 
                                   dependencies: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a single Python file containing the block."""
        try:
            # Create the output file path
            output_file = output_dir / f"{block_name}.py"
            
            # Get the block content from source
            block_content = source_info.get('content', '')
            if not block_content:
                return {
                    'file_path': str(output_file),
                    'files_generated': 0,
                    'total_files': 0,
                    'reason': 'No content available'
                }
            
            # Write the block content to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(block_content)
            
            # Create metadata
            metadata = self._create_metadata(block_name, source_info, dependencies)
            metadata_file = output_dir / self.metadata_file
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            return {
                'file_path': str(output_file),
                'files_generated': 1,
                'total_files': 1
            }
            
        except Exception as e:
            print(f"Error generating package: {e}")
            return {
                'file_path': '',
                'files_generated': 0,
                'total_files': 0,
                'reason': str(e)
            }
    
    def _create_metadata(self, block_name: str, source_info: Dict[str, Any],
                        dependencies: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for the block package."""
        metadata = {
            'block_name': block_name,
            'source': {
                'repository': source_info['repository'],
                'file_path': source_info['file_path'],
                'commit_sha': source_info.get('commit_sha', 'HEAD'),
                'url': source_info.get('url', ''),
                'extracted_at': self._get_current_timestamp()
            },
            'dependencies': {
                'total_files': dependencies.get('total_files', 0),
                'max_depth_reached': dependencies.get('max_depth_reached', False),
                'import_map': dependencies.get('import_map', {}),
                'files': list(dependencies.get('files', {}).keys())
            },
            'package': {
                'output_dir': self.base_output_dir,
                'files_generated': dependencies.get('total_files', 0),
                'compilation_status': 'pending'
            }
        }
        
        return metadata
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now().isoformat()
    
    def _identify_missing_references(self, content: str) -> List[str]:
        """Identify missing references in the content."""
        # Simple implementation - just return empty list for now
        return []
    
    def _should_skip_block(self, missing_refs: List[str], discovery_count: int) -> bool:
        """Determine if a block should be skipped."""
        # Skip if no discoveries or too many missing references
        if discovery_count == 0:
            return True
        if len(missing_refs) > 10:  # Arbitrary threshold
            return True
        return False
