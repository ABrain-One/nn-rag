#!/usr/bin/env python3
"""
Smart Dependency Resolver - Uses the smart GitHub client for efficient dependency resolution
"""

import asyncio
import logging
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
import ast
import re

from .github_client import SmartGitHubClient, CodeBlock

log = logging.getLogger(__name__)

class SmartDependencyResolver:
    """Smart dependency resolver using intelligent GitHub search and caching"""
    
    def __init__(self, github_token: str):
        self.github_token = github_token
        self.resolved_dependencies = {}
        
    async def resolve_block_smart(self, block_name: str) -> Optional[CodeBlock]:
        """
        Smart block resolution:
        1. Search for code directly
        2. Pick best candidate
        3. Cache for future use
        """
        try:
            async with SmartGitHubClient(self.github_token) as client:
                log.info(f"🔍 Smart searching for: {block_name}")
                
                # Search for the block
                code_block = await client.search_code_smart(block_name)
                
                if code_block:
                    log.info(f"✅ Found {block_name} in {code_block.repository}/{code_block.path}")
                    log.info(f"   📊 Score: {code_block.score:.2f}")
                    log.info(f"   📏 Lines: {code_block.line_count}")
                    log.info(f"   🔗 Dependencies: {len(code_block.dependencies)}")
                    log.info(f"   🚀 PyTorch: {code_block.has_pytorch}")
                    
                    return code_block
                else:
                    log.warning(f"❌ No suitable code found for {block_name}")
                    return None
                    
        except Exception as e:
            log.error(f"Smart resolution failed for {block_name}: {e}")
            return None
    
    async def resolve_dependencies_smart(self, code_block: CodeBlock, client: SmartGitHubClient) -> Dict[str, str]:
        """
        Smart dependency resolution with recursive resolution for nested dependencies
        """
        try:
            log.info(f"🔍 Starting smart dependency resolution for {code_block.path}")
            
            # Get initial missing dependencies
            missing_deps_categorized = self._extract_missing_dependencies(code_block.content)
            total_deps = sum(len(deps) for deps in missing_deps_categorized.values())
            
            if total_deps == 0:
                log.info("✅ No missing dependencies found")
                return {}
            
            log.info(f"📊 Found {total_deps} missing dependencies: {missing_deps_categorized}")
            
            # Initialize tracking variables
            resolved_deps = {}
            unresolved_deps = set()
            pending_recursive_resolution = set()
            
            # First pass: resolve all initial dependencies
            for category, deps in missing_deps_categorized.items():
                for dep in deps:
                    log.info(f"🔍 Resolving {category} dependency: {dep}")
                    dep_content = await client.resolve_dependency_smart("ShiHanQ/GFANet", code_block.path, dep)
                    if dep_content:
                        resolved_deps[dep] = dep_content
                        log.info(f"✅ Resolved {category}: {dep}")
                        
                        # Check if this resolved dependency introduces new undefined names
                        log.info(f"🔍 Checking if {dep} introduces new undefined names...")
                        log.info(f"🔍 Content length: {len(dep_content)}")
                        log.info(f"🔍 Content preview: {dep_content[:200]}...")
                        new_undefined = self._extract_new_undefined_names(dep_content)
                        log.info(f"🔍 Found new undefined names from {dep}: {new_undefined}")
                        if new_undefined:
                            log.info(f"🔄 Resolved {dep} introduces new undefined names: {new_undefined}")
                            pending_recursive_resolution.update(new_undefined)
                        else:
                            log.info(f"✅ {dep} does not introduce new undefined names")
                    else:
                        unresolved_deps.add(dep)
                        log.warning(f"❌ Could not resolve {category}: {dep}")
            
            # Recursive resolution for nested dependencies
            max_recursive_depth = 3
            current_depth = 0
            
            while pending_recursive_resolution and current_depth < max_recursive_depth:
                current_depth += 1
                log.info(f"🔄 Recursive resolution pass {current_depth}, pending: {pending_recursive_resolution}")
                
                current_pending = pending_recursive_resolution.copy()
                pending_recursive_resolution.clear()
                
                for dep in current_pending:
                    log.info(f"🔄 Recursively resolving nested dependency: {dep}")
                    dep_content = await client.resolve_dependency_smart("ShiHanQ/GFANet", code_block.path, dep)
                    if dep_content:
                        resolved_deps[dep] = dep_content
                        log.info(f"✅ Recursively resolved: {dep}")
                        
                        # Check if this resolved dependency introduces new undefined names
                        try:
                            new_undefined = self._extract_new_undefined_names(dep_content)
                            if new_undefined:
                                log.info(f"🔄 {dep} introduces new undefined names: {new_undefined}")
                                pending_recursive_resolution.update(new_undefined)
                        except Exception as e:
                            log.error(f"Failed to extract new undefined names: {e}")
                    else:
                        log.warning(f"❌ Could not recursively resolve: {dep}")
            
            if pending_recursive_resolution:
                log.warning(f"⚠️  Reached max recursive depth, remaining unresolved: {pending_recursive_resolution}")
                unresolved_deps.update(pending_recursive_resolution)
            
            log.info(f"📊 Resolution summary: {len(resolved_deps)}/{total_deps} dependencies resolved")
            if unresolved_deps:
                log.warning(f"❌ Unresolved dependencies: {unresolved_deps}")
            
            return resolved_deps
            
        except Exception as e:
            log.error(f"Dependency resolution failed: {e}")
            return {}
    
    def _extract_missing_dependencies(self, content: str) -> Dict[str, List[str]]:
        """Extract missing dependencies categorized by priority"""
        try:
            tree = ast.parse(content)
            
            # Collect all used names
            used_names = set()
            defined_names = set()
            
            for node in ast.walk(tree):
                # Collect defined names
                if isinstance(node, ast.ClassDef):
                    defined_names.add(node.name)
                elif isinstance(node, ast.FunctionDef):
                    defined_names.add(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            defined_names.add(target.id)
                
                # Collect used names
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    used_names.add(node.id)
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        used_names.add(node.func.id)
            
            # Find imported names but exclude external repo imports from being marked as "available"
            imported_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # Only mark as imported if it's not a repo-specific module
                        module = alias.name.split('.')[0]
                        if not self._is_repo_specific_module(module):
                            imported_names.add(alias.asname if alias.asname else alias.name)
                elif isinstance(node, ast.ImportFrom):
                    # For 'from X import Y', only mark Y as imported if X is not repo-specific
                    if node.module and not self._is_repo_specific_module(node.module):
                        for alias in node.names:
                            imported_names.add(alias.asname if alias.asname else alias.name)
                    # If it's repo-specific, the imported symbols need to be resolved
            
            # Built-in names
            builtin_names = {
                'True', 'False', 'None', 'len', 'range', 'int', 'float', 'str', 'list', 'dict', 'tuple', 'set',
                'super', 'isinstance', 'hasattr', 'getattr', 'setattr', 'print', 'abs', 'max', 'min', 'sum',
                '__name__', '__main__', '__file__', '__doc__', '__version__', '__author__'
            }
            
            # Find undefined names
            undefined_names = used_names - defined_names - imported_names - builtin_names
            
            # Remove obvious non-issues
            undefined_names = {name for name in undefined_names if not name.startswith('self')}
            
            # Remove constructor parameters
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                    for arg in node.args.args:
                        if hasattr(arg, 'arg') and arg.arg != 'self':
                            undefined_names.discard(arg.arg)
            
            # Remove loop variables
            for node in ast.walk(tree):
                if isinstance(node, ast.For):
                    if hasattr(node.target, 'id'):
                        undefined_names.discard(node.target.id)
                    elif isinstance(node.target, ast.Name):
                        undefined_names.discard(node.target.id)
            
            # Categorize dependencies by priority
            categorized = self._categorize_dependencies(list(undefined_names))
            
            return categorized
            
        except Exception as e:
            log.error(f"Failed to extract dependencies: {e}")
            return {"critical": [], "important": [], "optional": []}
    
    def _extract_new_undefined_names(self, content: str) -> Set[str]:
        """Extract undefined names from resolved dependency content that need further resolution"""
        try:
            tree = ast.parse(content)
            undefined_names = set()
            
            # Built-in names that are always available
            builtin_names = {
                'True', 'False', 'None', 'len', 'range', 'int', 'float', 'str', 'list', 'dict', 'tuple', 'set',
                'super', 'isinstance', 'hasattr', 'getattr', 'setattr', 'print', 'abs', 'max', 'min', 'sum',
                '__name__', '__main__', '__file__', '__doc__', '__version__', '__author__', '__init__', '__new__',
                '__call__', '__getattr__', '__setattr__', '__getitem__', '__setitem__', '__contains__', '__iter__',
                '__next__', '__enter__', '__exit__', '__repr__', '__str__', '__eq__', '__ne__', '__lt__', '__le__',
                '__gt__', '__ge__', '__hash__', '__bool__', '__len__', '__get__', '__set__', '__delete__'
            }
            
            # Common parameter names that don't need resolution
            param_names = {'self', 'cls', 'args', 'kwargs', 'other', 'key', 'value', 'item', 'obj', 'instance'}
            
            # Common loop/context variables
            context_names = {'i', 'j', 'k', 'n', 'm', 'x', 'y', 'z', 'idx', 'index', 'count', 'num', 'item', 'element'}
            
            # Common exception names
            exception_names = {'e', 'error', 'exc', 'exception', 'err', 'msg', 'message'}
            
            # Skip these categories entirely
            skip_categories = builtin_names | param_names | context_names | exception_names
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    name = node.id
                    
                    # Skip names that don't need resolution
                    if name in skip_categories:
                        continue
                    
                    # Special case: __C and similar names are clearly dependencies
                    if name.startswith('__') and name.endswith('__') and len(name) > 2:
                        if name not in builtin_names:  # Double-check it's not a built-in
                            undefined_names.add(name)
                        continue
                    
                    # Skip names that are clearly local/parameter variables
                    if len(name) <= 2 and name.islower():  # Very short names like 'v', 'a', 'b'
                        continue
                    
                    # Skip names that look like local variables
                    if name.startswith('_') and len(name) <= 3:  # Very short private names
                        continue
                    
                    # Only include names that look like they might be external dependencies
                    # These are typically: PascalCase classes, snake_case functions, or ALL_CAPS constants
                    if (name[0].isupper() or  # PascalCase classes
                        '_' in name or         # snake_case functions
                        name.isupper() or      # ALL_CAPS constants
                        len(name) > 3):        # Longer names that might be meaningful
                        
                        # Additional filtering for common patterns that don't need resolution
                        if not any(pattern in name.lower() for pattern in ['temp', 'tmp', 'var', 'val', 'arg', 'param']):
                            undefined_names.add(name)
            
            return undefined_names
            
        except Exception as e:
            log.error(f"Failed to extract new undefined names: {e}")
            return set()
    
    def _categorize_dependencies(self, dependencies: List[str]) -> Dict[str, List[str]]:
        """Categorize dependencies by priority"""
        import re
        
        critical = []
        important = []
        optional = []
        
        for dep in dependencies:
            if self._is_critical_dependency(dep):
                critical.append(dep)
            elif self._is_important_dependency(dep):
                important.append(dep)
            else:
                optional.append(dep)
        
        return {
            "critical": critical,
            "important": important,
            "optional": optional
        }
    
    def _is_critical_dependency(self, dep: str) -> bool:
        """Check if dependency is critical for execution"""
        import re
        
        # Critical patterns: PascalCase classes, snake_case functions, ALL_CAPS constants
        # Also include short lowercase names that are likely configuration objects
        critical_patterns = [
            r'^[A-Z][a-z]+$',  # PascalCase classes
            r'^[a-z]+_[a-z]+$',  # snake_case functions
            r'^[A-Z]+$',  # ALL_CAPS constants
            r'^[A-Z][a-z]+[A-Z][a-z]+$',  # MultiWord classes
            r'^[a-z]{1,4}$'  # Short lowercase names (like 'cfg', 'opt', 'args')
        ]
        
        return any(re.match(pattern, dep) for pattern in critical_patterns)
    
    def _is_important_dependency(self, dep: str) -> bool:
        """Check if dependency is important but not critical"""
        import re
        
        # Important patterns: camelCase, mixed case
        important_patterns = [
            r'^[a-z]+[A-Z][a-z]+$',  # camelCase
            r'^[A-Z][a-z]+[A-Z]',  # Mixed case starting with capital
        ]
        
        return any(re.match(pattern, dep) for pattern in important_patterns)
    
    def _is_repo_specific_module(self, module_name: str) -> bool:
        """Check if this is a repo-specific module that needs resolution."""
        # Common patterns for repo-specific imports
        repo_patterns = [
            'models',
            'utils', 
            'config',
            'helpers',
            'lib',
            'src',
            'core',
            'common'
        ]
        return any(module_name.startswith(pattern) for pattern in repo_patterns)
    
    async def create_executable_block(self, code_block: CodeBlock, resolved_deps: Dict[str, str]) -> str:
        """
        Create an executable block by combining original code with resolved dependencies
        """
        try:
            # Start with essential imports
            essential_imports = [
                "import torch",
                "import torch.nn as nn",
                "import torch.nn.functional as F"
            ]
            
            # Get missing dependencies for categorization
            missing_deps_categorized = self._extract_missing_dependencies(code_block.content)
            
            # Add resolved dependencies with priority labels
            dependency_code = []
            if resolved_deps:
                dependency_code.append("# Resolved dependencies")
                
                # Show all resolved dependencies, not just those from original missing dependencies
                # Group by priority for display
                critical_deps = []
                important_deps = []
                optional_deps = []
                
                for dep_name, dep_content in resolved_deps.items():
                    if self._is_critical_dependency(dep_name):
                        critical_deps.append((dep_name, dep_content))
                    elif self._is_important_dependency(dep_name):
                        important_deps.append((dep_name, dep_content))
                    else:
                        optional_deps.append((dep_name, dep_content))
                
                # Add critical dependencies
                if critical_deps:
                    dependency_code.append("\n# 🚨 Critical Dependencies")
                    for dep_name, dep_content in critical_deps:
                        dependency_code.append(f"# {dep_name} implementation")
                        dependency_code.append(dep_content)
                        dependency_code.append("")
                
                # Add important dependencies
                if important_deps:
                    dependency_code.append("# ⚠️  Important Dependencies")
                    for dep_name, dep_content in important_deps:
                        dependency_code.append(f"# {dep_name} implementation")
                        dependency_code.append(dep_content)
                        dependency_code.append("")
                
                # Add optional dependencies
                if optional_deps:
                    dependency_code.append("# 💡 Optional Dependencies")
                    for dep_name, dep_content in optional_deps:
                        dependency_code.append(f"# {dep_name} implementation")
                        dependency_code.append(dep_content)
                        dependency_code.append("")
            
            # Do not generate placeholders; keep code minimal and accurate.
            
            # Strip import lines for resolved symbols only
            cleaned_original = self._remove_resolved_imports(code_block.content, set(resolved_deps.keys()))

            # Combine everything
            final_code = "\n".join([
                "# Essential PyTorch imports",
                "\n".join(essential_imports),
                "",
                "# Dependency implementations",
                "\n".join(dependency_code),
                "",
                "# Original code block",
                cleaned_original
            ])
            
            return final_code
            
        except Exception as e:
            log.error(f"Failed to create executable block: {e}")
            return code_block.content

    def _remove_resolved_imports(self, content: str, resolved_names: set) -> str:
        """Remove import lines only for symbols we've actually resolved and inlined."""
        try:
            lines = content.split('\n')
            new_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('from ') and ' import ' in stripped:
                    # Parse 'from X import a, b, c'
                    try:
                        head, tail = stripped.split(' import ', 1)
                        names = [n.strip() for n in tail.split(',')]
                        remaining = [n for n in names if n.split(' as ')[0] not in resolved_names]
                        if not remaining:
                            # Drop the entire import line only if ALL symbols were resolved
                            continue
                        else:
                            # Rebuild the line with remaining names
                            rebuilt = f"{head} import {', '.join(remaining)}"
                            # Preserve original indentation
                            indent = line[: len(line) - len(line.lstrip())]
                            new_lines.append(indent + rebuilt)
                            continue
                    except Exception:
                        pass
                new_lines.append(line)
            return '\n'.join(new_lines)
        except Exception:
            return content
    
    def _create_minimal_placeholder(self, dep_name: str) -> str:
        """Create minimal placeholder for unresolved critical dependencies"""
        if dep_name[0].isupper():  # Class
            return f"class {dep_name}:\n    def __init__(self, *args, **kwargs):\n        pass\n    def forward(self, *args, **kwargs):\n        return None"
        else:  # Function
            return f"def {dep_name}(*args, **kwargs):\n        return None"
    
    def _create_smart_placeholder(self, dep_name: str) -> str:
        """Create smart placeholder for unresolved critical dependencies"""
        if dep_name[0].isupper():  # Class
            # Create a more intelligent class placeholder
            return f"""class {dep_name}:
    \"\"\"Smart placeholder for {dep_name} class\"\"\"
    
    def __init__(self, *args, **kwargs):
        # Initialize with default values for common PyTorch patterns
        if 'in_channels' in kwargs:
            self.in_channels = kwargs['in_channels']
        if 'out_channels' in kwargs:
            self.out_channels = kwargs['out_channels']
        if 'kernel_size' in kwargs:
            self.kernel_size = kwargs['kernel_size']
    
    def forward(self, x, *args, **kwargs):
        # Return input as-is for now (identity function)
        # This allows the code to run without errors
        return x
    
    def __repr__(self):
        return f'{dep_name}(placeholder=True)'"""
        else:  # Function
            # Create a more intelligent function placeholder
            dep_lower = dep_name.lower()
            if 'distance' in dep_lower:
                return f"def {dep_name}(*args, **kwargs):\n    return torch.tensor(0.0)  # Default distance"
            elif 'velocity' in dep_lower:
                return f"def {dep_name}(*args, **kwargs):\n    return torch.tensor(0.0)  # Default velocity"
            elif 'weight' in dep_lower:
                return f"def {dep_name}(*args, **kwargs):\n    return torch.tensor(1.0)  # Default weight"
            else:
                return f"def {dep_name}(*args, **kwargs):\n    return torch.tensor(0.0)  # Generic default"
    
    async def process_block_complete(self, block_name: str) -> Tuple[bool, str, Dict]:
        """
        Complete block processing pipeline:
        1. Find block
        2. Resolve dependencies
        3. Create executable code
        """
        try:
            log.info(f"🚀 Starting complete processing for: {block_name}")
            
            # Step 1: Find the block
            code_block = await self.resolve_block_smart(block_name)
            if not code_block:
                return False, "", {"error": "Block not found"}
            
            # Step 2: Resolve dependencies
            async with SmartGitHubClient(self.github_token) as client:
                resolved_deps = await self.resolve_dependencies_smart(code_block, client)
            
            # Step 3: Create executable code
            executable_code = await self.create_executable_block(code_block, resolved_deps)
            
            # Return results
            results = {
                "repository": code_block.repository,
                "path": code_block.path,
                "dependencies_resolved": len(resolved_deps),
                "total_dependencies": len(code_block.dependencies),
                "line_count": code_block.line_count,
                "has_pytorch": code_block.has_pytorch,
                "score": code_block.score
            }
            
            return True, executable_code, results
            
        except Exception as e:
            log.error(f"Complete processing failed for {block_name}: {e}")
            return False, "", {"error": str(e)}
