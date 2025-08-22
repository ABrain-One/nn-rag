#!/usr/bin/env python3
"""
Smart GitHub Client - Implements user's suggested strategy:
1. Search API + type=code for direct code search
2. Quick top 3 analysis based on criteria
3. Cache selected repos
4. Smart dependency resolution
"""

import asyncio
import aiohttp
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
import os
from pathlib import Path
from urllib.parse import quote_plus
import ast
import re

log = logging.getLogger(__name__)

@dataclass
class CodeBlock:
    """Represents a code block found via search"""
    repository: str
    path: str
    content: str
    score: float
    dependencies: List[str]
    line_count: int
    has_pytorch: bool

class SmartGitHubClient:
    """Smart GitHub client with intelligent code discovery and dependency resolution"""
    
    def __init__(self, token: str):
        self.token = token
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache_file = Path("cache/github_cache.json")
        self.cache_file.parent.mkdir(exist_ok=True)
        self.repo_cache = self._load_cache()
        
    def _load_cache(self) -> Dict[str, Dict]:
        """Load cached repository information"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to file"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.repo_cache, f, indent=2)
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={
                'Authorization': f'token {self.token}',
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'nn-rag-smart-client'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def search_code_smart(self, block_name: str) -> Optional[CodeBlock]:
        """
        Smart code search using user's strategy:
        1. Search API + type=code with enhanced queries
        2. Analyze top 3 results
        3. Pick best based on criteria
        """
        try:
            # Build robust queries generically for any block name
            search_strategies = self._build_search_queries(block_name)
            
            # Try each strategy until we find good Python code
            for i, strategy in enumerate(search_strategies):
                log.info(f"🔍 Trying search strategy {i+1}: {strategy}")
                
                encoded_q = quote_plus(strategy)
                search_url = f"https://api.github.com/search/code?q={encoded_q}&type=code&per_page=10"
                
                async with self.session.get(search_url) as response:
                    if response.status != 200:
                        log.warning(f"Search strategy {i+1} failed: {response.status}")
                        continue
                    
                    data = await response.json()
                    items = data.get('items', [])
                    
                    if not items:
                        log.info(f"Strategy {i+1}: No results found")
                        continue
                    
                    log.info(f"Strategy {i+1}: Found {len(items)} results")
                    
                    # Filter to Python files
                    python_files = [item for item in items if item.get('path', '').endswith('.py')]
                    if python_files:
                        log.info(f"✅ Strategy {i+1} yielded {len(python_files)} Python files")
                        return await self._process_search_results(python_files, block_name)
                    else:
                        log.info(f"Strategy {i+1}: No Python files after filtering")
            
            # If all strategies fail, try a fallback search
            log.warning(f"All search strategies failed for {block_name}, trying fallback")
            return await self._fallback_search(block_name)
            
        except Exception as e:
            log.error(f"Smart search failed: {e}")
            return None
    
    async def _process_search_results(self, python_files: List[Dict], block_name: str) -> Optional[CodeBlock]:
        """Process search results and find the best candidate"""
        try:
            # Analyze up to top N results
            candidates = []
            for item in python_files[:10]:  # Check more items to find good candidates
                candidate = await self._analyze_candidate(item, block_name)
                if candidate:
                    candidates.append(candidate)
            # Keep all viable candidates to improve selection
            
            if not candidates:
                return None
            
            # Pick best candidate based on criteria
            best_candidate = self._select_best_candidate(candidates)
            
            # Cache the selected repository
            self._cache_repository(best_candidate.repository, best_candidate.path)
            
            return best_candidate
            
        except Exception as e:
            log.error(f"Failed to process search results: {e}")
            return None
    
    async def _fallback_search(self, block_name: str) -> Optional[CodeBlock]:
        """Fallback search when all strategies fail"""
        try:
            # Try searching in known PyTorch repositories with broader queries
            known_repos = [
                'pytorch/vision',
                'pytorch/pytorch',
                'facebookresearch/detectron2',
                'open-mmlab/mmdetection',
                'ultralytics/yolov5',
                'rwightman/pytorch-image-models',
                'huggingface/transformers'
            ]
            
            # Try different fallback strategies
            fallback_strategies = [
                f'{block_name}',
                f'class {block_name}',
                f'{block_name} nn.Module'
            ]
            
            for repo in known_repos:
                for strategy in fallback_strategies:
                    log.info(f"🔍 Fallback: Searching {strategy} in {repo}")
                    search_query = f'repo:{repo} {strategy} language:python filename:*.py'
                    encoded_q = quote_plus(search_query)
                    search_url = f"https://api.github.com/search/code?q={encoded_q}&type=code&per_page=5"
                    
                    async with self.session.get(search_url) as response:
                        if response.status == 200:
                            data = await response.json()
                            items = data.get('items', [])
                            
                            if items:
                                python_files = [item for item in items if item['path'].endswith('.py')]
                                if python_files:
                                    log.info(f"✅ Fallback successful: {strategy} in {repo}")
                                    return await self._process_search_results(python_files, block_name)
            
            # Last resort: search for any Python file containing the name
            log.info(f"🔍 Last resort: Searching for any Python file containing {block_name}")
            search_query = f'{block_name} language:python filename:*.py size:>1000'
            encoded_q = quote_plus(search_query)
            search_url = f"https://api.github.com/search/code?q={encoded_q}&type=code&per_page=10"
            
            async with self.session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()
                    items = data.get('items', [])
                    
                    if items:
                        python_files = [item for item in items if item['path'].endswith('.py')]
                        if python_files:
                            log.info(f"✅ Last resort successful: {len(python_files)} Python files found")
                            return await self._process_search_results(python_files, block_name)
            
            log.warning(f"All fallback strategies failed for {block_name}")
            return None
            
        except Exception as e:
            log.error(f"Fallback search failed: {e}")
            return None
    
    async def _analyze_candidate(self, item: Dict, block_name: str) -> Optional[CodeBlock]:
        """Analyze a search result candidate"""
        try:
            repo = item['repository']['full_name']
            path = item['path']
            
            # Fetch file content
            content = await self._fetch_file_content(repo, path)
            if not content:
                return None
            
            # Analyze the candidate
            dependencies = self._extract_dependencies(content)
            line_count = len(content.split('\n'))
            has_pytorch = self._check_pytorch_usage(content)
            filename = os.path.basename(path)
            has_exact_class = bool(re.search(rf"^\s*class\s+{re.escape(block_name)}\b", content, re.MULTILINE))
            has_exact_def = bool(re.search(rf"^\s*def\s+{re.escape(block_name)}\b", content, re.MULTILINE))
            extends_nn_module = bool(re.search(rf"^\s*class\s+{re.escape(block_name)}\(.*?nn\.Module.*?\):", content, re.MULTILINE))
            filename_matches = block_name.lower() in filename.lower()
            
            # Calculate score (lower is better)
            score = self._calculate_score(
                dependencies=dependencies,
                line_count=line_count,
                has_pytorch=has_pytorch,
                has_exact_class=has_exact_class,
                has_exact_def=has_exact_def,
                extends_nn_module=extends_nn_module,
                filename_matches=filename_matches
            )
            
            return CodeBlock(
                repository=repo,
                path=path,
                content=content,
                score=score,
                dependencies=dependencies,
                line_count=line_count,
                has_pytorch=has_pytorch
            )
            
        except Exception as e:
            log.error(f"Failed to analyze candidate: {e}")
            return None
    
    def _calculate_score(
        self,
        dependencies: List[str],
        line_count: int,
        has_pytorch: bool,
        has_exact_class: bool,
        has_exact_def: bool,
        extends_nn_module: bool,
        filename_matches: bool
    ) -> float:
        """
        Calculate score for candidate selection (lower is better)
        Criteria: fewer dependencies, fewer lines, has PyTorch
        """
        score = 0.0
        
        # Fewer dependencies = better
        score += len(dependencies) * 0.5
        
        # Fewer lines = better (but not too few)
        if line_count < 10:
            score += 100  # Penalize very short files
        elif line_count < 100:
            score += line_count * 0.1
        else:
            score += line_count * 0.05
        
        # PyTorch usage bonus
        if not has_pytorch:
            score += 1000  # Heavily penalize non-PyTorch files
        else:
            score -= 20  # Bonus for PyTorch files
        # Exact definition bonuses
        if has_exact_class:
            score -= 80
        if extends_nn_module:
            score -= 40
        if has_exact_def:
            score -= 30
        if filename_matches:
            score -= 10
        
        return score

    def _build_search_queries(self, block_name: str) -> List[str]:
        """Build robust, generic search queries for any block name."""
        name = block_name.strip()
        queries: List[str] = []
        # Exact class/def
        queries.append(f'class {name} extension:py')
        queries.append(f'def {name} extension:py')
        # Co-occurrence with PyTorch hints
        queries.append(f'"{name}" nn.Module extension:py')
        queries.append(f'"{name}" torch.nn extension:py')
        # Filename and content
        queries.append(f'"{name}" in:file extension:py')
        queries.append(f'filename:{name}.py extension:py')
        # Generic name fallbacks
        if len(name) <= 5:
            queries.append(f'"{name}" neural network extension:py')
            queries.append(f'"{name}" deep learning extension:py')
        # Case/format variations (snake_case)
        snake = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        if snake != name:
            queries.append(f'"{snake}" extension:py')
            queries.append(f'filename:{snake}.py extension:py')
        return queries
    
    def _select_best_candidate(self, candidates: List[CodeBlock]) -> CodeBlock:
        """Select the best candidate based on score"""
        return min(candidates, key=lambda x: x.score)
    
    def _cache_repository(self, repo: str, path: str):
        """Cache repository information for future use"""
        if repo not in self.repo_cache:
            self.repo_cache[repo] = {}
        
        self.repo_cache[repo]['last_used'] = True
        self.repo_cache[repo]['successful_paths'] = self.repo_cache[repo].get('successful_paths', [])
        if path not in self.repo_cache[repo]['successful_paths']:
            self.repo_cache[repo]['successful_paths'].append(path)
        
        self._save_cache()
    
    async def _fetch_file_content(self, repo: str, path: str) -> Optional[str]:
        """Fetch file content from GitHub"""
        try:
            url = f"https://api.github.com/repos/{repo}/contents/{path}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    import base64
                    content = base64.b64decode(data['content']).decode('utf-8')
                    return content
                else:
                    log.warning(f"Failed to fetch {repo}/{path}: {response.status}")
                    return None
        except Exception as e:
            log.error(f"Error fetching file content: {e}")
            return None
    
    def _extract_dependencies(self, content: str) -> List[str]:
        """Extract dependencies from code content"""
        dependencies = []
        
        # Simple regex-based extraction (can be enhanced)
        import re
        
        # Find import statements
        import_pattern = r'^import\s+(\w+)'
        from_pattern = r'^from\s+(\w+)'
        
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('import '):
                match = re.match(import_pattern, line)
                if match:
                    dependencies.append(match.group(1))
            elif line.startswith('from '):
                match = re.match(from_pattern, line)
                if match:
                    dependencies.append(match.group(1))
        
        return dependencies
    
    def _check_pytorch_usage(self, content: str) -> bool:
        """Check if code uses PyTorch"""
        pytorch_indicators = [
            'torch', 'torch.nn', 'nn.Module', 'super().__init__()',
            'def forward', 'torch.cat', 'torch.tensor'
        ]
        
        return any(indicator in content for indicator in pytorch_indicators)
    
    async def resolve_dependency_smart(self, repo: str, file_path: str, dependency: str) -> Optional[str]:
        """
        Smart dependency resolution using user's strategy:
        1. Check same file first
        2. Recursively check imports of that file (AST-based, relative aware)
        """
        try:
            log.info(f"🔍 resolve_dependency_smart: {repo}/{file_path} -> {dependency}")
            
            # 1. Check same file first
            content = await self._fetch_file_content(repo, file_path)
            if content and dependency in content:
                log.info(f"✅ Found {dependency} in same file")
                # Extract the dependency definition from the same file
                extracted = self._extract_dependency_from_content(content, dependency)
                if extracted:
                    log.info(f"✅ Successfully extracted {dependency} from same file")
                    return extracted
                else:
                    log.info(f"❌ Failed to extract {dependency} from same file, will try imports")
            else:
                log.info(f"❌ {dependency} not found in same file")
            
            # 2. Recursively resolve through imports
            if content:
                log.info(f"🔍 Recursively searching imports for {dependency}")
                found = await self._find_dependency_via_imports(
                    repo=repo,
                    start_file_path=file_path,
                    dependency=dependency,
                    max_depth=2,
                    visited=set()
                )
                if found:
                    log.info(f"✅ Found {dependency} via imports")
                    return found
                else:
                    log.info(f"❌ {dependency} not found via imports")
            else:
                log.warning(f"❌ Could not fetch content for {file_path}")
            
            log.info(f"❌ Could not resolve {dependency}")
            return None
            
        except Exception as e:
            log.error(f"Smart dependency resolution failed: {e}")
            return None
    
    def _extract_import_paths(self, content: str, current_path: Optional[str] = None) -> List[str]:
        """Extract import paths via AST and resolve relative imports using the current file's directory.

        Returns a list of probable module file paths like 'pkg/submodule.py'.
        """
        import_paths: List[str] = []
        try:
            tree = ast.parse(content)
        except Exception:
            return []

        current_dir_parts: List[str] = []
        if current_path:
            current_dir = current_path.rsplit('/', 1)[0] if '/' in current_path else ''
            current_dir_parts = [p for p in current_dir.split('/') if p]

        def resolve_parts(module_parts: List[str], level: int) -> Optional[List[str]]:
            if level == 0:
                # Absolute import: use module_parts as-is
                return module_parts
            else:
                # Relative import: resolve against current directory
                base_parts = current_dir_parts
                if level and base_parts:
                    base_parts = base_parts[: max(0, len(base_parts) - level)]
                resolved = (base_parts or []) + (module_parts or [])
                return [p for p in resolved if p]

        seen = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module_parts = node.module.split('.') if node.module else []
                level = getattr(node, 'level', 0) or 0

                names = getattr(node, 'names', []) or []
                if module_parts:
                    base_parts = resolve_parts(module_parts, level)
                    if base_parts:
                        # Candidate: direct module file
                        cand = '/'.join(base_parts) + '.py'
                        if cand not in seen:
                            seen.add(cand)
                            import_paths.append(cand)
                        # Candidate: package __init__.py
                        pkg_init = '/'.join(base_parts + ['__init__']) + '.py'
                        if pkg_init not in seen:
                            seen.add(pkg_init)
                            import_paths.append(pkg_init)
                        # Candidates: submodules from 'from pkg import sub'
                        for alias in names:
                            alias_head = alias.name.split('.')[0]
                            sub_cand = '/'.join(base_parts + [alias_head]) + '.py'
                            if sub_cand not in seen:
                                seen.add(sub_cand)
                                import_paths.append(sub_cand)
                else:
                    # from . import submodule -> try each alias as a module file in current package level
                    for alias in names:
                        alias_name = alias.name.split('.')[0]
                        parts = resolve_parts([alias_name], level)
                        if parts:
                            path = '/'.join(parts) + '.py'
                            if path not in seen:
                                seen.add(path)
                                import_paths.append(path)
            elif isinstance(node, ast.Import):
                # Handle 'import pkg.sub as x' -> explore module file and package init
                for alias in getattr(node, 'names', []) or []:
                    mod = alias.name  # e.g., models.layers.config
                    parts = [p for p in mod.split('.') if p]
                    if parts:
                        cand = '/'.join(parts) + '.py'
                        if cand not in seen:
                            seen.add(cand)
                            import_paths.append(cand)
                        pkg_init = '/'.join(parts + ['__init__']) + '.py'
                        if pkg_init not in seen:
                            seen.add(pkg_init)
                            import_paths.append(pkg_init)

        return import_paths

    async def _find_dependency_via_imports(
        self,
        repo: str,
        start_file_path: str,
        dependency: str,
        max_depth: int = 2,
        visited: Optional[set] = None
    ) -> Optional[str]:
        """Recursively search for dependency by following import graph up to max_depth."""
        if visited is None:
            visited = set()
        if max_depth < 0:
            log.info(f"❌ Max depth reached for {dependency}")
            return None
        if start_file_path in visited:
            log.info(f"❌ Already visited {start_file_path}")
            return None
        visited.add(start_file_path)

        log.info(f"🔍 Searching {start_file_path} for {dependency} (depth: {max_depth})")
        content = await self._fetch_file_content(repo, start_file_path)
        if not content:
            log.info(f"❌ Could not fetch {start_file_path}")
            return None
        if dependency in content:
            log.info(f"✅ Found {dependency} in {start_file_path}")
            extracted = self._extract_dependency_from_content(content, dependency)
            if extracted:
                log.info(f"✅ Successfully extracted {dependency} from {start_file_path}")
                return extracted
            else:
                log.info(f"❌ Failed to extract {dependency} from {start_file_path}, will check imports")

        import_paths = self._extract_import_paths(content, current_path=start_file_path)
        log.info(f"🔍 Generated {len(import_paths)} import paths from {start_file_path}")
        
        for import_path in import_paths:
            log.info(f"🔍 Checking import path: {import_path}")
            dep_content = await self._fetch_file_content(repo, import_path)
            if dep_content and dependency in dep_content:
                log.info(f"✅ Found {dependency} in {import_path}")
                return self._extract_dependency_from_content(dep_content, dependency)
            # Recurse
            found = await self._find_dependency_via_imports(
                repo=repo,
                start_file_path=import_path,
                dependency=dependency,
                max_depth=max_depth - 1,
                visited=visited
            )
            if found:
                return found
        log.info(f"❌ {dependency} not found in any import paths from {start_file_path}")
        return None
    
    def _extract_dependency_from_content(self, content: str, dependency: str) -> Optional[str]:
        """Extract dependency definition (class/function/variable) from content using AST when possible."""
        try:
            tree = ast.parse(content)
            lines = content.split('\n')
            # Search for class, function, and variable assignments
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef) and node.name == dependency:
                    start = node.lineno - 1
                    end = getattr(node, 'end_lineno', start) - 1
                    return '\n'.join(lines[start:end + 1])
                if isinstance(node, ast.FunctionDef) and node.name == dependency:
                    start = node.lineno - 1
                    end = getattr(node, 'end_lineno', start) - 1
                    return '\n'.join(lines[start:end + 1])
                if isinstance(node, ast.Assign):
                    # Handle top-level assignments like NAME = ...
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == dependency:
                            start = node.lineno - 1
                            end = getattr(node, 'end_lineno', start) - 1
                            
                            # For assignments, try to include more context to capture related definitions
                            extracted_lines = lines[start:end + 1]
                            
                            # Look for related definitions in the same file (e.g., if cfg = __C, also include __C = ...)
                            if '=' in extracted_lines[0]:
                                # Find the right-hand side of the assignment
                                assignment_line = extracted_lines[0]
                                if '=' in assignment_line:
                                    rhs = assignment_line.split('=', 1)[1].strip()
                                    # If RHS is a simple name (like __C), look for its definition
                                    if rhs and not rhs.startswith('(') and not rhs.startswith('[') and not rhs.startswith('{'):
                                        rhs_name = rhs.split()[0]  # Get first word after =
                                        # Search for definition of rhs_name in the same file
                                        for i, line in enumerate(lines):
                                            if line.strip().startswith(f"{rhs_name} =") and i != start:
                                                # Include this definition in the extraction
                                                if i < start:
                                                    # Definition comes before assignment
                                                    extracted_lines = lines[i:end + 1]
                                                    start = i
                                                else:
                                                    # Definition comes after assignment
                                                    extracted_lines = lines[start:i + 1]
                                                    end = i
                                                break
                            
                            return '\n'.join(extracted_lines)
                if isinstance(node, ast.AnnAssign):
                    if isinstance(node.target, ast.Name) and node.target.id == dependency:
                        start = node.lineno - 1
                        end = getattr(node, 'end_lineno', start) - 1
                        return '\n'.join(lines[start:end + 1])
        except Exception:
            pass
        
        # Fallback to simple regex-based extraction
        lines = content.split('\n')
        pattern_defs = [
            rf'^\s*class\s+{re.escape(dependency)}\b',
            rf'^\s*def\s+{re.escape(dependency)}\b',
            rf'^\s*{re.escape(dependency)}\s*=\s*'
        ]
        for i, line in enumerate(lines):
            if any(re.search(p, line) for p in pattern_defs):
                start_line = i
                indent_level = len(line) - len(line.lstrip())
                end_line = start_line
                for j in range(start_line + 1, len(lines)):
                    if lines[j].strip() and (len(lines[j]) - len(lines[j].lstrip())) <= indent_level:
                        break
                    end_line = j
                return '\n'.join(lines[start_line:end_line + 1])
        return None
