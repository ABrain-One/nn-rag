"""
Health checker for generated code blocks
Validates syntax, imports, and PyTorch usage
"""

import os
import ast
import importlib.util
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

log = logging.getLogger(__name__)

class HealthChecker:
    """Comprehensive health checker for Python code files"""
    
    def __init__(self, output_dir: str = "output", blocks_dir: str = "blocks"):
        self.output_dir = Path(output_dir)
        self.blocks_dir = Path(blocks_dir)
        
        # Ensure directories exist
        self.output_dir.mkdir(exist_ok=True)
        self.blocks_dir.mkdir(exist_ok=True)
    
    def check_file_health(self, file_path: Path) -> Tuple[bool, Dict[str, str]]:
        """Check the health of a single file"""
        results = {}
        
        try:
            # File integrity check
            integrity_ok, integrity_msg = self._check_file_integrity(file_path)
            results['integrity'] = integrity_msg
            
            if not integrity_ok:
                return False, results
            
            # Python syntax check
            syntax_ok, syntax_msg = self._check_python_syntax(file_path)
            results['syntax'] = syntax_msg
            
            if not syntax_ok:
                return False, results
            
            # Import validation
            imports_ok, imports_msg = self._check_imports(file_path)
            results['imports'] = imports_msg
            results['imports_ok'] = imports_ok
            
            # Class definitions
            classes_ok, classes_msg = self._check_class_definitions(file_path)
            results['classes'] = classes_msg
            results['classes_ok'] = classes_ok
            
            # Function definitions
            functions_ok, functions_msg = self._check_function_definitions(file_path)
            results['functions'] = functions_msg
            results['functions_ok'] = functions_ok
            
            # PyTorch usage
            torch_ok, torch_msg = self._check_torch_usage(file_path)
            results['pytorch'] = torch_msg
            results['pytorch_ok'] = torch_ok
            
            # Executability check - can this code actually run?
            executable_ok, executable_msg = self._check_executability(file_path)
            results['executable'] = executable_msg
            results['executable_ok'] = executable_ok
            
            # Overall health assessment - must pass ALL checks
            overall_health = (
                integrity_ok and 
                syntax_ok and 
                imports_ok and 
                classes_ok and 
                functions_ok and 
                torch_ok and
                executable_ok
            )
            
            return overall_health, results
            
        except Exception as e:
            log.error(f"Health check failed for {file_path}: {e}")
            results['error'] = str(e)
            return False, results
    
    def _check_file_integrity(self, file_path: Path) -> Tuple[bool, str]:
        """Check basic file integrity"""
        try:
            if not file_path.exists():
                return False, "File does not exist"
            
            if file_path.stat().st_size == 0:
                return False, "File is empty"
            
            # Check if file is readable
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read()
            
            return True, "File exists and is readable"
            
        except Exception as e:
            return False, f"File integrity error: {e}"
    
    def _check_python_syntax(self, file_path: Path) -> Tuple[bool, str]:
        """Check if Python file has valid syntax"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to parse the AST
            ast.parse(content)
            return True, "Valid Python syntax"
            
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Error parsing file: {e}"
    
    def _check_imports(self, file_path: Path) -> Tuple[bool, str]:
        """Check if the file has valid imports that can be resolved"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse and check imports
            tree = ast.parse(content)
            imports = []
            unresolved_imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                        # Check if this import can be resolved
                        if not self._can_resolve_import(alias.name):
                            unresolved_imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                        # Check if this import can be resolved
                        if not self._can_resolve_import(node.module):
                            unresolved_imports.append(node.module)
            
            # Check for essential PyTorch imports
            essential_imports = ['torch', 'torch.nn']
            has_essential = any(imp.startswith('torch') for imp in imports)
            
            if not has_essential:
                return False, f"Missing essential PyTorch imports. Found: {imports}"
            
            # Check for unresolved imports
            if unresolved_imports:
                return False, f"Unresolved imports: {unresolved_imports}. All imports: {imports}"
            
            return True, f"Valid imports with PyTorch: {imports}"
                
        except Exception as e:
            return False, f"Error checking imports: {e}"
    
    def _can_resolve_import(self, module_name: str) -> bool:
        """Check if an import can be resolved"""
        # Built-in modules that are always available
        builtin_modules = {
            'os', 'sys', 'time', 'datetime', 'json', 'pathlib', 'typing',
            'collections', 'itertools', 'functools', 're', 'math', 'random',
            'numpy', 'torch', 'torch.nn', 'torch.nn.functional', 'torch.optim',
            'torch.utils', 'torchvision', 'PIL', 'cv2', 'matplotlib', 'seaborn'
        }
        
        # Standard library modules
        stdlib_modules = {
            'pathlib', 'typing', 'collections', 'itertools', 'functools',
            're', 'math', 'random', 'datetime', 'json', 'pickle', 'urllib'
        }
        
        # Check if it's a built-in or standard library module
        if module_name in builtin_modules or module_name in stdlib_modules:
            return True
        
        # Check if it's a PyTorch module
        if module_name.startswith('torch'):
            return True
        
        # Check if it's a common ML/vision library
        if module_name in {'numpy', 'PIL', 'cv2', 'matplotlib', 'seaborn'}:
            return True
        
        # Check if it's a relative import (starts with . or ..)
        if module_name.startswith('.'):
            return True
        
        # Check if it's an absolute import that might be repo-specific
        # These are typically unresolved and need dependency resolution
        if '.' in module_name and not module_name.startswith(('torch', 'numpy', 'PIL', 'cv2', 'matplotlib')):
            return False
        
        # For now, assume other imports are resolvable
        return True
    
    def _check_class_definitions(self, file_path: Path) -> Tuple[bool, str]:
        """Check if the file has valid class definitions"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            classes = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
            
            if classes:
                return True, f"Classes found: {classes}"
            else:
                return False, "No class definitions found"
                
        except Exception as e:
            return False, f"Error checking classes: {e}"
    
    def _check_function_definitions(self, file_path: Path) -> Tuple[bool, str]:
        """Check if the file has valid function definitions"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
            
            if functions:
                return True, f"Functions found: {functions}"
            else:
                return False, "No function definitions found"
                
        except Exception as e:
            return False, f"Error checking functions: {e}"
    
    def _check_torch_usage(self, file_path: Path) -> Tuple[bool, str]:
        """Check if the file uses PyTorch correctly"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for common PyTorch patterns
            torch_patterns = [
                'torch.nn',
                'nn.Module',
                'super().__init__()',
                'def forward',
                'torch.cat',
                'torch.tensor'
            ]
            
            found_patterns = []
            for pattern in torch_patterns:
                if pattern in content:
                    found_patterns.append(pattern)
            
            if found_patterns:
                return True, f"PyTorch patterns found: {found_patterns}"
            else:
                return False, "No PyTorch patterns detected"
                
        except Exception as e:
            return False, f"Error checking PyTorch usage: {e}"
    
    def check_all_files(self) -> Dict[str, Tuple[bool, Dict[str, str]]]:
        """Check health of all files in output directory"""
        results = {}
        
        if not self.output_dir.exists():
            log.warning(f"Output directory {self.output_dir} does not exist")
            return results
        
        python_files = list(self.output_dir.glob("*.py"))
        
        if not python_files:
            log.warning(f"No Python files found in {self.output_dir}")
            return results
        
        log.info(f"Checking health of {len(python_files)} Python files")
        
        for file_path in python_files:
            health_status, details = self.check_file_health(file_path)
            results[file_path.name] = (health_status, details)
            
            status_icon = "✅" if health_status else "❌"
            log.info(f"{status_icon} {file_path.name}: {'Healthy' if health_status else 'Unhealthy'}")
        
        return results
    
    def move_healthy_files(self, health_results: Dict[str, Tuple[bool, Dict[str, str]]]) -> Tuple[int, int]:
        """Move healthy files to blocks directory"""
        healthy_count = 0
        unhealthy_count = 0
        
        for filename, (is_healthy, details) in health_results.items():
            source_path = self.output_dir / filename
            
            if is_healthy:
                # Move to blocks directory
                target_path = self.blocks_dir / filename
                try:
                    import shutil
                    shutil.move(str(source_path), str(target_path))
                    healthy_count += 1
                    log.info(f"✅ Moved healthy file: {filename} -> blocks/")
                except Exception as e:
                    log.error(f"Failed to move {filename}: {e}")
            else:
                # Keep in output directory for inspection
                unhealthy_count += 1
                log.info(f"❌ Keeping unhealthy file: {filename} in output/")
        
        return healthy_count, unhealthy_count
    
    def generate_health_report(self, health_results: Dict[str, Tuple[bool, Dict[str, str]]]) -> str:
        """Generate a comprehensive health report"""
        report_lines = []
        report_lines.append("🚀 RAG-NN HEALTH CHECK REPORT")
        report_lines.append("=" * 60)
        
        total_files = len(health_results)
        
        if total_files == 0:
            report_lines.append("📊 SUMMARY:")
            report_lines.append("   No files found to check")
            report_lines.append("")
            return "\n".join(report_lines)
        
        healthy_files = sum(1 for _, (is_healthy, _) in health_results.items() if is_healthy)
        unhealthy_files = total_files - healthy_files
        
        report_lines.append(f"📊 SUMMARY:")
        report_lines.append(f"   Total files: {total_files}")
        report_lines.append(f"   ✅ Healthy: {healthy_files}")
        report_lines.append(f"   ❌ Unhealthy: {unhealthy_files}")
        report_lines.append(f"   Health rate: {(healthy_files/total_files)*100:.1f}%")
        report_lines.append("")
        
        # Detailed results
        report_lines.append("📋 DETAILED RESULTS:")
        report_lines.append("-" * 40)
        
        for filename, (is_healthy, details) in health_results.items():
            status_icon = "✅" if is_healthy else "❌"
            report_lines.append(f"{status_icon} {filename}")
            
            if not is_healthy:
                for check_name, result in details.items():
                    if check_name not in ['integrity', 'syntax']:  # Skip basic checks
                        report_lines.append(f"   ⚠️  {check_name}: {result}")
                report_lines.append("")
        
        return "\n".join(report_lines)
    
    def _check_executability(self, file_path: Path) -> Tuple[bool, str]:
        """Check if the code can actually be executed (missing dependencies)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST to find all used names
            tree = ast.parse(content)
            
            # Find all undefined names (used but not defined in this file)
            defined_names = set()
            used_names = set()
            
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
            
            # Find imported names
            imported_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_names.add(alias.asname if alias.asname else alias.name)
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        imported_names.add(alias.asname if alias.asname else alias.name)
            
            # Check for builtin Python names
            builtin_names = {
                'True', 'False', 'None', 'len', 'range', 'int', 'float', 'str', 'list', 'dict', 'tuple', 'set',
                'super', 'isinstance', 'hasattr', 'getattr', 'setattr', 'print', 'abs', 'max', 'min', 'sum',
                '__name__', '__main__', '__file__', '__doc__', '__version__', '__author__',
                # Common exceptions
                'Exception', 'BaseException', 'TypeError', 'ValueError', 'AttributeError', 'KeyError', 'IndexError',
                'RuntimeError', 'NameError', 'ImportError', 'ModuleNotFoundError', 'FileNotFoundError', 'OSError',
                'AssertionError', 'NotImplementedError', 'StopIteration', 'GeneratorExit', 'KeyboardInterrupt',
                # Common built-in functions and types
                'object', 'type', 'property', 'staticmethod', 'classmethod', 'enumerate', 'zip', 'map', 'filter',
                'reversed', 'sorted', 'any', 'all', 'chr', 'ord', 'bin', 'hex', 'oct', 'format', 'repr',
                'eval', 'exec', 'compile', 'globals', 'locals', 'vars', 'dir', 'callable', 'hash', 'id',
                # Common attributes and special methods
                '__init__', '__new__', '__del__', '__call__', '__getattr__', '__setattr__', '__getitem__',
                '__setitem__', '__delitem__', '__len__', '__iter__', '__next__', '__enter__', '__exit__',
                '__str__', '__repr__', '__format__', '__hash__', '__eq__', '__ne__', '__lt__', '__le__',
                '__gt__', '__ge__', '__add__', '__sub__', '__mul__', '__truediv__', '__floordiv__',
                '__mod__', '__pow__', '__lshift__', '__rshift__', '__and__', '__or__', '__xor__',
                '__radd__', '__rsub__', '__rmul__', '__rtruediv__', '__rfloordiv__', '__rmod__',
                '__rpow__', '__rlshift__', '__rrshift__', '__rand__', '__ror__', '__rxor__'
            }
            
            # Collect function parameter names from all functions (incl. *args/**kwargs)
            function_parameter_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Positional and keyword-only args
                    for arg in list(getattr(node.args, 'args', [])) + list(getattr(node.args, 'kwonlyargs', [])):
                        if hasattr(arg, 'arg'):
                            function_parameter_names.add(arg.arg)
                    # *args
                    if getattr(node.args, 'vararg', None) is not None and hasattr(node.args.vararg, 'arg'):
                        function_parameter_names.add(node.args.vararg.arg)
                    # **kwargs
                    if getattr(node.args, 'kwarg', None) is not None and hasattr(node.args.kwarg, 'arg'):
                        function_parameter_names.add(node.args.kwarg.arg)

            # Find undefined names
            undefined_names = used_names - defined_names - imported_names - builtin_names - function_parameter_names
            
            # Remove obvious non-issues
            undefined_names = {name for name in undefined_names if not name.startswith('self')}
            
            # Remove constructor parameters (redundant with generic function handling, kept for clarity)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                    for arg in getattr(node.args, 'args', []):
                        if hasattr(arg, 'arg') and arg.arg != 'self':
                            undefined_names.discard(arg.arg)
            
            # Remove loop variables (they're defined in for loops)
            for node in ast.walk(tree):
                if isinstance(node, ast.For):
                    if hasattr(node.target, 'id'):
                        undefined_names.discard(node.target.id)
                    elif isinstance(node.target, ast.Name):
                        undefined_names.discard(node.target.id)
            
            if undefined_names:
                # No hardcoded critical names; base severity purely on scale
                if len(undefined_names) > 10:
                    return False, f"Too many undefined names ({len(undefined_names)}): {sorted(list(undefined_names)[:10])}..."
                else:
                    return False, f"Undefined names: {sorted(undefined_names)}"
            
            return True, "All dependencies appear to be satisfied"
            
        except Exception as e:
            return False, f"Executability check error: {e}"
