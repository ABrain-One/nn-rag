import ast
import libcst as cst
from pathlib import Path

# Test file path
test_file = Path('.cache/repo_cache/open-mmlab_mmdetection/projects/EfficientDet/efficientdet/bifpn.py')

if test_file.exists():
    print(f"Testing file: {test_file}")
    content = test_file.read_text(encoding='utf-8')
    print(f"File size: {len(content)} characters")
    
    # Test AST parsing
    try:
        tree = ast.parse(content)
        print("✅ AST parsing successful")
        
        # Count classes and functions
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        print(f"Found {len(classes)} classes and {len(functions)} functions")
        
        # Check for BiFPN class
        bifpn_class = [c for c in classes if c.name == 'BiFPN']
        if bifpn_class:
            print("✅ Found BiFPN class")
        else:
            print("❌ BiFPN class not found")
            
    except Exception as e:
        print(f"❌ AST parsing failed: {e}")
    
    # Test LibCST parsing
    try:
        if len(content) <= 120_000 and "{{" not in content and "}}" not in content:
            tree = cst.parse_module(content)
            print("✅ LibCST parsing successful")
            
            # Count classes and functions
            classes = [n for n in tree.body if isinstance(n, cst.ClassDef)]
            functions = [n for n in tree.body if isinstance(n, cst.FunctionDef)]
            print(f"Found {len(classes)} classes and {len(functions)} functions")
            
            # Check for BiFPN class
            bifpn_class = [c for c in classes if c.name.value == 'BiFPN']
            if bifpn_class:
                print("✅ Found BiFPN class")
            else:
                print("❌ BiFPN class not found")
        else:
            print("⚠️ File too large or contains template syntax for LibCST")
            
    except Exception as e:
        print(f"❌ LibCST parsing failed: {e}")
        
else:
    print(f"File not found: {test_file}")
    print("Available files:")
    repo_root = Path('.cache/repo_cache/open-mmlab_mmdetection')
    if repo_root.exists():
        for f in repo_root.rglob("*.py"):
            if "bifpn" in f.name.lower():
                print(f"  {f}")
