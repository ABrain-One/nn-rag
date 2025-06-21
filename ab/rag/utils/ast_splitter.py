"""
Yield function / class level chunks using Python's AST.
"""
from __future__ import annotations
import ast, textwrap
from typing import Iterator

def iter_chunks(src: str) -> Iterator[str]:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            chunk = textwrap.dedent(src[node.lineno - 1 : node.end_lineno])
            yield chunk
