from __future__ import annotations
import ast
from typing import Iterable, List, Tuple
from .manual_specs import INTERNAL_PREFIXES

def _is_internal_module(mod: str | None, level: int) -> bool:
    if level and level > 0:
        return True
    if not mod:
        return False
    return mod.startswith(INTERNAL_PREFIXES)

def split_imports_and_body(src: str) -> tuple[list[str], str]:
    try:
        tree = ast.parse(src, type_comments=True)
    except SyntaxError:
        return [], src

    keep_lines: set[int] = set()
    drop_lines: set[int] = set()

    for node in tree.body:
        if isinstance(node, ast.Import):
            drop_this = False
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root.startswith(INTERNAL_PREFIXES):
                    drop_this = True
                    break
            if drop_this:
                drop_lines.update(range(node.lineno, node.end_lineno + 1))
            else:
                keep_lines.update(range(node.lineno, node.end_lineno + 1))

        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if _is_internal_module(mod, node.level):
                drop_lines.update(range(node.lineno, node.end_lineno + 1))
            else:
                keep_lines.update(range(node.lineno, node.end_lineno + 1))

    lines = src.splitlines()
    kept_imports = [lines[i - 1] for i in sorted(keep_lines)]
    body_lines = [
        lines[i]
        for i in range(len(lines))
        if (i + 1) not in keep_lines and (i + 1) not in drop_lines
    ]
    while kept_imports and not kept_imports[-1].strip():
        kept_imports.pop()
    body = "\n".join(body_lines).strip("\n") + "\n"
    return kept_imports, body

def dedup_preserve(seq: Iterable[str]) -> list[str]:
    seen = set(); out = []
    for s in seq:
        if s not in seen:
            out.append(s); seen.add(s)
    return out
