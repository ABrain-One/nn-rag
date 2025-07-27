from __future__ import annotations
import ast
import logging
from typing import List, Tuple

from ..utils.cache import load, save
from ..utils.github_utils import search_code, fetch_raw
from ..utils.query_helpers import extract_class

log = logging.getLogger(__name__)

def gh_hits(query: str) -> list[dict]:
    key = f"gh::{query}"
    hits = load(key)
    if hits is None:
        log.info("Searching GitHub for: %s", query)
        hits = search_code(query)
        save(key, hits)
    return hits

def fetch_body(repo: str, path: str) -> str:
    key = f"body::{repo}::{path}"
    body = load(key)
    if body is None:
        body = fetch_raw(repo, path)
        save(key, body)
    return body.replace("\x00", "")

def extract_named_functions(src: str, names: List[str]) -> list[str]:
    try:
        tree = ast.parse(src, type_comments=True)
    except SyntaxError:
        return []
    wanted = set(names)
    segs: list[tuple[int, int]] = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in wanted:
            if hasattr(n, "lineno") and hasattr(n, "end_lineno") and n.end_lineno is not None:
                segs.append((n.lineno, n.end_lineno))
    if not segs:
        return []
    lines = src.splitlines()
    out: list[str] = []
    for lo, hi in sorted(segs):
        out.append("\n".join(lines[lo - 1:hi]))
    return out

def extract_named_classes(src: str, names: List[str]) -> list[str]:
    out: list[str] = []
    for cls in names:
        try:
            out.append(extract_class(src, cls))
        except Exception:
            pass
    return out
