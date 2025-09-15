#!/usr/bin/env python3
"""
Build a JSON array of *neural network block* names (classes inheriting from nn.Module).

Strict requirements:
  - Top-level classes only.
  - Must define a 'forward' method.
  - Must inherit from Module-like classes:
      Direct: 'Module', 'nn.Module', 'torch.nn.Module'
      BaseModule: 'BaseModule', 'mmengine.model.BaseModule', 'mmcv.runner.BaseModule'
      Or any class ending with '.Module' or '.BaseModule'

No network calls. Uses your local clones (via RepoCache if available, otherwise best-effort path guesses).
Outputs a *names-only* JSON array, deduped & sorted.

Usage:
  python make_nn_block_names.py --config repo_config.json --out nn_block_names.json
Options:
  --include-registered       (Deprecated - registry decorators no longer sufficient)
  --allow-patterns REGEX...  (Deprecated - name patterns no longer sufficient)
  --strict-bases             (Deprecated - now always strict)
  --public-only              Drop names starting with "_"
  --no-paths-only            If a repo has no "paths" configured, scan whole repo (slower)
  --extra-globs GLOB [...]   Extra globs relative to repo root (e.g., "mmdet/**/*.py")
  --min-total N              Exit with nonzero if fewer than N names were found
  --quiet                    Less logging
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures as cf
import json
import logging
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

log = logging.getLogger("nn_blocks")

# ---------------- Repo roots (no network) ----------------

def _try_repo_cache():
    try:
        from ab.rag.utils.repo_cache import RepoCache  # your local helper used elsewhere
        return RepoCache()
    except Exception:
        return None

def _repo_root_from_cache(cache, repo_name: str) -> Optional[Path]:
    if cache is None:
        return None
    try:
        # Do NOT pull or update; just return existing cached path
        return cache.get_cached_repo(repo_name)
    except Exception:
        return None

def _guess_local_repo_root(repo_name: str) -> Optional[Path]:
    owner, name = repo_name.split("/", 1)
    # Get the script's directory to make paths relative to it
    from .utils.path_resolver import get_cache_dir
    script_dir = Path(__file__).parent
    cache_dir = get_cache_dir()
    candidates = [
        cache_dir / "repo_cache" / f"{owner}_{name}",  # Default repo cache location
        script_dir / "repo_cache" / f"{owner}_{name}",  # Fallback: relative to script
        Path("ab/rag/repo_cache") / f"{owner}_{name}",  # Fallback: relative to current working directory
        Path("repos") / owner / name,
        Path(".cache") / "repos" / owner / name,
        Path("external") / owner / name,
        Path(name),
    ]
    
    # Debug logging
    log.debug("Checking paths for %s:", repo_name)
    for i, p in enumerate(candidates):
        log.debug("  %d: %s (exists: %s, has .git: %s)", i, p, p.exists(), (p / ".git").exists())
    
    for p in candidates:
        if (p / ".git").exists():
            log.debug("Found repo at: %s", p)
            return p.resolve()
        if p.exists() and any(p.rglob("*.py")):
            log.debug("Found repo at: %s (no .git but has Python files)", p)
            return p.resolve()
    
    log.debug("No repo found for %s", repo_name)
    return None

# ---------------- AST helpers ----------------

def _expr_to_dotted(e: ast.AST) -> str:
    """Best-effort dotted name for base/decorator expressions."""
    if isinstance(e, ast.Name):
        return e.id
    if isinstance(e, ast.Attribute):
        parts = []
        cur = e
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        parts.reverse()
        return ".".join(parts)
    if isinstance(e, ast.Subscript):
        # Handle typing generics like BaseModule[Foo]
        return _expr_to_dotted(e.value)
    if isinstance(e, ast.Call):
        return _expr_to_dotted(e.func)
    return ""

MODULE_LIKE_BASES = {
    "Module",
    "nn.Module",
    "torch.nn.Module",
}

# Abstract base classes that should not be discovered as blocks
ABSTRACT_BASE_CLASSES = {
    "BaseModule",
    "mmengine.model.BaseModule",
    "mmcv.runner.BaseModule",
}

_OPENMMLAB_REG_DECORATOR_SUFFIX = "register_module"

_BLOCK_NAME_HINTS = re.compile(
    r"(Block|Bottleneck|Stem|Head|Neck|Backbone|Encoder|Decoder|Stage|Layer|"
    r"Attention|Transformer|Mixer|Residual|ResBlock|Conv|PatchEmbed|Downsample|Upsample|MLP)$",
    re.IGNORECASE,
)

def _has_forward(class_node: ast.ClassDef) -> bool:
    for n in class_node.body:
        if isinstance(n, ast.FunctionDef) and n.name == "forward":
            return True
    return False

def _bases_include_module_like(class_node: ast.ClassDef) -> bool:
    """Check if class inherits from nn.Module or Module (strict check)."""
    for b in class_node.bases:
        dotted = _expr_to_dotted(b)
        if not dotted:
            continue
        # Strict check: must be exactly Module, nn.Module, torch.nn.Module, or end with .Module or .BaseModule
        if (dotted in MODULE_LIKE_BASES or 
            dotted.endswith(".Module") or 
            dotted.endswith(".BaseModule") or
            dotted == "BaseModule"):
            return True
    return False

def _is_abstract_base_class(class_node: ast.ClassDef) -> bool:
    """Check if class is an abstract base class that should not be discovered."""
    # Check if the class name itself is an abstract base class
    if class_node.name in ABSTRACT_BASE_CLASSES:
        return True
    
    # Check if class inherits from any BaseModule variant
    for b in class_node.bases:
        dotted = _expr_to_dotted(b)
        if not dotted:
            continue
        if dotted in ABSTRACT_BASE_CLASSES or dotted.endswith(".BaseModule"):
            return True
    
    # Check for metaclass=ABCMeta (indicates abstract base class)
    for keyword in class_node.keywords:
        if keyword.arg == "metaclass":
            dotted = _expr_to_dotted(keyword.value)
            if dotted and "ABCMeta" in dotted:
                return True
    
    return False

def _has_openmmlab_register_decorator(class_node: ast.ClassDef) -> bool:
    for d in class_node.decorator_list:
        dotted = _expr_to_dotted(d)
        if dotted and dotted.endswith(_OPENMMLAB_REG_DECORATOR_SUFFIX):
            return True
    return False

def _looks_like_block_name(name: str) -> bool:
    return bool(_BLOCK_NAME_HINTS.search(name))

# ---------------- Per-file scan ----------------

def _iter_nn_block_names_from_file(
    py_path: Path,
    public_only: bool,
    include_registered: bool,
    strict_bases: bool,
    allow_patterns: List[re.Pattern],
) -> List[str]:
    try:
        text = py_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(text)
    except Exception:
        return []

    names: List[str] = []

    def is_public(n: str) -> bool:
        return not n.startswith("_")

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue

        cls_name = node.name
        if public_only and not is_public(cls_name):
            continue

        has_fwd = _has_forward(node)
        module_like = _bases_include_module_like(node)
        is_abstract = _is_abstract_base_class(node)

        # Strict criteria: Must inherit from Module-like class AND have forward() method AND not be abstract base class
        criteria = has_fwd and module_like and not is_abstract

        if criteria:
            names.append(cls_name)

    return names

# ---------------- File discovery ----------------

def _walk_paths(root: Path, subpaths: List[str], extra_globs: List[str]) -> List[Path]:
    files: List[Path] = []
    for sp in subpaths:
        base = (root / sp).resolve()
        if base.is_file() and base.suffix == ".py":
            files.append(base)
        elif base.is_dir():
            files.extend(p for p in base.rglob("*.py"))
    for g in extra_globs:
        files.extend(p for p in root.glob(g) if p.is_file() and p.suffix == ".py")
    # de-dupe
    seen: Set[Path] = set()
    uniq = []
    for f in files:
        if f not in seen:
            uniq.append(f)
            seen.add(f)
    return uniq

# ---------------- Build ----------------

def _load_repo_config(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("repo_config.json must be an object mapping 'owner/repo' -> {paths:[...], ...}")
    return data

def build_nn_block_names(
    config_path: Path,
    public_only: bool,
    include_registered: bool,
    strict_bases: bool,
    allow_patterns_raw: List[str],
    paths_only: bool,
    extra_globs: List[str],
) -> List[str]:
    cfg = _load_repo_config(config_path)
    cache = _try_repo_cache()

    allow_patterns = [re.compile(p, re.IGNORECASE) for p in allow_patterns_raw]

    all_files: List[Path] = []
    def allowed(p: Path) -> bool:
        # Light noise filter
        parts = {seg.lower() for seg in p.parts}
        return not ({"tests", "test", "docs", "examples"} & parts)

    for repo_name, repo_cfg in sorted(cfg.items(), key=lambda kv: kv[1].get("priority", 99)):
        root = _repo_root_from_cache(cache, repo_name) or _guess_local_repo_root(repo_name)
        if not root or not root.exists():
            log.warning("Repo not found locally for %s — skipping", repo_name)
            continue

        repo_paths = repo_cfg.get("paths", []) or []
        if not repo_paths and paths_only:
            log.info("No 'paths' for %s — skipping (pass --no-paths-only to scan whole repo)", repo_name)
            continue

        subpaths = repo_paths if paths_only else (repo_paths or [""])
        py_files = [p for p in _walk_paths(root, subpaths, extra_globs) if allowed(p)]
        all_files.extend(py_files)

    names: List[str] = []
    worker_count = min(32, (os.cpu_count() or 8) * 2)
    with cf.ThreadPoolExecutor(max_workers=worker_count) as ex:
        futs = [ex.submit(
            _iter_nn_block_names_from_file,
            f, public_only, include_registered, strict_bases, allow_patterns
        ) for f in all_files]
        for fut in cf.as_completed(futs):
            try:
                names.extend(fut.result())
            except Exception:
                pass

    return sorted(set(names))

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="Create a JSON array of neural network block names from configured repos.")
    from .utils.path_resolver import get_config_file_path
    ap.add_argument("--config", type=Path, default=get_config_file_path("repo_config.json"), help="Path to repo_config.json")
    ap.add_argument("--out", type=Path, default=get_config_file_path("nn_block_names.json"), help="Path to output JSON array")
    ap.add_argument("--include-registered", action="store_true",
                    help="(Deprecated) Registry decorators no longer sufficient - Module inheritance required")
    ap.add_argument("--allow-patterns", nargs="*", default=[],
                    help="(Deprecated) Name patterns no longer sufficient - Module inheritance required")
    ap.add_argument("--strict-bases", action="store_true",
                    help="(Deprecated) Now always strict - Module inheritance always required")
    ap.add_argument("--public-only", action="store_true", help="Drop names that start with '_'")
    ap.add_argument("--no-paths-only", dest="paths_only", action="store_false",
                    help="Scan full repo when 'paths' not set (can be slower)")
    ap.add_argument("--extra-globs", nargs="*", default=[],
                    help="Extra globs relative to repo root (e.g., 'mmdet/**/*.py')")
    ap.add_argument("--min-total", type=int, default=0, help="Require at least N names or exit 2")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    names = build_nn_block_names(
        config_path=args.config,
        public_only=args.public_only,
        include_registered=args.include_registered,
        strict_bases=args.strict_bases,
        allow_patterns_raw=args.allow_patterns,
        paths_only=args.paths_only,
        extra_globs=args.extra_globs,
    )

    if args.min_total and len(names) < args.min_total:
        log.error("Only %d names found (< %d required).", len(names), args.min_total)
        args.out.write_text(json.dumps(names, indent=2), encoding="utf-8")
        raise SystemExit(2)

    args.out.write_text(json.dumps(names, indent=2), encoding="utf-8")
    log.info("Wrote %d unique nn block names to %s", len(names), args.out)

def discover_nn_block_names() -> List[str]:
    """Discover neural network block names with sensible defaults for auto-discovery."""
    from .utils.path_resolver import get_config_file_path
    config_path = get_config_file_path("repo_config.json")
    
    return build_nn_block_names(
        config_path=config_path,
        public_only=False,  # Include private classes too
        include_registered=False,  # Deprecated - not needed
        strict_bases=True,  # Always strict - Module inheritance required
        allow_patterns_raw=[],  # Deprecated - not needed
        paths_only=True,  # Only scan configured paths for speed
        extra_globs=[],  # No extra globs
    )


if __name__ == "__main__":
    main()
