"""
Data models for the PyTorch Block Extractor.

This module contains all the data classes and structures used throughout
the block extraction pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SymbolInfo:
    """Information about a single symbol (class, function, or constant)."""
    name: str
    qualified_name: str         # e.g. "timm.layers.attention.Attention" or "timm.layers.config._USE_REENTRANT_CKPT"
    kind: str                   # 'class' | 'function' | 'const'
    location: str               # repo-relative file path (with .py)
    line_number: int
    source_code: str


@dataclass
class ModuleInfo:
    """Information about a Python module."""
    repo: str
    qual: str                   # dotted module path e.g. "timm.layers.attention"
    path: str                   # repo-relative path without suffix e.g. "timm/layers/attention"
    file: str                   # filename (module leaf)
    source_code: str = ""       # full module text (subset retained in SQLite too)
    all_exports: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    symbols: Dict[str, SymbolInfo] = field(default_factory=dict)   # name -> SymbolInfo


@dataclass
class ImportGraph:
    """Graph structure for tracking module imports and dependencies."""
    modules: Dict[str, ModuleInfo] = field(default_factory=dict)         # key: "repo:qual"
    symbol_table: Dict[str, SymbolInfo] = field(default_factory=dict)    # qual.name -> SymbolInfo
    dependents: Dict[str, List[str]] = field(default_factory=dict)       # imported_module -> [module_keys]


@dataclass
class ResolvedDependency:
    """Information about a resolved dependency."""
    name: str
    qualified_name: str
    source_code: str
    resolution_method: str
    confidence: float
    location: Optional[str] = None


@dataclass
class DependencyResolutionResult:
    """Result of dependency resolution for a target symbol."""
    target_symbol: str
    resolved_dependencies: Dict[str, ResolvedDependency]
    unresolved_dependencies: List[str]
    import_graph: ImportGraph
    resolution_stats: Dict[str, Any]
    topological_order: Optional[List[str]] = None
