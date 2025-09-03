from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

# The extractor imports ResolvedSymbol from here
@dataclass
class ResolvedSymbol:
    qname: str                         # e.g. "mmengine.model.base_module.BaseModule"
    name: str                          # "BaseModule"
    kind: str                          # 'class'|'function'|'const'
    repo: Optional[str]
    file_path: Optional[str]           # repo-relative path to file (e.g. "mmengine/model/base_module.py")
    module_qual: Optional[str]         # "mmengine.model.base_module"
    module_path: Optional[str]         # "mmengine/model/base_module"
    module_source: Optional[str]       # may be omitted (None) — we already have the slice
    source_code: str                   # exact block code for the symbol
    line: Optional[int]                # starting line if known
    module_imports: Optional[List[str]]# imports detected for its module (optional)


class DefinitionResolver:
    """
    Package-aware resolver that returns REAL definitions from the already-built import_graph.

    Key capabilities:
    - Understands package re-exports via __init__.py (e.g., mmengine.model re-exporting BaseModule).
    - Maps 'module.Member' to the actual definition qualified name.
    - Resolves bare names by consulting package export maps & a global reverse index.
    - Returns exact source slices from the index — no stubs.

    Assumptions:
    - The main extractor already indexed all repos listed in repo_config.json into `import_graph`.
      To resolve mmdetection dependencies cleanly, ensure you include:
        - "open-mmlab/mmengine"
        - "open-mmlab/mmcv"
        - "open-mmlab/mmdetection"
    """

    def __init__(self, repo_cache, import_graph, index, log):
        self.repo_cache = repo_cache
        self.import_graph = import_graph   # ImportGraph (modules + symbol_table)
        self.index = index                 # FileIndexStore
        self.log = log

        # Fast lookups populated at init (derived from current import_graph)
        self._module_index: Dict[str, Tuple[str, object]] = {}      # mod_qual -> (repo, ModuleInfo)
        self._name_to_qnames: Dict[str, List[str]] = {}             # "BaseModule" -> ["mmengine.model.base_module.BaseModule", ...]
        self._package_exports: Dict[str, Dict[str, str]] = {}       # "mmengine.model" -> {"BaseModule": "mmengine.model.base_module.BaseModule", ...}

        self._build_fast_indexes()

    # --------------------------------------------------------------------- #
    # Public API used by the extractor
    # --------------------------------------------------------------------- #

    def resolve_symbols(self, qnames: List[str]) -> Dict[str, ResolvedSymbol]:
        """
        Given unresolved names (bare or dotted), return concrete definitions with real source.
        """
        out: Dict[str, ResolvedSymbol] = {}
        for q in qnames:
            rs = self._resolve_one(q)
            if rs:
                out[rs.qname] = rs
        return out

    # --------------------------------------------------------------------- #
    # Core resolution
    # --------------------------------------------------------------------- #

    def _resolve_one(self, query: str) -> Optional[ResolvedSymbol]:
        """
        Resolve a single query, which may be:
          - a fully qualified symbol (exact match)
          - a package-level re-export (e.g., "mmengine.model.BaseModule")
          - a bare name ("BaseModule")
        Returns None if ambiguous or absent.
        """
        # 1) Exact qualified symbol (already in the symbol table)?
        si = self.import_graph.symbol_table.get(query)
        if si:
            return self._rs_from_symbolinfo(si)

        # 2) "module.Member" form that might be a re-export
        if "." in query:
            mod, name = query.rsplit(".", 1)
            # try module export map (if this module is a package that re-exports)
            target_q = self._resolve_export(mod, name)
            if target_q:
                si2 = self.import_graph.symbol_table.get(target_q)
                if si2:
                    return self._rs_from_symbolinfo(si2)
                # As a fallback, do a unique search constrained by module prefix
                target_q = self._unique_qname_under_module(mod, name)
                if target_q:
                    si3 = self.import_graph.symbol_table.get(target_q)
                    if si3:
                        return self._rs_from_symbolinfo(si3)
            # If "module.Member" points directly to a module-level constant captured via harvesting
            # (e.g., "timm.layers.config._USE_REENTRANT_CKPT"), allow a constrained unique search:
            target_q = self._unique_qname_under_module(mod, name)
            if target_q:
                si3 = self.import_graph.symbol_table.get(target_q)
                if si3:
                    return self._rs_from_symbolinfo(si3)

        # 3) Bare name — find a unique, best-effort match
        #    Prefer explicit re-exports uniqueness; otherwise a global unique name-only match
        candidates = self._name_to_qnames.get(query) or []
        if len(candidates) == 1:
            si4 = self.import_graph.symbol_table.get(candidates[0])
            if si4:
                return self._rs_from_symbolinfo(si4)

        # Try to disambiguate common package families (mmengine/mmcv/mmdet) via export maps:
        target_q = self._choose_via_exports(query)
        if target_q:
            si5 = self.import_graph.symbol_table.get(target_q)
            if si5:
                return self._rs_from_symbolinfo(si5)

        # Ambiguous or not available
        return None

    # --------------------------------------------------------------------- #
    # Index builders
    # --------------------------------------------------------------------- #

    def _build_fast_indexes(self) -> None:
        """Build module index, reverse name index, and package export maps."""
        # module index
        for key, mi in self.import_graph.modules.items():
            repo, mod = key.split(":", 1)
            self._module_index[mod] = (repo, mi)

        # name -> qnames
        for q, si in self.import_graph.symbol_table.items():
            self._name_to_qnames.setdefault(si.name, []).append(q)

        # package exports from __init__.py re-exports
        for mod_qual, tup in self._module_index.items():
            repo, mi = tup
            # Heuristic: treat directories as packages; `__init__` modules are typical aggregators
            # `mi.file` holds leaf name; when it's "__init__", parse for re-exports
            if mi.file == "__init__":
                self._package_exports[mod_qual] = self._collect_exports_from_init(mi, mod_qual)

    # --------------------------------------------------------------------- #
    # Export map construction
    # --------------------------------------------------------------------- #

    def _collect_exports_from_init(self, mi, mod_qual: str) -> Dict[str, str]:
        """
        Parse __init__.py and collect "from .sub import Name [as Alias]" and
        "from pkg.sub import Name [as Alias]" re-exports.

        Returns {exported_name -> concrete_qualified_symbol_qname}
        """
        out: Dict[str, str] = {}
        src = mi.source_code or ""
        if not src:
            return out

        try:
            tree = ast.parse(src)
        except Exception:
            return out

        def resolve_relative(base: str, level: int, sub: Optional[str]) -> str:
            # base is current module qual (package __init__), e.g. "mmengine.model"
            parts = base.split(".")
            up = max(0, level - 1)  # level=1 means "."
            head = parts[:len(parts) - up]
            if sub:
                head += sub.split(".")
            return ".".join([p for p in head if p])

        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                # Build the source module qual
                if node.module:
                    if node.level and node.level > 0:
                        src_mod = resolve_relative(mod_qual, node.level, node.module)
                    else:
                        src_mod = node.module
                else:
                    # from . import X
                    src_mod = resolve_relative(mod_qual, node.level or 1, None)

                for alias in node.names:
                    if alias.name == "*":
                        # We do not discover star-exports here (would require scanning the source module's __all__)
                        continue
                    exported_name = alias.asname or alias.name
                    # The *defining* qualified symbol must be src_mod + '.' + original_name
                    # e.g., "mmengine.model.base_module" + ".BaseModule"
                    target_q = f"{src_mod}.{alias.name}"
                    # If the target is itself a module (submodule re-export), try to refine by searching definitions under it
                    # Case: "from . import xyz" where xyz is submodule; often __init__ also re-exports symbols from it
                    if target_q not in self.import_graph.symbol_table:
                        refined = self._unique_qname_under_module(src_mod, alias.name)
                        if refined:
                            target_q = refined
                    out[exported_name] = target_q

        return out

    # --------------------------------------------------------------------- #
    # Resolution helpers
    # --------------------------------------------------------------------- #

    def _resolve_export(self, module_qual: str, name: str) -> Optional[str]:
        """
        Resolve `module_qual.name` when `module_qual` is a package that re-exports `name`.
        """
        # 1) direct export map hit for that exact package
        d = self._package_exports.get(module_qual)
        if d and name in d:
            return d[name]

        # 2) If not a known package init, attempt to find an ancestor package that re-exports
        #    Example: "mmengine.model.parallel" might export items from its own __init__.py
        parts = module_qual.split(".")
        for k in range(len(parts) - 1, 0, -1):
            base = ".".join(parts[:k])
            d2 = self._package_exports.get(base)
            if d2 and name in d2:
                # confirm the chosen target stays under the intended subtree
                return d2[name]

        # 3) Fallback: unique search constrained by the package prefix
        return self._unique_qname_under_module(module_qual, name)

    def _unique_qname_under_module(self, module_prefix: str, name: str) -> Optional[str]:
        """
        Find a unique symbol whose qname starts with `module_prefix + "."` and whose terminal name == `name`.
        """
        candidates = [
            q for q in self._name_to_qnames.get(name, [])
            if q.startswith(module_prefix + ".")
        ]
        return candidates[0] if len(candidates) == 1 else None

    def _choose_via_exports(self, bare_name: str) -> Optional[str]:
        """
        For a bare name, try to disambiguate using package export maps.
        If exactly one package re-exports this name to a single concrete target, return that target.
        """
        hits: List[str] = []
        # For each package that has an export map, see if it exports our bare name
        for pkg, mapping in self._package_exports.items():
            tgt = mapping.get(bare_name)
            if not tgt:
                continue
            # If the export target itself is a module (not definition), try to refine:
            if tgt not in self.import_graph.symbol_table:
                refined = self._unique_qname_under_module(pkg, bare_name)
                if refined:
                    hits.append(refined)
                else:
                    hits.append(tgt)
            else:
                hits.append(tgt)

        # If a single unambiguous target emerges, return it
        hits = list(dict.fromkeys(hits))  # dedupe, keep order
        if len(hits) == 1:
            return hits[0]

        # Otherwise, if the global name index is unique, use that
        qns = self._name_to_qnames.get(bare_name) or []
        if len(qns) == 1:
            return qns[0]

        return None

    # --------------------------------------------------------------------- #
    # Materialization helpers
    # --------------------------------------------------------------------- #

    def _rs_from_symbolinfo(self, si) -> ResolvedSymbol:
        """
        Convert a SymbolInfo from the import_graph into a ResolvedSymbol that the extractor can inject.
        """
        q = si.qualified_name
        mod_qual = q.rsplit(".", 1)[0]
        # retrieve the owning module to attach optional module_imports / module_source
        repo, mi = self._module_index.get(mod_qual, (None, None))
        mod_src = getattr(mi, "source_code", None) if mi else None

        # NOTE: Extractor doesn't strictly require module_imports here;
        # leaving None keeps things light. If needed, we can parse imports from mod_src.
        return ResolvedSymbol(
            qname=q,
            name=si.name,
            kind=si.kind if si.kind in ("class", "function", "const") else self._kind_from_code(si.source_code),
            repo=repo,
            file_path=si.location,                          # repo-relative path captured at index time
            module_qual=mod_qual,
            module_path=mod_qual.replace(".", "/"),
            module_source=mod_src,
            source_code=si.source_code,
            line=si.line_number,
            module_imports=None,
        )

    @staticmethod
    def _kind_from_code(snippet: str) -> str:
        try:
            t = ast.parse(snippet)
            if t.body and isinstance(t.body[0], ast.ClassDef): return "class"
            if t.body and isinstance(t.body[0], ast.FunctionDef): return "function"
        except Exception:
            pass
        return "const"
