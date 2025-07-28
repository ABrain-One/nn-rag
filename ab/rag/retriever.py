from __future__ import annotations
import ast
import logging
from pathlib import Path
from typing import Optional, Tuple, List

from .utils.query_helpers import _canonical, DEFAULT_SOURCES, BLOCKS_100, extract_class
from .helper.manual_specs import MANUAL_BUNDLES
from .helper.import_filter import split_imports_and_body, dedup_preserve
from .helper.extractors import (
    gh_hits, fetch_body, extract_named_functions, extract_named_classes
)
from .utils.cache import load, save
from .utils.github_utils import fetch_raw

log = logging.getLogger(__name__)

def _assemble_with_helpers(
    main_src: str,
    target_class: str,
    helpers: list[tuple[str, str]],
    keep_functions: list[str] | None = None,
    extra_classes: list[str] | None = None,
    strict_class_extract: bool = False,
) -> str:
    class_imports, main_wo_imports = split_imports_and_body(main_src)

    target_snippets = extract_named_classes(main_src, [target_class])
    if not target_snippets:
        if strict_class_extract:
            raise ValueError(f"class {target_class} not found")
        target_src = main_wo_imports
    else:
        target_src = target_snippets[0]

    same_file_funcs = extract_named_functions(main_src, keep_functions or [])
    same_file_classes = extract_named_classes(main_src, extra_classes or [])

    all_imports: list[str] = class_imports.copy()
    helper_bodies: list[str] = []
    for repo, path in helpers:
        helper_src = fetch_body(repo, path)
        imp, body = split_imports_and_body(helper_src)
        all_imports += imp
        helper_bodies.append(body)

    all_imports = dedup_preserve(all_imports)
    header = "\n".join(all_imports)
    if header:
        header += "\n\n"

    blocks: list[str] = []
    blocks.extend(helper_bodies)
    blocks.extend(same_file_funcs)
    blocks.extend(same_file_classes)
    blocks.append(target_src)

    return header + "\n\n".join(b.strip() for b in blocks if b.strip()) + "\n"


class Retriever:
    """
    Manual-first retriever:
      1) If block in MANUAL_BUNDLES -> assemble with rules.
      2) Else DEFAULT_SOURCES -> fetch there.
      3) Else REST search by class name.
    """

    def _manual_bundle(self, name: str) -> Optional[str]:
        spec = MANUAL_BUNDLES.get(name) or MANUAL_BUNDLES.get(_canonical(name))
        if not spec:
            return None

        repo = spec["repo"]; path = spec["path"]
        mode = spec.get("mode", "file")
        main_src = fetch_body(repo, path)
        target = _canonical(name)

        if mode == "file":
            log.info("[manual:file] %s ← %s/%s", name, repo, path)
            return main_src

        helpers = spec.get("helpers", [])
        keep_functions = spec.get("keep_functions", [])
        extra_classes = spec.get("extra_classes", [])
        strict = bool(spec.get("strict_class_extract", False))
        log.info(
            "[manual:class] %s ← %s/%s  (+%d helpers, %d same-file funcs, %d extra classes, strict=%s)",
            name, repo, path, len(helpers), len(keep_functions), len(extra_classes), strict,
        )
        return _assemble_with_helpers(
            main_src,
            target_class=target,
            helpers=helpers,
            keep_functions=keep_functions,
            extra_classes=extra_classes,
            strict_class_extract=strict,
        )

    def best_path(self, name: str) -> Tuple[str, str]:
        if name in DEFAULT_SOURCES:
            return DEFAULT_SOURCES[name]

        target = _canonical(name)
        query = f"class {target} in:file language:Python"
        for item in gh_hits(query):
            repo = item["repository"]["full_name"]
            path = item["path"]
            body = fetch_body(repo, path)
            if f"class {target}" in body:
                return repo, path
            try:
                t = ast.parse(body, type_comments=True)
                if any(isinstance(n, ast.ClassDef) and n.name == target for n in ast.walk(t)):
                    return repo, path
            except SyntaxError:
                continue
        raise FileNotFoundError(f"No suitable source for {name}")

    def file(self, repo: str, path: str) -> str:
        key = f"raw::{repo}::{path}"
        src = load(key)
        if src is None:
            src = fetch_raw(repo, path)
            save(key, src)
        return src.replace("\x00", "")

    def get_block(self, name: str) -> Optional[str]:
        manual = self._manual_bundle(name)
        if manual:
            return manual

        try:
            repo, path = self.best_path(name)
            src = self.file(repo, path)
            target = _canonical(name)
            try:
                return extract_class(src, target)
            except Exception:
                return src
        except Exception as e:
            log.warning("⚠️ skip %s – %s", name, e)
            return None

    def dump_all_blocks(self, dest: str | Path = "blocks") -> None:
        dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
        for name in BLOCKS_100:
            code = self.get_block(name)
            if code:
                (dest / f"{name}.py").write_text(code)
                log.info("✓ %s", name)

# convenient re-exports
__all__ = ["Retriever", "BLOCKS_100", "MANUAL_BUNDLES"]
