#!/usr/bin/env python3
"""
compile_blocks.py

Checks whether extracted blocks are:
  1) syntactically valid (AST compile)
  2) importable (executes the module) in a sandboxed subprocess with a timeout

Why subprocess?
- isolates side effects during import
- lets us set safe env vars (no GPU, offline modes)
- enforces a per-file timeout

Outputs a summary to stdout and (optionally) a JSON report.

Usage:
  python compile_blocks.py --dir ab/rag/generated_packages
Options:
  --dir DIR               Directory with generated .py files (default: ab/rag/generated_packages)
  --names FILE.json       Optional JSON array of block names (e.g. ["ConvMixer", ...]); we look for <name>.py
  --pattern GLOB          Only test files matching glob (default: *.py)
  --workers N             Parallel workers (default: min(32, 2 * CPU))
  --timeout SEC           Per-file import timeout in seconds (default: 15)
  --json-out PATH         Write a machine-readable report
  --verbose               Print per-file details
  --fail-fast             Stop on first failure (exit non-zero)
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures as cf
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------- CLI ----------------

def default_worker_count() -> int:
    """Return the default number of parallel workers: min(32, 2 * CPU), at least 2."""
    return max(2, min(32, (os.cpu_count() or 8) * 2))

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compile/import check for generated NN blocks.")
    ap.add_argument("--dir", type=Path, default=Path("ab/rag/generated_packages"))
    ap.add_argument("--names", type=Path, default=None, help="JSON array of block names; look for <name>.py inside --dir")
    ap.add_argument("--pattern", type=str, default="*.py")
    ap.add_argument("--workers", type=int, default=default_worker_count())
    ap.add_argument("--timeout", type=int, default=15)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--fail-fast", action="store_true")
    return ap.parse_args()

# ---------------- Data ----------------

@dataclass
class FileResult:
    file: str
    name: str
    ast_ok: bool
    import_ok: bool
    stage: str            # "ast" | "import" | "ok"
    err_type: Optional[str] = None
    err_brief: Optional[str] = None
    missing_module: Optional[str] = None
    stderr: Optional[str] = None

# ---------------- Helpers ----------------

_NO_MODULE_RE = re.compile(r"No module named ['\"]([^'\"/]+)['\"]")
_CUDA_RE = re.compile(r"CUDA|cuDNN|MPS|device.*(not|unavailable)", re.IGNORECASE)
_NUMPY_MKL_RE = re.compile(r"ImportError:.*(MKL|OpenBLAS)", re.IGNORECASE)

def _shorten(s: str, limit: int = 300) -> str:
    s = s.strip()
    if len(s) <= limit:
        return s
    return s[:limit].rstrip() + " ..."

def _classify(stderr: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (err_type, err_brief, missing_module) based on stderr."""
    if not stderr:
        return None, None, None
    brief = _shorten(stderr, 400)

    m = _NO_MODULE_RE.search(stderr)
    if m:
        return "ImportError", f"Missing module: {m.group(1)}", m.group(1)
    if "SyntaxError" in stderr:
        return "SyntaxError", brief, None
    if "NameError" in stderr:
        return "NameError", brief, None
    if _CUDA_RE.search(stderr):
        return "RuntimeError", "GPU/CUDA/MPS unavailable during import", None
    if _NUMPY_MKL_RE.search(stderr):
        return "ImportError", "BLAS/MKL backend missing", None
    if "Timeout" in stderr or "timed out" in stderr.lower():
        return "Timeout", "Import timed out", None
    if "MemoryError" in stderr:
        return "MemoryError", "Out of memory during import", None
    # fallback
    etype = stderr.splitlines()[0].split(":")[0] if ":" in stderr.splitlines()[0] else "Exception"
    return etype, brief, None

def _read_names_json(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("--names JSON must be a list.")
    out = []
    for x in data:
        if isinstance(x, str):
            out.append(x.strip())
    return out

def _discover_files(root: Path, pattern: str, names_json: Optional[Path]) -> List[Path]:
    if names_json and names_json.exists():
        names = _read_names_json(names_json)
        files = [(root / f"{n}.py") for n in names]
        return [p for p in files if p.is_file()]
    return sorted(root.glob(pattern))

# ---------------- Checks ----------------

def _ast_compile_check(py_file: Path) -> Tuple[bool, Optional[str]]:
    try:
        src = py_file.read_text(encoding="utf-8", errors="ignore")
        ast.parse(src)  # syntax check
        compile(src, str(py_file), "exec")  # bytecode check
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

_SANDBOX_SNIPPET = r"""
import os, sys, importlib.util, runpy, types, builtins
# Safety-ish environment: no GPU, offline-ish
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TORCH_HOME", os.path.join(os.getcwd(), ".torch"))
# keep threads small to avoid oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

path = sys.argv[1]
mod_name = os.path.splitext(os.path.basename(path))[0] + "_check"

spec = importlib.util.spec_from_file_location(mod_name, path)
m = importlib.util.module_from_spec(spec)

try:
    loader = spec.loader
    if loader is None:
        raise ImportError("No loader for spec")
    loader.exec_module(m)
    # Success signal
    print("<<<__IMPORT_OK__>>>")
except Exception as e:
    # Print full traceback to stderr
    import traceback
    traceback.print_exc()
    sys.exit(2)
"""

def _import_check_subprocess(py_file: Path, timeout: int) -> Tuple[bool, Optional[str]]:
    """
    Import in a subprocess, return (ok, stderr_text_if_any).
    """
    cmd = [sys.executable, "-c", _SANDBOX_SNIPPET, str(py_file)]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
            env={**os.environ},  # inherit env, but snippet overrides unsafe vars
        )
    except subprocess.TimeoutExpired as e:
        return False, f"Timeout: {e}"

    out = (proc.stdout or "")
    err = (proc.stderr or "")
    if "<<<__IMPORT_OK__>>>" in out and proc.returncode == 0:
        return True, None
    return False, err or out or f"Return code {proc.returncode}"

def _one_file(py_file: Path, timeout: int, verbose: bool) -> FileResult:
    name = py_file.stem
    ast_ok, ast_err = _ast_compile_check(py_file)
    if not ast_ok:
        if verbose:
            print(f"[AST ✗] {py_file} :: {ast_err}")
        et, brief, _ = _classify(ast_err or "")
        return FileResult(
            file=str(py_file), name=name,
            ast_ok=False, import_ok=False, stage="ast",
            err_type=et or "SyntaxError", err_brief=brief or ast_err
        )
    if verbose:
        print(f"[AST ✓] {py_file}")

    imp_ok, stderr = _import_check_subprocess(py_file, timeout=timeout)
    if not imp_ok:
        et, brief, missing = _classify(stderr or "")
        if verbose:
            print(f"[IMP ✗] {py_file} :: {et} :: {brief}")
        return FileResult(
            file=str(py_file), name=name,
            ast_ok=True, import_ok=False, stage="import",
            err_type=et, err_brief=brief, missing_module=missing,
            stderr=_shorten(stderr or "", 2000)
        )

    if verbose:
        print(f"[IMP ✓] {py_file}")
    return FileResult(file=str(py_file), name=name, ast_ok=True, import_ok=True, stage="ok")

# ---------------- Main ----------------

def main():
    args = parse_args()
    base = args.dir
    base.mkdir(parents=True, exist_ok=True)

    files = _discover_files(base, args.pattern, args.names)
    if not files:
        print(f"No files found under {base} matching {args.pattern}" + (f" (from {args.names})" if args.names else ""))
        sys.exit(1)

    print(f"Checking {len(files)} file(s) from: {base}")
    ok_count = 0
    fail_count = 0
    results: List[FileResult] = []

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one_file, f, args.timeout, args.verbose): f for f in files}
        for fut in cf.as_completed(futs):
            res = fut.result()
            results.append(res)
            if res.import_ok and res.ast_ok:
                ok_count += 1
            else:
                fail_count += 1
                if args.fail_fast:
                    break

    # Aggregate stats
    missing_by_mod: Dict[str, int] = {}
    err_by_type: Dict[str, int] = {}
    for r in results:
        if r.missing_module:
            missing_by_mod[r.missing_module] = missing_by_mod.get(r.missing_module, 0) + 1
        if r.err_type:
            err_by_type[r.err_type] = err_by_type.get(r.err_type, 0) + 1

    # Print summary
    print("\n=== Summary ===")
    print(f"OK:   {ok_count}")
    print(f"FAIL: {fail_count}")
    if err_by_type:
        for k, v in sorted(err_by_type.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {k:<16} {v}")
    if missing_by_mod:
        print("\nMost-missing modules:")
        for k, v in sorted(missing_by_mod.items(), key=lambda kv: (-kv[1], kv[0]))[:15]:
            print(f"  {k:<24} {v}")

    # Optional JSON report
    if args.json_out:
        payload = {
            "dir": str(base),
            "count": len(results),
            "ok": ok_count,
            "fail": fail_count,
            "errors_by_type": err_by_type,
            "missing_modules": missing_by_mod,
            "results": [asdict(r) for r in sorted(results, key=lambda x: x.file)],
        }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report → {args.json_out}")

    # Exit code: non-zero on any failure
    sys.exit(0 if fail_count == 0 else 2)

if __name__ == "__main__":
    main()
