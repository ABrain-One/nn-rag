#!/usr/bin/env python3
"""
Repository Cache Manager for PyTorch Block Extractor
- Shallow + sparse clones (Python-only by default) to minimize cold start
- Skips LFS weights via GIT_LFS_SKIP_SMUDGE=1 and blob-less filter
- Never pulls by default; only clones when missing (update_policy='missing')
- Optional safe update modes: 'ff-only' or 'force' if you really want to refresh
- Resilient to repos with huge assets; prunes weight-like files if any slip through
"""

import os
import re
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime

# ------------------------------ helpers -------------------------------- #

def _safe_run(cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None, check: bool = True) -> subprocess.CompletedProcess:
    base_env = os.environ.copy()
    if env:
        base_env.update(env)
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=base_env,
                         capture_output=True, text=True, check=check)

def _now_iso() -> str:
    return datetime.utcnow().isoformat()

# ------------------------------ RepoCache -------------------------------- #

class RepoCache:
    """
    Manages local caching of Git repositories with MINIMAL checkout:
      - shallow clone (--depth=1) + partial clone (--filter=blob:none)
      - sparse-checkout with patterns (Python-only by default)
      - skip LFS blobs on clone/checkout (GIT_LFS_SKIP_SMUDGE=1)

    update_policy:
      - 'missing' (default): clone if absent, never auto-update if present
      - 'ff-only'        : fetch + fast-forward to origin/<default-branch> if possible
      - 'force'          : fetch + hard reset to origin/<default-branch>
    download_policy:
      - 'sparse-py' (default): Python-only patterns + configured paths
      - 'paths-only'         : only configured paths from repo_config.json
      - 'full'               : full checkout (not recommended)
    """

    DEFAULT_BLOCK_EXTS = {".pt", ".pth", ".ckpt", ".bin", ".onnx", ".tflite", ".safetensors", ".npz", ".tar", ".gz", ".zip"}
    DEFAULT_MAX_FILE_MB = 64  # any file bigger than this gets pruned if not .py

    def __init__(
        self,
        cache_dir: str = None,
        config_file: str = None,
        update_policy: str = "missing",
        download_policy: str = "sparse-py",
        prune_weights: bool = True,
        max_file_mb: int = DEFAULT_MAX_FILE_MB,
        extra_skip_exts: Optional[List[str]] = None,
    ):
        # Set default paths using the path resolver
        if cache_dir is None:
            from .path_resolver import get_cache_dir
            cache_dir = str(get_cache_dir())
        if config_file is None:
            from .path_resolver import get_config_file_path
            config_file = str(get_config_file_path("repo_config.json"))
            
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure repo_cache subdirectory exists
        self.repo_cache_dir = self.cache_dir / "repo_cache"
        self.repo_cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_index_file = self.cache_dir / "cache_index.json"
        self.config_file = Path(config_file)

        self.update_policy = update_policy
        self.download_policy = download_policy
        self.prune_weights = prune_weights
        self.max_file_bytes = max_file_mb * 1024 * 1024
        self.skip_exts = set(self.DEFAULT_BLOCK_EXTS)
        if extra_skip_exts:
            self.skip_exts.update(e if e.startswith(".") else "." + e for e in extra_skip_exts)

        self.cache_index = self._load_cache_index()
        self.repos = self._load_repo_config()

    # ------------------------------ config/index I/O ------------------------------ #

    def _load_cache_index(self) -> Dict:
        if self.cache_index_file.exists():
            try:
                return json.loads(self.cache_index_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_cache_index(self):
        self.cache_index_file.write_text(json.dumps(self.cache_index, indent=2), encoding="utf-8")

    def _load_repo_config(self) -> Dict[str, Dict]:
        # First try to read from the config file location
        if self.config_file.exists():
            try:
                return json.loads(self.config_file.read_text(encoding="utf-8"))
            except Exception as e:
                pass
        
        # Try to read from package resources (installed package)
        try:
            from importlib.resources import files
            package_config = files('ab.rag.config').joinpath('repo_config.json')
            if package_config.is_file():
                cfg = json.loads(package_config.read_text(encoding='utf-8'))
                # Copy to cache location for future writes
                try:
                    self.config_file.parent.mkdir(parents=True, exist_ok=True)
                    self.config_file.write_text(package_config.read_text(encoding='utf-8'), encoding='utf-8')
                except Exception:
                    pass  # Read-only location is okay
                return cfg
        except (ImportError, FileNotFoundError, AttributeError):
            pass
        
        # Fallback to minimal default
        cfg = self._get_fallback_config()
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            self.config_file.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        except Exception as e:
            pass
        return cfg

    def _get_fallback_config(self) -> Dict[str, Dict]:
        # Minimal default if repo_config.json isn't present/parseable
        return {
            "huggingface/pytorch-image-models": {
                "url": "https://github.com/huggingface/pytorch-image-models.git",
                "priority": 1,
                "paths": ["timm/models", "timm/layers"],
                "description": "Essential for model/layer blocks"
            },
            "pytorch/vision": {
                "url": "https://github.com/pytorch/vision.git",
                "priority": 1,
                "paths": ["torchvision/models", "torchvision/ops"],
                "description": "TorchVision models & ops"
            },
        }

    # ------------------------------ public API ------------------------------ #

    def add_repository(self, name: str, url: str, priority: int = 2, paths: Optional[List[str]] = None, description: str = ""):
        if paths is None:
            paths = []
        self.repos[name] = {
            "url": url,
            "priority": priority,
            "paths": paths,
            "description": description,
        }
        try:
            # persist
            existing = {}
            if self.config_file.exists():
                existing = json.loads(self.config_file.read_text(encoding="utf-8"))
            existing[name] = self.repos[name]
            self.config_file.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        except Exception as e:
            pass

    def remove_repository(self, name: str):
        if name in self.repos:
            del self.repos[name]
            try:
                cfg = self.repos.copy()
                self.config_file.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            except Exception as e:
                pass
        else:
            pass

    def is_repo_cached(self, repo_name: str) -> bool:
        entry = self.cache_index.get(repo_name)
        if not entry:
            return False
        p = Path(entry.get("path", ""))
        if not (p.exists() and (p / ".git").exists()):
            return False
        # quick sanity: at least one .py somewhere
        for _ in p.rglob("*.py"):
            return True
        return False

    def get_cached_repo(self, repo_name: str) -> Optional[Path]:
        entry = self.cache_index.get(repo_name)
        if not entry:
            return None
        p = Path(entry.get("path", ""))
        return p if p.exists() else None

    def ensure_repo_cached(self, repo_name: str) -> Optional[Path]:
        """
        Ensure repo is present locally. By default, will NOT update an existing repo.
        """
        try:
            if self.is_repo_cached(repo_name):
                return self.get_cached_repo(repo_name)
            return self._update_repo_cache(repo_name)
        except Exception as e:
            # If there's any error, ensure the cache directory exists and try again
            self.repo_cache_dir.mkdir(parents=True, exist_ok=True)
            try:
                return self._update_repo_cache(repo_name)
            except Exception:
                # If still fails, return None to indicate the repo is not available
                return None

    def update_repo(self, repo_name: str, policy: Optional[str] = None) -> Optional[Path]:
        """
        Explicitly update a cached repo according to a policy.
        """
        return self._update_repo_cache(repo_name, update_policy=policy or self.update_policy)

    def clear_cache(self):
        for repo_name, info in list(self.cache_index.items()):
            p = Path(info["path"])
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)
            del self.cache_index[repo_name]
        self._save_cache_index()

    def get_cache_status(self) -> Dict:
        status = {"cache_dir": str(self.cache_dir), "total_size_mb": 0.0, "repos": {}}
        total = 0
        for name, info in self.cache_index.items():
            p = Path(info["path"])
            if p.exists():
                size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                total += size
                status["repos"][name] = {
                    "cached": True,
                    "size_mb": round(size / (1024 * 1024), 2),
                    "last_updated": info.get("last_updated", "unknown"),
                    "path": str(p),
                    "default_branch": info.get("default_branch", "unknown"),
                    "sparse": info.get("sparse", False),
                    "depth1": info.get("depth1", False),
                    "filter_blob_none": info.get("filter_blob_none", False),
                }
            else:
                status["repos"][name] = {"cached": False, "size_mb": 0.0, "last_updated": "never"}
        status["total_size_mb"] = round(total / (1024 * 1024), 2)
        return status

    # ------------------------------ clone/update internals ------------------------------ #

    def _update_repo_cache(self, repo_name: str, update_policy: Optional[str] = None) -> Optional[Path]:
        """
        Clone if missing. If present:
          - missing: do nothing (return as-is)
          - ff-only: fetch + fast-forward to origin/<default>
          - force  : fetch + hard reset to origin/<default>
        """
        policy = (update_policy or self.update_policy).lower()
        cfg = self.repos.get(repo_name)
        if not cfg:
            return None

        repo_path = self.repo_cache_dir / repo_name.replace("/", "_")
        url = cfg.get("url")
        if not url:
            return None

        # If exists already…
        if repo_path.exists():
            if policy == "missing":
                return repo_path
            elif policy in ("ff-only", "force"):
                try:
                    self._fetch_all(repo_path)
                    default = self._default_branch(repo_path) or "main"
                    if policy == "ff-only":
                        _safe_run(["git", "merge", "--ff-only", f"origin/{default}"], cwd=repo_path, check=True)
                    else:
                        _safe_run(["git", "reset", "--hard", f"origin/{default}"], cwd=repo_path, check=True)
                    self._post_update_index(repo_name, repo_path, sparse=True, depth1=True, blobless=True, default_branch=default)
                    if self.prune_weights:
                        self._prune_large_and_weights(repo_path)
                    return repo_path
                except subprocess.CalledProcessError as e:
                    # Keep the old checkout usable
                    return repo_path
            else:
                return repo_path

        # Fresh clone (minimal)
        try:
            repo_path.parent.mkdir(parents=True, exist_ok=True)
            env = {"GIT_LFS_SKIP_SMUDGE": "1"}  # don't download LFS blobs (weights)
            clone_cmd = ["git", "clone", "--filter=blob:none", "--no-checkout", "--depth=1", url, str(repo_path)]
            _safe_run(clone_cmd, env=env, check=True)
            _safe_run(["git", "config", "extensions.partialclone", "origin"], cwd=repo_path, check=True)

            # sparse checkout according to policy
            sparse = self.download_policy in ("sparse-py", "paths-only")
            if sparse:
                self._init_sparse(repo_path)
                patterns = self._sparse_patterns_for(repo_name)
                self._set_sparse_patterns(repo_path, patterns)

            # checkout default branch
            default = self._default_branch(repo_path) or "main"
            _safe_run(["git", "checkout", default], cwd=repo_path, env={"GIT_LFS_SKIP_SMUDGE": "1"}, check=True)

            # optional prune of lingering large/binary files
            if self.prune_weights:
                self._prune_large_and_weights(repo_path)

            self._post_update_index(repo_name, repo_path, sparse=sparse, depth1=True, blobless=True, default_branch=default)
            return repo_path

        except subprocess.CalledProcessError as e:
            # Clean broken directory to avoid future confusion
            if repo_path.exists():
                shutil.rmtree(repo_path, ignore_errors=True)
            return None

    # ------------------------------ sparse helpers ------------------------------ #

    def _init_sparse(self, repo_path: Path):
        """
        Initialize sparse-checkout in no-cone mode to allow glob patterns.
        """
        try:
            _safe_run(["git", "sparse-checkout", "init", "--no-cone"], cwd=repo_path, check=True)
        except subprocess.CalledProcessError:
            # Older git may not support --no-cone; fall back to cone mode
            _safe_run(["git", "sparse-checkout", "init"], cwd=repo_path, check=True)

    def _sparse_patterns_for(self, repo_name: str) -> List[str]:
        """
        Build sparse patterns:
          - 'sparse-py': all *.py + a few metadata files + configured 'paths'
          - 'paths-only': only configured 'paths'
        """
        info = self.repos.get(repo_name, {})
        conf_paths = [p.strip("/") for p in info.get("paths", []) if p and isinstance(p, str)]

        if self.download_policy == "paths-only":
            return conf_paths or ["**/*.py"]

        # sparse-py: glob for python & small project files + configured directories
        patterns = [
            "**/*.py",
            "**/*.pyi",
            "**/*.pyx",
            "**/py.typed",
            "README*",
            "LICENSE*",
            "setup.py",
            "pyproject.toml",
            "requirements*.txt",
        ]
        patterns.extend(conf_paths)
        # de-dup while preserving order
        out, seen = [], set()
        for p in patterns:
            if p not in seen:
                out.append(p); seen.add(p)
        return out

    def _set_sparse_patterns(self, repo_path: Path, patterns: List[str]):
        if not patterns:
            patterns = ["**/*.py"]
        try:
            _safe_run(["git", "sparse-checkout", "set", *patterns], cwd=repo_path, check=True)
        except subprocess.CalledProcessError as e:
            # On very old git versions, only directory paths work; fallback to top-level dirs of patterns
            tops = sorted({p.split("/")[0] for p in patterns if "/" in p} | {p for p in patterns if "/" not in p})
            try:
                _safe_run(["git", "sparse-checkout", "set", *tops], cwd=repo_path, check=True)
            except subprocess.CalledProcessError:
                # As a last resort, disable sparse
                _safe_run(["git", "sparse-checkout", "disable"], cwd=repo_path, check=False)

    # ------------------------------ git helpers ------------------------------ #

    def _fetch_all(self, repo_path: Path):
        _safe_run(["git", "fetch", "--all", "--tags", "--prune"], cwd=repo_path, check=True)

    def _default_branch(self, repo_path: Path) -> Optional[str]:
        # Try to detect origin/HEAD -> origin/<branch>
        try:
            cp = _safe_run(["git", "symbolic-ref", "refs/remotes/origin/HEAD"], cwd=repo_path, check=True)
            # output like: refs/remotes/origin/main
            ref = (cp.stdout or "").strip()
            if ref.startswith("refs/remotes/origin/"):
                return ref.split("/")[-1]
        except subprocess.CalledProcessError:
            pass
        # Fallback: remote show origin
        try:
            cp = _safe_run(["git", "remote", "show", "origin"], cwd=repo_path, check=True)
            m = re.search(r"HEAD branch:\s+(\S+)", cp.stdout or "")
            if m:
                return m.group(1)
        except subprocess.CalledProcessError:
            pass
        return None

    def _post_update_index(self, repo_name: str, repo_path: Path, *, sparse: bool, depth1: bool, blobless: bool, default_branch: str):
        self.cache_index[repo_name] = {
            "path": str(repo_path),
            "last_updated": _now_iso(),
            "sparse": bool(sparse),
            "depth1": bool(depth1),
            "filter_blob_none": bool(blobless),
            "default_branch": default_branch,
            "url": self.repos.get(repo_name, {}).get("url"),
            "download_policy": self.download_policy,
            "update_policy": self.update_policy,
        }
        self._save_cache_index()

    # ------------------------------ pruning (safety net) ------------------------------ #

    def _prune_large_and_weights(self, repo_path: Path):
        """
        As an extra guard, remove any non-.py file that looks like a weight/artifact or is huge.
        (Sparse + blobless means these rarely appear, but this keeps the cache lean if they do.)
        """
        removed = 0
        for f in repo_path.rglob("*"):
            if not f.is_file():
                continue
            ext = f.suffix.lower()
            if ext in self.skip_exts and f.exists():
                try:
                    f.unlink()
                    removed += 1
                    continue
                except Exception:
                    pass
            if ext not in {".py", ".pyi", ".pyx"}:
                try:
                    if f.stat().st_size > self.max_file_bytes:
                        f.unlink()
                        removed += 1
                except Exception:
                    pass
        if removed:
            print(f"🧽 Pruned {removed} large/weight files from {repo_path.name}")

    # ------------------------------ content/search utilities ------------------------------ #

    def get_file_content_from_cache(self, repo_name: str, file_path: str, commit_sha: Optional[str] = None) -> Optional[str]:
        """
        Read a file from the cached repo (optionally at a specific commit).
        NOTE: For partial clones, checking out arbitrary commits may need a fetch; we avoid it
              to keep cold start low. If commit_sha is given, we attempt a detached checkout.
        """
        repo = self.ensure_repo_cached(repo_name)
        if not repo:
            return None
        try:
            env = {"GIT_LFS_SKIP_SMUDGE": "1"}
            if commit_sha:
                _safe_run(["git", "checkout", commit_sha], cwd=repo, env=env, check=True)
            p = repo / file_path
            return p.read_text(encoding="utf-8") if p.exists() else None
        except Exception as e:
            return None
        finally:
            # Best-effort: return to default branch if we detached
            if commit_sha:
                try:
                    default = self.cache_index.get(repo_name, {}).get("default_branch") or "main"
                    _safe_run(["git", "checkout", default], cwd=repo, env={"GIT_LFS_SKIP_SMUDGE": "1"}, check=False)
                except Exception:
                    pass

    def search_in_cached_repo(self, repo_name: str, query: str) -> List[Dict[str, str]]:
        """
        Search for files containing a query using git grep (Python tree only due to sparse).
        """
        repo = self.get_cached_repo(repo_name)
        if not repo:
            return []
        try:
            result = _safe_run(["git", "grep", "-l", query], cwd=repo, check=False)
            files = [ln for ln in (result.stdout or "").splitlines() if ln.strip()]
            return [{"path": f, "repo": repo_name} for f in files]
        except Exception as e:
            return []

    def get_repo_tree(self, repo_name: str) -> Dict[str, str]:
        """
        Return a mapping of python files in the cached repo.
        """
        repo = self.ensure_repo_cached(repo_name)
        if not repo:
            return {}
        out = {}
        for p in repo.rglob("*.py"):
            try:
                out[str(p.relative_to(repo))] = str(p.relative_to(repo))
            except Exception:
                pass
        return out

    # ------------------------------ convenience ------------------------------ #

    def get_repositories_by_priority(self, priority: int) -> List[str]:
        return [name for name, info in self.repos.items() if info.get("priority") == priority]

    def get_essential_repositories(self) -> List[str]:
        return self.get_repositories_by_priority(1)

    def get_secondary_repositories(self) -> List[str]:
        return self.get_repositories_by_priority(2)


# # ------------------------------ demo -------------------------------- #

# if __name__ == "__main__":
#     # Example: keep defaults minimal (no auto-update, sparse python-only)
#     cache = RepoCache(update_policy="missing", download_policy="sparse-py")

#     print("Repository Cache Manager (lean mode)")
#     print("=" * 60)

#     # Pick a repo name that exists in your repo_config.json
#     test_repo = next(iter(cache.repos.keys()), "huggingface/pytorch-image-models")
#     print(f"Ensuring cache for: {test_repo}")

#     path = cache.ensure_repo_cached(test_repo)
#     print(f"→ Path: {path}")

#     tree = cache.get_repo_tree(test_repo)
#     print(f"Found {len(tree)} Python files (sparse). Example keys:")
#     for i, k in enumerate(tree.keys()):
#         print("  ", k)
#         if i >= 4:
#             break

#     status = cache.get_cache_status()
#     print("\nCache Status:")
#     print(json.dumps(status, indent=2))
