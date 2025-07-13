import json
import time
from pathlib import Path

_ROOT = Path.home() / ".cache" / "nn-rag"
_ROOT.mkdir(exist_ok=True)
_JSON = _ROOT / "json"
_JSON.mkdir(exist_ok=True)
TTL = 86_400  # seconds (24h)


def _cache_path(key: str) -> Path:
    import hashlib
    stable_hash = hashlib.sha1(key.encode()).hexdigest()
    return _JSON / f"{stable_hash}.json"


def load(key: str):
    p = _cache_path(key)
    if not p.exists():
        return None
    if time.time() - p.stat().st_mtime > TTL:
        return None
    return json.loads(p.read_text())


def save(key: str, obj) -> None:
    p = _cache_path(key)
    p.write_text(json.dumps(obj))


def in_cache(key: str) -> bool:
    return load(key) is not None