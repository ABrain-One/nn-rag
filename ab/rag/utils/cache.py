import hashlib, json
from typing import List, Optional
from pathlib import Path
from ..config.config import CACHE_DIR, TTL_SECONDS

def cpath(query: str) -> Path:
    h = hashlib.sha1(query.encode()).hexdigest()
    return CACHE_DIR / f"{h}.json"

def read_cache(query: str) -> Optional[List[str]]:
    fp = cpath(query)
    if not fp.exists():
        return None
    if fp.stat().st_mtime < __import__("time").time() - TTL_SECONDS:
        fp.unlink(missing_ok=True)
        return None
    try:
        return json.loads(fp.read_text())
    except json.JSONDecodeError:
        fp.unlink(missing_ok=True)
        return None

def save_cache(query: str, chunks: List[str]) -> None:
    try:
        cpath(query).write_text(json.dumps(chunks))
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("cache write failed for %s: %s", query, e)
