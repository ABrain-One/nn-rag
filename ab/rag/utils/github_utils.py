import os
import time
import logging
import requests
from .cache import load, save
from ..config.config import SEARCH_URL, HEADERS, _MIN_REMAIN

log = logging.getLogger(__name__)

def search_code(query: str, per_page: int = 10, page: int = 1) -> list:
    """
    Search GitHub code using the REST API, with simple rate-limit handling.
    """
    cache_key = f"gh::{query}"
    hits = load(cache_key)
    if hits is None:
        r = requests.get(
            SEARCH_URL,
            headers=HEADERS,
            params={"q": query, "per_page": per_page, "page": page},
            timeout=30,
        )
        remain = int(r.headers.get("X-RateLimit-Remaining", "0"))
        if remain < _MIN_REMAIN:
            reset = int(r.headers.get("X-RateLimit-Reset", 0))
            wait = max(1, reset - time.time() + 1)
            log.info("Sleeping %.1fs for rate-limit", wait)
            time.sleep(wait)
        r.raise_for_status()
        hits = r.json().get("items", [])
        save(cache_key, hits)
    return hits


def fetch_raw(repo: str, path: str, branch: str = "main") -> str:
    """
    Download raw file content from GitHub (try main then master).
    """
    url = f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"
    r = requests.get(url, headers=HEADERS, timeout=15)
    if r.status_code == 200:
        return r.text
    if branch == "main":
        return fetch_raw(repo, path, branch="master")
    r.raise_for_status()