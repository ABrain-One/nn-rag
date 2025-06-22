import base64, time, logging
import httpx
from typing import List, Dict
from ..config.config import HEADERS, SEARCH_URL

log = logging.getLogger(__name__)

def _get(url: str, *, params=None) -> httpx.Response:
    try:
        r = httpx.get(url, headers=HEADERS, params=params, timeout=30)
        if r.status_code == 403:
            wait = int(r.headers.get("Retry-After", 65))
            log.warning("rate-limit on %s – sleeping %ss", url, wait)
            time.sleep(wait)
            r = httpx.get(url, headers=HEADERS, params=params, timeout=30)
        r.raise_for_status()
        return r
    except httpx.HTTPError as e:
        log.error("HTTP error %s – returning empty result", e)
        fake = httpx.Response(status_code=200, request=None)
        fake._content = b'{"items": []}'
        return fake

def fetch_code_items(query: str) -> List[Dict]:
    resp = _get(SEARCH_URL, params={"q": query, "per_page": 100, "page": 1})
    return resp.json().get("items", [])

def fetch_body(contents_url: str) -> str | None:
    meta = _get(contents_url).json()
    if meta.get("encoding") != "base64" or meta.get("size", 0) > 1_000_000:
        return None
    try:
        raw = base64.b64decode(meta["content"])
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return None
