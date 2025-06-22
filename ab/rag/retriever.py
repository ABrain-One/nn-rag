# ab/rag/retriever.py
from typing import List
import logging

from .utils.cache import read_cache, save_cache
from .utils.http_helpers import fetch_code_items, fetch_body
from utils.query_helpers import make_queries, clean_chunk
from .utils.ast_splitter import iter_chunks

log = logging.getLogger(__name__)

class Retriever:
    """
    >>> r = Retriever()
    >>> r.search("AirNext-ff05392a…", k=3)
    """
    def search(self, arch: str, k: int = 3) -> List[str]:
        # 0) cache
        if (cached := read_cache(arch)) is not None:
            return cached[:k]

        # 1) build & run queries
        chunks: List[str] = []
        for q in make_queries(arch):
            for item in fetch_code_items(q):
                text = fetch_body(item["url"])
                if not text:
                    continue
                for chunk in iter_chunks(text):
                    chunks.append(chunk)
                    if len(chunks) >= k:
                        break
                if len(chunks) >= k:
                    break
            if chunks:
                log.info("✔ %s → %d snippet(s)", q, len(chunks))
                break

        if not chunks:
            log.warning("✘ no snippets for %s", arch)

        out = [clean_chunk(ch) for ch in chunks[:k]]
        save_cache(arch, out)
        return out
