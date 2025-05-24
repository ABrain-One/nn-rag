from __future__ import annotations
from typing import Dict, List
import json, pathlib, random, os

from pandas import DataFrame
from github import Github
from utils.github_utils import build_query, search_repositories_with_cache
from utils.neural_utils import neural_rank_repositories, generate_response_from_results
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ── lazy globals for the vector index --------------------------------------
_sentence_model = None   # type: ignore
_faiss_index    = None   # type: ignore
_chunks: List[str] | None = None

def _lazy_load_index() -> None:
    """Load MiniLM embedder, FAISS index, and chunk list once."""
    global _sentence_model, _faiss_index, _chunks
    if _faiss_index is not None:                         
        return


    root = pathlib.Path("rag.index")
    _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")   # 384-d vectors
    _faiss_index    = faiss.read_index(str(root / "faiss.index"))
    _chunks         = json.load(open(root / "chunks.json"))

def _nearest_blocks(query: str, k: int = 3) -> List[str]:
    """Return k code chunks semantically closest to the query string."""
    _lazy_load_index()
    
    qvec = _sentence_model.encode([query]).astype("float32")     # type: ignore
    _, ids = _faiss_index.search(qvec, k)                       # type: ignore
    lst = [_chunks[i] for i in ids[0]]                          # type: ignore
    random.shuffle(lst)                                         # tiny diversity
    return lst

# ── PUBLIC API (unchanged signature) ---------------------------------------
def retrieve_and_generate(
    keyword: str,
    language: str | None = None,
    owner: str | None = None,
    stars: str | None = None,
    max_results: int = 100,
    num_top: int = 3
) -> Dict:  
    qualifiers = {}
    if language: qualifiers["language"] = language
    if owner:    qualifiers["user"]     = owner
    if stars:    qualifiers["stars"]    = stars

    query = build_query(keyword, qualifiers)
    repo_results = search_repositories_with_cache(query, max_results=max_results)
    if not repo_results:
        return {"error": "No results found."}

    ranked = neural_rank_repositories(repo_results, keyword)
    summary = generate_response_from_results(keyword, ranked, num_results=num_top)

    # 2) NEW – attach code snippets
    blocks = _nearest_blocks(keyword, k=num_top)  # may be fewer than num_top
    for i, blk in enumerate(blocks):
        ranked[i]["code"] = blk

    return {"query": query, "results": ranked[:num_top], "generated_response": summary}

def data(
    keyword: str,
    language: str | None = None,
    owner: str | None = None,
    stars: str | None = None,
    max_results: int = 100,
    num_top: int = 3
) -> DataFrame:
    payload = retrieve_and_generate(keyword, language, owner, stars,
                                    max_results, num_top)
    if "error" in payload:
        return DataFrame()
    return DataFrame(payload["results"])