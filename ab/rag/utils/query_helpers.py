import re
from typing import List

_UUID_OR_HASH = re.compile(
    r"[-_]?(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$",
    re.I
)
_VERSION_TAG = re.compile(r"[-_]?v?\d+$|[-_](alpha|beta|rc)$", re.I)

def make_queries(arch: str) -> List[str]:
    clean = _UUID_OR_HASH.sub("", arch)
    clean = _VERSION_TAG.sub("", clean)
    slug = re.sub(r"[^A-Za-z0-9]", "", clean).lower()
    spaced = re.sub(r"[-_]+", " ", clean).strip().lower()
    if not slug:
        return [f"{arch} language:Python"]
    return [
        f"{slug} language:Python filename:{slug}.py",
        f"{spaced} language:Python",
        f"pytorch {spaced} language:Python",
        f"torch {spaced} language:Python",
    ]

def clean_chunk(chunk: str) -> str:
    lines = [
        ln for ln in chunk.splitlines()
        if not ln.lstrip().startswith(("```", "#", ">"))
    ]
    return "\n".join(lines).strip()
