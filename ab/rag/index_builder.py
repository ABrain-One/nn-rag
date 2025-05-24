from __future__ import annotations
import io, json, os, pathlib, random, re, sys, zipfile

import faiss, requests, tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

REPOS = [
    "pytorch/vision",
    "ultralytics/yolov5",        # default branch = master, not main
    "facebookresearch/detectron2",
    "mit-han-lab/once-for-all",
    "rwightman/pytorch-image-models",
]

OUT = pathlib.Path("rag.index")
OUT.mkdir(exist_ok=True)

BLANK = re.compile(r"\n\s*\n")
KEEP  = re.compile(r"\b(Conv2d|Linear)\b")

def regex_chunks(text: str) -> list[str]:
    return [blk.strip() for blk in BLANK.split(text) if KEEP.search(blk)]

def default_branch(repo: str) -> str | None:
    """Return GitHub repo default branch via REST; None if rate-limited."""
    url = f"https://api.github.com/repos/{repo}"
    try:
        j = requests.get(url, timeout=10).json()
        return j.get("default_branch")
    except Exception:
        return None

def fetch_chunks(repo: str) -> list[str]:
    """
    Download repo ZIP (default branch → main → master) and return chunks.
    Skips repo if all downloads fail or ZIP invalid.
    """
    for branch in filter(
        None,
        [default_branch(repo), "main", "master"]         # keep unique order
    ):
        zip_url = f"https://github.com/{repo}/archive/refs/heads/{branch}.zip"
        r = requests.get(zip_url, timeout=30)
        if r.status_code != 200 or "zip" not in r.headers.get("Content-Type", ""):
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                out = []
                for info in z.infolist():
                    if info.filename.endswith(".py"):
                        data = z.read(info).decode("utf-8", errors="ignore")
                        out.extend(regex_chunks(data))
                return out
        except zipfile.BadZipFile:
            continue
    print(f"[WARN] {repo}: no valid ZIP on default/main/master", file=sys.stderr)
    return []

def main() -> None:
    load_dotenv()                                    
    chunks: list[str] = []

    for repo in REPOS:
        print(f"[INFO] fetching {repo} …")
        c = fetch_chunks(repo)
        print(f"       → {len(c):,} chunks")
        chunks.extend(c)

    if not chunks:
        sys.exit("No code chunks extracted – check repo list/network")

    random.shuffle(chunks)
    print(f"[INFO] total chunks: {len(chunks):,} – embedding …")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")        
    vecs = embedder.encode(chunks, show_progress_bar=True).astype("float32")

    index = faiss.IndexFlatIP(vecs.shape[1])                   
    index.add(vecs)

    index_path = OUT / "faiss.index"
    faiss.write_index(index, str(index_path))                 
    (OUT / "chunks.json").write_text(json.dumps(chunks))

    print(f"✅  stored {len(chunks):,} chunks → {index_path}")

if __name__ == "__main__":
    main()
