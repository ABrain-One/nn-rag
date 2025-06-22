import os, pathlib, time, logging
from dotenv import load_dotenv

load_dotenv()

# GitHub API
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or "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"
CACHE_FILE = "search_cache.db"
MODEL_NAME = "all-MiniLM-L6-v2"
GENERATOR_MODEL = "gpt2"

HEADERS  = {
    "Accept":        "application/vnd.github+json",
    **({"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}),
}
SEARCH_URL = "https://api.github.com/search/code"

# cache settings
TTL_SECONDS = 86_400  # 24h
CACHE_DIR = pathlib.Path.home() / ".cache" / "nn-rag" / "rest"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# on import, purge old cache files
for fp in CACHE_DIR.glob("*.json"):
    try:
        if fp.stat().st_mtime < time.time() - TTL_SECONDS:
            fp.unlink()
    except Exception:
        logging.getLogger(__name__).warning("failed to purge %s", fp)
