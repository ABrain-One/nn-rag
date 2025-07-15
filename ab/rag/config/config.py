import os, pathlib, time, logging
from dotenv import load_dotenv
import re

load_dotenv()

from pathlib import Path

_ROOT = Path.home() / ".cache" / "nn-rag"
_ROOT.mkdir(exist_ok=True)
_JSON = _ROOT / "json"
_JSON.mkdir(exist_ok=True)
TTL = 86_400  # seconds (24h)

# GitHub API
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or "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"

_MAX_LINES = 800

_CLASS_RE = re.compile(
    r"\bclass\s+(?P<name>[A-Za-z_]\w*)\s*\([^)]*\):[\s\S]*?(?=\bclass\s|\Z)",
    re.M,
)
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
SEARCH_URL = "https://api.github.com/search/code"
_MIN_REMAIN = 2
