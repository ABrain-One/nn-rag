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
