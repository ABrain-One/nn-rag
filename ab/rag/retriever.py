from pathlib import Path
import logging
import sys
from typing import List, Tuple, Optional

from .utils.cache import load, save, in_cache
from .utils.github_utils import search_code, fetch_raw
from .utils.query_helpers import extract_class, _canonical, DEFAULT_SOURCES, BLOCKS_100

log = logging.getLogger(__name__)
# Logging configuration moved to the __main__ block to avoid interfering with user settings.
class Retriever:
    def best_path(self, name: str) -> Tuple[str, str]:
        # Use a default path if available
        if name in DEFAULT_SOURCES:
            return DEFAULT_SOURCES[name]

        target = _canonical(name)
        query = f"class {target} in:file language:Python"
        cache_key = f"gh::{query}"

        hits = load(cache_key)
        if hits is None:
            hits = search_code(query)
            save(cache_key, hits)

        for item in hits:
            repo = item["repository"]["full_name"]
            path = item["path"]
            url = item["url"]
            body_key = f"body::{url}"
            body = load(body_key)
            if body is None:
                raw = fetch_raw(repo, path)
                body = raw
                save(body_key, body)
            if f"class {target}" in body:
                return repo, path

        raise FileNotFoundError(f"No suitable source for {name}")

    def file(self, repo: str, path: str) -> str:
        cache_key = f"raw::{repo}::{path}"
        src = load(cache_key)
        if src is None:
            src = fetch_raw(repo, path)
            save(cache_key, src)
        return src

    def try_file(self, repo: str, path: str) -> Optional[str]:
        """
        Like `file()`, but returns None if the file isn't found.
        """
        try:
            return self.file(repo, path)
        except FileNotFoundError as e:
            log.debug("try_file: %s – %s", path, e)
            return None

    def get_block(self, name: str) -> Optional[str]:
        """
        Retrieve a single class definition by name.
        """
        try:
            repo, path = self.best_path(name)
            src = self.file(repo, path)
            return extract_class(src, _canonical(name))
        except Exception as e:
            log.warning("⚠️ skip %s – %s", name, e)
            return None

    def dump_all_blocks(self, dest: str | Path = "blocks") -> None:
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        for name in BLOCKS_100:
            code = self.get_block(name)
            if code:
                (dest / f"{name}.py").write_text(code)
                log.info("✓ %s", name)

    def load_cache(self, key: str):
        return load(key)

__all__ = ["Retriever","BLOCKS_100","sample_blocks","in_cache"]

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dump", default="blocks",
        help="Directory to dump all 100 blocks"
    )
    ap.add_argument(
        "--name",
        help="Fetch a single block by name (e.g. GLU) and print it"
    )
    args = ap.parse_args()
    retriever = Retriever()

    if args.name:
        code = retriever.get_block(args.name)
        if code:
            print(code)
        else:
            print(f"❌ failed to fetch block '{args.name}'", file=sys.stderr)
            sys.exit(1)
    else:
        retriever.dump_all_blocks(args.dump)