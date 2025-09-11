"""
File index storage using SQLite.

This module provides persistent storage for parsed Python files and their symbols
using SQLite database for efficient retrieval and caching.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .models import SymbolInfo


class FileIndexStore:
    """Persistent SQLite-based storage for file parsing results."""
    
    def __init__(self, db_path: Path = Path(".cache/index.sqlite3")):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    repo TEXT NOT NULL,
                    relpath TEXT NOT NULL,
                    mod_qual TEXT NOT NULL,
                    sha1 TEXT NOT NULL,
                    imports_json TEXT NOT NULL,
                    source TEXT,
                    PRIMARY KEY (repo, relpath, sha1)
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS symbols (
                    repo TEXT NOT NULL,
                    relpath TEXT NOT NULL,
                    sha1 TEXT NOT NULL,
                    name TEXT NOT NULL,
                    qual TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    line INTEGER NOT NULL,
                    code TEXT NOT NULL,
                    PRIMARY KEY (repo, relpath, sha1, qual)
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS repo_index_state (
                    repo TEXT PRIMARY KEY,
                    indexed_at TEXT NOT NULL,
                    file_count INTEGER NOT NULL
                )
            """)
            con.commit()

    @staticmethod
    def _sha1(data: str) -> str:
        """Generate SHA1 hash of the input string."""
        return hashlib.sha1(data.encode("utf-8", errors="ignore")).hexdigest()

    def get(self, repo: str, relpath: str, content: str) -> Optional[Tuple[str, List[str], List[SymbolInfo], str]]:
        """Retrieve cached parsing results for a file."""
        sha1 = self._sha1(content)
        with self._lock, sqlite3.connect(self.db_path) as con:
            cur = con.execute(
                "SELECT mod_qual, imports_json, source FROM files WHERE repo=? AND relpath=? AND sha1=?",
                (repo, relpath, sha1)
            )
            row = cur.fetchone()
            if not row:
                return None
            mod_qual, imports_json, source = row
            imports = json.loads(imports_json)
            sym_rows = con.execute(
                "SELECT name, qual, kind, line, code FROM symbols WHERE repo=? AND relpath=? AND sha1=?",
                (repo, relpath, sha1)
            ).fetchall()
            syms = [
                SymbolInfo(name=r[0], qualified_name=r[1], kind=r[2], line_number=r[3],
                           source_code=r[4], location=relpath)
                for r in sym_rows
            ]
            return mod_qual, imports, syms, source or ""

    def put(self, repo: str, relpath: str, mod_qual: str, imports: List[str],
            symbols: Iterable[SymbolInfo], source: str) -> None:
        """Store parsing results for a file."""
        sha1 = self._sha1(source)
        imports_json = json.dumps(sorted(set(imports)))
        with self._lock, sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT OR REPLACE INTO files(repo, relpath, mod_qual, sha1, imports_json, source) VALUES(?,?,?,?,?,?)",
                (repo, relpath, mod_qual, sha1, imports_json, source)
            )
            con.execute("DELETE FROM symbols WHERE repo=? AND relpath=? AND sha1=?", (repo, relpath, sha1))
            con.executemany(
                "INSERT INTO symbols(repo, relpath, sha1, name, qual, kind, line, code) VALUES(?,?,?,?,?,?,?,?)",
                [(repo, relpath, sha1, s.name, s.qualified_name, s.kind, s.line_number, s.source_code) for s in symbols]
            )
            con.commit()

    def repo_has_index(self, repo: str) -> bool:
        """Check if a repository has been indexed."""
        with sqlite3.connect(self.db_path) as con:
            row = con.execute(
                "SELECT file_count FROM repo_index_state WHERE repo=?",
                (repo,)
            ).fetchone()
            if row and int(row[0]) > 0:
                return True
            # Fallback check (older DBs): look for any file rows for this repo
            row2 = con.execute(
                "SELECT 1 FROM files WHERE repo=? LIMIT 1",
                (repo,)
            ).fetchone()
            return bool(row2)

    def mark_repo_indexed(self, repo: str) -> None:
        """Mark a repository as indexed with current file count."""
        with sqlite3.connect(self.db_path) as con:
            cnt = con.execute("SELECT COUNT(1) FROM files WHERE repo=?", (repo,)).fetchone()[0]
            con.execute(
                "INSERT OR REPLACE INTO repo_index_state(repo, indexed_at, file_count) VALUES(?,?,?)",
                (repo, datetime.utcnow().isoformat(), int(cnt))
            )
            con.commit()
