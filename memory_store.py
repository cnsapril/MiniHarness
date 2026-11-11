"""Small project-scoped, durable memory store independent of the execution ledger.

Only explicitly approved writes should reach this store through the tool layer.
Memory is contextual data, never an authorization or permission source.
"""

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


MEMORY_KINDS = frozenset({"fact", "decision", "preference"})
MAX_MEMORY_CHARS = 4096


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _content(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_MEMORY_CHARS:
        raise ValueError("Memory content must be 1-4096 characters")
    return value.strip()


class MemoryStore:
    """Project-local facts, versioned updates, and token-based FTS5 retrieval."""

    def __init__(self, db_path: str | Path):
        self.conn = sqlite3.connect(str(db_path), timeout=5)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA busy_timeout = 5000")
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    kind TEXT NOT NULL CHECK(kind IN ('fact', 'decision', 'preference')),
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active'
                        CHECK(status IN ('active', 'superseded', 'invalidated')),
                    supersedes_id TEXT REFERENCES memories(id),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(memory_id UNINDEXED, content, tokenize='unicode61')
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_scope_status
                ON memories(scope, status)
            """)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _insert(self, scope: str, kind: str, content: str, source: str,
                supersedes_id: str | None = None) -> str:
        memory_id = uuid4().hex
        timestamp = _now()
        self.conn.execute("""
            INSERT INTO memories
            (id, scope, kind, content, source, status, supersedes_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?)
        """, (memory_id, scope, kind, content, source, supersedes_id, timestamp, timestamp))
        self.conn.execute("INSERT INTO memories_fts(memory_id, content) VALUES (?, ?)",
                          (memory_id, content))
        return memory_id

    def save(self, scope: str, kind: str, content: str, *,
             source: str = "user_approved_tool_call") -> str:
        if not isinstance(scope, str) or not scope.strip():
            raise ValueError("Memory scope is required")
        if kind not in MEMORY_KINDS:
            raise ValueError("Invalid memory kind")
        text = _content(content)
        with self.conn:
            return self._insert(scope, kind, text, source)

    def update(self, scope: str, memory_id: str, content: str, *,
               source: str = "user_approved_tool_call") -> str:
        text = _content(content)
        with self.conn:
            old = self.conn.execute("""
                SELECT kind FROM memories WHERE id = ? AND scope = ? AND status = 'active'
            """, (memory_id, scope)).fetchone()
            if old is None:
                raise ValueError("Active memory not found in this project")
            self.conn.execute("""
                UPDATE memories SET status = 'superseded', updated_at = ? WHERE id = ?
            """, (_now(), memory_id))
            self.conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
            return self._insert(scope, old["kind"], text, source, supersedes_id=memory_id)

    def invalidate(self, scope: str, memory_id: str) -> None:
        with self.conn:
            result = self.conn.execute("""
                UPDATE memories SET status = 'invalidated', updated_at = ?
                WHERE id = ? AND scope = ? AND status = 'active'
            """, (_now(), memory_id, scope))
            if result.rowcount != 1:
                raise ValueError("Active memory not found in this project")
            self.conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))

    def get(self, scope: str, memory_id: str) -> dict | None:
        row = self.conn.execute("""
            SELECT * FROM memories WHERE scope = ? AND id = ?
        """, (scope, memory_id)).fetchone()
        return dict(row) if row else None

    def search(self, scope: str, query: str, limit: int = 5) -> list[dict]:
        if not isinstance(query, str) or not 1 <= len(query) <= 300:
            raise ValueError("Query must be 1-300 characters")
        if type(limit) is not int or not 1 <= limit <= 10:
            raise ValueError("Limit must be 1-10")
        # Quote tokens to avoid interpreting LLM-provided text as FTS syntax.
        # FTS5 unicode61 is lexical rather than embedding-based semantic search.
        tokens = list(dict.fromkeys(re.findall(r"[^\W_]+", query, flags=re.UNICODE)))[:16]
        if not tokens:
            return []
        match = " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)
        rows = self.conn.execute("""
            SELECT m.id, m.scope, m.kind, m.content, m.source, m.created_at,
                   m.updated_at, m.supersedes_id
            FROM memories_fts
            JOIN memories AS m ON m.id = memories_fts.memory_id
            WHERE memories_fts MATCH ? AND m.scope = ? AND m.status = 'active'
            ORDER BY bm25(memories_fts), m.updated_at DESC
            LIMIT ?
        """, (match, scope, limit)).fetchall()
        return [dict(row) for row in rows]
