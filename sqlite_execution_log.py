import json
import sqlite3

from context_manager import ContextHelper
from tool_model import ToolInteraction


class SQLiteExecutionLog:
    def __init__(self, db_path: str, session_id: str):
        self.session_id = session_id
        self.conn = sqlite3.connect(db_path)

        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_session
            ON events(session_id, id)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_context_snapshot
            ON events(session_id, kind, id);
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tool_interactions (
                session_id TEXT NOT NULL,
                call_id TEXT NOT NULL,
                event_id INTEGER NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (session_id, call_id),
                FOREIGN KEY (event_id) REFERENCES events(id)
            )
        """)
        self.conn.commit()

    def get_session_id(self) -> str:
        return self.session_id

    def append_event(self, kind: str, payload: dict):
        data = json.dumps(payload, ensure_ascii=False)
        with self.conn:
            self.conn.execute("""
                INSERT INTO events(session_id, kind, payload)
                VALUES (?, ?, ?)
            """, (self.session_id, kind, data))

    def append_items(self, items: list):
        serializable = ContextHelper.serialize_items(items)
        self.append_event("history_items", {"items": serializable})

    def append_tool_interaction(self, interaction: ToolInteraction) -> None:
        """Atomically journal the tool output and index the complete call/result pair."""
        payload = interaction.to_payload()
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO events(session_id, kind, payload) VALUES (?, ?, ?)",
                (self.session_id, "history_items",
                 json.dumps({"items": [payload["output"]]}, ensure_ascii=False))
            )
            self.conn.execute(
                "INSERT INTO tool_interactions(session_id, call_id, event_id, payload) VALUES (?, ?, ?, ?)",
                (self.session_id, interaction.call_id, cursor.lastrowid,
                 json.dumps(payload, ensure_ascii=False))
            )

    def get_tool_interaction(self, call_id: str) -> ToolInteraction | None:
        row = self.conn.execute(
            "SELECT payload FROM tool_interactions WHERE session_id = ? AND call_id = ?",
            (self.session_id, call_id)
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        return ToolInteraction(call_id=call_id, input=payload["input"], output=payload["output"])

    def append_step_complete(self, end: int):
        self.append_event("step_completed", {"end": end})

    def append_context_replaced(self, history: list, step_ends: list[int]):
        serializable = ContextHelper.serialize_items(history)
        self.append_event("context_replaced", {"history": serializable, "step_ends": step_ends})

    def get_events(self) -> list[dict]:
        rows = self.conn.execute("""
            SELECT kind, payload FROM events
            WHERE session_id = ?
            ORDER BY id ASC
        """, (self.session_id,)).fetchall()
        return self.serialize_return_rows(rows)

    def get_snapshots_for_recovery(self) -> list[dict]:
        rows = self.conn.execute("""
            SELECT kind, payload FROM events
            WHERE session_id = ?
            AND id >= COALESCE((
                SELECT MAX(id) FROM events WHERE session_id = ? AND kind = 'context_replaced'
            ),0) ORDER BY id ASC
        """, (self.session_id, self.session_id)).fetchall()
        return self.serialize_return_rows(rows)

    def close(self):
        self.conn.close()

    @staticmethod
    def serialize_return_rows(rows: list) -> list[dict]:
        return [
            {"kind": kind, **json.loads(payload)} for kind, payload in rows
        ]

