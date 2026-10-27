import json
import sqlite3

from context_manager import ContextHelper


class SQLiteExecutionLog:
    def __init__(self, db_path: str, session_id: str):
        self.session_id = session_id
        self.conn = sqlite3.connect(db_path)

        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
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
        with self.conn:
            recovery_snapshot_id = self.conn.execute("""
                SELECT id FROM events
                WHERE session_id = ? AND kind = 'context_replaced'
                ORDER BY id DESC LIMIT 1
            """, (self.session_id,)).fetchone()
            if recovery_snapshot_id is None:
                # 没有Snapshot来recover，只能返回全量history
                return self.get_events()
            # 返回最新的Snapshot和后续增量history
            rows = self.conn.execute("""
                SELECT kind, payload FROM events
                WHERE session_id = ? AND id >= ?
                ORDER BY id ASC
            """, (self.session_id, recovery_snapshot_id[0])).fetchall()
            return self.serialize_return_rows(rows)

    def close(self):
        self.conn.close()

    @staticmethod
    def serialize_return_rows(rows: list) -> list[dict]:
        return [
            {"kind": kind, **json.loads(payload)} for kind, payload in rows
        ]

