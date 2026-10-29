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
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tool_executions (
                session_id TEXT NOT NULL,
                call_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                input_json TEXT NOT NULL,
                state TEXT NOT NULL CHECK(
                    state IN ('RECORDED', 'STARTED', 'RETRY_READY', 'RESULT_RECORDED', 'DENIED')
                ),
                started_count INTEGER NOT NULL DEFAULT 0 CHECK(started_count >= 0),
                PRIMARY KEY (session_id, call_id)
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tool_executions_pending
            ON tool_executions(session_id, state)
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tool_approvals (
                session_id TEXT NOT NULL,
                call_id TEXT NOT NULL,
                expected_started_count INTEGER NOT NULL CHECK(expected_started_count >= 0),
                status TEXT NOT NULL CHECK(
                    status IN ('PENDING', 'APPROVED', 'CONSUMED', 'REJECTED')
                ),
                PRIMARY KEY (session_id, call_id, expected_started_count),
                FOREIGN KEY (session_id, call_id)
                  REFERENCES tool_executions(session_id, call_id)
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
        """Atomically journal history and register new function calls.

        The unique key on (session_id, call_id) prevents a duplicated call
        from ever entering the durable history as a new execution request.
        """
        serializable = ContextHelper.serialize_items(items)
        with self.conn:
            self.conn.execute(
                "INSERT INTO events(session_id, kind, payload) VALUES (?, ?, ?)",
                (self.session_id, "history_items",
                 json.dumps({"items": serializable}, ensure_ascii=False))
            )
            for item in serializable:
                if item.get("type") == "function_call":
                    self.conn.execute(
                        """INSERT INTO tool_executions(
                            session_id, call_id, tool_name, input_json, state
                        ) VALUES (?, ?, ?, ?, 'RECORDED')""",
                        (self.session_id, item["call_id"], item["name"],
                         json.dumps(item, ensure_ascii=False))
                    )

    def get_tool_execution(self, call_id: str) -> dict | None:
        row = self.conn.execute(
            """SELECT call_id, tool_name, input_json, state, started_count
               FROM tool_executions WHERE session_id = ? AND call_id = ?""",
            (self.session_id, call_id)
        ).fetchone()
        if row is None:
            return None
        return {
            "call_id": row[0], "tool_name": row[1], "input": json.loads(row[2]),
            "state": row[3], "started_count": row[4],
        }

    def list_tool_executions(self) -> list[dict]:
        rows = self.conn.execute(
            """SELECT call_id, tool_name, input_json, state, started_count
               FROM tool_executions WHERE session_id = ? ORDER BY rowid""",
            (self.session_id,)
        ).fetchall()
        return [
            {"call_id": row[0], "tool_name": row[1], "input": json.loads(row[2]),
             "state": row[3], "started_count": row[4]}
            for row in rows
        ]

    def get_tool_approval(self, call_id: str, expected_started_count: int) -> str | None:
        row = self.conn.execute(
            """SELECT status FROM tool_approvals
               WHERE session_id = ? AND call_id = ? AND expected_started_count = ?""",
            (self.session_id, call_id, expected_started_count)
        ).fetchone()
        return row[0] if row else None

    def create_pending_approval(self, call_id: str, expected_started_count: int,
                                *, retry: bool = False) -> None:
        """Create one approval request; duplicate attempts are NOT silently ignored.

        Existing PENDING/APPROVED approvals are read separately by the Manager.
        The INSERT is conditional on the current execution state and version.
        """
        expected_state = "RETRY_READY" if retry else "RECORDED"
        with self.conn:
            inserted = self.conn.execute(
                """INSERT INTO tool_approvals(
                       session_id, call_id, expected_started_count, status)
                   SELECT session_id, call_id, started_count, 'PENDING'
                   FROM tool_executions
                   WHERE session_id = ? AND call_id = ?
                     AND state = ? AND started_count = ?""",
                (self.session_id, call_id, expected_state, expected_started_count)
            )
            if inserted.rowcount != 1:
                raise ValueError(f"Cannot request approval for {call_id}")

    def approve_tool_execution(self, call_id: str, expected_started_count: int,
                               *, retry: bool = False) -> None:
        """Trusted user decision: persist APPROVED, not STARTED."""
        expected_state = "RETRY_READY" if retry else "RECORDED"
        with self.conn:
            updated = self.conn.execute(
                """UPDATE tool_approvals SET status = 'APPROVED'
                   WHERE session_id = ? AND call_id = ?
                     AND expected_started_count = ? AND status = 'PENDING'
                     AND EXISTS (
                         SELECT 1 FROM tool_executions
                         WHERE session_id = tool_approvals.session_id
                           AND call_id = tool_approvals.call_id
                           AND state = ? AND started_count = ?
                     )""",
                (self.session_id, call_id, expected_started_count,
                 expected_state, expected_started_count)
            )
            if updated.rowcount != 1:
                raise ValueError(f"Approval no longer pending for {call_id}")

    def acquire_tool_execution(self, call_id: str, expected_started_count: int,
                               *, retry: bool = False,
                               require_approval: bool = False) -> int:
        """Consume Approval (if required) and grant STARTED with the same CAS transaction."""
        expected_state = "RETRY_READY" if retry else "RECORDED"
        with self.conn:
            if require_approval:
                updated = self.conn.execute(
                    """UPDATE tool_approvals SET status = 'CONSUMED'
                       WHERE session_id = ? AND call_id = ?
                         AND expected_started_count = ? AND status = 'APPROVED'""",
                    (self.session_id, call_id, expected_started_count)
                )
                if updated.rowcount != 1:
                    raise ValueError(f"Approval not granted for {call_id}")
            updated = self.conn.execute(
                """UPDATE tool_executions
                   SET state = 'STARTED', started_count = started_count + 1
                   WHERE session_id = ? AND call_id = ?
                     AND state = ? AND started_count = ?""",
                (self.session_id, call_id, expected_state, expected_started_count)
            )
            if updated.rowcount != 1:
                raise ValueError(f"Execution admission rejected for {call_id}")
        return expected_started_count + 1

    def authorize_tool_retry(self, call_id: str, expected_started_count: int) -> None:
        """Explicit recovery transition; does not itself invoke the handler."""
        with self.conn:
            updated = self.conn.execute(
                """UPDATE tool_executions SET state = 'RETRY_READY'
                   WHERE session_id = ? AND call_id = ?
                     AND state = 'STARTED' AND started_count = ?""",
                (self.session_id, call_id, expected_started_count)
            )
            if updated.rowcount != 1:
                raise ValueError(f"Retry authorization rejected for {call_id}")

    def append_tool_denial(self, interaction: ToolInteraction,
                           expected_started_count: int, *,
                           retry: bool = False, rejected_approval: bool = False) -> None:
        """Atomically deny an unstarted attempt, including the final output and ledger.

        A retry DENIED means retry denied, NOT proof earlier external effects
        did not occur; started_count is intentionally preserved.
        """
        expected_state = "RETRY_READY" if retry else "RECORDED"
        payload = interaction.to_payload()
        with self.conn:
            if rejected_approval:
                updated = self.conn.execute(
                    """UPDATE tool_approvals SET status = 'REJECTED'
                       WHERE session_id = ? AND call_id = ?
                         AND expected_started_count = ? AND status = 'PENDING'""",
                    (self.session_id, interaction.call_id, expected_started_count)
                )
                if updated.rowcount != 1:
                    raise ValueError(f"Approval cannot be rejected: {interaction.call_id}")
            updated = self.conn.execute(
                """UPDATE tool_executions SET state = 'DENIED'
                   WHERE session_id = ? AND call_id = ?
                     AND state = ? AND started_count = ?""",
                (self.session_id, interaction.call_id, expected_state, expected_started_count)
            )
            if updated.rowcount != 1:
                raise ValueError(f"Denial rejected by execution CAS: {interaction.call_id}")
            self._append_output_and_ledger(interaction, payload)

    def _append_output_and_ledger(self, interaction: ToolInteraction, payload: dict) -> None:
        """Caller must hold the enclosing transaction."""
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

    def append_tool_interaction(self, interaction: ToolInteraction,
                                expected_started_count: int) -> None:
        """Atomically commit state, context-result event and pair ledger.

        A stale attempt's result cannot overwrite a newer STARTED version.
        """
        payload = interaction.to_payload()
        with self.conn:
            updated = self.conn.execute(
                """UPDATE tool_executions SET state = 'RESULT_RECORDED'
                   WHERE session_id = ? AND call_id = ?
                     AND state = 'STARTED' AND started_count = ?""",
                (self.session_id, interaction.call_id, expected_started_count)
            )
            if updated.rowcount != 1:
                raise ValueError(
                    f"Stale or invalid tool result commit: {interaction.call_id}"
                )
            self._append_output_and_ledger(interaction, payload)

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

