import json
import sqlite3

from context_manager import ContextHelper
from permission import DenialSource
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
                    state IN ('RECORDED', 'STARTED', 'RETRY_READY', 'RESULT_RECORDED', 'DENIED', 'CANCELLED')
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
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_turns (
                session_id TEXT PRIMARY KEY,
                state TEXT NOT NULL CHECK(state IN ('RUNNING', 'COMPLETED', 'INTERRUPTED', 'FAILED'))
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS turn_controls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES agent_turns(session_id),
                kind TEXT NOT NULL CHECK(kind IN ('STEER', 'INTERRUPT')),
                content TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'PENDING'
                    CHECK(state IN ('PENDING', 'APPLIED', 'UNHANDLED'))
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_turn_controls_pending
            ON turn_controls(session_id, state, id)
        """)
        self.conn.commit()

    def start_turn(self):
        """Register a live turn before the CLI announces its session ID."""
        with self.conn:
            self.conn.execute(
                "INSERT INTO agent_turns(session_id, state) VALUES (?, 'RUNNING')",
                (self.session_id,)
            )

    def post_control(self, kind: str, content: str = "") -> int:
        """Trusted local CLI submission, allowed only while the turn is active."""
        if kind not in ("STEER", "INTERRUPT"):
            raise ValueError("Unknown control kind")
        if not isinstance(content, str) or len(content) > 4096:
            raise ValueError("Invalid control content")
        if kind == "STEER" and not content.strip():
            raise ValueError("Steering text must not be blank")
        with self.conn:
            self.conn.execute("BEGIN IMMEDIATE")
            turn = self.conn.execute(
                "SELECT state FROM agent_turns WHERE session_id = ?", (self.session_id,)
            ).fetchone()
            if turn is None or turn[0] != "RUNNING":
                raise ValueError("No active turn for this session")
            cursor = self.conn.execute(
                "INSERT INTO turn_controls(session_id, kind, content) VALUES (?, ?, ?)",
                (self.session_id, kind, content.strip())
            )
            return cursor.lastrowid

    def get_pending_controls(self) -> list[dict]:
        rows = self.conn.execute("""
            SELECT id, kind, content FROM turn_controls
            WHERE session_id = ? AND state = 'PENDING' ORDER BY id
        """, (self.session_id,)).fetchall()
        return [{"id": r[0], "kind": r[1], "content": r[2]} for r in rows]

    def has_pending_control(self) -> bool:
        return self.conn.execute("""
            SELECT 1 FROM turn_controls WHERE session_id = ? AND state = 'PENDING'
            LIMIT 1
        """, (self.session_id,)).fetchone() is not None

    def apply_control(self, control_id: int, history_item: dict) -> None:
        """Atomically acknowledge the control and append its user input to history."""
        serializable = ContextHelper.serialize_items([history_item])
        with self.conn:
            updated = self.conn.execute("""
                UPDATE turn_controls SET state = 'APPLIED'
                WHERE id = ? AND session_id = ? AND state = 'PENDING'
            """, (control_id, self.session_id))
            if updated.rowcount != 1:
                raise ValueError("Control already consumed")
            self.conn.execute(
                "INSERT INTO events(session_id, kind, payload) VALUES (?, 'history_items', ?)",
                (self.session_id, json.dumps({"items": serializable}, ensure_ascii=False))
            )

    def finish_turn(self, state: str, *, require_no_pending: bool = False) -> bool:
        """Finalize atomically against concurrent control submission.

        Successful completion requires an empty control queue. Failure/interrupt
        terminates the turn and explicitly identifies any unhandled controls.
        """
        if state not in ("COMPLETED", "INTERRUPTED", "FAILED"):
            raise ValueError("Invalid terminal state")
        with self.conn:
            self.conn.execute("BEGIN IMMEDIATE")
            if require_no_pending and self.has_pending_control():
                return False
            updated = self.conn.execute("""
                UPDATE agent_turns SET state = ?
                WHERE session_id = ? AND state = 'RUNNING'
            """, (state, self.session_id))
            if updated.rowcount != 1:
                raise ValueError("Turn is not running")
            self.conn.execute("""
                UPDATE turn_controls SET state = 'UNHANDLED'
                WHERE session_id = ? AND state = 'PENDING'
            """, (self.session_id,))
            return True

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

    def create_pending_approval(self, call_id: str,
                                expected_started_count: int) -> None:
        """Insert a distinct Approval record. No execution-state checks here.

        The FK establishes call identity; execution admission is separately
        governed by the tool_executions CAS + approval consumption transaction.
        """
        with self.conn:
            self.conn.execute(
                """INSERT INTO tool_approvals(
                       session_id, call_id, expected_started_count, status)
                   VALUES (?, ?, ?, 'PENDING')""",
                (self.session_id, call_id, expected_started_count)
            )

    def approve_tool_execution(self, call_id: str,
                               expected_started_count: int) -> None:
        """Only advance the Approval state machine; do not claim execution."""
        with self.conn:
            updated = self.conn.execute(
                """UPDATE tool_approvals SET status = 'APPROVED'
                   WHERE session_id = ? AND call_id = ?
                     AND expected_started_count = ? AND status = 'PENDING'""",
                (self.session_id, call_id, expected_started_count)
            )
            if updated.rowcount != 1:
                raise ValueError(f"Approval no longer pending for {call_id}")

    def acquire_tool_execution(self, call_id: str, expected_started_count: int,
                               *, retry: bool = False,
                               require_approval: bool = False) -> int:
        """Consume Approval (if required) and grant STARTED with the same CAS transaction."""
        expected_state = "RETRY_READY" if retry else "RECORDED"
        with self.conn:
            self.conn.execute("BEGIN IMMEDIATE")
            # Admission and the steering/interrupt queue share the same SQLite
            # transaction boundary, so an already-arrived control wins over
            # launching another stale tool (including after an approval prompt).
            if self.has_pending_control():
                raise ValueError("Pending steering or interrupt; tool not admitted")
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
                           retry: bool = False, denial_source: DenialSource = DenialSource.POLICY) -> None:
        """Atomically deny an unstarted attempt, including the final output and ledger.

        A retry DENIED means retry denied, NOT proof earlier external effects
        did not occur; started_count is intentionally preserved.
        """
        expected_state = "RETRY_READY" if retry else "RECORDED"
        payload = interaction.to_payload()
        with self.conn:
            if denial_source == DenialSource.USER:
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

    def cancel_recorded_tool(self, interaction: ToolInteraction) -> None:
        """Finalize an unstarted stale call with a durable cancellation output.

        Never cancel STARTED: its external side effects may be unknown.
        """
        payload = interaction.to_payload()
        with self.conn:
            updated = self.conn.execute("""
                UPDATE tool_executions SET state = 'CANCELLED'
                WHERE session_id = ? AND call_id = ? AND state = 'RECORDED'
                  AND started_count = 0 AND input_json = ?
            """, (self.session_id, interaction.call_id,
                  json.dumps(interaction.input, ensure_ascii=False)))
            if updated.rowcount != 1:
                raise ValueError(f"Call already started or not recorded: {interaction.call_id}")
            self.conn.execute("""
                UPDATE tool_approvals SET status = 'REJECTED'
                WHERE session_id = ? AND call_id = ? AND status IN ('PENDING', 'APPROVED')
            """, (self.session_id, interaction.call_id))
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

