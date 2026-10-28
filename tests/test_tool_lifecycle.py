import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from context_manager import ContextBudget, ContextManager
from session import AgentSession
from sqlite_execution_log import SQLiteExecutionLog
from tool_execution import (RecoveryInspector, ToolExecutionManager, ToolPhase,
                            UnsafeToolRetryError, UnknownToolOutcomeError)
from tool_registry import ToolRegistry, ToolSafety, ToolSpec


class ToolLifecycleTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.db = str(Path(temp.name) / "agent.db")
        self.log = SQLiteExecutionLog(self.db, "session-1")
        self.addCleanup(self.log.close)
        self.session = AgentSession(
            ContextManager(ContextBudget(16384, 2048), 0.8), self.log
        )
        self.read_handler = lambda path: ["a.py"]
        self.side_handler = lambda: None
        self.registry = ToolRegistry([
            ToolSpec("list_files", lambda path: self.read_handler(path),
                     ToolSafety.READ_ONLY, "List files", {"type": "object"}),
            ToolSpec("increment_counter", lambda: self.side_handler(),
                     ToolSafety.SIDE_EFFECTING, "Increment", {"type": "object"}),
        ])
        self.manager = ToolExecutionManager(self.log, self.registry)
        self.session.append([{"role": "user", "content": "Run tool"}])

    def item(self, tool="list_files", call_id="call_1"):
        return {"type": "function_call", "call_id": call_id, "name": tool,
                "arguments": json.dumps({"path": "."} if tool == "list_files" else {})}

    def call(self, tool="list_files", call_id="call_1"):
        item = self.item(tool, call_id)
        self.session.append([item])
        return item

    def commit(self, outcome):
        self.session.record_tool_interaction(outcome.interaction, outcome.started_count)

    def run_tool(self, item, *, retry=False):
        outcome = self.manager.execute(item, retry=retry)
        self.commit(outcome)
        return outcome

    def inspect(self):
        return RecoveryInspector(self.log).inspect()

    def test_tool_registry_is_single_source(self):
        self.assertEqual([x["name"] for x in self.registry.llm_schemas()],
                         ["list_files", "increment_counter"])
        self.assertEqual(self.registry.get("list_files").safety, ToolSafety.READ_ONLY)
        with self.assertRaises(ValueError):
            ToolRegistry([self.registry.get("list_files"), self.registry.get("list_files")])

    def test_call_registration_and_history_are_atomic(self):
        item = self.item()
        self.session.append([item])
        self.assertEqual(self.inspect().get("call_1").phase, ToolPhase.RECORDED)
        before_events = self.log.get_events()
        before_history = self.session.get_history()
        with self.assertRaises(sqlite3.IntegrityError):
            self.session.append([item])
        self.assertEqual(self.log.get_events(), before_events)
        self.assertEqual(self.session.get_history(), before_history)

    def test_duplicate_in_same_response_rolls_back_whole_batch(self):
        item = self.item()
        before = self.log.get_events()
        with self.assertRaises(sqlite3.IntegrityError):
            self.session.append([item, item])
        self.assertIsNone(self.log.get_tool_execution("call_1"))
        self.assertEqual(self.log.get_events(), before)

    def test_not_recorded_cannot_execute(self):
        item = self.item()
        with self.assertRaises(UnsafeToolRetryError):
            self.manager.execute(item)
        self.assertEqual(self.log.list_tool_executions(), [])

    def test_restore_recovers_recorded_call_without_writes(self):
        item = self.call()
        events = self.log.get_events()
        recovered = AgentSession.restore(
            ContextManager(ContextBudget(16384, 2048), 0.8), self.log
        )
        self.assertEqual(recovered.get_history(), self.session.get_history())
        self.assertEqual(self.log.get_events(), events)
        self.assertEqual(self.inspect().get(item["call_id"]).started_count, 0)

    def test_started_persisted_before_handler_and_result_by_session(self):
        item = self.call()
        def handler(path):
            status = self.inspect().get("call_1")
            self.assertEqual(status.phase, ToolPhase.STARTED)
            self.assertEqual(status.started_count, 1)
            return ["main.py"]
        self.read_handler = handler
        outcome = self.manager.execute(item)
        self.assertTrue(outcome.handler_returned_normally)
        self.assertEqual(outcome.started_count, 1)
        self.assertEqual(self.inspect().get("call_1").phase, ToolPhase.STARTED)
        self.assertIsNone(self.log.get_tool_interaction("call_1"))
        self.commit(outcome)
        self.assertEqual(self.inspect().get("call_1").phase, ToolPhase.RESULT_RECORDED)
        self.assertEqual(self.log.get_tool_interaction("call_1"), outcome.interaction)
        self.assertNotIn("tool_execution_started", [x["kind"] for x in self.log.get_events()])

    def test_db_denies_start_before_handler(self):
        item = self.call()
        invoked = []
        self.read_handler = lambda path: invoked.append(path)
        self.log.conn.execute("""
            CREATE TRIGGER deny_started BEFORE UPDATE OF state ON tool_executions
            WHEN NEW.state = 'STARTED'
            BEGIN SELECT RAISE(ABORT, 'injected'); END
        """)
        with self.assertRaises(sqlite3.IntegrityError):
            self.manager.execute(item)
        self.assertEqual(invoked, [])
        self.assertEqual(self.inspect().get("call_1").phase, ToolPhase.RECORDED)

    def test_crash_after_started_commit_before_handler(self):
        item = self.call()
        original = self.log.acquire_tool_execution
        invoked = []
        self.read_handler = lambda path: invoked.append(path)
        def commit_then_crash(*args, **kwargs):
            original(*args, **kwargs)
            raise SystemExit("crash before handler")
        with patch.object(self.log, "acquire_tool_execution", commit_then_crash):
            with self.assertRaises(SystemExit):
                self.manager.execute(item)
        self.assertEqual(invoked, [])
        self.assertEqual(self.inspect().get("call_1").phase, ToolPhase.STARTED)

    def test_two_managers_cannot_execute_same_call(self):
        item = self.call()
        called = []
        self.read_handler = lambda path: called.append(path) or ["a.py"]
        first = self.manager.execute(item)  # STARTED, result intentionally uncommitted
        other = SQLiteExecutionLog(self.db, "session-1")
        try:
            second = ToolExecutionManager(other, self.registry)
            with self.assertRaises(UnsafeToolRetryError):
                second.execute(item)
            self.assertEqual(called, ["."])
            self.commit(first)
            self.assertEqual(self.inspect().get("call_1").started_count, 1)
        finally:
            other.close()

    def test_side_effect_crash_does_not_allow_retry(self):
        item = self.call("increment_counter")
        counter_db = Path(self.db).with_suffix(".counter.db")
        with sqlite3.connect(counter_db) as db:
            db.execute("CREATE TABLE counter(value INTEGER)")
            db.execute("INSERT INTO counter VALUES (0)")
        def handler():
            with sqlite3.connect(counter_db) as db:
                db.execute("UPDATE counter SET value = value + 1")
            raise SystemExit("crash after external effect")
        self.side_handler = handler
        with self.assertRaises(SystemExit):
            self.manager.execute(item)
        other = SQLiteExecutionLog(self.db, "session-1")
        try:
            recovered = ToolExecutionManager(other, self.registry)
            with self.assertRaises(UnsafeToolRetryError):
                recovered.authorize_retry("call_1", expected_started_count=1)
            with self.assertRaises(UnsafeToolRetryError):
                recovered.execute(item, retry=True)
            with sqlite3.connect(counter_db) as db:
                self.assertEqual(db.execute("SELECT value FROM counter").fetchone()[0], 1)
        finally:
            other.close()

    def test_explicit_read_only_retry_requires_authorization(self):
        item = self.call()
        self.read_handler = lambda path: (_ for _ in ()).throw(SystemExit())
        with self.assertRaises(SystemExit):
            self.manager.execute(item)
        with self.assertRaises(UnsafeToolRetryError):
            self.manager.execute(item, retry=True)
        self.manager.authorize_retry("call_1", expected_started_count=1)
        self.assertEqual(self.inspect().get("call_1").phase, ToolPhase.RETRY_READY)
        self.assertEqual(self.inspect().get("call_1").started_count, 1)
        with self.assertRaises(UnsafeToolRetryError):
            self.manager.execute(item)  # Ordinary first-time admission cannot use RETRY_READY
        self.read_handler = lambda path: ["retried.py"]
        outcome = self.manager.execute(item, retry=True)
        self.assertEqual(outcome.started_count, 2)
        self.commit(outcome)
        self.assertEqual(self.inspect().get("call_1").phase, ToolPhase.RESULT_RECORDED)
        self.assertEqual(self.inspect().get("call_1").started_count, 2)

    def test_retry_authorization_and_admission_are_cas_guarded(self):
        item = self.call()
        self.manager.execute(item)  # hold STARTED
        other_log = SQLiteExecutionLog(self.db, "session-1")
        try:
            second = ToolExecutionManager(other_log, self.registry)
            self.manager.authorize_retry("call_1", expected_started_count=1)
            with self.assertRaises(UnsafeToolRetryError):
                second.authorize_retry("call_1", expected_started_count=1)
            outcome = second.execute(item, retry=True)
            with self.assertRaises(UnsafeToolRetryError):
                self.manager.execute(item, retry=True)
            self.commit(outcome)
        finally:
            other_log.close()

    def test_stale_result_version_is_rejected(self):
        item = self.call()
        first = self.manager.execute(item)
        self.manager.authorize_retry("call_1", expected_started_count=1)
        second = self.manager.execute(item, retry=True)
        before = self.log.get_events()
        with self.assertRaisesRegex(ValueError, "Stale or invalid"):
            self.commit(first)
        self.assertEqual(self.log.get_events(), before)
        self.assertIsNone(self.log.get_tool_interaction("call_1"))
        self.commit(second)
        self.assertEqual(self.inspect().get("call_1").started_count, 2)

    def test_result_transaction_rolls_back_on_ledger_failure(self):
        item = self.call()
        outcome = self.manager.execute(item)
        self.log.conn.execute("""
            CREATE TRIGGER deny_ledger BEFORE INSERT ON tool_interactions
            BEGIN SELECT RAISE(ABORT, 'injected'); END
        """)
        before = self.log.get_events()
        with self.assertRaises(sqlite3.IntegrityError):
            self.commit(outcome)
        self.assertEqual(self.log.get_events(), before)
        self.assertEqual(self.inspect().get("call_1").phase, ToolPhase.STARTED)
        self.assertIsNone(self.log.get_tool_interaction("call_1"))

    def test_unknown_side_effect_timeout_keeps_started(self):
        item = self.call("increment_counter")
        self.side_handler = lambda: (_ for _ in ()).throw(TimeoutError())
        with self.assertRaises(UnknownToolOutcomeError):
            self.manager.execute(item)
        self.assertEqual(self.inspect().get("call_1").phase, ToolPhase.STARTED)
        self.assertIsNone(self.log.get_tool_interaction("call_1"))

    def test_result_before_step_completed_is_not_reexecuted(self):
        item = self.call()
        self.run_tool(item)
        other = SQLiteExecutionLog(self.db, "session-1")
        try:
            restored = AgentSession.restore(
                ContextManager(ContextBudget(16384, 2048), 0.8), other
            )
            self.assertEqual(restored.get_step_ends(), [])
            self.assertEqual(restored.get_history()[-1]["type"], "function_call_output")
            with self.assertRaises(UnsafeToolRetryError):
                ToolExecutionManager(other, self.registry).execute(item)
        finally:
            other.close()

    def test_compaction_does_not_erase_execution_identity(self):
        item = self.call(call_id="old")
        self.run_tool(item)
        self.session.mark_step_complete()
        self.session.replace_history([{"role": "user", "content": "Summary"}], [])
        before = self.log.get_events()
        with self.assertRaises(sqlite3.IntegrityError):
            self.session.append([item])
        self.assertEqual(self.log.get_events(), before)
        self.assertEqual(self.inspect().get("old").phase, ToolPhase.RESULT_RECORDED)

    def test_recovery_inspection_does_not_replay_event_tail(self):
        item = self.call()
        with patch.object(self.log, "get_snapshots_for_recovery",
                          side_effect=AssertionError("No replay should happen")):
            self.assertEqual(self.inspect().get("call_1").phase, ToolPhase.RECORDED)
            self.run_tool(item)
            self.assertEqual(self.inspect().get("call_1").phase, ToolPhase.RESULT_RECORDED)

    def test_handler_returned_normally_is_not_business_success(self):
        item = self.call("increment_counter")
        self.side_handler = lambda: {"payment_status": "declined"}
        outcome = self.run_tool(item)
        self.assertTrue(outcome.handler_returned_normally)
        self.assertEqual(outcome.result["payment_status"], "declined")


if __name__ == "__main__":
    unittest.main()
