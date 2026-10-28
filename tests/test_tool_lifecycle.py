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
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = str(Path(temporary.name) / "events.db")
        self.log = SQLiteExecutionLog(self.path, "session-1")
        self.addCleanup(self.log.close)
        self.session = AgentSession(ContextManager(ContextBudget(16384, 2048), 0.8), self.log)
        self.read_handler = lambda path: ["a.py"]
        self.side_handler = lambda: None
        self.registry = ToolRegistry([
            ToolSpec("list_files", lambda path: self.read_handler(path),
                     ToolSafety.READ_ONLY, "List files", {"type": "object"}),
            ToolSpec("increment_counter", lambda: self.side_handler(),
                     ToolSafety.SIDE_EFFECTING, "Increment counter", {"type": "object"}),
        ])
        self.manager = ToolExecutionManager(self.log, self.registry)
        self.session.append([{"role": "user", "content": "Run a tool"}])

    def call(self, name="list_files", call_id="call_1"):
        item = {"type": "function_call", "call_id": call_id, "name": name,
                "arguments": json.dumps({"path": "."} if name == "list_files" else {})}
        self.session.append([item])
        self.manager.register_recorded_calls([item])
        return item

    def commit(self, outcome):
        self.session.record_tool_interaction(outcome.interaction)
        self.manager.mark_result_recorded(outcome.interaction.call_id)

    def run_tool(self, item, retry=False):
        outcome = self.manager.execute(item, retry=retry)
        self.commit(outcome)
        return outcome

    def inspect(self):
        return RecoveryInspector(self.log).inspect()

    def test_registry_is_single_source_of_schema_and_safety(self):
        self.assertEqual([schema["name"] for schema in self.registry.llm_schemas()],
                         ["list_files", "increment_counter"])
        self.assertEqual(self.registry.get("list_files").safety, ToolSafety.READ_ONLY)
        with self.assertRaises(ValueError):
            ToolRegistry([self.registry.get("list_files"), self.registry.get("list_files")])

    def test_unregistered_call_cannot_execute(self):
        item = {"type": "function_call", "call_id": "absent", "name": "list_files",
                "arguments": '{"path":"."}'}
        before = self.log.get_events()
        with self.assertRaisesRegex(ValueError, "not registered"):
            self.manager.execute(item)
        self.assertEqual(self.log.get_events(), before)

    def test_recorded_not_executed_and_restore_is_read_only(self):
        item = self.call()
        self.assertEqual(self.inspect().get(item["call_id"]).phase, ToolPhase.RECORDED)
        before = self.log.get_events()
        recovered = AgentSession.restore(ContextManager(ContextBudget(16384, 2048), 0.8), self.log)
        self.assertEqual(recovered.get_history(), self.session.get_history())
        self.assertEqual(self.log.get_events(), before)

    def test_started_commits_before_handler_and_result_is_committed_by_session(self):
        item = self.call()
        def handler(path):
            self.assertEqual(self.inspect().get(item["call_id"]).phase, ToolPhase.STARTED)
            return ["main.py"]
        self.read_handler = handler
        outcome = self.manager.execute(item)
        self.assertTrue(outcome.handler_returned_normally)
        self.assertEqual(self.inspect().get(item["call_id"]).phase, ToolPhase.STARTED)
        self.assertIsNone(self.log.get_tool_interaction(item["call_id"]))
        self.commit(outcome)
        self.assertEqual(self.inspect().get(item["call_id"]).phase, ToolPhase.RESULT_RECORDED)

    def test_failed_started_commit_never_invokes_handler(self):
        item = self.call()
        invoked = []
        self.read_handler = lambda path: invoked.append(path)
        with patch.object(self.log, "append_tool_execution_started", side_effect=sqlite3.OperationalError):
            with self.assertRaises(sqlite3.OperationalError):
                self.manager.execute(item)
        self.assertEqual(invoked, [])
        self.assertEqual(self.inspect().get(item["call_id"]).phase, ToolPhase.RECORDED)

    def test_crash_after_started_commit_before_handler(self):
        item = self.call()
        original = self.log.append_tool_execution_started
        invoked = []
        def commit_then_crash(call_id):
            original(call_id)
            raise SystemExit("crash before handler")
        self.read_handler = lambda path: invoked.append(path)
        with patch.object(self.log, "append_tool_execution_started", commit_then_crash):
            with self.assertRaises(SystemExit):
                self.manager.execute(item)
        self.assertEqual(invoked, [])
        self.assertEqual(self.inspect().get(item["call_id"]).phase, ToolPhase.STARTED)

    def test_side_effect_crash_never_auto_retries_after_restart(self):
        item = self.call("increment_counter")
        counter = Path(self.path).with_suffix(".counter.db")
        with sqlite3.connect(counter) as conn:
            conn.execute("CREATE TABLE counters(n INTEGER)")
            conn.execute("INSERT INTO counters(n) VALUES (0)")
        def side_effect():
            with sqlite3.connect(counter) as conn:
                conn.execute("UPDATE counters SET n = n + 1")
            raise SystemExit("crash after externally committed side effect")
        self.side_handler = side_effect
        with self.assertRaises(SystemExit):
            self.manager.execute(item)
        self.log.close()
        self.log = SQLiteExecutionLog(self.path, "session-1")
        self.manager = ToolExecutionManager(self.log, self.registry)
        self.session = AgentSession.restore(ContextManager(ContextBudget(16384, 2048), 0.8), self.log)
        self.assertEqual(self.inspect().get(item["call_id"]).phase, ToolPhase.STARTED)
        with self.assertRaises(UnsafeToolRetryError):
            self.manager.execute(item, retry=True)
        with sqlite3.connect(counter) as conn:
            self.assertEqual(conn.execute("SELECT n FROM counters").fetchone()[0], 1)

    def test_read_only_explicit_retry_records_two_starts(self):
        item = self.call()
        self.read_handler = lambda path: (_ for _ in ()).throw(SystemExit())
        with self.assertRaises(SystemExit):
            self.manager.execute(item)
        with self.assertRaises(UnsafeToolRetryError):
            self.manager.execute(item)
        self.read_handler = lambda path: ["a.py"]
        outcome = self.manager.execute(item, retry=True)
        self.commit(outcome)
        status = self.inspect().get(item["call_id"])
        self.assertEqual(status.started_count, 2)
        self.assertEqual(status.phase, ToolPhase.RESULT_RECORDED)

    def test_side_effect_timeout_keeps_unknown_outcome(self):
        item = self.call("increment_counter")
        self.side_handler = lambda: (_ for _ in ()).throw(TimeoutError())
        with self.assertRaises(UnknownToolOutcomeError):
            self.manager.execute(item)
        self.assertEqual(self.inspect().get(item["call_id"]).phase, ToolPhase.STARTED)
        self.assertIsNone(self.log.get_tool_interaction(item["call_id"]))

    def test_result_write_failure_does_not_mark_manager_completed(self):
        item = self.call()
        outcome = self.manager.execute(item)
        with patch.object(self.log, "append_tool_interaction", side_effect=sqlite3.OperationalError):
            with self.assertRaises(sqlite3.OperationalError):
                self.session.record_tool_interaction(outcome.interaction)
        self.assertEqual(self.manager._calls[item["call_id"]].phase, ToolPhase.STARTED)
        self.assertEqual(self.inspect().get(item["call_id"]).phase, ToolPhase.STARTED)
        with self.assertRaises(UnsafeToolRetryError):
            self.manager.execute(item)

    def test_result_before_step_completed_is_not_reexecuted(self):
        item = self.call()
        self.run_tool(item)
        self.log.close()
        reopened = SQLiteExecutionLog(self.path, "session-1")
        try:
            recovered = AgentSession.restore(ContextManager(ContextBudget(16384, 2048), 0.8), reopened)
            self.assertEqual(recovered.get_step_ends(), [])
            self.assertEqual(recovered.get_history()[-1]["type"], "function_call_output")
            manager = ToolExecutionManager(reopened, self.registry)
            with self.assertRaises(UnsafeToolRetryError):
                manager.execute(item)
        finally:
            reopened.close()

    def test_duplicate_id_compacted_out_is_rejected(self):
        item = self.call(call_id="old")
        self.run_tool(item)
        self.session.mark_step_complete()
        self.session.replace_history([{"role": "user", "content": "Summary"}], [])
        # A fresh manager only sees the latest snapshot; the old call is gone
        # from its in-memory state. The durable ledger must still reject it.
        self.manager = ToolExecutionManager(self.log, self.registry)
        self.session.append([item])  # injected reused call ID
        with self.assertRaises(UnsafeToolRetryError):
            self.manager.register_recorded_calls([item])
        invoked = []
        self.read_handler = lambda path: invoked.append(path)
        with self.assertRaises(UnsafeToolRetryError):
            self.manager.execute(item)
        self.assertEqual(invoked, [])

    def test_snapshot_tail_inspection_and_restore(self):
        old = self.call(call_id="old")
        self.run_tool(old)
        self.session.mark_step_complete()
        self.session.replace_history(self.session.get_history(), self.session.get_step_ends())
        new = self.call(call_id="new")
        self.log.append_tool_execution_started("new")
        status = self.inspect()
        self.assertEqual(status.get("new").phase, ToolPhase.STARTED)
        self.assertEqual(status.get("old").phase, ToolPhase.RESULT_RECORDED)
        restored = AgentSession.restore(ContextManager(ContextBudget(16384, 2048), 0.8), self.log)
        self.assertEqual(restored.get_history(), self.session.get_history())

    def test_registering_duplicate_in_current_response_rejects_batch(self):
        item = {"type": "function_call", "call_id": "dup",
                "name": "list_files", "arguments": '{"path":"."}'}
        self.session.append([item, item])
        with self.assertRaises(UnsafeToolRetryError):
            self.manager.register_recorded_calls([item, item])
        self.assertNotIn("dup", self.manager._calls)

    def test_normal_execution_does_not_replay_snapshot_tail(self):
        item = self.call()
        with patch.object(self.log, "get_snapshots_for_recovery",
                          side_effect=AssertionError("Unexpected full replay in hot path")):
            self.run_tool(item)
        self.assertEqual(self.inspect().get(item["call_id"]).phase, ToolPhase.RESULT_RECORDED)

    def test_handler_returned_normally_is_not_business_success(self):
        item = self.call("increment_counter")
        self.side_handler = lambda: {"payment_status": "declined"}
        outcome = self.run_tool(item)
        self.assertTrue(outcome.handler_returned_normally)
        self.assertEqual(outcome.result["payment_status"], "declined")


if __name__ == "__main__":
    unittest.main()
