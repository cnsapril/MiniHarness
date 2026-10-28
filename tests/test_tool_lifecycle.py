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
                            ToolSafety, UnsafeToolRetryError, UnknownToolOutcomeError)


class ToolLifecycleTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = str(Path(temporary.name) / "events.db")
        self.log = SQLiteExecutionLog(self.path, "session-1")
        self.addCleanup(self.log.close)
        self.session = AgentSession(ContextManager(ContextBudget(16384, 2048), 0.8), self.log)
        self.manager = ToolExecutionManager(self.log, {"list_files": ToolSafety.READ_ONLY,
                                                        "increment_counter": ToolSafety.SIDE_EFFECTING})
        self.session.append([{"role": "user", "content": "Run a tool"}])

    def call(self, name="list_files", call_id="call_1"):
        item = {"type": "function_call", "call_id": call_id, "name": name,
                "arguments": json.dumps({"path": "."} if name == "list_files" else {})}
        self.session.append([item])
        return item

    def inspect(self):
        return RecoveryInspector(self.log).inspect()

    def test_recorded_not_executed_and_not_rewritten(self):
        item = self.call()
        self.assertEqual(self.inspect().get(item["call_id"]).phase, ToolPhase.RECORDED)
        before = self.log.get_events()
        recovered = AgentSession.restore(ContextManager(ContextBudget(16384, 2048), 0.8), self.log)
        self.assertEqual(recovered.get_history(), self.session.get_history())
        self.assertEqual(self.log.get_events(), before)

    def test_started_commits_before_handler(self):
        item = self.call()
        def handler(path):
            self.assertEqual(self.inspect().get(item["call_id"]).phase, ToolPhase.STARTED)
            return ["main.py"]
        result = self.manager.execute(self.session, item, handler)
        self.assertTrue(result.success)
        self.assertEqual(self.inspect().get(item["call_id"]).phase, ToolPhase.RESULT_RECORDED)
        self.assertIsNotNone(self.log.get_tool_interaction(item["call_id"]))

    def test_failure_after_started_before_handler_is_unresolved(self):
        item = self.call()
        original = self.log.append_tool_execution_started
        invoked = []
        def commit_then_crash(call_id):
            original(call_id)
            raise SystemExit("simulated crash before handler")
        with patch.object(self.log, "append_tool_execution_started", commit_then_crash):
            with self.assertRaises(SystemExit):
                self.manager.execute(self.session, item, lambda path: invoked.append(path))
        self.assertEqual(invoked, [])
        self.assertEqual(self.inspect().get(item["call_id"]).phase, ToolPhase.STARTED)

    def test_side_effect_crash_never_auto_retried(self):
        item = self.call("increment_counter")
        counter = Path(self.path).with_suffix(".counter.db")
        with sqlite3.connect(counter) as conn:
            conn.execute("CREATE TABLE counters(n INTEGER)")
            conn.execute("INSERT INTO counters(n) VALUES (0)")
        def side_effect():
            with sqlite3.connect(counter) as conn:
                conn.execute("UPDATE counters SET n = n + 1")
            raise SystemExit("crash after committed side effect")
        with self.assertRaises(SystemExit):
            self.manager.execute(self.session, item, side_effect)
        self.log.close()
        self.log = SQLiteExecutionLog(self.path, "session-1")
        self.manager = ToolExecutionManager(self.log, {"increment_counter": ToolSafety.SIDE_EFFECTING})
        self.session = AgentSession.restore(ContextManager(ContextBudget(16384, 2048), 0.8), self.log)
        self.assertEqual(self.inspect().get(item["call_id"]).phase, ToolPhase.STARTED)
        with self.assertRaises(UnsafeToolRetryError):
            self.manager.execute(self.session, item, side_effect, retry=True)
        with sqlite3.connect(counter) as conn:
            self.assertEqual(conn.execute("SELECT n FROM counters").fetchone()[0], 1)

    def test_read_only_explicit_retry_records_two_starts(self):
        item = self.call()
        with self.assertRaises(SystemExit):
            self.manager.execute(self.session, item, lambda path: (_ for _ in ()).throw(SystemExit()))
        with self.assertRaises(UnsafeToolRetryError):
            self.manager.execute(self.session, item, lambda path: ["a.py"])
        result = self.manager.execute(self.session, item, lambda path: ["a.py"], retry=True)
        self.assertTrue(result.success)
        status = self.inspect().get(item["call_id"])
        self.assertEqual(status.started_count, 2)
        self.assertEqual(status.phase, ToolPhase.RESULT_RECORDED)
        self.assertEqual(sum(e["kind"] == "tool_execution_started" for e in self.log.get_events()), 2)

    def test_unknown_outcome_side_effect_pauses_without_result(self):
        item = self.call("increment_counter")
        with self.assertRaises(UnknownToolOutcomeError):
            self.manager.execute(self.session, item, lambda: (_ for _ in ()).throw(TimeoutError()))
        self.assertEqual(self.inspect().get(item["call_id"]).phase, ToolPhase.STARTED)
        self.assertIsNone(self.log.get_tool_interaction(item["call_id"]))

    def test_result_before_step_completed_no_reexecution(self):
        item = self.call()
        self.manager.execute(self.session, item, lambda path: ["a.py"])
        self.log.close()
        reopened = SQLiteExecutionLog(self.path, "session-1")
        try:
            recovered = AgentSession.restore(ContextManager(ContextBudget(16384,2048),0.8), reopened)
            self.assertEqual(recovered.get_step_ends(), [])
            self.assertEqual(recovered.get_history()[-1]["type"], "function_call_output")
            self.assertEqual(RecoveryInspector(reopened).inspect().get(item["call_id"]).phase, ToolPhase.RESULT_RECORDED)
            with self.assertRaises(UnsafeToolRetryError):
                ToolExecutionManager(reopened, {"list_files": ToolSafety.READ_ONLY}).execute(recovered,item,lambda path: ["other"])
        finally:
            reopened.close()

    def test_old_call_id_cannot_be_reused_after_compaction(self):
        item = self.call(call_id="old")
        self.manager.execute(self.session, item, lambda path: ["a.py"])
        self.session.mark_step_complete()
        self.session.replace_history([{"role": "user", "content": "Summary"}], [])
        self.session.append([item])  # simulate a model reusing the old call_id
        invoked = []
        with self.assertRaises(UnsafeToolRetryError):
            self.manager.execute(self.session, item, lambda path: invoked.append(path))
        self.assertEqual(invoked, [])

    def test_snapshot_tail_inspection(self):
        old = self.call(call_id="old")
        self.manager.execute(self.session, old, lambda path: ["old.py"])
        self.session.mark_step_complete()
        self.session.replace_history(self.session.get_history(), self.session.get_step_ends())
        new = self.call(call_id="new")
        self.log.append_tool_execution_started("new")
        status = self.inspect()
        self.assertEqual(status.get("new").phase, ToolPhase.STARTED)
        self.assertEqual(status.get("old").phase, ToolPhase.RESULT_RECORDED)
        recovered = AgentSession.restore(ContextManager(ContextBudget(16384,2048),0.8),self.log)
        self.assertEqual(recovered.get_history(), self.session.get_history())


if __name__ == "__main__":
    unittest.main()
