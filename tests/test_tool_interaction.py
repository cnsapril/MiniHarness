import sqlite3
import tempfile
import unittest
from pathlib import Path

from context_manager import ContextBudget, ContextManager
from session import AgentSession
from sqlite_execution_log import SQLiteExecutionLog
from tool_model import ToolInteraction


class ToolInteractionLedgerTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = str(Path(self.temp_dir.name) / "agent.db")
        self.log = SQLiteExecutionLog(self.db_path, "session-1")
        self.addCleanup(self.log.close)
        manager = ContextManager(ContextBudget(16384, 2048), 0.8)
        self.session = AgentSession(manager, self.log)

    def interaction(self, call_id="call_1"):
        return ToolInteraction(
            call_id=call_id,
            input={"type": "function_call", "call_id": call_id, "name": "list_files",
                   "arguments": '{"path":"src"}'},
            output={"type": "function_call_output", "call_id": call_id,
                    "output": '["main.py"]'},
        )

    def test_complete_pair_is_recorded_and_queryable(self):
        item = self.interaction()
        self.session.append([{"role": "user", "content": "List src"}])
        self.session.append([item.input])
        version = self.log.acquire_tool_execution("call_1", 0)
        self.session.record_tool_interaction(item, version)
        self.session.mark_step_complete()
        self.assertEqual(self.log.get_tool_interaction("call_1"), item)
        self.assertEqual(self.session.get_history()[-1], item.output)
        events = self.log.get_events()
        self.assertEqual([e["kind"] for e in events],
                         ["history_items", "history_items", "history_items", "step_completed"])
        self.assertEqual(events[2]["items"], [item.output])

        # Compaction only replaces working context; the ledger is retained.
        self.session.replace_history(
            [{"role": "user", "content": "List src"},
             {"role": "assistant", "content": "Summary"}], []
        )
        self.assertEqual(self.log.get_tool_interaction("call_1"), item)

        other = SQLiteExecutionLog(self.db_path, "session-1")
        try:
            self.assertEqual(other.get_tool_interaction("call_1"), item)
        finally:
            other.close()

    def test_duplicate_rolls_back_event_and_ledger(self):
        item = self.interaction()
        self.session.append([item.input])
        version = self.log.acquire_tool_execution("call_1", 0)
        self.session.record_tool_interaction(item, version)
        before_history = self.session.get_history()
        before_events = self.log.get_events()
        with self.assertRaisesRegex(ValueError, "Stale or invalid"):
            self.session.record_tool_interaction(item, version)
        self.assertEqual(self.log.get_events(), before_events)
        self.assertEqual(self.session.get_history(), before_history)

    def test_invalid_pair_is_rejected_before_writing(self):
        with self.assertRaisesRegex(ValueError, "mismatch"):
            ToolInteraction(
                call_id="call_1",
                input=self.interaction().input,
                output={"type": "function_call_output", "call_id": "wrong", "output": "ok"},
            )
        self.assertEqual(self.log.get_events(), [])

    def test_original_mutations_cannot_change_saved_pair(self):
        input_item = self.interaction().input
        output_item = self.interaction().output
        item = ToolInteraction(call_id="call_1", input=input_item, output=output_item)
        input_item["name"] = "modified"
        output_item["output"] = "changed"
        self.session.append([item.input])
        version = self.log.acquire_tool_execution("call_1", 0)
        self.session.record_tool_interaction(item, version)
        self.assertEqual(self.log.get_tool_interaction("call_1").input["name"], "list_files")
        self.assertEqual(self.log.get_tool_interaction("call_1").output["output"], '["main.py"]')


if __name__ == "__main__":
    unittest.main()
