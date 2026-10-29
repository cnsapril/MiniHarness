"""Permission/approval integration tests on the persistent execution state machine."""
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from context_manager import ContextBudget, ContextManager
from permission import PermissionPolicy, PermissionRequirement
from session import AgentSession
from sqlite_execution_log import SQLiteExecutionLog
from tool_execution import (ApprovalRequiredError, RecoveryInspector, ToolDenialOutcome,
                            ToolExecutionManager, ToolPhase, UnsafeToolRetryError)
from tool_registry import ToolRegistry, ToolSafety, ToolSpec, write_file


class PermissionApprovalTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.log = SQLiteExecutionLog(str(self.root / "state.sqlite"), "session-1")
        self.addCleanup(self.log.close)
        self.session = AgentSession(
            ContextManager(ContextBudget(16384, 2048), 0.8), self.log
        )
        self.calls = []
        def handler(**args):
            self.calls.append(args)
            return {"handled": True}
        self.registry = ToolRegistry([
            ToolSpec("read", handler, ToolSafety.READ_ONLY, "Read", {},
                     permission=PermissionRequirement.ALLOW),
            ToolSpec("sensitive_read", handler, ToolSafety.READ_ONLY, "Sensitive", {},
                     permission=PermissionRequirement.ASK),
            ToolSpec("write_file", handler, ToolSafety.SIDE_EFFECTING, "Write", {},
                     permission=PermissionRequirement.ASK),
            ToolSpec("blocked", handler, ToolSafety.SIDE_EFFECTING, "Blocked", {},
                     permission=PermissionRequirement.DENY),
        ])
        self.manager = ToolExecutionManager(
            self.log, self.registry, PermissionPolicy(self.root)
        )

    def call(self, tool="write_file", args=None, call_id="call-1"):
        if args is None:
            args = {"path": "README.md", "content": "new"} if tool == "write_file" else {}
        item = {
            "type": "function_call", "name": tool,
            "call_id": call_id, "arguments": json.dumps(args),
        }
        self.session.append([item])
        return item

    def commit(self, outcome):
        if isinstance(outcome, ToolDenialOutcome):
            self.session.record_tool_denial(
                outcome.interaction, outcome.expected_started_count, retry=outcome.retry,
                rejected_approval=outcome.rejected_approval,
            )
        else:
            self.session.record_tool_interaction(outcome.interaction, outcome.started_count)

    def test_pending_is_durable_and_does_not_run_handler(self):
        item = self.call()
        with self.assertRaises(ApprovalRequiredError) as pending:
            self.manager.execute(item)
        self.assertEqual(pending.exception.expected_started_count, 0)
        self.assertEqual(self.log.get_tool_approval("call-1", 0), "PENDING")
        self.assertEqual(self.log.get_tool_execution("call-1")["state"], "RECORDED")
        self.assertEqual(self.calls, [])
        with self.assertRaises(ApprovalRequiredError):
            self.manager.execute(item)
        other_log = SQLiteExecutionLog(str(self.root / "state.sqlite"), "session-1")
        try:
            self.assertEqual(other_log.get_tool_approval("call-1", 0), "PENDING")
        finally:
            other_log.close()

    def test_approval_and_cas_are_atomic_and_single_use(self):
        item = self.call()
        with self.assertRaises(ApprovalRequiredError):
            self.manager.execute(item)
        self.manager.approve("call-1", expected_started_count=0)
        self.assertEqual(self.log.get_tool_approval("call-1", 0), "APPROVED")
        self.assertEqual(self.log.get_tool_execution("call-1")["state"], "RECORDED")
        outcome = self.manager.execute(item)
        self.assertEqual(outcome.started_count, 1)
        self.assertEqual(self.calls, [{"path": "README.md", "content": "new"}])
        self.assertEqual(self.log.get_tool_approval("call-1", 0), "CONSUMED")
        self.assertEqual(self.log.get_tool_execution("call-1")["state"], "STARTED")
        self.commit(outcome)
        self.assertEqual(self.log.get_tool_execution("call-1")["state"], "RESULT_RECORDED")
        with self.assertRaises(UnsafeToolRetryError):
            self.manager.execute(item)
        with self.assertRaises(UnsafeToolRetryError):
            self.manager.approve("call-1", expected_started_count=0)

    def test_two_managers_cannot_consume_same_approval(self):
        item = self.call()
        with self.assertRaises(ApprovalRequiredError):
            self.manager.execute(item)
        self.manager.approve("call-1", expected_started_count=0)
        other_log = SQLiteExecutionLog(str(self.root / "state.sqlite"), "session-1")
        try:
            second = ToolExecutionManager(other_log, self.registry, PermissionPolicy(self.root))
            outcome = second.execute(item)
            with self.assertRaises(UnsafeToolRetryError):
                self.manager.execute(item)
            self.assertEqual(len(self.calls), 1)
            self.commit(outcome)
        finally:
            other_log.close()

    def test_db_failure_does_not_consume_approval_or_execute(self):
        item = self.call()
        with self.assertRaises(ApprovalRequiredError):
            self.manager.execute(item)
        self.manager.approve("call-1", expected_started_count=0)
        self.log.conn.execute("""
            CREATE TRIGGER abort_start BEFORE UPDATE OF state ON tool_executions
            WHEN NEW.state = 'STARTED'
            BEGIN SELECT RAISE(ABORT, 'injected'); END
        """)
        with self.assertRaises(sqlite3.IntegrityError):
            self.manager.execute(item)
        self.assertEqual(self.log.get_tool_approval("call-1", 0), "APPROVED")
        self.assertEqual(self.log.get_tool_execution("call-1")["state"], "RECORDED")
        self.assertEqual(self.calls, [])

    def test_rejection_is_atomic_with_denied_event_and_ledger(self):
        item = self.call()
        with self.assertRaises(ApprovalRequiredError):
            self.manager.execute(item)
        denied = self.manager.reject(item, expected_started_count=0)
        self.assertEqual(self.log.get_tool_approval("call-1", 0), "PENDING")
        self.commit(denied)
        self.assertEqual(self.log.get_tool_approval("call-1", 0), "REJECTED")
        self.assertEqual(self.log.get_tool_execution("call-1")["state"], "DENIED")
        self.assertEqual(self.log.get_tool_execution("call-1")["started_count"], 0)
        self.assertEqual(self.log.get_tool_interaction("call-1"), denied.interaction)
        self.assertEqual(self.session.get_history()[-1], denied.interaction.output)
        self.assertEqual(RecoveryInspector(self.log).inspect().unresolved, ())
        self.assertEqual(self.calls, [])
        with self.assertRaises(UnsafeToolRetryError):
            self.manager.execute(item)

    def test_rejection_rolls_back_on_ledger_failure(self):
        item = self.call()
        with self.assertRaises(ApprovalRequiredError):
            self.manager.execute(item)
        denied = self.manager.reject(item, expected_started_count=0)
        self.log.conn.execute("""
            CREATE TRIGGER abort_denial BEFORE INSERT ON tool_interactions
            BEGIN SELECT RAISE(ABORT, 'injected'); END
        """)
        before = self.log.get_events()
        with self.assertRaises(sqlite3.IntegrityError):
            self.commit(denied)
        self.assertEqual(self.log.get_events(), before)
        self.assertEqual(self.log.get_tool_approval("call-1", 0), "PENDING")
        self.assertEqual(self.log.get_tool_execution("call-1")["state"], "RECORDED")

    def test_policy_denial_commits_without_approval(self):
        item = self.call("blocked")
        denied = self.manager.execute(item)
        self.assertIsInstance(denied, ToolDenialOutcome)
        self.assertIsNone(self.log.get_tool_approval("call-1", 0))
        self.commit(denied)
        self.assertEqual(self.log.get_tool_execution("call-1")["state"], "DENIED")
        self.assertEqual(self.calls, [])
        self.assertEqual(json.loads(denied.interaction.output["output"])["previous_outcome"],
                         "NOT_STARTED")

    def test_path_escaping_and_sensitive_paths_are_denied(self):
        for i, path in enumerate(("../outside", "/etc/hosts", ".git/config", ".env",
                                  "sub/../../secret")):
            item = self.call(args={"path": path, "content": "bad"}, call_id=f"call-{i}")
            denied = self.manager.execute(item)
            self.assertIsInstance(denied, ToolDenialOutcome)
            self.commit(denied)
            self.assertEqual(self.log.get_tool_execution(f"call-{i}")["state"], "DENIED")
        self.assertEqual(self.calls, [])

    def test_unregistered_tool_and_invalid_arguments_fail_closed(self):
        item = self.call("nonexistent")
        denied = self.manager.execute(item)
        self.assertIsInstance(denied, ToolDenialOutcome)
        self.commit(denied)
        malformed = {"type": "function_call", "name": "write_file",
                     "call_id": "bad-json", "arguments": "{bad"}
        self.session.append([malformed])
        denied = self.manager.execute(malformed)
        self.assertIsInstance(denied, ToolDenialOutcome)
        self.commit(denied)
        self.assertEqual(self.calls, [])

    def test_approval_cannot_be_reused_for_retry(self):
        item = self.call("sensitive_read")
        with self.assertRaises(ApprovalRequiredError):
            self.manager.execute(item)
        self.manager.approve("call-1", expected_started_count=0)
        first = self.manager.execute(item)
        self.assertEqual(first.started_count, 1)
        self.manager.authorize_retry("call-1", expected_started_count=1)
        with self.assertRaises(ApprovalRequiredError) as retry:
            self.manager.execute(item, retry=True)
        self.assertEqual(retry.exception.expected_started_count, 1)
        self.assertEqual(self.log.get_tool_approval("call-1", 0), "CONSUMED")
        self.assertEqual(self.log.get_tool_approval("call-1", 1), "PENDING")
        self.assertEqual(len(self.calls), 1)
        self.manager.approve("call-1", expected_started_count=1, retry=True)
        second = self.manager.execute(item, retry=True)
        self.assertEqual(second.started_count, 2)
        with self.assertRaisesRegex(ValueError, "Stale or invalid"):
            self.commit(first)
        self.commit(second)
        self.assertEqual(self.log.get_tool_approval("call-1", 1), "CONSUMED")

    def test_rejected_retry_preserves_unknown_previous_outcome(self):
        item = self.call("sensitive_read")
        with self.assertRaises(ApprovalRequiredError):
            self.manager.execute(item)
        self.manager.approve("call-1", expected_started_count=0)
        self.manager.execute(item)  # STARTED; no durable result
        self.manager.authorize_retry("call-1", expected_started_count=1)
        with self.assertRaises(ApprovalRequiredError):
            self.manager.execute(item, retry=True)
        rejected = self.manager.reject(item, expected_started_count=1, retry=True)
        self.commit(rejected)
        self.assertEqual(self.log.get_tool_execution("call-1")["state"], "DENIED")
        self.assertEqual(self.log.get_tool_execution("call-1")["started_count"], 1)
        self.assertEqual(self.log.get_tool_approval("call-1", 1), "REJECTED")
        response = json.loads(rejected.interaction.output["output"])
        self.assertEqual(response["previous_outcome"], "UNKNOWN")

    def test_denial_version_cas_prevents_stale_rejection(self):
        item = self.call("sensitive_read")
        with self.assertRaises(ApprovalRequiredError):
            self.manager.execute(item)
        denied = self.manager.reject(item, expected_started_count=0)
        self.manager.approve("call-1", expected_started_count=0)
        outcome = self.manager.execute(item)
        before = self.log.get_events()
        with self.assertRaises(ValueError):
            self.commit(denied)
        self.assertEqual(self.log.get_events(), before)
        self.assertEqual(self.log.get_tool_approval("call-1", 0), "CONSUMED")
        self.commit(outcome)

    def test_workspace_handler_does_not_follow_symlinks(self):
        (self.root / "real").mkdir()
        (self.root / "alias").symlink_to(self.root / "real", target_is_directory=True)
        with patch("tool_registry.PROJECT_ROOT", self.root):
            with self.assertRaises(OSError):
                write_file("alias/one.txt", "bad")
            self.assertFalse((self.root / "real" / "one.txt").exists())
            self.assertEqual(write_file("real/one.txt", "ok"), "real/one.txt")
            self.assertEqual((self.root / "real" / "one.txt").read_text(), "ok")

    def test_approved_arguments_cannot_be_swapped(self):
        item = self.call()
        with self.assertRaises(ApprovalRequiredError):
            self.manager.execute(item)
        self.manager.approve("call-1", expected_started_count=0)
        modified = dict(item)
        modified["arguments"] = json.dumps({"path": "another.txt", "content": "evil"})
        with self.assertRaises(UnsafeToolRetryError):
            self.manager.execute(modified)
        self.assertEqual(self.log.get_tool_approval("call-1", 0), "APPROVED")
        self.assertEqual(self.log.get_tool_execution("call-1")["state"], "RECORDED")
        self.assertEqual(self.calls, [])

    def test_cli_approval_and_rejection_use_trusted_input(self):
        from types import SimpleNamespace
        from main import execute_function_calls
        from trace_model import AgentTrace

        # The SDK supplies model_dump and attribute access; emulate that here.
        item = self.call(call_id="approved")
        sdk_item = SimpleNamespace(**item, model_dump=lambda mode="json": item)
        with patch("builtins.input", return_value="y"):
            count = execute_function_calls(
                self.session, self.manager, SimpleNamespace(output=[sdk_item]),
                0, 1, AgentTrace()
            )
        self.assertEqual(count, 1)
        self.assertEqual(self.log.get_tool_execution("approved")["state"], "RESULT_RECORDED")
        self.assertEqual(self.log.get_tool_approval("approved", 0), "CONSUMED")
        self.assertEqual(len(self.calls), 1)

        item2 = self.call(call_id="rejected")
        sdk_item2 = SimpleNamespace(**item2, model_dump=lambda mode="json": item2)
        with patch("builtins.input", return_value="n"):
            count = execute_function_calls(
                self.session, self.manager, SimpleNamespace(output=[sdk_item2]),
                1, 2, AgentTrace()
            )
        self.assertEqual(count, 1)
        self.assertEqual(self.log.get_tool_execution("rejected")["state"], "DENIED")
        self.assertEqual(self.log.get_tool_approval("rejected", 0), "REJECTED")
        self.assertEqual(len(self.calls), 1)



if __name__ == "__main__":
    unittest.main()
