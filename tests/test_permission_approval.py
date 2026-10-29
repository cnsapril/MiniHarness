"""Permission/approval integration tests on the persistent execution state machine."""
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from context_manager import ContextBudget, ContextManager
from permission import DenialSource, PermissionPolicy, PermissionRequirement
from session import AgentSession
from sqlite_execution_log import SQLiteExecutionLog
from tool_execution import (ApprovalDecision, ToolDispatchStatus, RecoveryInspector,
                            ToolExecutionManager, ToolPhase, UnsafeToolRetryError)
from tool_registry import (ToolRegistry, ToolSafety, ToolSpec, write_file,
                           write_file_permission_check, workspace_path_permission_check)


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
                     permission=PermissionRequirement.ASK,
                     permission_check=write_file_permission_check),
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
        if outcome.status == ToolDispatchStatus.DENIED:
            self.session.record_tool_denial(
                outcome.interaction, outcome.expected_started_count, retry=outcome.retry,
                denial_source=outcome.denial_source,
            )
        else:
            self.session.record_tool_interaction(outcome.interaction, outcome.started_count)

    def test_manager_owns_approval_workflow(self):
        item = self.call()
        seen = []

        def authorize(pending, recorded):
            seen.append((pending.call_id, pending.expected_started_count, recorded))
            self.assertEqual(self.log.get_tool_execution("call-1")["state"], "RECORDED")
            self.assertEqual(self.log.get_tool_approval("call-1", 0), "PENDING")
            return ApprovalDecision.APPROVE

        outcome = self.manager.run_with_approval(item, authorize)
        self.assertEqual(outcome.status, ToolDispatchStatus.COMPLETED)
        self.assertEqual(seen[0][0:2], ("call-1", 0))
        self.assertEqual(seen[0][2], item)
        self.assertEqual(self.log.get_tool_approval("call-1", 0), "CONSUMED")
        self.assertEqual(self.log.get_tool_execution("call-1")["state"], "STARTED")
        self.assertEqual(len(self.calls), 1)
        self.commit(outcome)

    def test_manager_owns_rejection_workflow(self):
        item = self.call()
        seen = []

        def reject(pending, recorded):
            seen.append((pending, recorded))
            return ApprovalDecision.REJECT

        outcome = self.manager.run_with_approval(item, reject)
        self.assertEqual(outcome.status, ToolDispatchStatus.DENIED)
        self.assertEqual(len(seen), 1)
        self.assertEqual(len(self.calls), 0)
        self.assertEqual(self.log.get_tool_approval("call-1", 0), "PENDING")
        self.commit(outcome)
        self.assertEqual(self.log.get_tool_approval("call-1", 0), "REJECTED")
        self.assertEqual(self.log.get_tool_execution("call-1")["state"], "DENIED")

    def test_manager_skips_approval_callback_when_policy_decides(self):
        def fail(*args):
            raise AssertionError("Approval callback must not run")
        allowed = self.call(tool="read", call_id="read-1")
        executed = self.manager.run_with_approval(allowed, fail)
        self.assertEqual(executed.status, ToolDispatchStatus.COMPLETED)
        self.commit(executed)
        denied = self.call(tool="blocked", call_id="blocked-1")
        declined = self.manager.run_with_approval(denied, fail)
        self.assertEqual(declined.status, ToolDispatchStatus.DENIED)
        self.commit(declined)
        self.assertEqual(self.log.get_tool_execution("blocked-1")["state"], "DENIED")

    def test_manager_invalid_ui_decision_stays_pending(self):
        item = self.call()
        for invalid in ("approve", "APPROVE"):
            with self.assertRaises(ValueError):
                self.manager.run_with_approval(item, lambda pending, recorded: invalid)
        self.assertEqual(self.log.get_tool_approval("call-1", 0), "PENDING")
        self.assertEqual(self.log.get_tool_execution("call-1")["state"], "RECORDED")
        self.assertEqual(self.calls, [])

    def test_pending_is_durable_and_does_not_run_handler(self):
        item = self.call()
        pending = self.manager.execute(item)
        self.assertEqual(pending.status, ToolDispatchStatus.APPROVAL_PENDING)
        self.assertEqual(pending.expected_started_count, 0)
        self.assertEqual(self.log.get_tool_approval("call-1", 0), "PENDING")
        self.assertEqual(self.log.get_tool_execution("call-1")["state"], "RECORDED")
        self.assertEqual(self.calls, [])
        # Existing PENDING is only read, not inserted again.
        writes_before = self.log.conn.total_changes
        self.assertEqual(self.manager.execute(item).status, ToolDispatchStatus.APPROVAL_PENDING)
        self.assertEqual(self.log.conn.total_changes, writes_before)
        # Explicit duplicate creation is a failure, not an ignored conflict.
        with self.assertRaises(sqlite3.IntegrityError):
            self.log.create_pending_approval("call-1", 0)
        self.assertEqual(self.log.get_tool_approval("call-1", 0), "PENDING")
        other_log = SQLiteExecutionLog(str(self.root / "state.sqlite"), "session-1")
        try:
            self.assertEqual(other_log.get_tool_approval("call-1", 0), "PENDING")
        finally:
            other_log.close()

    def test_approval_crud_does_not_own_execution_state(self):
        item = self.call()
        # Approval persistence identifies a recorded call, not a runnable attempt.
        # Admission CAS is the only authority for execution state/version.
        self.log.create_pending_approval("call-1", 1)
        self.assertEqual(self.log.get_tool_approval("call-1", 1), "PENDING")
        self.log.approve_tool_execution("call-1", 1)
        self.assertEqual(self.log.get_tool_approval("call-1", 1), "APPROVED")
        with self.assertRaises(ValueError):
            self.log.acquire_tool_execution("call-1", 1, require_approval=True)
        self.assertEqual(self.log.get_tool_approval("call-1", 1), "APPROVED")
        self.assertEqual(self.log.get_tool_execution("call-1")["state"], "RECORDED")
        self.assertEqual(self.log.get_tool_execution("call-1")["started_count"], 0)
        self.assertEqual(self.calls, [])
        # The Manager still creates the actual first-attempt approval (count=0).
        self.assertEqual(self.manager.execute(item).status,
                         ToolDispatchStatus.APPROVAL_PENDING)
        self.assertEqual(self.log.get_tool_approval("call-1", 0), "PENDING")

    def test_approval_foreign_key_rejects_unknown_call(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.log.create_pending_approval("does-not-exist", 0)

    def test_approval_manager_rejects_stale_execution_state(self):
        item = self.call()
        self.assertEqual(self.manager.execute(item).status,
                         ToolDispatchStatus.APPROVAL_PENDING)
        self.manager.approve("call-1", expected_started_count=0)
        outcome = self.manager.execute(item)
        with self.assertRaises(UnsafeToolRetryError):
            self.manager.approve("call-1", expected_started_count=0)
        self.commit(outcome)

    def test_approval_and_cas_are_atomic_and_single_use(self):
        item = self.call()
        self.assertEqual(self.manager.execute(item).status, ToolDispatchStatus.APPROVAL_PENDING)
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
        self.assertEqual(self.manager.execute(item).status, ToolDispatchStatus.APPROVAL_PENDING)
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
        self.assertEqual(self.manager.execute(item).status, ToolDispatchStatus.APPROVAL_PENDING)
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

    def test_denial_source_distinguishes_user_from_policy(self):
        user_call = self.call(call_id="user-reject")
        self.assertEqual(self.manager.execute(user_call).status,
                         ToolDispatchStatus.APPROVAL_PENDING)
        denied = self.manager.reject(user_call, expected_started_count=0)
        self.assertEqual(denied.denial_source, DenialSource.USER)
        self.commit(denied)
        self.assertEqual(self.log.get_tool_approval("user-reject", 0), "REJECTED")

        policy_call = self.call(tool="blocked", call_id="policy-deny")
        denied_by_policy = self.manager.execute(policy_call)
        self.assertEqual(denied_by_policy.denial_source, DenialSource.POLICY)
        self.commit(denied_by_policy)
        self.assertIsNone(self.log.get_tool_approval("policy-deny", 0))
        self.assertEqual(self.calls, [])

    def test_rejection_is_atomic_with_denied_event_and_ledger(self):
        item = self.call()
        self.assertEqual(self.manager.execute(item).status, ToolDispatchStatus.APPROVAL_PENDING)
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
        self.assertEqual(self.manager.execute(item).status, ToolDispatchStatus.APPROVAL_PENDING)
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
        self.assertEqual(denied.status, ToolDispatchStatus.DENIED)
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
            self.assertEqual(denied.status, ToolDispatchStatus.DENIED)
            self.commit(denied)
            self.assertEqual(self.log.get_tool_execution(f"call-{i}")["state"], "DENIED")
        self.assertEqual(self.calls, [])

    def test_unregistered_tool_and_invalid_arguments_fail_closed(self):
        item = self.call("nonexistent")
        denied = self.manager.execute(item)
        self.assertEqual(denied.status, ToolDispatchStatus.DENIED)
        self.commit(denied)
        malformed = {"type": "function_call", "name": "write_file",
                     "call_id": "bad-json", "arguments": "{bad"}
        self.session.append([malformed])
        denied = self.manager.execute(malformed)
        self.assertEqual(denied.status, ToolDispatchStatus.DENIED)
        self.commit(denied)
        self.assertEqual(self.calls, [])

    def test_approval_cannot_be_reused_for_retry(self):
        item = self.call("sensitive_read")
        self.assertEqual(self.manager.execute(item).status, ToolDispatchStatus.APPROVAL_PENDING)
        self.manager.approve("call-1", expected_started_count=0)
        first = self.manager.execute(item)
        self.assertEqual(first.started_count, 1)
        self.manager.authorize_retry("call-1", expected_started_count=1)
        pending_retry = self.manager.execute(item, retry=True)
        self.assertEqual(pending_retry.status, ToolDispatchStatus.APPROVAL_PENDING)
        self.assertEqual(pending_retry.expected_started_count, 1)
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
        self.assertEqual(self.manager.execute(item).status, ToolDispatchStatus.APPROVAL_PENDING)
        self.manager.approve("call-1", expected_started_count=0)
        self.manager.execute(item)  # STARTED; no durable result
        self.manager.authorize_retry("call-1", expected_started_count=1)
        self.assertEqual(self.manager.execute(item, retry=True).status, ToolDispatchStatus.APPROVAL_PENDING)
        rejected = self.manager.reject(item, expected_started_count=1, retry=True)
        self.commit(rejected)
        self.assertEqual(self.log.get_tool_execution("call-1")["state"], "DENIED")
        self.assertEqual(self.log.get_tool_execution("call-1")["started_count"], 1)
        self.assertEqual(self.log.get_tool_approval("call-1", 1), "REJECTED")
        response = json.loads(rejected.interaction.output["output"])
        self.assertEqual(response["previous_outcome"], "UNKNOWN")

    def test_denial_version_cas_prevents_stale_rejection(self):
        item = self.call("sensitive_read")
        self.assertEqual(self.manager.execute(item).status, ToolDispatchStatus.APPROVAL_PENDING)
        denied = self.manager.reject(item, expected_started_count=0)
        self.manager.approve("call-1", expected_started_count=0)
        outcome = self.manager.execute(item)
        before = self.log.get_events()
        with self.assertRaises(ValueError):
            self.commit(denied)
        self.assertEqual(self.log.get_events(), before)
        self.assertEqual(self.log.get_tool_approval("call-1", 0), "CONSUMED")
        self.commit(outcome)

    def test_policy_delegates_to_each_tool_spec_not_tool_name(self):
        def reject_specific_args(args, root):
            return "Too many records" if args.get("limit", 0) > 5 else None

        spec = ToolSpec(
            "report_query", lambda **kw: None, ToolSafety.READ_ONLY,
            "Read reports", {}, permission=PermissionRequirement.ALLOW,
            permission_check=reject_specific_args,
        )
        policy = PermissionPolicy(self.root)
        self.assertEqual(policy.evaluate(spec, {"limit": 6}).requirement,
                         PermissionRequirement.DENY)
        self.assertEqual(policy.evaluate(spec, {"limit": 3}).requirement,
                         PermissionRequirement.ALLOW)
        self.assertEqual(policy.evaluate(spec, {"limit": 6}).reason, "Too many records")
        # Names are irrelevant to the generic policy.
        unsafe_name = ToolSpec(
            "write_file", lambda **kw: None, ToolSafety.SIDE_EFFECTING,
            "Different handler", {}, permission=PermissionRequirement.ALLOW,
        )
        self.assertEqual(policy.evaluate(unsafe_name, {"path": "../outside"}).requirement,
                         PermissionRequirement.ALLOW)

    def test_tool_specific_rules_reject_bad_paths_and_oversized_content(self):
        policy = PermissionPolicy(self.root)
        read = ToolSpec("read_any", lambda **kw: None, ToolSafety.READ_ONLY, "", {},
                        permission=PermissionRequirement.ALLOW,
                        permission_check=workspace_path_permission_check)
        write = self.registry.get("write_file")
        self.assertEqual(policy.evaluate(read, {"path": "../outside"}).requirement,
                         PermissionRequirement.DENY)
        self.assertEqual(policy.evaluate(write, {"path": "x.txt", "content": "x" * 131073}).requirement,
                         PermissionRequirement.DENY)
        self.assertEqual(policy.evaluate(write, {"path": "x.txt", "content": "okay"}).requirement,
                         PermissionRequirement.ASK)

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
        self.assertEqual(self.manager.execute(item).status, ToolDispatchStatus.APPROVAL_PENDING)
        self.manager.approve("call-1", expected_started_count=0)
        modified = dict(item)
        modified["arguments"] = json.dumps({"path": "another.txt", "content": "evil"})
        with self.assertRaises(UnsafeToolRetryError):
            self.manager.execute(modified)
        self.assertEqual(self.log.get_tool_approval("call-1", 0), "APPROVED")
        self.assertEqual(self.log.get_tool_execution("call-1")["state"], "RECORDED")
        self.assertEqual(self.calls, [])

    def test_cli_policy_denial_does_not_prompt_for_approval(self):
        from types import SimpleNamespace
        from main import execute_function_calls
        from trace_model import AgentTrace

        item = self.call("blocked")
        sdk_item = SimpleNamespace(**item, model_dump=lambda mode="json": item)
        with patch("builtins.input", side_effect=AssertionError("Unexpected approval prompt")):
            result = execute_function_calls(
                self.session, self.manager, SimpleNamespace(output=[sdk_item]),
                0, 1, AgentTrace()
            )
        self.assertEqual(result, 1)
        self.assertEqual(self.log.get_tool_execution("call-1")["state"], "DENIED")
        self.assertIsNone(self.log.get_tool_approval("call-1", 0))
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
