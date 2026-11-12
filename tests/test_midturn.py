"""Mid-turn controls: durable admission, stale tool cancellation and stream safety."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from context_manager import ContextBudget, ContextManager
from main import (call_llm_with_retry_and_extend_history, execute_function_calls)
from permission import PermissionRequirement
from session import AgentSession
from sqlite_execution_log import SQLiteExecutionLog
from tool_execution import ApprovalDecision, RecoveryInspector, ToolExecutionManager
from tool_registry import ToolRegistry, ToolSafety, ToolSpec
from trace_model import AgentTrace
from turn_control import ControlAction, TurnController


class MidturnTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = str(Path(temp.name) / "agent.db")
        self.log = SQLiteExecutionLog(self.path, "session-a")
        self.addCleanup(self.log.close)
        self.log.start_turn()
        self.session = AgentSession(
            ContextManager(ContextBudget(16384, 2048), 0.8), self.log
        )
        self.session.append([{"role": "user", "content": "Refactor repository"}])
        self.controller = TurnController(self.session)

    def submit(self, kind="STEER", content="Only inspect the modified files"):
        with_external = SQLiteExecutionLog(self.path, "session-a")
        try:
            return with_external.post_control(kind, content)
        finally:
            with_external.close()

    @staticmethod
    def function_call(call_id, name="write_file"):
        data = {"type": "function_call", "name": name, "call_id": call_id,
                "arguments": json.dumps({"content": call_id})}
        return SimpleNamespace(**data, model_dump=lambda mode="json": data)

    def test_steering_persists_and_replays_as_user_message(self):
        control_id = self.submit()
        self.assertEqual(self.controller.consume_pending(), ControlAction.STEERED)
        self.assertEqual(self.session.get_history()[-1]["role"], "user")
        self.assertEqual(self.session.get_history()[-1]["content"], "Only inspect the modified files")
        self.assertEqual(self.controller.consume_pending(), ControlAction.NONE)
        restored = AgentSession.restore(
            ContextManager(ContextBudget(16384, 2048), 0.8), self.log
        )
        self.assertEqual(restored.get_history(), self.session.get_history())
        row = self.log.conn.execute("SELECT state FROM turn_controls WHERE id = ?",
                                    (control_id,)).fetchone()
        self.assertEqual(row[0], "APPLIED")

    def test_interrupt_prioritized_over_steering_and_new_control_rejected(self):
        steer = self.submit()
        interrupt = self.submit("INTERRUPT", "")
        self.assertEqual(self.controller.consume_pending(), ControlAction.INTERRUPTED)
        self.assertEqual(self.log.conn.execute(
            "SELECT state FROM agent_turns WHERE session_id = 'session-a'"
        ).fetchone()[0], "INTERRUPTED")
        self.assertEqual(self.log.conn.execute(
            "SELECT state FROM turn_controls WHERE id = ?", (steer,)
        ).fetchone()[0], "UNHANDLED")
        self.assertEqual(self.log.conn.execute(
            "SELECT state FROM turn_controls WHERE id = ?", (interrupt,)
        ).fetchone()[0], "APPLIED")
        with self.assertRaisesRegex(ValueError, "No active turn"):
            self.submit()
        restored = AgentSession.restore(
            ContextManager(ContextBudget(16384, 2048), 0.8), self.log
        )
        self.assertEqual(restored.get_history(), self.session.get_history())

    def test_unstarted_stale_tools_are_cancelled_without_handlers(self):
        calls = [self.function_call("A"), self.function_call("B")]
        self.session.append(calls)
        invoked = []
        reg = ToolRegistry([ToolSpec(
            "write_file", lambda content: invoked.append(content),
            ToolSafety.SIDE_EFFECTING, "test", {},
            permission=PermissionRequirement.ALLOW,
        )])
        self.submit("STEER", "Stop modifying files")
        executed = execute_function_calls(
            self.session, ToolExecutionManager(self.log, reg),
            SimpleNamespace(output=calls), 0, 1, AgentTrace(),
            controller=self.controller,
        )
        self.assertEqual(executed, 2)
        self.assertEqual(invoked, [])
        for call_id in ("A", "B"):
            self.assertEqual(self.log.get_tool_execution(call_id)["state"], "DENIED")
            output = self.log.get_tool_interaction(call_id).output
            self.assertEqual(json.loads(output["output"])["error"], "CANCELLED_BY_CONTROL")
        self.assertEqual(RecoveryInspector(self.log).inspect().unresolved, ())
        self.session.mark_step_complete()
        self.assertEqual(self.controller.consume_pending(), ControlAction.STEERED)

    def test_admission_denied_after_control_arrives_during_approval(self):
        call = self.function_call("A")
        self.session.append([call])
        invoked = []
        reg = ToolRegistry([ToolSpec(
            "write_file", lambda content: invoked.append(content),
            ToolSafety.SIDE_EFFECTING, "test", {},
            permission=PermissionRequirement.ASK,
        )])
        def approve(pending, recorded):
            self.submit("STEER", "Do not modify any more files")
            return ApprovalDecision.APPROVE
        with patch("main.cli_approval_provider", side_effect=approve):
            executed = execute_function_calls(
                self.session, ToolExecutionManager(self.log, reg),
                SimpleNamespace(output=[call]), 0, 1, AgentTrace(),
                controller=self.controller,
            )
        self.assertEqual(executed, 1)
        self.assertEqual(invoked, [])
        self.assertEqual(self.log.get_tool_execution("A")["state"], "DENIED")
        self.assertEqual(RecoveryInspector(self.log).inspect().unresolved, ())

    def test_stream_discards_partial_response_when_steered(self):
        final_response = SimpleNamespace(
            output=[], id="response-1", usage=None
        )
        def events():
            yield SimpleNamespace(type="response.output_text.delta", delta="partial")
            self.submit()
            yield SimpleNamespace(type="response.output_text.delta", delta="stale")
            yield SimpleNamespace(type="response.completed", response=final_response)
        stream = SimpleNamespace(__iter__=events)
        # Provide an iterable object with a close method.
        class Stream:
            def __iter__(self):
                return events()
            def close(self):
                pass
        fake_client = SimpleNamespace(
            responses=SimpleNamespace(create=lambda **kwargs: Stream())
        )
        received = []
        with patch("main.client", fake_client):
            result = call_llm_with_retry_and_extend_history(
                self.session, 0, AgentTrace(), controller=self.controller,
                stream=True, event_sink=received.append,
            )
        self.assertIsNone(result)
        self.assertEqual(self.session.get_history(), [{"role": "user", "content": "Refactor repository"}])
        self.assertEqual([event.data["text"] for event in received if event.type == "text_delta"],
                         ["partial"])
        self.assertEqual(self.controller.consume_pending(), ControlAction.STEERED)

    def test_steer_during_first_handler_preserves_first_effect_and_cancels_second(self):
        calls = [self.function_call("A"), self.function_call("B")]
        self.session.append(calls)
        invoked = []
        def handler(content):
            invoked.append(content)
            if content == "A":
                self.submit("STEER", "Do not run the remaining changes")
            return "done"
        reg = ToolRegistry([ToolSpec(
            "write_file", handler, ToolSafety.SIDE_EFFECTING, "test", {},
            permission=PermissionRequirement.ALLOW,
        )])
        executed = execute_function_calls(
            self.session, ToolExecutionManager(self.log, reg),
            SimpleNamespace(output=calls), 0, 1, AgentTrace(),
            controller=self.controller,
        )
        self.assertEqual(executed, 2)
        self.assertEqual(invoked, ["A"])
        self.assertEqual(self.log.get_tool_execution("A")["state"], "RESULT_RECORDED")
        self.assertEqual(self.log.get_tool_execution("B")["state"], "DENIED")
        self.assertEqual(RecoveryInspector(self.log).inspect().unresolved, ())

    def test_stream_commits_only_complete_model_responses(self):
        response = SimpleNamespace(
            output=[{"type": "message", "role": "assistant", "content": "All done"}],
            output_text="All done", id="response-final", usage=None,
        )
        class Stream:
            def __iter__(self):
                yield SimpleNamespace(type="response.output_text.delta", delta="All ")
                yield SimpleNamespace(type="response.output_text.delta", delta="done")
                yield SimpleNamespace(type="response.completed", response=response)
            def close(self):
                pass
        fake_client = SimpleNamespace(
            responses=SimpleNamespace(create=lambda **kwargs: Stream())
        )
        received = []
        with patch("main.client", fake_client):
            result = call_llm_with_retry_and_extend_history(
                self.session, 0, AgentTrace(), controller=self.controller,
                stream=True, event_sink=received.append,
            )
        self.assertIs(result, response)
        self.assertEqual(self.session.get_history()[-1]["content"], "All done")
        self.assertEqual("".join(e.data["text"] for e in received
                                 if e.type == "text_delta"), "All done")

    def test_turn_finishing_is_blocked_by_pending_control(self):
        self.submit()
        self.assertFalse(self.log.finish_turn("COMPLETED", require_no_pending=True))
        self.assertEqual(self.controller.consume_pending(), ControlAction.STEERED)
        self.assertTrue(self.log.finish_turn("COMPLETED", require_no_pending=True))
        with self.assertRaisesRegex(ValueError, "No active turn"):
            self.submit()


if __name__ == "__main__":
    unittest.main()
