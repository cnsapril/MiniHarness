"""Minimal durable tool lifecycle and explicit recovery inspection.

This module does not automatically resume sessions or retry side-effecting tools.
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from context_manager import ContextHelper
from tool_model import ToolInteraction


class ToolPhase(str, Enum):
    RECORDED = "RECORDED"
    STARTED = "STARTED"
    RESULT_RECORDED = "RESULT_RECORDED"


class ToolSafety(str, Enum):
    READ_ONLY = "READ_ONLY"
    SIDE_EFFECTING = "SIDE_EFFECTING"


class UnsafeToolRetryError(RuntimeError):
    """The recorded execution phase does not allow this invocation."""


class UnknownToolOutcomeError(RuntimeError):
    """The external operation may have taken effect; do not blindly retry."""


@dataclass(frozen=True)
class ToolCallInspection:
    call_id: str
    tool_name: str
    input: dict
    phase: ToolPhase
    started_count: int


@dataclass(frozen=True)
class RecoveryReport:
    session_id: str
    calls: tuple[ToolCallInspection, ...]

    @property
    def unresolved(self) -> tuple[ToolCallInspection, ...]:
        return tuple(call for call in self.calls if call.phase != ToolPhase.RESULT_RECORDED)

    def get(self, call_id: str) -> ToolCallInspection | None:
        return next((call for call in self.calls if call.call_id == call_id), None)


class RecoveryInspector:
    """Inspect the latest context snapshot and its tail, without executing tools.

    This scope is valid while compaction only occurs at completed-step boundaries,
    and tool calls are synchronous and serial.
    """

    def __init__(self, execution_log):
        self.execution_log = execution_log

    def inspect(self) -> RecoveryReport:
        states = {}

        def consume_items(items):
            for item in items:
                if item.get("type") == "function_call":
                    call_id = item["call_id"]
                    if call_id in states:
                        raise ValueError(f"Duplicate function call in recovery history: {call_id}")
                    states[call_id] = {"input": item, "started": 0, "result": False}
                elif item.get("type") == "function_call_output":
                    call_id = item["call_id"]
                    if call_id not in states:
                        raise ValueError(f"Tool result without function call: {call_id}")
                    if states[call_id]["result"]:
                        raise ValueError(f"Duplicate tool result: {call_id}")
                    states[call_id]["result"] = True

        for event in self.execution_log.get_snapshots_for_recovery():
            kind = event["kind"]
            if kind == "context_replaced":
                states.clear()
                consume_items(event["history"])
            elif kind == "history_items":
                consume_items(event["items"])
            elif kind == "tool_execution_started":
                call_id = event["call_id"]
                if call_id not in states:
                    raise ValueError(f"Tool started without function call: {call_id}")
                if states[call_id]["result"]:
                    raise ValueError(f"Tool started after result: {call_id}")
                states[call_id]["started"] += 1
            elif kind == "step_completed":
                pass
            else:
                raise ValueError(f"Unknown execution event: {kind}")

        calls = tuple(
            ToolCallInspection(
                call_id=call_id,
                tool_name=state["input"]["name"],
                input=state["input"],
                phase=(ToolPhase.RESULT_RECORDED if state["result"] else
                       ToolPhase.STARTED if state["started"] else ToolPhase.RECORDED),
                started_count=state["started"],
            )
            for call_id, state in states.items()
        )
        return RecoveryReport(self.execution_log.get_session_id(), calls)


@dataclass(frozen=True)
class ToolExecutionResult:
    result: object
    success: bool
    error: str | None


class ToolExecutionManager:
    def __init__(self, execution_log, tool_safety: dict[str, ToolSafety]):
        self.execution_log = execution_log
        self.tool_safety = tool_safety

    def execute(self, session, tool_call, handler: Callable | None, *, retry: bool = False) -> ToolExecutionResult:
        """Run an already-durably-recorded tool call. Retry requires explicit opt-in."""
        input_item = ContextHelper.serialize_items([tool_call])[0]
        if input_item.get("type") != "function_call":
            raise ValueError("Expected function_call")
        call_id = input_item["call_id"]
        tool_name = input_item["name"]
        safety = self.tool_safety.get(tool_name, ToolSafety.SIDE_EFFECTING)
        status = RecoveryInspector(self.execution_log).inspect().get(call_id)
        if status is None:
            raise ValueError(f"Tool call not durably recorded: {call_id}")
        if (status.tool_name != tool_name or status.input.get("arguments") != input_item.get("arguments")):
            raise ValueError(f"Tool call changed since recording: {call_id}")
        # A completed call may have been compacted out of the recovery tail.
        # The durable ledger protects its call_id from accidental reuse.
        if self.execution_log.get_tool_interaction(call_id) is not None or status.phase == ToolPhase.RESULT_RECORDED:
            raise UnsafeToolRetryError(f"Tool call already has result: {call_id}")
        if status.phase == ToolPhase.STARTED and not (retry and safety == ToolSafety.READ_ONLY):
            raise UnsafeToolRetryError(f"Unresolved tool call cannot be retried automatically: {call_id}")
        if retry and safety != ToolSafety.READ_ONLY:
            raise UnsafeToolRetryError(f"Retry is not allowed for side-effecting tool: {tool_name}")

        # Durable execution permission MUST commit before any handler invocation.
        self.execution_log.append_tool_execution_started(call_id)

        try:
            arguments = json.loads(input_item["arguments"])
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments not an object")
            if handler is None:
                raise KeyError(f"Unknown tool: {tool_name}")
        except Exception as exc:
            return self._record_result(session, input_item, {"error": repr(exc)}, False, repr(exc))

        try:
            result = handler(**arguments)
        except UnknownToolOutcomeError:
            raise
        except Exception as exc:
            if safety == ToolSafety.SIDE_EFFECTING:
                raise UnknownToolOutcomeError(
                    f"Outcome of side-effecting tool {tool_name} ({call_id}) is unknown"
                ) from exc
            return self._record_result(session, input_item, {"error": repr(exc)}, False, repr(exc))
        return self._record_result(session, input_item, result, True, None)

    @staticmethod
    def _record_result(session, input_item, result, success, error) -> ToolExecutionResult:
        output = {
            "type": "function_call_output",
            "call_id": input_item["call_id"],
            "output": json.dumps(result, ensure_ascii=False),
        }
        session.record_tool_interaction(
            ToolInteraction(call_id=input_item["call_id"], input=input_item, output=output)
        )
        return ToolExecutionResult(result=result, success=success, error=error)
