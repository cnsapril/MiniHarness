"""Minimal durable tool lifecycle and explicit recovery inspection.

This module does not automatically resume sessions or retry side-effecting tools.
"""

import json
from dataclasses import dataclass, replace
from enum import Enum

from context_manager import ContextHelper
from tool_model import ToolInteraction
from tool_registry import ToolRegistry, ToolSafety


class ToolPhase(str, Enum):
    RECORDED = "RECORDED"
    STARTED = "STARTED"
    RESULT_RECORDED = "RESULT_RECORDED"


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
class ToolExecutionOutcome:
    """A prepared result, NOT a persisted interaction or a business success."""
    interaction: ToolInteraction
    result: object
    handler_returned_normally: bool
    error: str | None


class ToolExecutionManager:
    """Per-session execution runtime; AgentSession persists prepared results.

    At construction, a single recovery replay establishes known call states.
    The normal hot path only touches in-memory states and an indexed ledger lookup.
    """

    def __init__(self, execution_log, registry: ToolRegistry):
        self.execution_log = execution_log
        self.registry = registry
        report = RecoveryInspector(execution_log).inspect()
        self._calls = {call.call_id: call for call in report.calls}

    def register_recorded_calls(self, items: list) -> None:
        """Register LLM tool requests AFTER AgentSession.append() commits them.

        Never invoke handlers if writing the LLM response to the log fails.
        """
        new_calls = {}
        for item in ContextHelper.serialize_items(items):
            if item.get("type") != "function_call":
                continue
            call_id = item["call_id"]
            if call_id in self._calls or call_id in new_calls:
                raise UnsafeToolRetryError(f"Duplicate recorded tool call ID: {call_id}")
            # An old, completed call may have been compacted out of the tail.
            if self.execution_log.get_tool_interaction(call_id) is not None:
                raise UnsafeToolRetryError(f"Tool call ID was already completed: {call_id}")
            new_calls[call_id] = ToolCallInspection(
                call_id=call_id, tool_name=item["name"], input=item,
                phase=ToolPhase.RECORDED, started_count=0,
            )
        self._calls.update(new_calls)

    def execute(self, tool_call, *, retry: bool = False) -> ToolExecutionOutcome:
        """Execute an already-persisted and registered call; never persist its result."""
        input_item = ContextHelper.serialize_items([tool_call])[0]
        if input_item.get("type") != "function_call":
            raise ValueError("Expected function_call")
        call_id = input_item["call_id"]
        status = self._calls.get(call_id)
        if status is None:
            raise ValueError(f"Tool call not registered after durable recording: {call_id}")
        if status.input != input_item:
            raise ValueError(f"Tool call changed since recording: {call_id}")
        # Cheap indexed query protects against a completed result whose in-memory
        # state update was interrupted. No Snapshot + Tail scan on this hot path.
        if (status.phase == ToolPhase.RESULT_RECORDED or
                self.execution_log.get_tool_interaction(call_id) is not None):
            raise UnsafeToolRetryError(f"Tool call already has result: {call_id}")

        spec = self.registry.get(input_item["name"])
        safety = spec.safety if spec is not None else ToolSafety.SIDE_EFFECTING
        if status.phase == ToolPhase.STARTED and not (retry and safety == ToolSafety.READ_ONLY):
            raise UnsafeToolRetryError(f"Unresolved tool call cannot be retried automatically: {call_id}")
        if retry and safety != ToolSafety.READ_ONLY:
            raise UnsafeToolRetryError(f"Retry is not allowed for side-effecting tool: {input_item['name']}")

        # Commit STARTED before invoking the handler. A failed commit must never
        # invoke the handler; a crash after commit leaves a safely unresolved call.
        self.execution_log.append_tool_execution_started(call_id)
        self._calls[call_id] = replace(
            status, phase=ToolPhase.STARTED, started_count=status.started_count + 1,
        )

        try:
            arguments = json.loads(input_item["arguments"])
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments not an object")
            if spec is None:
                raise KeyError(f"Unknown tool: {input_item['name']}")
        except Exception as exc:
            return self._prepare_result(input_item, {"error": repr(exc)}, False, repr(exc))

        try:
            result = spec.handler(**arguments)
        except UnknownToolOutcomeError:
            raise
        except Exception as exc:
            if safety == ToolSafety.SIDE_EFFECTING:
                raise UnknownToolOutcomeError(
                    f"Outcome of side-effecting tool {input_item['name']} ({call_id}) is unknown"
                ) from exc
            return self._prepare_result(input_item, {"error": repr(exc)}, False, repr(exc))
        return self._prepare_result(input_item, result, True, None)

    def mark_result_recorded(self, call_id: str) -> None:
        """Called only AFTER AgentSession.record_tool_interaction succeeds."""
        status = self._calls.get(call_id)
        if status is None or status.phase != ToolPhase.STARTED:
            raise ValueError(f"Unexpected result commit notification: {call_id}")
        self._calls[call_id] = replace(status, phase=ToolPhase.RESULT_RECORDED)

    @staticmethod
    def _prepare_result(input_item, result, returned_normally, error) -> ToolExecutionOutcome:
        output = {
            "type": "function_call_output",
            "call_id": input_item["call_id"],
            "output": json.dumps(result, ensure_ascii=False),
        }
        interaction = ToolInteraction(
            call_id=input_item["call_id"], input=input_item, output=output,
        )
        return ToolExecutionOutcome(
            interaction=interaction, result=result,
            handler_returned_normally=returned_normally, error=error,
        )
