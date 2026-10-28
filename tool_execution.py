"""Durable, minimal tool execution state machine.

SQLite is the execution-admission authority. Execution Log retains the agent
conversation; Tool Interaction Ledger retains completed input/output pairs.
"""

import json
from dataclasses import dataclass
from enum import Enum

from context_manager import ContextHelper
from tool_model import ToolInteraction
from tool_registry import ToolRegistry, ToolSafety


class ToolPhase(str, Enum):
    RECORDED = "RECORDED"
    STARTED = "STARTED"
    RETRY_READY = "RETRY_READY"
    RESULT_RECORDED = "RESULT_RECORDED"


class UnsafeToolRetryError(RuntimeError):
    """Tool execution permission or explicit retry authorization was denied."""


class UnknownToolOutcomeError(RuntimeError):
    """A side-effecting tool may have taken effect; stop rather than auto-retry."""


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
        return tuple(c for c in self.calls if c.phase != ToolPhase.RESULT_RECORDED)

    def get(self, call_id: str) -> ToolCallInspection | None:
        return next((c for c in self.calls if c.call_id == call_id), None)


class RecoveryInspector:
    """Read-only view of the durable tool_executions projection.

    Unlike working-context recovery, no event replay or snapshot interpretation
    is needed to determine tool execution state, even after compaction.
    """

    def __init__(self, execution_log):
        self.execution_log = execution_log

    def inspect(self) -> RecoveryReport:
        calls = tuple(
            ToolCallInspection(
                call_id=row["call_id"],
                tool_name=row["tool_name"],
                input=row["input"],
                phase=ToolPhase(row["state"]),
                started_count=row["started_count"],
            )
            for row in self.execution_log.list_tool_executions()
        )
        return RecoveryReport(self.execution_log.get_session_id(), calls)


@dataclass(frozen=True)
class ToolExecutionOutcome:
    """Prepared response; not yet committed and not necessarily business success."""
    interaction: ToolInteraction
    result: object
    handler_returned_normally: bool
    error: str | None
    started_count: int


class ToolExecutionManager:
    """Session-scoped executor. The database, not in-memory state, grants execution.

    The caller must commit the returned interaction using AgentSession with
    outcome.started_count. This manager never writes the final Tool Result.
    """

    def __init__(self, execution_log, registry: ToolRegistry):
        self.execution_log = execution_log
        self.registry = registry

    def authorize_retry(self, call_id: str, *, expected_started_count: int) -> None:
        """Explicit recovery step, for trusted READ_ONLY tools only.

        The caller must ensure the previous executor has stopped. Side-effecting
        tools require external reconciliation, which is not in this v0.1.
        """
        record = self.execution_log.get_tool_execution(call_id)
        if record is None:
            raise UnsafeToolRetryError(f"Unknown tool call: {call_id}")
        spec = self.registry.get(record["tool_name"])
        if spec is None or spec.safety != ToolSafety.READ_ONLY:
            raise UnsafeToolRetryError(f"Retry not authorized for {record['tool_name']}")
        try:
            self.execution_log.authorize_tool_retry(call_id, expected_started_count)
        except ValueError as exc:
            raise UnsafeToolRetryError(str(exc)) from exc

    def execute(self, tool_call, *, retry: bool = False) -> ToolExecutionOutcome:
        """Acquire durable permission before invoking any handler."""
        input_item = ContextHelper.serialize_items([tool_call])[0]
        if input_item.get("type") != "function_call":
            raise ValueError("Expected function_call")
        call_id = input_item["call_id"]
        record = self.execution_log.get_tool_execution(call_id)
        if record is None:
            raise UnsafeToolRetryError(f"Tool call not durably recorded: {call_id}")
        if record["input"] != input_item:
            raise UnsafeToolRetryError(f"Tool call changed since recording: {call_id}")

        spec = self.registry.get(record["tool_name"])
        safety = spec.safety if spec is not None else ToolSafety.SIDE_EFFECTING
        if retry and safety != ToolSafety.READ_ONLY:
            raise UnsafeToolRetryError(f"Retry forbidden for {record['tool_name']}")

        # Optimistic CAS: only one competing executor can claim this version.
        # The COMMIT must succeed before any handler is invoked.
        try:
            granted_version = self.execution_log.acquire_tool_execution(
                call_id, expected_started_count=record["started_count"], retry=retry
            )
        except ValueError as exc:
            raise UnsafeToolRetryError(str(exc)) from exc

        try:
            arguments = json.loads(input_item["arguments"])
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments not an object")
            if spec is None:
                raise KeyError(f"Unknown tool: {record['tool_name']}")
        except Exception as exc:
            return self._prepare_result(input_item, {"error": repr(exc)},
                                        False, repr(exc), granted_version)

        try:
            result = spec.handler(**arguments)
        except UnknownToolOutcomeError:
            raise
        except Exception as exc:
            if safety == ToolSafety.SIDE_EFFECTING:
                raise UnknownToolOutcomeError(
                    f"Outcome of side-effecting tool {record['tool_name']} ({call_id}) is unknown"
                ) from exc
            return self._prepare_result(input_item, {"error": repr(exc)},
                                        False, repr(exc), granted_version)
        return self._prepare_result(input_item, result, True, None, granted_version)

    @staticmethod
    def _prepare_result(input_item, result, returned_normally,
                        error, granted_version) -> ToolExecutionOutcome:
        output = {
            "type": "function_call_output",
            "call_id": input_item["call_id"],
            "output": json.dumps(result, ensure_ascii=False),
        }
        interaction = ToolInteraction(
            call_id=input_item["call_id"], input=input_item, output=output
        )
        return ToolExecutionOutcome(
            interaction=interaction, result=result,
            handler_returned_normally=returned_normally, error=error,
            started_count=granted_version
        )
