"""Durable, minimal tool execution state machine.

SQLite is the execution-admission authority. Execution Log retains the agent
conversation; Tool Interaction Ledger retains completed input/output pairs.
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Callable, ClassVar

from context_manager import ContextHelper
from permission import DenialSource, PermissionPolicy, PermissionRequirement
from tool_model import ToolInteraction
from tool_registry import PROJECT_ROOT, ToolRegistry, ToolSafety


class ToolPhase(str, Enum):
    RECORDED = "RECORDED"
    STARTED = "STARTED"
    RETRY_READY = "RETRY_READY"
    RESULT_RECORDED = "RESULT_RECORDED"
    DENIED = "DENIED"


class ToolDispatchStatus(str, Enum):
    COMPLETED = "COMPLETED"
    APPROVAL_PENDING = "APPROVAL_PENDING"
    DENIED = "DENIED"


class ApprovalDecision(Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"


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
        return tuple(c for c in self.calls
                     if c.phase not in (ToolPhase.RESULT_RECORDED, ToolPhase.DENIED))

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
    status: ClassVar[ToolDispatchStatus] = ToolDispatchStatus.COMPLETED
    interaction: ToolInteraction
    result: object
    handler_returned_normally: bool
    error: str | None
    started_count: int


@dataclass(frozen=True)
class ToolApprovalPending:
    """Ordinary control-flow outcome: an approval is needed before execution."""
    status: ClassVar[ToolDispatchStatus] = ToolDispatchStatus.APPROVAL_PENDING
    call_id: str
    expected_started_count: int
    retry: bool


@dataclass(frozen=True)
class ToolDenialOutcome:
    status: ClassVar[ToolDispatchStatus] = ToolDispatchStatus.DENIED
    interaction: ToolInteraction
    reason: str
    expected_started_count: int
    retry: bool
    denial_source: DenialSource


class ToolExecutionManager:
    """Session-scoped executor. The database, not in-memory state, grants execution.

    The caller must commit the returned interaction using AgentSession with
    outcome.started_count. This manager never writes the final Tool Result.
    """

    def __init__(self, execution_log, registry: ToolRegistry,
                 policy: PermissionPolicy | None = None):
        self.execution_log = execution_log
        self.registry = registry
        # Trusted, session-fixed policy; never sourced from model arguments.
        self.policy = policy if policy is not None else PermissionPolicy(PROJECT_ROOT)

    def _recorded(self, tool_call):
        input_item = ContextHelper.serialize_items([tool_call])[0]
        if input_item.get("type") != "function_call":
            raise ValueError("Expected function_call")
        call_id = input_item["call_id"]
        record = self.execution_log.get_tool_execution(call_id)
        if record is None:
            raise UnsafeToolRetryError(f"Tool call not durably recorded: {call_id}")
        if record["input"] != input_item:
            raise UnsafeToolRetryError(f"Tool call changed since recording: {call_id}")
        return input_item, record

    def _permission(self, record):
        spec = self.registry.get(record["tool_name"])
        try:
            arguments = json.loads(record["input"]["arguments"])
        except (TypeError, ValueError):
            arguments = None
        return spec, arguments, self.policy.evaluate(spec, arguments)

    def approve(self, call_id: str, *, expected_started_count: int,
                retry: bool = False) -> None:
        """Call only from a trusted user-interface approval decision."""
        record = self.execution_log.get_tool_execution(call_id)
        if (record is None or record["started_count"] != expected_started_count
                or record["state"] != ("RETRY_READY" if retry else "RECORDED")):
            raise UnsafeToolRetryError(f"Stale approval: {call_id}")
        _, _, decision = self._permission(record)
        if decision.requirement != PermissionRequirement.ASK:
            raise UnsafeToolRetryError(f"Tool does not require approval: {call_id}")
        self.execution_log.approve_tool_execution(
            call_id, expected_started_count
        )

    def reject(self, tool_call, *, expected_started_count: int,
               retry: bool = False) -> ToolDenialOutcome:
        """Prepare a rejection; Session atomically commits REJECTED + DENIED + output."""
        input_item, record = self._recorded(tool_call)
        if record["started_count"] != expected_started_count:
            raise UnsafeToolRetryError(f"Stale rejection: {input_item['call_id']}")
        _, _, decision = self._permission(record)
        if decision.requirement != PermissionRequirement.ASK:
            raise UnsafeToolRetryError("No approval is required for this call")
        return self._prepare_denial(
            input_item, "User rejected this tool invocation",
            expected_started_count, retry, denial_source=DenialSource.USER
        )

    @staticmethod
    def _prepare_denial(input_item, reason: str, count: int, retry: bool,
                        *, denial_source: DenialSource = DenialSource.POLICY) -> ToolDenialOutcome:
        # A retry denial does not imply the earlier execution had no effects.
        result = {
            "error": "PERMISSION_DENIED",
            "reason": reason,
            "previous_outcome": "UNKNOWN" if retry else "NOT_STARTED",
        }
        output_item = {
            "type": "function_call_output", "call_id": input_item["call_id"],
            "output": json.dumps(result, ensure_ascii=False),
        }
        interaction = ToolInteraction(
            call_id=input_item["call_id"], input=input_item, output=output_item
        )
        return ToolDenialOutcome(
            interaction, reason, count, retry, denial_source
        )

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

    def execute(self, tool_call, *, retry: bool = False
                ) -> ToolExecutionOutcome | ToolDenialOutcome | ToolApprovalPending:
        """Permission gateway then durable CAS; no Handler runs on ASK or DENY."""
        input_item, record = self._recorded(tool_call)
        call_id = input_item["call_id"]
        spec, arguments, decision = self._permission(record)
        safety = spec.safety if spec is not None else ToolSafety.SIDE_EFFECTING
        if retry and safety != ToolSafety.READ_ONLY:
            raise UnsafeToolRetryError(f"Retry forbidden for {record['tool_name']}")
        expected_count = record["started_count"]
        expected_state = "RETRY_READY" if retry else "RECORDED"
        if record["state"] != expected_state:
            raise UnsafeToolRetryError(
                f"Tool call {call_id} is {record['state']}, expected {expected_state}"
            )

        if decision.requirement == PermissionRequirement.DENY:
            return self._prepare_denial(
                input_item, decision.reason, expected_count, retry
            )
        requires_approval = decision.requirement == PermissionRequirement.ASK
        if requires_approval:
            status = self.execution_log.get_tool_approval(call_id, expected_count)
            if status is None:
                # Create exactly once; concurrent duplicate creation fails
                # rather than being silently swallowed by INSERT OR IGNORE.
                try:
                    self.execution_log.create_pending_approval(
                        call_id, expected_count
                    )
                except ValueError as exc:
                    raise UnsafeToolRetryError(str(exc)) from exc
                return ToolApprovalPending(call_id, expected_count, retry)
            if status == "PENDING":
                # A resumed/duplicate read is not a second approval request.
                return ToolApprovalPending(call_id, expected_count, retry)
            if status != "APPROVED":
                raise UnsafeToolRetryError(f"Approval is not usable: {call_id}")

        # The approval consumption and execution CAS COMMIT together.
        try:
            granted_version = self.execution_log.acquire_tool_execution(
                call_id, expected_started_count=expected_count, retry=retry,
                require_approval=requires_approval,
            )
        except ValueError as exc:
            raise UnsafeToolRetryError(str(exc)) from exc

        # No unexpected local policy/argument errors should reach a Handler.
        try:
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments not an object")
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

    def run_with_approval(
        self, tool_call,
        approval_provider: Callable[[ToolApprovalPending, dict], ApprovalDecision],
        *, retry: bool = False,
    ) -> ToolExecutionOutcome | ToolDenialOutcome:
        """Run the permission+approval workflow through a trusted UI port.

        Only presentation and the human decision live outside this manager.
        Persisting the final response still belongs to AgentSession.
        """
        outcome = self.execute(tool_call, retry=retry)
        if outcome.status != ToolDispatchStatus.APPROVAL_PENDING:
            return outcome

        # Give the UI the immutable recorded request, never a model-crafted
        # approval flag. The callback is provided by trusted host code.
        input_item, _ = self._recorded(tool_call)
        decision = approval_provider(outcome, input_item)
        if decision == ApprovalDecision.REJECT:
            return self.reject(
                tool_call, expected_started_count=outcome.expected_started_count,
                retry=outcome.retry,
            )
        if decision != ApprovalDecision.APPROVE:
            raise ValueError(f"Invalid approval decision: {decision!r}")

        self.approve(
            outcome.call_id, expected_started_count=outcome.expected_started_count,
            retry=outcome.retry,
        )
        resumed = self.execute(tool_call, retry=outcome.retry)
        if resumed.status == ToolDispatchStatus.APPROVAL_PENDING:
            raise RuntimeError("Approved invocation unexpectedly remains pending")
        return resumed

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
