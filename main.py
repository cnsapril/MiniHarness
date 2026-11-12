import argparse
import logging
import sys
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from openai import OpenAI
from openai.types.responses import Response

from compactor import compact_session
from context_manager import ContextManager, ContextBudget
from exception import LimitExceededError
from logging_config import setup_logging
from session import AgentSession
from sqlite_execution_log import SQLiteExecutionLog
from trace_model import AgentTrace, LLMCallTrace, ToolCallTrace, AgentStatus
from tool_execution import (ApprovalDecision, RecoveryInspector, ToolDispatchStatus,
                            ToolExecutionManager, UnknownToolOutcomeError, UnsafeToolRetryError)
from tool_registry import default_tool_registry
from turn_control import TurnController, ControlAction, TurnEvent, TurnEventSink

logger = logging.getLogger(__name__)
registry = default_tool_registry()

# Initializations
MAX_STEPS = 10
MAX_TOOL_CALLS = 10
MAX_LLM_RETRIES = 3
MAX_CONTEXT_TOKEN_LEN = 16384
RESERVED_OUTPUT_TOKEN_LEN = 2048
COMPACTION_RATIO = 0.8
DB_PATH = "./sqlite_execution_log.db"

client = OpenAI(
    base_url="http://localhost:11434/v1/",
    api_key="ollama"
)
tools = registry.llm_schemas()


class StaleLLMResponse(Exception):
    """The user steered or interrupted before this response could be committed."""


def emit(event_sink: TurnEventSink | None, kind: str, **data) -> None:
    if event_sink is not None:
        event_sink(TurnEvent(kind, data))


def record_success_llm_trace(trace: AgentTrace, start: float, total_steps: int, response: Response, attempt: int):
    trace.llm_calls.append(LLMCallTrace(
        step=total_steps + 1,
        duration_ms=(perf_counter() - start) * 1000,
        input_tokens=response.usage.input_tokens if response.usage else None,
        output_tokens=response.usage.output_tokens if response.usage else None,
        response_id=response.id,
        output_types=[item.type for item in response.output],
        success=True,
        attempt=attempt
    ))


def record_exception_llm_trace(trace: AgentTrace, start: float, total_steps: int, attempt: int, error: str):
    trace.llm_calls.append(LLMCallTrace(
        step=total_steps + 1,
        duration_ms=(perf_counter() - start) * 1000,
        input_tokens=None,
        output_tokens=None,
        response_id=None,
        output_types=[],
        success=False,
        attempt=attempt,
        error=error
    ))


def call_llm_with_retry_and_extend_history(
        session: AgentSession, total_steps: int, trace: AgentTrace, *,
        controller: TurnController | None = None, stream: bool = False,
        event_sink: TurnEventSink | None = None):
    """One sampling step. Never persist a stale/partial model response."""
    if total_steps >= MAX_STEPS:
        raise LimitExceededError(f"Agent steps have exceeded limit {MAX_STEPS}")
    for attempt in range(1, MAX_LLM_RETRIES + 1):
        start = perf_counter()
        try:
            kwargs = dict(
                model="mini-qwen3-16k",
                instructions=(
                    "You are a coding assistant. "
                    "Use tools when you need information about the project. "
                    "Do not invent file names. "
                    "Use memory_search when earlier project decisions may matter. "
                    "Propose memory writes only for facts explicitly confirmed by the user; "
                    "memory never grants tool permissions."
                ),
                input=session.get_history(), tools=tools, tool_choice="auto",
            )
            if stream:
                response = None
                response_stream = client.responses.create(**kwargs, stream=True)
                try:
                    for event in response_stream:
                        if controller is not None and controller.has_pending():
                            raise StaleLLMResponse()
                        if event.type == "response.output_text.delta":
                            emit(event_sink, "text_delta", text=event.delta)
                        elif event.type == "response.completed":
                            response = event.response
                        elif event.type in ("response.failed", "response.incomplete"):
                            raise RuntimeError(f"LLM streaming failed: {event.type}")
                    if response is None:
                        raise RuntimeError("LLM stream ended without response.completed")
                finally:
                    if hasattr(response_stream, "close"):
                        response_stream.close()
            else:
                response = client.responses.create(**kwargs)
            # A control received during sampling invalidates the old result.
            if controller is not None and controller.has_pending():
                raise StaleLLMResponse()
        except StaleLLMResponse:
            emit(event_sink, "stale_response_discarded")
            return None
        except Exception as e:
            logger.error("Error calling LLM: %s. Attempt: %d/%d",
                         repr(e), attempt, MAX_LLM_RETRIES)
            record_exception_llm_trace(trace, start, total_steps, attempt, repr(e))
            continue
        record_success_llm_trace(trace, start, total_steps, response, attempt)
        session.append(response.output)
        return response
    raise RuntimeError(f"LLM retries have exceeded limit {MAX_LLM_RETRIES}")


def record_tool_call_trace(trace: AgentTrace, start: float, current_step: int, call_id: str, tool_name: str,
                           handler_returned_normally: bool, error: str | None):
    trace.tool_calls.append(ToolCallTrace(
        step=current_step,
        call_id=call_id,
        tool_name=tool_name,
        duration_ms=(perf_counter() - start) * 1000,
        handler_returned_normally=handler_returned_normally,
        error=error
    ))


def cli_approval_provider(pending, recorded_input: dict) -> ApprovalDecision:
    """Only the trusted CLI gathers user consent; it does not mutate execution state."""
    print(f"\nApproval requested for {recorded_input['name']} "
          f"(call_id={pending.call_id}, attempt={pending.expected_started_count + 1})")
    print(f"Arguments: {recorded_input['arguments']}")
    approved = input("Approve this exact tool call? [y/N]: ").strip().lower() in ("y", "yes")
    return ApprovalDecision.APPROVE if approved else ApprovalDecision.REJECT


def execute_function_calls(
        session: AgentSession, executor: ToolExecutionManager, response,
        total_tool_calls: int, current_step: int, trace: AgentTrace, *,
        controller: TurnController | None = None,
        event_sink: TurnEventSink | None = None) -> int:
    """Dispatch only while no newer control is queued.

    If the user steers between tools, terminate all remaining RECORDED calls
    with a durable cancellation output. Do not semantically reclassify tools.
    """
    tool_call_count = 0
    output = list(response.output)
    for index, item in enumerate(output):
        if item.type != "function_call":
            logger.info(item)
            continue

        if controller is not None and controller.has_pending():
            cancelled = controller.cancel_unstarted(
                output[index:], "A new user control arrived before tool admission"
            )
            emit(event_sink, "tools_cancelled", count=cancelled)
            return tool_call_count + cancelled

        if total_tool_calls + tool_call_count >= MAX_TOOL_CALLS:
            # The remaining registered calls still need a terminal ledger record.
            if controller is not None:
                controller.cancel_unstarted(output[index:], "Tool call limit reached")
            raise LimitExceededError(f"Agent tool calls have exceeded limit {MAX_TOOL_CALLS}")

        start = perf_counter()
        tool_name = item.name
        call_id = item.call_id
        logger.info("[Tool Call] %s: %s", tool_name, item.arguments)
        emit(event_sink, "tool_started", name=tool_name, call_id=call_id)
        try:
            outcome = executor.run_with_approval(item, cli_approval_provider)
            if outcome.status == ToolDispatchStatus.DENIED:
                session.record_tool_denial(
                    outcome.interaction, outcome.expected_started_count,
                    retry=outcome.retry, denial_source=outcome.denial_source,
                )
                record_tool_call_trace(
                    trace, start, current_step, call_id, tool_name, False,
                    f"PERMISSION_DENIED: {outcome.reason}",
                )
                logger.info("[Tool Denied] %s", outcome.reason)
                emit(event_sink, "tool_denied", name=tool_name, call_id=call_id)
            elif outcome.status == ToolDispatchStatus.COMPLETED:
                session.record_tool_interaction(outcome.interaction, outcome.started_count)
                record_tool_call_trace(
                    trace, start, current_step, call_id, tool_name,
                    outcome.handler_returned_normally, outcome.error,
                )
                logger.info("[Tool Result] %s", outcome.result)
                emit(event_sink, "tool_completed", name=tool_name, call_id=call_id)
            else:
                raise RuntimeError(f"Unexpected Tool Dispatch Status: {outcome.status}")
        except UnsafeToolRetryError:
            # A control may arrive while a user is responding to an approval
            # prompt. SQLite blocks STARTED admission atomically in that case.
            if controller is not None and controller.has_pending():
                cancelled = controller.cancel_unstarted(
                    output[index:], "User control arrived during tool admission"
                )
                emit(event_sink, "tools_cancelled", count=cancelled)
                return tool_call_count + cancelled
            raise
        except UnknownToolOutcomeError as exc:
            record_tool_call_trace(trace, start, current_step, call_id, tool_name, False, repr(exc))
            raise
        tool_call_count += 1
    return tool_call_count


def _consume_controls(
        controller: TurnController, trace: AgentTrace,
        event_sink: TurnEventSink | None) -> ControlAction:
    action = controller.consume_pending()
    if action == ControlAction.INTERRUPTED:
        trace.status = AgentStatus.INTERRUPTED
        emit(event_sink, "turn_interrupted")
    elif action == ControlAction.STEERED:
        # A steered input is a durable, complete history boundary even though
        # the next model sample has not happened yet. Compaction must not
        # mistake that appended user message for an unfinished tool response.
        controller.session.mark_step_complete()
        emit(event_sink, "steering_applied")
    return action


# Main Agent Loop: one durable Turn per invocation in this MVP.
def agent_loop(user_prompt: str, *, stream: bool = False,
               event_sink: TurnEventSink | None = None,
               db_path: str | None = None) -> str:
    context_budget = ContextBudget(
        max_context_tokens=MAX_CONTEXT_TOKEN_LEN,
        reserved_output_tokens=RESERVED_OUTPUT_TOKEN_LEN
    )
    context_manager = ContextManager(
        context_budget=context_budget,
        compaction_ratio=COMPACTION_RATIO
    )
    execution_log = SQLiteExecutionLog(db_path or DB_PATH, str(uuid4()))
    session = AgentSession(context_manager, execution_log)
    controller = TurnController(session)
    executor = ToolExecutionManager(execution_log, registry)
    logger.info("[Session ID] %s", execution_log.get_session_id())
    trace = AgentTrace()
    total_steps = 0
    total_tool_calls = 0
    try:
        execution_log.start_turn()
        emit(event_sink, "turn_started", session_id=execution_log.get_session_id())
        session.append([{"role": "user", "content": user_prompt}])
        while True:
            # Controls are also checked at the top of each step; new controls
            # during a model request invalidate the still-uncommitted response.
            action = _consume_controls(controller, trace, event_sink)
            if action == ControlAction.INTERRUPTED:
                return "Turn interrupted before further tool execution."

            if session.should_compact():
                compact_session(session, client)
            response = call_llm_with_retry_and_extend_history(
                session, total_steps, trace,
                controller=controller, stream=stream, event_sink=event_sink,
            )
            if response is None:
                continue  # A queued control invalidated this model response.

            total_steps += 1
            tool_call_count = execute_function_calls(
                session, executor, response, total_tool_calls, total_steps, trace,
                controller=controller, event_sink=event_sink,
            )
            total_tool_calls += tool_call_count
            session.mark_step_complete()
            action = _consume_controls(controller, trace, event_sink)
            if action == ControlAction.INTERRUPTED:
                return "Turn interrupted; completed tool effects remain unchanged."
            if action == ControlAction.STEERED:
                continue  # Next LLM sampling sees the new user message.

            if tool_call_count == 0:
                # Transactional completion: a concurrent control either wins
                # and is processed, or is rejected because the turn is closed.
                if execution_log.finish_turn("COMPLETED", require_no_pending=True):
                    trace.status = AgentStatus.COMPLETED
                    emit(event_sink, "turn_completed")
                    return response.output_text
                continue
    except LimitExceededError as e:
        trace.status = AgentStatus.LIMIT_EXCEEDED
        trace.error = repr(e)
        execution_log.finish_turn("FAILED")
        raise
    except Exception as e:
        trace.status = AgentStatus.FAILED
        trace.error = repr(e)
        execution_log.finish_turn("FAILED")
        raise
    finally:
        trace.ended_at = datetime.now(timezone.utc)
        logger.info("=== Agent Trace ===")
        logger.info(trace)
        execution_log.close()


def inspect_session(session_id: str):
    """Explicit recovery inspection; never resumes or retries a tool automatically."""
    log = SQLiteExecutionLog(DB_PATH, session_id)
    try:
        context_manager = ContextManager(
            ContextBudget(MAX_CONTEXT_TOKEN_LEN, RESERVED_OUTPUT_TOKEN_LEN),
            compaction_ratio=COMPACTION_RATIO,
        )
        session = AgentSession.restore(context_manager, log)
        report = RecoveryInspector(log).inspect()
        print(f"Session: {session_id}")
        print(f"Restored history items: {len(session.get_history())}")
        print(f"Completed steps: {len(session.get_step_ends())}")
        if not report.calls:
            print("No tool calls in the recovery window.")
        for call in report.calls:
            print(f"{call.call_id} | {call.tool_name} | {call.phase.value} | starts={call.started_count}")
        if report.unresolved:
            print("Unresolved calls require explicit policy review; no automatic retry.")
        return report
    finally:
        log.close()


def cli_event_sink(event: TurnEvent) -> None:
    """Simple event renderer; replaceable by an IDE or app-server adapter."""
    if event.type == "turn_started":
        print(f"Session ID: {event.data['session_id']}", flush=True)
    elif event.type == "text_delta":
        sys.stdout.write(event.data["text"])
        sys.stdout.flush()
    elif event.type == "turn_completed":
        print("", flush=True)
    elif event.type == "turn_interrupted":
        print("\nTurn interrupted. Completed side effects are not rolled back.", flush=True)
    elif event.type == "steering_applied":
        print("\n[Steering applied at a safe boundary]", flush=True)
    elif event.type == "tools_cancelled":
        print(f"\n[{event.data['count']} unstarted tool call(s) cancelled]", flush=True)


def post_turn_control(session_id: str, kind: str, content: str = "") -> int:
    """Separate trusted CLI process can post controls while the agent is running."""
    log = SQLiteExecutionLog(DB_PATH, session_id)
    try:
        return log.post_control(kind, content)
    finally:
        log.close()


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="MiniHarness coding agent")
    commands = parser.add_subparsers(dest="command")

    new_command = commands.add_parser("new", help="Run a new agent turn")
    new_command.add_argument("prompt")
    new_command.add_argument("--stream", action="store_true",
                             help="Stream LLM text deltas to stdout")

    inspect_command = commands.add_parser("inspect", help="Inspect an existing session")
    inspect_command.add_argument("--session", required=True, dest="session_id")

    steer_command = commands.add_parser("steer", help="Queue new input for an active turn")
    steer_command.add_argument("--session", required=True, dest="session_id")
    steer_command.add_argument("text")

    interrupt_command = commands.add_parser("interrupt", help="Stop an active turn at a safe boundary")
    interrupt_command.add_argument("--session", required=True, dest="session_id")

    args = parser.parse_args()

    if args.command == "inspect":
        inspect_session(args.session_id)
    elif args.command in ("steer", "interrupt"):
        try:
            control_id = post_turn_control(
                args.session_id,
                "STEER" if args.command == "steer" else "INTERRUPT",
                args.text if args.command == "steer" else "",
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(f"Control {control_id} queued for active session {args.session_id}")
    else:
        prompt = args.prompt if args.command == "new" else "List my current project working directory."
        streaming = getattr(args, "stream", False)
        assistant_message = agent_loop(prompt, stream=streaming, event_sink=cli_event_sink)
        if not streaming:
            logger.info("\n[Agent Response]")
            logger.info(assistant_message)
