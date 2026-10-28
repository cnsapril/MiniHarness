import argparse
import logging
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
from tool_execution import RecoveryInspector, ToolExecutionManager, UnknownToolOutcomeError
from tool_registry import default_tool_registry

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


def call_llm_with_retry_and_extend_history(session: AgentSession, total_steps: int, trace: AgentTrace):
    # 检查是否超限
    if total_steps >= MAX_STEPS:
        raise LimitExceededError(f"Agent steps have exceeded limit {MAX_STEPS}")
    # 使用对话历史调用LLM并获取响应
    for attempt in range(1, MAX_LLM_RETRIES + 1):
        start = perf_counter()
        try:
            response = client.responses.create(
                model="mini-qwen3-16k",
                instructions=(
                    "You are a coding assistant. "
                    "Use tools when you need information about the project. "
                    "Do not invent file names."
                ),
                input=session.get_history(),
                tools=tools,
                tool_choice="auto"
            )
            # TODO: 应该区分可重试和不可重试异常
        except Exception as e:
            logger.error("Error calling LLM: %s. Attempt: %d/%d", repr(e), attempt, MAX_LLM_RETRIES)
            # 记录异常LLM调用trace
            record_exception_llm_trace(trace, start, total_steps, attempt, repr(e))
            continue
        # 记录成功LLM调用trace
        record_success_llm_trace(trace, start, total_steps, response, attempt)
        # 响应缓存至对话历史
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


def execute_function_calls(session: AgentSession, executor: ToolExecutionManager, response,
                           total_tool_calls: int, current_step: int, trace: AgentTrace) -> int:
    tool_call_count = 0
    for item in response.output:
        if item.type != "function_call":
            logger.info(item)
            continue

        if total_tool_calls + tool_call_count >= MAX_TOOL_CALLS:
            raise LimitExceededError(f"Agent tool calls have exceeded limit {MAX_TOOL_CALLS}")

        start = perf_counter()
        tool_name = item.name
        call_id = item.call_id
        logger.info("[Tool Call] %s: %s", tool_name, item.arguments)
        try:
            outcome = executor.execute(item)
            # Session owns the atomic Tool Result + Ledger write and Context update.
            session.record_tool_interaction(outcome.interaction)
            executor.mark_result_recorded(call_id)
        except UnknownToolOutcomeError as exc:
            # Do not turn an uncertain external outcome into a normal tool error.
            record_tool_call_trace(trace, start, current_step, call_id, tool_name, False, repr(exc))
            raise

        record_tool_call_trace(trace, start, current_step, call_id,
                               tool_name, outcome.handler_returned_normally, outcome.error)
        logger.info("[Tool Result] %s", outcome.result)
        tool_call_count += 1

    return tool_call_count


# Main Agent Loop
def agent_loop(user_prompt: str) -> str:
    context_budget = ContextBudget(
        max_context_tokens=MAX_CONTEXT_TOKEN_LEN,
        reserved_output_tokens=RESERVED_OUTPUT_TOKEN_LEN
    )
    context_manager = ContextManager(
        context_budget=context_budget,
        compaction_ratio=COMPACTION_RATIO
    )
    execution_log = SQLiteExecutionLog(DB_PATH, str(uuid4()))
    session = AgentSession(context_manager, execution_log)
    executor = ToolExecutionManager(execution_log, registry)
    logger.info("[Session ID] %s", execution_log.get_session_id())
    trace = AgentTrace()
    total_steps = 0
    total_tool_calls = 0
    try:
        session.append([{"role": "user", "content": user_prompt}])
        while True:
            # 处理上下文压缩
            if session.should_compact():
                compact_session(session, client)
            # LLM调用
            response = call_llm_with_retry_and_extend_history(session, total_steps, trace)
            # The response has been durably recorded by AgentSession; now register
            # its tool calls in the per-session runtime before any execution.
            executor.register_recorded_calls(response.output)
            total_steps += 1
            # Function Call处理
            tool_call_count = execute_function_calls(
                session, executor, response, total_tool_calls, total_steps, trace
            )
            total_tool_calls += tool_call_count
            # 记录Context Safe Boundary
            session.mark_step_complete()
            # 判断正常执行终止
            if tool_call_count == 0:
                trace.status = AgentStatus.COMPLETED
                return response.output_text
    except LimitExceededError as e:
        trace.status = AgentStatus.LIMIT_EXCEEDED
        trace.error = repr(e)
        raise
    except Exception as e:
        trace.status = AgentStatus.FAILED
        trace.error = repr(e)
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


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="MiniHarness coding agent")
    commands = parser.add_subparsers(dest="command")
    new_command = commands.add_parser("new", help="Run a new agent session")
    new_command.add_argument("prompt")
    inspect_command = commands.add_parser("inspect", help="Inspect an existing session")
    inspect_command.add_argument("--session", required=True, dest="session_id")
    args = parser.parse_args()

    if args.command == "inspect":
        inspect_session(args.session_id)
    else:
        prompt = args.prompt if args.command == "new" else "List my current project working directory."
        assistant_message = agent_loop(prompt)
        logger.info("\n[Agent Response]")
        logger.info(assistant_message)
