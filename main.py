import json
import logging
from pathlib import Path
from time import perf_counter

from openai import OpenAI
from openai.types.responses import Response

from logging_config import setup_logging
from trace_model import AgentTrace, LLMCallTrace, ToolCallTrace

PROJECT_ROOT = Path.cwd().resolve()
logger = logging.getLogger(__name__)


# Tool Functions
def list_files(path: str) -> list[str]:
    target = (PROJECT_ROOT / path).resolve()

    if not target.is_relative_to(PROJECT_ROOT):
        raise PermissionError("Path outside project root")

    if not target.is_dir():
        raise NotADirectoryError(path)

    return sorted(item.name for item in target.iterdir())


TOOL_HANDLERS = {
    "list_files": list_files
}

# Initializations
MAX_STEPS = 10
MAX_TOOL_CALLS = 10
MAX_LLM_RETRIES = 3
client = OpenAI(
    base_url="http://localhost:11434/v1/",
    api_key="ollama"
)
tools = [
    {
        "type": "function",
        "name": "list_files",
        "description": "List files and directories under a given path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative directory path, e.g. '.' or 'src'"
                }
            },
            "required": ["path"],
            "additionalProperties": False
        },
        "strict": True
    }
]


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


def record_exception_llm_trace(trace: AgentTrace, start: float, total_steps: int, retry_time: int, error: str):
    trace.llm_calls.append(LLMCallTrace(
        step=total_steps + 1,
        duration_ms=(perf_counter() - start) * 1000,
        input_tokens=None,
        output_tokens=None,
        response_id=None,
        output_types=[],
        success=False,
        retry_time=retry_time,
        error=error
    ))


def call_llm_with_retry_and_extend_history(history: list[dict[str, str]], total_steps: int, trace: AgentTrace):
    # 检查是否超限
    if total_steps >= MAX_STEPS:
        raise RuntimeError(f"Agent steps have exceeded limit {MAX_STEPS}")
    # 使用对话历史调用LLM并获取响应
    for attempt in range(1, MAX_LLM_RETRIES + 1):
        start = perf_counter()
        try:
            response = client.responses.create(
                model="qwen3:8b",
                instructions=(
                    "You are a coding assistant. "
                    "Use tools when you need information about the project. "
                    "Do not invent file names."
                ),
                input=history,
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
        history.extend(response.output)
        return response
    raise RuntimeError(f"LLM retries have exceeded limit {MAX_LLM_RETRIES}")


def record_tool_call_trace(trace: AgentTrace, start: float, current_step: int, call_id: str, tool_name: str,
                           success: bool, error: str | None):
    trace.tool_calls.append(ToolCallTrace(
        step=current_step,
        call_id=call_id,
        tool_name=tool_name,
        duration_ms=(perf_counter() - start) * 1000,
        success=success,
        error=error
    ))


def execute_function_calls(history, response, total_tool_calls: int, current_step: int, trace: AgentTrace) -> int:
    tool_call_count = 0
    for item in response.output:
        # 非function call的先跳过
        if item.type != "function_call":
            logger.info(item)
            continue

        # 额度超限检查
        if total_tool_calls + tool_call_count >= MAX_TOOL_CALLS:
            raise RuntimeError(f"Agent tool calls have exceeded limit {MAX_TOOL_CALLS}")
        # 进行function call调用
        start = perf_counter()
        tool_name = item.name
        call_id = item.call_id
        try:
            arguments = json.loads(item.arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments not an object")
            logger.info("[Tool Call] %s: %s", tool_name, arguments)
            handler = TOOL_HANDLERS[tool_name]
            result = handler(**arguments)
        except Exception as e:
            result = {"error": repr(e)}
            record_tool_call_trace(trace, start, current_step, call_id, tool_name, False, repr(e))
        # 处理function call结果
        record_tool_call_trace(trace, start, current_step, call_id, tool_name, True, None)
        logger.info("[Tool Result] %s", result)
        history.append({
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(result, ensure_ascii=False)
        })
        tool_call_count += 1

    return tool_call_count


# Main Agent Loop
def agent_loop(user_prompt: str) -> str:
    history = [{"role": "user", "content": user_prompt}]
    trace = AgentTrace()
    total_steps = 0
    total_tool_calls = 0
    while True:
        # LLM调用
        response = call_llm_with_retry_and_extend_history(history, total_steps, trace)
        total_steps += 1
        # Function Call处理
        tool_call_count = execute_function_calls(history, response, total_tool_calls, total_steps, trace)
        total_tool_calls += tool_call_count
        # 判断正常执行终止
        if tool_call_count == 0:
            logger.info("=== Agent Trace ===")
            logger.info(trace)
            return response.output_text


if __name__ == "__main__":
    setup_logging()
    assistant_message = agent_loop("List my current project working directory.")
    logger.info("\n[Agent Response]")
    logger.info(assistant_message)
