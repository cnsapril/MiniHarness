import json

from compactor import compact_session
from context_manager import (
    ContextBudget,
    ContextManager,
    ContextHelper,
)
from execution_log import ExecutionLog
from main import client
from session import AgentSession

manager = ContextManager(
    ContextBudget(16384, 2048),
    compaction_ratio=0.8
)
execution_log = ExecutionLog()
session = AgentSession(manager, execution_log)

session.append([{
    "role": "user",
    "content": "Analyze the files in my test project."
}])

# 构造 5 个完整 Steps

for i in range(5):
    if i == 0:
        path = "src/payment"
        files = [
            "config.py",
            "gateway.py",
            "validator.py"
        ]
    else:
        path = f"src/module_{i}"
        files = [
            f"file_{j}.py"
            for j in range(60)
        ]

    session.append([
        {
            "type": "function_call",
            "name": "list_files",
            "call_id": f"call_{i}",
            "arguments": json.dumps({"path": path})
        },
        {
            "type": "function_call_output",
            "call_id": f"call_{i}",
            "output": json.dumps(files)
        }
    ])

    session.mark_step_complete()

before = ContextHelper.estimate_tokens(
    session.get_history()
)
events_before = execution_log.get_events()
print(f"Tool Call Before: {execution_log.get_tool_result('call_0')}")
# 真实调用 Qwen3 执行 Compaction
compact_session(session, client)

after = ContextHelper.estimate_tokens(
    session.get_history()
)
events_after = execution_log.get_events()
print(f"Tool Call After: {execution_log.get_tool_result('call_0')}")
print(f"Before Token: {before}, After Token: {after}")
print("Boundaries:", session.get_step_ends())
print("Summary:", session.get_history()[1]["content"])
print(f"Before Events: {len(events_before)}, After Events: {len(events_after)}")
assert events_before == events_after
assert(execution_log.get_tool_result('call_0') is not None)
assert after < before

# 测试压缩后能否恢复关键信息
response = client.responses.create(
    model="mini-qwen3-16k",
    instructions="Answer using only the provided history.",
    input=session.get_history() + [{
        "role": "user",
        "content": (
            "Which file was discovered in the payment "
            "directory? Give its exact path."
        ),
    }],
    tools=[],
)

print("Continuation:", response.output_text)
