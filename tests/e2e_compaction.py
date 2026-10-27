
import json

from main import client
from compactor import compact_session
from context_manager import (
    ContextBudget,
    ContextManager,
    ContextHelper,
)
from session import AgentSession

manager = ContextManager(
    ContextBudget(16384, 2048),
    compaction_ratio=0.8
)
session = AgentSession(manager)

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

# 真实调用 Qwen3 执行 Compaction
compact_session(session, client)

after = ContextHelper.estimate_tokens(
    session.get_history()
)

print(f"Before: {before}, After: {after}")
print("Boundaries:", session.get_step_ends())
print("Summary:", session.get_history()[1]["content"])

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
