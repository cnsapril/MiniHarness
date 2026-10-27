import json
import tempfile
from pathlib import Path

from compactor import compact_session
from context_manager import ContextManager, ContextBudget
from main import client
from session import AgentSession
from sqlite_execution_log import SQLiteExecutionLog

with tempfile.TemporaryDirectory() as directory:
    db_path = str(Path(directory) / "agent.db")

    # First process
    log1 = SQLiteExecutionLog(db_path, "session-001")
    manager1 = ContextManager(
        ContextBudget(16384, 2048), 0.8
    )
    session1 = AgentSession(manager1, log1)
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

        session1.append([
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
        session1.mark_step_complete()
    compact_session(session1, client)
    session1.append([
        {"role": "user", "content": "List project files"}
    ])
    session1.mark_step_complete()
    expected_history = session1.get_history()
    expected_ends = session1.get_step_ends()
    log1.close()

    # Simulated restart
    log2 = SQLiteExecutionLog(db_path, "session-001")
    manager2 = ContextManager(
        ContextBudget(16384, 2048), 0.8
    )

    restored = AgentSession.restore(manager2, log2)

    assert restored.get_history() == expected_history
    assert restored.get_step_ends() == expected_ends
    assert restored.context_manager._cached_tokens is None

    print("Session recovery successful")

    log2.close()
