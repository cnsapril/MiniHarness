import unittest
from unittest import TestCase
from unittest.mock import Mock
from uuid import uuid4

from compactor import compact_session
from context_manager import ContextBudget, ContextManager, ContextHelper
from session import AgentSession
from sqlite_execution_log import SQLiteExecutionLog

DB_PATH = "tests/test_sqlite_execution_log.db"


class CompactorTest(TestCase):

    def test_compaction_llm_failure(self):
        # Arrange: 准备Session
        budget = ContextBudget(
            max_context_tokens=16384,
            reserved_output_tokens=2048
        )
        manager = ContextManager(budget, 0.8)
        execution_log = SQLiteExecutionLog(DB_PATH, str(UUID()))
        session = AgentSession(manager, execution_log)
        session.append([
            {"role": "user", "content": "List project files"}
        ])
        # 构造 3 个完整的 Agent Steps
        for i in range(3):
            session.append([
                {
                    "type": "function_call",
                    "name": "list_files",
                    "call_id": f"call_{i}",
                    "arguments": '{"path":"."}'
                },
                {
                    "type": "function_call_output",
                    "call_id": f"call_{i}",
                    "output": '["main.py"]'
                }
            ])
            session.mark_step_complete()
        # 保存执行前的状态
        history_before = session.get_history()
        step_ends_before = session.get_step_ends()
        cache_before = manager._cached_tokens

        # Arrange: 模拟LLM调用失败
        client = Mock()
        client.responses.create.side_effect = RuntimeError("Ollama unavailable")

        # Act + Assert: 验证异常
        with self.assertRaisesRegex(RuntimeError, "Ollama unavailable"):
            compact_session(session, client)

        # Assert: 验证Session没被修改
        self.assertEqual(session.get_history(), history_before)
        self.assertEqual(session.get_step_ends(), step_ends_before)
        self.assertEqual(cache_before, manager._cached_tokens)

        # Verify: LLM确实被调用过一次
        client.responses.create.assert_called_once()

    def test_compaction_success(self):
        # Arrange: 准备Session
        budget = ContextBudget(
            max_context_tokens=16384,
            reserved_output_tokens=2048
        )
        manager = ContextManager(budget, 0.8)
        execution_log = SQLiteExecutionLog(DB_PATH, str(uuid4()))
        session = AgentSession(manager, execution_log)
        session.append([
            {"role": "user", "content": "List project files"}
        ])
        for i in range(5):
            session.append([
                {
                    "type": "function_call",
                    "name": "list_files",
                    "call_id": f"call_{i}",
                    "arguments": '{"path":"src"}'
                },
                {
                    "type": "function_call_output",
                    "call_id": f"call_{i}",
                    "output": '["src/main.py", "src/config.py"]' * 50
                }
            ])
            session.mark_step_complete()
        # 保存执行前的状态
        history_before = session.get_history()
        recent_before = history_before[-4:]
        # Mock LLM Summary返回
        client = Mock()
        client.responses.create.return_value = Mock(
            output_text=(
                "Listed src directory. "
                "Found src/main.py and src/config.py."
            ),
            status="completed"
        )

        # Act
        compact_session(session, client)

        # Assert 1: History 长度
        self.assertEqual(len(session.get_history()), 6)
        # Assert 2: Step Boundaries
        self.assertEqual(session.get_step_ends(), [4, 6])
        # Assert 3: Pinned User Prompt 保持不变
        self.assertEqual(history_before[0], session.get_history()[0])
        # Assert 4: 最近两个 Steps 的内容保持不变
        self.assertEqual(recent_before, session.get_history()[-4:])
        # Assert 5: Summary 中包含预期的文件名
        summary_item = session.get_history()[1]
        self.assertEqual(summary_item["role"], "assistant")
        self.assertIn("src/main.py", summary_item["content"])
        self.assertIn("src/config.py", summary_item["content"])
        # Assert 6: Token Cache 在 replace_history 后失效
        self.assertIsNone(manager._cached_tokens)
        # 重新计算cached token
        session.should_compact()
        self.assertEqual(manager._cached_tokens, ContextHelper.estimate_tokens(session.get_history()))

    def test_compaction_no_eligible_history(self):
        # Arrange: 准备Session
        budget = ContextBudget(
            max_context_tokens=16384,
            reserved_output_tokens=2048
        )
        manager = ContextManager(budget, 0.8)
        execution_log = SQLiteExecutionLog(DB_PATH, str(uuid4()))
        session = AgentSession(manager, execution_log)
        session.append([
            {"role": "user", "content": "List project files"}
        ])
        # 构造 2 个完整的 Agent Steps
        for i in range(2):
            session.append([
                {
                    "type": "function_call",
                    "name": "list_files",
                    "call_id": f"call_{i}",
                    "arguments": '{"path":"."}'
                },
                {
                    "type": "function_call_output",
                    "call_id": f"call_{i}",
                    "output": '["main.py"]'
                }
            ])
            session.mark_step_complete()
        # 保存执行前的状态
        history_before = session.get_history()
        step_ends_before = session.get_step_ends()
        cache_before = manager._cached_tokens

        # Arrange: 模拟LLM调用失败
        client = Mock()

        # Act + Assert: 验证异常
        with self.assertRaisesRegex(ValueError, "Nothing to compact!"):
            compact_session(session, client)

        # Assert: 验证Session没被修改
        self.assertEqual(session.get_history(), history_before)
        self.assertEqual(session.get_step_ends(), step_ends_before)
        self.assertEqual(cache_before, manager._cached_tokens)

        # Verify: LLM没有被调用
        client.responses.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
