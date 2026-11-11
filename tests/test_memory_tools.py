"""Memory tools use the existing durable approval path and project scope."""
import json
import tempfile
import unittest
from pathlib import Path

from context_manager import ContextBudget, ContextManager
from memory_store import MemoryStore
from memory_tools import memory_tool_specs
from permission import PermissionPolicy, PermissionRequirement
from session import AgentSession
from sqlite_execution_log import SQLiteExecutionLog
from tool_execution import ToolDispatchStatus, ToolExecutionManager
from tool_registry import ToolRegistry


class MemoryToolIntegrationTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.db = self.root / "memory.db"
        self.log_path = str(self.root / "execution.db")
        self.registry = ToolRegistry(memory_tool_specs(self.db, "project-a"))
        self.log = SQLiteExecutionLog(self.log_path, "session-a")
        self.addCleanup(self.log.close)
        self.session = AgentSession(
            ContextManager(ContextBudget(16384, 2048), 0.8), self.log
        )
        self.manager = ToolExecutionManager(
            self.log, self.registry, PermissionPolicy(self.root)
        )

    def call(self, name: str, arguments: dict, call_id: str) -> dict:
        item = {"type": "function_call", "call_id": call_id, "name": name,
                "arguments": json.dumps(arguments)}
        self.session.append([item])
        return item

    def commit(self, result) -> dict:
        self.session.record_tool_interaction(result.interaction, result.started_count)
        return json.loads(result.interaction.output["output"])

    def test_memory_save_requires_real_approval_and_survives_new_session(self):
        item = self.call("memory_save", {"kind": "decision",
                    "content": "PaymentService uses PostgreSQL"}, "save-1")
        pending = self.manager.execute(item)
        self.assertEqual(pending.status, ToolDispatchStatus.APPROVAL_PENDING)
        with MemoryStore(self.db) as store:
            self.assertEqual(store.search("project-a", "PaymentService"), [])
        self.manager.approve("save-1", expected_started_count=0)
        saved = self.commit(self.manager.execute(item))
        self.assertEqual(saved["status"], "active")

        second_log = SQLiteExecutionLog(self.log_path, "session-b")
        try:
            second_session = AgentSession(
                ContextManager(ContextBudget(16384, 2048), 0.8), second_log
            )
            second_manager = ToolExecutionManager(
                second_log, self.registry, PermissionPolicy(self.root)
            )
            search = {"type": "function_call", "call_id": "search-1",
                      "name": "memory_search",
                      "arguments": json.dumps({"query": "PaymentService", "limit": 5})}
            second_session.append([search])
            result = second_manager.execute(search)
            self.assertEqual(result.status, ToolDispatchStatus.COMPLETED)
            second_session.record_tool_interaction(result.interaction, result.started_count)
            found = json.loads(result.interaction.output["output"])
            self.assertEqual(found[0]["id"], saved["memory_id"])
        finally:
            second_log.close()

    def test_bad_model_args_fail_permission_gate(self):
        policy = PermissionPolicy(self.root)
        save_spec = self.registry.get("memory_save")
        self.assertEqual(policy.evaluate(save_spec, {"kind": "fact", "content": "valid"}).requirement,
                         PermissionRequirement.ASK)
        self.assertEqual(policy.evaluate(save_spec, {"kind": "fact", "content": ""}).requirement,
                         PermissionRequirement.DENY)
        self.assertEqual(policy.evaluate(self.registry.get("memory_search"),
                                         {"query": "ledger", "limit": 5}).requirement,
                         PermissionRequirement.ALLOW)
        self.assertEqual(policy.evaluate(self.registry.get("memory_search"),
                                         {"query": "ledger", "limit": 999}).requirement,
                         PermissionRequirement.DENY)

    def test_memory_update_and_invalidation_keep_audit_history(self):
        with MemoryStore(self.db) as store:
            old = store.save("project-a", "fact", "PaymentService uses MySQL")
        item = self.call("memory_update", {"memory_id": old,
                         "content": "PaymentService uses PostgreSQL"}, "update-1")
        self.assertEqual(self.manager.execute(item).status,
                         ToolDispatchStatus.APPROVAL_PENDING)
        self.manager.approve("update-1", expected_started_count=0)
        updated = self.commit(self.manager.execute(item))
        new = updated["memory_id"]
        with MemoryStore(self.db) as store:
            self.assertEqual(store.get("project-a", old)["status"], "superseded")
            self.assertEqual(store.get("project-a", new)["supersedes_id"], old)
        item = self.call("memory_invalidate", {"memory_id": new}, "invalidate-1")
        self.assertEqual(self.manager.execute(item).status,
                         ToolDispatchStatus.APPROVAL_PENDING)
        self.manager.approve("invalidate-1", expected_started_count=0)
        self.commit(self.manager.execute(item))
        with MemoryStore(self.db) as store:
            self.assertEqual(store.search("project-a", "PostgreSQL"), [])
            self.assertEqual(store.get("project-a", new)["status"], "invalidated")


if __name__ == "__main__":
    unittest.main()
