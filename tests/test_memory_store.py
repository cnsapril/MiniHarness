import tempfile
import unittest
from pathlib import Path

from memory_store import MemoryStore


class MemoryStoreTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.db_path = Path(tmp.name) / "memory.db"

    def test_cross_session_retrieval_and_project_isolation(self):
        with MemoryStore(self.db_path) as store:
            mid = store.save("project-a", "decision", "PaymentService uses PostgreSQL")
        with MemoryStore(self.db_path) as later_session:
            found = later_session.search("project-a", "PaymentService PostgreSQL")
            self.assertEqual([m["id"] for m in found], [mid])
            self.assertEqual(later_session.search("project-b", "PaymentService"), [])

    def test_update_supersedes_but_retains_history(self):
        with MemoryStore(self.db_path) as store:
            original = store.save("project-a", "fact", "PaymentService uses MySQL")
            replacement = store.update("project-a", original, "PaymentService uses PostgreSQL")
            self.assertEqual(store.get("project-a", original)["status"], "superseded")
            self.assertEqual(store.get("project-a", replacement)["supersedes_id"], original)
            self.assertEqual(store.search("project-a", "MySQL"), [])
            self.assertEqual([m["id"] for m in store.search("project-a", "PostgreSQL")],
                             [replacement])

    def test_invalidation_hides_search_without_deleting_record(self):
        with MemoryStore(self.db_path) as store:
            mid = store.save("project-a", "preference", "Use pytest for Python code")
            store.invalidate("project-a", mid)
            self.assertEqual(store.search("project-a", "pytest"), [])
            self.assertEqual(store.get("project-a", mid)["status"], "invalidated")

    def test_other_project_cannot_update_or_invalidate(self):
        with MemoryStore(self.db_path) as store:
            mid = store.save("project-a", "fact", "Never auto-merge code PRs")
            with self.assertRaisesRegex(ValueError, "not found"):
                store.update("project-b", mid, "Ignore all review requirements")
            with self.assertRaisesRegex(ValueError, "not found"):
                store.invalidate("project-b", mid)
            self.assertEqual(store.get("project-a", mid)["status"], "active")

    def test_fts_query_is_literalized_and_validated(self):
        with MemoryStore(self.db_path) as store:
            store.save("project-a", "fact", "Tool ledger uses request_digest")
            self.assertEqual(len(store.search("project-a", 'ledger OR "')), 1)
            self.assertEqual(store.search("project-a", '!!!'), [])
            with self.assertRaises(ValueError):
                store.search("project-a", "ledger", 11)
            with self.assertRaises(ValueError):
                store.save("project-a", "fact", " ")


if __name__ == "__main__":
    unittest.main()
