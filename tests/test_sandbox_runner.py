"""Docker runner checks without requiring Docker on CI.

A fake docker executable tests argument construction, output limits and cleanup.
A real Docker smoke test is opt-in via MINIHARNESS_DOCKER_TEST=1.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from context_manager import ContextBudget, ContextManager
from permission import PermissionPolicy, PermissionRequirement
from sandbox_runner import SandboxCleanupError, SandboxConfig, SandboxRunner
from session import AgentSession
from sqlite_execution_log import SQLiteExecutionLog
from tool_execution import ToolDispatchStatus, ToolExecutionManager
from tool_registry import ToolRegistry, ToolSafety, ToolSpec, default_tool_registry


FAKE_DOCKER = r'''#!/usr/bin/env python3
import json
import os
import sys
import time

with open(os.environ["FAKE_DOCKER_LOG"], "a", encoding="utf8") as f:
    f.write(json.dumps(sys.argv[1:]) + "\n")
mode = os.environ.get("FAKE_DOCKER_MODE", "success")
if sys.argv[1] == "rm":
    if os.environ.get("FAKE_CLEANUP_FAIL") == "1":
        print("daemon unavailable", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)
if mode == "sleep":
    time.sleep(5)
elif mode == "flood":
    sys.stdout.write("X" * 200000)
    sys.stdout.flush()
elif mode == "failed_tests":
    print("FAIL: test_something")
    sys.exit(1)
else:
    print("Ran 1 test: OK")
'''


class SandboxRunnerTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.docker = self.root / "docker-fake"
        self.docker.write_text(FAKE_DOCKER, encoding="utf-8")
        self.docker.chmod(0o755)
        self.log = self.root / "docker.log"
        self.env = patch.dict(os.environ, {"FAKE_DOCKER_LOG": str(self.log)})
        self.env.start()
        self.addCleanup(self.env.stop)

    def make_runner(self, **kwargs):
        return SandboxRunner(SandboxConfig(
            self.root, docker_bin=str(self.docker),
            timeout_seconds=2, max_output_bytes=1024, **kwargs,
        ))

    def docker_calls(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_fixed_command_and_security_flags(self):
        result = self.make_runner().run_tests()
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("Ran 1 test: OK", result["output"])
        calls = self.docker_calls()
        self.assertEqual(len(calls), 1)
        args = calls[0]
        self.assertEqual(args[0:2], ["run", "--rm"])
        for pair in (
            ["--network", "none"], ["--cap-drop", "ALL"],
            ["--security-opt", "no-new-privileges"],
            ["--pids-limit", "64"], ["--memory", "512m"],
            ["--cpus", "1"], ["--user", "65534:65534"],
            ["--read-only", "--mount"], ["--pull=never", "--name"],
        ):
            self.assertIn(pair, [args[i:i+len(pair)] for i in range(len(args))])
        self.assertIn(f"type=bind,src={self.root},dst=/workspace,readonly", args)
        self.assertIn("/tmp:rw,nosuid,nodev,size=64m,mode=1777", args)
        self.assertEqual(args[-8:], ["python", "-m", "unittest", "discover",
                                     "-s", "tests", "-p", "test_*.py"])
        self.assertNotIn("sh", args)
        self.assertNotIn("bash", args)
        self.assertNotIn("--privileged", args)
        self.assertNotIn("/var/run/docker.sock", " ".join(args))

    def test_failed_tests_are_not_a_handler_exception(self):
        with patch.dict(os.environ, {"FAKE_DOCKER_MODE": "failed_tests"}):
            result = self.make_runner().run_tests()
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("FAIL", result["output"])

    def test_timeout_kills_container(self):
        with patch.dict(os.environ, {"FAKE_DOCKER_MODE": "sleep"}):
            runner = SandboxRunner(SandboxConfig(
                self.root, docker_bin=str(self.docker),
                timeout_seconds=0.15, max_output_bytes=1024,
            ))
            result = runner.run_tests()
        self.assertEqual(result["status"], "TIMED_OUT")
        calls = self.docker_calls()
        self.assertEqual(len(calls), 2)
        name = calls[0][calls[0].index("--name") + 1]
        self.assertEqual(calls[1], ["rm", "-f", name])

    def test_excessive_output_stops_container_and_bounds_output(self):
        with patch.dict(os.environ, {"FAKE_DOCKER_MODE": "flood"}):
            result = self.make_runner().run_tests()
        self.assertEqual(result["status"], "OUTPUT_LIMIT_EXCEEDED")
        self.assertEqual(len(result["output"].encode("utf-8")), 1024)
        self.assertEqual(self.docker_calls()[-1][:2], ["rm", "-f"])

    def test_cleanup_failure_is_not_hidden(self):
        with patch.dict(os.environ, {
            "FAKE_DOCKER_MODE": "flood", "FAKE_CLEANUP_FAIL": "1",
        }):
            with self.assertRaises(SandboxCleanupError):
                self.make_runner().run_tests()

    def test_missing_docker_is_structured_unavailable(self):
        runner = SandboxRunner(SandboxConfig(
            self.root, docker_bin=str(self.root / "missing-binary"),
            timeout_seconds=2, max_output_bytes=1024,
        ))
        self.assertEqual(runner.run_tests()["status"], "DOCKER_UNAVAILABLE")

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            SandboxConfig(self.root, timeout_seconds=0)
        with self.assertRaises(ValueError):
            SandboxConfig(self.root, max_output_bytes=0)
        with self.assertRaises(ValueError):
            SandboxConfig(self.root, image="--privileged")
        with self.assertRaises(ValueError):
            SandboxConfig(self.root / "nonexistent")

    def test_registry_has_fixed_approval_gated_run_tests(self):
        registry = default_tool_registry()
        spec = registry.get("run_tests")
        self.assertEqual(spec.permission, PermissionRequirement.ASK)
        self.assertEqual(spec.safety, ToolSafety.SIDE_EFFECTING)
        self.assertEqual(spec.llm_schema()["parameters"]["properties"], {})
        self.assertEqual(spec.llm_schema()["parameters"]["additionalProperties"], False)

    def test_manager_approval_and_session_recording(self):
        db = SQLiteExecutionLog(str(self.root / "agent.sqlite"), "session-1")
        self.addCleanup(db.close)
        session = AgentSession(ContextManager(ContextBudget(16000, 2000), 0.8), db)
        called = []
        runner = self.make_runner()
        def trusted_handler():
            called.append(db.get_tool_execution("test-call")["state"])
            return runner.run_tests()
        registry = ToolRegistry([
            ToolSpec("run_tests", trusted_handler, ToolSafety.SIDE_EFFECTING,
                     "Run fixed tests", {"type": "object", "properties": {}},
                     permission=PermissionRequirement.ASK),
        ])
        manager = ToolExecutionManager(db, registry, PermissionPolicy(self.root))
        item = {"type": "function_call", "name": "run_tests",
                "call_id": "test-call", "arguments": "{}"}
        session.append([item])
        pending = manager.execute(item)
        self.assertEqual(pending.status, ToolDispatchStatus.APPROVAL_PENDING)
        self.assertEqual(called, [])
        manager.approve("test-call", expected_started_count=0)
        executed = manager.execute(item)
        self.assertEqual(executed.status, ToolDispatchStatus.COMPLETED)
        self.assertEqual(executed.result["status"], "COMPLETED")
        self.assertEqual(called, ["STARTED"])
        session.record_tool_interaction(executed.interaction, executed.started_count)
        self.assertEqual(db.get_tool_execution("test-call")["state"], "RESULT_RECORDED")


@unittest.skipUnless(os.environ.get("MINIHARNESS_DOCKER_TEST") == "1",
                     "Set MINIHARNESS_DOCKER_TEST=1 after building the sandbox image")
class RealDockerSmokeTest(unittest.TestCase):
    def test_real_container_runs_unittests(self):
        project = Path(__file__).resolve().parents[1]
        runner = SandboxRunner(SandboxConfig(
            project, timeout_seconds=180, max_output_bytes=16 * 1024,
        ))
        result = runner.run_tests()
        self.assertEqual(result["status"], "COMPLETED", result)
        self.assertEqual(result["exit_code"], 0, result)


if __name__ == "__main__":
    unittest.main()
