"""Constrained Docker runner for MiniHarness' fixed run_tests tool.

The model never provides executable commands, mounts, image names or flags.
Docker is an isolation layer, not a guarantee against kernel/daemon exploits.
"""
import os
import selectors
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


class SandboxCleanupError(RuntimeError):
    """An aborted sandbox could not be conclusively removed."""


@dataclass(frozen=True)
class SandboxConfig:
    workspace_root: Path
    image: str = "miniharness-tests:local"
    timeout_seconds: float = 90.0
    max_output_bytes: int = 32 * 1024
    cleanup_timeout_seconds: float = 10.0
    docker_bin: str = "docker"

    def __post_init__(self):
        root = Path(self.workspace_root).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("Workspace must be a directory")
        if "," in str(root) or "\n" in str(root):
            raise ValueError("Workspace path cannot be encoded in Docker --mount")
        if self.timeout_seconds <= 0 or self.max_output_bytes < 1:
            raise ValueError("Timeout and output limits must be positive")
        if self.cleanup_timeout_seconds <= 0:
            raise ValueError("Cleanup timeout must be positive")
        if not self.image or self.image.startswith("-"):
            raise ValueError("Invalid trusted sandbox image")
        object.__setattr__(self, "workspace_root", root)


class SandboxRunner:
    """Runs one fixed test command with no host-shell interpolation.

    A bounded pipe reader prevents child output from exhausting host memory.
    When the command is interrupted, docker rm -f is mandatory. Failure to
    confirm cleanup raises rather than reporting a cleanly isolated result.
    """

    def __init__(self, config: SandboxConfig):
        self.config = config

    def _docker_command(self, name: str) -> list[str]:
        root = self.config.workspace_root
        return [
            self.config.docker_bin, "run", "--rm", "--pull=never",
            "--name", name,
            "--network", "none",
            "--read-only",
            "--mount", f"type=bind,src={root},dst=/workspace,readonly",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=64m,mode=1777",
            "--user", "65534:65534",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", "64",
            "--memory", "512m",
            "--cpus", "1",
            "--env", "PYTHONDONTWRITEBYTECODE=1",
            "--env", "HOME=/tmp",
            "--env", "PYTHONPATH=/workspace",
            "--workdir", "/workspace",
            self.config.image,
            "python", "-m", "unittest", "discover",
            "-s", "tests", "-p", "test_*.py",
        ]

    def _remove_container(self, name: str) -> None:
        try:
            removed = subprocess.run(
                [self.config.docker_bin, "rm", "-f", name],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, timeout=self.config.cleanup_timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxCleanupError(
                f"Cannot verify sandbox container {name} was removed"
            ) from exc
        if removed.returncode != 0 and b"No such container" not in removed.stderr:
            raise SandboxCleanupError(
                f"Cannot verify sandbox container {name} was removed"
            )

    @staticmethod
    def _stop_cli(proc: subprocess.Popen) -> None:
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)

    def run_tests(self) -> dict:
        """Return bounded output and a status; never execute a model-supplied shell."""
        name = "miniharness-" + uuid4().hex
        start = time.monotonic()
        output = bytearray()
        proc = None
        aborted = False
        status = None
        exit_code = None
        try:
            proc = subprocess.Popen(
                self._docker_command(name),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
            )
            with selectors.DefaultSelector() as selector:
                selector.register(proc.stdout, selectors.EVENT_READ)
                while True:
                    remaining = self.config.timeout_seconds - (time.monotonic() - start)
                    if remaining <= 0:
                        status = "TIMED_OUT"
                        aborted = True
                        break
                    events = selector.select(min(remaining, 0.2))
                    if not events:
                        continue
                    chunk = os.read(proc.stdout.fileno(), 4096)
                    if not chunk:  # EOF: Docker CLI has closed its output.
                        break
                    room = self.config.max_output_bytes - len(output)
                    output.extend(chunk[:room])
                    if len(chunk) > room:
                        status = "OUTPUT_LIMIT_EXCEEDED"
                        aborted = True
                        break
            if not aborted:
                remaining = max(0.001, self.config.timeout_seconds - (time.monotonic() - start))
                try:
                    exit_code = proc.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    status = "TIMED_OUT"
                    aborted = True
                else:
                    status = "COMPLETED" if exit_code == 0 else "FAILED"
        except FileNotFoundError:
            status = "DOCKER_UNAVAILABLE"
        except BaseException:
            # Fail closed: external effects may have occurred. Do not convert
            # an unknown outcome into a normal tool failure.
            aborted = proc is not None
            raise
        finally:
            if aborted and proc is not None:
                try:
                    self._stop_cli(proc)
                finally:
                    self._remove_container(name)
            if proc is not None and proc.stdout is not None:
                proc.stdout.close()

        return {
            "status": status,
            "exit_code": exit_code,
            "output": bytes(output).decode("utf-8", errors="replace"),
            "duration_ms": round((time.monotonic() - start) * 1000),
        }
