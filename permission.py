"""Trusted, session-fixed permission policy for MiniHarness v0.1.

This is application-level admission, NOT an OS sandbox. Every invocation
must still pass through the manager and the database execution CAS.
"""
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PureWindowsPath


class PermissionRequirement(str, Enum):
    ALLOW = "ALLOW"
    ASK = "ASK"
    DENY = "DENY"


@dataclass(frozen=True)
class PermissionDecision:
    requirement: PermissionRequirement
    reason: str = ""


class PermissionPolicy:
    """Immutable per-session policy with a single trusted workspace root."""

    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root).resolve()

    def evaluate(self, spec, arguments: dict) -> PermissionDecision:
        if spec is None:
            return PermissionDecision(PermissionRequirement.DENY, "Unknown tool")
        if not isinstance(arguments, dict):
            return PermissionDecision(PermissionRequirement.DENY, "Invalid tool arguments")
        requirement = spec.permission
        if requirement == PermissionRequirement.DENY:
            return PermissionDecision(requirement, "Tool not permitted by policy")

        if spec.name in ("list_files", "write_file"):
            path = arguments.get("path")
            if not isinstance(path, str) or not path or "\x00" in path:
                return PermissionDecision(PermissionRequirement.DENY, "Invalid workspace path")
            rel = Path(path)
            # Reject traversal outright, even if it happens to resolve inside root.
            if (rel.is_absolute() or PureWindowsPath(path).is_absolute()
                    or ".." in rel.parts or "\\" in path):
                return PermissionDecision(PermissionRequirement.DENY, "Path outside workspace policy")
            if any(part == ".git" or part == ".env" or part.startswith(".env.")
                   for part in rel.parts):
                return PermissionDecision(PermissionRequirement.DENY, "Protected workspace path")
            if not (self.workspace_root / rel).resolve().is_relative_to(self.workspace_root):
                return PermissionDecision(PermissionRequirement.DENY, "Path outside workspace")
            if spec.name == "write_file":
                content = arguments.get("content")
                if not isinstance(content, str) or len(content.encode("utf-8")) > 128 * 1024:
                    return PermissionDecision(PermissionRequirement.DENY, "Invalid or oversized content")
                if rel == Path("."):
                    return PermissionDecision(PermissionRequirement.DENY, "Expected a file path")
        return PermissionDecision(requirement)
