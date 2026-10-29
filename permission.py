"""Trusted, session-fixed permission policy for MiniHarness v0.1.

This is application-level admission, NOT an OS sandbox. Every invocation
must still pass through the manager and the database execution CAS.
"""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


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

        # The policy gate is tool-agnostic. Trusted ToolSpec owns any
        # argument-specific constraints (paths, content, resource scopes, ...).
        if spec.permission_check is not None:
            try:
                reason = spec.permission_check(arguments, self.workspace_root)
            except Exception:
                return PermissionDecision(
                    PermissionRequirement.DENY, "Tool permission validation failed"
                )
            if reason:
                return PermissionDecision(PermissionRequirement.DENY, reason)
        return PermissionDecision(requirement)
