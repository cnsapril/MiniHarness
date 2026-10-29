"""Trusted tool definitions shared by model schema generation and execution."""
from dataclasses import dataclass
from enum import Enum
import os
import stat
from pathlib import Path
from typing import Callable
from uuid import uuid4

from permission import PermissionRequirement


class ToolSafety(str, Enum):
    READ_ONLY = "READ_ONLY"
    SIDE_EFFECTING = "SIDE_EFFECTING"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    handler: Callable
    safety: ToolSafety
    description: str
    parameters: dict
    permission: PermissionRequirement = PermissionRequirement.DENY

    def llm_schema(self) -> dict:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": True,
        }


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec]):
        self._specs = {}
        for spec in specs:
            if not spec.name or spec.name in self._specs:
                raise ValueError(f"Invalid or duplicate tool registration: {spec.name}")
            self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def llm_schemas(self) -> list[dict]:
        return [spec.llm_schema() for spec in self._specs.values()]


PROJECT_ROOT = Path.cwd().resolve()


def list_files(path: str) -> list[str]:
    target = (PROJECT_ROOT / path).resolve()
    if not target.is_relative_to(PROJECT_ROOT):
        raise PermissionError("Path outside project root")
    if not target.is_dir():
        raise NotADirectoryError(path)
    return sorted(item.name for item in target.iterdir())


def write_file(path: str, content: str) -> str:
    """Write within the workspace via directory FDs; reject symlink parents.

    This protects this specific file operation against path traversal races.
    It is NOT a sandbox for arbitrary executable code.
    """
    if os.name != "posix" or not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("Safe workspace writes require POSIX O_NOFOLLOW")
    rel = Path(path)
    if (not path or rel.is_absolute() or ".." in rel.parts
            or len(rel.parts) == 0 or "\\x00" in path):
        raise PermissionError("Invalid workspace file path")

    dir_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    root_fd = os.open(PROJECT_ROOT, dir_flags)
    parent_fd = root_fd
    opened_dirs = [root_fd]
    tmp_name = None
    try:
        for parent in rel.parts[:-1]:
            next_fd = os.open(parent, dir_flags, dir_fd=parent_fd)
            opened_dirs.append(next_fd)
            parent_fd = next_fd
        name = rel.parts[-1]
        try:
            info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                raise PermissionError("Target is not a regular file")
        except FileNotFoundError:
            pass
        tmp_name = f".miniharness-{uuid4().hex}.tmp"
        tmp_fd = os.open(
            tmp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600, dir_fd=parent_fd
        )
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        tmp_name = None
        os.fsync(parent_fd)
        return str(rel)
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        for fd in reversed(opened_dirs):
            os.close(fd)


def default_tool_registry() -> ToolRegistry:
    return ToolRegistry([
        ToolSpec(
            name="list_files",
            handler=list_files,
            safety=ToolSafety.READ_ONLY,
            permission=PermissionRequirement.ALLOW,
            description="List files and directories under a given path.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative directory path, e.g. '.' or 'src'",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="write_file",
            handler=write_file,
            safety=ToolSafety.SIDE_EFFECTING,
            permission=PermissionRequirement.ASK,
            description="Write a UTF-8 file under the project workspace after user approval.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative workspace file path"},
                    "content": {"type": "string", "description": "New file contents"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        ),
    ])
