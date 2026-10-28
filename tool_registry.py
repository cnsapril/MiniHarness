"""Trusted tool definitions shared by model schema generation and execution."""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable


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


def default_tool_registry() -> ToolRegistry:
    return ToolRegistry([
        ToolSpec(
            name="list_files",
            handler=list_files,
            safety=ToolSafety.READ_ONLY,
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
    ])
