from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class AgentStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    LIMIT_EXCEEDED = "limit_exceeded"


@dataclass
class LLMCallTrace:
    step: int
    duration_ms: float
    input_tokens: int | None
    output_tokens: int | None
    response_id: str | None
    output_types: list[str]
    success: bool
    attempt: int
    error: str | None = None


@dataclass
class ToolCallTrace:
    step: int
    call_id: str
    tool_name: str
    duration_ms: float
    success: bool
    error: str | None = None


@dataclass
class AgentTrace:
    llm_calls: list[LLMCallTrace] = field(
        default_factory=list
    )
    tool_calls: list[ToolCallTrace] = field(
        default_factory=list
    )
    status: AgentStatus = AgentStatus.RUNNING
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    error: str | None = None

    @property
    def duration_ms(self) -> float | None:
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds() * 1000
