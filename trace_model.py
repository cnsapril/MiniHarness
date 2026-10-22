from dataclasses import dataclass, field


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
