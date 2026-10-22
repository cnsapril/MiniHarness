from dataclasses import dataclass, field


@dataclass
class LLMCallTrace:
    step: int
    duration_ms: float
    input_tokens: int | None
    output_tokens: int | None
    response_id: str | None
    output_types: list[str]


@dataclass
class AgentTrace:
    llm_calls: list[LLMCallTrace] = field(
        default_factory=list
    )
