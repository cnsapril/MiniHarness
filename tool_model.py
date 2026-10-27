from copy import deepcopy
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolInteraction:
    """A completed tool call/result pair, not an execution state."""

    call_id: str
    input: dict
    output: dict

    def __post_init__(self) -> None:
        if not isinstance(self.call_id, str) or not self.call_id:
            raise ValueError("Tool call ID must be a non-empty string")
        if not isinstance(self.input, dict) or self.input.get("type") != "function_call":
            raise ValueError("Invalid tool call input")
        if not isinstance(self.output, dict) or self.output.get("type") != "function_call_output":
            raise ValueError("Invalid tool call output")
        if self.input.get("call_id") != self.call_id or self.output.get("call_id") != self.call_id:
            raise ValueError("Tool call ID mismatch")

        # Decouple the interaction from the original SDK/handler dictionaries.
        object.__setattr__(self, "input", deepcopy(self.input))
        object.__setattr__(self, "output", deepcopy(self.output))

    def to_payload(self) -> dict:
        return {"input": deepcopy(self.input), "output": deepcopy(self.output)}
