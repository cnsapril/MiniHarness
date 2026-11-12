"""Minimal mid-turn control flow.

The harness never interprets whether old tool arguments satisfy new user intent.
Steering is injected for the next model sample; interrupts stop further dispatch.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from context_manager import ContextHelper
from tool_model import ToolInteraction


class ControlAction(str, Enum):
    NONE = "none"
    STEERED = "steered"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class TurnEvent:
    type: str
    data: dict[str, Any] = field(default_factory=dict)


TurnEventSink = Callable[[TurnEvent], None]


class TurnController:
    def __init__(self, session):
        self.session = session

    def has_pending(self) -> bool:
        return self.session.execution_log.has_pending_control()

    def consume_pending(self) -> ControlAction:
        pending = self.session.execution_log.get_pending_controls()
        if not pending:
            return ControlAction.NONE

        # An explicit interrupt takes priority over queued soft steering.
        interrupts = [item for item in pending if item["kind"] == "INTERRUPT"]
        if interrupts:
            self.session.execution_log.interrupt_turn(interrupts[0]["id"])
            return ControlAction.INTERRUPTED

        for item in pending:
            self.session.apply_steering(item["id"], item["content"])
        return ControlAction.STEERED

    def cancel_unstarted(self, tool_calls, reason: str) -> int:
        """Close every unstarted function call with a durable output + terminal state."""
        cancelled = 0
        for item in tool_calls:
            serialized = ContextHelper.serialize_items([item])[0]
            if serialized.get("type") != "function_call":
                continue
            output = {
                "type": "function_call_output",
                "call_id": serialized["call_id"],
                "output": json.dumps({
                    "error": "CANCELLED_BY_CONTROL",
                    "reason": reason,
                    "previous_outcome": "NOT_STARTED",
                }),
            }
            self.session.cancel_recorded_tool(ToolInteraction(
                call_id=serialized["call_id"], input=serialized, output=output
            ))
            cancelled += 1
        return cancelled
