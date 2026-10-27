from copy import deepcopy


class ExecutionLog:
    def __init__(self):
        self._events: list[dict] = []

    def append_items(self, items: list):
        serialized = [
            item.model_dump(mode="json")
            if hasattr(item, "model_dump")
            else item
            for item in items
        ]
        self._events.append({
            "kind": "history_items",
            "items": deepcopy(serialized)
        })

    def get_events(self) -> list[dict]:
        return deepcopy(self._events)

    def get_tool_result(self, call_id: str) -> dict | None:
        matches = []
        for event in self._events:
            if event["kind"] != "history_items":
                continue
            for item in event["items"]:
                if item.get("type") == "function_call_output" and item.get("call_id") == call_id:
                    matches.append(item)
        if len(matches) > 1:
            raise ValueError(f"Duplicate tool result: {call_id}")
        return deepcopy(matches[0]) if matches else None
