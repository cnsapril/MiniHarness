from copy import deepcopy

from context_manager import ContextManager


class AgentSession:

    def __init__(self, context_manager: ContextManager):
        self._history = []
        self.context_manager = context_manager

    def append(self, items: list):
        deep_copy_items = deepcopy(items)
        self._history.extend(deep_copy_items)
        try:
            self.context_manager.append(self._history, deep_copy_items)
        except Exception:
            self.context_manager.invalidate()
            raise

    def replace_history(self, new_history: list):
        deep_copy_new_history = deepcopy(new_history)
        self._history = deep_copy_new_history
        self.context_manager.invalidate()

    def get_history(self) -> list:
        return deepcopy(self._history)

    def should_compact(self) -> bool:
        return self.context_manager.should_compact(self._history)
