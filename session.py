from copy import deepcopy

from context_manager import ContextManager


class AgentSession:

    def __init__(self, context_manager: ContextManager):
        self.context_manager = context_manager
        self._history = []
        self._completed_step_ends: list[int] = []

    def append(self, items: list):
        deep_copy_items = deepcopy(items)
        self._history.extend(deep_copy_items)
        try:
            self.context_manager.append(self._history, deep_copy_items)
        except Exception:
            self.context_manager.invalidate()
            raise

    def replace_history(self, new_history: list, new_step_ends: list[int]):
        deep_copy_new_history = deepcopy(new_history)
        self._history = deep_copy_new_history
        deep_copy_new_step_ends = deepcopy(new_step_ends)
        self._completed_step_ends = deep_copy_new_step_ends
        self.context_manager.invalidate()

    def get_history(self) -> list:
        return deepcopy(self._history)

    def get_step_ends(self) -> list[int]:
        return self._completed_step_ends.copy()

    def should_compact(self) -> bool:
        return self.context_manager.should_compact(self._history)

    def get_compaction_threshold(self) -> int:
        return self.context_manager.get_compaction_threshold()

    def mark_step_complete(self):
        # 记录当前完整Step在History中的结束位置
        self._completed_step_ends.append(len(self._history))

    def split_for_compaction(self, keep_recent_steps: int = 2) -> tuple[list, list, list]:
        # 返回：pinned / old_history / recent_history
        # TODO: 后续优化pinned history的获取方式，先写死
        pinned_history_idx = 1
        old_history_end_boundary = len(self._completed_step_ends) - keep_recent_steps
        copied_history = deepcopy(self._history)

        pinned_history = copied_history[:pinned_history_idx]
        if old_history_end_boundary < 0:
            return pinned_history, [], copied_history[pinned_history_idx:]
        old_history = []
        last_step_end_idx = pinned_history_idx
        for current_step_end_idx in self._completed_step_ends[:old_history_end_boundary]:
            step_history = copied_history[last_step_end_idx:current_step_end_idx]
            old_history.extend(step_history)
            last_step_end_idx = current_step_end_idx
        recent_history = copied_history[last_step_end_idx:]
        return pinned_history, old_history, recent_history
