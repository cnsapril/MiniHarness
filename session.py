from copy import deepcopy

from context_manager import ContextManager
from permission import DenialSource
from sqlite_execution_log import SQLiteExecutionLog
from tool_model import ToolInteraction


class AgentSession:

    @classmethod
    def restore(cls, context_manager: ContextManager, execution_log: SQLiteExecutionLog) -> "AgentSession":
        session = cls(context_manager, execution_log)
        recovery_history = execution_log.get_snapshots_for_recovery()
        for item in recovery_history:
            if item["kind"] == "context_replaced":
                # 处理全量SNAPSHOT
                session._restore_snapshot(item["history"], item["step_ends"])
            elif item["kind"] == "step_completed":
                # 处理step end replay
                end = item["end"]
                last_end = session._completed_step_ends[-1] if session._completed_step_ends else 0
                if not last_end < end <= len(session._history):
                    raise ValueError(f"Invalid restored step boundary: {end}" )
                session._replay_step_end(end)
            elif item["kind"] == "history_items":
                # 处理历史消息replay
                session._replay_history_items(item["items"])
            else:
                raise ValueError(f"Unrecognized kind {item['kind']}")
        # restore结束，直接重置token cache，下次lazy recalculate
        context_manager.invalidate()
        return session

    def __init__(self, context_manager: ContextManager, execution_log: SQLiteExecutionLog):
        self.context_manager = context_manager
        self.execution_log = execution_log
        self._history = []
        self._completed_step_ends: list[int] = []

    def _replay_history_items(self, items: list):
        deep_copy_items = deepcopy(items)
        self._history.extend(deep_copy_items)

    def _replay_step_end(self, end: int):
        self._completed_step_ends.append(end)

    def _restore_snapshot(self, history_snapshot: list, step_ends_snapshot: list[int]):
        deep_copy_history = deepcopy(history_snapshot)
        deep_copy_step_ends = deepcopy(step_ends_snapshot)
        self._history = deep_copy_history
        self._completed_step_ends = deep_copy_step_ends

    def append(self, items: list):
        deep_copy_items = deepcopy(items)
        self.execution_log.append_items(deep_copy_items)
        self._history.extend(deep_copy_items)
        try:
            self.context_manager.append(self._history, deep_copy_items)
        except Exception:
            self.context_manager.invalidate()
            raise

    def record_tool_interaction(self, interaction: ToolInteraction,
                                expected_started_count: int) -> None:
        """Commit the STARTED version, output and ledger before updating context."""
        output_item = deepcopy(interaction.output)
        self.execution_log.append_tool_interaction(interaction, expected_started_count)
        self._history.append(output_item)
        try:
            self.context_manager.append(self._history, [output_item])
        except Exception:
            self.context_manager.invalidate()
            raise

    def record_tool_denial(self, interaction: ToolInteraction,
                           expected_started_count: int, *,
                           retry: bool = False,
                           denial_source: DenialSource = DenialSource.POLICY) -> None:
        """Atomically persist denial and tool output before updating context."""
        output_item = deepcopy(interaction.output)
        self.execution_log.append_tool_denial(
            interaction, expected_started_count, retry=retry,
            denial_source=denial_source
        )
        self._history.append(output_item)
        try:
            self.context_manager.append(self._history, [output_item])
        except Exception:
            self.context_manager.invalidate()
            raise

    def replace_history(self, new_history: list, new_step_ends: list[int]):
        deep_copy_new_history = deepcopy(new_history)
        deep_copy_new_step_ends = deepcopy(new_step_ends)
        self.execution_log.append_context_replaced(deep_copy_new_history, deep_copy_new_step_ends)
        self._history = deep_copy_new_history
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

    def mark_step_complete(self,):
        # 记录当前完整Step在History中的结束位置
        end = len(self._history)
        self.execution_log.append_step_complete(end)
        self._completed_step_ends.append(end)

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
